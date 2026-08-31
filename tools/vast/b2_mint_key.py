#!/usr/bin/env python3
"""Mint / revoke ephemeral, capability-scoped B2 application keys.

Implements docs/plans/keyless-b2-ingest.md Option 1: rented boxes never see a
standing credential — the launcher mints a bucket-restricted, no-delete,
self-expiring key per launch and ships THAT under the unchanged on-box names
(B2_KEY_ID/B2_APPLICATION_KEY). stdlib-only (urllib against the native B2 API,
no b2 CLI / SDK dependency).

Requires in env / repo-root .env:
  B2_MINTER_KEY_ID / B2_MINTER_APPLICATION_KEY  account-level key with ONLY
        listBuckets,listKeys,writeKeys,deleteKeys (no file access) —
        workstation-only, never shipped to a box.
  B2_BUCKET                                     default bucket to restrict to.

Subcommands:
  mint    --name N|--run RID [--hours 48] [--caps ...] [--prefix P] [--json]
          Revoke-then-mint on name collision (relaunches get fresh keys).
          Default caps: listFiles,readFiles,writeFiles — NO deleteFiles.
          Prints `export B2_KEY_ID=... B2_APPLICATION_KEY=...` lines by
          default (eval-able); --var-prefix renames; --json for machines.
          THE SECRET IS PRINTED EXACTLY ONCE, HERE — never logged elsewhere.
  mint-pair --name N --write-prefix P [--hours 48] [--json]
          Mint the Option-1b (read, write) pair via mint_pair(): bucket-wide
          RO '<name>-ro' + prefix-scoped RW '<name>-rw'. Default output is the
          four eval-able export lines (B2_KEY_ID / B2_APPLICATION_KEY /
          B2_WRITE_KEY_ID / B2_WRITE_APPLICATION_KEY — the §2.1 wire names in
          docs/plans/cred-broker-buildout.md); --json for machines.
  revoke  --name N|--run RID       delete all keys with that (sanitized) name
  ls      [--all]                  list keys + expiry (ephemerals only unless --all)
  gc                               delete expired ephemeral keys (run-*/box-*/serve-*/smoke-*)
  whoami  [--pair box|ops|minter]  authorize with an .env pair, print its
                                   capabilities/bucket/expiry (cap-check tool)

Security notes: the no-delete guarantee holds only while the bucket lifecycle
keeps hidden/shadowed versions (daysFromHidingToDeleting >= 30, set
2026-07-10); `writeFiles` can hide or shadow objects but the prior version
stays recoverable with the ops key for that window.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

# v4 is REQUIRED: bucket-restricted keys minted via the v4 API/CLI carry the
# multi-bucket `allowed.buckets` structure and 400 on v1-v3 authorize.
AUTH_URL = "https://api.backblazeb2.com/b2api/v4/b2_authorize_account"
DEFAULT_CAPS = "listFiles,readFiles,writeFiles"
# Per-role scoping (Option 1b, tools/vast/CREDENTIAL_LIFECYCLE.md): a box that
# needs prefix-scoped writes gets a PAIR — a bucket-wide read key + a write key
# restricted to one namePrefix. B2 keys carry a single namePrefix applied to all
# caps, so read (bucket-wide) and write (one prefix) cannot share a key.
READ_CAPS = "listFiles,readFiles"                 # bucket-wide reader (no writes)
WRITE_CAPS = "listFiles,readFiles,writeFiles"     # scoped to a namePrefix
EPHEMERAL_PREFIXES = ("run-", "box-", "serve-", "smoke-", "job-")
# PUBLISH grant (2026-08-05). A training bundle's publish stage writes the named
# adapter to checkpoints/<RUN_NAME>/ — outside the jobs/-scoped write key, and a
# B2 key carries exactly ONE namePrefix, so a box that publishes needs a SECOND
# scoped write key rather than a widened one. Same no-delete caps, same TTL, same
# revoke-on-destroy (`_revoke_box_keys` prefix-matches '<base>-'). Set
# B2_PUBLISH_PREFIX='' to ship a box with no publish grant at all.
# Incident + design: docs/plans/witness/g2_push/B2_PUBLISH_KEY_SCOPE_FIX_2026-08-05.md
PUBLISH_PREFIX = "checkpoints/"
MAX_HOURS = 24 * 999          # b2 validDurationInSeconds cap is < 1000 days


class MintError(RuntimeError):
    pass


def load_env() -> None:
    """Populate os.environ from the nearest .env walking up (mirror herdd)."""
    d = os.getcwd()
    for _ in range(6):
        p = os.path.join(d, ".env")
        if os.path.isfile(p):
            for line in open(p):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.split(" #", 1)[0].strip().strip('"').strip("'")
                os.environ.setdefault(k.strip(), v)
            return
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd


def _http(url: str, body: dict | None = None, headers: dict | None = None) -> dict:
    """One JSON request. Seam for tests (monkeypatched)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode()).get("message", "")
        except Exception:
            detail = ""
        raise MintError(f"B2 API {e.code} on {url.rsplit('/', 1)[-1]}: {detail}")


def authorize(kid: str, key: str) -> dict:
    """b2_authorize_account (v4) normalized to a flat dict:
    accountId, apiUrl, authorizationToken, allowed{capabilities,buckets,
    namePrefix}, expiration."""
    tok = base64.b64encode(f"{kid}:{key}".encode()).decode()
    r = _http(AUTH_URL, headers={"Authorization": f"Basic {tok}"})
    sa = r["apiInfo"]["storageApi"]
    al = sa.get("allowed") or {}
    return {
        "accountId": r["accountId"],
        "apiUrl": sa["apiUrl"],
        "authorizationToken": r["authorizationToken"],
        "allowed": {"capabilities": al.get("capabilities", []),
                    "buckets": al.get("buckets"),
                    "namePrefix": al.get("namePrefix")},
        "expiration": r.get("applicationKeyExpirationTimestamp"),
    }


def _minter_auth() -> dict:
    kid = os.environ.get("B2_MINTER_KEY_ID")
    key = os.environ.get("B2_MINTER_APPLICATION_KEY")
    if not (kid and key):
        raise MintError("B2_MINTER_KEY_ID/B2_MINTER_APPLICATION_KEY unset "
                        "(see docs/plans/keyless-b2-ingest.md)")
    return authorize(kid, key)


def _api(auth: dict, call: str, body: dict) -> dict:
    return _http(f"{auth['apiUrl']}/b2api/v4/{call}", body=body,
                 headers={"Authorization": auth["authorizationToken"]})


def sanitize_name(s: str) -> str:
    """B2 key names allow only [A-Za-z0-9-], <=100 chars; RUN_IDs allow ._ too."""
    out = re.sub(r"-{2,}", "-", re.sub(r"[^A-Za-z0-9-]", "-", s)).strip("-")
    if not out:
        raise MintError(f"key name {s!r} sanitizes to nothing")
    return out[:100]


def resolve_bucket_id(auth: dict, bucket_name: str) -> str:
    r = _api(auth, "b2_list_buckets",
             {"accountId": auth["accountId"], "bucketName": bucket_name})
    for b in r.get("buckets", []):
        if b["bucketName"] == bucket_name:
            return b["bucketId"]
    raise MintError(f"bucket {bucket_name!r} not found (minter needs listBuckets)")


def list_keys(auth: dict) -> list[dict]:
    keys, start = [], None
    while True:
        body = {"accountId": auth["accountId"], "maxKeyCount": 1000}
        if start:
            body["startApplicationKeyId"] = start
        r = _api(auth, "b2_list_keys", body)
        keys.extend(r.get("keys", []))
        start = r.get("nextApplicationKeyId")
        if not start:
            return keys


def delete_key(auth: dict, application_key_id: str) -> None:
    _api(auth, "b2_delete_key", {"applicationKeyId": application_key_id})


def revoke_by_name(auth: dict, name: str) -> int:
    """Delete every key whose keyName == name. Returns count deleted."""
    n = 0
    for k in list_keys(auth):
        if k["keyName"] == name:
            delete_key(auth, k["applicationKeyId"])
            n += 1
    return n


def mint(name: str, hours: float = 48, caps: str = DEFAULT_CAPS,
         prefix: str | None = None, bucket: str | None = None) -> tuple[str, str]:
    """Mint a bucket-restricted ephemeral key; returns (keyId, applicationKey).
    Revokes same-name keys first so a relaunch always gets a fresh secret."""
    name = sanitize_name(name)
    hours = max(1.0, min(float(hours), MAX_HOURS))
    cap_list = [c.strip() for c in caps.split(",") if c.strip()]
    if "deleteFiles" in cap_list:
        raise MintError("refusing to mint a box key with deleteFiles "
                        "(the whole point is G1 no-destruction)")
    bucket = bucket or os.environ.get("B2_BUCKET")
    if not bucket:
        raise MintError("no bucket (pass --bucket or set B2_BUCKET)")
    auth = _minter_auth()
    revoked = revoke_by_name(auth, name)
    if revoked:
        print(f">> b2_mint_key: revoked {revoked} prior key(s) named {name}",
              file=sys.stderr)
    body = {
        "accountId": auth["accountId"],
        "capabilities": cap_list,
        "keyName": name,
        "validDurationInSeconds": int(hours * 3600),
        "bucketIds": [resolve_bucket_id(auth, bucket)],   # v4: list, not bucketId
    }
    if prefix:
        body["namePrefix"] = prefix
    r = _api(auth, "b2_create_key", body)
    return r["applicationKeyId"], r["applicationKey"]


def mint_pair(base_name: str, hours: float = 48, write_prefix: str = "",
              bucket: str | None = None
              ) -> tuple[tuple[str, str], tuple[str, str]]:
    """Mint a (read, write) key pair for a box that needs prefix-scoped writes.

      read  = bucket-wide READ_CAPS, name '<base>-ro', NO namePrefix — the box
              reads shared assets (base-models/, eval-env/, its own prefix, …).
      write = WRITE_CAPS restricted to namePrefix=<write_prefix>, name '<base>-rw'
              — the box can only create/overwrite objects under that prefix.

    Returns ((read_kid, read_key), (write_kid, write_key)). `write_prefix` is
    REQUIRED (that is the whole point); for a box that never writes, call
    `mint(base, caps=READ_CAPS)` and ship a single read-only key instead. Each
    half is revoke-then-mint on its own '<base>-ro'/'<base>-rw' name, so a
    relaunch rotates both. `_revoke_box_keys` prefix-matches '<base>-' to tear
    both down (tools/vast/herdd.py)."""
    if not write_prefix:
        raise MintError("mint_pair needs write_prefix (use mint() for RO-only)")
    read = mint(f"{base_name}-ro", hours=hours, caps=READ_CAPS, bucket=bucket)
    write = mint(f"{base_name}-rw", hours=hours, caps=WRITE_CAPS,
                 prefix=write_prefix, bucket=bucket)
    return read, write


def publish_prefix(env=None) -> str | None:
    """The namePrefix of a box's PUBLISH key, or None when the grant is switched
    off. Default PUBLISH_PREFIX ('checkpoints/'); override or disable with
    B2_PUBLISH_PREFIX (empty / '0' / 'none' / 'off' / 'false' -> no publish key).
    Pure w.r.t. the mapping passed in (defaults to os.environ)."""
    env = os.environ if env is None else env
    v = (env.get("B2_PUBLISH_PREFIX", PUBLISH_PREFIX) or "").strip()
    if not v or v.lower() in ("0", "none", "off", "false"):
        return None
    if v.startswith("/") or ".." in v.split("/"):
        raise MintError(f"B2_PUBLISH_PREFIX {v!r} must be a relative prefix")
    return v if v.endswith("/") else v + "/"


def mint_publish(base_name: str, hours: float = 48, prefix: str | None = None,
                 bucket: str | None = None, env=None):
    """Mint the box's PUBLISH write key: '<base>-pub', WRITE_CAPS, restricted to
    ONE namePrefix (default 'checkpoints/'). Returns (keyId, applicationKey), or
    None when the grant is disabled. Never carries deleteFiles (mint() refuses
    it) and self-expires on the same TTL as the rest of the box's keys — this is
    a THIRD narrowly-scoped ephemeral key, never a widening of an existing one."""
    p = prefix or publish_prefix(env)
    if not p:
        return None
    return mint(f"{base_name}-pub", hours=hours, caps=WRITE_CAPS, prefix=p,
                bucket=bucket)


def _fmt_exp(ts_ms) -> str:
    if not ts_ms:
        return "never"
    left = ts_ms / 1000 - time.time()
    when = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime(ts_ms / 1000))
    return f"{when} ({left / 3600:+.1f}h)"


# ---------------------------------------------------------------- CLI verbs #
def cmd_mint(a) -> None:
    name = a.name or (f"run-{a.run}" if a.run else None)
    if not name:
        sys.exit("error: --name or --run required")
    kid, key = mint(name, hours=a.hours, caps=a.caps, prefix=a.prefix,
                    bucket=a.bucket)
    if a.json:
        print(json.dumps({"keyId": kid, "applicationKey": key,
                          "name": sanitize_name(name), "hours": a.hours}))
    else:
        vp = a.var_prefix
        print(f"export {vp}KEY_ID={kid}")
        print(f"export {vp}APPLICATION_KEY={key}")
    print(f">> minted ephemeral B2 key {sanitize_name(name)!r} "
          f"(ttl {a.hours}h, caps {a.caps}, no deleteFiles)", file=sys.stderr)


def cmd_mint_pair(a) -> None:
    (rkid, rkey), (wkid, wkey) = mint_pair(
        a.name, hours=a.hours, write_prefix=a.write_prefix, bucket=a.bucket)
    if a.json:
        print(json.dumps({"read": {"keyId": rkid, "applicationKey": rkey},
                          "write": {"keyId": wkid, "applicationKey": wkey},
                          "name": sanitize_name(a.name),
                          "writePrefix": a.write_prefix, "hours": a.hours}))
    else:
        print(f"export B2_KEY_ID={rkid}")
        print(f"export B2_APPLICATION_KEY={rkey}")
        print(f"export B2_WRITE_KEY_ID={wkid}")
        print(f"export B2_WRITE_APPLICATION_KEY={wkey}")
    print(f">> minted ephemeral B2 pair {sanitize_name(a.name)!r} "
          f"(-ro bucket-wide read, -rw prefix={a.write_prefix}, "
          f"ttl {a.hours}h, no deleteFiles)", file=sys.stderr)


def cmd_revoke(a) -> None:
    name = sanitize_name(a.name or (f"run-{a.run}" if a.run else ""))
    n = revoke_by_name(_minter_auth(), name)
    print(f"revoked {n} key(s) named {name}")


def cmd_ls(a) -> None:
    for k in list_keys(_minter_auth()):
        eph = k["keyName"].startswith(EPHEMERAL_PREFIXES)
        if not a.all and not eph:
            continue
        print(f"{k['applicationKeyId']}  {k['keyName']:<24} "
              f"exp={_fmt_exp(k.get('expirationTimestamp')):<28} "
              f"caps={','.join(k['capabilities'])}")


def cmd_gc(a) -> None:
    auth = _minter_auth()
    now_ms = time.time() * 1000
    n = 0
    for k in list_keys(auth):
        exp = k.get("expirationTimestamp")
        if (k["keyName"].startswith(EPHEMERAL_PREFIXES) and exp
                and exp < now_ms):
            delete_key(auth, k["applicationKeyId"])
            print(f"gc: deleted expired {k['keyName']} ({k['applicationKeyId']})")
            n += 1
    print(f"gc: {n} expired ephemeral key(s) removed")


def cmd_whoami(a) -> None:
    pair = {"box": ("B2_BOX_KEY_ID", "B2_BOX_APPLICATION_KEY"),
            "ops": ("B2_KEY_ID", "B2_APPLICATION_KEY"),
            "minter": ("B2_MINTER_KEY_ID", "B2_MINTER_APPLICATION_KEY")}[a.pair]
    kid, key = os.environ.get(pair[0]), os.environ.get(pair[1])
    if not (kid and key):
        sys.exit(f"error: {pair[0]}/{pair[1]} unset")
    r = authorize(kid, key)
    al = r["allowed"]
    buckets = ",".join(b["name"] for b in al["buckets"]) if al.get("buckets") \
        else "<account-wide>"
    print(f"keyId       : {kid}")
    print(f"capabilities: {','.join(al['capabilities'])}")
    print(f"buckets     : {buckets}")
    print(f"namePrefix  : {al.get('namePrefix') or '<none>'}")
    print(f"expires     : {_fmt_exp(r.get('expiration'))}")


def main(argv=None) -> None:
    load_env()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mint", help="mint an ephemeral scoped key")
    m.add_argument("--name"); m.add_argument("--run")
    m.add_argument("--hours", type=float, default=48)
    m.add_argument("--caps", default=DEFAULT_CAPS)
    m.add_argument("--prefix", help="restrict to one namePrefix (Option 1b)")
    m.add_argument("--bucket")
    m.add_argument("--var-prefix", default="B2_",
                   help="env-var prefix for the export lines (default B2_)")
    m.add_argument("--json", action="store_true")
    m.set_defaults(func=cmd_mint)

    mp = sub.add_parser("mint-pair",
                        help="mint an Option-1b read+write key pair")
    mp.add_argument("--name", required=True, help="base key name (-ro/-rw appended)")
    mp.add_argument("--write-prefix", required=True,
                    help="namePrefix the -rw key is restricted to (e.g. serve/)")
    mp.add_argument("--hours", type=float, default=48)
    mp.add_argument("--bucket")
    mp.add_argument("--json", action="store_true")
    mp.set_defaults(func=cmd_mint_pair)

    r = sub.add_parser("revoke", help="delete key(s) by name")
    r.add_argument("--name"); r.add_argument("--run")
    r.set_defaults(func=cmd_revoke)

    l = sub.add_parser("ls", help="list ephemeral keys (+expiry)")
    l.add_argument("--all", action="store_true", help="include standing keys")
    l.set_defaults(func=cmd_ls)

    g = sub.add_parser("gc", help="delete EXPIRED ephemeral keys")
    g.set_defaults(func=cmd_gc)

    w = sub.add_parser("whoami", help="print an .env pair's capabilities")
    w.add_argument("--pair", choices=("box", "ops", "minter"), default="box")
    w.set_defaults(func=cmd_whoami)

    a = p.parse_args(argv)
    try:
        a.func(a)
    except MintError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
