#!/usr/bin/env python3
"""onstart/cred_client.py — box-side credential client (cred-broker C3).

Fetches fresh B2 creds from the workstation cred broker and installs them
locally, per docs/plans/cred-broker-buildout.md §2.5 (decision order) + §2.4
(B2-mediated lane). SELF-CONTAINED by design: boxes receive only onstart/
files, so the §2.4 envelope + proof constructions are byte-compatible COPIES
of tools/vast/credbroker.py (C3's tests cross-round-trip the two — edit them
in lockstep):

  nonce_bytes            = nonce.encode("utf-8")   (the hex string as-is)
  proof                  = HMAC-SHA256(nonce_bytes,
                           b"credreq|<iid>|<ts>|<role>").hexdigest()
  enc_key / mac_key      = SHA256(nonce_bytes + b"enc"/b"mac").digest()
  keystream block i      = HMAC-SHA256(enc_key, b"<ts>|<i>")   (i decimal, 0-based)
  ciphertext             = keystream XOR canonical-JSON plaintext
  mac                    = HMAC-SHA256(mac_key, b"<ts>|" + ciphertext).hexdigest()

Decision order (§2.5): 1) direct POST $CRED_BROKER_URL/v1/creds (10 s);
2) on failure with TS_AUTHKEY set: tailnet_join.sh (same dir), retry through
curl --socks5-hostname localhost:1055 (curl ONLY for the SOCKS path);
3) role=jobs: B2-mediated credreq/creds under jobs/nodes/<IID>/ via rclone.

Install is VERIFY-THEN-SWAP: the new [b2]/[b2w] rclone remotes are written to
a temp config, probed with `rclone --config <temp> lsf`, and only then
os.replace()d over the live config + jobd.env (0600, exact jobd_boot.sh
formats). Any failure exits nonzero with the working config untouched.
Progress goes to stderr as structured lines; key material is NEVER printed.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import subprocess
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
RCLONE_CONF = os.path.expanduser("~/.config/rclone/rclone.conf")
ROLE_PROBE_PREFIX = {"jobs": "jobs/", "serve": "serve/"}
# B2-lane freshness: the broker seals the response ts from ITS clock, so a
# box clock ahead of the broker's would misread every genuine response as
# "stale prior cycle". Vast containers are not NTP-tight — tolerate this much
# box-ahead skew (broker-side replay bound is 600 s, so this stays well under).
CLOCK_SKEW_S = 120


def _log(msg):
    sys.stderr.write(">> [cred-client] %s %s\n"
                     % (time.strftime("%FT%TZ", time.gmtime()), msg))
    sys.stderr.flush()


# ------------------------------------- envelope (COPY of credbroker.py §2.4) #
class EnvelopeError(ValueError):
    pass


def _nonce_keys(nonce):
    nb = str(nonce).encode()
    return (hashlib.sha256(nb + b"enc").digest(),
            hashlib.sha256(nb + b"mac").digest())


def _keystream(enc_key, ts, n) -> bytes:
    out = b""
    i = 0
    while len(out) < n:
        out += hmac.new(enc_key, f"{ts}|{i}".encode(), hashlib.sha256).digest()
        i += 1
    return out[:n]


def seal_envelope(nonce, obj, ts=None) -> dict:
    """§2.4 envelope, byte-identical to credbroker.seal_envelope."""
    ts = int(time.time() if ts is None else ts)
    enc_key, mac_key = _nonce_keys(nonce)
    pt = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ct = bytes(a ^ b for a, b in zip(pt, _keystream(enc_key, ts, len(pt))))
    mac = hmac.new(mac_key, f"{ts}|".encode() + ct, hashlib.sha256).hexdigest()
    return {"ts": ts, "ciphertext": ct.hex(), "mac": mac}


def open_envelope(nonce, blob):
    """Constant-time MAC check, then decrypt (MAC-then-decrypt — any jobs box
    can shadow-write the response object). Raises EnvelopeError otherwise."""
    try:
        ts = int(blob["ts"])
        ct = bytes.fromhex(blob["ciphertext"])
        mac = str(blob["mac"])
    except Exception:
        raise EnvelopeError("malformed envelope")
    enc_key, mac_key = _nonce_keys(nonce)
    want = hmac.new(mac_key, f"{ts}|".encode() + ct, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(want.encode(), mac.encode()):
        raise EnvelopeError("bad mac")
    pt = bytes(a ^ b for a, b in zip(ct, _keystream(enc_key, ts, len(ct))))
    try:
        return json.loads(pt.decode())
    except Exception:
        raise EnvelopeError("bad plaintext")


def make_credreq_proof(nonce, instance_id, ts, role) -> str:
    """HMAC proof for jobs/nodes/<IID>/credreq — the raw nonce NEVER goes on
    the bucket; only this MAC does. COPY of credbroker.make_credreq_proof."""
    msg = f"credreq|{instance_id}|{int(ts)}|{role}".encode()
    return hmac.new(str(nonce).encode(), msg, hashlib.sha256).hexdigest()


# -------------------------------------------------------------- transports #
def _instance_id(env=None):
    """JOBD_IID > INSTANCE_ID > CONTAINER_ID (same chain as jobd.sh:53)."""
    env = os.environ if env is None else env
    for k in ("JOBD_IID", "INSTANCE_ID", "CONTAINER_ID"):
        v = (env.get(k) or "").strip()
        if v:
            try:
                return int(v)
            except ValueError:
                continue
    return None


def _direct_fetch(url, body, timeout=10):
    """§2.5 step 1: plain urllib POST /v1/creds — works once on the tailnet."""
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/creds", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _tailnet_join():
    """Run tailnet_join.sh from this file's dir (idempotent; exits 0 if up)."""
    script = os.path.join(_HERE, "tailnet_join.sh")
    subprocess.run(["bash", script], check=True, stdout=sys.stderr,
                   timeout=300)


def _socks_fetch(url, body, timeout=15):
    """§2.5 step 2 retry: curl through tailscaled's SOCKS5 (urllib has no
    SOCKS support — curl is used ONLY here). Body via stdin, never argv."""
    p = subprocess.run(
        ["curl", "-fsS", "--max-time", str(int(timeout)),
         "--socks5-hostname", "localhost:1055",
         "-H", "Content-Type: application/json",
         "--data-binary", "@-", url.rstrip("/") + "/v1/creds"],
        input=json.dumps(body).encode(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("curl socks fetch failed (exit %d)" % p.returncode)
    return json.loads(p.stdout.decode())


def _b2_lane(iid, nonce, role, bucket, deadline_s=300, poll_s=10,
             run=subprocess.run):
    """§2.4 box side: write jobs/nodes/<IID>/credreq (proof, no raw nonce),
    poll .../creds <= 5 min, MAC-verify + decrypt. Writes go through [b2w]
    when the box has the scoped pair, else [b2] (jobd.sh:56-61 pattern)."""
    ts = int(time.time())
    req = {"instance_id": iid, "ts": ts, "role": role,
           "want": {"write_prefix": None},
           "proof": make_credreq_proof(nonce, iid, ts, role)}
    lr = run(["rclone", "listremotes"], stdout=subprocess.PIPE,
             stderr=subprocess.DEVNULL, timeout=60)
    remotes = lr.stdout.decode().split() if lr.returncode == 0 else []
    wr = "b2w" if "b2w:" in remotes else "b2"
    p = run(["rclone", "rcat", "%s:%s/jobs/nodes/%d/credreq" % (wr, bucket, iid)],
            input=json.dumps(req).encode(), stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=120)
    if p.returncode != 0:
        raise RuntimeError("credreq write via [%s] failed" % wr)
    _log("credreq posted (ts=%d); polling jobs/nodes/%d/creds" % (ts, iid))
    end = time.time() + deadline_s
    stale_logged = False
    while True:
        c = run(["rclone", "cat", "b2:%s/jobs/nodes/%d/creds" % (bucket, iid)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=120)
        if c.returncode == 0 and c.stdout.strip():
            try:
                blob = json.loads(c.stdout.decode())
                bts = int(blob.get("ts") or 0)
                # Freshness with skew tolerance: only drop clearly-prior
                # cycles; the MAC (keyed on ts) still binds the envelope.
                if bts >= ts - CLOCK_SKEW_S:
                    return open_envelope(nonce, blob)
                if not stale_logged:
                    stale_logged = True
                    _log("ignoring stale creds blob (ts=%d < req ts=%d - %ds)"
                         % (bts, ts, CLOCK_SKEW_S))
            except (ValueError, EnvelopeError, TypeError):
                pass   # garbage / shadow-write — keep polling
        if time.time() >= end:
            break
        time.sleep(poll_s)
    raise RuntimeError("timed out waiting for creds response")


def fetch_creds(cfg, direct=_direct_fetch, join=_tailnet_join,
                socks=_socks_fetch, b2lane=_b2_lane):
    """§2.5 decision order. cfg: {iid, nonce, role, broker_url, ts_authkey,
    bucket}. Transports are injectable (tests). Raises when all lanes fail."""
    body = {"instance_id": cfg["iid"], "nonce": cfg["nonce"],
            "role": cfg["role"], "want": {"write_prefix": None}}
    url = cfg.get("broker_url") or ""
    if url:
        try:
            _log("direct POST %s/v1/creds" % url.rstrip("/"))
            return direct(url, body)
        except Exception as e:
            _log("direct fetch failed: %s" % type(e).__name__)
        if cfg.get("ts_authkey"):
            try:
                _log("joining tailnet via tailnet_join.sh")
                join()
                _log("retrying through SOCKS5 localhost:1055")
                return socks(url, body)
            except Exception as e:
                _log("tailnet lane failed: %s" % type(e).__name__)
    if cfg.get("role") == "jobs" and cfg.get("bucket"):
        _log("falling back to B2-mediated lane")
        return b2lane(cfg["iid"], cfg["nonce"], cfg["role"], cfg["bucket"])
    raise RuntimeError("all credential transports failed")


# ------------------------------------------- install (verify-then-swap) #
def _remote_section(name, kid, secret, endpoint, region):
    """EXACT byte format of jobd_boot.sh:67-97's [b2]/[b2w] heredocs."""
    return ("[%s]\n"
            "type = s3\n"
            "provider = Other\n"
            "access_key_id = %s\n"
            "secret_access_key = %s\n"
            "endpoint = %s\n"
            "region = %s\n"
            "acl = private\n"
            "no_check_bucket = true\n") % (name, kid, secret, endpoint, region)


def _strip_sections(text, names):
    """Drop the named INI sections (header through next header) from text."""
    out, skip = [], False
    for line in text.splitlines(True):
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            skip = s[1:-1] in names
        if not skip:
            out.append(line)
    return "".join(out)


def build_rclone_conf(existing_text, creds):
    """New rclone.conf text: everything else preserved, [b2]/[b2w] replaced
    in jobd_boot.sh byte format ([b2w] only when a write pair was issued)."""
    b2 = creds["b2"]
    region = creds.get("region") or "us-west-004"
    txt = _strip_sections(existing_text or "", {"b2", "b2w"})
    if txt and not txt.endswith("\n"):
        txt += "\n"
    txt += _remote_section("b2", b2["key_id"], b2["application_key"],
                           creds["s3_endpoint"], region)
    if b2.get("write_key_id"):
        txt += _remote_section("b2w", b2["write_key_id"],
                               b2["write_application_key"],
                               creds["s3_endpoint"], region)
    return txt


_MANAGED_ENV = ("B2_BUCKET", "B2_KEY_ID", "B2_APPLICATION_KEY",
                "B2_S3_ENDPOINT", "B2_REGION", "B2_WRITE_KEY_ID",
                "B2_WRITE_APPLICATION_KEY", "B2_KEY_EXPIRES_AT")


def build_jobd_env(existing_text, creds):
    """jobd.env text: managed B2_* exports first (jobd_boot.sh:135-149 order
    + format), then every non-managed line preserved (INSTANCE_ID, JOBD_*)."""
    b2 = creds["b2"]
    lines = ["export B2_BUCKET=%s" % creds["bucket"],
             "export B2_KEY_ID=%s" % b2["key_id"],
             "export B2_APPLICATION_KEY=%s" % b2["application_key"],
             "export B2_S3_ENDPOINT=%s" % creds["s3_endpoint"],
             "export B2_REGION=%s" % (creds.get("region") or "us-west-004")]
    if b2.get("write_key_id"):
        lines.append("export B2_WRITE_KEY_ID=%s" % b2["write_key_id"])
        lines.append("export B2_WRITE_APPLICATION_KEY=%s"
                     % b2["write_application_key"])
    lines.append("export B2_KEY_EXPIRES_AT=%d" % int(creds["expires_at"]))
    for line in (existing_text or "").splitlines():
        s = line.strip()
        if s.startswith("export ") and \
                s[len("export "):].split("=", 1)[0] in _MANAGED_ENV:
            continue
        if s:
            lines.append(line)
    return "\n".join(lines) + "\n"


def _write_private(path, text):
    """Write <path>.credtmp with 0600 from birth; caller os.replace()s it."""
    tmp = path + ".credtmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return tmp


def _probe_conf(conf_path, creds, role, run=subprocess.run):
    """`rclone lsf` against the TEMP config — the §2.5 gate before any swap.
    Probes [b2] at the bucket root; [b2w] (scoped key: list works only under
    its namePrefix) at the role prefix. True only if every issued key works."""
    bucket = creds["bucket"]
    p = run(["rclone", "--config", conf_path, "lsf", "--max-depth", "1",
             "b2:%s" % bucket], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=60)
    if p.returncode != 0:
        return False
    if creds["b2"].get("write_key_id"):
        prefix = ROLE_PROBE_PREFIX.get(role, "")
        p = run(["rclone", "--config", conf_path, "lsf", "--max-depth", "1",
                 "b2w:%s/%s" % (bucket, prefix)], stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=60)
        if p.returncode != 0:
            return False
    return True


def apply_creds(creds, role, conf_path=None, jobd_env_path=None,
                probe=_probe_conf):
    """Verify-then-swap install, transactional across BOTH files: the new
    rclone config AND the new jobd.env are fully built up front (so a
    malformed response — e.g. bad expires_at — fails before anything moves),
    written to temps, the temp config probed (--config <temp>), and only then
    are BOTH os.replace()d over the live files back-to-back. Any earlier
    failure leaves rclone.conf and jobd.env untouched AND in sync; temps
    (0600, secret-bearing) are always cleaned up."""
    # RCLONE_CONFIG is read here and not at import: rclone honours it, so a
    # swap onto the hardcoded $HOME path would leave every later call elsewhere.
    conf_path = (conf_path or os.environ.get("CRED_CLIENT_RCLONE_CONF")
                 or os.environ.get("RCLONE_CONFIG") or RCLONE_CONF)
    jobd_dir = (os.environ.get("JOBD_BOOT_DIR") or os.environ.get("JOBD_DIR")
                or "/workspace/jobd")
    jobd_env_path = jobd_env_path or os.path.join(jobd_dir, "jobd.env")
    if not creds.get("bucket"):
        creds = dict(creds, bucket=os.environ.get("B2_BUCKET") or "")
    if not creds["bucket"]:
        raise RuntimeError("no bucket in response or B2_BUCKET env")
    existing = ""
    if os.path.isfile(conf_path):
        with open(conf_path) as f:
            existing = f.read()
    old_env = ""
    if os.path.isfile(jobd_env_path):
        with open(jobd_env_path) as f:
            old_env = f.read()
    # Build BOTH texts before touching anything — build_jobd_env parses
    # expires_at, so an incomplete broker response dies here, not mid-swap.
    conf_text = build_rclone_conf(existing, creds)
    env_text = build_jobd_env(old_env, creds)
    expires_at = int(creds["expires_at"])
    os.makedirs(os.path.dirname(conf_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(jobd_env_path) or ".", exist_ok=True)
    tmp = env_tmp = None
    try:
        tmp = _write_private(conf_path, conf_text)
        _log("probing new key(s) via rclone lsf (temp config)")
        if not probe(tmp, creds, role):
            raise RuntimeError(
                "new-key rclone probe FAILED — existing config untouched")
        env_tmp = _write_private(jobd_env_path, env_text)
        os.replace(tmp, conf_path)
        os.replace(env_tmp, jobd_env_path)
    finally:
        for t in (tmp, env_tmp):     # no-ops after a successful os.replace
            if t and os.path.exists(t):
                try:
                    os.unlink(t)
                except OSError:
                    pass
    os.environ["B2_KEY_EXPIRES_AT"] = str(expires_at)
    _log("installed [b2]%s + jobd.env; expires_at=%d"
         % ("/[b2w]" if creds["b2"].get("write_key_id") else "", expires_at))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="fetch fresh B2 creds from the cred broker and install "
                    "them (rclone remotes + jobd.env), verify-then-swap")
    ap.add_argument("--rclone-conf", help="rclone config path override")
    ap.add_argument("--jobd-env", help="jobd.env path override")
    args = ap.parse_args(argv)
    nonce = os.environ.get("BOX_IDENTITY_NONCE") or ""
    if not nonce:
        _log("!! BOX_IDENTITY_NONCE not set — cannot authenticate to broker")
        return 2
    iid = _instance_id()
    if iid is None:
        _log("!! no instance id (JOBD_IID/INSTANCE_ID/CONTAINER_ID)")
        return 2
    cfg = {"iid": iid, "nonce": nonce,
           "role": os.environ.get("CRED_ROLE") or "jobs",
           "broker_url": os.environ.get("CRED_BROKER_URL") or "",
           "ts_authkey": os.environ.get("TS_AUTHKEY") or "",
           "bucket": os.environ.get("B2_BUCKET") or ""}
    try:
        creds = fetch_creds(cfg)
    except Exception as e:
        _log("!! credential fetch failed: %s: %s" % (type(e).__name__, e))
        return 3
    try:
        apply_creds(creds, cfg["role"], args.rclone_conf, args.jobd_env)
    except Exception as e:
        _log("!! install failed (working config untouched): %s" % e)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
