"""vastlib.launch.spec — the launch CONTRACT: what gets recorded, and the
credentials a box is handed.

Why this exists
---------------
Two things happen just before a box is created, and both are one-way doors:

* **The spec is frozen.** `_build_launch_spec` returns the
  `runs/<RUN_ID>/spec.json` v=1 body (SPOT_DESIGN §3.1) — the declarative
  record a relaunch replays byte-for-byte. That object is DURABLE and never
  deleted, so a secret VALUE that reaches it is a secret that lives forever in
  B2. `_split_env_secrets` is the one place that decision is made, and it is
  deliberately conservative: name family OR credential-shaped value, values
  dropped, names kept.
* **Keys are minted.** `_ship_b2_pair` / `_ship_b2_env` hand a rented box its
  B2 credentials, preferring a freshly minted, bucket-restricted, no-delete,
  self-expiring key over any standing one (`docs/plans/keyless-b2-ingest.md`).
  The mint caches below hold LIVE KEY MATERIAL for the process lifetime.

Both halves are separated from `launch.py` for the same reason: the spec and
the credential decision are testable without a network, and the five-phase
create sequence is not.

THE `_MINTED_*` CACHES ARE A SECOND, INDEPENDENT SET — READ THIS
----------------------------------------------------------------
Plan §8 steps 2-5 are ADD-ONLY: `herdd.py` keeps its own
`_MINTED_PAIRS` / `_MINTED_SCOPED` / `_MINTED_PUBLISH` / `_MINT_ANNOUNCED`
until step 6, and the four dicts below are **different objects in a different
module**. Nothing synchronises them. Consequences, all of which have a way of
looking like a passing test:

* A test that drives `vastlib.launch.spec._ship_b2_env` and then asserts on
  `herdd._MINTED_SCOPED` is asserting the WRONG DICT — it will read `{}` and
  pass or fail for reasons unrelated to the code under test. Assert on the
  ledger belonging to the module you called.
* `test_broker_env.py`'s autouse `_clean_seams` fixture clears the `herdd`
  copies only. A test in this file that mints must clear these itself
  (`test_vastlib_launch.py` has its own autouse fixture doing exactly that),
  or process-lifetime cache state leaks between tests.
* Two live caches means a key minted through one path is invisible to the
  other, so `_minted_expiry` on this module can answer `None` for a mint that
  `herdd` performed, and vice versa. During the add-only phase every launch
  still runs through `herdd`, so the FLEET only ever uses one of them; this
  copy is exercised by tests until step 6 rewires the CLI.

Independence is pinned, not assumed:
`test_vastlib_launch.py::test_mint_ledgers_are_independent_of_the_herdd_copies`.

What is deliberately NOT here
-----------------------------
* **`MintLedger` is not this change.** Plan §5 designs the four globals into an
  instance owned by the launch context. That is a TEST-BREAKING refactor, not a
  rename: `test_broker_env.py` mutates the module attributes directly at nine
  sites, including an autouse fixture that `.clear()`s two of them, and an
  instance removes the attribute they bind to. Ported as module globals first
  (behavior identical, plan §8 step-3 add-only rule); the ledger object is a
  separate, test-repointing change.
* **No revoke.** `_revoke_box_keys` is the teardown half of the same lifecycle
  and lives in `boxes/lifecycle.py`, next to `_destroy_and_revoke` and
  `cmd_destroy` — the callers that fire it (plan §5 "destroy+revoke";
  `test_label_grammar.py` already names that home). It reads none of the caches
  below, so the split costs nothing.
* **No broker.** `credbroker*` / `cred_client` are box-side and explicitly NOT
  absorbed (plan §3). This module ships credentials AT LAUNCH; refreshing them
  on a live box is the broker's job.
* **No offer, no price, no body.** Everything about choosing and creating the
  instance is `launch/launch.py`.
* **No `_capture_launch_spec` / `_read_onstart` / `_init_state`.** They sit in
  the same textual block in `herdd.py` under a "supervise" banner and are
  supervise-side (plan §8 step 4). They are here in neither code nor spirit —
  the block was cut by NAME, not by line range.

Provenance: verbatim-with-types move from `tools/vast/herdd.py`, plan §8
step 3 (`launch/`) of `docs/plans/vast-tooling-refactor-v2.md`. Every symbol
carries its `# moved-from:` marker. Step 3 is ADD-ONLY, so `herdd.py` keeps
its own copies until step 6 and both are live meanwhile.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from typing import Any, Callable, Iterable, Sequence

import b2_mint_key
import imageref

from vastlib.storage import b2

import runmeta

# --------------------------------------------------------------------------- #
# The secret split — what may and may not land in a durable B2 object.
# --------------------------------------------------------------------------- #
# An env var holds a secret iff its NAME matches a secret family (case-insensitive,
# a superset of what _do_launch redacts on) OR its VALUE embeds a URL credential
# (scheme://user:pass@host). Either way its VALUE must never land in the durable,
# never-deleted B2 spec (invariant §5.8) — only the NAME does. The name families
# are broad on purpose: an oddly-named --env passthrough (DATABASE_URL, DOCKER_AUTH,
# lowercase hf_token, …) must not leak its value into a permanent B2 object.
# moved-from: herdd._SECRET_ENV_RE
# B2_CDN_PREFIX is here by NAME because nothing about its shape says "secret":
# it is a 144-bit random path segment, so it reads as an ordinary opaque value
# while being a URL BEARER credential — anyone holding it can read the mirror.
# Without this line it would land verbatim in the never-deleted runs/<id>/spec.json
# and in `launch --dry-run` stdout.
_SECRET_ENV_RE = re.compile(
    r"TOKEN|KEY|SECRET|PASS|PWD|CRED|AUTH|PRIVATE|SIGNATURE|SESSION"
    r"|CDN_PREFIX", re.I)
# moved-from: herdd._SECRET_VAL_RE
_SECRET_VAL_RE = re.compile(r"://[^/\s:@]+:[^/\s@]+@")  # scheme://user:pass@host


# moved-from: herdd._is_secret_env
def _is_secret_env(k: str | None, v: str | None) -> bool:
    """True if KEY=VALUE must be treated as a secret (name family or credential-
    shaped value). Conservative: when in doubt the value is withheld from B2.

    ALSO the dry-run redactor. `launch.py`'s `_do_launch` calls it as
    `spec._is_secret_env(...)` — by MODULE ATTRIBUTE, never `from … import` —
    because it is the only thing keeping token values out of `launch --dry-run`
    stdout, and a stale binding there fails OPEN (it prints the secret) rather
    than failing loudly. `test_vastlib_launch.py` pins that with a fake secret
    driven through the vastlib dry-run path.
    """
    return bool(_SECRET_ENV_RE.search(k or "") or _SECRET_VAL_RE.search(v or ""))


# moved-from: herdd._split_env_secrets
def _split_env_secrets(
    env_list: Sequence[str] | None,
) -> tuple[dict[str, str], list[str]]:
    """Split assembled KEY=VALUE launch-env strings into a NON-secret env dict and
    a list of secret NAMES (values dropped). Secret VALUES never leave here — the
    spec carries names only; supervise re-injects values from the local env."""
    # TYPING-FORCED SPLIT of the verbatim `env, secret_keys = {}, []` — a
    # tuple-unpacking assignment cannot carry two annotations. Same values.
    env: dict[str, str] = {}
    secret_keys: list[str] = []
    for kv in env_list or []:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        if _is_secret_env(k, v):
            if k not in secret_keys:
                secret_keys.append(k)
        else:
            env[k] = v
    return env, secret_keys


# moved-from: herdd._build_launch_spec
def _build_launch_spec(*, run_id: object, runset: object, image: object,
                       image_login_ref: object, disk: object, runtype: object,
                       gpu: Iterable[object] | None, gpu_ram: object,
                       num_gpus: object, env_list: Sequence[str] | None,
                       onstart: str,
                       orig_bid: object, max_bid: object,
                       defend_at: object = None, rescue_wait_s: object = None,
                       cuda: object = None,
                       image_digest: object = None) -> dict[str, Any]:
    """The declarative runs/<RUN_ID>/spec.json body (SPOT_DESIGN §3.1) — the
    launch contract a relaunch reproduces byte-for-byte. Secrets NEVER land here:
    only secret_env_keys (names). `image_login_ref` is the token-redacted docker
    login string (presence marker only); the real login is re-derived at relaunch
    from the local signing secret the image's registry needs
    (REGISTRY_AUTH_SECRET — the R2 Worker registry is the only one left).
    onstart ships verbatim as base64 so a relaunch
    boots the exact same script even if the local checkout moved on mid-run.
    defend_at/rescue_wait_s (SPOT_DESIGN §3.4, both optional) are the runset's
    spot: policy at launch time — a durable record alongside the bid cap; not
    read back by supervise today (cmd_train passes them explicitly on the
    --supervise handoff instead).

    `image_digest` is what makes "byte-for-byte" above true rather than
    aspirational. `image` is almost always a MUTABLE tag (`train-vast-latest`),
    so replaying it after an env push lands the run's second half on a
    DIFFERENT env than its first — silently, with nothing in the spec able to
    detect it. Recording the launch-time content digest lets _relaunch_body
    compare and pin (velvet P4a). Best-effort: None for an image on a registry
    we cannot resolve, which degrades to exactly today's tag replay.

    FROZEN CONTRACT (plan §4, "B2 event schemas"). The returned dict IS the
    durable object: key names, the key SET, the nesting of `bid`, and the
    base64 encoding of `onstart` are read back by `_read_spec_soft` and by
    supervise's `_capture_launch_spec`. A typed wrapper that drops or reorders
    unknown keys breaks a relaunch, so this stays a plain dict.
    """
    env, secret_keys = _split_env_secrets(env_list)
    return {
        "v": 1, "run_id": run_id, "runset": runset,
        "image": image, "image_login": image_login_ref,
        "image_digest": image_digest,
        "disk": disk, "runtype": runtype,
        "gpu": list(gpu or []), "gpu_ram": gpu_ram, "num_gpus": num_gpus,
        "cuda": cuda,
        "env": env, "secret_env_keys": secret_keys,
        "bid": {"orig": orig_bid, "max": max_bid,
                "defend_at": defend_at, "rescue_wait_s": rescue_wait_s},
        "onstart_b64": base64.b64encode(onstart.encode("utf-8")).decode("ascii"),
    }


# --------------------------------------------------------------------------- #
# The mint caches. LIVE SECRET MATERIAL — see the module docstring for why they
# are module globals and not a MintLedger, and for the second-set hazard.
# Nothing here (or anywhere) may print a VALUE out of them: the four existing
# prints below emit the key NAME, its ttl and its prefix, and that is the
# whole permitted surface.
# --------------------------------------------------------------------------- #
# moved-from: herdd._MINTED_PAIRS
_MINTED_PAIRS: dict[str, tuple[str, str]] = {}   # key name -> (kid, key); one mint per process
# moved-from: herdd._MINTED_SCOPED
_MINTED_SCOPED: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {}
#                                    # base name -> (read_pair, write_pair); scoped mints
# moved-from: herdd._MINTED_PUBLISH
_MINTED_PUBLISH: dict[str, tuple[str, str]] = {}  # base name -> (kid, key); checkpoints/ grant
# moved-from: herdd._MINT_ANNOUNCED
_MINT_ANNOUNCED: set[str] = set()  # dry-run "would mint" lines already printed


# moved-from: herdd._b2_eu_pairs
def _b2_eu_pairs() -> list[tuple[str, str]]:
    """EU read-replica vars to inject into a box for region-aware static-asset
    reads (tools/vast/B2_REGIONS.md). Returns [(K, V), ...] when the EU replica
    is configured in the workstation env, else []. Ships the STANDING READ-ONLY
    B2_*_EU key — boxes only ever READ the replica; writes always go to the US
    source (the region gate forces it, and the EU key is read-only anyway). No-op
    when EU is unconfigured, so default launches are unchanged. B2_REGION_MODE=us
    in the workstation env opts a launch out (forces US-only reads)."""
    kid = os.environ.get("B2_KEY_ID_EU"); key = os.environ.get("B2_APPLICATION_KEY_EU")  # noqa: E702 — verbatim body (plan §7.4)
    ep = os.environ.get("B2_S3_ENDPOINT_EU"); bkt = os.environ.get("B2_BUCKET_EU")  # noqa: E702 — verbatim body (plan §7.4)
    if not (kid and key and ep and bkt):
        return []
    return [
        ("B2_KEY_ID_EU", kid), ("B2_APPLICATION_KEY_EU", key),
        ("B2_S3_ENDPOINT_EU", ep), ("B2_BUCKET_EU", bkt),
        ("B2_REGION_EU", os.environ.get("B2_REGION_EU", "eu-central-003")),
        ("B2_REGION_MODE", os.environ.get("B2_REGION_MODE", "auto")),
    ]


# moved-from: herdd._r2_tc_pairs
def _r2_tc_pairs() -> list[tuple[str, str]]:
    """Shared-Triton-cache R2 vars for a box (tools/vast/triton_cache.py's
    `resolve_remote`). Returns [(K, V), ...] when the workstation env carries
    the bucket-scoped R2 token (R2_TC_*), else [] — in which case the box's
    triton cache falls back to B2 (its scoped key can read/write triton-cache/
    only if the bucket policy allows; in practice R2 is the intended home and
    B2 fallback simply reports cold). The token is scoped to the
    shared-triton-cache bucket alone: it cannot touch the registry bucket or
    any B2 data. It is a STANDING key (not per-launch minted); rotation is a
    CF-dashboard/API operation — see triton_cache.py's TRUST NOTE."""
    kid = os.environ.get("R2_TC_KEY_ID")
    sec = os.environ.get("R2_TC_SECRET_ACCESS_KEY")
    ep = os.environ.get("R2_TC_ENDPOINT")
    if not (kid and sec and ep):
        return []
    return [
        ("R2_TC_KEY_ID", kid), ("R2_TC_SECRET_ACCESS_KEY", sec),
        ("R2_TC_ENDPOINT", ep),
        ("R2_TC_BUCKET", os.environ.get("R2_TC_BUCKET", "shared-triton-cache")),
    ]


def _cdn_pairs() -> list[tuple[str, str]]:
    """CDN weights-mirror vars for a box (`onstart/b2x_boot.sh`'s rung-0 CDN tier
    in `b2x_pull`). Returns [(K, V), ...] when the workstation env carries all
    three, else [] — in which case a box's base-model pull is exactly today's
    b2x -> rclone ladder, because the tier refuses to engage without all three.
    ALL-OR-NOTHING on purpose: a partial set would make every base-model pull
    log a miss and then fall through anyway, which is cost with no benefit.

    B2_CDN_PREFIX is a URL-BEARER SECRET (a 144-bit anti-discovery path segment;
    the content behind it is public upstream weights, so this is anti-scanning,
    not confidentiality). It lives in .env and is classified secret by
    `_SECRET_ENV_RE` above, which is what keeps it out of the durable B2 spec and
    out of dry-run stdout. Host and bucket are not sensitive."""
    host = os.environ.get("B2_CDN_HOST")
    bucket = os.environ.get("B2_CDN_BUCKET")
    prefix = os.environ.get("B2_CDN_PREFIX")
    if not (host and bucket and prefix):
        return []
    return [("B2_CDN_HOST", host), ("B2_CDN_BUCKET", bucket),
            ("B2_CDN_PREFIX", prefix)]


# moved-from: herdd._ship_b2_pair
def _ship_b2_pair(name: str | None, hours: float,
                  dry_run: bool = False) -> tuple[str, str]:
    """The B2 pair to ship to a rented box (docs/plans/keyless-b2-ingest.md).
    Preference: (1) a fresh EPHEMERAL key minted with the workstation minter
    pair — bucket-restricted, listFiles/readFiles/writeFiles, NO deleteFiles,
    self-expiring after `hours`; (2) the standing B2_BOX_* no-delete pair;
    (3) the full-capability ops key with a loud warning. dry_run never mints
    (the read-only promise) and falls through to (2)/(3).

    KEEPS ITS `sys.exit`. A hard exit inside a library function is unusual and
    deliberate (plan §7.4: no behavior changes in a port) — `test_lifecycle.py`
    and `test_launch_eval_env_pin.py` assert on `SystemExit` message TEXT
    across this path's callers, and converting it to a raise is a behavior
    change with a test break behind it.
    """
    try:
        name = b2_mint_key.sanitize_name(name) if name else ""
    except b2_mint_key.MintError:
        name = ""
    minter = (os.environ.get("B2_MINTER_KEY_ID")
              and os.environ.get("B2_MINTER_APPLICATION_KEY"))
    if name and minter:
        if dry_run:
            if name not in _MINT_ANNOUNCED:
                _MINT_ANNOUNCED.add(name)
                print(f">> [dry-run] would mint ephemeral B2 key {name} "
                      f"(ttl {hours:g}h, no-delete) — standing pair used for the preview")
        elif name in _MINTED_PAIRS:
            return _MINTED_PAIRS[name]
        else:
            try:
                pair = b2_mint_key.mint(name, hours=hours)
                print(f">> minted ephemeral B2 key {name} (ttl {hours:g}h, "
                      f"no deleteFiles, self-expires)")
                _MINTED_PAIRS[name] = pair
                return pair
            except Exception as e:
                print(f">> WARNING: ephemeral B2 key mint failed ({e}) — "
                      f"falling back to the standing box key", file=sys.stderr)
    kid = os.environ.get("B2_BOX_KEY_ID")
    key = os.environ.get("B2_BOX_APPLICATION_KEY")
    if kid and key:
        return kid, key
    kid = os.environ.get("B2_KEY_ID")
    key = os.environ.get("B2_APPLICATION_KEY")
    if not (kid and key):
        sys.exit("error: no B2 credentials in env/.env (need B2_MINTER_*, "
                 "B2_BOX_*, or the B2_KEY_ID pair)")
    print(">> WARNING: no minter and no B2_BOX_* pair — shipping the "
          "full-capability ops key to a rented box "
          "(see docs/plans/keyless-b2-ingest.md)", file=sys.stderr)
    return kid, key


# moved-from: herdd._ephemeral_hours
def _ephemeral_hours(timeout_s: float | None = None) -> float:
    """TTL (hours) for an ephemeral box key. A high floor so a long park or an
    EXTENDED job never outlives its key (the sole suspend/outbid breakage
    vector), optionally scaled to a declared job/run timeout + slack. Floor =
    B2_EPHEMERAL_HOURS (default 168h/7d); with a timeout, max(floor, timeout_h +
    72h slack). Rotation (auto-reattach / supervise-relaunch) is the primary
    freshness mechanism; this TTL is the backstop. (CREDENTIAL_LIFECYCLE.md)

    `credbroker.py`:156 carries an acknowledged DUPLICATE of this function.
    credbroker* is box-side and explicitly not absorbed (plan §3); the drift is
    deliberate and audited separately. Do not unify them here."""
    floor = float(os.environ.get("B2_EPHEMERAL_HOURS", 168))
    if timeout_s:
        return max(floor, timeout_s / 3600.0 + 72.0)
    return floor


# moved-from: herdd._ship_b2_env
def _ship_b2_env(base: str | None, hours: float, write_prefix: str | None = None,
                 dry_run: bool = False,
                 publish: bool = True) -> list[tuple[str, str]]:
    """B2 env vars to inject into a box, as an ordered list of (name, value).

    With a minter AND a write_prefix (a box whose writes are confined to one
    prefix, e.g. a jobs box -> 'jobs/'): mints a SCOPED PAIR — a bucket-wide
    read key (B2_KEY_ID/B2_APPLICATION_KEY, no writes) plus a write key
    restricted to write_prefix (B2_WRITE_KEY_ID/B2_WRITE_APPLICATION_KEY). The
    box's [b2]/[b2w] rclone remotes route reads vs writes accordingly
    (tools/vast/CREDENTIAL_LIFECYCLE.md).

    `publish=True` additionally mints the PUBLISH key —
    B2_PUBLISH_KEY_ID/B2_PUBLISH_APPLICATION_KEY, namePrefix `checkpoints/`,
    the box's [b2p] remote. WHY A THIRD KEY: a B2 application key carries
    exactly ONE namePrefix and `jobs/`+`checkpoints/` share none, so the only
    single-key alternative is a bucket-wide write — strictly worse. A training
    bundle's publish stage (adapter -> checkpoints/<RUN_NAME>/) 403'd on both v7
    arms AFTER training completed because this key did not exist
    (docs/plans/witness/g2_push/B2_PUBLISH_KEY_SCOPE_FIX_2026-08-05.md).
    No-delete, self-expiring, revoked with the rest on destroy. Disable per
    fleet with B2_PUBLISH_PREFIX=''.

    Without a write_prefix, or on dry_run / no minter / mint failure: falls back
    to the single bucket-wide pair from _ship_b2_pair (no B2_WRITE_*), so the
    box's [b2w] degrades to [b2] and behavior is byte-identical to before.

    FROZEN SHAPE: the ORDER of the returned pairs, and the 2-vs-4-vs-6 length,
    are what the box's [b2]/[b2w]/[b2p] rclone remotes key off (consumed
    positionally-by-name into box env and jobd.env). The bare `rk = None` in
    the except below, tested by the following `if rk:`, is the fallthrough —
    there is no else-branch reset, and an early-return refactor would ship a
    scoped env full of `None`s on a mint failure."""
    try:
        base = b2_mint_key.sanitize_name(base) if base else ""
    except b2_mint_key.MintError:
        base = ""
    minter = (os.environ.get("B2_MINTER_KEY_ID")
              and os.environ.get("B2_MINTER_APPLICATION_KEY"))
    if base and write_prefix and minter and not dry_run:
        if base in _MINTED_SCOPED:
            (rk, rs), (wk, ws) = _MINTED_SCOPED[base]
        else:
            try:
                (rk, rs), (wk, ws) = b2_mint_key.mint_pair(
                    base, hours=hours, write_prefix=write_prefix)
                print(f">> minted scoped B2 key pair {base}-ro/-rw "
                      f"(read bucket-wide, write namePrefix={write_prefix!r}, "
                      f"ttl {hours:g}h, no deleteFiles, self-expires)")
                _MINTED_SCOPED[base] = ((rk, rs), (wk, ws))
            except Exception as e:
                print(f">> WARNING: scoped B2 key mint failed ({e}) — falling "
                      f"back to one bucket-wide key (no write scoping)",
                      file=sys.stderr)
                rk = None  # the fallthrough sentinel, verbatim — tested by `if rk:` below
        if rk:
            out = [("B2_KEY_ID", rk), ("B2_APPLICATION_KEY", rs),
                   ("B2_WRITE_KEY_ID", wk), ("B2_WRITE_APPLICATION_KEY", ws)]
            pub = _MINTED_PUBLISH.get(base)
            if publish and pub is None:
                try:
                    pub = b2_mint_key.mint_publish(base, hours=hours)
                    if pub:
                        _MINTED_PUBLISH[base] = pub
                        print(f">> minted scoped B2 publish key {base}-pub "
                              f"(write namePrefix="
                              f"{b2_mint_key.publish_prefix()!r}, ttl {hours:g}h, "
                              f"no deleteFiles, self-expires)")
                except Exception as e:
                    # NEVER fail a launch on the publish grant: a box without it
                    # runs every non-publishing job exactly as before, and the
                    # submit-time write-scope preflight is what stops a
                    # publishing bundle from reaching such a box.
                    print(f">> WARNING: B2 publish key mint failed ({e}) — this "
                          f"box has NO checkpoints/ write grant; a bundle with a "
                          f"publish stage will 403", file=sys.stderr)
                    pub = None
            if publish and pub:
                out += [("B2_PUBLISH_KEY_ID", pub[0]),
                        ("B2_PUBLISH_APPLICATION_KEY", pub[1])]
            return out
    kid, key = _ship_b2_pair(base, hours=hours, dry_run=dry_run)
    return [("B2_KEY_ID", kid), ("B2_APPLICATION_KEY", key)]


# moved-from: herdd._minted_expiry
def _minted_expiry(base: str | None, hours: float) -> int | None:
    """Unix-epoch expiry (int) of the ephemeral key minted THIS process for
    `base`, or None when no mint happened (dry-run / standing-key fallback — the
    standing key has no known expiry, so B2_KEY_EXPIRES_AT must NOT be shipped).
    Mint-cache membership is the witness that a mint actually occurred
    (docs/plans/cred-broker-buildout.md §2.1).

    Checks _MINTED_SCOPED and _MINTED_PAIRS and deliberately NOT
    _MINTED_PUBLISH: the publish grant rides along with a scoped mint, and a
    box whose read/write keys are STANDING must not be told its keys expire.
    `test_broker_env.py` pins the asymmetry."""
    try:
        base = b2_mint_key.sanitize_name(base) if base else ""
    except b2_mint_key.MintError:
        return None
    if base and (base in _MINTED_SCOPED or base in _MINTED_PAIRS):
        return int(time.time() + hours * 3600)
    return None


# moved-from: herdd._resolve_secret
def _resolve_secret(name: str, run_id: object = None, mint: bool = False,
                    key_name: str | None = None) -> str | None:
    """Local value for a secret env NAME the spec recorded (never stored in B2).
    B2_KEY_ID/B2_APPLICATION_KEY resolve through _ship_b2_pair — on a real
    relaunch (mint=True) that mints a FRESH ephemeral run key (the evicted
    box's pair may be near expiry, and revoke-then-mint retires it); preflight
    checks (mint=False) only prove a shippable pair exists. Everything else is
    a plain lookup. load_env has already folded .env into os.environ.

    `key_name` overrides the minted B2 key NAME (default `run-<run_id>`). The
    handoff understudy passes a nonce-suffixed name so its mint does NOT
    revoke-then-mint the PRIMARY's live `run-<run_id>` key (T3; the
    box-44566398 incident class)."""
    if name in ("B2_KEY_ID", "B2_APPLICATION_KEY"):
        base = key_name or (f"run-{run_id}" if run_id else "")
        kid, key = _ship_b2_pair(base, hours=_ephemeral_hours(), dry_run=not mint)
        return kid if name == "B2_KEY_ID" else key
    return os.environ.get(name)


# moved-from: herdd._read_spec_soft
def _read_spec_soft(run_id: object) -> dict[str, Any]:
    """Soft-read runs/<RUN_ID>/spec.json (the declarative launch contract). {} on
    ANY failure — a B2 blip or a pre-spec run must degrade to the event-scrape
    fallback, never abort supervise startup.

    The reciprocal reader of `_build_launch_spec`'s frozen contract, which is
    why it is here and not in `storage/` — `storage.b2._rclone_soft` is its
    transport, not its subject. Its callers are supervise-side
    (`_capture_launch_spec`) and repoint at plan §8 step 4."""
    bucket = os.environ.get("B2_BUCKET")
    if not bucket:
        return {}
    rc, out, _ = b2._rclone_soft(["cat", f"b2:{bucket}/runs/{run_id}/spec.json"])
    if rc != 0 or not (out or "").strip():
        return {}
    try:
        d = json.loads(out)
    except (ValueError, TypeError):
        return {}
    return d if isinstance(d, dict) else {}


# Homed here, not in `supervise/`, on the same reasoning as `_read_spec_soft`
# directly above: it is the OTHER half of the launch-spec read (the degraded
# path `_capture_launch_spec` falls back to when there is no v=1 spec.json),
# and its remaining callers sit in three different supervise modules
# (`run_lane._capture_launch_spec`, `replacement._has_relaunched_after_last_evicted`,
# the handoff adopt-backfill) plus the `runs` CLI. Putting it in any one of
# those would force the other two to import a peer lane; putting it in the ring
# BELOW them is the only cycle-free home. No port manifest claimed it — recorded
# here so the step-6 rename table has one entry, not four copies.
# moved-from: herdd._raw_events_soft
def _raw_events_soft(run_id: object) -> list[dict[str, Any]]:
    """Raw event dicts from the local runmeta cache (already refreshed by a prior
    _read_run_soft/read_run). Needed for launch-spec/original-bid capture and the
    stopping-actor read, all of which the fold_events whitelist drops. [] on any
    failure. Sorted (ts, nonce)."""
    import glob as _glob
    cache = os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "vast-runmeta", str(run_id), "events")
    out: list[dict[str, Any]] = []
    try:
        for p in _glob.glob(os.path.join(cache, "*.json")):
            try:
                with open(p) as fh:
                    out.append(json.load(fh))
            except Exception:
                pass
    except Exception:
        return []
    out.sort(key=lambda e: (e.get("ts", ""), e.get("nonce", "")))
    return out


# --------------------------------------------------------------------------- #
# The default image, and the constant the failure message quotes
# --------------------------------------------------------------------------- #
# Landed here by the step-6 `cli/` port (cli-surface.json hazard H1): the
# `launch --image` help text and `cmd_train`'s refusal both INTERPOLATE
# `_EXPECTED_DEFAULT_IMAGE`, so it is printed output and an input to the §8
# CLI-surface byte diff. It had no vastlib home and no port manifest claimed
# it; `launch/spec.py` is the home §5 names for the launch spec's constants.
#
# H1's other half is a test repoint, NOT done here and still owed:
# `test_rehearse.py:138` regex-reads the LITERAL FILE `herdd.py` for
# `^_TRAIN_FALLBACK_IMAGE = "([^"]+)"` and asserts it equals herdd.yaml's
# `default_image`. That guard exists because a half-landed image flip is
# invisible. It keeps passing while step 6 is ADD-ONLY (the flat file still
# holds its copy); it must be repointed at this module IN THE SAME COMMIT that
# thins `herdd.py`, and the parametrized row must not be deleted.
#
# `_require_image` — the fail-closed gate that quotes this — LANDED at the foot
# of this file later in the same step (it was still a raising seam in
# `launch/launch.py` when the paragraph above was written). One name, one home:
# it reads the constant from here, and `launch.py` now binds the ported
# function instead of raising.

# Default image: the BAKED train image on our R2 registry (herdd.yaml
# default_image —
# full train env + CUDA nvcc baked in, onstart sources
# /workspace/.train_env_activate). That IS the fast boot: one authenticated
# registry pull (no Docker-Hub throttle), no B2 rehydrate step, no
# base/tarball ABI seam.
#
# The image herdd.yaml is EXPECTED to name in default_image — quoted in the
# error when it is missing, not silently substituted. This was
# axolotlai/axolotl:main-latest until 2026-08-02; that fallback was a trap once
# the baked image became the only one carrying nvcc and the training env, and
# axolotl stopped being any lane's image when the eval env unified onto t211.
# moved-from: herdd._TRAIN_FALLBACK_IMAGE
_TRAIN_FALLBACK_IMAGE = "registry.example.com/train:latest"

# moved-from: herdd._EXPECTED_DEFAULT_IMAGE
_EXPECTED_DEFAULT_IMAGE = _TRAIN_FALLBACK_IMAGE


# --------------------------------------------------------------------------- #
# Stale-pin gate — the invoking checkout is not a source of truth
# --------------------------------------------------------------------------- #
# `default_image` resolves from whatever checkout the CLI was invoked in, and
# the primary checkout is SHARED: any session can leave it on a branch that
# predates an image roll, and every launch from it then silently rents the old
# image. Measured 2026-08-20: the primary sat on a peer branch 266 commits
# behind `main`, so `--jobs` launches pulled t214 for a day after t215 shipped.
#
# Nothing local could catch it. A stale checkout is INTERNALLY CONSISTENT — a
# roll moves `herdd.yaml` and `_TRAIN_FALLBACK_IMAGE` in one commit, so on an
# old branch the two agree with each other and disagree only with the world.
# `classify_image_staleness` does not cover this either: it asks whether a
# running box's image drifted from the tag it launched with, which is a
# different axis, and it is inert for our registry besides (task #33).
#
# So the comparison has to reach OUTSIDE the working tree, and `origin/main` is
# the cheapest truth that exists locally. The read is SOFT by construction: no
# git, no remote ref, a detached checkout or a box with no repo all return None
# and the gate stands down. It fires only on a POSITIVE disagreement, and only
# when the image came from the default — naming `--image` explicitly is intent
# (a deliberate rollback is exactly that) and is never second-guessed.
_CANONICAL_PIN_REF = "origin/main"
_CANONICAL_PIN_PATH = "tools/vast/herdd.yaml"


def canonical_default_image(
        runner: Callable[[Sequence[str]], Any] | None = None,
        repo_root: object = None) -> str | None:
    """`default_image` as `origin/main` spells it, or None if unknowable.

    Soft everywhere: a missing git, an absent remote ref, a fetch that has
    never run, or a checkout outside a repo all yield None. A gate that cannot
    read the truth must not invent a disagreement.
    """
    if runner is None:
        def _run(args: Sequence[str]) -> Any:  # noqa: ANN401 — CompletedProcess-shaped
            import subprocess
            return subprocess.run(args, capture_output=True, text=True,
                                  timeout=10)
        runner = _run
    root = repo_root or os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    try:
        cp = runner(["git", "-C", str(root), "show",
                     f"{_CANONICAL_PIN_REF}:{_CANONICAL_PIN_PATH}"])
    except Exception:  # noqa: BLE001 — every failure is "unknowable"
        return None
    if getattr(cp, "returncode", 1) != 0:
        return None
    m = re.search(r"^default_image:\s*(\S+)", getattr(cp, "stdout", "") or "",
                  re.M)
    return m.group(1) if m else None


def image_pin_verdict(resolved: object, local_default: object,
                      canonical: object) -> str | None:
    """PURE. `None` when the launch may proceed, else the refusal text.

    Fires only when all three hold: the image came from the checkout's own
    default (not an explicit `--image`), the canonical pin is READABLE, and
    the two positively disagree.
    """
    if not resolved or not canonical:
        return None
    if str(resolved) != str(local_default):
        return None                      # explicit --image: operator intent
    if str(resolved) == str(canonical):
        return None
    return (f"stale image pin: this checkout's default_image is "
            f"{resolved}, but {_CANONICAL_PIN_REF} says {canonical}.\n"
            f"       The checkout you invoked herdd from is behind an image "
            f"roll, so this launch would rent the OLD image and every result "
            f"off it would describe a stack we no longer ship.\n"
            f"       Fix the checkout (git fetch && git checkout main), run "
            f"from an up-to-date worktree, or — if you mean the old image — "
            f"name it: --image {resolved}")


# --------------------------------------------------------------------------- #
# The credentials a box is handed, and the gate in front of them
# --------------------------------------------------------------------------- #
# Five helpers, four commands (`launch`, `supervise`, `train`, `job supervise`),
# `cli-surface.json` hazard H3: no `cli/<command>.py` can own them. They are
# also five of the seven seams `launch/launch.py` was still RAISING, and this is
# the step that closes them — `launch.py` binds each by module-level assignment
# at the foot of its seam block, so `monkeypatch.setattr(launch, …)` keeps
# steering `_do_launch` exactly as before.
#
# WHY HERE AND NOT `core/config.py` (which cli-surface.json proposed for the two
# `hf_*` helpers): `hf_token_text` resolves a LIVE SECRET and `image_login_arg`
# MINTS one, and both are handed to a rented box at create time. That is this
# module's subject — "the credentials a box is handed", the same reason
# `_ship_b2_pair` / `_ship_b2_env` live here — while `core/config.py` is the
# workstation's own configuration and sits at the bottom of the DAG, where a
# credential mint would drag `registry.mint_token` under every importer.
# `image_login_arg` and `_mask_image_login` move as a PAIR by rule: the first
# returns a string containing a token and the second is the only thing that
# makes it printable.
#
# `_require_image` sits with them because it is the fail-closed gate in front of
# the same create call, and because it interpolates `_EXPECTED_DEFAULT_IMAGE`
# from the block above — one module, one image contract.


# moved-from: herdd.hf_token_text
def hf_token_text(explicit: str | None = None) -> str | None:
    """Resolve a HuggingFace token to upload to a rented box, so model-weight
    pulls authenticate (full bandwidth, gated repos). Order:
      1. explicit value (--hf-token)
      2. HF_TOKEN / HUGGING_FACE_HUB_TOKEN / HUGGINGFACE_TOKEN env (incl. .env via load_env)
      3. ~/.config/herdd/hf_token   (dedicated file, matches the *.key convention)
      4. ~/.cache/huggingface/token        (this box's own `hf` CLI login)
    Returns the token string or None.
    """
    if explicit:
        return explicit.strip()
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(k)
        if v and v.strip():
            return v.strip()
    for c in ("~/.config/herdd/hf_token", "~/.cache/huggingface/token"):
        p = os.path.expanduser(c)
        if os.path.isfile(p):
            t = open(p).read().strip()
            if t:
                return t
    return None


# `tools/vast/` — `hf_login_snippet` read `onstart/hf_login.sh` relative to
# `dirname(abspath(__file__))`, which was `tools/vast` in the flat module and is
# `tools/vast/vastlib/launch` here. Copied verbatim the probe would miss the
# file and SILENTLY fall through to the inline fallback: no error, no missing
# token on the box — just a second copy of the snippet that no longer tracks
# edits to `onstart/hf_login.sh`, which is exactly what the file exists to
# prevent. Hoisted for the same reason `core.config._HERE` and
# `jobs.bundle.TOOLS_VAST_DIR` are, and pinned by
# `test_vastlib_cli_helpers.py::test_hf_login_snippet_reads_the_repo_file`.
_TOOLS_VAST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


# moved-from: herdd.hf_login_snippet
def hf_login_snippet() -> str:
    """The secret-free onstart prelude that installs $HF_TOKEN on the box.
    Lives beside this script so it can be curated/read independently."""
    here = _TOOLS_VAST_DIR
    path = os.path.join(here, "onstart", "hf_login.sh")
    if os.path.isfile(path):
        return open(path).read()
    # Fallback inline (keep in sync with onstart/hf_login.sh) so a stray copy of
    # herdd.py without the helper still injects the token.
    return (
        'if [ -n "${HF_TOKEN:-}" ]; then '
        '_hf_home="${HF_HOME:-$HOME/.cache/huggingface}"; mkdir -p "$_hf_home" 2>/dev/null || true; '  # noqa: E501 — verbatim shell fallback, byte-identical to onstart/hf_login.sh
        'printf %s "$HF_TOKEN" > "$_hf_home/token" 2>/dev/null && chmod 600 "$_hf_home/token" 2>/dev/null || true; '  # noqa: E501 — verbatim shell fallback, byte-identical to onstart/hf_login.sh
        "grep -q '^HF_TOKEN=' /etc/environment 2>/dev/null || echo \"HF_TOKEN=${HF_TOKEN}\" >> /etc/environment 2>/dev/null || true; "  # noqa: E501 — verbatim shell fallback, byte-identical to onstart/hf_login.sh
        'export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"; fi\n'
    )


# moved-from: herdd.image_login_arg
def image_login_arg(image: object, explicit: str | None = None) -> str | None:
    """Docker-login credential string for Vast's `image_login` field, so a box
    can pull a PRIVATE image (the prebaked train/serve/eval image).

    Vast wants a literal `docker login` argument string: `-u USER -p TOKEN HOST`.
    It can only be set at launch (not updated later). Whatever goes here is
    handed to vast's API and to the rented host's docker daemon — treat it as
    DISCLOSED TO A STRANGER. Both branches below are shaped by that.

    LIVE PATH — the R2 Worker registry (registry.example.com; registry/
    R2_WORKER_REGISTRY_PLAN.md D3), which is where every fleet image lives
    since the 2026-08-12 cutover. We mint a 6 h HMAC token scoped to the
    image's repo; username is the literal "vast". Signing secret:
    REGISTRY_AUTH_SECRET from the environment — it lives in .env,
    which load_env does NOT read, so export it or add it to .env before
    launching.

    THERE IS NO SECOND PATH. The GitLab lane and its deploy-token auto-attach
    were cut on 2026-08-22 (owner: "GitLab as a registry needs to be cut. Focus
    on R2 and B2"), so a `registry.gitlab.com` ref gets no creds here and is
    REFUSED outright upstream by `_require_image`. Nothing falls back: an
    unrecognised private registry returns None, the pull fails visibly on the
    box, and no credential is invented for a host we do not control.

    Returns the arg string, or None when the image is public / no creds apply.
      - explicit: a full override string (from --login), returned verbatim.
    """
    if explicit:
        return explicit.strip()
    if not image:
        return None
    # Only attach creds for images actually hosted on the private registry — a
    # public base image (pytorch/…, vllm/…) must stay anonymous.
    host = str(image).split("/", 1)[0]
    if host != imageref.R2_REGISTRY_HOST:
        return None
    secret = os.environ.get(imageref.R2_SECRET_ENV)
    if not secret or "/" not in str(image):
        return None
    # registry.example.com/<repo>:<tag> → <repo> (may be multi-segment)
    repo = str(image).split("/", 1)[1].split("@", 1)[0].rsplit(":", 1)[0]
    from registry.mint_token import mint
    tok = mint(secret, repo, ttl_hours=6, instance="herdd")  # type: ignore[no-untyped-call]
    return f"-u vast -p {tok} {host}"


# moved-from: herdd._mask_image_login
def _mask_image_login(s: str | None) -> str | None:
    """Redact the -p TOKEN in an image_login string for safe printing."""
    if not s:
        return s
    return re.sub(r"(-p\s+)\S+", r"\1<redacted>", s)


# moved-from: herdd._require_image
def _require_image(image: object, what: str) -> Any:  # noqa: ANN401 — returns `image` unchanged
    """Fail-closed image gate for every path that creates a container.

    Returns `image` unchanged when it is set and pullable. Two refusals, both
    non-zero exits rather than a POSTed instance:

      1. no image at all — never fall back to a stock one;
      2. an image on a RETIRED registry (`imageref.is_retired_registry`). This
         refusal is worth its own branch because the alternative is not an
         error, it is a BILL: vast accepts the create, the host tries to pull,
         and the box sits in `loading` on `denied: access forbidden` until
         somebody notices. Refusing here costs nothing and names the move.

    Both refusals are `sys.exit`, INCLUDING on the relaunch path that reaches
    this from the supervisor — deliberately the same class of exit case (1)
    has always had there, and `_relaunch` already catches SystemExit. Case (2)
    is also unreachable from a daemon in practice: a spec can only record a
    retired ref if a launch accepted one, and this gate is what a launch goes
    through.
    """
    host = str(image).split("/", 1)[0] if image else ""
    if image and imageref.is_retired_registry(host):
        sys.exit(
            f"error: refusing to {what} with {image} — {host} is a RETIRED "
            "registry.\n"
            "       GitLab was cut as a registry on 2026-08-22; the project, "
            "its images and its\n"
            "       token are gone, so this ref can never pull and the box "
            "would bill while it\n"
            "       waits in `loading`. Every image lives on "
            f"{imageref.R2_REGISTRY_HOST} now — see the\n"
            "       `push-train-image` skill and tools/vast/registry/README.md.\n"
            f"       Expected default_image: {_EXPECTED_DEFAULT_IMAGE}")
    if image:
        return image
    sys.exit(
        f"error: no image to {what} with — herdd.yaml has no readable "
        "`default_image` and no --image was passed.\n"
        f"       Expected default_image: {_EXPECTED_DEFAULT_IMAGE}\n"
        "       (the unified t211 train+serve+eval image). There is deliberately\n"
        "       NO fallback: stock pytorch has no nvcc, no baked env and not our\n"
        "       vLLM fork, so it would rent a box that cannot train or serve.\n"
        "       Pass --image explicitly, or restore tools/vast/herdd.yaml.")


# --------------------------------------------------------------------------- #
# Run-metadata soft reads — the third and fourth halves of `_raw_events_soft`
# --------------------------------------------------------------------------- #
# `_observe` (the `supervise` poll body, `cli/supervise.py`) is the only caller
# of all three, so `cli-surface.json` filed them under `cli/supervise.py`. They
# land one ring DOWN, beside `_read_spec_soft` and `_raw_events_soft`, on that
# module's own argument: these are the RECIPROCAL READERS of the durable B2 run
# record, `storage.b2._rclone_soft` is their transport and not their subject,
# and putting a run-record reader in a command module means the next reader
# (`cli/runs.py`, `boxstate.py`) either imports a sibling command or copies it.
# `_last_stopping_actor` in particular is a pure fold over `_raw_events_soft`,
# which is already here.
#
# S2 CONTRACT, do not "simplify": a FAILED read never fabricates a terminal.
# `_read_run_soft` tags the view `_cache_stale` and forces a NON-terminal
# 'unknown' when the refresh failed and no cached events survive — a supervisor
# that reads 'done' off a network blip stops babysitting a live, billing run.


# moved-from: herdd._read_run_soft
def _read_run_soft(run_id: object, live_iids: Iterable[Any] = ()) -> dict[str, Any]:
    """runmeta.read_run that never sys.exits and NEVER fabricates a terminal from
    a failed read (S2). Records the rclone rc of the incremental copy; if the
    refresh failed AND there are no cached events to fold, the view is tagged
    _cache_stale and forced to non-terminal 'unknown'. A warm cache of immutable
    events stays authoritative even if the momentary refresh failed."""
    rcs: dict[str, Any] = {"copy": None, "cat": None}

    def _runner(args: Sequence[str], input: Any = None) -> Any:  # noqa: ANN401 — Zone S runner shape
        rc, out, errtxt = runmeta._default_runner(args, input=input)  # type: ignore[no-untyped-call]
        if args and args[0] in rcs:
            rcs[args[0]] = rc
        return rc, out, errtxt

    try:
        view = runmeta.read_run(run_id, runner=_runner, live_iids=live_iids)
    except Exception as e:
        return {"status": "unknown", "display_status": "unknown",
                "_cache_stale": True, "_read_error": str(e),
                "instance_id": None, "n_events": 0}
    stale = (rcs["copy"] not in (0, None)) or (rcs["cat"] not in (0, None))
    if stale:
        view["_cache_stale"] = True
        if not view.get("n_events"):
            view["status"] = "unknown"
            view["display_status"] = "unknown"
    return view


# moved-from: herdd._last_stopping_actor
def _last_stopping_actor(run_id: object) -> Any:  # noqa: ANN401 — actor string as the event carried it
    """Actor of the latest LIVE `stopping` event (the operator-intent signal
    poll 2b reads). A later `resumed`/`launched`/`relaunched` event CLEARS the
    intent — a parked-then-resumed box (herdd stop -> start) or a manually
    re-launched run is back under normal supervision policy, so a stale park
    must never read as operator_destroy. None if no live intent."""
    last = None
    for e in _raw_events_soft(run_id):            # (ts, nonce)-sorted
        ev = e.get("event")
        if ev == "stopping":
            last = e
        elif ev in ("resumed", "launched", "relaunched"):
            last = None
    return last.get("actor") if last else None


# moved-from: herdd._status_marker_soft
def _status_marker_soft(run_id: object) -> str | None:
    """Legacy checkpoints/<id>/STATUS marker (for I4 done/failed-inferred when the
    box is gone). None if absent/unreadable."""
    bucket = os.environ.get("B2_BUCKET")
    if not bucket:
        return None
    rc, out, _ = b2._rclone_soft(["cat", f"b2:{bucket}/checkpoints/{run_id}/STATUS"])
    return (out or None) if rc == 0 else None


# The `ensure_base_model.sh` stdout contract, homed with the launch spec because
# its output SIZES THE BOX: `cmd_train` feeds the byte count to the disk
# calculation that becomes the create call's `--disk`. Pure, no I/O, no deps —
# `cli/train.py` is its only caller today, and it lives below `cli` so a second
# caller (a sizing dry-run, the workflow lane) never has to import a command.
# moved-from: herdd.parse_base_gate_stdout
def parse_base_gate_stdout(out: str | None) -> tuple[str, int | None]:
    r"""PURE. `(subpath, bytes|None)` from `ensure_base_model.sh` stdout.

    That script's contract is the subpath alone; `--print-bytes` widens it to
    `"<subpath>\t<bytes>"`. Partition rather than split/index so an OLD script
    that ignores the flag and prints only the subpath still parses — the flag
    ships from the launcher but the script can lag on a stale checkout.

    An unparseable, absent, or zero size is None, never 0: a caller sizing a
    box's disk must be able to tell "we could not measure this model" from "this
    model is empty", and 0 would silently under-size every box."""
    sub, _, size = (out or "").partition("\t")
    try:
        n = int(size.strip()) or None
    except ValueError:
        n = None
    return sub.strip(), n
