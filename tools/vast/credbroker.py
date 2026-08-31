#!/usr/bin/env python3
"""Cred-broker core library (transport-agnostic) — verify / policy / mint.

Implements docs/plans/cred-broker-buildout.md §2.2-§2.4 + §3 state dir + §4
invariants. The daemon (credbrokerd.py) wires this to HTTP + the B2-mediated
lane; nothing in here does network I/O except through injected callables and
b2_mint_key's _http seam.

Wire-format pinning (cred_client.py carries a byte-compatible copy; C3
cross-tests the two):

  nonce_bytes            = nonce.encode("utf-8")   (the hex string as-is,
                           NOT bytes.fromhex — registry nonces need not be hex)
  proof                  = HMAC-SHA256(nonce_bytes,
                           b"credreq|<iid>|<ts>|<role>").hexdigest()
  enc_key / mac_key      = SHA256(nonce_bytes + b"enc"/b"mac").digest()
  keystream block i      = HMAC-SHA256(enc_key, b"<ts>|<i>")   (i decimal, 0-based)
  ciphertext             = keystream XOR canonical-JSON plaintext
                           (json.dumps(obj, sort_keys=True, separators=(",",":")))
  mac                    = HMAC-SHA256(mac_key, b"<ts>|" + ciphertext).hexdigest()
  envelope blob          = {"ts": int, "ciphertext": hex, "mac": hex}

Security invariants (§4): nothing a box asserts is trusted (instance_id is a
lookup key only); nonce compares are constant-time; MAC-then-decrypt; the
broker only ever revokes brk-keys it minted itself; audit records carry key
IDs/names but NEVER applicationKey values.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import b2_mint_key

LIVE_STATES = {"running", "loading", "created"}   # mirror herdd.LIVE_STATES
ROLE_WRITE_PREFIX = {"jobs": "jobs/", "serve": "serve/", "train": None}
REPLAY_WINDOW_S = 600
RATE_MIN_SPACING_S = 60
RATE_DAILY_CAP = 24


# ------------------------------------------------------------- state files #
def state_dir() -> str:
    """~/.local/state/upstream-monorepo/credbroker (created), or CRED_BROKER_STATE."""
    d = (os.environ.get("CRED_BROKER_STATE")
         or os.path.expanduser("~/.local/state/upstream-monorepo/credbroker"))
    os.makedirs(d, exist_ok=True)
    return d


class _JsonFile:
    """Tiny persisted-dict base: load on init, atomic save on request."""

    def __init__(self, path):
        self.path = path
        self.data = {}
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=0, sort_keys=True)
        os.replace(tmp, self.path)


class Registry(_JsonFile):
    """Per-instance broker state: registered nonce hashes (POST /v1/register
    fallback for pre-nonce boxes) + the brk-keys we last minted per iid (so a
    reissue can revoke its predecessors — and ONLY them)."""

    def __init__(self, path=None):
        super().__init__(path or os.path.join(state_dir(), "registry.json"))

    def _ent(self, instance_id) -> dict:
        return self.data.setdefault(str(instance_id), {})

    def nonce_sha256(self, instance_id):
        return (self.data.get(str(instance_id)) or {}).get("nonce_sha256")

    def register_nonce(self, instance_id, nonce_sha256_hex):
        self._ent(instance_id)["nonce_sha256"] = str(nonce_sha256_hex).lower()
        self.save()

    def last_keys(self, instance_id) -> list:
        return list((self.data.get(str(instance_id)) or {}).get("keys") or [])

    def set_last_keys(self, instance_id, keys):
        """keys: [{'name': ..., 'key_id': ...}] — names+IDs only, no secrets."""
        self._ent(instance_id)["keys"] = list(keys)
        self.save()


class RateLimiter(_JsonFile):
    """Per-iid mint throttle, persisted: >=60 s between issues, <=24/day.
    Bounds a compromised box to bounded mint volume (§4.7)."""

    def __init__(self, path=None):
        super().__init__(path or os.path.join(state_dir(), "ratelimit.json"))

    def try_acquire(self, instance_id, now=None):
        """(ok, reason). Records the issue timestamp on success."""
        now = time.time() if now is None else float(now)
        k = str(instance_id)
        stamps = [t for t in (self.data.get(k) or []) if now - t < 86400]
        if stamps and now - max(stamps) < RATE_MIN_SPACING_S:
            self.data[k] = stamps
            self.save()
            return False, "spacing"
        if len(stamps) >= RATE_DAILY_CAP:
            self.data[k] = stamps
            self.save()
            return False, "daily_cap"
        stamps.append(now)
        self.data[k] = stamps
        self.save()
        return True, None


# ------------------------------------------------------------------- audit #
def _redact(obj):
    """Drop any key that smells like an application-key SECRET (key IDs and
    names stay). Recursive; defense-in-depth for audit_append."""
    if isinstance(obj, dict):
        return {k: _redact(v) for k, v in obj.items()
                if "application_key" not in k.lower()
                and "applicationkey" not in k.lower()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def audit_append(record, path=None):
    """Append one JSONL audit record ({ts, transport, remote, instance_id,
    role, verdict, reason?, keys?, expires_at?}). NEVER writes applicationKey
    values — redacted structurally, whatever the caller passes."""
    path = path or os.path.join(state_dir(), "audit.jsonl")
    rec = dict(record)
    rec.setdefault("ts", int(time.time()))
    with open(path, "a") as f:
        f.write(json.dumps(_redact(rec), sort_keys=True) + "\n")


def ephemeral_hours(timeout_s=None) -> float:
    """TTL policy for broker-minted keys — duplicate of herdd._ephemeral_hours
    (deliberately NOT imported): floor B2_EPHEMERAL_HOURS (default 168 h);
    with a declared timeout, max(floor, timeout_h + 72 h slack)."""
    floor = float(os.environ.get("B2_EPHEMERAL_HOURS", 168))
    if timeout_s:
        return max(floor, timeout_s / 3600.0 + 72.0)
    return floor


# ------------------------------------------------------ verification core #
def _instance_env(i) -> dict:
    """extra_env readback (list of [K, V] pairs on the wire; tolerate a dict
    too) — mirrors herdd._instance_env. {} when absent."""
    ee = i.get("extra_env") or []
    if isinstance(ee, dict):
        return dict(ee)
    out = {}
    for kv in ee:
        if isinstance(kv, (list, tuple)) and len(kv) == 2:
            out[kv[0]] = kv[1]
    return out


def verify_instance(instance_id, nonce, fetch_instances, registry=None):
    """(ok, reason). Live-state + nonce binding per §2.3 steps 1-2.

    fetch_instances: injectable zero-arg callable returning the vast
    instances list (the daemon binds it to the account API; tests fake it).
    Nothing the box asserts is trusted — instance_id is only a lookup key;
    the presented nonce must match the launch-recorded extra_env nonce
    (constant-time) OR sha256(nonce) must match the registry entry."""
    try:
        instances = fetch_instances() or []
    except Exception:
        return False, "instances_unavailable"
    inst = None
    for i in instances:
        if i.get("id") == instance_id:
            inst = i
            break
    if inst is None:
        return False, "unknown_instance"
    if (inst.get("actual_status") or "").lower() not in LIVE_STATES:
        return False, "not_live"
    nonce = str(nonce or "")
    if not nonce:
        return False, "nonce_mismatch"
    expected = str(_instance_env(inst).get("BOX_IDENTITY_NONCE") or "")
    if expected and hmac.compare_digest(nonce.encode(), expected.encode()):
        return True, "nonce"
    reg = registry.nonce_sha256(instance_id) if registry is not None else None
    if reg:
        presented = hashlib.sha256(nonce.encode()).hexdigest()
        if hmac.compare_digest(presented.encode(), str(reg).lower().encode()):
            return True, "registry"
    return False, "nonce_mismatch"


def check_policy(role, want_write_prefix=None):
    """(ok, effective_write_prefix|None) per §2.3 step 3. jobs->'jobs/',
    serve->'serve/', train->None (bucket-wide single key until the
    runs/<RID>/ layout, C8). A want may equal or EXTEND (startswith) the
    policy prefix, e.g. 'jobs/<JOB_ID>/'; anything else -> deny."""
    if role not in ROLE_WRITE_PREFIX:
        return False, None
    policy = ROLE_WRITE_PREFIX[role]
    want = want_write_prefix or None
    if policy is None:
        return (True, None) if want is None else (False, None)
    if want is None:
        return True, policy
    if want == policy or want.startswith(policy):
        return True, want
    return False, None


# --------------------------------------------- envelope (B2-mediated lane) #
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
    """§2.4 response envelope: HMAC-SHA256 keystream XOR over canonical-JSON
    plaintext, MAC over ts||ciphertext. World-readable-safe: only the nonce
    holder (that box + the broker) can decrypt or forge."""
    ts = int(time.time() if ts is None else ts)
    enc_key, mac_key = _nonce_keys(nonce)
    pt = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ct = bytes(a ^ b for a, b in zip(pt, _keystream(enc_key, ts, len(pt))))
    mac = hmac.new(mac_key, f"{ts}|".encode() + ct, hashlib.sha256).hexdigest()
    return {"ts": ts, "ciphertext": ct.hex(), "mac": mac}


def open_envelope(nonce, blob):
    """Constant-time MAC check, then decrypt (MAC-then-decrypt — any jobs box
    can shadow-write the response object). Raises EnvelopeError on anything
    malformed, tampered, or sealed under a different nonce."""
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
    the bucket; only this MAC does."""
    msg = f"credreq|{instance_id}|{int(ts)}|{role}".encode()
    return hmac.new(str(nonce).encode(), msg, hashlib.sha256).hexdigest()


def verify_credreq_proof(nonce, instance_id, ts, role, proof) -> bool:
    want = make_credreq_proof(nonce, instance_id, ts, role)
    return hmac.compare_digest(want.encode(), str(proof).encode())


def ts_fresh(ts, now=None, window=REPLAY_WINDOW_S) -> bool:
    """Replay bound: |now - ts| <= window (600 s). Dedupe on (iid, ts) is the
    daemon's job on top of this."""
    now = time.time() if now is None else now
    try:
        return abs(now - int(ts)) <= window
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------- issue #
def _revoke_previous_brk(instance_id, registry):
    """Best-effort revoke of the brk-keys WE last minted for this iid (from
    the registry) — bounded key count on reissue. Guarded to names starting
    'box-<iid>-brk' so a corrupted registry can never make the broker revoke
    a launch-shipped (non-brk) key (§4.4). Never raises."""
    if registry is None:
        return
    prev = registry.last_keys(instance_id)
    if not prev:
        return
    guard = f"box-{instance_id}-brk"
    try:
        auth = b2_mint_key._minter_auth()
        for k in prev:
            if not str(k.get("name", "")).startswith(guard):
                continue
            kid = k.get("key_id")
            if not kid:
                continue
            try:
                b2_mint_key.delete_key(auth, kid)
            except Exception:
                pass
    except Exception:
        pass


def issue_keys(instance_id, role, write_prefix, hours, registry=None,
               now=None) -> dict:
    """Mint per §2.2/§2.3 step 4 and return the /v1/creds response body.

    write_prefix is the EFFECTIVE prefix from check_policy: truthy -> scoped
    pair (mint_pair; read half bucket-wide, write half prefix-scoped);
    None -> single bucket-wide DEFAULT_CAPS key (train). Base name
    'box-<iid>-brk<epoch>' starts with 'box-' so cmd_gc classes it ephemeral
    and destroy's _revoke_box_keys('box-<iid>') prefix-match tears it down
    with no herdd change. Previous brk-keys for this iid are best-effort
    revoked first; the new key names/IDs (never secrets) go back into the
    registry."""
    now = int(time.time() if now is None else now)
    base = f"box-{instance_id}-brk{now}"
    _revoke_previous_brk(instance_id, registry)
    if write_prefix:
        (rk, rs), (wk, ws) = b2_mint_key.mint_pair(
            base, hours=hours, write_prefix=write_prefix)
        b2 = {"key_id": rk, "application_key": rs,
              "write_key_id": wk, "write_application_key": ws}
        minted = [{"name": f"{base}-ro", "key_id": rk},
                  {"name": f"{base}-rw", "key_id": wk}]
    else:
        kid, key = b2_mint_key.mint(base, hours=hours)
        b2 = {"key_id": kid, "application_key": key}
        minted = [{"name": base, "key_id": kid}]
    if registry is not None:
        registry.set_last_keys(instance_id, minted)
    return {
        "b2": b2,
        "bucket": os.environ.get("B2_BUCKET", ""),
        "s3_endpoint": os.environ.get("B2_S3_ENDPOINT", ""),
        "region": os.environ.get("B2_REGION", "us-west-004"),
        "expires_at": now + int(float(hours) * 3600),
    }
