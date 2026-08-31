#!/usr/bin/env python3
"""Cred-broker daemon — tailnet-only HTTP + B2-mediated lane + install-unit.

Wires credbroker.py (the transport-agnostic core) to the wire per
docs/plans/cred-broker-buildout.md §2.2 (HTTP API), §2.4 (B2-mediated lane,
broker side) and §3 (deployment). Serving is Tailscale-only by design: the
bind address comes from `tailscale ip -4` unless CRED_BROKER_BIND overrides,
and 0.0.0.0 is refused without CRED_BROKER_BIND_ANY=1.

Subcommands:
  serve          ThreadingHTTPServer on the tailnet IPv4, port 8651
                 (CRED_BROKER_PORT). Routes: GET /v1/health, POST /v1/creds,
                 POST /v1/register. Every verification failure on /v1/creds
                 returns a UNIFORM 403 body (no oracle, §4.1). Starts the
                 60 s B2 credreq poll thread unless --no-b2-lane.
                 SIGTERM/SIGINT shut down cleanly.
  install-unit   Generate ~/.config/systemd/user/credbrokerd.service AT
                 RUNTIME (absolute paths stay out of git) + print the
                 systemctl next-steps.

Security invariants (§4): stderr logs and audit records carry key IDs/names
but NEVER applicationKey values; the raw nonce never goes on the bucket
(HMAC proof in, sealed envelope out); 403s are uniform.

Vast API note: instances are fetched via GET /api/v1/instances/ (the v0 LIST
endpoint is 410 Gone — see herdd.py header), Bearer-authed with
VASTAI_API_KEY. B2 lane transport: a local 8-line rclone runner duplicating
runmeta._default_runner's (rc, stdout, stderr) contract — chosen over
importing jobmeta to keep this daemon's import surface tiny; tests inject
their own runner.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import b2_mint_key
import credbroker

DEFAULT_PORT = 8651
VAST_API = "https://console.vast.ai/api"
MAX_BODY = 65536
DENY_BODY = {"error": "verification failed"}    # uniform — no oracle (§4.1)
B2_POLL_INTERVAL_S = 60

LOG_STREAM = sys.stderr        # injectable for tests (assert no key material)


def _log(rec):
    """One structured JSON line per request/event to stderr. NO key material
    ever goes through here — callers pass status/verdict metadata only."""
    rec = dict(rec)
    rec.setdefault("ts", int(time.time()))
    print(json.dumps(rec, sort_keys=True), file=LOG_STREAM, flush=True)


def fetch_vast_instances():
    """GET vast v1/instances/ with VASTAI_API_KEY (production fetch_instances
    binding for credbroker.verify_instance). Raises on any failure — the core
    maps that to a deny."""
    key = os.environ.get("VASTAI_API_KEY") or os.environ.get("VAST_API_KEY")
    if not key:
        raise RuntimeError("VASTAI_API_KEY unset")
    req = urllib.request.Request(
        VAST_API + "/v1/instances/",
        headers={"Authorization": "Bearer " + key,
                 "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read().decode())
    return d.get("instances", d) if isinstance(d, dict) else d


def _rclone_runner(args, input=None):
    """rclone transport for the B2 lane (workstation [b2] ops remote);
    duplicates runmeta._default_runner's contract — see module docstring."""
    try:
        p = subprocess.run(["rclone", *args], capture_output=True, text=True,
                           input=input)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", "rclone not found on PATH"


def resolve_bind():
    """Tailnet IPv4 via `tailscale ip -4`; CRED_BROKER_BIND overrides; refuse
    all-interfaces binds unless CRED_BROKER_BIND_ANY=1 (§3)."""
    b = os.environ.get("CRED_BROKER_BIND")
    if b:
        if b in ("0.0.0.0", "::", "") and \
                os.environ.get("CRED_BROKER_BIND_ANY") != "1":
            sys.exit("error: refusing to bind all interfaces "
                     "(set CRED_BROKER_BIND_ANY=1 to override)")
        return b
    try:
        p = subprocess.run(["tailscale", "ip", "-4"], capture_output=True,
                           text=True, timeout=10)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    sys.exit("error: cannot determine tailnet IPv4 (`tailscale ip -4` "
             "failed); set CRED_BROKER_BIND")


# ------------------------------------------------------------------ broker #
def _authorized_role(inst, registry, iid):
    """The role a box is ALLOWED to claim — never the one it asserts (§4.1).
    Authority order: launch-recorded CRED_ROLE from extra_env (shipped by
    herdd at launch, box can't rewrite it), then an admin-registered role
    (POST /v1/register, for pre-CRED_ROLE boxes). '' when neither is
    recorded -> the request is denied."""
    r = str(credbroker._instance_env(inst or {}).get("CRED_ROLE") or "")
    if r or registry is None:
        return r
    return str((registry.data.get(str(iid)) or {}).get("role") or "")


class CredreqLog(credbroker._JsonFile):
    """PERSISTED B2-lane replay state (state dir — survives daemon restarts):
    'seen' = {"<iid>|<ts>": seen_at} dedupe entries, pruned past 2x the replay
    window; 'issued_ts' = {"<iid>": ts} high-water mark of the last credreq ts
    that MINTED. Persistence + the monotonic issued-ts bind mean a
    world-readable credreq replayed across a restart (or re-written by
    another jobs box) can never re-mint — and therefore never revoke — a
    newer active key."""

    def __init__(self, path=None):
        super().__init__(path or os.path.join(credbroker.state_dir(),
                                              "credreq_seen.json"))
        self.data.setdefault("seen", {})
        self.data.setdefault("issued_ts", {})

    def seen(self, iid, ts):
        return f"{iid}|{ts}" in self.data["seen"]

    def mark(self, iid, ts, now):
        self.data["seen"][f"{iid}|{ts}"] = now
        self.save()

    def issued_ts(self, iid):
        return int(self.data["issued_ts"].get(str(iid), 0))

    def mark_issued(self, iid, ts):
        k = str(iid)
        self.data["issued_ts"][k] = max(int(ts),
                                        int(self.data["issued_ts"].get(k, 0)))
        self.save()

    def prune(self, cutoff):
        stale = [k for k, v in self.data["seen"].items() if v < cutoff]
        for k in stale:
            del self.data["seen"][k]
        if stale:
            self.save()


class Broker:
    """Request handling shared by both transports. One lock serializes the
    file-backed state (registry / ratelimit / credreq dedupe) across the
    ThreadingHTTPServer worker threads and the B2 poll thread.

    fetch_instances / mint are injectable (tests fake both; production binds
    fetch_vast_instances + credbroker.issue_keys)."""

    def __init__(self, fetch_instances=None, mint=None):
        self.fetch_instances = fetch_instances or fetch_vast_instances
        self.mint = mint or credbroker.issue_keys
        self.registry = credbroker.Registry()
        self.ratelimit = credbroker.RateLimiter()
        self.lock = threading.Lock()
        self.credreqs = CredreqLog()  # persisted (iid, ts) replay dedupe

    def _audit(self, transport, remote, iid, role, verdict, reason=None,
               **extra):
        rec = {"transport": transport, "remote": remote, "instance_id": iid,
               "role": role, "verdict": verdict}
        if reason:
            rec["reason"] = reason
        rec.update(extra)
        try:
            credbroker.audit_append(rec)
        except Exception:
            pass                      # audit failure never blocks a verdict

    def handle_creds(self, body, remote, transport="http"):
        """POST /v1/creds → (status, response-body). §2.2: 400 malformed,
        uniform 403 on ANY verification failure, 429 rate limit, 200 issue."""
        if not isinstance(body, dict):
            return 400, {"error": "bad request"}
        iid = body.get("instance_id")
        nonce = body.get("nonce")
        role = body.get("role")
        want = body.get("want") or {}
        if (not isinstance(iid, int) or isinstance(iid, bool)
                or not isinstance(nonce, str) or not isinstance(role, str)
                or not isinstance(want, dict)):
            return 400, {"error": "bad request"}
        wp = want.get("write_prefix")
        if wp is not None and not isinstance(wp, str):
            return 400, {"error": "bad request"}
        with self.lock:
            try:
                instances = list(self.fetch_instances() or [])
            except Exception:
                self._audit(transport, remote, iid, role, "denied",
                            "instances_unavailable")
                return 403, dict(DENY_BODY)
            ok, why = credbroker.verify_instance(
                iid, nonce, lambda: instances, self.registry)
            if not ok:
                self._audit(transport, remote, iid, role, "denied", why)
                return 403, dict(DENY_BODY)
            inst = next((i for i in instances if i.get("id") == iid), None)
            granted = _authorized_role(inst, self.registry, iid)
            if not granted or role != granted:
                # role is box-asserted in the body; the launch-recorded
                # CRED_ROLE (or admin-registered role) is authoritative —
                # a jobs box asserting 'train' must never get a bucket-wide
                # write key (§4.1)
                self._audit(transport, remote, iid, role, "denied",
                            "role_mismatch")
                return 403, dict(DENY_BODY)
            ok, prefix = credbroker.check_policy(role, wp)
            if not ok:
                self._audit(transport, remote, iid, role, "denied", "policy")
                return 403, dict(DENY_BODY)
            ok, why = self.ratelimit.try_acquire(iid)
            if not ok:
                self._audit(transport, remote, iid, role, "rate_limited", why)
                return 429, {"error": "rate limited"}
            try:
                resp = self.mint(iid, role, prefix,
                                 credbroker.ephemeral_hours(),
                                 registry=self.registry)
            except Exception as e:
                self._audit(transport, remote, iid, role, "denied",
                            "mint_failed:" + type(e).__name__)
                return 403, dict(DENY_BODY)
            self._audit(transport, remote, iid, role, "issued",
                        keys=self.registry.last_keys(iid),
                        expires_at=resp.get("expires_at"))
            return 200, resp

    def handle_register(self, token, body, remote):
        """POST /v1/register → (status, body). Constant-time admin-token
        check; registers sha256(nonce) for a pre-nonce box (§2.2), plus an
        OPTIONAL 'role' — the admin-asserted role authority for boxes with no
        launch-recorded CRED_ROLE (without it, such a box can register a
        nonce but /v1/creds denies with role_mismatch)."""
        admin = os.environ.get("CRED_BROKER_ADMIN_TOKEN") or ""
        if not admin or not hmac.compare_digest(
                str(token or "").encode(), admin.encode()):
            self._audit("http", remote, None, None, "register_denied")
            return 401, {"error": "unauthorized"}
        if not isinstance(body, dict):
            return 400, {"error": "bad request"}
        iid = body.get("instance_id")
        ns = body.get("nonce_sha256")
        role = body.get("role")
        if (not isinstance(iid, int) or isinstance(iid, bool)
                or not isinstance(ns, str) or len(ns) != 64
                or any(c not in "0123456789abcdefABCDEF" for c in ns)):
            return 400, {"error": "bad request"}
        if role is not None and role not in credbroker.ROLE_WRITE_PREFIX:
            return 400, {"error": "bad request"}
        with self.lock:
            self.registry.register_nonce(iid, ns)
            if role:
                self.registry._ent(iid)["role"] = role
                self.registry.save()
        self._audit("http", remote, iid, role, "registered")
        return 200, {"ok": True}


# ---------------------------------------------------- B2-mediated lane (§2.4) #
def b2_sweep(broker, runner, bucket=None, now=None):
    """One poll pass over jobs/nodes/*/credreq. Pure given (broker, runner,
    now) — tests drive it directly with an in-memory runner. Returns
    [(iid_str, verdict), ...] for everything acted on this pass.

    The lane trusts NOTHING on the bucket: requests are HMAC-proved with the
    launch nonce (from extra_env, fetched once per sweep), role-bound to the
    launch-recorded CRED_ROLE, and replay-bounded (ts_fresh + PERSISTED
    (iid, ts) dedupe + monotonic issued-ts — see CredreqLog); responses go
    out ONLY as sealed envelopes — neither the raw nonce nor unencrypted key
    material is ever written to the bucket."""
    bucket = bucket or os.environ.get("B2_BUCKET", "")
    now = time.time() if now is None else now
    out = []
    rc, listing, _ = runner(["lsf", "-R", f"b2:{bucket}/jobs/nodes/",
                             "--include", "*/credreq"], input=None)
    if rc != 0:
        return out
    rels = [ln.strip() for ln in listing.splitlines()
            if ln.strip().endswith("/credreq")]
    if not rels:
        return out
    try:
        instances = broker.fetch_instances() or []
    except Exception:
        return out                       # vast API blip: retry next sweep
    by_id = {i.get("id"): i for i in instances}
    for rel in rels:
        iid_s = rel.split("/", 1)[0]
        v = _sweep_one(broker, runner, bucket, iid_s, by_id, now)
        if v is not None:
            out.append((iid_s, v))
    return out


def _sweep_one(broker, runner, bucket, iid_s, by_id, now):
    """Handle one credreq object; returns a verdict string or None (already
    handled — silent skip so a lingering credreq doesn't spam the audit)."""
    def deny(iid, role, reason, remember=True):
        if remember:
            broker.credreqs.mark(iid, ts, now)
        broker._audit("b2", f"jobs/nodes/{iid_s}", iid, role, "denied", reason)
        return "denied"

    rc, body, _ = runner(["cat", f"b2:{bucket}/jobs/nodes/{iid_s}/credreq"],
                         input=None)
    if rc != 0:
        return "cat_failed"
    ts = None
    try:
        req = json.loads(body)
        iid = int(req["instance_id"])
        ts = int(req["ts"])
        role = str(req["role"])
        proof = str(req["proof"])
        want = (req.get("want") or {}).get("write_prefix")
    except Exception:
        broker._audit("b2", f"jobs/nodes/{iid_s}", None, None, "denied",
                      "malformed")
        return "denied"
    with broker.lock:
        if broker.credreqs.seen(iid, ts):
            return None
        # prune dedupe entries safely past the replay window
        broker.credreqs.prune(now - 2 * credbroker.REPLAY_WINDOW_S)
        if str(iid) != iid_s:
            return deny(iid, role, "iid_path_mismatch")
        if not credbroker.ts_fresh(ts, now=now):
            return deny(iid, role, "stale_ts")
        if ts <= broker.credreqs.issued_ts(iid):
            # at-or-before the last ts that MINTED for this iid: a replay
            # (e.g. re-written by another box across a daemon restart, when
            # the seen-set alone wouldn't remember it) — must never re-mint,
            # which would revoke the box's newer active brk-key
            return deny(iid, role, "replayed_ts")
        inst = by_id.get(iid)
        if inst is None:
            return deny(iid, role, "unknown_instance")
        if (inst.get("actual_status") or "").lower() \
                not in credbroker.LIVE_STATES:
            return deny(iid, role, "not_live")
        nonce = str(credbroker._instance_env(inst)
                    .get("BOX_IDENTITY_NONCE") or "")
        if not nonce:
            # registry sha256 can't key an HMAC proof — B2 lane needs the
            # launch nonce itself (§2.4); pre-nonce boxes must use HTTP
            return deny(iid, role, "no_launch_nonce")
        if not credbroker.verify_credreq_proof(nonce, iid, ts, role, proof):
            return deny(iid, role, "bad_proof")
        granted = _authorized_role(inst, broker.registry, iid)
        if not granted or role != granted:
            # the proof only shows the box holds ITS OWN nonce — the role
            # inside the credreq is still box-asserted; bind it to the
            # launch-recorded CRED_ROLE (§4.1)
            return deny(iid, role, "role_mismatch")
        ok, prefix = credbroker.check_policy(role, want)
        if not ok:
            return deny(iid, role, "policy")
        ok, why = broker.ratelimit.try_acquire(iid, now=now)
        if not ok:
            # NOT remembered: spacing may clear by the next sweep
            return deny(iid, role, why, remember=False)
        try:
            resp = broker.mint(iid, role, prefix,
                               credbroker.ephemeral_hours(),
                               registry=broker.registry)
        except Exception as e:
            return deny(iid, role, "mint_failed:" + type(e).__name__)
        sealed = credbroker.seal_envelope(nonce, resp, ts=int(now))
        rc, _, err = runner(
            ["rcat", f"b2:{bucket}/jobs/nodes/{iid}/creds"],
            input=json.dumps(sealed, sort_keys=True) + "\n")
        if rc != 0:
            # keys minted but undeliverable; retry next credreq (new ts) —
            # issued-ts still advances so a replay of THIS ts can't re-mint
            broker.credreqs.mark(iid, ts, now)
            broker.credreqs.mark_issued(iid, ts)
            broker._audit("b2", f"jobs/nodes/{iid_s}", iid, role, "denied",
                          "rcat_failed")
            return "rcat_failed"
        broker.credreqs.mark(iid, ts, now)
        broker.credreqs.mark_issued(iid, ts)
        broker._audit("b2", f"jobs/nodes/{iid_s}", iid, role, "issued",
                      keys=broker.registry.last_keys(iid),
                      expires_at=resp.get("expires_at"))
        return "issued"


def _b2_poll_loop(broker, stop, interval=B2_POLL_INTERVAL_S):
    """Daemon-thread loop: sweep, then wait; any sweep exception is logged
    and the loop keeps going."""
    while not stop.is_set():
        try:
            for iid_s, verdict in b2_sweep(broker, _rclone_runner):
                _log({"transport": "b2", "instance": iid_s,
                      "verdict": verdict})
        except Exception as e:
            _log({"event": "b2_sweep_error", "error": str(e)[:200]})
        if stop.wait(interval):
            break


# ------------------------------------------------------------ HTTP surface #
class Handler(BaseHTTPRequestHandler):
    server_version = "credbrokerd/1"

    def log_message(self, *a):
        pass                              # replaced by _log structured lines

    def _read_json(self):
        """Parse the request body; None on anything malformed/oversized."""
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if n <= 0 or n > MAX_BODY:
            return None
        try:
            return json.loads(self.rfile.read(n).decode())
        except Exception:
            return None

    def _send(self, status, obj, verdict=None):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except Exception:
            pass
        rec = {"transport": "http", "remote": self.client_address[0],
               "method": self.command, "path": self.path, "status": status}
        if verdict:
            rec["verdict"] = verdict
        _log(rec)

    def do_GET(self):
        if self.path == "/v1/health":
            self.server.broker._audit("http", self.client_address[0], None,
                                      None, "health")
            self._send(200, {"ok": True, "version": 1})
        else:
            self.server.broker._audit("http", self.client_address[0], None,
                                      None, "not_found", self.path)
            self._send(404, {"error": "not found"})

    def do_POST(self):
        broker = self.server.broker
        remote = self.client_address[0]
        if self.path == "/v1/creds":
            body = self._read_json()
            if body is None:
                broker._audit("http", remote, None, None, "bad_request")
                self._send(400, {"error": "bad request"})
                return
            status, obj = broker.handle_creds(body, remote)
            self._send(status, obj,
                       verdict="issued" if status == 200 else "denied")
        elif self.path == "/v1/register":
            body = self._read_json()
            if body is None:
                broker._audit("http", remote, None, None, "bad_request")
                self._send(400, {"error": "bad request"})
                return
            status, obj = broker.handle_register(
                self.headers.get("X-Broker-Admin"), body, remote)
            self._send(status, obj)
        else:
            broker._audit("http", remote, None, None, "not_found", self.path)
            self._send(404, {"error": "not found"})


def make_server(bind, port, broker):
    srv = ThreadingHTTPServer((bind, port), Handler)
    srv.daemon_threads = True
    srv.broker = broker
    return srv


# -------------------------------------------------------------------- CLI #
def cmd_serve(args):
    bind = resolve_bind()
    port = int(os.environ.get("CRED_BROKER_PORT", DEFAULT_PORT))
    broker = Broker()
    server = make_server(bind, port, broker)
    stop = threading.Event()
    if not args.no_b2_lane:
        threading.Thread(target=_b2_poll_loop,
                         args=(broker, stop, args.b2_interval),
                         daemon=True).start()

    def _sig(signum, frame):
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    _log({"event": "serve", "bind": bind, "port": port,
          "b2_lane": not args.no_b2_lane})
    server.serve_forever()
    server.server_close()
    _log({"event": "shutdown"})
    return 0


UNIT_TEMPLATE = """\
[Unit]
Description=upstream-monorepo credential broker (tailnet-only)
After=network-online.target

[Service]
ExecStart={python} {script} serve
WorkingDirectory={repo}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""


def cmd_install_unit(args):
    """Generate the systemd user unit AT RUNTIME — the absolute paths it
    embeds (this interpreter, this checkout) never enter git (§3)."""
    script = os.path.abspath(__file__)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(script)))
    unit_dir = os.path.expanduser("~/.config/systemd/user")
    os.makedirs(unit_dir, exist_ok=True)
    path = os.path.join(unit_dir, "credbrokerd.service")
    with open(path, "w") as f:
        f.write(UNIT_TEMPLATE.format(python=sys.executable, script=script,
                                     repo=repo))
    print(f"wrote {path}")
    print("next steps:")
    print("  systemctl --user daemon-reload")
    print("  systemctl --user enable --now credbrokerd.service")
    print("  loginctl enable-linger $USER   # keep it running after logout")
    print("  # health check: curl http://$(tailscale ip -4):"
          + os.environ.get("CRED_BROKER_PORT", str(DEFAULT_PORT))
          + "/v1/health")
    return 0


def main(argv=None):
    b2_mint_key.load_env()
    ap = argparse.ArgumentParser(
        description="cred-broker daemon (see docs/plans/"
                    "cred-broker-buildout.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the broker HTTP server")
    s.add_argument("--no-b2-lane", action="store_true",
                   help="disable the B2-mediated credreq poll thread")
    s.add_argument("--b2-interval", type=float, default=B2_POLL_INTERVAL_S,
                   help="B2 credreq poll interval, seconds (default 60)")
    s.set_defaults(fn=cmd_serve)
    u = sub.add_parser("install-unit",
                       help="generate the systemd user unit (runtime paths)")
    u.set_defaults(fn=cmd_install_unit)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
