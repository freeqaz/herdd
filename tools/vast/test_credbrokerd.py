"""Unit tests for credbrokerd.py — real HTTP server on 127.0.0.1:0 with faked
core deps (fetch_instances / mint); B2 sweep driven as a pure function with an
in-memory rclone runner. No network, no creds."""
import io
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import credbroker as cb
import credbrokerd as cbd

NONCE = "ab" * 16
IID = 42


def _inst(iid=IID, status="running", nonce=NONCE, cred_role="jobs"):
    ee = [["FOO", "1"]] + ([["BOX_IDENTITY_NONCE", nonce]] if nonce else [])
    if cred_role:
        ee.append(["CRED_ROLE", cred_role])
    return {"id": iid, "actual_status": status, "extra_env": ee}


def _fake_mint(counter=None):
    """issue_keys-shaped fake: returns a response carrying sentinel secrets
    so leak assertions can grep for 'sekrit'."""
    def mint(instance_id, role, write_prefix, hours, registry=None, now=None):
        if counter is not None:
            counter.append((instance_id, role, write_prefix))
        if registry is not None:
            registry.set_last_keys(instance_id, [
                {"name": f"box-{instance_id}-brk1-ro", "key_id": "kid1"}])
        b2 = {"key_id": "kid1", "application_key": "sekrit1"}
        if write_prefix:
            b2.update({"write_key_id": "kid2",
                       "write_application_key": "sekrit2"})
        return {"b2": b2, "bucket": "bkt", "s3_endpoint": "https://s3.fake",
                "region": "us-west-004", "expires_at": 1234567890}
    return mint


class Srv:
    def __init__(self, url, broker, log, instances, mints, tmp):
        self.url = url
        self.broker = broker
        self.log = log
        self.instances = instances
        self.mints = mints
        self.tmp = tmp

    def audit_text(self):
        p = os.path.join(str(self.tmp), "audit.jsonl")
        return open(p).read() if os.path.isfile(p) else ""


@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("CRED_BROKER_STATE", str(tmp_path))
    monkeypatch.setenv("CRED_BROKER_ADMIN_TOKEN", "admintok")
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.delenv("B2_EPHEMERAL_HOURS", raising=False)
    log = io.StringIO()
    monkeypatch.setattr(cbd, "LOG_STREAM", log)
    instances = [_inst()]
    mints = []
    broker = cbd.Broker(fetch_instances=lambda: list(instances),
                        mint=_fake_mint(mints))
    server = cbd.make_server("127.0.0.1", 0, broker)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield Srv(f"http://127.0.0.1:{server.server_address[1]}", broker, log,
              instances, mints, tmp_path)
    server.shutdown()
    server.server_close()
    t.join(timeout=5)


def _req(url, body=None, headers=None, method=None):
    """(status, parsed-json). Raw bytes body passes through unencoded."""
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
    r = urllib.request.Request(url, data=data, headers=headers or {},
                               method=method)
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def _creds_body(iid=IID, nonce=NONCE, role="jobs", want=None):
    b = {"instance_id": iid, "nonce": nonce, "role": role}
    if want is not None:
        b["want"] = want
    return b


# -------------------------------------------------------------------- HTTP #
def test_health(srv):
    status, body = _req(srv.url + "/v1/health")
    assert (status, body) == (200, {"ok": True, "version": 1})


def test_unknown_path_404(srv):
    assert _req(srv.url + "/nope")[0] == 404
    assert _req(srv.url + "/v1/other", body={})[0] == 404


def test_creds_happy_path(srv):
    status, body = _req(srv.url + "/v1/creds",
                        body=_creds_body(want={"write_prefix": "jobs/J1/"}))
    assert status == 200
    assert body["b2"]["application_key"] == "sekrit1"
    assert body["b2"]["write_key_id"] == "kid2"
    assert body["bucket"] == "bkt"
    assert srv.mints == [(IID, "jobs", "jobs/J1/")]
    # verdict is on the audit trail; no secret anywhere but the response
    assert '"verdict": "issued"' in srv.audit_text()


def test_creds_wrong_nonce_uniform_403(srv):
    status, body = _req(srv.url + "/v1/creds",
                        body=_creds_body(nonce="cd" * 16))
    assert (status, body) == (403, {"error": "verification failed"})
    assert srv.mints == []


def test_creds_dead_instance_same_403_body(srv):
    srv.instances[0]["actual_status"] = "exited"
    s1, b1 = _req(srv.url + "/v1/creds", body=_creds_body())
    srv.instances[0]["actual_status"] = "running"
    s2, b2 = _req(srv.url + "/v1/creds", body=_creds_body(nonce="cd" * 16))
    # uniform deny: dead instance and bad nonce are indistinguishable (§4.1)
    assert (s1, b1) == (403, {"error": "verification failed"})
    assert (s1, b1) == (s2, b2)


def test_creds_bad_policy_403(srv):
    status, body = _req(srv.url + "/v1/creds",
                        body=_creds_body(want={"write_prefix": "serve/"}))
    assert (status, body) == (403, {"error": "verification failed"})


def test_creds_role_escalation_uniform_403(srv):
    # launch-recorded CRED_ROLE=jobs; asserting train (bucket-wide key) or
    # serve (another role's namespace) must be a uniform 403 — the role in
    # the body is box-asserted and never trusted (§4.1)
    for role, want in (("train", None), ("serve", {"write_prefix": "serve/"})):
        status, body = _req(srv.url + "/v1/creds",
                            body=_creds_body(role=role, want=want))
        assert (status, body) == (403, {"error": "verification failed"})
    assert srv.mints == []


def test_creds_no_recorded_role_uniform_403(srv):
    # valid nonce but the box has NO launch CRED_ROLE and no registered role
    srv.instances[0] = _inst(cred_role=None)
    status, body = _req(srv.url + "/v1/creds", body=_creds_body())
    assert (status, body) == (403, {"error": "verification failed"})
    assert srv.mints == []


def test_creds_rate_limited_429(srv):
    assert _req(srv.url + "/v1/creds", body=_creds_body())[0] == 200
    status, body = _req(srv.url + "/v1/creds", body=_creds_body())
    assert status == 429
    assert "rate" in body["error"]
    assert len(srv.mints) == 1


def test_creds_malformed_400(srv):
    assert _req(srv.url + "/v1/creds", body=b"{not json")[0] == 400
    assert _req(srv.url + "/v1/creds", body=b"")[0] == 400
    # wrong types are 400, not 403
    assert _req(srv.url + "/v1/creds",
                body={"instance_id": "42", "nonce": NONCE,
                      "role": "jobs"})[0] == 400
    assert _req(srv.url + "/v1/creds",
                body=_creds_body(want={"write_prefix": 7}))[0] == 400
    assert srv.mints == []


def test_creds_mint_failure_uniform_403(srv):
    def boom(*a, **k):
        raise RuntimeError("b2 down")
    srv.broker.mint = boom
    status, body = _req(srv.url + "/v1/creds", body=_creds_body())
    assert (status, body) == (403, {"error": "verification failed"})


def test_register_bad_token_401(srv):
    body = {"instance_id": 7, "nonce_sha256": "0" * 64}
    assert _req(srv.url + "/v1/register", body=body)[0] == 401
    assert _req(srv.url + "/v1/register", body=body,
                headers={"X-Broker-Admin": "wrong"})[0] == 401
    assert srv.broker.registry.nonce_sha256(7) is None


def test_register_good_token_then_creds_via_registry(srv):
    import hashlib
    # pre-nonce-injection box: no launch nonce AND no launch CRED_ROLE —
    # the admin registration supplies both authorities
    srv.instances[0] = _inst(nonce=None, cred_role=None)
    body = {"instance_id": IID, "role": "jobs",
            "nonce_sha256": hashlib.sha256(NONCE.encode()).hexdigest()}
    status, resp = _req(srv.url + "/v1/register", body=body,
                        headers={"X-Broker-Admin": "admintok"})
    assert (status, resp) == (200, {"ok": True})
    assert _req(srv.url + "/v1/creds", body=_creds_body())[0] == 200


def test_register_without_role_denies_creds(srv):
    import hashlib
    # nonce registered but NO role authority anywhere -> the asserted role
    # is never trusted on its own
    srv.instances[0] = _inst(nonce=None, cred_role=None)
    body = {"instance_id": IID,
            "nonce_sha256": hashlib.sha256(NONCE.encode()).hexdigest()}
    assert _req(srv.url + "/v1/register", body=body,
                headers={"X-Broker-Admin": "admintok"})[0] == 200
    status, resp = _req(srv.url + "/v1/creds", body=_creds_body())
    assert (status, resp) == (403, {"error": "verification failed"})
    assert srv.mints == []


def test_register_role_binds_registry_lane(srv):
    import hashlib
    srv.instances[0] = _inst(nonce=None, cred_role=None)
    body = {"instance_id": IID, "role": "jobs",
            "nonce_sha256": hashlib.sha256(NONCE.encode()).hexdigest()}
    assert _req(srv.url + "/v1/register", body=body,
                headers={"X-Broker-Admin": "admintok"})[0] == 200
    # registered role=jobs: asserting train is still a uniform 403
    status, resp = _req(srv.url + "/v1/creds", body=_creds_body(role="train"))
    assert (status, resp) == (403, {"error": "verification failed"})
    assert srv.mints == []


def test_register_malformed_400(srv):
    hdr = {"X-Broker-Admin": "admintok"}
    assert _req(srv.url + "/v1/register", body=b"junk", headers=hdr)[0] == 400
    assert _req(srv.url + "/v1/register", headers=hdr,
                body={"instance_id": 7, "nonce_sha256": "xyz"})[0] == 400
    # unknown role is 400, not silently registered
    assert _req(srv.url + "/v1/register", headers=hdr,
                body={"instance_id": 7, "nonce_sha256": "0" * 64,
                      "role": "root"})[0] == 400


def test_no_key_material_in_logs_or_audit(srv):
    # exercise every verdict path, then grep the whole exhaust
    _req(srv.url + "/v1/health")
    assert _req(srv.url + "/v1/creds",
                body=_creds_body(want={"write_prefix": "jobs/J1/"}))[0] == 200
    _req(srv.url + "/v1/creds", body=_creds_body(nonce="cd" * 16))
    _req(srv.url + "/v1/creds", body=_creds_body())            # 429
    _req(srv.url + "/v1/register",
         body={"instance_id": 7, "nonce_sha256": "0" * 64},
         headers={"X-Broker-Admin": "admintok"})
    _req(srv.url + "/v1/creds", body=b"{oops")
    stderr = srv.log.getvalue()
    audit = srv.audit_text()
    assert stderr and audit
    for exhaust in (stderr, audit):
        assert "sekrit" not in exhaust
        assert "application_key" not in exhaust
    # ids/names survive in the audit trail (accountability without secrets)
    assert "kid1" in audit


def _log_lines(srv, n, timeout=10):
    """The server's first `n` log lines, waited for rather than assumed.

    `_req` returns when the CLIENT has read the response; the handler writes
    its structured line on the server THREAD, so on a loaded box the read can
    win the race and the line is not there yet. Waiting on the count makes the
    assertion about behaviour instead of about scheduling.
    """
    end = time.time() + timeout
    while True:
        raw = srv.log.getvalue().splitlines()
        if len(raw) >= n or time.time() >= end:
            return [json.loads(l) for l in raw]
        time.sleep(0.05)


def test_stderr_one_line_per_request(srv):
    _req(srv.url + "/v1/health")
    _req(srv.url + "/nope")
    lines = _log_lines(srv, 2)
    assert len(lines) == 2
    assert lines[0]["path"] == "/v1/health" and lines[0]["status"] == 200
    assert lines[1]["status"] == 404
    assert all(l["transport"] == "http" and l["ts"] for l in lines)


# ------------------------------------------------------------ bind + unit #
def test_resolve_bind_env_override(monkeypatch):
    monkeypatch.setenv("CRED_BROKER_BIND", "100.64.0.9")
    assert cbd.resolve_bind() == "100.64.0.9"


def test_resolve_bind_refuses_any(monkeypatch):
    monkeypatch.setenv("CRED_BROKER_BIND", "0.0.0.0")
    monkeypatch.delenv("CRED_BROKER_BIND_ANY", raising=False)
    with pytest.raises(SystemExit):
        cbd.resolve_bind()
    monkeypatch.setenv("CRED_BROKER_BIND_ANY", "1")
    assert cbd.resolve_bind() == "0.0.0.0"


def test_install_unit_generated_paths(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cbd.cmd_install_unit(None) == 0
    unit = open(tmp_path / ".config/systemd/user/credbrokerd.service").read()
    assert f"ExecStart={sys.executable} " in unit
    assert unit.count("credbrokerd.py serve") == 1
    assert "Restart=on-failure" in unit and "RestartSec=5" in unit
    assert "WorkingDirectory=" in unit
    out = capsys.readouterr().out
    assert "daemon-reload" in out and "enable-linger" in out


# ---------------------------------------------------------- B2 sweep (§2.4) #
class FakeRunner:
    """In-memory rclone: objects keyed by bucket-relative path; records every
    op. Contract mirrors runmeta._default_runner: (rc, stdout, stderr)."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.ops = []

    @staticmethod
    def _key(args):
        remote = next(a for a in args if a.startswith("b2:"))
        return remote.split("/", 1)[1]           # strip 'b2:bkt/'

    def __call__(self, args, input=None):
        self.ops.append((args[0], self._key(args)))
        if args[0] == "lsf":
            pre = self._key(args)
            hits = [k[len(pre):] for k in sorted(self.objects)
                    if k.startswith(pre) and k.endswith("/credreq")]
            return 0, "".join(h + "\n" for h in hits), ""
        if args[0] == "cat":
            k = self._key(args)
            if k not in self.objects:
                return 3, "", "not found"
            return 0, self.objects[k], ""
        if args[0] == "rcat":
            self.objects[self._key(args)] = input
            return 0, "", ""
        raise AssertionError(f"unexpected rclone op {args}")


@pytest.fixture
def bro(tmp_path, monkeypatch):
    monkeypatch.setenv("CRED_BROKER_STATE", str(tmp_path))
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.delenv("B2_EPHEMERAL_HOURS", raising=False)
    monkeypatch.setattr(cbd, "LOG_STREAM", io.StringIO())
    mints = []
    b = cbd.Broker(fetch_instances=lambda: [_inst()],
                   mint=_fake_mint(mints))
    b._mints = mints
    return b


NOW = 1_700_000_000


def _credreq(iid=IID, ts=NOW - 5, role="jobs", nonce=NONCE, proof=None,
             want=None):
    return json.dumps({
        "instance_id": iid, "ts": ts, "role": role, "want": want,
        "proof": proof if proof is not None
        else cb.make_credreq_proof(nonce, iid, ts, role)})


def test_sweep_happy_path_seals_creds(bro):
    r = FakeRunner({f"jobs/nodes/{IID}/credreq": _credreq()})
    assert cbd.b2_sweep(bro, r, now=NOW) == [(str(IID), "issued")]
    blob_text = r.objects[f"jobs/nodes/{IID}/creds"]
    # never on the bucket: the raw nonce or unencrypted key material
    assert NONCE not in blob_text and "sekrit" not in blob_text
    resp = cb.open_envelope(NONCE, json.loads(blob_text))
    assert resp["b2"]["application_key"] == "sekrit1"
    assert resp["b2"]["write_key_id"] == "kid2"    # jobs -> scoped pair
    assert bro._mints == [(IID, "jobs", "jobs/")]


def test_sweep_empty_listing_skips_instance_fetch(bro):
    calls = []
    bro.fetch_instances = lambda: calls.append(1) or [_inst()]
    assert cbd.b2_sweep(bro, FakeRunner(), now=NOW) == []
    assert calls == []                     # fetch once per sweep, only if work


def test_sweep_bad_proof_no_creds(bro):
    r = FakeRunner({f"jobs/nodes/{IID}/credreq":
                    _credreq(proof="0" * 64)})
    assert cbd.b2_sweep(bro, r, now=NOW) == [(str(IID), "denied")]
    assert f"jobs/nodes/{IID}/creds" not in r.objects
    assert bro._mints == []


def test_sweep_forged_nonce_proof_rejected(bro):
    # a rogue jobs box forges a credreq for iid 42 without 42's nonce
    r = FakeRunner({f"jobs/nodes/{IID}/credreq":
                    _credreq(nonce="cd" * 16)})
    assert cbd.b2_sweep(bro, r, now=NOW) == [(str(IID), "denied")]
    assert f"jobs/nodes/{IID}/creds" not in r.objects


def test_sweep_stale_ts_rejected(bro):
    ts = NOW - 700                                   # outside the 600 s window
    r = FakeRunner({f"jobs/nodes/{IID}/credreq": _credreq(ts=ts)})
    assert cbd.b2_sweep(bro, r, now=NOW) == [(str(IID), "denied")]
    assert f"jobs/nodes/{IID}/creds" not in r.objects


def test_sweep_replay_dedupe(bro):
    r = FakeRunner({f"jobs/nodes/{IID}/credreq": _credreq()})
    assert cbd.b2_sweep(bro, r, now=NOW) == [(str(IID), "issued")]
    # same credreq still on the bucket next sweep: no re-mint, no audit spam
    assert cbd.b2_sweep(bro, r, now=NOW + 70) == []
    assert len(bro._mints) == 1
    # a FRESH request (new ts) from the same box mints again
    r.objects[f"jobs/nodes/{IID}/credreq"] = _credreq(ts=NOW + 130)
    assert cbd.b2_sweep(bro, r, now=NOW + 135) == [(str(IID), "issued")]
    assert len(bro._mints) == 2


def test_sweep_dead_instance_and_no_nonce_denied(bro):
    bro.fetch_instances = lambda: [_inst(status="exited")]
    r = FakeRunner({f"jobs/nodes/{IID}/credreq": _credreq()})
    assert cbd.b2_sweep(bro, r, now=NOW) == [(str(IID), "denied")]
    bro.fetch_instances = lambda: [_inst(iid=77, nonce=None)]
    r2 = FakeRunner({"jobs/nodes/77/credreq": _credreq(iid=77)})
    # no launch nonce -> B2 lane cannot verify or seal; deny
    assert cbd.b2_sweep(bro, r2, now=NOW) == [("77", "denied")]
    assert bro._mints == []


def test_sweep_iid_path_mismatch_denied(bro):
    # credreq for iid 42 planted under another box's directory
    r = FakeRunner({"jobs/nodes/99/credreq": _credreq()})
    assert cbd.b2_sweep(bro, r, now=NOW) == [("99", "denied")]
    assert bro._mints == []


def test_sweep_malformed_credreq(bro):
    r = FakeRunner({f"jobs/nodes/{IID}/credreq": "{not json"})
    assert cbd.b2_sweep(bro, r, now=NOW) == [(str(IID), "denied")]


def test_sweep_ratelimit_not_remembered(bro):
    # exhaust the spacing window via a prior HTTP-side issue
    bro.ratelimit.try_acquire(IID, now=NOW - 10)
    r = FakeRunner({f"jobs/nodes/{IID}/credreq": _credreq()})
    assert cbd.b2_sweep(bro, r, now=NOW) == [(str(IID), "denied")]
    # spacing clears -> the SAME credreq succeeds on a later sweep
    assert cbd.b2_sweep(bro, r, now=NOW + 60) == [(str(IID), "issued")]


def test_sweep_listing_failure_soft(bro):
    def down(args, input=None):
        return 1, "", "b2 down"
    assert cbd.b2_sweep(bro, down, now=NOW) == []


def test_sweep_role_escalation_denied(bro):
    # VALID proof under the box's OWN nonce, but role=train while the launch
    # recorded CRED_ROLE=jobs — the proof authenticates the box, not the
    # role; no bucket-wide key comes back
    r = FakeRunner({f"jobs/nodes/{IID}/credreq": _credreq(role="train")})
    assert cbd.b2_sweep(bro, r, now=NOW) == [(str(IID), "denied")]
    assert f"jobs/nodes/{IID}/creds" not in r.objects
    assert bro._mints == []


def test_sweep_replay_across_restart_no_remint(bro):
    r = FakeRunner({f"jobs/nodes/{IID}/credreq": _credreq()})
    assert cbd.b2_sweep(bro, r, now=NOW) == [(str(IID), "issued")]
    # daemon restart: fresh Broker over the SAME state dir; the lingering
    # credreq (still inside the 600 s window) must not re-mint — a re-mint
    # would revoke the box's active key out from under it
    mints2 = []
    bro2 = cbd.Broker(fetch_instances=lambda: [_inst()],
                      mint=_fake_mint(mints2))
    assert cbd.b2_sweep(bro2, r, now=NOW + 70) == []
    assert mints2 == []


def test_sweep_older_ts_replay_cannot_revoke(bro):
    assert cbd.b2_sweep(
        bro, FakeRunner({f"jobs/nodes/{IID}/credreq": _credreq(ts=NOW - 5)}),
        now=NOW) == [(str(IID), "issued")]
    # attacker re-writes a captured OLDER credreq (unseen (iid, ts), still
    # inside the freshness window): monotonic issued-ts bind denies it
    r = FakeRunner({f"jobs/nodes/{IID}/credreq": _credreq(ts=NOW - 50)})
    assert cbd.b2_sweep(bro, r, now=NOW + 70) == [(str(IID), "denied")]
    assert f"jobs/nodes/{IID}/creds" not in r.objects
    assert len(bro._mints) == 1


def test_sweep_audit_no_secrets(bro, tmp_path):
    r = FakeRunner({f"jobs/nodes/{IID}/credreq": _credreq()})
    cbd.b2_sweep(bro, r, now=NOW)
    audit = open(os.path.join(str(tmp_path), "audit.jsonl")).read()
    assert '"verdict": "issued"' in audit
    assert "sekrit" not in audit and NONCE not in audit
    assert '"transport": "b2"' in audit
