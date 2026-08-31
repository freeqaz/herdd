"""Tests for onstart/cred_client.py (C3) — envelope/proof CROSS-compat with
credbroker.py, §2.5 decision order (injectable transports), jobd_boot.sh
byte-format-compatible config rewrite, and verify-then-swap install."""
import configparser
import json
import os
import stat
import subprocess
import sys

import pytest

_VAST = os.path.dirname(os.path.abspath(__file__))
for _p in (_VAST, os.path.join(_VAST, "onstart")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import credbroker as cb
import cred_client as cc

NONCE = "ab" * 16
CREDS_PAIR = {
    "b2": {"key_id": "kidRO", "application_key": "sekRO",
           "write_key_id": "kidRW", "write_application_key": "sekRW"},
    "bucket": "example-runs-bucket",
    "s3_endpoint": "https://s3.us-west-004.backblazeb2.com",
    "region": "us-west-004",
    "expires_at": 1900000000,
}
CREDS_SINGLE = {
    "b2": {"key_id": "kid1", "application_key": "sek1"},
    "bucket": "example-runs-bucket",
    "s3_endpoint": "https://s3.us-west-004.backblazeb2.com",
    "region": "us-west-004",
    "expires_at": 1900000000,
}


# ---------------------------------------------- envelope + proof cross-compat #
def test_broker_seal_client_open():
    blob = cb.seal_envelope(NONCE, CREDS_PAIR, ts=1234)
    assert cc.open_envelope(NONCE, blob) == CREDS_PAIR


def test_client_seal_broker_open():
    blob = cc.seal_envelope(NONCE, CREDS_PAIR, ts=1234)
    assert cb.open_envelope(NONCE, blob) == CREDS_PAIR


def test_envelopes_byte_identical():
    assert cc.seal_envelope(NONCE, CREDS_PAIR, ts=99) \
        == cb.seal_envelope(NONCE, CREDS_PAIR, ts=99)


def _tamper(blob):
    bad = dict(blob)
    bad["mac"] = ("0" if blob["mac"][0] != "0" else "1") + blob["mac"][1:]
    return bad


def test_tampered_mac_rejected_both_ways():
    blob = cb.seal_envelope(NONCE, CREDS_PAIR, ts=1234)
    with pytest.raises(cc.EnvelopeError):
        cc.open_envelope(NONCE, _tamper(blob))
    blob = cc.seal_envelope(NONCE, CREDS_PAIR, ts=1234)
    with pytest.raises(cb.EnvelopeError):
        cb.open_envelope(NONCE, _tamper(blob))


def test_wrong_nonce_rejected_both_ways():
    blob = cb.seal_envelope(NONCE, CREDS_PAIR, ts=1234)
    with pytest.raises(cc.EnvelopeError):
        cc.open_envelope("cd" * 16, blob)
    blob = cc.seal_envelope(NONCE, CREDS_PAIR, ts=1234)
    with pytest.raises(cb.EnvelopeError):
        cb.open_envelope("cd" * 16, blob)


def test_proof_matches_broker_verifier():
    proof = cc.make_credreq_proof(NONCE, 42, 1111, "jobs")
    assert proof == cb.make_credreq_proof(NONCE, 42, 1111, "jobs")
    assert cb.verify_credreq_proof(NONCE, 42, 1111, "jobs", proof)
    assert not cb.verify_credreq_proof(NONCE, 42, 1112, "jobs", proof)
    assert not cb.verify_credreq_proof("cd" * 16, 42, 1111, "jobs", proof)


# ------------------------------------------------------- instance id chain #
def test_instance_id_chain(monkeypatch):
    for k in ("JOBD_IID", "INSTANCE_ID", "CONTAINER_ID"):
        monkeypatch.delenv(k, raising=False)
    assert cc._instance_id() is None
    monkeypatch.setenv("CONTAINER_ID", "77")
    assert cc._instance_id() == 77
    monkeypatch.setenv("INSTANCE_ID", "66")
    assert cc._instance_id() == 66
    monkeypatch.setenv("JOBD_IID", "55")
    assert cc._instance_id() == 55
    monkeypatch.setenv("JOBD_IID", "not-a-number")   # falls through the chain
    assert cc._instance_id() == 66


# ------------------------------------------------- §2.5 decision order #
def _cfg(**kw):
    base = dict(iid=42, nonce=NONCE, role="jobs",
                broker_url="http://rig.example.ts.net:8651",
                ts_authkey="", bucket="example-runs-bucket")
    base.update(kw)
    return base


def _fail(*a, **k):
    raise RuntimeError("boom")


def test_direct_wins(monkeypatch):
    calls = []

    def direct(url, body):
        calls.append(("direct", url, body))
        return CREDS_PAIR

    got = cc.fetch_creds(_cfg(), direct=direct, join=_fail, socks=_fail,
                         b2lane=_fail)
    assert got == CREDS_PAIR
    assert len(calls) == 1
    body = calls[0][2]
    assert body == {"instance_id": 42, "nonce": NONCE, "role": "jobs",
                    "want": {"write_prefix": None}}


def test_direct_fail_then_tailnet_socks():
    order = []
    got = cc.fetch_creds(
        _cfg(ts_authkey="tskey-auth-x"),
        direct=lambda u, b: order.append("direct") or _fail(),
        join=lambda: order.append("join"),
        socks=lambda u, b: order.append("socks") or CREDS_PAIR,
        b2lane=_fail)
    assert got == CREDS_PAIR
    assert order == ["direct", "join", "socks"]


def test_no_authkey_skips_tailnet_goes_b2(monkeypatch):
    order = []
    got = cc.fetch_creds(
        _cfg(),
        direct=lambda u, b: order.append("direct") or _fail(),
        join=lambda: order.append("join"),
        socks=lambda u, b: order.append("socks"),
        b2lane=lambda i, n, r, bkt: order.append("b2") or CREDS_PAIR)
    assert got == CREDS_PAIR
    assert order == ["direct", "b2"]


def test_tailnet_fail_falls_to_b2():
    order = []
    got = cc.fetch_creds(
        _cfg(ts_authkey="tskey-auth-x"),
        direct=lambda u, b: order.append("direct") or _fail(),
        join=lambda: order.append("join") or _fail(),
        socks=lambda u, b: order.append("socks"),
        b2lane=lambda i, n, r, bkt: order.append("b2") or CREDS_PAIR)
    assert got == CREDS_PAIR
    assert order == ["direct", "join", "b2"]


def test_no_broker_url_jobs_goes_straight_to_b2():
    got = cc.fetch_creds(_cfg(broker_url=""), direct=_fail, join=_fail,
                         socks=_fail,
                         b2lane=lambda i, n, r, bkt: CREDS_PAIR)
    assert got == CREDS_PAIR


def test_non_jobs_role_never_uses_b2_lane():
    with pytest.raises(RuntimeError, match="all credential transports"):
        cc.fetch_creds(_cfg(role="train"), direct=_fail, join=_fail,
                       socks=_fail, b2lane=lambda *a: CREDS_PAIR)


# --------------------------------------------------- §2.4 B2 lane (box side) #
class _R:
    def __init__(self, rc=0, out=b""):
        self.returncode = rc
        self.stdout = out


def test_b2_lane_roundtrip_against_broker():
    """Full lane: credreq proof verified by the BROKER's verifier; response
    sealed by the BROKER's seal; client MAC-verifies and decrypts."""
    store = {}

    def run(args, **kw):
        if args[:2] == ["rclone", "listremotes"]:
            return _R(out=b"b2:\nb2w:\n")
        if args[1] == "rcat":
            assert args[2].startswith("b2w:example-runs-bucket/jobs/nodes/42/")
            req = json.loads(kw["input"].decode())
            assert "nonce" not in req            # raw nonce never on the bucket
            assert cb.verify_credreq_proof(NONCE, req["instance_id"],
                                           req["ts"], req["role"],
                                           req["proof"])
            store["creds"] = json.dumps(
                cb.seal_envelope(NONCE, CREDS_PAIR, ts=req["ts"])).encode()
            return _R()
        if args[1] == "cat":
            return _R(out=store.get("creds", b""))
        raise AssertionError("unexpected rclone call %r" % (args,))

    got = cc._b2_lane(42, NONCE, "jobs", "example-runs-bucket",
                      deadline_s=5, poll_s=0, run=run)
    assert got == CREDS_PAIR


def test_b2_lane_ignores_tampered_then_times_out():
    def run(args, **kw):
        if args[:2] == ["rclone", "listremotes"]:
            return _R(out=b"b2:\n")
        if args[1] == "rcat":
            return _R()
        if args[1] == "cat":   # shadow-written garbage under a wrong nonce
            blob = cb.seal_envelope("cd" * 16, {"evil": 1}, ts=2**33)
            return _R(out=json.dumps(blob).encode())
        raise AssertionError("unexpected call %r" % (args,))

    with pytest.raises(RuntimeError, match="timed out"):
        cc._b2_lane(42, NONCE, "jobs", "example-runs-bucket",
                    deadline_s=0.2, poll_s=0.05, run=run)


def _skewed_broker_run(skew_s):
    """rclone shim whose broker seals the response ts skew_s BEHIND req.ts
    (i.e. the box clock is ahead of the broker clock by skew_s)."""
    state = {}

    def run(args, **kw):
        if args[:2] == ["rclone", "listremotes"]:
            return _R(out=b"b2:\n")
        if args[1] == "rcat":
            req = json.loads(kw["input"].decode())
            state["blob"] = json.dumps(cb.seal_envelope(
                NONCE, CREDS_PAIR, ts=req["ts"] - skew_s)).encode()
            return _R()
        if args[1] == "cat":
            return _R(out=state.get("blob", b""))
        raise AssertionError("unexpected call %r" % (args,))

    return run


def test_b2_lane_tolerates_box_ahead_clock_skew():
    """Broker stamps envelope ts from ITS clock; a box clock ahead by less
    than CLOCK_SKEW_S must still accept the genuine response."""
    got = cc._b2_lane(42, NONCE, "jobs", "example-runs-bucket", deadline_s=5,
                      poll_s=0, run=_skewed_broker_run(cc.CLOCK_SKEW_S - 1))
    assert got == CREDS_PAIR


def test_b2_lane_still_drops_prior_cycle():
    """Beyond the skew window the blob is a prior cycle — still rejected."""
    with pytest.raises(RuntimeError, match="timed out"):
        cc._b2_lane(42, NONCE, "jobs", "example-runs-bucket", deadline_s=0.2,
                    poll_s=0.05, run=_skewed_broker_run(cc.CLOCK_SKEW_S + 1))


# --------------------------- config rewrite (jobd_boot.sh byte formats) #
# Fixture strings modeled LITERALLY on jobd_boot.sh:67-97 heredoc output —
# if jobd_boot.sh's format changes, change these AND cred_client together.
BOOT_B2 = """[b2]
type = s3
provider = Other
access_key_id = kidRO
secret_access_key = sekRO
endpoint = https://s3.us-west-004.backblazeb2.com
region = us-west-004
acl = private
no_check_bucket = true
"""
BOOT_B2W = BOOT_B2.replace("[b2]", "[b2w]").replace("kidRO", "kidRW") \
                  .replace("sekRO", "sekRW")
# jobd_boot.sh:135-149 env-file shape
BOOT_ENV = """export B2_BUCKET=example-runs-bucket
export B2_KEY_ID=oldkid
export B2_APPLICATION_KEY=oldsek
export B2_S3_ENDPOINT=https://s3.us-west-004.backblazeb2.com
export B2_REGION=us-west-004
export INSTANCE_ID=42
export JOBD_IDLE_PARK=1
"""


def test_remote_section_matches_boot_bytes():
    assert cc._remote_section(
        "b2", "kidRO", "sekRO",
        "https://s3.us-west-004.backblazeb2.com", "us-west-004") == BOOT_B2


def test_build_conf_fresh_pair_is_boot_bytes():
    assert cc.build_rclone_conf("", CREDS_PAIR) == BOOT_B2 + BOOT_B2W


def test_build_conf_single_key_no_b2w():
    txt = cc.build_rclone_conf("", CREDS_SINGLE)
    assert "[b2w]" not in txt
    cp = configparser.ConfigParser()
    cp.read_string(txt)
    assert cp["b2"]["access_key_id"] == "kid1"


def test_build_conf_replaces_and_preserves():
    old = ("[other]\ntype = local\n"
           + BOOT_B2.replace("kidRO", "deadkid").replace("sekRO", "deadsek")
           + BOOT_B2W.replace("kidRW", "deadw"))
    txt = cc.build_rclone_conf(old, CREDS_PAIR)
    assert "deadkid" not in txt and "deadsek" not in txt and "deadw" not in txt
    cp = configparser.ConfigParser()
    cp.read_string(txt)
    assert set(cp.sections()) == {"other", "b2", "b2w"}
    assert cp["other"]["type"] == "local"
    # emitted sections parse IDENTICALLY to a jobd_boot.sh-produced config
    boot = configparser.ConfigParser()
    boot.read_string(BOOT_B2 + BOOT_B2W)
    for sec in ("b2", "b2w"):
        assert dict(cp[sec]) == dict(boot[sec])


def test_build_jobd_env_boot_format_and_preserve():
    txt = cc.build_jobd_env(BOOT_ENV, CREDS_PAIR)
    lines = txt.splitlines()
    assert lines[0] == "export B2_BUCKET=example-runs-bucket"
    assert "export B2_KEY_ID=kidRO" in lines
    assert "export B2_APPLICATION_KEY=sekRO" in lines
    assert "export B2_WRITE_KEY_ID=kidRW" in lines
    assert "export B2_WRITE_APPLICATION_KEY=sekRW" in lines
    assert "export B2_KEY_EXPIRES_AT=1900000000" in lines
    assert "export INSTANCE_ID=42" in lines          # non-managed preserved
    assert "export JOBD_IDLE_PARK=1" in lines
    assert "oldkid" not in txt and "oldsek" not in txt
    # every managed key appears exactly once
    for k in ("B2_KEY_ID", "B2_APPLICATION_KEY", "B2_BUCKET"):
        assert sum(1 for ln in lines
                   if ln.startswith("export %s=" % k)) == 1


# --------------------------------------------------- verify-then-swap #
ORIG_CONF = "[b2]\ntype = s3\naccess_key_id = LIVEKEY\n"
ORIG_ENV = "export B2_KEY_ID=LIVEKEY\nexport INSTANCE_ID=42\n"


def _paths(tmp_path):
    conf = tmp_path / "rclone.conf"
    jenv = tmp_path / "jobd.env"
    conf.write_text(ORIG_CONF)
    jenv.write_text(ORIG_ENV)
    return conf, jenv


def test_failed_probe_leaves_everything_untouched(tmp_path):
    conf, jenv = _paths(tmp_path)
    with pytest.raises(RuntimeError, match="probe FAILED"):
        cc.apply_creds(CREDS_PAIR, "jobs", conf_path=str(conf),
                       jobd_env_path=str(jenv),
                       probe=lambda *a, **k: False)
    assert conf.read_text() == ORIG_CONF
    assert jenv.read_text() == ORIG_ENV
    assert not (tmp_path / "rclone.conf.credtmp").exists()


def test_probe_exception_leaves_everything_untouched(tmp_path):
    conf, jenv = _paths(tmp_path)
    with pytest.raises(RuntimeError, match="probe boom"):
        cc.apply_creds(CREDS_PAIR, "jobs", conf_path=str(conf),
                       jobd_env_path=str(jenv),
                       probe=_fail_probe)
    assert conf.read_text() == ORIG_CONF
    assert jenv.read_text() == ORIG_ENV
    assert not (tmp_path / "rclone.conf.credtmp").exists()


def _fail_probe(*a, **k):
    raise RuntimeError("probe boom")


def test_probe_success_swaps_atomically(tmp_path):
    conf, jenv = _paths(tmp_path)
    seen = {}

    def probe(path, creds, role):
        # probe runs against the TEMP config while the LIVE one is untouched
        seen["path"] = path
        seen["live_at_probe"] = conf.read_text()
        assert path != str(conf)
        return True

    cc.apply_creds(CREDS_PAIR, "jobs", conf_path=str(conf),
                   jobd_env_path=str(jenv), probe=probe)
    assert seen["path"] == str(conf) + ".credtmp"
    assert seen["live_at_probe"] == ORIG_CONF          # verify-THEN-swap
    assert conf.read_text() == BOOT_B2 + BOOT_B2W      # [b2] section replaced
    envtxt = jenv.read_text()
    assert "export B2_KEY_ID=kidRO" in envtxt
    assert "export B2_KEY_EXPIRES_AT=1900000000" in envtxt
    assert "export INSTANCE_ID=42" in envtxt
    assert "LIVEKEY" not in envtxt
    for p in (conf, jenv):
        assert stat.S_IMODE(os.stat(str(p)).st_mode) == 0o600
    assert not (tmp_path / "rclone.conf.credtmp").exists()
    assert not (tmp_path / "jobd.env.credtmp").exists()
    assert os.environ["B2_KEY_EXPIRES_AT"] == "1900000000"


def test_bad_expires_at_leaves_both_files_untouched(tmp_path):
    """Transactional swap: a response whose keys probe fine but whose
    expires_at is missing/non-numeric must fail BEFORE either file moves —
    previously rclone.conf was replaced first, leaving it out of sync with
    jobd.env. Temps (secret-bearing) must not leak either."""
    conf, jenv = _paths(tmp_path)
    no_exp = {k: v for k, v in CREDS_PAIR.items() if k != "expires_at"}
    for bad, exc in ((no_exp, KeyError),
                     (dict(CREDS_PAIR, expires_at="soon"), ValueError)):
        with pytest.raises(exc):
            cc.apply_creds(bad, "jobs", conf_path=str(conf),
                           jobd_env_path=str(jenv),
                           probe=lambda *a, **k: True)
        assert conf.read_text() == ORIG_CONF
        assert jenv.read_text() == ORIG_ENV
        assert not (tmp_path / "rclone.conf.credtmp").exists()
        assert not (tmp_path / "jobd.env.credtmp").exists()


def test_apply_requires_bucket(tmp_path, monkeypatch):
    conf, jenv = _paths(tmp_path)
    monkeypatch.delenv("B2_BUCKET", raising=False)
    creds = dict(CREDS_PAIR, bucket="")
    with pytest.raises(RuntimeError, match="no bucket"):
        cc.apply_creds(creds, "jobs", conf_path=str(conf),
                       jobd_env_path=str(jenv), probe=lambda *a, **k: True)
    assert conf.read_text() == ORIG_CONF


def test_probe_conf_command_shape():
    """_probe_conf drives rclone lsf with --config <temp>; write probe hits
    the role prefix (a scoped key can only list under its namePrefix)."""
    calls = []

    def run(args, **kw):
        calls.append(args)
        return _R(rc=0)

    assert cc._probe_conf("/tmp/x.credtmp", CREDS_PAIR, "jobs", run=run)
    assert calls[0][:3] == ["rclone", "--config", "/tmp/x.credtmp"]
    assert calls[0][-1] == "b2:example-runs-bucket"
    assert calls[1][-1] == "b2w:example-runs-bucket/jobs/"
    calls.clear()
    assert cc._probe_conf("/tmp/x.credtmp", CREDS_SINGLE, "train", run=run)
    assert len(calls) == 1                              # no write key, no b2w probe


def test_probe_conf_write_key_failure_fails():
    def run(args, **kw):
        return _R(rc=1 if args[-1].startswith("b2w:") else 0)

    assert not cc._probe_conf("/tmp/x", CREDS_PAIR, "jobs", run=run)


# ------------------------------------------------------------- shell lint #
def test_tailnet_join_bash_n():
    p = subprocess.run(
        ["bash", "-n", os.path.join(_VAST, "onstart", "tailnet_join.sh")],
        capture_output=True)
    assert p.returncode == 0, p.stderr.decode()
