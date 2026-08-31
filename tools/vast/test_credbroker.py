"""Unit tests for credbroker.py — no network, no creds, tmp state dir."""
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2_mint_key as bmk
import credbroker as cb


class FakeB2:
    """Stub for bmk._http (pattern from test_b2_mint_key.py): canned auth +
    in-memory key store with distinct incrementing key ids/secrets."""

    def __init__(self, keys=None):
        self.keys = list(keys or [])
        self.created = []
        self.deleted = []
        self.n = 0

    def __call__(self, url, body=None, headers=None):
        call = url.rsplit("/", 1)[-1]
        if call == "b2_authorize_account":
            return {"accountId": "acct1", "authorizationToken": "tok",
                    "apiInfo": {"storageApi": {
                        "apiUrl": "https://api.fake",
                        "allowed": {"capabilities": [
                            "listKeys", "writeKeys",
                            "deleteKeys", "listBuckets"]}}}}
        if call == "b2_list_buckets":
            return {"buckets": [{"bucketName": body["bucketName"],
                                 "bucketId": "bkt1"}]}
        if call == "b2_list_keys":
            return {"keys": list(self.keys), "nextApplicationKeyId": None}
        if call == "b2_delete_key":
            self.deleted.append(body["applicationKeyId"])
            self.keys = [k for k in self.keys
                         if k["applicationKeyId"] != body["applicationKeyId"]]
            return {}
        if call == "b2_create_key":
            self.n += 1
            kid, sec = f"kid{self.n}", f"sekrit{self.n}"
            self.created.append(body)
            self.keys.append({"applicationKeyId": kid,
                              "keyName": body["keyName"],
                              "capabilities": body["capabilities"]})
            return {"applicationKeyId": kid, "applicationKey": sec}
        raise AssertionError(f"unexpected call {call}")


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("CRED_BROKER_STATE", str(tmp_path))
    return tmp_path


@pytest.fixture
def fake(monkeypatch, state):
    f = FakeB2()
    monkeypatch.setattr(bmk, "_http", f)
    monkeypatch.setenv("B2_MINTER_KEY_ID", "mk")
    monkeypatch.setenv("B2_MINTER_APPLICATION_KEY", "ms")
    monkeypatch.setenv("B2_BUCKET", "example-runs-bucket")
    monkeypatch.setenv("B2_S3_ENDPOINT", "https://s3.us-west-004.fakeb2.com")
    monkeypatch.setenv("B2_REGION", "us-west-004")
    return f


def _inst(iid=42, status="running", nonce="ab" * 16, env_shape="pairs"):
    ee = [["FOO", "1"], ["BOX_IDENTITY_NONCE", nonce]]
    if env_shape == "dict":
        ee = {"FOO": "1", "BOX_IDENTITY_NONCE": nonce}
    elif env_shape == "none":
        ee = None
    return {"id": iid, "actual_status": status, "extra_env": ee}


NONCE = "ab" * 16


# ------------------------------------------------------------------ verify #
def test_verify_nonce_match_pairs(state):
    ok, why = cb.verify_instance(42, NONCE, lambda: [_inst()], None)
    assert (ok, why) == (True, "nonce")


def test_verify_nonce_match_dict_extra_env(state):
    ok, why = cb.verify_instance(
        42, NONCE, lambda: [_inst(env_shape="dict")], None)
    assert (ok, why) == (True, "nonce")


def test_verify_nonce_mismatch(state):
    ok, why = cb.verify_instance(42, "cd" * 16, lambda: [_inst()], None)
    assert (ok, why) == (False, "nonce_mismatch")


def test_verify_empty_nonce_denied(state):
    ok, why = cb.verify_instance(42, "", lambda: [_inst()], None)
    assert not ok


def test_verify_dead_instance_denied(state):
    ok, why = cb.verify_instance(
        42, NONCE, lambda: [_inst(status="exited")], None)
    assert (ok, why) == (False, "not_live")


def test_verify_unknown_instance_denied(state):
    ok, why = cb.verify_instance(99, NONCE, lambda: [_inst()], None)
    assert (ok, why) == (False, "unknown_instance")


def test_verify_fetch_failure_denied(state):
    def boom():
        raise RuntimeError("api down")
    ok, why = cb.verify_instance(42, NONCE, boom, None)
    assert (ok, why) == (False, "instances_unavailable")


def test_verify_registry_fallback(state):
    import hashlib
    reg = cb.Registry()
    reg.register_nonce(42, hashlib.sha256(NONCE.encode()).hexdigest())
    # box launched before nonce injection: no extra_env nonce
    ok, why = cb.verify_instance(
        42, NONCE, lambda: [_inst(env_shape="none")], reg)
    assert (ok, why) == (True, "registry")
    ok, _ = cb.verify_instance(
        42, "cd" * 16, lambda: [_inst(env_shape="none")], reg)
    assert not ok


def test_verify_registry_survives_reload(state):
    import hashlib
    cb.Registry().register_nonce(7, hashlib.sha256(b"n7").hexdigest())
    assert cb.Registry().nonce_sha256(7) == \
        hashlib.sha256(b"n7").hexdigest()


# ------------------------------------------------------------------ policy #
@pytest.mark.parametrize("role,want,exp", [
    ("jobs", None, (True, "jobs/")),
    ("jobs", "jobs/", (True, "jobs/")),
    ("jobs", "jobs/J1/", (True, "jobs/J1/")),         # extension ok
    ("jobs", "serve/", (False, None)),
    ("jobs", "jobs", (False, None)),                  # not an extension
    ("serve", None, (True, "serve/")),
    ("serve", "serve/s1/", (True, "serve/s1/")),
    ("serve", "jobs/", (False, None)),
    ("train", None, (True, None)),
    ("train", "jobs/", (False, None)),                # train can't ask a prefix
    ("root", None, (False, None)),                    # unknown role
    ("", "jobs/", (False, None)),
])
def test_check_policy(role, want, exp):
    assert cb.check_policy(role, want) == exp


# ------------------------------------------------------------------- issue #
def test_issue_scoped_pair_naming_and_shape(fake):
    reg = cb.Registry()
    r = cb.issue_keys(42, "jobs", "jobs/J1/", hours=200, registry=reg,
                      now=1000)
    ro, rw = fake.created
    assert ro["keyName"] == "box-42-brk1000-ro"
    assert rw["keyName"] == "box-42-brk1000-rw"
    assert rw["namePrefix"] == "jobs/J1/"
    assert "namePrefix" not in ro
    assert r["b2"]["key_id"] == "kid1" and r["b2"]["write_key_id"] == "kid2"
    assert r["b2"]["application_key"] and r["b2"]["write_application_key"]
    assert r["bucket"] == "example-runs-bucket"
    assert r["s3_endpoint"].startswith("https://")
    assert r["region"] == "us-west-004"
    assert r["expires_at"] == 1000 + 200 * 3600
    assert reg.last_keys(42) == [
        {"name": "box-42-brk1000-ro", "key_id": "kid1"},
        {"name": "box-42-brk1000-rw", "key_id": "kid2"}]


def test_issue_train_single_key(fake):
    r = cb.issue_keys(7, "train", None, hours=168, registry=cb.Registry(),
                      now=2000)
    assert [b["keyName"] for b in fake.created] == ["box-7-brk2000"]
    assert set(r["b2"]) == {"key_id", "application_key"}


def test_reissue_revokes_prior_brk_keys_only(fake):
    reg = cb.Registry()
    cb.issue_keys(42, "jobs", "jobs/", hours=168, registry=reg, now=1000)
    first = [k["applicationKeyId"] for k in fake.keys]
    # a launch-shipped key for the same box sits alongside — must survive
    fake.keys.append({"applicationKeyId": "launchkid",
                      "keyName": "box-42-ro", "capabilities": []})
    cb.issue_keys(42, "jobs", "jobs/", hours=168, registry=reg, now=2000)
    assert set(first) <= set(fake.deleted)
    assert "launchkid" not in fake.deleted
    assert reg.last_keys(42) == [
        {"name": "box-42-brk2000-ro", "key_id": "kid3"},
        {"name": "box-42-brk2000-rw", "key_id": "kid4"}]


def test_revoke_guard_ignores_non_brk_registry_rows(fake):
    reg = cb.Registry()
    # poisoned registry: names outside our brk namespace must never be revoked
    reg.set_last_keys(42, [{"name": "run-important", "key_id": "runkid"},
                           {"name": "box-42-ro", "key_id": "launchkid"}])
    cb.issue_keys(42, "jobs", "jobs/", hours=168, registry=reg, now=3000)
    assert "runkid" not in fake.deleted
    assert "launchkid" not in fake.deleted


# ---------------------------------------------------------------- envelope #
def test_envelope_round_trip():
    obj = {"b2": {"key_id": "k1", "application_key": "s1"},
           "bucket": "b", "expires_at": 123}
    blob = cb.seal_envelope(NONCE, obj, ts=5000)
    assert set(blob) == {"ts", "ciphertext", "mac"}
    assert blob["ts"] == 5000
    assert cb.open_envelope(NONCE, blob) == obj
    # secret never appears in the sealed form
    assert "s1" not in json.dumps(blob)


def test_envelope_multiblock_plaintext():
    obj = {"pad": "x" * 500}          # > several SHA256 keystream blocks
    assert cb.open_envelope(NONCE, cb.seal_envelope(NONCE, obj)) == obj


def test_envelope_tamper_rejected():
    blob = cb.seal_envelope(NONCE, {"a": 1}, ts=5000)
    ct = bytearray(bytes.fromhex(blob["ciphertext"]))
    ct[0] ^= 0xFF
    with pytest.raises(cb.EnvelopeError, match="mac"):
        cb.open_envelope(NONCE, dict(blob, ciphertext=ct.hex()))
    with pytest.raises(cb.EnvelopeError, match="mac"):
        cb.open_envelope(NONCE, dict(blob, mac="0" * 64))
    with pytest.raises(cb.EnvelopeError, match="mac"):
        cb.open_envelope(NONCE, dict(blob, ts=5001))     # MAC binds ts
    with pytest.raises(cb.EnvelopeError):
        cb.open_envelope("cd" * 16, blob)                # wrong nonce
    with pytest.raises(cb.EnvelopeError, match="malformed"):
        cb.open_envelope(NONCE, {"ts": 1})


def test_credreq_proof():
    p = cb.make_credreq_proof(NONCE, 42, 5000, "jobs")
    assert cb.verify_credreq_proof(NONCE, 42, 5000, "jobs", p)
    assert not cb.verify_credreq_proof(NONCE, 43, 5000, "jobs", p)
    assert not cb.verify_credreq_proof(NONCE, 42, 5001, "jobs", p)
    assert not cb.verify_credreq_proof(NONCE, 42, 5000, "train", p)
    assert not cb.verify_credreq_proof("cd" * 16, 42, 5000, "jobs", p)
    assert p != NONCE and len(p) == 64        # raw nonce never on the bucket


def test_ts_fresh_window():
    assert cb.ts_fresh(10_000, now=10_000)
    assert cb.ts_fresh(10_000, now=10_600)
    assert cb.ts_fresh(10_000, now=9_500)     # broker clock behind box
    assert not cb.ts_fresh(10_000, now=10_601)
    assert not cb.ts_fresh(10_000, now=9_399)
    assert not cb.ts_fresh("junk", now=10_000)


# -------------------------------------------------------------- rate limit #
def test_rate_limit_spacing_and_cap(state):
    rl = cb.RateLimiter()
    assert rl.try_acquire(42, now=1000) == (True, None)
    assert rl.try_acquire(42, now=1030) == (False, "spacing")
    assert rl.try_acquire(43, now=1030) == (True, None)   # per-iid
    ok, why = rl.try_acquire(42, now=1061)
    assert ok
    now = 2000
    for _ in range(22):                       # 2 done + 22 = 24 today
        now += 61
        ok, why = rl.try_acquire(42, now=now)
        assert ok, why
    assert rl.try_acquire(42, now=now + 61) == (False, "daily_cap")
    # window rolls: a day later the cap clears
    assert rl.try_acquire(42, now=now + 90000) == (True, None)


def test_rate_limit_persisted_across_instances(state):
    cb.RateLimiter().try_acquire(42, now=1000)
    assert cb.RateLimiter().try_acquire(42, now=1030) == (False, "spacing")


# ------------------------------------------------------------------- audit #
def test_audit_redacts_application_keys(state):
    # even a careless caller passing the full issue response leaks nothing
    rec = {"transport": "http", "remote": "100.1.2.3", "instance_id": 42,
           "role": "jobs", "verdict": "issued",
           "keys": [{"name": "box-42-brk1000-ro", "key_id": "kid1"}],
           "resp": {"b2": {"key_id": "kid1", "application_key": "sekrit1",
                           "write_key_id": "kid2",
                           "write_application_key": "sekrit2"}},
           "expires_at": 999}
    cb.audit_append(rec)
    line = open(os.path.join(cb.state_dir(), "audit.jsonl")).read()
    assert "sekrit" not in line
    assert "application_key" not in line
    got = json.loads(line)
    assert got["keys"][0]["key_id"] == "kid1"          # ids/names survive
    assert got["resp"]["b2"] == {"key_id": "kid1", "write_key_id": "kid2"}
    assert got["ts"]                                    # stamped
    assert got["verdict"] == "issued"


def test_audit_appends_jsonl(state):
    cb.audit_append({"verdict": "denied", "reason": "nonce_mismatch"})
    cb.audit_append({"verdict": "issued"})
    lines = open(os.path.join(cb.state_dir(), "audit.jsonl")).readlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["reason"] == "nonce_mismatch"


# ------------------------------------------------------------- misc policy #
def test_ephemeral_hours(monkeypatch):
    monkeypatch.delenv("B2_EPHEMERAL_HOURS", raising=False)
    assert cb.ephemeral_hours() == 168
    assert cb.ephemeral_hours(timeout_s=3600) == 168        # floor wins
    assert cb.ephemeral_hours(timeout_s=200 * 3600) == 272  # timeout + 72h
    monkeypatch.setenv("B2_EPHEMERAL_HOURS", "300")
    assert cb.ephemeral_hours() == 300


def test_state_dir_override(state):
    assert cb.state_dir() == str(state)
    assert os.path.isdir(cb.state_dir())
