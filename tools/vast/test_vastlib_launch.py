"""Portable tests for `vastlib.launch` — the spec/secrets/mint half and the
`_do_launch` create sequence, ported at plan §8 step 3.

Four jobs:

1. **Pin the ported copies against the live flat ones — while that window was
   open.** Step 3 was ADD-ONLY, so `herdd.py` still carried its own
   definitions and both could be fed the same inputs: stronger than a
   hand-written expectation, because it failed on drift in either direction.
   Plan §8 step 6d closed the window; those sweeps are deleted (each site says
   so) and the properties they proxied for are asserted directly.

2. **Prove the two mint ledgers are INDEPENDENT.** `vastlib.launch.spec` has
   its own `_MINTED_PAIRS` / `_MINTED_SCOPED` / `_MINTED_PUBLISH` /
   `_MINT_ANNOUNCED`; nothing synchronises them with `herdd`'s. The failure
   mode is a test that drives one and asserts on the other, reads `{}`, and
   passes for the wrong reason. `test_broker_env.py`'s autouse `_clean_seams`
   clears the `herdd` copies only, so this file carries its own fixture.

3. **Prove the dry-run redaction still bites on the vastlib path.**
   `_do_launch` calls `spec._is_secret_env` by MODULE ATTRIBUTE. If that
   binding ever goes stale the redaction fails OPEN — `launch --dry-run` starts
   printing HF/B2 token VALUES to stdout — and no other assertion in the suite
   would notice. Two tests here: a fake secret must not appear in stdout, and
   patching `spec._is_secret_env` must visibly change the output (which is what
   proves the call goes through the module attribute rather than a snapshot).

4. **Prove the not-yet-ported seams RAISE — and that the ported ones no
   longer do.** `launch.py` opened with ten stub names; plan §8 step 4 rebound
   three of them (`_launch_preflight`, `launch_instance`,
   `_emit_launched_soft`) to `boxes.lifecycle`. The remaining SEVEN must still
   raise: a silent no-op stub would let a test drive `_do_launch` to a green
   "launched" that never resolved an image — the vacuous pass plan §7.3 exists
   to kill. The three that landed get the mirror-image assertion (identity with
   the `lifecycle` definition, and no `NotImplementedError`), plus the
   launch-through tests in the last section.

5. **Prove the money path is live code behind the guard.** With
   `launch_instance` rebound, `_do_launch` can reach a real `PUT v0/asks/`.
   The offline lane now rests on conftest's `_block_mutating_api_calls`, which
   wraps the ATTRIBUTE `vastlib.core.api.request_soft` — so the last section
   asserts both that an unstubbed `_do_launch` is REFUSED by the guard (and
   exits, rather than passing vacuously) and that the refusal names
   `v0/asks/`, which is only true if the PUT went through that attribute. A
   `from … import request_soft` in `lifecycle.launch_instance` would bind past
   the guard and bill a real box; this is the test that would catch it.

Offline lane: no vast API, no network, no B2, $0. Every mint is stubbed at
`b2_mint_key`; nothing in this file ever holds or prints real key material, and
the fixture strips the workstation's B2/broker env before each test so a folded
`.env` cannot leak into an assertion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import b2_mint_key  # noqa: E402
import herdd as v  # noqa: E402
from vastlib.boxes import lifecycle  # noqa: E402
from vastlib.core import result  # noqa: E402
from vastlib.launch import launch, spec  # noqa: E402
from vastlib.storage import b2  # noqa: E402

# Obvious fakes. Nothing in this file may carry, print or assert on a real key.
FAKE_KID, FAKE_KEY = "kid-FAKE-0", "key-FAKE-0"
FAKE_RO = ("kid-FAKE-ro", "key-FAKE-ro")
FAKE_RW = ("kid-FAKE-rw", "key-FAKE-rw")
FAKE_PUB = ("kid-FAKE-pub", "key-FAKE-pub")

_B2_ENV = ("B2_MINTER_KEY_ID", "B2_MINTER_APPLICATION_KEY", "B2_BOX_KEY_ID",
           "B2_BOX_APPLICATION_KEY", "B2_KEY_ID", "B2_APPLICATION_KEY",
           "B2_BUCKET", "B2_EPHEMERAL_HOURS", "B2_PUBLISH_PREFIX",
           "B2_KEY_ID_EU", "B2_APPLICATION_KEY_EU", "B2_S3_ENDPOINT_EU",
           "B2_BUCKET_EU", "B2_REGION_EU", "B2_REGION_MODE",
           "R2_TC_KEY_ID", "R2_TC_SECRET_ACCESS_KEY", "R2_TC_ENDPOINT",
           "R2_TC_BUCKET", "CRED_BROKER_URL", "TS_AUTHKEY")


@pytest.fixture(autouse=True)
def _clean_launch_seams(monkeypatch):
    """Start every test from "nothing configured", and clear THIS module's mint
    ledgers (not `herdd`'s — they are different objects; see the header)."""
    for k in _B2_ENV:
        monkeypatch.delenv(k, raising=False)
    for led in (spec._MINTED_PAIRS, spec._MINTED_SCOPED, spec._MINTED_PUBLISH):
        led.clear()
    spec._MINT_ANNOUNCED.clear()
    yield
    for led in (spec._MINTED_PAIRS, spec._MINTED_SCOPED, spec._MINTED_PUBLISH):
        led.clear()
    spec._MINT_ANNOUNCED.clear()


# =============================================================================
# spec.py — the secret split and the frozen spec.json contract
# =============================================================================
@pytest.mark.parametrize("k,val,secret", [
    ("HF_TOKEN", "x", True), ("B2_APPLICATION_KEY", "x", True),
    ("hf_token", "x", True), ("DOCKER_AUTH", "x", True),
    ("MY_SESSION", "x", True), ("SOME_SIGNATURE", "x", True),
    ("PRIVATE_THING", "x", True), ("PASSWORD", "x", True),
    ("PWD_X", "x", True), ("CRED_X", "x", True),
    # VALUE-based, not key-based: an embedded credential in a URL.
    ("DATABASE_URL", "postgres://u:p@host/db", True),
    ("DATABASE_URL", "x", False),
    ("PLAIN", "value", False), ("RUN_ID", "r7", False),
    ("EVAL_ENV_VER", "v9", False),
    ("HTTP_PROXY", "http://proxy:8080", False),
])
def test_is_secret_env_classifies_the_key(k, val, secret):
    """Was `…_agrees_with_the_flat_copy` over the same fifteen keys. The
    launcher re-exports this predicate since plan §8 step 6d, so the comparison
    ran one classifier twice; the verdict per key is what it stood for, and a
    key silently falling off the secret side is the failure that matters —
    `_build_launch_spec` would then ship its VALUE to a rented box."""
    assert bool(spec._is_secret_env(k, val)) is secret


def test_is_secret_env_handles_none():
    assert spec._is_secret_env(None, None) is False


def test_split_env_secrets_drops_values_and_keeps_names_in_order():
    env, keys = spec._split_env_secrets(
        ["A=1", "HF_TOKEN=supersecret", "B=2", "B2_KEY_ID=alsosecret",
         "HF_TOKEN=again"])
    assert env == {"A": "1", "B": "2"}
    assert keys == ["HF_TOKEN", "B2_KEY_ID"]          # order preserved, deduped
    assert "supersecret" not in json.dumps([env, keys])


def _spec_kw(**over):
    base = dict(run_id="r1", runset="rs", image="img:tag",
                image_login_ref="-u u -p <redacted> host", disk=40,
                runtype="ssh_direct", gpu=("H100",), gpu_ram=80, num_gpus=2,
                env_list=["A=1", "HF_TOKEN=tok"], onstart="#!/bin/sh\necho hi\n",
                orig_bid=0.3, max_bid=0.6)
    base.update(over)
    return base


def test_build_launch_spec_keeps_its_frozen_key_set():
    """Was `…_is_byte_identical_to_the_flat_copy`, which also took the frozen
    key set (plan §4, B2 event schema) from the flat copy — so post-6d BOTH
    arms read this module. The key set is pinned literally instead: it is a
    persisted event schema, and the reader is a dashboard nobody re-runs this
    suite against."""
    body = spec._build_launch_spec(**_spec_kw())
    assert set(body) == {
        "bid", "cuda", "disk", "env", "gpu", "gpu_ram", "image",
        "image_digest", "image_login", "num_gpus", "onstart_b64", "run_id",
        "runset", "runtype", "secret_env_keys", "v"}


def test_build_launch_spec_never_carries_a_secret_value():
    body = spec._build_launch_spec(**_spec_kw(env_list=["HF_TOKEN=supersecret"]))
    assert "supersecret" not in json.dumps(body)
    assert body["secret_env_keys"] == ["HF_TOKEN"]
    assert body["env"] == {}


def test_build_launch_spec_bid_nesting_and_onstart_b64_round_trip():
    import base64
    body = spec._build_launch_spec(**_spec_kw(defend_at=0.5, rescue_wait_s=90))
    assert body["v"] == 1
    assert body["bid"] == {"orig": 0.3, "max": 0.6,
                           "defend_at": 0.5, "rescue_wait_s": 90}
    assert base64.b64decode(body["onstart_b64"]).decode() == "#!/bin/sh\necho hi\n"


# =============================================================================
# spec.py — the standing-credential pair builders
# =============================================================================
def test_b2_eu_pairs_needs_the_four_and_adds_the_region(monkeypatch):
    """Was `…_agrees_with_the_flat_copy`; one builder since step 6d."""
    assert spec._b2_eu_pairs() == []
    for k in ("B2_KEY_ID_EU", "B2_APPLICATION_KEY_EU", "B2_S3_ENDPOINT_EU",
              "B2_BUCKET_EU"):
        monkeypatch.setenv(k, f"fake-{k.lower()}")
    assert dict(spec._b2_eu_pairs())["B2_REGION_EU"] == "eu-central-003"
    assert dict(spec._b2_eu_pairs())["B2_REGION_MODE"] == "auto"


def test_b2_eu_pairs_needs_all_four(monkeypatch):
    monkeypatch.setenv("B2_KEY_ID_EU", "fake")
    monkeypatch.setenv("B2_APPLICATION_KEY_EU", "fake")
    assert spec._b2_eu_pairs() == []


def test_r2_tc_pairs_needs_the_three_and_adds_the_bucket(monkeypatch):
    """Was `…_agrees_with_the_flat_copy`; one builder since step 6d."""
    assert spec._r2_tc_pairs() == []
    for k in ("R2_TC_KEY_ID", "R2_TC_SECRET_ACCESS_KEY", "R2_TC_ENDPOINT"):
        monkeypatch.setenv(k, f"fake-{k.lower()}")
    assert dict(spec._r2_tc_pairs())["R2_TC_BUCKET"] == "shared-triton-cache"


CDN_KEYS = ("B2_CDN_HOST", "B2_CDN_BUCKET", "B2_CDN_PREFIX")


@pytest.mark.parametrize("drop", CDN_KEYS)
def test_cdn_pairs_is_all_three_or_nothing(monkeypatch, drop):
    """ALL-OR-NOTHING, because the box-side tier refuses to engage on a partial
    set: shipping two of three is pure cost. Explicit delenv — a workstation
    .env carries these, so an implicit-empty assertion would test the runner."""
    for k in CDN_KEYS:
        monkeypatch.setenv(k, "wtestdeadbeef")
    monkeypatch.delenv(drop, raising=False)
    assert spec._cdn_pairs() == []


def test_cdn_pairs_never_ships_an_empty_value(monkeypatch):
    for k in CDN_KEYS:
        monkeypatch.setenv(k, "wtestdeadbeef")
    pairs = spec._cdn_pairs()
    assert [k for k, _ in pairs] == list(CDN_KEYS)
    assert all(v for _, v in pairs)
    # an empty string is not "set" — it would ship a var the tier then trips on
    monkeypatch.setenv("B2_CDN_PREFIX", "")
    assert spec._cdn_pairs() == []


def test_the_cdn_prefix_is_classified_secret_and_the_host_is_not():
    """It is a URL BEARER credential wearing an opaque-value disguise. Without
    the classification it lands verbatim in the never-deleted runs/<id>/spec.json
    and in `launch --dry-run` stdout."""
    assert spec._is_secret_env("B2_CDN_PREFIX", "wtestdeadbeef")
    assert not spec._is_secret_env("B2_CDN_HOST", "weights.example.com")
    assert not spec._is_secret_env("B2_CDN_BUCKET", "weights-cdn")
    env, secrets = spec._split_env_secrets(
        ["B2_CDN_HOST=h", "B2_CDN_BUCKET=b", "B2_CDN_PREFIX=wtestdeadbeef"])
    assert "B2_CDN_PREFIX" in secrets
    assert "wtestdeadbeef" not in str(env)


# `test_ephemeral_hours_agrees_with_the_flat_copy` swept four timeouts through
# both copies. One copy since step 6d; the floor and the slack are asserted by
# value immediately below.


def test_ephemeral_hours_floor_and_slack(monkeypatch):
    assert spec._ephemeral_hours() == 168.0
    monkeypatch.setenv("B2_EPHEMERAL_HOURS", "10")
    assert spec._ephemeral_hours() == 10.0
    assert spec._ephemeral_hours(3600 * 100) == 172.0   # 100h + 72h slack


# =============================================================================
# spec.py — the mint. Every call to b2_mint_key is stubbed: no network.
# =============================================================================
def _arm_minter(monkeypatch):
    monkeypatch.setenv("B2_MINTER_KEY_ID", "fake-minter-id")
    monkeypatch.setenv("B2_MINTER_APPLICATION_KEY", "fake-minter-key")


def test_ship_b2_pair_prefers_a_fresh_mint_and_caches_it(monkeypatch, capsys):
    _arm_minter(monkeypatch)
    calls = []
    monkeypatch.setattr(b2_mint_key, "mint",
                        lambda name, hours=48, **kw: (calls.append((name, hours))
                                                      or (FAKE_KID, FAKE_KEY)))
    base = b2_mint_key.sanitize_name("run-abc")
    assert spec._ship_b2_pair("run-abc", hours=5) == (FAKE_KID, FAKE_KEY)
    assert spec._MINTED_PAIRS[base] == (FAKE_KID, FAKE_KEY)
    # second call is served from the cache — ONE mint per process
    assert spec._ship_b2_pair("run-abc", hours=5) == (FAKE_KID, FAKE_KEY)
    assert len(calls) == 1
    out = capsys.readouterr().out
    assert "minted ephemeral B2 key" in out and base in out
    assert FAKE_KEY not in out and FAKE_KID not in out   # NAME only, never the value


def test_ship_b2_pair_dry_run_never_mints_and_announces_once(monkeypatch, capsys):
    _arm_minter(monkeypatch)
    monkeypatch.setenv("B2_BOX_KEY_ID", "fake-box-id")
    monkeypatch.setenv("B2_BOX_APPLICATION_KEY", "fake-box-key")
    monkeypatch.setattr(b2_mint_key, "mint",
                        lambda *a, **k: pytest.fail("dry_run must never mint"))
    assert spec._ship_b2_pair("run-abc", hours=5, dry_run=True) == (
        "fake-box-id", "fake-box-key")
    spec._ship_b2_pair("run-abc", hours=5, dry_run=True)
    assert capsys.readouterr().out.count("would mint ephemeral B2 key") == 1


def test_ship_b2_pair_falls_back_to_the_ops_key_with_a_warning(monkeypatch, capsys):
    monkeypatch.setenv("B2_KEY_ID", "fake-ops-id")
    monkeypatch.setenv("B2_APPLICATION_KEY", "fake-ops-key")
    assert spec._ship_b2_pair("run-abc", hours=5) == ("fake-ops-id", "fake-ops-key")
    assert "full-capability ops key" in capsys.readouterr().err


def test_ship_b2_pair_exits_with_the_exact_flat_message(monkeypatch):
    with pytest.raises(SystemExit) as e:
        spec._ship_b2_pair("run-abc", hours=5)
    assert str(e.value) == ("error: no B2 credentials in env/.env (need B2_MINTER_*, "
                            "B2_BOX_*, or the B2_KEY_ID pair)")


def test_ship_b2_env_scoped_mint_returns_six_pairs_in_order(monkeypatch, capsys):
    _arm_minter(monkeypatch)
    monkeypatch.setattr(b2_mint_key, "mint_pair",
                        lambda base, hours=48, write_prefix="", **kw: (FAKE_RO, FAKE_RW))
    monkeypatch.setattr(b2_mint_key, "mint_publish",
                        lambda base, hours=48, **kw: FAKE_PUB)
    monkeypatch.setattr(b2_mint_key, "publish_prefix", lambda *a, **k: "checkpoints/")
    pairs = spec._ship_b2_env("run-abc", hours=5, write_prefix="jobs/")
    assert [k for k, _val in pairs] == [
        "B2_KEY_ID", "B2_APPLICATION_KEY",
        "B2_WRITE_KEY_ID", "B2_WRITE_APPLICATION_KEY",
        "B2_PUBLISH_KEY_ID", "B2_PUBLISH_APPLICATION_KEY"]
    out = capsys.readouterr().out
    for _k, val in pairs:
        assert val not in out          # names, ttls and prefixes only


def test_ship_b2_env_publish_mint_failure_never_kills_the_launch(monkeypatch, capsys):
    _arm_minter(monkeypatch)
    monkeypatch.setattr(b2_mint_key, "mint_pair",
                        lambda base, hours=48, write_prefix="", **kw: (FAKE_RO, FAKE_RW))

    def _boom(*a, **k):
        raise b2_mint_key.MintError("nope")
    monkeypatch.setattr(b2_mint_key, "mint_publish", _boom)
    pairs = spec._ship_b2_env("run-abc", hours=5, write_prefix="jobs/")
    assert len(pairs) == 4
    assert "publish key mint failed" in capsys.readouterr().err


def test_ship_b2_env_scoped_mint_failure_falls_through_to_two_pairs(monkeypatch):
    """The bare `rk = None` sentinel: a failed scoped mint must fall all the way
    through to the single bucket-wide pair, never ship a scoped env of Nones."""
    _arm_minter(monkeypatch)
    monkeypatch.setenv("B2_BOX_KEY_ID", "fake-box-id")
    monkeypatch.setenv("B2_BOX_APPLICATION_KEY", "fake-box-key")

    def _boom(*a, **k):
        raise b2_mint_key.MintError("nope")
    monkeypatch.setattr(b2_mint_key, "mint_pair", _boom)
    pairs = spec._ship_b2_env("run-abc", hours=5, write_prefix="jobs/")
    assert pairs == [("B2_KEY_ID", "fake-box-id"),
                     ("B2_APPLICATION_KEY", "fake-box-key")]
    assert all(val is not None for _k, val in pairs)


def test_ship_b2_env_without_a_write_prefix_is_the_two_pair_shape(monkeypatch):
    _arm_minter(monkeypatch)
    monkeypatch.setattr(b2_mint_key, "mint",
                        lambda name, hours=48, **kw: (FAKE_KID, FAKE_KEY))
    assert spec._ship_b2_env("run-abc", hours=5) == [
        ("B2_KEY_ID", FAKE_KID), ("B2_APPLICATION_KEY", FAKE_KEY)]


def test_minted_expiry_is_none_without_a_mint():
    assert spec._minted_expiry("run-abc", 5) is None


def test_minted_expiry_fires_on_pairs_and_scoped_but_not_publish(monkeypatch):
    base = b2_mint_key.sanitize_name("run-abc")
    spec._MINTED_PUBLISH[base] = FAKE_PUB
    assert spec._minted_expiry("run-abc", 5) is None, (
        "_MINTED_PUBLISH must NOT witness an expiry — the read/write keys may "
        "be standing (test_broker_env.py pins the same asymmetry on herdd)")
    spec._MINTED_PAIRS[base] = (FAKE_KID, FAKE_KEY)
    assert spec._minted_expiry("run-abc", 5) is not None
    spec._MINTED_PAIRS.clear()
    spec._MINTED_SCOPED[base] = (FAKE_RO, FAKE_RW)
    assert spec._minted_expiry("run-abc", 5) is not None


def test_resolve_secret_only_special_cases_the_b2_pair(monkeypatch):
    monkeypatch.setenv("SOME_OTHER", "plain-value")
    assert spec._resolve_secret("SOME_OTHER") == "plain-value"
    monkeypatch.setenv("B2_BOX_KEY_ID", "fake-box-id")
    monkeypatch.setenv("B2_BOX_APPLICATION_KEY", "fake-box-key")
    assert spec._resolve_secret("B2_KEY_ID", run_id="r1") == "fake-box-id"
    assert spec._resolve_secret("B2_APPLICATION_KEY", run_id="r1") == "fake-box-key"


# =============================================================================
# spec.py — the mint ledgers are a SECOND, INDEPENDENT set
# =============================================================================
def test_there_is_exactly_one_mint_ledger_per_name(monkeypatch):
    """INVERTED AT STEP 6d, deliberately. While `herdd.py` had a body these
    four ledgers were a SECOND, INDEPENDENT set and this test asserted `is not`
    — driving the vastlib copy and asserting on `herdd._MINTED_*` was reading
    the wrong dict. The thin launcher makes that impossible: whatever it
    exposes is THIS module's object. A live B2 key minted through either name
    must therefore be recorded once, and both spellings must see it — the
    property that actually protects against a double mint.
    """
    for name in ("_MINTED_PAIRS", "_MINTED_SCOPED", "_MINTED_PUBLISH",
                 "_MINT_ANNOUNCED"):
        twin = getattr(v, name, None)
        assert twin is None or twin is getattr(spec, name), (
            f"{name}: the launcher grew a SECOND ledger — a mint recorded in "
            f"one would be invisible to the other, which is how a run mints "
            f"twice and leaks a key")

    _arm_minter(monkeypatch)
    monkeypatch.setattr(b2_mint_key, "mint",
                        lambda name, hours=48, **kw: (FAKE_KID, FAKE_KEY))
    base = b2_mint_key.sanitize_name("run-independence")
    spec._ship_b2_pair("run-independence", hours=5)
    assert base in spec._MINTED_PAIRS
    assert base in v._MINTED_PAIRS, "one ledger, both spellings"
    assert spec._minted_expiry("run-independence", 5) is not None
    assert v._minted_expiry("run-independence", 5) is not None


def test_clearing_the_ledger_through_either_spelling_clears_the_one_ledger():
    """The other half of the inversion. `test_broker_env.py::_clean_seams`
    clears the dicts by their `herdd` names; before 6d that reached a
    different pair of dicts than this file's `_clean_launch_seams`, and each
    file had to own its own cleanup. Now it is one object, so either fixture
    cleans up after either file — asserted structurally rather than trusted."""
    spec._MINTED_PAIRS["sentinel"] = (FAKE_KID, FAKE_KEY)
    assert v._MINTED_PAIRS["sentinel"] == (FAKE_KID, FAKE_KEY)
    v._MINTED_PAIRS.clear()
    assert "sentinel" not in spec._MINTED_PAIRS


# =============================================================================
# spec.py — _read_spec_soft, the reciprocal reader
# =============================================================================
def test_read_spec_soft_returns_empty_without_a_bucket():
    assert spec._read_spec_soft("r1") == {}


@pytest.mark.parametrize("rc,out", [(1, ""), (0, ""), (0, "   "),
                                    (0, "not json"), (0, "[1,2]")])
def test_read_spec_soft_degrades_to_empty_on_anything_odd(monkeypatch, rc, out):
    monkeypatch.setenv("B2_BUCKET", "fake-bucket")
    monkeypatch.setattr(b2, "_rclone_soft",
                        lambda args: result.ProcResult(rc, out, ""))
    assert spec._read_spec_soft("r1") == {}


def test_read_spec_soft_reads_through_the_storage_b2_module_attribute(monkeypatch):
    monkeypatch.setenv("B2_BUCKET", "fake-bucket")
    seen = []
    monkeypatch.setattr(b2, "_rclone_soft",
                        lambda args: (seen.append(args)
                                      or result.ProcResult(0, '{"v": 1}', "")))
    assert spec._read_spec_soft("r1") == {"v": 1}
    assert seen == [["cat", "b2:fake-bucket/runs/r1/spec.json"]]


# =============================================================================
# launch.py — the seams RAISE (no silent no-ops)
# =============================================================================
# TWO, down from ten. Plan §8 step 4 rebound `_launch_preflight`,
# `launch_instance` and `_emit_launched_soft` to `boxes.lifecycle`; step 6
# rebound the image gate and the four credential helpers to `launch.spec`
# (cli-surface.json H3 — four commands reach them, so they could not live in any
# `cli/<command>.py`). Asserting any of those eight raises would now be
# asserting the port did NOT happen, so they move to `_REBOUND_SEAMS` below,
# which asserts the opposite.
#
# The two left are blocked on the DIRECTION of an import, not on timing: both
# definitions exist (`jobs.bundle`, `fleet.client`) and both sit in a ring ABOVE
# `launch`, so import-linter forbids the edge at module scope AND inside a
# function. Only the `cli/` composition root can bind them.
#
# WHICH IT NOW DOES — and the two names below still belong here, deliberately.
# `cli/_compose.py::bind()` closes both, but it is called when a COMMAND RUNS
# (`cli.main.main()`, and `cli/launch.py` / `cli/train.py` / `cli/supervise.py`
# at the top of their `run()`), NOT at `cli` import. That is what keeps this
# census a property of the module rather than of pytest's collection order: any
# test file that imports `vastlib.cli` does so in this same process, and if the
# import bound these names, the assertions below would pass or fail depending on
# which file was collected first. `test_vastlib_cli_launch.py` drives a real
# `launch --jobs` through the binding and restores all three attributes around
# every one of its tests for exactly this reason.
_SEAM_NAMES = ["compose_jobs_launch_env", "fleet_watch_best_effort"]

# name in `launch` -> the module that now owns the definition
_REBOUND_SEAMS = {"_launch_preflight": lifecycle,
                  "launch_instance": lifecycle,
                  "_emit_launched_soft": lifecycle,
                  "_require_image": spec,
                  "hf_token_text": spec,
                  "hf_login_snippet": spec,
                  "image_login_arg": spec,
                  "_mask_image_login": spec}


@pytest.mark.parametrize("name", _SEAM_NAMES)
def test_every_seam_exists_and_raises(name):
    """A seam that quietly returned None would let `_do_launch` reach a green
    'launched' having resolved no image and PUT nothing (plan §7.3)."""
    fn = getattr(launch, name)
    assert callable(fn)
    with pytest.raises(NotImplementedError) as e:
        fn(*([None] * fn.__code__.co_argcount))
    assert name in str(e.value)


@pytest.mark.parametrize("name", _SEAM_NAMES)
def test_every_seam_names_a_live_flat_definition(name):
    """Each seam is a placeholder for a real function reachable as
    `herdd.<name>` — a re-export of its vastlib home since plan §8 step 6d,
    and still the spelling the flat-module consumers use. If a rename lands and
    the launcher's surface is not updated with it, this fails and names the
    seam."""
    assert callable(getattr(v, name)), f"{name} no longer exists on herdd"


def test_importing_the_cli_ring_does_not_move_a_name_off_the_raising_list():
    """The census must survive an import of the ring that closes it.

    This is the ordering trap in `_SEAM_NAMES`' banner, asserted rather than
    described: importing `vastlib.cli.launch` pulls in `cli/_compose.py` and
    every module it binds, and if any of that ran `bind()` at import time the
    two seams above would already be live — silently, and only in runs where a
    `cli` test file happened to be collected."""
    from vastlib.cli import launch as cli_launch  # noqa: F401 — the import IS the test
    for name in _SEAM_NAMES:
        with pytest.raises(NotImplementedError):
            getattr(launch, name)(*([None] * 2))


def test_the_two_seam_sets_are_disjoint_and_cover_the_ten():
    """The bookkeeping itself: ten names, no name in both lists. A rebind that
    forgot to leave `_SEAM_NAMES` would make `test_every_seam_exists_and_raises`
    fail loudly, but a name dropped from BOTH lists would just stop being
    checked — which is the failure this asserts against."""
    assert not set(_SEAM_NAMES) & set(_REBOUND_SEAMS)
    assert len(_SEAM_NAMES) + len(_REBOUND_SEAMS) == 10


@pytest.mark.parametrize("name", sorted(_REBOUND_SEAMS))
def test_rebound_seam_is_the_ported_definition(name):
    """The rebind is an ASSIGNMENT, so the two attributes are the SAME object.

    That identity is the whole contract: `monkeypatch.setattr(launch, name, …)`
    keeps steering `_do_launch` (the `_wire` idiom below), while the function
    under characterization in `test_vastlib_boxes_lifecycle.py` (or, for the
    five step-6 names, `test_vastlib_cli_helpers.py`) is literally this one. It
    also means a patch of `lifecycle.<name>` / `spec.<name>` is NOT visible
    here — deliberate, and stated in `launch.py`'s banner."""
    fn = getattr(launch, name)
    assert fn is getattr(_REBOUND_SEAMS[name], name)
    assert callable(fn)


@pytest.mark.parametrize("name, call", [
    # an unlabelled launch: the preflight fast-returns before any instance GET
    ("_launch_preflight", lambda: launch._launch_preflight(None, False)),
    # the PUT is real code now; conftest's guard refuses it and the failure
    # triple comes back, which is the point of the next test
    ("launch_instance", lambda: launch.launch_instance(1, {})),
    # no B2_BUCKET (the autouse fixture strips it) -> the emitter no-ops
    ("_emit_launched_soft",
     lambda: launch._emit_launched_soft(argparse.Namespace(gpu=[]),
                                        {"label": "run:r1"}, 1, 2, 0.5)),
    # the step-6 five. Every call below is chosen to touch no network, no
    # credential store and no mint: an image that IS set never reaches the gate's
    # sys.exit; an explicit token short-circuits every file probe; a PUBLIC image
    # host is not the private registry, so no deploy token and no HMAC mint.
    ("_require_image", lambda: launch._require_image("img:tag", "launch")),
    ("hf_token_text", lambda: launch.hf_token_text("explicit-not-a-real-token")),
    ("hf_login_snippet", lambda: launch.hf_login_snippet()),
    ("image_login_arg", lambda: launch.image_login_arg("pytorch/pytorch:2.4.0")),
    ("_mask_image_login", lambda: launch._mask_image_login("-u u -p T host")),
])
def test_rebound_seam_no_longer_raises(name, call):
    """The negative half of the rebind: none of the eight may raise
    `NotImplementedError` any more. Each call is chosen to touch no network —
    the preflight fast-returns on a non-`run:` label, the emitter fast-returns
    without `B2_BUCKET`, the PUT is intercepted by conftest's guard, and the
    five step-6 names take their no-I/O branch (see the parametrization)."""
    call()   # a NotImplementedError here means the rebind was reverted


# =============================================================================
# launch.py — _do_launch
# =============================================================================
def _launch_ns(**over):
    """The Namespace `_do_launch` reads (mirrors `test_broker_env._launch_ns`)."""
    base = dict(offer=None, type="ondemand", price=None, env=None, port=None,
                jupyter=False, onstart=None, no_hf_token=True, hf_token=None,
                ssh=False, ssh_key_file=None, jobs=False, image="img:tag",
                disk=40, runtype="ssh_direct", label=None, template_id=None,
                no_registry_login=True, login=None, dry_run=False, wait=None,
                force=False, eval_env_ver=None, cuda=0, num_gpus=1)
    base.update(over)
    return argparse.Namespace(**base)


def _wire(monkeypatch, *, offers_rows=None, searched=None):
    """Stub every seam and every cross-module read `_do_launch` touches, and
    return the list the captured launch body lands in. Nothing here reaches the
    network: `imageref.image_tag_digest` is stubbed too (it would otherwise
    query the GitLab registry)."""
    import imageref

    from vastlib.market import offers as offers_mod

    bodies = []
    rows = [{"id": 123, "min_bid": 0.20, "dph_total": 1.00}] \
        if offers_rows is None else offers_rows

    def _search(a):
        if searched is not None:
            searched.append(a)
        return rows

    monkeypatch.setattr(offers_mod, "search_offers", _search)
    monkeypatch.setattr(launch.fmt, "fmt_offer", lambda o: "offer-123")
    monkeypatch.setattr(imageref, "image_tag_digest", lambda img: None)
    monkeypatch.setattr(launch, "_require_image", lambda image, what: image)
    monkeypatch.setattr(launch, "_launch_preflight", lambda label, force: None)
    monkeypatch.setattr(launch, "_emit_launched_soft",
                        lambda a, body, cid, oid, dph: None)
    monkeypatch.setattr(launch, "launch_instance",
                        lambda oid, body: (bodies.append(body) or (True, 42, None)))
    return bodies


def test_do_launch_returns_and_mutates_the_namespace(monkeypatch):
    """`a.image` and `a.price` are written back onto the CALLER's Namespace —
    `cmd_train` reads `a.price` off the same object afterwards."""
    _wire(monkeypatch)
    ns = _launch_ns(type="bid", image=None)
    monkeypatch.setattr(launch, "_require_image", lambda image, what: "resolved:tag")
    monkeypatch.setattr(launch.pricing, "_offer_ondemand_ref", lambda o, n=None: 1.00)
    monkeypatch.setattr(launch.pricing, "_auto_bid_price", lambda mb, od=None: 0.24)
    cid, offer_id, dph = launch._do_launch(ns)
    assert (cid, offer_id, dph) == (42, 123, 0.24)
    assert ns.image == "resolved:tag"       # mutated in place
    assert ns.price == 0.24                 # cmd_train reads this back


def test_do_launch_stamps_the_nonce_and_entry_floor(monkeypatch):
    bodies = _wire(monkeypatch)
    launch._do_launch(_launch_ns())
    env = bodies[0]["env"]
    assert len(env["BOX_IDENTITY_NONCE"]) == 32
    assert env["ENTRY_FLOOR"] == "0.2000"


def test_do_launch_stamps_the_disk_it_ASKED_FOR(monkeypatch):
    """`disk_space` on the instance is what vast DELIVERED, and a host with less
    to give clamps rather than refusing — so the supervise lanes need the
    request itself, and the box env is the channel that survives a restart."""
    bodies = _wire(monkeypatch)
    launch._do_launch(_launch_ns(disk=50))
    assert bodies[0]["env"]["LAUNCH_DISK_GB"] == "50"
    assert bodies[0]["disk"] == 50


def test_do_launch_stamps_the_ARCHITECTURE_it_was_launched_under(monkeypatch):
    """`--cc-allow` is a statement about the WORKLOAD — which silicon its
    kernels have an image for — so it rides the same immutable box env as the
    disk request. Without it the replacement lane has nothing to read off an
    evicted box, and twice in two days a rehost landed a job on an sm_120 RTX
    PRO 6000 that could not run it (2026-08-17, 2026-08-18)."""
    bodies = _wire(monkeypatch)
    launch._do_launch(_launch_ns(cc_allow="90,80,86,89"))
    assert bodies[0]["env"]["LAUNCH_CC_ALLOW"] == "80,86,89,90"


def test_do_launch_stamps_no_architecture_when_none_was_declared(monkeypatch):
    """Additive: a launch that names no allowlist stamps nothing, and every box
    that predates the flag keeps behaving exactly as it did."""
    bodies = _wire(monkeypatch)
    launch._do_launch(_launch_ns())
    assert "LAUNCH_CC_ALLOW" not in bodies[0]["env"]
    bodies.clear()
    launch._do_launch(_launch_ns(cc_allow=""))
    assert "LAUNCH_CC_ALLOW" not in bodies[0]["env"]


def test_do_launch_warns_when_the_picked_offer_is_out_of_the_allowlist(monkeypatch,
                                                                      capsys):
    """The search narrows client-side, and a `--offer`/`--machine` pin skips the
    search — the same hole `_warn_disk_shortfall` covers for disk. A warning,
    not a refusal: the stamp is what binds the replacement lane, and an operator
    pinning a box has already chosen the hardware."""
    _wire(monkeypatch, offers_rows=[{"id": 123, "min_bid": 0.20,
                                     "dph_total": 1.00, "compute_cap": 1200}])
    launch._do_launch(_launch_ns(cc_allow="80,90"))
    err = capsys.readouterr().err
    assert "sm_120" in err and "--cc-allow" in err


def test_do_launch_is_quiet_when_the_offer_is_in_the_allowlist(monkeypatch, capsys):
    _wire(monkeypatch, offers_rows=[{"id": 123, "min_bid": 0.20,
                                     "dph_total": 1.00, "compute_cap": 900}])
    launch._do_launch(_launch_ns(cc_allow="80,90"))
    assert "cc-allow" not in capsys.readouterr().err


def test_do_launch_hands_the_disk_request_to_the_offer_search(monkeypatch):
    """The prevention half: `build_search_query` floors `disk_space` on
    `a.disk` (pinned in test_vastlib_market), so the SEARCH must see the same
    namespace the launch body is built from."""
    from vastlib.market import offers as offers_mod

    seen = {}

    def _search(a):
        seen["disk"] = getattr(a, "disk", None)
        return [{"id": 123, "min_bid": 0.20, "dph_total": 1.00,
                 "disk_space": 600.0}]

    _wire(monkeypatch)
    monkeypatch.setattr(offers_mod, "search_offers", _search)
    launch._do_launch(_launch_ns(disk=50))
    assert seen["disk"] == 50


def test_do_launch_warns_when_the_picked_offer_cannot_hold_the_disk(monkeypatch, capsys):
    """Box 48006308 asked for 50 GB, was picked onto a host advertising 13, and
    the first symptom was `insufficient_disk` on an already-billing box. The
    search floor cannot bind a `--offer` pin, so the row we hold is checked
    too."""
    _wire(monkeypatch, offers_rows=[{"id": 123, "min_bid": 0.20,
                                     "dph_total": 1.00, "disk_space": 13.0}])
    launch._do_launch(_launch_ns(disk=50))
    err = capsys.readouterr().err
    assert "13G" in err and "50G" in err


def test_do_launch_is_quiet_when_the_offer_has_the_room(monkeypatch, capsys):
    _wire(monkeypatch, offers_rows=[{"id": 123, "min_bid": 0.20,
                                     "dph_total": 1.00, "disk_space": 600.0}])
    launch._do_launch(_launch_ns(disk=50))
    assert "container disk" not in capsys.readouterr().err


def test_do_launch_nonce_is_fresh_per_call(monkeypatch):
    bodies = _wire(monkeypatch)
    launch._do_launch(_launch_ns())
    launch._do_launch(_launch_ns())
    assert bodies[0]["env"]["BOX_IDENTITY_NONCE"] != bodies[1]["env"]["BOX_IDENTITY_NONCE"]


def test_do_launch_explicit_env_wins_over_setdefault(monkeypatch):
    bodies = _wire(monkeypatch)
    launch._do_launch(_launch_ns(env=["BOX_IDENTITY_NONCE=mine"]))
    assert bodies[0]["env"]["BOX_IDENTITY_NONCE"] == "mine"


def test_do_launch_rejects_a_malformed_env(monkeypatch):
    _wire(monkeypatch)
    with pytest.raises(SystemExit) as e:
        launch._do_launch(_launch_ns(env=["NOEQUALS"]))
    assert str(e.value) == "error: --env expects KEY=VALUE, got 'NOEQUALS'"


def test_do_launch_refuses_when_no_offers_match(monkeypatch):
    _wire(monkeypatch, offers_rows=[])
    with pytest.raises(SystemExit) as e:
        launch._do_launch(_launch_ns())
    assert "no offers match filters" in str(e.value)


# --- the fail-closed prologue runs BEFORE any offer search -------------------
def test_image_gate_refuses_before_the_offer_search(monkeypatch):
    """test_lifecycle.py pins the same ordering on the flat copy: a launch the
    image gate refuses must not have searched."""
    searched = []
    _wire(monkeypatch, searched=searched)

    def _refuse(image, what):
        sys.exit(f"error: no image to {what} with")
    monkeypatch.setattr(launch, "_require_image", _refuse)
    with pytest.raises(SystemExit):
        launch._do_launch(_launch_ns(image=None))
    assert not searched, "the offer search must not have run"


def test_empty_eval_env_ver_refuses_before_the_offer_search(monkeypatch):
    searched = []
    _wire(monkeypatch, searched=searched)
    with pytest.raises(SystemExit) as e:
        launch._do_launch(_launch_ns(eval_env_ver="  "))
    assert "--eval-env-ver was given an empty value" in str(e.value)
    assert not searched, "the offer search must not have run"


def test_contradicting_eval_env_ver_refuses_before_the_offer_search(monkeypatch):
    searched = []
    _wire(monkeypatch, searched=searched)
    with pytest.raises(SystemExit) as e:
        launch._do_launch(_launch_ns(eval_env_ver="v9", env=["EVAL_ENV_VER=v8"]))
    assert str(e.value) == ("error: --eval-env-ver v9 contradicts "
                            "--env EVAL_ENV_VER=v8. One box, one baked env: "
                            "say it once.")
    assert not searched, "the offer search must not have run"


def test_eval_env_ver_folds_into_env_when_it_agrees(monkeypatch):
    bodies = _wire(monkeypatch)
    launch._do_launch(_launch_ns(eval_env_ver="v9"))
    assert bodies[0]["env"]["EVAL_ENV_VER"] == "v9"


# =============================================================================
# launch.py — THE DRY-RUN REDACTION. Fails OPEN if the binding goes stale.
# =============================================================================
def _dry_run_body(out):
    """The dry-run JSON, past the `picked: ...` line the offer pick prints."""
    return json.loads(out[out.index("{"):])["body"]


def test_dry_run_never_prints_a_secret_value(monkeypatch, capsys):
    """A fake secret handed in through --env must render `<redacted>` in
    `launch --dry-run` stdout, on the VASTLIB path. This is the only guard
    keeping HF/B2 token values off the terminal and out of shell history."""
    _wire(monkeypatch)
    launch._do_launch(_launch_ns(
        dry_run=True,
        env=["HF_TOKEN=FAKE-SECRET-VALUE-DO-NOT-PRINT",
             "B2_APPLICATION_KEY=FAKE-SECRET-TWO",
             "DATABASE_URL=postgres://u:FAKE-SECRET-THREE@h/db",
             "PLAIN_SETTING=visible"]))
    out = capsys.readouterr().out
    for leaked in ("FAKE-SECRET-VALUE-DO-NOT-PRINT", "FAKE-SECRET-TWO",
                   "FAKE-SECRET-THREE"):
        assert leaked not in out, "dry-run leaked a secret VALUE"
    shown = _dry_run_body(out)["env"]
    assert shown["HF_TOKEN"] == "<redacted>"
    assert shown["B2_APPLICATION_KEY"] == "<redacted>"
    assert shown["DATABASE_URL"] == "<redacted>"
    assert shown["PLAIN_SETTING"] == "visible"   # non-secrets stay visible
    assert "HF_TOKEN" in shown                   # NAMES are deliberately kept


def test_dry_run_redaction_goes_through_the_spec_module_attribute(monkeypatch, capsys):
    """`_do_launch` must call `spec._is_secret_env`, not a `from … import`
    snapshot: patching the module attribute has to change the output. If this
    passes while `test_dry_run_never_prints_a_secret_value` also passes, the
    redaction is live rather than incidentally correct."""
    _wire(monkeypatch)
    monkeypatch.setattr(spec, "_is_secret_env", lambda k, val: k == "PLAIN_SETTING")
    launch._do_launch(_launch_ns(
        dry_run=True, env=["PLAIN_SETTING=now-secret", "HF_TOKEN=now-visible"]))
    shown = _dry_run_body(capsys.readouterr().out)["env"]
    assert shown["PLAIN_SETTING"] == "<redacted>"
    assert shown["HF_TOKEN"] == "now-visible"


def test_dry_run_returns_without_launching(monkeypatch):
    bodies = _wire(monkeypatch)
    monkeypatch.setattr(launch, "_launch_preflight",
                        lambda label, force: pytest.fail("dry-run must not preflight"))
    cid, offer_id, dph = launch._do_launch(_launch_ns(dry_run=True))
    assert (cid, dph) == (None, None)
    assert offer_id == 123
    assert bodies == [], "dry-run must not PUT"


# =============================================================================
# launch.py — LAUNCH-THROUGH: the money path is real code now (plan §8 step 4)
#
# Before the rebind these were impossible: `_do_launch` could not reach a PUT
# at all, because `launch_instance` raised. Every test here drives the whole
# assembly and stops at a different point on the create sequence.
# =============================================================================
def _wire_through(monkeypatch, *, instances=None):
    """`_wire`, minus the three step-4 stubs: the REAL rebound preflight, PUT
    and runmeta emitter run. `lifecycle._instances` is stubbed because the
    preflight would otherwise issue a hard instance GET (a read — conftest's
    guard passes those THROUGH to the live API), and `B2_BUCKET` is already
    stripped by the autouse fixture, which is what makes the emitter a no-op."""
    bodies = _wire(monkeypatch)
    monkeypatch.setattr(launch, "_launch_preflight", lifecycle._launch_preflight)
    monkeypatch.setattr(launch, "_emit_launched_soft", lifecycle._emit_launched_soft)
    monkeypatch.setattr(lifecycle, "_instances", lambda: list(instances or []))
    return bodies


def test_do_launch_reaches_a_stubbed_launch_instance_end_to_end(monkeypatch, capsys):
    """The happy path, with only the PUT itself stubbed: real image gate
    (`_wire`'s identity stub), real offer pick, real body assembly, REAL
    preflight, REAL runmeta emitter. The captured body is the one the PUT would
    have carried."""
    bodies = _wire_through(monkeypatch)
    cid, offer_id, dph = launch._do_launch(_launch_ns(label="run:r1"))
    assert (cid, offer_id) == (42, 123)
    assert dph == 1.00                    # the picked offer's on-demand dph_total
    assert len(bodies) == 1 and bodies[0]["label"] == "run:r1"
    assert "launched instance 42" in capsys.readouterr().out


def test_do_launch_is_refused_by_the_real_preflight_when_a_twin_is_live(monkeypatch):
    """The rebound preflight is REACHED from `_do_launch`, not merely importable:
    a live `run:r1` twin must abort the launch before the PUT, with the message
    text `test_lifecycle.py` pins."""
    bodies = _wire_through(monkeypatch, instances=[
        {"id": 11, "label": "run:r1", "actual_status": "running"}])
    with pytest.raises(SystemExit) as e:
        launch._do_launch(_launch_ns(label="run:r1"))
    assert "live instance [11]" in str(e.value)
    assert bodies == [], "the preflight must refuse BEFORE the PUT"


def test_do_launch_put_is_intercepted_by_the_conftest_guard(monkeypatch):
    """THE MONEY MOVE, unstubbed. Nothing replaces `launch_instance` here, so
    `_do_launch` runs the real `PUT v0/asks/<offer>/` — and conftest's
    `_block_mutating_api_calls` refuses it, `launch_instance` reports the
    failure, and `_do_launch` exits.

    Two things are proved at once. (1) The guard is IN FRONT of the money move:
    an unstubbed launch test is red, not a real rental. (2) The PUT went through
    the `vastlib.core.api.request_soft` module ATTRIBUTE — the guard's message
    carries the method and path, and only the wrapped attribute can produce it.
    A `from vastlib.core.api import request_soft` inside `launch_instance` would
    bind past the guard and this test would issue a live, billable PUT."""
    _wire(monkeypatch)
    monkeypatch.setattr(launch, "launch_instance", lifecycle.launch_instance)
    monkeypatch.setattr(launch, "_launch_preflight", lambda label, force: None)
    with pytest.raises(SystemExit) as e:
        launch._do_launch(_launch_ns())
    msg = str(e.value)
    assert "test isolation" in msg and "blocked" in msg
    assert "PUT v0/asks/123/" in msg


def test_do_launch_exits_on_a_launch_failure_without_emitting(monkeypatch):
    """`sys.exit(f"error: {err}")` sits BETWEEN the PUT and the runmeta emit, so
    a failed launch records nothing — asserted here because the emitter is real
    in this file now and a reordering would be invisible otherwise."""
    _wire_through(monkeypatch)
    emitted = []
    monkeypatch.setattr(launch, "launch_instance",
                        lambda oid, body: (False, None, "no available gpus"))
    monkeypatch.setattr(launch, "_emit_launched_soft",
                        lambda *a, **k: emitted.append(a))
    with pytest.raises(SystemExit) as e:
        launch._do_launch(_launch_ns(label="run:r1"))
    assert "no available gpus" in str(e.value)
    assert emitted == []


# =============================================================================
# the `compose_jobs_launch_env` seam — RAISING, but with the REAL signature
# =============================================================================
# Added 2026-08-16 with `vastlib/jobs/bundle.py` (plan §8 step 5). The seam stays
# in `_SEAM_NAMES` above — `jobs` is the ring ABOVE `launch`, so this name can
# never be rebound the way the three `boxes.lifecycle` ones were, and the `cli/`
# composition root binds it at step 6. What DID change is the stub's signature:
# as first written it declared only `(env, onstart, *, dry_run, no_idle_park,
# idle_park_grace, no_job_deadline)` and dropped `key_base`, `timeout_s` and
# `bootstrap_stager`, all three of which live callers pass. A narrow stub is a
# silent trap for a composition root: binding the real function behind it
# type-checks clean and breaks the workflow lane at runtime, because
# `workflowctl.build_box_resolver` injects `bootstrap_stager` positionally by
# keyword and nothing else would notice.


def test_compose_jobs_seam_signature_matches_the_real_function():
    """The seam and its eventual implementation must be substitutable.

    The third arm (`flat = inspect.signature(v.compose_jobs_launch_env)`) went
    at plan §8 step 6d: the launcher re-exports `bundle.compose_jobs_launch_env`,
    so it was `real`'s signature under another name. Seam vs implementation is
    the comparison with content — they are two genuinely different objects."""
    import inspect

    from vastlib.jobs import bundle

    seam = inspect.signature(launch.compose_jobs_launch_env)
    real = inspect.signature(bundle.compose_jobs_launch_env)
    assert list(seam.parameters) == list(real.parameters)
    for name, p in real.parameters.items():
        assert seam.parameters[name].kind == p.kind
        assert seam.parameters[name].default == p.default


def test_compose_jobs_seam_still_raises_with_the_real_arguments():
    """`_SEAM_NAMES` calls every seam with positional `None`s; that would pass
    even if the three keyword-only arguments were still missing. This drives the
    exact call shape `workflowctl` uses."""
    with pytest.raises(NotImplementedError) as e:
        launch.compose_jobs_launch_env({}, "", dry_run=True, key_base="k",
                                       timeout_s=3600,
                                       bootstrap_stager=lambda **kw: "sha")
    assert "compose_jobs_launch_env" in str(e.value)
