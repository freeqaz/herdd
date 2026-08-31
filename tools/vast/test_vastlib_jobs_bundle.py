"""Portable tests for `vastlib.jobs.bundle` — what ships to a box, ported at
plan §8 step 5.

Why this file is shaped the way it is
-------------------------------------
Every failure mode in `bundle.py` is SILENT by construction, so "the tests pass"
is not evidence unless each one is provoked deliberately:

1. **The two `__file__` walk-backs.** `_job_attach_files`, `_jobd_import_gate`
   and `_jobd_boot_snippet` all resolved paths from `tools/vast/` in the flat
   module; this file sits two directories deeper. A wrong depth does not raise —
   it yields a bundle of files that do not exist, or a `repo_root` that makes
   `shipcheck`'s relpaths produce `../..` keys that match nothing. The gate then
   swallows the resulting exception as a one-line NOTE. So both constants are
   pinned against the SAME expression applied to `herdd.py`'s own PATH. That
   pin outlived the thinning: plan §4 freezes the path (the reaper unit's
   `ExecStart` names it literally), so `dirname(abspath(herdd.__file__))` is
   still the anchor these two constants have to land on, whatever is inside the
   file.

2. **The import gate's blanket `except Exception`.** Asserting it "passes on the
   real bundle" is worthless on its own: a gate that can no longer resolve
   `shipcheck` passes too, silently, forever. Every gate test here therefore
   comes in a pair — the positive control (a bundle missing `bidpolicy.py` MUST
   exit) next to the negative one, plus an explicit assertion that a pass
   printed no NOTE.

3. **The raw `rclone rcat`.** `_stage_jobd_bootstrap` does NOT route its upload
   through `storage.b2._b2_rcat`, which passes `text=True` and would decode the
   tar. Stubbing the storage seam therefore leaves the write LIVE. This file
   stubs `bundle.subprocess` — the module's own attribute — and asserts the body
   is still `bytes` and that no `text=`/`capture_output=` crept in.

Parity spine, and its end (plan §8 step 6d): while the flat original existed,
the ported function and `herdd`'s were run over the same input and compared —
drift in either direction failed, with no hand-written expectation. The
thinning left one body per name, so five of those comparisons became
self-comparisons and are deleted (`_job_attach_files`, `_jobd_import_gate`'s
refusal string, `_jobd_boot_snippet`, `_stage_jobd_bootstrap`'s dry-run sha,
`compose_jobs_launch_env` key-for-key). Each sits next to a characterization
test that states the property outright, which is what the parity arm was
standing in for; the bindings themselves are asserted at the bottom of this
file. The `compose` one had additionally gone VACUOUS: it steered its nonce
sources with `monkeypatch.setattr(herdd, …)`, and a re-export is not a patch
point (launcher docstring, rule 2) — post-thinning those patches steered
nothing and the test would have run the real minting path.

Offline lane: no network, no B2, no rclone, no vast API, $0. Nothing here writes
outside `tmp_path`.
"""
from __future__ import annotations

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import herdd as v  # noqa: E402
from vastlib.jobs import bundle  # noqa: E402
from vastlib.launch import spec  # noqa: E402
from vastlib.storage import b2  # noqa: E402

TOOLS_VAST = os.path.dirname(os.path.abspath(v.__file__))


# =============================================================================
# the two __file__ walk-backs — the pins, and nothing else fails when they drift
# =============================================================================
def test_tools_vast_dir_matches_the_herdd_computation():
    """`os.path.dirname(os.path.abspath(__file__))` applied to `herdd.py`.
    Three `dirname`s from `vastlib/jobs/bundle.py` must reach the same place."""
    assert bundle.TOOLS_VAST_DIR == TOOLS_VAST
    assert os.path.isdir(bundle.TOOLS_VAST_DIR)
    assert os.path.isfile(os.path.join(bundle.TOOLS_VAST_DIR, "jobmeta.py"))


def test_repo_root_matches_the_herdd_computation():
    """`_jobd_import_gate`'s `root` — three `dirname`s above `herdd.py`, five
    above this module. It is the base every `shipcheck` relpath is taken from."""
    flat_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(v.__file__))))
    assert bundle._REPO_ROOT == flat_root
    assert os.path.isdir(os.path.join(bundle._REPO_ROOT, "tools", "vast"))


def test_tools_vast_dir_agrees_with_the_other_two_copies():
    """`storage.b2` and `core.config` walk the same path for `b2_sync.sh` and
    `herdd.yaml`. Three copies of one path is fine; three ANSWERS is not."""
    from vastlib.core import config
    assert bundle.TOOLS_VAST_DIR == b2.TOOLS_VAST_DIR == config._HERE


# =============================================================================
# _job_attach_files — THE delivery manifest, read by three other checkers
# =============================================================================
def test_attach_files_is_the_object_shipcheck_loads():
    """Was `…_is_identical_to_the_flat_original`, comparing the two lists.

    The list is read by the flat-bundle test, `shipcheck.JOBD_FLAT_NAMESPACES`
    and `test_broker_env._PINNED_JOBD_BUNDLE`, so a drift is a drift in all
    four at once — which is why the binding still matters after step 6d
    collapsed the two copies into one: `shipcheck.py` loads `herdd.py` by
    `spec_from_file_location` and calls `mod._job_attach_files()`, so it
    reaches this function THROUGH the launcher's namespace."""
    assert v._job_attach_files is bundle._job_attach_files


def test_every_bundled_file_actually_exists():
    """`_stage_jobd_bootstrap` `sys.exit`s on the first miss, so a wrong
    `TOOLS_VAST_DIR` presents as a refusal naming one file rather than as a
    depth bug. Fail here, with the whole list, instead."""
    missing = [f for f in bundle._job_attach_files() if not os.path.isfile(f)]
    assert missing == []


def test_attach_files_is_not_the_last_def_in_the_module():
    """`test_preempt_save.py` slices the function's SOURCE with the regex
    `def _job_attach_files\\(\\):(.*?)\\ndef ` — it needs a following `def` to
    terminate the match. When that test is repointed here at step 6 (it carries
    its own "re-point this test" instruction), a `_job_attach_files` that ended
    the file would make the regex fail to match and the test would report a
    missing `preempt_save.py` literal that is right there."""
    src = inspect.getsource(bundle)
    head, _, tail = src.partition("def _job_attach_files(")
    assert head and tail, "the def moved or was renamed"
    assert "\ndef " in tail


def test_the_preempt_and_serve_pairs_ship_together():
    """Each of these three pairs is a delivery defect that already shipped once:
    a file whose SIBLING resolver silently degrades when the partner is absent."""
    names = {os.path.basename(f) for f in bundle._job_attach_files()}
    for a, b in (("preempt_save.py", "preempt_trap.sh"),
                 ("job_serve.sh", "serve_vllm.sh"),
                 ("gemm_probe.py", "metrics_probe.py")):
        assert a in names and b in names, f"{a}/{b} must ship together"
    assert "bidpolicy.py" in names       # the 84d09ab1 incident itself


# =============================================================================
# _jobd_import_gate — fail-closed, with the positive control next to every pass
# =============================================================================
def test_import_gate_passes_the_real_bundle_and_prints_no_note(capsys):
    """The pass and the SKIP look identical from the outside — a gate that can
    no longer import `shipcheck`, or that computed a wrong `repo_root`, also
    "passes". The absence of the NOTE is the real assertion here."""
    bundle._jobd_import_gate(bundle._job_attach_files())      # must not exit
    err = capsys.readouterr().err
    assert "IMPORT CLOSURE BROKEN" not in err
    assert "check skipped" not in err, (
        "the gate degraded to a NOTE — it is not actually checking anything; "
        "suspect _REPO_ROOT, TOOLS_VAST_DIR or the sys.path.insert")


def test_import_gate_refuses_a_bundle_missing_bidpolicy(capsys):
    """THE positive control (mirrors `test_jobd_bundle_imports_flat.py`'s). This
    is the 2026-08-14 bundle: `jobmeta.py` imports `bidpolicy` at module scope,
    so without it every `python3 jobd.py` on a box dies — swallowed by jobd.sh's
    `|| true` into a box that looks idle and bills."""
    files = [f for f in bundle._job_attach_files()
             if os.path.basename(f) != "bidpolicy.py"]
    with pytest.raises(SystemExit) as e:
        bundle._jobd_import_gate(files)
    assert "not import-closed" in str(e.value)
    err = capsys.readouterr().err
    assert "bidpolicy" in err and "jobmeta.py" in err


def test_import_gate_refuses_a_bundle_broken_at_the_box_floor(capsys):
    """A bundle can be import-closed on THIS python and still unparseable on the
    box's (PEP 701 is 3.12+; stock images ship 3.11). The gate must refuse at
    stage/attach time because the shipper is not always a tested checkout —
    fleetd re-stages from its own tree (boxes 48094838/48132001, 2026-08-19)."""
    probe = os.path.join(bundle.TOOLS_VAST_DIR, "_floor_probe_test_only.py")
    with open(probe, "w") as f:
        f.write('x = "observed"\n'
                's = f"{\'a\'\n'
                '     if x else \'b\'}"\n')
    try:
        with pytest.raises(SystemExit) as e:
            bundle._jobd_import_gate(bundle._job_attach_files() + [probe])
        assert "not import-closed" in str(e.value)
        err = capsys.readouterr().err
        assert "the box floor" in err and "_floor_probe_test_only.py" in err
    finally:
        os.unlink(probe)


# `test_import_gate_refusal_matches_the_flat_original` was here: it raised out
# of both copies on the same truncated file list and compared `str(e.value)`,
# because tests match that refusal by substring (plan §7.4). Step 6d left one
# copy, so it compared a string with itself. Deleted — the test above asserts
# the substring (`"not import-closed"`) outright, which is what the flat side
# was standing in for.


def test_import_gate_escape_hatches_are_explicit(monkeypatch, capsys):
    files = [f for f in bundle._job_attach_files()
             if os.path.basename(f) != "bidpolicy.py"]
    monkeypatch.setenv("JOBD_NO_IMPORT_CHECK", "1")
    bundle._jobd_import_gate(files)                           # must not exit
    assert "shipping the broken bundle anyway" in capsys.readouterr().err
    monkeypatch.delenv("JOBD_NO_IMPORT_CHECK")
    bundle._jobd_import_gate(files, warn_only=True)           # must not exit
    assert "shipping the broken bundle anyway" in capsys.readouterr().err


def test_import_gate_degrades_to_a_note_when_shipcheck_explodes(monkeypatch,
                                                                capsys):
    """The blanket `except Exception` is load-bearing — a guard must never be the
    reason a box cannot be attached — and it is also why every port mistake in
    this function is invisible. Provoke it once, on purpose, so the NOTE text is
    pinned and the tests above can trust its ABSENCE."""
    import shipcheck

    def boom(*a, **k):
        raise RuntimeError("shipcheck exploded")

    monkeypatch.setattr(shipcheck, "jobd_import_closure_gaps", boom)
    bundle._jobd_import_gate(bundle._job_attach_files())      # must not exit
    assert "jobd import-closure check skipped" in capsys.readouterr().err


def test_import_gate_passes_files_explicitly_so_shipcheck_cannot_re_exec():
    """`shipped=None` makes `shipcheck` `exec_module()` the source file again
    under a synthetic name, producing a second live copy of the module with its
    own state. The gate exists to prevent that; assert the argument is threaded."""
    import shipcheck
    seen = {}

    real = shipcheck.jobd_import_closure_gaps

    def spy(root, shipped=None):
        seen["root"], seen["shipped"] = root, shipped
        return real(root, shipped)

    shipcheck.jobd_import_closure_gaps = spy
    try:
        bundle._jobd_import_gate(bundle._job_attach_files())
    finally:
        shipcheck.jobd_import_closure_gaps = real
    assert seen["shipped"] is not None and len(seen["shipped"]) > 10
    assert seen["root"] == bundle._REPO_ROOT
    # relpaths, not `../..`-escaped absolutes — the silent-drift tell
    assert not any(k.startswith("..") for k in seen["shipped"])
    assert "tools/vast/jobmeta.py" in seen["shipped"]


# =============================================================================
# _jobd_boot_snippet — the second __file__-anchored read
# =============================================================================
# `test_boot_snippet_matches_the_flat_original` was here (one call each, same
# sha, compared). One body since step 6d; the substitution properties it stood
# in for are asserted directly below.


def test_boot_snippet_substitutes_the_sha_and_leaves_no_placeholder():
    out = bundle._jobd_boot_snippet("deadbeef" * 8)
    assert "@JOBD_BUNDLE_SHA@" not in out
    assert "deadbeef" * 8 in out
    assert out.endswith("\n")


# =============================================================================
# _stage_jobd_bootstrap — dry-run, dedupe, and the RAW binary upload
# =============================================================================
class _FakeProc:
    def __init__(self, rc=0):
        self.returncode = rc


class _FakeSubprocess:
    """Stands in for `bundle.subprocess`. Records the whole call so the test can
    assert on the body's TYPE and on the absence of `text=`/`capture_output=`."""
    def __init__(self, rc=0):
        self.rc = rc
        self.calls = []

    def run(self, args, **kw):
        self.calls.append((list(args), kw))
        return _FakeProc(self.rc)


@pytest.fixture
def staged(monkeypatch):
    """`_stage_jobd_bootstrap` with the B2 seam and the raw subprocess stubbed,
    and a bucket in env. Returns the fake subprocess for inspection."""
    fake = _FakeSubprocess()
    monkeypatch.setenv("B2_BUCKET", "fake-bucket")
    monkeypatch.setattr(bundle, "subprocess", fake)
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(b2, "_rclone", lambda args: (1, ""))   # dedupe MISS
    return fake


def test_stage_dry_run_writes_nothing_and_returns_the_sha(monkeypatch, capsys):
    """`workflowctl.py:2735` calls this in a plan/preview path and relies on the
    sha coming back with no B2 write."""
    calls = []
    monkeypatch.setattr(bundle, "subprocess", _FakeSubprocess())
    monkeypatch.setattr(b2, "_ensure_b2_remote",
                        lambda: calls.append("ensure"))
    monkeypatch.setattr(b2, "_rclone", lambda args: calls.append(args) or (1, ""))
    sha = bundle._stage_jobd_bootstrap(dry_run=True)
    assert len(sha) == 64 and int(sha, 16) >= 0
    assert calls == []
    assert "would stage jobd bundle" in capsys.readouterr().out


def test_stage_dry_run_sha_is_deterministic(monkeypatch, capsys):
    """Content-addressed over the FOLDER: the same 18 files must hash the same
    on every call, or a `launch --jobs` and a `job attach` disagree about which
    bootstrap object the box should pull.

    The third arm (`c = v._stage_jobd_bootstrap(dry_run=True)`, "the same hash
    from either module") went at step 6d — `v._stage_jobd_bootstrap` IS this
    function, so it was a third call to the same code, not a second module.
    Note the surviving arms still catch the real hazard: the hash must not
    depend on call order, cwd, or anything else that varies between the two
    entry points."""
    monkeypatch.setattr(bundle, "subprocess", _FakeSubprocess())
    a = bundle._stage_jobd_bootstrap(dry_run=True)
    b = bundle._stage_jobd_bootstrap(dry_run=True)
    capsys.readouterr()
    assert a == b


def test_stage_uploads_with_a_raw_rcat_and_a_BYTES_body(staged, capsys):
    """THE hazard. `storage.b2._b2_rcat` passes `text=True`; routing this through
    it would decode the tar into mojibake, the box would pull a corrupt bundle,
    and `jobd.sh` would swallow it with `|| true`. Assert the call shape."""
    sha = bundle._stage_jobd_bootstrap()
    assert len(staged.calls) == 1
    args, kw = staged.calls[0]
    assert args[:2] == ["rclone", "rcat"]
    assert args[2] == f"b2:fake-bucket/jobs/jobd-boot/{sha}.tar"
    assert isinstance(kw["input"], bytes), "the tar must not be decoded"
    assert "text" not in kw and "capture_output" not in kw
    assert "staged jobd bundle" in capsys.readouterr().out


def test_stage_reuses_an_existing_object_without_uploading(monkeypatch, capsys):
    """Immutable content-addressed object: an `lsf` hit means the bytes are
    already there, and re-uploading them is money for nothing."""
    fake = _FakeSubprocess()
    monkeypatch.setenv("B2_BUCKET", "fake-bucket")
    monkeypatch.setattr(bundle, "subprocess", fake)
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(b2, "_rclone", lambda args: (0, "jobd-boot.tar\n"))
    bundle._stage_jobd_bootstrap()
    assert fake.calls == []
    assert "reusing" in capsys.readouterr().out


def test_stage_exits_when_the_upload_fails(monkeypatch):
    monkeypatch.setenv("B2_BUCKET", "fake-bucket")
    monkeypatch.setattr(bundle, "subprocess", _FakeSubprocess(rc=1))
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(b2, "_rclone", lambda args: (1, ""))
    with pytest.raises(SystemExit) as e:
        bundle._stage_jobd_bootstrap()
    assert "failed to stage jobd bootstrap bundle" in str(e.value)


def test_stage_refuses_a_missing_bundle_file(monkeypatch):
    """Pre-spend refusal, and the shape a wrong `TOOLS_VAST_DIR` would take."""
    monkeypatch.setattr(bundle, "_job_attach_files",
                        lambda: ["/nonexistent/jobd.sh"])
    with pytest.raises(SystemExit) as e:
        bundle._stage_jobd_bootstrap(dry_run=True)
    assert "missing jobd file to stage" in str(e.value)


def test_stage_runs_the_import_gate_before_touching_b2(monkeypatch):
    """Order matters: an unimportable bundle must be refused for $0, before the
    remote is configured and before the PUT."""
    order = []
    monkeypatch.setenv("B2_BUCKET", "fake-bucket")
    monkeypatch.setattr(bundle, "_jobd_import_gate",
                        lambda files, **kw: order.append("gate"))
    monkeypatch.setattr(bundle, "subprocess", _FakeSubprocess())
    monkeypatch.setattr(b2, "_ensure_b2_remote",
                        lambda: order.append("ensure"))
    monkeypatch.setattr(b2, "_rclone",
                        lambda args: order.append("lsf") or (1, ""))
    bundle._stage_jobd_bootstrap()
    assert order[0] == "gate"


# =============================================================================
# compose_jobs_launch_env — the defect-#6 parity seam
# =============================================================================
@pytest.fixture
def composable(monkeypatch):
    """A hermetic compose: no mint, no staging, no B2. `_ship_b2_env` and
    `_minted_expiry` are patched on `launch.spec` as MODULE ATTRIBUTES, which is
    the whole point of the module-attribute calling convention (plan §8(b)) —
    `test_broker_env.py`'s three `_ship_b2_env` patch sites survive the port
    only because of it."""
    monkeypatch.setenv("B2_BUCKET", "fake-bucket")
    monkeypatch.setenv("B2_S3_ENDPOINT", "https://s3.example")
    monkeypatch.delenv("B2_REGION", raising=False)
    monkeypatch.delenv("CRED_BROKER_URL", raising=False)
    monkeypatch.delenv("TS_AUTHKEY", raising=False)
    monkeypatch.setattr(spec, "_ship_b2_env",
                        lambda base, hours, write_prefix=None, dry_run=False:
                        [("B2_KEY_ID", "kid"), ("B2_APP_KEY", "secret")])
    monkeypatch.setattr(spec, "_minted_expiry", lambda base, hours: 1234567890)
    monkeypatch.setattr(spec, "_b2_eu_pairs", list)
    monkeypatch.setattr(spec, "_r2_tc_pairs", list)
    monkeypatch.setattr(bundle, "_jobd_boot_snippet", lambda sha: f"#boot {sha}\n")


def test_compose_mutates_env_in_place_and_prepends_the_boot_snippet(composable):
    env = {}
    onstart, sha = bundle.compose_jobs_launch_env(
        env, "echo hi\n", bootstrap_stager=lambda **kw: "SHA")
    assert sha == "SHA"
    assert onstart == "#boot SHA\necho hi\n"
    assert env["B2_BUCKET"] == "fake-bucket"
    assert env["B2_S3_ENDPOINT"] == "https://s3.example"
    assert env["B2_REGION"] == "us-west-004"          # the documented default
    assert env["CRED_ROLE"] == "jobs"
    assert env["B2_KEY_ID"] == "kid" and env["B2_APP_KEY"] == "secret"
    assert env["B2_KEY_EXPIRES_AT"] == "1234567890"
    assert len(env["BOX_IDENTITY_NONCE"]) == 32


def test_compose_never_overwrites_an_explicit_value(composable):
    """`setdefault` throughout — an explicit value already present always wins.
    A `--jobs` launch that had already stamped its nonce must be unchanged."""
    env = {"BOX_IDENTITY_NONCE": "mine", "B2_REGION": "eu-central-003",
           "CRED_ROLE": "run"}
    bundle.compose_jobs_launch_env(env, None, bootstrap_stager=lambda **kw: "s")
    assert env["BOX_IDENTITY_NONCE"] == "mine"
    assert env["B2_REGION"] == "eu-central-003"
    assert env["CRED_ROLE"] == "run"


def test_compose_mints_a_UNIQUE_key_base_per_launch(composable, monkeypatch):
    """The 2026-07-12 box-44566398 incident: a second `--jobs`/workflow launch
    reusing one mint base REVOKES the still-running box's key."""
    bases = []
    monkeypatch.setattr(spec, "_ship_b2_env",
                        lambda base, hours, write_prefix=None, dry_run=False:
                        bases.append(base) or [])
    for _ in range(4):
        bundle.compose_jobs_launch_env({}, None, bootstrap_stager=lambda **kw: "s")
    assert len(set(bases)) == 4
    assert all(b.startswith("job-launch-") for b in bases)


def test_compose_honours_an_explicit_key_base(composable, monkeypatch):
    bases = []
    monkeypatch.setattr(spec, "_ship_b2_env",
                        lambda base, hours, write_prefix=None, dry_run=False:
                        bases.append(base) or [])
    bundle.compose_jobs_launch_env({}, None, key_base="wf-7",
                                   bootstrap_stager=lambda **kw: "s")
    assert bases == ["wf-7"]


def test_compose_scopes_the_minted_write_to_the_jobs_prefix(composable,
                                                            monkeypatch):
    """Bucket-wide read + `jobs/`-restricted write. A wider grant is what the
    submit-side B2 write-scope gate is measuring against."""
    seen = {}
    monkeypatch.setattr(spec, "_ship_b2_env",
                        lambda base, hours, write_prefix=None, dry_run=False:
                        seen.update(prefix=write_prefix, hours=hours,
                                    dry=dry_run) or [])
    bundle.compose_jobs_launch_env({}, None, timeout_s=7200, dry_run=True,
                                   bootstrap_stager=lambda **kw: "s")
    assert seen["prefix"] == "jobs/"
    assert seen["dry"] is True
    assert seen["hours"] == spec._ephemeral_hours(7200)


def test_compose_threads_dry_run_into_the_injected_stager(composable):
    """`workflowctl.build_box_resolver` injects `bootstrap_stager`; the preview
    path relies on `dry_run` reaching it."""
    seen = {}
    bundle.compose_jobs_launch_env(
        {}, None, dry_run=True,
        bootstrap_stager=lambda **kw: seen.update(kw) or "s")
    assert seen == {"dry_run": True}


def test_compose_defaults_the_stager_to_stage_jobd_bootstrap(composable,
                                                             monkeypatch):
    """`bootstrap_stager=None` must fall through to the real one — the default is
    resolved at CALL time (a module attribute), so a patch of
    `bundle._stage_jobd_bootstrap` steers it."""
    monkeypatch.setattr(bundle, "_stage_jobd_bootstrap",
                        lambda dry_run=False: "DEFAULT")
    _, sha = bundle.compose_jobs_launch_env({}, None)
    assert sha == "DEFAULT"


def test_compose_refuses_without_queue_transport(monkeypatch):
    """A jobd box with no queue can never claim a job, so this is a hard error
    rather than a degraded launch."""
    monkeypatch.delenv("B2_BUCKET", raising=False)
    monkeypatch.setenv("B2_S3_ENDPOINT", "https://s3.example")
    with pytest.raises(SystemExit) as e:
        bundle.compose_jobs_launch_env({}, None)
    assert "--jobs needs B2_BUCKET/B2_S3_ENDPOINT" in str(e.value)


def test_compose_ships_the_broker_pair_only_together(composable, monkeypatch):
    """`TS_AUTHKEY` rides only when `CRED_BROKER_URL` is set — the tailnet key is
    useless without the broker and must not be sprayed onto every box."""
    env = {}
    monkeypatch.setenv("TS_AUTHKEY", "tskey-fake")
    bundle.compose_jobs_launch_env(env, None, bootstrap_stager=lambda **kw: "s")
    assert "TS_AUTHKEY" not in env and "CRED_BROKER_URL" not in env
    env = {}
    monkeypatch.setenv("CRED_BROKER_URL", "https://broker.example")
    bundle.compose_jobs_launch_env(env, None, bootstrap_stager=lambda **kw: "s")
    assert env["CRED_BROKER_URL"] == "https://broker.example"
    assert env["TS_AUTHKEY"] == "tskey-fake"


def test_compose_idle_park_knobs_are_assignments_not_setdefaults(composable):
    """The three park knobs use `env[...] = `, deliberately: `--no-idle-park` is
    an explicit operator instruction and must beat anything already in env."""
    env = {"JOBD_IDLE_PARK": "1", "JOBD_IDLE_PARK_S": "60",
           "JOBD_NO_JOB_PARK_S": "60"}
    bundle.compose_jobs_launch_env(env, None, no_idle_park=True,
                                   idle_park_grace=900, no_job_deadline=1800,
                                   bootstrap_stager=lambda **kw: "s")
    assert env["JOBD_IDLE_PARK"] == "0"
    assert env["JOBD_IDLE_PARK_S"] == "900"
    assert env["JOBD_NO_JOB_PARK_S"] == "1800"


# `test_compose_matches_the_flat_original_key_for_key` was here: it composed
# the launch env through both copies with the nonce sources pinned, popped the
# deliberately-random `BOX_IDENTITY_NONCE`, and compared the mutated dicts key
# for key. It is deleted for TWO reasons, and the second one is why it is not
# merely redundant:
#
#   1. Step 6d left one `compose_jobs_launch_env`, so it composed twice through
#      the same function.
#   2. It pinned its nonce sources with `monkeypatch.setattr(herdd, …)`. A
#      re-export is not a patch point (launcher docstring, rule 2): those five
#      patches rebound names in the launcher's namespace that
#      `vastlib.jobs.bundle` never reads, so post-thinning they steered
#      nothing — and `_ship_b2_env` unstubbed is the LIVE B2 key mint. It went
#      green because both arms ran the same unpatched code.
#
# The mutation itself keeps its coverage in the `composable`-fixture tests
# above, which stub the seams on the OWNING module.


# =============================================================================
# the launcher's bindings — the residue of the five deleted parity tests
# =============================================================================
def test_the_launcher_re_exports_rather_than_redefines():
    """`shipcheck.py` loads `herdd.py` BY PATH (`spec_from_file_location`) and
    calls `mod._job_attach_files()`; `launch_serve.sh`'s heredoc imports the
    module by name. Those consumers make the launcher's namespace a live
    delivery surface, so a second body under any of these names ships a
    different bundle from the one this file tests."""
    for name in ("_job_attach_files", "_jobd_boot_snippet", "_jobd_import_gate",
                 "_stage_jobd_bootstrap", "_sync_file_list",
                 "compose_jobs_launch_env"):
        assert getattr(v, name) is getattr(bundle, name), (
            f"herdd.{name} is a second body again — the launcher must "
            f"re-export vastlib.jobs.bundle's object, never redefine it")
