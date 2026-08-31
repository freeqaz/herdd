"""`vastlib.jobs.runlocal` — the ported local-GPU lane, held to its traps.

Why this file exists
--------------------
`test_joblocal.py` is the lane's end-to-end coverage and it drives
`tools/vast/herdd.py` as a SUBPROCESS — zero imports, so it neither needed
nor got a repoint, and it did not bind to the ported code until `herdd.py`
became a thin launcher at plan §8 step 6d (it does now). That left four
properties of the ported module with nothing checking them in the meantime,
and they stay checked here because a subprocess end-to-end test cannot see
them:

1. **`_JOB_LOCAL` is a MUTABLE CROSS-MODULE GLOBAL.** It lives here and is read
   by `view._live_iids_set` / `view._present_iids_set` — the only two places in
   the jobs lane that reach for the vast API. `_job_local_activate()` must flip
   it in a way the READERS see; if `view` had done
   `from .runlocal import _JOB_LOCAL`, the flip would be invisible and the local
   lane would silently start hitting the real API with real credentials, which
   is exactly what `LOCAL_GPU_LANE.md` promises never to happen.
2. **`view` and `runlocal` import each other.** The cycle is legal only because
   neither touches the other's attributes at import time. Both orders are run in
   a fresh interpreter.
3. **The GPU refusal is authorized by CONFIG, and checked FIRST** — before any
   directory is created or any card probed, so a refusal leaves no debris. One
   switch (`allow_local_gpu`), one home (`core.config`), owner ruling
   2026-08-11.
4. **`_JOB_LOCAL_SUBCOMMANDS` is a CLI-surface contract** — `attach`/`retarget`/
   `requeue`/`supervise` are box concepts and deliberately have no `--local`.

What is deliberately NOT here
-----------------------------
* No real jobd, no `nvidia-smi`, no subprocess that touches a card.
  `joblocal.*` is stubbed as module attributes throughout, and the one
  `subprocess.call(["bash", JOBD_SH])` is intercepted.
* No re-testing of `joblocal.py` itself (`test_joblocal.py` owns it) and no
  re-testing of `require_local_gpu`'s policy (`core.config`'s own tests own it) —
  only that this module calls DOWN into it rather than re-deciding.

Provenance: created 2026-08-16 alongside `vastlib/jobs/runlocal.py`, plan §8
step 5.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

import jobmeta                                         # noqa: E402  Zone S
import joblocal                                        # noqa: E402  absorbed sibling

from vastlib.boxes import lifecycle                    # noqa: E402
from vastlib.core import api, config                   # noqa: E402
from vastlib.jobs import runlocal, submit, view        # noqa: E402


def _ns(**kw):
    return argparse.Namespace(**kw)


@pytest.fixture(autouse=True)
def _flag_is_restored(monkeypatch):
    """`_JOB_LOCAL` is a module global that `_job_local_activate` writes with
    `global`, so monkeypatch cannot see the assignment — restore it by hand or
    one test leaks the local lane into every later one."""
    monkeypatch.setattr(runlocal, "_JOB_LOCAL", False)
    yield
    runlocal._JOB_LOCAL = False


# --------------------------------------------------------------------------- #
# 1. THE MUTABLE CROSS-MODULE GLOBAL
# --------------------------------------------------------------------------- #
def test_activate_flips_the_flag_and_returns_the_home(monkeypatch):
    monkeypatch.setattr(joblocal, "activate", lambda home=None: "/tmp/lane-home")
    assert runlocal._JOB_LOCAL is False
    assert runlocal._job_local_activate() == "/tmp/lane-home"
    assert runlocal._JOB_LOCAL is True


def test_activate_is_idempotent(monkeypatch):
    calls = []
    monkeypatch.setattr(joblocal, "activate",
                        lambda home=None: calls.append(1) or "/tmp/h")
    runlocal._job_local_activate()
    runlocal._job_local_activate()
    assert runlocal._JOB_LOCAL is True and len(calls) == 2   # activate itself is idempotent


def test_the_flip_is_visible_to_views_api_readers(monkeypatch):
    """THE HAZARD. `view` must read `runlocal._JOB_LOCAL` at CALL time. A
    `from .runlocal import _JOB_LOCAL` binds False at import, and both readers
    below would then reach for the real vast API with real credentials."""
    monkeypatch.setattr(joblocal, "activate", lambda home=None: "/tmp/h")
    monkeypatch.setattr(joblocal, "live_boxes", lambda home=None: {"local-box"})
    monkeypatch.setattr(lifecycle, "_instances_soft",
                        lambda: pytest.fail("the LOCAL lane must not reach the API"))
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: pytest.fail("the LOCAL lane must not reach the API"))

    runlocal._job_local_activate()
    assert view._live_iids_set() == {"local-box"}
    assert view._present_iids_set() is None


def test_view_and_runlocal_import_in_either_order():
    """Neither module may touch the other's attributes at IMPORT time, or the
    cycle becomes an ImportError that depends on which entry point ran first."""
    for first, second in (("vastlib.jobs.runlocal", "vastlib.jobs.view"),
                          ("vastlib.jobs.view", "vastlib.jobs.runlocal")):
        code = (f"import sys; sys.path.insert(0, {str(VAST_DIR)!r}); "
                f"import {first} as a; import {second} as b; "
                f"print(a.__name__, b.__name__)")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, cwd=str(VAST_DIR))
        assert r.returncode == 0, f"{first} first: {r.stderr}"


def test_the_flag_starts_false_at_import():
    """Import must not activate anything: `herdd ls` imports this module too."""
    code = (f"import sys; sys.path.insert(0, {str(VAST_DIR)!r}); "
            "from vastlib.jobs import runlocal; print(runlocal._JOB_LOCAL)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=str(VAST_DIR))
    assert r.returncode == 0 and r.stdout.strip() == "False"


# --------------------------------------------------------------------------- #
# 2. The dispatcher and the --local surface
# --------------------------------------------------------------------------- #
def test_cmd_job_activates_the_lane_only_on_local(monkeypatch):
    activated = []
    monkeypatch.setattr(runlocal, "_job_local_activate",
                        lambda: activated.append(1))
    ran = []
    runlocal.cmd_job(_ns(local=False, jobfunc=lambda a: ran.append("plain")))
    assert activated == [] and ran == ["plain"]

    runlocal.cmd_job(_ns(local=True, jobfunc=lambda a: ran.append("local")))
    assert activated == [1] and ran == ["plain", "local"]


def test_cmd_job_activates_BEFORE_dispatching(monkeypatch):
    """The subcommand's first B2 read must already be pointed at the local
    bucket — activating afterwards would send it at the real one."""
    order = []
    monkeypatch.setattr(runlocal, "_job_local_activate",
                        lambda: order.append("activate"))
    runlocal.cmd_job(_ns(local=True, jobfunc=lambda a: order.append("dispatch")))
    assert order == ["activate", "dispatch"]


def test_cmd_job_treats_a_missing_local_attr_as_false(monkeypatch):
    monkeypatch.setattr(runlocal, "_job_local_activate",
                        lambda: pytest.fail("no --local attr => no activation"))
    runlocal.cmd_job(_ns(jobfunc=lambda a: None))


def test_the_local_subcommand_roster_excludes_every_box_concept():
    """`attach`/`retarget`/`requeue`/`supervise` install a daemon on / move work
    between / babysit the bid of a RENTED machine. A CLI-surface contract checked
    at step 6 by the §4 full-help diff; pinned here so a drift is caught earlier."""
    assert runlocal._JOB_LOCAL_SUBCOMMANDS == (
        "submit", "status", "wait", "logs", "pull", "ls", "cancel")
    for box_only in ("attach", "retarget", "requeue", "supervise", "run-local"):
        assert box_only not in runlocal._JOB_LOCAL_SUBCOMMANDS


# --------------------------------------------------------------------------- #
# 3. The preflight — config authorizes, this module does not
# --------------------------------------------------------------------------- #
def _pre_args(**kw):
    base = dict(root=None, gpus=None, force=False)
    base.update(kw)
    return _ns(**base)


def test_the_gpu_refusal_is_checked_first_and_leaves_no_debris(monkeypatch,
                                                               tmp_path):
    """Before any directory is created and before any card is probed."""
    def _refuse(lane, cfg=None):
        raise SystemExit(f"error: local GPU lane disabled ({lane})")

    monkeypatch.setattr(config, "require_local_gpu", _refuse)
    monkeypatch.setattr(joblocal, "probe_gpus",
                        lambda: pytest.fail("refusal comes before the probe"))
    root = tmp_path / "never-created"
    with pytest.raises(SystemExit) as ei:
        runlocal._run_local_preflight(_pre_args(root=str(root)), None, None)
    assert "`job run-local`" in str(ei.value.code)
    assert not root.exists()


def test_the_preflight_refuses_when_no_card_is_visible(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "require_local_gpu", lambda lane, cfg=None: None)
    monkeypatch.setattr(joblocal, "probe_gpus", lambda: [])
    with pytest.raises(SystemExit) as ei:
        runlocal._run_local_preflight(_pre_args(root=str(tmp_path / "r")), None, None)
    assert "no GPU visible to nvidia-smi" in str(ei.value.code)


def test_the_preflight_refuses_a_gpus_flag_that_selects_nothing(monkeypatch,
                                                                tmp_path):
    monkeypatch.setattr(config, "require_local_gpu", lambda lane, cfg=None: None)
    monkeypatch.setattr(joblocal, "probe_gpus", lambda: [(0, 24576, "3090")])
    with pytest.raises(SystemExit) as ei:
        runlocal._run_local_preflight(_pre_args(root=str(tmp_path / "r"), gpus="7"),
                                      None, [7])
    assert "selects no probed card" in str(ei.value.code)


def test_the_preflight_refuses_a_busy_card_and_never_kills_anything(monkeypatch,
                                                                    tmp_path,
                                                                    capsys):
    """jobd's own orphan reaper is disabled for this lane (JOBD_GPU_REAP=0); this
    refusal is the second half — do not silently contend for a card someone is
    already training on."""
    monkeypatch.setattr(config, "require_local_gpu", lambda lane, cfg=None: None)
    monkeypatch.setattr(joblocal, "probe_gpus", lambda: [(0, 24576, "3090")])
    monkeypatch.setattr(joblocal, "foreign_gpu_procs",
                        lambda idxs: [(4242, 0, "python train.py")])
    with pytest.raises(SystemExit) as ei:
        runlocal._run_local_preflight(_pre_args(root=str(tmp_path / "r")), None, None)
    assert "we will NOT kill anything either way" in str(ei.value.code)
    assert "pid=4242" in capsys.readouterr().err


def test_force_downgrades_the_busy_refusal_to_a_warning(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "require_local_gpu", lambda lane, cfg=None: None)
    monkeypatch.setattr(joblocal, "probe_gpus", lambda: [(0, 24576, "3090")])
    monkeypatch.setattr(joblocal, "foreign_gpu_procs",
                        lambda idxs: [(4242, 0, "python train.py")])
    root, warns = runlocal._run_local_preflight(
        _pre_args(root=str(tmp_path / "r"), force=True), None, None)
    assert root == str(tmp_path / "r")
    assert any("--force over busy GPUs" in w for w in warns)


def test_the_preflight_falls_back_to_joblocals_workspace_dir(monkeypatch,
                                                             tmp_path):
    monkeypatch.setattr(config, "require_local_gpu", lambda lane, cfg=None: None)
    monkeypatch.setattr(joblocal, "probe_gpus", lambda: [(0, 24576, "3090")])
    monkeypatch.setattr(joblocal, "foreign_gpu_procs", lambda idxs: [])
    monkeypatch.setattr(joblocal, "workspace_dir",
                        lambda home=None: str(tmp_path / "ws"))
    root, warns = runlocal._run_local_preflight(_pre_args(), "/tmp/home", None)
    assert root == str(tmp_path / "ws") and warns == []
    assert (tmp_path / "ws").is_dir()


# --------------------------------------------------------------------------- #
# 4. Asset-dest warnings
# --------------------------------------------------------------------------- #
def test_an_absolute_dest_outside_the_local_root_is_named(tmp_path):
    """`_link_asset_dest` refuses any dest outside $JOBD_ROOT, so it would
    silently not link — the entrypoint then fails confusingly."""
    out = runlocal._run_local_asset_warnings(
        {"assets": [{"name": "base", "dest": "/workspace/models"}]}, str(tmp_path))
    assert len(out) == 1 and "jobd will REFUSE to link it" in out[0]


def test_a_relative_dest_and_an_in_root_absolute_dest_are_quiet(tmp_path):
    cfg = {"assets": [{"name": "a", "dest": "models/base"},
                      {"name": "b", "dest": str(tmp_path / "models")},
                      {"name": "c"}]}
    assert runlocal._run_local_asset_warnings(cfg, str(tmp_path)) == []


def test_no_assets_means_no_warnings():
    assert runlocal._run_local_asset_warnings({}, "/workspace") == []


# --------------------------------------------------------------------------- #
# 5. cmd_job_run_local — the drain, without touching a card
# --------------------------------------------------------------------------- #
def _wire_lane(monkeypatch, tmp_path, *, rc=0, amap=None):
    monkeypatch.setattr(joblocal, "activate", lambda home=None: str(tmp_path / "home"))
    monkeypatch.setattr(joblocal, "differences_banner", lambda: "-- differences --")
    monkeypatch.setattr(joblocal, "parse_gpu_allow", lambda spec: None)
    monkeypatch.setattr(joblocal, "load_asset_map", lambda home=None: dict(amap or {}))
    monkeypatch.setattr(joblocal, "save_asset_map", lambda m, home=None: "")
    monkeypatch.setattr(joblocal, "seed_asset", lambda root, n, p: "")
    monkeypatch.setattr(joblocal, "parse_asset_arg", lambda s: tuple(s.split("=", 1)))
    monkeypatch.setattr(joblocal, "local_box_id", lambda host=None: "local-box")
    monkeypatch.setattr(joblocal, "bucket_dir", lambda home=None: str(tmp_path / "bkt"))
    monkeypatch.setattr(joblocal, "jobd_env", lambda *a, **k: {"JOBD_ROOT": "x"})
    monkeypatch.setattr(joblocal, "JOBD_SH", str(tmp_path / "jobd.sh"))
    monkeypatch.setattr(runlocal, "_run_local_preflight",
                        lambda a, home, allow: (str(tmp_path / "root"), []))
    calls = []
    monkeypatch.setattr(runlocal.subprocess, "call",
                        lambda argv, env=None: calls.append((argv, env)) or rc)
    return calls


def _lane_args(**kw):
    base = dict(dir=None, gpus=None, asset=None, dry_run=False, watch=False,
                cpu_slots=1, name=None, timeout=None, env=None, root=None,
                force=False)
    base.update(kw)
    return _ns(**base)


def test_run_local_drains_without_submitting_when_given_no_folder(monkeypatch,
                                                                  tmp_path,
                                                                  capsys):
    """That IS the resume path — re-running is jobd's own, same JOB_ID."""
    calls = _wire_lane(monkeypatch, tmp_path)
    monkeypatch.setattr(submit, "cmd_job_submit",
                        lambda a: pytest.fail("no --dir => nothing is submitted"))
    runlocal.cmd_job_run_local(_lane_args())
    assert len(calls) == 1 and calls[0][0][0] == "bash"
    assert "running jobd" in capsys.readouterr().out


def test_run_local_dry_run_starts_no_daemon(monkeypatch, tmp_path, capsys):
    calls = _wire_lane(monkeypatch, tmp_path)
    runlocal.cmd_job_run_local(_lane_args(dry_run=True))
    assert calls == []
    assert "[dry-run] preflight only" in capsys.readouterr().out


def test_run_local_submits_through_jobs_submit_with_the_local_namespace(
        monkeypatch, tmp_path):
    """The staleness preflight compares a LOCAL source against a B2 prefix, and
    there is no B2 in this lane — so `no_asset_check` and `local` are pinned."""
    _wire_lane(monkeypatch, tmp_path)
    monkeypatch.setattr(jobmeta, "load_job_config", lambda d: {"name": "x"})
    monkeypatch.setattr(jobmeta, "validate_job_config",
                        lambda raw, d, **k: ({"name": "x", "assets": []}, []))
    seen = {}

    def _sub(ns):
        seen["ns"] = ns
        return "20260816T000000-x-aaaa"

    monkeypatch.setattr(submit, "cmd_job_submit", _sub)
    monkeypatch.setattr(jobmeta, "read_job",
                        lambda jid, live_iids=None: {"status": "done"})
    monkeypatch.setattr(view, "_print_job_view", lambda v: None)
    monkeypatch.setattr(view, "_live_iids_set", lambda: {"local-box"})
    runlocal.cmd_job_run_local(_lane_args(dir="/some/bundle"))
    ns = seen["ns"]
    assert ns.box == "local-box" and ns.no_asset_check is True and ns.local is True
    assert ns.dry_run is False and ns.strict_assets is False


def test_run_local_warns_about_an_asset_with_no_local_override(monkeypatch,
                                                               tmp_path, capsys):
    _wire_lane(monkeypatch, tmp_path)
    monkeypatch.setattr(jobmeta, "load_job_config", lambda d: {})
    monkeypatch.setattr(jobmeta, "validate_job_config",
                        lambda raw, d, **k: ({"assets": [{"name": "base"}]}, []))
    monkeypatch.setattr(submit, "cmd_job_submit", lambda ns: None)
    runlocal.cmd_job_run_local(_lane_args(dir="/some/bundle"))
    assert "asset_stage_failed:base" in capsys.readouterr().err


def test_run_local_never_masks_jobds_exit_code(monkeypatch, tmp_path):
    _wire_lane(monkeypatch, tmp_path, rc=3)
    with pytest.raises(SystemExit) as ei:
        runlocal.cmd_job_run_local(_lane_args())
    assert "jobd exited 3" in str(ei.value.code)


def test_a_fold_failure_after_the_run_never_masks_jobds_rc(monkeypatch, tmp_path,
                                                           capsys):
    _wire_lane(monkeypatch, tmp_path, rc=0)
    monkeypatch.setattr(jobmeta, "load_job_config", lambda d: {})
    monkeypatch.setattr(jobmeta, "validate_job_config",
                        lambda raw, d, **k: ({"assets": []}, []))
    monkeypatch.setattr(submit, "cmd_job_submit", lambda ns: "j-1")
    monkeypatch.setattr(view, "_live_iids_set", lambda: {"local-box"})
    monkeypatch.setattr(jobmeta, "read_job",
                        lambda jid, live_iids=None: (_ for _ in ()).throw(
                            RuntimeError("fold blew up")))
    runlocal.cmd_job_run_local(_lane_args(dir="/some/bundle"))    # no raise
    cap = capsys.readouterr()
    assert "could not fold j-1" in cap.err
    assert "resume: " in cap.out and "job run-local" in cap.out


def test_a_stale_asset_override_is_dropped_with_a_warning(monkeypatch, tmp_path,
                                                          capsys):
    _wire_lane(monkeypatch, tmp_path, amap={"base": "/gone/from/disk"})
    runlocal.cmd_job_run_local(_lane_args())
    assert "no longer exists — ignoring" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# 6. TWIN IDENTITY — one copy since plan §8 step 6d
# --------------------------------------------------------------------------- #
# The header here said `herdd.py` "keeps its originals until plan step 6".
# Step 6d thinned it, so the roster comparison, the warning-text comparison and
# the two-copies-both-False assertion all became statements about one object
# and are deleted. `test_joblocal.py` still drives `tools/vast/herdd.py` as a
# SUBPROCESS, which is now this module executing — the end-to-end arm the
# deleted comparisons were standing in for until the thinning landed.
import herdd                                        # noqa: E402


def test_the_launcher_re_exports_rather_than_redefines():
    """`_JOB_LOCAL` in particular: a re-export is a BINDING, not a copy.

    The mutable-global hazard at §1 of this file has a launcher-shaped
    variant. `herdd._JOB_LOCAL` is a name bound to the same False object at
    import; `_job_local_activate()` rebinds `runlocal._JOB_LOCAL`, and the
    launcher's name does NOT follow — which is fine, because nothing reads the
    launcher's copy. What must never happen is the launcher DEFINING these
    names, because then `herdd.py job --local` and every other entry point
    would run two different rosters.
    """
    for name in ("_JOB_LOCAL_SUBCOMMANDS", "_run_local_asset_warnings"):
        assert getattr(herdd, name) is getattr(runlocal, name), (
            f"herdd.{name} is a second body again — the launcher must "
            f"re-export vastlib.jobs.runlocal's object, never redefine it")
    assert "_JOB_LOCAL" in vars(herdd), (
        "the launcher dropped its _JOB_LOCAL re-export; joblocal's subprocess "
        "lane and the six flat-module consumers read it by that name")
