"""`vastlib.jobs.view` — the ported job READ layer, held to its traps.

Why this file exists
--------------------
Five properties of this module survive a move only if something checks them,
and the existing flat tests could not check the ported copy while it was being
written (they drove `herdd`'s, live through the add-only phase and left
untouched here; at plan §8 step 6d that copy is gone and they reach this
module):

1. **`_present_iids_set` is TRI-STATE.** `set` / `None` because the listing was
   unreadable / `None` because we are in the local lane. Every one of the three
   is pinned, plus the reason the function calls `api.request_soft` DIRECTLY:
   `lifecycle._instances_soft` flattens an API error to `[]`, and reading `[]`
   as "every box is destroyed" is what would let `job orphans --resolve -y`
   cancel a whole fleet's queue on one HTTP 500. Highest-blast-radius contract
   in the jobs lane and it had no direct test before this file.
2. **The repo-root depth.** `herdd._repo_root()` is three `os.path.dirname`
   calls from `tools/vast/herdd.py`; from `tools/vast/vastlib/jobs/view.py`
   the same expression yields `tools/vast` and NOTHING FAILS — `find_job_defs`
   would just find zero bundles, which is indistinguishable from a repo with no
   bundles.
3. **`_JOB_LOCAL` is read as `runlocal._JOB_LOCAL` at CALL time.** A
   `from .runlocal import _JOB_LOCAL` binds `False` at import and the local lane
   silently starts hitting the real vast API. The test flips the flag AFTER
   import and asserts the reader follows.
4. **`cmd_job_wait`'s exit codes** (0 / 2 / 124 / SystemExit-with-a-string) are
   a frozen shell contract with ZERO coverage anywhere in the tree until now.
5. **`_fold_fleet_jobs` swallows everything into `{}`** and tolerates the RAW
   INT box ids `fleetd.py` passes it. A port that raises a new exception type
   still yields `{}` there; a port that changes the RETURN SHAPE turns every
   box's health verdict blind and silently. So it is asserted on OUTPUT.

What is deliberately NOT here
-----------------------------
* No repoint of any existing test. `test_jobprogress_rate.py`,
  `test_job_logs_provenance.py`, `test_job_orphans.py` and friends still target
  `herdd.<name>` and still steer `herdd`'s callers; they migrate with their
  callers at plan steps 6-7 and are re-run UNEDITED as this port's gate.
* No network, no box, no rclone. `api.request_soft` is stubbed as a MODULE
  ATTRIBUTE on `vastlib.core.api`, `b2._ensure_b2_remote`/`b2._rclone` on
  `vastlib.storage.b2`, and every `jobmeta` touch on the Zone S module.
* No re-testing of `_tqdm_points` / `_step_delta_s` — `test_vastlib_jobs_risk.py`
  owns those; only the three display functions that consume them are here.

Provenance: created 2026-08-16 alongside `vastlib/jobs/view.py`, plan §8 step 5.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

import jobmeta                                         # noqa: E402  Zone S
import joblocal                                        # noqa: E402  absorbed sibling

from vastlib.boxes import lifecycle, reap              # noqa: E402
from vastlib.core import api                           # noqa: E402
from vastlib.jobs import risk, runlocal, scan, view          # noqa: E402
from vastlib.storage import b2                         # noqa: E402


def _ns(**kw):
    import argparse
    return argparse.Namespace(**kw)


@pytest.fixture(autouse=True)
def _no_b2(monkeypatch):
    """`_ensure_b2_remote` shells rclone config; nothing here needs a remote."""
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)


# --------------------------------------------------------------------------- #
# 1. The repo root — the one line of the port that is not verbatim
# --------------------------------------------------------------------------- #
def test_repo_root_matches_herdd_computation():
    """The ported constant == what `herdd.py` computes for itself at runtime."""
    herdd_py = os.path.abspath(str(VAST_DIR / "herdd.py"))
    expected = os.path.dirname(os.path.dirname(os.path.dirname(herdd_py)))
    assert view._REPO_ROOT == expected
    assert os.path.isdir(os.path.join(view._REPO_ROOT, "tools", "vast"))


def test_naive_file_arithmetic_here_would_be_wrong():
    """Copying the three-dirname expression verbatim lands in tools/vast.

    Not an error — a silently different directory, so only a comparison catches
    it. `find_job_defs` under the naive root finds zero bundles, which reads
    exactly like a repo that has none.
    """
    naive = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(view.__file__))))
    assert naive.endswith(os.path.join("tools", "vast"))
    assert naive != view._REPO_ROOT
    assert view.find_job_defs(naive) == []
    assert view.find_job_defs() != []          # ... and the real root finds them


# --------------------------------------------------------------------------- #
# 2. _present_iids_set — THE TRI-STATE
# --------------------------------------------------------------------------- #
def test_present_iids_set_returns_the_account_as_strings(monkeypatch):
    monkeypatch.setattr(runlocal, "_JOB_LOCAL", False)
    monkeypatch.setattr(api, "request_soft",
                        lambda m, p, *a, **k: (True, {"instances": [
                            {"id": 41}, {"id": 42, "actual_status": "exited"}]}, None))
    assert view._present_iids_set() == {"41", "42"}


@pytest.mark.parametrize("soft", [
    (False, None, "http 500"),                 # the API refused
    (True, {"instances": "not-a-list"}, None),  # the API answered nonsense
    (True, None, None),                        # ... or answered nothing at all
])
def test_present_iids_set_is_none_when_the_listing_is_unreadable(monkeypatch, soft):
    """The load-bearing None. `[]` would mean "the account is empty", i.e. every
    box destroyed — and `job orphans --resolve -y` cancels on that reading."""
    monkeypatch.setattr(runlocal, "_JOB_LOCAL", False)
    monkeypatch.setattr(api, "request_soft", lambda m, p, *a, **k: soft)
    assert view._present_iids_set() is None


def test_present_iids_set_is_none_in_the_local_lane_without_touching_the_api(
        monkeypatch):
    """Third state, different reason: presence is meaningless when the machine
    is always there — and the lane promises never to touch a credential."""
    monkeypatch.setattr(runlocal, "_JOB_LOCAL", True)

    def _boom(*a, **k):
        pytest.fail("the LOCAL lane must not reach the vast API")

    monkeypatch.setattr(api, "request_soft", _boom)
    assert view._present_iids_set() is None


def test_present_iids_set_does_not_go_through_the_soft_instances_wrapper(
        monkeypatch):
    """`_instances_soft` flattens an API error into `[]`, which is exactly the
    value the tri-state exists to avoid. Prove the wrapper is not in the path."""
    monkeypatch.setattr(runlocal, "_JOB_LOCAL", False)
    monkeypatch.setattr(lifecycle, "_instances_soft",
                        lambda: pytest.fail("must call request_soft directly"))
    monkeypatch.setattr(api, "request_soft", lambda m, p, *a, **k: (False, None, "boom"))
    assert view._present_iids_set() is None


# --------------------------------------------------------------------------- #
# 3. _live_iids_set — strings, and the local lane
# --------------------------------------------------------------------------- #
def test_live_iids_set_stringifies_the_int_ids_the_api_returns(monkeypatch):
    """The str() is the fix for `job ls` calling every box in the account dead."""
    monkeypatch.setattr(runlocal, "_JOB_LOCAL", False)
    monkeypatch.setattr(lifecycle, "_instances_soft", lambda: [
        {"id": 41, "actual_status": "running"},
        {"id": 42, "actual_status": "loading"},
        {"id": 43, "actual_status": "exited"}])
    got = view._live_iids_set()
    assert got == {"41", "42"}
    assert all(isinstance(x, str) for x in got)
    assert 41 not in got                      # the bug: int membership never hits


def test_live_iids_set_takes_the_local_answer(monkeypatch):
    monkeypatch.setattr(runlocal, "_JOB_LOCAL", True)
    monkeypatch.setattr(joblocal, "live_boxes", lambda home=None: {"local-box"})
    monkeypatch.setattr(lifecycle, "_instances_soft",
                        lambda: pytest.fail("the LOCAL lane must not reach the API"))
    assert view._live_iids_set() == {"local-box"}


def test_the_job_local_flag_is_read_at_call_time_not_bound_at_import(monkeypatch):
    """Plan §8b. Flipping `runlocal._JOB_LOCAL` AFTER `view` was imported must
    steer both readers; a `from .runlocal import _JOB_LOCAL` would freeze False
    and send the local lane at the real vast API."""
    monkeypatch.setattr(joblocal, "live_boxes", lambda home=None: {"local-box"})
    monkeypatch.setattr(lifecycle, "_instances_soft", lambda: [])
    monkeypatch.setattr(api, "request_soft", lambda m, p, *a, **k: (True, {"instances": []}, None))

    monkeypatch.setattr(runlocal, "_JOB_LOCAL", False)
    assert view._live_iids_set() == set() and view._present_iids_set() == set()

    monkeypatch.setattr(runlocal, "_JOB_LOCAL", True)
    assert view._live_iids_set() == {"local-box"}
    assert view._present_iids_set() is None


def test_view_and_runlocal_import_in_either_order():
    """The two modules import each other. Neither may touch the other's
    attributes at IMPORT time, or the cycle becomes an ImportError that depends
    on which entry point ran first."""
    for first, second in (("vastlib.jobs.view", "vastlib.jobs.runlocal"),
                          ("vastlib.jobs.runlocal", "vastlib.jobs.view")):
        code = (f"import sys; sys.path.insert(0, {str(VAST_DIR)!r}); "
                f"import {first}; import {second}; print('ok')")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, cwd=str(VAST_DIR))
        assert r.returncode == 0, f"{first} first: {r.stderr}"
        assert "ok" in r.stdout


# --------------------------------------------------------------------------- #
# 4. The view cache — what may be frozen
# --------------------------------------------------------------------------- #
def test_only_done_and_cancelled_are_cacheable():
    """`failed` is re-openable by `job requeue`; freezing it is what raised
    ZOMBIE_NO_JOBD against a healthy box at step 142/156 on 2026-08-07."""
    assert view._JOB_VIEW_STICKY == frozenset({"done", "cancelled"})
    for st in ("done", "cancelled"):
        assert view._job_view_cacheable({"status": st})
    for st in ("failed", "running", "interrupted", "submitted", "claimed"):
        assert not view._job_view_cacheable({"status": st})
    assert not view._job_view_cacheable({})
    assert not view._job_view_cacheable(None)


def test_cacheable_reads_the_raw_fold_status_not_display_status():
    """The distinction IS the bug fix: a `display_status` of `done` over a fold
    of `failed` must not freeze."""
    assert not view._job_view_cacheable({"status": "failed", "display_status": "done"})


def _fold_env(monkeypatch, tmp_path, pairs, jobs):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(jobmeta, "list_all_queued", lambda: pairs)
    monkeypatch.setattr(jobmeta, "read_job",
                        lambda jid, live_iids=None: dict(jobs[jid]))


def test_fold_fleet_jobs_attaches_to_target_and_claiming_box(monkeypatch, tmp_path):
    _fold_env(monkeypatch, tmp_path, [("41", "j-1")],
              {"j-1": {"status": "running", "instance_id": 42}})
    out = view._fold_fleet_jobs({"41"})
    assert set(out) == {"41", "42"}            # BOTH, and both as strings
    assert out["41"][0]["status"] == "running"


def test_fold_fleet_jobs_lists_a_retargeted_ticket_once_per_box(monkeypatch, tmp_path):
    """A retargeted ticket sits in BOTH queue prefixes, and the old target's
    pair attaches it to the claiming box as well — so without a dedupe the box
    running it once renders it twice."""
    _fold_env(monkeypatch, tmp_path, [("41", "j-1"), ("42", "j-1")],
              {"j-1": {"job_id": "j-1", "status": "running", "instance_id": 42}})
    out = view._fold_fleet_jobs({"41", "42"})
    assert [len(v) for v in (out["41"], out["42"])] == [1, 1]


def test_fold_fleet_jobs_dedupes_by_job_id_not_by_view_identity(monkeypatch, tmp_path):
    """Two DISTINCT tickets on one box are two rows — the dedupe must not
    collapse them (that would hide genuine concurrent work)."""
    _fold_env(monkeypatch, tmp_path, [("42", "j-1"), ("42", "j-2")],
              {"j-1": {"job_id": "j-1", "status": "running", "instance_id": 42},
               "j-2": {"job_id": "j-2", "status": "running", "instance_id": 42}})
    out = view._fold_fleet_jobs({"42"})
    assert sorted(v["job_id"] for v in out["42"]) == ["j-1", "j-2"]


def test_fold_fleet_jobs_writes_and_stamps_the_sticky_cache(monkeypatch, tmp_path):
    _fold_env(monkeypatch, tmp_path, [("41", "j-1")], {"j-1": {"status": "done"}})
    view._fold_fleet_jobs({"41"})
    body = json.loads((tmp_path / "vast-jobmeta" / "j-1" / "view.json").read_text())
    assert body[view._JOB_VIEW_CACHE_KEY] == view._JOB_VIEW_CACHE_V


def test_fold_fleet_jobs_does_not_cache_a_failed_view(monkeypatch, tmp_path):
    _fold_env(monkeypatch, tmp_path, [("41", "j-1")], {"j-1": {"status": "failed"}})
    view._fold_fleet_jobs({"41"})
    assert not (tmp_path / "vast-jobmeta" / "j-1" / "view.json").exists()


def test_an_unstamped_cache_body_is_re_read_rather_than_trusted(monkeypatch, tmp_path):
    """`_JOB_VIEW_CACHE_V` is a format stamp: entries written under the OLD rule
    are still on disk, frozen at a `failed` that has since been requeued."""
    d = tmp_path / "vast-jobmeta" / "j-1"
    d.mkdir(parents=True)
    (d / "view.json").write_text(json.dumps({"status": "done", "stale": True}))
    _fold_env(monkeypatch, tmp_path, [("41", "j-1")],
              {"j-1": {"status": "done", "stale": False}})
    out = view._fold_fleet_jobs({"41"})
    assert out["41"][0]["stale"] is False      # unstamped => re-read from B2


@pytest.mark.parametrize("boom", ["list", "ensure"])
def test_fold_fleet_jobs_swallows_everything_into_an_empty_dict(monkeypatch, boom):
    """fleetd's `Hooks.health` wraps this in try/except and `gather_fleet_health`
    is ALARMS-only, so a raise degrades to "no jobs fold" — but the RETURN SHAPE
    is what every box's health verdict reads. Assert on output, not on "it ran"."""
    def _die(*a, **k):
        raise RuntimeError("b2 is down")

    if boom == "list":
        monkeypatch.setattr(jobmeta, "list_all_queued", _die)
    else:
        monkeypatch.setattr(b2, "_ensure_b2_remote", _die)
    out = view._fold_fleet_jobs({"41"})
    assert out == {} and isinstance(out, dict)


def test_fold_fleet_jobs_tolerates_the_raw_int_ids_fleetd_passes(monkeypatch, tmp_path):
    """`fleetd.py`'s Hooks builds `live` from raw `i.get('id')` (ints) while this
    fold is string-keyed. The mismatch is PRE-EXISTING and must not be "fixed" in
    the refactor — it only reaches `jobmeta.read_job`'s live injection, and the
    keys of the result stay strings either way."""
    _fold_env(monkeypatch, tmp_path, [(41, "j-1")], {"j-1": {"status": "running"}})
    out = view._fold_fleet_jobs({41, 42})
    assert set(out) == {"41"}


def test_a_job_whose_fold_raises_is_dropped_not_fatal(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(jobmeta, "list_all_queued", lambda: [("41", "bad"), ("41", "ok")])

    def _read(jid, live_iids=None):
        if jid == "bad":
            raise RuntimeError("unreadable log")
        return {"status": "running"}

    monkeypatch.setattr(jobmeta, "read_job", _read)
    out = view._fold_fleet_jobs({"41"})
    assert [v["status"] for v in out["41"]] == ["running"]


def test_the_reap_seam_is_bound_and_steerable(monkeypatch):
    """`boxes` may not import `jobs`, so `view` injects itself into reap's slot.
    The forwarder must resolve `view._fold_fleet_jobs` at CALL time."""
    assert reap._FOLD_FLEET_JOBS is not None
    monkeypatch.setattr(view, "_fold_fleet_jobs", lambda live, prog=None: {"sentinel": []})
    assert reap._fold_fleet_jobs({"41"}) == {"sentinel": []}


# --------------------------------------------------------------------------- #
# 5. The progress renderers
# --------------------------------------------------------------------------- #
_BAR = (" 50%|#####     | 5/10 [00:50<00:50, 10.00s/it]\r"
        " 60%|######    | 6/10 [01:41<00:40, 10.10s/it]")


def test_job_progress_prefers_the_consecutive_step_delta():
    out = view._job_progress({"last_tail": _BAR})
    assert out["pct"] == 60 and out["step"] == 6 and out["total"] == 10
    assert out["rate_kind"] == "delta" and out["rate"] == "51s/it"


def test_job_progress_labels_the_tqdm_fallback_as_an_average():
    """The `~…(avg)` marking is a user-visible contract."""
    one = " 60%|######    | 6/10 [01:41<00:40, 10.10s/it]"
    out = view._job_progress({"last_tail": one})
    assert out["rate"] == "~10.1s/it(avg)" and out["rate_kind"] == "avg"


def test_job_progress_reads_the_LAST_num_tokens_match():
    """The `for tk in ...: pass` idiom leaks the loop variable deliberately —
    the newest cumulative count is the one that divides by elapsed."""
    tail = ("{'loss': 1.0, 'num_tokens': '1.0e+03'}\n"
            " 60%|######    | 6/10 [00:10<00:40, 1.00s/it]\n"
            "{'loss': 0.5, 'num_tokens': '5.0e+03'}\n")
    out = view._job_progress({"last_tail": tail})
    assert out["toks"] == pytest.approx(500.0)          # 5000 / 10s, not 1000/10


def test_job_progress_is_empty_when_nothing_parses():
    assert view._job_progress({}) == {}
    assert view._job_progress({"last_tail": "no bars here"}) == {}


def test_job_progress_borrows_risks_bar_parser_rather_than_forking_it(monkeypatch):
    """One tqdm regex in the tree. If `view` grew its own, this patch would not
    steer it."""
    monkeypatch.setattr(risk, "_tqdm_points", lambda tail: [])
    assert view._job_progress({"last_tail": _BAR}) == {}


def test_job_cell_emits_only_parsed_scalars():
    """`last_tail` must never reach the dashboard cache."""
    cell = view._job_cell({"name": "train", "display_status": "running",
                           "last_tail": _BAR, "n_checkpoints": 3})
    assert cell == "train:running:60%:51s/it:ckpt3"
    assert "|" not in cell and "#" not in cell


def test_job_cell_falls_back_to_the_job_id_when_unnamed():
    assert view._job_cell({"job_id": "j-1", "display_status": "queued"}) == "j-1:queued"


def test_step_rate_is_none_on_no_points():
    assert view._step_rate([]) is None
    assert view._step_rate(None) is None


# --------------------------------------------------------------------------- #
# 6. cmd_job_wait — FOUR frozen exit codes, first coverage
# --------------------------------------------------------------------------- #
def _wait_args(**kw):
    base = dict(job_id="j-1", until="terminal", timeout=30, interval=0,
                json=False)
    base.update(kw)
    return _ns(**base)


def _pin_view(monkeypatch, *views):
    """Feed `_job_view` a script of folds, repeating the last one forever."""
    seq = list(views)

    def _v(jid):
        return seq.pop(0) if len(seq) > 1 else seq[0]

    monkeypatch.setattr(view, "_job_view", _v)


def test_job_wait_exits_0_when_the_state_is_reached(monkeypatch, capsys):
    _pin_view(monkeypatch, {"status": "done", "display_status": "done", "rc": 0})
    view.cmd_job_wait(_wait_args())            # returns, does not raise
    assert "reached terminal" in capsys.readouterr().out


def test_job_wait_exits_2_on_terminal_but_FAILED(monkeypatch):
    """`--until terminal` on a job that FAILED: a shell `&&` chain must stop."""
    _pin_view(monkeypatch, {"status": "failed", "display_status": "failed", "rc": 7})
    with pytest.raises(SystemExit) as ei:
        view.cmd_job_wait(_wait_args())
    assert ei.value.code == 2


def test_job_wait_does_not_exit_2_for_a_non_terminal_until(monkeypatch):
    """The 2 is scoped to `--until terminal`; `--until failed` REACHED its ask."""
    _pin_view(monkeypatch, {"status": "failed", "display_status": "failed", "rc": 7})
    view.cmd_job_wait(_wait_args(until="failed"))


def test_job_wait_exits_124_on_timeout(monkeypatch, capsys):
    """124 is the coreutils convention and the reason a caller can tell "not yet"
    apart from "never"."""
    _pin_view(monkeypatch, {"status": "running", "display_status": "running", "rc": None})
    with pytest.raises(SystemExit) as ei:
        view.cmd_job_wait(_wait_args(timeout=-1))
    assert ei.value.code == 124
    assert "did not reach" in capsys.readouterr().err


def test_job_wait_exits_with_a_message_when_the_state_is_unreachable(monkeypatch):
    """A string exit code is 1 at the shell — "will never get there"."""
    _pin_view(monkeypatch, {"status": "failed", "display_status": "failed",
                            "rc": 7, "fail_reason": "oom"})
    with pytest.raises(SystemExit) as ei:
        view.cmd_job_wait(_wait_args(until="done"))
    assert "will never reach" in str(ei.value.code)


def test_job_wait_rejects_an_unknown_until_state(monkeypatch):
    with pytest.raises(SystemExit) as ei:
        view.cmd_job_wait(_wait_args(until="finished-ish"))
    assert "not one of" in str(ei.value.code)


def test_job_wait_never_consults_the_done_marker_and_so_cannot_false_terminal(
        monkeypatch):
    """`wait` reads the FOLD only — `_job_view`/`read_job` never probe
    results.DONE.json — so the stale-marker false terminal cannot reach it. This
    pins that as a property rather than an accident: a re-opened job carrying a
    dead attempt's marker keeps WAITING, and times out rather than claiming the
    job reached terminal."""
    _pin_view(monkeypatch, {"status": "claimed", "display_status": "running",
                            "rc": None, "reopened": True,
                            "reopened_at": "20260828T104316128Z",
                            "done_marker": True})
    with pytest.raises(SystemExit) as ei:
        view.cmd_job_wait(_wait_args(timeout=-1))
    assert ei.value.code == 124


def test_job_wait_polls_until_the_state_arrives(monkeypatch, capsys):
    _pin_view(monkeypatch,
              {"status": "running", "display_status": "running", "rc": None},
              {"status": "running", "display_status": "running", "rc": None},
              {"status": "done", "display_status": "done", "rc": 0})
    view.cmd_job_wait(_wait_args())
    assert "reached terminal" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 7. status / fold error contract
# --------------------------------------------------------------------------- #
def test_job_view_exits_on_a_jobmeta_error(monkeypatch):
    monkeypatch.setattr(view, "_live_iids_set", lambda: set())

    def _raise(jid, live_iids=None):
        raise jobmeta.JobmetaError("no such job")

    monkeypatch.setattr(jobmeta, "read_job", _raise)
    with pytest.raises(SystemExit) as ei:
        view._job_view("j-1")
    assert "error: no such job" in str(ei.value.code)


def test_job_view_fresh_uses_the_uncached_reader(monkeypatch):
    monkeypatch.setattr(view, "_live_iids_set", lambda: {"41"})
    seen = {}

    def _fresh(jid, live_iids=None):
        seen["live"] = live_iids
        return {"ok": 1}

    monkeypatch.setattr(jobmeta, "read_job_fresh", _fresh)
    monkeypatch.setattr(jobmeta, "read_job",
                        lambda *a, **k: pytest.fail("--fresh must not use the cached read"))
    assert view._job_view_fresh("j-1") == {"ok": 1}
    assert seen["live"] == {"41"}


_FULL = {"job_id": "j-1", "display_status": "running", "status": "started",
         "live": True, "instance_id": "41", "target_box": "41", "name": "train",
         "entrypoint": "run.sh", "bundle_sha256": "a" * 64, "n_events": 5,
         "parse_errors": 0, "rc": None, "fail_reason": None,
         "last_heartbeat_ts": "20260806T170440975Z", "results": [],
         "last_event": "heartbeat", "last_event_ts": "20260806T170440975Z"}


def test_print_job_view_renders_the_required_keys():
    buf = io.StringIO()
    with redirect_stdout(buf):
        view._print_job_view(dict(_FULL))
    out = buf.getvalue()
    assert "== job j-1 ==" in out
    assert "status=running (fold=started)" in out
    assert "bundle=aaaaaaaaaaaa " in out            # sha12, not the full digest


def test_print_job_view_keyerrors_on_a_missing_required_key():
    """Indexing with `[]` is by design: a fold missing `rc` is a fold bug, and a
    silent `-` would hide it."""
    v = dict(_FULL)
    del v["rc"]
    with pytest.raises(KeyError):
        view._print_job_view(v)


def test_print_fresh_notes_says_the_b2_list_lagged(capsys):
    view._print_fresh_notes({"status": "started", "done_marker": True})
    out = capsys.readouterr().out
    assert "the job FINISHED" in out and "--fresh" in out


def test_print_fresh_notes_disclaims_liveness_on_an_unclaimed_job(capsys):
    view._print_fresh_notes({"status": "submitted", "unclaimed": True})
    assert "live=n/a" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 7b. The DONE marker beside a non-terminal fold has TWO causes
#
# B2 LIST lag (the job finished) and a re-opened job carrying the DEAD attempt's
# marker (the job is running). The note said "the job FINISHED" for both, which
# on 2026-08-28 reported a v16-r64 training arm complete at 42% of 273 steps.
# --------------------------------------------------------------------------- #
def test_a_prior_attempts_done_marker_is_never_rendered_as_finished(capsys):
    view._print_fresh_notes({
        "job_id": "j-1", "status": "claimed", "done_marker": True,
        "done_marker_verdict": jobmeta.DONE_MARKER_STALE,
        "done_marker_ts": "20260828T100012000Z", "done_marker_rc": 3,
        "done_marker_box": "41", "reopened_at": "20260828T104316128Z"})
    out = capsys.readouterr().out
    assert "the job FINISHED" not in out
    assert "PRIOR attempt" in out and "20260828T104316128Z" in out
    assert "rc=3" in out and "box 41" in out
    assert "checkpoints/" in out                    # where the live work IS


def test_an_undatable_marker_on_a_reopened_job_refuses_to_call_it_either_way(capsys):
    view._print_fresh_notes({
        "job_id": "j-1", "status": "started", "done_marker": True,
        "done_marker_verdict": jobmeta.DONE_MARKER_UNKNOWN,
        "done_marker_ts": None, "done_marker_rc": None,
        "reopened_at": "20260828T104316128Z"})
    out = capsys.readouterr().out
    assert "the job FINISHED" not in out
    assert "could not be DATED" in out and "lsjson" in out


def test_a_view_with_no_verdict_field_renders_the_pre_2026_08_28_wording(capsys):
    """A fold from an older reader (or `scan.fold_many`, which fills the bool
    from a listing) has no verdict. That is the never-re-opened case and must
    read exactly as it always did."""
    view._print_fresh_notes({"status": "started", "done_marker": True})
    assert "the job FINISHED" in capsys.readouterr().out


def test_cmd_job_status_json_is_indented_one_shot_and_compact_under_watch(
        monkeypatch, capsys):
    """The indent difference is a real output-shape contract: `--watch` emits one
    JSON object per LINE so a reader can stream it."""
    _pin_view(monkeypatch, {"status": "done", "display_status": "done", "rc": 0})
    view.cmd_job_status(_ns(job_id="j-1", watch=False, json=True, fresh=False,
                            interval=0))
    assert "\n" in capsys.readouterr().out.strip()

    view.cmd_job_status(_ns(job_id="j-1", watch=True, json=True, fresh=False,
                            interval=0))
    assert capsys.readouterr().out.strip().count("\n") == 0


# --------------------------------------------------------------------------- #
# 8. logs — presence is not provenance
# --------------------------------------------------------------------------- #
def test_hb_age_s_parses_the_colon_free_ms_stamp():
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    ts = now.strftime("%Y%m%dT%H%M%S") + "123Z"
    assert view._hb_age_s(ts) == pytest.approx(0, abs=5)


@pytest.mark.parametrize("ts", [None, "", "not-a-stamp", 12345])
def test_hb_age_s_is_none_on_anything_unparseable(ts):
    assert view._hb_age_s(ts) is None


def test_job_log_provenance_warns_when_the_emitting_box_is_not_the_target(
        monkeypatch):
    monkeypatch.setattr(view, "_hb_age_s", lambda ts: 12.0)
    lines = view._job_log_provenance(
        {"instance_id": "41", "target_box": "42", "display_status": "running",
         "last_heartbeat_ts": "20260806T170440975Z"}, "j-1")
    assert any("!! PROVENANCE" in ln for ln in lines)
    assert any("emitted by box 41" in ln for ln in lines)
    assert any("now targeted at box 42" in ln for ln in lines)


def test_job_log_provenance_flags_a_stale_heartbeat(monkeypatch):
    monkeypatch.setattr(view, "_hb_age_s", lambda ts: 601.0)
    lines = view._job_log_provenance(
        {"instance_id": "41", "target_box": "41", "display_status": "running",
         "last_heartbeat_ts": "20260806T170440975Z"}, "j-1")
    assert any(ln.startswith("!! STALE") for ln in lines)
    assert not any("PROVENANCE" in ln for ln in lines)   # same box: no warning


def test_cmd_job_logs_reads_B2_BUCKET_from_the_raw_environment(monkeypatch, capsys):
    """Ported verbatim: this is the ONE bucket read in the tree that does not go
    through the config/b2 layer. Pinned so a later "cleanup" is a red test rather
    than a silent precedence change."""
    monkeypatch.setenv("B2_BUCKET", "bkt-from-env")
    _pin_view(monkeypatch, {"status": "done", "display_status": "done"})
    calls = []

    def _rclone(args):
        calls.append(list(args))
        return (0, "  12345 2026-08-06 17:04:40 log.txt\n" if args[0] == "lsl"
                else "the log bytes\n")

    monkeypatch.setattr(b2, "_rclone", _rclone)
    view.cmd_job_logs(_ns(job_id="j-1"))
    assert calls[0] == ["lsl", "b2:bkt-from-env/jobs/j-1/log.txt"]
    assert calls[1] == ["cat", "b2:bkt-from-env/jobs/j-1/log.txt"]
    out = capsys.readouterr().out
    assert "whichever attempt FINALIZED" in out and "the log bytes" in out


def test_cmd_job_logs_on_a_running_job_prints_provenance_then_the_tail(
        monkeypatch, capsys):
    _pin_view(monkeypatch, {"status": "started", "display_status": "running",
                            "instance_id": "41", "target_box": "41",
                            "last_heartbeat_ts": None, "last_tail": "TAIL-BYTES"})
    monkeypatch.setattr(b2, "_rclone",
                        lambda a: pytest.fail("a running job never reads log.txt"))
    view.cmd_job_logs(_ns(job_id="j-1"))
    out = capsys.readouterr().out
    assert out.index("== job j-1") < out.index("TAIL-BYTES")


# --------------------------------------------------------------------------- #
# 9. pull — an empty pull is exit 0, and must say why
# --------------------------------------------------------------------------- #
def test_cmd_job_pull_defaults_dest_under_the_repo_out_tree(monkeypatch, capsys):
    seen = {}

    def _pull(jid, dest):
        seen["dest"] = dest
        return ["a.json"]

    monkeypatch.setattr(jobmeta, "pull_results", _pull)
    view.cmd_job_pull(_ns(job_id="j-1", dest=None))
    assert seen["dest"] == os.path.join(view._REPO_ROOT, "out", "jobs", "j-1")
    assert "pulled 1 result file(s)" in capsys.readouterr().out


def test_an_empty_pull_stays_exit_0_and_explains_itself(monkeypatch, capsys):
    monkeypatch.setattr(jobmeta, "pull_results", lambda jid, dest: [])
    monkeypatch.setattr(view, "_live_iids_set", lambda: set())
    monkeypatch.setattr(jobmeta, "read_job", lambda jid, live_iids=None: {"status": "started"})
    monkeypatch.setattr(jobmeta, "list_checkpoints", lambda jid: ["ck-1", "ck-2"])
    view.cmd_job_pull(_ns(job_id="j-1", dest="/tmp/x"))    # no SystemExit
    out = capsys.readouterr().out
    assert "NOT lost work" in out and "2 file(s) ARE durable" in out


def _pin_pull(monkeypatch, *, reopened_at, probe, display_status="running"):
    """Wire the guard's two reads and make the pull itself observable."""
    pulled = []
    monkeypatch.setattr(view, "_live_iids_set", lambda: {"42"})
    monkeypatch.setattr(jobmeta, "read_job", lambda jid, live_iids=None: {
        "reopened_at": reopened_at, "display_status": display_status})
    monkeypatch.setattr(jobmeta, "probe_done_marker",
                        lambda jid, reopened_at=None: dict(probe))
    monkeypatch.setattr(jobmeta, "pull_results",
                        lambda jid, dest: (pulled.append(dest), ["a.json"])[1])
    return pulled


_STALE_PROBE = {"present": True, "ts": "20260828T100012000Z", "rc": 3,
                "box": "41", "verdict": jobmeta.DONE_MARKER_STALE}


def test_pull_refuses_a_results_tree_that_belongs_to_a_dead_attempt(
        monkeypatch, capsys):
    """A warning on `status` buys nothing if `pull` still ships the debris at
    exit 0 — 57.7 KiB of the failed attempt's scheduler/rng files, read as a
    finished training arm (2026-08-28)."""
    pulled = _pin_pull(monkeypatch, reopened_at="20260828T104316128Z",
                       probe=_STALE_PROBE)
    with pytest.raises(SystemExit) as e:
        view.cmd_job_pull(_ns(job_id="j-1", dest="/tmp/x", allow_stale=False))
    assert not pulled                       # nothing was written
    msg = str(e.value)
    assert "REFUSING" in msg and "DEAD" in msg
    assert "20260828T104316128Z" in msg and "--allow-stale" in msg
    assert "checkpoints/" in msg


def test_allow_stale_pulls_the_dead_attempt_but_says_so(monkeypatch, capsys):
    pulled = _pin_pull(monkeypatch, reopened_at="20260828T104316128Z",
                       probe=_STALE_PROBE)
    view.cmd_job_pull(_ns(job_id="j-1", dest="/tmp/x", allow_stale=True))
    assert pulled == ["/tmp/x"]
    out = capsys.readouterr().out
    assert "--allow-stale" in out and "PRIOR attempt" in out


def test_an_undatable_marker_warns_and_still_pulls(monkeypatch, capsys):
    """Refusing every pull on a B2 listing hiccup is a worse failure than the
    one being prevented, so only a POSITIVE stale verdict refuses."""
    pulled = _pin_pull(monkeypatch, reopened_at="20260828T104316128Z",
                       probe={"present": True, "ts": None, "rc": 3, "box": None,
                              "verdict": jobmeta.DONE_MARKER_UNKNOWN})
    view.cmd_job_pull(_ns(job_id="j-1", dest="/tmp/x", allow_stale=False))
    assert pulled == ["/tmp/x"]
    assert "could not be dated" in capsys.readouterr().out


def test_a_job_that_was_never_reopened_pulls_without_probing_the_marker(
        monkeypatch, capsys):
    monkeypatch.setattr(view, "_live_iids_set", lambda: {"42"})
    monkeypatch.setattr(jobmeta, "read_job",
                        lambda jid, live_iids=None: {"reopened_at": None})
    monkeypatch.setattr(jobmeta, "probe_done_marker",
                        lambda *a, **k: pytest.fail("no marker probe is owed here"))
    monkeypatch.setattr(jobmeta, "pull_results", lambda jid, dest: ["a.json"])
    view.cmd_job_pull(_ns(job_id="j-1", dest="/tmp/x", allow_stale=False))
    assert "pulled 1 result file(s)" in capsys.readouterr().out


def test_the_pull_guard_fails_open_when_its_own_probe_raises(monkeypatch, capsys):
    """Same contract as the empty-pull diagnostic: a guard that raises would turn
    a healthy pull into a failure."""
    monkeypatch.setattr(view, "_live_iids_set",
                        lambda: (_ for _ in ()).throw(RuntimeError("api down")))
    monkeypatch.setattr(jobmeta, "pull_results", lambda jid, dest: ["a.json"])
    view.cmd_job_pull(_ns(job_id="j-1", dest="/tmp/x", allow_stale=False))
    assert "pulled 1 result file(s)" in capsys.readouterr().out


def test_the_empty_pull_diagnostic_never_changes_the_outcome(monkeypatch, capsys):
    """Fully guarded by design: a diagnostic that raises would turn a healthy
    pull into a failure."""
    monkeypatch.setattr(view, "_live_iids_set",
                        lambda: (_ for _ in ()).throw(RuntimeError("api down")))
    monkeypatch.setattr(jobmeta, "list_checkpoints",
                        lambda jid: (_ for _ in ()).throw(RuntimeError("b2 down")))
    view._job_pull_explain_empty("j-1")
    assert "(job status=unknown)" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 10. bundle definitions
# --------------------------------------------------------------------------- #
def test_the_three_bundle_homes_are_not_consolidated():
    """Consolidating them was considered and REJECTED (find_job_defs' docstring);
    the tuple is the registry."""
    assert view.JOB_DEF_HOMES == ("tools/witness/jobs", "tools/vast/jobs",
                                  "tools/pipeline/jobs")


def test_find_job_defs_reports_an_unparseable_bundle_rather_than_dropping_it(
        monkeypatch, tmp_path):
    home = tmp_path / "tools" / "vast" / "jobs"
    (home / "good").mkdir(parents=True)
    (home / "bad").mkdir()
    (home / "good" / "job-config.yaml").write_text("name: good\n")
    (home / "bad" / "job-config.yaml").write_text(": not yaml\n")
    (home / "notabundle").mkdir()
    (home / "_scratch").mkdir()

    def _load(bundle):
        if bundle.endswith("bad"):
            raise ValueError("unparseable")
        return {"name": os.path.basename(bundle)}

    monkeypatch.setattr(jobmeta, "load_job_config", _load)
    defs = view.find_job_defs(str(tmp_path))
    assert [(p, c, e is not None) for p, c, e in defs] == [
        ("tools/vast/jobs/bad", None, True),
        ("tools/vast/jobs/good", {"name": "good"}, False)]
    assert view.find_job_def_strays(str(tmp_path)) == ["tools/vast/jobs/notabundle"]


@pytest.mark.parametrize("cfg,want", [
    ({}, "CPU"),
    ({"needs": {"gpu": True, "gpus": "all"}, "env": {"MODE": "autotune"}}, "whole-box DDP"),
    ({"needs": {"gpu": True, "gpus": "all"}}, "1-GPU pinned"),
    ({"needs": {"gpu": True}, "env": {"MODE": "autotune"}}, "1-GPU autotune"),
    ({"needs": {"gpu": True}}, "1-GPU pinned"),
    (None, "CPU"),
])
def test_job_shape_defaults_mode_to_pinned_fail_closed(cfg, want):
    assert view._job_shape(cfg) == want


def test_cmd_job_defs_json_schema_is_the_contract(monkeypatch, capsys):
    monkeypatch.setattr(view, "find_job_defs",
                        lambda: [("tools/vast/jobs/x",
                                  {"name": "x", "entrypoint": "run.sh",
                                   "needs": {"gpu": True}, "assets": [1, 2]}, None)])
    monkeypatch.setattr(view, "find_job_def_strays", lambda: ["tools/vast/jobs/y"])
    monkeypatch.setattr(jobmeta, "collect_tracked", lambda c: ["t"])
    view.cmd_job_defs(_ns(json=True))
    body = json.loads(capsys.readouterr().out)
    assert body["not_bundles"] == ["tools/vast/jobs/y"]
    assert body["bundles"] == [{"path": "tools/vast/jobs/x", "name": "x",
                                "entrypoint": "run.sh", "gpu": True,
                                "shape": "1-GPU pinned", "assets": 2,
                                "tracks": 1, "error": None}]


# --------------------------------------------------------------------------- #
# 11. job ls — the tri-state reaches the operator
# --------------------------------------------------------------------------- #
def _ls_env(monkeypatch, present, statuses=("submitted",)):
    monkeypatch.setattr(jobmeta, "list_all_queued",
                        lambda: [("41", f"j-{i}") for i in range(len(statuses))])
    monkeypatch.setattr(jobmeta, "read_box", lambda b: {"parked": False,
                                                        "drained_pending": False})
    monkeypatch.setattr(view, "_live_iids_set", lambda: set())
    monkeypatch.setattr(view, "_present_iids_set", lambda: present)
    st = {f"j-{i}": s for i, s in enumerate(statuses)}
    # `cmd_job_ls` folds the whole queue through `scan.fold_many` (ONE bulk
    # listing), not `jobmeta.read_job` per ticket — see `jobs/scan.py`. Patching
    # the old seam here would leave this test asserting against nothing.
    monkeypatch.setattr(scan, "fold_many", lambda jids, live_iids=(): {
        jid: {"status": st[jid], "display_status": st[jid], "n_events": 3,
              "done_marker": False} for jid in jids})


def test_job_ls_skips_orphan_detection_when_presence_is_unreadable(
        monkeypatch, capsys):
    _ls_env(monkeypatch, None)
    view.cmd_job_ls(_ns(box=None))
    out = capsys.readouterr().out
    assert "instance listing unreadable — orphan detection skipped" in out
    assert "ORPHAN" not in out.replace("orphan detection skipped", "")


def test_job_ls_headlines_a_gone_box_and_counts_its_orphans(monkeypatch, capsys):
    _ls_env(monkeypatch, {"99"})               # 41 is NOT in the account
    view.cmd_job_ls(_ns(box=None))
    out = capsys.readouterr().out
    assert "GONE — instance destroyed" in out
    assert "1 ORPHANED ticket(s)" in out


def test_job_ls_says_live_when_the_box_is_present(monkeypatch, capsys):
    _ls_env(monkeypatch, {"41"})
    view.cmd_job_ls(_ns(box=None))
    out = capsys.readouterr().out
    assert "live=False" in out and "GONE" not in out and "ORPHANED" not in out


# --------------------------------------------------------------------------- #
# 12. _box_lifecycle_soft — the seam handoff/job_lane read
# --------------------------------------------------------------------------- #
def test_box_lifecycle_soft_never_raises(monkeypatch):
    monkeypatch.setattr(jobmeta, "read_box",
                        lambda b: (_ for _ in ()).throw(RuntimeError("b2 down")))
    assert view._box_lifecycle_soft(41) == {"parked": False,
                                            "drained_pending": False,
                                            "park_reason": None}


def test_box_lifecycle_soft_stringifies_the_iid(monkeypatch):
    seen = []
    monkeypatch.setattr(jobmeta, "read_box",
                        lambda b: seen.append(b) or {"parked": True})
    assert view._box_lifecycle_soft(41) == {"parked": True}
    assert seen == ["41"]


# --------------------------------------------------------------------------- #
# 13. TWIN IDENTITY — one copy since plan §8 step 6d
# --------------------------------------------------------------------------- #
# The header here read: "`herdd.py` keeps its originals until plan step 6, so
# during this window two implementations of each of these is REACHABLE: `ls
# --minimal` and the dash-cache projection go through one, `vastlib` consumers
# through the other." Step 6d closed that window. Five parity tests are deleted
# with it — `_job_cell` and `_job_progress` over the `_TWIN_VIEWS` corpus, the
# three view-cache constants, `JOB_DEF_HOMES`, and the provenance warning text.
# The last of those had also gone VACUOUS: it steered the clock with
# `monkeypatch.setattr(herdd, "_hb_age_s", …)`, and a re-export is not a
# patch point (launcher docstring, rule 2), so post-thinning that patch bound a
# name nothing reads.
#
# `_job_cell`'s real protection is unchanged and lives elsewhere: it is SHARED
# rather than duplicated, so the dashboard string and the CLI string cannot
# disagree, and `test_job_logs_provenance.py` asserts the warning substrings
# against this module.
import herdd                                        # noqa: E402


def test_the_launcher_re_exports_rather_than_redefines():
    """A second body here would put `ls --minimal` and the dash-cache
    projection back on two renderers — the exact drift `_job_cell` was made
    shared to prevent."""
    for name in ("JOB_DEF_HOMES", "_JOB_VIEW_CACHE_KEY", "_JOB_VIEW_CACHE_V",
                 "_JOB_VIEW_STICKY", "_hb_age_s", "_job_cell",
                 "_job_log_provenance", "_job_progress"):
        assert getattr(herdd, name) is getattr(view, name), (
            f"herdd.{name} is a second body again — the launcher must "
            f"re-export vastlib.jobs.view's object, never redefine it")
