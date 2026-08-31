"""Unit pins for `vastlib.jobs.risk` — the work-at-risk metrics, characterized.

Why this file exists
--------------------
`risk.py` is a verbatim-with-types move of cluster C23 out of `herdd.py`
(plan §8 step 3, ADD-ONLY at the time: `herdd.py` kept its live copies until
step 6d thinned it — see `_flat_twin` at the bottom for what that did to the
parity half of this file).
The port manifest's coverage sweep found that **10 of its 15 symbols had ZERO
direct tests** — `_attempt_start_epoch`, `_step_delta_s`, `_job_eta_s`,
`_job_pct`, `_jobs_work_at_risk_h`, `_jobs_unresumable_running`,
`_jobs_ckpt_stale`, `_jobs_min_running_eta_s`, `_jobs_work_horizon_h` and
`CKPT_STALL_MULT` were covered only transitively, through `job_supervise_tick`
and `_job_rebid_ladder`. Plan §5 names this module the package's first to reach
100% typed AND tested; the typing was a port, and this file is the testing.

Three jobs, in order of what they protect:

1. **The tri-state direction of every reader.** `None` means UNKNOWN and must
   never arrive as `0.0` (about to finish) or as a large number (plenty of
   time) — defect #67, the 2026-08-08 22:17Z incident that read a hang detector
   as a work estimate and inflated a projected saving ~5x. Two readers go the
   other way on purpose (`_jobs_work_at_risk_h` is 0.0-never-None,
   `_jobs_unresumable_running` is 0-never-None) and that direction is pinned
   too, because a "consistency" cleanup in either direction is a money bug.
2. **The asymmetries a typed port is tempted to smooth over.** The
   `isinstance(x, bool)` rejection present in `_jobs_ckpt_stale` /
   `_jobs_remaining_wall_h` and ABSENT from `_ckpt_watchdog_alarm`; the
   def-time `mult=CKPT_STALL_MULT` bind versus the call-time `mult=None`
   sentinel; `_job_pct` not gating on `display_status`; `_job_eta_s` taking a
   `now` it deliberately ignores; `_jobs_unresumable_running` taking no `now` at
   all.
3. **A drift tripwire.** For the whole porting window there are two copies of
   every function. The parity block at the bottom feeds one table of views to
   both and asserts identical output, so a rebase that lands a peer's edit in
   `herdd.py` fails here instead of silently forking what the fleet believes
   is at risk.

The `_tqdm_tail` fixture is lifted from `test_supervise.py` (the only place in
the suite that synthesizes two consecutive training bars) so `_job_eta_s` and
`_step_delta_s` can be exercised on the real bar shape rather than on a
hand-built points tuple.

Toolchain-free lane (`pytest -m "not integration"`): pure functions, no
network, no vast API, no B2, no clock — every function under test takes `now`
as an argument.
"""

from __future__ import annotations

import datetime as _dt
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bidpolicy  # noqa: E402
import herdd as v  # noqa: E402
from vastlib.jobs import risk  # noqa: E402

NOW = 1_000_000.0


# --- helpers ---------------------------------------------------------------- #
def _ts(epoch):
    """A jobmeta-format ts ('YYYYMMDDTHHMMSSmmmZ') for a given UTC epoch second."""
    return _dt.datetime.fromtimestamp(epoch, tz=_dt.timezone.utc) \
        .strftime("%Y%m%dT%H%M%S") + "000Z"


def _hms(s):
    return f"{int(s) // 3600}:{(int(s) % 3600) // 60:02d}:{int(s) % 60:02d}"


def _tqdm_tail(step, elapsed_s, *, total=100, step_delta=60.0, rate=1.0):
    """Two consecutive training bars, `step_delta` seconds apart — the ONLY
    honest step time available without SSH (`_step_delta_s`). Remaining steps x
    that delta is the ETA the handoff prices against.

    Lifted verbatim in shape from `test_supervise.py`'s fixture of the same
    name, so both suites exercise the same bar syntax."""
    def bar(st, el):
        return (f"\r {int(100 * st / total):>3}%|##        | {st}/{total} "
                f"[{_hms(el)}<1:00:00, {rate:.2f}s/it]")
    return bar(step - 1, elapsed_s - step_delta) + "\n" + bar(step, elapsed_s)


def _running(**over):
    """A RUNNING job view with every field the cluster reads present and sane."""
    view = {"job_id": "job-x", "display_status": "running",
            "checkpoint_s": 300, "n_checkpoints": 3,
            "started_at": _ts(NOW - 10_000), "last_resumed_ts": None,
            "last_checkpoint_ts": _ts(NOW - 200), "last_event": "checkpoint",
            "timeout_s": 36_000, "last_tail": ""}
    view.update(over)
    return view


# =============================================================================
# 1. Constants and the tqdm bar parser.
# =============================================================================
def test_constants_match_the_herdd_originals():
    """The `== v.<name>` arms went at step 6d (see `_flat_twin` below): those
    three names are the launcher's re-exports of these objects. The literals
    are what they were standing in for."""
    assert risk.CKPT_STALL_MULT == 3
    assert risk._STEP_DELTA_FLOOR_S == 2.0
    assert risk._TQDM_RE.search(" 50%|##| 5/10 [00:50<00:50, 10.00s/it]")


def test_ckpt_stall_mult_is_bound_at_def_time_not_read_at_call_time(monkeypatch):
    # The manifest's warning made executable: `mult=CKPT_STALL_MULT` is
    # evaluated when the def runs, so rebinding the module attribute later does
    # NOT move the alarm threshold. (`_jobs_ckpt_stale`'s `mult=None` sentinel
    # is the opposite; see its own test.)
    monkeypatch.setattr(risk, "CKPT_STALL_MULT", 10_000)
    view = _running(last_checkpoint_ts=_ts(NOW - 1200), last_resumed_ts=None,
                    started_at=_ts(NOW - 1200))
    assert risk._ckpt_watchdog_alarm(view, NOW) is not None    # still 3x, not 10000x


def test_tqdm_points_parses_two_consecutive_bars():
    pts = risk._tqdm_points(_tqdm_tail(50, 3000.0))
    assert len(pts) == 2
    assert [p[1] for p in pts] == [49, 50]
    assert [p[2] for p in pts] == [100, 100]
    assert pts[-1] == (50, 50, 100, 3000, 1.0, "s/it")


def test_tqdm_points_drops_described_bars():
    tail = ("Loading weights: 100%|##| 339/339 [00:00<00:00, 9415.48it/s]\n"
            + _tqdm_tail(50, 3000.0))
    pts = risk._tqdm_points(tail)
    assert [p[1] for p in pts] == [49, 50]          # the described bar is gone


def test_tqdm_points_on_none_and_empty_and_unparseable_elapsed():
    assert risk._tqdm_points(None) == []
    assert risk._tqdm_points("") == []
    assert risk._tqdm_points("no bar here") == []
    # `[::<` matches the elapsed group but `int('')` raises -> the point is
    # skipped rather than the whole tail failing.
    assert risk._tqdm_points("\r 50%|##| 5/10 [::<1:00:00, 1.00s/it]") == []


# `test_tqdm_points_parity_with_herdd` swept five tails through both copies.
# One copy since step 6d (`_flat_twin`); the bar-shape coverage those five
# tails carried is asserted by value in the tests above and below.


# =============================================================================
# 2. _step_delta_s — the consecutive-step delta, and its 2.0s floor.
# =============================================================================
def test_step_delta_s_needs_two_points():
    assert risk._step_delta_s(None) is None
    assert risk._step_delta_s([]) is None
    assert risk._step_delta_s(risk._tqdm_points(_tqdm_tail(50, 3000.0))[:1]) is None


def test_step_delta_s_is_the_consecutive_step_delta():
    pts = risk._tqdm_points(_tqdm_tail(50, 3000.0, step_delta=60.0))
    assert risk._step_delta_s(pts) == pytest.approx(60.0)


def test_step_delta_s_refuses_below_the_resolution_floor():
    # 1 s/step is under _STEP_DELTA_FLOOR_S (2.0): tqdm's 1-second elapsed
    # stamp cannot resolve it, so the honest answer is None, not 1.0.
    pts = [(49, 49, 100, 2999, 1.0, "s/it"), (50, 50, 100, 3000, 1.0, "s/it")]
    assert risk._step_delta_s(pts) is None


def test_step_delta_s_ignores_points_from_a_different_total():
    # A bar with a different `total` is a different phase (eval vs train); it
    # must not be paired with the training bar.
    pts = [(50, 5, 10, 2000, 1.0, "s/it"), (50, 50, 100, 3000, 1.0, "s/it")]
    assert risk._step_delta_s(pts) is None


def test_step_delta_s_ignores_non_advancing_or_time_travelling_points():
    same_step = [(50, 50, 100, 2000, 1.0, "s/it"), (50, 50, 100, 3000, 1.0, "s/it")]
    assert risk._step_delta_s(same_step) is None
    back_in_time = [(49, 49, 100, 3500, 1.0, "s/it"), (50, 50, 100, 3000, 1.0, "s/it")]
    assert risk._step_delta_s(back_in_time) is None


def test_step_delta_s_takes_the_nearest_usable_earlier_point():
    pts = [(10, 10, 100, 1000, 1.0, "s/it"),      # far older
           (40, 40, 100, 2400, 1.0, "s/it"),      # the one it should pair with
           (50, 50, 100, 3000, 1.0, "s/it")]
    assert risk._step_delta_s(pts) == pytest.approx(60.0)


# `test_step_delta_s_parity_with_herdd` swept four point lists through both
# copies. One copy since step 6d; the 2.0s floor and the empty/one-point
# degradations are asserted by value in this section.


# =============================================================================
# 3. _ckpt_watchdog_alarm — advisory string, two triggers, no bool guard.
# =============================================================================
def test_watchdog_is_silent_when_checkpoints_are_fresh():
    assert risk._ckpt_watchdog_alarm(_running(), NOW) is None


def test_watchdog_is_silent_for_non_dicts_and_non_running_views():
    assert risk._ckpt_watchdog_alarm(None, NOW) is None
    assert risk._ckpt_watchdog_alarm("junk", NOW) is None
    assert risk._ckpt_watchdog_alarm(
        _running(display_status="interrupted",
                 last_checkpoint_ts=_ts(NOW - 9000)), NOW) is None


def test_watchdog_silence_alarm_names_the_job_and_the_shortfall():
    view = _running(last_checkpoint_ts=_ts(NOW - 1200))
    alarm = risk._ckpt_watchdog_alarm(view, NOW)
    assert alarm is not None
    assert "NO checkpoint" in alarm and "job-x" in alarm
    assert "3x checkpoint_s=300s" in alarm and "1200s" in alarm
    # This used to add `assert alarm == v._ckpt_watchdog_alarm(view, NOW)` —
    # byte-for-byte with the live copy, because the substrings above are what
    # `test_supervise.py` matches and an em dash or a space moved across the
    # implicit string concatenation would break those asserts silently. Step 6d
    # left one body, and `test_supervise.py` now matches THIS one, so the
    # substring assertions above are the whole check.


def test_watchdog_takes_the_LATEST_of_checkpoint_resume_start():
    # A fresh resume means the box is alive again, even with an ancient
    # checkpoint stamp.
    view = _running(last_checkpoint_ts=_ts(NOW - 9000),
                    last_resumed_ts=_ts(NOW - 200))
    assert risk._ckpt_watchdog_alarm(view, NOW) is None


def test_watchdog_fires_on_pure_silence_before_the_first_checkpoint():
    view = _running(n_checkpoints=0, last_checkpoint_ts=None,
                    started_at=_ts(NOW - 5000), last_event="started")
    alarm = risk._ckpt_watchdog_alarm(view, NOW)
    assert alarm is not None and "NO checkpoint" in alarm


def test_watchdog_explicit_trigger_fires_without_any_checkpoint_s():
    for over in ({"last_event": "checkpoint_sync_failed"},
                 {"checkpoint_sync_failed": True}):
        view = _running(checkpoint_s=None, **over)
        alarm = risk._ckpt_watchdog_alarm(view, NOW)
        assert alarm is not None and "checkpoint_sync_failed" in alarm


def test_watchdog_job_id_falls_back_to_question_mark():
    view = _running(job_id=None, last_event="checkpoint_sync_failed")
    assert risk._ckpt_watchdog_alarm(view, NOW).startswith("?: ")


def test_watchdog_silence_path_is_off_without_a_positive_checkpoint_s():
    for cps in (None, 0, -5, "300"):
        view = _running(checkpoint_s=cps, last_checkpoint_ts=_ts(NOW - 9000))
        assert risk._ckpt_watchdog_alarm(view, NOW) is None


def test_watchdog_needs_at_least_one_readable_stamp():
    view = _running(last_checkpoint_ts="junk", last_resumed_ts=None, started_at=None)
    assert risk._ckpt_watchdog_alarm(view, NOW) is None


def test_watchdog_mult_is_overridable_by_keyword():
    view = _running(last_checkpoint_ts=_ts(NOW - 1200))     # 4x checkpoint_s
    assert risk._ckpt_watchdog_alarm(view, NOW, mult=10) is None
    assert risk._ckpt_watchdog_alarm(view, NOW, mult=1) is not None


def test_watchdog_accepts_a_bool_checkpoint_s_and_ckpt_stale_does_not():
    # THE ASYMMETRY (manifest hazard, plan §7.4). `isinstance(True, int)` is
    # True, so `checkpoint_s: True` is a 1-second interval HERE and is rejected
    # as absent by `_jobs_ckpt_stale`. Ported as found, not "fixed".
    view = _running(checkpoint_s=True, last_checkpoint_ts=_ts(NOW - 60))
    assert risk._ckpt_watchdog_alarm(view, NOW) is not None
    assert risk._jobs_ckpt_stale([view], NOW) is False


def test_watchdog_stubs_still_work_with_the_two_arg_positional_shape():
    # Three suite sites neuter this with `lambda vw, now: None`; a port that
    # made `mult` positional would break them.
    import inspect
    sig = inspect.signature(risk._ckpt_watchdog_alarm)
    assert sig.parameters["mult"].kind is inspect.Parameter.KEYWORD_ONLY
    assert [p for p, s in sig.parameters.items()
            if s.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD] == ["view", "now"]


# =============================================================================
# 4. _attempt_start_epoch — one of the four copies of the precedence rule.
# =============================================================================
def test_attempt_start_epoch_prefers_the_later_stamp():
    assert risk._attempt_start_epoch(
        {"started_at": _ts(NOW - 10_000), "last_resumed_ts": _ts(NOW - 200)}
    ) == pytest.approx(NOW - 200)
    # ...and "later" is a max, not a field priority: an older resume loses.
    assert risk._attempt_start_epoch(
        {"started_at": _ts(NOW - 200), "last_resumed_ts": _ts(NOW - 10_000)}
    ) == pytest.approx(NOW - 200)


def test_attempt_start_epoch_tolerates_either_stamp_missing():
    assert risk._attempt_start_epoch({"started_at": _ts(NOW - 300)}) == pytest.approx(NOW - 300)
    assert risk._attempt_start_epoch(
        {"last_resumed_ts": _ts(NOW - 300)}) == pytest.approx(NOW - 300)


def test_attempt_start_epoch_is_none_when_nothing_parses():
    assert risk._attempt_start_epoch({}) is None
    assert risk._attempt_start_epoch({"started_at": "junk", "last_resumed_ts": None}) is None


# `test_attempt_start_epoch_parity_with_herdd` compared three views through
# both copies; one copy since step 6d, and the two degenerate views are pinned
# by value immediately above.


# =============================================================================
# 5. _job_eta_s / _job_pct — strict tri-state over one view.
# =============================================================================
def test_job_eta_s_is_remaining_steps_times_the_step_delta():
    view = _running(last_tail=_tqdm_tail(50, 3000.0, total=100, step_delta=60.0))
    assert risk._job_eta_s(view, NOW) == pytest.approx((100 - 50) * 60.0)


def test_job_eta_s_ignores_now_entirely():
    view = _running(last_tail=_tqdm_tail(50, 3000.0))
    assert risk._job_eta_s(view, 0.0) == risk._job_eta_s(view, 1e12) == risk._job_eta_s(view)


def test_job_eta_s_is_none_for_every_unknown_shape():
    # Tri-state: each of these is UNKNOWN, and None is the only honest answer —
    # 0.0 would read as "about to finish" at the fence.
    assert risk._job_eta_s(None, NOW) is None
    assert risk._job_eta_s("junk", NOW) is None
    assert risk._job_eta_s(_running(display_status="queued",
                                    last_tail=_tqdm_tail(50, 3000.0)), NOW) is None
    assert risk._job_eta_s(_running(last_tail=""), NOW) is None            # no bar
    assert risk._job_eta_s(_running(last_tail=None), NOW) is None
    # one bar only -> no delta -> no ETA (tqdm's own rate is NOT a fallback)
    one_bar = _tqdm_tail(50, 3000.0).splitlines()[-1]
    assert risk._job_eta_s(_running(last_tail=one_bar), NOW) is None


def test_job_eta_s_is_none_when_the_bar_is_past_its_own_total():
    view = _running(last_tail=_tqdm_tail(110, 6600.0, total=100, step_delta=60.0))
    assert risk._job_eta_s(view, NOW) is None


def test_job_eta_s_is_zero_not_none_on_the_last_step():
    view = _running(last_tail=_tqdm_tail(100, 6000.0, total=100, step_delta=60.0))
    assert risk._job_eta_s(view, NOW) == 0.0        # measured and finished != unknown


def test_job_pct_reads_the_last_bar_and_ignores_display_status():
    tail = _tqdm_tail(50, 3000.0)
    assert risk._job_pct(_running(last_tail=tail)) == 50
    # THE ASYMMETRY: unlike every other reader here, `_job_pct` does not gate on
    # display_status — a terminal job's last bar still yields a percent.
    assert risk._job_pct(_running(display_status="exited", last_tail=tail)) == 50


def test_job_pct_is_none_without_a_bar_or_a_dict():
    assert risk._job_pct(None) is None
    assert risk._job_pct("junk") is None
    assert risk._job_pct(_running(last_tail="")) is None
    assert risk._job_pct(_running(last_tail=None)) is None


# `test_job_eta_and_pct_parity_with_herdd` drove five views (including the
# two non-dict ones) through both copies of both readers. One copy each since
# step 6d; the non-dict tolerance and the bar arithmetic are asserted by value
# in this section.


# =============================================================================
# 6. Queue readers: the 0.0-never-None and 0-never-None pair.
# =============================================================================
def test_work_at_risk_h_is_zero_never_none():
    # Deliberate UNDER-statement — it is a price, never the protection.
    for views in (None, [], ["junk", None], [_running(display_status="queued")],
                  [_running(n_checkpoints=0, started_at=None, last_resumed_ts=None,
                            last_checkpoint_ts=None)]):
        got = risk._jobs_work_at_risk_h(views, NOW)
        assert got == 0.0 and isinstance(got, float)


def test_work_at_risk_h_measures_from_the_last_checkpoint_when_there_is_one():
    view = _running(n_checkpoints=2, last_checkpoint_ts=_ts(NOW - 1800))
    assert risk._jobs_work_at_risk_h([view], NOW) == pytest.approx(0.5)


def test_work_at_risk_h_falls_back_to_the_attempt_start():
    never = _running(n_checkpoints=0, last_checkpoint_ts=_ts(NOW - 60),
                     started_at=_ts(NOW - 3600))
    assert risk._jobs_work_at_risk_h([never], NOW) == pytest.approx(1.0)
    # n_checkpoints truthy but the stamp is unreadable -> same fallback
    junk = _running(n_checkpoints=2, last_checkpoint_ts="junk",
                    started_at=_ts(NOW - 3600))
    assert risk._jobs_work_at_risk_h([junk], NOW) == pytest.approx(1.0)


def test_work_at_risk_h_is_the_max_across_tickets_not_the_sum():
    a = _running(job_id="a", n_checkpoints=1, last_checkpoint_ts=_ts(NOW - 3600))
    b = _running(job_id="b", n_checkpoints=1, last_checkpoint_ts=_ts(NOW - 1800))
    assert risk._jobs_work_at_risk_h([a, b], NOW) == pytest.approx(1.0)


def test_work_at_risk_h_clamps_a_future_dated_stamp_to_zero():
    view = _running(n_checkpoints=1, last_checkpoint_ts=_ts(NOW + 3600))
    assert risk._jobs_work_at_risk_h([view], NOW) == 0.0


def test_unresumable_running_is_int_never_none():
    for views in (None, [], ["junk", None]):
        got = risk._jobs_unresumable_running(views)
        assert got == 0 and isinstance(got, int)


def test_unresumable_running_counts_only_running_tickets_with_no_checkpoint():
    views = [_running(job_id="a", n_checkpoints=0),
             _running(job_id="b", n_checkpoints=None),
             _running(job_id="c", n_checkpoints=3),                 # resumable
             _running(job_id="d", display_status="queued", n_checkpoints=0),
             "junk"]
    assert risk._jobs_unresumable_running(views) == 2


def test_unresumable_running_takes_no_now_argument():
    # job_supervise_tick calls it with ONE argument; adding `now` for symmetry
    # would break that call site.
    import inspect
    assert list(inspect.signature(risk._jobs_unresumable_running).parameters) == ["views"]


# =============================================================================
# 7. _jobs_ckpt_stale / _jobs_min_running_eta_s / _jobs_work_horizon_h.
# =============================================================================
def test_ckpt_stale_is_false_when_nothing_declares_or_nothing_is_late():
    assert risk._jobs_ckpt_stale(None, NOW) is False
    assert risk._jobs_ckpt_stale([], NOW) is False
    assert risk._jobs_ckpt_stale(["junk", None], NOW) is False
    assert risk._jobs_ckpt_stale([_running()], NOW) is False              # 200s < 1.5x300
    assert risk._jobs_ckpt_stale([_running(checkpoint_s=None,
                                           last_checkpoint_ts=_ts(NOW - 9000))],
                                 NOW) is False                            # opted out
    assert risk._jobs_ckpt_stale([_running(display_status="queued",
                                           last_checkpoint_ts=_ts(NOW - 9000))],
                                 NOW) is False


def test_ckpt_stale_fires_past_the_fresh_multiple():
    # 1.5 x 300 = 450s; 600s of silence is stale.
    assert risk._jobs_ckpt_stale([_running(last_checkpoint_ts=_ts(NOW - 600))],
                                 NOW) is True


def test_ckpt_stale_falls_back_to_the_attempt_start():
    view = _running(last_checkpoint_ts=None, started_at=_ts(NOW - 600))
    assert risk._jobs_ckpt_stale([view], NOW) is True


def test_ckpt_stale_needs_a_readable_base_stamp():
    view = _running(last_checkpoint_ts="junk", started_at=None, last_resumed_ts=None)
    assert risk._jobs_ckpt_stale([view], NOW) is False


def test_ckpt_stale_mult_sentinel_is_resolved_at_CALL_time(monkeypatch):
    # `mult=None` is a sentinel, not a default: the policy constant is read
    # inside the body, so a rebind moves the threshold for calls made after it.
    # (Contrast `_ckpt_watchdog_alarm`, whose default binds at def time.)
    views = [_running(last_checkpoint_ts=_ts(NOW - 600))]
    assert risk._jobs_ckpt_stale(views, NOW) is True
    monkeypatch.setattr(bidpolicy, "HANDOFF_CKPT_FRESH_MULT", 100.0)
    assert risk._jobs_ckpt_stale(views, NOW) is False
    assert risk._jobs_ckpt_stale(views, NOW, mult=1.5) is True       # explicit wins


def test_ckpt_stale_rejects_a_bool_checkpoint_s():
    view = _running(checkpoint_s=True, last_checkpoint_ts=_ts(NOW - 9000))
    assert risk._jobs_ckpt_stale([view], NOW) is False


def test_min_running_eta_s_is_none_when_no_ticket_yields_an_estimate():
    # A None must reach `_handoff_fence_hold` AS None — the fence ignores an
    # unknown ETA, and a 0.0 here would read as "about to finish".
    for views in (None, [], ["junk", None], [_running(last_tail="")],
                  [_running(display_status="queued", last_tail=_tqdm_tail(50, 3000.0))]):
        assert risk._jobs_min_running_eta_s(views, NOW) is None


def test_min_running_eta_s_takes_the_tightest_estimate():
    near = _running(job_id="near", last_tail=_tqdm_tail(90, 5400.0, step_delta=60.0))
    far = _running(job_id="far", last_tail=_tqdm_tail(10, 600.0, step_delta=60.0))
    assert risk._jobs_min_running_eta_s([far, near, "junk"], NOW) == pytest.approx(600.0)


def test_work_horizon_h_sums_the_running_etas():
    a = _running(job_id="a", last_tail=_tqdm_tail(90, 5400.0, step_delta=60.0))
    b = _running(job_id="b", last_tail=_tqdm_tail(80, 4800.0, step_delta=60.0))
    # (10 + 20) steps x 60s = 1800s = 0.5h, and the ceiling (2 x 36000s minus
    # elapsed) is far larger, so the work estimate stands.
    assert risk._jobs_work_horizon_h([a, b], NOW) == pytest.approx(0.5)


def test_work_horizon_h_is_none_on_an_empty_queue():
    assert risk._jobs_work_horizon_h(None, NOW) is None
    assert risk._jobs_work_horizon_h([], NOW) is None
    assert risk._jobs_work_horizon_h(["junk", None], NOW) is None


def test_work_horizon_h_is_poisoned_by_one_unmeasured_ticket():
    measured = _running(job_id="a", last_tail=_tqdm_tail(90, 5400.0, step_delta=60.0))
    queued = _running(job_id="b", display_status="queued")
    assert risk._jobs_work_horizon_h([measured], NOW) is not None
    assert risk._jobs_work_horizon_h([measured, queued], NOW) is None
    unmeasured = _running(job_id="c", last_tail="")
    assert risk._jobs_work_horizon_h([measured, unmeasured], NOW) is None


def test_work_horizon_h_is_capped_by_the_timeout_ceiling():
    view = _running(timeout_s=1800, started_at=_ts(NOW - 900), last_resumed_ts=None,
                    last_tail=_tqdm_tail(10, 600.0, step_delta=60.0))
    # work = 90 steps x 60s = 5400s = 1.5h; ceiling = (1800 - 900)s = 0.25h.
    assert risk._jobs_work_horizon_h([view], NOW) == pytest.approx(0.25)


def test_work_horizon_h_is_capped_by_the_wall_budget_remainder():
    view = _running(last_tail=_tqdm_tail(10, 600.0, step_delta=60.0))
    assert risk._jobs_work_horizon_h([view], NOW, wall_remaining_h=0.1) == pytest.approx(0.1)


def test_work_horizon_h_is_not_the_timeout_ceiling_defect_67():
    # THE 2026-08-08 22:17Z INCIDENT, as a test. A running ticket with a 10h
    # `timeout_s` and 345s elapsed but NO measurable progress: the ceiling reads
    # ~9.9h, and reading that as a work estimate inflated the projected saving
    # ~5x. The horizon refuses instead.
    view = _running(timeout_s=36_000, started_at=_ts(NOW - 345), last_resumed_ts=None,
                    last_tail="")
    assert risk._jobs_remaining_wall_h([view], NOW) == pytest.approx(9.904, abs=1e-3)
    assert risk._jobs_work_horizon_h([view], NOW) is None


def test_work_horizon_h_keeps_wall_remaining_h_keyword_only():
    # Two suite sites stub this as `lambda views, now, **kw: ...`.
    import inspect
    sig = inspect.signature(risk._jobs_work_horizon_h)
    assert sig.parameters["wall_remaining_h"].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        risk._jobs_work_horizon_h([], NOW, 1.0)


# =============================================================================
# 8. _jobs_defend_hint / _jobs_prior_runtime_h / _jobs_remaining_wall_h.
# =============================================================================
def test_defend_hint_dear_wins_over_cheap():
    views = [_running(job_id="a", defend=bidpolicy.DEFEND_CHEAP),
             _running(job_id="b", defend=bidpolicy.DEFEND_DEAR)]
    assert risk._jobs_defend_hint(views) == bidpolicy.DEFEND_DEAR


def test_defend_hint_cheap_only_when_every_declaration_is_cheap():
    assert risk._jobs_defend_hint(
        [_running(defend="  CHEAP  ")]) == bidpolicy.DEFEND_CHEAP     # trimmed+lowered


def test_defend_hint_is_none_when_nothing_declares():
    # None is NOT "cheap": it leaves the derivation to bidpolicy.resolve_defend,
    # and answering "cheap" would silently disarm every pre-2026-08-14 job.
    assert risk._jobs_defend_hint(None) is None
    assert risk._jobs_defend_hint([]) is None
    assert risk._jobs_defend_hint([_running()]) is None
    assert risk._jobs_defend_hint([_running(defend="expensive")]) is None
    assert risk._jobs_defend_hint([_running(defend=1), "junk", None]) is None


def test_defend_hint_does_not_filter_on_display_status():
    assert risk._jobs_defend_hint(
        [_running(display_status="queued",
                  defend=bidpolicy.DEFEND_DEAR)]) == bidpolicy.DEFEND_DEAR


def test_prior_runtime_h_is_none_when_nothing_is_measurable():
    assert risk._jobs_prior_runtime_h(None, NOW) is None
    assert risk._jobs_prior_runtime_h([], NOW) is None
    assert risk._jobs_prior_runtime_h(["junk", None], NOW) is None
    assert risk._jobs_prior_runtime_h([_running(display_status="queued")], NOW) is None
    assert risk._jobs_prior_runtime_h(
        [_running(started_at=None, last_resumed_ts=None)], NOW) is None
    # The elapsed test is STRICT `> 0`, so a bare running view whose stamps do
    # not parse — and one whose attempt starts exactly now — contribute nothing.
    assert risk._jobs_prior_runtime_h([{"display_status": "running"}], NOW) is None
    assert risk._jobs_prior_runtime_h([_running(started_at=_ts(NOW))], NOW) is None


def test_prior_runtime_h_is_the_longest_running_attempt():
    a = _running(job_id="a", started_at=_ts(NOW - 3600), last_resumed_ts=None)
    b = _running(job_id="b", started_at=_ts(NOW - 7200), last_resumed_ts=None)
    assert risk._jobs_prior_runtime_h([a, b], NOW) == pytest.approx(2.0)


def test_prior_runtime_h_counts_from_the_resume_not_the_first_attempt():
    view = _running(started_at=_ts(NOW - 7200), last_resumed_ts=_ts(NOW - 1800))
    assert risk._jobs_prior_runtime_h([view], NOW) == pytest.approx(0.5)


def test_remaining_wall_h_is_none_when_no_ticket_yields_a_bound():
    assert risk._jobs_remaining_wall_h(None, NOW) is None
    assert risk._jobs_remaining_wall_h([], NOW) is None
    assert risk._jobs_remaining_wall_h(["junk", None], NOW) is None
    assert risk._jobs_remaining_wall_h([_running(timeout_s=None)], NOW) is None
    assert risk._jobs_remaining_wall_h([_running(timeout_s=0)], NOW) is None
    assert risk._jobs_remaining_wall_h([_running(timeout_s=True)], NOW) is None   # bool
    # a running ticket with no readable attempt stamp is skipped, not defaulted
    assert risk._jobs_remaining_wall_h(
        [_running(started_at=None, last_resumed_ts=None)], NOW) is None


def test_remaining_wall_h_is_none_even_when_a_wall_budget_is_supplied():
    # A --wall-budget remainder bounds our SPEND, not the work: with nothing
    # left to run there is no saving to accrue.
    assert risk._jobs_remaining_wall_h([], NOW, wall_remaining_h=5.0) is None


def test_remaining_wall_h_gives_a_queued_ticket_its_whole_budget():
    view = _running(display_status="queued", timeout_s=3600)
    assert risk._jobs_remaining_wall_h([view], NOW) == pytest.approx(1.0)


def test_remaining_wall_h_spends_a_running_tickets_budget():
    view = _running(timeout_s=3600, started_at=_ts(NOW - 900), last_resumed_ts=None)
    assert risk._jobs_remaining_wall_h([view], NOW) == pytest.approx(0.75)


def test_remaining_wall_h_clamps_a_straggler_at_zero():
    # Past its own timeout: contributes nothing rather than subtracting from a
    # neighbour's genuine runway.
    late = _running(job_id="late", timeout_s=600, started_at=_ts(NOW - 7200),
                    last_resumed_ts=None)
    ok = _running(job_id="ok", display_status="queued", timeout_s=3600)
    assert risk._jobs_remaining_wall_h([late], NOW) == 0.0
    assert risk._jobs_remaining_wall_h([late, ok], NOW) == pytest.approx(1.0)


def test_remaining_wall_h_sums_and_a_skipped_ticket_does_not_poison():
    a = _running(job_id="a", display_status="queued", timeout_s=3600)
    b = _running(job_id="b", display_status="queued", timeout_s=1800)
    unreadable = _running(job_id="c", timeout_s=3600, started_at=None,
                          last_resumed_ts=None)
    assert risk._jobs_remaining_wall_h([a, b], NOW) == pytest.approx(1.5)
    assert risk._jobs_remaining_wall_h([a, b, unreadable], NOW) == pytest.approx(1.5)


def test_remaining_wall_h_is_capped_by_the_wall_budget_remainder():
    view = _running(display_status="queued", timeout_s=36_000)
    assert risk._jobs_remaining_wall_h([view], NOW, wall_remaining_h=2.0) == pytest.approx(2.0)


def test_remaining_wall_h_keeps_wall_remaining_h_keyword_only():
    import inspect
    sig = inspect.signature(risk._jobs_remaining_wall_h)
    assert sig.parameters["wall_remaining_h"].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        risk._jobs_remaining_wall_h([], NOW, 1.0)


# =============================================================================
# 9. Port integrity: parity with the live copies, and the seams that must stay
#    patchable.
# =============================================================================
def _view_tables():
    """Queue shapes that exercise every branch the readers can take."""
    return [
        None,
        [],
        ["junk", None, 42],
        [_running()],
        [_running(last_tail=_tqdm_tail(50, 3000.0))],
        [_running(job_id="a", last_tail=_tqdm_tail(90, 5400.0)),
         _running(job_id="b", last_tail=_tqdm_tail(10, 600.0))],
        [_running(display_status="queued"), _running(last_tail=_tqdm_tail(50, 3000.0))],
        [_running(n_checkpoints=0, last_checkpoint_ts=None)],
        [_running(checkpoint_s=True), _running(timeout_s=True)],
        [_running(last_checkpoint_ts=_ts(NOW - 9000))],
        [_running(timeout_s=600, started_at=_ts(NOW - 7200), last_resumed_ts=None)],
        [_running(defend=bidpolicy.DEFEND_CHEAP), _running(defend=bidpolicy.DEFEND_DEAR)],
        [_running(started_at=None, last_resumed_ts=None, last_checkpoint_ts=None)],
    ]


def _flat_twin(name):
    """`herdd.<name>` when it is a SECOND implementation, else None.

    The parity tests below existed for the ADD-ONLY window, when every reader
    had two bodies (plan §8 steps 2-5). Step 6d ended it: `herdd.py` is a thin
    launcher, so each of these names is now either absent from it or the SAME
    OBJECT this module already called — a function compared with itself. Rather
    than delete the tripwire, it is made conditional: it fires the moment a
    genuinely distinct body reappears under one of these names (a peer
    re-adding a copy to the launcher, or a shim that wraps instead of
    re-exporting), and is inert otherwise. Every reader below also has direct
    behavior coverage earlier in this file, so nothing is left unasserted.
    """
    old = getattr(v, name, None)
    return None if old is None or old is getattr(risk, name) else old


@pytest.mark.parametrize("name", ["_jobs_work_at_risk_h", "_jobs_min_running_eta_s",
                                  "_jobs_prior_runtime_h"])
def test_parity_two_arg_readers(name):
    old = _flat_twin(name)
    if old is None:
        pytest.skip(f"{name} has one body since step 6d — see _flat_twin")
    for views in _view_tables():
        assert getattr(risk, name)(views, NOW) == old(views, NOW), name


@pytest.mark.parametrize("name", ["_jobs_unresumable_running", "_jobs_defend_hint"])
def test_parity_one_arg_readers(name):
    old = _flat_twin(name)
    if old is None:
        pytest.skip(f"{name} has one body since step 6d — see _flat_twin")
    for views in _view_tables():
        assert getattr(risk, name)(views) == old(views), name


def test_parity_ckpt_stale():
    old = _flat_twin("_jobs_ckpt_stale")
    if old is None:
        pytest.skip("_jobs_ckpt_stale has one body since step 6d")
    for views in _view_tables():
        assert risk._jobs_ckpt_stale(views, NOW) == old(views, NOW)
        assert (risk._jobs_ckpt_stale(views, NOW, mult=4.0)
                == old(views, NOW, mult=4.0))


@pytest.mark.parametrize("name", ["_jobs_work_horizon_h", "_jobs_remaining_wall_h"])
@pytest.mark.parametrize("wall", [None, 0.1, 100.0])
def test_parity_capped_readers(name, wall):
    old = _flat_twin(name)
    if old is None:
        pytest.skip(f"{name} has one body since step 6d — see _flat_twin")
    for views in _view_tables():
        assert (getattr(risk, name)(views, NOW, wall_remaining_h=wall)
                == old(views, NOW, wall_remaining_h=wall)), name


def test_parity_watchdog_over_the_whole_table():
    old = _flat_twin("_ckpt_watchdog_alarm")
    if old is None:
        pytest.skip("_ckpt_watchdog_alarm has one body since step 6d")
    for views in _view_tables():
        for view in views or ():
            assert risk._ckpt_watchdog_alarm(view, NOW) == old(view, NOW)


def test_ts_to_epoch_is_reached_as_a_module_attribute(monkeypatch):
    # Plan §8(b): cross-module calls stay module-attribute form so the patch
    # idiom survives the port. A `from vastlib.core.fmt import _ts_to_epoch`
    # would bind at import time and this test would go vacuously green.
    from vastlib.core import fmt
    monkeypatch.setattr(fmt, "_ts_to_epoch", lambda ts: None)
    assert risk._attempt_start_epoch({"started_at": _ts(NOW - 300)}) is None
    assert risk._jobs_work_at_risk_h([_running()], NOW) == 0.0
    assert risk._ckpt_watchdog_alarm(_running(last_checkpoint_ts=_ts(NOW - 9000)),
                                     NOW) is None


def test_hms_secs_is_reached_as_a_module_attribute(monkeypatch):
    from vastlib.core import fmt

    def _boom(t):
        raise ValueError(t)

    monkeypatch.setattr(fmt, "_hms_secs", _boom)
    assert risk._tqdm_points(_tqdm_tail(50, 3000.0)) == []


def test_the_module_carries_no_or_zero_coercion():
    # Defect #67's mechanism in one grep: `or 0.0` / `or 0` on a tri-state
    # reader turns UNKNOWN into "about to finish". The tests above pin the
    # behavior; this pins the shape, because the coercion is usually added as a
    # "type cleanup" rather than as a decision.
    src = open(risk.__file__, encoding="utf-8").read()
    body = src.split('"""', 2)[2]           # skip the module docstring
    for line in body.splitlines():
        code = line.split("#", 1)[0]
        assert " or 0.0" not in code and " or 0)" not in code, line
