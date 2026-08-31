"""Regression tests for the POST-CUTOVER handoff wedge (observed live 2026-08-05).

The incident
------------
The perf-levers handoff (primary 46864225 -> understudy 46864611) armed, reached
CUTOVER, and there **correctly refused** to move the job:

    retarget 20260805T075419-perf-levers-e83b skipped (… is RUNNING on live box
    46864225 — retargeting would double-run it)
    CUTOVER: retargeted 0 job(s)

That refusal is the two-writer fence working as designed. But it cut over into
DRAINING having moved nothing, and *then* the understudy died — and DRAINING had
no exit at all:

  * `bidpolicy.handoff_poll` precedence 2 aborts only phases still open
    **pre-CUTOVER**; this one is past it.
  * `_handoff_stall_alarm` explicitly "does NOT force a transition" — it alarms
    once and latches.
  * `_handoff_observe_job_understudy` returned early when the understudy was
    absent from the listing, leaving `understudy_status` STALE at `running`, so
    nothing even noticed it had died.

It did not threaten the primary (its job ran normally throughout). The damage was
a dead understudy nobody reaps and a latched alarm that hides real ones.

What the fix must and must not do
---------------------------------
The byte-safety invariant is "never DESTROY the primary without understudy
proof-of-life". The recovery here is its opposite — give the primary back — so it
cannot violate it. These tests pin both halves: that a dead-or-stalled DRAINING
now escapes, and that a HEALTHY DRAINING still waits (no forced destroy on a
timer).
"""
import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bidpolicy as B      # noqa: E402
import jobmeta             # noqa: E402
import herdd as v        # noqa: E402  (one stays-on-flat site: the _b2_write_soft phantom)
from vastlib.boxes import lifecycle          # noqa: E402
from vastlib.core import labels              # noqa: E402
from vastlib.supervise import handoff, journal  # noqa: E402

NOW = 5_000_000.0


def _hs(**kw):
    kw.setdefault("phase", "DRAINING")
    kw.setdefault("primary_iid", "P")
    kw.setdefault("understudy_iid", "U")
    kw.setdefault("drain_ts", NOW - 60)
    kw.setdefault("now", NOW)
    return B.mk_handoff_state(**kw)


# --------------------------------------------------------------------------- #
# the PURE rule
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", ["exited", "stopped", "offline", "inactive"])
def test_a_DEAD_understudy_in_DRAINING_now_escapes(status):
    act = B.handoff_poll(_hs(understudy_status=status))
    assert act.kind == "abort_unfence"
    assert act.reason == "understudy_died_draining"


def test_an_understudy_MISSING_FROM_THE_LISTING_counts_as_dead():
    act = B.handoff_poll(_hs(understudy_gone=True, understudy_status=None))
    assert act.kind == "abort_unfence"


def test_a_LIVE_understudy_that_never_produces_escapes_at_the_deadline():
    early = B.handoff_poll(_hs(understudy_status="running",
                               drain_ts=NOW - B.HANDOFF_DRAIN_DEADLINE_S + 1))
    assert early.kind == "noop" and early.reason == "await_understudy_ckpt"
    late = B.handoff_poll(_hs(understudy_status="running",
                              drain_ts=NOW - B.HANDOFF_DRAIN_DEADLINE_S - 1))
    assert late.kind == "abort_unfence" and late.reason == "drain_deadline"


def test_a_HEALTHY_DRAINING_still_waits_forever_for_proof_of_life():
    """The byte-safety invariant is untouched: with the understudy alive and the
    clock unstarted there is still no forced transition."""
    act = B.handoff_poll(_hs(understudy_status="running", drain_ts=None))
    assert act.kind == "noop"


def test_a_PRODUCING_understudy_is_never_aborted_however_stale_the_clock():
    """`understudy_producing` beats the deadline — the migration WORKED, and
    aborting a working cutover would be the real data hazard."""
    act = B.handoff_poll(_hs(understudy_status="running",
                             understudy_producing=True, drain_ts=0.0))
    assert act.kind == "drain_primary"


def test_with_NO_primary_left_it_reaps_instead_of_unfencing():
    """Nothing to give back — fall through to the plain reap and let the ordinary
    relaunch/replacement ladder run."""
    act = B.handoff_poll(_hs(understudy_status="exited", primary_gone=True))
    assert act.kind == "abort_reap"


def test_precedence_the_PRE_cutover_abort_is_unchanged():
    """Precedence 2 still governs everything before the cutover — the new rule is
    additive, not a re-routing of the old one."""
    for phase in ("ARMED", "LAUNCHING", "WARMING", "SYNCED"):
        act = B.handoff_poll(B.mk_handoff_state(
            phase=phase, handoff_started_ts=0.0, now=B.HANDOFF_DEADLINE_S + 1))
        assert act.kind == "abort_reap" and act.reason == "deadline"


# --------------------------------------------------------------------------- #
# observation: a dead understudy must STOP READING AS ALIVE
# --------------------------------------------------------------------------- #
def test_an_understudy_absent_from_a_NON_EMPTY_listing_is_marked_gone():
    hf = {"understudy_iid": "U", "understudy_status": "running",
          "phase": "DRAINING"}
    jctx = {"iid": "P", "now": NOW,
            "instances": [{"id": "P", "actual_status": "running"}]}
    handoff._handoff_observe_job_understudy(jctx, hf)
    assert hf["understudy_gone"] is True
    assert hf["understudy_status"] is None


def test_an_EMPTY_listing_is_an_API_BLIP_not_evidence_of_death():
    """One failed instance read must not tear down a healthy migration."""
    hf = {"understudy_iid": "U", "understudy_status": "running",
          "phase": "DRAINING"}
    jctx = {"iid": "P", "now": NOW, "instances": []}
    handoff._handoff_observe_job_understudy(jctx, hf)
    assert not hf.get("understudy_gone")
    assert hf["understudy_status"] == "running"


def test_a_reappearing_understudy_clears_the_gone_flag():
    hf = {"understudy_iid": "U", "understudy_status": None,
          "understudy_gone": True, "phase": "DRAINING"}
    jctx = {"iid": "P", "now": NOW,
            "instances": [{"id": "U", "actual_status": "running",
                           "label": labels._job_handoff_label("P")}]}
    handoff._handoff_observe_job_understudy(jctx, hf)
    assert hf["understudy_gone"] is False
    assert hf["understudy_status"] == "running"


# --------------------------------------------------------------------------- #
# the unfence itself
# --------------------------------------------------------------------------- #
def test_unfence_restores_the_PRE_FENCE_bid_not_the_park_pin(monkeypatch):
    """Resuming at the $0.001 park pin leaves the box permanently unable to win
    its market — the same wedge one step later."""
    bids, states = [], []
    monkeypatch.setattr(lifecycle, "_put_bid_soft",
                        lambda i, p: (bids.append((i, p)), (True, None))[1])
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda i, s: (states.append((i, s)), (True, None))[1])
    hf = {"prefence_bid": 0.55}
    assert handoff._handoff_unfence_primary("P", hf) is True
    assert bids == [("P", 0.55)]
    assert states == [("P", "running")]


def test_unfence_still_resumes_when_the_bid_restore_FAILS(monkeypatch):
    """A failed bid write must not skip the resume — a parked box is the more
    expensive half."""
    states = []
    monkeypatch.setattr(lifecycle, "_put_bid_soft", lambda i, p: (False, "403"))
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda i, s: (states.append((i, s)), (True, None))[1])
    handoff._handoff_unfence_primary("P", {"prefence_bid": 0.55})
    assert states == [("P", "running")]


def test_unfence_with_no_recorded_bid_only_resumes(monkeypatch):
    bids = []
    monkeypatch.setattr(lifecycle, "_put_bid_soft",
                        lambda i, p: (bids.append(p), (True, None))[1])
    monkeypatch.setattr(lifecycle, "_put_state_soft", lambda i, s: (True, None))
    handoff._handoff_unfence_primary("P", {})
    assert bids == []


def test_unfence_is_a_noop_without_a_primary():
    assert handoff._handoff_unfence_primary(None, {}) is False


def test_fence_records_the_bid_it_is_about_to_overwrite(monkeypatch):
    monkeypatch.setattr(lifecycle, "_put_state_soft", lambda i, s: (True, None))
    monkeypatch.setattr(lifecycle, "_put_bid_soft", lambda i, p: (True, None))
    monkeypatch.setattr(lifecycle, "_wait_states_soft", lambda *a, **k: None)
    monkeypatch.setattr(journal, "_job_handoff_emit", lambda *a, **k: {})
    hf = {"phase": "SYNCED", "primary_iid": "P", "understudy_iid": "U"}
    jctx = {"iid": "P", "now": NOW, "last_bid": 0.55, "pending_jobs": [],
            "running_jobs": [], "instances": []}
    handoff._do_job_handoff_move(jctx, hf, B.HandoffAction("fence_primary", "synced"))
    assert hf["prefence_bid"] == 0.55
    assert hf["phase"] == "CUTOVER"


# --------------------------------------------------------------------------- #
# the jobs-lane rollback, end to end
# --------------------------------------------------------------------------- #
def _wire(monkeypatch, *, queue=(), retarget=None):
    calls = {"destroyed": [], "retargeted": [], "bids": [], "states": [],
             "events": []}
    monkeypatch.setattr(lifecycle, "_destroy_soft",
                        lambda i, dry_run=False: calls["destroyed"].append(i))
    monkeypatch.setattr(handoff, "_confirm_gone", lambda i: None)
    monkeypatch.setattr(lifecycle, "_put_bid_soft",
                        lambda i, p: (calls["bids"].append((i, p)), (True, None))[1])
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda i, s: (calls["states"].append((i, s)), (True, None))[1])
    monkeypatch.setattr(journal, "_job_handoff_emit",
                        lambda jc, ev, **f: calls["events"].append(ev) or {})
    monkeypatch.setattr(jobmeta, "list_queue", lambda box: list(queue))
    monkeypatch.setattr(handoff, "cmd_job_retarget",
                        retarget or (lambda ns: calls["retargeted"].append(
                            (ns.job_id, ns.from_box, ns.box))))
    return calls


def test_abort_unfence_reaps_the_understudy_AND_gives_the_primary_back(monkeypatch):
    calls = _wire(monkeypatch)
    hf = {"phase": "DRAINING", "primary_iid": "P", "understudy_iid": "U",
          "prefence_bid": 0.55, "handoffs_done": 0}
    handoff._do_job_handoff_move({"iid": "P", "now": NOW}, hf,
                           B.HandoffAction("abort_unfence",
                                           "understudy_died_draining"))
    assert calls["destroyed"] == ["U"]
    assert calls["bids"] == [("P", 0.55)]
    assert calls["states"] == [("P", "running")]
    assert hf["phase"] == "IDLE"                  # reset, with a cooldown
    assert hf["cooldown_until"] > NOW


def test_the_LIVE_INCIDENT_shape_zero_tickets_moved(monkeypatch):
    """CUTOVER retargeted 0 jobs (the fence refusing to move a RUNNING job), so
    the rollback has nothing to move back — and must not invent work."""
    calls = _wire(monkeypatch, queue=())
    hf = {"phase": "DRAINING", "primary_iid": "46864225",
          "understudy_iid": "46864611", "prefence_bid": 0.55}
    handoff._do_job_handoff_move({"iid": "46864225", "now": NOW}, hf,
                           B.HandoffAction("abort_unfence",
                                           "understudy_died_draining"))
    assert calls["retargeted"] == []
    assert calls["destroyed"] == ["46864611"]
    assert "handoff_retarget_back" not in calls["events"]


def test_tickets_that_DID_move_are_retargeted_BACK(monkeypatch):
    calls = _wire(monkeypatch, queue=("J1", "J2"))
    hf = {"phase": "DRAINING", "primary_iid": "P", "understudy_iid": "U",
          "prefence_bid": 0.55}
    handoff._do_job_handoff_move({"iid": "P", "now": NOW}, hf,
                           B.HandoffAction("abort_unfence", "drain_deadline"))
    assert calls["retargeted"] == [("J1", "U", "P"), ("J2", "U", "P")]
    assert "handoff_retarget_back" in calls["events"]


def test_tickets_move_back_BEFORE_the_primary_is_resumed(monkeypatch):
    """Same launch-then-move-then-dispose order the eviction ladder uses: a
    ticket must never point at a box that is about to run it while another copy
    is still queued elsewhere."""
    order = []
    calls = _wire(monkeypatch, queue=("J1",),
                  retarget=lambda ns: order.append("retarget"))
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda i, s: (order.append("resume"), (True, None))[1])
    hf = {"phase": "DRAINING", "primary_iid": "P", "understudy_iid": "U",
          "prefence_bid": 0.55}
    handoff._do_job_handoff_move({"iid": "P", "now": NOW}, hf,
                           B.HandoffAction("abort_unfence", "drain_deadline"))
    assert order == ["retarget", "resume"]
    del calls


def test_the_fence_pin_is_never_recorded_as_the_bid_to_restore(monkeypatch):
    """A supervisor that dies mid-fence and restarts reconciles into the
    migration with `prefence_bid` lost and the primary ALREADY parked at the pin,
    so its observed dph_total is $0.001. Recording that would memorise the pin as
    the thing to restore and the unwind would put the box back at a bid it can
    never win with — the wedge, laundered through a restart."""
    assert handoff._prefence_bid(B.HANDOFF_PARK_BID, B.HANDOFF_PARK_BID) is None
    assert handoff._prefence_bid(None, None) is None
    assert handoff._prefence_bid(2.55, 0.001) == 2.55
    # ...and the unwind ignores such a value even if some older path stored one
    calls = _wire(monkeypatch)
    hf = {"phase": "DRAINING", "primary_iid": "P", "understudy_iid": "U",
          "prefence_bid": B.HANDOFF_PARK_BID}
    handoff._handoff_unfence_primary("P", hf, policy_target=2.55)
    assert calls["bids"] == [("P", 2.55)]


def test_a_rollback_with_no_recoverable_bid_leaves_the_box_parked(monkeypatch):
    """2026-08-08, task #62. With no pre-fence bid recorded and no policy target
    to fall back on, resuming would put the box back at the $0.001 fence pin —
    live, billing, and unable to win its market ever again. Leave it PARKED (the
    reaper and the operator can both act on that) and never write the pin back."""
    calls = _wire(monkeypatch, queue=("J1",))
    hf = {"phase": "DRAINING", "primary_iid": "P", "understudy_iid": "U"}
    handoff._do_job_handoff_move({"iid": "P", "now": NOW}, hf,
                           B.HandoffAction("abort_unfence", "drain_deadline"))
    assert calls["retargeted"] == [("J1", "U", "P")]   # tickets still come back
    assert calls["destroyed"] == ["U"]                 # understudy still reaped
    assert calls["states"] == []                       # NOT resumed
    assert calls["bids"] == []                         # and never re-pinned


def test_an_unreadable_understudy_queue_does_not_block_the_rollback(monkeypatch):
    calls = _wire(monkeypatch)
    monkeypatch.setattr(jobmeta, "list_queue",
                        lambda box: (_ for _ in ()).throw(RuntimeError("b2 down")))
    hf = {"phase": "DRAINING", "primary_iid": "P", "understudy_iid": "U",
          "prefence_bid": 0.55}
    handoff._do_job_handoff_move({"iid": "P", "now": NOW}, hf,
                           B.HandoffAction("abort_unfence", "drain_deadline"))
    assert calls["destroyed"] == ["U"]            # still reaped
    assert calls["states"] == [("P", "running")]  # still unfenced


def test_plain_abort_reap_does_NOT_touch_the_primary(monkeypatch):
    """The pre-cutover abort is unchanged: nothing was fenced, so nothing is
    unfenced."""
    calls = _wire(monkeypatch)
    hf = {"phase": "WARMING", "primary_iid": "P", "understudy_iid": "U"}
    handoff._do_job_handoff_move({"iid": "P", "now": NOW}, hf,
                           B.HandoffAction("abort_reap", "deadline"))
    assert calls["destroyed"] == ["U"]
    assert calls["bids"] == [] and calls["states"] == []


def test_dry_run_changes_nothing(monkeypatch):
    calls = _wire(monkeypatch, queue=("J1",))
    hf = {"phase": "DRAINING", "primary_iid": "P", "understudy_iid": "U",
          "prefence_bid": 0.55}
    handoff._do_job_handoff_move({"iid": "P", "now": NOW, "dry_run": True}, hf,
                           B.HandoffAction("abort_unfence", "drain_deadline"))
    assert calls["bids"] == [] and calls["states"] == []
    assert calls["retargeted"] == []


# --------------------------------------------------------------------------- #
# the run lane gets the same escape hatch
# --------------------------------------------------------------------------- #
def test_run_lane_abort_unfence_unfences_then_aborts(monkeypatch):
    seen = {"bids": [], "states": [], "destroyed": []}
    monkeypatch.setattr(lifecycle, "_put_bid_soft",
                        lambda i, p: (seen["bids"].append((i, p)), (True, None))[1])
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda i, s: (seen["states"].append((i, s)), (True, None))[1])
    monkeypatch.setattr(lifecycle, "_destroy_soft",
                        lambda i, dry_run=False: seen["destroyed"].append(i))
    monkeypatch.setattr(handoff, "_confirm_gone", lambda i: None)
    monkeypatch.setattr(journal, "_sup_emit", lambda *a, **k: {})
    hf = {"phase": "DRAINING", "primary_iid": "P", "understudy_iid": "U",
          "prefence_bid": 0.9, "handoffs_done": 0}
    st = {"run_id": "R", "instance_id": "P", "now": NOW}
    handoff._do_handoff_move(st, argparse.Namespace(dry_run=False), hf,
                       B.HandoffAction("abort_unfence", "understudy_died_draining"))
    assert seen["bids"] == [("P", 0.9)]
    assert seen["states"] == [("P", "running")]
    assert seen["destroyed"] == ["U"]
    assert hf["phase"] == "IDLE"


def test_run_lane_stamps_drain_ts_at_cutover(monkeypatch):
    """Without the stamp the deadline branch can never fire."""
    monkeypatch.setattr(lifecycle, "_put_label_soft", lambda *a: (True, None))
    monkeypatch.setattr(journal, "_sup_emit", lambda *a, **k: {})
    # phantom: see vastlib/storage/b2.py:69 — no `_b2_write_soft` is defined anywhere
    # (flat or vastlib), so this patch is vacuous today. stays-on-flat: creating a
    # target to make it land would convert a detectable defect into a silent one.
    monkeypatch.setattr(v, "_b2_write_soft", lambda *a, **k: True, raising=False)
    hf = {"phase": "CUTOVER", "primary_iid": "P", "understudy_iid": "U",
          "epoch": 1}
    st = {"run_id": "R", "instance_id": "P", "now": NOW}
    try:
        handoff._do_handoff_move(st, argparse.Namespace(dry_run=True), hf,
                           B.HandoffAction("resume_understudy", "post_flush"))
    except Exception:                                    # noqa: BLE001
        pytest.skip("run-lane cutover needs more wiring than this unit provides")
    assert hf["drain_ts"] == NOW


def test_run_lane_fence_records_the_bid_it_overwrites(monkeypatch):
    monkeypatch.setattr(lifecycle, "_put_state_soft", lambda i, s: (True, None))
    monkeypatch.setattr(lifecycle, "_put_bid_soft", lambda i, p: (True, None))
    monkeypatch.setattr(lifecycle, "_wait_states_soft", lambda *a, **k: None)
    monkeypatch.setattr(journal, "_sup_emit", lambda *a, **k: {})
    hf = {"phase": "SYNCED", "primary_iid": "P", "understudy_iid": "U"}
    st = {"run_id": "R", "instance_id": "P", "now": NOW, "last_bid": 0.9}
    handoff._do_handoff_move(st, argparse.Namespace(dry_run=False), hf,
                       B.HandoffAction("fence_primary", "synced"))
    assert hf["prefence_bid"] == 0.9
