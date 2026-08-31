#!/usr/bin/env python3
"""test_rebid_ladder — the escalation rung between a stalled rescue and a rental.

Before 2026-08-08 an outbid box got exactly ONE `rescue_bid` (aimed at the
ordinary standing target) and then, if that did not bring it back, the ladder
went straight to `unrecoverable` -> rent a replacement. There was no second bid
at any price, and the alarm text told the operator to "raise the bid" by hand.

That ordering is backwards on cost: a replacement pays a MEASURED 11m35s of
setup on a cold disk and loses whatever never reached B2, while a re-bid moves
the price on a box that still holds its rehydrated env, base model, dataset and
newest checkpoint.

Every test here FAILS on the pre-2026-08-08 code — `bidpolicy.rebid_ladder` and
`herdd._job_rebid_ladder` did not exist, and the supervise tick went from a
stalled rescue to `_job_eviction_replace` with nothing in between.
"""
import argparse
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bidpolicy as bp  # noqa: E402
from vastlib.boxes import lifecycle  # noqa: E402
from vastlib.core import config  # noqa: E402
from vastlib.supervise import journal, replacement  # noqa: E402


def _lad(**kw):
    base = dict(last_bid=0.80, market_min_bid=0.90, on_demand=2.40,
                max_bid=None, rungs_used=0, launch_dph_anchor=0.76,
                budget_usd=10.0, spend_usd=1.0)
    base.update(kw)
    return bp.rebid_ladder(**base)


# --------------------------------------------------------------------------- #
# the rung itself
# --------------------------------------------------------------------------- #
def test_a_risen_floor_gets_a_second_bid_at_all():
    """The whole point. v7 eviction 1's floor went $0.7599 -> $0.9099 under a
    $0.747 bid and the ladder had no move left. Now it has three."""
    d = _lad()
    assert d.action == "rebid"
    assert d.price > 0.80
    assert d.rungs_left == bp.REBID_MAX_RUNGS - 1


def test_the_rung_clears_the_new_floor_not_just_the_old_bid():
    """A flat +25% step is not enough when the floor jumped further than that.
    The rung is max(step, the ordinary cushioned target for the CURRENT floor)."""
    d = _lad(last_bid=0.50, market_min_bid=1.20, on_demand=4.00)
    assert d.action == "rebid"
    assert d.price > 1.20, "a rung under the live floor cannot win"
    assert d.price >= 0.50 * (1 + bp.REBID_STEP)


def test_the_step_binds_when_the_floor_barely_moved():
    """When the ordinary cushioned target is BELOW a +25% step (a tight
    on-demand price caps it), the step is what moves. Note the corollary: on a
    machine with a lot of on-demand headroom the ordinary target already
    outruns the step, and the ladder's value there is the FRESH floor read plus
    the fact that there is a second attempt at all."""
    d = _lad(last_bid=0.80, market_min_bid=0.81, on_demand=1.40)
    assert d.action == "rebid"
    assert d.price == pytest.approx(1.0)          # 0.80 x 1.25


def test_the_hard_ondemand_ceiling_stops_the_ladder_climbing(monkeypatch):
    """Recalibration item A: a rescue rung is an emitted bid like any other, so it
    is bounded by the SAME hard ceiling the defend path is
    (BID_CEILING_ONDEMAND_FRAC x on-demand). "The box is warm" is not a licence to
    pay a price the standing-bid policy is forbidden to pay.

    Same shape as the test above but with on-demand at $1.30, whose ceiling is
    0.75 x 1.30 = $0.975. The +25% step wants $1.00 and is clamped; the ladder
    still places the partial rung (clamp, do not filter — D8) and the NEXT rung
    has no headroom, which is the fall-through to `replacement_decision`."""
    d = _lad(last_bid=0.80, market_min_bid=0.81, on_demand=1.30)
    assert d.action == "rebid"
    assert d.price == pytest.approx(0.975)        # 0.75 x 1.30, not 0.80 x 1.25
    assert d.ceiling == pytest.approx(0.975)
    # one rung later there is nothing left to climb: stop, and say which bound
    d2 = _lad(last_bid=0.975, market_min_bid=0.81, on_demand=1.30, rungs_used=1)
    assert d2.action == "stop"
    assert "HARD on-demand ceiling" in d2.reason
    assert "replacement rung" in d2.reason


def test_a_floor_above_the_hard_ceiling_escalates_rather_than_bidding():
    """The other half: the machine's floor is so close to its on-demand rate that
    no survivable bid fits under the ceiling. Bidding into that buys a
    preemptible box at ~on-demand prices; the ladder must say so by name so the
    replacement rung's on-demand escalation reads as a decision, not a failure."""
    d = _lad(last_bid=1.90, market_min_bid=1.95, on_demand=2.10,
             launch_dph_anchor=1.90)
    assert d.action == "stop"
    assert "escalate_over_ceiling" in d.reason
    assert "only a different box wins this back" in d.reason


# --------------------------------------------------------------------------- #
# every refusal is bounded, named, and carries its arithmetic
# --------------------------------------------------------------------------- #
def test_ondemand_displacement_is_never_bid_into():
    """v7 eviction 2: a $1.05 bid lost to a $1.0017 on-demand rate. No rung wins
    that, and spending one on it is the exact mistake the alarm text used to
    recommend."""
    d = _lad(eviction_class=bp.EVICTION_ONDEMAND)
    assert d.action == "stop"
    assert "on-demand claim" in d.reason


def test_rungs_are_exhaustible_and_the_refusal_says_so():
    d = _lad(rungs_used=bp.REBID_MAX_RUNGS)
    assert d.action == "stop" and d.rungs_left == 0
    assert "exhausted" in d.reason and f"{bp.REBID_MAX_RUNGS}" in d.reason


def test_the_ceiling_is_the_replacement_ceiling_and_it_stops_the_ladder():
    """Ladder ceiling == 2.0 x the ORIGINAL launch anchor == what
    `replacement_decision` may spend. Money we would authorise for a COLD
    replacement we authorise for the WARM box; above that both rungs stop."""
    d = _lad(last_bid=1.515, launch_dph_anchor=0.76, market_min_bid=1.50,
             on_demand=4.00)
    assert d.action == "stop"
    assert d.ceiling == pytest.approx(1.52)
    assert "ceiling" in d.reason and "headroom" in d.reason
    assert bp.REBID_CEILING_MULT == bp.REPLACE_CEILING_MULT


def test_a_partial_rung_under_the_ceiling_is_taken_not_refused():
    """Clamp, do not filter: with $0.20 of headroom left the ladder places a
    $0.20 rung rather than refusing into a replacement."""
    d = _lad(last_bid=1.32, launch_dph_anchor=0.76, market_min_bid=1.30,
             on_demand=4.00)
    assert d.action == "rebid"
    assert d.price == pytest.approx(1.52)         # the ceiling exactly


def test_no_anchor_and_no_max_bid_is_a_refusal_not_a_guess():
    d = _lad(launch_dph_anchor=None, max_bid=None)
    assert d.action == "stop" and "cannot derive a re-bid ceiling" in d.reason


def test_an_unwinnable_floor_stops_the_ladder():
    """Floor at/over on-demand: no legal bid takes the machine back, only a
    different box does. This is the v7-eviction-1 market AFTER the spike."""
    d = _lad(market_min_bid=0.95, on_demand=0.95, launch_dph_anchor=5.0)
    assert d.action == "stop"
    assert "no winnable bid" in d.reason or "ceiling" in d.reason


def test_budget_that_cannot_buy_the_runtime_floor_refuses():
    d = _lad(budget_usd=1.0, spend_usd=0.9)
    assert d.action == "stop"
    assert "buys only" in d.reason


def test_an_uncapped_watch_may_still_re_bid():
    """Deliberate asymmetry with `replacement_decision`, which REQUIRES a budget
    cap: a rental starts a SECOND meter, a re-bid moves the price of one already
    running and already bounded by the watch's hard spend stop. Requiring a cap
    here would silently disarm rescue on every uncapped watch."""
    d = _lad(budget_usd=None)
    assert d.action == "rebid"


def test_max_rungs_zero_disables_the_ladder():
    d = _lad(max_rungs=0)
    assert d.action == "stop" and "disabled" in d.reason


def test_max_bid_is_never_exceeded_by_a_rung():
    d = _lad(max_bid=0.85, launch_dph_anchor=5.0)
    if d.action == "rebid":
        assert d.price <= 0.85
    d2 = _lad(max_bid=0.805, launch_dph_anchor=5.0)
    assert d2.action == "stop"                    # no material raise left


def test_a_rung_never_reaches_ondemand():
    for od in (0.9, 1.2, 2.4):
        d = _lad(last_bid=0.80, market_min_bid=0.85, on_demand=od,
                 launch_dph_anchor=10.0)
        if d.action == "rebid":
            assert d.price < od


def test_the_wall_budget_is_one_replacement_setup():
    """REBID_MAX_RUNGS x REBID_WAIT_S must not exceed a replacement's measured
    setup cost by more than slack — preferring the ladder can never cost more
    wall time than the thing it avoids (11m35s = 695 s, v11 eval lane)."""
    assert bp.REBID_MAX_RUNGS * bp.REBID_WAIT_S <= 1000


# --------------------------------------------------------------------------- #
# driver wiring: the ladder runs BEFORE the replacement rung
# --------------------------------------------------------------------------- #
def _jc(**kw):
    jc = {"iid": "700", "last_bid": 0.80, "max_bid": None,
          "launch_dph_anchor": 0.76, "spend_usd": 1.0, "rebid_rungs": 0,
          "a": argparse.Namespace(budget=10.0, dry_run=False)}
    jc.update(kw)
    return jc


def test_driver_places_the_rung_and_arms_a_bounded_wait(monkeypatch):
    puts = []
    emits = []
    monkeypatch.setattr(lifecycle, "_put_bid_soft",
                        lambda iid, p: (puts.append((iid, p)) or (True, None)))
    monkeypatch.setattr(journal, "_job_handoff_emit",
                        lambda jc, kind, **kw: emits.append((kind, kw)))
    jc = _jc()
    kept = replacement._job_rebid_ladder(jc, jc["a"], "700", 0.90, 2.40,
                               bp.EVICTION_OUTBID, 1000.0)
    assert kept is True
    assert puts and puts[0][0] == "700"
    assert jc["last_bid"] == puts[0][1] > 0.80
    assert jc["rebid_rungs"] == 1
    assert jc["rescue_deadline"] == 1000.0 + bp.REBID_WAIT_S
    assert [k for k, _ in emits] == ["rebid_ladder"]


def test_driver_refusal_is_journaled_and_falls_through(monkeypatch):
    emits = []
    monkeypatch.setattr(lifecycle, "_put_bid_soft",
                        lambda iid, p: pytest.fail("must not PUT on a refusal"))
    monkeypatch.setattr(journal, "_job_handoff_emit",
                        lambda jc, kind, **kw: emits.append((kind, kw)))
    jc = _jc(rebid_rungs=bp.REBID_MAX_RUNGS)
    kept = replacement._job_rebid_ladder(jc, jc["a"], "700", 0.90, 2.40,
                               bp.EVICTION_OUTBID, 1000.0)
    assert kept is False                          # -> the replacement rung runs
    assert jc["rebid_refused"] and "exhausted" in jc["rebid_refused"]
    assert [k for k, _ in emits] == ["rebid_refused"]


def test_driver_keeps_the_box_when_the_put_itself_fails(monkeypatch):
    """A 429 or transient PUT failure must NOT be read as "the bid ladder is out
    of moves" and hand the box to the replacement rung — the rung was never
    actually placed. Retry next tick instead."""
    monkeypatch.setattr(lifecycle, "_put_bid_soft", lambda iid, p: (False, "HTTP 429"))
    monkeypatch.setattr(journal, "_job_handoff_emit", lambda jc, kind, **kw: None)
    jc = _jc()
    assert replacement._job_rebid_ladder(jc, jc["a"], "700", 0.90, 2.40,
                               bp.EVICTION_OUTBID, 1000.0) is True
    assert jc["rebid_rungs"] == 0                 # not consumed by a failed PUT


def test_driver_dry_run_moves_no_money(monkeypatch):
    monkeypatch.setattr(lifecycle, "_put_bid_soft",
                        lambda iid, p: pytest.fail("dry-run must not PUT"))
    monkeypatch.setattr(journal, "_job_handoff_emit", lambda jc, kind, **kw: None)
    jc = _jc(a=argparse.Namespace(budget=10.0, dry_run=True))
    assert replacement._job_rebid_ladder(jc, jc["a"], "700", 0.90, 2.40,
                               bp.EVICTION_OUTBID, 1000.0) is False


# --------------------------------------------------------------------------- #
# the parameters are CONFIG, not constants
# --------------------------------------------------------------------------- #
def test_ladder_knobs_resolve_namespace_then_env_then_yaml(monkeypatch):
    jc = {"a": argparse.Namespace(rebid_max_rungs=7)}
    assert replacement._rebid_knob(jc, "rebid_max_rungs", bp.REBID_MAX_RUNGS) == 7

    jc = {"a": argparse.Namespace(rebid_max_rungs=None)}
    monkeypatch.setenv("JOB_REBID_MAX_RUNGS", "5")
    assert replacement._rebid_knob(jc, "rebid_max_rungs", bp.REBID_MAX_RUNGS) == 5
    monkeypatch.delenv("JOB_REBID_MAX_RUNGS")

    monkeypatch.setattr(config, "load_herdd_config",
                        lambda: {"JOB_REBID_MAX_RUNGS": "4"})
    assert replacement._rebid_knob(jc, "rebid_max_rungs", bp.REBID_MAX_RUNGS) == 4

    monkeypatch.setattr(config, "load_herdd_config", lambda: {})
    assert replacement._rebid_knob(jc, "rebid_max_rungs",
                         bp.REBID_MAX_RUNGS) == bp.REBID_MAX_RUNGS


def test_a_malformed_knob_never_disarms_the_ladder(monkeypatch):
    jc = {"a": argparse.Namespace(rebid_step=None)}
    monkeypatch.setenv("JOB_REBID_STEP", "not-a-number")
    assert replacement._rebid_knob(jc, "rebid_step", bp.REBID_STEP) == bp.REBID_STEP


def test_the_rung_counter_is_durable_fleetd_state():
    """A restart that forgot the counter would let a box that already spent its
    whole ladder start another one — an unbounded loop of bounded stalls."""
    import fleetd
    assert "rebid_rungs" in fleetd.REPLACEMENT_STATE_KEYS
