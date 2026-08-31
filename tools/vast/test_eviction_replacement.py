"""Portable tests for AUTOMATIC EVICTION REPLACEMENT (owner directive
2026-08-05, after the v7 training run cost two hand-rescues in one night —
docs/plans/witness/g2_push/V7_TRAIN_RUN_2026-08-05.md).

What is under test, in three layers:

  * `bidpolicy.classify_eviction` — outbid vs on-demand displacement vs host
    failure. The discriminator is load-bearing: only one of the three is
    winnable by bidding, and calling an on-demand claim an "outbid" is what made
    the v7 incident read as a pricing problem.
  * `bidpolicy.replacement_decision` — the spend bounds (budget remainder,
    replacement count cap, price ceiling derived from the ORIGINAL launch,
    minimum affordable runtime) and the spot -> on-demand rung ladder.
  * `herdd._job_eviction_replace` + the `job_supervise_tick` seam — the
    hand-off to the existing jobs-v2 machinery (launch -> retarget -> destroy ->
    re-anchor), and the guarantee that a REFUSAL leaves the pre-existing
    `unrecoverable` behavior exactly as it was.
  * `fleetd` — the replacement counters survive a daemon restart (they are spend
    bounds, so a restart must not reset them).

Toolchain-free lane: no vast API, no B2 — every seam monkeypatched, no clock.
"""
import argparse
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bidpolicy as bp  # noqa: E402
import disksize  # noqa: E402
import fleetd  # noqa: E402
import imageref  # noqa: E402
import jobmeta  # noqa: E402
import train_rates  # noqa: E402
from vastlib.boxes import lifecycle  # noqa: E402
from vastlib.core import config, labels, models  # noqa: E402
from vastlib.jobs import risk as jobs_risk  # noqa: E402
from vastlib.launch import launch  # noqa: E402
from vastlib.market import hostrep  # noqa: E402
from vastlib.market import offers as market_offers  # noqa: E402
from vastlib.market import pricing  # noqa: E402
from vastlib.supervise import (handoff, job_lane, journal,  # noqa: E402
                               replacement, retention)

NOW = 3_000_000.0

# The two v7 evictions, as numbers. Quoted from the run readout's incident
# section so a change in policy that would have re-broken that night fails here.
V7_E1 = dict(bid=0.747, floor_before=0.7599, floor_after=0.9099, on_demand=0.748)
V7_E2 = dict(bid=1.05, on_demand=1.0017)


# --------------------------------------------------------------------------- #
# 1. eviction classification
# --------------------------------------------------------------------------- #
def test_gone_from_the_listing_is_host_failure():
    assert bp.classify_eviction(present=False) == bp.EVICTION_HOST_FAILURE


def test_still_live_is_never_an_eviction():
    """The caller debounces (NOT_LIVE_DEBOUNCE); a blip must not classify."""
    assert bp.classify_eviction(present=True, actual_status="running",
                                market_min_bid=9.0, last_bid=0.5) \
        == bp.EVICTION_UNKNOWN


def test_v7_eviction_1_is_an_outbid_not_an_ondemand_claim():
    """Regression on the ORDER of the two tests. The v7 eviction-1 box was bid
    at $0.747 against a $0.748 on-demand price — i.e. `bid >= on_demand - eps`
    was ALSO true — but the floor had risen to $0.9099, which is direct evidence
    of another bidder. Floor-risen must win, or a plain outbid routes to the
    expensive rung for a reason it did not earn."""
    assert bp.classify_eviction(
        present=True, actual_status="exited",
        market_min_bid=V7_E1["floor_after"], on_demand=V7_E1["on_demand"],
        last_bid=V7_E1["bid"]) == bp.EVICTION_OUTBID


def test_v7_eviction_2_is_ondemand_displacement():
    """Bidding ABOVE on-demand and losing anyway: the only renter who can do
    that is an on-demand one. Unwinnable by price, by construction."""
    assert bp.classify_eviction(
        present=True, actual_status="exited", market_min_bid=0.6,
        on_demand=V7_E2["on_demand"], last_bid=V7_E2["bid"]) \
        == bp.EVICTION_ONDEMAND


def test_no_market_read_is_unknown_never_outbid():
    """An eviction event that asserts a cause it could not observe is the log
    line that sends the next postmortem the wrong way."""
    assert bp.classify_eviction(present=True, actual_status="exited",
                                last_bid=0.7) == bp.EVICTION_UNKNOWN


def test_bid_can_win_is_false_once_the_floor_clears_on_demand():
    assert bp.bid_can_win(V7_E1["floor_before"], 2.0) is True
    # post-spike floor above the machine's on-demand price: no legal bid wins
    assert bp.bid_can_win(V7_E1["floor_after"], V7_E1["on_demand"]) is False


# --------------------------------------------------------------------------- #
# 2. replacement decision math (spend bounds + rung ladder)
# --------------------------------------------------------------------------- #
def _dec(**kw):
    base = dict(eviction_class=bp.EVICTION_OUTBID, replacements_used=0,
                budget_usd=5.0, spend_usd=0.10, launch_dph_anchor=0.76,
                offer_min_bid=0.50, offer_ondemand=1.20)
    base.update(kw)
    return bp.replacement_decision(**base)


def test_healthy_market_rents_spot():
    d = _dec()
    assert (d.action, d.rental) == ("rent", "bid")
    assert d.price == pytest.approx(0.60)          # 1.2 x the $0.50 floor (the 0.65
                                                   # x od cap, $0.78, stays dormant)
    assert d.ceiling == pytest.approx(1.52)        # 2.0x the $0.76 launch price


def test_no_budget_cap_refuses_every_autonomous_rental():
    """The one gate no knob may widen: an uncapped watch does not spend."""
    d = _dec(budget_usd=None)
    assert d.action == "stop" and "no budget cap" in d.reason


def test_replacement_count_cap_stops_the_loop():
    d = _dec(replacements_used=3, max_replacements=3)
    assert d.action == "stop" and "replacement cap" in d.reason
    assert _dec(replacements_used=2, max_replacements=3).action == "rent"


def test_max_replacements_zero_disables_the_feature():
    assert _dec(replacements_used=0, max_replacements=0).action == "stop"


def test_exhausted_budget_stops():
    d = _dec(spend_usd=5.0)
    assert d.action == "stop" and "budget exhausted" in d.reason


def test_budget_that_buys_only_minutes_is_refused():
    """Renting a box the remaining budget can afford for four minutes spends
    money for zero progress and then budget-parks."""
    d = _dec(spend_usd=4.95)                       # $0.05 left, $0.60/hr spot
    assert d.action == "stop" and "0.25h floor" in d.reason


def test_price_ceiling_is_derived_from_the_original_launch():
    d = _dec(offer_min_bid=2.0, offer_ondemand=4.0)   # both way over 2x $0.76
    assert d.action == "stop"
    assert "over the $1.520 ceiling" in d.reason and "launch price" in d.reason
    # a bigger multiplier admits the same market
    assert _dec(offer_min_bid=2.0, offer_ondemand=4.0,
                ceiling_mult=6.0).action == "rent"


def test_missing_launch_anchor_refuses_rather_than_guesses():
    d = _dec(launch_dph_anchor=None)
    assert d.action == "stop" and "price ceiling" in d.reason


# --------------------------------------------------------------------------- #
# 2b. replacement CEILING re-pricing (REPLACEMENT_CEILING_WEDGE_2026-08-24)
#
# The incident numbers, so the arithmetic is graded against the real market and
# not against a scenario invented to pass: anchor $0.1933 -> base ceiling
# $0.387; the only qualifying offer billed $0.4000/hr; the daemon held a
# $0.5333 `p_alt` read it never consulted; the machine's on-demand was $1.20.
# The pull-reschedule lane refused for 33 minutes over a 3.4% gap.
# --------------------------------------------------------------------------- #
ANCHOR = 0.19333333333333333          # box 48537477's launch price
BASE = 0.387                          # 2.0 x ANCHOR


def _ceil(**kw):
    base = dict(launch_dph_anchor=ANCHOR)
    base.update(kw)
    return bp.replacement_ceiling(**base)


def test_no_market_evidence_leaves_the_base_ceiling_exactly_as_it_was():
    """The whole pre-2026-08-24 behaviour, unchanged: with nothing to price
    against, the ceiling is `mult x anchor` and a market over it is refused."""
    c = _ceil()
    assert (c.price, c.base, c.escalated) == (BASE, BASE, False)
    assert c.market_ref is None and "no live market evidence" in c.bound


def test_a_market_under_the_base_never_escalates():
    """Evidence is not a licence. A market the base already covers moves
    nothing — the ceiling is a cost bound, not a market tracker."""
    c = _ceil(market_floor=0.30, p_alt=0.35)
    assert (c.price, c.escalated) == (BASE, False)
    assert "market under the base ceiling" in c.bound


def test_a_stale_anchor_against_a_moved_market_is_repriced_within_bounds():
    """The incident, exactly. The re-derived ceiling must clear the $0.4000
    offer and must stay far under every safety bound."""
    c = _ceil(market_floor=0.40, p_alt=0.5333333, on_demand=1.2,
              budget_left=59.74)
    assert c.escalated is True and c.source == "market_floor"
    assert c.price == pytest.approx(0.44)        # 1.10 x the $0.4000 read
    assert c.price > 0.40, "the re-priced ceiling must clear the offer that was refused"
    assert c.price < bp.REPLACE_ESCALATION_CAP_MULT * ANCHOR
    assert c.price < bp.BID_CEILING_ONDEMAND_FRAC * 1.2


def test_the_reference_is_the_cheapest_evidence_not_the_loudest():
    """Two readings of the same market: price against the one that actually
    unblocks the lane. `p_alt` is a class-wide floor and reads $0.5333 here;
    paying 1.10x THAT would buy $0.15/hr of headroom nobody needs."""
    c = _ceil(market_floor=0.40, p_alt=0.5333333)
    assert c.source == "market_floor" and c.market_ref == pytest.approx(0.40)
    # ...and with no observed offer (the `no_offer` shape — 29 of 36 refusals,
    # because the ceiling is pushed into the search and empties it) p_alt is
    # what remains, which is the whole point of consulting it.
    c = _ceil(p_alt=0.5333333)
    assert c.source == "p_alt" and c.price == pytest.approx(0.587)


def test_escalation_is_capped_at_an_absolute_multiple_of_the_ORIGINAL_anchor():
    """A runaway market may not be followed. The cap is on the anchor, never on
    the previous ceiling, which is what makes N swaps unable to compound."""
    c = _ceil(market_floor=5.0, on_demand=100.0, budget_left=10_000.0)
    assert c.price == pytest.approx(round(bp.REPLACE_ESCALATION_CAP_MULT * ANCHOR, 3))
    assert "absolute escalation cap" in c.bound


def test_re_deriving_from_an_already_escalated_ceiling_cannot_ratchet():
    """The property the anchor's immutability was written to protect. Feed the
    escalated ceiling back in as the market and the answer does not climb: the
    function is of (anchor, market), never of its own history."""
    c1 = _ceil(market_floor=0.40)
    c2 = _ceil(market_floor=0.40)
    assert c1.price == c2.price
    # three consecutive swaps, each seeing the last one's ceiling as the market
    p = _ceil(market_floor=0.40).price
    for _ in range(3):
        p = _ceil(market_floor=p).price
    assert p <= round(bp.REPLACE_ESCALATION_CAP_MULT * ANCHOR, 3)


def test_the_hard_ondemand_line_binds_the_escalation():
    """0.75x on-demand is the hard ceiling on every emitted bid from every path
    (recalibration 2026-08-09 item A); a re-pricing may not step over it."""
    c = _ceil(market_floor=0.60, on_demand=0.60)     # want $0.66, hard line $0.45
    assert c.escalated is True
    assert c.price == pytest.approx(bp.effective_bid_ceiling(0.60))
    assert "hard" in c.bound
    # and when the hard line lands UNDER the base, the base still stands — a
    # cheap machine's on-demand price does not retract an authorised bound.
    assert _ceil(market_floor=0.60, on_demand=0.50).price == BASE


def test_the_budget_bounds_the_escalation_over_the_queues_projected_runtime():
    """A ceiling the remaining budget cannot sustain for the work still queued
    is not an affordable ceiling. $1.00 left over a 10h horizon buys $0.10/hr —
    under the base, so the escalation is REFUSED and the base stands."""
    c = _ceil(market_floor=0.60, budget_left=1.0, horizon_h=10.0)
    assert (c.price, c.escalated) == (BASE, False)
    assert "escalation refused" in c.bound and "budget" in c.bound
    # a budget that CAN sustain it escalates normally
    c = _ceil(market_floor=0.60, budget_left=60.0, horizon_h=10.0)
    assert c.escalated is True and c.price == pytest.approx(0.66)


def test_an_unaffordable_market_stays_a_REFUSAL():
    """Requirement (c). The escalation cap admits $0.773; a market at $2.00/hr
    is still refused, and the refusal still names the arithmetic."""
    c = _ceil(market_floor=2.0, on_demand=4.0, budget_left=60.0)
    assert c.price == pytest.approx(0.773)
    d = _dec(launch_dph_anchor=ANCHOR, offer_min_bid=2.0, offer_ondemand=4.0,
             ceiling=c.price, ceiling_basis="re-derived replacement ceiling")
    assert d.action == "stop"
    assert "over the $0.773 ceiling" in d.reason
    assert "re-derived replacement ceiling" in d.reason


def test_re_derivation_never_LOWERS_a_ceiling_the_operator_authorised():
    """Every bound only ever tightens the ESCALATION; none of them may pull the
    ceiling below `mult x anchor`, which is a figure a human chose."""
    for kw in ({"on_demand": 0.01}, {"budget_left": 0.01, "horizon_h": 100.0},
               {"market_floor": 0.001}, {"p_alt": 0.001, "on_demand": 0.02}):
        assert _ceil(**kw).price >= BASE, kw


def test_no_anchor_is_still_a_refusal_not_a_market_derived_ceiling():
    """`p_alt` stands in as a ceiling anchor inside `rebid_ladder` when nothing
    else exists. It deliberately does NOT here: an autonomous REPLACEMENT with
    no launch price to bound it is the "unknown ceiling is not a licence to
    spend" case, and a market read is not an authorisation."""
    c = _ceil(launch_dph_anchor=None, market_floor=0.40, p_alt=0.53)
    assert c.price is None and "no launch price anchor" in c.bound


def test_a_handed_in_ceiling_is_used_verbatim_and_omitting_it_is_unchanged():
    """The probe and the decision must price against ONE market. Omitting the
    argument reproduces every pre-2026-08-24 caller byte for byte."""
    assert _dec(ceiling=3.0).ceiling == pytest.approx(3.0)
    assert _dec().ceiling == pytest.approx(1.52)     # 2.0 x $0.76, as before
    assert "2x the $0.7600 launch price" in _dec(offer_min_bid=2.0,
                                                 offer_ondemand=4.0).reason


def test_ondemand_displacement_routes_to_the_ondemand_rung():
    """A spot replacement after an on-demand claim buys the same loss again."""
    d = _dec(eviction_class=bp.EVICTION_ONDEMAND)
    assert (d.action, d.rental) == ("rent", "ondemand")
    assert d.price == pytest.approx(1.20)
    assert "outranks any bid" in d.reason


def test_repeated_fast_deaths_flip_the_ladder_to_ondemand():
    d = _dec(fast_deaths=2, max_fast_deaths=2)
    assert (d.action, d.rental) == ("rent", "ondemand")
    assert "died inside" in d.reason


def test_inverted_spread_flips_the_ladder_to_ondemand():
    """The winnable bid is at or above on-demand — paying more for a box that
    an on-demand renter can take from you at any moment is strictly worse."""
    d = _dec(offer_min_bid=1.00, offer_ondemand=1.05)   # 1.2x1.00 clamps to 1.049
    assert (d.action, d.rental) == ("rent", "ondemand")


def test_razor_thin_spot_cushion_routes_to_ondemand():
    """The v7 eviction-1 ROOT CAUSE as a policy test. `min(1.2 x floor,
    on-demand - eps)` on a machine whose on-demand price sits a tenth of a cent
    over its bid floor yields a bid with no cushion at all. Re-renting into that
    market buys the same eviction; the ladder must see the compressed cushion,
    not the nominal 1.2x multiplier.

    **Which rail catches it moved on 2026-08-09** (recalibration item A) and the
    OUTCOME is what this test is about, so the outcome assertion is unchanged. It
    used to be `thin` — a bid computed, then measured against the floor and found
    compressed. It is now the hard ceiling, one step earlier: surviving a $0.746
    floor needs $0.8206 and the ceiling is 0.75 x $0.748 = $0.561, so no spot
    price is computed at all and the on-demand rung is reached by escalation. That
    is the stronger form of the same judgement — the ladder never has to inspect a
    bid it should not have been willing to place.

    `thin` is NOT dead: it still fires where the REPLACEMENT ceiling (2 x the
    launch anchor) is what compressed the cushion, which is a budget bound rather
    than a market one."""
    d = _dec(offer_min_bid=0.746, offer_ondemand=0.748)
    assert (d.action, d.rental) == ("rent", "ondemand")
    assert "escalate_over_ceiling" in d.reason
    # sanity: the SAME floor with real headroom under on-demand stays on spot
    assert _dec(offer_min_bid=0.746, offer_ondemand=2.0).rental == "bid"


def test_the_hard_ceiling_SUBSUMES_the_thin_cushion_rung_selector():
    """What became of `thin` on 2026-08-09, established rather than assumed —
    a dead rail that still reads as live is how a policy quietly loses a rung.

    `thin` still FIRES (it is emitted here, in the reason, on a bid the
    replacement ceiling compressed under the cushion), but it can no longer
    CHOOSE the on-demand rung, because the two conditions are now mutually
    exclusive. Proof, from the rails as shipped:

        no escalation      => cushion <= 0.75 x on_demand   (the hard ceiling)
        `thin`             => spot_price < cushion, and the only rail that can
                              push spot_price under the cushion is the
                              replacement ceiling => repl_ceiling < cushion
        on-demand rung ok  => on_demand <= repl_ceiling

        chain: on_demand <= repl_ceiling < cushion <= 0.75 x on_demand  — false.

    So a `thin` that survives the hard ceiling always finds the on-demand rung
    unaffordable and falls back to spot. Every case `thin` used to route to
    on-demand is now caught one step earlier, by `escalate_over_ceiling`, where
    the on-demand rung is genuinely reachable.

    Floor $0.76 at 29% of a $2.60 on-demand rate — a perfectly ordinary machine —
    with a $0.40 launch anchor, so the $0.80 replacement ceiling clamps the bid
    under the $0.836 cushion."""
    d = _dec(offer_min_bid=0.76, offer_ondemand=2.60, launch_dph_anchor=0.40)
    assert "cushion too thin" in d.reason               # still diagnosed
    assert (d.action, d.rental) == ("rent", "bid")      # but never rung-selecting
    assert "falling back to spot" in d.reason


def test_ondemand_preferred_but_unaffordable_falls_back_to_spot():
    d = _dec(eviction_class=bp.EVICTION_ONDEMAND, offer_min_bid=0.50,
             offer_ondemand=9.99)
    assert (d.action, d.rental) == ("rent", "bid")
    assert "falling back to spot" in d.reason


def test_no_offers_at_all_stops():
    d = _dec(offer_min_bid=None, offer_ondemand=None)
    assert d.action == "stop" and "no affordable replacement" in d.reason


def test_every_decision_carries_its_arithmetic():
    """An autonomous rental decision nobody can reconstruct is not a bounded
    one — every outcome, refusal included, must name numbers."""
    for d in (_dec(), _dec(spend_usd=5.0), _dec(replacements_used=9),
              _dec(offer_min_bid=2.0, offer_ondemand=4.0)):
        assert d.reason and any(ch.isdigit() for ch in d.reason)


# --------------------------------------------------------------------------- #
# 3. the herdd driver: launch -> retarget -> destroy -> re-anchor
# --------------------------------------------------------------------------- #
def _args(**kw):
    base = dict(id=41, dry_run=False, budget=5.0, max_bid=None, handoff=True,
                strict_ceiling=False, keep=False, max_replacements=None,
                replace_ceiling_mult=None, replacement_retention_hours=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _inst(iid=41, status="exited", machine=7, dph=0.76):
    return {"id": iid, "actual_status": status, "machine_id": machine,
            "dph_total": dph, "num_gpus": 2, "gpu_name": "RTX PRO 6000",
            "label": "upstream-monorepo", "start_date": NOW - 600, "is_bid": True}


def _jc(**kw):
    jc, hf = job_lane.job_supervise_init(_args(**kw.pop("args", {})))
    jc["now"] = NOW
    jc["launch_dph_anchor"] = 0.76
    jc["instances"] = [_inst()]
    jc.update(kw)
    return jc, hf


_UNSET = object()


def _wire(monkeypatch, *, spot=_UNSET, od=_UNSET, od_uncapped=_UNSET,
          unfloored=None, launch=(88, 0.9, None),
          retarget=(["j1", "j2"], []), destroy_fail=None, label_fail=None):
    """Stub every seam `_job_eviction_replace` touches; return the call log.

    `od` answers the CEILINGED on-demand probe (`max_dph` set), `od_uncapped`
    the un-ceilinged REFERENCE probe the 2026-08-05 fix added — pass
    `od=None, od_uncapped={...}` to model the night's market: a real on-demand
    book that sits entirely above the ceiling. `od_uncapped` defaults to `od`,
    so a healthy market needs neither.

    The stubbed seam is `_job_replacement_offers` (the CANDIDATE SET, 2026-08-16)
    — `_job_replacement_offer` is left REAL so its `limit=1` delegation, which
    the un-ceilinged reference re-probe still uses, is exercised rather than
    mocked away. A single-offer stub is a candidate set of one, i.e. exactly
    the market shape these tests were written against. `spot`/`od` may be a
    LIST to model a real multi-candidate market.

    `unfloored` answers the `disk_gb=0` SHORTFALL probe (2026-08-16) — the
    market as it looks with the container-disk floor lifted. It only ever runs
    when both rungs came back empty, so a test that does not model an empty
    market never sees it."""
    calls = []
    spot = {"id": 1, "min_bid": 0.50, "dph_total": 1.20} \
        if spot is _UNSET else spot
    od = {"id": 2, "dph_total": 1.20} if od is _UNSET else od
    od_uncapped = od if od_uncapped is _UNSET else od_uncapped

    def _as_list(x):
        if x is None:
            return []
        return list(x) if isinstance(x, list) else [x]

    def _offers(jctx, excl=None, rental="bid", max_dph=None, cuda=None,
                limit=None, disk_gb=None):
        calls.append(("offer", rental, tuple(excl or ()), max_dph, cuda,
                      disk_gb))
        if disk_gb == 0:
            # the unfloored SHORTFALL re-probe (2026-08-16) — never a pick
            return _as_list(unfloored)
        if rental != "ondemand":
            return _as_list(spot)
        return _as_list(od if max_dph is not None else od_uncapped)

    monkeypatch.setattr(replacement, "_job_replacement_offers", _offers)
    monkeypatch.setattr(replacement, "_launch_job_replacement",
        lambda jctx, excl, offer=None, rental="bid", price=None, max_dph=None: (
            calls.append(("launch", rental, price, tuple(excl or ()), max_dph,
                          (offer or {}).get("id"))),
            launch)[1])
    monkeypatch.setattr(replacement, "_retarget_pending_tickets",
        lambda old, new, reason="pull_condemned": (
            calls.append(("retarget", old, new, reason)), retarget)[1])
    monkeypatch.setattr(lifecycle, "_destroy_and_revoke",
        lambda ids, ins, intent, noun="": (
            calls.append(("destroy", list(ids), intent)), destroy_fail or [])[1])
    monkeypatch.setattr(journal, "_job_handoff_emit",
                        lambda jctx, event, **kw: calls.append(("emit", event, kw)))
    # A label PUT is a real mutation of a real box: this seam is stubbed so the
    # portable lane can never reach the vast API (it did, once, during
    # development — a 404 against instance 41, which someone else owns).
    monkeypatch.setattr(lifecycle, "_put_label_soft",
        lambda iid, label: (calls.append(("label", str(iid), label)),
                            (not label_fail, label_fail))[1])
    return calls


def test_replacement_moves_the_queue_before_disposing_of_the_old_box(monkeypatch):
    """ORDER is the safety property, exactly as on the pull-condemn lane: a
    ticket must never point at a box that is already gone (the 46590907 orphan
    shape). Disposal is the RETENTION label by default (2026-08-05) and the
    destroy with retention off — the ordering constraint is the same either
    way."""
    jc, hf = _jc(args={"replacement_retention_hours": 0})
    calls = _wire(monkeypatch)
    assert replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid") is True
    kinds = [c[0] for c in calls]
    assert kinds.index("retarget") < kinds.index("destroy")
    assert ("retarget", "41", 88, "evicted") in calls
    assert [c for c in calls if c[0] == "destroy"][0][1] == ["41"]

    jc, hf = _jc()                                   # default: retain
    calls = _wire(monkeypatch)
    assert replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid") is True
    kinds = [c[0] for c in calls]
    assert kinds.index("retarget") < kinds.index("label")
    assert "destroy" not in kinds


def test_replacement_re_anchors_the_supervisor_on_the_new_box(monkeypatch):
    jc, hf = _jc()
    _wire(monkeypatch)
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    assert jc["iid"] == "88"
    assert jc["replacements"] == 1
    assert jc["last_bid"] == 0.9 and jc["first_seen_dph"] == 0.9
    assert jc["rescue_deadline"] is None and jc["not_live"] == 0
    assert 7 in jc["evicted_machines"]          # never re-rent the lost machine


def test_the_price_ceiling_anchor_never_ratchets(monkeypatch):
    """Three replacements at 2x each would license an 8x box if the ceiling
    re-derived from the replacement's price. It is anchored to the ORIGINAL
    launch and must survive every swap."""
    jc, hf = _jc()
    _wire(monkeypatch, launch=(88, 1.40, None))
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    assert jc["launch_dph_anchor"] == 0.76
    assert jc["first_seen_dph"] == 1.40          # bid ladder DOES track the new box


def test_ondemand_rung_launches_on_demand(monkeypatch):
    jc, hf = _jc()
    calls = _wire(monkeypatch)
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_ONDEMAND, "on-demand claim")
    launch = [c for c in calls if c[0] == "launch"][0]
    assert launch[1] == "ondemand"


def test_offer_search_is_geo_open_bandwidth_and_cuda_gated(monkeypatch):
    """Owner directive 2026-08-05: replacement rentals search GLOBALLY, gated on
    bandwidth (pick_cheapest_offer's inet_down floor, applied inside it) and the
    host CUDA floor — never on geography. `_job_replacement_offer` takes no geo
    argument at all, which is the enforcement."""
    jc, hf = _jc()
    calls = _wire(monkeypatch)
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    offers = [c for c in calls if c[0] == "offer"]
    assert offers, "no offer search happened"
    for _, rental, excl, max_dph, cuda, _disk in offers:
        assert cuda == 12.8                      # the image is cu129 (VLLM_PIN)
        assert max_dph == pytest.approx(1.52)    # ceiling pushed into the query
        assert 7 in excl                         # the lost machine is excluded
    assert {o[1] for o in offers} == {"bid", "ondemand"}
    import inspect
    assert "geo" not in inspect.signature(replacement._job_replacement_offer).parameters


def test_refusal_leaves_the_old_box_and_its_queue_untouched(monkeypatch):
    jc, hf = _jc(args={"budget": None})           # uncapped => always refused
    calls = _wire(monkeypatch)
    assert replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid") is False
    assert not [c for c in calls if c[0] in ("launch", "retarget", "destroy")]
    assert jc["iid"] == "41"
    assert "no budget cap" in jc["replacement_refused"]


def test_a_failed_launch_never_orphans_the_queue(monkeypatch):
    jc, hf = _jc()
    calls = _wire(monkeypatch, launch=(None, None, "no_offer"))
    assert replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid") is False
    kinds = [c[0] for c in calls]
    assert "retarget" not in kinds and "destroy" not in kinds
    assert jc["iid"] == "41" and jc["replacements"] == 0


def test_a_DECISION_that_declines_to_rent_is_not_a_wedge(monkeypatch):
    """SCOPE of the `replacement_wedged` alarm. A decision that refuses already
    escalates — `replacement_refused` -> `unrecoverable` -> `rescue_stalled`,
    which names the bound. Counting it toward the wedge would put a SECOND alarm
    on every budget-exhausted or cap-reached watch, i.e. fire on the normal
    workflow. Only an ATTEMPTED launch that failed is a wedge."""
    for args in ({"budget": None}, {}):
        jc, hf = _jc(args=args)
        if not args:
            jc["replacements"] = bp.MAX_REPLACEMENTS      # cap reached
        _wire(monkeypatch)
        assert replacement._job_eviction_replace(
            jc, hf, bp.EVICTION_OUTBID, "outbid") is False
        assert jc["replacement_refused"], "the refusal is still recorded and loud"
        assert "replacement_refusals" not in jc, args

    # ...while a launch that was ATTEMPTED and failed does count, on both lanes.
    jc, hf = _jc()
    _wire(monkeypatch, launch=(None, None, "over_ceiling"))
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    assert jc["replacement_refusals"] == 1


def test_host_failure_skips_the_destroy_there_is_nothing_to_destroy(monkeypatch):
    """The box already left the listing; the retarget is the whole job."""
    jc, hf = _jc(instances=[])
    calls = _wire(monkeypatch)
    assert replacement._job_eviction_replace(
        jc, hf, bp.EVICTION_HOST_FAILURE, "box gone") is True
    assert "destroy" not in [c[0] for c in calls]
    assert ("retarget", "41", 88, "evicted") in calls


def test_dry_run_decides_but_never_spends(monkeypatch):
    jc, hf = _jc(args={"dry_run": True})
    calls = _wire(monkeypatch)
    assert replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid") is False
    assert not [c for c in calls if c[0] == "launch"]
    assert [c for c in calls if c[0] == "emit"
            and c[1] == "eviction_replacement_decision"]


def test_every_decision_is_journaled_with_its_price_math(monkeypatch):
    jc, hf = _jc()
    calls = _wire(monkeypatch)
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    dec = [c for c in calls if c[0] == "emit"
           and c[1] == "eviction_replacement_decision"]
    assert dec, "an autonomous rental happened with no decision event"
    f = dec[0][2]
    for k in ("action", "rental", "price", "ceiling", "budget_left",
              "budget_usd", "spend_usd", "replacements_used",
              "max_replacements", "launch_dph_anchor", "eviction_class"):
        assert k in f, f"decision event is missing {k}"
    assert [c for c in calls if c[0] == "emit" and c[1] == "eviction_replaced"]


def test_only_spot_replacements_count_as_fast_deaths():
    """An on-demand box that dies early is a host failure, not evidence about
    the spot market — it must not push the ladder further toward on-demand."""
    jc = {"replacement_history": [
        {"rental": "ondemand", "ts": NOW, "died_ts": NOW + 10},
        {"rental": "bid", "ts": NOW, "died_ts": NOW + 10},
        {"rental": "bid", "ts": NOW, "died_ts": NOW + 10_000},   # long-lived
        {"rental": "bid", "ts": NOW, "died_ts": None},           # still alive
    ]}
    assert replacement._job_replacement_fast_deaths(jc, NOW) == 1


def test_knob_precedence_namespace_then_env_then_default(monkeypatch):
    jc = {"a": argparse.Namespace(max_replacements=None)}
    monkeypatch.delenv("JOB_MAX_REPLACEMENTS", raising=False)
    assert replacement._job_replacement_knob(jc, "max_replacements",
                                  bp.MAX_REPLACEMENTS) == bp.MAX_REPLACEMENTS
    monkeypatch.setenv("JOB_MAX_REPLACEMENTS", "1")
    assert replacement._job_replacement_knob(jc, "max_replacements", bp.MAX_REPLACEMENTS) == 1
    jc["a"].max_replacements = 0                  # an explicit 0 is NOT "unset"
    assert replacement._job_replacement_knob(jc, "max_replacements", bp.MAX_REPLACEMENTS) == 0


# --------------------------------------------------------------------------- #
# 4. the job_supervise_tick seam
# --------------------------------------------------------------------------- #
def _tick_env(monkeypatch, inst, *, market=None, on_demand=None, listed=None):
    # MIGRATED (was MIGRATION-BLOCKED, step 6e batch B3): both named blockers
    # landed — `_sticky_on_demand` at `vastlib.market.pricing` and
    # `_serve_self_park_soft` at `vastlib.supervise.replacement` (the jobs tick
    # reaches each by module attribute), so subject and seams move together.
    # Placement follows what `job_lane.job_supervise_tick` RESOLVES, not what
    # owns the name: `lifecycle.<name>` for the instance read and the bid PUT,
    # `pricing.<name>` for the three market reads, `handoff.<name>` for the
    # reconcile, `risk.<name>` for the checkpoint alarm, and bare in `job_lane`
    # for `_box_lifecycle_soft` (still a raising SEAM stub there; its body is at
    # `vastlib.jobs.view`) and `_job_sup_reattach`.
    monkeypatch.setattr(lifecycle, "_instances_soft", lambda: ([inst] if inst else []))
    monkeypatch.setattr(handoff, "_job_handoff_reconcile", lambda jc, hf: None)
    monkeypatch.setattr(job_lane, "_box_lifecycle_soft",
                        lambda iid: {"parked": False, "drained_pending": False})
    monkeypatch.setattr(jobmeta, "list_queue", lambda iid: ["j1"])
    monkeypatch.setattr(jobmeta, "read_job",
                        lambda j, **kw: {"job_id": j, "display_status": "running",
                                         "status": "running"})
    monkeypatch.setattr(pricing, "_market_min_bid_soft", lambda m, n: market)
    monkeypatch.setattr(pricing, "_market_min_bid_read",
                        lambda m, n=None: models.MarketRead(listed is not None,
                                                            bool(listed), market))
    monkeypatch.setattr(pricing, "_market_ondemand_soft", lambda m, n: on_demand)
    monkeypatch.setattr(lifecycle, "_put_bid_soft", lambda iid, p: (True, None))
    monkeypatch.setattr(job_lane, "_job_sup_reattach", lambda jc, iid: None)
    monkeypatch.setattr(jobs_risk, "_ckpt_watchdog_alarm", lambda vw, now: None)


def test_tick_calls_the_replacement_before_returning_unrecoverable(monkeypatch):
    """The whole point: an evicted jobs box whose bid ladder is out of moves now
    gets a replacement instead of a retarget checklist."""
    seen = {}

    def _replace(jc, hf, ecls, why, exclusion_class=None):
        seen["class"], seen["why"] = ecls, why
        seen["exclusion_class"] = exclusion_class
        jc["iid"] = "999"
        return True

    monkeypatch.setattr(replacement, "_job_eviction_replace", _replace)
    # floor above on-demand => `_bid_action` returns None (unwinnable), which is
    # the v7 eviction-1 shape: the ladder never even bids.
    _tick_env(monkeypatch, _inst(status="exited"),
              market=V7_E1["floor_after"], on_demand=V7_E1["on_demand"])
    jc, hf = job_lane.job_supervise_init(_args())
    verdict = None
    for _ in range(2 * bp.NOT_LIVE_DEBOUNCE + 1):
        verdict = job_lane.job_supervise_tick(jc, hf)
        if jc["iid"] == "999":
            break
    assert verdict is None, "a successful replacement must keep supervising"
    assert jc["iid"] == "999"
    assert seen["class"] == bp.EVICTION_OUTBID


def test_tick_falls_through_to_unrecoverable_when_replacement_refuses(monkeypatch):
    """The manual path is never WORSE than before — only no longer first."""
    monkeypatch.setattr(replacement, "_job_eviction_replace",
                        lambda jc, hf, ecls, why, exclusion_class=None: False)
    _tick_env(monkeypatch, _inst(status="exited"),
              market=V7_E1["floor_after"], on_demand=V7_E1["on_demand"])
    jc, hf = job_lane.job_supervise_init(_args())
    verdicts = [job_lane.job_supervise_tick(jc, hf)
                for _ in range(2 * bp.NOT_LIVE_DEBOUNCE + 1)]
    assert "unrecoverable" in verdicts


def test_a_raising_replacement_never_kills_the_babysitter(monkeypatch, capsys):
    def _boom(jc, hf, ecls, why, exclusion_class=None):
        raise RuntimeError("vast API on fire")

    monkeypatch.setattr(replacement, "_job_eviction_replace", _boom)
    _tick_env(monkeypatch, _inst(status="exited"),
              market=V7_E1["floor_after"], on_demand=V7_E1["on_demand"])
    jc, hf = job_lane.job_supervise_init(_args())
    verdicts = [job_lane.job_supervise_tick(jc, hf)
                for _ in range(2 * bp.NOT_LIVE_DEBOUNCE + 1)]
    assert "unrecoverable" in verdicts
    assert "vast API on fire" in capsys.readouterr().out


def test_serve_mode_is_excluded_from_the_jobs_replacement(monkeypatch):
    """A serve box has no queue to retarget; its replacement is
    launch_serve.sh's own SLA relaunch spec, which already exists."""
    called = []
    monkeypatch.setattr(replacement, "_job_eviction_replace",
                        lambda *a, **k: called.append(1) or True)
    _tick_env(monkeypatch, _inst(status="exited"),
              market=V7_E1["floor_after"], on_demand=V7_E1["on_demand"])
    jc, hf = job_lane.job_supervise_init(_args(serve_mode=True))
    verdicts = [job_lane.job_supervise_tick(jc, hf)
                for _ in range(2 * bp.NOT_LIVE_DEBOUNCE + 1)]
    assert "unrecoverable" in verdicts and not called


def test_launch_price_anchor_is_captured_once_from_the_original_box(monkeypatch):
    _tick_env(monkeypatch, _inst(status="running", dph=0.76), market=0.5,
              on_demand=1.2)
    jc, hf = job_lane.job_supervise_init(_args())
    job_lane.job_supervise_tick(jc, hf)
    assert jc["launch_dph_anchor"] == 0.76
    monkeypatch.setattr(lifecycle, "_instances_soft",
                        lambda: [_inst(status="running", dph=2.20)])
    job_lane.job_supervise_tick(jc, hf)
    assert jc["launch_dph_anchor"] == 0.76, "the anchor must never be rewritten"


# --------------------------------------------------------------------------- #
# 5. fleetd: the counters are spend bounds, so they are DURABLE
# --------------------------------------------------------------------------- #
def test_replacement_state_round_trips_through_a_restart():
    """A restart that reset `replacements` would hand the ladder a fresh budget
    of autonomous rentals; one that reset `launch_dph_anchor` would re-derive
    the ceiling from the replacement's price and ratchet it."""
    jc = {"replacements": 2, "launch_dph_anchor": 0.76,
          "evicted_machines": {7, 9},
          "replacement_history": [{"iid": "88", "rental": "bid"}]}
    w = {}
    fleetd._replacement_state_persist(jc, w)
    assert w["replacement"]["evicted_machines"] == [7, 9]   # JSON-safe

    fresh = {"replacements": 0, "launch_dph_anchor": None,
             "evicted_machines": set(), "replacement_history": []}
    fleetd._replacement_state_restore(fresh, w)
    assert fresh["replacements"] == 2
    assert fresh["launch_dph_anchor"] == 0.76
    assert fresh["evicted_machines"] == {7, 9}
    assert fresh["replacement_history"] == [{"iid": "88", "rental": "bid"}]


def test_a_watch_that_predates_the_feature_restores_cleanly():
    fresh = {"replacements": 0, "launch_dph_anchor": None,
             "evicted_machines": set()}
    fleetd._replacement_state_restore(fresh, {"profile": "jobs"})
    assert fresh["replacements"] == 0 and fresh["launch_dph_anchor"] is None


def test_jobs_policy_defaults_leave_the_knobs_unset_not_zero():
    """`None` means "this watch did not choose", which the ladder reads as the
    bidpolicy default. A 0 default would silently disable replacement."""
    assert fleetd.JOBS_POLICY_DEFAULTS["max_replacements"] is None
    assert fleetd.JOBS_POLICY_DEFAULTS["replace_ceiling_mult"] is None
    a = fleetd.make_policy("jobs", {}, "41", budget_usd=5.0)
    assert a.max_replacements is None and a.budget == 5.0


# --------------------------------------------------------------------------- #
# 6. RETENTION of the lost box (owner directive 2026-08-05)
#
#   "don't have fleetd destroy the box immediately please. we should eat a few
#    hours of parked host time just in case we have bugs that lose data."
#
# The lost box's disk can hold state that never reached B2 (checkpoint sync is
# periodic), so the replacement flow HOLDS it for a bounded window instead of
# destroying it. Two properties carry the whole feature and both are tested
# against the real parsers rather than a copy of them: the keep label must be
# honored by `herdd reap`, and the window must actually END.
# --------------------------------------------------------------------------- #
def _retention_rec(jc):
    return (jc.get("retained_boxes") or [{}])[-1]


def test_retention_is_the_default_the_lost_box_is_held_not_destroyed(monkeypatch):
    jc, hf = _jc()
    calls = _wire(monkeypatch)
    assert replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid") is True
    assert "destroy" not in [c[0] for c in calls]
    rec = _retention_rec(jc)
    assert rec["status"] == "retained" and rec["iid"] == "41"
    assert rec["replacement_iid"] == "88"
    assert rec["deadline_ts"] == pytest.approx(
        NOW + bp.REPLACEMENT_RETENTION_H * 3600.0)


def test_the_keep_label_is_the_one_reap_actually_parses(monkeypatch):
    """The retention window is a LIE unless `herdd reap` honors the label:
    reap destroys any stopped box idle past 2h without a `keep` token, so a
    3h window would silently lose the box at 2h. Asserted against `_reap_kept`
    itself — reap's own parser — never a re-implementation of it."""
    jc, hf = _jc()
    calls = _wire(monkeypatch)
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    label = [c for c in calls if c[0] == "label"][0][2]
    assert labels._reap_kept(label, now=NOW) is True
    assert labels._reap_kept(label, now=NOW + 60) is True
    # ... and the SAME parser stops keeping it once the window closes, which is
    # what makes the reaper the expiry mechanism.
    assert labels._reap_kept(label, now=NOW + 3 * 3600 + 1) is False
    # the pre-existing label is preserved (a group is APPENDED, never fused):
    # `_label_value` still has to read `run:<RID>` for the B2 key revocation.
    assert label.startswith("upstream-monorepo ")
    assert labels._keep_retention_info(label, now=NOW)["reason"] == "evicted-outbid"


def test_retention_hours_zero_restores_the_immediate_destroy(monkeypatch):
    jc, hf = _jc(args={"replacement_retention_hours": 0})
    calls = _wire(monkeypatch)
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    assert [c for c in calls if c[0] == "destroy"][0][1] == ["41"]
    assert "label" not in [c[0] for c in calls]
    assert _retention_rec(jc)["status"] == "destroyed"


def test_retention_hours_knob_reads_env_then_default(monkeypatch):
    jc = {"a": argparse.Namespace(replacement_retention_hours=None)}
    monkeypatch.delenv("JOB_REPLACEMENT_RETENTION_HOURS", raising=False)
    assert replacement._job_replacement_knob(jc, "replacement_retention_hours",
                                   bp.REPLACEMENT_RETENTION_H) == 3.0
    monkeypatch.setenv("JOB_REPLACEMENT_RETENTION_HOURS", "0.5")
    assert replacement._job_replacement_knob(jc, "replacement_retention_hours",
                                   bp.REPLACEMENT_RETENTION_H) == 0.5
    jc["a"].replacement_retention_hours = 0        # an explicit 0 is NOT "unset"
    assert replacement._job_replacement_knob(jc, "replacement_retention_hours",
                                   bp.REPLACEMENT_RETENTION_H) == 0


def test_retention_is_journaled_with_its_cost_and_deadline(monkeypatch):
    """Cost disclosure: a box nobody chose to rent must never be a surprise."""
    jc, hf = _jc()
    calls = _wire(monkeypatch)
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    ev = [c for c in calls if c[0] == "emit" and c[1] == "eviction_box_retained"]
    assert ev, "a box was retained with no journal event"
    f = ev[0][2]
    for k in ("box", "deadline", "retention_h", "est_cost_usd",
              "est_cost_hi_usd", "keep_labeled", "eviction_class"):
        assert k in f, f"retention event is missing {k}"
    assert f["deadline"].endswith("Z")
    assert 0 < f["est_cost_usd"] <= f["est_cost_hi_usd"] < 1.0   # ~3h of disk


def test_a_label_put_failure_keeps_the_box_and_says_the_window_is_unsafe(
        monkeypatch, capsys):
    """Failing to defend the box is not a reason to destroy it — the owner
    asked for the disk. But the operator must be told the reaper may take it at
    2h, because the label is the only thing that would have stopped it."""
    jc, hf = _jc()
    calls = _wire(monkeypatch, label_fail="HTTP 500")
    assert replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid") is True
    assert "destroy" not in [c[0] for c in calls]
    rec = _retention_rec(jc)
    assert rec["status"] == "retained" and rec["keep_labeled"] is False
    assert "2h idle mark" in capsys.readouterr().out


def test_a_404_on_the_lost_box_never_fails_the_replacement(monkeypatch, capsys):
    """Retention is BEST-EFFORT on an evicted spot box (box 44612403: a parked
    bid instance answered 404 `no_such_instance` minutes later). The
    replacement is the important half and must land regardless."""
    jc, hf = _jc()
    calls = _wire(monkeypatch)
    monkeypatch.setattr(retention, "_job_retain_or_destroy",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("404 no_such_instance")))
    assert replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid") is True
    assert jc["iid"] == "88" and jc["replacements"] == 1
    assert ("retarget", "41", 88, "evicted") in calls
    assert "404 no_such_instance" in capsys.readouterr().out


def test_a_box_already_out_of_the_listing_is_already_gone_not_retained(monkeypatch):
    """Host failure / spot reclaim: nothing to retain, nothing to destroy — and
    it is journaled as its OWN class, because folding it into `retained` would
    overstate how often the disk was actually available to salvage."""
    jc, hf = _jc(instances=[])
    calls = _wire(monkeypatch)
    assert replacement._job_eviction_replace(
        jc, hf, bp.EVICTION_HOST_FAILURE, "box gone") is True
    assert _retention_rec(jc)["status"] == "already_gone"
    assert [c for c in calls if c[0] == "emit"
            and c[1] == "eviction_box_already_gone"]
    assert "destroy" not in [c[0] for c in calls]


# --- the sweep: every retained box reaches a terminal outcome ---------------- #
def _swept(monkeypatch, rec, *, instances, now, **knobs):
    calls = []
    monkeypatch.setattr(lifecycle, "_destroy_and_revoke",
        lambda ids, ins, intent, noun="": (
            calls.append(("destroy", list(ids), intent)), [])[1])
    monkeypatch.setattr(journal, "_job_handoff_emit",
                        lambda jctx, event, **kw: calls.append(("emit", event, kw)))
    jc = {"a": argparse.Namespace(**dict({"replacement_retention_hours": None,
                                          "retention_backstop_hours": None},
                                         **knobs)),
          "retained_boxes": [rec], "instances": instances}
    retention._job_retention_sweep(jc, now)
    return jc, calls


def _rec(deadline_ts, status="retained"):
    return {"iid": "41", "status": status, "class": bp.EVICTION_OUTBID,
            "retained_ts": NOW, "deadline_ts": deadline_ts, "retention_h": 3.0,
            "cost_usd": 0.27, "cost_hi_usd": 0.58, "keep_labeled": True}


def test_sweep_leaves_an_open_window_alone(monkeypatch):
    jc, calls = _swept(monkeypatch, _rec(NOW + 3 * 3600),
                       instances=[_inst()], now=NOW + 600)
    assert jc["retained_boxes"][0]["status"] == "retained"
    assert not calls


def test_sweep_marks_the_window_expired_and_leaves_reap_to_reclaim(monkeypatch):
    """Expiry does NOT need fleetd: the label stopped keeping the box, and
    `herdd reap`'s 15-minute timer owns idle-storage reclamation. The sweep
    only records it."""
    jc, calls = _swept(monkeypatch, _rec(NOW + 3 * 3600),
                       instances=[_inst()], now=NOW + 3 * 3600 + 60)
    assert jc["retained_boxes"][0]["status"] == "expired"
    assert "destroy" not in [c[0] for c in calls]
    assert [c for c in calls if c[0] == "emit"
            and c[1] == "eviction_retention_expired"]


def test_the_backstop_destroys_a_box_the_reaper_never_took(monkeypatch):
    """A retention that never expires recreates the orphaned-billing problem.
    Past deadline + grace with the box STILL listed, the ladder finishes it —
    the case where the reap timer is simply not installed."""
    dl = NOW + 3 * 3600
    jc, calls = _swept(monkeypatch, _rec(dl, status="expired"),
                       instances=[_inst()],
                       now=dl + bp.RETENTION_BACKSTOP_GRACE_H * 3600 + 1)
    assert [c for c in calls if c[0] == "destroy"][0][1] == ["41"]
    assert jc["retained_boxes"][0]["status"] == "destroyed"


def test_the_backstop_never_kills_a_box_someone_resumed_to_salvage(monkeypatch):
    """A LIVE retained box is a human mid-salvage; destroying it is exactly the
    data loss retention exists to prevent."""
    dl = NOW + 3 * 3600
    jc, calls = _swept(monkeypatch, _rec(dl, status="expired"),
                       instances=[_inst(status="running")],
                       now=dl + 10 * 3600)
    assert "destroy" not in [c[0] for c in calls]


def test_a_box_that_vanishes_before_its_deadline_is_retention_lost(monkeypatch):
    """The measured failure rate of the retention promise on SPOT boxes. It is
    never folded into the clean outcome — a host that reclaims the stopped bid
    instance took the disk with it."""
    jc, calls = _swept(monkeypatch, _rec(NOW + 3 * 3600), instances=[],
                       now=NOW + 600)
    assert jc["retained_boxes"][0]["status"] == "retention_lost"
    ev = [c for c in calls if c[0] == "emit"
          and c[1] == "eviction_retention_ended"][0][2]
    assert ev["outcome"] == "retention_lost"


def test_a_box_gone_after_its_deadline_is_the_designed_outcome(monkeypatch):
    jc, calls = _swept(monkeypatch, _rec(NOW + 3 * 3600), instances=[],
                       now=NOW + 4 * 3600)
    assert jc["retained_boxes"][0]["status"] == "reaped"


def test_the_backstop_grace_is_configurable(monkeypatch):
    dl = NOW + 3 * 3600
    jc, calls = _swept(monkeypatch, _rec(dl, status="expired"),
                       instances=[_inst()], now=dl + 600,
                       retention_backstop_hours=0.05)      # 3 min
    assert [c for c in calls if c[0] == "destroy"]


# --- durability + surfacing -------------------------------------------------- #
def test_retention_deadlines_survive_a_daemon_restart():
    """A deadline a restart forgot is a box nobody follows to an outcome. (The
    keep label on the box still self-expires — that is deliberately independent
    of this file — but the backstop and `fleet status` read from here.)"""
    jc = {"replacements": 1, "launch_dph_anchor": 0.76, "evicted_machines": {7},
          "replacement_history": [], "retained_boxes": [_rec(NOW + 3 * 3600)]}
    w = {}
    fleetd._replacement_state_persist(jc, w)
    fresh = {"replacements": 0, "evicted_machines": set()}
    fleetd._replacement_state_restore(fresh, w)
    assert fresh["retained_boxes"][0]["deadline_ts"] == NOW + 3 * 3600
    assert fresh["retained_boxes"][0]["status"] == "retained"


def test_fleet_status_surfaces_retained_boxes_but_not_finished_ones():
    """Nobody chose to rent these, so they are printed unconditionally."""
    state = {"watches": {"41": {"replacement": {"retained_boxes": [
        _rec(NOW + 3 * 3600),
        dict(_rec(NOW), iid="42", status="reaped")]}}}}
    rows = fleetd.retention_rows(state, NOW + 600)
    assert [r["iid"] for r in rows] == ["41"]
    assert rows[0]["left_s"] == pytest.approx(3 * 3600 - 600)
    assert rows[0]["est_cost_usd"] == 0.27


def test_retention_policy_default_is_unset_not_zero():
    assert fleetd.JOBS_POLICY_DEFAULTS["replacement_retention_hours"] is None
    a = fleetd.make_policy("jobs", {}, "41", budget_usd=5.0)
    assert a.replacement_retention_hours is None
    a = fleetd.make_policy("jobs", {"replacement_retention_hours": 0}, "41",
                           budget_usd=5.0)
    assert a.replacement_retention_hours == 0


def test_retention_plan_cost_uses_the_boxs_own_storage_rate_when_known():
    """The measured $2.13-$4.62/day fleet range is a FALLBACK; a real per-box
    `storage_total_cost` always wins, and then the estimate is one number."""
    p = bp.retention_plan(retention_h=3, present=True, now=0,
                          storage_day_usd=4.0)
    assert p.cost_usd == p.cost_hi_usd == pytest.approx(0.5)
    p = bp.retention_plan(retention_h=3, present=True, now=0)
    assert p.cost_usd == pytest.approx(2.13 * 3 / 24, abs=1e-4)
    assert p.cost_hi_usd == pytest.approx(4.62 * 3 / 24, abs=1e-4)
    # the owner-quoted headline: a 3h window costs cents, not dollars
    assert 0.26 <= p.cost_usd <= 0.28 and 0.57 <= p.cost_hi_usd <= 0.59


# --------------------------------------------------------------------------- #
# 6. THE 2026-08-05 ON-DEMAND ESCALATION (doc 50)
#
# fleetd escalated a q6 training watch to an on-demand box billing $3.4741/hr —
# 1.6x over its own $2.164 ceiling — and journaled it as `ondemand @ $None/hr`.
# Root cause was a single `or` in the caller:
#
#     offer_ondemand=_num_dph((od_offer or {}).get("dph_total")
#                             or (spot_offer or {}).get("dph_total"))
#
# On a BID-type vast offer `dph_total` is the current interruptible price
# (~min_bid + 0.5%), not the machine's on-demand rate. So when every real
# on-demand offer sat above the ceiling and the ceilinged probe returned None,
# the ladder priced its on-demand reference at the SPOT price: the bid clamped
# to $1.602 against a $1.600 floor, the cushion read 1.001x, `thin` fired, and
# the expensive rung was taken on arithmetic describing no rentable offer.
#
# The numbers below are the decision record's own
# (b2:example-runs-bucket/jobs/nodes/46914272/events/, 20:42:04Z) plus the
# same-machine cross-market read from doc 50 §3.
# --------------------------------------------------------------------------- #
Q6 = dict(anchor=1.081888888888889,      # launch_dph_anchor, journaled
          ceiling=2.164,                 # 2.0x, journaled
          floor=1.5999999999999999,      # cheapest eligible bid floor (m81589)
          spot_dph=1.6029629629629630,   # the BID offer's dph_total = the bug's
                                         # fake "on-demand"
          true_od=2.670,                 # m81589's REAL on-demand rate
          paid=3.4740740740740743,       # what the escalation actually bought
          budget=12.0, spend=0.0203)


def _q6(**kw):
    base = dict(eviction_class=bp.EVICTION_UNKNOWN, replacements_used=0,
                budget_usd=Q6["budget"], spend_usd=Q6["spend"],
                launch_dph_anchor=Q6["anchor"], offer_min_bid=Q6["floor"],
                offer_ondemand=Q6["true_od"], fast_deaths=0)
    base.update(kw)
    return bp.replacement_decision(**base)


def test_q6_with_the_true_ondemand_price_stays_on_spot():
    """THE regression. Fed the real on-demand number for the machine it was
    pricing against, the same ladder that escalated bids $1.92 on spot with a
    healthy 1.20x cushion and never reaches the on-demand rung."""
    d = _q6()
    assert (d.action, d.rental) == ("rent", "bid")
    assert d.price == pytest.approx(1.76)              # 1.10 x $1.60 survival cushion
    assert d.price / Q6["floor"] >= bp.REPLACE_MIN_CUSHION
    assert d.ceiling == pytest.approx(Q6["ceiling"])
    assert "on-demand rung" not in d.reason


def test_q6_the_arithmetic_was_right_and_the_INPUT_was_the_bug():
    """Documents the defect rather than the fix, so nobody re-introduces the
    fallback thinking the predicate was at fault. Hand `replacement_decision`
    the SPOT offer's dph_total as if it were an on-demand quote and it
    faithfully reproduces the night: a "market" whose on-demand rate sits a tenth
    of a cent over its own floor, and the on-demand rung taken on that arithmetic.
    The pure function is behaving correctly; the caller lied to it. `bidpolicy`
    may not defend against this (a price is a price), which is why the ONLY
    defense is the caller's probe — see
    `test_the_ondemand_reference_is_never_sourced_from_the_spot_offer`.

    Which rail catches it moved 2026-08-09 (`thin` -> the hard ceiling; see
    `test_razor_thin_spot_cushion_routes_to_ondemand`). The rung and the price are
    unchanged, and they are what this test is about."""
    d = _q6(offer_ondemand=Q6["spot_dph"])
    assert (d.action, d.rental) == ("rent", "ondemand")
    assert "escalate_over_ceiling" in d.reason
    assert d.price == pytest.approx(Q6["spot_dph"])    # priced off a bid offer


def test_q6_the_real_ondemand_book_is_refused_on_the_ceiling():
    """R2's payoff: with the TRUE cheapest on-demand price in hand — $3.4741,
    the rate the escalation actually paid — the on-demand rung is refused by the
    ceiling rail, out loud and with the arithmetic, instead of being bought.
    Spot is taken instead; had spot also been unavailable the ladder stops."""
    d = _q6(eviction_class=bp.EVICTION_ONDEMAND, offer_ondemand=Q6["paid"])
    assert (d.action, d.rental) == ("rent", "bid")
    assert "over the $2.164 ceiling" in d.reason and "falling back to spot" in d.reason
    d = _q6(eviction_class=bp.EVICTION_ONDEMAND, offer_ondemand=Q6["paid"],
            offer_min_bid=None)
    assert d.action == "stop" and "over the $2.164 ceiling" in d.reason


# --- unknown on-demand: a first-class state, never a guess ------------------- #
def test_unknown_ondemand_does_not_clamp_the_bid():
    """No on-demand quote => no clamp => the spot target keeps its full
    BID_TARGET_MULT cushion. (With the old fallback this case could not even
    arise: `offer_ondemand` was never None, it was the spot price.)"""
    d = _q6(offer_ondemand=None)
    assert (d.action, d.rental) == ("rent", "bid")
    assert d.price == pytest.approx(1.92)
    assert d.price / Q6["floor"] >= bp.REPLACE_MIN_CUSHION


def test_unknown_ondemand_can_never_make_a_cushion_thin():
    """`thin` names exactly one mechanism — the on-demand clamp compressing the
    1.20x target onto the floor. With no on-demand price there is no clamp, so
    the branch must be structurally unreachable, not merely unlikely."""
    for floor in (0.05, 0.746, 1.5999999999999999, 9.0):
        d = bp.replacement_decision(
            eviction_class=bp.EVICTION_UNKNOWN, replacements_used=0,
            budget_usd=100.0, spend_usd=0.0, launch_dph_anchor=floor,
            offer_min_bid=floor, offer_ondemand=None, ceiling_mult=4.0)
        assert d.rental == "bid", f"floor {floor} escalated with no OD price"
        assert "cushion too thin" not in d.reason


def test_unknown_ondemand_never_rents_the_ondemand_rung():
    """Even when a REAL signal prefers on-demand (the eviction was an on-demand
    claim, or two spot replacements died fast), an unknown price cannot be
    rented. The ladder falls back to spot and says why; with no spot either it
    stops, naming the missing quote."""
    for kw in ({"eviction_class": bp.EVICTION_ONDEMAND},
               {"fast_deaths": 2}):
        d = _q6(offer_ondemand=None, **kw)
        assert (d.action, d.rental) == ("rent", "bid")
        assert "no on-demand price known" in d.reason
        d = _q6(offer_ondemand=None, offer_min_bid=None, **kw)
        assert d.action == "stop" and "no on-demand price known" in d.reason


def test_unknown_ondemand_is_never_inverted():
    """`inverted` compares two prices; one of them is missing."""
    d = _q6(offer_ondemand=None, offer_min_bid=99.0, ceiling_mult=1000.0)
    assert "INVERTED" not in d.reason


# --- the caller: where the price comes from --------------------------------- #
def test_the_ondemand_reference_is_never_sourced_from_the_spot_offer(monkeypatch):
    """R1 at the seam that broke. The market is exactly the night's: a spot
    offer at floor $1.60 (whose own `dph_total` is $1.6030), NO on-demand offer
    under the $2.164 ceiling, and a real on-demand book starting at $2.670.
    The decision must price against $2.670 and rent SPOT at $1.92.

    2026-08-16: the SURVIVAL rail now prices against the candidate's OWN
    machine on-demand rate (`_market_ondemand_soft`), so the fixture states it
    explicitly instead of leaning on the conftest constant — here the machine
    lists at the same $2.670 the book starts at, which is the night's shape and
    keeps the expected $1.76 (the 1.1x cushion) exactly as it was. The poison
    value the defect substituted, the BID offer's own $1.6030 `dph_total`, is
    asserted absent below."""
    jc, hf = _jc()
    jc["launch_dph_anchor"] = Q6["anchor"]
    jc["a"].budget = Q6["budget"]
    jc["spend_usd"] = Q6["spend"]
    monkeypatch.setattr(pricing, "_market_ondemand_soft",
                        lambda mid, n=None: Q6["true_od"])
    calls = _wire(monkeypatch,
                  spot={"id": 1, "min_bid": Q6["floor"],
                        "dph_total": Q6["spot_dph"]},
                  od=None,                                  # nothing under the ceiling
                  od_uncapped={"id": 9, "dph_total": Q6["true_od"]},
                  launch=(88, 1.92, None))
    assert replacement._job_eviction_replace(jc, hf, bp.EVICTION_UNKNOWN, "outbid") is True
    launch = [c for c in calls if c[0] == "launch"][0]
    assert launch[1] == "bid" and launch[2] == pytest.approx(1.76)
    dec = [c for c in calls if c[0] == "emit"
           and c[1] == "eviction_replacement_decision"][0][2]
    assert dec["rental"] == "bid" and dec["price"] == pytest.approx(1.76)
    # the two market reads, journaled SEPARATELY: an auditor must be able to
    # see that the on-demand book was read AND that none of it was affordable.
    assert dec["offer_ondemand"] == pytest.approx(Q6["true_od"])
    assert dec["ondemand_under_ceiling"] is False
    assert dec["offer_min_bid"] == pytest.approx(Q6["floor"])
    # THE defect, restated at the seam it moved to: neither reference may ever
    # be the bid offer's own interruptible `dph_total`.
    assert dec["spot_ondemand"] != pytest.approx(Q6["spot_dph"])
    assert dec["offer_ondemand"] != pytest.approx(Q6["spot_dph"])


def test_no_ondemand_offer_under_the_ceiling_triggers_an_uncapped_reprobe(monkeypatch):
    """R2: "nothing affordable" and "no market" must be distinguishable, so the
    on-demand market is re-read WITHOUT the ceiling when the ceilinged probe
    comes back empty — and that second probe still carries the CUDA floor."""
    jc, hf = _jc()
    calls = _wire(monkeypatch, od=None,
                  od_uncapped={"id": 9, "dph_total": 5.0})
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    od_probes = [c for c in calls if c[0] == "offer" and c[1] == "ondemand"]
    assert len(od_probes) == 2, "the un-ceilinged reference probe did not run"
    assert od_probes[0][3] == pytest.approx(1.52)       # ceilinged
    assert od_probes[1][3] is None                      # reference: no ceiling
    assert all(p[4] == 12.8 for p in od_probes)         # CUDA floor on both


def test_a_healthy_ondemand_market_is_probed_once(monkeypatch):
    """The reference re-probe is a fallback, not a second API call per tick."""
    jc, hf = _jc()
    calls = _wire(monkeypatch)
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    assert len([c for c in calls if c[0] == "offer" and c[1] == "ondemand"]) == 1


def test_the_launch_path_is_handed_the_ceiling(monkeypatch):
    """R3: the price rail must reach the launcher. A ceiling enforced only
    inside the pure decision function is not a rail — it is a comment."""
    jc, hf = _jc()
    calls = _wire(monkeypatch)
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    launch = [c for c in calls if c[0] == "launch"][0]
    assert launch[4] == pytest.approx(1.52)             # max_dph == the ceiling


def test_the_replacement_record_carries_the_realized_price(monkeypatch):
    """R4: `ondemand @ $None/hr` is not a spend record. The realized rate lands
    in the journal event, the replacement history and the bid ladder's anchor."""
    jc, hf = _jc()
    calls = _wire(monkeypatch, launch=(88, 3.4741, None))
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_ONDEMAND, "on-demand claim")
    ev = [c for c in calls if c[0] == "emit" and c[1] == "eviction_replaced"][0][2]
    assert ev["dph"] == pytest.approx(3.4741)
    assert ev["decided_price"] is not None and ev["ceiling"] == pytest.approx(1.52)
    assert jc["replacement_history"][-1]["dph"] == pytest.approx(3.4741)
    # ...but the BID-ladder anchors stay empty on the on-demand rung: an
    # on-demand box has no standing bid to defend, decay or classify against.
    assert jc["last_bid"] is None and jc["first_seen_dph"] is None
    # on the spot rung the same realized number IS the ladder's anchor
    jc, hf = _jc()
    _wire(monkeypatch, launch=(88, 0.61, None))
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    assert jc["last_bid"] == pytest.approx(0.61)
    assert jc["first_seen_dph"] == pytest.approx(0.61)


def test_a_realized_price_over_the_decision_price_alarms(monkeypatch, capsys):
    """The market can move between decide and launch. That is not a silent
    field: it gets its own loud event, because $1.60 decided / $3.47 realized is
    exactly the shape nobody noticed on 2026-08-05."""
    jc, hf = _jc()
    calls = _wire(monkeypatch, launch=(88, 3.4741, None))
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    assert "OVERPRICED" in capsys.readouterr().out
    over = [c for c in calls if c[0] == "emit"
            and c[1] == "eviction_replacement_overpriced"]
    assert over and over[0][2]["realized_dph"] == pytest.approx(3.4741)
    # ... and a replacement that lands at or under its decided price does not
    jc, hf = _jc()
    calls = _wire(monkeypatch, launch=(88, 0.60, None))
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    assert not [c for c in calls if c[0] == "emit"
                and c[1] == "eviction_replacement_overpriced"]


# --- the launcher: the ceiling binds on the offer it actually rents ---------- #
def _launch_env(monkeypatch, *, offer_pick=None, launched=(88, 5, None),
                ondemand=None):
    """Stub `_launch_job_replacement`'s I/O: the offer search, the image-pin
    probe and `_do_launch` itself. Returns (calls, ns_holder)."""
    calls = []

    def _pick(jctx, excl=None, rental="bid", max_dph=None, cuda=None):
        calls.append(("pick", rental, max_dph, cuda))
        return offer_pick

    def _launch(ns):
        calls.append(("do_launch", ns))
        return launched

    monkeypatch.setattr(replacement, "_job_replacement_offer", _pick)
    monkeypatch.setattr(imageref, "image_tag_digest", lambda img: None)
    # the on-demand clamp reference is a real market read; the portable lane
    # never touches the API, and `None` (unknown) is the honest default here.
    monkeypatch.setattr(pricing, "_market_ondemand_soft",
                        lambda mid, ngpu=None: (calls.append(("od_probe", mid)),
                                                ondemand)[1])
    monkeypatch.setattr(launch, "_do_launch", _launch)
    return calls


def test_launcher_refuses_an_ondemand_offer_over_the_ceiling(monkeypatch, capsys):
    """The night's second defect, as a test: the box that was actually rented.
    A $3.4741/hr offer against a $2.164 ceiling must REFUSE — no API call, no
    box, a named reason on the context."""
    jc, _hf = _jc()
    calls = _launch_env(monkeypatch)
    cid, dph, reason = replacement._launch_job_replacement(
        jc, [7], offer={"id": 5, "dph_total": Q6["paid"]}, rental="ondemand",
        price=None, max_dph=Q6["ceiling"])
    assert (cid, dph, reason) == (None, None, "over_ceiling")
    assert not [c for c in calls if c[0] == "do_launch"], "it rented anyway"
    assert "over the $2.164" in jc["last_error"]
    assert "REFUSED" in capsys.readouterr().out


def test_launcher_refuses_a_spot_bid_over_the_ceiling(monkeypatch):
    """The spot rung is not exempt: the offer search filters on `min_bid`, so
    1.20x a floor just under the ceiling lands over it."""
    jc, _hf = _jc()
    _launch_env(monkeypatch)
    cid, _dph, reason = replacement._launch_job_replacement(
        jc, [7], offer={"id": 5, "min_bid": 1.90, "dph_total": 1.91},
        rental="bid", max_dph=2.0)                     # 1.20 x 1.90 = 2.28 > 2.0
    assert (cid, reason) == (None, "over_ceiling")


def test_launcher_prices_a_bid_against_the_ONDEMAND_market(monkeypatch):
    """R1 on the launch path. Auto-pricing a replacement bid used the OFFER's
    own `dph_total` as the on-demand clamp — but this offer came from a BID
    search, where that field is the interruptible price (~min_bid + 0.5%). The
    clamp then pinned the bid a tenth of a cent over the floor: the razor-thin
    understudy shape ($1.071 over a $1.0667 floor, evicted 45 min later),
    manufactured by the code that was supposed to prevent it.

    The reference now comes from the on-demand market. Unknown (soft read
    failed) means NO clamp — a missing price never invents one."""
    offer = {"id": 5, "min_bid": 1.90, "dph_total": 1.91, "machine_id": 36726}
    jc, _hf = _jc()
    _launch_env(monkeypatch, ondemand=None)             # no on-demand read
    _cid, dph, _r = replacement._launch_job_replacement(jc, [7], offer=offer,
                                              rental="bid")
    assert dph == pytest.approx(2.28)                   # 1.20 x floor, unclamped
    # With a REAL on-demand price the clamp binds, as designed — and since
    # 2026-08-09 it binds all the way to a REFUSAL here. This offer's floor is
    # $1.90 against an on-demand rate of $2.10, i.e. 90% of list: the survival
    # cushion wants $2.09 (99.5% of on-demand) and the hard ceiling is 0.75 x 2.10
    # = $1.575. The old code priced the $2.09 and rented it. Paying 99.5% of
    # on-demand for a preemptible box is strictly dominated, so the launcher now
    # declines to price this offer at all and the caller takes the on-demand rung.
    jc, _hf = _jc()
    _launch_env(monkeypatch, ondemand=2.10)
    _cid, dph, _r = replacement._launch_job_replacement(jc, [7], offer=offer,
                                              rental="bid")
    assert dph is None, "a 0.90 floor/on-demand offer must not be bid on"
    # a machine with real headroom is still priced and rented
    jc, _hf = _jc()
    _launch_env(monkeypatch, ondemand=5.00)
    _cid, dph, _r = replacement._launch_job_replacement(jc, [7], offer=offer,
                                              rental="bid")
    assert dph == pytest.approx(2.28)                   # 1.2 x floor; the 0.65 x od
                                                        # cap ($3.25) stays dormant


def test_launcher_rents_inside_the_ceiling(monkeypatch):
    jc, _hf = _jc()
    calls = _launch_env(monkeypatch)
    cid, dph, reason = replacement._launch_job_replacement(
        jc, [7], offer={"id": 5, "dph_total": 2.0}, rental="ondemand",
        max_dph=Q6["ceiling"])
    assert (cid, reason) == (88, None)
    assert dph == pytest.approx(2.0)
    assert [c for c in calls if c[0] == "do_launch"]


def test_launcher_reports_the_realized_ondemand_price(monkeypatch):
    """`_do_launch` reads a price off SEARCHED offers only and this lane always
    PINS one, so it returns None for on-demand — which journaled as
    `ondemand @ $None/hr` beside a $3.4741/hr meter. The offer row we launched
    from is the price."""
    jc, _hf = _jc()
    _launch_env(monkeypatch, launched=(88, 5, None))    # _do_launch: dph=None
    _cid, dph, _r = replacement._launch_job_replacement(
        jc, [7], offer={"id": 5, "dph_total": 1.75}, rental="ondemand")
    assert dph == pytest.approx(1.75)


def test_launcher_internal_repick_carries_the_ceiling_and_cuda_floor(monkeypatch):
    """The line that bought the $3.4741 box: with no offer handed in, the
    re-pick used neither the ceiling nor the CUDA floor the decision probes
    both passed."""
    jc, _hf = _jc()
    calls = _launch_env(monkeypatch, offer_pick=None)
    _cid, _dph, reason = replacement._launch_job_replacement(
        jc, [7], rental="ondemand", max_dph=Q6["ceiling"])
    assert reason == "no_offer"
    pick = [c for c in calls if c[0] == "pick"][0]
    assert pick[1] == "ondemand"
    assert pick[2] == pytest.approx(Q6["ceiling"]) and pick[3] == 12.8


def test_replacement_disk_is_floored_at_the_primarys_allocation(monkeypatch):
    """R7. Box 46914272 was evicted ~2 min into boot holding ~5.7 GB of a 50 GB
    workspace; `used x 1.4 + 12` sized its replacement at 20 GB, 85% full before
    the next training arm started. On a FORCED rehost a usage snapshot is not a
    measurement of need."""
    jc, _hf = _jc(instances=[dict(_inst(), disk_space=50.0, disk_usage=5.7)])
    calls = _launch_env(monkeypatch)
    replacement._launch_job_replacement(jc, [7], offer={"id": 5, "min_bid": 0.5,
                                              "dph_total": 0.7}, rental="bid")
    ns = [c for c in calls if c[0] == "do_launch"][0][1]
    assert ns.disk == 50, "the replacement shrank below the primary's allocation"


def test_replacement_disk_is_floored_at_the_LAUNCH_allocation(monkeypatch):
    """Task #69, the driftr3 H200 failure (DRIFT_ROSTER_R3_H200_COHORT §8): the
    lane launched `--disk 110` and a later hop put the workload on a 60 GB box,
    which died on the bundle's own disk guard with rc 5.

    The primary's allocation is evidence about whatever box last held the job —
    once ANY hop lands smaller, flooring at it propagates the shrink for the
    rest of the chain. The launch-time size is the statement of need and it is
    carried on the WATCH, so it survives every hop."""
    jc, _hf = _jc(instances=[dict(_inst(), disk_space=60.0, disk_usage=25.0)],
                  launch_disk_gb=110.0)
    calls = _launch_env(monkeypatch)
    replacement._launch_job_replacement(jc, [7], offer={"id": 5, "min_bid": 0.5,
                                              "dph_total": 0.7}, rental="bid")
    ns = [c for c in calls if c[0] == "do_launch"][0][1]
    assert ns.disk == 110, "the replacement shrank below the LAUNCH allocation"


def test_replacement_disk_survives_a_primary_that_left_the_listing(monkeypatch):
    """`classify_eviction(present=False)` is EVICTION_HOST_FAILURE — a `rent`
    class — so the ladder routinely sizes a replacement for a box that is no
    longer in the instance listing. Reading `disk_space` off nothing then
    reverts to the hardcoded launch default, silently. With the anchor there is
    nothing to read it off the box for."""
    jc, _hf = _jc(instances=[], launch_disk_gb=200.0)
    calls = _launch_env(monkeypatch)
    replacement._launch_job_replacement(jc, [7], offer={"id": 5, "min_bid": 0.5,
                                              "dph_total": 0.7}, rental="bid")
    ns = [c for c in calls if c[0] == "do_launch"][0][1]
    assert ns.disk == 200, "a vanished primary reverted the sizing to a default"


def test_replacement_disk_with_no_evidence_at_all_is_LOUD(monkeypatch, capsys):
    """Missing data fails safe but never silently: with neither an anchor nor a
    readable primary the fallback is the module default AND a warning naming
    the risk. A quiet default is exactly how a 110 GB job ended up on a 60 GB
    box for three hours."""
    jc, _hf = _jc(instances=[])
    calls = _launch_env(monkeypatch)
    replacement._launch_job_replacement(jc, [7], offer={"id": 5, "min_bid": 0.5,
                                              "dph_total": 0.7}, rental="bid")
    ns = [c for c in calls if c[0] == "do_launch"][0][1]
    assert ns.disk == int(disksize.REPLACEMENT_FALLBACK_GB)
    err = capsys.readouterr().err
    assert "replacement disk" in err and "default" in err


def test_replacement_disk_grows_when_the_primary_is_nearly_full(monkeypatch):
    """The other direction, and the reason this is a max() and not a floor: a
    box 92% full is evidence its allocation was WRONG, and the rehost is the
    only moment it can be corrected (a running instance cannot be resized)."""
    jc, _hf = _jc(instances=[dict(_inst(), disk_space=60.0, disk_usage=55.0)],
                  launch_disk_gb=60.0)
    calls = _launch_env(monkeypatch)
    replacement._launch_job_replacement(jc, [7], offer={"id": 5, "min_bid": 0.5,
                                              "dph_total": 0.7}, rental="bid")
    ns = [c for c in calls if c[0] == "do_launch"][0][1]
    assert ns.disk == 90                        # 55 x 1.4 + 12 -> 90


def test_launch_disk_anchor_is_captured_once_from_the_original_box(monkeypatch):
    """Same rule as the price anchor: written on the first observation of the
    ORIGINAL box and never rewritten, so a chain of ever-smaller replacements
    cannot ratchet the sizing down the way it ratcheted 110 -> 110 -> 60."""
    _tick_env(monkeypatch, dict(_inst(status="running"), disk_space=110.0,
                                disk_usage=30.0), market=0.5, on_demand=1.2)
    jc, hf = job_lane.job_supervise_init(_args())
    job_lane.job_supervise_tick(jc, hf)
    assert jc["launch_disk_gb"] == 110.0
    monkeypatch.setattr(lifecycle, "_instances_soft",
                        lambda: [dict(_inst(status="running"), disk_space=60.0,
                                      disk_usage=30.0)])
    job_lane.job_supervise_tick(jc, hf)
    assert jc["launch_disk_gb"] == 110.0, "the disk anchor must never be rewritten"


def test_the_disk_anchor_is_the_size_ASKED_FOR_not_the_size_DELIVERED(monkeypatch):
    """A host advertising less container disk than `--disk` clamps the
    allocation instead of refusing the rental, so `disk_space` is a fact about
    one box and not about the workload. The anchor takes the launch's own
    request off the box env."""
    _tick_env(monkeypatch, dict(_inst(status="running"), disk_space=10.0,
                                disk_usage=1.0,
                                extra_env=[["LAUNCH_DISK_GB", "50"]]),
              market=0.5, on_demand=1.2)
    jc, hf = job_lane.job_supervise_init(_args())
    job_lane.job_supervise_tick(jc, hf)
    assert jc["launch_disk_gb"] == 50.0


def test_an_under_delivered_container_disk_is_said_out_loud(monkeypatch, capsys):
    """The clamp is silent on vast's side; it must not be silent on ours."""
    _tick_env(monkeypatch, dict(_inst(status="running"), disk_space=10.0,
                                disk_usage=1.0,
                                extra_env=[["LAUNCH_DISK_GB", "50"]]),
              market=0.5, on_demand=1.2)
    jc, hf = job_lane.job_supervise_init(_args())
    job_lane.job_supervise_tick(jc, hf)
    out = capsys.readouterr().out
    assert "--disk 50G" in out and "10G" in out
    assert jc["disk_shortfall_said"] is True


def test_a_delivered_disk_ABOVE_the_request_still_wins(monkeypatch):
    """The anchor is a max(), not a swap: a box holding more than was asked for
    is still evidence the replacement must not shrink below."""
    _tick_env(monkeypatch, dict(_inst(status="running"), disk_space=120.0,
                                disk_usage=30.0,
                                extra_env=[["LAUNCH_DISK_GB", "50"]]),
              market=0.5, on_demand=1.2)
    jc, hf = job_lane.job_supervise_init(_args())
    job_lane.job_supervise_tick(jc, hf)
    assert jc["launch_disk_gb"] == 120.0


def test_a_box_with_no_disk_stamp_keeps_the_old_anchor_behaviour(monkeypatch):
    """Boxes launched before the stamp existed must degrade, not break."""
    _tick_env(monkeypatch, dict(_inst(status="running"), disk_space=60.0,
                                disk_usage=10.0), market=0.5, on_demand=1.2)
    jc, hf = job_lane.job_supervise_init(_args())
    job_lane.job_supervise_tick(jc, hf)
    assert jc["launch_disk_gb"] == 60.0
    assert not jc.get("disk_shortfall_said")


def test_the_replacement_rehosts_at_the_disk_the_LAUNCH_ASKED_FOR(monkeypatch):
    """The incident shape, end to end on the eviction lane: box 48005604 asked
    for 50 GB, was handed 10, and its rehost was sized at 10 — too small for the
    job's own 19.3 GB base-model asset, so the rescued queue was doomed before
    it started. Both the launch AND the offer search must carry 50."""
    clamped = dict(_inst(status="running"), disk_space=10.0, disk_usage=1.0,
                   extra_env=[["LAUNCH_DISK_GB", "50"]])
    _tick_env(monkeypatch, clamped, market=0.5, on_demand=1.2)
    jc, hf = job_lane.job_supervise_init(_args())
    job_lane.job_supervise_tick(jc, hf)          # anchors off the box env
    jc["instances"] = [clamped]
    calls = _launch_env(monkeypatch)
    replacement._launch_job_replacement(jc, [7], offer={"id": 5, "min_bid": 0.5,
                                              "dph_total": 0.7}, rental="bid")
    ns = [c for c in calls if c[0] == "do_launch"][0][1]
    assert ns.disk == 50, "the rehost inherited the SHORTFALL, not the request"
    need, _why, known = replacement._replacement_disk_need(jc, clamped)
    assert (need, known) == (50.0, True)


# --------------------------------------------------------------------------- #
# THE ARCHITECTURE ALLOWLIST (2026-08-18). Two runs voided in two days by the
# same blind spot: a replacement honoured the VRAM floor and landed on an RTX
# PRO 6000 (sm_120), where the baked flash_attn 2.8.3 has no kernel image — the
# import succeeds and the first forward dies. Once after an SLA condemn
# (worked around by hand with `--gpu h200 --max-replacements 0`), once after a
# pk2 A100 was outbid, caught by a bundle gate only AFTER the swap.
#
# Making the workloads arch-tolerant is the primary fix and is elsewhere. This
# is the other half: a replacement stays inside what the launch DECLARED, and an
# arch change is never silent.
# --------------------------------------------------------------------------- #
def _cc_inst(**kw):
    """A primary launched with `--cc-allow`: the list is in its box env, which
    is where the supervise tick reads it from."""
    base = dict(_inst(status="running"), gpu_name="A100 PCIE", gpu_ram=81920,
                disk_space=50.0, disk_usage=5.0,
                extra_env=[["LAUNCH_CC_ALLOW", "80,86,89,90"]])
    base.update(kw)
    return base


def test_the_arch_allowlist_is_read_off_the_box_env(monkeypatch):
    """Same channel and the same rule as the disk anchor: which silicon the
    workload can run on is a property of the WORKLOAD, so the watch takes it off
    the launch's own stamp rather than off whatever card is in the box now."""
    _tick_env(monkeypatch, _cc_inst(), market=0.5, on_demand=1.2)
    jc, hf = job_lane.job_supervise_init(_args())
    job_lane.job_supervise_tick(jc, hf)
    assert jc["launch_cc_allow"] == [80, 86, 89, 90]


def test_a_box_with_no_arch_stamp_stays_unconstrained(monkeypatch):
    """Additive, like every anchor before it: a box launched without
    `--cc-allow` — i.e. every box that exists today — behaves as it always did."""
    _tick_env(monkeypatch, _cc_inst(extra_env=[]), market=0.5, on_demand=1.2)
    jc, hf = job_lane.job_supervise_init(_args())
    job_lane.job_supervise_tick(jc, hf)
    assert jc["launch_cc_allow"] == []
    assert replacement._replacement_cc_allow(jc) == ()


def test_the_replacement_search_only_accepts_declared_architectures(monkeypatch):
    """THE fix, at the seam the defect went through. With an allowlist active
    the candidate set holds only in-list silicon — and an offer whose
    compute_cap the market did not advertise is EXCLUDED, because an unknown is
    exactly how an sm_120 gets in."""
    book = [dict(_a100(1, 56764, 128.0, 0.10, name="RTX PRO 6000 WS"),
                 compute_cap=1200, gpu_ram=98304),
            dict(_a100(2, 56759, 128.0, 0.15), compute_cap=None),
            dict(_a100(3, 56760, 128.0, 0.20), compute_cap=800)]
    jc, _hf = _a100_jc(launch_cc_allow=[80, 86, 89, 90])
    seen = _market(monkeypatch, bid=book)
    offers = replacement._job_replacement_offers(jc, [], rental="bid",
                                                 max_dph=1.52, cuda=12.8)
    assert seen[0]["cc_allow"] == (80, 86, 89, 90)
    assert [o["id"] for o in offers] == [3], (
        "the sm_120 offer and the one that would not say are both out")


def test_no_arch_stamp_means_no_arch_FILTER(monkeypatch):
    """The regression that keeps the search additive: with no declared list the
    query carries no allowlist and the cheapest qualifying offer still wins,
    unknown compute_cap included."""
    book = [dict(_a100(1, 56764, 128.0, 0.10, name="RTX PRO 6000 WS"),
                 compute_cap=1200, gpu_ram=98304),
            dict(_a100(3, 56760, 128.0, 0.20), compute_cap=800)]
    jc, _hf = _a100_jc()
    seen = _market(monkeypatch, bid=book)
    offers = replacement._job_replacement_offers(jc, [], rental="bid",
                                                 max_dph=1.52, cuda=12.8)
    assert seen[0]["cc_allow"] == ()
    assert [o["id"] for o in offers] == [1, 3]


def test_the_replacement_carries_the_arch_stamp_TO_THE_NEXT_BOX(monkeypatch):
    """A constraint that dies at the first hop is not a constraint. The rehost's
    own launch re-stamps the list, so hop 2 is bounded by hop 0's declaration —
    the disk anchor's lesson, applied to the architecture."""
    box = _cc_inst()
    _tick_env(monkeypatch, box, market=0.5, on_demand=1.2)
    jc, hf = job_lane.job_supervise_init(_args())
    job_lane.job_supervise_tick(jc, hf)
    jc["instances"] = [box]
    calls = _launch_env(monkeypatch)
    replacement._launch_job_replacement(
        jc, [7], offer={"id": 5, "min_bid": 0.5, "dph_total": 0.7,
                        "gpu_name": "A100 PCIE", "compute_cap": 800},
        rental="bid")
    ns = [c for c in calls if c[0] == "do_launch"][0][1]
    assert ns.cc_allow == "80,86,89,90"


def test_a_watch_adopted_after_the_stamp_reads_it_off_the_primary(monkeypatch):
    """The anchor is seeded by the tick, and a watch can be handed a box it
    never ticked (fleetd reconcile-adopt). The primary's env is the fallback,
    for the same reason `_replacement_disk_need` has one."""
    jc, _hf = _jc(instances=[_cc_inst()])
    assert jc["launch_cc_allow"] == []
    assert replacement._replacement_cc_allow(jc) == (80, 86, 89, 90)


def test_a_cross_arch_replacement_is_LOUD(monkeypatch, capsys):
    """Not a block — the swap already happened, and refusing is the stamp's job.
    But the two incidents were both invisible at the moment they mattered, and
    the boundary is a measurement boundary as well as a kernel one."""
    jc, _hf = _a100_jc()
    calls = _launch_env(monkeypatch)
    replacement._launch_job_replacement(
        jc, [7], offer={"id": 5, "min_bid": 0.5, "dph_total": 0.7,
                        "gpu_name": "RTX PRO 6000 WS", "compute_cap": 1200},
        rental="bid")
    assert [c for c in calls if c[0] == "do_launch"], "it must still rent"
    err = capsys.readouterr().err
    assert "ARCHITECTURE CHANGE" in err and "sm_120" in err
    ev = [f for name, f in jc["ladder_journal"] if name == "arch_change"]
    assert len(ev) == 1
    assert ev[0]["old_arch"] == "A100 PCIE" and "sm_120" in ev[0]["new_arch"]
    assert "not comparable" in ev[0]["note"]


def test_a_same_arch_replacement_says_nothing(monkeypatch, capsys):
    """An alarm on every rehost is an alarm nobody reads. Same alias family —
    the SKU string may differ — is not an architecture change."""
    jc, _hf = _a100_jc()
    _launch_env(monkeypatch)
    replacement._launch_job_replacement(
        jc, [7], offer={"id": 5, "min_bid": 0.5, "dph_total": 0.7,
                        "gpu_name": "A100 SXM4"}, rental="bid")
    assert "ARCHITECTURE CHANGE" not in capsys.readouterr().err
    assert not [f for name, f in jc.get("ladder_journal", [])
                if name == "arch_change"]


def test_the_replacement_event_journals_the_size_it_rented(monkeypatch):
    """§8 of the drift-roster write-up could only INFER which lane sized the
    60 GB box, from its label and the observed disk — "not something read off a
    decision event". Now the event carries what was rented AND the anchor it
    inherited from."""
    jc, hf = _jc(launch_disk_gb=110.0, last_replacement_disk_gb=110.0)
    calls = _wire(monkeypatch)
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    ev = [c[2] for c in calls if c[0] == "emit" and c[1] == "eviction_replaced"]
    assert ev and ev[0]["disk_gb"] == 110.0 and ev[0]["launch_disk_gb"] == 110.0


def test_the_disk_anchor_survives_a_daemon_restart():
    """It is the same class of state as `launch_dph_anchor` — a bound the
    ladder must not re-derive from whatever box it happens to be holding — so
    it rides the durable half of the watch record too."""
    assert "launch_disk_gb" in fleetd.REPLACEMENT_STATE_KEYS
    w = {}
    fleetd._replacement_state_persist({"launch_disk_gb": 110.0}, w)
    fresh = {"launch_disk_gb": None}
    fleetd._replacement_state_restore(fresh, w)
    assert fresh["launch_disk_gb"] == 110.0


def test_the_ceiling_helper_never_ratchets_off_the_current_box(monkeypatch):
    jc, _hf = _jc()
    assert replacement._job_replacement_ceiling(jc) == pytest.approx(1.52)   # 2.0 x 0.76
    jc["launch_dph_anchor"] = None
    assert replacement._job_replacement_ceiling(jc) is None                  # refuse, don't guess


# --------------------------------------------------------------------------- #
# 6. minimum requirements + tokens-per-dollar (incident 2026-08-16)
#
# The night, as numbers: the rung pinned the primary's exact vast gpu_name
# ("H100 NVL"), took limit:1, and got ONE offer whose bid floor sat at 91.5% of
# its own on-demand price. `bid_decision`'s cushion rail correctly refused to
# hold THAT machine on spot, the rung read the refusal as "spot is unsafe", and
# the ladder rented on-demand at $1.603/hr while a $0.4027 H200 NVL spot offer
# sat on the same market, unqueried. The rail was right. The candidate set was a
# sample of one. Doc of record:
# docs/plans/vast-fleet-sessions/2026-08-16-replacement-probe-tpd/SESSION.md
# --------------------------------------------------------------------------- #
N_H100 = dict(id=101, machine_id=1, gpu_name="H100 NVL", num_gpus=2,
              min_bid=1.4667, dph_total=1.4740, on_demand=1.603)
N_H200 = dict(id=102, machine_id=2, gpu_name="H200 NVL", num_gpus=2,
              min_bid=0.4027, dph_total=0.4047, on_demand=3.34)


def _offer_row(d):
    return {k: val for k, val in d.items() if k != "on_demand"}


def _od_by_machine(monkeypatch, table):
    """Per-machine on-demand market, the way `_market_ondemand_soft` answers.
    Returns the probe log so a test can assert WHICH machine was priced."""
    seen = []

    def _probe(mid, num_gpus=None):
        seen.append((mid, num_gpus))
        return table.get(mid)

    monkeypatch.setattr(pricing, "_market_ondemand_soft", _probe)
    return seen


def _fake_rates(monkeypatch, table):
    """Install a stub `gpu_rates` for the lazy import in `_gpu_rate_soft`. The
    interface under test is the one the real module is being built to:
    `rate_for(gpu_name, num_gpus=1, shape=None) -> float | None`."""
    mod = types.SimpleNamespace(
        rate_for=lambda gpu_name, num_gpus=1, shape=None: table.get(gpu_name))
    monkeypatch.setitem(sys.modules, "gpu_rates", mod)
    return mod


def test_the_candidate_set_picks_the_cheap_h200_over_the_vetoed_h100(monkeypatch):
    """THE regression. Both cards clear the minimum requirements; the H100's own
    floor/on-demand spread makes it structurally unsafe to hold on spot, the
    H200's does not. With no rate table installed the survivors rank on
    effective price — and that alone rents the $0.48 spot box instead of the
    $1.603 on-demand one."""
    monkeypatch.delitem(sys.modules, "gpu_rates", raising=False)
    _od_by_machine(monkeypatch, {1: N_H100["on_demand"], 2: N_H200["on_demand"]})
    jc, hf = _jc()
    calls = _wire(monkeypatch,
                  spot=[_offer_row(N_H100), _offer_row(N_H200)],
                  od=None,                       # $1.603 is over the $1.52 ceiling
                  od_uncapped={"id": 9, "gpu_name": "H100 NVL", "num_gpus": 2,
                               "dph_total": N_H100["on_demand"]})
    assert replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid") is True
    launch = [c for c in calls if c[0] == "launch"][0]
    assert launch[1] == "bid"                    # NOT the on-demand rung
    assert launch[5] == N_H200["id"]             # the H200, not the pinned SKU
    assert launch[2] == pytest.approx(0.483)     # 1.2 x the $0.4027 floor
    dec = [c[2] for c in calls if c[0] == "emit"
           and c[1] == "eviction_replacement_decision"][0]
    assert dec["rental"] == "bid" and dec["spot_gpu"] == "H200 NVL"
    assert dec["spot_candidates"] == 2 and dec["spot_survivors"] == 1
    assert dec["ranked_by"] == "price"           # no rate table installed
    # the survival rail priced against the H200's OWN machine, not the class
    assert dec["spot_ondemand"] == pytest.approx(N_H200["on_demand"])


def test_tokens_per_dollar_may_prefer_the_pricier_faster_card(monkeypatch):
    """Owner ruling 2026-08-16: rank on measured tok/s per effective $/hr, and
    an UPGRADE is the right answer when it is the better deal. Both candidates
    survive the rail here, so the ranking is the whole test: on price the H100
    wins ($0.72 vs $1.08), on tokens-per-dollar the H200 does (185 vs 139
    tok/$)."""
    _od_by_machine(monkeypatch, {1: 3.0, 2: 4.0})
    h100 = dict(_offer_row(N_H100), min_bid=0.60)
    h200 = dict(_offer_row(N_H200), min_bid=0.90)

    monkeypatch.delitem(sys.modules, "gpu_rates", raising=False)
    jc, hf = _jc(args={"budget": 50.0})
    jc["launch_dph_anchor"] = 2.0                        # ceiling $4.00
    calls = _wire(monkeypatch, spot=[h100, h200], od=None,
                  od_uncapped={"id": 9, "dph_total": 3.0, "num_gpus": 2})
    assert replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid") is True
    assert [c for c in calls if c[0] == "launch"][0][5] == h100["id"]

    _fake_rates(monkeypatch, {"H100 NVL": 100.0, "H200 NVL": 200.0})
    jc, hf = _jc(args={"budget": 50.0})
    jc["launch_dph_anchor"] = 2.0
    calls = _wire(monkeypatch, spot=[h100, h200], od=None,
                  od_uncapped={"id": 9, "dph_total": 3.0, "num_gpus": 2})
    assert replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid") is True
    launch = [c for c in calls if c[0] == "launch"][0]
    assert launch[5] == h200["id"]
    dec = [c[2] for c in calls if c[0] == "emit"
           and c[1] == "eviction_replacement_decision"][0]
    assert dec["ranked_by"] == "tokens_per_dollar"


def test_a_missing_rate_for_any_candidate_falls_back_to_price(monkeypatch):
    """All-or-nothing: mixing a measured tok/s against an assumed one ranks
    money on the assumption. One unmeasured class demotes the WHOLE set back to
    cheapest-effective-price, which is the pre-2026-08-16 pick."""
    _od_by_machine(monkeypatch, {1: 3.0, 2: 4.0})
    _fake_rates(monkeypatch, {"H100 NVL": 100.0})        # H200 unmeasured
    h100 = dict(_offer_row(N_H100), min_bid=0.60)
    h200 = dict(_offer_row(N_H200), min_bid=0.90)
    jc, hf = _jc(args={"budget": 50.0})
    jc["launch_dph_anchor"] = 2.0
    calls = _wire(monkeypatch, spot=[h100, h200], od=None,
                  od_uncapped={"id": 9, "dph_total": 3.0, "num_gpus": 2})
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    assert [c for c in calls if c[0] == "launch"][0][5] == h100["id"]
    dec = [c[2] for c in calls if c[0] == "emit"
           and c[1] == "eviction_replacement_decision"][0]
    assert dec["ranked_by"] == "price"


def test_gpu_rate_soft_is_none_without_the_module(monkeypatch):
    """The module is being built in parallel; its ABSENCE must be a degraded
    ranking, never an exception and never an invented number."""
    monkeypatch.delitem(sys.modules, "gpu_rates", raising=False)
    assert replacement._gpu_rate_soft("H200 NVL", 2) is None
    _fake_rates(monkeypatch, {"H200 NVL": 0.0})          # zero is not a rate
    assert replacement._gpu_rate_soft("H200 NVL", 2) is None
    _fake_rates(monkeypatch, {"H200 NVL": 12.5})
    assert replacement._gpu_rate_soft("H200 NVL", 2) == 12.5
    assert replacement._gpu_rate_soft("", 2) is None


# --------------------------------------------------------------------------- #
# 3b. the ladder ranks on THIS JOB's shape (2026-08-28)
#
# `_gpu_rate_soft` passed no shape, so `gpu_rates.entry_for` resolved
# DEFAULT_SHAPE — the slowest measured shape for the card, and not necessarily
# the shape of the run being replaced. `train_rates` answers per (card, JOB), and
# the job is identified from the queue TICKET's env: the training shape is a
# BUNDLE env jobd exports box-side and is never in the instance record.
# --------------------------------------------------------------------------- #
_TRAIN_ENV = {"BASE_SLUG": "qwen35-9b", "MAX_SEQ": "20480", "BATCH": "1",
              "GRAD_ACCUM": "32", "QUANT": "bf16"}
_JID = "20260828T000000-9b-w20480-a0"


def _queued_ticket(monkeypatch, env=_TRAIN_ENV, jid=_JID):
    """Put one jobs-v2 TICKET on the watch's queue and return the pending views
    the poll would have folded. `read_ticket` is the seam because that is where
    `jobmeta.make_ticket` puts the canonical config."""
    reads = []

    def _read(box, job_id, **k):
        reads.append((str(box), job_id))
        return {"config": {"env": dict(env)}} if job_id == jid else None

    monkeypatch.setattr(jobmeta, "read_ticket", _read)
    return [{"job_id": jid, "status": "running"}], reads


def _fake_train_rates(monkeypatch, table, tier="measured"):
    """Stub the ANCHOR LOOKUP only — the real module keeps answering
    `family_from_env`, so these tests pin the real env->Family mapping."""
    def _rate(family, gpu_name, num_gpus=1, gpu_ram_gb=None, **k):
        tok = table.get(gpu_name)
        if tok is None:
            return None
        return train_rates.RateEstimate(tok_s=tok, tier=tier, n=3, spread=1.0,
                                        runs=(), op_point="b1xga32 gc=on",
                                        why="stub")
    monkeypatch.setattr(train_rates, "rate_for_offer", _rate)


def _tpd_market(monkeypatch):
    """The two-candidate market both rate tables are read against: H100 cheaper
    ($0.72 effective), H200 pricier ($1.08), both surviving the rail."""
    _od_by_machine(monkeypatch, {1: 3.0, 2: 4.0})
    return (dict(_offer_row(N_H100), min_bid=0.60),
            dict(_offer_row(N_H200), min_bid=0.90))


def _run_tpd(monkeypatch, *, views=None):
    h100, h200 = _tpd_market(monkeypatch)
    jc, hf = _jc(args={"budget": 50.0})
    jc["launch_dph_anchor"] = 2.0                        # ceiling $4.00
    if views is not None:
        jc["pending_views"] = views
    calls = _wire(monkeypatch, spot=[h100, h200], od=None,
                  od_uncapped={"id": 9, "dph_total": 3.0, "num_gpus": 2})
    assert replacement._job_eviction_replace(
        jc, hf, bp.EVICTION_OUTBID, "outbid") is True
    dec = [c[2] for c in calls if c[0] == "emit"
           and c[1] == "eviction_replacement_decision"][0]
    return [c for c in calls if c[0] == "launch"][0][5], dec


def test_the_job_shaped_rate_outranks_the_default_shape_one(monkeypatch):
    """THE defect. The card-class table (DEFAULT_SHAPE) says the H100 is the
    better deal; anchors at the shape this box is ACTUALLY training say the
    H200 is. The ladder must buy the box the job's own numbers pick."""
    _fake_rates(monkeypatch, {"H100 NVL": 200.0, "H200 NVL": 100.0})
    _fake_train_rates(monkeypatch, {"H100 NVL": 100.0, "H200 NVL": 200.0})

    oid, dec = _run_tpd(monkeypatch)                     # no ticket -> no family
    assert oid == N_H100["id"]
    assert (dec["ranked_by"], dec["rate_source"]) == ("tokens_per_dollar",
                                                      "gpu_rates")

    views, reads = _queued_ticket(monkeypatch)
    oid, dec = _run_tpd(monkeypatch, views=views)
    assert oid == N_H200["id"]                           # the ranking flipped
    assert dec["rate_source"] == "train_rates:measured"
    assert reads and reads[0] == ("41", _JID)            # off the WATCH's queue


def test_a_provisional_anchor_labels_the_whole_set_provisional(monkeypatch):
    """A stale anchor is a FLOOR, so it is ranked on and labeled — and one of
    them makes the comparison a floor, which is what the journal must say."""
    _fake_rates(monkeypatch, {"H100 NVL": 200.0, "H200 NVL": 100.0})
    _fake_train_rates(monkeypatch, {"H100 NVL": 100.0, "H200 NVL": 200.0},
                      tier="provisional")
    views, _ = _queued_ticket(monkeypatch)
    oid, dec = _run_tpd(monkeypatch, views=views)
    assert oid == N_H200["id"] and dec["rate_source"] == "train_rates:provisional"


@pytest.mark.parametrize("env", [
    {"MAX_SEQ": "20480"},                                # eval bundle: no base
    {"BASE_SLUG": "qwen35-9b"},                          # no window, no ladder
    {},
])
def test_an_underivable_family_is_byte_identical_to_the_card_class_path(
        monkeypatch, env):
    """A watch whose ticket names no training shape — an eval, a generation
    sweep, a probe — must rank EXACTLY as it did before this seam existed, and
    must not even consult the anchor table."""
    _fake_rates(monkeypatch, {"H100 NVL": 200.0, "H200 NVL": 100.0})
    monkeypatch.setattr(train_rates, "rate_for_offer",
                        lambda *a, **k: pytest.fail("asked for a job-shaped "
                                                    "rate without a family"))
    views, _ = _queued_ticket(monkeypatch, env=env)
    oid, dec = _run_tpd(monkeypatch, views=views)
    assert oid == N_H100["id"]
    assert (dec["ranked_by"], dec["rate_source"]) == ("tokens_per_dollar",
                                                      "gpu_rates")


def test_an_unreadable_ticket_degrades_to_the_card_class_path(monkeypatch):
    """B2 is down / the ticket moved / the queue is empty: a rate is a ranking
    signal, so its source failing must cost accuracy, never a replacement."""
    _fake_rates(monkeypatch, {"H100 NVL": 200.0, "H200 NVL": 100.0})
    _fake_train_rates(monkeypatch, {"H100 NVL": 100.0, "H200 NVL": 200.0})
    monkeypatch.setattr(jobmeta, "read_ticket",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("b2")))
    oid, dec = _run_tpd(monkeypatch, views=[{"job_id": _JID, "status": "running"}])
    assert oid == N_H100["id"] and dec["rate_source"] == "gpu_rates"


def test_the_rate_source_is_all_or_nothing_across_the_candidate_set(monkeypatch):
    """`_replacement_rank` refuses to mix a measured rate with an assumed one;
    mixing two rate SOURCES has the same defect one level down, because a
    job-shaped rate for one card against DEFAULT_SHAPE for another compares two
    different training jobs. One unanswered card demotes the WHOLE set to the
    card-class table — which is a complete, comparable ranking, not price."""
    _fake_rates(monkeypatch, {"H100 NVL": 200.0, "H200 NVL": 100.0})
    _fake_train_rates(monkeypatch, {"H100 NVL": 100.0})   # H200 has no anchor
    views, _ = _queued_ticket(monkeypatch)
    oid, dec = _run_tpd(monkeypatch, views=views)
    assert oid == N_H100["id"]
    assert (dec["ranked_by"], dec["rate_source"]) == ("tokens_per_dollar",
                                                      "gpu_rates")


def test_neither_source_covering_the_set_still_falls_back_to_price(monkeypatch):
    """The pre-existing all-or-nothing check, unchanged: when the fallback table
    is short too, no rate ranked anything and the record says so with a null
    source rather than naming a table that decided nothing."""
    _fake_rates(monkeypatch, {"H100 NVL": 200.0})         # H200 unmeasured
    _fake_train_rates(monkeypatch, {"H100 NVL": 100.0})
    views, _ = _queued_ticket(monkeypatch)
    oid, dec = _run_tpd(monkeypatch, views=views)
    assert oid == N_H100["id"]                            # cheapest effective
    assert (dec["ranked_by"], dec["rate_source"]) == ("price", None)


def test_the_ticket_is_read_once_per_job_across_stuck_eviction_ticks(monkeypatch):
    """A stuck eviction re-runs this ~every 50 s and the read is a B2 `cat`.
    Box 47398836 spent 66 minutes in that state."""
    _fake_rates(monkeypatch, {"H100 NVL": 200.0, "H200 NVL": 100.0})
    _fake_train_rates(monkeypatch, {"H100 NVL": 100.0, "H200 NVL": 200.0})
    views, reads = _queued_ticket(monkeypatch)
    jc, _hf = _jc()
    jc["pending_views"] = views
    for _ in range(3):
        assert replacement._job_train_family(jc, 2) is not None
    assert len(reads) == 1
    jc["pending_views"] = [dict(views[0], job_id="20260828T000000-other-a0")]
    assert replacement._job_train_family(jc, 2) is None   # ticket unknown
    assert len(reads) == 2                                # ...and re-read


def test_a_serve_watch_never_looks_for_a_training_family(monkeypatch):
    """A serve box has no queue and no training shape; asking for one would be a
    B2 read per tick that can only ever answer None."""
    monkeypatch.setattr(jobmeta, "read_ticket",
                        lambda *a, **k: pytest.fail("read a queue on a serve box"))
    jc, _hf = _jc()
    jc["serve_mode"] = True
    jc["pending_views"] = [{"job_id": _JID, "status": "running"}]
    assert replacement._job_train_family(jc, 2) is None


def test_rank_rates_is_set_level_and_tiers_down_to_the_worst_member(monkeypatch):
    """The unit under the ladder tests: source selection is over the SET, and a
    train_rates tier is the worst tier in it."""
    fam = train_rates.family_from_env(_TRAIN_ENV, world_size=2)
    assert fam is not None
    rows = [{"gpu_name": "H100 NVL"}, {"gpu_name": "H200 NVL"}]
    monkeypatch.delitem(sys.modules, "gpu_rates", raising=False)
    assert replacement._rank_rates(rows, 2, fam) == ([None, None], None)

    _fake_rates(monkeypatch, {"H100 NVL": 200.0, "H200 NVL": 100.0})
    assert replacement._rank_rates(rows, 2, None) == ([200.0, 100.0],
                                                      "gpu_rates")
    _fake_train_rates(monkeypatch, {"H100 NVL": 10.0})       # covers 1 of 2
    assert replacement._rank_rates(rows, 2, fam) == ([200.0, 100.0],
                                                     "gpu_rates")
    _fake_train_rates(monkeypatch, {"H100 NVL": 10.0, "H200 NVL": 20.0})
    assert replacement._rank_rates(rows, 2, fam) == ([10.0, 20.0],
                                                     "train_rates:measured")

    def _mixed_tier(family, gpu_name, num_gpus=1, gpu_ram_gb=None, **k):
        return train_rates.RateEstimate(
            tok_s=10.0, tier=("measured" if gpu_name == "H100 NVL"
                              else "provisional"),
            n=1, spread=1.0, runs=(), op_point="b1xga32 gc=on", why="stub")
    monkeypatch.setattr(train_rates, "rate_for_offer", _mixed_tier)
    assert replacement._rank_rates(rows, 2, fam)[1] == "train_rates:provisional"
    assert replacement._rank_rates([], 2, fam) == ([], None)


def test_rank_rates_passes_the_offer_s_own_per_card_vram(monkeypatch):
    """`gpu_ram` is MiB on a vast row and the fit filter is in GB — reading it
    raw would admit every operating point on a card that fits none of them."""
    seen = []
    monkeypatch.setattr(train_rates, "rate_for_offer",
                        lambda f, g, n=1, ram=None, **k: seen.append(ram))
    fam = train_rates.family_from_env(_TRAIN_ENV, world_size=2)
    replacement._rank_rates([{"gpu_name": "H200 NVL", "gpu_ram": 143771}], 2, fam)
    assert seen == [pytest.approx(140.4, abs=0.1)]


def test_a_family_underivable_from_the_env_is_not_an_error():
    """`family_from_env` is documented never to raise; the ladder's guard adds
    the import failure, and both answer None."""
    assert replacement._train_family_soft({"NOT": "a training bundle"}) is None
    assert replacement._train_family_soft(None) is None
    assert replacement._train_rate_soft(None, "H200 NVL", 2) is None
    assert replacement._train_rate_soft(object(), "", 2) is None


def test_train_rates_absence_is_a_degraded_ranking_never_an_exception(monkeypatch):
    """`train_rates` is a sibling of the package, not a dependency of it."""
    assert replacement._train_family_soft(_TRAIN_ENV, world_size=2) is not None
    monkeypatch.delitem(sys.modules, "train_rates", raising=False)
    monkeypatch.setattr(sys, "path", [p for p in sys.path
                                      if not os.path.isfile(
                                          os.path.join(p, "train_rates.py"))])
    assert replacement._train_family_soft(_TRAIN_ENV, world_size=2) is None
    assert replacement._train_rate_soft(object(), "H200 NVL", 2) is None


def test_a_single_pathological_candidate_still_falls_through_to_ondemand(monkeypatch):
    """The OTHER half of the fix must not regress: when the candidate set really
    does hold one structurally-unsafe offer, the rung still escalates — and the
    refusal an operator reads is the rail's own arithmetic, not a bare "no
    market read" from an empty survivor list."""
    _od_by_machine(monkeypatch, {1: N_H100["on_demand"]})
    jc, hf = _jc()
    calls = _wire(monkeypatch, spot=[_offer_row(N_H100)],
                  od={"id": 9, "gpu_name": "H100 NVL", "num_gpus": 2,
                      "dph_total": 1.40})
    assert replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid") is True
    launch = [c for c in calls if c[0] == "launch"][0]
    assert launch[1] == "ondemand" and launch[5] == 9
    dec = [c[2] for c in calls if c[0] == "emit"
           and c[1] == "eviction_replacement_decision"][0]
    assert dec["spot_candidates"] == 1 and dec["spot_survivors"] == 0
    assert "escalate_over_ceiling" in dec["reason"]     # the rail, not silence
    assert dec["offer_min_bid"] == pytest.approx(N_H100["min_bid"])


# --------------------------------------------------------------------------- #
# 6b. the candidate CLASS itself
# --------------------------------------------------------------------------- #
def test_the_candidate_class_is_minimum_requirements_not_a_sku_pin(monkeypatch):
    """An H100-class primary admits every bf16 card AT OR ABOVE its VRAM — the
    query carries a `gpu_ram_gb` floor and a name allowlist, never the primary's
    exact vast gpu_name."""
    seen = {}

    def _search(**kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(market_offers, "pick_offers", _search)
    jc, _hf = _jc()
    jc["instances"] = [dict(_inst(), gpu_name="H100 NVL", gpu_ram=81920)]
    replacement._job_replacement_offers(jc, [7], rental="bid", max_dph=1.52, cuda=12.8)
    assert seen["gpu_ram_gb"] == pytest.approx(80.0)
    names = set(seen["gpu"])
    assert {"H100 NVL", "H100 SXM", "H200", "H200 NVL", "B200"} <= names
    assert seen["num_gpus"] == 2 and seen["cuda"] == 12.8
    assert seen["max_dph"] == pytest.approx(1.52)
    assert seen["exclude_machines"] == [7]
    assert seen["limit"] == replacement.REPLACEMENT_CANDIDATES
    assert seen["verified"] is True


def test_an_unreadable_primary_vram_narrows_to_the_alias_family():
    """No VRAM floor means a name-open class could DOWNGRADE us, so the class
    falls back to the primary's own alias family — still wider than the exact
    SKU pin it replaced, never smaller than the card we lost."""
    names, ram = market_offers._replacement_candidate_class({"gpu_name": "H100 NVL"})
    assert ram is None
    assert set(names) == {"H100 SXM", "H100 PCIE", "H100 NVL"}
    # a SKU in no alias family replaces itself and nothing else
    assert market_offers._replacement_candidate_class({"gpu_name": "RTX WEIRD 9000"}) \
        == (("RTX WEIRD 9000",), None)


def test_a_vram_field_that_is_not_plausibly_a_card_is_unknown_not_zero():
    """The number becomes a search FLOOR. A megabyte value read as gigabytes
    floors at ~0.08, which silently admits every SMALLER card on the market —
    the one direction the class must never widen in."""
    assert models._gpu_ram_gb(81920) == pytest.approx(80.0)      # MB, the vast unit
    assert models._gpu_ram_gb(80) == pytest.approx(80.0)         # already GB
    assert models._gpu_ram_gb(0) is None and models._gpu_ram_gb(None) is None
    assert models._gpu_ram_gb("nonsense") is None
    assert market_offers._replacement_candidate_class(
        {"gpu_name": "H100 NVL", "gpu_ram": 0})[1] is None  # -> family fallback


def test_gpu_family_names_is_the_alias_inverse_index():
    assert set(market_offers.gpu_family_names("H100 NVL")) == {"H100 SXM", "H100 PCIE",
                                                   "H100 NVL"}
    assert set(market_offers.gpu_family_names("H200")) == {"H200", "H200 NVL"}
    assert market_offers.gpu_family_names("") == []


def test_the_verified_knob_defaults_true_and_can_be_widened(monkeypatch):
    """Knob precedence, and the reason it is not routed through `_rebid_knob`:
    that helper coerces with `type(default)(v)`, and `bool("0")` is True — a
    bool knob through it would ignore every disable."""
    jc = {"a": argparse.Namespace(replacement_verified=None)}
    monkeypatch.delenv("JOB_REPLACEMENT_VERIFIED", raising=False)
    monkeypatch.setattr(config, "load_herdd_config", dict)
    assert replacement._job_replacement_verified(jc) is True
    monkeypatch.setenv("JOB_REPLACEMENT_VERIFIED", "0")
    assert replacement._job_replacement_verified(jc) is False
    jc["a"].replacement_verified = "1"                  # namespace wins over env
    assert replacement._job_replacement_verified(jc) is True


def test_the_size_filter_prefers_an_exact_gpu_count_but_never_empties():
    a, b = {"id": 1, "num_gpus": 8}, {"id": 2, "num_gpus": 2}
    assert replacement._replacement_fit([a, b], 2) == [b]
    assert replacement._replacement_fit([a], 2) == [a]            # a box beats no box
    assert replacement._replacement_fit([], 2) == []


def test_the_candidate_walk_never_prices_off_a_bid_offers_dph_total(monkeypatch):
    """DOC-50 GUARD, at the seam the walk moved it to. A bid-type offer's
    `dph_total` is the current INTERRUPTIBLE price (~min_bid + 0.5%); read as an
    on-demand rate it makes every candidate look razor-thin and routes the
    ladder to the expensive rung on arithmetic describing no real offer."""
    poisoned = dict(_offer_row(N_H100), dph_total=1.4740)
    seen = _od_by_machine(monkeypatch, {1: 3.20})
    cands = replacement._replacement_spot_walk([poisoned], 4.0, 2)
    assert seen == [(1, 2)]                              # machine + gpu count
    assert cands[0].ondemand == pytest.approx(3.20)      # NOT 1.4740
    assert cands[0].price == pytest.approx(
        bp.bid_decision(N_H100["min_bid"], 4.0, 3.20).price)


def test_the_ondemand_probe_is_deduped_per_machine_and_bounded(monkeypatch):
    """One eviction tick must not become 16 market reads: the walk probes at
    most `max_probes` DISTINCT machines and drops the tail rather than pricing
    it off a number it did not read."""
    seen = _od_by_machine(monkeypatch, {i: 3.0 for i in range(20)})
    many = [{"id": i, "machine_id": i, "gpu_name": "H200 NVL", "num_gpus": 2,
             "min_bid": 0.4 + i / 100.0} for i in range(20)]
    walked = replacement._replacement_spot_walk(many, 4.0, 2, max_probes=3)
    assert len(seen) == 3 and len(walked) == 3
    seen2 = _od_by_machine(monkeypatch, {1: 3.0})
    dupes = [dict(o, machine_id=1) for o in many[:5]]
    assert len(replacement._replacement_spot_walk(dupes, 4.0, 2, max_probes=2)) == 5
    assert len(seen2) == 1


def test_the_ondemand_rung_reads_dph_total_because_that_offer_IS_ondemand():
    """The doc-50 ban is on a BID offer's `dph_total`. On an ondemand-type offer
    that field is the list price we would actually be billed, so the on-demand
    rung needs no per-machine probe."""
    cands = replacement._replacement_ondemand_walk(
        [{"id": 1, "gpu_name": "H200", "dph_total": 2.0},
         {"id": 2, "gpu_name": "H200", "dph_total": None}], 2)
    assert [c.offer["id"] for c in cands] == [1]
    assert cands[0].price == 2.0 and cands[0].ondemand == 2.0


# --------------------------------------------------------------------------- #
# 6c. the CONTAINER-DISK fit requirement (the 23 GB / 47 GB trap)
#
# 2026-08-16, the same night, one axis further down. The minimum-requirements
# class carried per-card VRAM, GPU count, the cuda floor and the inet floor —
# and NOT the container disk, though `_launch_job_replacement` computes exactly
# how many GB it is about to ask vast for. So the rung ranked by price over a
# book it could not use: the eviction ladder minted 47845212 on a 47 GB machine
# and the pull-reschedule ladder minted 47845159 on a 23 GB one, both against a
# 50 GB request, and vast does not refuse a rental it cannot fit — it hands
# back a smaller container. One died `insufficient_disk: 22GB free < 24GB
# required`; the other was destroyed by hand off `ls` before it could.
#
# The market rows below are the REAL bundles response, quoted from
# <upstream-bench>/archive/runs/2026-08-16-gpu-rate-bench-a100/offers/
# a100_offers_bid_2026-08-16T0605Z.json — the cheap A100 PCIe supply that night
# was one host (67231) advertising 18/23/33/47/128/275 GB.
# Readout: docs/plans/vast-fleet-sessions/2026-08-16-replacement-probe-tpd/
# GPU_BENCH_RESULTS.md Part 2 §5.
# --------------------------------------------------------------------------- #
def _a100(oid, machine, disk, bid, ram=81920, ngpu=1, name="A100 PCIE"):
    return {"id": oid, "machine_id": machine, "host_id": 67231,
            "gpu_name": name, "num_gpus": ngpu, "gpu_ram": ram,
            "disk_space": disk, "min_bid": bid,
            "dph_total": round(bid * 1.0167, 4), "inet_down": 9086.3}


#: host 67231's book, cheapest-first, exactly as the API returned it
H67231_BOOK = [
    _a100(32304443, 56764, 23.0, 0.13333333333333333),
    _a100(32303769, 56759, 47.0, 0.154),
    _a100(45716572, 56760, 128.20000000000007, 0.19999999999999998),
    # a different host, and a 40 GB card — the VRAM floor already rejects it,
    # which is the axis that WAS carried, kept here as the contrast
    _a100(42182413, 141396, 570.0, 0.304, ram=40960),
]

#: what the primary looked like: an 80 GB A100 PCIe on host 67231's m56748,
#: launched `--disk 50` (`jobs_box_launched … disk_gb: 50.0` in the journal)
def _a100_jc(**kw):
    inst = dict(_inst(machine=56748), gpu_name="A100 PCIE", gpu_ram=81920,
                num_gpus=1, disk_space=50.0, disk_usage=5.0)
    kw.setdefault("launch_disk_gb", 50.0)
    kw.setdefault("instances", [inst])
    return _jc(**kw)


#: the REAL search, captured before `_wire` stubs it — this section is about
#: what the search filters on, so the seam that gets faked is the MARKET
_REAL_OFFERS = replacement._job_replacement_offers


def _market(monkeypatch, bid=(), ondemand=()):
    """A vast bundles market that HONOURS the filters `pick_offers` builds:
    gpu-name allowlist, the per-card VRAM floor, num_gpus, the container-disk
    floor, the price ceiling and the machine exclusions, sorted cheapest-first
    and truncated to `limit`.

    Faked at `pick_offers` rather than at `_job_replacement_offers`, so the
    candidate-class construction and every filter it passes are under test
    rather than mocked away. Returns the query log."""
    seen = []

    def _pick(*, gpu=(), num_gpus=1, gpu_ram_gb=None, disk_gb=None,
              max_dph=None, rental="bid", verified=True, inet_down=None,
              exclude_machines=None, geo=None, any_gpu=False, any_inet=False,
              cuda=None, cc_allow=None, limit=1):
        seen.append({"rental": rental, "disk_gb": disk_gb, "max_dph": max_dph,
                     "gpu_ram_gb": gpu_ram_gb, "num_gpus": num_gpus,
                     "cc_allow": tuple(cc_allow or ()),
                     "excluded": tuple(exclude_machines or ())})
        rows = list(ondemand if rental in ("ondemand", "on-demand") else bid)
        price = (lambda o: o["dph_total"]) if rental in (
            "ondemand", "on-demand") else (lambda o: o["min_bid"])
        out = []
        for o in rows:
            if gpu and o["gpu_name"] not in gpu:
                continue
            if gpu_ram_gb and o["gpu_ram"] < market_offers.gpu_ram_floor_mib(gpu_ram_gb):
                continue
            if o["num_gpus"] < num_gpus:
                continue
            if disk_gb and o["disk_space"] < float(disk_gb):
                continue
            if max_dph is not None and price(o) > max_dph:
                continue
            if o["machine_id"] in set(exclude_machines or ()):
                continue
            # the arch allowlist is applied client-side by the real picker, and
            # an offer with no compute_cap is EXCLUDED while it is active
            if not market_offers.cc_allow_ok(o, cc_allow):
                continue
            out.append(o)
        return sorted(out, key=price)[:max(1, int(limit or 1))]

    monkeypatch.setattr(market_offers, "pick_offers", _pick)
    monkeypatch.setattr(replacement, "_job_replacement_offers", _REAL_OFFERS)
    return seen


def test_the_replacement_search_carries_the_workloads_disk_floor(monkeypatch):
    """THE regression, at the seam. The 50 GB the launch is about to request
    must reach the bundles query as a `disk_space` floor — the axis that was
    missing while VRAM, GPU count, cuda and inet were all carried."""
    jc, _hf = _a100_jc()
    seen = _market(monkeypatch, bid=H67231_BOOK)
    offers = replacement._job_replacement_offers(jc, [], rental="bid", max_dph=1.52,
                                       cuda=12.8)
    assert seen[0]["disk_gb"] == 50.0
    assert [o["id"] for o in offers] == [45716572], (
        "the 23 GB and 47 GB machines are still in the candidate set")


def test_the_ladder_rents_the_128GB_machine_over_the_cheaper_23GB_one(
        monkeypatch):
    """End to end on the night's own market: the 23 GB offer is cheapest and
    the ladder must not rent it. $0.20 instead of $0.1333 is the correct
    trade — a box that cannot stage the assets is not a cheaper box, it is a
    retry loop."""
    jc, hf = _a100_jc()
    _od_by_machine(monkeypatch, {56764: 1.0, 56759: 1.0, 56760: 1.0})
    calls = _wire(monkeypatch)
    _market(monkeypatch, bid=H67231_BOOK)        # after _wire: un-stub the search
    assert replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid") is True
    launch = [c for c in calls if c[0] == "launch"][0]
    assert launch[5] == 45716572, "rented a machine too small for the job"
    assert launch[2] == pytest.approx(0.24)      # 1.2 x the $0.20 floor
    dec = [c[2] for c in calls if c[0] == "emit"
           and c[1] == "eviction_replacement_decision"][0]
    assert dec["disk_floor_gb"] == 50.0 and dec["disk_blocked"] is False
    assert dec["spot_machine"] == 56760


def test_a_market_with_no_big_enough_box_REFUSES_and_names_the_disk_bound(
        monkeypatch):
    """Fail-closed, and say which bound stopped it. With m56760 excluded (it is
    the machine we were just evicted from) host 67231 has nothing over 47 GB,
    so there is no replacement to rent — and the refusal must say DISK, not
    `no_market_read`. `_report_stalled` puts this string in the operator alarm,
    where 'raise the bid' would be actively wrong advice."""
    jc, hf = _a100_jc()
    _od_by_machine(monkeypatch, {})
    calls = _wire(monkeypatch)
    _market(monkeypatch, bid=[o for o in H67231_BOOK
                              if o["machine_id"] != 56760])
    assert replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid") is False
    assert not [c for c in calls if c[0] == "launch"], "rented anyway"
    dec = [c[2] for c in calls if c[0] == "emit"
           and c[1] == "eviction_replacement_decision"][0]
    assert dec["action"] == "stop" and dec["disk_blocked"] is True
    assert dec["disk_floor_gb"] == 50.0
    for fragment in ("DISK FLOOR", "50G", "47G", "56759"):
        assert fragment in dec["reason"], fragment
    # ...and the same string is what fleetd's rescue_stalled alarm reads
    assert jc["replacement_refused"] == dec["reason"]


def test_an_empty_market_is_not_blamed_on_disk(monkeypatch):
    """The shortfall probe answers a question, it does not assume an answer: a
    market with no offers at all (or one whose biggest box DOES fit and was
    refused on price) must keep the price arithmetic as its reason. A rail that
    always blames the last thing added is worse than no rail."""
    jc, hf = _a100_jc()
    _od_by_machine(monkeypatch, {})
    calls = _wire(monkeypatch)
    _market(monkeypatch, bid=[])
    assert replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid") is False
    dec = [c[2] for c in calls if c[0] == "emit"
           and c[1] == "eviction_replacement_decision"][0]
    assert dec["disk_blocked"] is False and "DISK FLOOR" not in dec["reason"]
    assert "no affordable replacement" in dec["reason"]


def test_the_shortfall_probe_runs_only_when_the_ladder_is_already_refusing(
        monkeypatch):
    """It costs one extra bundles read. It must never run on a tick that can
    rent — and it is the ONLY caller allowed to search with the floor lifted."""
    jc, hf = _a100_jc()
    _od_by_machine(monkeypatch, {56760: 1.0})
    _wire(monkeypatch)
    seen = _market(monkeypatch, bid=H67231_BOOK)
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    assert [q for q in seen if q["disk_gb"] == 0] == []

    jc, hf = _a100_jc()
    seen = _market(monkeypatch, bid=[o for o in H67231_BOOK
                                     if o["machine_id"] != 56760])
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    assert len([q for q in seen if q["disk_gb"] == 0]) == 1


def test_the_search_floor_is_the_number_the_launch_asks_for(monkeypatch):
    """The property that makes this correct rather than merely stricter: one
    function feeds both, so a search cannot qualify a machine the very next
    call then over-requests. Includes the grow case — a nearly-full primary
    resizes the rental UP, and the floor moves with it."""
    for kw, want in (({}, 50.0),
                     (dict(launch_disk_gb=110.0), 110.0),
                     (dict(instances=[dict(_inst(machine=56748),
                                           gpu_name="A100 PCIE", gpu_ram=81920,
                                           num_gpus=1, disk_space=60.0,
                                           disk_usage=55.0)],
                           launch_disk_gb=60.0), 90.0)):
        jc, _hf = _a100_jc(**kw)
        seen = _market(monkeypatch, bid=H67231_BOOK)
        replacement._job_replacement_offers(jc, [], rental="bid")
        launch_calls = _launch_env(monkeypatch)
        replacement._launch_job_replacement(jc, [], offer={"id": 5, "min_bid": 0.5,
                                                 "dph_total": 0.7},
                                  rental="bid")
        ns = [c for c in launch_calls if c[0] == "do_launch"][0][1]
        assert seen[0]["disk_gb"] == want == ns.disk


def test_an_unknowable_disk_need_still_floors_the_search(monkeypatch):
    """No anchor, no readable box: the launcher already falls back to
    REPLACEMENT_FALLBACK_GB out loud, so the search must carry the same number.
    Searching un-floored and then requesting 120 GB is the two-call disagreement
    this exists to prevent."""
    jc, _hf = _jc(instances=[], launch_disk_gb=None)
    seen = _market(monkeypatch, bid=H67231_BOOK)
    replacement._job_replacement_offers(jc, [], rental="bid")
    assert seen[0]["disk_gb"] == disksize.REPLACEMENT_FALLBACK_GB


def test_the_palt_poll_prices_a_box_the_job_could_actually_use(monkeypatch):
    """`p_alt` is "what would moving cost right now" and it feeds the one-shot
    defense's ceiling. Priced off a 23 GB machine it is not the cost of moving,
    it is the cost of a box we would have to throw away."""
    jc, _hf = _a100_jc()
    seen = _market(monkeypatch, bid=H67231_BOOK)
    replacement._job_palt_poll(jc, NOW, own_machine=56748)
    assert seen[0]["disk_gb"] == 50.0
    assert jc["p_alt"] == pytest.approx(0.19999999999999998)
    assert jc["p_alt_machine"] == 56760


def test_the_forced_rehost_says_disk_when_it_finds_no_offer(monkeypatch):
    """The pull-reschedule lane (box 47845159's lane) goes through the same
    launcher. Its `no_offer` error is what reaches the journal, so it names the
    floor rather than leaving the operator hunting for a price problem."""
    jc, _hf = _a100_jc()
    _launch_env(monkeypatch, offer_pick=None)
    cid, dph, reason = replacement._launch_job_replacement(jc, [], rental="bid")
    assert (cid, dph, reason) == (None, None, "no_offer")
    assert "50G of container disk" in jc["last_error"]


def test_offer_disk_gb_reads_GB_and_refuses_to_invent_one():
    """`disk_space` is GB on an offer while `gpu_ram` one field over is MiB.
    Unreadable is None — a zero here would read as "this machine has no disk"
    and could never be the biggest candidate, quietly disabling the diagnosis."""
    assert replacement._offer_disk_gb({"disk_space": 47.0}) == 47.0
    assert replacement._offer_disk_gb({"disk_space": "128.2"}) == pytest.approx(128.2)
    assert replacement._offer_disk_gb({"disk_space": 0}) is None
    assert replacement._offer_disk_gb({}) is None and replacement._offer_disk_gb(None) is None


# --------------------------------------------------------------------------- #
# 6d. the evicted-machine exclusion, TTL'd by class
# --------------------------------------------------------------------------- #
def test_an_outbid_exclusion_expires_but_a_host_failure_never_does():
    """On 2026-08-16 the lost machine's floor never moved ($0.80 all night);
    only its listing blinked during a host stop. A permanent exclusion there
    removes the machine we know the most about from every later probe. A broken
    host, or one an on-demand renter is sitting on, stays excluded.

    A machine id PER LEG, not one reused: the strike-earning classes share one
    durable reputation store within a test, so reusing an id would stack three
    condemnations onto one host and read its BLOCK back as a TTL result."""
    jc = {}
    replacement._job_note_evicted_machine(jc, 7, bp.EVICTION_OUTBID, NOW)
    assert replacement._job_excluded_machines(jc, NOW) == {7}
    assert replacement._job_excluded_machines(jc, NOW + 60) == {7}
    assert replacement._job_excluded_machines(
        jc, NOW + replacement.EVICTED_EXCLUSION_TTL_S + 1) == set()
    assert 7 in jc["evicted_machines"]          # the record itself is kept

    for mid, cls in ((71, bp.EVICTION_HOST_FAILURE), (72, bp.EVICTION_ONDEMAND)):
        jc = {}
        replacement._job_note_evicted_machine(jc, mid, cls, NOW)
        assert replacement._job_excluded_machines(jc, NOW + 86_400) == {mid}

    jc = {}
    replacement._job_note_evicted_machine(jc, 73, bp.EVICTION_HOST_STOP, NOW)
    assert replacement._job_excluded_machines(jc, NOW + 86_400) == set()


def test_a_pull_bad_machine_is_never_ttld():
    """That host failed to pull our image — a property of the host, not of the
    market, and the whole point of the reschedule is a DIFFERENT one."""
    jc = {"pull_bad_machines": {5}}
    replacement._job_note_evicted_machine(jc, 7, bp.EVICTION_OUTBID, NOW)
    assert replacement._job_excluded_machines(jc, NOW + 86_400) == {5}


def test_the_ttl_survives_a_daemon_restart_and_tolerates_the_old_shape():
    """The set survives a restart already; without the timestamps beside it a
    restart would silently promote every TTL'd exclusion back to permanent."""
    assert "evicted_machine_ts" in fleetd.REPLACEMENT_STATE_KEYS
    jc = {"replacements": 1, "launch_dph_anchor": 0.76}
    replacement._job_note_evicted_machine(jc, 7, bp.EVICTION_OUTBID, NOW)
    replacement._job_note_evicted_machine(jc, 9, bp.EVICTION_HOST_FAILURE, NOW)
    w = {}
    fleetd._replacement_state_persist(jc, w)
    w = json.loads(json.dumps(w))               # state.json is JSON, and only JSON
    assert w["replacement"]["evicted_machines"] == [7, 9]
    fresh = {"evicted_machines": set(), "evicted_machine_ts": {}}
    fleetd._replacement_state_restore(fresh, w)
    assert fresh["evicted_machines"] == {7, 9}
    assert replacement._job_excluded_machines(fresh, NOW + 60) == {7, 9}
    assert replacement._job_excluded_machines(
        fresh, NOW + replacement.EVICTED_EXCLUSION_TTL_S + 1) == {9}      # 7 aged out

    # OLD SHAPE: a watch persisted before the sidecar existed. No timestamps
    # means no TTL means permanent — degraded to the pre-2026-08-16 behaviour,
    # never a silently un-excluded broken host.
    old = {"replacement": {"evicted_machines": [7, 9]}}
    fresh = {"evicted_machines": set(), "evicted_machine_ts": {}}
    fleetd._replacement_state_restore(fresh, old)
    assert replacement._job_excluded_machines(fresh, NOW + 86_400) == {7, 9}


def test_the_eviction_rung_stamps_the_class_it_excluded_on(monkeypatch):
    jc, hf = _jc()
    _wire(monkeypatch)
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    assert jc["evicted_machine_ts"]["7"]["class"] == bp.EVICTION_OUTBID
    assert jc["evicted_machine_ts"]["7"]["ts"] == NOW


# --------------------------------------------------------------------------- #
# 6e. host_stop is HOST evidence: it earns a durable strike, and a repeat
#     escalates the exclusion
# --------------------------------------------------------------------------- #
def _kinds(machine, now=NOW):
    rows = [r for r in hostrep.summary(now)
            if r["machine_id"] == str(machine)]
    return rows[0]["kinds"] if rows else []


def test_a_host_stop_earns_a_durable_strike_where_a_price_class_does_not():
    """`host_stop`'s own classification is "box present, chunk still listed and
    rentable, our bid still clears the floor" — nobody took the box, the host
    stopped it. That is a claim about the HOST and it has to outlive the watch,
    or a stopping host is met from a clean slate by every later session.

    `outbid` is a price we lost and `no_credit` names our own account, so
    neither may leave a mark on a machine's record."""
    replacement._job_note_evicted_machine({}, 7, bp.EVICTION_HOST_STOP, NOW)
    assert _kinds(7) == ["host_stop"]

    for mid, cls in ((81, bp.EVICTION_OUTBID), (82, bp.EVICTION_NO_CREDIT)):
        replacement._job_note_evicted_machine({}, mid, cls, NOW)
        assert _kinds(mid) == [], cls
    assert bp.EVICTION_HOST_STOP not in replacement.STRIKE_FREE_EVICTION_CLASSES

    # everything else still reads as the generic host failure it always did
    replacement._job_note_evicted_machine({}, 83, bp.EVICTION_HOST_FAILURE, NOW)
    assert _kinds(83) == ["host_failure"]


def test_a_second_host_stop_from_one_machine_blocks_it_across_watches():
    """The durable half of the answer: the second stop crosses the block score,
    so a FRESH watch — which has no `evicted_machines` at all — still refuses
    the host that took the last two boxes."""
    replacement._job_note_evicted_machine({}, 7, bp.EVICTION_HOST_STOP, NOW - 600)
    assert hostrep.blocked_machines(NOW - 600) == set()
    replacement._job_note_evicted_machine({}, 7, bp.EVICTION_HOST_STOP, NOW)
    assert hostrep.blocked_machines(NOW) == {7}
    assert replacement._job_excluded_machines({}, NOW) == {7}


def test_a_repeat_host_stop_escalates_the_exclusion_ttl(monkeypatch):
    """One stop is a minute-scale event and ages out on the flat TTL. A repeat
    is a host that will stop the replacement too, so the exclusion grows —
    bounded, because even a serial stopper has to be retryable the same day.

    Measured with the DURABLE layer OFF, through its own escape hatch, because
    the two layers are independent and the block would otherwise hide every
    result here. Reputation can be disabled, its store unwritable, or its score
    discounted by a good boot in between; the watch's own memory is what has to
    hold in all three."""
    monkeypatch.setenv(hostrep.DISABLE_ENV, "1")
    base = replacement.EVICTED_EXCLUSION_TTL_S
    jc = {}
    replacement._job_note_evicted_machine(jc, 7, bp.EVICTION_HOST_STOP, NOW)
    assert replacement._job_excluded_machines(jc, NOW + base + 1) == set()

    replacement._job_note_evicted_machine(jc, 7, bp.EVICTION_HOST_STOP, NOW)
    assert jc["evicted_machine_ts"]["7"]["host_stops"] == 2
    assert replacement._job_excluded_machines(jc, NOW + base + 1) == {7}
    grown = base * replacement.HOST_STOP_TTL_ESCALATION
    assert replacement._job_excluded_machines(jc, NOW + grown - 1) == {7}
    assert replacement._job_excluded_machines(jc, NOW + grown + 1) == set()

    # ...and it is capped, not unbounded
    for _ in range(6):
        replacement._job_note_evicted_machine(jc, 7, bp.EVICTION_HOST_STOP, NOW)
    assert replacement._job_excluded_machines(
        jc, NOW + replacement.HOST_STOP_TTL_MAX_S + 1) == set()


def test_the_stop_count_is_not_laundered_by_a_later_outbid(monkeypatch):
    """A host that stops us, wins a contested rebid, then stops us again has
    stopped us twice. Counting only consecutive records would let one price
    event reset the escalation. Durable layer off for the same reason as
    above — this pins the WATCH's memory, not the store's."""
    monkeypatch.setenv(hostrep.DISABLE_ENV, "1")
    jc = {}
    replacement._job_note_evicted_machine(jc, 7, bp.EVICTION_HOST_STOP, NOW)
    replacement._job_note_evicted_machine(jc, 7, bp.EVICTION_OUTBID, NOW)
    assert jc["evicted_machine_ts"]["7"]["host_stops"] == 1
    replacement._job_note_evicted_machine(jc, 7, bp.EVICTION_HOST_STOP, NOW)
    assert jc["evicted_machine_ts"]["7"]["host_stops"] == 2
    assert replacement._job_excluded_machines(
        jc, NOW + replacement.EVICTED_EXCLUSION_TTL_S + 1) == {7}


def test_a_restored_record_without_the_stop_count_keeps_the_old_ttl():
    """The daemon in flight holds sidecars written before `host_stops` existed.
    A missing key means the old behaviour, never a crash and never a promotion
    to the escalated TTL."""
    jc = {"evicted_machines": {7},
          "evicted_machine_ts": {"7": {"ts": NOW, "class": bp.EVICTION_HOST_STOP}}}
    assert replacement._job_excluded_machines(jc, NOW + 60) == {7}
    assert replacement._job_excluded_machines(
        jc, NOW + replacement.EVICTED_EXCLUSION_TTL_S + 1) == set()

    jc["evicted_machine_ts"]["7"]["host_stops"] = "not a number"
    assert replacement._job_excluded_machines(
        jc, NOW + replacement.EVICTED_EXCLUSION_TTL_S + 1) == set()


def test_the_replacement_probe_refuses_the_machine_that_just_stopped_us(monkeypatch):
    """"One host_stop is enough — move." Price-first replacement re-converges on
    the cheapest machine, which is how one host took nine boxes in a night, so
    the machine has to leave the candidate set BEFORE the search — the exclusion
    is what the launcher and the journal both read."""
    jc, hf = _jc()
    calls = _wire(monkeypatch)
    assert replacement._job_eviction_replace(
        jc, hf, bp.EVICTION_HOST_STOP, "host stopped the box") is True
    probes = [c for c in calls if c[0] == "offer"]
    assert probes and all(7 in c[2] for c in probes), probes
    launches = [c for c in calls if c[0] == "launch"]
    assert launches and all(7 in c[3] for c in launches), launches


# --------------------------------------------------------------------------- #
# 6d. an ON-DEMAND box cannot be displaced by an on-demand renter
# --------------------------------------------------------------------------- #
def test_a_non_bid_box_can_never_classify_as_ondemand_displaced():
    """2026-08-16: the ladder journaled `ondemand_displaced` with `is_bid:
    false`. That class is the single strongest input to the expensive rung
    ("the eviction was an on-demand claim, which outranks any bid"), so a
    misfire buys an on-demand replacement on evidence that cannot exist."""
    args = dict(present=True, actual_status="exited", on_demand=1.0,
                last_bid=1.5)                    # a STALE bid off a previous box
    assert bp.classify_eviction(**args) == bp.EVICTION_ONDEMAND      # unstated
    assert bp.classify_eviction(is_bid=True, **args) == bp.EVICTION_ONDEMAND
    assert bp.classify_eviction(is_bid=False, **args) == bp.EVICTION_UNKNOWN
    # the fall-through arm asserts the same thing and is gated the same way
    tail = dict(present=True, actual_status="exited", market_min_bid=0.5,
                on_demand=1.0, last_bid=0.6, market_listed=None)
    assert bp.classify_eviction(**tail) == bp.EVICTION_ONDEMAND
    assert bp.classify_eviction(is_bid=False, **tail) == bp.EVICTION_UNKNOWN


def test_the_is_bid_gate_never_touches_the_other_classes():
    """It shuts off ONE arm. Host failure, outbid and host-stop are unchanged
    for a non-bid box (they describe the box leaving, not who took it)."""
    assert bp.classify_eviction(present=False, is_bid=False) \
        == bp.EVICTION_HOST_FAILURE
    assert bp.classify_eviction(present=True, actual_status="exited",
                                market_min_bid=0.9, last_bid=0.7,
                                is_bid=False) == bp.EVICTION_OUTBID
    assert bp.classify_eviction(present=True, actual_status="running",
                                is_bid=False) == bp.EVICTION_UNKNOWN


# --------------------------------------------------------------------------- #
# 8. THE RETAINED BOX THAT CAME BACK TO LIFE
#
# INCIDENT 2026-08-16, box 47833510 (doc of record:
# docs/plans/vast-fleet-sessions/2026-08-16-replacement-probe-tpd/SESSION.md).
#
# The eviction ladder retained an evicted H100 NVL spot box at 03:39:50Z. It
# left the standing bid at the $1.20 the rescue rung had just raised it to
# (floor $0.80), and it left outstanding the two in-place `start` calls vast had
# answered "Required resources are currently unavailable, state change queued."
# At ~04:40Z machine 34985 had capacity again and vast honoured the queued
# start. The box came back RUNNING with zero jobs, zero GPU util, billing
# $0.8407/hr against a retention disclosure of $0.1222 for the whole 3h window.
#
# Nothing caught it, and each miss had its own excuse:
#   * `herdd reap` — exempt, by the `keep:` label retention itself stamps;
#   * the retention sweep — skipped LIVE boxes as "a human mid-salvage", and
#     inside the window did not even look;
#   * the stray sweep — auto-adopt refused (a watch is filed under this very id
#     while supervising the REPLACEMENT), and the refusal path pops the stray
#     record, resetting the idle-park fuse's grace clock every cycle.
#
# Observed 04:42:05Z (jobd's `scratch_probe` on B2) to the operator's manual
# destroy at 04:50:57Z. Had nobody been watching, the retention window ran to
# 06:39:38Z (~$2.5) and past it the box was still LIVE, i.e. still skipped by
# both the reaper and the backstop — indefinitely.
# --------------------------------------------------------------------------- #
def _quiesce_wire(monkeypatch):
    """Record the stop/bid-pin PUTs the quiesce path issues.

    conftest's `_block_box_mutating_puts` refuses them suite-wide; this is the
    per-test override that lets us assert on them."""
    seen = {"stop": [], "bid": []}
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda iid, state: (seen["stop"].append((str(iid), state)),
                                            (True, None))[1])
    monkeypatch.setattr(lifecycle, "_put_bid_soft",
                        lambda iid, price: (seen["bid"].append((str(iid), price)),
                                            (True, None))[1])
    return seen


def test_a_retained_box_is_stopped_and_its_bid_dropped(monkeypatch):
    """THE REGRESSION. Retention must leave the box unable to come back.

    Both moves are required and neither substitutes for the other: the `stop`
    withdraws the queued start vast had accepted but not yet executed, and the
    bid pin stops vast auto-resuming a stopped bid instance whose standing bid
    clears the floor. On 2026-08-16 the box billed at $0.8407/hr — BELOW the
    $0.96 it launched at — so "the rescue rung raised the bid to $1.20" is not
    the whole story: the original bid would have cleared too. Dropping the bid
    without stopping the box, or vice versa, is half a fix."""
    jc, hf = _jc()
    _wire(monkeypatch)
    seen = _quiesce_wire(monkeypatch)
    assert replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid") is True
    rec = _retention_rec(jc)
    assert rec["status"] == "retained"
    assert seen["stop"] == [("41", "stopped")]
    assert seen["bid"] == [("41", bp.RETENTION_PARK_BID)]
    assert rec["quiesce"]["stopped"] is True
    assert rec["quiesce"]["bid_pinned"] == bp.RETENTION_PARK_BID
    assert not rec["quiesce"]["errors"]


def test_the_park_pin_is_below_any_floor_not_below_the_current_one():
    """A floor-relative pin auto-resumes the moment the floor DROPS, which is
    exactly when a chunk frees up. Same constant, same reasoning as the handoff
    fence's HANDOFF_PARK_BID (the box-44566398 stuck-bid leak) — that lane
    learned this in July and retention never got the lesson."""
    assert bp.RETENTION_PARK_BID == bp.HANDOFF_PARK_BID == 0.001
    assert bp.RETENTION_PARK_BID < 0.8      # m34985's floor the whole night


def test_an_ondemand_box_is_stopped_but_never_bid_pinned(monkeypatch):
    """An on-demand instance has no standing bid; PUT-ing a price at one is a
    move against a box that cannot be outbid. The `stop` still applies — a
    queued start is not a bid concept."""
    inst = _inst(dph=1.6)
    inst["is_bid"] = False
    jc, hf = _jc(instances=[inst])
    _wire(monkeypatch)
    seen = _quiesce_wire(monkeypatch)
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    assert seen["stop"] == [("41", "stopped")]
    assert seen["bid"] == []
    assert _retention_rec(jc)["quiesce"]["bid_pinned"] is None


def test_the_prior_bid_is_recorded_so_a_manual_resume_can_restore_it(
        monkeypatch, capsys):
    """Pinning to $0.001 leaves the box unable to win its market — which is the
    point, and which a human resuming it for salvage has to know to undo."""
    jc, hf = _jc()
    _wire(monkeypatch)
    _quiesce_wire(monkeypatch)
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    assert _retention_rec(jc)["quiesce"]["prior_bid"] == pytest.approx(0.76)
    assert "standing bid was $0.7600" in capsys.readouterr().out


def test_a_failed_quiesce_still_retains_the_box_and_says_so(monkeypatch, capsys):
    """Best-effort, like the label PUT: failing to defend the box is never a
    reason to throw away the disk the owner asked for. But it must be LOUD —
    an un-quiesced retained box is the incident, live."""
    jc, hf = _jc()
    _wire(monkeypatch)
    monkeypatch.setattr(lifecycle, "_put_state_soft", lambda i, s: (False, "HTTP 500"))
    monkeypatch.setattr(lifecycle, "_put_bid_soft", lambda i, p: (False, "HTTP 429"))
    assert replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid") is True
    rec = _retention_rec(jc)
    assert rec["status"] == "retained"
    assert rec["quiesce"]["stopped"] is False
    assert "QUIESCE FAILED" in retention._quiesce_summary(rec["quiesce"])
    assert "HTTP 500" in capsys.readouterr().out


def test_the_quiesce_is_journaled_where_fleet_log_can_see_it(monkeypatch):
    """`_job_handoff_emit` writes to B2 only. A money-relevant rung has to reach
    `fleet log`, which is the ladder journal (task #78)."""
    jc, hf = _jc()
    _wire(monkeypatch)
    _quiesce_wire(monkeypatch)
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid")
    ev = [f for name, f in (jc.get("ladder_journal") or [])
          if name == "jobs_box_quiesced"]
    assert ev, "the quiesce never reached the ladder journal"
    assert ev[0]["stopped"] is True
    assert ev[0]["bid_pinned"] == bp.RETENTION_PARK_BID


def test_dry_run_issues_no_stop_and_no_bid_pin(monkeypatch):
    jc, hf = _jc()
    jc["dry_run"] = True
    seen = _quiesce_wire(monkeypatch)
    q = retention._job_quiesce_box(jc, "41", _inst(), why="retention")
    assert seen == {"stop": [], "bid": []}
    assert q["stopped"] is None and q["bid_pinned"] is None


# --- the retention cost model is only true of a STOPPED box ----------------- #
def test_the_disclosed_retention_cost_prices_a_stopped_box_only():
    """Hypothesis 2, settled. `retention_plan` prices ALLOCATED DISK. Box
    47833510 reported `storage_total_cost` $0.9778/day, so retention disclosed
    $0.1222 for a 3h window ($0.0407/hr) — and the resurrected box billed
    $0.8407/hr, 20.6x the model. The model was never wrong about storage; it was
    wrong to assume, with nothing enforcing it, that the box would stay
    stopped."""
    p = bp.retention_plan(retention_h=3, present=True, now=0,
                          storage_day_usd=0.9777777777777781)
    assert p.cost_usd == p.cost_hi_usd == pytest.approx(0.1222, abs=1e-4)
    assert "while the box stays STOPPED" in p.reason
    usd, mult = bp.retention_live_cost(0.8407, 3 * 3600.0,
                                       storage_day_usd=0.9777777777777781)
    assert usd == pytest.approx(2.5221, abs=1e-3)      # what 3h would have cost
    assert mult == pytest.approx(20.6, abs=0.1)


def test_retention_live_cost_reports_an_unreadable_rate_as_unknown():
    """A box we cannot price is unpriced, never $0.00 — a zero here reads as
    "this resurrection was free", which is the reassurance that let the real one
    run."""
    assert bp.retention_live_cost(None, 3600) == (None, None)
    assert bp.retention_live_cost(0.0, 3600) == (None, None)
    assert bp.retention_live_cost("x", 3600) == (None, None)


# --- the sweep now sees a live retained box --------------------------------- #
def _live_rec(deadline_ts, **kw):
    r = _rec(deadline_ts)
    r["storage_day_usd"] = 0.9777777777777781
    r["quiesce"] = {"stopped": True, "bid_pinned": bp.RETENTION_PARK_BID,
                    "prior_bid": 1.2, "errors": [], "why": "retention"}
    r.update(kw)
    return r


def test_a_live_retained_box_inside_its_window_is_caught_and_re_parked(
        monkeypatch, capsys):
    """THE OBSERVABILITY REGRESSION. The old sweep returned at `now < deadline`
    without ever looking at `actual_status`, so the whole 2h59m of the 47833510
    window was a blind spot by construction."""
    seen = _quiesce_wire(monkeypatch)
    jc, calls = _swept(monkeypatch, _live_rec(NOW + 3 * 3600),
                       instances=[_inst(status="running", dph=0.8407)],
                       now=NOW + 3600)
    r = jc["retained_boxes"][0]
    assert r["status"] == "retained"           # still retained, NOT destroyed
    assert r["live_since_ts"] == NOW + 3600
    assert r["resurrections"] == 1 and r["requiesces"] == 1
    assert seen["stop"] == [("41", "stopped")]
    assert seen["bid"] == [("41", bp.RETENTION_PARK_BID)]
    assert "RUNNING again" in capsys.readouterr().out


def test_the_resurrection_reaches_fleet_log_with_the_money_on_it(monkeypatch):
    _quiesce_wire(monkeypatch)
    jc, _calls = _swept(monkeypatch, _live_rec(NOW + 3 * 3600),
                        instances=[_inst(status="running", dph=0.8407)],
                        now=NOW + 3600)
    ev = [f for name, f in (jc.get("ladder_journal") or [])
          if name == "jobs_retained_box_resurrected"]
    assert ev, "a retained box came back to life and `fleet log` said nothing"
    assert ev[0]["dph"] == pytest.approx(0.8407)
    assert ev[0]["live_multiple"] == pytest.approx(20.6, abs=0.1)


def test_a_live_retained_box_is_never_destroyed_by_the_sweep(monkeypatch):
    """Re-parking is not destroying. The disk is the whole reason the box is
    being held, and a human mid-salvage must not lose it."""
    _quiesce_wire(monkeypatch)
    dl = NOW + 3 * 3600
    jc, calls = _swept(monkeypatch, _live_rec(dl, status="expired"),
                       instances=[_inst(status="running")],
                       now=dl + 10 * 3600)
    assert "destroy" not in [c[0] for c in calls]


def test_the_re_park_is_bounded_and_then_only_alarms(monkeypatch):
    """A host that keeps re-placing the instance is not winnable by PUT-ing at
    it every 45s. Past RETENTION_REQUIESCE_MAX the ladder stops acting and the
    standing alarm owns it."""
    seen = _quiesce_wire(monkeypatch)
    rec = _live_rec(NOW + 3 * 3600, requiesces=retention.RETENTION_REQUIESCE_MAX)
    jc, _calls = _swept(monkeypatch, rec, now=NOW + 3600,
                        instances=[_inst(status="running", dph=0.8407)])
    assert seen == {"stop": [], "bid": []}
    assert jc["retained_boxes"][0]["requiesces"] == retention.RETENTION_REQUIESCE_MAX
    ev = [f for name, f in (jc.get("ladder_journal") or [])
          if name == "jobs_retained_box_resurrected"]
    assert "destroy or salvage it by hand" in ev[0]["note"]


def test_a_box_the_ladder_never_quiesced_is_alarmed_but_not_touched(monkeypatch):
    """No `quiesce` record means someone else owns this box's state — a record
    written before this change, or a hand-managed retention. Alarm, hands off."""
    seen = _quiesce_wire(monkeypatch)
    rec = _rec(NOW + 3 * 3600)                       # no `quiesce` key at all
    jc, _calls = _swept(monkeypatch, rec, now=NOW + 3600,
                        instances=[_inst(status="running")])
    assert seen == {"stop": [], "bid": []}
    assert jc["retained_boxes"][0]["live_since_ts"] == NOW + 3600
    ev = [f for name, f in (jc.get("ladder_journal") or [])
          if name == "jobs_retained_box_resurrected"]
    assert "never quiesced by the ladder" in ev[0]["note"]


def test_the_liveness_flag_retracts_itself_when_the_box_goes_back_down(
        monkeypatch):
    """Derived, not latched: the alarm has to disappear on its own when the
    re-park lands, or an operator ends up acking a condition that fixed itself."""
    _quiesce_wire(monkeypatch)
    rec = _live_rec(NOW + 3 * 3600)
    _swept(monkeypatch, rec, instances=[_inst(status="running", dph=0.8407)],
           now=NOW + 3600)
    assert rec["live_since_ts"] == NOW + 3600
    _swept(monkeypatch, rec, instances=[_inst(status="exited")], now=NOW + 3700)
    assert "live_since_ts" not in rec
    assert rec["resurrections"] == 1               # the COUNT is not retracted


def test_one_resurrection_journals_once_not_once_per_tick(monkeypatch):
    """45s ticks over a 3h window is 240 identical lines — a log nobody reads."""
    _quiesce_wire(monkeypatch)
    rec = _live_rec(NOW + 3 * 3600)
    lines = 0
    for i in range(5):
        jc, _c = _swept(monkeypatch, rec, now=NOW + 3600 + i * 45,
                        instances=[_inst(status="running", dph=0.8407)])
        lines += len([1 for name, _f in (jc.get("ladder_journal") or [])
                      if name == "jobs_retained_box_resurrected"])
    assert lines == 1
    assert rec["requiesces"] == 1


def test_a_failed_re_park_is_retried_up_to_the_bound(monkeypatch):
    """A `stop` that 500s must not be a one-shot — but it must not be an
    unbounded retry loop either."""
    monkeypatch.setattr(lifecycle, "_put_state_soft", lambda i, s: (False, "HTTP 500"))
    monkeypatch.setattr(lifecycle, "_put_bid_soft", lambda i, p: (False, "HTTP 500"))
    rec = _live_rec(NOW + 3 * 3600)
    for i in range(6):
        _swept(monkeypatch, rec, now=NOW + 3600 + i * 45,
               instances=[_inst(status="running", dph=0.8407)])
    assert rec["requiesces"] == retention.RETENTION_REQUIESCE_MAX


# --- fleet status / fleet log tell the truth -------------------------------- #
def test_fleet_status_alarms_on_a_live_retained_box():
    state = {"watches": {"41": {"replacement": {"retained_boxes": [
        dict(_live_rec(NOW + 3 * 3600), live_since_ts=NOW, live_dph=0.8407,
             live_cost_usd=0.14, live_multiple=20.6, requiesces=1)]}}}}
    alarms = fleetd.retention_alarms(state, NOW + 600)
    assert len(alarms) == 1
    key, msg = alarms[0]
    assert key == "retention:41:live"
    assert "RETAINED box is RUNNING again" in msg
    assert "$0.8407/hr" in msg and "20.6x" in msg


def test_fleet_status_does_not_alarm_on_a_retained_box_that_is_asleep():
    state = {"watches": {"41": {"replacement": {"retained_boxes": [
        _live_rec(NOW + 3 * 3600)]}}}}
    assert fleetd.retention_alarms(state, NOW + 600) == []


def test_fleet_status_rows_carry_the_live_rate_not_just_the_disclosed_one():
    """The disclosed `est_cost_usd` is storage. A row that shows only that,
    on a box billing GPU rate, repeats the reassurance the incident hid behind."""
    state = {"watches": {"41": {"replacement": {"retained_boxes": [
        dict(_live_rec(NOW + 3 * 3600), live_since_ts=NOW, live_dph=0.8407,
             live_cost_usd=0.14)]}}}}
    row = fleetd.retention_rows(state, NOW + 600)[0]
    assert row["live_dph"] == pytest.approx(0.8407)
    assert row["live_cost_usd"] == pytest.approx(0.14)
    assert row["live_since"].endswith("Z")


def test_jobs_replaced_no_longer_claims_a_retained_box_was_destroyed():
    """The note said "the old box destroyed" unconditionally — false on every
    default-configuration replacement since retention shipped 2026-08-05, and
    on 2026-08-16 it is why a box that was still alive read, in the only log an
    operator checks, as a box that no longer existed."""
    ret = {"41": _rec(NOW + 3 * 3600)}
    status, clause = fleetd._retention_fate(ret, "41")
    assert status == "retained"
    assert "RETAINED" in clause and "destroy" not in clause.lower()

    ret = {"41": dict(_rec(NOW), status="destroyed")}
    assert fleetd._retention_fate(ret, "41")[1] == "the old box destroyed"


def test_an_unknown_old_box_fate_is_vague_not_confidently_wrong():
    """An SLA relaunch or a pull-reschedule writes no retention record. Saying
    "handed off" is honest; naming an outcome we cannot see is how this bug
    was written in the first place."""
    assert fleetd._retention_fate({}, "41") == (None, "the old box handed off")


# --------------------------------------------------------------------------- #
# The eval-env pin across an automatic re-rent (2026-08-16).
# --------------------------------------------------------------------------- #

EEV = "20260816-1813-3c0a5f5b"


def _inst_pinned(**kw):
    """A primary rented by `launch_jobs_box.sh`: the eval-env pin is in its
    launch env, which is where jobd's boot-time fetch reads it from."""
    return dict(_inst(**kw), extra_env=[["EVAL_ENV_VER", EEV]])


def test_replacement_inherits_the_eval_env_pin(monkeypatch):
    """MEASURED 2026-08-16. Box 47887414 was rented WITH the pin, outbid before
    it ran, and the replacement came up with `env=[]` — so jobd fetched
    eval-env/LATEST (20260807-0503-84d35a08) while the queued job pinned
    20260816-1813-3c0a5f5b, and both E3 legs died rc 6 on the content-identity
    gate after paying for boot. The pin is part of the box's SHAPE, exactly
    like the image and the disk this lane already inherits."""
    jc, _hf = _jc(instances=[_inst_pinned()])
    calls = _launch_env(monkeypatch)
    replacement._launch_job_replacement(jc, [7], offer={"id": 5, "min_bid": 0.5,
                                              "dph_total": 0.7}, rental="bid")
    ns = [c for c in calls if c[0] == "do_launch"][0][1]
    assert f"EVAL_ENV_VER={EEV}" in ns.env


def test_no_pin_on_the_primary_stays_no_pin(monkeypatch):
    """Never invent a version. An unpinned replacement resolves LATEST and dies
    LOUDLY on the content gate; a wrong-but-plausible pin is the failure that
    gate exists to catch, so guessing would be strictly worse than not."""
    jc, _hf = _jc(instances=[_inst()])            # no extra_env at all
    calls = _launch_env(monkeypatch)
    replacement._launch_job_replacement(jc, [7], offer={"id": 5, "min_bid": 0.5,
                                              "dph_total": 0.7}, rental="bid")
    ns = [c for c in calls if c[0] == "do_launch"][0][1]
    assert not [s for s in ns.env if s.startswith("EVAL_ENV_VER=")]


def test_only_the_allowlisted_keys_are_inherited():
    """A copy of the primary's env would carry per-instance state (image digest
    stamps, B2 key nonces, handoff epochs) onto a different box. The allowlist
    is the point of the helper."""
    primary = {"extra_env": [["EVAL_ENV_VER", EEV],
                             [imageref.IMAGE_DIGEST_ENV, "sha256:deadbeef"],
                             ["B2_APPLICATION_KEY", "should-never-travel"]]}
    env = replacement._inherited_launch_env(primary)
    assert env == [f"EVAL_ENV_VER={EEV}"]


def test_the_watch_anchor_supplies_the_pin_when_the_primary_is_gone(monkeypatch):
    """RECURRED 2026-08-17: v13chain arm A died rc 6 twice across a 3-hop eviction
    chain. `_launch_job_replacement` reads the primary through
    `_job_primary_shape(jctx, None)`, and an EVICTED box has left the tick
    snapshot — so `primary` is {} precisely when a replacement is due and the
    inheritance had nothing to read. The watch anchor is that same box's own
    launch value, recorded while it was still visible."""
    jc, _hf = _jc(instances=[], launch_env_pin={"EVAL_ENV_VER": EEV})
    calls = _launch_env(monkeypatch)
    replacement._launch_job_replacement(jc, [7], offer={"id": 5, "min_bid": 0.5,
                                              "dph_total": 0.7}, rental="bid")
    ns = [c for c in calls if c[0] == "do_launch"][0][1]
    assert f"EVAL_ENV_VER={EEV}" in ns.env


def test_the_launch_env_anchor_never_captures_a_secret():
    """The anchor is persisted to state.json, so it is an allowlist projection and
    not a copy of extra_env."""
    inst = _inst_pinned()
    inst["extra_env"] = list(inst["extra_env"]) + [
        ["B2_APPLICATION_KEY", "must-never-be-persisted"],
        ["HF_TOKEN", "must-never-be-persisted"],
        [imageref.IMAGE_DIGEST_ENV, "sha256:deadbeef"]]
    assert replacement.launch_env_pin_from(inst) == {"EVAL_ENV_VER": EEV}


def test_the_launch_env_anchor_is_durable_across_a_fleetd_restart():
    """Without this the daemon reloads a watch mid-chain and loses the pin, which
    is the same failure one restart later."""
    assert "launch_env_pin" in fleetd.REPLACEMENT_STATE_KEYS


def test_explicit_extra_wins_over_the_inherited_value():
    """A caller that names a key itself has decided it; inheritance fills gaps,
    it does not override."""
    primary = {"extra_env": [["EVAL_ENV_VER", EEV]]}
    env = replacement._inherited_launch_env(primary, ("EVAL_ENV_VER=20260807-0503-84d35a08",))
    assert env == ["EVAL_ENV_VER=20260807-0503-84d35a08"]
