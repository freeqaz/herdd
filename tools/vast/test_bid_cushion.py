#!/usr/bin/env python3
"""test_bid_cushion — the SURVIVAL CUSHION rail on every bid we place.

The defect these pin (AUTOBID_AUDIT_2026-08-08.md §2): the old target was

    min(BID_TARGET_MULT x floor, max_bid, on_demand - $0.001)

so whenever the on-demand reference sat just above the floor the `min` picked
the SECOND term and handed back a bid one rounding unit over the floor — the
lowest-priority bid the market can hold — while the launch banner printed a
"1.2x floor" cushion. Four measured instances, every one of them dead:

  * 46848347  $0.747 over a $0.746 floor   (v7 eviction 1; two hand-rescues)
  * 46909754  $1.071 over a $1.0667 floor  (q6 understudy; dead in 45 min)
  * 46880245  $0.401 over a $0.400 floor   (v8; outbid before it finished booting)
  * v11 resume  $1.043 over a $1.040 floor (banner: "1.2x floor $1.040")

Every test here FAILS on the pre-2026-08-08 arithmetic; each one names the
number the old formula produced so the regression is legible without git.
"""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bidpolicy as bp  # noqa: E402
import herdd  # noqa: E402


# --------------------------------------------------------------------------- #
# the four real incidents, replayed with their TRUE on-demand price
# --------------------------------------------------------------------------- #
def test_v7_eviction1_standing_bid_and_the_defend_rebid_that_answers_the_spike():
    """46848347: floor $0.746, true on-demand ~$1.50 (the $0.748 in the incident
    record was the BID-view dph_total, doc 50 R1). The floor then spiked to
    $0.9099 and we lost the box because our standing bid was $0.747.

    Pre-2026-08-08: min(1.2 x 0.746, 1.50 - 0.001) = $0.895 with the true od,
    but $0.746 with the misread one — and the misread one is what shipped.
    2026-08-08 hardening: the 2.00x preference hit the 0.65 x od cost cap =
    $0.975, ABOVE the post-spike floor — the standing margin alone absorbed it.
    2026-08-09 OWNER ruling (recalibration decision 3): back to a 1.20x
    market-anchored bid = $0.895. That standing bid does NOT clear the $0.9099
    spike — and that is the ACCEPTED trade: margin over the floor is paid every
    hour, so spike-absorption moved from the standing bid to the DEFEND re-bid,
    which at the post-spike floor prices at the survival cushion ($1.001) and
    holds the box if the poll catches the climb. A spike that lands between
    polls costs an eviction + the recovery ladder, by choice."""
    bid = bp._bid_target(0.746, None, 1.50)
    assert bid == 0.895                       # 1.2 x floor; cost cap not binding
    assert bid / 0.746 >= bp.BID_MIN_CUSHION_MULT
    assert bid < 0.9099, (
        "at a 1.20x margin the v7 spike DOES displace the standing bid — if this "
        "starts passing the multiple was raised; re-read decision 3 before keeping it")
    rebid = bp._bid_target(0.9099, None, 1.50)
    assert rebid == 1.001                     # cushion-priced: 1.10 x post-spike floor
    assert rebid > 0.9099, "the defend re-bid must clear the post-spike floor"


def test_v11_resume_box_gets_a_real_cushion_at_the_true_ondemand_price():
    """v11 resume box: banner read `auto bid price $1.043 = 1.2x floor $1.040,
    capped below on-demand $1.044`. The true on-demand price of that offer was
    $2.137; $1.044 was the bid-view dph_total. The point of this test is the
    FLOOR-RELATIVE number: the incident bid $1.0029x the floor while its banner
    CLAIMED 1.2x; with the true od the policy now delivers the full 1.20x it
    prints (the 2026-08-08..09 interval priced this at the 0.65 x od cost cap,
    $1.389 = 1.336x; the 2026-08-09 ruling returned to the market anchor)."""
    bid = bp._bid_target(1.040, None, 2.137)
    assert bid == 1.248                       # 1.2 x floor; cap (1.389) not binding
    assert round(bid / 1.040, 3) == 1.2
    assert bid / 1.040 >= bp.BID_MIN_CUSHION_MULT


@pytest.mark.parametrize("floor,od,old_bid", [
    (0.746, 0.748, 0.747),                    # 46848347, as actually priced
    (1.0667, 1.0719, 1.071),                  # 46909754 understudy
    (0.400, 0.401, 0.400),                    # 46880245
    (1.040, 1.044, 1.043),                    # v11 resume box
])
def test_an_unsatisfiable_cushion_is_an_ESCALATION_not_a_near_ondemand_bid(
        floor, od, old_bid):
    """**Doctrine reversal, 2026-08-09 (recalibration item A).** When on-demand
    genuinely sits within the cushion of the floor there is no cushion to be had.

    Until 2026-08-09 the rule was "take ALL the remaining room" — bid
    `on_demand - EPS` — justified as deterrence: at that price no rational bidder
    outbids us. That argument was about *rank* and ignored *price*. A bid nobody
    outbids, at 99.9% of on-demand, is an on-demand box bought through the
    preemptible queue: it carries the full eviction risk of spot (on-demand still
    outranks it, SPOT_DESIGN #6) at the full price of on-demand, and
    `spot_breakeven` says it never breaks even at ANY lifetime.

    So the hard ceiling (BID_CEILING_ONDEMAND_FRAC x on-demand) now refuses it and
    the decision ESCALATES: the price-only `_bid_target` is None, which holds a
    live box's standing bid and routes a rescue/replacement to the on-demand rung.

    These four rows are the incident numbers as observed, with the (wrong)
    bid-view on-demand reference. Their primary FIX is still the on-demand
    REFERENCE (`_offer_ondemand_ref`), not this rail — but had the reference stayed
    wrong, this rail would now refuse the bid rather than place the razor-thin one
    that killed all four boxes. `old_bid` is kept so the regression stays legible."""
    dec = bp.bid_decision(floor, None, od)
    assert dec.price is None, f"a bid was emitted where the old code put ${old_bid}"
    assert dec.escalate is True
    assert "escalate_over_ceiling" in dec.reason
    assert dec.ceiling == pytest.approx(
        round(bp.BID_CEILING_ONDEMAND_FRAC * od, 3), abs=1e-9)
    assert bp._bid_target(floor, None, od) is None      # the price-only alias


# --------------------------------------------------------------------------- #
# the rail itself
# --------------------------------------------------------------------------- #
def test_cushion_outranks_the_cost_cap_when_ondemand_leaves_room():
    """floor $0.60 / on-demand $1.00. The 0.65 x on-demand COST cap prices this at
    $0.65; the survival cushion wants 1.10 x 0.60 = $0.66 and RAISES over the cost
    cap, landing under the $0.75 hard ceiling.

    This is the whole reason the cushion outranks the cost cap: a bid that cannot
    survive is not cheap, it is a 12-15 minute setup bill for nothing."""
    assert bp._bid_target(0.60, None, 1.00) == 0.66


def test_the_cushion_may_raise_only_up_to_the_hard_ceiling():
    """Recalibration item A: rails 2 and 3 compose badly by construction — the
    cushion is a survival rail and outranks the cost rail, so on a tight machine
    it was the only thing setting the price and nothing but `on_demand - EPS` sat
    under it. That is the 47214941 precedence defect, and it is real on genuine
    floors, not only on the self-referential ones.

    floor $0.90 / on-demand $1.00 (floor at 90% of on-demand). The cushion wants
    $0.99, the hard ceiling is $0.75. OLD emitted $0.99 — 99% of on-demand for a
    preemptible box. NEW escalates."""
    dec = bp.bid_decision(0.90, None, 1.00)
    assert dec.price is None and dec.escalate is True
    assert dec.ceiling == 0.75
    assert "structurally unsafe" in dec.reason


def test_the_escalation_boundary_is_where_the_cushion_crosses_the_ceiling():
    """The exact frontier, stated as arithmetic so a constants move relocates it
    visibly: escalation begins when `BID_MIN_CUSHION_MULT x floor` passes
    `BID_CEILING_ONDEMAND_FRAC x on_demand`, i.e. at floor/on-demand =
    0.75/1.10 = 0.6818. Below it we bid; at or above it we escalate.

    The measured floor/on-demand distribution is 0.36-0.53 across 51 real bid
    records (AUTOBID_AUDIT §2), so the ordinary machine is nowhere near this line
    — which is why the rail costs us almost no boxes and catches exactly the
    tight ones where spot buys nothing."""
    od = 2.00
    frontier = bp.BID_CEILING_ONDEMAND_FRAC / bp.BID_MIN_CUSHION_MULT
    assert round(frontier, 4) == 0.6818
    assert bp.bid_decision(round(od * (frontier - 0.02), 4), None, od).price is not None
    assert bp.bid_decision(round(od * (frontier + 0.02), 4), None, od).escalate is True


def test_cheap_machine_takes_the_full_preference_multiple():
    """floor 25% of on-demand: 1.2 x floor = 0.30 x on-demand, far under the
    0.65 cost cap, so the preference rung binds. (The 2026-08-08..09 2.00x
    interval bid $0.50 here; the 2026-08-09 ruling returned to the market
    anchor — a 90%-off machine now gets bid at 1.2x its floor, not 2x.)"""
    assert bp._bid_target(0.25, None, 1.00) == 0.30


def test_typical_machine_is_priced_by_the_MULTIPLE_not_the_cost_cap():
    """The measured regime: floor/on-demand 0.36-0.53 across 51 real bid records.
    Since the 2026-08-09 return to 1.20x, the multiple prices this ENTIRE range
    (1.2 x 0.53 = 0.636 < 0.65) and the cost cap binds only on the expensive
    tail (floor/od > ~0.54). This machine sits at floor/od 0.44: bid = 1.2 x
    floor, and the cap ($1.56) stays dormant."""
    bid = bp._bid_target(1.0667, None, 2.40)
    assert bid == 1.28                        # 1.2 x 1.0667, not 0.65 x 2.40
    assert round(bid / 1.0667, 2) == 1.20
    # the cap DOES bind past the crossover (floor/od 0.54-0.59): at 0.55 the
    # preference (1.584) exceeds the $1.56 cap and the cushion (1.452) fits under
    capped = bp._bid_target(1.32, None, 2.40)
    assert capped == 1.56                     # 0.65 x 2.40
    # past floor/od ~0.591 the SURVIVAL cushion outranks the cap (audit §2)
    cushioned = bp._bid_target(1.44, None, 2.40)
    assert cushioned == round(bp.BID_MIN_CUSHION_MULT * 1.44, 3) == 1.584


def test_unknown_ondemand_keeps_the_conservative_multiple():
    """No on-demand read => no cost cap. Taking the 2.0x preference there would
    bid 1.2x on-demand on a machine whose floor is 60% of on-demand — pure waste.
    The unpriced path stays at 1.20x (unchanged from before the audit)."""
    assert bp._bid_target(0.20, None, None) == 0.24
    assert bp._bid_target(0.20, None, 0) == 0.24


def test_max_bid_still_binds_from_every_path():
    """Invariant SPOT_DESIGN §5.4: no action may ever exceed max_bid. The cushion
    RAISES, so it is the one rail that could have broken this."""
    assert bp._bid_target(1.0667, 1.20, 2.40) == 1.20
    # and when max_bid is under the floor entirely the target is unwinnable
    assert bp._bid_target(1.0667, 0.50, 2.40) is None


def test_bid_never_reaches_ondemand_from_any_rail():
    """Exhaustive-ish sweep of the floor/on-demand plane: the hard invariant must
    survive the two new raising rails."""
    for od in (0.05, 0.4, 1.0, 2.4, 4.46):
        for ratio in (0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95, 0.999, 1.05):
            floor = round(od * ratio, 4)
            bid = bp._bid_target(floor, None, od)
            if bid is None:
                continue
            assert bid < od, f"bid {bid} >= on-demand {od} (floor {floor})"


def test_no_path_emits_a_bid_over_the_hard_ceiling():
    """**Recalibration item A, the property.** Every rail, every entry point, every
    (floor, on-demand, max_bid, caller-cap) combination: nothing may be emitted
    above `effective_bid_ceiling`, and nothing may reach on-demand.

    This is the invariant the survival cushion broke — on 47214941 it emitted
    $2.818 (and the ladder reached $3.410) against a 0.75 x $3.876 = $2.907 line —
    so the sweep deliberately covers the tight end of the plane where the cushion
    is the binding rail.

    The ceiling is written out longhand from the CONSTANT rather than read back
    from `effective_bid_ceiling`: a property test that asks the code under test
    where its own line is cannot fail when that line moves, and this one has to
    fail if anybody neuters the helper."""
    for od in (0.05, 0.4, 1.0, 2.4, 3.876, 4.46):
        ceiling = min(round(bp.BID_CEILING_ONDEMAND_FRAC * od, 3),
                      round(od - bp.BID_ONDEMAND_EPS, 3))
        assert bp.effective_bid_ceiling(od) == ceiling
        for ratio in (0.05, 0.2, 0.4, 0.5, 0.6, 0.65, 0.68, 0.7, 0.8, 0.95,
                      0.999, 1.05):
            floor = round(od * ratio, 4)
            for max_bid in (None, round(0.5 * od, 3), round(2.0 * od, 3)):
                for cap in (None, round(0.55 * od, 3)):
                    dec = bp.bid_decision(floor, max_bid, od, ondemand_cap=cap)
                    if dec.price is None:
                        continue
                    assert dec.price <= ceiling + 1e-9, (
                        f"floor {floor} od {od} max_bid {max_bid} cap {cap} -> "
                        f"${dec.price} over the ${ceiling} hard ceiling "
                        f"({dec.reason})")
                    assert dec.price < od, (
                        f"${dec.price} >= on-demand ${od} — strictly dominated")


def test_the_47214941_shape_the_cushion_fits_under_the_ceiling():
    """The incident's own numbers, the PASSING half. On-demand $3.876, preferred
    ceiling 0.75 x 3.876 = **$2.907**, and the bid the cushion wanted at the second
    rung was **$2.818**. That one is UNDER the ceiling, so it is a legal bid and
    the clamp must not refuse it — the defect was never "the cushion" but the
    absence of anything under the cushion.

    (The floor it was computed from was our own standing bid read back — the #73
    self-floor defect, fixed separately. This test is about the ceiling arithmetic
    only, so it feeds the number as a genuine floor.)"""
    od = 3.876
    assert bp.effective_bid_ceiling(od) == 2.907
    floor = round(2.818 / bp.BID_MIN_CUSHION_MULT, 4)      # 2.5618
    dec = bp.bid_decision(floor, None, od)
    assert dec.escalate is False
    assert dec.price == pytest.approx(2.818, abs=0.002)
    assert dec.price <= dec.ceiling
    # and the rung ABOVE it — the one the ladder actually reached — is refused
    assert bp.bid_decision(round(3.410 / bp.BID_MIN_CUSHION_MULT, 4), None,
                           od).escalate is True


def test_cushion_holds_whenever_the_ceiling_leaves_room_for_it():
    """The rail stated as a property: if 1.10 x floor fits under the hard ceiling,
    the emitted bid clears 1.10 x floor. This is the assertion the old formula
    violated on every machine whose on-demand reference sat near its floor.

    The skip condition moved from `on_demand - EPS` to the hard ceiling on
    2026-08-09: above that line the answer is no bid at all (escalation), not a
    smaller cushion."""
    for od in (0.4, 1.0, 2.4, 4.46):
        for ratio in (0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9):
            floor = round(od * ratio, 4)
            if bp.BID_MIN_CUSHION_MULT * floor > bp.effective_bid_ceiling(od):
                assert bp.bid_decision(floor, None, od).escalate is True
                continue                      # cushion over the ceiling -> escalate
            bid = bp._bid_target(floor, None, od)
            assert bid is not None
            assert bid >= round(bp.BID_MIN_CUSHION_MULT * floor, 3) - 1e-9, (
                f"floor {floor} / od {od} -> {bid} is inside the cushion")


# --------------------------------------------------------------------------- #
# launch price == steady-state target (SPOT_DESIGN §3.2 P2)
# --------------------------------------------------------------------------- #
def test_launch_price_is_the_same_function_as_the_defend_target():
    """`_auto_bid_price` used to be a hand-synced COPY of `_bid_target`. If the
    cushion had landed only in the launch copy, the decay ladder would have walked
    every freshly cushioned bid back down to the old target within BID_DECAY_POLLS
    ticks — the fix would have been silently undone in ~3 minutes."""
    for floor, od in ((0.746, 1.50), (1.0667, 2.40), (0.20, None),
                      (0.90, 1.00), (0.080, 0.082), (0.28, 0.30)):
        assert herdd._auto_bid_price(floor, od) == bp._bid_target(floor, None, od)


def test_a_freshly_launched_bid_is_not_a_decay_candidate():
    """The same invariant from the ladder's side: a box launched at
    `_auto_bid_price` must not be seen as stale-high on its first poll."""
    floor, od = 1.0667, 2.40
    launch = herdd._auto_bid_price(floor, od)
    st = bp.mk_poll_state(present=True, actual_status="running",
                          market_min_bid=floor, on_demand=od, last_bid=launch)
    assert bp._decay_candidate(st) is False


# --------------------------------------------------------------------------- #
# the preferred ceiling must stay above the target, or it latches fleet-wide
# --------------------------------------------------------------------------- #
def test_preferred_ceiling_sits_above_the_standing_bid_target():
    """A preferred ceiling BELOW the target would make every freshly launched box
    breach it on its first tick (latching `bid_over_preferred_ceiling` fleet-wide)
    and would dead-arm handoff, whose candidate filter requires a candidate target
    at or under this line. OLD: ceiling 0.50 x od vs a 0.65 x od target."""
    assert bp.BID_CEILING_ONDEMAND_FRAC > bp.BID_TARGET_ONDEMAND_FRAC
    od = 2.40
    for ratio in (0.1, 0.3, 0.45, 0.5):
        floor = round(od * ratio, 4)
        st = bp.mk_poll_state(present=True, actual_status="running",
                              on_demand=od,
                              last_bid=bp._bid_target(floor, None, od))
        over, _pref = bp._preferred_ceiling_alarm(st)
        assert over is False, f"fresh launch at floor/od={ratio} breaches the ceiling"


def test_replace_min_cushion_is_bound_to_the_launch_cushion():
    """One number, two sides. A drift between them would let the replacement
    ladder rent a bid the target function is not allowed to place."""
    assert bp.REPLACE_MIN_CUSHION == bp.BID_MIN_CUSHION_MULT
