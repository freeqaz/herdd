#!/usr/bin/env python3
"""test_spot_breakeven_wiring — the livelock trigger, from advisory to structural.

`bidpolicy.spot_breakeven` shipped 2026-08-06 as a pure function that nothing
called ("Advisory — not wired into any ladder yet"), and its own docstring named
the wiring point: `replacement_decision`, fed the lane's OBSERVED inter-eviction
lifetime from `replacement_history`.

Why it matters, in the lane's own numbers: the v11 chat arm took four spot
evictions with 11-13 minute realised lifetimes against an 11m35s setup, and one
full cycle moved the banked-row count from 40 to 40. Nothing about the PRICES
was wrong there — `thin`, `inverted` and the cushion rails all compare prices,
and all four of them were satisfied. Only a cost-per-USEFUL-hour comparison can
see that shape, and until 2026-08-08 the ladder did not make one; it re-rented
spot and bought the same cycle again. The operator broke the loop by hand with
`--max-replacements 0` and a manual on-demand box.

Every test here fails on the pre-2026-08-08 code (`replacement_decision` had no
`observed_lifetime_h`/`setup_h` parameters and `_job_observed_lifetime_h` did
not exist).
"""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bidpolicy as bp  # noqa: E402
import herdd as v  # noqa: E402


def _dec(**kw):
    # a market where every PRICE rail is satisfied: floor $1.00, on-demand
    # $3.00, so the bid is cushioned ~1.95x and nothing is thin or inverted.
    base = dict(eviction_class=bp.EVICTION_UNKNOWN, replacements_used=0,
                budget_usd=20.0, spend_usd=1.0, launch_dph_anchor=1.60,
                offer_min_bid=1.00, offer_ondemand=3.00)
    base.update(kw)
    return bp.replacement_decision(**base)


def test_prices_alone_keep_the_ladder_on_spot():
    """The control: with no lifetime history the ladder behaves exactly as it
    did before, so the trigger is additive and cannot be blamed for a rung it
    did not choose."""
    d = _dec()
    assert (d.action, d.rental) == ("rent", "bid")
    assert "LOSES per useful hour" not in d.reason


def test_a_livelock_lifetime_flips_the_rung_to_ondemand():
    """The v11 shape: 12-minute realised lifetime against 11m35s of setup. Spot
    delivers ~0.007 h of useful work per cycle at full price — its cost per
    useful hour is ~30x on-demand, and no bid changes that."""
    d = _dec(observed_lifetime_h=0.20, setup_h=bp.SPOT_SETUP_H)
    assert (d.action, d.rental) == ("rent", "ondemand")
    assert "LOSES per useful hour" in d.reason
    assert "0.20h" in d.reason and "0.19h" in d.reason


def test_a_lifetime_at_or_under_setup_is_the_hard_livelock():
    d = _dec(observed_lifetime_h=bp.SPOT_SETUP_H, setup_h=bp.SPOT_SETUP_H)
    assert d.rental == "ondemand"


def test_a_healthy_lifetime_leaves_the_ladder_on_spot():
    """2.5 h of realised life at 1/3 of on-demand: spot wins comfortably per
    useful hour and the trigger must stay silent."""
    d = _dec(observed_lifetime_h=2.5, setup_h=bp.SPOT_SETUP_H)
    assert (d.action, d.rental) == ("rent", "bid")


def test_an_unknown_lifetime_falls_back_to_the_prior_not_to_silence():
    """**Changed 2026-08-09 (recalibration item B).** It used to read "an unknown
    lifetime can never fire the trigger", by the same doctrine
    `offer_ondemand=None` follows. That was right about an unknown MARKET and
    wrong here, because the lifetime is not unknown in the same sense: it is
    simply not yet MEASURED, and it stays unmeasured until this lane has rented
    and lost a spot box. Since `_job_observed_lifetime_h` only counts DEAD BID
    REPLACEMENTS, the rung was structurally dead on the FIRST eviction — which is
    the only eviction most watches ever have, and is exactly the 2026-08-08
    moment where the NO-GO had to be made by a human.

    So an absent observation now falls back to SPOT_PRIOR_LIFETIME_H and the
    decision says which it ran on. `setup_h` is still mandatory: it is a MEASURED
    lane property, not an assumption, and without it there is no arithmetic.

    At these prices (floor 1/3 of on-demand, bid at the 0.65 cost cap) the prior
    does NOT fire, which is the point of where it was set — see
    `test_the_prior_fires_on_a_first_eviction_above_the_ratified_band`."""
    assert _dec(setup_h=bp.SPOT_SETUP_H).lifetime_basis == "prior"
    assert _dec(observed_lifetime_h=0.20).lifetime_basis is None   # no setup_h
    for kw in ({}, {"observed_lifetime_h": 0.20}, {"setup_h": bp.SPOT_SETUP_H},
               {"observed_lifetime_h": None, "setup_h": bp.SPOT_SETUP_H}):
        assert _dec(**kw).rental == "bid", kw
    # and an explicitly disabled prior restores the pre-2026-08-09 behaviour
    assert _dec(setup_h=bp.SPOT_SETUP_H,
                prior_lifetime_h=0).lifetime_basis is None


def test_the_prior_fires_on_a_first_eviction_above_the_ratified_band():
    """Item B's whole point: a livelock verdict on the FIRST eviction, with no
    history at all.

    A tight machine — floor at 65% of a $2.00 on-demand rate — where the survival
    cushion prices the bid at 1.10 x 1.30 = $1.43, i.e. 0.715 x on-demand. That is
    above the 0.70 top of the owner-ratified band, and SPOT_PRIOR_LIFETIME_H is
    derived as the L_min of exactly that ratio (0.193/0.30 = 0.643 h), so the
    arithmetic reads: at the lifetime the ratified cost cap already assumes, a bid
    priced above that band loses per useful hour. Take the on-demand rung.

    The reason string must say ASSUMED, not measured — an escalation made on a
    prior that reads like one made on evidence is how a prior becomes a fact."""
    d = _dec(offer_min_bid=1.30, offer_ondemand=2.00, launch_dph_anchor=1.60,
             setup_h=bp.SPOT_SETUP_H)
    assert (d.action, d.rental) == ("rent", "ondemand")
    assert d.lifetime_basis == "prior"
    assert "LOSES per useful hour" in d.reason
    assert "ASSUMED" in d.reason and "SPOT_PRIOR_LIFETIME_H" in d.reason
    assert "OBSERVED" not in d.reason


def test_an_observation_always_overrides_the_prior():
    """Evidence outranks the assumption in BOTH directions — that is what makes
    the prior safe. Same tight machine: a measured 4 h lifetime says spot is fine
    here after all and the ladder goes back to spot, while a measured 12 min
    lifetime reaches the same on-demand verdict the prior did but records it as
    OBSERVED."""
    tight = dict(offer_min_bid=1.30, offer_ondemand=2.00, launch_dph_anchor=1.60,
                 setup_h=bp.SPOT_SETUP_H)
    healthy = _dec(observed_lifetime_h=4.0, **tight)
    assert (healthy.action, healthy.rental) == ("rent", "bid")
    assert healthy.lifetime_basis == "observed"
    livelocked = _dec(observed_lifetime_h=0.20, **tight)
    assert livelocked.rental == "ondemand"
    assert livelocked.lifetime_basis == "observed"
    assert "OBSERVED" in livelocked.reason and "ASSUMED" not in livelocked.reason


def test_the_prior_is_derived_from_the_ratified_band_not_from_the_median():
    """The derivation, pinned so a constants move relocates it visibly and so the
    number is never mistaken for a measurement.

    The measured realised spot lifetimes on this fleet have a median of ~0.20 h —
    at or below SPOT_SETUP_H, which `spot_breakeven` scores as the hard livelock
    at ANY price ratio. A prior set from that median would route every first
    eviction to the expensive rung, which is a fleet-wide spend decision and not
    one a prior gets to make. So the prior comes from what the POLICY already
    asserts: ratifying a 55-70% bid/on-demand band asserts that a box lives long
    enough for that band to pay, i.e. `setup / (1 - 0.70)`.

    The resulting firing threshold is the top of the band itself."""
    assert bp.SPOT_PRIOR_LIFETIME_H == pytest.approx(
        bp.SPOT_SETUP_H / (1.0 - bp.SPOT_PRIOR_BAND_TOP_FRAC), abs=5e-4)
    threshold = 1.0 - bp.SPOT_SETUP_H / bp.SPOT_PRIOR_LIFETIME_H
    assert threshold == pytest.approx(bp.SPOT_PRIOR_BAND_TOP_FRAC, abs=1e-3)
    # the prior is well ABOVE the measured median (0.20 h): its error direction
    # is spend, so it is deliberately set at the spot-favouring end
    assert bp.SPOT_PRIOR_LIFETIME_H > 3 * 0.20
    # and its live domain is a COMPLEMENT of item A's hard ceiling, not an
    # overlap: above 0.75 x on-demand nothing is ever priced at all
    assert threshold < bp.BID_CEILING_ONDEMAND_FRAC


def test_an_unknown_ondemand_price_can_never_fire_the_trigger():
    d = _dec(offer_ondemand=None, observed_lifetime_h=0.20,
             setup_h=bp.SPOT_SETUP_H)
    assert d.rental == "bid"


def test_the_checkpoint_loss_term_is_carried_through():
    """A lane that re-does half a checkpoint interval per eviction pays that on
    top of setup; at the margin it is what decides the rung."""
    base = dict(observed_lifetime_h=0.60, setup_h=bp.SPOT_SETUP_H)
    assert _dec(**base).rental == "bid"
    assert _dec(ckpt_interval_h=1.0, **base).rental == "ondemand"


# --------------------------------------------------------------------------- #
# the observed-lifetime input
# --------------------------------------------------------------------------- #
def test_observed_lifetime_is_the_median_of_dead_bid_replacements():
    jc = {"replacement_history": [
        {"rental": "bid", "ts": 0.0, "died_ts": 3600.0},      # 1.0 h
        {"rental": "bid", "ts": 0.0, "died_ts": 720.0},       # 0.2 h
        {"rental": "bid", "ts": 0.0, "died_ts": 1800.0},      # 0.5 h
    ]}
    assert v._job_observed_lifetime_h(jc) == pytest.approx(0.5)


def test_ondemand_rentals_and_live_boxes_are_excluded():
    """An on-demand box's lifetime says nothing about the spot market, and a box
    still running has no lifetime yet — counting it would read as a very short
    one and bias the ladder toward the expensive rung."""
    jc = {"replacement_history": [
        {"rental": "ondemand", "ts": 0.0, "died_ts": 60.0},
        {"rental": "bid", "ts": 0.0, "died_ts": None},
        {"rental": "bid", "ts": 100.0, "died_ts": 100.0},     # zero-length
    ]}
    assert v._job_observed_lifetime_h(jc) is None


def test_no_history_is_no_lifetime():
    assert v._job_observed_lifetime_h({}) is None
    assert v._job_observed_lifetime_h({"replacement_history": []}) is None


def test_the_v11_history_would_have_flipped_the_rung():
    """Replayed against the lane's real shape: 11-13 minute lifetimes, the
    trigger fires and the second replacement goes on-demand instead of buying
    the same cycle again."""
    jc = {"replacement_history": [
        {"rental": "bid", "ts": 0.0, "died_ts": 11 * 60.0},
        {"rental": "bid", "ts": 0.0, "died_ts": 13 * 60.0},
    ]}
    lt = v._job_observed_lifetime_h(jc)
    assert 0.18 <= lt <= 0.22
    assert _dec(observed_lifetime_h=lt, setup_h=bp.SPOT_SETUP_H).rental == "ondemand"


def test_setup_cost_is_the_measured_v11_number():
    assert bp.SPOT_SETUP_H == pytest.approx(695 / 3600.0, abs=5e-4)
