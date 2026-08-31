"""S2b — notification rows as EVIDENCE in the classifier and the rescue quote.

Spec: `NOTIFY_DESIGN.md` §6. This slice touches the two seams that took a
three-lane adversarial review last time (the eviction classifier and the money
path), so the tests are organised the way §6.6 says the review will be:

  1. **precedence and regressions** — every `notify=None` path byte-identical to
     pre-S2b (proved against a REFERENCE COPY of the old function over an
     exhaustive matrix, not against a handful of remembered cases), the arms a
     row may never outrank, and the two rows the captured feed proves exist:
     the one displaced BELOW our bid, and the box evicted twice in a night;
  2. **races, latches, seams** — match/latch/consume, return-to-live, box swap,
     restart, and a row that belongs to another box;
  3. **money-path rails** — no input combination emits a bid `_bid_target`
     would not have emitted, `rescue_attempted` unbypassable, and the refusal
     journaled as a refusal.

Plus the boundary that is the whole doctrine (D2): with no rows — endpoint
retired, switch off, or nothing matching — the daemon is byte-for-byte its S2a
self.

Every number is from the captured feed (`testfixtures/notify/`) or from the
2026-08-16 06:24Z field case in §2. No network, no clock dependence beyond a
`now` the tests own.
"""
from __future__ import annotations

import argparse
import collections
import copy
import itertools
import json
import math
import os
import pathlib
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bidpolicy as bp                                            # noqa: E402
import fleetd                                                     # noqa: E402
import notify                                                     # noqa: E402
import jobmeta                                                    # noqa: E402
from vastlib.boxes import lifecycle                               # noqa: E402
from vastlib.core import models as vl_models                      # noqa: E402
from vastlib.jobs import risk as jobs_risk                        # noqa: E402
from vastlib.market import pricing                                # noqa: E402
from vastlib.supervise import handoff, job_lane, replacement      # noqa: E402
from vastlib.supervise import journal as sup_journal              # noqa: E402
from test_fleetd import FakeHooks, journal                        # noqa: E402
from test_fleetd_notify import NotifyHooks, _envelope             # noqa: E402

_FIX = pathlib.Path(__file__).resolve().parent / "testfixtures" / "notify"


def _feed():
    with open(_FIX / "inbox_2026-08-16.json") as fh:
        return json.load(fh)["notifications"]


def _outbids():
    return [r for r in _feed() if r["notif_type"] == "outbid"]


def _row(iid):
    rows = [r for r in _outbids() if r["associated_id"]["instance_id"] == iid]
    assert rows, f"fixture has no outbid row for {iid}"
    return sorted(rows, key=lambda r: r["created_at"])


# --------------------------------------------------------------------------- #
# the field case, as numbers (NOTIFY_DESIGN §2)
# --------------------------------------------------------------------------- #
#: Instance 47845356, machine 56748, 2026-08-16. The inbox row at 06:24:21Z said
#: we were displaced at $1.00/hr against our $0.45. SEVENTEEN SECONDS LATER
#: fleetd's own market read said the chunk was still listed at $0.14 — below our
#: own standing bid — and `classify_eviction` returned `host_stop`. `fleet
#: report` scored the episode UNRESOLVED. Only one of those two numbers is a
#: price somebody paid.
FIELD = dict(iid="47845356", machine=56748,
             your_bid=0.45, new_min_bid=1.00,
             listing_floor_at_stop=0.14,      # what the offers read said
             on_demand=3.0)                   # ceiling anchor; not in the record

#: Instance 47840057, machine 56759: displaced with `your_bid 0.16` against
#: `new_min_bid 0.15`. A displacing price BELOW our own bid — an on-demand taker
#: or a host action. §6.1: not an outbid in OUR vocabulary, and not a price a
#: bid can beat, so it may mint no class and quote no rescue.
BELOW = dict(iid="47840057", your_bid=0.16, new_min_bid=0.15)

#: Instance 47833510 evicted TWICE in one night, 02:40:08Z ($0.96 -> $2.33) and
#: 04:48:47Z ($1.20 -> $1.44). The reason there is a consumed set as well as a
#: freshness window.
TWICE = dict(iid="47833510",
             first=dict(your_bid=0.96, new_min_bid=2.33),
             second=dict(your_bid=1.20, new_min_bid=1.44))


def _ev(iid, your_bid, new_min_bid, created_at, event_id="e" * 32,
        machine_id=56748):
    """A matched-evidence dict in exactly the shape the driver hands around."""
    return {"event_id": event_id, "iid": str(iid), "machine_id": machine_id,
            "your_bid": your_bid, "new_min_bid": new_min_bid,
            "created_at": created_at}


# =========================================================================== #
# LANE 1 — classifier precedence and regressions
# =========================================================================== #
def _classify_pre_s2b(*, present, actual_status=None, market_min_bid=None,
                      on_demand=None, last_bid=None, market_listed=None,
                      is_bid=None):
    """VERBATIM copy of `bidpolicy.classify_eviction` at 34a78e09, the S1+S2a
    merge — the commit S2b promised not to change with `notify=None`.

    A copy, not a git call: the invariant is "this behaviour", and a test that
    shells out to git to find out what the behaviour was would stop testing the
    day the file moves. When this function and the real one disagree, exactly
    one of them is wrong and the diff says which."""
    if not present:
        return bp.EVICTION_HOST_FAILURE
    if (actual_status or "").lower() in bp.LIVE_STATES:
        return bp.EVICTION_UNKNOWN
    od = on_demand if (on_demand and on_demand > 0) else None
    lb = last_bid if (last_bid and last_bid > 0) else None
    mmb = market_min_bid if (market_min_bid and market_min_bid > 0) else None
    od_claim_possible = is_bid is not False
    if mmb is not None and lb is not None and mmb > lb:
        return bp.EVICTION_OUTBID
    if (od_claim_possible and od is not None and lb is not None
            and lb >= od - bp.BID_ONDEMAND_EPS):
        return bp.EVICTION_ONDEMAND
    if mmb is None and market_listed is False:
        return bp.EVICTION_OUTBID
    if mmb is None:
        return bp.EVICTION_UNKNOWN
    if market_listed is True and lb is not None and lb >= mmb:
        return bp.EVICTION_HOST_STOP
    return (bp.EVICTION_ONDEMAND if (od is not None and od_claim_possible)
            else bp.EVICTION_UNKNOWN)


#: The classifier's whole input space, coarsened to the values that can change
#: an arm: absent / zero / below-our-bid / above-our-bid prices, both sides of
#: the on-demand clamp, and every tri-state.
_MATRIX = dict(
    present=(True, False),
    actual_status=(None, "running", "loading", "exited", "stopped"),
    market_min_bid=(None, 0.0, 0.14, 1.0, 2.81),
    on_demand=(None, 0.0, 1.0017, 3.0),
    last_bid=(None, 0.0, 0.45, 1.05, 2.55),
    market_listed=(None, True, False),
    is_bid=(None, True, False),
)


def _matrix_cases():
    keys = sorted(_MATRIX)
    for combo in itertools.product(*(_MATRIX[k] for k in keys)):
        yield dict(zip(keys, combo))


def test_every_notify_none_path_is_byte_identical_to_pre_s2b():
    """THE invariant, exhaustively. 9,000 input combinations; with no row, the
    new classifier IS the old one. Anything else and S2b changed a live fleet's
    behaviour the moment it merged, review or no review."""
    n = 0
    for case in _matrix_cases():
        n += 1
        assert bp.classify_eviction(**case) == _classify_pre_s2b(**case), case
        # ...and passing the argument EXPLICITLY as None is the same path as
        # not passing it at all (every pre-S2b caller does the latter).
        assert bp.classify_eviction(notify=None, **case) == \
            _classify_pre_s2b(**case), case
    assert n == 9000, f"the matrix shrank to {n} — did a dimension get dropped?"


def _ondemand_discriminator_fires(case):
    """The REAL on-demand arm: our standing bid was already at or above the
    machine's on-demand price and we lost the box anyway. Not the trailing
    fallback at the bottom of the function, which returns `ondemand` merely
    because an on-demand price is KNOWN."""
    od = case["on_demand"] if (case["on_demand"] and case["on_demand"] > 0) else None
    lb = case["last_bid"] if (case["last_bid"] and case["last_bid"] > 0) else None
    return (case["is_bid"] is not False and od is not None and lb is not None
            and lb >= od - bp.BID_ONDEMAND_EPS)


@pytest.mark.parametrize("row", [
    _ev(FIELD["iid"], 0.10, 9.99, 0.0),   # maximally "outbid-looking"
    _ev(FIELD["iid"], 0.10, 0.90, 0.0),   # ...and one UNDER every matrix od
])
def test_a_row_may_only_move_a_verdict_toward_outbid(row):
    """The precedence rule as a property, over the same matrix: with an
    outbid-looking row present, every verdict either stays put or becomes
    `outbid` — nothing else is reachable, in either direction.

    And the arms that must survive, survive:

      * wherever the ON-DEMAND DISCRIMINATOR fires (our bid was already at or
        above on-demand and we lost the box anyway), the row changes nothing.
        That is the class no bid can undo and the only class
        `replacement_decision` and the re-bid ladder branch on — lane 1's hunt,
        answered by construction rather than by sampling;
      * wherever the box is still LIVE, the row changes nothing either. That arm
        sits above the notify arm and was untested against a row until review
        round 1 (F5): a mutant returning `outbid` for live+supported-row survived
        all eight of the original lane-1 tests. It is unreachable from the
        shipped call sites today, which is exactly why it needs an assertion —
        the day a caller classifies a live box, this is the leg that catches it.

    Parametrized over TWO rows since round 1 (F1): the $9.99 one is above every
    on-demand price in the matrix, so with the `< on_demand` clause restored it
    is only supported where on-demand is unreadable. The $0.90 one sits under
    both real on-demand values, so the refinement leg stays exercised where the
    clamp is live."""
    seen = collections.Counter()
    for case in _matrix_cases():
        bare = bp.classify_eviction(**case)
        withrow = bp.classify_eviction(notify=row, **case)
        assert withrow in (bare, bp.EVICTION_OUTBID), (case, bare, withrow)
        if bare == bp.EVICTION_ONDEMAND and _ondemand_discriminator_fires(case):
            seen["discriminator"] += 1
            assert withrow == bp.EVICTION_ONDEMAND, case
        if bare == bp.EVICTION_HOST_FAILURE:
            seen["host_failure"] += 1
            assert withrow == bp.EVICTION_HOST_FAILURE, case
        if (bare == bp.EVICTION_UNKNOWN
                and (case["actual_status"] or "").lower() in bp.LIVE_STATES
                and case["present"]):
            seen["live"] += 1
            assert withrow == bp.EVICTION_UNKNOWN, case
        if bare != withrow:
            seen["refined"] += 1
    # the property is not vacuous on any of its four legs
    assert seen["discriminator"] and seen["host_failure"] and seen["live"] \
        and seen["refined"], seen


def test_a_row_at_or_above_on_demand_supports_nothing():
    """§6.1's second clause, dropped in the first cut and restored by review
    round 1 (F1/m5). Nobody rationally bids past on-demand — an on-demand claim
    outranks every interruptible bid at any price — so a displacing price at or
    above the machine's on-demand rate is not a genuine higher BIDDER. It is
    displacement of unknown class, exactly like a below-bid one.

    This is not a labelling nicety. `outbid` is in `EVICTED_TTL_CLASSES`, so
    minting it off such a row shortens the evicted-MACHINE exclusion from
    permanent to thirty minutes on a machine an on-demand taker may be holding
    (F2/M2), and the replacement probe then walks straight back into it."""
    args = dict(present=True, actual_status="exited", is_bid=True,
                market_min_bid=1.0, market_listed=True, last_bid=None,
                on_demand=3.0)
    assert bp.classify_eviction(**args) == bp.EVICTION_ONDEMAND
    for nmb, want in ((1.0, bp.EVICTION_OUTBID),      # genuine higher bidder
                      (2.9, bp.EVICTION_OUTBID),
                      (3.0, bp.EVICTION_ONDEMAND),    # AT on-demand: not a bid
                      (3.5, bp.EVICTION_ONDEMAND),
                      (99.0, bp.EVICTION_ONDEMAND)):
        row = {"your_bid": 0.45, "new_min_bid": nmb}
        assert bp.notify_outbid_supported(row, on_demand=3.0) is (nmb < 3.0)
        assert bp.classify_eviction(notify=row, **args) == want, nmb
        # ...and with NO on-demand read the predicate degrades to the
        # row-internal test alone, which is what shipped and what §6.7 documents
        assert bp.notify_outbid_supported(row) is True


def test_the_predicate_normalizes_prices_the_way_the_module_does():
    """F6/m4. `notify_outbid_supported` was a bare `nmb > yb` on raw floats, and
    the feed is a hidden endpoint parsed by `json.loads` — which mints `inf`
    from the string `"1e309"`. Every one of these shapes used to read as
    EVIDENCE."""
    assert bp.notify_outbid_supported({"your_bid": 0.0,
                                       "new_min_bid": 0.01}) is False
    assert bp.notify_outbid_supported({"your_bid": -1,
                                       "new_min_bid": 0.001}) is False
    assert bp.notify_outbid_supported({"your_bid": 0.45,
                                       "new_min_bid": float("inf")}) is False
    assert bp.notify_outbid_supported({"your_bid": 0.45,
                                       "new_min_bid": "1e309"}) is False
    assert bp.notify_outbid_supported({"your_bid": float("nan"),
                                       "new_min_bid": 9.0}) is False
    # a SUB-GRID raise is still a raise: the price grid is $0.001, and
    # `notify_rescue_floor` opens a whole cent above it anyway
    assert bp.notify_outbid_supported({"your_bid": 0.45,
                                       "new_min_bid": 0.4501}) is True
    # ...and the same normalization on the floor the rescue prices off
    assert bp.notify_rescue_floor(float("inf")) is None
    assert bp.notify_rescue_floor("1e309") is None
    assert bp.notify_rescue_floor(0.0) is None
    assert bp.notify_rescue_floor(-1) is None
    assert bp.notify_rescue_floor(1.0) == 1.0 + bp.BID_MIN_STEP
    # an infinite MARKET floor cannot poison the max() either
    assert bp.notify_rescue_floor(1.0, float("inf")) == 1.0 + bp.BID_MIN_STEP


def test_the_one_ondemand_verdict_a_row_may_refine_is_the_FALLBACK():
    """The exception, named out loud because §6.6 lane 1 will look for it.

    `classify_eviction` ends with a guess: nothing above matched, so if an
    on-demand price is merely KNOWN, answer `ondemand`. A row outranks THAT —
    §6.2 puts the notify arm above every listing-derived arm, and this one is
    weaker still. The consequence is real and deliberate: `ondemand` stops the
    re-bid ladder dead, so refining a fallback `ondemand` to `outbid` lets the
    (bounded, rail-clamped) re-bid rungs run on a box vast says was outbid.
    Direction of the other consumer is the safe one — `replacement_decision`
    prefers the EXPENSIVE on-demand rung only on `ondemand`, so this can only
    make that rung less reachable."""
    case = dict(present=True, actual_status="exited", market_min_bid=1.0,
                market_listed=None, on_demand=3.0, last_bid=None, is_bid=True)
    assert bp.classify_eviction(**case) == bp.EVICTION_ONDEMAND
    assert not _ondemand_discriminator_fires(case), "the FALLBACK, not the arm"
    row = _ev(FIELD["iid"], 0.45, 1.0, 0.0)
    assert bp.classify_eviction(notify=row, **case) == bp.EVICTION_OUTBID


def test_the_0624z_field_case_flips_from_host_stop_to_outbid():
    """NOTIFY_DESIGN §2, on this account, on an ordinary morning.

    The classifier saw a chunk still listed at a floor BELOW our own bid and
    concluded the host had stopped us. Vast's own record says we were displaced
    at $1.00/hr. This assertion is the entire point of S2b."""
    args = dict(present=True, actual_status="exited", is_bid=True,
                market_min_bid=FIELD["listing_floor_at_stop"],
                market_listed=True, last_bid=FIELD["your_bid"],
                on_demand=FIELD["on_demand"])
    assert bp.classify_eviction(**args) == bp.EVICTION_HOST_STOP
    row = _ev(FIELD["iid"], FIELD["your_bid"], FIELD["new_min_bid"], 0.0)
    assert bp.classify_eviction(notify=row, **args) == bp.EVICTION_OUTBID


def test_a_below_bid_row_mints_no_class():
    """Instance 47840057: `your_bid 0.16, new_min_bid 0.15`. Vast calls it an
    outbid; our vocabulary does not (§6.1), because the displacing price is
    below our own — which is what an on-demand taker or a host action looks
    like. Every verdict is exactly the one the row-less classifier gave."""
    row = _ev(BELOW["iid"], BELOW["your_bid"], BELOW["new_min_bid"], 0.0)
    assert bp.notify_outbid_supported(row) is False
    for case in _matrix_cases():
        assert bp.classify_eviction(notify=row, **case) == \
            bp.classify_eviction(**case), case


def test_a_row_never_outranks_the_ondemand_discriminator():
    """v7 eviction 2: a $1.05 bid on a box whose on-demand rate was $1.0017. No
    price wins that back, and a notification that says "outbid" must not
    relabel it — that reading is what made a supply problem look like a bidding
    problem for a whole run."""
    args = dict(present=True, actual_status="exited", market_min_bid=None,
                market_listed=False, on_demand=1.0017, last_bid=1.05,
                is_bid=True)
    assert bp.classify_eviction(**args) == bp.EVICTION_ONDEMAND
    row = _ev("1", 1.05, 2.50, 0.0)
    assert bp.classify_eviction(notify=row, **args) == bp.EVICTION_ONDEMAND


def test_a_row_never_reorders_the_risen_floor_arm():
    """v7 eviction 1: the floor went $0.7599 -> $0.9099 while our bid sat on the
    on-demand clamp. Risen floor is tested FIRST so that case reads `outbid` and
    not `ondemand`; a row must not be able to move it either way.

    WHAT THIS PINS, AND WHAT IT DOES NOT (round 1 F4, sharpened by round 2).
    Round 1 replaced a verdict-comparison here with arm-reached probes, and
    round 2 re-ran the mutation: **the hoist-the-notify-arm-above-risen-floor
    mutant still passes this test.** It has to. Both arms return `outbid`, and
    the notify arm can only ever return `outbid` or fall through, so the two can
    never DISAGREE about the answer — only about which of them said it, and that
    is not observable in a return value. This test therefore pins the property
    it can pin: **no row can move the risen-floor verdict**, in either
    direction, including from a row that supports nothing (below our bid AND
    above on-demand) and one that is fully supported.

    The ORDERING is pinned one arm lower down, by the ON-DEMAND DISCRIMINATOR
    that sits between them: hoisting the notify arm past risen-floor also hoists
    it past the discriminator, and
    `test_a_row_may_only_move_a_verdict_toward_outbid[row1]` kills that mutant
    over the whole matrix (verified by mutation, 2026-08-16 — `[row0]` does not,
    because a $9.99 row is unsupported wherever on-demand is readable, which is
    exactly where the discriminator fires)."""
    args = dict(present=True, actual_status="exited", market_min_bid=0.9099,
                market_listed=True, on_demand=0.748, last_bid=0.747,
                is_bid=True)
    assert bp.classify_eviction(**args) == bp.EVICTION_OUTBID
    # ARM-REACHED probe: an unsupportable row (below our bid, and above
    # on-demand) leaves nothing but the risen-floor arm able to say `outbid`.
    dead_row = _ev("1", 0.747, 0.10, 0.0)
    assert bp.notify_outbid_supported(dead_row, on_demand=0.748) is False
    assert bp.classify_eviction(notify=dead_row, **args) == bp.EVICTION_OUTBID
    # ...and a row that IS supported cannot reorder the arm either. Mutation
    # check: hoisting the notify arm above risen-floor still yields `outbid`, so
    # the discriminating assertion is the one below — drop the floor under our
    # bid and the risen-floor arm goes quiet, at which point the two arms give
    # DIFFERENT answers and the placement is observable.
    live_row = _ev("1", 0.30, 0.74, 0.0)
    assert bp.notify_outbid_supported(live_row, on_demand=0.748) is True
    assert bp.classify_eviction(notify=live_row, **args) == bp.EVICTION_OUTBID
    # ARM-SILENCE state: risen-floor quiet (floor under our bid), on-demand arm
    # quiet (our bid under the clamp) — the notify arm is the ONLY thing that
    # can say `outbid` here, which is how we know it is reached at all rather
    # than shadowed by the arm above it.
    quiet = dict(args, market_min_bid=0.10, last_bid=0.20)
    assert bp.classify_eviction(**quiet) == bp.EVICTION_HOST_STOP
    assert bp.classify_eviction(notify=live_row, **quiet) == bp.EVICTION_OUTBID
    assert bp.classify_eviction(notify=dead_row, **quiet) == bp.EVICTION_HOST_STOP


def test_the_stale_last_bid_incident_keeps_its_fix():
    """2026-08-16: the ladder journaled `ondemand_displaced` with `is_bid:
    false`, off a STALE `last_bid` left by a previous bid box, and that class is
    the single strongest input to the expensive replacement rung.

    `is_bid=False` shuts both on-demand arms, and a notification cannot re-open
    them.

    UPDATED by review round 1 (F1). With §6.1's `< on_demand` clause restored,
    this row (`new_min_bid 2.50` on a machine whose on-demand rate is $1.0017)
    supports nothing at all, so the answer is `unknown` — strictly MORE
    conservative than the `outbid` the first cut minted, and the reason is the
    incident's own arithmetic: our standing bid was $1.05, already ABOVE
    on-demand, so no displacing price can be both above ours and below
    on-demand. On this box no notification can ever support `outbid`, which is
    the correct reading of "we were displaced on a machine we were already
    over-bidding". The regression bar §6.2 sets — `is_bid=False` shuts the od
    arms — is what it always was."""
    args = dict(present=True, actual_status="exited", market_min_bid=None,
                market_listed=None, on_demand=1.0017, last_bid=1.05)
    assert bp.classify_eviction(is_bid=False, **args) == bp.EVICTION_UNKNOWN
    row = _ev("1", 1.05, 2.50, 0.0)
    assert bp.notify_outbid_supported(row, on_demand=1.0017) is False
    assert bp.classify_eviction(is_bid=False, notify=row, **args) \
        == bp.EVICTION_UNKNOWN, "no row can mint a class on this box"
    assert bp.classify_eviction(is_bid=True, notify=row, **args) \
        == bp.EVICTION_ONDEMAND, "a BID box is still an on-demand displacement"
    # ...and the arm stays shut for a row that WOULD be supported elsewhere:
    # `is_bid=False` is upstream of everything, the notify arm included.
    cheap = _ev("1", 0.20, 0.90, 0.0)
    assert bp.notify_outbid_supported(cheap, on_demand=1.0017) is True
    assert bp.classify_eviction(is_bid=False, notify=cheap, **args) \
        == bp.EVICTION_OUTBID, "the notify arm is not an ON-DEMAND arm"


@pytest.mark.parametrize("row", [
    None, {}, "not-a-dict", 17, [],
    {"your_bid": 0.45},                       # half a comparison
    {"new_min_bid": 1.0},
    {"your_bid": None, "new_min_bid": 1.0},
    {"your_bid": "0.45", "new_min_bid": "abc"},
    {"your_bid": 0.45, "new_min_bid": 0.45},    # equal is not GREATER
    {"your_bid": 0.0, "new_min_bid": 0.01},     # zeroed bid: F6
    {"your_bid": -1.0, "new_min_bid": 0.001},   # negative bid
    {"your_bid": 0.45, "new_min_bid": 0.0},
    {"your_bid": 0.45, "new_min_bid": -2.0},
    {"your_bid": 0.45, "new_min_bid": float("inf")},
    {"your_bid": 0.45, "new_min_bid": "1e309"},  # json.loads mints inf from this
    {"your_bid": float("nan"), "new_min_bid": 9.0},
    {"your_bid": 0.45, "new_min_bid": float("nan")},
])
def test_an_unusable_row_supports_nothing(row):
    """The feed is a hidden endpoint written by a service we do not control, so
    every degenerate shape must read as "no evidence" — never as evidence.

    The zero / negative / infinite / NaN rows are review round 1's F6: the first
    cut's bare `nmb > yb` on raw floats read four of them as evidence, and the
    zeroed `your_bid` one disabled the only conflation guard the row carries."""
    assert bp.notify_outbid_supported(row) is False
    assert bp.notify_outbid_supported(row, on_demand=3.0) is False
    args = dict(present=True, actual_status="exited", market_min_bid=0.14,
                market_listed=True, last_bid=0.45, on_demand=3.0, is_bid=True)
    assert bp.classify_eviction(notify=row, **args) == bp.EVICTION_HOST_STOP


def test_the_fallback_arm_can_only_be_reached_with_no_standing_bid():
    """Deviation 4's hazard is unreachable, and by a COINCIDENCE nothing stated
    (review round 1, F3). A row can refine the trailing `ondemand` FALLBACK, and
    `ondemand` is what stops the re-bid ladder dead — so refining it makes the
    re-bid rungs reachable on a box the classifier had written off. Lane 3
    measured the added spend as empty, but only because of two structural facts
    no test asserted, either of which could be changed by an unrelated edit.

    FACT 1 — `MarketRead` never reports a floor it has not listed. All three
    classify call sites source market from `_job_market_read`, and the fallback
    arm needs `market_min_bid is not None` with the risen-floor and host-stop
    arms both declining; with `listed is True` those two partition on
    `lb >= mmb`, so every driver-reachable fallback state has no usable
    `last_bid`.

    FACT 2 — `rebid_ladder` stops dead on `last_bid` None. So the rung the
    refinement unlocks refuses on its own next check, and the worst-case added
    bid spend is nothing.

    Assert both, so the day either changes this test says so instead of the
    ledger."""
    # FACT 1: the MarketRead invariant, on the constructor itself.
    for read in (vl_models.MarketRead(False, False, None),
                 vl_models.MarketRead(True, False, None),
                 vl_models.MarketRead(True, True, 0.14)):
        assert read.min_bid is None or read.listed is True, read
    # ...and the arm is genuinely unreachable with a standing bid + a listing
    # (below the on-demand clamp, or the DISCRIMINATOR answers first).
    for lb in (0.10, 0.14, 0.45, 2.0):
        got = bp.classify_eviction(present=True, actual_status="exited",
                                   market_min_bid=0.14, market_listed=True,
                                   on_demand=3.0, last_bid=lb, is_bid=True)
        assert got in (bp.EVICTION_OUTBID, bp.EVICTION_HOST_STOP), lb
    # the fallback needs market_listed None (nobody asked) or no standing bid
    assert bp.classify_eviction(present=True, actual_status="exited",
                                market_min_bid=0.14, market_listed=None,
                                on_demand=3.0, last_bid=None,
                                is_bid=True) == bp.EVICTION_ONDEMAND
    # FACT 2: and the rung it unlocks refuses on its own.
    stop = bp.rebid_ladder(last_bid=None, market_min_bid=0.14, on_demand=3.0,
                           max_bid=2.0, rungs_used=0, launch_dph_anchor=1.0,
                           eviction_class=bp.EVICTION_OUTBID)
    assert stop.action == "stop" and "no standing bid" in stop.reason


# =========================================================================== #
# LANE 3 (pure half) — the rescue quote and the rails
# =========================================================================== #
def _bid_action_pre_s2b(s):
    """VERBATIM copy of `bidpolicy._bid_action` at 34a78e09."""
    mmb = s.get("market_min_bid")
    target = bp._bid_target(mmb, s.get("max_bid"), s.get("on_demand"))
    if target is None:
        return None
    last_bid = s.get("last_bid")
    if last_bid is None:
        return None
    if s.get("now", 0.0) - s.get("last_bid_put_ts", 0.0) < bp.BID_RATE_LIMIT_S:
        return None
    if bool(s.get("present")) and s.get("actual_status") in bp.LIVE_STATES:
        defend_at = s.get("defend_at")
        defend_at = bp.DEFEND_AT if defend_at is None else defend_at
        if mmb >= defend_at * last_bid and target - last_bid >= bp.BID_MIN_STEP:
            return bp.Action("raise_bid", f"defend:{target}")
        if s.get("decay_streak", 0) >= bp.BID_DECAY_POLLS \
                and target + bp.BID_MIN_STEP < last_bid:
            return bp.Action("lower_bid", f"decay:{target}")
        return None
    if s.get("present") and not s.get("rescue_attempted") and target > last_bid:
        return bp.Action("rescue_bid", f"rescue:{target}")
    return None


#: `defend_at` and `launch_dph_anchor` joined the matrix in review round 1 (m7,
#: M3). `defend_at` is the only one of `_bid_action`'s remaining un-varied inputs
#: it actually reads; lane 3 re-ran the byte-identity claim with it varied (9,216
#: states, 0 divergences) and the fix round encodes that rather than re-deriving
#: it. `launch_dph_anchor` is the new ceiling input, and the None column is what
#: proves the pre-S2b half of the function never looks at it.
_BID_MATRIX = dict(
    present=(True, False),
    actual_status=("running", "exited"),
    market_min_bid=(None, 0.14, 1.0, 2.81),
    max_bid=(None, 2.0),
    last_bid=(None, 0.45, 2.55),
    on_demand=(None, 0.5, 3.0),
    rescue_attempted=(False, True),
    decay_streak=(0, 9),
    last_bid_put_ts=(0.0, 990.0),
    defend_at=(None, 0.5, 0.9, 1.5),
    launch_dph_anchor=(None, 1.2),
    # the job-aware defense ceiling (review round 2). $0.606 is the verifier's
    # own number: p_alt $0.60 with 20 h of work left.
    defense_cap=(None, 0.606),
)


def _bid_cases():
    keys = sorted(_BID_MATRIX)
    for combo in itertools.product(*(_BID_MATRIX[k] for k in keys)):
        yield dict(zip(keys, combo), now=1000.0)


def test_bid_action_is_byte_identical_without_a_row():
    """The money path's half of the same invariant, over 36,864 states: with no
    `notify_min_bid`, `_bid_action` returns exactly what it returned at
    34a78e09 — same kind, same price, same None. `launch_dph_anchor`,
    `defense_cap`, `rebid_ceiling_mult` and the budget fields are on the state
    and are never read here, which is the property M3's fix had to preserve and
    this is where it is checked."""
    n = 0
    for case in _bid_cases():
        n += 1
        s = bp.mk_poll_state(budget_usd=5.0, spend_usd=4.99,
                             rebid_ceiling_mult=1.2, **case)
        assert bp._bid_action(s) == _bid_action_pre_s2b(s), case
        for k in ("notify_min_bid", "launch_dph_anchor", "defense_cap",
                  "rebid_ceiling_mult"):
            s.pop(k)                          # a hand-built pre-S2b state dict
        assert bp._bid_action(s) == _bid_action_pre_s2b(s), case
    assert n == 36864, f"the bid matrix shrank to {n}"


#: Row prices for the exhaustive rails sweep. Everything the review supplied by
#: probe (m4): zero, negative, None, NaN, inf, the JSON string that parses to
#: inf, sub-cent, at-ceiling and one step below a ceiling in the matrix.
_ROW_PRICES = (0.0, -1.0, None, float("nan"), float("inf"), "1e309", "1.99",
               1e-4, 0.15, 1.0, 2.24, 2.25, 2.33, 99.0, 1e9)


def test_no_row_can_emit_a_price_the_rails_would_not():
    """LANE 3, exhaustively. For every state and every row, whatever comes out
    is a price `_bid_target` itself produced from SOME floor at that state's own
    ceiling and on-demand — so the preference, the cost cap, the survival
    cushion and the BID_CEILING_ONDEMAND_FRAC clamp all bind exactly as they do
    without a row. A row proposes; it never widens.

    The `ceiling is None` cell is no longer vacuous (m4): where no ceiling is
    derivable the assertion is that the quote is REFUSED, which is the M3 fix.
    It used to assert nothing at all over 120 rescue outcomes, and with
    adversarial rows those outcomes included `rescue:inf`."""
    seen = collections.Counter()
    for case in _bid_cases():
        for nmb in _ROW_PRICES:
            s = bp.mk_poll_state(notify_min_bid=nmb, **case)
            act = bp._bid_action(s)
            bound = bp.notify_rescue_bound(s)
            if bound.floor is not None and bound.ceiling is None:
                # no anchor and no max_bid: an unknown ceiling is not a licence
                seen["no_ceiling"] += 1
                assert bound.price is None, (case, nmb)
                assert act is None or act.kind != "rescue_bid" \
                    or float(act.reason.split(":", 1)[1]) \
                    == bp._bid_target(case["market_min_bid"], case["max_bid"],
                                      case["on_demand"]), (case, nmb)
            if act is None:
                continue
            price = float(act.reason.split(":", 1)[1])
            assert math.isfinite(price), (case, nmb, act)
            ceiling = bp.effective_bid_ceiling(case["on_demand"],
                                               case["max_bid"])
            assert ceiling is None or price <= ceiling + 1e-9, (case, nmb, act)
            # a rescue is the ONLY arm a row can reach, and it still has to beat
            # the standing bid to exist at all
            if act.kind == "rescue_bid":
                seen["rescue"] += 1
                assert price > case["last_bid"]
                if bound.floor is not None:
                    seen["row_priced"] += 1
                    # ...and a ROW-priced rescue is additionally under the SAME
                    # ceiling the next rung up obeys (M3)
                    assert bound.ceiling is not None
                    assert price <= bound.ceiling + 1e-9, (case, nmb, act)
            else:
                assert act == _bid_action_pre_s2b(s), (case, nmb)
    assert seen["no_ceiling"] and seen["rescue"] and seen["row_priced"], seen


def test_the_notify_quote_obeys_the_rebid_rungs_ceiling():
    """M3, on the 2026-08-16 field box's real numbers. The machine lists
    nothing, the row says $1.00, our launch anchor was $0.45 and on-demand is
    $3.00. The first cut quoted $1.212 — 1.35x the $0.900 ceiling
    (`min(2 x anchor, max_bid, 0.75 x on-demand)`) the very NEXT rung would have
    refused, and 2.69x the price we launched at. It is refused now, and by
    exactly the ceiling `rebid_ladder` builds."""
    s = bp.mk_poll_state(present=True, actual_status="exited",
                         market_min_bid=None, last_bid=FIELD["your_bid"],
                         max_bid=2.999, on_demand=FIELD["on_demand"],
                         launch_dph_anchor=0.45, now=1000.0,
                         last_bid_put_ts=0.0,
                         notify_min_bid=FIELD["new_min_bid"])
    bound = bp.notify_rescue_bound(s)
    assert bound.ceiling == 0.9 == bp.rebid_ceiling(
        launch_dph_anchor=0.45, max_bid=2.999, on_demand=FIELD["on_demand"])
    assert bound.floor == 1.01 and bound.price is None
    assert "rails refused" in bound.refusal
    assert bp._bid_action(s) is None
    # ...and the SAME row on a box we launched at $1.20 is affordable: the
    # ceiling is 2x the LAUNCH price, and that is the number that decides.
    rich = bp.notify_rescue_bound(bp.mk_poll_state(
        present=True, actual_status="exited", market_min_bid=None,
        last_bid=FIELD["your_bid"], max_bid=2.999,
        on_demand=FIELD["on_demand"], launch_dph_anchor=1.20, now=1000.0,
        last_bid_put_ts=0.0, notify_min_bid=FIELD["new_min_bid"]))
    assert rich.ceiling == 2.25 and rich.price == bp._bid_target(
        1.01, 2.25, FIELD["on_demand"])


def test_the_rescue_ceiling_IS_the_rebid_rungs_ceiling_in_every_state():
    """M3's claim, as an equality rather than a resemblance (review round 2).

    Round 1 built the rescue's ceiling from `rebid_ceiling` but fed it only two
    of the four inputs `rebid_ladder` feeds it: it used the MODULE DEFAULT
    `ceiling_mult` instead of the per-watch `rebid_ceiling_mult` knob, and it
    omitted the job-aware `defense_cap` entirely. Both gaps let the rescue quote
    sit ABOVE the rung that runs next — and the rescue PUT happens FIRST on the
    tick, so the defense controller never got a chance to bind it (measured: 11
    of 18 states with a live defense, 31 of 64 at a tightened knob).

    So the test is the identity itself, swept over the whole bound space. If a
    future edit adds a bound to `rebid_ladder` and not to the rescue, this
    fails — which is the only way the §6.7-6 equality stays true."""
    n = 0
    for anchor in (None, 0.45, 1.20, 3.0):
        for max_bid in (None, 0.9, 2.999):
            for od in (None, 0.5, 3.0):
                for mult in (None, 1.2, 2.0, 4.0):
                    for p_alt, rem in ((None, None), (0.60, 20.0), (2.0, 1.0)):
                        n += 1
                        cap, _basis = bp.defense_ceiling(p_alt=p_alt,
                                                         remaining_h=rem)
                        s = bp.mk_poll_state(
                            present=True, actual_status="exited",
                            market_min_bid=None, last_bid=0.45,
                            max_bid=max_bid, on_demand=od,
                            launch_dph_anchor=anchor, now=1000.0,
                            last_bid_put_ts=0.0, notify_min_bid=1.00,
                            rebid_ceiling_mult=mult, defense_cap=cap)
                        reb = bp.rebid_ladder(
                            last_bid=0.45, market_min_bid=1.01, on_demand=od,
                            max_bid=max_bid, rungs_used=0,
                            launch_dph_anchor=anchor,
                            eviction_class=bp.EVICTION_OUTBID,
                            ceiling_mult=(bp.REBID_CEILING_MULT if mult is None
                                          else mult),
                            p_alt=p_alt, remaining_h=rem)
                        got = bp.notify_rescue_bound(s)
                        assert got.ceiling == reb.ceiling, (
                            anchor, max_bid, od, mult, p_alt, got.ceiling,
                            reb.ceiling)
                        # ...and a quote that survives its ceiling is a price
                        # the re-bid rung would also have been allowed to pay
                        if got.price is not None:
                            assert got.price <= reb.ceiling + 1e-9
    assert n == 432, n


def test_a_live_job_aware_defense_binds_the_notify_quote(monkeypatch):
    """The verifier's exact state, end to end: anchor $1.20, `p_alt` $0.60, 20 h
    of work remaining. `defense_ceiling` puts the line at $0.606 — replacing
    this box is rationally cheaper than holding it — and the re-bid rung stops
    there. Before round 2 the rescue quoted $1.212 against a $2.25 ceiling and
    PUT it, because the rescue runs first and the defense bound only the rung
    that ran second."""
    cap, _basis = bp.defense_ceiling(p_alt=0.60, remaining_h=20.0)
    assert cap == 0.606
    base = dict(present=True, actual_status="exited", market_min_bid=None,
                last_bid=0.45, max_bid=2.999, on_demand=3.0,
                launch_dph_anchor=1.20, now=1000.0, last_bid_put_ts=0.0,
                notify_min_bid=1.00)
    defended = bp.notify_rescue_bound(bp.mk_poll_state(defense_cap=cap, **base))
    assert defended.ceiling == 0.606 and defended.price is None
    reb = bp.rebid_ladder(last_bid=0.45, market_min_bid=1.01, on_demand=3.0,
                          max_bid=2.999, rungs_used=0, launch_dph_anchor=1.20,
                          eviction_class=bp.EVICTION_OUTBID,
                          p_alt=0.60, remaining_h=20.0)
    assert reb.action == "stop" and reb.ceiling == 0.606, "the rung agrees"
    assert bp._bid_action(bp.mk_poll_state(defense_cap=cap, **base)) is None

    # CONTROL: no live defense (no fresh p_alt) and the quote is untouched —
    # this bound only ever TIGHTENS, exactly like the one it mirrors.
    plain = bp.notify_rescue_bound(bp.mk_poll_state(**base))
    assert plain.ceiling == 2.25 and plain.price == 1.212
    assert bp.defense_ceiling(p_alt=None, remaining_h=20.0)[0] is None


def test_a_tightened_ceiling_knob_reaches_the_notify_quote():
    """The other half of round 2's finding: `rebid_ceiling_mult` is a per-watch
    knob (`JOB_REBID_CEILING_MULT` / herdd.yaml), and round 1's rescue used
    the module default no matter what the operator had set. Latent today — the
    knob is unset everywhere — and 31 of 64 states diverged at mult 1.2."""
    base = dict(present=True, actual_status="exited", market_min_bid=None,
                last_bid=0.45, max_bid=2.999, on_demand=3.0,
                launch_dph_anchor=1.20, now=1000.0, last_bid_put_ts=0.0,
                notify_min_bid=1.00)
    assert bp.notify_rescue_bound(bp.mk_poll_state(**base)).ceiling == 2.25
    tight = bp.notify_rescue_bound(bp.mk_poll_state(rebid_ceiling_mult=1.2,
                                                    **base))
    assert tight.ceiling == round(1.2 * 1.20, 3) == 1.44
    # ...and tight enough, it refuses what the default allowed
    tighter = bp.notify_rescue_bound(bp.mk_poll_state(rebid_ceiling_mult=0.5,
                                                      **base))
    assert tighter.ceiling == 0.6 and tighter.price is None


def test_the_driver_hands_the_rescue_the_same_bounds_the_rung_gets(monkeypatch):
    """...and the wiring is real. `_job_defense_inputs` has both callers now, so
    the rescue's ceiling and the re-bid rung's are derived from one set of six
    numbers on one tick rather than two derivations of the same idea."""
    iid = FIELD["iid"]
    inst = _inst(iid, FIELD["machine"], FIELD["your_bid"])
    puts = _tick_env(monkeypatch, inst, market=None, listed=False,
                     on_demand=FIELD["on_demand"])
    jc, hf = job_lane.job_supervise_init(_args(iid))
    jc["launch_dph_anchor"] = 1.20
    jc["notify_rows"] = [_ev(iid, FIELD["your_bid"], FIELD["new_min_bid"],
                             time.time(), machine_id=FIELD["machine"])]
    # a FRESH replacement-market read: this is what arms the job-aware defense
    jc["p_alt"], jc["p_alt_ts"] = 0.60, time.time()
    job_lane.job_supervise_tick(jc, hf)
    assert replacement._job_palt_fresh(jc, jc["now"]) == 0.60, "the defense is live"
    assert replacement._job_defense_cap(jc, jc["now"]) is not None
    job_lane.job_supervise_tick(jc, hf)
    assert puts == [], "the defended ceiling binds the rescue, not just the rung"
    q = _fields(jc, notify.RESCUE_QUOTE_EVENT)
    assert q["emitted"] is None and q["ceiling"] == replacement._job_defense_cap(
        jc, jc["now"])


def test_an_underivable_ceiling_refuses_the_quote():
    """m4's `rescue:inf` cell, closed. No launch anchor and no `--max-bid` is
    `rebid_ladder`'s refusal 5 — an unknown ceiling is not a licence to spend —
    and the rescue rung now answers to the same rule. The on-demand clamp alone
    does not rescue that state: it says only "below on-demand", which on an
    expensive machine is not a bound anybody authorised."""
    base = dict(present=True, actual_status="exited", market_min_bid=None,
                last_bid=0.45, max_bid=None, launch_dph_anchor=None,
                now=1000.0, last_bid_put_ts=0.0)
    for nmb in (1.0, 99.0, float("inf"), "1e309", 1e9):
        s = bp.mk_poll_state(on_demand=3.0, notify_min_bid=nmb, **base)
        assert bp.notify_rescue_bound(s).price is None, nmb
        assert bp._bid_action(s) is None, nmb
        # ...and with no on-demand read either, which is where `rescue:inf` was
        s2 = bp.mk_poll_state(on_demand=None, notify_min_bid=nmb, **base)
        assert bp._bid_action(s2) is None, nmb
    # one bound is enough: --max-bid alone derives a ceiling
    ok = bp.mk_poll_state(on_demand=3.0, notify_min_bid=1.0,
                          **dict(base, max_bid=2.0))
    assert bp.notify_rescue_bound(ok).ceiling == 2.0
    assert bp._bid_action(ok).kind == "rescue_bid"


def test_the_quote_refuses_what_the_remaining_budget_cannot_run():
    """M3's second half. `rebid_ladder` has refused a rung the residual budget
    cannot run for REPLACE_MIN_RUNTIME_H since it was written (refusal 6); the
    rescue rung never did, so a --budget 5.00 watch with $4.90 spent would raise
    its bid to a price that budget-parks the box inside five minutes. The only
    guard upstream was the binary budget-park, which fires AFTER the money."""
    base = dict(present=True, actual_status="exited", market_min_bid=None,
                last_bid=0.45, max_bid=2.999, on_demand=3.0,
                launch_dph_anchor=1.20, now=1000.0, last_bid_put_ts=0.0,
                notify_min_bid=1.00)
    rich = bp.notify_rescue_bound(bp.mk_poll_state(budget_usd=5.0,
                                                   spend_usd=0.0, **base))
    assert rich.price == 1.212 and rich.budget_left == 5.0
    poor = bp.notify_rescue_bound(bp.mk_poll_state(budget_usd=5.0,
                                                   spend_usd=4.90, **base))
    assert poor.price is None and poor.budget_left == 0.10
    assert "0.25h floor" in poor.refusal
    assert bp._bid_action(bp.mk_poll_state(budget_usd=5.0, spend_usd=4.90,
                                           **base)) is None
    # an UNCAPPED watch is unaffected — same rule as the re-bid ladder: a rescue
    # moves the price of a meter that is already running
    assert bp.notify_rescue_bound(
        bp.mk_poll_state(budget_usd=None, spend_usd=99.0, **base)).price == 1.212


def test_the_row_prices_a_rescue_the_market_could_not():
    """The reason §6.4 exists. A just-taken machine lists nothing, so
    `market_min_bid` is None at exactly the moment the rescue needs a number and
    the rung simply does not fire. The notification carries the price that
    actually won."""
    base = dict(present=True, actual_status="exited", market_min_bid=None,
                last_bid=FIELD["your_bid"], max_bid=2.999,
                launch_dph_anchor=1.20,     # the ceiling input, M3
                on_demand=FIELD["on_demand"], now=1000.0, last_bid_put_ts=0.0)
    assert bp._bid_action(bp.mk_poll_state(**base)) is None
    act = bp._bid_action(bp.mk_poll_state(notify_min_bid=FIELD["new_min_bid"],
                                          **base))
    assert act is not None and act.kind == "rescue_bid"
    # exactly `_bid_target(new_min_bid + one cent, the REBID CEILING)`, nothing
    # hand-rolled — `rebid_ceiling` is the same function `rebid_ladder` calls
    ceiling = bp.rebid_ceiling(launch_dph_anchor=1.20, max_bid=2.999,
                               on_demand=FIELD["on_demand"])
    want = bp._bid_target(FIELD["new_min_bid"] + bp.BID_MIN_STEP, ceiling,
                          FIELD["on_demand"])
    assert float(act.reason.split(":", 1)[1]) == want


def test_the_quote_never_lowers_the_floor_the_market_gave():
    """`max(market_floor, new_min_bid + step)`: a row whose displacing price is
    below a floor we can actually see changes nothing.

    This is also the BYTE-IDENTITY BOUNDARY of the M3 fix, and the reason the
    bound is keyed on "did the row RAISE the floor" rather than on "is a row
    present". A rescue priced off a readable market floor keeps its pre-S2b
    behaviour — anchor-free ceiling and all — whether or not a row happens to be
    latched beside it. That anchor-free hole PREDATES S2b and closing it is an
    owner question (NOTIFY_DESIGN §6.7), not something a review of this slice
    gets to change under cover of a fix."""
    base = dict(present=True, actual_status="exited", market_min_bid=2.81,
                last_bid=0.45, max_bid=None, on_demand=9.0, now=1000.0,
                last_bid_put_ts=0.0)
    plain = bp._bid_action(bp.mk_poll_state(**base))
    withrow = bp._bid_action(bp.mk_poll_state(notify_min_bid=0.20, **base))
    assert plain == withrow
    assert plain is not None and plain.kind == "rescue_bid"
    # no anchor, no max_bid — and yet NOT refused, because the row raised
    # nothing and this is not the notify-priced path
    assert bp.notify_rescue_bound(
        bp.mk_poll_state(notify_min_bid=0.20, **base)).floor is None


def test_the_rails_refuse_an_unaffordable_quote_and_stay_refused():
    """The displacing price is above what this machine can safely be held at, so
    `_bid_target` escalates instead of pricing. The answer is escalation, not a
    bigger number — the row buys nothing."""
    s = bp.mk_poll_state(present=True, actual_status="exited",
                         market_min_bid=None, last_bid=0.45, max_bid=0.499,
                         launch_dph_anchor=1.20, on_demand=0.5, now=1000.0,
                         last_bid_put_ts=0.0, notify_min_bid=1.00)
    assert bp._bid_target(1.01, None, 0.5) is None      # the rail says no
    assert bp._bid_action(s) is None


def test_the_one_shot_and_the_rate_limit_still_gate_the_row():
    """`rescue_attempted` is the one-shot latch per eviction cycle and the 60 s
    rate limit is vast's 429 guard. A row is evidence, not an exemption."""
    base = dict(present=True, actual_status="exited", market_min_bid=None,
                last_bid=0.45, max_bid=2.999, launch_dph_anchor=1.20,
                on_demand=3.0, notify_min_bid=1.00)
    assert bp._bid_action(bp.mk_poll_state(rescue_attempted=True, now=1000.0,
                                           last_bid_put_ts=0.0, **base)) is None
    assert bp._bid_action(bp.mk_poll_state(rescue_attempted=False, now=1000.0,
                                           last_bid_put_ts=999.0,
                                           **base)) is None


def test_a_live_box_never_sees_the_notify_floor():
    """The defend and decay arms are not a rescue and must not read the row —
    including the case that used to be impossible to reach at all (a live box
    with no market read)."""
    live = dict(present=True, actual_status="running", last_bid=0.45,
                max_bid=None, on_demand=3.0, now=1000.0, last_bid_put_ts=0.0)
    assert bp._bid_action(bp.mk_poll_state(market_min_bid=None,
                                           notify_min_bid=99.0, **live)) is None
    for mmb, streak in ((1.0, 0), (0.14, 9)):
        a = bp._bid_action(bp.mk_poll_state(market_min_bid=mmb,
                                            decay_streak=streak, **live))
        b = bp._bid_action(bp.mk_poll_state(market_min_bid=mmb,
                                            decay_streak=streak,
                                            notify_min_bid=99.0, **live))
        assert a == b


# =========================================================================== #
# LANE 2 — the driver: matching, latching, cycles, seams
# =========================================================================== #
def test_match_is_by_instance_and_inside_the_window():
    now = 1_000_000.0
    rows = [_ev("700", 0.45, 1.0, now - 60, event_id="a" * 32),
            _ev("701", 0.45, 9.0, now - 60, event_id="b" * 32)]
    assert notify.match_outbid(rows, "700", now)["event_id"] == "a" * 32
    assert notify.match_outbid(rows, "702", now) is None
    # machine_id is NEVER the key: a machine hosts sibling chunks
    assert notify.match_outbid(rows, 56748, now) is None
    # freshness, both directions
    stamp = rows[0]["created_at"]
    assert notify.match_outbid(rows, "700", stamp + notify.FRESH_WINDOW_S + 1) is None
    assert notify.match_outbid(rows, "700", stamp - notify.FRESH_WINDOW_S - 1) is None
    assert notify.match_outbid(rows, "700", stamp - 300) is not None
    # the newest fresh row wins
    rows.append(_ev("700", 0.45, 2.0, now - 10, event_id="c" * 32))
    assert notify.match_outbid(rows, "700", now)["new_min_bid"] == 2.0
    # ...and a consumed id is invisible
    assert notify.match_outbid(rows, "700", now,
                               exclude_ids=["c" * 32])["event_id"] == "a" * 32


def test_match_reads_raw_feed_rows_and_retained_records_alike():
    """One matcher, two shapes: the rows as vast ships them and the records as
    `state.json` gives them back. A drift between those two is a row that
    matches in the daemon and not in a test, or the reverse."""
    raw = _row(int(FIELD["iid"]))[0]
    now = raw["created_at"] + 30
    got = notify.match_outbid([raw], FIELD["iid"], now)
    assert got["new_min_bid"] == FIELD["new_min_bid"]
    assert got["your_bid"] == FIELD["your_bid"]
    round_tripped = json.loads(json.dumps([got]))
    assert notify.match_outbid(round_tripped, FIELD["iid"], now) == got
    # non-outbid rows are not evidence for this seam at all
    started = [r for r in _feed() if r["notif_type"] != "outbid"][0]
    assert notify.outbid_evidence(started) is None


def test_the_lookaside_dedupes_ages_out_and_is_bounded():
    now = 1_000_000.0
    rows = [_ev("700", 0.4, 1.0, now - i * 10, event_id=f"{i:032x}")
            for i in range(50)]
    kept = notify.retain_outbid([], rows, now)
    assert len(kept) == notify.RETAIN_MAX
    assert kept == notify.retain_outbid(kept, rows, now), "idempotent on re-poll"
    old = _ev("700", 0.4, 1.0, now - notify.RETAIN_S - 1, event_id="f" * 32)
    assert notify.retain_outbid([old], [], now) == []
    assert notify.retain_outbid("garbage", [{"nope": 1}, None], now) == []


# --- tick harness (modelled on test_eviction_blindspot's) ------------------- #
_MarketRead = collections.namedtuple("_MarketRead",
                                     ["ok", "listed", "min_bid", "floors",
                                      "scaled"])
_MarketRead.__new__.__defaults__ = ((), False)


def _args(iid, **kw):
    base = dict(id=int(iid), dry_run=False, budget=5.0, max_bid=None,
                handoff=False, strict_ceiling=False, keep=False,
                max_replacements=None, replace_ceiling_mult=None,
                replacement_retention_hours=None, rescue_wait=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _inst(iid, machine, bid, *, status="exited", intended="stopped"):
    return {"id": int(iid), "actual_status": status, "machine_id": machine,
            "intended_status": intended, "dph_total": bid, "dph_base": bid,
            "num_gpus": 1, "gpu_name": "H200 SXM", "label": "upstream-monorepo",
            "start_date": time.time() - 3600, "is_bid": True}


# MIGRATED (was MIGRATION-BLOCKED, step 6e batch B7) — the whole job-supervise
# TICK lane below moved, subject AND seams together. The blocker named here is
# closed: `_sticky_on_demand` landed at `vastlib.market.pricing` (the home
# job_lane's seam comment named) and `_job_announce_eviction` reaches it as
# `pricing._sticky_on_demand`, so nothing dies inside the port.
#
# Seam placement is by RESOLUTION, not ownership — what module
# `job_lane.job_supervise_tick` looks the name up on: `lifecycle.<name>` for the
# instance read and the bid PUT, `pricing.<name>` for the market reads,
# `handoff.<name>` for the reconcile, `risk.<name>` for the checkpoint alarm,
# `journal.<name>` for the handoff emitter, `replacement.<name>` for the re-bid
# ladder / eviction replacement / defense cap, and bare in `job_lane` for
# `_box_lifecycle_soft` (still a raising SEAM stub there; body at
# `vastlib.jobs.view`) and `_job_resume_in_place`.
#
# The tests that drive `_job_eviction_replace` through `test_eviction_replacement._wire`
# (a batch-B3 helper) moved with B3 — their subject is
# `vastlib.supervise.replacement`, and they follow whichever namespace `_wire`
# patches, not this banner.
def _tick_env(monkeypatch, inst, *, market, listed, on_demand, replaces=None):
    """Every seam that would touch the network, B2 or money, stubbed. Returns
    the list of bid PUTs the ladder issued.

    `replaces`, when a list is passed, collects the
    `(eviction_class, exclusion_class)` pair each `_job_eviction_replace` call
    was made with — the F2/M2 seam."""
    puts = []
    replaces = [] if replaces is None else replaces
    monkeypatch.setattr(lifecycle, "_instances_soft", lambda: ([inst] if inst else []))
    monkeypatch.setattr(handoff, "_job_handoff_reconcile", lambda jc, hf: None)
    monkeypatch.setattr(job_lane, "_box_lifecycle_soft",
                        lambda iid: {"parked": False, "drained_pending": False})
    monkeypatch.setattr(jobmeta, "list_queue", lambda iid: ["j1"])
    monkeypatch.setattr(jobmeta, "read_job",
                        lambda j, **kw: {"job_id": j, "display_status": "running",
                                         "status": "running"})
    monkeypatch.setattr(pricing, "_market_min_bid_soft", lambda m, n=None: market)
    monkeypatch.setattr(pricing, "_market_min_bid_read", lambda m, n=None:
                        _MarketRead(listed is not None, bool(listed), market))
    monkeypatch.setattr(pricing, "_market_bid_listed_soft", lambda m, n=None: listed)
    monkeypatch.setattr(pricing, "_market_ondemand_soft", lambda m, n=None: on_demand)
    monkeypatch.setattr(
        lifecycle, "_put_bid_soft",
        lambda iid, p: (puts.append((str(iid), p)), (True, None))[1])
    monkeypatch.setattr(job_lane, "_job_sup_reattach", lambda jc, iid: None)
    monkeypatch.setattr(jobs_risk, "_ckpt_watchdog_alarm", lambda vw, now: None)
    monkeypatch.setattr(sup_journal, "_job_handoff_emit", lambda jc, ev, **kw: None)
    monkeypatch.setattr(replacement, "_job_rebid_ladder", lambda *a_, **k_: False)
    monkeypatch.setattr(replacement, "_job_eviction_replace",
                        lambda jc, hf, ecls, why, exclusion_class=None: (
                            replaces.append((ecls, exclusion_class)), False)[1])
    # RUNG ZERO (resume-in-place) is upstream of everything S2b touches and is
    # deliberately out of scope: it spends nothing and reads no notification.
    # Held OFF so these tests measure the rungs S2b actually changed.
    monkeypatch.setattr(job_lane, "_job_resume_in_place",
                        lambda *a_, **k_: False)
    return puts


def _events(jc, name=None):
    rows = list(jc.get("ladder_journal") or [])
    return [(ev, f) for ev, f in rows if name is None or ev == name]


def _fields(jc, name):
    got = _events(jc, name)
    return got[0][1] if got else None


def test_the_0624z_field_case_through_the_whole_tick(monkeypatch):
    """§2 end to end, on the real numbers: the inbox row, the listing that
    disagreed with it, and the eviction event that used to say `host_stop`.

    The `notify_outbid_matched` row is the field instrument — it carries BOTH
    verdicts, so the question "does the notification ever change anything?" is
    answered by reading the journal rather than by re-deriving it."""
    inst = _inst(FIELD["iid"], FIELD["machine"], FIELD["your_bid"],
                 status="running", intended="running")
    _tick_env(monkeypatch, inst, market=FIELD["listing_floor_at_stop"],
              listed=True, on_demand=FIELD["on_demand"])
    jc, hf = job_lane.job_supervise_init(_args(FIELD["iid"]))
    assert job_lane.job_supervise_tick(jc, hf) is None            # live; ticket seen

    now = time.time()
    jc["notify_rows"] = [_ev(FIELD["iid"], FIELD["your_bid"],
                             FIELD["new_min_bid"], now - 17,
                             machine_id=FIELD["machine"])]
    inst["actual_status"], inst["intended_status"] = "exited", "stopped"
    job_lane.job_supervise_tick(jc, hf)

    ev = _fields(jc, "jobs_box_evicted")
    assert ev["eviction_class"] == bp.EVICTION_OUTBID
    assert ev["market_listed"] is True
    assert ev["market_min_bid"] == FIELD["listing_floor_at_stop"]
    assert ev["notify_event_id"] == "e" * 32

    m = _fields(jc, notify.MATCHED_EVENT)
    assert m["class_without_notify"] == bp.EVICTION_HOST_STOP
    assert m["class_with_notify"] == bp.EVICTION_OUTBID
    assert (m["your_bid"], m["new_min_bid"]) == (FIELD["your_bid"],
                                                 FIELD["new_min_bid"])
    # §6.5: the calibration read lands with it, and changes nothing
    fc = _fields(jc, notify.FLOOR_CHECK_EVENT)
    assert fc["listing_floor_at_stop"] == FIELD["listing_floor_at_stop"]
    assert fc["new_min_bid"] == FIELD["new_min_bid"]


def test_the_match_and_its_journal_are_latched_per_cycle(monkeypatch):
    """Seventeen not-live ticks are ONE eviction, so they are one match, one
    matched row, and one floor check."""
    inst = _inst(FIELD["iid"], FIELD["machine"], FIELD["your_bid"])
    _tick_env(monkeypatch, inst, market=None, listed=False,
              on_demand=FIELD["on_demand"])
    jc, hf = job_lane.job_supervise_init(_args(FIELD["iid"]))
    jc["notify_rows"] = [_ev(FIELD["iid"], FIELD["your_bid"],
                             FIELD["new_min_bid"], time.time())]
    for _ in range(6):
        job_lane.job_supervise_tick(jc, hf)
    assert len(_events(jc, notify.MATCHED_EVENT)) == 1
    assert len(_events(jc, notify.FLOOR_CHECK_EVENT)) == 1
    assert len(_events(jc, notify.RESCUE_QUOTE_EVENT)) == 1
    assert jc["notify_matched"]["iid"] == FIELD["iid"]


def test_a_double_eviction_never_reuses_the_first_cycles_row(monkeypatch):
    """Instance 47833510, evicted twice in one night. Here the two evictions are
    put SIX MINUTES apart — inside one freshness window, which the window alone
    cannot separate — and cycle 2 must refuse cycle 1's row and its price."""
    iid = TWICE["iid"]
    inst = _inst(iid, 34985, TWICE["first"]["your_bid"])
    _tick_env(monkeypatch, inst, market=None, listed=False, on_demand=5.0)
    jc, hf = job_lane.job_supervise_init(_args(iid))
    now = time.time()
    first = _ev(iid, event_id="1" * 32, created_at=now - 30,
                machine_id=34985, **TWICE["first"])
    jc["notify_rows"] = [first]
    job_lane.job_supervise_tick(jc, hf)
    assert len(_events(jc, notify.MATCHED_EVENT)) == 1

    inst["actual_status"], inst["intended_status"] = "running", "running"
    job_lane.job_supervise_tick(jc, hf)                      # rescued
    assert jc.get("notify_matched") is None, "the cycle latch clears"
    assert jc["notify_consumed_ids"] == ["1" * 32], "the consumed set does NOT"

    inst["actual_status"], inst["intended_status"] = "exited", "stopped"
    job_lane.job_supervise_tick(jc, hf)                      # cycle 2, same window
    assert len(_events(jc, notify.MATCHED_EVENT)) == 1, \
        "cycle 1's row must not label cycle 2"
    assert len(_events(jc, "jobs_box_evicted")) == 2

    second = _ev(iid, event_id="2" * 32, created_at=time.time(),
                 machine_id=34985, **TWICE["second"])
    jc["notify_rows"] = [first, second]
    inst["actual_status"] = "exited"
    job_lane.job_supervise_tick(jc, hf)
    matched = _events(jc, notify.MATCHED_EVENT)
    assert len(matched) == 2 and matched[1][1]["event_id"] == "2" * 32
    assert matched[1][1]["new_min_bid"] == TWICE["second"]["new_min_bid"]


def test_the_whole_cycles_rows_are_consumed_not_just_the_latched_one(monkeypatch):
    """Review round 1, 2-2 — demonstrated on 47833510's real prices.

    A cycle can mint more than one row. Our rescue raise is PUT against a
    STOPPED instance, so if that raise is itself outbid before the box resumes,
    vast mints a second outbid row mid-cycle. The latch takes one row and
    returned early, so the second stayed matchable for the rest of the 900 s
    freshness window — straight into cycle 2, which then labelled and PRICED
    itself off a row describing neither cycle.

    The rule is the cycle, not the row: every fresh in-window row for this box
    belongs to the cycle we are living through, and the cycle spends them all."""
    iid = TWICE["iid"]
    inst = _inst(iid, 34985, TWICE["first"]["your_bid"])
    _tick_env(monkeypatch, inst, market=None, listed=False, on_demand=5.0)
    jc, hf = job_lane.job_supervise_init(_args(iid))
    now = time.time()
    r1 = _ev(iid, event_id="1" * 32, created_at=now - 30, machine_id=34985,
             **TWICE["first"])
    r2 = _ev(iid, event_id="2" * 32, created_at=now - 5, machine_id=34985,
             **TWICE["second"])
    jc["notify_rows"] = [r1, r2]                  # BOTH minted in cycle 1
    job_lane.job_supervise_tick(jc, hf)
    assert len(_events(jc, notify.MATCHED_EVENT)) == 1
    assert set(jc["notify_consumed_ids"]) == {"1" * 32, "2" * 32}, \
        "the cycle spends every row it minted, not only the one it latched"

    inst["actual_status"], inst["intended_status"] = "running", "running"
    job_lane.job_supervise_tick(jc, hf)                                  # rescued
    inst["actual_status"], inst["intended_status"] = "exited", "stopped"
    job_lane.job_supervise_tick(jc, hf)                                  # cycle 2
    assert len(_events(jc, notify.MATCHED_EVENT)) == 1, \
        "cycle 1's leftovers must not label OR price cycle 2"
    assert _events(jc, notify.RESCUE_QUOTE_EVENT) == []

    # ...and the GOOD case still works: cycle 2's own row matches.
    r3 = _ev(iid, event_id="3" * 32, created_at=time.time(), machine_id=34985,
             **TWICE["second"])
    jc["notify_rows"] = [r1, r2, r3]
    job_lane.job_supervise_tick(jc, hf)
    matched = _events(jc, notify.MATCHED_EVENT)
    assert len(matched) == 2 and matched[1][1]["event_id"] == "3" * 32


def test_a_row_landing_LATE_in_a_cycle_is_also_consumed(monkeypatch):
    """The other half of 2-2: the sweep runs on the LATCHED path too, so a row
    minted three ticks into a cycle is that cycle's even though the latch is
    already spent. Without this the sweep would only ever see rows that happened
    to be in the lookaside on the tick the match landed."""
    iid = TWICE["iid"]
    inst = _inst(iid, 34985, TWICE["first"]["your_bid"])
    _tick_env(monkeypatch, inst, market=None, listed=False, on_demand=5.0)
    jc, hf = job_lane.job_supervise_init(_args(iid))
    r1 = _ev(iid, event_id="1" * 32, created_at=time.time(), machine_id=34985,
             **TWICE["first"])
    jc["notify_rows"] = [r1]
    job_lane.job_supervise_tick(jc, hf)
    assert jc["notify_consumed_ids"] == ["1" * 32]
    late = _ev(iid, event_id="2" * 32, created_at=time.time(), machine_id=34985,
               **TWICE["second"])
    jc["notify_rows"] = [r1, late]
    job_lane.job_supervise_tick(jc, hf)                    # still cycle 1, still down
    assert set(jc["notify_consumed_ids"]) == {"1" * 32, "2" * 32}
    assert len(_events(jc, notify.MATCHED_EVENT)) == 1, "one match per cycle"


def test_a_row_that_lands_AFTER_the_stop_is_still_matched(monkeypatch):
    """The race, in the direction the field case happened not to take.

    Our market read and vast's notification are two independent observations of
    one displacement; on 2026-08-16 the row was seventeen seconds EARLY, and
    nothing makes that the rule. A match that only ran on the announcing tick
    would drop every late row on the floor — so the attempt is retried each
    not-live tick, and the late match still labels the cycle and still prices
    the rescue (`jobs_box_evicted` keeps the class it was honestly emitted
    with; `notify_outbid_matched` carries the correction, with both verdicts)."""
    iid = FIELD["iid"]
    inst = _inst(iid, FIELD["machine"], FIELD["your_bid"])
    puts = _tick_env(monkeypatch, inst, market=None, listed=False,
                     on_demand=FIELD["on_demand"])
    jc, hf = job_lane.job_supervise_init(_args(iid))
    job_lane.job_supervise_tick(jc, hf)                       # eviction announced, no row
    assert _events(jc, notify.MATCHED_EVENT) == []
    evicted = _fields(jc, "jobs_box_evicted")
    assert "notify_event_id" not in evicted

    jc["launch_dph_anchor"] = 1.20      # M3: the ceiling is 2x the LAUNCH price
    jc["notify_rows"] = [_ev(iid, FIELD["your_bid"], FIELD["new_min_bid"],
                             time.time(), machine_id=FIELD["machine"])]
    job_lane.job_supervise_tick(jc, hf)                       # the row lands a tick late
    m = _fields(jc, notify.MATCHED_EVENT)
    assert m is not None and m["new_min_bid"] == FIELD["new_min_bid"]
    assert m["class_without_notify"] == bp.EVICTION_OUTBID    # D7 already had it
    assert (m["match_path"], m["floor_source"]) == ("late", "guarded")
    assert puts == [(iid, bp._bid_target(
        FIELD["new_min_bid"] + bp.BID_MIN_STEP,
        bp.rebid_ceiling(launch_dph_anchor=1.20, max_bid=jc["max_bid"],
                         on_demand=FIELD["on_demand"]),
        FIELD["on_demand"]))]


def test_a_box_swap_retires_the_latch_and_the_consumed_set():
    """A replacement / rehost / handoff promotion lands a NEW instance id, so
    nothing the old box consumed can ever match again — and a latch keyed to a
    contract we no longer hold is a liability, exactly like the echo window
    `ladder_core.box_swap_reset` clears beside it."""
    jc = {"notify_matched": _ev("700", 0.45, 1.0, 1.0),
          "notify_consumed_ids": ["a" * 32], "notify_quote_said": True}
    job_lane._job_notify_box_swap_reset(jc)
    assert jc == {}
    # the CYCLE reset is the narrower one: the consumed set survives it
    jc = {"notify_matched": _ev("700", 0.45, 1.0, 1.0),
          "notify_consumed_ids": ["a" * 32], "notify_quote_said": True}
    job_lane._job_notify_cycle_reset(jc)
    assert jc == {"notify_consumed_ids": ["a" * 32]}


def test_a_real_replacement_swap_retires_the_latch(monkeypatch):
    """...and now through a REAL swap path (review round 1, 2-5). The round-1
    test called the reset on a hand-built dict and so covered zero of the three
    seams that change `jc["iid"]`; the wiring was verified by reading. This
    drives the most valuable of the three — `_job_eviction_replace`, the one
    that rents money — end to end and watches the latch go."""
    import test_eviction_replacement as ter
    jc, hf = ter._jc()
    ter._wire(monkeypatch)
    jc["notify_matched"] = _ev("41", 0.45, 1.0, ter.NOW)
    jc["notify_consumed_ids"] = ["a" * 32]
    jc["notify_quote_said"] = True
    assert replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid") is True
    assert jc["iid"] == "88", "the swap really happened"
    for k in ("notify_matched", "notify_consumed_ids", "notify_quote_said"):
        assert k not in jc, k


# --------------------------------------------------------------------------- #
# F2 / M2 — the third class consumer: the evicted-MACHINE exclusion TTL
# --------------------------------------------------------------------------- #
def test_a_row_may_not_shorten_the_machine_exclusion(monkeypatch):
    """The finding both lane 1 (F2) and lane 3 (M2) landed on, with the spend
    trace: `EVICTED_TTL_CLASSES = (outbid, host_stop)` gives those two classes a
    30-minute exclusion and everything else a permanent one. A row refining
    `unknown -> outbid` therefore un-excluded the machine we were just displaced
    from at t+30m, and the next replacement probe could re-rent it. Two deaths
    inside SPOT_FASTDEATH_S flip `prefer_od` to the on-demand rung (measured
    8.3x at anchor $2.00).

    The fix: the notification refines the class we ACT on; the exclusion reads
    the BARE class. Both directions, through the real `_job_eviction_replace`."""
    import test_eviction_replacement as ter
    # (a) bare `unknown`, refined to `outbid` by a row -> STILL permanent
    jc, hf = ter._jc()
    ter._wire(monkeypatch)
    assert replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid",
                                   exclusion_class=bp.EVICTION_UNKNOWN) is True
    assert replacement._job_excluded_machines(jc, ter.NOW + 86_400) == {7}
    assert jc["evicted_machine_ts"]["7"]["class"] == bp.EVICTION_UNKNOWN

    # (b) a GENUINE bare outbid -> the 30-minute TTL, exactly as before S2b
    jc, hf = ter._jc()
    ter._wire(monkeypatch)
    assert replacement._job_eviction_replace(jc, hf, bp.EVICTION_OUTBID, "outbid",
                                   exclusion_class=bp.EVICTION_OUTBID) is True
    assert replacement._job_excluded_machines(jc, ter.NOW + 60) == {7}
    assert replacement._job_excluded_machines(
        jc, ter.NOW + replacement.EVICTED_EXCLUSION_TTL_S + 1) == set()

    # (c) no `exclusion_class` at all = the pre-S2b caller, unchanged
    jc, hf = ter._jc()
    ter._wire(monkeypatch)
    replacement._job_eviction_replace(jc, hf, bp.EVICTION_HOST_FAILURE, "gone")
    assert replacement._job_excluded_machines(jc, ter.NOW + 86_400) == {7}


def test_the_ladder_hands_the_bare_class_to_the_exclusion(monkeypatch):
    """...and the driver really passes it. The row flips this eviction's class
    from `host_stop` to `outbid`, the ladder ACTS on `outbid` — and the
    exclusion still gets `host_stop`... which is itself a TTL'd class, so the
    sharper case is the one below it: the same tick with the listing read
    failing, where bare is `unknown` (permanent) and refined is `outbid`."""
    iid = FIELD["iid"]
    for market, listed, want_bare in ((FIELD["listing_floor_at_stop"], True,
                                       bp.EVICTION_HOST_STOP),
                                      (None, None, bp.EVICTION_UNKNOWN)):
        replaces = []
        with pytest.MonkeyPatch.context() as mp:
            inst = _inst(iid, FIELD["machine"], FIELD["your_bid"])
            _tick_env(mp, inst, market=market, listed=listed,
                      on_demand=FIELD["on_demand"], replaces=replaces)
            jc, hf = job_lane.job_supervise_init(_args(iid))
            jc["notify_rows"] = [_ev(iid, FIELD["your_bid"],
                                     FIELD["new_min_bid"], time.time(),
                                     machine_id=FIELD["machine"])]
            for _ in range(6):                  # ...to the `dead` verdict
                job_lane.job_supervise_tick(jc, hf)
        assert replaces, (market, listed)
        assert replaces[0] == (bp.EVICTION_OUTBID, want_bare), (market, listed)


def test_a_row_can_never_make_the_expensive_rung_more_reachable():
    """`prefer_od` (the on-demand replacement rung) and `rebid_ladder`'s hard
    stop are the other two class consumers, and both branch on
    `EVICTION_ONDEMAND` alone. A row can only move a verdict TOWARD `outbid`, so
    it can only make the expensive rung LESS reachable — the direction that
    costs nothing. Asserted over the whole matrix rather than argued."""
    row = _ev(FIELD["iid"], 0.10, 0.90, 0.0)
    moved = 0
    for case in _matrix_cases():
        bare = bp.classify_eviction(**case)
        withrow = bp.classify_eviction(notify=row, **case)
        if withrow == bp.EVICTION_ONDEMAND:
            assert bare == bp.EVICTION_ONDEMAND, case
        moved += bare != withrow
    assert moved, "the property would be vacuous if no row ever refined"


def test_rows_for_other_boxes_and_stale_rows_change_nothing(monkeypatch):
    """D2 at the driver: a feed full of real rows, none of them ours or none of
    them fresh, and the tick is its pre-S2b self — same class, same events."""
    iid, now = FIELD["iid"], time.time()
    others = [_ev(str(r["associated_id"]["instance_id"]),
                  r["associated_id"].get("your_bid"),
                  r["associated_id"].get("new_min_bid"), now - 60,
                  event_id=r["event_id"])
              for r in _outbids()
              if str(r["associated_id"]["instance_id"]) != iid]
    stale = _ev(iid, FIELD["your_bid"], FIELD["new_min_bid"],
                now - notify.FRESH_WINDOW_S - 60, event_id="z" * 32)

    def run(rows):
        inst = _inst(iid, FIELD["machine"], FIELD["your_bid"])
        with pytest.MonkeyPatch.context() as mp:
            _tick_env(mp, inst, market=FIELD["listing_floor_at_stop"],
                      listed=True, on_demand=FIELD["on_demand"])
            jc, hf = job_lane.job_supervise_init(_args(iid))
            if rows is not None:
                jc["notify_rows"] = rows
            for _ in range(3):
                job_lane.job_supervise_tick(jc, hf)
            return [(ev, dict(f)) for ev, f in _events(jc)]

    assert run(others + [stale]) == run(None)
    assert _fields({"ladder_journal": run(None)}, "jobs_box_evicted") \
        ["eviction_class"] == bp.EVICTION_HOST_STOP


def test_the_rescue_prices_off_the_row_and_puts_it(monkeypatch):
    """§6.4 on the money path: the machine lists nothing (the normal shape for a
    box just taken), so today the rescue rung has no number and does not fire.
    With the row it aims at the price that won — THROUGH `_bid_target`, and
    since review round 1 (M3) under the same ceiling the next rung obeys.

    The box here was launched at $1.20, so `2 x anchor = $2.40` is not the
    binding bound and the $1.212 quote fits. The SAME row on the real field box
    — launched at $0.45 — is refused; that is the test below."""
    iid = FIELD["iid"]
    inst = _inst(iid, FIELD["machine"], FIELD["your_bid"])
    puts = _tick_env(monkeypatch, inst, market=None, listed=False,
                     on_demand=FIELD["on_demand"])
    jc, hf = job_lane.job_supervise_init(_args(iid))
    jc["launch_dph_anchor"] = 1.20
    jc["notify_rows"] = [_ev(iid, FIELD["your_bid"], FIELD["new_min_bid"],
                             time.time(), machine_id=FIELD["machine"])]
    # NOT_LIVE_DEBOUNCE: a blip is not an eviction, so the rescue rung
    # only opens on the SECOND consecutive not-live tick.
    job_lane.job_supervise_tick(jc, hf)
    job_lane.job_supervise_tick(jc, hf)
    ceiling = bp.rebid_ceiling(launch_dph_anchor=1.20, max_bid=jc["max_bid"],
                               on_demand=FIELD["on_demand"])
    want = bp._bid_target(FIELD["new_min_bid"] + bp.BID_MIN_STEP, ceiling,
                          FIELD["on_demand"])
    assert puts == [(iid, want)]
    q = _fields(jc, notify.RESCUE_QUOTE_EVENT)
    assert q["new_min_bid"] == FIELD["new_min_bid"] and q["emitted"] == want
    assert q["market_floor"] is None
    # M3: the field record can score the bound, because the bound is IN it
    assert (q["ceiling"], q["launch_dph_anchor"]) == (ceiling, 1.20)
    assert q["row_raised"] is True and q["quoted"] == want and q["refused"] is None
    assert q["budget_left"] == round(5.0 - jc["spend_usd"], 4)


def test_the_field_boxs_own_anchor_refuses_its_own_quote(monkeypatch):
    """M3 end to end, on the 2026-08-16 box exactly as it was: launched at
    $0.45, displaced at $1.00, on-demand $3.00, machine lists nothing.

    The first cut PUT $1.212 here — 1.35x the $0.900 ceiling the very next rung
    (`rebid_ladder`) would have refused, and 2.69x what we launched at. The
    honest answer on that box is that no legal bid takes it back and the ladder
    should escalate to a replacement, which is what a bounded rescue now says.
    This is the one behaviour the fix round deliberately took AWAY from S2b's
    headline case, and it is the finding, not a regression."""
    iid = FIELD["iid"]
    inst = _inst(iid, FIELD["machine"], FIELD["your_bid"])
    puts = _tick_env(monkeypatch, inst, market=None, listed=False,
                     on_demand=FIELD["on_demand"])
    jc, hf = job_lane.job_supervise_init(_args(iid))
    jc["notify_rows"] = [_ev(iid, FIELD["your_bid"], FIELD["new_min_bid"],
                             time.time(), machine_id=FIELD["machine"])]
    job_lane.job_supervise_tick(jc, hf)
    assert jc["launch_dph_anchor"] == FIELD["your_bid"]   # written from dph
    job_lane.job_supervise_tick(jc, hf)
    assert puts == []
    q = _fields(jc, notify.RESCUE_QUOTE_EVENT)
    assert q["ceiling"] == 0.9 and q["proposed_floor"] == 1.01
    assert q["emitted"] is None and q["quoted"] is None
    assert "rails refused" in q["refused"]


def test_without_the_row_that_same_eviction_buys_nothing(monkeypatch):
    """The control for the test above, same fleet, same market: no row, no
    rescue PUT at all. This is the delta S2b is asking to be allowed to make."""
    iid = FIELD["iid"]
    inst = _inst(iid, FIELD["machine"], FIELD["your_bid"])
    puts = _tick_env(monkeypatch, inst, market=None, listed=False,
                     on_demand=FIELD["on_demand"])
    jc, hf = job_lane.job_supervise_init(_args(iid))
    # NOT_LIVE_DEBOUNCE: a blip is not an eviction, so the rescue rung
    # only opens on the SECOND consecutive not-live tick.
    job_lane.job_supervise_tick(jc, hf)
    job_lane.job_supervise_tick(jc, hf)
    assert puts == []
    assert _events(jc, notify.RESCUE_QUOTE_EVENT) == []


def test_a_refused_quote_is_journaled_as_a_refusal(monkeypatch):
    """The rails say the displacing price cannot be safely held on this machine.
    `emitted: null`, no PUT, and the ladder walks on to escalation — which is
    the outcome §6.4 promises and the one worth being able to CHECK.

    The machine here is cheap (on-demand $0.90) and we were displaced at $0.85,
    so the row IS supported (§6.1's `< on_demand` clause holds) and it is the
    RAILS that refuse: the survival cushion on a $0.86 floor is $0.946, over the
    $0.675 hard ceiling. Round 1's version used a row above on-demand, which the
    F1 fix now rejects one seam earlier — the refusal is still a refusal, but it
    would no longer have been the rails'."""
    iid = FIELD["iid"]
    inst = _inst(iid, FIELD["machine"], FIELD["your_bid"])
    puts = _tick_env(monkeypatch, inst, market=None, listed=False,
                     on_demand=0.90)
    jc, hf = job_lane.job_supervise_init(_args(iid))
    jc["launch_dph_anchor"] = 1.20         # not the binding bound; the rails are
    jc["notify_rows"] = [_ev(iid, FIELD["your_bid"], 0.85,
                             time.time(), machine_id=FIELD["machine"])]
    # NOT_LIVE_DEBOUNCE: a blip is not an eviction, so the rescue rung
    # only opens on the SECOND consecutive not-live tick.
    job_lane.job_supervise_tick(jc, hf)
    job_lane.job_supervise_tick(jc, hf)
    assert puts == []
    q = _fields(jc, notify.RESCUE_QUOTE_EVENT)
    assert q["emitted"] is None and q["new_min_bid"] == 0.85
    assert q["refused"] and q["ceiling"] == bp.effective_bid_ceiling(0.90)


def test_a_below_bid_row_labels_but_never_prices(monkeypatch):
    """Instance 47840057 again, now on the money path: the row is matched and
    journaled (it still rules things out), but it quotes no rescue — there is no
    `notify_rescue_quote` at all, because a displacing price below our own bid
    is not a price a bid can beat."""
    iid = BELOW["iid"]
    inst = _inst(iid, 56759, BELOW["your_bid"])
    puts = _tick_env(monkeypatch, inst, market=None, listed=False,
                     on_demand=3.0)
    jc, hf = job_lane.job_supervise_init(_args(iid))
    jc["notify_rows"] = [_ev(iid, BELOW["your_bid"], BELOW["new_min_bid"],
                             time.time(), machine_id=56759)]
    job_lane.job_supervise_tick(jc, hf)
    m = _fields(jc, notify.MATCHED_EVENT)
    assert m["class_without_notify"] == m["class_with_notify"] \
        == bp.EVICTION_OUTBID, "the D7 listing arm already had this one"
    assert _events(jc, notify.RESCUE_QUOTE_EVENT) == []
    assert puts == []


def test_a_bid_disagreement_is_journaled_and_never_believed(monkeypatch):
    """§6.3's cross-check. Vast's record of OUR standing bid is the one place we
    could learn that our belief drifted — and it is exactly the input belief
    reconciliation must not grow a second writer for."""
    iid = FIELD["iid"]
    inst = _inst(iid, FIELD["machine"], FIELD["your_bid"])
    _tick_env(monkeypatch, inst, market=None, listed=False,
              on_demand=FIELD["on_demand"])
    jc, hf = job_lane.job_supervise_init(_args(iid))
    jc["notify_rows"] = [_ev(iid, 0.96, 2.33, time.time(),
                             machine_id=FIELD["machine"])]
    job_lane.job_supervise_tick(jc, hf)
    mm = _fields(jc, notify.BID_MISMATCH_EVENT)
    assert mm["believed_bid"] == FIELD["your_bid"] and mm["vast_your_bid"] == 0.96
    # the belief itself is untouched by the row; only a PUT or `dph_base` moves it
    assert jc["last_bid"] in (FIELD["your_bid"],
                              bp._bid_target(2.33 + bp.BID_MIN_STEP,
                                             jc["max_bid"], FIELD["on_demand"]))
    assert _fields(jc, "jobs_box_evicted")["standing_bid"] == FIELD["your_bid"]


def test_a_matching_row_with_no_clock_matches_nothing(monkeypatch):
    """The inline `job supervise` CLI and every direct caller of
    `_job_announce_eviction` pass a `jc` with no `now`. Freshness cannot be
    evaluated there, and unevaluated freshness is NOT a match."""
    jc = {"last_bid": 0.45, "notify_rows": [
        _ev("700", 0.45, 1.0, time.time())]}
    assert job_lane._job_notify_match(jc, "700") is None


# =========================================================================== #
# the fleetd boundary (D2) — the daemon half
# =========================================================================== #
@pytest.fixture
def fleet(tmp_path, monkeypatch):
    monkeypatch.setenv("FLEETD_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("FLEETD_GLOBAL_BUDGET_USD", raising=False)
    monkeypatch.delenv("FLEETD_NOTIFY", raising=False)
    monkeypatch.delenv("FLEETD_NOTIFY_POLICY", raising=False)
    return fleetd.Fleet(str(tmp_path / "state"), hooks=NotifyHooks())


def test_the_policy_switch_is_off_by_default(fleet, monkeypatch):
    """S2b is blocked on the §6.6 review and this daemon runs from a checkout,
    so the default has to be OFF: merging must not arm a money-path change."""
    assert fleetd.notify_policy_enabled() is False
    fleet.hooks.notify_queue = [(_envelope(_outbids()), None)]
    fleet.tick()
    assert fleet.state["notify"]["outbid"], "rows are still RETAINED"
    jc = {"notify_rows": ["stale"]}
    fleet._notify_feed(jc)
    assert "notify_rows" not in jc, "off means REMOVED, not emptied"
    monkeypatch.setenv("FLEETD_NOTIFY_POLICY", "1")
    fleet._notify_feed(jc)
    assert len(jc["notify_rows"]) == len(_outbids())


def _armed_latch_tick(monkeypatch, *, anchor=1.20):
    """A box mid-eviction with a MATCHED, latched, affordable row — the state
    lane 2 and lane 3 both drove the gate probe from. Returns (jc, hf, puts)."""
    iid = FIELD["iid"]
    inst = _inst(iid, FIELD["machine"], FIELD["your_bid"])
    puts = _tick_env(monkeypatch, inst, market=None, listed=False,
                     on_demand=FIELD["on_demand"])
    jc, hf = job_lane.job_supervise_init(_args(iid))
    jc["launch_dph_anchor"] = anchor
    jc["notify_rows"] = [_ev(iid, FIELD["your_bid"], FIELD["new_min_bid"],
                             time.time(), machine_id=FIELD["machine"])]
    job_lane.job_supervise_tick(jc, hf)                   # announce + match + latch
    assert jc["notify_matched"]["iid"] == iid
    return jc, hf, puts


def test_the_gate_off_disarms_a_latch_placed_while_it_was_on(fleet, monkeypatch):
    """Review round 1, 2-1 / M1 — the same defect found independently by two
    lanes, each with a working PUT to show for it.

    The gate guarded the FEED, not the LATCH. `notify_matched` is durable state
    and `_job_notify_rescue_min_bid` reads it ungated, so a box that latched a
    row while armed kept pricing its rescue off that row after the switch went
    off — and `.env` hot-reloads, so flip-on/flip-off is the EXPECTED
    operational shape, not an exotic one. This is exactly the emergency-off
    path: turn it on, see something wrong, set the env to 0.

    (a) the switch goes off between the announce tick and the rescue tick."""
    monkeypatch.setenv("FLEETD_NOTIFY_POLICY", "1")
    jc, hf, puts = _armed_latch_tick(monkeypatch)
    assert puts == [], "no PUT yet — NOT_LIVE_DEBOUNCE has not opened the rung"

    monkeypatch.setenv("FLEETD_NOTIFY_POLICY", "0")
    fleet._notify_feed(jc)                          # the one gated seam
    for k in ("notify_rows", "notify_matched", "notify_quote_said"):
        assert k not in jc, k
    assert jc["notify_consumed_ids"], "dedup MEMORY survives; evidence does not"
    job_lane.job_supervise_tick(jc, hf)
    assert puts == [], "the row that was latched must buy nothing"
    assert _events(jc, notify.RESCUE_QUOTE_EVENT) == []


def test_the_gate_off_disarms_a_latch_restored_from_state_json(fleet,
                                                               monkeypatch):
    """(b) the harder half: the daemon RESTARTS with the switch already off and
    the latch comes back out of `state.json`, with no rows fed at all. Probed at
    round 1 as `PUTS with the gate OFF: [('47845356', 1.212)]`."""
    monkeypatch.setenv("FLEETD_NOTIFY_POLICY", "1")
    jc, hf, _ = _armed_latch_tick(monkeypatch)
    w = {}
    fleetd._replacement_state_persist(jc, w)
    w = json.loads(json.dumps(w))                   # state.json is JSON only

    monkeypatch.setenv("FLEETD_NOTIFY_POLICY", "0")
    inst = _inst(FIELD["iid"], FIELD["machine"], FIELD["your_bid"])
    puts = _tick_env(monkeypatch, inst, market=None, listed=False,
                     on_demand=FIELD["on_demand"])
    reborn, hf2 = job_lane.job_supervise_init(_args(FIELD["iid"]))
    fleetd._replacement_state_restore(reborn, w)
    assert reborn["notify_matched"]["iid"] == FIELD["iid"], "the latch is durable"
    fleet._notify_feed(reborn)                      # gate off, no rows
    for _ in range(3):
        job_lane.job_supervise_tick(reborn, hf2)
    assert puts == []
    assert [ev for ev, _f in _events(reborn) if ev.startswith("notify_")] == []


def test_a_gate_flap_cannot_re_latch_a_consumed_row(fleet, monkeypatch):
    """(c) and re-arming does not resurrect the cycle. `notify_consumed_ids` is
    deliberately NOT popped by the gate — it is dedup memory, not evidence, and
    dropping it would let a flap re-match and re-price a row an earlier cycle
    already spent. Keeping it can only ever refuse."""
    monkeypatch.setenv("FLEETD_NOTIFY_POLICY", "1")
    jc, hf, puts = _armed_latch_tick(monkeypatch)
    rows = list(jc["notify_rows"])
    monkeypatch.setenv("FLEETD_NOTIFY_POLICY", "0")
    fleet._notify_feed(jc)
    monkeypatch.setenv("FLEETD_NOTIFY_POLICY", "1")
    jc["notify_rows"] = rows                        # the SAME row, re-fed
    job_lane.job_supervise_tick(jc, hf)
    assert jc.get("notify_matched") is None, "a spent row may not re-latch"
    assert puts == []
    assert len(_events(jc, notify.MATCHED_EVENT)) == 1


def test_the_consumed_set_survives_a_drifted_state_file(monkeypatch):
    """Review round 1, 2-9. `notify_consumed_ids` is durable, so a hand-edited
    or schema-drifted `state.json` can hand back any JSON shape — and BOTH
    readers iterate it. A non-iterable there raised inside the tick, which
    fleetd catches per-watch as `watch_error`: one box wedged forever, never
    rescued. `notify_matched` has been isinstance-guarded since it was written;
    this is the same guard for its sibling."""
    for junk in (17, "abc", None, {"a": 1}, True):
        jc = {"notify_consumed_ids": junk, "now": 1_000_000.0,
              "notify_rows": [_ev("700", 0.45, 1.0, 1_000_000.0)]}
        assert job_lane._job_notify_consumed_ids(jc) == []
        got = job_lane._job_notify_match(jc, "700")         # must not raise
        assert got is not None
    # a well-formed one still excludes
    jc = {"notify_consumed_ids": ["e" * 32], "now": 1_000_000.0,
          "notify_rows": [_ev("700", 0.45, 1.0, 1_000_000.0)]}
    assert job_lane._job_notify_match(jc, "700") is None


def test_the_poll_off_switch_also_starves_the_policy(fleet, monkeypatch):
    monkeypatch.setenv("FLEETD_NOTIFY_POLICY", "1")
    monkeypatch.setenv("FLEETD_NOTIFY", "0")
    jc = {}
    fleet._notify_feed(jc)
    assert "notify_rows" not in jc


def test_the_lookaside_survives_a_restart_and_ages_out(fleet, tmp_path):
    fleet.hooks.notify_queue = [(_envelope(_outbids()), None)]
    fleet.tick()
    kept = fleet.state["notify"]["outbid"]
    assert 0 < len(kept) <= notify.RETAIN_MAX
    assert all(set(r) >= {"event_id", "iid", "new_min_bid"} for r in kept)
    reborn = fleetd.Fleet(fleet.dir, hooks=NotifyHooks())
    assert reborn.state["notify"]["outbid"] == kept
    # ...and a poll far in the future drops them rather than matching forever
    newest = max(r["created_at"] for r in _feed())
    reborn.hooks.t = newest + 10 * notify.RETAIN_S
    reborn.hooks.notify_queue = [(_envelope([]), None)]
    reborn.tick()
    assert reborn.state["notify"]["outbid"] == []


def test_the_latch_keys_are_durable_state():
    """A restart mid-eviction must not re-consume a row this process already
    spent — the `evicted_announced` lesson (three re-announcements of one
    eviction across two deploy restarts on 2026-08-14).

    `rescue_deadline` / `rescue_put_failures` joined them in review round 1
    (2-4): the rescue rung's one-shot latch IS `rescue_deadline is not None`,
    and it lived only in memory while `notify_matched` beside it did not — so a
    restart mid-cycle re-armed a rescue the cycle had already spent, and S2b
    handed it a price to re-spend it at. The behaviour change is strictly
    spend-reducing, and it also closes a pre-S2b duplicate-rescue-after-restart
    shape that had gone unnoticed because pre-S2b that state usually had no
    price to fire on."""
    for k in ("notify_matched", "notify_consumed_ids",
              "rescue_deadline", "rescue_put_failures"):
        assert k in fleetd.REPLACEMENT_STATE_KEYS
    jc = {"notify_matched": _ev("700", 0.45, 1.0, 1.0),
          "notify_consumed_ids": ["a" * 32], "notify_rows": ["ephemeral"],
          "rescue_deadline": 1_000_300.0, "rescue_put_failures": 2}
    w = {}
    fleetd._replacement_state_persist(jc, w)
    assert "notify_rows" not in w["replacement"], "the FEED is not durable state"
    reborn = {}
    fleetd._replacement_state_restore(reborn, json.loads(json.dumps(w)))
    assert reborn["notify_consumed_ids"] == ["a" * 32]
    assert reborn["notify_matched"]["event_id"] == "e" * 32
    assert reborn["rescue_deadline"] == 1_000_300.0
    assert reborn["rescue_put_failures"] == 2


def test_a_restart_does_not_re_arm_a_rescue_the_cycle_already_spent(monkeypatch):
    """2-4 through the ladder: the box is still down, the one-shot is spent, and
    the daemon restarts. Before this the reborn watch saw `rescue_deadline=None`
    and fired a SECOND rescue for the same eviction cycle."""
    monkeypatch.setenv("FLEETD_NOTIFY_POLICY", "1")
    jc, hf, puts = _armed_latch_tick(monkeypatch)
    job_lane.job_supervise_tick(jc, hf)                    # the rung opens; PUT lands
    assert len(puts) == 1 and jc["rescue_deadline"] is not None
    w = {}
    fleetd._replacement_state_persist(jc, w)

    inst = _inst(FIELD["iid"], FIELD["machine"], FIELD["your_bid"])
    puts2 = _tick_env(monkeypatch, inst, market=None, listed=False,
                      on_demand=FIELD["on_demand"])
    reborn, hf2 = job_lane.job_supervise_init(_args(FIELD["iid"]))
    reborn["launch_dph_anchor"] = 1.20
    fleetd._replacement_state_restore(reborn, json.loads(json.dumps(w)))
    reborn["notify_rows"] = list(jc.get("notify_rows") or [])
    for _ in range(2):
        job_lane.job_supervise_tick(reborn, hf2)
    assert puts2 == [], "the one-shot was already spent, and it survived"


def test_rows_that_match_nothing_change_no_fleet_behaviour(tmp_path,
                                                           monkeypatch):
    """The S2b analogue of S2a's boundary test, and the sharper one: the policy
    switch ARMED, the real feed flowing, and not one row matching a watched box
    — identical actions, identical journal, notify rows included.

    Round 1 (2-6) caught this registering the watch on profile `bare`, which is
    NOT in `POLICY_PROFILES` — so `_tick_policy_watch` and with it `_notify_feed`
    never ran, and the test's docstring was not what it measured. It watches
    `jobs` now, which really does drive the feed into the ladder."""
    monkeypatch.delenv("FLEETD_NOTIFY", raising=False)
    monkeypatch.setenv("FLEETD_NOTIFY_POLICY", "1")
    assert "jobs" in fleetd.POLICY_PROFILES and "bare" not in fleetd.POLICY_PROFILES

    def run(with_rows):
        d = tmp_path / ("on" if with_rows else "off")
        h = NotifyHooks()
        h.box(5001, label="jobs:one", dph=1.0)
        h.box(5002, label=None, dph=3.0)
        if with_rows:
            h.notify_queue = [(_envelope(_feed()), None)]
        f = fleetd.Fleet(str(d), hooks=h)
        f.watch("5001", "jobs", budget_usd=5.0)
        for _ in range(4):
            f.tick()
            h.advance(300)
        return f, h

    on, hon = run(True)
    off, hoff = run(False)
    assert (hon.parked, hon.resumed, hon.destroyed, hon.kept) == \
           (hoff.parked, hoff.resumed, hoff.destroyed, hoff.kept)
    strip = lambda f: [r["event"] for r in journal(f)
                       if not r["event"].startswith("notify_")]
    assert strip(on) == strip(off)
    assert on.alarms == off.alarms
    # the ON arm really did consume the feed, and matched none of it
    ons = [r for r in journal(on) if r["event"] == notify.SEEN_EVENT]
    assert len(ons) == len(_feed())
    assert [r for r in journal(on)
            if r["event"] in (notify.MATCHED_EVENT, notify.RESCUE_QUOTE_EVENT)] == []


def test_the_report_schema_knows_every_s2b_event():
    """`fleet report` counts an unschema'd event as `unknown_events`, which is
    an alarm. A new journal row that trips it teaches operators to ignore it."""
    import fleet_report
    for ev in (notify.MATCHED_EVENT, notify.BID_MISMATCH_EVENT,
               notify.RESCUE_QUOTE_EVENT, notify.FLOOR_CHECK_EVENT):
        assert ev in fleet_report.EVENT_SCHEMA
        assert ev in fleet_report.TIMELINE_EVENTS
    spec = fleet_report.EVENT_SCHEMA[notify.MATCHED_EVENT]
    assert {"class_without_notify", "class_with_notify"} <= spec.optional
    # review round 1: the calibration fields (2-3) and the quote's BOUNDS (M3)
    assert {"match_path", "floor_source"} <= spec.optional
    assert {"match_path", "floor_source"} <= \
        fleet_report.EVENT_SCHEMA[notify.FLOOR_CHECK_EVENT].optional
    assert {"ceiling", "launch_dph_anchor", "budget_left", "quoted", "refused",
            "row_raised"} <= \
        fleet_report.EVENT_SCHEMA[notify.RESCUE_QUOTE_EVENT].optional


# =========================================================================== #
# LANE 5 — an outbid must be arithmetically possible (owner, 2026-08-26)
# =========================================================================== #
#: Instance 48392216, 2026-08-22 12:26Z: the matched row said we were displaced
#: at $0.60 having bid $0.48, while the standing bid the rungs had walked us to
#: was $0.724. Fifteen rows in one month of journal carry a `new_min_bid` below
#: our own bid at match time; each one could mint `outbid` off the row-internal
#: test alone, which is how an eviction gets that class at a market floor
#: arithmetically below the bid it supposedly lost to.
STALE = dict(iid="48392216", your_bid=0.48, new_min_bid=0.60, our_bid=0.724)

#: Instance 48712232, 2026-08-26 03:04:06Z: eviction recorded `market_min_bid
#: $0.407` against our $0.24 and was classified `outbid`. The read 54 s later
#: said $0.20 and the box came back at 03:06:11 — a sibling chunk's price on a
#: multi-chunk machine, not a competing bidder.
SPIKE = dict(iid="48712232", spike=0.407, our_bid=0.24, neighbours=(0.20, 0.24))


def test_a_row_whose_price_loses_to_our_own_bid_mints_no_class():
    """The owner rule: nobody paid more than we bid => not an outbid. The row's
    `your_bid` goes stale the moment a rung raises us, so the row-internal test
    alone cannot see this."""
    row = _ev(STALE["iid"], STALE["your_bid"], STALE["new_min_bid"], 0.0)
    assert bp.notify_outbid_supported(row) is True          # row-internal: yes
    assert bp.notify_outbid_supported(row, last_bid=STALE["our_bid"]) is False
    args = dict(present=True, actual_status="exited", is_bid=True,
                market_min_bid=0.28, market_listed=True,
                last_bid=STALE["our_bid"], on_demand=1.5)
    assert bp.classify_eviction(**args) == bp.EVICTION_HOST_STOP
    assert bp.classify_eviction(notify=row, **args) == bp.EVICTION_HOST_STOP


def test_no_arm_can_mint_outbid_below_our_own_standing_bid():
    """The property, over the whole matrix and every row shape: a verdict of
    `outbid` requires SOME price above our bid — the listing's, or one vast
    recorded. `market_min_bid` None is not a price and keeps its own arm, and
    `is_bid=False` is excluded because there the `last_bid` is not a bid we hold
    (see `test_the_last_bid_gate_is_withheld_on_a_non_bid_box`)."""
    rows = [None,
            _ev("1", 0.10, 0.20, 0.0), _ev("1", 0.48, 0.60, 0.0),
            _ev("1", 0.10, 9.99, 0.0), _ev("1", 2.0, 1.0, 0.0)]
    for case in _matrix_cases():
        if case["is_bid"] is False:
            continue
        lb = case["last_bid"] if (case["last_bid"] and case["last_bid"] > 0) else None
        mmb = (case["market_min_bid"]
               if (case["market_min_bid"] and case["market_min_bid"] > 0) else None)
        for row in rows:
            if bp.classify_eviction(notify=row, **case) != bp.EVICTION_OUTBID:
                continue
            if lb is None or mmb is None:
                continue            # no bid, or no listing price: other arms
            nmb = bp.notify_price((row or {}).get("new_min_bid"))
            assert mmb > lb or (nmb is not None and nmb > lb), (case, row)


def test_a_row_still_outranks_a_listing_it_genuinely_beats():
    """The 06:24Z precedence is NOT withdrawn: a displacing price above OUR bid
    is a price somebody paid, and it still beats a listing that came back below
    our bid seventeen seconds later."""
    args = dict(present=True, actual_status="exited", is_bid=True,
                market_min_bid=FIELD["listing_floor_at_stop"],
                market_listed=True, last_bid=FIELD["your_bid"],
                on_demand=FIELD["on_demand"])
    row = _ev(FIELD["iid"], FIELD["your_bid"], FIELD["new_min_bid"], 0.0)
    assert bp.classify_eviction(notify=row, **args) == bp.EVICTION_OUTBID


def test_the_last_bid_gate_is_withheld_on_a_non_bid_box():
    """`is_bid=False` means the `last_bid` on the watch is a stale leftover of a
    previous rental, not a price we are paying — so it may not gate a row (the
    2026-08-16 stale-last_bid incident, from the other side)."""
    args = dict(present=True, actual_status="exited", market_min_bid=None,
                market_listed=None, on_demand=1.0017, last_bid=1.05)
    cheap = _ev("1", 0.20, 0.90, 0.0)
    assert bp.classify_eviction(is_bid=False, notify=cheap, **args) \
        == bp.EVICTION_OUTBID


def test_an_uncorroborated_floor_spike_reads_as_host_stop():
    """A single offers read is one sample of a multi-chunk machine. When every
    other observation sits at or under our bid, the rise is uncorroborated and
    the conservative class wins."""
    args = dict(present=True, actual_status="exited", is_bid=True,
                market_min_bid=SPIKE["spike"], market_listed=True,
                last_bid=SPIKE["our_bid"], on_demand=1.2)
    assert bp.classify_eviction(**args) == bp.EVICTION_OUTBID       # no samples
    assert bp.classify_eviction(floor_samples=SPIKE["neighbours"], **args) \
        == bp.EVICTION_HOST_STOP
    # a rise READ TWICE is a rise: the sample under test is discounted once, and
    # the second sighting still corroborates
    assert bp.classify_eviction(
        floor_samples=(0.20, SPIKE["spike"], SPIKE["spike"]), **args) \
        == bp.EVICTION_OUTBID
    # ...and so does a notification whose price beats our bid
    assert bp.classify_eviction(
        floor_samples=SPIKE["neighbours"],
        notify=_ev("1", SPIKE["our_bid"], 0.35, 0.0), **args) == bp.EVICTION_OUTBID


def test_corroboration_is_tri_state_and_silent_without_evidence():
    """`None` = nothing to corroborate with, and it must leave every existing
    caller alone: no samples, no standing bid, or samples that are all the read
    under test."""
    assert bp.floor_rise_corroborated(0.407, 0.24) is None
    assert bp.floor_rise_corroborated(0.407, None, floor_samples=(0.2,)) is None
    assert bp.floor_rise_corroborated(0.407, 0.24, floor_samples=(0.407,)) is None
    assert bp.floor_rise_corroborated(0.407, 0.24, floor_samples=(0.2, 0.24)) is False
    assert bp.floor_rise_corroborated(0.407, 0.24, floor_samples=(0.2, 0.41)) is True
    # junk in the sample list is ignored, never counted as evidence
    assert bp.floor_rise_corroborated(
        0.407, 0.24, floor_samples=(None, "x", 0.0, -1, float("inf"))) is None


def test_the_corroboration_argument_is_inert_when_unused():
    """Every pre-2026-08-26 caller passes no samples; the whole matrix must be
    bit-identical with the argument spelled out as empty."""
    for case in _matrix_cases():
        assert bp.classify_eviction(floor_samples=(), **case) == \
            bp.classify_eviction(**case), case
