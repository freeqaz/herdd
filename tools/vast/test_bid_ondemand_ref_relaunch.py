#!/usr/bin/env python3
"""test_bid_ondemand_ref_relaunch — two ways the on-demand REFERENCE went wrong.

Both are the doc 50 R1 family (a BID-view offer row's `dph_total` is the current
INTERRUPTIBLE price, ~min_bid + a storage sliver — nothing on that row carries
the machine's on-demand rate), and both were still live when the 2026-08-08
autobid audit ran:

  A. `_do_relaunch` (the RUN lane's eviction relaunch) priced every candidate as
     `min(1.2 x min_bid + 1e-4, offer.dph_total - 0.001)`. The launch, handoff
     and eviction-replacement paths were fixed 2026-08-06 / 2026-08-05; this one
     was missed, so a run-lane relaunch still bid a rounding unit over its own
     floor. It also still carried the `+1e-4` nudge that P2 (2026-07-18) removed
     everywhere else.

  B. A FAILED on-demand probe returned None, which disables the clamp entirely
     and drops the ceiling onto the 3.0x-median-floor fallback. Four boxes are on
     record carrying standing bids ABOVE their machine's on-demand price.

Both tests fail on the pre-2026-08-08 code.
"""
import argparse
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bidpolicy as bp  # noqa: E402
# NOTE no `import herdd`: with `_sticky_on_demand` repointed to
# `vastlib.market.pricing` (step 6 leftovers) this file's last flat reference is
# gone. Both subjects here — the relaunch pricing rung and the sticky clamp —
# now live in the market ring, and `bidpolicy` is a sibling module both arms
# share, so nothing left needs the 10k-line flat namespace imported.
from vastlib.boxes import lifecycle, ssh  # noqa: E402
from vastlib.market import offers as market_offers  # noqa: E402
from vastlib.market import pricing  # noqa: E402
from vastlib.supervise import journal, replacement  # noqa: E402


# A real-shape BID-view offer: dph_total is the interruptible price, and the
# machine's true on-demand rate (2.40) exists only market-side.
_OFFER = {"id": 999, "machine_id": 777, "num_gpus": 1,
          "min_bid": 1.0667, "dph_total": 1.0713, "dph_base": 1.0667}
_TRUE_OD = 2.40


# --------------------------------------------------------------------------- #
# A. the run-lane relaunch
# --------------------------------------------------------------------------- #
def _relaunch_st():
    return {"run_id": "r1", "husk_id": None, "max_bid": None,
            "relaunch_count": 0, "spend_usd": 0.0,
            "launch_spec": {"image": "reg/img:tag", "disk": 100,
                            "runtype": "ssh_direct", "env": {"RUN_ID": "r1"},
                            "secret_env_keys": []}}


def _wire_relaunch(monkeypatch, offer, market_od):
    seen = {}
    monkeypatch.setattr(market_offers, "_search_offers_soft", lambda a: [dict(offer)])
    monkeypatch.setattr(pricing, "_market_ondemand_soft",
                        lambda mid, g=None: (seen.setdefault("od_probe", [])
                                             .append((mid, g)) or market_od))
    monkeypatch.setattr(replacement, "_relaunch_body",
                        lambda st, a, bid, **kw: ({"price": bid}, []))
    monkeypatch.setattr(replacement, "_reset_run_markers", lambda run_id, dry_run=False: None)
    monkeypatch.setattr(ssh, "attach_ssh_key_soft", lambda cid: None)
    monkeypatch.setattr(lifecycle, "launch_instance",
                        lambda oid, body: (seen.update(body=body, offer=oid),
                                           (True, "9100", None))[1])
    monkeypatch.setattr(journal, "_sup_emit",
                        lambda run_id, kind, **kw: seen.setdefault(
                            "emits", []).append((kind, kw)))
    return seen


def test_relaunch_prices_against_the_ONDEMAND_market_not_the_bid_row(monkeypatch):
    """THE regression. With the offer row as the clamp this bid $1.070 — floor
    plus a rounding unit. It must come from the market probe instead."""
    seen = _wire_relaunch(monkeypatch, _OFFER, _TRUE_OD)
    st = _relaunch_st()
    a = argparse.Namespace(dry_run=False, num_gpus=1)
    assert replacement._relaunch(st, a) == "relaunched"
    assert seen["od_probe"] == [(777, 1)], "the machine's on-demand market must be read"
    assert st["last_bid"] == bp._bid_target(1.0667, None, _TRUE_OD)
    assert st["last_bid"] > 1.0713, "still priced off the bid row's dph_total"
    assert st["last_bid"] / 1.0667 >= bp.BID_MIN_CUSHION_MULT


def test_relaunch_has_no_epsilon_nudge_left(monkeypatch):
    """P2 (2026-07-18): launch price == steady-state target, or the first
    supervised poll decays the fresh bid. The `+1e-4` survived only here."""
    _wire_relaunch(monkeypatch, _OFFER, _TRUE_OD)
    st = _relaunch_st()
    replacement._relaunch(st, argparse.Namespace(dry_run=False, num_gpus=1))
    s = bp.mk_poll_state(present=True, actual_status="running",
                         market_min_bid=1.0667, on_demand=_TRUE_OD,
                         last_bid=st["last_bid"])
    assert bp._decay_candidate(s) is False


def test_relaunch_falls_back_to_the_rank_price_when_the_probe_fails(monkeypatch):
    """A failed market probe must not fail the relaunch — a rescue that cannot
    find a fast on-demand quote still needs a box. No clamp, full cushion."""
    _wire_relaunch(monkeypatch, _OFFER, None)
    st = _relaunch_st()
    assert replacement._relaunch(st, argparse.Namespace(dry_run=False,
                                                 num_gpus=1)) == "relaunched"
    assert st["last_bid"] == bp._bid_target(1.0667, None, None)


def test_relaunch_still_honors_max_bid(monkeypatch):
    _wire_relaunch(monkeypatch, _OFFER, _TRUE_OD)
    st = _relaunch_st()
    st["max_bid"] = 1.20
    assert replacement._relaunch(st, argparse.Namespace(dry_run=False,
                                                 num_gpus=1)) == "relaunched"
    assert st["last_bid"] <= 1.20


# --------------------------------------------------------------------------- #
# B. a failed on-demand probe must not unclamp the bid
# --------------------------------------------------------------------------- #
def test_sticky_on_demand_survives_a_failed_probe():
    # REPOINTED (step 6 leftovers): `_sticky_on_demand` now has a vastlib home —
    # `market.pricing`, beside `_auto_bid_price` and `_market_ondemand_soft`,
    # because BOTH supervise lanes call it and two copies of the clamp is the
    # drift the market ring exists to prevent. The subject moves with the body;
    # the flat `herdd._sticky_on_demand` is the add-only twin that step 7's
    # shim retires.
    st = {}
    assert pricing._sticky_on_demand(st, 2.40) == 2.40
    assert pricing._sticky_on_demand(st, None) == 2.40  # was None -> clamp disabled
    assert pricing._sticky_on_demand(st, 0) == 2.40
    assert pricing._sticky_on_demand(st, 3.10) == 3.10  # a real read always wins
    assert pricing._sticky_on_demand(st, None) == 3.10


def test_sticky_on_demand_starts_empty():
    assert pricing._sticky_on_demand({}, None) is None  # nothing to be stale about


@pytest.mark.parametrize("iid,bid,od", [
    ("44962074", 0.15014814814814814, 0.10666666666666667),
    ("44965461", 0.15014814814814814, 0.10666666666666667),
    ("46177923", 0.2277777777777778, 0.152),
    ("47018759", 1.21, 1.1706666666666667),
])
def test_recorded_over_ondemand_bids_are_unreachable_with_a_clamp(iid, bid, od):
    """The four `bid_over_preferred_ceiling` records where our standing bid was
    ABOVE the machine's on-demand price. With the on-demand number in hand at the
    moment of the raise — which is what stickiness guarantees — `_bid_target`
    cannot produce any of them from any floor."""
    for floor_mult in (0.5, 0.8, 1.0, 1.2, 1.5):
        floor = round(bid / 1.2 * floor_mult, 4)
        t = bp._bid_target(floor, None, od)
        assert t is None or t < od, f"{iid}: target {t} >= on-demand {od}"


def test_run_lane_observe_uses_the_sticky_value(monkeypatch):
    """Wiring check on the state field the ladder actually reads."""
    st = {"machine_id": 777, "num_gpus": 1, "on_demand_last": 2.40}
    # REPOINTED with the subject (step 6 leftovers): both the clamp and the probe
    # it consumes are `market.pricing` names now, and the test calls the probe
    # itself — so the patch has to sit in the namespace the call below resolves
    # through, which is `pricing`, exactly as the relaunch test above already
    # patches it. Patching `herdd` here would leave the real probe running.
    monkeypatch.setattr(pricing, "_market_ondemand_soft", lambda mid, g=None: None)
    assert pricing._sticky_on_demand(
        st, pricing._market_ondemand_soft(777, 1)) == 2.40
