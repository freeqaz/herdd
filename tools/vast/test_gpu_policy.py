"""Portable tests for the default preferred-GPU policy (owner directive
2026-08-03, widened 2026-08-07) — no real vast API, no network.

The incident this pins: a bf16 training launch with only a VRAM floor (no
--gpu) auto-picked the cheapest fitting card — a Quadro RTX 8000, Turing
sm_75, NO bf16 — and the box had to be destroyed before it wasted the run.
The --cuda floor cannot catch this class: cuda_max_good measures the host
DRIVER, not the silicon, so a Turing card behind a new driver passes 13.0.

The allowlist draws exactly ONE line — bf16-capable silicon (Ampere or newer)
— and is not an architecture preference. It was Blackwell-only from 2026-08-03
to 2026-08-07, which made an H100/A100 unreachable without --any-gpu even when
it was the best value on the board; the owner ruled that restriction off
2026-08-07. `test_cheap_hopper_wins_on_price` is that ruling's regression test:
it must FAIL if anyone re-narrows tier 0 to Blackwell.

Contract under test (policy table: vastlib.market.offers.GPU_DEFAULT_POLICY_TIERS):
  * no --gpu / pin  -> auto-pick restricted to the tier allowlist, cheapest
    within tier 0 (>=32 GB bf16 cards, ranked on price alone — Hopper/Ampere
    datacenter included); pre-Ampere can never be picked (by construction);
  * tier fallback   -> tier 0 dry falls through to the <32 GB / older tail;
  * explicit --gpu, --machine/--host pin -> policy fully bypassed;
  * --any-gpu       -> old unrestricted cheapest-overall behavior.
Covers all three seams: `search_offers` (search/launch/train),
`_search_offers_soft` (supervise relaunch/handoff), `pick_cheapest_offer`
(workflow / jobs understudy).
"""
import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vastlib.cli import search as cli_search  # noqa: E402
from vastlib.core import api  # noqa: E402
from vastlib.market import offers as market_offers  # noqa: E402


# A fake market: the pre-Ampere card is the CHEAPEST — exactly the shape that
# produced the incident. Prices ascend Turing < 5090 < PRO 6000 < B200, and
# id 50 (RTX 5070, tier 1) exists only to exercise the tier fallback.
MARKET = [
    {"id": 10, "gpu_name": "Quadro RTX 8000", "dph_total": 0.12, "min_bid": 0.06},
    {"id": 20, "gpu_name": "RTX 5090",        "dph_total": 0.35, "min_bid": 0.18},
    {"id": 30, "gpu_name": "RTX PRO 6000 WS", "dph_total": 0.60, "min_bid": 0.30},
    {"id": 40, "gpu_name": "B200",            "dph_total": 2.40, "min_bid": 1.20},
    {"id": 50, "gpu_name": "RTX 5070",        "dph_total": 0.20, "min_bid": 0.10},
]


def _bundles_reply(body, market):
    """Emulate the v0/bundles/ endpoint over an in-memory market: apply the
    gpu_name in-filter and the ascending price sort the query asks for."""
    offers = list(market)
    gn = body.get("gpu_name")
    if gn:
        offers = [o for o in offers if o["gpu_name"] in gn["in"]]
    field = body["order"][0][0]
    offers.sort(key=lambda o: o[field])
    return {"offers": offers[: body.get("limit") or 20]}


def _wire(monkeypatch, market=MARKET):
    """Point both request paths at the fake market; return the query log."""
    calls = []

    def fake_request(method, path, body=None, **kw):
        calls.append(body)
        return _bundles_reply(body, market)

    def fake_request_soft(method, path, body=None, **kw):
        calls.append(body)
        return True, _bundles_reply(body, market), None

    monkeypatch.setattr(api, "request", fake_request)
    monkeypatch.setattr(api, "request_soft", fake_request_soft)
    return calls


def _ns(*argv):
    """A real argparse namespace from the real shared search-filter flags, so
    these tests exercise the actual defaults (incl. the new --any-gpu)."""
    p = argparse.ArgumentParser()
    cli_search.add_search_filters(p)
    return p.parse_args(list(argv))


# =============================================================================
# search_offers (the search/launch/train auto-pick seam)
# =============================================================================
def test_default_pick_prefers_bf16_over_cheaper_turing(monkeypatch):
    """No --gpu: the cheapest PREFERRED card (RTX 5090) wins even though a
    Turing card is cheaper overall — the incident regression."""
    calls = _wire(monkeypatch)
    offers = market_offers.search_offers(_ns())
    assert offers[0]["gpu_name"] == "RTX 5090"
    assert all(o["gpu_name"] != "Quadro RTX 8000" for o in offers)
    # the query itself was restricted to tier 0 — the exclusion is by
    # allowlist construction, not client-side filtering luck
    assert calls[0]["gpu_name"] == {"in": list(market_offers.GPU_DEFAULT_POLICY_TIERS[0])}


def test_cheap_hopper_wins_on_price(monkeypatch):
    """Owner ruling 2026-08-07: Hopper/Ampere datacenter cards are ordinary
    tier-0 candidates, so an H100 priced under the 5090 is simply the pick.
    Under the old Blackwell-only tier 0 it was unreachable without --any-gpu —
    this test is what fails if anyone re-narrows the tier."""
    market = MARKET + [{"id": 60, "gpu_name": "H100 PCIE",
                        "dph_total": 0.30, "min_bid": 0.15}]
    _wire(monkeypatch, market)
    assert market_offers.search_offers(_ns())[0]["gpu_name"] == "H100 PCIE"
    # and it is in the FIRST tier — not a fallback the 5090's presence hides
    assert "H100 PCIE" in market_offers.GPU_DEFAULT_POLICY_TIERS[0]


def test_default_pick_falls_through_to_smaller_tier(monkeypatch):
    """Tier 0 dry (nothing >=32 GB on the market): the tier-1 tail (RTX 5070)
    is picked; the cheap Turing card still never surfaces."""
    market = [o for o in MARKET if o["id"] in (10, 50)]
    calls = _wire(monkeypatch, market)
    offers = market_offers.search_offers(_ns())
    assert [o["gpu_name"] for o in offers] == ["RTX 5070"]
    assert len(calls) == 2                       # tier 0 probed first, then tier 1
    assert calls[1]["gpu_name"] == {"in": list(market_offers.GPU_DEFAULT_POLICY_TIERS[1])}


def test_default_pick_returns_empty_not_turing_when_no_bf16(monkeypatch):
    """Only pre-Ampere on the market: the default pick finds NOTHING (the
    caller errors out with guidance) rather than landing on Turing."""
    _wire(monkeypatch, [MARKET[0]])
    assert market_offers.search_offers(_ns()) == []


def test_explicit_gpu_bypasses_policy(monkeypatch):
    """--gpu names a card: the policy steps aside entirely — even a Turing
    card is honored when the operator asked for it by name."""
    calls = _wire(monkeypatch)
    offers = market_offers.search_offers(_ns("--gpu", "Quadro RTX 8000"))
    assert offers[0]["id"] == 10
    assert calls[0]["gpu_name"] == {"in": ["Quadro RTX 8000"]}
    assert len(calls) == 1                       # one query, no tier loop


def test_any_gpu_restores_cheapest_overall(monkeypatch):
    """--any-gpu: old behavior — cheapest offer of ANY family, Turing included."""
    calls = _wire(monkeypatch)
    offers = market_offers.search_offers(_ns("--any-gpu"))
    assert offers[0]["id"] == 10                 # the cheap Quadro RTX 8000
    assert "gpu_name" not in calls[0]


def test_machine_pin_bypasses_policy(monkeypatch):
    """A --machine pin means the operator chose the hardware — no gpu_name
    restriction is injected on top of it."""
    calls = _wire(monkeypatch)
    market_offers.search_offers(_ns("--machine", "12345"))
    assert "gpu_name" not in calls[0]
    assert calls[0]["machine_id"] == {"in": [12345]}


# =============================================================================
# _search_offers_soft (supervise eviction-relaunch / handoff pick seam)
# =============================================================================
def test_soft_search_applies_policy(monkeypatch):
    """A relaunch search with no captured gpu name must not land the
    replacement box on pre-Ampere either."""
    _wire(monkeypatch)
    offers = market_offers._search_offers_soft(_ns())
    assert offers[0]["gpu_name"] == "RTX 5090"


def test_soft_search_explicit_gpu_unchanged(monkeypatch):
    _wire(monkeypatch)
    offers = market_offers._search_offers_soft(_ns("--gpu", "Quadro RTX 8000"))
    assert [o["id"] for o in offers] == [10]


# =============================================================================
# pick_cheapest_offer (argparse-free workflow / jobs-understudy seam)
# =============================================================================
def test_pick_cheapest_offer_default_prefers_preferred(monkeypatch):
    calls = _wire(monkeypatch)
    o = market_offers.pick_cheapest_offer(rental="bid")
    assert o["gpu_name"] == "RTX 5090"
    assert calls[0]["gpu_name"] == {"in": list(market_offers.GPU_DEFAULT_POLICY_TIERS[0])}


def test_pick_cheapest_offer_tier_fallback(monkeypatch):
    _wire(monkeypatch, [o for o in MARKET if o["id"] in (10, 50)])
    o = market_offers.pick_cheapest_offer(rental="bid")
    assert o["gpu_name"] == "RTX 5070"


def test_pick_cheapest_offer_none_when_only_preampere(monkeypatch):
    _wire(monkeypatch, [MARKET[0]])
    assert market_offers.pick_cheapest_offer(rental="bid") is None


def test_pick_cheapest_offer_explicit_gpu_bypasses(monkeypatch):
    calls = _wire(monkeypatch)
    o = market_offers.pick_cheapest_offer(gpu=("Quadro RTX 8000",), rental="bid")
    assert o["id"] == 10
    assert len(calls) == 1


def test_pick_cheapest_offer_any_gpu_escape_hatch(monkeypatch):
    calls = _wire(monkeypatch)
    o = market_offers.pick_cheapest_offer(rental="bid", any_gpu=True)
    assert o["id"] == 10                          # cheapest overall again
    assert "gpu_name" not in calls[0]


# =============================================================================
# policy table hygiene
# =============================================================================
def test_policy_tiers_contain_no_preampere_names():
    """The allowlist IS the exclusion mechanism — make sure nobody edits a
    Turing/Volta/Pascal name into it when updating for market shifts."""
    banned = ("quadro", "titan", "gtx", "tesla", "v100", "p100", "p40", "t4",
              "2080", "2070", "2060", "1080", "1070")
    for tier in market_offers.GPU_DEFAULT_POLICY_TIERS:
        for name in tier:
            assert not any(b in name.lower() for b in banned), name
