"""Portable tests for `vastlib.market` — the offer/pricing ring, ported at plan
§8 step 3.

Two jobs, and the second one is why this file is worth reading.

1. **Pin the ported copies against the live flat ones — WHILE THAT WAS
   POSSIBLE.** Step 3 was ADD-ONLY: `herdd.py` kept its own definitions, so
   both existed and could be compared input for input, which is stronger than
   any hand-written expectation because it fails on drift in either direction.
   Plan §8 step 6d ended that window — the launcher re-exports this ring by
   identity — so the nine pure-parity sweeps here are deleted and their
   comments say which. What is NOT deleted: every invariant that names a
   THIRD party (`ladder_core` aliases, `bidpolicy._bid_target` delegation,
   the two-builder VRAM-floor agreement, the conftest guard roster), because
   those still compare two distinct objects. The invariants the flat suite
   pins (`test_gpu_ram_floor.py`, `test_gpu_name_aliases.py`,
   `test_bid_cushion.py`) reach this ring now that the launcher is thin.

2. **Prove the live-market guard actually reaches the new module.**
   `conftest._isolate_market_ondemand` stubs `_market_ondemand_soft` because,
   left live, that probe walks up to the repo `.env`, finds a real API key and
   queries the real market from a unit test. It degrades by `hasattr`, so a
   module it does not know about is not a failing guard — it is a silent hole.
   The vastlib copy is now on its roster; `test_every_market_ondemand_guard_
   target_exists` asserts the roster resolves, and
   `test_vastlib_ondemand_probe_is_intercepted` proves the interception on the
   vastlib path specifically, with the network rigged to fail the test if it is
   ever reached.

Offline lane: no vast API, no network, $0. Every bundles read in this ring is a
**POST** to `v0/bundles/`, which `conftest._block_mutating_api_calls` refuses —
so a test that wants rows stubs `vastlib.core.api.request_soft` by module
ATTRIBUTE (the port calls it that way precisely so the patch idiom survives),
and a test that stubs nothing asserts the soft degradation instead.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bidpolicy  # noqa: E402
import conftest  # noqa: E402
import ladder_core  # noqa: E402
import herdd as v  # noqa: E402
from vastlib.core import api, models  # noqa: E402
from vastlib.market import offers, pricing  # noqa: E402


def _search_ns(**kw):
    """The argparse namespace `build_search_query` reads, with the same defaults
    the CLI parser produces (mirrors `test_gpu_ram_floor._search_ns`)."""
    base = dict(limit=20, type="bid", num_gpus=1, unverified=False, gpu=None,
                gpu_ram=0, max_dph=None, host_disk=0, reliability=0, cuda=0,
                inet_down=None, machine=None, host=None, geo=None,
                exclude_machines=None, any_inet=False)
    base.update(kw)
    return argparse.Namespace(**base)


# The REAL probe, captured at import time — i.e. before any function-scoped
# fixture runs. `conftest._isolate_market_ondemand` replaces the module
# attribute for every test, which is the point (and which
# `test_vastlib_ondemand_probe_is_intercepted` proves), so a test of the probe
# ITSELF has to put this back deliberately.
_REAL_MARKET_ONDEMAND_SOFT = pricing._market_ondemand_soft


def _rows(*offer_dicts):
    """A `request_soft` stub that answers every POST with these offer rows and
    records the query bodies it was given."""
    seen = []

    def _fake(method, path, body=None, *a, **k):
        seen.append((method, path, body))
        return True, {"offers": list(offer_dicts)}, None

    return _fake, seen


# --------------------------------------------------------------------------- #
# offers.py — GPU names and the policy tiers
# --------------------------------------------------------------------------- #
def test_gpu_aliases_and_tiers_are_the_same_table():
    """The alias table and the policy tiers are DATA, so equality is the whole
    check. An entry silently dropped in the move is an empty market, not an
    error — the failure mode the table exists to prevent."""
    # Post-6d these four are the launcher's re-exported objects, so equality is
    # identity; a second table in `herdd.py` is the drop-an-entry hazard
    # coming back, and that is what this now checks.
    for name in ("GPU_ALIASES", "GPU_DEFAULT_POLICY_TIERS",
                 "VRAM_SEARCH_TOLERANCE", "OFFER_SCAN_LIMIT"):
        assert getattr(v, name) is getattr(offers, name), name
    assert offers.GPU_ALIASES and offers.GPU_DEFAULT_POLICY_TIERS


# `test_normalize_gpu_agrees_with_the_flat_copy` swept twelve spellings through
# both copies (the whitespace pass, the alias families, the exact name, the
# unfamilied passthrough). One copy since step 6d; the round-trip invariant
# below is the independent statement that survives.


def test_normalize_gpu_dedups_preserving_order():
    """Two aliases of one family must collapse to one ordered list — the query
    filter is `{"in": [...]}` and a duplicate is a wasted slot, not an error."""
    got = offers.normalize_gpu(["h100", "H100 SXM", "a100"])
    assert len(got) == len(set(got))
    assert got[0] == "H100 SXM"


def test_every_policy_tier_name_round_trips_through_normalize_gpu():
    """The anti-drift invariant `test_gpu_name_aliases.py` pins on the flat
    copy, re-asserted on the ported one: every card the default policy will
    auto-pick must be nameable — spelled exactly at minimum (the passthrough),
    or reachable inside its own alias family (`H200` widens to the H200 pair)."""
    for tier in offers.GPU_DEFAULT_POLICY_TIERS:
        for name in tier:
            assert offers.normalize_gpu([name]) == [name] or \
                name in offers.normalize_gpu([name]), name


# `test_gpu_family_names_agrees_with_the_flat_copy` (five names, both copies)
# was here. One copy since step 6d; the widening property it stood for is
# asserted outright below.


def test_gpu_family_names_widens_an_h100_pin():
    """The reason the function exists: an exact-SKU pin builds a one-offer
    candidate set out of what is really one 80 GB card class."""
    fam = offers.gpu_family_names("H100 NVL")
    assert set(fam) == {"H100 SXM", "H100 PCIE", "H100 NVL"}


@pytest.mark.parametrize("kw,tiered", [
    ({}, True),                              # no name, no pin -> policy applies
    ({"gpu": ["h100"]}, False),              # explicit --gpu
    ({"any_gpu": True}, False),              # escape hatch
    ({"machine": [123]}, False),             # operator pinned the hardware
    ({"host": [7]}, False),
    ({"exclude_machines": [123]}, True),     # rotation is NOT a pin
])
def test_gpu_policy_tiers_bypass_matrix(kw, tiered):
    ns = _search_ns(**dict({"any_gpu": False}, **kw))
    got = offers._gpu_policy_tiers(ns)
    assert (got is not None) is tiered


# --------------------------------------------------------------------------- #
# offers.py — the VRAM floor and the two query builders
# --------------------------------------------------------------------------- #
# `test_gpu_ram_floor_agrees_with_the_flat_copy` swept sixteen declared sizes
# through both copies. One copy since step 6d — and the 2026-08-16 regression it
# guarded is pinned by value in the test below, which is the arm that matters.


def test_gpu_ram_floor_admits_the_48gb_class_and_rejects_a_smaller_card():
    """The 2026-08-16 regression, re-pinned on the ported copy: a "48 GB" card
    advertises 49140 MiB, the naive 49152 floor excluded every one of them, and
    an ECC-capacity 46068 MiB part must still NOT satisfy a declared 48."""
    floor = offers.gpu_ram_floor_mib(48)
    assert floor <= 49140
    assert floor <= 48935                      # RTX PRO 5000
    assert floor > 46068                       # ECC-on L40/L40S/A6000
    assert floor > 32760


# `test_build_search_query_matches_the_flat_copy` built one fully-populated
# namespace and compared the two builders' dicts. One builder since step 6d;
# the two tests below pin the query keys that drift has actually cost us
# (machine pin vs exclude, and the VRAM floor shared with `pick_offers`).


def test_build_search_query_machine_pin_beats_exclude_machines():
    """Mutually exclusive by construction: excluding a pinned machine searches
    nothing, so an operator pin wins."""
    q = offers.build_search_query(_search_ns(machine=[42], exclude_machines=[42]))
    assert q["machine_id"] == {"in": [42]}
    q2 = offers.build_search_query(_search_ns(exclude_machines=[42, 43]))
    assert q2["machine_id"] == {"notin": [42, 43]}


def test_search_query_floors_container_disk_at_the_requested_disk():
    """The CLI lane's offer search must carry the `--disk` request as a
    `disk_space` floor: a machine advertising less does not refuse the rental,
    it hands back a smaller container."""
    q = offers.build_search_query(_search_ns(disk=50))
    assert q["disk_space"] == {"gte": 50.0}
    # --host-disk still works, and the two are a max(), not a choice
    assert offers.build_search_query(
        _search_ns(disk=50, host_disk=200))["disk_space"] == {"gte": 200.0}
    assert offers.build_search_query(
        _search_ns(disk=200, host_disk=50))["disk_space"] == {"gte": 200.0}
    # a namespace with no --disk at all (search/dash surveys) stays permissive
    assert "disk_space" not in offers.build_search_query(_search_ns())
    for junk in (None, 0, -5, "", "big"):
        assert "disk_space" not in offers.build_search_query(_search_ns(disk=junk))


def test_the_two_query_builders_still_agree_on_the_vram_floor(monkeypatch):
    """The invariant `test_gpu_ram_floor.py` pins between `build_search_query`
    and `pick_offers` — two hand-written builders whose drift (0.96 vs exact)
    made the whole 48 GB class unrentable. They are deliberately NOT unified by
    the port, so the agreement has to be re-asserted on the ported pair."""
    fake, seen = _rows()
    monkeypatch.setattr(api, "request_soft", fake)
    offers.pick_offers(gpu=["a6000"], gpu_ram_gb=48, any_gpu=False, any_inet=True)
    picker_q = seen[0][2]
    cli_q = offers.build_search_query(
        _search_ns(gpu=["a6000"], gpu_ram=48, any_inet=True))
    assert picker_q["gpu_ram"] == cli_q["gpu_ram"] == {"gte": offers.gpu_ram_floor_mib(48)}
    assert picker_q["gpu_name"] == cli_q["gpu_name"]


def test_pick_offers_accepts_both_ondemand_spellings(monkeypatch):
    """`on-demand` is workflow.RENTAL_CHOICES' frozen external spelling and
    `ondemand` is vast-native; normalizing to one inside the port would break
    the workflow controller."""
    for spelling in ("ondemand", "on-demand"):
        fake, seen = _rows()
        monkeypatch.setattr(api, "request_soft", fake)
        offers.pick_offers(rental=spelling, gpu=["h100"], any_inet=True)
        assert seen[0][2]["type"] == "ondemand"
        assert seen[0][2]["order"] == [["dph_total", "asc"]]


def test_pick_offers_returns_the_whole_candidate_set_cheapest_first(monkeypatch):
    """`limit` is what makes a candidate SET possible — a per-offer rail
    evaluated against a sample of one is a coin flip (2026-08-16)."""
    fake, seen = _rows({"id": 1, "min_bid": 0.40}, {"id": 2, "min_bid": 0.55})
    monkeypatch.setattr(api, "request_soft", fake)
    got = offers.pick_offers(gpu=["h200"], limit=5, any_inet=True)
    assert [o["id"] for o in got] == [1, 2]
    assert seen[0][2]["limit"] == 5


def test_pick_cheapest_offer_survives_as_its_own_name(monkeypatch):
    """workflowctl binds this name and `test_workflow.py` patches it as a
    raiser; collapsing it into `pick_offers` breaks an external binding and a
    guard-only seam at once."""
    calls = []
    monkeypatch.setattr(offers, "pick_offers",
                        lambda **kw: (calls.append(kw) or [{"id": 9}]))
    assert offers.pick_cheapest_offer(gpu=["h100"]) == {"id": 9}
    assert calls == [{"limit": 1, "gpu": ["h100"]}]
    monkeypatch.setattr(offers, "pick_offers", lambda **kw: [])
    assert offers.pick_cheapest_offer() is None


# --------------------------------------------------------------------------- #
# offers.py — the inet floor, the tier search, and the soft/hard split
# --------------------------------------------------------------------------- #
def test_inet_floor_precedence_matches_the_flat_copy(monkeypatch):
    """Explicit wins verbatim (0 disables), a pin or --any-inet suppresses the
    default, otherwise the knob applies. The knob is read through
    `vastlib.core.config._boot_knob`, so setting the env var steers both."""
    monkeypatch.setenv("LAUNCH_INET_DOWN_MBPS", "1500")
    assert offers._inet_floor(None) == 1500.0
    assert offers._inet_floor(250) == 250.0
    assert offers._inet_floor(0) is None
    assert offers._inet_floor(None, pinned=True) is None
    assert offers._inet_floor(None, any_inet=True) is None
    assert offers._inet_floor("junk") is None


def test_inet_floor_for_reads_the_namespace_getattr_safely(monkeypatch):
    """Relaunch and probe namespaces predate several flags — an AttributeError
    here would take a launch down."""
    monkeypatch.setenv("LAUNCH_INET_DOWN_MBPS", "1000")
    assert offers._inet_floor_for(argparse.Namespace()) == 1000.0
    assert offers._inet_floor_for(_search_ns(machine=[1])) is None
    assert offers._inet_floor_for(_search_ns(any_inet=True)) is None
    assert offers._inet_floor_for(_search_ns(inet_down=0)) is None


def test_search_offers_walks_the_tiers_through_the_module_attribute(monkeypatch):
    """The HARD path: `search_offers` goes through `api.request` (which
    sys.exits on error) and it must be called by ATTRIBUTE, or every
    monkeypatch that steers it is a silent no-op. Tier 0 is dry here, so the
    walk has to reach tier 1 and stop there."""
    seen = []

    def _fake_request(method, path, body=None, timeout=60):
        seen.append(body["gpu_name"]["in"])
        return {"offers": [{"id": 7}] if "RTX 4090" in body["gpu_name"]["in"] else []}

    monkeypatch.setattr(api, "request", _fake_request)
    got = offers.search_offers(_search_ns())
    assert got == [{"id": 7}]
    assert len(seen) == 2                          # tier 0 empty, tier 1 hit
    assert seen[0] == list(offers.GPU_DEFAULT_POLICY_TIERS[0])


def test_search_offers_single_query_when_the_policy_is_bypassed(monkeypatch):
    seen = []

    def _fake_request(method, path, body=None, timeout=60):
        seen.append(body)
        return {"offers": []}

    monkeypatch.setattr(api, "request", _fake_request)
    assert offers.search_offers(_search_ns(gpu=["h100"])) == []
    assert len(seen) == 1
    assert seen[0]["gpu_name"] == {"in": offers.normalize_gpu(["h100"])}


def test_search_offers_soft_retries_unfloored_and_says_so(monkeypatch, capsys):
    """The rescue/relaunch contract: a pick must never fail outright for want
    of a fast host, and the fallback prints a user-visible note. An EXPLICIT
    --inet-down stays hard, so only the DEFAULT floor gets the second pass."""
    monkeypatch.setenv("LAUNCH_INET_DOWN_MBPS", "1000")
    seen = []

    def _fake(method, path, body=None, *a, **k):
        seen.append(body)
        if "inet_down" in body:
            return True, {"offers": []}, None
        return True, {"offers": [{"id": 3}]}, None

    monkeypatch.setattr(api, "request_soft", _fake)
    got = offers._search_offers_soft(_search_ns(gpu=["h100"]))
    assert got == [{"id": 3}]
    assert len(seen) == 2 and "inet_down" in seen[0] and "inet_down" not in seen[1]
    assert ">> note: no offer clears the default inet-down floor" in capsys.readouterr().out


def test_search_offers_soft_keeps_an_explicit_floor_hard(monkeypatch):
    seen = []

    def _fake(method, path, body=None, *a, **k):
        seen.append(body)
        return True, {"offers": []}, None

    monkeypatch.setattr(api, "request_soft", _fake)
    assert offers._search_offers_soft(_search_ns(gpu=["h100"], inet_down=5000)) == []
    assert len(seen) == 1 and seen[0]["inet_down"] == {"gte": 5000.0}


def test_offer_machine_scan_matches_the_id_in_python(monkeypatch):
    """The API's `id` filter is dead, so the row is recovered by scanning an
    unfiltered query with a raised limit and matching in Python."""
    fake, seen = _rows({"id": 11, "min_bid": 0.1}, {"id": 22, "min_bid": 0.2})
    monkeypatch.setattr(api, "request_soft", fake)
    ns = _search_ns(offer=22, offer_machine=None)
    assert offers._offer_machine_scan_soft(ns) == {"id": 22, "min_bid": 0.2}
    assert seen[0][2]["limit"] == offers.OFFER_SCAN_LIMIT
    assert "id" not in seen[0][2]
    assert offers._offer_machine_scan_soft(_search_ns(offer=999)) is None
    assert offers._offer_machine_scan_soft(_search_ns(offer=None)) is None


def test_offer_cuda_soft_prefers_the_recovered_row(monkeypatch):
    """The `row` path is the one that works; the id-filtered fallback is a rung
    expected to answer nothing, and callers must degrade to a warning."""
    def _boom(*a, **k):
        raise AssertionError("the row path must not hit the API")

    monkeypatch.setattr(api, "request_soft", _boom)
    assert offers._offer_cuda_soft(1, {"cuda_max_good": 12.8}) == 12.8

    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (True, {"offers": []}, None))
    assert offers._offer_cuda_soft(1) is None


def test_pick_offers_is_soft_when_nothing_stubs_the_api():
    """No stub: `conftest._block_mutating_api_calls` refuses the POST (every
    bundles read in this ring is a POST) and the picker degrades to [] rather
    than raising or reaching the network."""
    assert offers.pick_offers(gpu=["h100"], any_inet=True) == []
    assert offers.pick_cheapest_offer(gpu=["h100"], any_inet=True) is None


# --------------------------------------------------------------------------- #
# pricing.py — the ladder glue and the bid number
# --------------------------------------------------------------------------- #
def test_ladder_glue_is_an_identity_alias_not_a_reimplementation():
    """The five names are re-exports of ONE copy of the state transitions. `is`,
    not `==`: several of them mutate lane dicts in place and their keys are a
    wire format fleetd persists, so a second implementation would drift the
    durable state, not just the behavior."""
    assert pricing.BID_HISTORY_MAX == ladder_core.BID_HISTORY_MAX
    assert pricing._note_standing_bid is ladder_core.note_standing_bid
    assert pricing._hist_field is ladder_core.hist_field
    assert pricing._bid_history_for is ladder_core.bid_history_for
    assert pricing._self_floor_reset is ladder_core.self_floor_reset
    # ...and the same objects the flat module re-exports.
    assert pricing._note_standing_bid is v._note_standing_bid
    assert pricing._hist_field is v._hist_field
    assert pricing._bid_history_for is v._bid_history_for
    assert pricing._self_floor_reset is v._self_floor_reset


@pytest.mark.parametrize("floor,od", [(0.10, None), (0.60, 1.00), (0.25, 1.00),
                                      (1.0667, 2.40), (1.32, 2.40), (1.44, 2.40),
                                      (0.746, 1.50), (None, 1.0), ("junk", 1.0)])
def test_auto_bid_price_is_a_pure_delegation(floor, od):
    """`test_bid_cushion.py:301`'s anti-drift identity, re-asserted on the
    ported copy: the launch price IS `_bid_target(floor, None, on_demand)`.
    That delegation is what enforces SPOT_DESIGN §3.2's "launch price ==
    steady-state target" invariant — local arithmetic here would drift the
    moment a rail changes in bidpolicy."""
    got = pricing._auto_bid_price(floor, od)
    # The `== v._auto_bid_price(...)` arm went at step 6d (one body). The
    # delegation identity below is the one that was never parity: it compares
    # this ring against `bidpolicy`, a Zone S leaf that was never ported.
    if models._num_dph(floor) is not None:
        assert got == bidpolicy._bid_target(models._num_dph(floor), None,
                                            models._num_dph(od))
    else:
        assert got is None


def test_odprobe_caps_are_the_same_numbers():
    assert pricing.HANDOFF_ODPROBE_MAX == 8
    assert pricing.RELAUNCH_ODPROBE_MAX == 5
    assert v.HANDOFF_ODPROBE_MAX is pricing.HANDOFF_ODPROBE_MAX
    assert v.RELAUNCH_ODPROBE_MAX is pricing.RELAUNCH_ODPROBE_MAX


# --------------------------------------------------------------------------- #
# pricing.py — the market reads
# --------------------------------------------------------------------------- #
_CHUNKS = [{"num_gpus": 1, "min_bid": 0.1333},
           {"num_gpus": 2, "min_bid": 0.2667},
           {"num_gpus": 4, "min_bid": 0.5334}]


# `test_market_chunk_floors_agree_with_the_flat_copy` swept six `num_gpus`
# values through both copies of both readers. One copy each since step 6d; the
# D5 defect they were guarding is pinned by value in the tests below.


def test_market_chunk_floors_reads_our_own_chunk_not_the_cheapest():
    """Defect D5: the bare min() read the 1-GPU floor ($0.1333) for a 2-GPU box
    (real floor $0.2667) and vast underbid-parked it a poll later."""
    floors, scaled = pricing._market_chunk_floors(_CHUNKS, 2)
    assert floors == [0.2667] and scaled is False
    # No exact-count offer -> per-GPU rescale, and the row is FLAGGED as
    # synthesized (F8): such a number can never match our bid history.
    floors, scaled = pricing._market_chunk_floors(_CHUNKS, 3)
    assert scaled is True and floors == [round(0.1333 * 3, 4)]


def test_market_chunk_floors_keeps_the_rows_apart():
    """Review F3: on a machine we are a tenant of, one query returns both our
    own bid echo and a genuine sibling floor. A min() collapse hides the
    sibling, so the guards filter ROWS, not the scalar."""
    rows = [{"num_gpus": 2, "min_bid": 0.30}, {"num_gpus": 2, "min_bid": 0.45}]
    floors, scaled = pricing._market_chunk_floors(rows, 2)
    assert floors == [0.30, 0.45] and scaled is False
    assert pricing._market_chunk_floor(rows, 2) == 0.30


def test_market_min_bid_read_tri_state(monkeypatch):
    """`ok=False` is IGNORANCE and must never advance eviction state;
    `ok=True, listed=False` is positive displacement evidence (defect D7)."""
    assert pricing._market_min_bid_read(None) == (False, False, None, (), False)

    monkeypatch.setattr(api, "request_soft", lambda *a, **k: (False, None, "HTTP 500"))
    r = pricing._market_min_bid_read(555, 2)
    assert r.ok is False and r.listed is False and r.min_bid is None
    assert pricing._market_bid_listed_soft(555, 2) is None
    assert pricing._market_min_bid_soft(555, 2) is None

    monkeypatch.setattr(api, "request_soft", lambda *a, **k: (True, {"offers": []}, None))
    r = pricing._market_min_bid_read(555, 2)
    assert r.ok is True and r.listed is False and r.min_bid is None
    assert pricing._market_bid_listed_soft(555, 2) is False

    fake, seen = _rows(*_CHUNKS)
    monkeypatch.setattr(api, "request_soft", fake)
    r = pricing._market_min_bid_read(555, 2)
    assert (r.ok, r.listed, r.min_bid, r.floors, r.scaled) == (True, True, 0.2667,
                                                               [0.2667], False)
    assert pricing._market_bid_listed_soft(555, 2) is True
    assert pricing._market_min_bid_soft(555, 2) == 0.2667
    assert seen[0][2]["type"] == "bid" and seen[0][2]["order"] == [["min_bid", "asc"]]


def test_market_min_bid_read_returns_the_shared_marketread(monkeypatch):
    """The type is `core.models.MarketRead` — imported, never redeclared, so
    the 3-positional construction and the defaulted `floors`/`scaled` stay one
    contract across the package."""
    from vastlib.core import models
    monkeypatch.setattr(api, "request_soft", lambda *a, **k: (True, {"offers": []}, None))
    assert isinstance(pricing._market_min_bid_read(555), models.MarketRead)


def test_market_min_bid_read_drops_unreadable_rows(monkeypatch):
    """The pre-filter the pure chunk helper depends on: a row whose `min_bid`
    is junk is dropped BEFORE the arithmetic, never coerced to a floor."""
    fake, _ = _rows({"num_gpus": 2, "min_bid": None},
                    {"num_gpus": 2, "min_bid": "junk"},
                    {"num_gpus": 2, "min_bid": 0.5})
    monkeypatch.setattr(api, "request_soft", fake)
    assert pricing._market_min_bid_read(555, 2).min_bid == 0.5


def test_machine_offers_soft_projects_the_three_field_row(monkeypatch):
    """One no-`type` POST answers reserved price, spot price and capacity at
    once; the `{g, base, bid}` keys are `core.models.MachineRow` and a wire
    shape `_rates` reads."""
    fake, seen = _rows({"num_gpus": 2, "dph_base": 0.80, "min_bid": 0.27},
                       {"num_gpus": None, "dph_base": 1.0},         # skipped
                       {"num_gpus": 4, "dph_base": None, "min_bid": None})
    monkeypatch.setattr(api, "request_soft", fake)
    got = pricing._machine_offers_soft(777)
    assert got == [{"g": 2, "base": 0.80, "bid": 0.27},
                   {"g": 4, "base": None, "bid": None}]
    assert "type" not in seen[0][2] and seen[0][2]["limit"] == 64
    assert pricing._machine_offers_soft(None) is None

    monkeypatch.setattr(api, "request_soft", lambda *a, **k: (False, None, "boom"))
    assert pricing._machine_offers_soft(777) is None


def test_market_ondemand_soft_picks_our_gpu_config(monkeypatch):
    """Exact GPU-count match, else the smallest covering chunk, else the min
    base seen — mirroring `models._rates`. None disables the clamp."""
    fake, _ = _rows({"num_gpus": 1, "dph_base": 0.50, "min_bid": 0.1},
                    {"num_gpus": 2, "dph_base": 0.90, "min_bid": 0.2},
                    {"num_gpus": 4, "dph_base": 1.70, "min_bid": 0.4})
    monkeypatch.setattr(api, "request_soft", fake)
    monkeypatch.setattr(pricing, "_market_ondemand_soft", _REAL_MARKET_ONDEMAND_SOFT)
    assert pricing._market_ondemand_soft(777, 2) == 0.90
    assert pricing._market_ondemand_soft(777, 3) == 1.70      # smallest covering
    assert pricing._market_ondemand_soft(777) == 0.50         # min base seen
    assert pricing._market_ondemand_soft(None, 2) is None


def test_offer_ondemand_ref_never_reads_the_bid_rows_dph_total(monkeypatch):
    """The doc 50 R1 razor-thin-bid defect: on a BID row `dph_total` is the
    current interruptible price (min_bid + storage sliver) and `dph_base`
    equals min_bid, so the on-demand reference has to come off the machine's
    ON-DEMAND offers instead."""
    row = {"machine_id": 777, "num_gpus": 2, "min_bid": 0.2667,
           "dph_total": 0.2711, "dph_base": 0.2667}
    monkeypatch.setattr(pricing, "_market_ondemand_soft",
                        lambda mid, num_gpus=None: 0.5111 if mid == 777 else None)
    assert pricing._offer_ondemand_ref(row) == 0.5111
    assert pricing._offer_ondemand_ref({"machine_id": 1}) is None
    assert pricing._offer_ondemand_ref(None) is None
    assert pricing._offer_ondemand_ref("not-a-row") is None


def test_offer_pricing_soft_is_the_dead_rung_it_documents(monkeypatch):
    """MEASURED DEAD 2026-08-09: the `id` filter returns zero rows in every
    view. Ported as a rung; its real contract is "(None, None, None), almost
    always" and nothing may depend on it resolving."""
    monkeypatch.setattr(api, "request_soft", lambda *a, **k: (True, {"offers": []}, None))
    assert pricing._offer_pricing_soft(123) == (None, None, None)
    monkeypatch.setattr(api, "request_soft", lambda *a, **k: (False, None, "HTTP 404"))
    assert pricing._offer_pricing_soft(123) == (None, None, None)

    fake, seen = _rows({"min_bid": 0.2, "dph_total": 0.21, "machine_id": 9})
    monkeypatch.setattr(api, "request_soft", fake)
    assert pricing._offer_pricing_soft("123") == (0.2, 0.21, 9)
    assert seen[0][2]["id"] == {"in": [123]}       # coerced to int when it can be


# --------------------------------------------------------------------------- #
# the live-market guard — the fixture that must not degrade quietly
# --------------------------------------------------------------------------- #
def test_every_market_ondemand_guard_target_exists():
    """A guard that cannot find its target does not fail — it stops guarding.

    `_isolate_market_ondemand` walks `sys.modules` and skips anything absent or
    lacking `_market_ondemand_soft`, which is correct for an unimported module
    and is exactly why a RENAME would be silent: `hasattr` goes False, the
    fixture returns, and the next unit test queries the real market with a real
    key off the repo `.env`. So the roster is asserted here, by importing each
    target and demanding a callable — the same meta-test the api wave added for
    `request_soft`, for the same failure mode.

    Adding a third copy of the probe means adding it to
    `conftest._GUARDED_MARKET_ONDEMAND_MODULES`, and this test is what says so.
    """
    assert "herdd" in conftest._GUARDED_MARKET_ONDEMAND_MODULES
    assert "vastlib.market.pricing" in conftest._GUARDED_MARKET_ONDEMAND_MODULES
    for modname in conftest._GUARDED_MARKET_ONDEMAND_MODULES:
        mod = importlib.import_module(modname)
        fn = getattr(mod, "_market_ondemand_soft", None)
        assert callable(fn), f"{modname}._market_ondemand_soft is not a callable guard target"


def test_vastlib_ondemand_probe_is_intercepted(monkeypatch):
    """Interception on the VASTLIB path specifically, not just on `herdd`.

    Fail-safe by construction: `request_soft` is rigged to fail the test, so if
    the fixture were NOT covering this module the probe would try the API and
    blow up here rather than quietly billing a real read. The constant is
    conftest's own, so this cannot pass by coincidence."""
    def _boom(*a, **k):
        raise AssertionError("unguarded live-market probe through vastlib.market.pricing")

    monkeypatch.setattr(api, "request_soft", _boom)
    assert pricing._market_ondemand_soft(47214941, 2) == conftest._TEST_MARKET_ONDEMAND
    # ...and the caller that reads the clamp reference gets the stub too, since
    # `_offer_ondemand_ref` resolves the name on this module at call time.
    assert pricing._offer_ondemand_ref({"machine_id": 47214941, "num_gpus": 2}) == \
        conftest._TEST_MARKET_ONDEMAND


def test_the_flat_copy_is_still_guarded_too():
    """NOT a parity test, and not tautological post-6d — keep it.

    One body, two guarded BINDINGS: `conftest._GUARDED_MARKET_ONDEMAND_MODULES`
    stubs each module attribute separately, and `herdd._market_ondemand_soft`
    is the spelling the flat-module consumers reach. Drop `"herdd"` from that
    tuple and this probe walks up to the repo `.env`, finds a real key and
    queries the real market from a unit test."""
    assert v._market_ondemand_soft(47214941, 2) == conftest._TEST_MARKET_ONDEMAND


def test_a_mutating_call_through_the_market_ring_is_refused():
    """`_block_mutating_api_calls` wraps `vastlib.core.api.request_soft`, which
    is the seam every read in this ring funnels through by module attribute. An
    unstubbed POST comes back in the `(False, None, err)` shape the soft
    contract already survives — which is why the pickers degrade to [] rather
    than reaching the network."""
    ok, data, err = api.request_soft("POST", "v0/bundles/", {"limit": 1})
    assert ok is False and data is None
    assert "test isolation" in err



# --------------------------------------------------------------------------- #
# Re-port discipline: peer drift on main must go red here, not silent
# --------------------------------------------------------------------------- #
def test_pick_offers_signature_is_the_re_ported_one():
    """The 2026-08-16 rebase carried peer commit 49bc0103 (pick_offers grew a
    disk_gb container-disk floor) and the behavior parity tests stayed GREEN
    because none exercised the new axis — silent drift, the exact fork risk of
    the two-copies window. This was the drift tripwire for that window: a
    signature diff against `v.pick_offers`, red the moment a peer edited the
    flat copy.

    Plan §8 step 6d closed the window it watched — there is no flat copy left
    for a peer to edit, and `v.pick_offers` IS this function. What replaces it
    is the disk_gb parameter pinned by NAME and default, so the re-ported axis
    cannot quietly vanish from the surviving copy either."""
    import inspect
    ours = inspect.signature(offers.pick_offers).parameters
    assert "disk_gb" in ours, "the 49bc0103 re-port was lost"
    assert ours["disk_gb"].default in (None, 0), ours["disk_gb"].default
    assert v.pick_offers is offers.pick_offers


def test_pick_offers_disk_gb_floor_reaches_the_query(monkeypatch):
    """disk_gb=N adds {"disk_space": {"gte": float(N)}} (GB, NOT MiB like
    gpu_ram); falsy disk_gb adds nothing. Pinned on both copies so the filter
    that stops 23GB machines being rented for 50GB workloads cannot be lost in
    either home."""
    fake, seen = _rows({"id": 1, "min_bid": 0.5, "dph_total": 0.6})
    monkeypatch.setattr(api, "request_soft", fake)
    # The second arm drove `v.pick_offers` with `monkeypatch.setattr(v,
    # "request_soft", fake)`. Post-6d both were the same object as the first
    # arm's — and worse, that patch bound a name in the launcher's namespace
    # that nothing reads (a re-export is not a patch point). Dropped: one call,
    # one seam, patched on the module that owns it.
    offers.pick_offers(gpu=("RTX_5090",), disk_gb=50, limit=1)
    assert len(seen) == 1
    for _m, _p, body in seen:
        assert body["disk_space"] == {"gte": 50.0}
    seen.clear()
    offers.pick_offers(gpu=("RTX_5090",), disk_gb=0, limit=1)
    offers.pick_offers(gpu=("RTX_5090",), limit=1)
    assert all("disk_space" not in body for _m, _p, body in seen)


# --------------------------------------------------------------------------- #
# the sm ARCHITECTURE allowlist (2026-08-18)
# --------------------------------------------------------------------------- #
def test_cc_allow_parses_every_spelling_it_will_be_handed():
    """The stamp is a string, a restored watch key is a list, and a hand-typed
    value may be either an sm level or vast's compute_cap (sm x10) spelling of
    the same thing."""
    assert offers.parse_cc_allow("80,86,89,90") == (80, 86, 89, 90)
    assert offers.parse_cc_allow([90, "sm_80", "SM86"]) == (80, 86, 90)
    assert offers.parse_cc_allow(" 90 , 80 ") == (80, 90)
    # compute_cap spelling folds onto the same sm levels; sm_120 stays 120
    assert offers.parse_cc_allow("800,1200") == (80, 120)
    assert offers.parse_cc_allow("120") == (120,)
    # absent / garbage is NO CONSTRAINT, never an empty allowlist that excludes
    for junk in (None, "", "  ", [], "nvidia", 0, -5):
        assert offers.parse_cc_allow(junk) == ()


def test_a_card_name_and_an_allowlist_that_cannot_both_hold_are_detectable():
    """A `--gpu NAME` filter and a `--cc-allow` list are an AND. When the named
    card's silicon is outside the list the search returns nothing and vast
    reports that as an empty market, so launch_jobs_box.sh asks this question
    BEFORE searching. Unknown names answer "cannot tell", never "conflict"."""
    assert offers.gpu_alias_sm("h100") == (90,)
    assert offers.gpu_alias_sm("rtxpro6000") == (120,)
    assert offers.gpu_alias_sm("a100") == (80,)
    assert offers.gpu_alias_sm("l40") == (89,)          # alias -> L40 + L40S
    assert offers.gpu_alias_sm("RTX 6000Ada") == (89,)  # exact vast spelling
    assert offers.gpu_alias_sm("some-2027-card") == ()
    assert offers.gpu_alias_sm(None) == ()

    fa2 = (80, 86, 89, 90)
    assert offers.gpu_alias_conflicts("rtxpro6000", fa2) == (120,)
    assert offers.gpu_alias_conflicts("b200", fa2) == (100,)
    assert offers.gpu_alias_conflicts("h100", fa2) == ()
    assert offers.gpu_alias_conflicts("a100", (90,)) == (80,)
    # no allowlist, or a card we cannot place, is not a conflict
    assert offers.gpu_alias_conflicts("rtxpro6000", ()) == ()
    assert offers.gpu_alias_conflicts("some-2027-card", (90,)) == ()


def test_an_offer_with_no_compute_cap_is_excluded_by_an_active_allowlist():
    """`compute_cap` is sm x10 and MAY be absent on a row. An unknown must not
    be the thing that smuggles an sm_120 past a list that exists precisely
    because one architecture cannot run the job."""
    assert offers.offer_sm({"compute_cap": 900}) == 90
    assert offers.offer_sm({"compute_cap": 1200}) == 120
    assert offers.offer_sm({"compute_cap": None}) is None
    assert offers.offer_sm({}) is None
    allow = (80, 86, 89, 90)
    assert offers.cc_allow_ok({"compute_cap": 900}, allow) is True
    assert offers.cc_allow_ok({"compute_cap": 1200}, allow) is False
    assert offers.cc_allow_ok({"compute_cap": None}, allow) is False
    # no allowlist = no filtering, including for the unknown row
    assert offers.cc_allow_ok({"compute_cap": None}, ()) is True


def test_pick_offers_keeps_only_in_list_architectures(monkeypatch, capsys):
    """The defect, at the picker: an A100 (sm_80) workload must not be handed an
    RTX PRO 6000 (sm_120) whose flash_attn has no kernel image for it. The
    filter is CLIENT-SIDE, so the query over-fetches and the survivors are
    trimmed to `limit`."""
    fake, seen = _rows({"id": 1, "min_bid": 0.30, "compute_cap": 1200},
                       {"id": 2, "min_bid": 0.40, "compute_cap": None},
                       {"id": 3, "min_bid": 0.50, "compute_cap": 800})
    monkeypatch.setattr(api, "request_soft", fake)
    got = offers.pick_offers(gpu=["a100"], cc_allow=(80, 86, 89, 90), limit=1,
                             any_inet=True)
    assert [o["id"] for o in got] == [3], "the cheapest rows are out of list"
    # over-fetch: a limit-1 query would have returned only the sm_120 row and
    # the filter would have emptied the market
    assert seen[0][2]["limit"] == offers.CC_ALLOW_SCAN_LIMIT
    out = capsys.readouterr().out
    assert "excluded 2 offer(s)" in out and "sm 80,86,89,90" in out


def test_no_allowlist_filters_nothing(monkeypatch, capsys):
    """The regression that keeps this additive: a box with no stamp, and every
    caller that predates the axis, sees exactly the pre-2026-08-18 behaviour —
    the cheapest row wins, unknown compute_cap and all, and nothing is said."""
    fake, seen = _rows({"id": 1, "min_bid": 0.30, "compute_cap": 1200},
                       {"id": 2, "min_bid": 0.40, "compute_cap": None})
    monkeypatch.setattr(api, "request_soft", fake)
    got = offers.pick_offers(gpu=["a100"], limit=2, any_inet=True)
    assert [o["id"] for o in got] == [1, 2]
    assert seen[0][2]["limit"] == 2, "no allowlist, no over-fetch"
    assert "allowlist" not in capsys.readouterr().out


def test_the_cli_search_lane_honours_the_same_allowlist(monkeypatch):
    """`--cc-allow` narrows the LAUNCH's own search too, or the launch would
    stamp a box with a list its own hardware violates."""
    def _fake(method, path, body=None, *a, **k):
        return True, {"offers": [{"id": 1, "min_bid": 0.3, "compute_cap": 1200},
                                 {"id": 2, "min_bid": 0.4, "compute_cap": 890}]}, None

    monkeypatch.setattr(api, "request", lambda m, p, b=None, *a, **k: _fake(m, p, b)[1])
    got = offers.search_offers(_search_ns(gpu=["a100"], cc_allow="89,90"))
    assert [o["id"] for o in got] == [2]
    assert [o["id"] for o in offers.search_offers(_search_ns(gpu=["a100"]))] == [1, 2]


def test_arch_change_is_reported_only_when_it_can_be_seen():
    """The alarm's predicate. compute_cap decides when both rows carry it;
    otherwise the gpu_name alias family does, because an instance body does not
    advertise compute_cap at all. Unknowable is NOT a change — an alarm that
    fires on ignorance is one nobody reads."""
    a100 = {"gpu_name": "A100 PCIE", "compute_cap": 800}
    pro6000 = {"gpu_name": "RTX PRO 6000 WS", "compute_cap": 1200}
    assert offers.arch_changed(a100, pro6000) is True
    assert offers.arch_changed(a100, dict(a100)) is False
    # same alias family, different SKU string, no compute_cap on either side
    assert offers.arch_changed({"gpu_name": "H100 SXM"},
                               {"gpu_name": "H100 NVL"}) is False
    assert offers.arch_changed({"gpu_name": "A100 PCIE"},
                               {"gpu_name": "RTX PRO 6000 WS"}) is True
    assert offers.arch_changed({}, pro6000) is False
    assert offers.arch_changed(None, None) is False
    assert offers.arch_label(pro6000) == "RTX PRO 6000 WS (sm_120)"
    assert offers.arch_label({"gpu_name": "H200"}) == "H200"
    assert offers.arch_label({}) == "?"


# --------------------------------------------------------------------------- #
# cpu_score — a RANKING PRIOR for CPU-shaped work
# --------------------------------------------------------------------------- #

def _offer(cores, ghz, dph=0.012):
    return {"cpu_cores_effective": cores, "cpu_ghz": ghz, "dph_total": dph}


def test_cpu_score_prefers_many_slow_cores_over_few_fast_ones():
    """The reason a raw core count is the wrong unit, and so is a raw clock.
    Live board figures, 2026-08-21, all at $0.012/hr — a cores-only floor of 64
    would have hidden the 5950X, a GHz-only floor of 5 would have hidden the
    EPYC, and for parallel compiles the EPYC is the box you want."""
    epyc, mid, ryzen = _offer(256, 3.7), _offer(64, 3.9), _offer(32, 5.7)
    assert offers.cpu_score(epyc) == 947.2
    assert offers.cpu_score(mid) == 249.6
    assert offers.cpu_score(ryzen) == 182.4
    assert offers.cpu_score(epyc) > offers.cpu_score(mid) > offers.cpu_score(ryzen)


def test_cpu_score_reads_the_slice_not_the_host():
    """An offer is a SLICE. Three live boxes advertising cpu_cores 256/64/256
    were all 32-core slices, so scoring the raw field over-states them ~8x."""
    o = _offer(32, 4.0)
    o["cpu_cores"] = 256                      # the host's advertised count
    assert offers.cpu_score(o) == 128.0


def test_cpu_score_is_none_when_a_term_is_missing():
    """No term, no score — and a caller sorting on it must put these LAST, not
    treat them as zero. A missing field is an unknown box, not a bad one."""
    assert offers.cpu_score({"cpu_cores_effective": 64}) is None
    assert offers.cpu_score({"cpu_ghz": 3.5}) is None
    assert offers.cpu_score({}) is None
    assert offers.cpu_score(None) is None
    assert offers.cpu_score(_offer(0, 3.5)) is None


# --------------------------------------------------------------------------- #
# cpu_perf — the MEASURED correction cpu_score is a stand-in for
# --------------------------------------------------------------------------- #

_TABLE = {
    "units": "pyops", "rate_is": "per_core_s", "generated": "t",
    "n_machines": 2, "n_models": 2,
    "fleet_median": 4.0e6, "fleet_spread": 3.0,
    "by_machine": {"140799": {"rate": 6.0e6, "n_records": 1,
                              "cpu_name": "Fast CPU"}},
    "by_model": {"Fast CPU": {"rate": 5.0e6, "n_machines": 3, "spread": 1.2},
                 "Slow CPU": {"rate": 1.0e6, "n_machines": 1, "spread": None}},
}


def _cpu_offer(machine=0, name="", cores=32, dph=0.05):
    return {"machine_id": machine, "cpu_name": name,
            "cpu_cores_effective": cores, "cpu_ghz": 3.0, "dph_total": dph}


def test_an_exactly_measured_machine_beats_its_own_model_average():
    """The machine tier is the same silicon, actually measured — no
    extrapolation — so it must win over a model median built from others."""
    p = offers.cpu_perf(_cpu_offer(140799, "Fast CPU"), _TABLE)
    assert p == {"rate": 6.0e6, "tier": "machine", "n": 1, "spread": None}


def test_the_model_tier_carries_how_far_it_is_being_stretched():
    """A model estimate is only as good as its spread, and a caller that cannot
    see the spread cannot tell a tight generalisation from a coin flip."""
    p = offers.cpu_perf(_cpu_offer(999, "Fast CPU"), _TABLE)
    assert p["tier"] == "model" and p["n"] == 3 and p["spread"] == 1.2


def test_an_offer_nothing_resembles_is_unmeasured_not_guessed():
    """No family tier and no silent fall back to cpu_score: mixing a measured
    rate with a modelled prior in one number is how a ranking stops meaning
    anything. The 9xx4 line spans 2.58x, so a family guess would be worse than
    admitting ignorance."""
    assert offers.cpu_perf(_cpu_offer(1, "AMD EPYC 9654 96-Core"), _TABLE) is None
    assert offers.cpu_throughput(_cpu_offer(1, "nope"), _TABLE) is None
    assert offers.cpu_value(_cpu_offer(1, "nope"), _TABLE) is None


def test_throughput_multiplies_the_rate_by_the_slice_not_the_host():
    """Both terms count THREADS — vast advertises a 48-core EPYC 7K62 as 96 —
    so the product is sound, but an offer is a slice and the host figure would
    over-state it."""
    o = _cpu_offer(140799, "Fast CPU", cores=8)
    o["cpu_cores"] = 128                          # the whole host
    assert offers.cpu_throughput(o, _TABLE) == pytest.approx(4.8e7)
    assert offers.cpu_value(o, _TABLE) == pytest.approx(4.8e7 / 0.05)


def test_the_floor_refuses_measured_slow_silicon():
    slow = _cpu_offer(1, "Slow CPU")               # 1.0e6 vs a 1.4e6 floor
    keep, why = offers.cpu_floor_verdict(slow, _TABLE)
    assert keep is False and "floor" in why and "model" in why


def test_the_floor_never_drops_an_offer_it_has_not_measured():
    """70% of the cheap market is unmeasured. A gate that dropped those would
    delete the board and call it selectivity — a box we cannot measure is an
    UNKNOWN box, not a bad one, so it ranks last and is kept."""
    keep, why = offers.cpu_floor_verdict(_cpu_offer(7, "Who Knows"), _TABLE)
    assert keep is True and why == "unmeasured"


def test_with_no_calibration_the_floor_cannot_fire_at_all():
    """A gate that fires on ignorance empties the board. No table, no floor."""
    assert offers.cpu_perf_floor({}, 0.35) is None
    keep, _ = offers.cpu_floor_verdict(_cpu_offer(1, "Slow CPU"), {})
    assert keep is True


def test_a_zero_ratio_disarms_the_floor():
    assert offers.cpu_perf_floor(_TABLE, 0.0) is None
    keep, _ = offers.cpu_floor_verdict(_cpu_offer(1, "Slow CPU"), _TABLE, 0.0)
    assert keep is True


def test_filter_cpu_perf_is_three_way_like_the_host_ram_floor():
    """Measured-fast keeps its place, measured-slow is dropped, UNMEASURED is
    kept but ranked after every measured row — never dropped, never zero."""
    fast = _cpu_offer(1, "Fast CPU")
    slow = _cpu_offer(2, "Slow CPU")
    unknown = _cpu_offer(3, "Who Knows")
    kept, dropped = offers.filter_cpu_perf([slow, unknown, fast], 0.35, _TABLE)
    assert dropped == 1
    assert [r["cpu_name"] for r in kept] == ["Fast CPU", "Who Knows"]


def test_filter_cpu_perf_is_a_no_op_without_a_ratio_or_a_table():
    rows = [_cpu_offer(1, "Slow CPU"), _cpu_offer(2, "Fast CPU")]
    assert offers.filter_cpu_perf(rows, 0, _TABLE) == (rows, 0)
    assert offers.filter_cpu_perf(rows, None, _TABLE) == (rows, 0)
    assert offers.filter_cpu_perf(rows, 0.35, {}) == (rows, 0)


def test_the_floor_is_a_ratio_of_the_median_not_a_pinned_constant():
    """So it re-derives itself when the kernel or the fleet moves, instead of
    pinning a number to one probe version."""
    assert offers.cpu_perf_floor(_TABLE, 0.35) == pytest.approx(1.4e6)
    doubled = dict(_TABLE, fleet_median=8.0e6)
    assert offers.cpu_perf_floor(doubled, 0.35) == pytest.approx(2.8e6)


def test_cpu_floors_are_absent_from_the_query_unless_asked_for():
    """They bound a fetch; they must not narrow every other lane's search."""
    q = offers.build_search_query(_search_ns())
    assert "cpu_cores_effective" not in q and "cpu_ghz" not in q


def test_cpu_floors_filter_on_the_effective_slice():
    q = offers.build_search_query(_search_ns(cpu_cores=32, cpu_ghz=3.0))
    assert q["cpu_cores_effective"] == {"gte": 32}
    assert q["cpu_ghz"] == {"gte": 3.0}
    assert "cpu_cores" not in q               # never the host-wide field


# --------------------------------------------------------------------------- #
# cpu_name normalisation (2026-08-28): some hosts advertise the marketing
# spelling instead of the /proc/cpuinfo one the probe banks.
# --------------------------------------------------------------------------- #
def test_the_marketing_spelling_joins_to_the_proc_cpuinfo_one():
    """`Xeon(R) E5-2699 v4 @ 2.20GHz` and `Xeon® E5-2699 v4 ` are one part.
    26 of 229 offers on a live board used the second form, and the two slowest
    machines we have measured are among them -- so this gap was letting exactly
    the silicon the floor exists to refuse through as `unmeasured`."""
    t = {"fleet_median": 4.0, "by_machine": {},
         "by_model": {"Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz":
                      {"rate": 3.8, "n_machines": 1, "spread": None}}}
    p = offers.cpu_perf(_cpu_offer(0, "Xeon® E5-2699 v4 "), t)
    assert p is not None and p["tier"] == "model" and p["rate"] == 3.8


def test_cpu_name_key_never_merges_two_distinct_parts():
    """The model number IS the identity. An early cut of this normaliser
    stripped it, collapsing EPYC 7352/7402/7702 to one key -- which would have
    reported a Rome part as a measured Genoa one, i.e. the silent family tier
    this module refuses by design."""
    parts = ["AMD EPYC 7352 24-Core Processor", "AMD EPYC 7402 24-Core Processor",
             "AMD EPYC 7702 64-Core Processor", "AMD EPYC 9534 64-Core Processor",
             "AMD EPYC 9754 128-Core Processor", "AMD EPYC 9B14",
             "Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz",
             "Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz",
             "Intel(R) Xeon(R) 6952P", "Intel(R) Xeon(R) 6767P",
             "AMD Ryzen 9 9950X 16-Core Processor",
             "AMD Ryzen 9 7945HX with Radeon Graphics"]
    keys = [offers.cpu_name_key(p) for p in parts]
    assert len(set(keys)) == len(parts), \
        [p for p, k in zip(parts, keys) if keys.count(k) > 1]


def test_the_shipped_table_has_no_normalisation_collisions():
    """Guards the real table, not a fixture: if two measured models ever reduce
    to one key the index drops it, so a regression here reads as coverage
    quietly falling rather than as a wrong answer -- easy to miss."""
    for arm in ("pyops", "compile_tu"):
        t = offers.cpu_calibration(reload=True, arm=arm)
        assert t, f"shipped table has no {arm} arm"
        names = list(t["by_model"])
        assert len(names) == len({offers.cpu_name_key(n) for n in names})


def test_a_colliding_key_reports_unmeasured_rather_than_guessing():
    """Two measured parts reducing to one key means the normalisation is not
    identity-preserving. Picking one of them would be a coin flip presented as
    a measurement, so the key is dropped and the offer reads unmeasured."""
    t = {"fleet_median": 4.0, "by_machine": {},
         "by_model": {"Weird Part": {"rate": 9.0, "n_machines": 1, "spread": None},
                      "WEIRD  PART": {"rate": 1.0, "n_machines": 1, "spread": None}}}
    assert offers.cpu_name_key("Weird Part") == offers.cpu_name_key("WEIRD  PART")
    assert offers.cpu_perf(_cpu_offer(0, "weird part"), t) is None


# --------------------------------------------------------------------------- #
# the floor moved to the compile arm (2026-08-28)
# --------------------------------------------------------------------------- #
def test_the_floor_reads_the_compile_arm_and_the_ranking_reads_pyops():
    """They are different arms on purpose: pyops is all-core and may be scaled
    by a slice width; compile_tu is a serial single-compile latency and may
    not. Reading one for the other's question is the defect this pins."""
    assert offers.THROUGHPUT_ARM == "pyops"
    assert offers.FLOOR_ARM == "compile_tu"
    blob = {"schema": 2, "arms": {
        "pyops": {"units": "pyops", "fleet_median": 4.0e6, "by_machine":
                  {"7": {"rate": 8.0e6}}, "by_model": {}},
        "compile_tu": {"units": "compile_tu", "fleet_median": 6.0, "by_machine":
                       {"7": {"rate": 1.0}}, "by_model": {}}}}
    import hostfacts as hf
    thr = hf.calibration_arm(blob, offers.THROUGHPUT_ARM)
    flr = hf.calibration_arm(blob, offers.FLOOR_ARM)
    # Fast all-core, slow serial: the ranking likes it, the floor refuses it.
    assert offers.cpu_perf(_cpu_offer(7), thr)["rate"] == 8.0e6
    keep, why = offers.cpu_floor_verdict(_cpu_offer(7), flr, 0.60)
    assert not keep and "compile_tu" in why


def test_the_floor_ratio_sits_in_the_measured_gap_not_on_an_edge():
    """0.60 was chosen because four machines land at 0.44-0.57x and the next is
    0.711x. Anything in 0.57-0.71 refuses the same four, so the constant is not
    knife-edge -- if a future re-cut narrows that gap, this should be re-read
    rather than nudged."""
    assert 0.57 < offers.CPU_PERF_FLOOR_RATIO < 0.71
