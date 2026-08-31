"""Portable tests for the host-RAM floor — the CPU-side twin of the VRAM one.

Two properties, and the second is the one that costs money.

The FLOOR (`host_ram_floor_mib`) is the GB -> `cpu_ram`-MiB translation, and it
must stay a single helper shared by `build_search_query` and `pick_offers` for
the reason `test_gpu_ram_floor.py` records at length: those two query builders
are hand-maintained copies, and their VRAM translations silently diverged for a
month until one helper fixed it. This file asserts the same anti-drift property
before the copies have a chance to diverge.

The SLICE (`effective_host_ram_gb`) is the half a floor alone gets wrong.
`cpu_ram` is the WHOLE MACHINE's memory and vast publishes no
`cpu_ram_effective` to go with its `cpu_cores_effective`, so a filter read
straight off it over-admits by 1/gpu_frac — it buys a "768 GB" host for a job
needing 128 and hands it a 96 GB slice.

Offline lane: no vast API, no network, $0. Every assertion is on a constructed
query dict or a pure function.
"""
import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import herdd as v  # noqa: E402

from vastlib.core import api  # noqa: E402
from vastlib.market import offers as mo  # noqa: E402


# `cpu_ram` as vast reports it (MiB) for the host classes the CPU lane actually
# rents, measured off the live board 2026-08-24. A "128 GB" host advertises
# 126 GiB — firmware and the OS take the rest — which is why a declaration is
# read against the ADVERTISED number and not the marketing one.
ADVERTISED_MIB = {
    "126 GB host": (126, 129024),
    "252 GB host": (252, 258048),
    "512 GB host": (512, 524288),
}


def _search_ns(**kw):
    base = dict(limit=20, type="bid", num_gpus=1, unverified=False, gpu=None,
                gpu_ram=0, max_dph=None, host_disk=0, reliability=0, cuda=0,
                inet_down=None, machine=None, host=None, geo=None,
                exclude_machines=None, any_inet=False, cpu_cores=0, cpu_ghz=0,
                host_ram=0)
    base.update(kw)
    return argparse.Namespace(**base)


def _pick_offers_query(monkeypatch, **kw):
    """The FIRST bundles query `pick_offers` puts on the wire, with no network.
    `pick_offers` is soft by contract, so returning a miss is enough."""
    seen = []

    def _fake(method, path, body=None, *a, **k):
        seen.append((method, path, body))
        return True, {"offers": []}, None

    monkeypatch.setattr(api, "request_soft", _fake)
    mo.pick_offers(**kw)
    assert seen, "pick_offers issued no query"
    return seen[0][2]


# --------------------------------------------------------------------------- #
# the floor itself
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("host,decl,advertised",
                         [(k, d, m) for k, (d, m) in ADVERTISED_MIB.items()])
def test_every_host_class_is_admitted_by_its_own_declaration(host, decl,
                                                             advertised):
    assert mo.host_ram_floor_mib(decl) <= advertised, host


def test_the_floor_is_exact_gib_with_no_tolerance_band():
    """The deliberate asymmetry with `gpu_ram_floor_mib`, pinned so that making
    the two symmetric is a DECISION rather than a cleanup.

    VRAM carries a 1% band because a card's advertised framebuffer sits a fixed
    ~0.5% under its marketing capacity, so an exact floor excludes a whole
    DISCRETE class by 12 MiB. Host RAM has no classes — a slice is whatever the
    host had left — so there is no cliff for a band to step over and slack buys
    only an OOM."""
    assert mo.host_ram_floor_mib(96) == 96 * 1024
    assert mo.host_ram_floor_mib(126) == 126 * 1024
    assert mo.host_ram_floor_mib(125.5) == int(125.5 * 1024)


def test_the_floor_is_a_floor():
    assert mo.host_ram_floor_mib(126) > 65536      # a 64 GB host must not pass
    assert mo.host_ram_floor_mib(252) > 129024     # nor a 126 GB one


def test_floor_is_soft_on_junk_input():
    """Never raises into a query builder: unusable input means no floor, not a
    traceback and not a floor of zero smuggled in as a filter."""
    assert mo.host_ram_floor_mib(None) == 0
    assert mo.host_ram_floor_mib("") == 0
    assert mo.host_ram_floor_mib("lots") == 0
    assert mo.host_ram_floor_mib(0) == 0
    assert mo.host_ram_floor_mib(-96) == 0


# --------------------------------------------------------------------------- #
# anti-drift: the two call sites are ONE translation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gb", [16, 32, 64, 96, 126, 128, 252, 512])
def test_search_and_picker_paths_agree_on_the_threshold(monkeypatch, gb):
    """`build_search_query` (the CLI/launcher path) and `pick_offers` (the
    workflow / eviction-replacement path) must produce the SAME cpu_ram floor
    for the same declaration. The VRAM pair did not, for a month, and the
    divergence — not the arithmetic — was the defect."""
    cli = v.build_search_query(_search_ns(host_ram=gb))["cpu_ram"]
    picker = _pick_offers_query(monkeypatch, gpu=("RTX A6000",),
                                host_ram_gb=gb)["cpu_ram"]
    assert cli == picker == {"gte": mo.host_ram_floor_mib(gb)}


def test_no_host_ram_means_no_filter(monkeypatch):
    assert "cpu_ram" not in v.build_search_query(_search_ns(host_ram=0))
    assert "cpu_ram" not in _pick_offers_query(monkeypatch, gpu=("RTX A6000",))


# --------------------------------------------------------------------------- #
# the SLICE — what the server-side floor cannot see
# --------------------------------------------------------------------------- #
def test_the_slice_is_the_host_figure_times_gpu_frac():
    """The over-admission a raw `cpu_ram` filter buys: one card of an 8-GPU
    768 GB host advertises 768 and hands you 96."""
    assert mo.effective_host_ram_gb(
        {"cpu_ram": 786432, "gpu_frac": 0.125}) == 96.0
    assert mo.effective_host_ram_gb(
        {"cpu_ram": 129024, "gpu_frac": 1.0}) == 126.0
    assert mo.effective_host_ram_gb(
        {"cpu_ram": 258048, "gpu_frac": 0.5}) == 126.0


@pytest.mark.parametrize("offer", [
    {},                                          # nothing at all
    {"cpu_ram": 129024},                         # no gpu_frac
    {"gpu_frac": 0.5},                           # no cpu_ram
    {"cpu_ram": 0, "gpu_frac": 0.5},             # the volume-listing shape
    {"cpu_ram": 129024, "gpu_frac": 0},          # unusable fraction
    {"cpu_ram": "lots", "gpu_frac": 0.5},        # unparseable
])
def test_an_unmeasurable_offer_reads_as_UNKNOWN_and_never_as_ZERO(offer):
    """`None`, never `0.0`. A box we cannot measure is an unknown box, not an
    empty one — the same house rule the disk precheck states as "a measurement
    we could not take is not evidence".

    `cpu_ram: 0` is not hypothetical: every `num_gpus=0` row on the board is a
    `resource_type: disk` VOLUME listing and carries exactly that."""
    assert mo.effective_host_ram_gb(offer) is None


# --------------------------------------------------------------------------- #
# the three-way client-side narrowing
# --------------------------------------------------------------------------- #
_BIG = {"id": 1, "cpu_ram": 786432, "gpu_frac": 0.5}      # 384 G slice
_FITS = {"id": 2, "cpu_ram": 258048, "gpu_frac": 0.5}     # 126 G slice
_SMALL = {"id": 3, "cpu_ram": 786432, "gpu_frac": 0.125}  # 96 G slice
_UNKNOWN = {"id": 4, "cpu_ram": 0}                        # unmeasurable


def test_a_measurably_small_slice_is_dropped():
    kept, dropped = mo.filter_host_ram([_BIG, _SMALL], 126)
    assert [o["id"] for o in kept] == [1]
    assert dropped == 1


def test_an_unmeasurable_offer_is_kept_but_ranked_last():
    """Kept, because refusing what we cannot measure empties the market on
    ignorance. Last, because an offer we can PROVE fits must always win."""
    kept, dropped = mo.filter_host_ram([_UNKNOWN, _FITS], 126)
    assert [o["id"] for o in kept] == [2, 4]
    assert dropped == 0


def test_market_order_is_preserved_within_each_class():
    """Cheapest-first is the contract every caller of pick_offers relies on, so
    the narrowing may only PARTITION, never re-sort."""
    a = {"id": 10, "cpu_ram": 258048, "gpu_frac": 1.0}
    b = {"id": 11, "cpu_ram": 258048, "gpu_frac": 1.0}
    kept, _ = mo.filter_host_ram([a, b, _UNKNOWN, _SMALL], 126)
    assert [o["id"] for o in kept] == [10, 11, 4]


def test_a_market_of_nothing_but_unknowns_still_yields_a_box():
    kept, dropped = mo.filter_host_ram([_UNKNOWN, dict(_UNKNOWN, id=5)], 512)
    assert len(kept) == 2 and dropped == 0


def test_no_floor_narrows_nothing():
    rows = [_SMALL, _UNKNOWN]
    for junk in (0, None, "", "lots", -8):
        kept, dropped = mo.filter_host_ram(rows, junk)
        assert len(kept) == 2 and dropped == 0


def test_the_narrowing_does_not_mutate_the_callers_rows():
    rows = [dict(_SMALL), dict(_FITS)]
    mo.filter_host_ram(rows, 126)
    assert rows[0] == _SMALL and rows[1] == _FITS


# --------------------------------------------------------------------------- #
# end to end through pick_offers
# --------------------------------------------------------------------------- #
def test_pick_offers_applies_the_slice_check_not_just_the_query(monkeypatch):
    """The whole point of the second filter. The server can only bound on the
    host figure, so a big host with a thin slice comes back from a correct
    query and must be dropped HERE."""
    def _fake(method, path, body=None, *a, **k):
        return True, {"offers": [dict(_SMALL), dict(_FITS)]}, None

    monkeypatch.setattr(api, "request_soft", _fake)
    got = mo.pick_offers(gpu=("RTX A6000",), host_ram_gb=126, limit=5,
                         hostrep=False)
    assert [o["id"] for o in got] == [2]


def test_pick_offers_over_fetches_so_the_slice_check_can_bite(monkeypatch):
    """Fetching exactly `limit` and then dropping rows client-side hands back
    fewer offers than asked for — cc_allow's bug, and the RAM floor inherits it
    unless it inherits the over-fetch too."""
    q = _pick_offers_query(monkeypatch, gpu=("RTX A6000",), host_ram_gb=126,
                           limit=1)
    assert q["limit"] >= mo.CC_ALLOW_SCAN_LIMIT


# --------------------------------------------------------------------------- #
# the live board, trimmed: why there is no GPU-less lane
# --------------------------------------------------------------------------- #
#: Five rows from a `num_gpus=0` market query, 2026-08-24, trimmed to the fields
#: this file reasons about. Kept because the two facts below are load-bearing
#: and easy to "fix" wrongly from first principles.
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "testfixtures", "cpu_offers_20260824.json")


def _fixture():
    import json
    with open(FIXTURE) as fh:
        return json.load(fh)


def test_every_gpuless_ask_on_the_board_is_a_volume_listing():
    """vast sells NO GPU-less compute. `num_gpus=0` asks exist — 200 of them on
    this query — and every one carries `resource_type: disk`: they are VOLUME
    listings, and a rent against one is refused (`no_such_ask`).

    So the CPU lane is 1-GPU offers with big CPU slices and a throwaway card,
    which is what `--host-ram` and `--cpu-cores` are for. Anyone tempted to
    build a `num_gpus=0` launch path should read this test first."""
    rows = _fixture()
    assert rows, "fixture is empty"
    assert {r["resource_type"] for r in rows} == {"disk"}
    assert {r["num_gpus"] for r in rows} == {0}


def test_the_volume_listings_are_exactly_the_unmeasurable_shape():
    """These rows advertise hundreds of cores and `cpu_ram: 0` — the wide-and-
    apparently-empty box that a floor treating unknown as zero would either
    silently admit or silently drop. `effective_host_ram_gb` calls them
    UNKNOWN, which is the only honest reading."""
    for r in _fixture():
        assert r["cpu_cores_effective"] >= 256      # not a small machine
        assert not r["cpu_ram"]                     # ...and no memory figure
        assert mo.effective_host_ram_gb(r) is None


def test_a_ram_floor_ranks_the_volume_listings_last_rather_than_dropping_them():
    real = {"id": 99, "cpu_ram": 258048, "gpu_frac": 0.5}   # 126 G slice
    kept, dropped = mo.filter_host_ram(_fixture() + [real], 96)
    assert kept[0]["id"] == 99, "a measurable box must outrank an unknown one"
    assert dropped == 0, "unknown is not a refusal"
    assert len(kept) == len(_fixture()) + 1
