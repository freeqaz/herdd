"""Portable tests for the GB -> vast-`gpu_ram`-MiB search floor.

The defect, found 2026-08-16: `build_search_query` translated `--gpu-ram 48`
to `gte 49152` while a "48 GB" card advertises 49140 MiB. Off by 12 MiB, and
every bundle declaring `needs.gpu_ram_gb: 48` (w5a, w5b, v12, w4t) was
therefore unable to rent a 48 GB card AT ALL — the launcher saw an empty
market, not an error. Reproduced live: `--gpu a6000 --type ondemand --max-dph
3.0` returned 4 offers, and adding `--gpu-ram 48` returned 0.

The ROOT CAUSE is not the arithmetic: `pick_offers` already carried a
tolerance (0.96, added 2026-07-15 for the same class of bug on RTX 5090s) and
the two translations had silently diverged. Hence one helper,
`gpu_ram_floor_mib`, and the anti-drift test below.

A second, DISTINCT defect found in the same pass — vast's exact-match
`gpu_name` filter and the missing "RTX 6000 Ada" alias — lives in
`test_gpu_name_aliases.py`. The two present identically (0 offers).

Offline lane: no vast API, no network, $0. Every assertion is on the
constructed query dict; the fixtures are the advertised `gpu_ram` values
measured off the live board on 2026-08-16 (1465 verified on-demand offers).
"""
import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import herdd as v  # noqa: E402

# The HTTP seam is patched at its owner (`core.api`), not through the `herdd`
# re-export: since plan §8 step 6d `market.offers.pick_offers` resolves
# `api.request_soft` from its OWN globals, so a patch of `v.request_soft` would
# leave the real request live and this helper would hit the board.
from vastlib.core import api  # noqa: E402


# Advertised per-card `gpu_ram` (MiB) as vast reports it, measured off the live
# board 2026-08-16. These are the numbers the floor has to admit or reject; do
# not "round them up" to the marketing capacity, that IS the bug.
ADVERTISED_MIB = {
    "RTX 5060": (8, 8151),
    "RTX 5070": (12, 12227),
    "RTX 5080": (16, 16303),
    "RTX PRO 4000": (24, 24467),
    "RTX 3090": (24, 24576),
    "RTX 4090": (24, 24564),
    "RTX 5090": (32, 32607),
    "RTX 4080S 32G": (32, 32760),
    "RTX A6000": (48, 49140),
    "RTX 6000Ada": (48, 49140),
    "RTX 5880Ada": (48, 49140),
    "RTX PRO 5000": (48, 48935),
    "H100 SXM": (80, 81559),
    "RTX PRO 6000 WS": (96, 97887),
    "H200": (141, 143771),
    "B200": (180, 183359),
}

# ECC-on GDDR6 costs 1/16 of capacity, so these "48 GB" parts really do expose
# only 44.99 GiB. They MUST NOT satisfy a declared 48 — that would be an OOM,
# not a fix.
ECC_CAPACITY_MIB = 46068          # L40, L40S, and ECC-on RTX A6000


def _search_ns(**kw):
    base = dict(limit=20, type="bid", num_gpus=1, unverified=False, gpu=None,
                gpu_ram=0, max_dph=None, host_disk=0, reliability=0, cuda=0,
                inet_down=None, machine=None, host=None, geo=None,
                exclude_machines=None, any_inet=False)
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
    v.pick_offers(**kw)
    assert seen, "pick_offers issued no query"
    return seen[0][2]


# --------------------------------------------------------------------------- #
# the floor itself
# --------------------------------------------------------------------------- #
def test_48gb_declaration_admits_the_48gb_class():
    """The regression. 49140 MiB is what an A6000 / 6000Ada / 5880Ada
    advertises; 49152 is the naive floor that excluded all of them."""
    floor = v.gpu_ram_floor_mib(48)
    assert floor <= 49140, f"floor {floor} excludes the 48 GB class (49140 MiB)"
    assert floor <= 48935, f"floor {floor} excludes RTX PRO 5000 (48935 MiB)"


def test_48gb_declaration_still_rejects_a_smaller_card():
    """The floor is a FLOOR. A 32 GB part must not satisfy a declared 48."""
    floor = v.gpu_ram_floor_mib(48)
    assert floor > 32760, "a 32 GB card satisfies a declared 48 GB need"
    assert floor > 24576, "a 24 GB card satisfies a declared 48 GB need"


def test_48gb_declaration_rejects_ecc_capacity_cards():
    """L40/L40S/ECC-A6000 expose 44.99 GiB and must NOT pass a declared 48.
    This is the tolerance's upper bound: anything looser than ~6% admits them
    and buys an OOM. They stay reachable from an honest declaration of 45."""
    assert v.gpu_ram_floor_mib(48) > ECC_CAPACITY_MIB
    assert v.gpu_ram_floor_mib(44) <= ECC_CAPACITY_MIB


@pytest.mark.parametrize("card,decl,advertised",
                         [(k, d, m) for k, (d, m) in ADVERTISED_MIB.items()])
def test_every_measured_card_class_is_admitted_by_its_own_declaration(
        card, decl, advertised):
    """Every card on the live board satisfies a declaration equal to its own
    marketing capacity. The carve-out is ~0.5% at every size, so this is the
    property the tolerance exists to hold."""
    assert v.gpu_ram_floor_mib(decl) <= advertised, card


@pytest.mark.parametrize("card,decl,advertised",
                         [(k, d, m) for k, (d, m) in ADVERTISED_MIB.items()])
def test_the_floor_never_admits_a_full_class_downgrade(card, decl, advertised):
    """Declared N GB, the floor stays above 0.98*N. The measured carve-out is
    ~0.5% at every size and never exceeded 0.53% on the whole board, so 2% is
    already 4x it — anything looser is buying OOM risk for no supply."""
    assert v.gpu_ram_floor_mib(decl) > decl * 1024 * 0.98, card


def test_the_tolerance_band_is_a_deliberate_choice():
    """Pin the band so widening it is a DECISION, not a cleanup.

    Lower bound (0.98): slack is OOM risk. `gpu_ram_gb` is a floor for a real
    VRAM requirement, and the carve-out this absorbs is ~0.5%.
    Upper bound (< 1.0): no tolerance at all is the 12-MiB bug — a "48 GB"
    card advertises 49140 MiB and 48*1024 is 49152."""
    assert 0.98 <= v.VRAM_SEARCH_TOLERANCE < 1.0


def test_floor_is_soft_on_junk_input():
    """Never raises into a query builder: unusable input means no floor, not a
    traceback and not a floor of zero smuggled in as a filter."""
    assert v.gpu_ram_floor_mib(None) == 0
    assert v.gpu_ram_floor_mib("") == 0
    assert v.gpu_ram_floor_mib(0) == 0
    assert v.gpu_ram_floor_mib(-8) == 0


# --------------------------------------------------------------------------- #
# anti-drift: the two call sites are ONE translation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gb", [16, 24, 32, 40, 48, 80, 96, 141, 180])
def test_search_and_picker_paths_agree_on_the_threshold(monkeypatch, gb):
    """`build_search_query` (the CLI/launcher path) and `pick_offers` (the
    workflow / eviction-replacement path) must produce the SAME gpu_ram floor
    for the same declaration. They did not from 2026-07-15 to 2026-08-16, and
    the divergence — not the arithmetic — is what made the 48 GB class
    unrentable from the launcher while the replacement probe could reach it."""
    cli = v.build_search_query(_search_ns(gpu_ram=gb))["gpu_ram"]
    picker = _pick_offers_query(monkeypatch, gpu=("RTX A6000",),
                                gpu_ram_gb=gb)["gpu_ram"]
    assert cli == picker == {"gte": v.gpu_ram_floor_mib(gb)}


def test_cli_search_query_admits_the_48gb_class_end_to_end():
    """The reproduction, as a query: `--gpu a6000 --gpu-ram 48` must not
    exclude a 49140 MiB offer."""
    q = v.build_search_query(_search_ns(gpu=["a6000"], gpu_ram=48))
    assert q["gpu_name"] == {"in": ["RTX A6000"]}
    assert q["gpu_ram"]["gte"] <= 49140
    assert q["gpu_ram"]["gte"] > 32760


def test_no_gpu_ram_means_no_filter():
    assert "gpu_ram" not in v.build_search_query(_search_ns(gpu_ram=0))


# --------------------------------------------------------------------------- #
# agreement with the BOX-SIDE gate
# --------------------------------------------------------------------------- #
def _jobd_card_gb(mib):
    """jobd's own view of a card's size, from `tools/vast/onstart/jobd.sh`:
    `GPU_MEM+=($(( (_mib + 512) / 1024 )))` — whole GB, rounded to NEAREST. It
    fails a ticket when the largest card is below `needs.gpu_ram_gb`."""
    return (mib + 512) // 1024


@pytest.mark.parametrize("decl", [8, 12, 16, 20, 24, 32, 40, 45, 48, 64, 80,
                                  96])
def test_search_floor_never_admits_a_card_the_box_would_refuse(decl):
    """The filter must not buy a box whose ticket jobd then refuses — that is
    a paid box with no run, the exact failure `launch_jobs_box.sh`'s
    `--force-gpu-ram` refusal exists to prevent, just arriving by a different
    door. Holds for every declaration in the range this fleet actually uses."""
    floor = v.gpu_ram_floor_mib(decl)
    for card, advertised in sorted((c, m) for c, (_, m) in
                                   ADVERTISED_MIB.items()):
        if advertised >= floor:
            assert _jobd_card_gb(advertised) >= decl, (
                f"declared {decl} GB would rent {card} ({advertised} MiB), "
                f"which jobd sizes at {_jobd_card_gb(advertised)} GB and "
                f"would refuse")


@pytest.mark.parametrize("decl,card_mib", [(141, 143771), (180, 183359)])
def test_hbm_marketing_sizes_diverge_from_the_box_gate(decl, card_mib):
    """KNOWN GAP, documented rather than silently fixed. An H200's "141 GB" is
    140.40 GiB and a B200's "180 GB" is 179.06 GiB — more than half a GiB below
    the marketing number, so jobd's round-to-nearest sizes them 140 and 179 and
    would REFUSE a ticket declaring 141 / 180. The search floor admits them, so
    those two declarations rent a box that will not run.

    Not fixed here because the fix is a supply decision, not a rounding one:
    clamping the floor to jobd's threshold makes `gpu_ram_gb: 141` unrentable
    outright. The honest declaration for those cards is 140 / 179, which both
    gates agree on. No bundle declares either value today.

    This gap is NOT introduced by the tolerance — an exact `gb * 1024` floor
    excluded those cards entirely, which is a different wrong answer."""
    assert v.gpu_ram_floor_mib(decl) <= card_mib      # search says yes
    assert _jobd_card_gb(card_mib) < decl             # the box says no
    honest = _jobd_card_gb(card_mib)
    assert v.gpu_ram_floor_mib(honest) <= card_mib    # and both agree here
    assert _jobd_card_gb(card_mib) >= honest
