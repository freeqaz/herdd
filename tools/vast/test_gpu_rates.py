"""Pins the measured-rate table's contract for the fleet replacement lane.

Portable lane: no marks, no GPU, no network, no fixtures. The consumer ranks
candidate boxes by tokens-per-dollar and falls back to price-only whenever a
rate is None, so the three things that must never break are (a) a raw vast
`gpu_name` resolving to the class we measured, (b) an unmeasured class
answering None instead of a guess, and (c) the arithmetic.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_rates  # noqa: E402


# --------------------------------------------------------------------------- #
# alias normalization
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,want", [
    # vast API `gpu_name` — the vocabulary the consumer actually receives
    ("H100 NVL", "H100 NVL"),
    ("H100 PCIE", "H100 PCIE"),
    ("H100 SXM", "H100 SXM"),
    ("H200 NVL", "H200 NVL"),
    ("RTX 5090", "RTX 5090"),
    ("A100 SXM4", "A100 SXM4"),
    ("B200", "B200"),
    # torch `get_device_name` — the vocabulary train_summary.json records
    ("NVIDIA H100 NVL", "H100 NVL"),
    ("NVIDIA H100 PCIe", "H100 PCIE"),
    ("NVIDIA H100 80GB HBM3", "H100 SXM"),
    ("NVIDIA GeForce RTX 5090", "RTX 5090"),
    ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
     "RTX PRO 6000 WS 600W"),
    ("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
     "RTX PRO 6000 WS MAXQ"),
    # shorthands, as herdd's own --gpu flag takes them
    ("5090", "RTX 5090"),
    ("rtx5090", "RTX 5090"),
    ("pro6000", "RTX PRO 6000 WS"),
    # whitespace / case are not signal
    ("  h100 nvl  ", "H100 NVL"),
    ("rtx pro 6000 ws", "RTX PRO 6000 WS"),
])
def test_normalize_maps_both_vocabularies_onto_one_class(raw, want):
    assert gpu_rates.normalize_gpu_name(raw) == want


def test_bare_h100_does_not_borrow_a_variants_rate():
    """PCIe and NVL measured different numbers and SXM was never measured, so
    an unqualified "h100" must fall through to None rather than pick one."""
    assert gpu_rates.normalize_gpu_name("h100") == "H100"
    assert gpu_rates.rate_for("h100", 1, "9b-w20480-dec") is None
    assert gpu_rates.rate_for("H100 NVL", 1, "9b-w20480-dec") == 4050.0


def test_the_two_workstation_skus_stay_apart():
    """vast sells them under one name; they measured 1.39x apart, so the table
    keeps three keys and the ambiguous one answers with the slower part."""
    fast = gpu_rates.rate_for("NVIDIA RTX PRO 6000 Blackwell Workstation "
                              "Edition", 1, "7b-w12288-fit")
    slow = gpu_rates.rate_for("NVIDIA RTX PRO 6000 Blackwell Max-Q "
                              "Workstation Edition", 1, "7b-w12288-fit")
    assert fast is not None and slow is not None
    assert fast > slow
    # the vast offer string has no cell of its own at this shape -> SKU floor
    assert gpu_rates.rate_for("RTX PRO 6000 WS", 1, "7b-w12288-fit") == slow
    # ...but where it DOES have its own cell, that wins over the floor
    assert gpu_rates.rate_for("RTX PRO 6000 WS", 1, "9b-w40960-dec") == 2519.0


@pytest.mark.parametrize("bad", [None, "", "   ", 5090, object()])
def test_normalize_survives_junk(bad):
    assert gpu_rates.normalize_gpu_name(bad) == ""


# --------------------------------------------------------------------------- #
# None for the unknown — the designed price-only fallback
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gpu,n,shape", [
    # H200 NVL and B200 USED to sit here as the canonical never-measured
    # cells; the 2026-08-16 gpu-rate-9b-w20480 sweep measured both, so they
    # moved out and these took their place. The examples are meant to be real
    # gaps, not decoration -- a card whose cell exists teaches this test
    # nothing.
    ("H200 NVL", 4, "9b-w20480-dec"),      # measured card, unmeasured count
    ("B200", 1, "7b-w12288-fit"),           # measured card, unmeasured shape
    ("L40S", 1, None),                      # class exists, never rentable on
                                            # spot in the 08-16 window
    ("H100 SXM", 1, "9b-w20480-dec"),       # attempted 08-16, evicted twice
    ("Totally Made Up 9000", 1, None),      # unknown class
    ("H100 NVL", 3, "9b-w20480-dec"),       # measured card, unmeasured count
    ("H100 NVL", 1, "7b-w4096-fit"),        # measured card, unmeasured shape
    ("RTX 5090", 2, "7b-w12288-fit"),       # the ladder has 1/4/8, not 2
    ("H100 NVL", 0, None),                  # nonsense count
    ("H100 NVL", -1, None),
    (None, 1, None),
])
def test_unknown_cells_are_none_never_a_guess(gpu, n, shape):
    assert gpu_rates.rate_for(gpu, n, shape) is None
    assert gpu_rates.entry_for(gpu, n, shape) is None
    assert gpu_rates.tokens_per_dollar(gpu, n, 1.5, shape) is None


def test_no_entry_is_ever_interpolated_from_the_scaling_table():
    """MULTI_GPU_SCALING is published for a caller who asks for it BY NAME; it
    must never leak into rate_for and manufacture a W=2 or W=4 H100 cell."""
    assert 4 in gpu_rates.MULTI_GPU_SCALING["tok_s_multiple_vs_1_card"]
    assert gpu_rates.rate_for("H100 NVL", 4, "9b-w20480-dec") is None
    assert gpu_rates.rate_for("RTX 5090", 2, "7b-w12288-fit") is None


# --------------------------------------------------------------------------- #
# shape resolution
# --------------------------------------------------------------------------- #
def test_shape_none_prefers_the_default_shape():
    assert gpu_rates.DEFAULT_SHAPE == "9b-w20480-dec"
    assert (gpu_rates.rate_for("H100 NVL", 1)
            == gpu_rates.rate_for("H100 NVL", 1, gpu_rates.DEFAULT_SHAPE))


def test_shape_none_falls_back_to_the_slowest_measured_shape():
    """Conservative on purpose: a number used to justify spend must not be the
    luckiest shape we happened to bench."""
    gpu = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
    shapes = gpu_rates.shapes_for(gpu, 1)
    assert len(shapes) > 1  # more than one shape measured, so the rule bites
    rates = [gpu_rates.rate_for(gpu, 1, s) for s in shapes]
    assert gpu_rates.rate_for(gpu, 1) == min(rates)


def test_every_shape_key_in_the_table_is_described():
    for r in gpu_rates.RATES:
        assert r.shape in gpu_rates.SHAPES, r.shape


# --------------------------------------------------------------------------- #
# tokens_per_dollar arithmetic
# --------------------------------------------------------------------------- #
def test_tokens_per_dollar_is_rate_times_an_hour_over_price():
    tpd = gpu_rates.tokens_per_dollar("H100 NVL", 1, 2.0, "9b-w20480-dec")
    assert tpd == pytest.approx(4050.0 * 3600.0 / 2.0)


def test_tokens_per_dollar_prices_the_ddp_premium_instead_of_assuming_it():
    """The owner ruling this table serves: buy the box with the best tokens
    per dollar, and a bigger box is fine when it IS that box. Four 5090s buy
    3.03x the tokens of one, so the answer flips on where the price sits
    against 3.03 -- which is the whole reason a rate table exists."""
    one = gpu_rates.tokens_per_dollar("RTX 5090", 1, 0.50, "7b-w12288-fit")
    cheap4 = gpu_rates.tokens_per_dollar("RTX 5090", 4, 1.40, "7b-w12288-fit")
    dear4 = gpu_rates.tokens_per_dollar("RTX 5090", 4, 1.60, "7b-w12288-fit")
    assert cheap4 > one   # 2.8x the price for 3.03x the tokens -> upgrade
    assert dear4 < one    # 3.2x the price for 3.03x the tokens -> do not
    # 8 cards at 4.44x are a worse buy than 4 at 3.03x per dollar at any price
    # proportional to card count -- TRAINING.md's "stop at 4 cards", in money.
    per_card_1_60 = 1.60 / 4
    assert (gpu_rates.tokens_per_dollar("RTX 5090", 8, per_card_1_60 * 8,
                                        "7b-w12288-fit") < dear4)


@pytest.mark.parametrize("dph", [0, -1, 0.0, None, "free", float("nan")])
def test_tokens_per_dollar_refuses_an_unpriced_box(dph):
    assert gpu_rates.tokens_per_dollar("H100 NVL", 1, dph,
                                       "9b-w20480-dec") is None


# --------------------------------------------------------------------------- #
# the provisional flag has to reach the caller
# --------------------------------------------------------------------------- #
def test_provisional_surfaces_on_the_entry_not_just_in_the_docs():
    firm = gpu_rates.entry_for("H100 NVL", 1, "9b-w20480-dec")
    prov = gpu_rates.entry_for("RTX 5090", 4, "7b-w12288-fit")
    assert firm.provisional is False and firm.measured is True
    assert prov.provisional is True and prov.measured is True
    assert prov.note  # a provisional cell must say WHY it is provisional


def test_every_multi_card_entry_is_provisional_and_post_boundary():
    """The DDP path was optimized on 2026-08-14 (e48def36). Nothing measured
    before that date may sit in this table as a multi-card rate."""
    multi = [r for r in gpu_rates.RATES if r.num_gpus > 1]
    assert multi, "the multi-card rows are the point of the caveat"
    for r in multi:
        assert r.provisional is True, r
        assert r.date >= "2026-08-14", r


def test_every_entry_is_measured_and_carries_datable_provenance():
    for r in gpu_rates.RATES:
        assert r.measured is True, r        # nothing modelled ships here
        assert r.tok_s > 0
        assert len(r.provenance) > 40, r    # a path or a run id, not a shrug
        assert len(r.date) == 10 and r.date[4] == "-"


def test_provenance_carries_no_absolute_machine_paths():
    """House rule: repo-relative paths or b2: URIs. An absolute /home path in
    a checked-in table is unreadable on every box but the one that wrote it."""
    for r in gpu_rates.RATES:
        for field in (r.provenance, r.note):
            assert "/home/" not in field, r
            assert not field.startswith("/"), r


def test_entries_are_uniquely_keyed():
    keys = [(r.gpu, r.num_gpus, r.shape) for r in gpu_rates.RATES]
    assert len(keys) == len(set(keys))


# --------------------------------------------------------------------------- #
# the null vector — an empty table answers None to everything and throws at
# nothing. Bands and fallbacks control the analyst, not the code.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gpu", ["H100 NVL", "RTX 5090", "", None, "junk"])
@pytest.mark.parametrize("shape", [None, "9b-w20480-dec", "nonexistent"])
def test_empty_table_returns_none_everywhere_and_raises_nothing(gpu, shape):
    empty = gpu_rates.build_index(())
    assert empty == {}
    assert gpu_rates.rate_for(gpu, 1, shape, table=empty) is None
    assert gpu_rates.entry_for(gpu, 1, shape, table=empty) is None
    assert gpu_rates.tokens_per_dollar(gpu, 1, 3.0, shape, table=empty) is None
    assert gpu_rates.shapes_for(gpu, 1, table=empty) == ()


def test_build_index_round_trips_the_shipped_table():
    idx = gpu_rates.build_index(gpu_rates.RATES)
    assert len(idx) == len(gpu_rates.RATES)
    for r in gpu_rates.RATES:
        assert idx[(r.gpu, r.num_gpus, r.shape)] is r


def test_the_module_stays_import_light():
    """It is imported for its side-effect-free constants from anywhere,
    including paths that must not touch the network or the vast API."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "gpu_rates.py")).read()
    for banned in ("import herdd", "import requests", "import urllib",
                   "import yaml", "subprocess", "open("):
        assert banned not in src, banned
