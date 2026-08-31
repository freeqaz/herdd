"""Tests for train_rates — the derived tok/s lookup over vram_facts.json.

Runs standalone from the lane root and under `scripts/test_tools.py`.
"""
from __future__ import annotations

import os
import statistics
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gpu_rates                                        # noqa: E402
import train_rates as tr                                # noqa: E402
import vram_facts                                       # noqa: E402


@pytest.fixture(scope="module")
def facts():
    return vram_facts.load_facts()


@pytest.fixture(scope="module")
def pool(facts):
    return tr.usable_anchors(facts)


# --------------------------------------------------------------------------- #
# synthetic anchors — the unit tests build their own table
# --------------------------------------------------------------------------- #
_CUR = ("2.13.0+cu129", "2.8.3", "5.15.1", "1.7.1")
_OLD = ("2.11.0+cu129", "2.8.3+cu128torch2.11", "5.13.0", "1.7.1")


def _anchor(run, gpu="H200 NVL", stack=_CUR, tok_step=100_000.0, p50=20.0,
            steps=12, peak=30.0, world_size=1, **shape):
    s = {"quant_mode": "bf16", "max_seq": 20480, "batch": 1, "grad_accum": 32,
         "eff_batch": 32, "world_size": world_size, "grad_checkpointing": True,
         "ce_chunk_matmul": "bf16", "target_modules": "all-linear",
         "lora_r": 32, "packing": "off", "attn_impl": "sdpa",
         "sdpa_backends": "flashmeff"}
    s.update(shape)
    return {
        "base_slug": "qwen35-9b", "run": run, "shape": s,
        "measured": {"peak_vram_alloc_gb": peak - vram_facts.RESERVED_HEADROOM_GB,
                     "peak_vram_reserved_gb": peak,
                     "peak_vram_alloc_gb_per_gpu": [],
                     "peak_vram_reserved_gb_per_gpu": []},
        "telemetry": {
            "hardware": {"gpu_names": [gpu] * world_size,
                         "torch_version": stack[0],
                         "flash_attn_version": stack[1],
                         "transformers_version": stack[2],
                         "trl_version": stack[3]},
            "throughput": {"tokens_seen": tok_step * steps,
                           "step_time_p50_s": p50, "n_steps_timed": steps,
                           "tokens_per_second": 1.0},
        },
    }


def _doc(*anchors):
    return {"schema": 1, "anchors": list(anchors)}


PROD = tr.Family(base_slug="qwen35-9b", quant_mode="bf16", max_seq=20480,
                 eff_batch=32, packing="off", target_modules_class="all-linear",
                 lora_r=32, ce_chunk_matmul="bf16")


# --------------------------------------------------------------------------- #
# import-lightness
# --------------------------------------------------------------------------- #
def test_import_reads_no_files_and_needs_no_cwd():
    """Importable from anywhere with nothing but its own directory on the path,
    and it opens the 1.2 MB anchor file lazily — `fleetd` and the submit path
    import this per candidate offer."""
    prog = ("import sys; sys.path.insert(0, %r);"
            "import train_rates, vram_facts;"
            "assert vram_facts._FACTS_CACHE == {}, 'read facts at import';"
            "print('ok')" % HERE)
    out = subprocess.run([sys.executable, "-c", prog], cwd=os.sep,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


# --------------------------------------------------------------------------- #
# card-name vocabulary: anchors speak torch, offers speak vast
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("torch_name,vast_name", [
    ("NVIDIA H200 NVL", "H200 NVL"),
    ("NVIDIA H200", "H200"),
    ("NVIDIA H100 NVL", "H100 NVL"),
    ("NVIDIA H100 PCIe", "H100 PCIE"),
    ("NVIDIA H100 80GB HBM3", "H100 SXM"),
    ("NVIDIA GeForce RTX 5090", "RTX 5090"),
    ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "RTX PRO 6000 S"),
    ("NVIDIA B200", "B200"),
])
def test_gpu_names_round_trip_between_anchor_and_offer(torch_name, vast_name):
    """The census's top cards must land on the same class key from either
    vocabulary, or a lookup keyed on an offer can never find its own anchors."""
    assert (gpu_rates.normalize_gpu_name(torch_name)
            == gpu_rates.normalize_gpu_name(vast_name) == vast_name)


@pytest.mark.parametrize("vast_name,anchor_name", [
    ("RTX PRO 6000 WS", "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"),
    ("A100 PCIE", "NVIDIA A100-PCIE-40GB"),
    ("A100 SXM4", "NVIDIA A100-SXM4-40GB"),
])
def test_ambiguous_offer_names_resolve_to_the_slower_part(vast_name, anchor_name):
    """vast's offer string does not name the SKU for these (the memory size and
    the Max-Q binning live elsewhere), so the fallback answers with the floor."""
    offer = gpu_rates.normalize_gpu_name(vast_name)
    assert offer != gpu_rates.normalize_gpu_name(anchor_name)
    assert gpu_rates.normalize_gpu_name(anchor_name) in tr.SKU_FALLBACK[offer]


def test_sku_fallback_agrees_with_gpu_rates():
    """Two tables, one doctrine. Pinned so they cannot drift apart silently."""
    assert tr.SKU_FALLBACK == gpu_rates._SKU_FALLBACK


# --------------------------------------------------------------------------- #
# tiering
# --------------------------------------------------------------------------- #
def test_current_stack_is_the_newest_dated_anchors_fingerprint():
    doc = _doc(_anchor("20260810T000000-a", stack=_OLD),
               _anchor("20260822T000000-b", stack=_CUR))
    assert tr.current_stack(tr.usable_anchors(doc)) == _CUR


def test_older_stack_is_provisional_and_current_stack_is_measured():
    """The epoch is FLEET-WIDE: a card whose only anchors predate the current
    image is provisional even though it is the newest thing on that card."""
    doc = _doc(_anchor("20260810T000000-old", stack=_OLD, p50=40.0),
               _anchor("20260822T000000-new", stack=_CUR, p50=20.0))
    assert tr.rate_for_offer(PROD, "H200 NVL", 1, 141, facts=doc).tier == "measured"

    doc = _doc(_anchor("20260810T000000-old", gpu="L40S", stack=_OLD, p50=40.0),
               _anchor("20260822T000000-new", stack=_CUR, p50=20.0))
    assert tr.rate_for_offer(PROD, "L40S", 1, 48, facts=doc).tier == "provisional"


def test_undated_anchor_is_provisional_even_on_the_current_stack():
    """Recency is what ages a rate out; an anchor that cannot be placed in time
    cannot be shown to be current. `local-smoke` and `rsft-4b-v0` are real."""
    doc = _doc(_anchor("local-smoke", stack=_CUR))
    assert tr.rate_for_offer(PROD, "H200 NVL", 1, 141, facts=doc).tier == "provisional"


def test_tiers_are_never_averaged():
    """A provisional floor and a measured reading do not have a meaningful mean.
    Three fast stale anchors must not pull the measured cell up."""
    doc = _doc(*[_anchor(f"2026081{i}T000000-old", stack=_OLD, p50=10.0)
                 for i in range(1, 4)],
               _anchor("20260822T000000-new", stack=_CUR, p50=20.0))
    est = tr.rate_for_offer(PROD, "H200 NVL", 1, 141, facts=doc)
    assert est.tier == "measured" and est.n == 1
    assert est.tok_s == pytest.approx(100_000.0 / 20.0, rel=1e-6)


def test_median_of_the_three_most_recent(monkeypatch):
    doc = _doc(_anchor("20260801T000000-a", p50=100.0),   # oldest, ignored
               _anchor("20260818T000000-b", p50=20.0),
               _anchor("20260819T000000-c", p50=25.0),
               _anchor("20260820T000000-d", p50=50.0))
    est = tr.rate_for_offer(PROD, "H200 NVL", 1, 141, facts=doc)
    assert est.n == tr.K_RECENT == 3
    assert est.tok_s == pytest.approx(100_000.0 / 25.0, rel=1e-6)
    assert est.spread == pytest.approx(2.5, rel=1e-3)     # 5000 / 2000
    assert "20260801T000000-a" not in est.runs


# --------------------------------------------------------------------------- #
# the 2026-08-14 multi-card boundary
# --------------------------------------------------------------------------- #
def test_real_pre_boundary_multi_card_anchors_are_excluded(facts, pool):
    """PINS A REAL ANCHOR, not a synthetic one.

    `20260813T203655-perf-levers-ddp3-2b4b` is a W=4 run dated 2026-08-13 — one
    day before `e48def36` made the DDP metric-gather defaults, worth +40.7% at
    W=4. It is in the shipped anchor file with a throughput block and it must
    not reach a rate. Its post-boundary sibling `20260814T090233-...-ddp3d-655e`
    (also W=4) must. Keyed on (run, world_size): these bench jobs ran W=1 and
    W=4 cells under one run id, and only the multi-card cells are excluded.
    """
    def multi(anchors):
        return {(a["run"], (a.get("shape") or {}).get("world_size"))
                for a in anchors if (a.get("shape") or {}).get("world_size", 1) > 1}

    raw, kept = multi(facts["anchors"]), multi(pool)
    for run in ("20260813T203655-perf-levers-ddp3-2b4b",
                "20260813T230654-perf-levers-ddp3b-3750"):
        assert (run, 4) in raw, f"{run} left the anchor file — re-pick the pin"
        assert (run, 4) not in kept
    assert ("20260814T090233-perf-levers-ddp3d-655e", 4) in kept
    assert ("20260814T090840-perf-levers-w8-135c", 8) in kept
    # nothing pre-boundary survives at any world size > 1
    assert all(d >= tr.MULTI_GPU_EPOCH
               for d in (tr.anchor_date(a) for a in pool
                         if tr._world_size(a) > 1))


def test_boundary_leaves_single_card_anchors_alone():
    """The W=1 null reproduced 3x inside 0.05% across the boundary, so a
    single-card anchor from 2026-08-01 is still evidence."""
    doc = _doc(_anchor("20260801T000000-w1", world_size=1))
    assert tr.rate_for_offer(PROD, "H200 NVL", 1, 141, facts=doc) is not None


def test_undated_multi_card_anchor_is_excluded():
    """Cannot be shown post-boundary, and the pre-boundary error is bigger than
    the gap between card classes — so it would be a wrong ranking, not a floor."""
    fam4 = tr.Family(**{**PROD.__dict__, "eff_batch": 128})
    doc = _doc(_anchor("ddp-smoke", world_size=4, eff_batch=128))
    assert tr.rate_for_offer(fam4, "H200 NVL", 4, 141, facts=doc) is None


# --------------------------------------------------------------------------- #
# operating points and the VRAM fit
# --------------------------------------------------------------------------- #
def test_fits_in_vram_picks_the_wider_operating_point_on_a_bigger_card():
    """The whole point of the VRAM axis: grad-checkpointing OFF costs peak and
    buys tok/s, so a card with room gets the faster operating point and a card
    without it gets the one that fits — never a rate it cannot run."""
    doc = _doc(
        _anchor("20260820T000000-gcon", p50=25.0, peak=35.0,
                grad_checkpointing_flag="on"),
        _anchor("20260820T000001-gcoff", p50=12.0, peak=80.0,
                grad_checkpointing_flag="0.0"),
    )
    big = tr.rate_for_offer(PROD, "H200 NVL", 1, 141, facts=doc)
    small = tr.rate_for_offer(PROD, "H200 NVL", 1, 48, facts=doc)
    assert "gc=none" in big.op_point and big.tok_s > small.tok_s
    assert "gc=full" in small.op_point
    assert "1 of 2 operating point(s) fit" in small.why


def test_no_vram_budget_skips_the_fit_filter():
    doc = _doc(_anchor("20260820T000000-fat", p50=12.0, peak=180.0))
    assert tr.rate_for_offer(PROD, "H200 NVL", 1, 141, facts=doc) is None
    assert tr.rate_for_offer(PROD, "H200 NVL", 1, None, facts=doc) is not None


def test_operating_point_split_keeps_eff_batch_in_the_family():
    """b1xga32 and b2xga16 are the same job at two operating points; the ranker
    may pick either. b1xga8 is a DIFFERENT job (eff_batch 8) and must not
    answer a query for eff_batch 32."""
    doc = _doc(_anchor("20260820T000000-split", batch=2, grad_accum=16,
                       eff_batch=32, p50=10.0))
    assert tr.rate_for_offer(PROD, "H200 NVL", 1, 141, facts=doc).op_point \
        .startswith("b2xga16")
    other = tr.Family(**{**PROD.__dict__, "eff_batch": 8})
    assert tr.rate_for_offer(other, "H200 NVL", 1, 141, facts=doc) is None


def test_per_card_peak_reads_the_max_of_the_per_gpu_list_never_a_sum(facts):
    """`peak_vram_reserved_gb` is a MAX over cards, so nothing divides by world
    size. Verified against every anchor that carries both."""
    n = 0
    for a in facts["anchors"]:
        m = a.get("measured") or {}
        for whole, per in (("peak_vram_reserved_gb", "peak_vram_reserved_gb_per_gpu"),
                           ("peak_vram_alloc_gb", "peak_vram_alloc_gb_per_gpu")):
            if m.get(per):
                n += 1
                assert max(m[per]) == pytest.approx(m[whole], abs=0.011), a["run"]
    assert n > 100


# --------------------------------------------------------------------------- #
# measurement-quality floor
# --------------------------------------------------------------------------- #
def test_two_step_probe_cells_are_refused():
    """A p50 over 2 steps is warmup. The real `fit_qla` cells report 423 tok/s
    where the same declared shape's 6- and 30-step cells report 3,400-5,200."""
    doc = _doc(_anchor("20260820T000000-fit", steps=2, p50=100.0))
    assert tr.usable_anchors(doc) == []
    assert tr.rate_for_offer(PROD, "H200 NVL", 1, 141, facts=doc) is None


def test_rate_is_the_p50_division_not_the_reported_segment_average():
    a = _anchor("20260820T000000-x", tok_step=100_000.0, p50=20.0, steps=12)
    a["telemetry"]["throughput"]["tokens_per_second"] = 999.0
    assert tr.anchor_rate(a) == pytest.approx(5000.0)


# --------------------------------------------------------------------------- #
# unmeasured is None, never a guess
# --------------------------------------------------------------------------- #
def test_unmeasured_card_returns_none_and_never_transfers_a_rate():
    doc = _doc(_anchor("20260820T000000-h200"))
    assert tr.rate_for_offer(PROD, "L40S", 1, 48, facts=doc) is None
    assert tr.rate_for_offer(PROD, "RTX 3090", 1, 24, facts=doc) is None
    assert tr.rate_for_offer(PROD, "", 1, 48, facts=doc) is None


def test_unmeasured_card_count_returns_none(facts):
    """A 4-card box is not four 1-card rates. `MULTI_GPU_SCALING` exists in
    gpu_rates for a caller who wants to make that leap by name."""
    assert tr.rate_for_offer(PROD, "H200 NVL", 4, 141, facts=facts) is None


def test_bad_arguments_return_none_rather_than_raising(facts):
    assert tr.rate_for_offer(None, "H200 NVL", 1, 141, facts=facts) is None
    assert tr.rate_for_offer(PROD, "H200 NVL", 0, 141, facts=facts) is None
    assert tr.rate_for_offer(PROD, "H200 NVL", True, 141, facts=facts) is None


def test_probe_hint_names_a_runnable_bundle():
    hint = tr.probe_hint(PROD, "L40S")
    assert "L40S" in hint and "gpu-rate-9b-w20480" in hint and "harvest_vram" in hint
    seven = tr.Family(**{**PROD.__dict__, "target_modules_class": "list-7"})
    assert "fit-ladder" in tr.probe_hint(seven, "L40S")


# --------------------------------------------------------------------------- #
# tokens_per_dollar
# --------------------------------------------------------------------------- #
def test_tokens_per_dollar():
    est = tr.RateEstimate(4578.0, "measured", 3, 1.01, ("r",), "b1xga32", "why")
    assert tr.tokens_per_dollar(est, 0.4969) == pytest.approx(4578.0 * 3600 / 0.4969)
    # None, not 0.0: a zero sorts as "worst buy", which is a claim we cannot make.
    for bad in (0.0, -1.0, None, "free"):
        assert tr.tokens_per_dollar(est, bad) is None
    assert tr.tokens_per_dollar(None, 1.0) is None


# --------------------------------------------------------------------------- #
# family_from_env
# --------------------------------------------------------------------------- #
def test_family_from_env_reads_a_production_bundle():
    fam = tr.family_from_env({
        "BASE_SLUG": "qwen35-9b", "MAX_SEQ": "20480", "BATCH": "1",
        "GRAD_ACCUM": "32", "GRAD_CKPT": "on", "TARGET_MODULES": "all-linear",
        "LORA_R": "32", "QUANT": "bf16", "CE_CHUNK_MATMUL": "bf16"})
    assert fam == PROD


def test_family_from_env_fills_trainer_defaults():
    """A bundle records only what it overrode; an anchor records what the
    trainer resolved. Without the defaults every bundle reads as unmeasured."""
    fam = tr.family_from_env({"BASE_SLUG": "qwen35-9b", "MAX_SEQ": "12288",
                              "BATCH": "1", "GRAD_ACCUM": "8"})
    assert fam.ce_chunk_matmul == "bf16" and fam.packing == "off"
    assert fam.lora_r == 32 and fam.target_modules_class == "list-7"


def test_family_from_env_uses_assets_and_the_ladder_floor():
    fam = tr.family_from_env(
        {"WINDOW_LADDER": "20480,16384", "BATCH": "1", "GRAD_ACCUM": "8"},
        assets=[{"name": "base", "b2": "base-models/qwen35-9b"}])
    assert fam.base_slug == "qwen35-9b" and fam.max_seq == 16384


def test_family_from_env_folds_world_size_into_eff_batch():
    env = {"BASE_SLUG": "qwen35-9b", "MAX_SEQ": "20480", "BATCH": "1",
           "GRAD_ACCUM": "8"}
    assert tr.family_from_env(env).eff_batch == 8
    assert tr.family_from_env(env, world_size=4).eff_batch == 32


@pytest.mark.parametrize("env", [
    None, {}, {"MAX_SEQ": "20480"},                      # no base
    {"BASE_SLUG": "qwen35-9b"},                          # no window, no ladder
    {"BASE_SLUG": "qwen35-9b", "MAX_SEQ": "not-a-number"},
    {"BASE_SLUG": "qwen35-9b", "MAX_SEQ": ["20480"]},
    "not-a-dict", 17,
])
def test_family_from_env_returns_none_never_raises(env):
    """An env this cannot map is an eval, a generation sweep or a probe — the
    caller degrades to price-only ranking, which is today's behaviour."""
    assert tr.family_from_env(env) is None


# --------------------------------------------------------------------------- #
# the shipped anchor file
# --------------------------------------------------------------------------- #
def test_index_cache_does_not_serve_one_document_for_another(facts):
    """`search` interleaves lookups; a memo keyed on the anchor document must
    not answer a synthetic table with the shipped one (or the reverse)."""
    empty = {"schema": 1, "anchors": []}
    assert tr.rate_for_offer(PROD, "H200 NVL", 1, 141, facts=empty) is None
    real = tr.rate_for_offer(PROD, "H200 NVL", 1, 141, facts=facts)
    assert real is not None
    assert tr.rate_for_offer(PROD, "H200 NVL", 1, 141, facts=empty) is None
    assert tr.rate_for_offer(PROD, "H200 NVL", 1, 141, facts=facts) == real


def test_shipped_census_is_derivable(pool):
    rows = tr.rate_census()
    assert len(pool) > 200
    assert len(rows) > 25
    assert all(r["tok_s"] > 0 for r in rows)
    assert {r["tier"] for r in rows} <= {"measured", "provisional"}


def test_derived_rates_reproduce_the_hand_curated_gpu_rates_cells(facts):
    """Two independent derivations of the same cell. `gpu_rates.py`'s A100 PCIE
    40GB and H100 NVL rows at `9b-w20480-dec` were hand-derived from the run
    artifacts; this module re-derives them from the harvested anchors. Loose
    band because the curated rows are means of two named repeats and this is a
    median of the K most recent."""
    for gpu, ram in (("A100 PCIE 40GB", 40), ("H100 NVL", 94)):
        curated = gpu_rates.rate_for(gpu, 1, "9b-w20480-dec")
        est = tr.rate_for_offer(PROD, gpu, 1, ram, facts=facts)
        assert est is not None, gpu
        assert est.tok_s == pytest.approx(curated, rel=0.10), (
            f"{gpu}: derived {est.tok_s} vs curated {curated} ({est.why})")


# --------------------------------------------------------------------------- #
# THE BACKTEST
# --------------------------------------------------------------------------- #
def _backtest(pool):
    cur = tr.current_stack(pool)
    cells: dict = {}
    for a in pool:
        cells.setdefault((tr.anchor_gpu(a), tr._world_size(a),
                          tr.anchor_family(a)), []).append(a)
    out = []
    for _key, group in cells.items():
        if len(group) < 4:
            continue
        dated = [a for a in group if tr.anchor_date(a)]
        if not dated:
            continue
        held = max(dated, key=lambda a: (tr.anchor_date(a), a.get("run", "")))
        op = tr._op_point(held["shape"])
        rest = [a for a in group if a is not held and tr._op_point(a["shape"]) == op]
        if not rest:
            continue
        est = tr._estimate_op_point(rest, cur, op, "")
        actual = tr.anchor_rate(held)
        out.append((abs(est.tok_s - actual) / actual, held["run"],
                    est.tok_s, actual))
    return out


def test_backtest_leave_one_out_newest_anchor(pool):
    """Hold out the newest anchor of every (gpu, world_size, family) cell with
    >= 4 anchors and re-derive it from the rest at its own operating point.

    TOLERANCE, AND WHY IT IS THIS LOOSE. Asserted: median relative error <= 15%
    and >= 75% of points within 25%. Measured 2026-08-28 over 22 cells: median
    6.2%, 86.4% within 25%. A tight max bound is NOT assertable and the two
    points that break it are both understood, both structural, and neither is a
    defect in the derivation:

      * `2026-08-17-v13-chain-train/chainonly` — predicted 4,915, measured
        2,379. Same declared family, `token_stats.row_tokens_mean` 748.6 against
        3,900 for every other anchor in the cell. `max_seq` is a CAP, not a
        length, and at BATCH=1 tok/s tracks the rows a corpus actually produces.
        Same lesson `vram_facts.anchor_tokens` records for peak VRAM, one metric
        over. A submit cannot know its corpus's length mix, so this residual is
        irreducible here; `RateEstimate.spread` is what surfaces it (2.03 on
        this cell).
      * `20260821T001641-gc-hybrid-accept.../f1_b4k_smoke_r1` — predicted 674,
        measured 4,276. The held-out anchor is the cell's ONLY current-stack
        witness, so holding it out leaves five 2026-08-16 gc-ladder cells, and
        those are a bucketed window ladder (2k/4k/.../16k rows inside one
        declared 20480 family) as well as an older epoch. The miss is in the
        FLOOR direction, which is the direction this module's provisional tier
        promises.

    So the honest statement is: the central tendency is good to single digits,
    and the tail is set by two things the anchor schema does not record — the
    corpus row-length mix and a bucketed-window bench harness. Narrowing it is
    stream C's job (stamp the epoch key), not a tolerance to tighten here.
    """
    res = _backtest(pool)
    assert len(res) >= 15, f"only {len(res)} backtest points"
    errs = sorted(r[0] for r in res)
    median = statistics.median(errs)
    within25 = sum(1 for e in errs if e <= 0.25) / len(errs)
    detail = "\n".join(f"  {e:6.1%} {run} derived {d:.0f} vs actual {a:.0f}"
                       for e, run, d, a in sorted(res, reverse=True)[:5])
    assert median <= 0.15, f"median relative error {median:.1%}\n{detail}"
    assert within25 >= 0.75, f"only {within25:.1%} within 25%\n{detail}"


def test_backtest_points_carry_provenance(pool):
    """A rate with no runs behind it is not evidence."""
    cur = tr.current_stack(pool)
    for a in pool[:50]:
        est = tr._estimate_op_point([a], cur, tr._op_point(a["shape"]), "")
        assert est.runs and all(est.runs)
        assert est.n == 1 and est.spread == pytest.approx(1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
