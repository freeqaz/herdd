"""Tests for vram_facts.py — the measured-anchor VRAM estimator.

The estimator's whole claim is that it answers from measurement rather than
arithmetic, so the tests are mostly *against the shipped anchors*: they assert
it reproduces numbers that appear in the campaign record, and that it refuses
where it has no data. Two negative tests exist specifically to stop the
estimator regressing into the model that was already measured wrong.

CPU-only, stdlib + pytest. Run: pytest tools/vast/test_vram_facts.py
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import vram_facts as vf  # noqa: E402

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]


@pytest.fixture(scope="module")
def facts():
    return vf.load_facts()


def _q(**over):
    base = dict(base_slug="qwen35-9b", quant_mode="bf16", grad_checkpointing=True,
                ce_chunk_matmul="fp32", target_modules=DEFAULT_TARGETS, lora_r=32,
                packing="off", fsdp="", world_size=1, batch=1)
    base.update(over)
    return base


# --- reproduces the record ----------------------------------------------------

def test_reproduces_the_fit_ladder_headline(facts):
    """fit-ladder measured 9B bf16 @ seq 12288 at 26.34 GB
    (FITTING_9B_ON_A_5090 §8). The estimate must contain that measurement."""
    r = vf.estimate_peak_gb(facts=facts, max_seq=12288, **_q())
    assert r["min"] <= 26.34 <= r["max"], r
    assert r["n"] >= 10, "the fit-ladder ladder should contribute many anchors"


def test_ce_chunk_matmul_is_the_measured_fitting_lever(facts):
    """§8.3: `--ce-chunk-matmul bf16` took 26.34 -> 22.33, a 4.01 GB saving.
    The estimator must SEE that as a distinct group, not average it away."""
    fp32 = vf.estimate_peak_gb(facts=facts, max_seq=12288, **_q())
    bf16 = vf.estimate_peak_gb(facts=facts, max_seq=12288,
                               **_q(ce_chunk_matmul="bf16"))
    assert bf16["gb"] < fp32["gb"], (bf16, fp32)


def test_grad_checkpointing_dominates_sequence_length(facts):
    """Measured on qwen25-coder-7b bf16 @ 4096: 20.87 on vs 52.20 off. That one
    knob is worth more than any window choice, which is the opposite of what
    the refuted analytic model implied."""
    on = vf.estimate_peak_gb(facts=facts, base_slug="qwen25-coder-7b-instruct",
                             quant_mode="bf16", grad_checkpointing=True,
                             ce_chunk_matmul="fp32", target_modules=DEFAULT_TARGETS,
                             lora_r=32, packing="off", world_size=1, max_seq=4096)
    off = vf.estimate_peak_gb(facts=facts, base_slug="qwen25-coder-7b-instruct",
                              quant_mode="bf16", grad_checkpointing=False,
                              ce_chunk_matmul="fp32", target_modules=DEFAULT_TARGETS,
                              lora_r=32, packing="off", world_size=1, max_seq=4096)
    assert off["gb"] > on["gb"] * 2, (on, off)


def test_sharding_changes_the_per_card_peak_but_ddp_does_not(facts):
    """fit-ladder wave 2: FSDP2 9B @ 12288 went 26.34 -> 18.00 -> 13.46 at ws
    1/2/4. Under DDP each rank holds a full replica, so world size is NOT part
    of the group key there — and must not be, or a 2-card DDP query would
    refuse against perfectly good 1-card anchors."""
    ddp1 = vf.estimate_peak_gb(facts=facts, max_seq=12288, **_q(world_size=1))
    ddp2 = vf.estimate_peak_gb(facts=facts, max_seq=12288, **_q(world_size=2))
    assert ddp1["gb"] == ddp2["gb"]

    fsdp4 = vf.estimate_peak_gb(facts=facts, max_seq=12288,
                                **_q(fsdp="full_shard auto_wrap", world_size=4))
    assert fsdp4["gb"] < ddp1["gb"], (fsdp4, ddp1)


# --- the flat-slice slope must not come back as an extrapolator ---------------

def test_the_flat_slice_slope_is_never_charged_for_a_window():
    """0.063 GB/1k is not a window slope and must not be used as one.

    It was read off a ladder whose rows had stopped growing; exercised windows
    measure 0.412-0.446 (GPU5090_SIZING_2026-08-29 §2.1). The constant survives
    only as the grad-ckpt-OFF counter-example, so this pins that the number the
    extrapolator actually charges is the measured floor and not it.
    """
    assert vf.DISPROVEN_FLAT_SLICE_SLOPE_GB_PER_1K == 0.063
    assert vf.EXTRAPOLATION_SLOPE_FLOOR_GB_PER_1K >= 0.41
    assert vf.group_slope_gb_per_1k([]) == vf.EXTRAPOLATION_SLOPE_FLOOR_GB_PER_1K


def test_a_shorter_window_is_never_charged_a_longer_ones_peak(facts):
    """Interpolation DOWN an anchored axis stays sound: seq 2048 must not
    inherit seq 12288's measured peak. (This test used to assert 'window growth
    is nearly free', which was the flat-slice artifact.)"""
    short = vf.estimate_peak_gb(facts=facts, max_seq=2048, **_q())
    assert short["gb"] < vf.estimate_peak_gb(facts=facts, max_seq=12288,
                                             **_q())["gb"]


# --- extrapolation fails closed -----------------------------------------------
# The measured misses these pin are GPU5090_SIZING_2026-08-29 §5.2, RTX 5090,
# 9B bf16 LoRA r32 / ce-chunk-matmul bf16 / grad-ckpt on, target modules =
# the trainer's default seven projections, rows minted to fill each window:
#
#   window   this table said   the card measured
#   12,288   22.33 GB (n=1)    26.08 GB
#   20,480   22.83 EXTRAPOLATED 30.55
#   24,576   23.09 EXTRAPOLATED 31.02
_5090_LIST7_MEASURED = {12288: 26.08, 20480: 30.55, 24576: 31.02}


def _list7(**over):
    """The exact query GPU5090_SIZING §2.2 shows `vram_facts` answering."""
    return _q(ce_chunk_matmul="bf16", **over)


def test_past_the_longest_anchor_it_refuses_by_default(facts):
    """The default answer to a window nobody measured is a refusal, because the
    old answer was an under-read and an under-read here is an OOM."""
    for seq in (20480, 24576):
        with pytest.raises(vf.Unmeasured) as e:
            vf.estimate_peak_gb(facts=facts, max_seq=seq, **_list7())
        assert "12288" in str(e.value), "it must name the longest anchor it has"
        assert e.value.probe_cmd, "a refusal must say how to mint the anchor"


def test_the_refusal_reaches_the_card_sizing_entry_point(facts):
    with pytest.raises(vf.Unmeasured):
        vf.required_gpu_ram_gb(facts=facts, max_seq=20480, **_list7())


def test_the_anchored_window_still_answers_and_is_the_sound_path(facts):
    """Only the extrapolation changed. 12,288 has an anchor, so it answers --
    and this is the row the 5090 run found sound to 2.8% in the all-linear
    group."""
    r = vf.estimate_peak_gb(facts=facts, max_seq=12288, **_list7())
    assert not r["extrapolated"] and r["extrapolated_gb"] == 0.0
    assert r["gb"] == 22.33


def test_opting_in_charges_the_measured_floor_not_the_flat_slice_slope(facts):
    """The regression: at 0.063 GB/1k these answered 22.83 and 23.09 against
    30.55 and 31.02 measured. The floor cannot reach the measurement -- the
    n=1 anchor it builds on is itself 14.4% low -- but it must at least move
    the number by the measured slope."""
    for seq in (20480, 24576):
        r = vf.estimate_peak_gb(facts=facts, max_seq=seq, allow_extrapolation=True,
                                **_list7())
        assert r["extrapolated"]
        gap_k = (seq - 12288) / 1024.0
        assert r["extrapolated_gb"] == pytest.approx(
            vf.EXTRAPOLATION_SLOPE_FLOOR_GB_PER_1K * gap_k, abs=0.01)
        old = 22.33 + vf.DISPROVEN_FLAT_SLICE_SLOPE_GB_PER_1K * gap_k
        assert r["gb"] > old + 2.0, "must not reproduce the old under-read"


def test_the_risk_band_covers_the_5090_measurement(facts):
    """An opt-in extrapolation is only honest if the band it reports reaches
    the number the card actually produced. `required_gb + risk_gb` does, at
    both windows where the old point estimate did not."""
    for seq, actual in ((20480, _5090_LIST7_MEASURED[20480]),
                        (24576, _5090_LIST7_MEASURED[24576])):
        r = vf.required_gpu_ram_gb(facts=facts, max_seq=seq,
                                   allow_extrapolation=True, **_list7())
        assert r["risk_gb"] >= r["extrapolated_gb"] > 0, \
            "a single-anchor group has no spread, so risk used to read 0.00"
        assert r["required_gb"] + r["risk_gb"] >= actual


def test_the_backtest_still_scores_the_extrapolated_arm(facts):
    """Holding out a group's longest anchor IS the refused case, so the
    backtest must opt in -- otherwise it silently stops measuring the arm it
    exists to measure and reports a cleaner number for having gone blind."""
    r = vf.backtest(facts=facts)
    assert r["skipped_unmeasured"] == 0
    assert r["n"] > 400


# --- refusal ------------------------------------------------------------------

def test_unknown_base_refuses_and_names_the_probe(facts):
    with pytest.raises(vf.Unmeasured) as e:
        vf.estimate_peak_gb(facts=facts, max_seq=4096,
                            **_q(base_slug="llama-99b-imaginary"))
    assert "fit-ladder" in e.value.probe_cmd


def test_unnamed_base_refuses(facts):
    """An unidentifiable base (`assets/base`, a content-addressed snapshot dir)
    must not silently match anything."""
    with pytest.raises(vf.Unmeasured):
        vf.estimate_peak_gb(facts=facts, max_seq=4096, **_q(base_slug=""))


def test_known_base_unknown_shape_says_which_dimension_is_missing(facts):
    with pytest.raises(vf.Unmeasured) as e:
        vf.estimate_peak_gb(facts=facts, max_seq=4096, **_q(quant_mode="fp8"))
    assert "fp8" in str(e.value)


# --- sizing -------------------------------------------------------------------

def test_required_rounds_up_to_a_rentable_card(facts):
    r = vf.required_gpu_ram_gb(facts=facts, max_seq=12288, **_q())
    assert r["card_class"] in vf.CARD_CLASSES
    assert r["card_class"] >= r["required_gb"]
    assert r["required_gb"] == pytest.approx(r["gb"] + r["headroom_gb"])


def test_headroom_does_not_inflate_with_spread(facts):
    """A spread-proportional headroom was tried and rejected on measurement: it
    halved card-class accuracy (65% -> 34%) while preventing zero additional
    class-level under-sizes. Spread is REPORTED as risk instead. This pins that
    decision so it is not quietly re-introduced as an obvious safety win."""
    assert vf.headroom_for({"spread": 0.0}) == vf.headroom_for({"spread": 6.32})


def test_spread_is_reported_as_risk(facts):
    r = vf.required_gpu_ram_gb(facts=facts, max_seq=12288, **_q())
    assert r["risk_gb"] == r["spread"]
    assert r["risk_gb"] > 0, "this group is known to be heterogeneous"


def test_card_class_never_rounds_down():
    for gb in (0.5, 23.9, 24.0, 24.1, 47.9, 95.5, 200.0):
        assert vf.card_class_for(gb) >= gb


def test_estimate_is_the_group_max_not_the_mean(facts):
    """The failure mode of the mean is an OOM. Within-group spread here reaches
    6.3 GB at identical declared shape, so the conservative end is the only
    safe one to size on."""
    r = vf.estimate_peak_gb(facts=facts, max_seq=12288, **_q())
    assert r["gb"] == r["max"]
    assert r["spread"] > 0, "this group is known to be heterogeneous"


# --- the tokens-in-flight coordinate ------------------------------------------
# `anchor_tokens` prefers a run's MEASURED longest row (schema-2
# `token_stats.row_tokens_max`) over its declared `max_seq`. The A/B that
# followed (below, `test_the_estimate_path_ignores_the_measured_row_coordinate`)
# took the preference OFF the estimate path and left it on the anchor-to-anchor
# knob report. It was measured on 2026-08-13 within ONE job (36 anchors) and
# RE-MEASURED out of sample on 2026-08-16, when the table went 160 -> 247
# anchors and the instrumented subset went 36 -> 111 across 15 jobs. Same
# verdict, wider margin. These cases are synthetic so they pin the function's
# contract rather than any one job's numbers.

def _synth(run="r", peak=20.0, *, ce="fp32", quant="bf16", ws=1, seq=4096,
           batch=1, grad_accum=4, corpus="c1", lora_r=32, packing="off",
           gck=True, targets=None, step=10.0, telemetry=None,
           base="synth-1b", fsdp=""):
    a = {
        "base_slug": base, "run": run, "sources": [],
        "shape": {"base": f"assets/{base}", "quant_mode": quant, "max_seq": seq,
                  "batch": batch, "grad_accum": grad_accum, "world_size": ws,
                  "grad_checkpointing": gck, "ce_chunk_matmul": ce,
                  "target_modules": targets or DEFAULT_TARGETS,
                  "lora_r": lora_r, "packing": packing, "fsdp": fsdp,
                  "device_map_mode": "single"},
        "context": {"dataset_content_sha256": corpus, "step_time_seconds": step},
        "measured": {"peak_vram_alloc_gb": peak},
    }
    if telemetry:
        a["telemetry"] = telemetry
    return a


def _tok(row_max):
    return {"token_stats": {"row_tokens_max": row_max}}


def test_tokens_falls_back_to_the_declared_window():
    """Every anchor in the shipped table takes this path."""
    assert vf.anchor_tokens(_synth(seq=4096, batch=2)) == 8192
    assert vf.anchor_tokens({"shape": {}}) is None


def test_measured_row_max_is_preferred_when_present():
    """A 12288 cap that the corpus never reached is a 5120-token measurement,
    and pretending otherwise is what the within-group spread is made of."""
    a = _synth(seq=12288, telemetry=_tok(5120))
    assert vf.anchor_tokens(a) == 5120


def test_measured_row_max_is_capped_at_the_declared_window():
    """The trainer truncates at `max_seq`, so a longer row does not put more
    than the cap in flight — reading the raw row length would invent tokens the
    run never held."""
    assert vf.anchor_tokens(_synth(seq=4096, telemetry=_tok(9000))) == 4096


def test_measured_row_max_is_multiplied_by_batch():
    a = _synth(seq=8192, batch=2, telemetry=_tok(3000))
    assert vf.anchor_tokens(a) == 6000


def test_a_junk_row_max_falls_back_rather_than_zeroing_the_coordinate():
    for bad in (0, -1, None, "long"):
        assert vf.anchor_tokens(_synth(seq=4096, telemetry=_tok(bad))) == 4096


def _strip_token_stats(facts):
    anchors = []
    for a in facts["anchors"]:
        tel = a.get("telemetry")
        if tel and "token_stats" in tel:
            a = dict(a)
            a["telemetry"] = {k: v for k, v in tel.items() if k != "token_stats"}
        anchors.append(a)
    return {"schema": 1, "anchors": anchors}


def test_the_estimate_path_ignores_the_measured_row_coordinate(facts):
    """THE ACCEPTANCE GATE, RESOLVED (2026-08-13, n=36; VRAM_SIZING.md,
    "Refining the coordinate"). It ran, and the answer was no.

    Scored on the 36 anchors that carry `token_stats`, moving the ANCHORS onto
    their measured longest row while the QUERY stays on its declared window
    made the estimate worse in every statistic: card class 83.3% -> 58.3%,
    median |error| 1.83 -> 2.84 GB, p90 5.10 -> 14.06, worst 11.48 -> 23.38.
    Mechanism, not noise: the query cannot know its corpus's longest row, so
    the two sides end up on different axes and 27 of the 36 points the declared
    axis covered exactly became extrapolations off the end of a compressed one.

    So the estimate path now asks for the DECLARED coordinate on both sides,
    and this pins that: `token_stats` present or stripped, every held-out
    prediction is identical. The measured coordinate keeps its default in
    `anchor_tokens` for the knob report, where both sides are anchors and the
    comparison is symmetric."""
    with_ts = vf.backtest(facts)
    without = vf.backtest(_strip_token_stats(facts))
    assert with_ts["errors"] == without["errors"], (
        "an anchor's token_stats moved a production estimate — the estimate "
        "path must stay on the declared coordinate, see the A/B in "
        "VRAM_SIZING.md")


def test_the_measured_coordinate_arm_is_the_one_that_measured_worse(facts):
    """Pins the A/B's direction so the preference cannot be flipped back into
    the estimate path on the strength of how sensible it sounds. It sounds very
    sensible; it rents one class too big 33 times in 110.

    MEASURED THREE TIMES, TWICE OUT OF SAMPLE. 2026-08-13, n=36, one sweep,
    one base: declared 83.3% / measured 58.3% / oracle 91.7%. 2026-08-16,
    111 instrumented across fifteen jobs: declared 80.9% / measured 70.0% /
    oracle 94.5%. 2026-08-25, after the gc_flag group key, the query fix that
    let fractional cells find their own groups, and the no-op-resume filter
    (323 usable instrumented; 320 score + 3 singleton + 0 unmeasured):
    declared **64.7%** / measured **57.2%** / oracle **83.4%**, median |e|
    5.05 / 5.36 / 4.35, p90 29.02 / 44.95 / 18.46. The absolute level dropped
    because the scored set now contains the fractional-GC ladder probes in
    their own small groups — a harder subset, not a worse estimator. The
    ORDERING is the pinned claim, and it has held through every widening.

    The oracle arm (query on the measured axis too) is still the BEST of the
    three, which is the honest reason this is not simply "the idea was wrong":
    it is not shippable, because nothing at submit time knows the longest row a
    corpus will produce. Making it shippable is the owed work — declare
    `row_tokens_max` in the bundle at prep time, then re-run `--backtest`.

    ON THE NEVER-UNDER CLAIM, re-scoped 2026-08-25. Under-estimates are no
    longer one bookkeeping point: the declared arm under-predicts 3 held-out
    points, the worst by 30.72 GB, and 2 land a card class short. But every
    one of them, in every arm, is a `gc-ladder` worst-case probe cell — a run
    whose cells exist to be their group's extremes, so holding one out removes
    the only evidence that could have covered it (`vf.backtest.__doc__`, the
    designed self-bias, now at probe scale). The claim with teeth is
    therefore SCOPED: on production shapes, no arm under-estimates past the
    reserved headroom and no arm lands a class short. A probe cell defeating
    leave-one-out is the probe doing its job; a production shape doing so is
    the alarm this test exists to raise."""
    def scored(a):
        return bool((a.get("telemetry") or {}).get("token_stats"))

    declared = vf.backtest(facts, score_only=scored)
    measured = vf.backtest(facts, prefer_measured_tokens=True,
                           score_only=scored)
    oracle = vf.backtest(facts, prefer_measured_tokens=True,
                         oracle_query_tokens=True, score_only=scored)
    assert measured["class_accuracy"] < declared["class_accuracy"]
    assert measured["p90_abs_gb"] > declared["p90_abs_gb"]
    assert oracle["class_accuracy"] > declared["class_accuracy"]

    for name, arm in (("declared", declared), ("measured", measured),
                      ("oracle", oracle)):
        stray = [u for u in arm["under_points"]
                 if "gc-ladder" not in u["run"]
                 and abs(u["signed_gb"]) >= vf.RESERVED_HEADROOM_GB]
        assert stray == [], (
            f"{name} arm under-estimates a PRODUCTION shape past the "
            f"{vf.RESERVED_HEADROOM_GB} GB headroom: {stray}. The self-bias "
            f"absorbed by the headroom is expected; this is not it — a new "
            f"group maximum landed and the verdict has to be re-taken.")
        stray_class = [u for u in arm["under_class"]
                       if "gc-ladder" not in u["run"]]
        assert stray_class == [], (
            f"{name} arm rents a card class too small for a production "
            f"shape: {stray_class}")
    # The two arms that share a query axis must fail on the SAME probe cells —
    # a divergence would mean the anchor-side coordinate started costing
    # safety, not just money.
    assert ({(u["run"], u["unit"]) for u in declared["under_class"]}
            == {(u["run"], u["unit"]) for u in measured["under_class"]})


def test_no_op_resumes_are_filtered_at_harvest(facts):
    """REPLACES `test_the_table_admits_idle_resumes_and_they_inflate_the_
    reported_risk`, a tripwire retired 2026-08-25 by the fix it demanded.

    A resume that re-entered a finished segment wrote a train_summary whose
    18.60 GB peak was a model sitting in memory; a second such row
    (`tuner-v0`, 5.79 GB) landed and proved the defect was spreading. Both
    could only ever be a group's MIN (the estimate is the max), but the min
    feeds `spread`, spread is `risk_gb`, and `risk_gb` widens jobmeta's
    refusal band — so an idle row silently bought wrong declarations ~13 GB
    of tolerance. `harvest_vram.MIN_LIVE_STEP_TIME_S` now excludes them at
    admission, on the one signal both rows carried.

    Two regressions pinned: the committed table holds no idle rows, and the
    threshold still sits in a wide gap below the smallest REAL training step
    on file (0.484 s at the retire date, 160x the idle rows' 2-3 ms). If real
    step times ever drift toward the threshold, the second assert fires and
    the threshold has to be re-adjudicated, not nudged."""
    import harvest_vram

    step_times = [a["context"]["step_time_seconds"] for a in facts["anchors"]
                  if isinstance((a.get("context") or {})
                                .get("step_time_seconds"), (int, float))]
    assert step_times, "step-time telemetry vanished from the table entirely"
    assert min(step_times) >= harvest_vram.MIN_LIVE_STEP_TIME_S, (
        f"an idle-resume anchor is back in the committed table "
        f"(step_time {min(step_times)}s) — the harvest filter regressed, or "
        f"the table was written by a stale harvest_vram.py")
    assert min(step_times) > 4 * harvest_vram.MIN_LIVE_STEP_TIME_S, (
        "the gap between the liveness threshold and the smallest real "
        "training step has narrowed — re-adjudicate the threshold before a "
        "real run gets filtered as idle")


# --- knob-impact report -------------------------------------------------------

def test_report_finds_a_matched_pair_and_measures_only_the_knob():
    facts = {"schema": 1, "anchors": [
        _synth("a1", 20.0, ce="fp32"), _synth("a2", 21.0, ce="fp32"),
        _synth("b1", 16.0, ce="bf16")]}
    (f,) = vf.knob_findings(facts, "ce_chunk_matmul")
    assert (f["from"], f["to"]) == ("bf16", "fp32")
    assert f["delta_gb"] == 5.0                     # max vs max: 21.0 - 16.0
    assert f["delta_gb_min"] == 4.0                 # min vs min: 20.0 - 16.0
    assert (f["a"]["n"], f["b"]["n"]) == (1, 2)
    assert f["a"]["runs"] == ["b1"] and f["b"]["runs"] == ["a1", "a2"]


def test_report_says_unmeasured_where_no_pair_isolates_the_knob():
    """The whole point of the mode: no matched pair is reported as no answer,
    never as an interpolation across unmatched groups."""
    facts = {"schema": 1, "anchors": [
        _synth("a1", 20.0, lora_r=32), _synth("a2", 30.0, lora_r=64, seq=8192)]}
    assert vf.knob_findings(facts, "lora_r") == []
    rep = vf.knob_report(facts, ("lora_r",))
    assert rep["knobs"]["lora_r"] == []
    assert "lora_r: unmeasured" in "\n".join(vf.format_knob_report(rep))


def test_report_does_not_pair_across_corpora_by_default():
    """Two runs of one declared shape measured 20.77 and 24.56 GB purely by
    which rows survived the drop policy. A pair straddling that would report
    3.8 GB of corpus as a knob effect."""
    facts = {"schema": 1, "anchors": [
        _synth("a1", 20.0, ce="fp32", corpus="c1"),
        _synth("b1", 16.0, ce="bf16", corpus="c2")]}
    assert vf.knob_findings(facts, "ce_chunk_matmul") == []
    loose = vf.knob_findings(facts, "ce_chunk_matmul", match_corpus=False)
    assert len(loose) == 1 and loose[0]["delta_gb"] == 4.0


def test_report_does_not_pair_a_setting_against_an_unrecorded_field():
    """`unknown` is what normalization produces when the summary never wrote
    the field — older trainers recorded no ce_chunk_matmul at all. A
    `fp32 -> unknown` pair publishes a missing field as a measured setting."""
    facts = {"schema": 1, "anchors": [
        _synth("a1", 20.0, ce="fp32"), _synth("b1", 16.0, ce=None)]}
    assert vf.knob_findings(facts, "ce_chunk_matmul") == []
    ws = {"schema": 1, "anchors": [
        _synth("a1", 20.0, ws=1), _synth("b1", 16.0, ws=None)]}
    assert vf.knob_findings(ws, "world_size") == []


def test_report_does_not_pair_across_windows():
    facts = {"schema": 1, "anchors": [
        _synth("a1", 20.0, ce="fp32", seq=4096),
        _synth("b1", 16.0, ce="bf16", seq=12288)]}
    assert vf.knob_findings(facts, "ce_chunk_matmul") == []


def test_report_uses_the_measured_coordinate_when_matching_windows():
    """Two runs declaring different caps that measured the same longest row are
    the same point on the token axis, and pairing them is the refinement's whole
    purpose. ANCHOR against ANCHOR — both sides carry the block, so the
    comparison is symmetric, which is exactly the case the 2026-08-13 A/B left
    the preference switched on for."""
    facts = {"schema": 1, "anchors": [
        _synth("a1", 20.0, ce="fp32", seq=12288, telemetry=_tok(4096)),
        _synth("b1", 16.0, ce="bf16", seq=4096)]}
    (f,) = vf.knob_findings(facts, "ce_chunk_matmul")
    assert f["delta_gb"] == 4.0


def test_step_time_is_not_compared_across_different_grad_accum():
    """A per-step wall time is denominated in micro-batches per step, and
    grad_accum is not in the VRAM group key. On the real table this guard is
    what stops ws 1->2 being published at +239% on one pair and -45% on
    another."""
    facts = {"schema": 1, "anchors": [
        _synth("a1", 20.0, ce="fp32", grad_accum=16, step=20.0),
        _synth("b1", 16.0, ce="bf16", grad_accum=4, step=5.0)]}
    (f,) = vf.knob_findings(facts, "ce_chunk_matmul")
    assert "delta_step_pct" not in f
    assert "grad_accum" in f["step_time_incomparable"] or \
        "micro-batches" in f["step_time_incomparable"]

    same = {"schema": 1, "anchors": [
        _synth("a1", 20.0, ce="fp32", grad_accum=4, step=10.0),
        _synth("b1", 16.0, ce="bf16", grad_accum=4, step=5.0)]}
    (g,) = vf.knob_findings(same, "ce_chunk_matmul")
    assert g["delta_step_pct"] == 100.0             # bf16 5s -> fp32 10s


def test_report_carries_throughput_only_when_both_sides_measured_it():
    both = {"schema": 1, "anchors": [
        _synth("a1", 20.0, ce="fp32",
               telemetry={"throughput": {"tokens_per_second": 1000.0}}),
        _synth("b1", 16.0, ce="bf16",
               telemetry={"throughput": {"tokens_per_second": 1200.0}})]}
    (f,) = vf.knob_findings(both, "ce_chunk_matmul")
    assert f["delta_tps"] == -200.0 and f["delta_tps_pct"] == pytest.approx(-16.7)

    one = {"schema": 1, "anchors": [
        _synth("a1", 20.0, ce="fp32"),
        _synth("b1", 16.0, ce="bf16",
               telemetry={"throughput": {"tokens_per_second": 1200.0}})]}
    (g,) = vf.knob_findings(one, "ce_chunk_matmul")
    assert "delta_tps" not in g and "delta_tps_pct" not in g


def test_report_reproduces_the_measured_ce_chunk_matmul_saving(facts):
    """Against the SHIPPED anchors, not a fixture: fit-ladder measured
    26.34 -> 22.33 for `--ce-chunk-matmul bf16` on the 9B at seq 12288
    (FITTING_9B_ON_A_5090 §8.3, VRAM_SIZING fact 2). The report must recover
    that 4.01 GB from the table alone."""
    found = [f for f in vf.knob_findings(facts, "ce_chunk_matmul")
             if f["context"]["base_slug"] == "qwen35-9b"
             and f["context"]["tokens"] == 12288
             and (f["from"], f["to"]) == ("bf16", "fp32")]
    assert found, "the 9B ce-chunk-matmul pair vanished from the anchor table"
    assert any(f["delta_gb_min"] == pytest.approx(4.01, abs=0.01) for f in found)


def test_report_covers_every_knob_the_owner_asked_about():
    for knob in ("ce_chunk_matmul", "grad_checkpointing", "quant_mode",
                 "packing", "world_size", "lora_r", "target_modules_class"):
        assert knob in vf.REPORT_KNOBS


def test_report_runs_over_the_shipped_table(facts):
    """End to end on real anchors: it must produce a readout, and every knob
    must be either measured or explicitly unmeasured — no silent omission."""
    rep = vf.knob_report(facts)
    text = "\n".join(vf.format_knob_report(rep))
    for knob in vf.REPORT_KNOBS:
        assert knob in text
        assert isinstance(rep["knobs"][knob], list)
    assert rep["n_anchors"] > 50


# --- backtest -----------------------------------------------------------------

def test_leave_one_out_picks_the_right_card_class(facts):
    """Drop each anchor, estimate its shape from the rest, and score the
    decision that costs money: which card class you would have rented.

    SCORED ON CLASS, NOT GB, on purpose. Classes are coarse (24/32/48/80/96),
    so a 3 GB over-estimate is usually free and a 3 GB one straddling 32 is not.
    Optimizing the GB error is optimizing a number nobody spends.

    THIS TEST IS BIASED AGAINST ITSELF, and the bias is worth naming. The
    estimate is a group MAX, so holding out the group's maximum GUARANTEES an
    under-prediction of that point — the anchor being predicted is the very
    evidence that would have covered it. In production every measured shape is
    in the table. So the under-count here is a pessimistic proxy for "a new run
    lands above everything seen", not a defect rate.

    It is a REGRESSION gate, not a validation: it says a change to grouping,
    slope, or headroom did not make us rent worse than we do today. The harness
    moved into `vram_facts.backtest` on 2026-08-13, because the coordinate A/B
    needed to point it at two arms and a harness that only exists inside an
    assertion cannot be pointed anywhere."""
    r = vf.backtest(facts)
    assert r["n"] >= 40, f"backtest too thin to mean anything: n={r['n']}"
    print("\n" + "\n".join(vf.format_backtest({"shipped": r})))

    # The bar is comparative, not absolute: today's rule lands the right class
    # about two thirds of the time, and a change that makes it materially
    # worse is a regression. Class-level under-calls are scoped the same way
    # as the A/B test above: a `gc-ladder` worst-case probe cell defeating
    # leave-one-out is the designed self-bias at probe scale; the standing
    # NON-probe under-call is one (`local-smoke .../i_on_2048_r1`, 16.33 GB
    # measured vs 15.96 predicted — a 0.37 GB straddle of the 16 GB edge).
    assert r["class_accuracy"] >= 0.55, (
        f"card-class accuracy regressed: {r['exact']}/{r['n']}")
    stray = [u for u in r["under_class"] if "gc-ladder" not in u["run"]]
    assert len(stray) <= 1, (
        f"class-level under-sizing on production shapes: {stray}")


def test_backtest_scores_two_arms_on_the_same_held_out_points(facts):
    """`score_only` is what makes the A/B an A/B. An arm evaluated on a
    different subset than the other is not a comparison, so the harness has to
    hold the scored set fixed while the coordinate moves.

    THE INVARIANT IS BOOKKEEPING, NOT A COUNT. This used to assert
    `n == len(token_stats anchors)`, which was true only while every
    instrumented anchor happened to have a groupmate. The 2026-08-16 harvest
    landed 111 instrumented anchors of which 110 score: the odd one out is a
    `local-smoke` cell of the w5a bundle at **quant 8bit**, the table's only
    8-bit qwen35-9b anchor, so its group is a singleton and leave-one-out has
    literally nothing left to predict from. `backtest` reports that as
    `skipped_singleton` rather than silently shrinking `n` — so the invariant
    with teeth is that the three numbers ADD UP, and that whatever is dropped is
    dropped identically in both arms.

    Written against the result's own skip fields on purpose: a magic 110 would
    go stale on the next harvest and would not have caught the failure it is
    guarding — an arm that quietly scores a different subset than its
    counterpart. `skipped_unmeasured` is pinned at 0 separately because that one
    is not benign: it would mean a held-out query stopped matching its own
    group, which is a grouping bug wearing a skip's clothes."""
    def scored(a):
        return bool((a.get("telemetry") or {}).get("token_stats"))

    a_arm = vf.backtest(facts, score_only=scored)
    b_arm = vf.backtest(facts, prefer_measured_tokens=True, score_only=scored)
    # Over the anchors backtest can SEE: a row with no base_slug is invisible
    # to the estimator (there is no group to put it in), and both arms drop it
    # identically before scoring — so it belongs outside the invariant, not
    # inside `skipped_*`.
    n_scorable = sum(1 for x in facts["anchors"]
                     if scored(x) and x.get("base_slug"))

    assert a_arm["n"] == b_arm["n"] > 0
    for name, arm in (("A", a_arm), ("B", b_arm)):
        assert (arm["n"] + arm["skipped_singleton"]
                + arm["skipped_unmeasured"]) == n_scorable, (
            f"arm {name} lost held-out points it did not account for: scored "
            f"{arm['n']} + singleton {arm['skipped_singleton']} + unmeasured "
            f"{arm['skipped_unmeasured']} != {n_scorable} instrumented anchors")
        assert arm["skipped_unmeasured"] == 0, (
            "a held-out query failed to match its own anchor group — that is a "
            "grouping bug, not a skip")
    assert a_arm["skipped_singleton"] == b_arm["skipped_singleton"], (
        "the arms dropped different held-out points; this is no longer an A/B")


# --- the grad-ckpt-OFF calibration autotune reads ------------------------------

def test_grad_ckpt_off_calibration_is_a_lookup_not_a_model(facts):
    """`grad_ckpt_off_calibration` must return anchors that are literally on
    file, with their provenance — no arithmetic, no extrapolation.

    This is what `autotune.GRAD_CKPT_OFF_REF_VRAM_GB` / `_OFF_MIN_VRAM_GB` are
    pinned to (test_autotune.py). The value it replaces was an analytic 32 GB
    that the reference run measured at 52.20.

    Declared coordinate, matching the function: `ref_tokens` is a declared
    `batch x seq` (autotune's bash mirror has nothing else), so the anchors are
    read on that axis — same argument as the estimate path."""
    cal = vf.grad_ckpt_off_calibration(facts=facts)
    off = [a for a in facts["anchors"]
           if vf._anchor_shape(a)["grad_checkpointing"] == "false"]
    peaks_at_ref = [a["measured"]["peak_vram_alloc_gb"] for a in off
                    if vf.anchor_tokens(a, prefer_measured=False)
                    == cal["ref_tokens"]]
    peaks_below = [a["measured"]["peak_vram_alloc_gb"] for a in off
                   if (vf.anchor_tokens(a, prefer_measured=False) or 0)
                   < cal["ref_tokens"]]
    assert cal["ref_gb"] == max(peaks_at_ref)          # worst case, as elsewhere
    assert cal["floor_gb"] == (max(peaks_below) if peaks_below else 0.0)
    assert cal["ref_gb"] in peaks_at_ref               # an anchor, not a fit
    assert cal["ref_run"] and cal["ref_base"]
    assert cal["n_off_anchors"] == len(off)


def test_grad_ckpt_off_calibration_refuses_when_unmeasured():
    """No OFF anchor at the reference window -> Unmeasured, never a guess."""
    with pytest.raises(vf.Unmeasured):
        vf.grad_ckpt_off_calibration(facts={"anchors": []})


def test_global_slope_is_not_applicable_to_grad_ckpt_off(facts):
    """0.063 GB/1k came off a checkpointed-ON ladder whose rows had stopped
    growing (so it is not a valid ON slope either — see
    `test_the_flat_slice_slope_is_never_charged_for_a_window`). Applying it to
    an OFF shape predicts ~52.7 GB at 12288 tokens; the measurement there is an
    OOM on a 94.97 GiB card. This pins the gap so nobody wires the two
    together."""
    cal = vf.grad_ckpt_off_calibration(facts=facts)
    naive = cal["ref_gb"] + vf.MEASURED_SLOPE_GB_PER_1K * (12288 - 4096) / 1024.0
    assert naive < 95.0, "the naive slope must still look 'fits' — that is the trap"
    import autotune  # noqa: E402  (same directory, stdlib-only)
    assert autotune.grad_ckpt_off_vram_gb(1, 12288) > 95.0


# --- partial grad checkpointing is a separate group ---------------------------

def test_gc_flag_absent_reads_as_full():
    """Every run predating the partial-GC lever checkpointed everything. A
    third bucket for "unrecorded" would split months of anchors away from the
    shape every current bundle declares."""
    assert vf.gc_flag_class(None) == "full"
    assert vf.gc_flag_class("on") == "full"
    assert vf.gc_flag_class("1.0") == "full"
    assert vf.gc_flag_class("0.0") == "none"
    assert vf.gc_flag_class("off") == "none"
    assert vf.gc_flag_class("0.5") == "partial-0.5"
    assert vf.gc_flag_class("hybrid") == "partial-hybrid"


def test_a_partial_gc_cell_does_not_join_the_full_gc_group():
    """The confound this key exists for: one 9B/20480 shape measured 147 GB at
    fraction 0.0 and 29 GB at 1.0, both recording grad_checkpointing true."""
    full = vf.normalize_shape(**_q(grad_checkpointing_flag="on"))
    part = vf.normalize_shape(**_q(grad_checkpointing_flag="0.0"))
    assert vf._group_of(full) != vf._group_of(part)
    unflagged = vf.normalize_shape(**_q())
    assert vf._group_of(full) == vf._group_of(unflagged)


def test_the_27b_anchor_is_measured_not_extrapolated(facts):
    """The v10 run is the only 27B measurement we own and it lives only on B2
    (`harvest_vram.py --b2-sync`). Losing it puts a v14-class 27B arm back on
    an extrapolation — and extrapolation now refuses by default because the
    slope it used was 7x too shallow, measured 2026-08-29."""
    got = vf.estimate_peak_gb(facts=facts, base_slug="qwen36-27b",
                              quant_mode="bf16", grad_checkpointing=True,
                              ce_chunk_matmul="fp32", target_modules="all-linear",
                              lora_r=32, packing="off", fsdp="", world_size=1,
                              batch=1, max_seq=20480)
    assert got["n"] >= 1 and not got["extrapolated"]
    assert got["gb"] == 82.77
    assert "tuner-v10-qwen36-27b-dec" in got["runs"]
