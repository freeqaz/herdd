"""Regression pins for `mfu.py`.

The point of these tests is not that the code runs — it is that the arithmetic
still lands on the numbers
`docs/plans/witness/perf/TRAINING_THROUGHPUT_REVIEW_2026-08-06.md` §3 published, so
that a config edit or a refactor cannot silently move a denominator that four
other documents' ratios hang off. No GPU, no torch, no network.
"""
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mfu  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_MFU = os.path.join(_HERE, "mfu.py")
_G4_CONFIG = os.path.join(_HERE, "testfixtures", "gemma4-12b-text.config.json")


# ---------------------------------------------------------------------------
# The cost-model constants: which are measured, which are derived, and the
# boundary between them. These exist so a future edit cannot quietly collapse
# a seconds ratio into a FLOP count, in either direction.
# ---------------------------------------------------------------------------

#: The one profiled optimizer step every measured constant here comes from.
#: G4_ATTENTION_PROFILE_2026-08-07 (branch `t50-attn-profile`, bundle
#: `tools/witness/jobs/g4-attn-profile`, job
#: `20260807T031101-g4-attn-profile-12f0`, cell `a_sdpa_prof`).
#: gemma-4-12b-text, BATCH 1 x GRAD_ACCUM 32, MAX_SEQ 40960, bf16, grad-ckpt on,
#: ATTN_IMPL=sdpa / SDPA_BACKENDS=flashmeff, on one NVIDIA RTX PRO 6000
#: Blackwell **Server Edition** (sm_120, 188 SMs, torch 2.11.0+cu129).
PROFILE = dict(
    tw=8303.0, tokens=175_144, label_share=0.0508,
    step_s=166.65,                 # total self-CUDA, = 89.710 / 0.5383
    gemm_bucket_s=33.811,          # §1 bucket table: body/MLP/LoRA GEMMs
    attn_fwd_plus_recompute_s=21.700,   # §0, the 40 sliding head_dim-256 layers
    attn_bwd_s=52.615,                  # §0, same layers
    gemm_ceiling_tflops=419.2,     # §3, measured in-run on this box
)


def test_the_attention_backward_time_ratio_is_pinned_to_its_measurement():
    """`ATTENTION_BWD_TIME_RATIO = 4.85` is MEASURED, and this is the arithmetic
    that produced it (G4_ATTENTION_PROFILE §0):

        forward + grad-ckpt recompute   21.700 s  => one forward  10.850 s
        backward                        52.615 s  => 52.615/10.850 = 4.849

    Halving the fwd+recompute figure is the step that assumes the grad-ckpt
    recompute costs the same as the original forward — same kernel, same shapes,
    so it holds here, and it is the only inference in the chain.
    """
    one_fwd = PROFILE["attn_fwd_plus_recompute_s"] / 2
    assert one_fwd == pytest.approx(10.850, abs=0.001)
    assert PROFILE["attn_bwd_s"] / one_fwd == pytest.approx(
        mfu.ATTENTION_BWD_TIME_RATIO, abs=0.005)
    # and the time-equivalent multiplier built from it
    assert mfu.PASSES_ATTENTION_TIME == pytest.approx(6.85, abs=0.001)


def test_passes_attention_is_the_derived_matmul_count_not_the_measured_seconds():
    """**The load-bearing test.** `PASSES_ATTENTION` is 4.5 and must not become
    6.85.

    4.85 is a ratio of SECONDS. The attention backward executes 2.5x the
    forward's FLOPs (5 matmuls — S recomputed, dV, dP, dQ, dK — against the
    forward's 2) while taking 4.85x as long, so it runs at 2.5/4.85 = 52% of the
    forward's FLOP/s. That is cutlass's memory-efficient backward being
    inefficient, not the GPU doing more arithmetic.

    `flops_per_token` feeds an MFU numerator. Billing an efficiency as FLOPs
    would inflate it — see the sibling test for by how much.
    """
    assert mfu.PASSES_ATTENTION == 4.5
    assert mfu.PASSES_ATTENTION != mfu.PASSES_ATTENTION_TIME
    # 1 fwd + 1 grad-ckpt recompute + a backward worth 2.5 forwards in FLOPs
    assert mfu.PASSES_ATTENTION == pytest.approx(1 + 1 + 2.5)
    # the implied efficiency of the backward, which is what 4.85 really records
    assert 2.5 / mfu.ATTENTION_BWD_TIME_RATIO == pytest.approx(0.515, abs=0.005)


def test_using_the_time_ratio_as_a_flop_multiplier_inflates_mfu_by_15_percent():
    """Guard on the exact failure mode the constant's comment describes. Nobody
    should have to take "it would inflate the numerator" on trust.

    +15.0% against the shipped 4.5 at the corpus tw (it is +20% against the old
    4, and grows with tw because the attention term does).
    """
    s = mfu.gemma4_12b_text()
    honest = mfu.flops_per_token(s, mfu.TW_V9_GEMMA4)
    if_we_billed_time = mfu.flops_per_token(
        s, mfu.TW_V9_GEMMA4, passes_attention=mfu.PASSES_ATTENTION_TIME)
    assert if_we_billed_time.raw / honest.raw == pytest.approx(1.150, abs=0.005)
    # ...and an MFU is linear in the numerator, so it moves by the same factor.
    c = mfu.Ceiling(device="d", tflops=PROFILE["gemm_ceiling_tflops"])
    u = mfu.utilisation(s, tw=mfu.TW_V9_GEMMA4, tok_s=1333, device="d",
                        ceiling=c)
    assert u.mfu_raw == pytest.approx(0.293, abs=0.005)
    assert if_we_billed_time.raw * 1333 / 1e12 / c.tflops == pytest.approx(
        0.337, abs=0.005)


def test_passes_weights_3_is_the_only_value_the_profile_bucket_table_admits():
    """`PASSES_WEIGHTS = 3` was derived the same way the attention constant was,
    so it needs its own check rather than a shrug. It gets a decisive one.

    Divide this model's weight-GEMM FLOPs by the profile's measured `gemm`
    bucket (33.811 s of a 166.65 s step, 175,144 tokens at Tw 8,303) and compare
    against the 419.2 TFLOP/s dense-bf16 ceiling measured on the same box:

        2 -> 227.2 TFLOP/s   54% of ceiling   implausibly slow for pure GEMM
        3 -> 340.9 TFLOP/s   81%              where a well-fed GEMM stack sits
        4 -> 454.5 TFLOP/s  108%              ABOVE the measured peak: impossible

    So the profile does not merely fail to refute 3, it excludes both
    neighbours. Nothing about the weight term is left underived.
    """
    s = mfu.gemma4_12b_text()

    def implied_tflops(passes):
        per_tok = (passes * mfu.FLOPS_PER_MAC
                   * (s.body_params
                      + s.lm_head_params * PROFILE["label_share"]))
        return per_tok * PROFILE["tokens"] / PROFILE["gemm_bucket_s"] / 1e12

    ceiling = PROFILE["gemm_ceiling_tflops"]
    assert implied_tflops(2) == pytest.approx(227.2, abs=1.0)
    assert implied_tflops(3) == pytest.approx(340.9, abs=1.0)
    assert implied_tflops(4) == pytest.approx(454.5, abs=1.0)

    assert mfu.PASSES_WEIGHTS == 3
    assert implied_tflops(4) > ceiling          # excluded: faster than the card
    assert 0.75 < implied_tflops(3) / ceiling < 0.90
    assert implied_tflops(2) / ceiling < 0.60   # excluded: absurdly inefficient


def test_banding_speedup_timed_uses_the_measured_constant_and_beats_the_flop_ratio():
    """The measured 4.85 is not discarded — it lives here, where a ratio of
    seconds belongs. G4_ATTENTION_PROFILE §4 measured `g4_hybrid` at **1.51x**
    at the profiled shape. The FLOP ratio predicts 1.291x; re-weighting
    attention to the measured time cost predicts 1.420x.

    Still short — the residual is the non-matmul work inside the attention
    kernel, which this model bills at zero — but on the right side of the
    measurement instead of 1.9x short of it.
    """
    f = mfu.flops_per_token(mfu.gemma4_12b_text(), PROFILE["tw"],
                            label_share=PROFILE["label_share"])
    assert f.banding_speedup == pytest.approx(1.291, abs=0.01)
    assert f.banding_speedup_timed == pytest.approx(1.420, abs=0.01)
    assert f.banding_speedup < f.banding_speedup_timed < 1.51
    assert f.as_dict()["banding_speedup_timed"] == f.banding_speedup_timed


def test_banding_speedup_timed_is_invariant_to_the_flop_multiplier():
    """It re-weights from whatever multiplier built the `Flops`, so passing a
    non-default `passes_attention` must not change the TIME answer. Without the
    stored `passes_attention` field this would silently double-count."""
    s = mfu.gemma4_12b_text()
    a = mfu.flops_per_token(s, 8192, passes_attention=4.5)
    b = mfu.flops_per_token(s, 8192, passes_attention=4.0)
    assert a.banding_speedup != b.banding_speedup          # the FLOP ratio moves
    assert a.banding_speedup_timed == pytest.approx(b.banding_speedup_timed)


# ---------------------------------------------------------------------------
# The published numbers
# ---------------------------------------------------------------------------

def test_gemma4_body_params_match_the_review_doc():
    """§3: "decoder body 10.916 B params, embedding 1.007 B, total 11.92 B
    (⇒ 23.85 GB bf16, matching the bundle's measured 23.81 GB)"."""
    s = mfu.gemma4_12b_text()
    assert s.body_params / 1e9 == pytest.approx(10.916, abs=0.001)
    assert s.embedding_params / 1e9 == pytest.approx(1.007, abs=0.001)
    assert s.total_params / 1e9 == pytest.approx(11.92, abs=0.01)
    # tied embeddings: the lm_head is the embedding matrix, counted once.
    assert s.tied_embeddings


def test_gemma4_flops_per_token_is_92_2():
    """§3 published 89.1 GFLOP/token at tw = 7,461 with a 4x attention
    multiplier. The multiplier is 4.5 since 2026-08-07 (the backward is 5
    matmuls, not 4 — see `mfu.PASSES_ATTENTION`), so the figure is **92.17**.

    The body term is untouched at 65.49: only the O(T²) attention term moved,
    23.50 -> 26.40. §3 carries the correction in place.
    """
    f = mfu.flops_per_token(mfu.gemma4_12b_text(), mfu.TW_V9_GEMMA4)
    assert f.raw / 1e9 == pytest.approx(92.17, abs=0.05)
    assert f.body / 1e9 == pytest.approx(65.49, abs=0.05)      # unchanged
    assert (f.attn_dense + f.attn_sliding_executed) / 1e9 == pytest.approx(
        26.40, abs=0.05)                                        # was 23.50


def test_gemma4_the_wasted_dense_attention_share():
    """§3's "16.9 of the 89.1 GFLOP/token — 19% of all training work — is dense
    attention over key positions the architecture has already decided to
    ignore", re-derived at the corrected 4.5 multiplier: **18.98 of 92.17, or
    20.6%**. That split is the reason this module models sliding separately.

    Note this is the FLOP share. The measured TIME share of the same waste is
    ~39% (G4_ATTENTION_PROFILE §0) — a bigger number describing the same work,
    because attention runs far below GEMM efficiency. Do not treat the two as
    rival estimates of one quantity.
    """
    f = mfu.flops_per_token(mfu.gemma4_12b_text(), mfu.TW_V9_GEMMA4)
    assert f.wasted / 1e9 == pytest.approx(18.98, abs=0.05)
    assert f.wasted_share == pytest.approx(0.206, abs=0.005)


@pytest.mark.parametrize("tw,expect", [
    (4096, 1.127), (8192, 1.287), (16384, 1.577), (32768, 2.063),
    (mfu.TW_V9_GEMMA4, 1.259),
])
def test_g4_hybrid_banding_speedup_curve(tw, expect):
    """§4: "The win grows with row length (1.12× at 4k, 1.26× at 8k, 1.53× at
    16k, 1.98× at 32k)", and ~1.24× at the corpus's own tw. Five independent
    points is a much stronger pin on the model than the single total.

    Restated at the 4.5 attention multiplier: 1.127 / 1.287 / 1.577 / 2.063,
    and 1.259 at the corpus tw. Every point moved UP, because banding removes
    attention work and attention work now costs more.
    """
    f = mfu.flops_per_token(mfu.gemma4_12b_text(), tw)
    assert f.banding_speedup == pytest.approx(expect, abs=0.01)


def test_qwen35_9b_flops_diverges_from_the_doc_and_this_is_the_measured_value():
    """§3's table prints 48.6 GFLOP/token for qwen3.5-9B. The same cost model
    that reproduces gemma-4's 89.1 (and all five of its banding ratios) gives
    **43.7** here, 10.1% lower, and the doc shows no derivation for its figure.

    Cross-checked against the shipped weights: the decoder body is
    6,919,290,880 projection+MLP parameters, read off Qwen3.5-9B's safetensors
    headers (6,919,561,728 including norms, which carry no GEMM work).

    Consequences, if 43.7 is right:
      - achieved 134 TFLOP/s at the measured 3,049 tok/s, not 148;
      - MFU ~51% against the 261.9 TFLOP/s weighted ceiling, not 57%
        (that ceiling is the Max-Q borrow, retracted for the gemma-4 lane by
        G4_ATTENTION_PROFILE §3; qwen3.5-9B has no measured ceiling of its own
        yet, so this row is still quoted against the borrow and says so);
      - the 12B-vs-9B decomposition is 2.04× (FLOPs/token) × 1.12× (achieved
        TFLOP/s), not 1.83× × 1.25×. The product, 2.29×, is unchanged — it is
        an identity, so it cannot arbitrate between the two.

    43.71 here until 2026-08-07; **43.95** at the corrected 4.5 attention
    multiplier. The move is small because qwen3.5-9B has only 8 attention layers
    and no sliding ones, so almost all of its cost is weight GEMM — the same
    reason the gemma-4 gap is a gemma-4 problem.

    This test pins OUR value. Do not "fix" it to 48.6 without a derivation.
    """
    f = mfu.flops_per_token(mfu.qwen35_9b_text(), mfu.TW_V9_QWEN35)
    assert f.raw / 1e9 == pytest.approx(43.95, abs=0.05)
    assert mfu.qwen35_9b_text().body_params == 6_919_290_880
    # no sliding layers -> nothing to band, raw == required
    assert f.raw == f.required and f.wasted == 0


def test_qwen35_linear_layers_carry_no_quadratic_term():
    """Gated DeltaNet is O(T). Its cost is entirely in the projections, so the
    attention term must come only from the 8 full-attention layers."""
    s = mfu.qwen35_9b_text()
    lin = [b for b in s.blocks if b.kind == "linear"]
    assert lin and sum(b.count for b in lin) == 24
    a = mfu.flops_per_token(s, 4096).attn_dense
    b = mfu.flops_per_token(s, 8192).attn_dense
    assert b == pytest.approx(2 * a)          # strictly linear in tw, 8 layers


# ---------------------------------------------------------------------------
# tw, and why it is not the median
# ---------------------------------------------------------------------------

def test_time_weighted_length_is_the_input_not_the_median():
    """§1's table: median 3,811, time-weighted mean 7,461. Quoting the median
    understates the attention bill by ~12.9 GFLOP/token — 14% of the total.
    (~11.5 and 11% before the attention multiplier was corrected to 4.5; the
    penalty for quoting a median grows with the cost of attention.)"""
    s = mfu.gemma4_12b_text()
    at_median = mfu.flops_per_token(s, 3811).raw
    at_tw = mfu.flops_per_token(s, 7461).raw
    assert (at_tw - at_median) / 1e9 == pytest.approx(12.92, abs=0.1)
    assert at_tw / at_median == pytest.approx(1.163, abs=0.005)


def test_tw_must_be_positive():
    with pytest.raises(ValueError):
        mfu.flops_per_token(mfu.gemma4_12b_text(), 0)


def test_label_share_moves_only_the_lm_head_term():
    s = mfu.gemma4_12b_text()
    a = mfu.flops_per_token(s, 7461, label_share=0.0448)
    b = mfu.flops_per_token(s, 7461, label_share=1.0)
    assert a.body == b.body and a.attn_dense == b.attn_dense
    assert b.lm_head / a.lm_head == pytest.approx(1 / 0.0448, rel=1e-9)


# ---------------------------------------------------------------------------
# config.json parsing
# ---------------------------------------------------------------------------

def test_the_checked_in_gemma4_config_reproduces_the_hardcoded_shape():
    """The hardcoded `gemma4_12b_text()` is a convenience; the config file is
    the truth. If they ever disagree, the convenience is wrong."""
    a = mfu.load_config(_G4_CONFIG, name="gemma-4-12b-text")
    b = mfu.gemma4_12b_text()
    assert a.body_params == b.body_params
    assert a.total_params == b.total_params
    assert sorted((x.kind, x.count, x.head_dim, x.window) for x in a.blocks) == \
        sorted((x.kind, x.count, x.head_dim, x.window) for x in b.blocks)


def test_config_with_an_unknown_layer_type_refuses_rather_than_guessing():
    cfg = {"num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
           "vocab_size": 100, "num_attention_heads": 4, "head_dim": 16,
           "layer_types": ["full_attention", "moe_attention"]}
    with pytest.raises(ValueError, match="moe_attention"):
        mfu.shape_from_config(cfg, name="x")


def test_config_layer_type_count_mismatch_refuses():
    cfg = {"num_hidden_layers": 4, "hidden_size": 64, "intermediate_size": 128,
           "vocab_size": 100, "num_attention_heads": 4, "head_dim": 16,
           "layer_types": ["full_attention"]}
    with pytest.raises(ValueError, match="layer_types"):
        mfu.shape_from_config(cfg, name="x")


def test_multimodal_wrapper_is_unwrapped_to_the_text_decoder():
    inner = {"num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
             "vocab_size": 100, "num_attention_heads": 4, "head_dim": 16}
    s = mfu.shape_from_config({"text_config": inner, "vision_config": {"depth": 27},
                               "tie_word_embeddings": False}, name="wrapped")
    assert s.n_layers == 2 and s.hidden == 64


# ---------------------------------------------------------------------------
# Denominator discipline
# ---------------------------------------------------------------------------

def test_a_ceiling_without_a_device_is_refused():
    """gemm_ceiling.py: "A TFLOP/s figure with no device attached is not
    quotable." Enforced here as a constructor error, not a comment."""
    with pytest.raises(mfu.DenominatorError):
        mfu.Ceiling(device="", tflops=261.9)
    with pytest.raises(mfu.DenominatorError):
        mfu.Ceiling(device="   ", tflops=261.9)


def test_a_nonpositive_ceiling_is_refused():
    with pytest.raises(mfu.DenominatorError):
        mfu.Ceiling(device="RTX 3090", tflops=0)


def test_utilisation_without_a_run_device_is_refused():
    c = mfu.Ceiling(device="RTX 3090", tflops=79.4)
    with pytest.raises(mfu.DenominatorError):
        mfu.utilisation(mfu.gemma4_12b_text(), tw=7461, tok_s=1000,
                        device="", ceiling=c)


def test_a_borrowed_ceiling_is_marked_provisional_and_says_so():
    """The live 2026-08-06 case: the only measured RTX PRO 6000 ceiling is the
    Max-Q part; v9-gemma4-dec runs on the Server Edition. Silently substituting
    one for the other is exactly the error this module exists to stop."""
    c = mfu.Ceiling(device="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
                    tflops=261.9, source="V7_PERF_LEVERS §2, Qwen shapes")
    u = mfu.utilisation(mfu.gemma4_12b_text(), tw=mfu.TW_V9_GEMMA4, tok_s=1430,
                        device="NVIDIA RTX PRO 6000 Blackwell Server Edition",
                        ceiling=c)
    assert u.provisional
    assert "Max-Q" in u.note and "Server Edition" in u.note
    assert u.as_dict()["ceiling_device"] == c.device


def test_a_matching_device_is_not_provisional():
    dev = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
    c = mfu.Ceiling(device=dev, tflops=261.9)
    u = mfu.utilisation(mfu.gemma4_12b_text(), tw=7461, tok_s=1430,
                        device="  " + dev.upper() + " ", ceiling=c)
    assert not u.provisional and u.note == ""


@pytest.mark.parametrize("ceiling,raw,req,what", [
    (419.2, 0.293, 0.233, "max across shapes, measured on the Server Edition"),
    (358.8, 0.342, 0.272, "FLOP-weighted over gemma-4's own shapes"),
    (261.9, 0.469, 0.373, "the Max-Q borrow — RETRACTED, kept only as the "
                          "provenance of the '45%' that used to be quoted"),
])
def test_mfu_depends_on_which_denominator_and_all_three_are_pinned(
        ceiling, raw, req, what):
    """§3 published "45% MFU on raw work; on useful work it is 37%" at 1,333
    tok/s. Both halves of that have moved, for two independent reasons, and the
    test carries all three denominators so nobody has to guess which one a
    quoted MFU used.

    The numerator moved: 89.23 -> 92.17 GFLOP/token at the corrected 4.5
    attention multiplier, so achieved goes 119 -> 122.9 TFLOP/s.

    The denominator is disputed and the answers are NOT rivals — 261.9 is a
    borrow from the Max-Q part at Qwen's shapes and is retracted for this lane;
    419.2 is the max-across-shapes peak measured in-run on the Server Edition;
    358.8 is the FLOP-weighted figure over gemma-4's own shapes. Every one of
    these divisions is arithmetically correct. Quoting one without naming it is
    the bug.
    """
    c = mfu.Ceiling(device="d", tflops=ceiling, source=what)
    u = mfu.utilisation(mfu.gemma4_12b_text(), tw=mfu.TW_V9_GEMMA4, tok_s=1333,
                        device="d", ceiling=c)
    assert u.achieved_tflops == pytest.approx(122.9, abs=0.5)   # was 119
    assert u.mfu_raw == pytest.approx(raw, abs=0.005)
    assert u.mfu_required == pytest.approx(req, abs=0.005)
    assert u.mfu_required < u.mfu_raw
    # whichever denominator is used, it travels with the figure
    assert u.as_dict()["ceiling_tflops"] == ceiling
    assert u.as_dict()["ceiling_source"] == what


# ---------------------------------------------------------------------------
# Weighted ceiling
# ---------------------------------------------------------------------------

def test_harmonic_weighting_is_harmonic_not_arithmetic():
    """Averaging RATES over fixed work: half the FLOPs at 100 and half at 300
    is 150 effective, not 200. The arithmetic mean overstates by 33%."""
    got = mfu.harmonic_weighted({"a": 0.5, "b": 0.5}, {"a": 100.0, "b": 300.0})
    assert got == pytest.approx(150.0)


def test_a_partial_weighting_is_refused():
    with pytest.raises(mfu.DenominatorError, match="mlp_down"):
        mfu.harmonic_weighted({"mlp_up": 0.5, "mlp_down": 0.5},
                              {"mlp_up": 100.0})


def test_mac_mix_sums_to_one_and_mlp_dominates():
    m = mfu.mac_mix(mfu.gemma4_12b_text())
    assert sum(m.values()) == pytest.approx(1.0)
    assert m["mlp_up"] > m["mlp_down"] > m["attn_proj"] > m["lm_head"]
    # gate+up is exactly twice down for a 3-matrix SwiGLU MLP
    assert m["mlp_up"] == pytest.approx(2 * m["mlp_down"])


def test_ceiling_from_gemm_ceiling_json_weights_over_the_model_mix():
    blob = {"device": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
            "capability": "sm_120", "torch": "2.11.0+cu129", "cuda": "12.9",
            "shapes": [{"m": 12288, "k": 3840, "n": 3840, "tflops": 283.3},
                       {"m": 12288, "k": 3840, "n": 15360, "tflops": 274.1},
                       {"m": 12288, "k": 15360, "n": 3840, "tflops": 231.6}]}
    s = mfu.gemma4_12b_text()
    c = mfu.Ceiling.from_gemm_ceiling_json(blob, weights=mfu.mac_mix(s))
    # strictly inside [min, max] and below the headline — the whole reason the
    # headline is the wrong number to divide by.
    assert 231.6 < c.tflops < 283.3
    assert c.tflops < blob["shapes"][0]["tflops"]
    assert "FLOP-weighted" in c.source and "sm_120" in c.source
    assert c.device == blob["device"]


def test_a_gemm_probe_record_reads_as_a_ceiling_and_keeps_its_shape_basis():
    """`tools/vast/gemm_probe.py` emits this schema as a superset so the boot
    instrument and the bench instrument need no format branch. The one thing
    that must survive is `shape_basis`: a boot probe measures a GENERIC shape
    set (no base is known at boot), and an MFU divided by it is an
    approximation that has to say so."""
    blob = {"probe_version": 1, "shape_basis": "generic", "status": "ok",
            "device": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "capability": "sm_120", "torch": "2.11.0+cu129", "cuda": "12.9",
            "power_limit_w": 600,
            "shapes": [{"m": 8192, "k": 4096, "n": 4096, "tflops": 269.4},
                       {"m": 8192, "k": 4096, "n": 16384, "tflops": 268.2},
                       {"m": 8192, "k": 16384, "n": 4096, "tflops": 229.1}]}
    s = mfu.gemma4_12b_text()
    c = mfu.Ceiling.from_gemm_ceiling_json(blob, weights=mfu.mac_mix(s))
    assert c.shape_basis == "generic"
    assert "gemm_probe.py" in c.source and "generic shapes" in c.source
    assert 229.1 < c.tflops < 269.4


def test_generic_shapes_are_flagged_but_are_NOT_provisional():
    """Two different defects, and conflating them would be wrong in both
    directions. `provisional` means the ceiling belongs to a DIFFERENT DEVICE —
    it could be off by 2x. A generic shape basis means the right device measured
    at approximate within-class K/N — a few percent, and the note names the
    exact command that removes it."""
    dev = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    c = mfu.Ceiling(device=dev, tflops=261.9, shape_basis="generic")
    u = mfu.utilisation(mfu.gemma4_12b_text(), tw=mfu.TW_V9_GEMMA4,
                        tok_s=1430, device=dev, ceiling=c)
    assert u.provisional is False
    assert "APPROXIMATE SHAPES" in u.note and "--gemm-cmd" in u.note
    assert u.as_dict()["ceiling_shape_basis"] == "generic"


def test_a_model_shape_ceiling_on_the_right_device_carries_no_caveat_at_all():
    dev = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    u = mfu.utilisation(mfu.gemma4_12b_text(), tw=mfu.TW_V9_GEMMA4, tok_s=1430,
                        device=dev, ceiling=mfu.Ceiling(device=dev, tflops=261.9))
    assert u.note == "" and not u.provisional


def test_both_caveats_can_apply_at_once_and_both_are_stated():
    c = mfu.Ceiling(device="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation "
                           "Edition", tflops=261.9, shape_basis="generic")
    u = mfu.utilisation(mfu.gemma4_12b_text(), tw=mfu.TW_V9_GEMMA4, tok_s=1430,
                        device="NVIDIA RTX PRO 6000 Blackwell Server Edition",
                        ceiling=c)
    assert u.provisional
    assert "PROVISIONAL" in u.note and "APPROXIMATE SHAPES" in u.note


def test_gemm_ceiling_json_without_a_device_is_refused():
    blob = {"device": "", "shapes": [{"m": 1, "k": 2, "n": 2, "tflops": 10.0}]}
    with pytest.raises(mfu.DenominatorError):
        mfu.Ceiling.from_gemm_ceiling_json(blob)


def test_gemm_shapes_are_the_models_own_not_a_square():
    s = mfu.gemma4_12b_text()
    assert mfu.gemm_shapes(s) == [(12288, 3840, 3840), (12288, 3840, 15360),
                                  (12288, 15360, 3840)]
    assert [mfu.classify_gemm(k, n) for _, k, n in mfu.gemm_shapes(s)] == \
        ["attn_proj", "mlp_up", "mlp_down"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run(*argv):
    return subprocess.run([sys.executable, _MFU, *argv],
                          capture_output=True, text=True, timeout=120)


def test_cli_json_shape():
    r = _run("--model", "gemma-4-12b-text", "--tok-s", "1430",
             "--device", "NVIDIA RTX PRO 6000 Blackwell Server Edition",
             "--ceiling-tflops", "261.9",
             "--ceiling-device", "NVIDIA RTX PRO 6000 Blackwell Max-Q "
                                 "Workstation Edition",
             "--json")
    assert r.returncode == 0, r.stderr
    blob = json.loads(r.stdout)
    assert blob["flops"]["raw"] / 1e9 == pytest.approx(92.17, abs=0.05)
    assert blob["flops"]["banding_speedup_timed"] > blob["flops"][
        "banding_speedup"]
    assert blob["utilisation"]["provisional"] is True
    assert blob["utilisation"]["mfu_raw"] > blob["utilisation"]["mfu_required"]


def test_cli_refuses_a_ceiling_with_no_device():
    r = _run("--model", "gemma-4-12b-text", "--tok-s", "1430",
             "--ceiling-tflops", "261.9")
    assert r.returncode != 0
    assert "not quotable" in (r.stderr + r.stdout)


def test_cli_tok_s_without_a_ceiling_reports_numerator_only_and_warns():
    r = _run("--model", "gemma-4-12b-text", "--tok-s", "1430", "--json")
    assert r.returncode == 0, r.stderr
    assert "utilisation" not in json.loads(r.stdout)
    assert "gemm_ceiling.py" in r.stderr


def test_cli_gemm_cmd_prints_a_runnable_invocation():
    r = _run("--model", "gemma-4-12b-text", "--gemm-cmd")
    assert r.returncode == 0, r.stderr
    assert "--shape 12288x3840x15360" in r.stdout
    assert "gemm_ceiling.py" in r.stdout


def test_cli_config_requires_tw():
    r = _run("--config", _G4_CONFIG)
    assert r.returncode != 0
    assert "--tw" in r.stderr
