"""Pins for the vLLM width calculator against banked boot logs.

Every constant here is a line printed by a real engine boot in
`<upstream-bench>/archive/runs/2026-08-22-mergeddemoa-think-probe/logs/` and
`.../2026-08-21-arm3-q36-cont8k/gen.log`; see
`docs/plans/throughput/WIDTH_SIZING_LEVER_2026-08-22.md` §3.
"""

import pytest

import width_calc as wc

# Qwen3.6-27B merged (MERGEDDEMOA), bf16, vLLM 0.27.1.post1+fork.gfb8e9ed57, H200 NVL.
H200_53K = wc.BootPoint(70.64 * wc.GIB, 53_248, 19.95)
H200_20K = wc.BootPoint(70.64 * wc.GIB, 20_480, 44.73)
# Same model, DIFFERENT card (RTX PRO 6000 WS) — the transfer check.
PRO6000_28K = wc.BootPoint(30.64 * wc.GIB, 28_672, 14.88)


def test_parse_boot_log_pulls_both_lines():
    text = (
        "INFO [gpu_worker.py:563] Available KV cache memory: 70.64 GiB\n"
        "INFO [kv_cache_utils.py:2235] GPU KV cache size: 1,062,081 tokens\n"
        "INFO [kv_cache_utils.py:2236] Maximum concurrency for 53,248 tokens "
        "per request: 19.95x\n"
    )
    pt = wc.parse_boot_log(text)
    assert pt.max_model_len == 53_248
    assert pt.max_concurrency == 19.95
    assert pt.kv_avail_bytes == pytest.approx(70.64 * wc.GIB)


def test_parse_boot_log_refuses_a_log_without_the_lines():
    with pytest.raises(ValueError, match="Available KV cache memory"):
        wc.parse_boot_log("INFO nothing useful here\n")


def test_calibration_needs_two_distinct_windows():
    with pytest.raises(ValueError, match="vary the window"):
        wc.calibrate([H200_53K, H200_53K])


def test_two_h200_boots_recover_the_model_constants():
    """The 53k and 20k boots on one card solve for KV/token and state/seq."""
    cal = wc.calibrate([H200_53K, H200_20K])
    assert cal.kv_bytes_per_token / wc.KIB == pytest.approx(62.8, abs=0.3)
    assert cal.state_bytes_per_seq / wc.GIB == pytest.approx(0.353, abs=0.01)


def test_calibration_transfers_to_a_different_card():
    """The constants are the MODEL's, so an H200 fit predicts the PRO 6000 boot.

    This is the load-bearing claim: it means one calibration sizes every card,
    and a per-card boot is a check rather than a prerequisite.
    """
    cal = wc.calibrate([H200_53K, H200_20K])
    predicted = cal.concurrency(PRO6000_28K.kv_avail_bytes, PRO6000_28K.max_model_len)
    assert predicted == pytest.approx(PRO6000_28K.max_concurrency, rel=0.02)


def test_conservative_width_reproduces_the_pinned_16_at_k1():
    """At k=1 over a 53k window the H200 really does only afford ~16."""
    cal = wc.calibrate([H200_53K, H200_20K])
    assert wc.width_from_length(70.64 * wc.GIB, 53_248, cal) == 16


def test_rescale_from_the_measured_forcethink_occupancy():
    """64 running at 40.2% peak KV -> 128 at the 0.80 target, on a graph step."""
    assert wc.width_from_observed(64, 0.402) == 128
    assert wc.width_from_observed(64, 0.402, target_util=0.85) == 136


def test_rescale_refuses_an_impossible_occupancy():
    with pytest.raises(ValueError):
        wc.width_from_observed(64, 0.0)
    with pytest.raises(ValueError):
        wc.width_from_observed(64, 1.4)


def test_capture_rounding_lands_on_a_graph_step():
    assert wc.round_to_capture(127) == 128
    assert wc.round_to_capture(64) == 64
    assert wc.round_to_capture(5) == 5  # below the step, keep the exact width
    assert wc.round_to_capture(0.4) == 1


def test_cli_rescale(capsys):
    assert wc.main(["rescale", "--width-now", "64", "--kv-util-now", "0.402"]) == 0
    assert "--max-num-seqs 128" in capsys.readouterr().out


# ── the 9B, calibrated 2026-08-23 on a local RTX 3090 (vLLM 0.27.2.dev87) ──────
# <upstream-bench>/archive/runs/2026-08-23-width-optimization-legs/gates/A_tp1_*
NINEB_4K = wc.BootPoint(0.96 * wc.GIB, 4_096, 4.21)
NINEB_8K = wc.BootPoint(1.01 * wc.GIB, 8_192, 2.82)
NINEB_12K = wc.BootPoint(0.97 * wc.GIB, 12_288, 2.00)
NINEB_20K = wc.BootPoint(1.01 * wc.GIB, 20_480, 1.38)   # held out


def test_nine_b_constants_match_its_config_json():
    """8 full-attn layers x 2 x 4 kv_heads x 256 head_dim x 2 B = 32.00 KiB/token."""
    cal = wc.calibrate([NINEB_4K, NINEB_8K, NINEB_12K])
    assert cal.kv_bytes_per_token / wc.KIB == pytest.approx(32.0, rel=0.05)
    assert cal.state_bytes_per_seq / (1024**2) == pytest.approx(102, abs=8)


def test_nine_b_calibration_predicts_a_held_out_window():
    """Fit on 4k/8k/12k, predict the 20,480 boot the fit never saw."""
    cal = wc.calibrate([NINEB_4K, NINEB_8K, NINEB_12K])
    pred = cal.concurrency(NINEB_20K.kv_avail_bytes, NINEB_20K.max_model_len)
    assert pred == pytest.approx(NINEB_20K.max_concurrency, rel=0.03)


def test_per_seq_is_robust_to_the_pools_own_boot_noise():
    """Two identical-config boots gave 0.89 and 0.97 GiB of pool — 9% apart —
    but per_seq agrees to 0.3%, because max_concurrency moves with it."""
    a = wc.BootPoint(0.97 * wc.GIB, 12_288, 2.00).per_seq_bytes
    b = wc.BootPoint(0.89 * wc.GIB, 12_288, 1.83).per_seq_bytes
    assert a == pytest.approx(b, rel=0.01)


def test_block_ceiling_matches_the_engines_own_refusals():
    """vLLM refused width 64 saying '63 blocks' and width 256 saying '58'."""
    cal = wc.calibrate([NINEB_4K, NINEB_8K, NINEB_12K])
    # gates/B_tp2_len12288_w64.json: kv_avail 1.03 GiB -> engine said 63
    assert wc.block_ceiling(1.03 * wc.GIB, 528, cal) == pytest.approx(63, abs=2)
    # gates/B_tp2_len12288_w256_cap64.json: kv_avail 0.93 GiB -> engine said 58
    assert wc.block_ceiling(0.93 * wc.GIB, 528, cal) == pytest.approx(58, abs=2)


def test_block_ceiling_under_reads_which_is_the_safe_direction():
    cal = wc.calibrate([NINEB_4K, NINEB_8K, NINEB_12K])
    assert wc.block_ceiling(1.03 * wc.GIB, 528, cal) <= 63
    assert wc.block_ceiling(0.93 * wc.GIB, 528, cal) <= 58


def test_the_block_gate_is_only_reachable_by_a_HAND_PICKED_width():
    """A width from the formula never trips the boot gate; a hand-picked one does.

    Provable from the constants: the memory bound is <= the block ceiling exactly
    when `state_per_seq >= target_util * attn_block_size * kv_per_token`. Both
    models measured clear that by a wide margin (9B: 102 MiB vs 13.6; 27B:
    362 MiB vs 38.5), so `block_ceiling` is a VALIDITY CHECK on an operator's
    number, not a term that ever binds ours. That is why the 3090 refusals were
    all at widths someone chose (64, 128, 256, and vLLM's own default) and never
    at the 16 the formula gives.
    """
    for cal, blk in ((wc.calibrate([NINEB_4K, NINEB_8K, NINEB_12K]), 528),
                     (wc.calibrate([H200_53K, H200_20K]), 784)):
        assert cal.state_bytes_per_seq > (
            wc.DEFAULT_TARGET_UTIL * blk * cal.kv_bytes_per_token)
        for kv_gib in (0.97, 8.0, 30.64, 70.64):
            for L in (2_048, 8_192, 20_480, 53_248):
                kv = kv_gib * wc.GIB
                assert (wc.width_from_length(kv, L, cal)
                        == wc.width_from_length(kv, L, cal, attn_block_size=blk))


def test_the_engine_refused_exactly_the_widths_above_its_ceiling():
    """Boot-gate arithmetic against the four banked 3090 refusals."""
    cal = wc.calibrate([NINEB_4K, NINEB_8K, NINEB_12K])
    assert wc.block_ceiling(1.03 * wc.GIB, 528, cal) < 64      # refused
    assert wc.block_ceiling(0.93 * wc.GIB, 528, cal) < 256     # refused
    assert wc.block_ceiling(0.97 * wc.GIB, 528, cal) >= 32     # width 32 booted


# ── what `state_per_seq` is a property of ─────────────────────────────────────
# Two campaigns fitted the 9B's state term ~2x apart and the gap was read as a
# minting bug. It is neither a bug nor card variance: vLLM sets
# `mamba_cache_mode='align'` for Qwen3_5 when prefix caching is on, and
# `MambaSpec.max_memory_usage_bytes` charges `page_size x (2 + spec)` there
# against `page_size x 1` in `none` mode. The cells below are the A/B.
#
# <upstream-bench>/archive/runs/2026-08-23-width-optimization-legs/gates/C_apc*
# — one RTX 3090, one day, one engine, prefix caching the only variable.
APC_OFF_4K = wc.BootPoint(0.66 * wc.GIB, 4_096, 3.64)
APC_OFF_12K = wc.BootPoint(0.94 * wc.GIB, 12_288, 2.15)
APC_OFF_20K = wc.BootPoint(1.01 * wc.GIB, 20_480, 1.48)
APC_OFF_3090 = [APC_OFF_4K, APC_OFF_12K, APC_OFF_20K]
# The A_tp1_* cells are the same card at vLLM's default (APC on) and are the
# arm STATE_PER_SEQ_GIB ships from.
APC_ON_3090 = [NINEB_4K, NINEB_8K, NINEB_12K, NINEB_20K]

# GPU5090_SIZING_2026-08-29 §1 — RTX 5090, one campaign, prefix caching OFF on
# every boot (its §8 says so). fp8 is vLLM's load-time `--quantization fp8`;
# bf16 is the same base unquantized. Same card, same day, same engine build.
FP8_8K = wc.BootPoint(14.72 * wc.GIB, 8_192, 48.05)
FP8_32K = wc.BootPoint(14.72 * wc.GIB, 32_768, 13.83)
BF16_8K = wc.BootPoint(7.94 * wc.GIB, 8_192, 25.95)
BF16_32K = wc.BootPoint(7.95 * wc.GIB, 32_768, 7.47)

MIB = 1024**2


def test_weight_dtype_does_not_move_the_state_term():
    """The confound that was proposed for the ~2x gap, and it is not the cause.

    fp8 weights free 6.8 GiB of KV pool, so the two arms boot with pools 85%
    apart -- and fit the same per-sequence state to under 1%.
    """
    fp8 = wc.calibrate([FP8_8K, FP8_32K])
    bf16 = wc.calibrate([BF16_8K, BF16_32K])
    assert bf16.state_bytes_per_seq == pytest.approx(
        fp8.state_bytes_per_seq, rel=0.02)
    assert bf16.kv_bytes_per_token == pytest.approx(
        fp8.kv_bytes_per_token, rel=0.01)


def test_prefix_caching_is_what_moves_the_state_term():
    """`align` mode costs 1.8x the per-sequence state of `none`, on one card
    with prefix caching as the only variable -- while the per-token term, which
    architecture forbids moving, does not move."""
    on = wc.calibrate(APC_ON_3090)
    off = wc.calibrate(APC_OFF_3090)
    ratio = on.state_bytes_per_seq / off.state_bytes_per_seq
    assert 1.7 < ratio < 2.0, ratio
    assert on.kv_bytes_per_token == pytest.approx(
        off.kv_bytes_per_token, rel=0.01)


def test_the_none_mode_fits_agree_across_two_cards():
    """Once the modes are separated the constants DO transfer: the 3090's
    APC-off fit and the 5090's APC-off fits are one number, and the '55 vs 102
    MiB' gap was never a cross-card disagreement."""
    off_3090 = wc.calibrate(APC_OFF_3090).state_bytes_per_seq / MIB
    off_5090 = wc.calibrate([FP8_8K, FP8_32K]).state_bytes_per_seq / MIB
    assert off_3090 == pytest.approx(59.2, abs=1.0)
    assert off_5090 == pytest.approx(55.0, abs=1.0)
    assert off_3090 == pytest.approx(off_5090, rel=0.10)


def test_adopting_the_none_mode_state_on_an_apc_box_over_widens():
    """Why the 5090 doc's 'just re-derive both state_per_seq values' must not
    be applied to an APC-on bundle: the same pool and window resolve a wider
    width, and on a hybrid a too-high width is a boot ValueError rather than a
    degradation."""
    align = wc.calibrate(APC_ON_3090)
    none = wc.Calibration(align.kv_bytes_per_token,
                          wc.calibrate(APC_OFF_3090).state_bytes_per_seq)
    # An APC-on box really affords `align`'s concurrency; `none` claims more.
    for kv_gib, L in ((8.0, 20_480), (14.7, 8_192), (60.0, 32_768)):
        kv = kv_gib * wc.GIB
        assert none.concurrency(kv, L) > align.concurrency(kv, L)
    # And it reaches the width knob, at the pool sizes this bundle rents.
    assert (wc.width_from_length(60.0 * wc.GIB, 8_192, none)
            > wc.width_from_length(60.0 * wc.GIB, 8_192, align))
