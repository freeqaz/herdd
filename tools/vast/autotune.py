#!/usr/bin/env python3
"""autotune — pure launch-shape planning for training jobs (AUTOTUNE_DESIGN.md).

The CANONICAL implementation of the static autotune formulas: given hardware
facts (GPU count, per-card VRAM, CPU cores) and the authored optimization shape
(BATCH, GRAD_ACCUM), decide the launch shape — DDP nproc, rebalanced grad-accum
that HOLDS the authored effective batch, dataloader workers, quant-by-VRAM.

Stdlib-only and import-light on purpose: this file is meant to ship in the jobd
bootstrap bundle (phase 2) and be callable box-side with zero deps. Job-bundle
entrypoints that cannot import it (a bundle ships only its own folder) mirror
these formulas in a sourceable `launch_plan.sh`; `test_autotune.py` cross-checks
that bash mirror against this module over a grid, so the two cannot drift
silently.

TWO MODES (the reproducibility boundary — see AUTOTUNE_DESIGN.md §6):

  * ``pinned``    — paired/of-record runs. NEVER auto-scales: single process,
                    grad-accum verbatim, workers explicit-or-historical-default.
                    Grad-accum + padding + per-token mean-reduction is NOT
                    invariant to micro-batch composition or world size, so a
                    paired comparison (e.g. v2<->v2.1, eff-batch 32 at 1x32)
                    must freeze batch, composition, AND world_size.
  * ``autotune``  — fresh/bakeoff runs. Uses every assigned card via DDP,
                    divides grad-accum by world size (refusing loudly when it
                    does not divide), auto-sizes dataloader workers.

Absent/empty mode resolves to ``pinned`` (fail-closed: never silently perturb
an existing caller); authoring surfaces opt fresh work into ``autotune``.

CLI (emits shell assignments for `eval`):

    python3 autotune.py plan --mode autotune --gpus 2 --batch 1 \
        --grad-accum 32 --cpu-cores 32
    # PLAN_MODE=autotune PLAN_NPROC=2 PLAN_GRAD_ACCUM=16 ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

VALID_MODES = ("pinned", "autotune")

# Historical trainer-side defaults (run.sh / train_proposer_lora.py): pinned
# mode must reproduce these byte-for-byte when the env leaves them unset.
HISTORICAL_NUM_WORKERS = 2

# Per-card quant policy (matrix.py owner principle, 2026-07-12): 4-bit exists
# ONLY to fit VRAM-constrained cards; on big cards it pays a per-matmul
# dequantization tax that bf16/8-bit avoid.
#
# UPDATE 2026-08-04 — TRAINING PRECISION DEFAULT IS bf16; the small-card table
# below is now OPT-IN ONLY (`allow_quantized=True`). The VRAM-tiered defaults
# were authored when training ran on the local 2x3090s, where 24 GB per card
# made quantization the only way to fit a 7B-class QLoRA. Under the 2026-07-30
# "rent the big box" posture that constraint no longer binds: runs are sized to
# rented large-VRAM cards. Quantizing afterwards for local INFERENCE is cheap
# and lossy only where it does not matter; quantizing during TRAINING silently
# changes the recipe. This extends the 2026-07-30 opt-in ruling ("the old
# implicit 4-bit default silently trained a config nobody chose") from *no
# implicit quantization* to *an explicit bf16 choice*.
QUANT_4BIT_MAX_GB = 32     # opt-in table: <= this per card -> 4bit
QUANT_BF16_MIN_GB = 48     # opt-in table: >= this -> bf16; the gap -> 8bit

# Resumable-spot runs prefer bf16 wherever a 7B-class QLoRA fits: bnb 4-bit/8-bit
# CANNOT reliably reload a paged-optimizer state on --resume (crashes at step 0,
# bitsandbytes pythonInterface.cpp 'invalid argument'; memory
# bnb-4bit-qlora-resume-crash), so a mid-run spot reclaim of a quantized run
# needs the trainer's discard-optimizer fallback. bf16 sidesteps bnb entirely.
# Below this floor even a bf16 7B QLoRA (seq 4096, grad-ckpt on) won't fit, so a
# VRAM-constrained resumable card still gets 4-bit (its reclaim-resume then
# leans on the trainer fallback). 7B-class assumption — matches the design's
# implicit repair-lifter sizing (AUTOTUNE_DESIGN.md §4).
RESUMABLE_BF16_MIN_GB = 24

WORKERS_FLOOR = 2          # below this the grad-accum loop input-starves
WORKERS_CAP = 8            # diminishing returns + host-RAM pressure above

# Gradient-checkpointing fit rule (AUTOTUNE_DESIGN.md §4, phase 1). Recompute is
# numerically identical to storing activations (only VRAM-for-throughput), so
# grad-ckpt is a THROUGHPUT knob: keep it OFF (store activations, no recompute
# serialization / ~2x forward FLOPs) whenever the activations fit; turn it ON
# only to make a run fit.
#
# RE-ANCHORED ON MEASUREMENT 2026-08-11. These constants used to read
# OFF_REF 32 GB, derived from the design's per-card table (24 -> on; 32 -> off
# @ B1; >=48 -> on + big batch) and never reconciled against a run. It was
# wrong, and wrong in the direction that OOMs a paid box: the same shape it
# describes — 7B-class LoRA, BATCH 1 x seq 4096, grad-ckpt OFF — measured
# **52.20 GB**, and the 12288 arm of the same run OOMed a 94.97 GiB card that
# the 32 rule licensed OFF on (docs/plans/witness/
# TRAINING_DEFAULTS_REVIEW_2026-08-09.md §2-§3).
#
# Both numbers below are ANCHOR READINGS, not arithmetic — the house rule after
# FITTING_9B_ON_A_5090_2026-08-06 §1 derived a slope that §8.2 then measured 7x
# lower. They are the output of `vram_facts.grad_ckpt_off_calibration()`, and
# `test_autotune.py::test_grad_ckpt_off_constants_are_the_measured_anchors`
# binds them to `vram_facts.json` so a re-harvest that moves either one fails
# the suite instead of silently drifting. They are duplicated here as literals
# (rather than imported) because the SAME rule has to evaluate box-side in
# `jobcommon/launch_plan.sh` and the runsets' `train.sh`, where there is no
# python and no facts file; a shared literal that a test pins is the only shape
# in which the bash mirrors and the python cannot disagree.
#
# TWO TERMS, because peak = (activations, ~linear in tokens-in-flight) +
# (weights + optimizer + static, flat in tokens). A rule proportional through
# the origin gets the second term wrong at short windows: proportional-from-52.2
# predicts 13.05 GB at 1024 tokens, where the measurement is 21.87 GB, so a
# 16 GB card would have been told OFF fits. Hence
#
#     OFF fits iff gpu_ram_gb >= max(OFF_REF * tokens / REF_TOKENS, OFF_MIN)
#
# which is consistent with every grad-ckpt-OFF measurement we hold (see the
# table in TRAINING_DEFAULTS_REVIEW §3 plus the 1024-token row).
#
# SCOPE. Both anchors are 7B-class and the heavier of them is bf16, so the rule
# is conservative for a quantized 7B (its weights term is smaller) and
# OPTIMISTIC for a >7B base — pin GRAD_CKPT explicitly for a 9B/12B/27B run on
# a tight card, exactly as before. A shape with its own measured OFF anchor
# should be sized from `vram_facts.estimate_peak_gb` at submit time rather than
# from this rule; today only 2 of 116 anchors are grad-ckpt-OFF and no live
# training bundle's OFF shape is among them. Phase 2 replaces the heuristic
# with the in-process fit-probe.
GRAD_CKPT_OFF_REF_VRAM_GB = 52.2   # MEASURED: qwen25-coder-7b bf16 LoRA r32,
                                   # BATCH 1 x seq 4096, grad-ckpt OFF, peak
                                   # allocated, run 20260805T075419-perf-levers
                                   # -e83b arm g_off_4096...
GRAD_CKPT_REF_TOKENS = 4096        # ...at this tokens-in-flight (batch x seq).
GRAD_CKPT_OFF_MIN_VRAM_GB = 21.87  # MEASURED floor below the reference window:
                                   # same base 8-bit at seq 1024 OFF (local-
                                   # smoke). Tokens-independent terms live here.


class AutotuneError(ValueError):
    """A plan that cannot be honored (refuse loudly, never guess)."""


def resolve_mode(mode: str | None) -> str:
    """Absent/empty -> 'pinned' (fail-closed). Unknown values are an error, not
    a guess — a typo'd 'auto-tune' silently landing in pinned would surprise."""
    m = (mode or "").strip().lower()
    if not m:
        return "pinned"
    if m not in VALID_MODES:
        raise AutotuneError(
            f"unknown MODE {mode!r} (expected one of {', '.join(VALID_MODES)})")
    return m


def effective_batch(batch: int, grad_accum: int, world_size: int = 1) -> int:
    """eff_batch = per_device_batch x grad_accum x world_size."""
    return int(batch) * int(grad_accum) * int(world_size)


def rebalance_grad_accum(grad_accum: int, world_size: int) -> int:
    """Divide the authored (world_size==1 denominated) grad-accum across ranks
    so the EFFECTIVE batch is held: B x (GA/W) x W == B x GA. REFUSES when it
    does not divide — a silent round would change the effective batch, which is
    an optimization change, not a throughput knob."""
    grad_accum, world_size = int(grad_accum), int(world_size)
    if world_size < 1:
        raise AutotuneError(f"world_size must be >= 1, got {world_size}")
    if grad_accum < 1:
        raise AutotuneError(f"grad_accum must be >= 1, got {grad_accum}")
    if grad_accum % world_size:
        raise AutotuneError(
            f"grad_accum {grad_accum} not divisible by world_size {world_size} "
            f"— refusing to auto-scale (effective batch would change). Author a "
            f"grad-accum divisible by the GPU count, or pin the run to fewer "
            f"cards.")
    return grad_accum // world_size


def pick_num_workers(cpu_cores: int, nproc: int,
                     floor: int = WORKERS_FLOOR, cap: int = WORKERS_CAP) -> int:
    """Dataloader workers PER RANK: half the cores split across ranks, clamped
    to [floor, cap]. Half, because each rank's main process + tokenizer/pin
    threads also need cores, and an oversubscribed collator starves EVERY
    tenant (memory: cpu-farm-saturator-starves-training). floor=2 because 0-1
    workers input-starve a grad-accum loop (the measured 18%-util smoke)."""
    cpu_cores, nproc = max(1, int(cpu_cores)), max(1, int(nproc))
    return max(floor, min(cap, cpu_cores // (2 * nproc)))


def quant_for_vram(gpu_ram_gb: float, resumable: bool = False,
                   allow_quantized: bool = False) -> str:
    """Per-assigned-card default precision. **bf16 unless asked otherwise.**

    This function used to be the biggest silent quantizer in the stack: it
    returned 4bit at <=32 GB and 8bit at 33-47 GB for every autotuned run, so a
    box that happened to land on a 40 GB card trained a different recipe than
    the same job on an 80 GB card — a config nobody chose, decided by the
    rental market. Owner standing rule 2026-08-04: **always train bf16 LoRAs.**
    So the default is now bf16 at EVERY size, and the VRAM table survives only
    behind `allow_quantized=True`, for a caller that has decided to pay the
    dequant tax to fit a small card.

    There is an independent correctness argument for bf16 that has nothing to
    do with VRAM, and it is why `resumable` existed: bnb 4-bit AND 8-bit both
    hit the paged-optimizer --resume crash, and bf16 is the only precision that
    reloads cleanly after a spot reclaim. So even the opt-in table refuses 8bit
    for a resumable run.

    allow_quantized=True reinstates the historical policy:
      resumable=False: <=32 GB -> 4bit, 33-47 GB -> 8bit, >=48 GB -> bf16.
      resumable=True:  bf16 wherever a 7B-class QLoRA fits
                       (>= RESUMABLE_BF16_MIN_GB), else 4bit — NEVER 8bit.

    (A pinned of-record run sets QUANT explicitly and never consults this.)"""
    if not allow_quantized:
        return "bf16"
    gb = float(gpu_ram_gb)
    if resumable:
        return "bf16" if gb >= RESUMABLE_BF16_MIN_GB else "4bit"
    if gb <= QUANT_4BIT_MAX_GB:
        return "4bit"
    if gb < QUANT_BF16_MIN_GB:
        return "8bit"
    return "bf16"


def grad_ckpt_off_vram_gb(batch: int, max_seq: int,
                          off_ref_vram_gb: float = GRAD_CKPT_OFF_REF_VRAM_GB,
                          ref_tokens: int = GRAD_CKPT_REF_TOKENS,
                          off_min_vram_gb: float = GRAD_CKPT_OFF_MIN_VRAM_GB) -> float:
    """Per-card VRAM a grad-ckpt-OFF run of this (batch, max_seq) needs.

    ``max(off_ref * tokens/ref_tokens, off_min)`` — see the constants above for
    why it is two terms and where each number was measured. Factored out so the
    bash mirrors have one expression to copy and one function to be tested
    against."""
    tokens = max(1, int(batch) * int(max_seq))
    proportional = float(off_ref_vram_gb) * tokens / max(1, int(ref_tokens))
    return max(proportional, float(off_min_vram_gb))


def pick_grad_ckpt(gpu_ram_gb: float, batch: int, max_seq: int,
                   off_ref_vram_gb: float = GRAD_CKPT_OFF_REF_VRAM_GB,
                   ref_tokens: int = GRAD_CKPT_REF_TOKENS,
                   off_min_vram_gb: float = GRAD_CKPT_OFF_MIN_VRAM_GB) -> str:
    """Gradient-checkpointing default: 'off' (throughput) when the activations
    fit at this (per-card VRAM, batch, seq), else 'on' (fit). See the constants
    above for the measured calibration. Returns 'on'/'off' (the trainer's
    --grad-checkpointing values); numerically identical either way, so this is
    always a safe default — explicit on/off from the caller must still win.

    Fit rule: OFF needs ``gpu_ram_gb >= grad_ckpt_off_vram_gb(batch, max_seq)``.
    At the measured anchors: 24 GB @ B1/1024 -> off (needs 21.87, measured
    21.87); 24/32/48 GB @ B1/4096 -> on (needs 52.2, measured 52.20); 96 GB @
    B1/4096 -> off; 96 GB @ B1/12288 -> on (needs 156.6 — the arm that OOMed a
    94.97 GiB card); big batch and long seq both scale into on."""
    gb = float(gpu_ram_gb)
    needed = grad_ckpt_off_vram_gb(batch, max_seq, off_ref_vram_gb, ref_tokens,
                                   off_min_vram_gb)
    return "off" if gb >= needed else "on"


def grad_ckpt_vram_safe(requested: str, gpu_ram_gb: float, batch: int,
                        max_seq: int,
                        off_ref_vram_gb: float = GRAD_CKPT_OFF_REF_VRAM_GB,
                        ref_tokens: int = GRAD_CKPT_REF_TOKENS,
                        off_min_vram_gb: float = GRAD_CKPT_OFF_MIN_VRAM_GB) -> str:
    """VRAM-safety FLOOR for gradient checkpointing: force 'on' when the caller
    pinned 'off' but the activations do NOT fit at this (per-card VRAM, batch,
    seq). Returns the safe value ('on'/'off').

    Distinct from pick_grad_ckpt, which is a THROUGHPUT suggestion free to go
    either direction. The safety floor only ever flips off->on, and grad-ckpt
    is numerically IDENTICAL to off (it trades VRAM for recompute, never the
    math), so applying this floor is safe in EVERY mode — including a pinned
    of-record paired run: it decides only whether the run OOMs, never what it
    computes. A caller's 'on' or a fitting 'off' pass through unchanged, so a
    card with headroom keeps the authored (saturated) shape."""
    if requested != "off":
        return requested
    return "on" if pick_grad_ckpt(gpu_ram_gb, batch, max_seq, off_ref_vram_gb,
                                  ref_tokens, off_min_vram_gb) == "on" else "off"


def plan(*, mode: str | None, gpus: int, batch: int, grad_accum: int,
         cpu_cores: int | None = None, num_workers: int | None = None,
         gpu_ram_gb: float | None = None, resumable: bool = False,
         max_seq: int | None = None, grad_ckpt: str | None = None,
         allow_quantized: bool = False) -> dict:
    """Compose the launch shape. Returns a dict of PLAN_* values.

    pinned:   nproc=1, grad_accum verbatim, workers explicit-or-2. gpus>1 is
              tolerated (warned by callers) but never used — a paired run must
              not change world_size.
    autotune: nproc=gpus (>=1), grad_accum rebalance (refusing non-divisible),
              workers explicit-else-formula. quant suggestion only when
              gpu_ram_gb is known AND the caller did not choose one; grad-ckpt
              THROUGHPUT suggestion only in autotune when BOTH gpu_ram_gb and
              max_seq are known (the fit rule needs the seq). Since 2026-08-04
              the quant suggestion is bf16 at every card size unless
              allow_quantized=True asks for the small-card table. Pinned never
              suggests quant or a throughput grad-ckpt — BUT the grad-ckpt
              VRAM-SAFETY floor (grad_ckpt='off' that will not fit -> 'on')
              applies in BOTH modes, because it is numerically identical to off
              and only prevents an OOM (never perturbs a paired of-record run).
    """
    m = resolve_mode(mode)
    batch, grad_accum = int(batch), int(grad_accum)
    if batch < 1 or grad_accum < 1:
        raise AutotuneError(f"batch/grad_accum must be >= 1 "
                            f"(got batch={batch} grad_accum={grad_accum})")
    eff = effective_batch(batch, grad_accum, 1)   # authored, world_size==1
    if m == "pinned":
        nproc = 1
        ga = grad_accum
        workers = HISTORICAL_NUM_WORKERS if num_workers is None else int(num_workers)
        quant = None
    else:
        nproc = max(1, int(gpus))
        ga = rebalance_grad_accum(grad_accum, nproc)
        if num_workers is not None:
            workers = int(num_workers)
        else:
            workers = pick_num_workers(cpu_cores if cpu_cores else 1, nproc)
        quant = (quant_for_vram(gpu_ram_gb, resumable=resumable,
                                allow_quantized=allow_quantized)
                 if gpu_ram_gb else None)
    out = {
        "PLAN_MODE": m,
        "PLAN_NPROC": nproc,
        "PLAN_BATCH": batch,
        "PLAN_GRAD_ACCUM": ga,
        "PLAN_NUM_WORKERS": workers,
        "PLAN_EFF_BATCH": eff,      # invariant across modes by construction
    }
    if quant:
        out["PLAN_QUANT"] = quant
    # grad-ckpt. Two independent mechanisms, both needing per-card VRAM + seq:
    #   (a) THROUGHPUT suggestion (autotune, caller left it unset/auto):
    #       'hybrid' since 2026-08-28 — it subsumes the pick_grad_ckpt on/off
    #       guess by measuring in-process (per-micro-batch fraction, never
    #       OOMs where full GC fits, loss-identical; 1.41-1.45x measured on
    #       big cards) and retires the 7B-fitted-constants footgun for >7B
    #       bases. pick_grad_ckpt survives as the safety floor's fit rule.
    #   (b) VRAM-SAFETY floor (BOTH modes): flip to 'on' when the EFFECTIVE
    #       request resolves to 'off' but 'off' will not fit. The effective
    #       request is 'off' both when the caller pinned it AND when they left
    #       it unset/auto in pinned mode — run.sh falls back to
    #       ${GRAD_CKPT:-off}, so an unevaluated floor is an OOM waiting for a
    #       long-seq run (live instance: phase1-cot 2026-08-02, pinned + unset
    #       at seq 16384 on 96 GB — needs 128 by the fit rule, planner stayed
    #       silent, both arms OOM'd at step 2). Numerically identical to off,
    #       so safe even for a pinned of-record run.
    # Emitted only when it actually resolves a value; a fitting card keeps the
    # authored GRAD_CKPT (run.sh falls back to ${GRAD_CKPT:-off}).
    if gpu_ram_gb and max_seq:
        if m != "pinned" and grad_ckpt in (None, "", "auto"):
            out["PLAN_GRAD_CKPT"] = "hybrid"
        elif grad_ckpt in ("off", None, "", "auto"):
            if grad_ckpt_vram_safe("off", gpu_ram_gb, batch, max_seq) == "on":
                out["PLAN_GRAD_CKPT"] = "on"   # safety flip (off would OOM here)
    # invariant check (cheap, prevents any future formula edit from breaking it)
    assert effective_batch(batch, ga, nproc) == eff, "eff-batch invariant broke"
    return out


def cache_key(*, model: str, max_seq: int, gpu_name: str, quant: str,
              world_size: int, packing: bool = False,
              trainer_rev: str = "") -> str:
    """Stable key for the phase-3 throughput-tune cache (box disk + B2).
    Keyed on everything that moves the tokens/sec knee; packing is IN the key
    because it changes tokens/step at a given micro-batch."""
    raw = json.dumps({
        "model": model, "max_seq": int(max_seq),
        "gpu": gpu_name.strip().lower().replace(" ", "-"),
        "quant": quant, "world_size": int(world_size),
        "packing": bool(packing), "trainer_rev": trainer_rev,
    }, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="emit PLAN_* shell assignments")
    p.add_argument("--mode", default=None)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--grad-accum", type=int, required=True)
    p.add_argument("--cpu-cores", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--gpu-ram-gb", type=float, default=None)
    p.add_argument("--max-seq", type=int, default=None,
                   help="MAX_SEQ (tokens): with --gpu-ram-gb, emits a "
                        "PLAN_GRAD_CKPT on/off suggestion (autotune only)")
    p.add_argument("--grad-ckpt", default=None,
                   help="requested GRAD_CKPT (on/off/auto): with --gpu-ram-gb + "
                        "--max-seq, a pinned/explicit 'off' that will not fit is "
                        "flipped to 'on' (VRAM-safety floor, numerically identical)")
    p.add_argument("--resumable", action="store_true",
                   help="checkpointing spot run: prefer bf16 quant wherever it "
                        "fits (bnb quant cannot reload optimizer state on --resume)")
    a = ap.parse_args(argv)
    try:
        out = plan(mode=a.mode, gpus=a.gpus, batch=a.batch,
                   grad_accum=a.grad_accum, cpu_cores=a.cpu_cores,
                   num_workers=a.num_workers, gpu_ram_gb=a.gpu_ram_gb,
                   resumable=a.resumable, max_seq=a.max_seq, grad_ckpt=a.grad_ckpt)
    except AutotuneError as e:
        print(f"!! autotune: {e}", file=sys.stderr)
        return 12
    for k, v in out.items():
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
