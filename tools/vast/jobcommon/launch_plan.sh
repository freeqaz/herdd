#!/usr/bin/env bash
# launch_plan.sh — sourceable launch-shape planner for run.sh (jobs-v2 bundle).
#
# BASH MIRROR of tools/vast/autotune.py (the canonical implementation — a job
# bundle ships only its own folder, so it cannot import that module box-side).
# tools/vast/test_autotune.py cross-checks this file against the python over a
# grid, so the two cannot drift silently. Keep the formulas in lockstep.
#
# Contract: `plan_launch` reads MODE BATCH GRAD_ACCUM JOB_GPU_COUNT NUM_WORKERS
# CPU_CORES from the environment and sets:
#   PLAN_MODE PLAN_NPROC PLAN_BATCH PLAN_GRAD_ACCUM PLAN_NUM_WORKERS PLAN_EFF_BATCH
# Returns non-zero (with a "!! plan:" line on stderr) on a plan it cannot honor
# — the caller must abort, never guess.
#
# Modes (AUTOTUNE_DESIGN.md §6 — the reproducibility boundary):
#   pinned   (DEFAULT when MODE unset — fail-closed): single process, grad-accum
#            verbatim, workers explicit-or-2 (historical). Paired/of-record runs
#            (v2<->v2.1) live here: grad-accum + padding + per-token mean
#            reduction is NOT invariant to micro-batch composition or world
#            size, so batch, composition AND world_size stay frozen.
#   autotune fresh/bakeoff runs: DDP over every assigned card
#            (nproc = JOB_GPU_COUNT), grad-accum divided by nproc to HOLD the
#            authored effective batch (REFUSES when not divisible), dataloader
#            workers auto-sized clamp(cores/(2*nproc), 2, 8) unless explicit.

plan_launch() {
  local mode="${MODE:-}" batch="${BATCH:-1}" ga="${GRAD_ACCUM:-32}"
  local gpus="${JOB_GPU_COUNT:-1}" workers="${NUM_WORKERS:-}"
  local cores="${CPU_CORES:-}"
  # per-card VRAM (jobd exports JOB_GPU_RAM_GB in phase 2; empty in phase 1 ->
  # no quant suggestion). RESUMABLE: this bundle checkpoints + --resume auto, so
  # default ON — prefer bf16 wherever it fits since bnb 4/8-bit cannot reload
  # optimizer state on --resume (memory bnb-4bit-qlora-resume-crash).
  local ram="${JOB_GPU_RAM_GB:-}" resumable="${RESUMABLE:-1}"
  PLAN_QUANT=""     # empty = no suggestion (run.sh falls back to ${QUANT:-bf16})

  # mode resolution: absent/empty -> pinned (never silently perturb an existing
  # caller); anything else must be an exact known mode (a typo must not land in
  # pinned silently).
  case "$mode" in
    "")        mode=pinned
               echo ">> plan: MODE unset -> pinned (historical behaviour; set MODE=autotune for fresh/bakeoff runs)" >&2 ;;
    pinned|autotune) : ;;
    *) echo "!! plan: unknown MODE '$mode' (expected pinned|autotune)" >&2; return 12 ;;
  esac

  case "$batch$ga" in (*[!0-9]*) echo "!! plan: BATCH/GRAD_ACCUM must be integers (got '$batch'/'$ga')" >&2; return 12 ;; esac
  [ "$batch" -ge 1 ] && [ "$ga" -ge 1 ] \
    || { echo "!! plan: BATCH/GRAD_ACCUM must be >= 1 (got $batch/$ga)" >&2; return 12; }

  PLAN_MODE="$mode"
  PLAN_BATCH="$batch"
  PLAN_EFF_BATCH=$(( batch * ga ))      # authored (world_size==1) — held invariant

  if [ "$mode" = "pinned" ]; then
    PLAN_NPROC=1
    PLAN_GRAD_ACCUM="$ga"
    PLAN_NUM_WORKERS="${workers:-2}"    # historical run.sh default
    if [ "${gpus:-1}" -gt 1 ] 2>/dev/null; then
      echo ">> plan: MODE=pinned on a ${gpus}-card assignment — using 1 card (a paired run must not change world_size)" >&2
    fi
  else
    case "$gpus" in (*[!0-9]*|"") gpus=1 ;; esac
    [ "$gpus" -ge 1 ] || gpus=1
    PLAN_NPROC="$gpus"
    if [ $(( ga % gpus )) -ne 0 ]; then
      echo "!! plan: GRAD_ACCUM $ga not divisible by nproc $gpus — refusing to auto-scale (effective batch would change). Author a grad-accum divisible by the GPU count, or pin the run to fewer cards." >&2
      return 12
    fi
    PLAN_GRAD_ACCUM=$(( ga / gpus ))
    if [ -n "$workers" ]; then
      PLAN_NUM_WORKERS="$workers"       # explicit env always wins
    else
      [ -n "$cores" ] || cores="$(nproc 2>/dev/null || echo 1)"
      case "$cores" in (*[!0-9]*|"") cores=1 ;; esac
      local w=$(( cores / (2 * gpus) ))
      [ "$w" -lt 2 ] && w=2             # floor: <2 input-starves grad-accum
      [ "$w" -gt 8 ] && w=8             # cap: diminishing returns + host RAM
      PLAN_NUM_WORKERS="$w"
    fi
    # quant suggestion (autotune only). MIRROR of autotune.quant_for_vram.
    # DEFAULT since 2026-08-04 is bf16 at EVERY card size — owner standing rule
    # "always train bf16 LoRAs". The VRAM-tiered table below is OPT-IN
    # (ALLOW_QUANTIZED=1) and survives only for a caller that has decided to pay
    # the dequant tax to fit a small card; its thresholds 24 (resumable bf16
    # floor) / 32 (4bit max) / 48 (bf16 min) must stay in lockstep with
    # autotune.py (test_autotune.py cross-checks). awk compares floats.
    case "$ram" in
      ''|*[!0-9.]*) : ;;                              # unknown -> no suggestion
      *)
        if [ "${ALLOW_QUANTIZED:-0}" != "1" ]; then PLAN_QUANT=bf16
        elif [ "$resumable" = "1" ]; then
          if awk "BEGIN{exit !($ram >= 24)}"; then PLAN_QUANT=bf16; else PLAN_QUANT=4bit; fi
        elif awk "BEGIN{exit !($ram <= 32)}"; then PLAN_QUANT=4bit
        elif awk "BEGIN{exit !($ram < 48)}"; then PLAN_QUANT=8bit
        else PLAN_QUANT=bf16
        fi ;;
    esac
  fi

  # grad-ckpt (MIRROR of autotune.plan grad-ckpt block). Two mechanisms, both
  # needing per-card VRAM (JOB_GPU_RAM_GB) + seq (MAX_SEQ):
  #   (a) THROUGHPUT suggestion (autotune, GRAD_CKPT unset/auto): 'hybrid'
  #       since 2026-08-28 — self-calibrates in-process, subsumes the on/off
  #       guess, loss-identical (1.41-1.45x measured on big cards).
  #   (b) VRAM-SAFETY floor (BOTH modes, GRAD_CKPT=off that won't fit -> on).
  #       grad-ckpt is numerically identical to off, so the floor is safe even
  #       for the pinned of-record run (decides OOM-vs-fit, not the math).
  # Emitted only when it resolves a value; else run.sh keeps ${GRAD_CKPT:-off}.
  # Fit rule (== autotune.grad_ckpt_off_vram_gb / pick_grad_ckpt):
  #   off fits iff ram >= max(52.2*(batch*seq)/4096, 21.87)
  # Both constants are MEASURED anchor readings, not arithmetic, and the pair is
  # bound to tools/vast/vram_facts.json by test_autotune.py — keep them in
  # lockstep with autotune.GRAD_CKPT_OFF_REF_VRAM_GB / _OFF_MIN_VRAM_GB.
  # 52.2 = qwen25-coder-7b bf16 LoRA, BATCH 1 x seq 4096, grad-ckpt OFF (run
  # 20260805T075419-perf-levers-e83b). It read 32 until 2026-08-11, which is
  # 20 GB below the measurement of the very shape it describes and licensed OFF
  # for the 12288 arm that OOMed a 94.97 GiB card. 21.87 = the same base 8-bit
  # at seq 1024 OFF: the flat weights/optimizer term a proportional rule misses
  # at short windows.
  PLAN_GRAD_CKPT=""
  local gc_req="${GRAD_CKPT:-}" seq="${MAX_SEQ:-}" ram_ok=0 seq_ok=0
  case "$ram" in ''|*[!0-9.]*) ;; *) ram_ok=1 ;; esac
  case "$seq" in ''|*[!0-9]*) ;; *) seq_ok=1 ;; esac
  if [ "$ram_ok" = 1 ] && [ "$seq_ok" = 1 ]; then
    local fit_off=1                                       # 1 == off fits
    awk "BEGIN{n=52.2*($batch*$seq)/4096; if(n<21.87)n=21.87; exit !($ram >= n)}" \
      || fit_off=0
    if [ "$mode" = "autotune" ] && { [ -z "$gc_req" ] || [ "$gc_req" = "auto" ]; }; then
      PLAN_GRAD_CKPT="hybrid"
    elif { [ "$gc_req" = "off" ] || [ -z "$gc_req" ] || [ "$gc_req" = "auto" ]; } \
         && [ "$fit_off" -eq 0 ]; then
      # unset/auto in pinned mode resolves to 'off' in run.sh (${GRAD_CKPT:-off}),
      # so the floor must see it too (phase1-cot 2026-08-02 OOM: pinned + unset
      # at seq 16384 on 96 GB stayed silent, both arms OOM'd at step 2)
      PLAN_GRAD_CKPT="on"                                 # VRAM-safety flip
    fi
  fi

  # invariant: eff batch unchanged by the plan (B x GA' x NPROC == B x GA)
  if [ $(( PLAN_BATCH * PLAN_GRAD_ACCUM * PLAN_NPROC )) -ne "$PLAN_EFF_BATCH" ]; then
    echo "!! plan: eff-batch invariant broke (bug in launch_plan.sh)" >&2; return 12
  fi
  export PLAN_MODE PLAN_NPROC PLAN_BATCH PLAN_GRAD_ACCUM PLAN_NUM_WORKERS PLAN_EFF_BATCH PLAN_QUANT PLAN_GRAD_CKPT
  echo ">> plan: mode=$PLAN_MODE nproc=$PLAN_NPROC batch=$PLAN_BATCH grad_accum=$PLAN_GRAD_ACCUM workers=$PLAN_NUM_WORKERS eff_batch=$PLAN_EFF_BATCH quant=${PLAN_QUANT:-<env>} grad_ckpt=${PLAN_GRAD_CKPT:-<env>}" >&2
  return 0
}
