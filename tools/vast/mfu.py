#!/usr/bin/env python3
"""mfu.py — GPU FLOP utilisation, with a denominator you are allowed to quote.

Companion to `tools/vast/jobcommon/gemm_ceiling.py`, which measures
the DENOMINATOR (dense-bf16 GEMM ceiling, per device). This module owns the
NUMERATOR (FLOP/token, per base) and the division.

WHAT THIS METRIC IS, AND WHAT IT IS NOT — read before quoting it anywhere
------------------------------------------------------------------------
**The quantity this module computes is `roof-HFU`, not MFU.** The name `mfu` in
the filename, the `mfu_raw` / `mfu_required` JSON keys and the
`DenominatorError` message are FROZEN WIRE/API SURFACE (`hostfacts.py`,
`test_hostfacts.py`, `gemm_probe.py` records) and are left alone; every
HUMAN-FACING label says `roof-HFU`. Definition, once, here — every quoting
site in `docs/plans/witness/` points back to this paragraph:

  **roof-HFU** = (FLOPs the GPU actually EXECUTES per second)
               / (a MEASURED dense-bf16 GEMM ceiling for that device).

It departs from conventional MFU in BOTH terms, and both departures push the
number UP:

  * NUMERATOR — it is a HARDWARE-FLOP count, not a model-FLOP count. It bills
    gradient-checkpoint recomputation as work (`PASSES_WEIGHTS = 3` is
    fwd + recompute + bwd(dx); the un-recomputed LoRA count is 2), and the
    `raw` variant additionally bills the dense `T x T` that gemma-4's sliding
    layers EXECUTE but the architecture does not require (20.6% of all training
    work — that gap is the whole point of `mfu_required`, which removes the
    sliding waste but still counts recompute). PyTorch's own MFU-vs-HFU
    distinction is exactly this: activation recomputation makes HFU rise while
    MFU falls. By that convention this is HFU.
  * DENOMINATOR — a measured dense-bf16 GEMM roof on the specific card, not a
    vendor theoretical peak. That roof is BELOW peak by construction (see the
    four competing RTX PRO 6000 denominators in discipline 1), which raises the
    ratio again.

Consequences, and they are load-bearing:

  * **Never compare a roof-HFU figure to a published MFU number.** Conventional
    MFU for the same run is lower on both axes; the two are not the same
    statistic and the gap is not a small one.
  * **Do not conclude "software headroom is limited" from a high roof-HFU.**
    A roof-HFU near the roof means the GEMMs are well fed against a ceiling we
    lowered to what this card really achieves — it says nothing about whether
    the FLOPs being executed were WORTH executing. Levers that delete work
    (banding, recompute-free regimes, padding-free batching) reduce the
    numerator and are invisible to this ratio, or move it the wrong way.
    `Flops.wasted_share` is the honest place to look for that headroom.
  * roof-HFU is still the right instrument for what it was built for: comparing
    two runs on the SAME card against the SAME measured roof, and catching a
    numerator or denominator that has silently rotted.

Recorded 2026-08-09 in response to an external review
(`docs/plans/witness/perf/reviews/EXTERNAL_REVIEW_RESPONSE_2026-08-09.md`, claim 1),
which is correct on this point. LABELS AND PROSE ONLY — no arithmetic in this
module changed, and every historical figure stands at the value it was
published with.

Three files, one denominator, and it is worth knowing which produced yours:

  * `gemm_ceiling.py`  — the BENCH instrument, shared from `tools/vast/jobcommon/`
    and pulled into the bundles that use it via `includes:`.
    Runs at whatever shapes you give it, so it is the one to use for a model's
    own `--gemm-cmd` shapes. It only runs when someone rents a box to bench.
  * `tools/vast/gemm_probe.py` — the BOOT instrument, in the jobd bundle. Runs
    on every box, guarded (refuses a busy GPU), bounded, fail-open, at a generic
    three-aspect-class shape set because no base is known at boot. Emits this
    module's `--ceiling-json` format as a superset, tagged `shape_basis:
    generic` so the approximation is stated rather than assumed away.
  * `tools/vast/hostfacts.py` — the STORE. `hostfacts.py ceiling --machine <M>`
    prints the newest quotable record for a host, ready to pipe into
    `--ceiling-json`, which is how an MFU stops being PROVISIONAL for a box that
    has been probed.

Why it exists
-------------
`docs/plans/witness/perf/TRAINING_THROUGHPUT_REVIEW_2026-08-06.md` §3 built a
per-base FLOP model in prose and arithmetic. Prose does not survive a config
change: gemma-4-12b-text's 92.2 GFLOP/token depends on 48 layers, 40 of them
sliding at head_dim 256 and 8 global at head_dim 512, a time-weighted mean
sequence length of 7,461, and a 4.48% label mass. Change any of those and every
downstream ratio in that document silently rots. This encodes it, parameterised
from a HuggingFace `config.json`, so the arithmetic is re-derivable and pinned
by tests.

Four disciplines are enforced in code, not by convention:

1. **No TFLOP/s or MFU without a device name.** Verbatim from `gemm_ceiling.py`:
   *"A TFLOP/s figure with no device attached is not quotable."* `Ceiling`
   refuses to construct without one, and a ceiling measured on device A but
   applied to a run on device B is emitted as `provisional` with the borrow
   recorded.

   **A device name is necessary and no longer sufficient: NAME THE DENOMINATOR
   EVERY TIME.** Four are in circulation for the RTX PRO 6000 Blackwell and
   they are not interchangeable. An MFU quoted without saying which one it
   divided by is unreadable, and two of the pairs below have already been
   mistaken for a contradiction:

     * **262 TFLOP/s** — the **Max-Q Workstation** part at *Qwen's* shapes
       (`V7_PERF_LEVERS_2026-08-05` section 2). A borrow from a different SKU.
       **Retracted** as this card's ceiling by `G4_ATTENTION_PROFILE` section 3;
       do not requote it.
     * **419.2 TFLOP/s** — **max across shapes**, measured in-run on the
       **Server Edition** itself (`G4_ATTENTION_PROFILE` section 3; the three
       rates were 366.4 / 419.2 / 386.0). The strictest denominator, and the
       headline this module's own `from_gemm_ceiling_json` calls "honest but
       optimistic".
     * **397.7 TFLOP/s** — those same three rates, FLOP-weighted over gemma-4's
       MAC mix by `harmonic_weighted`. Note the profile benched
       K/N = 3,584 / 18,944, which are **not** gemma-4's 3,840 / 15,360, so this
       is a `shape_basis="generic"` number for this base and says so.
     * **358.8 TFLOP/s** — FLOP-weighted over gemma-4's *own* shapes
       (owner-measured 2026-08-07). The most representative of the four. It has
       no `gemm_ceiling.py` record checked in yet; give it one and the
       PROVISIONAL/APPROXIMATE machinery below picks it up automatically.

   So `99.3 / 419.2 = 23.7%` and `99.3 / 358.8 = 27.7%` (27.6% if the achieved
   figure is rounded to 99) are both arithmetically right and describe the same
   run. Different denominators, not a disagreement.

2. **The O(T^2) term is quoted at the TIME-WEIGHTED mean sequence length,
   `sum(T^2)/sum(T)`, not the median.** The tail is where the attention bill is:
   for the v9 corpus that is 7,461 against a median of 3,811, a 1.96x difference
   in the term it multiplies. `attention_flops` takes `tw` and nothing else.

3. **Sliding attention is modelled separately from dense.** `ATTN_IMPL=sdpa`
   runs gemma-4's 40 sliding layers as dense `T x T` against a bool mask, so the
   work we EXECUTE and the work the architecture REQUIRES differ. Both are
   reported: `mfu_raw` divides by the ceiling using what runs today, `mfu_required`
   uses the banded cost. The gap is the point -- 19.0 of 92.2 GFLOP/token, 20.6%
   of all training work, spent on key positions the architecture has already
   decided to ignore. Priced in SECONDS rather than FLOPs it is bigger still:
   `G4_ATTENTION_PROFILE_2026-08-07` measures ~39% of the step at its own
   `Tw = 8,303` (~36% at production `Tw`), because the attention kernels run at
   a fraction of GEMM efficiency -- see discipline 4.

4. **FLOPs and SECONDS are kept apart, and every constant says which it is.**
   Each cost-model constant below is tagged DERIVED (counted from the algorithm,
   portable) or MEASURED (read off one profiled step on one card, portable to
   neither), and only DERIVED constants may enter `flops_per_token`. A measured
   ratio of seconds is a statement about EFFICIENCY; billing it as FLOPs inflates
   the MFU numerator by exactly the inefficiency it was recording. The live case
   is `PASSES_ATTENTION`, which is **4.5 (derived)** and deliberately **not 6.85
   (measured, seconds)**; the measured number is not discarded, it lives in
   `PASSES_ATTENTION_TIME` and is used by `Flops.banding_speedup_timed`, which is
   a time prediction and the only place it belongs.

CLI
---
    # numerator only, from a checked-in config
    python3 tools/vast/mfu.py --config <path>/gemma4-12b-text.config.json --tw 7461

    # full MFU, borrowing a ceiling measured on a DIFFERENT device (flagged)
    python3 tools/vast/mfu.py --model gemma-4-12b-text --tw 7461 --tok-s 1430 \
        --device 'NVIDIA RTX PRO 6000 Blackwell Server Edition' \
        --ceiling-tflops 261.9 \
        --ceiling-device 'NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition' \
        --ceiling-source 'V7_PERF_LEVERS_2026-08-05 section 2, Qwen shapes'

    # the gemm_ceiling.py invocation that would measure THIS model's own shapes
    python3 tools/vast/mfu.py --model gemma-4-12b-text --gemm-cmd

    # machine-readable
    python3 tools/vast/mfu.py --model gemma-4-12b-text --tw 7461 --tok-s 1430 --json

Stdlib only (argparse/json/dataclasses/math). No torch, no GPU, no network.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Cost-model constants. Named, because every one of them is a decision the
# review document argued for and any of them could reasonably be different for
# a full-finetune or a non-checkpointed run.
#
# TWO KINDS LIVE HERE AND THEY MUST NOT BE MIXED (see discipline 4 above):
#
#   DERIVED   counted from the algorithm -- matmuls, passes. Exact, portable to
#             any hardware, re-checkable with a pencil.
#   MEASURED  read off a profile of one real step. Belongs to that SHAPE and
#             that DEVICE and is portable to neither, so it carries both.
#
# `flops_per_token` is a FLOP model. Only DERIVED constants may enter it.
# ---------------------------------------------------------------------------

#: **DERIVED**, and since 2026-08-07 **CONFIRMED** against a kernel profile.
#:
#: Forward-equivalent passes over the weight GEMMs, LoRA + gradient
#: checkpointing. Base weights are FROZEN so the backward is dx-only (~1x
#: forward, no dW GEMM); recompute adds one more forward.
#: fwd + recompute + bwd(dx) = 3. (V7_PERF_LEVERS_2026-08-05 section 3's
#: corrected accounting. A full finetune would be 4: the dW GEMM comes back.)
#:
#: The confirmation, because "it was derived the same way as the attention
#: constant that turned out wrong" is a fair objection and this is the answer to
#: it. `G4_ATTENTION_PROFILE_2026-08-07` section 1's bucket table puts the
#: body/MLP/LoRA `gemm` bucket at 33.811 s of a 166.65 s step over 175,144
#: tokens at `Tw = 8,303`. Divide this model's weight-GEMM FLOPs by that time:
#:
#:     passes_weights=2  ->  227.2 TFLOP/s   54% of the measured ceiling
#:     passes_weights=3  ->  340.9 TFLOP/s   81%   <-- shipped
#:     passes_weights=4  ->  454.5 TFLOP/s  108%   IMPOSSIBLE
#:
#: against the 419.2 TFLOP/s dense-bf16 GEMM ceiling measured in-run on the same
#: box. 4 would require the body GEMMs to run ABOVE the card's measured peak; 2
#: would have a pure bf16 GEMM stack idling at half rate. 81% is exactly where a
#: well-fed GEMM stack sits, so 3 is not merely derived here -- it is the only
#: one of the three the measurement admits. Left underived: nothing. This
#: constant is now the better-evidenced of the two.
PASSES_WEIGHTS = 3

#: **DERIVED.** Forward-equivalent passes over the O(T^2) attention math,
#: counted in MATMULS, because the two O(T^2) matmuls are all this model bills:
#:
#:     forward    2   S = Q K^T  and  O = P V
#:     backward   5   S recomputed, dV = P^T dO, dP = dO V^T, dQ = dS K,
#:                    dK = dS^T Q                      => 2.5x the forward
#:     grad-ckpt  +1  one more forward
#:     total      1 + 1 + 2.5 = 4.5
#:
#: This was **4** until 2026-08-07 -- the standard "backward is 2x forward"
#: convention, which undercounts because a flash / memory-efficient backward
#: RECOMPUTES S rather than storing it. 4 -> 4.5 is +12.5% on the attention term
#: and +3.3% on the total. Note this correction is derived, not measured: it is
#: the matmul count, and anyone can recheck it against the list above.
#:
#: **It is NOT 6.85, and the reason is the whole point of discipline 4.**
#: `G4_ATTENTION_PROFILE_2026-08-07` section 0 measured the attention backward at
#: 4.85x the forward and proposed `1 + 1 + 4.85 = 6.85` for this constant. 4.85
#: is a ratio of SECONDS (`ATTENTION_BWD_TIME_RATIO` below), not of FLOPs. The
#: backward executes 2.5x the forward's FLOPs while taking 4.85x as long, i.e.
#: it runs at `2.5 / 4.85 =` **52% of the forward's FLOP/s** -- 45.3 against
#: 87.8 TFLOP/s on that trace. That is a known property of cutlass's
#: memory-efficient backward (atomics accumulating dQ, heavy dS elementwise),
#: not extra arithmetic. Putting 6.85 here would bill the inefficiency as work
#: the GPU performed: `Flops.raw` +15.0% at the corpus `tw` (+20% against the
#: old 4, and it grows with `tw`), and every MFU derived from it inflated by the
#: same factor -- which is the error the whole module exists to prevent, merely
#: relocated from the denominator to the numerator.
PASSES_ATTENTION = 4.5

#: **MEASURED** -- one profiled optimizer step. Not portable to another shape or
#: another card, so both are recorded.
#:
#: Self-CUDA time of the 40 sliding `head_dim 256` attention layers, attributed
#: kernel -> aten op through the profiler's `External id` linkage (836,065
#: kernels, 0.01% unlinked, so an attribution rather than a sample):
#:
#:     forward + grad-ckpt recompute   21.700 s   => one forward   10.850 s
#:     backward                        52.615 s   => 52.615 / 10.850 = 4.849
#:
#: PROVENANCE branch `t50-attn-profile`,
#:   `docs/plans/witness/perf/G4_ATTENTION_PROFILE_2026-08-07.md` section 0,
#:   bundle `tools/witness/jobs/g4-attn-profile`, job
#:   `20260807T031101-g4-attn-profile-12f0`, cell `a_sdpa_prof`.
#: SHAPE gemma-4-12b-text; BATCH 1 x GRAD_ACCUM 32; MAX_SEQ 40960; bf16;
#:   gradient checkpointing ON; `ATTN_IMPL=sdpa`, `SDPA_BACKENDS=flashmeff`;
#:   32 real corpus rows, 175,144 tokens, `Tw = 8,303`.
#: HARDWARE one NVIDIA RTX PRO 6000 Blackwell **Server Edition**, sm_120,
#:   188 SMs, torch 2.11.0+cu129.
#:
#: The same trace puts the 8 global `head_dim 512` layers at the same
#: FLOP/s (55.4 vs 57.4 TFLOP/s blended), which is why this reads as a property
#: of the kernel rather than of the architecture -- but it is still one shape on
#: one card, and a flash-3 / flex backward would not have this ratio at all.
ATTENTION_BWD_TIME_RATIO = 4.85

#: **MEASURED-derived, for TIME predictions ONLY** -- never for `Flops.raw`.
#: 1 fwd + 1 grad-ckpt recompute + `ATTENTION_BWD_TIME_RATIO` = 6.85
#: forward-EQUIVALENT SECONDS per attention pass. Used by
#: `Flops.banding_speedup_timed`; the FLOP numerator does not see it. Same shape
#: and hardware caveat as the constant it is built from.
PASSES_ATTENTION_TIME = 2.0 + ATTENTION_BWD_TIME_RATIO

# A second reason not to reach for 6.85, recorded because the readout says
# otherwise and someone will re-derive it. Substituting 6.85 does NOT close the
# gap between the model's attention FLOP share (28.4% at Tw = 8,303) and the
# measured attention TIME share (53.8%). It moves the predicted share to 40.5%
# -- 1.42x of a 1.91x gap. The readout compared the multiplier ratio
# 6.85/4 = 1.71x against the share ratio 1.91x and called them nearly equal, but
# a share carries the corrected term in its own denominator, so bumping
# attention by 1.71x moves its SHARE by only 1.42x. The two are not comparable
# quantities.
#
# Forcing FLOP share to equal time share would need 11.9, and would be worse
# still: the profiler's `attention` bucket is whole-kernel self time (softmax,
# mask apply, dS elementwise, rescaling) while this model bills only the two
# O(T^2) matmuls, so an attention TIME share must exceed its matmul-FLOP share
# whatever constant is chosen. The residual is unbilled non-matmul work, not a
# missing multiplier, and no value of this constant can absorb it.

#: MACs -> FLOPs.
FLOPS_PER_MAC = 2

#: Share of positions that carry loss on the witness repair corpus, i.e. the
#: share of positions the chunked CE lm_head GEMM actually runs on. TRL's
#: `_chunked_cross_entropy_loss` drops `-100` positions BEFORE chunking.
#: Measured: 1,033,015 of 23,061,435 tokens (TRAINING_THROUGHPUT_REVIEW section 2).
DEFAULT_LABEL_SHARE = 0.0448

#: Time-weighted mean sequence length, sum(T^2)/sum(T), of the v9 corpus under
#: each base's own tokenizer (TRAINING_THROUGHPUT_REVIEW section 1). NOT the
#: median (3,811) -- see the module docstring.
TW_V9_GEMMA4 = 7461.0
TW_V9_QWEN35 = 7338.8


class DenominatorError(ValueError):
    """Raised when a TFLOP/s or MFU figure would be emitted without provenance."""


# ---------------------------------------------------------------------------
# Model shape
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Block:
    """One homogeneous group of decoder layers, described by its token mixer.

    `kind` drives the O(T^2) term and nothing else:
      ``dense``   full causal attention. Cost ~ 0.5*T per query (the causal
                  half-triangle); this is what SDPA's `is_causal` skip buys when
                  `attn_mask is None`.
      ``sliding`` sliding-window attention. Required cost ~ min(T, window) keys
                  per query; EXECUTED cost is a full dense T x T today, because
                  handing SDPA a bool window mask forfeits the causal skip.
      ``linear``  linear/recurrent mixer (gated DeltaNet). No quadratic term at
                  all -- its whole cost is in `mixer_params`.

    `mixer_params` is the per-LAYER parameter count of the token mixer's weight
    matrices (q/k/v/o, or the GDN in_proj/out_proj stack). Norms and biases are
    excluded: they are <0.01% of the body and zero GEMM work.
    """
    kind: str
    count: int
    mixer_params: int
    n_heads: int = 0
    head_dim: int = 0
    window: int | None = None

    def __post_init__(self):
        if self.kind not in ("dense", "sliding", "linear"):
            raise ValueError(f"unknown block kind {self.kind!r}")
        if self.kind == "sliding" and not self.window:
            raise ValueError("a sliding block needs a window")


@dataclass(frozen=True)
class ModelShape:
    """Everything the FLOP model needs about a base, and nothing else."""
    name: str
    n_layers: int
    hidden: int
    intermediate: int
    vocab: int
    tied_embeddings: bool
    blocks: tuple[Block, ...]
    #: gate+up+down; 3 for a SwiGLU/GeGLU MLP, 2 for a plain one.
    mlp_matrices: int = 3

    def __post_init__(self):
        got = sum(b.count for b in self.blocks)
        if got != self.n_layers:
            raise ValueError(f"{self.name}: blocks cover {got} layers, "
                             f"n_layers is {self.n_layers}")

    # -- parameters ---------------------------------------------------------

    @property
    def mlp_params(self) -> int:
        return self.n_layers * self.mlp_matrices * self.hidden * self.intermediate

    @property
    def mixer_params(self) -> int:
        return sum(b.count * b.mixer_params for b in self.blocks)

    @property
    def body_params(self) -> int:
        """Decoder-body weight parameters: MLP + token mixers, no norms.

        Excludes the input embedding (a GATHER, zero FLOPs -- billing it is the
        error V7_PERF_LEVERS section 3 corrected) and the lm_head, which is billed
        separately because it only runs on the labelled positions.
        """
        return self.mlp_params + self.mixer_params

    @property
    def lm_head_params(self) -> int:
        return self.hidden * self.vocab

    @property
    def embedding_params(self) -> int:
        return self.hidden * self.vocab

    @property
    def total_params(self) -> int:
        n = self.body_params + self.embedding_params
        if not self.tied_embeddings:
            n += self.lm_head_params
        return n


# ---------------------------------------------------------------------------
# config.json -> ModelShape
# ---------------------------------------------------------------------------

def _text_config(cfg: dict) -> dict:
    """Unwrap a multimodal wrapper. Qwen3.5 and gemma-4 both ship the text
    decoder under `text_config`; a text-only extract has it at top level."""
    inner = cfg.get("text_config")
    return inner if isinstance(inner, dict) else cfg


def shape_from_config(cfg: dict, name: str = "") -> ModelShape:
    """Build a `ModelShape` from a HuggingFace `config.json` dict.

    Handles the three families in flight: gemma-4 unified text (sliding/global
    interleave with two head_dims and two kv-head counts), qwen3.5/3.6 text
    (linear gated-DeltaNet interleaved with gated full attention), and plain
    dense GQA (llama / qwen2 / qwen3 / qwen2.5-coder).

    Raises on anything it cannot describe rather than guessing -- a silently
    wrong numerator is the failure mode this whole module exists to prevent.
    """
    tc = _text_config(cfg)
    name = name or tc.get("model_type") or cfg.get("model_type") or "unnamed"
    n_layers = int(tc["num_hidden_layers"])
    hidden = int(tc["hidden_size"])
    inter = int(tc["intermediate_size"])
    vocab = int(tc["vocab_size"])
    tied = bool(cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings", False)))
    heads = int(tc["num_attention_heads"])
    kv_heads = int(tc.get("num_key_value_heads", heads))
    head_dim = int(tc.get("head_dim") or (hidden // heads))
    window = tc.get("sliding_window")
    layer_types = tc.get("layer_types") or ["full_attention"] * n_layers
    if len(layer_types) != n_layers:
        raise ValueError(f"{name}: layer_types has {len(layer_types)} entries, "
                         f"num_hidden_layers is {n_layers}")

    # gemma-4 gives the global layers their own head_dim and kv-head count.
    g_head_dim = int(tc.get("global_head_dim") or head_dim)
    g_kv_heads = int(tc.get("num_global_key_value_heads") or kv_heads)
    # qwen3.5's `attn_output_gate` widens q_proj to 2 * heads * head_dim; the
    # gate is not a separate module (NEW_BASE_MODELS_LORA_2026-08-05).
    q_mult = 2 if tc.get("attn_output_gate") else 1

    def attn_params(h, hd, kvh):
        q = hidden * (q_mult * h * hd)
        k = hidden * (kvh * hd)
        v = hidden * (kvh * hd)
        o = (h * hd) * hidden
        return q + k + v + o

    def gdn_params():
        """Gated DeltaNet, from the shipped module shapes (verified against
        Qwen3.5-9B's safetensors headers): in_proj_qkv, in_proj_z, in_proj_a,
        in_proj_b, conv1d, out_proj."""
        kh = int(tc["linear_num_key_heads"])
        kd = int(tc["linear_key_head_dim"])
        vh = int(tc["linear_num_value_heads"])
        vd = int(tc["linear_value_head_dim"])
        conv_k = int(tc.get("linear_conv_kernel_dim", 4))
        qkv = 2 * kh * kd + vh * vd
        return (hidden * qkv                 # in_proj_qkv
                + hidden * (vh * vd)         # in_proj_z (output gate)
                + hidden * vh * 2            # in_proj_a + in_proj_b
                + qkv * conv_k               # depthwise conv1d
                + (vh * vd) * hidden)        # out_proj

    counts: dict[str, int] = {}
    for lt in layer_types:
        counts[lt] = counts.get(lt, 0) + 1

    blocks: list[Block] = []
    for lt, n in counts.items():
        if lt == "sliding_attention":
            if not window:
                raise ValueError(f"{name}: sliding layers but no sliding_window")
            blocks.append(Block("sliding", n, attn_params(heads, head_dim, kv_heads),
                                n_heads=heads, head_dim=head_dim, window=int(window)))
        elif lt in ("full_attention", "attention"):
            blocks.append(Block("dense", n, attn_params(heads, g_head_dim, g_kv_heads),
                                n_heads=heads, head_dim=g_head_dim))
        elif lt == "linear_attention":
            blocks.append(Block("linear", n, gdn_params()))
        else:
            raise ValueError(f"{name}: unhandled layer type {lt!r}")

    return ModelShape(name=name, n_layers=n_layers, hidden=hidden,
                      intermediate=inter, vocab=vocab, tied_embeddings=tied,
                      blocks=tuple(sorted(blocks, key=lambda b: b.kind)))


def load_config(path: str, name: str = "") -> ModelShape:
    with open(path) as fh:
        return shape_from_config(json.load(fh), name=name)


# ---------------------------------------------------------------------------
# The FLOP model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Flops:
    """FLOP/token, decomposed. `raw` is what the run executes; `required` swaps
    the sliding term for its banded cost and is what the architecture asks for."""
    model: str
    tw: float
    label_share: float
    body: float
    lm_head: float
    attn_dense: float
    attn_sliding_executed: float
    attn_sliding_required: float
    #: the attention multiplier these terms were built with, so a TIME-weighted
    #: variant can re-scale them without recomputing the shape.
    passes_attention: float = PASSES_ATTENTION

    @property
    def raw(self) -> float:
        return (self.body + self.lm_head + self.attn_dense
                + self.attn_sliding_executed)

    @property
    def required(self) -> float:
        return (self.body + self.lm_head + self.attn_dense
                + self.attn_sliding_required)

    @property
    def wasted(self) -> float:
        """FLOP/token spent on key positions the architecture ignores."""
        return self.attn_sliding_executed - self.attn_sliding_required

    @property
    def wasted_share(self) -> float:
        return self.wasted / self.raw if self.raw else 0.0

    @property
    def banding_speedup(self) -> float:
        """Ceiling on what `ATTN_IMPL=g4_hybrid` can buy at this `tw`, in FLOPs.

        A ratio of FLOPs, so it implicitly prices attention math at the same
        rate as everything else. It does not run at the same rate --
        `banding_speedup_timed` is the same question asked in seconds, and is
        the one to compare against a measured A/B.
        """
        return self.raw / self.required if self.required else 1.0

    @property
    def banding_speedup_timed(self) -> float:
        """Same banding ceiling, priced in SECONDS rather than FLOPs.

        Re-weights only the attention terms from `PASSES_ATTENTION` (4.5,
        derived, FLOPs) to `PASSES_ATTENTION_TIME` (6.85, measured, seconds).
        The measured attention backward costs 4.85x a forward in time against
        2.5x in FLOPs, so a FLOP ratio under-credits any lever that makes the
        backward cheaper -- which banding is.

        At the profiled `Tw = 8,303` this reads **1.420x** against a measured
        **1.51x** (`G4_ATTENTION_PROFILE_2026-08-07` section 4); the FLOP ratio
        reads 1.291x. Closer, and still an underestimate: the residual is the
        non-matmul work inside the kernel, which this model does not bill at
        all. Inherits the shape/hardware caveat on `ATTENTION_BWD_TIME_RATIO` --
        it is a prediction for sm_120 + memory-efficient sdpa, not a law.
        """
        k = PASSES_ATTENTION_TIME / self.passes_attention
        raw_t = self.body + self.lm_head + k * (self.attn_dense
                                                + self.attn_sliding_executed)
        req_t = self.body + self.lm_head + k * (self.attn_dense
                                                + self.attn_sliding_required)
        return raw_t / req_t if req_t else 1.0

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d.update(raw=self.raw, required=self.required, wasted=self.wasted,
                 wasted_share=self.wasted_share,
                 banding_speedup=self.banding_speedup,
                 banding_speedup_timed=self.banding_speedup_timed)
        return d


def flops_per_token(shape: ModelShape, tw: float, *,
                    label_share: float = DEFAULT_LABEL_SHARE,
                    passes_weights: float = PASSES_WEIGHTS,
                    passes_attention: float = PASSES_ATTENTION) -> Flops:
    """FLOP/token for one training token, LoRA + gradient checkpointing.

    `tw` is the TIME-WEIGHTED mean sequence length sum(T^2)/sum(T). Passing a
    median here understates the attention term; see the module docstring.

    Both multipliers are DERIVED counts of algorithmic passes. Do not pass a
    measured time ratio for `passes_attention` -- that makes the result a
    seconds estimate wearing a FLOP label, and `Utilisation` will divide it by a
    FLOP ceiling. `Flops.banding_speedup_timed` is the supported way to get a
    time answer out of this model.
    """
    if tw <= 0:
        raise ValueError("tw (time-weighted mean sequence length) must be > 0")
    if not 0.0 <= label_share <= 1.0:
        raise ValueError("label_share must be in [0, 1]")

    body = passes_weights * FLOPS_PER_MAC * shape.body_params
    lm_head = (passes_weights * FLOPS_PER_MAC
               * shape.lm_head_params * label_share)

    dense = sliding_x = sliding_req = 0.0
    for b in shape.blocks:
        if b.kind == "linear":
            continue
        # QK^T and A@V: 2 matmuls x FLOPS_PER_MAC x heads x head_dim x keys.
        per_key = (passes_attention * FLOPS_PER_MAC * 2
                   * b.n_heads * b.head_dim * b.count)
        if b.kind == "dense":
            # causal skip: each query attends ~half the sequence on average.
            dense += per_key * tw * 0.5
        else:
            # executed: a bool window mask forfeits the causal skip entirely,
            # so every query pays the full T. required: a banded kernel touches
            # ~`window` keys per query.
            sliding_x += per_key * tw
            sliding_req += per_key * min(tw, float(b.window or tw))

    return Flops(model=shape.name, tw=float(tw), label_share=label_share,
                 body=body, lm_head=lm_head, attn_dense=dense,
                 attn_sliding_executed=sliding_x,
                 attn_sliding_required=sliding_req,
                 passes_attention=float(passes_attention))


# ---------------------------------------------------------------------------
# The denominator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ceiling:
    """A dense-bf16 GEMM ceiling AND the device it belongs to.

    Constructing one without a device name raises. That is the whole point: a
    reviewer's 25-30% and our 48% for the same run differed only in this number
    (V7_THROUGHPUT_AUDIT section 4), and vendor headline TFLOPS are routinely
    sparse and/or FP8.
    """
    device: str
    tflops: float
    source: str = ""
    #: per-GEMM-class TFLOP/s this was weighted from, when it was weighted
    per_shape: dict = field(default_factory=dict)
    #: ``model``   the GEMMs were run at THIS model's own (M,K,N) — `gemm_shapes`.
    #: ``generic`` they were run at a fixed three-aspect-class set, which is what
    #:             `tools/vast/gemm_probe.py` does at box boot because no base is
    #:             known there. The class-resolved weighting below still applies
    #:             (that is the whole point of `mac_mix`), but the within-class
    #:             K/N differ from this model's, so the result is an
    #:             approximation and `utilisation()` says so in its note.
    shape_basis: str = "model"

    def __post_init__(self):
        if not (self.device or "").strip():
            raise DenominatorError(
                "a TFLOP/s figure with no device attached is not quotable — "
                "pass the device string from "
                "torch.cuda.get_device_properties().name (gemm_ceiling.py "
                "records it for exactly this reason)")
        if self.tflops <= 0:
            raise DenominatorError(f"ceiling must be > 0, got {self.tflops}")

    @classmethod
    def from_gemm_ceiling_json(cls, blob: dict, *, weights: dict | None = None
                               ) -> "Ceiling":
        """Build from `gemm_ceiling.py --json` output.

        Also reads `tools/vast/gemm_probe.py`'s record, which emits the same
        schema as a superset (plus `shape_basis`, `power_limit_w`, throttle bits)
        so the boot-time and bench-time instruments need no format branch here.
        `hostfacts.py ceiling --machine <M>` prints exactly such a blob.

        With `weights` (a GEMM-class -> FLOP-share map, e.g. `ModelShape.mac_mix`)
        the result is the FLOP-WEIGHTED HARMONIC MEAN over the model's own MAC
        mix, which is the correct denominator -- not the headline max. Without
        weights it is the max across shapes, i.e. the honest but optimistic
        "ceiling", and `source` says so.
        """
        device = (blob.get("device") or "").strip()
        rows = blob.get("shapes") or []
        if not rows:
            raise DenominatorError("gemm_ceiling json carries no shapes")
        by_class = {classify_gemm(r["k"], r["n"]): float(r["tflops"]) for r in rows}
        env = (f"torch {blob.get('torch')} / cuda {blob.get('cuda')} / "
               f"{blob.get('capability')}")
        basis = str(blob.get("shape_basis") or "model")
        if basis != "model":
            env += f" / {basis} shapes"
        tool = "gemm_probe.py" if blob.get("probe_version") else "gemm_ceiling.py"
        if weights:
            tf = harmonic_weighted(weights, by_class)
            return cls(device=device, tflops=tf,
                       source=f"{tool} FLOP-weighted, {env}",
                       per_shape=by_class, shape_basis=basis)
        return cls(device=device, tflops=max(by_class.values()),
                   source=f"{tool} max-across-shapes, {env}",
                   per_shape=by_class, shape_basis=basis)


def classify_gemm(k: int, n: int) -> str:
    """Coarse GEMM class from its K/N aspect. The three shapes gemm_ceiling.py
    benches by default map onto exactly these."""
    if n > k * 2:
        return "mlp_up"
    if k > n * 2:
        return "mlp_down"
    return "attn_proj"


#: GEMM classes with no bench shape of their own, and the measured class whose
#: aspect ratio stands in for them. The lm_head is (hidden x vocab) — a wide-N
#: shape like gate/up, and V7_PERF_LEVERS §2 weighted it at the gate/up rate for
#: exactly this reason. Substituting is a stated approximation, not a guess.
GEMM_CLASS_FALLBACK = {"lm_head": "mlp_up"}


def harmonic_weighted(weights: dict, tflops_by_class: dict) -> float:
    """FLOP-weighted harmonic mean. Harmonic because we are averaging RATES over
    a fixed amount of work: total_time = sum(w_i / r_i), so the effective rate is
    sum(w_i) / sum(w_i / r_i). An arithmetic mean here overstates the ceiling.
    """
    num = 0.0
    den = 0.0
    for cls, w in weights.items():
        if w <= 0:
            continue
        r = tflops_by_class.get(cls)
        if r is None:
            r = tflops_by_class.get(GEMM_CLASS_FALLBACK.get(cls, ""))
        if r is None:
            raise DenominatorError(
                f"no measured TFLOP/s for GEMM class {cls!r} "
                f"({100 * w:.1f}% of this model's weight FLOPs) — measure it or "
                f"drop the weighting; a partial weighting is a silent lie")
        num += w
        den += w / r
    if den <= 0:
        raise DenominatorError("empty weighting")
    return num / den


def mac_mix(shape: ModelShape, *,
            label_share: float = DEFAULT_LABEL_SHARE) -> dict:
    """This model's own weight-GEMM FLOP mix, as fractions summing to 1.

    The O(T^2) attention math is deliberately EXCLUDED: it is not a weight GEMM
    and does not run at GEMM efficiency, so folding it into a GEMM-ceiling
    weighting would be comparing two different machines.
    """
    p = PASSES_WEIGHTS * FLOPS_PER_MAC
    parts = {
        "attn_proj": p * shape.mixer_params,
        "mlp_up": p * shape.n_layers * (shape.mlp_matrices - 1)
        * shape.hidden * shape.intermediate,
        "mlp_down": p * shape.n_layers * shape.hidden * shape.intermediate,
        "lm_head": p * shape.lm_head_params * label_share,
    }
    tot = sum(parts.values())
    return {k: v / tot for k, v in parts.items()}


def gemm_shapes(shape: ModelShape, m: int = 12288) -> list[tuple[int, int, int]]:
    """The (M, K, N) triples `gemm_ceiling.py` should be run at to produce a
    denominator for THIS model. Measuring at a square 8192^3 instead is how a
    ceiling ends up 12% off before any architecture question enters."""
    return [(m, shape.hidden, shape.hidden),
            (m, shape.hidden, shape.intermediate),
            (m, shape.intermediate, shape.hidden)]


# ---------------------------------------------------------------------------
# The division
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Utilisation:
    model: str
    device: str
    tok_s: float
    flops: Flops
    ceiling: Ceiling
    #: True when the ceiling was measured on a device other than `device`.
    provisional: bool
    note: str = ""

    @property
    def achieved_tflops(self) -> float:
        return self.flops.raw * self.tok_s / 1e12

    @property
    def achieved_tflops_required(self) -> float:
        return self.flops.required * self.tok_s / 1e12

    @property
    def mfu_raw(self) -> float:
        """**roof-HFU on executed work.** Key name frozen for the wire schema;
        the QUANTITY is HFU-like, not MFU — see the module docstring's
        "WHAT THIS METRIC IS" block. Numerator bills grad-ckpt recompute AND
        gemma-4's executed-but-unrequired dense sliding attention; denominator
        is a measured GEMM roof. Not comparable to published MFU."""
        return self.achieved_tflops / self.ceiling.tflops

    @property
    def mfu_required(self) -> float:
        """**roof-HFU on architecture-required work.** Same caveats as
        `mfu_raw` minus the sliding waste; grad-ckpt recompute is still billed,
        so this is still HFU-like and still not published-MFU."""
        return self.achieved_tflops_required / self.ceiling.tflops

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "device": self.device,
            "tok_s": self.tok_s,
            "gflop_per_token_raw": round(self.flops.raw / 1e9, 3),
            "gflop_per_token_required": round(self.flops.required / 1e9, 3),
            "achieved_tflops": round(self.achieved_tflops, 1),
            "achieved_tflops_required": round(self.achieved_tflops_required, 1),
            "ceiling_tflops": self.ceiling.tflops,
            "ceiling_device": self.ceiling.device,
            "ceiling_source": self.ceiling.source,
            "ceiling_shape_basis": self.ceiling.shape_basis,
            "mfu_raw": round(self.mfu_raw, 4),
            "mfu_required": round(self.mfu_required, 4),
            "provisional": self.provisional,
            "note": self.note,
        }


def utilisation(shape: ModelShape, *, tw: float, tok_s: float,
                device: str, ceiling: Ceiling,
                label_share: float = DEFAULT_LABEL_SHARE) -> Utilisation:
    """roof-HFU for one run (see the module docstring: HFU-like, not MFU).
    Refuses without a device name for the RUN as well as for the ceiling --
    "which box was this?" is half of every throughput postmortem we have
    written."""
    if not (device or "").strip():
        raise DenominatorError(
            "utilisation() needs the device the run was measured on; an MFU "
            "with no device is not quotable (see gemm_ceiling.py)")
    if tok_s <= 0:
        raise ValueError("tok_s must be > 0")
    f = flops_per_token(shape, tw, label_share=label_share)
    borrowed = _norm_device(device) != _norm_device(ceiling.device)
    notes = []
    if borrowed:
        notes.append(
            f"PROVISIONAL: ceiling measured on {ceiling.device!r}, run on "
            f"{device!r}. The two SKUs differ; treat the MFU as a band, not "
            f"a point estimate, until gemm_ceiling.py is run on the run's "
            f"own device.")
    if ceiling.shape_basis != "model":
        # NOT provisional: the device is right, only the within-class K/N are
        # approximated. Named anyway, with the exact upgrade path.
        notes.append(
            f"APPROXIMATE SHAPES: the ceiling was measured at "
            f"{ceiling.shape_basis} GEMM shapes, not {shape.name}'s own. The "
            f"per-class weighting still applies (that is what mac_mix is for), "
            f"but the within-class K/N differ. To remove the approximation run "
            f"the shapes `mfu.py --model {shape.name} --gemm-cmd` prints.")
    return Utilisation(model=shape.name, device=device, tok_s=float(tok_s),
                       flops=f, ceiling=ceiling, provisional=borrowed,
                       note="  ".join(notes))


def _norm_device(s: str) -> str:
    return " ".join((s or "").split()).lower()


# ---------------------------------------------------------------------------
# Known bases (regression pins live in test_mfu.py)
# ---------------------------------------------------------------------------

def gemma4_12b_text() -> ModelShape:
    """google/gemma-4-12B, text decoder. Mirrors
    tools/vast/testfixtures/gemma4-12b-text.config.json.
    """
    return shape_from_config({
        "num_hidden_layers": 48, "hidden_size": 3840, "intermediate_size": 15360,
        "vocab_size": 262144, "tie_word_embeddings": True,
        "num_attention_heads": 16, "num_key_value_heads": 8,
        "head_dim": 256, "global_head_dim": 512,
        "num_global_key_value_heads": 1, "sliding_window": 1024,
        "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 8,
    }, name="gemma-4-12b-text")


def qwen35_9b_text() -> ModelShape:
    """Qwen/Qwen3.5-9B, text decoder (the multimodal wrapper's vision tower and
    the MTP head are not on the training path and carry no FLOPs here)."""
    return shape_from_config({
        "num_hidden_layers": 32, "hidden_size": 4096, "intermediate_size": 12288,
        "vocab_size": 248320, "tie_word_embeddings": False,
        "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
        "attn_output_gate": True,
        "linear_num_key_heads": 16, "linear_key_head_dim": 128,
        "linear_num_value_heads": 32, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 8,
    }, name="qwen3.5-9b-text")


KNOWN = {
    "gemma-4-12b-text": (gemma4_12b_text, TW_V9_GEMMA4),
    "qwen3.5-9b": (qwen35_9b_text, TW_V9_QWEN35),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt_g(x: float) -> str:
    return f"{x / 1e9:,.2f}"


def _report(shape: ModelShape, f: Flops, u: Utilisation | None) -> str:
    out = [f"model            {shape.name}",
           f"  layers         {shape.n_layers} "
           + " + ".join(f"{b.count}x{b.kind}"
                        + (f"(hd {b.head_dim})" if b.head_dim else "")
                        + (f"(win {b.window})" if b.window else "")
                        for b in shape.blocks),
           f"  hidden/inter   {shape.hidden} / {shape.intermediate}",
           f"  vocab          {shape.vocab:,}"
           + ("  (tied)" if shape.tied_embeddings else "  (untied lm_head)"),
           f"  body params    {shape.body_params / 1e9:.3f} B"
           f"   total {shape.total_params / 1e9:.3f} B",
           "",
           f"FLOP/token at tw={f.tw:,.0f}, label_share={f.label_share:.4f}",
           f"  body (weights)         {_fmt_g(f.body):>8} GFLOP",
           f"  lm_head                {_fmt_g(f.lm_head):>8}",
           f"  attention, dense       {_fmt_g(f.attn_dense):>8}",
           f"  attention, sliding     {_fmt_g(f.attn_sliding_executed):>8}"
           f"   (executed, dense T x T)",
           f"                         {_fmt_g(f.attn_sliding_required):>8}"
           f"   (required, banded)",
           f"  ---------------------- {'-' * 8}",
           f"  RAW (executed)         {_fmt_g(f.raw):>8} GFLOP/token",
           f"  REQUIRED (architecture){_fmt_g(f.required):>8}",
           f"  wasted                 {_fmt_g(f.wasted):>8}"
           f"   ({100 * f.wasted_share:.1f}% of raw; banding ceiling "
           f"{f.banding_speedup:.3f}x)",
           f"  banding ceiling in TIME{f.banding_speedup_timed:>8.3f}x"
           f"   (attention re-weighted 4.5 -> 6.85, MEASURED on sm_120 +"
           f" memeff sdpa)"]
    if u is None:
        out += ["", "no --tok-s / ceiling given: numerator only. A roof-HFU "
                    "needs a measured ceiling AND a device name."]
        return "\n".join(out)
    out += ["",
            f"throughput       {u.tok_s:,.0f} tok/s on {u.device!r}",
            f"  achieved       {u.achieved_tflops:.1f} TFLOP/s (raw work)   "
            f"{u.achieved_tflops_required:.1f} TFLOP/s (required work)",
            f"  ceiling        {u.ceiling.tflops:.1f} TFLOP/s on "
            f"{u.ceiling.device!r}",
            f"                 {u.ceiling.source}" if u.ceiling.source else "",
            f"  roof-HFU raw   {100 * u.mfu_raw:.1f}%"
            f"   (executed FLOPs incl. grad-ckpt recompute / measured GEMM "
            f"roof — NOT comparable to published MFU)",
            f"  roof-HFU req.  {100 * u.mfu_required:.1f}%"
            f"   (sliding waste removed; recompute still billed)"]
    if u.note:
        out += ["", f"** {u.note}"]
    return "\n".join(x for x in out if x != "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", choices=sorted(KNOWN),
                     help="a base with a checked-in shape")
    src.add_argument("--config", help="path to a HuggingFace config.json")
    ap.add_argument("--name", default="", help="label for --config")
    ap.add_argument("--tw", type=float, default=0.0,
                    help="TIME-WEIGHTED mean sequence length sum(T^2)/sum(T). "
                         "Not the median. Defaults to the v9 corpus value for "
                         "a --model.")
    ap.add_argument("--label-share", type=float, default=DEFAULT_LABEL_SHARE,
                    help=f"share of positions carrying loss "
                         f"(default {DEFAULT_LABEL_SHARE}, the v9 corpus)")
    ap.add_argument("--tok-s", type=float, default=0.0,
                    help="measured training tokens/s")
    ap.add_argument("--device", default="",
                    help="device the run was measured on "
                         "(torch.cuda.get_device_properties().name)")
    ap.add_argument("--ceiling-tflops", type=float, default=0.0)
    ap.add_argument("--ceiling-device", default="",
                    help="device the CEILING was measured on; defaults to "
                         "--device. Differing marks the MFU provisional.")
    ap.add_argument("--ceiling-source", default="",
                    help="where the ceiling number came from")
    ap.add_argument("--ceiling-json", default="",
                    help="a gemm_ceiling.py --json / gemm_probe.py record "
                         "(same schema); weighted over this model's own MAC "
                         "mix. `hostfacts.py ceiling --machine <M>` prints one "
                         "for a host we have probed.")
    ap.add_argument("--gemm-cmd", action="store_true",
                    help="print the gemm_ceiling.py invocation that measures "
                         "this model's own shapes, and exit")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.model:
        builder, default_tw = KNOWN[args.model]
        shape = builder()
    else:
        shape, default_tw = load_config(args.config, name=args.name), 0.0
    tw = args.tw or default_tw
    if not tw:
        ap.error("--tw is required for --config (the time-weighted mean "
                 "sequence length sum(T^2)/sum(T) of the corpus)")

    if args.gemm_cmd:
        shapes = " ".join(f"--shape {m}x{k}x{n}"
                          for m, k, n in gemm_shapes(shape))
        print(f"python3 tools/vast/jobcommon/gemm_ceiling.py "
              f"{shapes} --json gemm_ceiling.json")
        mix = mac_mix(shape, label_share=args.label_share)
        print("# weight-FLOP mix for the harmonic weighting: "
              + ", ".join(f"{k} {100 * v:.1f}%" for k, v in sorted(mix.items())))
        return 0

    f = flops_per_token(shape, tw, label_share=args.label_share)

    ceil = None
    if args.ceiling_json:
        with open(args.ceiling_json) as fh:
            ceil = Ceiling.from_gemm_ceiling_json(
                json.load(fh), weights=mac_mix(shape, label_share=args.label_share))
    elif args.ceiling_tflops:
        dev = args.ceiling_device or args.device
        if not dev.strip():
            ap.error("--ceiling-tflops needs --ceiling-device (or --device): a "
                     "TFLOP/s figure with no device attached is not quotable")
        ceil = Ceiling(device=dev, tflops=args.ceiling_tflops,
                       source=args.ceiling_source)

    u = None
    if args.tok_s and ceil is not None:
        u = utilisation(shape, tw=tw, tok_s=args.tok_s,
                        device=args.device or ceil.device, ceiling=ceil,
                        label_share=args.label_share)
    elif args.tok_s:
        print("!! --tok-s given with no ceiling: reporting the numerator only. "
              "Run gemm_ceiling.py on the run's own device (see --gemm-cmd) "
              "rather than borrowing a number from another SKU.", file=sys.stderr)

    if args.json:
        blob = {"shape": {"name": shape.name, "n_layers": shape.n_layers,
                          "hidden": shape.hidden,
                          "intermediate": shape.intermediate,
                          "vocab": shape.vocab,
                          "body_params": shape.body_params,
                          "total_params": shape.total_params},
                "flops": f.as_dict(),
                "mac_mix": mac_mix(shape, label_share=args.label_share)}
        if u is not None:
            blob["utilisation"] = u.as_dict()
        print(json.dumps(blob, indent=2))
    else:
        print(_report(shape, f, u))
    return 0


if __name__ == "__main__":
    sys.exit(main())
