"""gpu_rates.py — MEASURED training throughput per (GPU class, job shape, world size).

WHY THIS FILE EXISTS
--------------------
Owner ruling 2026-08-16 (`docs/plans/vast-fleet-sessions/
2026-08-16-replacement-probe-tpd/SESSION.md`): a box is selected by **minimum
requirements first, then best tokens per dollar** — upgrading the card is the
right call whenever it is the better deal. Ranking on `$/hr` alone cannot say
that, because $/hr is a price and the thing being bought is *tokens*. This
module is the missing numerator: `tok/s`, per card class, per job shape,
per card count, every number derived from a run artifact this repo can point at.

It is deliberately **stdlib-only and import-light** (no `herdd` import, no
network, no file reads at import) so any caller — `herdd`, `fleetd`, a
one-off probe — can import it for the cost of parsing this file.

WHAT A RATE HERE IS, EXACTLY
----------------------------
    tok_s = tokens_per_optimizer_step / seconds_per_optimizer_step

measured in **steady state** — after model load and step-1 JIT, before publish.
It EXCLUDES per-job fixed costs (boot, base-model pull, tokenizer pass, the
~107 s `fla` cold compile on the 9B, final upload), which on a real job are
minutes, not seconds: w5a spent ~8 min of a 107 min job outside stepping. A
tokens-per-dollar figure from this table is therefore an **upper bound on the
tokens a rental hour actually buys**; it is the right quantity for *ranking*
candidate boxes against each other (the fixed costs are card-independent to
first order) and the wrong quantity for budgeting a job's absolute cost.

Tokens are **non-padding** tokens (`attention_mask.sum()`, TRL's `num_tokens`),
so a shape with `BATCH=1` (every production bundle here) has no padding waste
and `tok_s` is directly comparable across cards at the same shape.

READ THIS BEFORE COMPARING TWO CARDS
------------------------------------
**Only compare rates at the SAME shape key.** `tok_s` is not shape-invariant:
the same RTX PRO 6000 WS 600W does 4,336 tok/s on the 7B at a 12,288 window and
2,519 tok/s on the 9B at 40,960. Ranking an H100 NVL's `9b-w20480-dec` number
against a 5090's `7b-w12288-fit` number measures the *shapes*, not the cards.
`rate_for(..., shape=None)` exists for the caller who has no shape in hand and
is documented below to be conservative — but a cross-card ranking built on it
is only as good as the shapes that happened to be measured.

MULTI-CARD IS PROVISIONAL, AND THE BOUNDARY HAS A DATE
------------------------------------------------------
The DDP path was optimized on **2026-08-14**: commit `e48def36` made
`TELEMETRY_TOKEN_COUNT=0` + `DDP_METRIC_GATHER=deferred` the trainer defaults,
worth a measured **+40.7% at W=4 / +26.7% at W=8** (`docs/plans/witness/perf/
W8_LADDER_RESULT_2026-08-14.md`), followed same-day by `fb164ed6` (detach the
held fold — a VRAM fix, time-neutral and time-verified by D3D). **Every
multi-card measurement taken before `e48def36` understates the current path by
those factors and is not in this table.** The multi-card entries below are all
post-boundary, and are still flagged `provisional` for the token-count reason
recorded on each. Single-card rates are unaffected by the boundary — the W=1
null was measured 3× inside 0.05% (W8 ladder §1).

Standing procurement rule this table does NOT encode, and callers must apply
separately: **stop a single DDP job at 4 cards on PCIe boxes** — cards 5–8
deliver 33.6% marginal (`tools/vast/TRAINING.md` §1). The W=8 entry exists so a
whole-box hold can be priced, not to recommend one.

UPDATING
--------
`tools/vast/GPU_RATES.md` — "Adding a rate" — is the procedure, and it is one
paragraph. Do not hand-author an entry: every `tok_s` below is division, and
the two numbers divided are named in its `provenance`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "Rate",
    "RATES",
    "SHAPES",
    "DEFAULT_SHAPE",
    "MULTI_GPU_SCALING",
    "normalize_gpu_name",
    "entry_for",
    "rate_for",
    "tokens_per_dollar",
    "shapes_for",
    "build_index",
]


# --------------------------------------------------------------------------- #
# shape classes — the job shapes we actually run
# --------------------------------------------------------------------------- #
# A shape key names everything that changes tok/s at a fixed card: base model,
# window, micro-batch/accumulation, precision, and grad-checkpointing. Two runs
# sharing a key are comparable; two runs that do not, are not.
SHAPES = {
    "9b-w20480-dec": (
        "qwen35-9b (GDN, fla 0.4.2), MAX_SEQ 20480, B=1 x GA=32, bf16, "
        "grad-ckpt on, LoRA r32 all-linear, sdpa/flashmeff. The CURRENT "
        "production training shape: v12 / w4t / w5a / w5b bundles."
    ),
    "9b-w40960-dec": (
        "as 9b-w20480-dec but MAX_SEQ 40960 — the v8 / v9 window."
    ),
    "9b-w32768-v16": (
        "qwen35-9b (GDN, fla 0.5.2), MAX_SEQ 32768, B=1 x GA=32, bf16, "
        "grad-ckpt on, LoRA r64 all-linear, sdpa/flashmeff, v16 v5chat "
        "prod-mix. The v14/v16 rank-ladder lineage's shape."
    ),
    "9b-w32768-v16-hybridgc": (
        "as 9b-w32768-v16 but GRAD_CKPT=hybrid (runtime self-calib) — the "
        "shape doctrine's recommended big-card variant of the v16 recipe."
    ),
    "7b-w32768-prod": (
        "qwen25-coder-7b-instruct, MAX_SEQ 32768, B=1 x GA=32, bf16, "
        "grad-ckpt on, sdpa. The v11 lane's production shape."
    ),
    "7b-w12288-fit": (
        "qwen25-coder-7b-instruct, MAX_SEQ 12288, B=1 x GA=32, bf16, "
        "grad-ckpt on, sdpa. The perf-levers bench slice — the most-measured "
        "shape in the bank and the one with a self-anchored W=1/4/8 ladder."
    ),
    "7b-w4096-fit": (
        "as 7b-w12288-fit but MAX_SEQ 4096."
    ),
    "9b-w12288-fit": (
        "qwen35-9b, MAX_SEQ 12288, B=1 x GA=8, bf16, grad-ckpt on, sdpa, "
        "fla 0.4.2. The k1-flashqla bench slice."
    ),
}

# What `shape=None` resolves to first. The current production training shape:
# if a caller has not said what it is about to run, this is what it is most
# likely about to run.
DEFAULT_SHAPE = "9b-w20480-dec"


@dataclass(frozen=True)
class Rate:
    """One measured (gpu, num_gpus, shape) -> tokens/sec cell.

    `measured`     — True iff tok_s is a division of two numbers read out of a
                     run artifact. There is no other kind of entry in this file
                     today; a modelled or vendor-spec number would set it False
                     and must never be added without saying so here.
    `provisional`  — the number is real but something about it is weaker than a
                     production run: few timed steps, a token count borrowed
                     from a paired arm, an unresolved card SKU. `note` says
                     which. Consumers should prefer a non-provisional cell when
                     both exist, and may want to discount provisional ones.
    `provenance`   — repo-relative path or `b2:` URI, never an absolute path.
    """

    gpu: str
    num_gpus: int
    shape: str
    tok_s: float
    measured: bool
    provisional: bool
    provenance: str
    date: str
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "gpu": self.gpu, "num_gpus": self.num_gpus, "shape": self.shape,
            "tok_s": self.tok_s, "measured": self.measured,
            "provisional": self.provisional, "provenance": self.provenance,
            "date": self.date, "note": self.note,
        }


# --------------------------------------------------------------------------- #
# THE TABLE
# --------------------------------------------------------------------------- #
# Ordered by card, then card count, then shape. Every tok_s is
# tokens_per_step / seconds_per_step with both terms named in `provenance`.
RATES: tuple[Rate, ...] = (
    # ======================================================================= #
    # The 2026-08-16 `gpu-rate-9b-w20480` sweep. Five of the six cells below
    # come from ONE bundle run unchanged on five cards inside one hour, which
    # is what makes them cross-comparable in a way the pre-existing rows are
    # not: same trainer, same knobs, same 1,024-row slice, IDENTICAL
    # 1,632,983 tokens over 13 steps on every card. Bundle:
    # tools/witness/jobs/gpu-rate-9b-w20480 (see its run.sh header for the
    # three deliberate differences from its fit-ladder ancestor). Session
    # record: <upstream-bench>/archive/runs/2026-08-16-gpu-rate-bench-48gb/.
    #
    # tok/step is 125,614 for every one of them, and that number is the
    # sweep's own validity check: the H100 NVL anchor below measured 126,739
    # tok/step off a full w5a training run, so the slice reproduces the
    # production length mix to 0.9% and these rows land in the SAME shape cell
    # rather than merely sharing its name.
    #
    # SECONDS come from `throughput.step_time_p50_s` (GPU_RATES.md's "Adding a
    # rate"), meaned over the two repeats. A p50 over 13 steps already excludes
    # the one ~165 s Triton-compile step, which is why it agrees with the
    # warmup-excluding tail figure to 2-7% on every card; where they differ the
    # p50 is the SLOWER of the two and is used, so each cell is conservative.
    # ======================================================================= #
    # ---- H200 NVL ---------------------------------------------------------
    Rate(
        gpu="H200 NVL", num_gpus=1, shape="9b-w20480-dec", tok_s=4578.0,
        measured=True, provisional=False, date="2026-08-16",
        provenance=(
            "job 20260816T043729-gpu-rate-9b-w20480-8b49 (box 47839942, offer "
            "45172707 / machine 144381, spot $0.4969/hr); 125,614 tok/step "
            "(1,632,983 / 13) / 27.4421 s mean step p50 (27.2967, 27.5875). "
            "out/jobs/20260816T043729-gpu-rate-9b-w20480-8b49/out/"
            "prod_w20480_r{1,2}/train_summary.json"
        ),
        note=(
            "13 timed steps x 2 repeats (26 total), p50 spread 1.06%. Card "
            "confirmed `NVIDIA H200 NVL` in hardware.gpu_names. Peak VRAM "
            "32.94 GB reserved. GEMM ceiling 704.3 TFLOP/s bf16. THE CELL THE "
            "2026-08-16 INCIDENT NEEDED: machine 144381 is the $0.4027 H200 "
            "offer fleetd's replacement probe dropped on a GPU-name pin before "
            "buying an on-demand H100 NVL at $1.603 (SESSION.md). At the price "
            "actually paid this is 33.2M tok/$ against that box's 9.0M — the "
            "H200 was ~3.7x the better buy, which is the ruling's whole point."
        ),
    ),
    Rate(
        gpu="H200 NVL", num_gpus=1, shape="9b-w32768-v16", tok_s=4982.1,
        measured=True, provisional=True, date="2026-08-28",
        provenance=(
            "job 20260828T110646-h200-v16w32k-gc-probe-6828 (box 49003486, "
            "on-demand $3.6139/hr); gc_collect tok_s_p50 mean of gcon_v16_a/b "
            "(4,683.5 / 5,280.7). ../upstream-bench/archive/runs/"
            "2026-08-28-h200-v16w32k-gc-probe/"
        ),
        note=(
            "The v16 rank-ladder recipe verbatim (GRAD_CKPT on, MAX_SEQ 32768, "
            "r64, sdpa/flashmeff, v16 v5chat prod-mix slice). Provisional: 13 "
            "timed steps per twin and an 11% twin spread (25.11 vs 22.27 s/it)."
        ),
    ),
    Rate(
        gpu="H200 NVL", num_gpus=1, shape="9b-w32768-v16-hybridgc", tok_s=7142.6,
        measured=True, provisional=True, date="2026-08-28",
        provenance=(
            "same job, hybridrt_v16_a/b tok_s_p50 mean (7,012.6 / 7,272.5); "
            "../upstream-bench/archive/runs/2026-08-28-h200-v16w32k-gc-probe/"
        ),
        note=(
            "Same recipe with GRAD_CKPT=hybrid via the RUNTIME self-calibrator "
            "(trainer ac51ed6dc, calib_source=runtime, no calib file) — 1.41x "
            "over the gcon twin mean, loss-identity PASS, peak 44.57 GiB. "
            "Provisional: 13 timed steps per twin. First banked self-calib "
            "number; the w20480 file-calib ladder read 1.45x on this card."
        ),
    ),
    # ---- B200 -------------------------------------------------------------
    Rate(
        gpu="B200", num_gpus=1, shape="9b-w20480-dec", tok_s=5555.0,
        measured=True, provisional=False, date="2026-08-16",
        provenance=(
            "job 20260816T045324-gpu-rate-9b-w20480-2abf (box 47840777, "
            "machine 56359, spot $5.6064/hr); 125,614 tok/step / 22.6150 s "
            "mean step p50 (22.8066, 22.4233). "
            "out/jobs/20260816T045324-gpu-rate-9b-w20480-2abf/out/"
            "prod_w20480_r{1,2}/train_summary.json"
        ),
        note=(
            "26 timed steps, p50 spread 1.69%. THE FASTEST CARD MEASURED AND "
            "THE WORST BUY: 5,555 tok/s is 1.21x the H200's, at 11.3x the "
            "price actually paid, so 3.6M tok/$ vs 33.2M — an order of "
            "magnitude behind. Benchmarked because the owner asked to KNOW, "
            "not because it should be rented. Second finding, and the one with "
            "a shelf life: fla 0.4.2's GDN chunk rule PASSES the 9/9 gradient "
            "probe on **sm_100** (forward + all five gradients vs an fp32 "
            "torch reference, 3 seq lengths x 3 seeds). Production pins "
            "EXPECT_SM '90,120' and had never seen this silicon."
        ),
    ),
    # ---- H100 NVL ---------------------------------------------------------
    Rate(
        gpu="H100 NVL", num_gpus=1, shape="9b-w20480-dec", tok_s=4050.0,
        measured=True, provisional=False, date="2026-08-16",
        provenance=(
            "run 20260816T003445-w5a-chainmass-rb3-9b-w20480-dec-train-fb51; "
            "126,739 tok/step / 31.30 s/step over steps 16->176. Step times "
            "from the SAVE_STEPS=16 checkpoint ModTimes in "
            "b2:example-runs-bucket/jobs/20260816T003445-w5a-chainmass-rb3-9b-"
            "w20480-dec-train-fb51/checkpoints/out/checkpoint-*/"
            "trainer_state.json; token counts from the cumulative `num_tokens` "
            "in out/jobs/20260816T003445-w5a-chainmass-rb3-9b-w20480-dec-train"
            "-fb51/out/checkpoint-181/trainer_state.json"
        ),
        note=(
            "INDEPENDENTLY CONFIRMED 2026-08-16 to 0.15%. The gpu-rate-9b-"
            "w20480 bundle, a different harness on a different box (job "
            "20260816T043818-...-cc7f, box 47841874), measured 31.000 and "
            "30.930 s tail s/it on this card at this shape => 4,052 / 4,061 "
            "tok/s against this row's 4,050. No cell was added for it: this "
            "anchor comes from 160 steps of a real training run and is the "
            "better number; the 08-16 run's value is that it AGREES. Its own "
            "p50 (32.58 s => 3,855 tok/s, 4.9% slower) is NOT the card -- that "
            "box landed on machine 147577, co-tenant to the live w5b run, and "
            "its step time drifted 33->40 s/it while the neighbour was hot and "
            "recovered to 31 s/it when w5b finished, at 83% GPU util "
            "throughout. Tenancy, not silicon; see "
            "docs/plans/vast-fleet-sessions/2026-08-16-replacement-probe-tpd/"
            "GPU_BENCH_RESULTS.md finding 6. "
            "11 consecutive 16-step intervals gave 3,998-4,119 tok/s (spread "
            "3.0%). INCLUDES the 10 checkpoint saves in that window (~6 s each, "
            "~1.2% of wall), so it is a hair conservative vs a pure step p50 -- "
            "which is exactly the ~1% by which the 08-16 pure-step tail sits "
            "above it. "
            "The run's own train_summary.json throughput block is UNUSABLE — "
            "the synced artifact is a 1-step resume segment (`segment_steps`: "
            "1, `wall_seconds`: 0.47) and reports 48.8M tok/s. Card confirmed "
            "`NVIDIA H100 NVL` in that file's `hardware.gpu_names`."
        ),
    ),
    # ======================================================================= #
    # The 2026-08-16 A100 sweep — the SAME bundle, one field changed.
    # <upstream-bench>/archive/runs/2026-08-16-gpu-rate-bench-a100/.
    #
    # `needs.gpu_ram_gb` went 48 -> 40, because jobd fails a ticket outright
    # when the declaration exceeds the largest card on the box and 48 therefore
    # made a 40 GB card unrentable AND unrunnable. That field is a rental /
    # admission filter, not a shape knob: every knob the trainer sees is
    # byte-identical to the four cells above, and the proof is in the data —
    # all four A100s reported the SAME 1,632,983 tokens over 13 steps
    # (125,614 tok/step) as the B200, H200 NVL and Max-Q did. Same cell, not a
    # look-alike.
    #
    # WHAT THE SWEEP SETTLED, beyond four rates:
    #   * THE PRODUCTION SHAPE RUNS ON A 40 GB CARD. Peak was 32.900 GB
    #     reserved / 32.060 GB allocated on all four A100s — the same figure to
    #     three decimals as every 48 GB+ card measured, i.e. eight cards across
    #     four architectures now agree. On the 40 GB parts (42.4 GB usable) that
    #     is ~9.5 GB of headroom, and all eight are harvested into
    #     vram_facts.json. NOT a clearance for w5a/w5b/v12/w4t, which pin
    #     `gpu_ram_gb: 41` and exclude the A100 40 GB deliberately: their policy
    #     carries a +3.80 GB CORPUS-SCATTER term, and this sweep ran one slice
    #     of one corpus eight times, so it cannot see that term. The eight
    #     anchors land BELOW the 35.13 group max they are compared against and
    #     therefore confirm the model rather than lower it. Closing the gap
    #     needs same-shape peaks on DIFFERENT corpora, not more repeats.
    #   * fla 0.4.2's GDN chunk rule PASSES on sm_80 — 4/4 boxes, verdict PASS,
    #     worst cosine 0.999973743651698 on every one of them, bit-identical
    #     across PCIe and SXM4 and both memory sizes. Production pins
    #     EXPECT_SM "90,120"; Ampere is a WIDENING DECISION for the owner, not
    #     a measurement gap. See GPU_BENCH_RESULTS.md.
    #   * The four parts are NOT one class. 2,134 -> 2,549 tok/s is a 1.19x
    #     spread, which is why normalize_gpu_name resolves the memory size.
    #     The ordering is SXM4 80 > PCIE 80 > SXM4 40 > PCIE 40, i.e. memory
    #     generation first (HBM2e 80 GB parts ahead of HBM2 40 GB parts) and
    #     board power second — and it does NOT track the GEMM ceiling, which
    #     orders SXM4 40 (273.0) > SXM4 80 (258.9) > PCIE 80 (254.4) >
    #     PCIE 40 (213.5). Same lesson as the H200-vs-Max-Q pair: at this
    #     shape the cards are bandwidth-bound, not GEMM-bound.
    # ======================================================================= #
    # ---- A100 SXM4 80GB ---------------------------------------------------
    Rate(
        gpu="A100 SXM4 80GB", num_gpus=1, shape="9b-w20480-dec", tok_s=2549.0,
        measured=True, provisional=False, date="2026-08-16",
        provenance=(
            "job 20260816T060758-gpu-rate-9b-w20480-5894 (box 47844684, "
            "machine 28415 / host 4535, spot $0.8806/hr); 125,614 tok/step "
            "(1,632,983 / 13) / 49.2724 s mean step p50 (49.2250, 49.3198). "
            "<upstream-bench>/archive/runs/2026-08-16-gpu-rate-bench-a100/jobs/"
            "20260816T060758-gpu-rate-9b-w20480-5894/prod_w20480_r{1,2}/"
            "train_summary.json"
        ),
        note=(
            "13 timed steps x 2 repeats (26 total), p50 spread 0.192% — the "
            "tightest repeat pair in the whole table. Card confirmed "
            "`NVIDIA A100-SXM4-80GB`, 85.09 GB usable. Peak VRAM 32.900 GB "
            "reserved. GEMM ceiling 258.9 TFLOP/s bf16. 100% util, power-capped "
            "at its own 400 W limit (mean 330.7 / 383.2 W across the repeats), "
            "so this is the card and not a busy neighbour. THE FASTEST A100 "
            "PART MEASURED, and at the $0.8806 paid it is also the worst A100 "
            "buy: 10.4M tok/$ against the PCIe 40GB's 19.4M."
        ),
    ),
    # ---- A100 PCIE 80GB ---------------------------------------------------
    Rate(
        gpu="A100 PCIE 80GB", num_gpus=1, shape="9b-w20480-dec", tok_s=2388.0,
        measured=True, provisional=False, date="2026-08-16",
        provenance=(
            "job 20260816T062836-gpu-rate-9b-w20480-3462 (box 47845680, "
            "machine 71654 / host 458442, ON-DEMAND $1.5819/hr); 125,614 "
            "tok/step (1,632,983 / 13) / 52.6108 s mean step p50 (52.7427, "
            "52.4790). <upstream-bench>/archive/runs/"
            "2026-08-16-gpu-rate-bench-a100/jobs/"
            "20260816T062836-gpu-rate-9b-w20480-3462/prod_w20480_r{1,2}/"
            "train_summary.json"
        ),
        note=(
            "26 timed steps, p50 spread 0.501%. Card confirmed "
            "`NVIDIA A100 80GB PCIe`, 85.09 GB usable. Peak VRAM 32.900 GB "
            "reserved. GEMM ceiling 254.4 TFLOP/s bf16. 300 W part, mean "
            "252.8 / 288.9 W. THE ONLY ON-DEMAND CELL IN THE SWEEP, and the "
            "$1.5819 is a MARKET price not a card price: every cheap spot "
            "A100 PCIe 80GB on 2026-08-16 sat on ONE host (67231) which "
            "outbid this lane twice at $0.25 and $0.45 and then dropped an "
            "on-demand box to `offline` mid-boot. The rate is the silicon; "
            "the 5.4M tok/$ that falls out of this price is NOT the class's "
            "tokens-per-dollar and must be recomputed from a live offer. At "
            "the $0.2022/hr spot this lane first tried it would be 42.5M "
            "tok/$ — the best on the whole board — which is exactly why the "
            "ranker divides by the price you actually pay."
        ),
    ),
    # ---- A100 SXM4 40GB ---------------------------------------------------
    Rate(
        gpu="A100 SXM4 40GB", num_gpus=1, shape="9b-w20480-dec", tok_s=2352.0,
        measured=True, provisional=False, date="2026-08-16",
        provenance=(
            "job 20260816T061942-gpu-rate-9b-w20480-d9ae (box 47845221, "
            "machine 145857 / host 554156, spot $0.4806/hr); 125,614 tok/step "
            "(1,632,983 / 13) / 53.4000 s mean step p50 (53.5918, 53.2081). "
            "<upstream-bench>/archive/runs/2026-08-16-gpu-rate-bench-a100/jobs/"
            "20260816T061942-gpu-rate-9b-w20480-d9ae/prod_w20480_r{1,2}/"
            "train_summary.json"
        ),
        note=(
            "26 timed steps, p50 spread 0.719%. Card confirmed "
            "`NVIDIA A100-SXM4-40GB`, 42.4 GB usable. FITS: peak VRAM 32.900 GB "
            "reserved, ~9.5 GB spare. GEMM ceiling 273.0 TFLOP/s bf16 — HIGHER "
            "than the 80 GB part's 258.9 while delivering 8% FEWER tokens, "
            "which is the same lesson the H200-vs-Max-Q pair taught: this shape "
            "is not GEMM-bound, it follows memory bandwidth (HBM2 ~1,555 GB/s "
            "here vs HBM2e 2,039 on the 80 GB SXM4). 390 W limit, mean 280/332 W."
        ),
    ),
    # ---- A100 PCIE 40GB ---------------------------------------------------
    Rate(
        gpu="A100 PCIE 40GB", num_gpus=1, shape="9b-w20480-dec", tok_s=2134.0,
        measured=True, provisional=False, date="2026-08-16",
        provenance=(
            "job 20260816T060655-gpu-rate-9b-w20480-0aae (box 47844629, "
            "machine 67966 / host 458442, spot $0.3967/hr); 125,614 tok/step "
            "(1,632,983 / 13) / 58.8579 s mean step p50 (59.0887, 58.6271). "
            "<upstream-bench>/archive/runs/2026-08-16-gpu-rate-bench-a100/jobs/"
            "20260816T060655-gpu-rate-9b-w20480-0aae/prod_w20480_r{1,2}/"
            "train_summary.json"
        ),
        note=(
            "26 timed steps, p50 spread 0.784%. Card confirmed "
            "`NVIDIA A100-PCIE-40GB`, 42.41 GB usable. THE HEADLINE FIT: peak "
            "VRAM 32.900 GB reserved, ~9.5 GB spare, so the cheapest and most "
            "plentiful datacenter class on the market runs the production shape. "
            "SLOWEST card in the whole table and, at the $0.3967 paid, still "
            "19.4M tok/$ — second only to the H200 NVL's 33.2M and better than "
            "the PRO 6000 Max-Q that is today's launch default. 100% util, "
            "power-capped at its own 250 W limit (mean 222.7 / 239.6 W) at "
            "1,140-1,170 MHz: the 250 W PCIe part IS this slow, it was not "
            "throttled or co-tenanted. GEMM ceiling 213.5 TFLOP/s bf16, the "
            "lowest measured."
        ),
    ),
    # ---- H100 PCIe --------------------------------------------------------
    Rate(
        gpu="H100 PCIE", num_gpus=1, shape="7b-w32768-prod", tok_s=3944.0,
        measured=True, provisional=True, date="2026-08-12",
        provenance=(
            "out/jobs/20260812T023847-m1-step-attribution-3a9b/out/q7_tp/"
            "train_summary.json; 131,299 tok/step / 33.296 s step p50"
        ),
        note=(
            "5 timed steps, single cell, no repeat — and the job it came from "
            "died rc=124 on its hang detector (m1 bundle header). The paired "
            "q9 cell from the same run is EXCLUDED from this table: it "
            "measured 149.8 s/step for 9b-w40960, ~2.6x off the same shape's "
            "PRO 6000 anchor and off its own bundle's prediction, so it is an "
            "unexplained artifact rather than an H100 PCIe rate."
        ),
    ),
    Rate(
        gpu="H100 PCIE", num_gpus=1, shape="9b-w12288-fit", tok_s=3150.0,
        measured=True, provisional=True, date="2026-08-12",
        provenance=(
            "out/jobs/20260812T024148-k1-flashqla-7391/out/flaold_r{1,2}/"
            "train_summary.json; 24,704 tok/step / 7.844 s mean step p50"
        ),
        note=(
            "6 timed steps per cell. `flaold` = fla 0.4.2, the version every "
            "production bundle pins. The r1 cell of the earlier run of the "
            "same bundle (20260812T021707-k1-flashqla-9d9e) shows 11-12 s p50 "
            "on first-cell warmup, so a 6-step p50 on this bundle is "
            "warmup-sensitive."
        ),
    ),
    # ---- RTX PRO 6000 Blackwell, Workstation, 600 W ------------------------
    Rate(
        gpu="RTX PRO 6000 WS 600W", num_gpus=1, shape="7b-w12288-fit",
        tok_s=4336.0, measured=True, provisional=False, date="2026-08-11",
        provenance=(
            "out/jobs/20260811T083823-perf-levers-padfree-0e92/out/"
            "s12288_a_b1ga32_r{1,2}/train_summary.json; 92,373 tok/step / "
            "21.3035 s mean step p50"
        ),
        note=(
            "12 timed steps x 2 repeats, p50 agreeing to 0.002%. Card string "
            "`NVIDIA RTX PRO 6000 Blackwell Workstation Edition` (no Max-Q), "
            "power 514-516 W mean."
        ),
    ),
    Rate(
        gpu="RTX PRO 6000 WS 600W", num_gpus=1, shape="7b-w4096-fit",
        tok_s=4335.0, measured=True, provisional=False, date="2026-08-11",
        provenance=(
            "out/jobs/20260811T083823-perf-levers-padfree-0e92/out/"
            "s04096_a_b1ga32_r{1,2}/train_summary.json; 60,492 tok/step / "
            "13.953 s mean step p50"
        ),
        note=(
            "12 timed steps x 2 repeats. Within 0.03% of the same card's "
            "12,288 rate — at these windows this shape is FLOP-bound, not "
            "attention-bound, which is why the two agree."
        ),
    ),
    # ---- RTX PRO 6000 Blackwell, Max-Q (300 W class) -----------------------
    Rate(
        gpu="RTX PRO 6000 WS MAXQ", num_gpus=1, shape="9b-w20480-dec",
        tok_s=2984.0, measured=True, provisional=False, date="2026-08-16",
        provenance=(
            "job 20260816T045239-gpu-rate-9b-w20480-b33c (box 47840746, "
            "machine 54830, spot $0.5657/hr); 125,614 tok/step / 42.0991 s "
            "mean step p50 (42.1200, 42.0781). "
            "out/jobs/20260816T045239-gpu-rate-9b-w20480-b33c/out/"
            "prod_w20480_r{1,2}/train_summary.json"
        ),
        note=(
            "26 timed steps, p50 spread 0.10% — the tightest cell in the "
            "sweep. Card string `NVIDIA RTX PRO 6000 Blackwell Max-Q "
            "Workstation Edition`, so the SKU is RESOLVED, not inferred. "
            "BOTH `RTX PRO 6000 WS` offers rented that night delivered Max-Q "
            "parts (the first, box 47839964 on machine 98261, was preempted at "
            "step 1 and also reported Max-Q at a 250 W limit) — two for two, "
            "which is the strongest evidence yet that the ambiguous vast name "
            "should keep answering with the slower part. NO 600 W cell exists "
            "at this shape: the vast market does not let you ask for one."
        ),
    ),
    Rate(
        gpu="RTX PRO 6000 WS MAXQ", num_gpus=1, shape="7b-w12288-fit",
        tok_s=3110.0, measured=True, provisional=True, date="2026-08-11",
        provenance=(
            "out/jobs/20260811T081845-perf-levers-padfree-04f5/out/"
            "fit_s12288_a_b1ga32/train_summary.json; 92,373 tok/step / "
            "29.7026 s step p50"
        ),
        note=(
            "2 timed steps, one cell (the sibling run 20260811T075108-...-bd42 "
            "produced no summary for this cell). 1.39x SLOWER than the 600 W "
            "part at the same shape, on 237 W mean vs 516 W."
        ),
    ),
    Rate(
        gpu="RTX PRO 6000 WS MAXQ", num_gpus=1, shape="7b-w4096-fit",
        tok_s=3237.0, measured=True, provisional=True, date="2026-08-11",
        provenance=(
            "out/jobs/20260811T075108-perf-levers-padfree-bd42/out/"
            "fit_s04096_a_b1ga32/train_summary.json and "
            "out/jobs/20260811T081845-perf-levers-padfree-04f5/out/"
            "fit_s04096_a_b1ga32/train_summary.json; 60,492 tok/step / "
            "18.682 s mean step p50 (18.8947, 18.4689)"
        ),
        note="2 timed steps per cell, two independent runs of the same bundle.",
    ),
    # ---- RTX PRO 6000 WS, SKU UNRESOLVED (the vast offer name) -------------
    Rate(
        gpu="RTX PRO 6000 WS", num_gpus=1, shape="9b-w40960-dec", tok_s=2519.0,
        measured=True, provisional=True, date="2026-08-05",
        provenance=(
            "run 20260805T124023-v8-qwen35-dec-train-8d00 (box 46880356, "
            "1x RTX PRO 6000 WS 96 GB, on-demand); 146,567 tok/step "
            "(22,864,505 corpus tokens / 156 steps) / 58.18 s warm step. "
            "docs/plans/witness/V8_QWEN35_DEC_TRAIN_2026-08-05.md sections "
            "0 and 3; artifact out/jobs/20260805T124023-v8-qwen35-dec-train-"
            "8d00/out/train_summary.json (whole-run wall 9,183.5 s / 156 steps "
            "= 58.87 s/step including the 165.69 s cold step 1)"
        ),
        note=(
            "PROVISIONAL FOR SKU, NOT FOR RIGOUR: 155 warm steps is the "
            "longest single-card window in the bank, but the run predates "
            "train_summary schema v2 (2026-08-09) so it records no "
            "`hardware.gpu_names`, and `RTX PRO 6000 WS` on the vast market "
            "covers BOTH the 600 W part and the Max-Q part, measured 1.39x "
            "apart. Which one box 46880356 was is unrecoverable — its offer "
            "left the market at rent time."
        ),
    ),
    # ---- RTX 5090 ---------------------------------------------------------
    # The self-anchored W=1/4/8 ladder, one 8x5090 PCIe host, same slice, same
    # seed, same image, both knob settings. W=4/W=8 rows are the POST-e48def36
    # default; the pre-boundary numbers are in GPU_RATES.md and not here.
    Rate(
        gpu="RTX 5090", num_gpus=1, shape="7b-w12288-fit", tok_s=3190.0,
        measured=True, provisional=False, date="2026-08-14",
        provenance=(
            "docs/plans/witness/perf/W8_LADDER_RESULT_2026-08-14.md section "
            "1 (job 20260814T090840-perf-levers-w8-135c, box 47694876); "
            "95,245.5 tok/step / 29.855 s mean tail s/it (29.850, 29.860). "
            "Token count from out/jobs/20260814T090840-perf-levers-w8-135c/out/"
            "n1_shipped_r{1,2}/train_summary.json"
        ),
        note=(
            "10 timed steps x 2 repeats. W=1 is BOUNDARY-INDEPENDENT: the "
            "optimized knobs remove per-micro-batch collectives, of which a "
            "single rank has none (W8 ladder: the W=1 null reproduced 3x "
            "inside 0.05%). The train_summary p50 (30.17 s) gives 3,157 tok/s; "
            "the doc's tail figure is used so this row and the W=4/W=8 rows "
            "come from one self-anchored ladder."
        ),
    ),
    Rate(
        gpu="RTX 5090", num_gpus=4, shape="7b-w12288-fit", tok_s=9669.0,
        measured=True, provisional=True, date="2026-08-14",
        provenance=(
            "docs/plans/witness/perf/D3D_DEFAULT_VERIFICATION_RESULT_"
            "2026-08-14.md section 1 (job 20260814T090233-perf-levers-ddp3d-"
            "655e, box 47694876), `n4_newdef` cell; 100,122.6 tok/step / "
            "10.355 s mean tail s/it (10.350, 10.360)"
        ),
        note=(
            "POST-BOUNDARY (e48def36). Provisional because the tokens/step is "
            "borrowed from the PAIRED `n4_shipped` cell of the same job: same "
            "bundle, slice, seed and world size, so the rows processed are "
            "identical, but the `deferred` counter under-reports its own total "
            "(638,821 vs 1,001,226 over 10 steps). 3.03x the W=1 rate = 75.8% "
            "raw scaling per token (72.1% per second, the ladder's own figure "
            "-- W=1 draws ~5% fewer non-padding tokens per step because the "
            "distributed sampler pads the slice differently)."
        ),
    ),
    Rate(
        gpu="RTX 5090", num_gpus=8, shape="7b-w12288-fit", tok_s=14172.0,
        measured=True, provisional=True, date="2026-08-14",
        provenance=(
            "docs/plans/witness/perf/W8_LADDER_RESULT_2026-08-14.md section "
            "1 (job 20260814T090840-perf-levers-w8-135c, box 47694876), "
            "`n8_newdef` cell; 100,122.6 tok/step / 7.065 s mean tail s/it "
            "(7.070, 7.060)"
        ),
        note=(
            "POST-BOUNDARY (e48def36). Same borrowed-token-count caveat as the "
            "W=4 row (n8_newdef self-reports 817,014 vs n8_shipped's "
            "1,001,226). 4.44x the W=1 rate. TRAINING.md's standing rule says "
            "stop a single DDP job at 4 cards on PCIe boxes — cards 5-8 are "
            "33.6% marginal; this row prices a whole-box hold, it does not "
            "recommend one."
        ),
    ),
)


# --------------------------------------------------------------------------- #
# card-name normalization
# --------------------------------------------------------------------------- #
# Callers hand us names from two vocabularies that do not agree:
#
#   vast API  `gpu_name`      "H100 NVL", "RTX 5090", "RTX PRO 6000 WS"
#   torch     `get_device_name` "NVIDIA H100 NVL", "NVIDIA GeForce RTX 5090",
#                               "NVIDIA RTX PRO 6000 Blackwell Max-Q
#                                Workstation Edition"
#
# plus the short hands `herdd`'s own `--gpu` flag takes ("5090", "h100").
# This mirrors the family logic of herdd.py's GPU_ALIASES without importing
# it: that map expands a shorthand into the SET of vast names to SEARCH for,
# which is a different job from collapsing one name into the one class whose
# rate we measured. Keeping them separate is deliberate — herdd's "h100"
# legitimately searches SXM+PCIE+NVL together, and those three do not share a
# throughput number.
_STRIP = re.compile(r"\b(nvidia|geforce|blackwell|edition|corporation)\b")
_NONWORD = re.compile(r"[^a-z0-9]+")

# exact shorthand -> canonical, for the forms a human or a flag hands us
_SHORTHAND = {
    "3090": "RTX 3090", "rtx3090": "RTX 3090",
    "4090": "RTX 4090", "rtx4090": "RTX 4090",
    "5090": "RTX 5090", "rtx5090": "RTX 5090",
    "4080": "RTX 4080", "3080": "RTX 3080",
    "a6000": "RTX A6000", "a5000": "RTX A5000", "a40": "A40",
    "pro6000": "RTX PRO 6000 WS", "rtxpro6000": "RTX PRO 6000 WS",
    "6000blackwell": "RTX PRO 6000 WS",
}


def normalize_gpu_name(name: str | None) -> str:
    """Collapse a vast / torch / shorthand GPU name to this table's class key.

    Returns "" for an empty or non-string input, and returns an UNKNOWN name
    unchanged (upper-cased, whitespace-collapsed) rather than guessing: an
    unknown key simply has no entries, and every lookup below then returns
    None, which is the designed "caller falls back to price-only" path.

    The class keys are as narrow as the measurements are. `H100` bare does NOT
    resolve to `H100 NVL`: the PCIe part measured a different number and the
    SXM part was never measured here, so a bare "h100" must return None rather
    than borrow a variant's rate.
    """
    if not isinstance(name, str) or not name.strip():
        return ""
    raw = name.strip()
    s = _STRIP.sub(" ", raw.lower())
    s = _NONWORD.sub(" ", s).strip()
    s = re.sub(r"\s+", " ", s)
    if not s:
        return ""

    key = s.replace(" ", "")
    if key in _SHORTHAND:
        return _SHORTHAND[key]

    # --- RTX PRO 6000 Blackwell: three SKUs, and vast names only two -------
    # The discriminator between the two Workstation parts is which vocabulary
    # the caller spoke. torch spells the part out ("... Workstation Edition",
    # "... Max-Q Workstation Edition") and so resolves the SKU; vast's offer
    # string is the abbreviation "RTX PRO 6000 WS", which covers BOTH parts and
    # is therefore its own, deliberately separate, class key. Box 47010337
    # (2026-08-06) is the standing example of an offer that advertised WS and
    # delivered the Max-Q part; see herdd.py's note at the instance-scorecard
    # capture, and tools/vast/mfu.py on the four circulating ceilings.
    if "pro 6000" in s or "pro6000" in key:
        if "max q" in s or "maxq" in key:
            return "RTX PRO 6000 WS MAXQ"
        if "server" in s or re.search(r"\bs\b", s):
            return "RTX PRO 6000 S"
        if "workstation" in s:
            return "RTX PRO 6000 WS 600W"
        return "RTX PRO 6000 WS"

    if "h100" in key:
        if "nvl" in s:
            return "H100 NVL"
        if "pcie" in key:
            return "H100 PCIE"
        if "sxm" in s or "hbm3" in s:
            return "H100 SXM"
        return "H100"
    if "h200" in key:
        return "H200 NVL" if "nvl" in s else "H200"
    if "b200" in key:
        return "B200"
    if "b300" in key:
        return "B300"
    # --- A100: FOUR parts, and vast's offer string names only two ----------
    # Same shape as the PRO 6000 problem above, one axis further out. vast sells
    # `A100 PCIE` and `A100 SXM4` and carries the memory size in a SEPARATE
    # offer field (`gpu_ram`), so the offer NAME cannot tell a 40 GB card from
    # an 80 GB one — but the two are different silicon for this workload: the
    # 40 GB parts are HBM2 at ~1,555 GB/s and the 80 GB parts HBM2e at
    # 1,935 (PCIe) / 2,039 (SXM4) GB/s, and 2026-08-16 measured the four of them
    # spread across a real range at one shape. torch's device name DOES carry
    # the size ("NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-40GB"), so a
    # post-rent read resolves the part and a pre-rent offer string does not.
    # Bare `A100 PCIE` / `A100 SXM4` therefore stay their own class keys and
    # answer, via _SKU_FALLBACK, with the SLOWER (40 GB) part — the same
    # "ranking a box you might not get is a purchase decision" rule.
    if "a100" in key:
        _mem = "80GB" if "80gb" in key else ("40GB" if "40gb" in key else "")
        if "sxm" in s:
            return ("A100 SXM4 " + _mem) if _mem else "A100 SXM4"
        if "pcie" in key:
            return ("A100 PCIE " + _mem) if _mem else "A100 PCIE"
        return ("A100 " + _mem) if _mem else "A100"
    if "l40s" in key:
        return "L40S"
    if re.search(r"\brtx (3090|4090|5090|4080|3080)\b", s):
        return "RTX " + re.search(r"\b(3090|4090|5090|4080|3080)\b", s).group(1)

    return s.upper()


# --------------------------------------------------------------------------- #
# lookup
# --------------------------------------------------------------------------- #
def build_index(rates) -> dict:
    """(gpu, num_gpus, shape) -> Rate. Later entries win, so a fresher row can
    be appended above without deleting the one it supersedes."""
    return {(r.gpu, r.num_gpus, r.shape): r for r in rates}


_INDEX = build_index(RATES)

# Fallback order for an unresolved SKU: the vast offer name "RTX PRO 6000 WS"
# is ambiguous between two parts measured 1.39x apart, so when it has no cell
# of its own we answer with the SLOWER part. Ranking a box you might not get is
# a purchase decision, and the floor is the honest number for one.
#
# `A100 PCIE` / `A100 SXM4` are the same problem: vast's offer NAME omits the
# memory size (it lives in the separate `gpu_ram` field), and the 40 GB and
# 80 GB parts measured 1.19x / 1.09x apart at `9b-w20480-dec` on 2026-08-16. A
# caller holding only the offer string gets the 40 GB floor. A caller who has
# already rented and read torch's device name gets the exact part, because
# normalize_gpu_name resolves the size out of that string.
_SKU_FALLBACK = {
    "RTX PRO 6000 WS": ("RTX PRO 6000 WS MAXQ",),
    "A100 PCIE": ("A100 PCIE 40GB",),
    "A100 SXM4": ("A100 SXM4 40GB",),
}


def entry_for(gpu_name: str, num_gpus: int = 1, shape: str | None = None,
              *, table: dict | None = None) -> Rate | None:
    """The full `Rate` cell, or None when this (card, count, shape) is unknown.

    `shape=None` resolves in this order, and each step is a documented choice
    rather than a fallback chain that grew:
      1. DEFAULT_SHAPE — what a caller who did not say is most likely running.
      2. the SLOWEST measured shape for that (card, count). Conservative on
         purpose: a rate used to justify spending money should not be the
         luckiest shape we happened to bench.
      3. the SKU floor for an ambiguous vast name (see _SKU_FALLBACK).
    It never scales, interpolates or extrapolates. `pass table={}` (or any
    mapping) to look up against something other than the shipped table — the
    empty mapping is the null vector and every function here returns None on it.
    """
    idx = _INDEX if table is None else table
    gpu = normalize_gpu_name(gpu_name)
    if (not gpu or isinstance(num_gpus, bool)
            or not isinstance(num_gpus, int) or num_gpus < 1):
        return None

    candidates = (gpu,) + _SKU_FALLBACK.get(gpu, ())
    for key in candidates:
        if shape is not None:
            hit = idx.get((key, num_gpus, shape))
            if hit is not None:
                return hit
            continue
        hit = idx.get((key, num_gpus, DEFAULT_SHAPE))
        if hit is not None:
            return hit
        rows = [r for (g, n, _s), r in idx.items() if g == key and n == num_gpus]
        if rows:
            return min(rows, key=lambda r: r.tok_s)
    return None


def rate_for(gpu_name: str, num_gpus: int = 1, shape: str | None = None,
             *, table: dict | None = None) -> float | None:
    """Tokens/sec for this (card, count, shape), or None when unknown.

    None is a first-class answer and means "we have never measured this" — the
    caller is expected to fall back to price-only ranking, NOT to substitute a
    guess. Nothing in this module ever invents a rate for an unmeasured cell.
    """
    hit = entry_for(gpu_name, num_gpus, shape, table=table)
    return None if hit is None else hit.tok_s


def tokens_per_dollar(gpu_name: str, num_gpus: int, dph: float,
                      shape: str | None = None,
                      *, table: dict | None = None) -> float | None:
    """Training tokens bought per USD at `dph` dollars/hour for the WHOLE box.

    `dph` is the box price for `num_gpus` cards — vast quotes per-instance, so
    pass what vast quotes. Returns None when the rate is unknown or `dph` is
    not a positive number (a free or unpriced box is not a division we can do).

    Steady-state, per this module's header: fixed per-job costs are excluded,
    so this over-states absolute tokens per dollar and is intended for RANKING.
    """
    tok_s = rate_for(gpu_name, num_gpus, shape, table=table)
    if tok_s is None:
        return None
    try:
        dph = float(dph)
    except (TypeError, ValueError):
        return None
    if not dph > 0:
        return None
    return tok_s * 3600.0 / dph


def shapes_for(gpu_name: str, num_gpus: int = 1,
               *, table: dict | None = None) -> tuple[str, ...]:
    """Shape keys measured for this (card, count), fastest last. Empty when
    none — useful for a caller that wants to pick a shape both candidate cards
    actually have, which is the only licensed way to compare two of them."""
    idx = _INDEX if table is None else table
    gpu = normalize_gpu_name(gpu_name)
    if not gpu:
        return ()
    rows = [r for (g, n, _s), r in idx.items() if g == gpu and n == num_gpus]
    return tuple(r.shape for r in sorted(rows, key=lambda r: r.tok_s))


# --------------------------------------------------------------------------- #
# measured multi-GPU scaling — EXPLICIT, opt-in, never used by rate_for()
# --------------------------------------------------------------------------- #
# One card class, one shape, one host, one ladder. Offered so a caller who
# WANTS to estimate an unmeasured multi-card cell has to reach for it by name
# and can see what it is: a 5090/PCIe/7B measurement, not a law. NVLink-class
# hosts are explicitly out of scope (W8 ladder section 3) — the term that
# dominates the W=8 residual is architecturally different there.
MULTI_GPU_SCALING = {
    "basis": "RTX 5090 (PCIe) @ 7b-w12288-fit, post-e48def36 default, "
             "docs/plans/witness/perf/W8_LADDER_RESULT_2026-08-14.md",
    "date": "2026-08-14",
    "tok_s_multiple_vs_1_card": {1: 1.00, 4: 3.03, 8: 4.44},
}
