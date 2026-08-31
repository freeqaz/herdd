#!/usr/bin/env python3
"""vram_facts — how much VRAM a training shape needs, answered from MEASUREMENT.

The inverse of `autotune.py`, which it sits beside: autotune asks "given this
card, which knobs?"; this asks "given this shape, which card?". Together they
are the two directions of one question, and neither should be answered by
hand-authoring a number into a bundle.

    python3 tools/vast/vram_facts.py --base qwen35-9b --quant bf16 --max-seq 12288
    # -> 26.34 GB measured (n=11, spread 0.00) + 1.5 headroom -> needs a 32 GB card

WHY THIS IS A LOOKUP AND NOT A FORMULA. We tried the formula.
`FITTING_9B_ON_A_5090_2026-08-06.md` §1 derived
`peak_GB ~= 22.82 + 0.455*(seq/1024)` from one measured anchor plus a term
decomposition; §8.2 read a slope of **0.063** off the same ladder and called
the 0.455 refuted. **Correction 2026-08-29: the 0.063 is the artifact.** §8.2's
ladder walked `max_seq`, but peak tracks the LONGEST ROW ACTUALLY PROCESSED and
its slice's rows stopped growing — its own `n_rows` saturates (21/47/58/61 over
2048/4096/8192/12288) and no anchor in it records `row_tokens_max` at all, so
nothing in that run establishes that the window was ever exercised. A flat slice
measures a flat curve. On windows minted to fill themselves, `GPU5090_SIZING_2026-08-29.md`
§2.1 measures **0.412–0.446 GB/1k**, i.e. §1's original slope, and every row this
module EXTRAPOLATED at 0.063 under-read the measurement by 10–26% — the sign
that says "fits" about a shape that OOMs. §8.4's OOM negative control did not
OOM. The doc's own conclusion — do not calibrate a multi-term model against a
single point — survives; its slope constant does not. So the shape of this
module is: match an anchor group, take its measured worst case, and REFUSE when
there is no group and when the window runs past the group's longest anchor —
never extrapolate from arithmetic.

THREE PROPERTIES THAT COME STRAIGHT FROM THE DATA, not from theory:

  * `grad_checkpointing` is the dominant lever, not sequence length. Measured
    on qwen25-coder-7b bf16 at seq 4096: 20.87 GB on, **52.20 GB off**. Window
    over the same span costs well under 1 GB.
  * Within a group, identical declared shapes still spread by up to **3.8 GB**
    (qwen25-coder-7b bf16 12288: 20.77 vs 24.56, differing by no recorded shape
    field). Peak is set by the longest row ACTUALLY PROCESSED, which depends on
    the corpus and the drop policy. So an estimate is the group's MAX. Sizing
    on the mean would under-size half the runs it claims to cover.
  * Without FSDP, world size barely moves the per-card peak (+0.4 GB from 1 to
    4 cards) — each rank holds a full replica. With FSDP it changes everything
    (9B bf16 12288: 26.34 -> 18.00 -> 13.46 at ws 1/2/4). So world size is part
    of the group key only when sharding is on.

Stdlib-only and import-light, matching `autotune.py`: this is meant to be
callable from the submit path with no dependencies.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FACTS_PATH = os.path.join(HERE, "vram_facts.json")

# Reserved-vs-allocated headroom. The anchors measure `max_memory_allocated`;
# the caching allocator's RESERVED pool runs above it, and fragmentation makes
# the gap uneven. 1.5 GB is the convention `FITTING_9B_ON_A_5090` §2 sizes with
# ("usable budget = card GB - 1.5"), applied here in the inverse direction.
RESERVED_HEADROOM_GB = 1.5

# `FITTING_9B_ON_A_5090` §8.2 read this off a ladder whose rows stopped growing
# (2048 -> 12288 "cost +0.65 GB"). It is NOT a window slope and MUST NOT be used
# as one — see the module docstring. It survives only as the tripwire constant
# `test_does_not_reproduce_the_refuted_slope` and the grad-ckpt-OFF note below
# refer to, and as the thing this module used to extrapolate at.
DISPROVEN_FLAT_SLICE_SLOPE_GB_PER_1K = 0.063
MEASURED_SLOPE_GB_PER_1K = DISPROVEN_FLAT_SLICE_SLOPE_GB_PER_1K   # legacy alias

# The floor an opt-in extrapolation is charged at, per 1k tokens.
# Measured on EXERCISED windows (rows minted to fill each one), RTX 5090,
# `GPU5090_SIZING_2026-08-29.md` §2.1: 0.412 GB/1k over 12,288 -> 24,576 for the
# 7-projection target set and 0.446 for all-linear. Individual segments of those
# ladders run 0.12 to 0.67, so 0.45 is the endpoint fit rounded up and NOT an
# upper bound — which is half of why extrapolation is opt-in. The other half is
# that the ANCHOR can under-read too: the same run measured list-7 @12,288 at
# 26.08 GB where this table's single anchor says 22.33, so no slope, however
# steep, rescues an extrapolation off a base that is already 14% low.
EXTRAPOLATION_SLOPE_FLOOR_GB_PER_1K = 0.45

# What we can actually rent (per card). An estimate rounds UP to one of these:
# a floor of 27 GB is unactionable, "needs a 32" is the decision.
CARD_CLASSES = (16, 24, 32, 48, 80, 96, 141, 192)

# Dimensions an anchor must match EXACTLY to be usable for a shape. Everything
# else in the summary is descriptive. `world_size` is conditional — see
# `_group_of`.
GROUP_KEYS = ("base_slug", "quant_mode", "grad_checkpointing", "gc_flag",
              "ce_chunk_matmul", "target_modules_class", "lora_r", "packing",
              "sharded")


class VramFactsError(Exception):
    """The facts file is unusable (missing, malformed)."""


class Unmeasured(Exception):
    """No measured anchor covers this shape.

    Deliberately an exception and not a fallback estimate. The two times this
    project guessed a VRAM number from first principles it was wrong by 7x on
    the slope and wrong in direction on the OOM control, and both guesses were
    quoted downstream as though measured. Carries `probe_cmd` — what to run to
    turn the guess into an anchor."""

    def __init__(self, message: str, *, probe_cmd: str = "", near=()):
        super().__init__(message)
        self.probe_cmd = probe_cmd
        self.near = list(near)


_FACTS_CACHE: dict = {}


def load_facts(path: str = FACTS_PATH) -> dict:
    """Read (and memoize) the anchor file. Cached on (path, mtime, size): a
    matrix asks once per arm, and the document is ~150 KB of JSON. The mtime
    key means a `harvest_vram.py --write` in the same process is picked up
    rather than serving stale anchors."""
    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
        if key in _FACTS_CACHE:
            return _FACTS_CACHE[key]
    except OSError:
        key = None
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except FileNotFoundError:
        raise VramFactsError(
            f"{path} not found — build it with `python3 "
            f"{os.path.join('tools', 'vast', 'harvest_vram.py')} --write`")
    except ValueError as e:
        raise VramFactsError(f"{path}: {e}") from e
    if not isinstance(doc, dict) or "anchors" not in doc:
        raise VramFactsError(f"{path}: not a vram_facts document")
    if key is not None:
        _FACTS_CACHE.clear()          # one entry: the current file
        _FACTS_CACHE[key] = doc
    return doc


# --- shape normalization ------------------------------------------------------

def target_modules_class(tm) -> str:
    """`all-linear` vs the trainer's default 7-name list vs anything else.

    A class, not the literal list: the exact names differ per architecture
    (gemma and qwen spell their MLP projections differently) while the VRAM
    consequence — how much adapter + optimizer state exists — tracks the
    CATEGORY. Grouping on the raw list would split every base into singletons."""
    if tm is None:
        return "unknown"
    if isinstance(tm, str):
        return "all-linear" if tm.strip() == "all-linear" else tm.strip()
    names = list(tm)
    if len(names) == 1 and str(names[0]).strip() == "all-linear":
        return "all-linear"
    return f"list-{len(names)}"


def _norm_bool(v) -> str:
    if v is None:
        return "unknown"
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "on", "yes"):
            return "true"
        if s in ("0", "false", "off", "no", ""):
            return "false"
        return s
    return "true" if v else "false"


def gc_flag_class(flag) -> str:
    """How much of the model was actually checkpointed: `full`, `none`, or
    `partial-<f>`.

    `grad_checkpointing` is the switch; this is the FRACTION behind it, and the
    fraction is what the memory tracks — one 9B/20480 shape measures 147 GB at
    fraction 0.0 and 29 GB at 1.0 while both record `grad_checkpointing: true`.
    ABSENT NORMALIZES TO `full`, not to `unknown`: every run that predates the
    partial-GC lever checkpointed everything, so a third bucket would split the
    existing anchors away from the shape today's bundles declare and refuse
    queries that months of measurement already answer."""
    if flag is None:
        return "full"
    s = str(flag).strip().lower()
    if s in ("", "on", "true", "yes", "1", "1.0", "all", "full"):
        return "full"
    if s in ("off", "false", "no", "0", "0.0", "none"):
        return "none"
    return f"partial-{s}"


def _sharded(fsdp, device_map_mode) -> str:
    """Whether the model is SPLIT across cards (FSDP or a pipeline device map)
    or replicated (DDP / single). Only in the former does world size change the
    per-card peak, so this is what the group keys on."""
    if str(fsdp or "").strip():
        return "fsdp"
    if str(device_map_mode or "single") not in ("single", "None", ""):
        return str(device_map_mode)
    return "replica"


def normalize_shape(*, base_slug, quant_mode=None, grad_checkpointing=None,
                    grad_checkpointing_flag=None,
                    ce_chunk_matmul=None, target_modules=None, lora_r=None,
                    packing=None, fsdp=None, device_map_mode=None,
                    world_size=None, max_seq=None, batch=None, **_ignored) -> dict:
    """One canonical dict from either a train_summary or a caller's kwargs, so
    an anchor and a query are compared on identical terms."""
    sharded = _sharded(fsdp, device_map_mode)
    return {
        "base_slug": str(base_slug or ""),
        "quant_mode": str(quant_mode or "").strip() or "unknown",
        "grad_checkpointing": _norm_bool(grad_checkpointing),
        "gc_flag": gc_flag_class(grad_checkpointing_flag),
        "ce_chunk_matmul": str(ce_chunk_matmul or "unknown").strip() or "unknown",
        "target_modules_class": target_modules_class(target_modules),
        "lora_r": int(lora_r) if lora_r else None,
        "packing": str(packing or "unknown").strip() or "unknown",
        "sharded": sharded,
        "world_size": int(world_size) if world_size else None,
        "tokens": (int(batch or 1) * int(max_seq)) if max_seq else None,
    }


def _anchor_shape(anchor: dict) -> dict:
    s = dict(anchor.get("shape") or {})
    s["base_slug"] = anchor.get("base_slug") or ""
    return normalize_shape(**s)


def anchor_tokens(anchor: dict, *, prefer_measured: bool = True) -> int | None:
    """The tokens-in-flight coordinate for an anchor: MEASURED where the run
    recorded it, declared where it did not.

    `max_seq` is a CAP, and the peak is set by the longest row ACTUALLY
    PROCESSED (`FITTING_9B_ON_A_5090_2026-08-06.md` §8.2). `summary_schema: 2`'s
    `token_stats.row_tokens_max` is the first time the trainer records the
    longest row, so this prefers it — capped at the declared window, because the
    trainer truncates there, and multiplied by batch, which is what is resident
    at once.

    BACKTESTED 2026-08-13, AND THE ANSWER WAS "ONLY WHERE BOTH SIDES ARE
    MEASURED" (36 anchors carry `token_stats`; `--backtest`; VRAM_SIZING.md,
    "Refining the coordinate"). A submit-time QUERY knows its declared window
    and cannot know the longest row its corpus will produce, so preferring the
    measured value on the anchor side alone puts the two sides on different
    axes: the anchors' coordinate shrinks, the query's does not, and 27 of 36
    held-out points that the declared axis covered EXACTLY (0 extrapolations)
    fell off the end of the compressed axis and were extrapolated at the
    group's own slope. Card-class accuracy fell 83.3% -> 58.3%, p90 absolute
    error 5.10 -> 14.06 GB, worst 11.48 -> 23.38 GB — all of it over-sizing (0
    under-estimates either way). Both sides on the measured axis scored BEST
    (91.7%) — the coordinate is not wrong, the asymmetry is.

    So `prefer_measured=False` is what the estimate path passes: query and
    anchor on one axis. The measured value stays the default here because the
    knob report pairs ANCHOR against ANCHOR, where both sides carry it and the
    comparison is symmetric."""
    shape = dict((anchor or {}).get("shape") or {})
    max_seq = shape.get("max_seq")
    batch = int(shape.get("batch") or 1)
    declared = int(batch * int(max_seq)) if max_seq else None
    if not prefer_measured:
        return declared
    stats = ((anchor or {}).get("telemetry") or {}).get("token_stats") or {}
    rows = stats.get("row_tokens_max")
    if not isinstance(rows, (int, float)) or rows <= 0:
        return declared
    measured = int(batch * int(rows))
    return min(measured, declared) if declared else measured


def _group_of(shape: dict) -> tuple:
    """The equality key. World size joins it ONLY when the model is sharded —
    with a full replica per rank the per-card peak is ~flat in world size
    (measured +0.4 GB across 1->4 cards), so demanding an exact match would
    refuse a perfectly good anchor."""
    key = tuple(shape.get(k) for k in GROUP_KEYS)
    if shape.get("sharded") != "replica":
        key += (shape.get("world_size"),)
    return key


# --- the estimate -------------------------------------------------------------

def group_slope_gb_per_1k(anchors, *, prefer_measured: bool = False) -> float:
    """GB per 1k tokens, measured on THIS group rather than borrowed.

    The global constant is the 9B ladder's 0.063, and it does not transfer: the
    qwen25-coder-7b 8-bit group's own 1024 -> 2048 step measures ~1.9 GB/1k,
    30x steeper. That is not a contradiction — `max_seq` is a CAP, and peak
    tracks the longest row actually seen, so a corpus whose rows outrun a short
    cap makes the cap bite while a long cap leaves it slack. Borrowing one
    group's slope for another is the same error as the refuted analytic model,
    one level down. Falls back to the global constant only when the group has
    fewer than two distinct window sizes to measure between.

    On the DECLARED axis, like everything else on the estimate path — a slope
    fitted over measured-row coordinates would be charged against a gap
    measured in declared ones. The qwen25-coder-7b padfree group fits 1.37
    GB/1k declared and 0.72 GB/1k measured. See `anchor_tokens`.

    The floor is `EXTRAPOLATION_SLOPE_FLOOR_GB_PER_1K`, not the old 0.063: a
    group that reads flat over its own range has usually measured a slice whose
    rows stopped growing, which is exactly the defect that produced 0.063 in the
    first place. A flat GROUP and a flat CURVE are indistinguishable from here,
    so the floor assumes the former."""
    pts = []
    for a in anchors:
        t = anchor_tokens(a, prefer_measured=prefer_measured)
        if t:
            pts.append((t, a["measured"]["peak_vram_alloc_gb"]))
    by_tok = {}
    for t, gb in pts:
        by_tok[t] = max(by_tok.get(t, 0.0), gb)     # worst case per window
    if len(by_tok) < 2:
        return EXTRAPOLATION_SLOPE_FLOOR_GB_PER_1K
    xs = sorted(by_tok)
    worst = 0.0
    for lo, hi in zip(xs, xs[1:]):
        worst = max(worst, (by_tok[hi] - by_tok[lo]) / ((hi - lo) / 1024.0))
    return max(worst, EXTRAPOLATION_SLOPE_FLOOR_GB_PER_1K)


def _probe_cmd(shape: dict) -> str:
    """The instrument that can actually mint the missing anchor.

    NOT always `fit-ladder`. `fit-ladder` never passes `--target-modules`, so
    the trainer applies DEFAULT_TARGET_MODULES (7 projections) and every anchor
    it mints lands in the `list-7` group. Pointed at an `all-linear` query it
    sends the operator to run a probe that cannot answer the question — the
    resulting anchor keys to a different group and the query still refuses.
    `gpu-rate-9b-w20480` pins `TARGET_MODULES: all-linear` and refuses to start
    on anything else, so it is the one that mints an all-linear anchor.
    (Measured the hard way on 2026-08-16; run of record
    `<upstream-bench>/archive/runs/2026-08-16-gradckpt-off-anchor/`.)"""
    seq = shape.get("tokens") or 12288
    if shape.get("target_modules_class") == "all-linear":
        bundle = "gpu-rate-9b-w20480"
        why = ("  # all-linear: fit-ladder would mint a list-7 anchor and the "
               "query would still refuse")
    else:
        bundle = "fit-ladder"
        why = ""
    return (f"tools/vast/rehearse.sh tools/witness/jobs/{bundle} --image  "
            f"# then run {bundle} with BASE={shape['base_slug'] or '<base>'} "
            f"MAX_SEQ={seq} QUANT={shape['quant_mode']} to mint the anchor{why}")


def estimate_peak_gb(*, facts=None, prefer_measured_tokens: bool = False,
                     allow_extrapolation: bool = False, **query) -> dict:
    """Measured peak-allocated GB per card for `query`'s shape.

    Returns {gb, n, min, max, spread, extrapolated, tokens_measured, anchors}.
    Raises `Unmeasured` when no anchor group matches — see that class for why
    this refuses instead of falling back to arithmetic.

    `allow_extrapolation` is FALSE by default and that is the whole safety
    property. A window past the group's longest anchor used to be answered by
    adding a slope, and the slope was measured on a slice that had stopped
    growing: measured against exercised windows on a 5090 (2026-08-29) those
    answers under-read by 10–26%, which is a tool telling you a shape fits a
    card it OOMs on. An under-read here surfaces as an OOM on rented hardware,
    so the refusal is cheaper than the answer. Opting in charges
    `EXTRAPOLATION_SLOPE_FLOOR_GB_PER_1K` at minimum and folds the whole
    extrapolated increment into `risk_gb`; it is still not a measurement.

    `prefer_measured_tokens` exists for `backtest()` and is FALSE in production:
    the query side is a declared window, so the anchor side is too. Turning it
    on is the arm that measured worse — `anchor_tokens`."""
    doc = facts if facts is not None else load_facts()
    want = normalize_shape(**query)
    if not want["base_slug"]:
        raise Unmeasured(
            "no base model named — a VRAM estimate keyed on an unidentified "
            "base would be a guess wearing a measurement's clothes")
    key = _group_of(want)
    group = [a for a in doc["anchors"]
             if a.get("base_slug") and _group_of(_anchor_shape(a)) == key]
    if not group:
        near = sorted({a["base_slug"] for a in doc["anchors"]
                       if a.get("base_slug") == want["base_slug"]})
        detail = (f"base {want['base_slug']!r} has anchors, but none at "
                  f"quant={want['quant_mode']} grad_ckpt={want['grad_checkpointing']} "
                  f"gc={want['gc_flag']} "
                  f"ce={want['ce_chunk_matmul']} targets={want['target_modules_class']} "
                  f"r={want['lora_r']} packing={want['packing']} "
                  f"sharded={want['sharded']}"
                  if near else
                  f"no anchor at all for base {want['base_slug']!r}")
        raise Unmeasured(f"unmeasured shape: {detail}",
                         probe_cmd=_probe_cmd(want), near=near)

    want_tokens = want.get("tokens")
    # Only anchors at or below the requested window. Window costs almost nothing
    # (§8.2) but it is not NEGATIVE, so a longer measured run is an upper bound
    # on a shorter one and charging it would over-size — 9B at seq 2048 would
    # otherwise inherit seq 12288's peak. If every anchor sits above the request
    # they are still valid upper bounds, so use them rather than refusing.
    # DECLARED tokens on both sides. `want_tokens` is a declared window (a
    # query cannot know its corpus's longest row), so comparing it against a
    # measured anchor coordinate compares two different axes — backtested at
    # -25 points of card-class accuracy, `anchor_tokens`.
    def _tok(a):
        return anchor_tokens(a, prefer_measured=prefer_measured_tokens)

    at_or_below = [a for a in group
                   if not want_tokens or not _tok(a) or _tok(a) <= want_tokens]
    used = at_or_below or group
    bounded_above = not at_or_below

    peaks = [a["measured"]["peak_vram_alloc_gb"] for a in used]
    toks = [_tok(a) for a in used]
    measured_max_tokens = max([t for t in toks if t] or [0])
    # MAX, not mean — the within-group spread is real and unexplained (up to
    # 6.3 GB on this data), and the failure mode of the mean is an OOM.
    gb = max(peaks)
    extrapolated = False
    extrapolated_gb = 0.0
    if want_tokens and measured_max_tokens and want_tokens > measured_max_tokens:
        gap = want_tokens - measured_max_tokens
        if not allow_extrapolation:
            raise Unmeasured(
                f"unmeasured window: this group's longest anchor is "
                f"{measured_max_tokens} tokens and the query asks for "
                f"{want_tokens} ({gap} past it). Extrapolating that gap is "
                f"what under-read every 5090 row by 10-26% "
                f"(GPU5090_SIZING_2026-08-29 §5.2), so it refuses: mint the "
                f"anchor, or pass --allow-extrapolation and read the answer as "
                f"a guess with a floor under it",
                probe_cmd=_probe_cmd(dict(want, tokens=want_tokens)),
                near=sorted({a.get("run", "") for a in used}))
        slope = group_slope_gb_per_1k(used, prefer_measured=prefer_measured_tokens)
        extrapolated_gb = slope * gap / 1024.0
        gb += extrapolated_gb
        extrapolated = True
    group = used
    return {
        "gb": round(gb, 2),
        "extrapolated_gb": round(extrapolated_gb, 2),
        "n": len(group),
        "min": round(min(peaks), 2),
        "max": round(max(peaks), 2),
        "spread": round(max(peaks) - min(peaks), 2),
        "extrapolated": extrapolated,
        "bounded_above": bounded_above,
        "tokens_measured": measured_max_tokens,
        "tokens_requested": want_tokens,
        "runs": sorted({a.get("run", "") for a in group}),
    }


# --- the grad-checkpointing-OFF calibration autotune reads --------------------
# `autotune.pick_grad_ckpt` has to answer "does OFF fit on this card?" box-side,
# from a bash mirror, knowing only (per-card VRAM, batch, seq) — no base, no
# quant, no facts file. It therefore cannot do the group lookup above, and ships
# two SCALAR constants instead. This function is where those scalars come from:
# a pure lookup over the grad-ckpt-OFF anchors, so `test_autotune.py` can bind
# the shipped constants to measurement and fail loudly if a re-harvest moves
# either one. Nothing here extrapolates.
#
# NB 0.063 GB/1k (`DISPROVEN_FLAT_SLICE_SLOPE_GB_PER_1K`) MUST NOT be used for a
# grad-ckpt-OFF shape, and since 2026-08-29 it is not used for an ON shape
# either — it was read off a slice whose rows had stopped growing, and even the
# checkpointed 9B measures 0.412-0.446 GB/1k on exercised windows. OFF is worse
# again: the activation stack IS the footprint and scales ~linearly with
# tokens-in-flight (measured 4.18x for 4x the tokens,
# TRAINING_DEFAULTS_REVIEW_2026-08-09.md §2). Charging 0.063 GB/1k to an OFF
# shape predicts ~52.7 GB at 12288 tokens, where the measurement is an OOM on a
# 94.97 GiB card — which is why this path is a pure lookup with no slope in it.
GRAD_CKPT_OFF_REF_TOKENS = 4096   # BATCH 1 x seq 4096, the reference window


def grad_ckpt_off_calibration(facts=None, ref_tokens: int = GRAD_CKPT_OFF_REF_TOKENS) -> dict:
    """The measured grad-checkpointing-OFF anchors, as two scalars.

    ``ref_gb``  — worst measured OFF peak AT ``ref_tokens`` tokens-in-flight.
                  This is the proportional rule's anchor point.
    ``floor_gb``— worst measured OFF peak BELOW ``ref_tokens``. A proportional
                  rule through the origin under-predicts short windows, because
                  weights + optimizer + static state do not scale with tokens:
                  proportional-from-52.2 says 13.05 GB at 1024 tokens and the
                  measurement there is 21.87. So the shipped rule takes the max
                  of the two terms. 0.0 when nothing below the reference is
                  measured (no floor claimed rather than a guessed one).

    MAX within each bucket, matching `estimate_peak_gb`: the within-group spread
    is real and the failure mode of the mean is an OOM. Raises `Unmeasured` when
    no OFF anchor sits at the reference window at all — the calibration would
    then be a guess, and this module does not make those."""
    doc = facts if facts is not None else load_facts()
    off = [a for a in doc.get("anchors", [])
           if _anchor_shape(a)["grad_checkpointing"] == "false"]
    at_ref, below = [], []
    for a in off:
        # Declared coordinate: `ref_tokens` is a declared window (autotune's
        # bash mirror knows batch x seq and nothing else), so the anchors have
        # to be on that same axis. Same argument as the estimate path.
        t = anchor_tokens(a, prefer_measured=False)
        if t == int(ref_tokens):
            at_ref.append(a)
        elif t and t < int(ref_tokens):
            below.append(a)
    if not at_ref:
        raise Unmeasured(
            f"no grad-ckpt-OFF anchor at {ref_tokens} tokens-in-flight "
            f"({len(off)} OFF anchors on file, none at the reference window) — "
            f"autotune's fit rule cannot be calibrated from measurement",
            probe_cmd=_probe_cmd({"base_slug": "", "quant_mode": "bf16",
                                  "tokens": int(ref_tokens)}))

    def _peak(a):
        return a["measured"]["peak_vram_alloc_gb"]

    ref = max(at_ref, key=_peak)
    flo = max(below, key=_peak) if below else None
    return {
        "ref_gb": round(_peak(ref), 2),
        "ref_tokens": int(ref_tokens),
        "ref_run": ref.get("run", ""),
        "ref_base": ref.get("base_slug", ""),
        "ref_quant": _anchor_shape(ref)["quant_mode"],
        "floor_gb": round(_peak(flo), 2) if flo else 0.0,
        "floor_tokens": anchor_tokens(flo) if flo else 0,
        "floor_run": flo.get("run", "") if flo else "",
        "n_off_anchors": len(off),
    }


def card_class_for(gb: float) -> int:
    """Round up to a card we can rent. Above the biggest class, return the need
    itself rather than a class — a shape that fits nothing should say so."""
    for c in CARD_CLASSES:
        if gb <= c:
            return c
    return int(math.ceil(gb))


def headroom_for(est: dict, base_headroom_gb: float = RESERVED_HEADROOM_GB) -> float:
    """Margin over a group's measured max: a flat reserved-pool constant.

    A spread-proportional term was tried and REJECTED on measurement. Adding a
    group's observed spread (up to 6.3 GB) on top of its max looks prudent and
    is not: scored on the decision that actually costs money — which card class
    you rent — it picked the right class for 34% of held-out anchors against
    65% for the flat rule, while preventing exactly ZERO additional class-level
    under-sizes (1 either way). It bought GB of paper safety by routinely
    renting one class too big.

    The spread is still the honest measure of residual risk, so it is REPORTED
    (`spread`, `risk_gb`) for a caller to act on rather than silently folded
    into the number. `jobmeta`'s submit gate widens its refusal band by it."""
    return base_headroom_gb


def required_gpu_ram_gb(*, facts=None, headroom_gb=None, **query) -> dict:
    """What `needs.gpu_ram_gb` should be: the measured peak plus headroom,
    rounded up to a rentable card class.

    NB `needs.gpu_ram_gb` is a PER-CARD floor — jobd compares it against the
    largest single card on the box (`onstart/jobd.sh`), not the box total. A
    2x24 GB box does not satisfy 48."""
    est = estimate_peak_gb(facts=facts, **query)
    head = headroom_for(est) if headroom_gb is None else float(headroom_gb)
    need = est["gb"] + head
    # How far a NEW run of this shape could plausibly land above the estimate:
    # the group's own unexplained scatter, PLUS the whole extrapolated increment
    # when the window ran past the longest anchor. That second term is what this
    # comment promised and the code did not do until 2026-08-29 — the arm that
    # under-read the 5090 by 10-26% reported `risk_gb` 0.00, because a
    # single-anchor group has no spread and the extrapolation contributed
    # nothing. Not added to the number (see `headroom_for`) — it is what a gate
    # should widen its band by.
    risk = (max(0.0, float(est.get("spread") or 0.0))
            + max(0.0, float(est.get("extrapolated_gb") or 0.0)))
    return dict(est, headroom_gb=round(head, 2), required_gb=round(need, 2),
                risk_gb=round(risk, 2), card_class=card_class_for(need))


# --- knob-impact report -------------------------------------------------------
# "Which knobs were used, and what did they cost?" answered from the anchor
# table alone — no GPU, no rerun, no network. The whole method is MATCHED PAIRS:
# two anchor sets identical in every recorded dimension except the knob under
# test. Nothing else is admissible here, because the alternative — comparing a
# knob's mean across the table — averages over shapes that differ in more than
# the axis being measured. (This comment used to cite the `0.455 GB/1k` slope as
# that confound's output. It was not: 0.455 was a single-anchor term
# decomposition, and 2026-08-29 measured it essentially RIGHT — 0.412-0.446 on
# exercised windows. The module docstring carries the correction. The matched-
# pair rule stands on its own.) Where no matched pair exists the answer is
# "unmeasured", printed as such and never interpolated.

REPORT_KNOBS = ("ce_chunk_matmul", "grad_checkpointing", "quant_mode", "packing",
                "world_size", "lora_r", "target_modules_class")


def _context_of(anchor: dict, knob: str, *, match_corpus: bool = True) -> tuple:
    """Everything held FIXED while `knob` varies.

    That is: every group key except the knob, plus the tokens coordinate (a
    window difference is not the knob's doing), plus — by default — the corpus
    hash. The corpus belongs in the key because the largest unexplained effect
    in this table is corpus-driven: two runs of one byte-identical declared
    shape measured 20.77 and 24.56 GB, differing only in which rows survived the
    drop policy. A "pair" straddling that is reporting 3.8 GB of corpus as if it
    were the knob. `--ignore-corpus` relaxes it and the report says so."""
    shape = _anchor_shape(anchor)
    key = tuple((k, shape.get(k)) for k in GROUP_KEYS if k != knob)
    key += (("tokens", anchor_tokens(anchor)),)
    if knob != "world_size":
        key += (("world_size", shape.get("world_size")),)
    if match_corpus:
        key += (("corpus", (anchor.get("context") or {})
                 .get("dataset_content_sha256")),)
    return key


def _median(values):
    v = sorted(values)
    if not v:
        return None
    mid = len(v) // 2
    return v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2.0


def _timing_key(anchor: dict) -> str:
    """What a per-step wall time is denominated in: micro-batches per step.

    `batch` x `grad_accum` are NOT part of the VRAM group key (grad_accum moves
    no memory), so two anchors can be a legitimate matched VRAM pair and still
    be timing-incomparable. This is the guard."""
    s = (anchor or {}).get("shape") or {}
    return f"b{s.get('batch')}xga{s.get('grad_accum')}"


def _side(anchors: list) -> dict:
    """One arm of a matched pair, summarised on the statistics each measure
    actually supports.

    VRAM is a MAX, for the reason the estimator takes a max: the number that
    matters is the worst case a card has to hold, and the mean's failure mode is
    an OOM. Throughput is a MEDIAN: it is a rate, several anchors of one shape
    are repeats of the same measurement, and one preempted or contended run
    would drag a mean.

    `step_time_s` comes from the v1 `step_time_seconds` field and is a WALL
    TIME PER OPTIMIZER STEP, so it is not a rate: a run with grad_accum 16 takes
    16 micro-batches per step and one with grad_accum 1 takes one. Neither of
    those is in the anchor group key (they do not move the peak), so a matched
    VRAM pair can straddle them — measured here, ws 1 -> 2 read +239% on one
    pair and -45% on another. `timing_key` is carried so the caller can refuse
    the comparison rather than publish that as a knob effect."""
    peaks = [a["measured"]["peak_vram_alloc_gb"] for a in anchors]
    tps = [((a.get("telemetry") or {}).get("throughput") or {}).get("tokens_per_second")
           for a in anchors]
    tps = [t for t in tps if isinstance(t, (int, float))]
    steps = [(a.get("context") or {}).get("step_time_seconds") for a in anchors]
    steps = [s for s in steps if isinstance(s, (int, float)) and s > 0]
    power = [((a.get("telemetry") or {}).get("gpu_power") or {}).get("power_mean_w")
             for a in anchors]
    power = [p for p in power if isinstance(p, (int, float))]
    out = {
        "n": len(anchors),
        "peak_max": round(max(peaks), 2),
        "peak_min": round(min(peaks), 2),
        "runs": sorted({a.get("run", "") for a in anchors}),
        "timing_key": sorted({_timing_key(a) for a in anchors}),
    }
    if tps:
        out["tokens_per_second"] = round(_median(tps), 1)
        out["n_throughput"] = len(tps)
    if steps:
        out["step_time_s"] = round(_median(steps), 3)
        out["n_step_time"] = len(steps)
    if power:
        out["power_mean_w"] = round(_median(power), 1)
    return out


def _value_sort_key(v):
    """Deterministic ordering of a knob's values across mixed types — a knob
    reads as ints (`lora_r`, `world_size`), strings, or None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return (1, 0.0, str(v))
    return (0, float(v), "")


def knob_findings(facts: dict, knob: str, *, match_corpus: bool = True) -> list:
    """Every matched pair in the table for one knob. Pure.

    Returns a list of {knob, context, from, to, a, b, delta_gb, delta_tps,
    delta_tps_pct, delta_step_pct} — one per (context, ordered value pair).
    Empty means unmeasured: no two anchors differ in this knob alone."""
    anchors = [a for a in (facts.get("anchors") or []) if a.get("base_slug")]
    contexts = {}
    for a in anchors:
        val = _anchor_shape(a).get(knob)
        # An UNRECORDED knob is not a knob SETTING. `unknown` / None is what
        # `normalize_shape` produces when the summary never wrote the field
        # (older trainers did not record ce_chunk_matmul or world_size at all),
        # and pairing `fp32 -> unknown` publishes "the field was missing" as if
        # it were a measured value.
        if val is None or (isinstance(val, str) and val.strip() == "unknown"):
            continue
        ctx = _context_of(a, knob, match_corpus=match_corpus)
        contexts.setdefault(ctx, {}).setdefault(val, []).append(a)

    findings = []
    for ctx, by_val in contexts.items():
        if len(by_val) < 2:
            continue
        vals = sorted(by_val, key=_value_sort_key)
        for i, lo in enumerate(vals):
            for hi in vals[i + 1:]:
                a, b = _side(by_val[lo]), _side(by_val[hi])
                f = {
                    "knob": knob,
                    "context": dict(ctx),
                    "from": lo, "to": hi, "a": a, "b": b,
                    # BOTH ends, because they answer different questions and
                    # can disagree by a lot. max-vs-max is the sizing number
                    # (what a card must hold either way); min-vs-min is the
                    # conservative reading of the effect. On the shipped table
                    # ce fp32->bf16 reads 9.68 GB on the max and 3.36 on the
                    # min at seq 12288 — the gap is the group's own spread, not
                    # the knob, and quoting only the max would overstate it.
                    "delta_gb": round(b["peak_max"] - a["peak_max"], 2),
                    "delta_gb_min": round(b["peak_min"] - a["peak_min"], 2),
                }
                if "tokens_per_second" in a and "tokens_per_second" in b:
                    f["delta_tps"] = round(
                        b["tokens_per_second"] - a["tokens_per_second"], 1)
                    if a["tokens_per_second"]:
                        f["delta_tps_pct"] = round(
                            100.0 * f["delta_tps"] / a["tokens_per_second"], 1)
                if "step_time_s" in a and "step_time_s" in b and a["step_time_s"]:
                    # Only when both sides are one and the same denomination —
                    # see `_timing_key`. Otherwise the pair carries the reason
                    # the comparison was refused, so the readout can say
                    # "not compared" instead of printing a confound.
                    if (len(a["timing_key"]) == 1 == len(b["timing_key"])
                            and a["timing_key"] == b["timing_key"]):
                        f["delta_step_pct"] = round(
                            100.0 * (b["step_time_s"] - a["step_time_s"])
                            / a["step_time_s"], 1)
                    else:
                        f["step_time_incomparable"] = (
                            f"micro-batches per step differ "
                            f"({'/'.join(a['timing_key'])} vs "
                            f"{'/'.join(b['timing_key'])})")
                findings.append(f)
    findings.sort(key=lambda f: (-abs(f["delta_gb"]), str(f["context"]),
                                 str(f["from"]), str(f["to"])))
    return findings


def knob_report(facts: dict = None, knobs=REPORT_KNOBS, *,
                match_corpus: bool = True) -> dict:
    """`{knob: [findings]}` for every knob, plus the table's telemetry census."""
    doc = facts if facts is not None else load_facts()
    anchors = [a for a in (doc.get("anchors") or []) if a.get("base_slug")]
    return {
        "match_corpus": match_corpus,
        "n_anchors": len(anchors),
        "n_with_throughput": sum(
            1 for a in anchors
            if ((a.get("telemetry") or {}).get("throughput") or {})
            .get("tokens_per_second") is not None),
        "n_with_token_stats": sum(
            1 for a in anchors if (a.get("telemetry") or {}).get("token_stats")),
        "n_with_step_time": sum(
            1 for a in anchors if (a.get("context") or {}).get("step_time_seconds")),
        "knobs": {k: knob_findings(doc, k, match_corpus=match_corpus)
                  for k in knobs},
    }


def _ctx_str(ctx: dict, knob: str) -> str:
    short = {"base_slug": "", "quant_mode": "", "grad_checkpointing": "gck",
             "ce_chunk_matmul": "ce", "target_modules_class": "tm",
             "lora_r": "r", "packing": "pack", "sharded": "shard",
             "world_size": "ws", "tokens": "tok", "corpus": "corpus"}
    bits = []
    for k, v in ctx.items():
        if k == knob or v is None and k != "corpus":
            continue
        if k == "corpus":
            bits.append(f"corpus={str(v)[:8]}" if v else "corpus=none")
        elif short.get(k) == "":
            bits.append(str(v))
        else:
            bits.append(f"{short.get(k, k)}={v}")
    return " ".join(bits)


def format_knob_report(rep: dict, *, max_pairs: int = 6) -> list:
    """The human readout. One block per knob; `unmeasured` where the table has
    no pair that isolates it."""
    lines = [f"knob impact from {rep['n_anchors']} usable anchors  "
             f"({rep['n_with_step_time']} carry step time, "
             f"{rep['n_with_throughput']} carry tokens/s, "
             f"{rep['n_with_token_stats']} carry token_stats)"]
    lines.append(
        "matched pairs only: two anchor sets identical in every recorded "
        "dimension except the knob"
        + (", corpus hash included." if rep["match_corpus"] else
           " — but NOT the corpus (--ignore-corpus), so a delta here may be "
           "which rows ran, not the knob.")
        + "  A pair reading corpus=none matched two runs that recorded no "
          "dataset hash; they may still differ in corpus.")
    for knob, findings in rep["knobs"].items():
        lines.append("")
        if not findings:
            lines.append(f"{knob}: unmeasured — no two anchors differ in this "
                         f"knob alone")
            continue
        lines.append(f"{knob}: {len(findings)} matched pair(s)")
        for f in findings[:max_pairs]:
            a, b = f["a"], f["b"]
            sign = "+" if f["delta_gb"] >= 0 else ""
            extra = ""
            if "delta_tps_pct" in f:
                extra += (f", tokens/s {a['tokens_per_second']:.0f} -> "
                          f"{b['tokens_per_second']:.0f} "
                          f"({f['delta_tps_pct']:+.1f}%)")
            if "delta_step_pct" in f:
                extra += (f", step {a['step_time_s']:.2f}s -> "
                          f"{b['step_time_s']:.2f}s "
                          f"({f['delta_step_pct']:+.1f}%)")
            elif "step_time_incomparable" in f:
                extra += f", step time not compared ({f['step_time_incomparable']})"
            span = ""
            if f["delta_gb"] != f["delta_gb_min"]:
                span = (f", {f['delta_gb_min']:+.2f} on the min "
                        f"({a['peak_min']:.2f} -> {b['peak_min']:.2f})")
            lines.append(f"  {f['from']} -> {f['to']}: {sign}{f['delta_gb']:.2f} GB "
                         f"on the max ({a['peak_max']:.2f} -> {b['peak_max']:.2f})"
                         f"{span}, n={a['n']} vs {b['n']}{extra}")
            lines.append(f"      where: {_ctx_str(f['context'], knob)}")
            lines.append(f"      runs:  {', '.join(a['runs'][:2])}"
                         f"  vs  {', '.join(b['runs'][:2])}")
        if len(findings) > max_pairs:
            lines.append(f"  ... {len(findings) - max_pairs} more pair(s) "
                         f"(--json for all)")
    return lines


# --- leave-one-out backtest ---------------------------------------------------
# Drop each anchor, estimate its shape from the rest, and score the decision
# that costs money: which card class you would have rented. Lives here rather
# than inside `test_vram_facts.py` because it has two callers now — the
# regression test, and the A/B that decides whether a coordinate change earns
# its place (`--backtest`). A harness that only exists inside an assertion
# cannot be pointed at two arms.

def _anchor_query(anchor: dict) -> dict:
    """The submit-time query a run of this anchor's shape would have made.
    Deliberately built from the DECLARED shape: a bundle asking `job submit`
    for a card knows its `--max-seq` and batch, and does not know the longest
    row its corpus will produce."""
    shape = _anchor_shape(anchor)
    raw = anchor.get("shape") or {}
    return dict(
        base_slug=anchor["base_slug"], quant_mode=shape["quant_mode"],
        grad_checkpointing=shape["grad_checkpointing"],
        # The GC FRACTION is declared in the bundle exactly like the switch, so
        # the submit-time query knows it; omitting it defaulted every query to
        # gc=full and parted 64 held-out anchors from their own fractional
        # groups the day _group_of started keying on it.
        grad_checkpointing_flag=raw.get("grad_checkpointing_flag"),
        ce_chunk_matmul=shape["ce_chunk_matmul"],
        target_modules=raw.get("target_modules"), lora_r=shape["lora_r"],
        packing=shape["packing"], fsdp=raw.get("fsdp"),
        device_map_mode=raw.get("device_map_mode"),
        world_size=shape["world_size"], max_seq=raw.get("max_seq"),
        batch=raw.get("batch"))


def backtest(facts=None, *, prefer_measured_tokens: bool = False,
             oracle_query_tokens: bool = False, score_only=None) -> dict:
    """Leave-one-out over the anchor table. Returns the error DISTRIBUTION.

    `prefer_measured_tokens` is the arm switch of the coordinate A/B: FALSE
    (production) puts anchors and query alike on the declared `batch x max_seq`
    axis; TRUE moves the anchors onto their measured longest row and leaves the
    query where it is.

    `oracle_query_tokens` additionally moves the QUERY onto the held-out run's
    own measured coordinate. Not shippable — a submit does not know the longest
    row its corpus will produce — but it separates "the coordinate is wrong"
    from "applying it to one side only is wrong", and those have different
    fixes.

    `score_only` is a predicate on an anchor: which held-out points are SCORED.
    The evidence set is the whole table either way; restricting the scored
    points is what makes two arms comparable (an arm scored on a different
    subset than the other is not a comparison).

    BIASED AGAINST ITSELF, deliberately. The estimate is a group MAX, so
    holding out a group's maximum GUARANTEES an under-prediction of that point:
    the anchor being predicted is the evidence that would have covered it. In
    production every measured shape is in the table, so the under-count here is
    a pessimistic proxy for "a new run lands above everything seen", not a
    defect rate."""
    doc = facts if facts is not None else load_facts()
    anchors = [a for a in doc["anchors"] if a.get("base_slug")]
    groups = {}
    for a in anchors:
        groups.setdefault(_group_of(_anchor_shape(a)), []).append(a)

    errors, under_class, under_points, over, exact = [], [], [], 0, 0
    singleton = unmeasured = 0
    for a in anchors:
        if score_only is not None and not score_only(a):
            continue
        group = groups[_group_of(_anchor_shape(a))]
        if len(group) < 2:
            singleton += 1
            continue
        held = {"schema": 1, "anchors": [x for x in group if x is not a]}
        query = _anchor_query(a)
        if oracle_query_tokens:
            t = anchor_tokens(a, prefer_measured=True)
            if t:
                query["max_seq"], query["batch"] = t, 1
        try:
            # Scored WITH extrapolation on: holding out a group's longest
            # anchor is exactly the case the production default now refuses, and
            # a backtest that skipped those points would stop measuring the arm
            # it exists to measure.
            r = required_gpu_ram_gb(
                facts=held, prefer_measured_tokens=prefer_measured_tokens,
                allow_extrapolation=True, **query)
        except Unmeasured:
            unmeasured += 1
            continue
        actual = a["measured"]["peak_vram_alloc_gb"]
        signed = round(r["required_gb"] - actual, 4)
        errors.append(signed)
        if signed < 0:
            under_points.append({
                "run": a.get("run", ""),
                "unit": (a.get("sources") or [{}])[0].get("path", ""),
                "actual_gb": actual, "predicted_gb": r["required_gb"],
                "signed_gb": signed})
        truth = card_class_for(actual + RESERVED_HEADROOM_GB)
        if r["card_class"] == truth:
            exact += 1
        elif r["card_class"] > truth:
            over += 1
        else:
            under_class.append({
                "base": a.get("base_slug", ""), "run": a.get("run", ""),
                "unit": (a.get("sources") or [{}])[0].get("path", ""),
                "actual_gb": actual, "predicted_gb": r["required_gb"],
                "truth_class": truth, "predicted_class": r["card_class"]})

    n = len(errors)
    absolute = sorted(abs(e) for e in errors)

    def _pct(p):
        """Nearest-rank, stated because a 36-point distribution's p90 moves by
        a whole GB between conventions and these numbers get quoted."""
        if not absolute:
            return 0.0
        i = min(len(absolute) - 1, int(math.ceil(p * len(absolute))) - 1)
        return absolute[max(0, i)]

    return {
        "prefer_measured_tokens": prefer_measured_tokens,
        "oracle_query_tokens": oracle_query_tokens,
        "n": n,
        "exact": exact,
        "over": over,
        "under_class": under_class,
        "under_points": under_points,
        "n_under_gb": len(under_points),
        "class_accuracy": round(exact / n, 4) if n else 0.0,
        "median_abs_gb": round(statistics.median(absolute), 2) if absolute else 0.0,
        "p90_abs_gb": round(_pct(0.9), 2),
        "max_abs_gb": round(absolute[-1], 2) if absolute else 0.0,
        "median_signed_gb": round(statistics.median(errors), 2) if errors else 0.0,
        "worst_under_gb": round(min(errors), 2) if errors else 0.0,
        "worst_over_gb": round(max(errors), 2) if errors else 0.0,
        "skipped_singleton": singleton,
        "skipped_unmeasured": unmeasured,
        "errors": errors,
    }


def format_backtest(arms: dict) -> list:
    """Side-by-side readout of named arms, distribution first."""
    lines = ["arm                      n  class-ok   median|e|   p90|e|   "
             "max|e|   under(GB)  under(class)"]
    for name, r in arms.items():
        lines.append(
            f"{name:20s} {r['n']:5d}  {100 * r['class_accuracy']:5.1f}%  "
            f"{r['median_abs_gb']:9.2f} {r['p90_abs_gb']:8.2f} "
            f"{r['max_abs_gb']:8.2f} {r['n_under_gb']:10d} "
            f"{len(r['under_class']):12d}")
    for name, r in arms.items():
        for u in r["under_class"]:
            lines.append(f"  {name}: class under-size {u['unit'] or u['run']} "
                         f"measured {u['actual_gb']:.2f} -> predicted "
                         f"{u['predicted_gb']:.2f} "
                         f"({u['predicted_class']} < {u['truth_class']})")
    return lines


# --- CLI ----------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # `--base` stays a flag rather than becoming a subcommand's positional:
    # `vram_facts.py --base X --quant bf16` is the documented invocation and is
    # called from docs, the gate's advice strings and muscle memory. `--report`
    # is a second mode on the same parser, so nothing existing moves.
    p.add_argument("--base", help="base model slug, e.g. qwen35-9b")
    p.add_argument("--quant", default="bf16")
    p.add_argument("--max-seq", type=int)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-ckpt", default="true")
    # Mirrors train_proposer_lora.py's own default (bf16 since 2026-08-10): a
    # query with no --ce-chunk-matmul asks about the shape a run would ACTUALLY
    # take. Pass fp32 explicitly to size a bundle that pins fp32.
    p.add_argument("--ce-chunk-matmul", default="bf16")
    p.add_argument("--target-modules", default=None,
                   help="'all-linear', or a comma list; default = the 7-name list")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--packing", default="off")
    p.add_argument("--fsdp", default="")
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--json", action="store_true")
    p.add_argument("--allow-extrapolation", action="store_true",
                   help="answer a window PAST the group's longest anchor by "
                        f"charging at least "
                        f"{EXTRAPOLATION_SLOPE_FLOOR_GB_PER_1K} GB/1k tokens. "
                        "Off by default: the old 0.063 GB/1k extrapolation "
                        "under-read every exercised-window measurement by "
                        "10-26%, i.e. it said 'fits' about shapes that OOM")
    p.add_argument("--facts", default=FACTS_PATH)
    p.add_argument("--report", action="store_true",
                   help="knob-impact report: matched anchor pairs differing in "
                        "one knob, with the measured VRAM/throughput delta")
    p.add_argument("--knob", action="append", default=[],
                   choices=list(REPORT_KNOBS),
                   help="restrict --report to these knobs (repeatable)")
    p.add_argument("--ignore-corpus", action="store_true",
                   help="--report: allow pairs whose sides ran different "
                        "corpora. Widens the search and admits the 6.3 GB "
                        "same-shape spread into the delta")
    p.add_argument("--max-pairs", type=int, default=6,
                   help="--report: pairs printed per knob (default 6)")
    p.add_argument("--backtest", action="store_true",
                   help="leave-one-out over the anchor table, WITH and WITHOUT "
                        "the measured-row token coordinate, scored on the "
                        "anchors that carry token_stats")
    a = p.parse_args(argv)

    if a.backtest:
        try:
            facts = load_facts(a.facts)
        except VramFactsError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        def scored(anchor):
            return bool((anchor.get("telemetry") or {}).get("token_stats"))

        arms = {
            "A declared (shipped)": backtest(facts, score_only=scored),
            "B measured anchors": backtest(facts, prefer_measured_tokens=True,
                                           score_only=scored),
            "B' measured both": backtest(facts, prefer_measured_tokens=True,
                                         oracle_query_tokens=True,
                                         score_only=scored),
            "full table (declared)": backtest(facts),
            "full table (measured)": backtest(facts,
                                              prefer_measured_tokens=True),
        }
        if a.json:
            print(json.dumps(arms, indent=2))
        else:
            n_ts = sum(1 for x in facts["anchors"]
                       if (x.get("telemetry") or {}).get("token_stats"))
            print(f"leave-one-out backtest — {len(facts['anchors'])} anchors, "
                  f"{n_ts} carry token_stats")
            print("A/B/B' are scored on the SAME held-out points (the "
                  "token_stats subset); the evidence set is the whole table in "
                  "every arm. B' is an ORACLE — it hands the query the "
                  "held-out run's own measured coordinate, which a submit "
                  "cannot know.")
            print("\n".join(format_backtest(arms)))
        return 0

    if a.report:
        try:
            facts = load_facts(a.facts)
        except VramFactsError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        rep = knob_report(facts, tuple(a.knob) or REPORT_KNOBS,
                          match_corpus=not a.ignore_corpus)
        if a.json:
            print(json.dumps(rep, indent=2, default=str))
        else:
            print("\n".join(format_knob_report(rep, max_pairs=a.max_pairs)))
        return 0

    if not a.base:
        p.error("--base is required (or use --report)")

    tm = a.target_modules
    if tm and tm != "all-linear":
        tm = [x.strip() for x in tm.split(",") if x.strip()]
    elif tm is None:
        tm = ["q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"]

    try:
        r = required_gpu_ram_gb(
            facts=load_facts(a.facts), base_slug=a.base, quant_mode=a.quant,
            max_seq=a.max_seq, batch=a.batch, grad_checkpointing=a.grad_ckpt,
            ce_chunk_matmul=a.ce_chunk_matmul, target_modules=tm, lora_r=a.lora_r,
            packing=a.packing, fsdp=a.fsdp, world_size=a.world_size,
            allow_extrapolation=a.allow_extrapolation)
    except Unmeasured as e:
        print(f"UNMEASURED: {e}", file=sys.stderr)
        if e.probe_cmd:
            print(f"  measure it: {e.probe_cmd}", file=sys.stderr)
        return 3
    except VramFactsError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if a.json:
        print(json.dumps(r, indent=2))
        return 0
    flag = ""
    if r["extrapolated"]:
        flag = (f" — {r['extrapolated_gb']:.2f} GB of this is EXTRAPOLATED "
                f"{r['tokens_requested'] - r['tokens_measured']} tokens past the "
                f"longest anchor and is NOT measured")
    print(f"measured peak   {r['gb']:6.2f} GB   (n={r['n']}, observed "
          f"{r['min']:.2f}-{r['max']:.2f}, spread {r['spread']:.2f}){flag}")
    print(f"+ headroom      {r['headroom_gb']:6.2f} GB   (reserved pool over allocated)")
    print(f"= needs         {r['required_gb']:6.2f} GB per card  ->  "
          f"gpu_ram_gb: {r['card_class']}")
    if r["risk_gb"] > 0.5:
        why = []
        if r["spread"] > 0:
            why.append(f"{r['spread']:.2f} GB of scatter between identical "
                       f"declared shapes")
        if r["extrapolated_gb"] > 0:
            why.append(f"{r['extrapolated_gb']:.2f} GB of unmeasured "
                       f"extrapolation")
        print(f"  risk          {r['risk_gb']:6.2f} GB   " + " + ".join(why))
    print(f"  anchors from: {', '.join(r['runs'][:4])}"
          + (" ..." if len(r["runs"]) > 4 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
