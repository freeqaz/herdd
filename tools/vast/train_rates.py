"""train_rates — training tok/s DERIVED from `vram_facts.json`, per card and job.

The measured half of the tokens-per-dollar ranker. `gpu_rates.py` is the
hand-curated table (~20 cells, authored shape strings); this reads the same
numbers the harvester already banks off every real run, so coverage grows
without anyone editing a table.

TWO RULES CARRY THE WHOLE DESIGN.

**No extrapolation.** A (card, family) cell with no anchor is UNMEASURED and
`rate_for_offer` returns None. Nothing here scales a rate across cards, windows
or card counts — same doctrine as `vram_facts.Unmeasured`, for the same reason
(the one analytic slope this project fitted was measured 7x wrong). Unmeasured
ranks last; it never becomes a zero and never becomes a guess.

**Staleness is one-directional, so a stale rate is a FLOOR.** Every lever landed
here — DDP metric-gather, padfree, chunked CE, the sm_120 flash build — made
training faster, so an anchor from an older stack under-states what that card
does today. It is therefore labeled `provisional` and ranked below `measured`
rather than dropped: ranking on a floor is honest, and dropping it would leave
the card unmeasured, which is worse. The one exception is a boundary big enough
to invert a ranking (`MULTI_GPU_EPOCH`), and that one excludes.
"""
from __future__ import annotations

import os
import re
import statistics
import sys
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:                 # sibling imports, jobmeta's pattern
    sys.path.insert(0, _HERE)

import gpu_rates                          # noqa: E402
import vram_facts                         # noqa: E402

__all__ = [
    "Family", "RateEstimate", "family_from_env", "rate_for_offer",
    "tokens_per_dollar", "probe_hint", "usable_anchors", "current_stack",
    "rate_census", "MULTI_GPU_EPOCH", "K_RECENT", "MIN_TIMED_STEPS",
    "SKU_FALLBACK",
]


# --------------------------------------------------------------------------- #
# constants — every one of these is a measurement decision, not a tuning knob
# --------------------------------------------------------------------------- #

# Anchors behind one rate: the median of the K most recent. New runs age old
# points out on their own, and a median survives one preempted or co-tenanted
# repeat (the H100 NVL row in gpu_rates.py is the worked example: one box drifted
# 33->40 s/it under a hot neighbour and recovered).
K_RECENT = 3

# A p50 over 2 timed steps is a warmup reading, not a steady-state rate. Measured
# on this file: the k3/k4 `fit_qla` probe cells report 423-726 tok/s where the
# same declared shape's 6- and 30-step cells report 3,400-5,200 — an 8x error
# that no tier or epoch key can see, because nothing in `shape` distinguishes a
# fit probe from a run. GPU_RATES.md already flags its own 2-step cells
# provisional; this refuses them outright.
MIN_TIMED_STEPS = 4

# THE HARD BOUNDARY, generalized from gpu_rates.py's header. `e48def36`
# (2026-08-14) made TELEMETRY_TOKEN_COUNT=0 + DDP_METRIC_GATHER=deferred the
# trainer defaults, worth a measured +40.7% at W=4 / +26.7% at W=8
# (docs/plans/witness/perf/W8_LADDER_RESULT_2026-08-14.md). A pre-boundary
# multi-card anchor under-states the current path by more than the gap between
# adjacent card classes, so it is EXCLUDED rather than demoted: it would not be a
# floor, it would be a wrong ranking. Single-card rates are unaffected (the W=1
# null reproduced 3x inside 0.05%). An undated multi-card anchor cannot be shown
# to be post-boundary, so it goes the same way.
MULTI_GPU_EPOCH = "2026-08-14"

# vast's offer NAME is ambiguous for these; answer with the SLOWER part, because
# ranking a box you might not get is a purchase decision. Mirrors
# gpu_rates._SKU_FALLBACK (pinned equal by test_train_rates).
SKU_FALLBACK = {
    "RTX PRO 6000 WS": ("RTX PRO 6000 WS MAXQ",),
    "A100 PCIE": ("A100 PCIE 40GB",),
    "A100 SXM4": ("A100 SXM4 40GB",),
}

TIERS = ("measured", "provisional")

# train_proposer_lora.py's own argparse defaults, for knobs a bundle leaves
# unset. Mirrors jobmeta._TRAINER_DEFAULTS and MUST track the trainer for the
# same reason: a stale value here mis-keys every bundle that leaves the knob out.
_TRAINER_DEFAULTS = {"ce_chunk_matmul": "bf16", "packing": "off",
                     "batch": 1, "grad_accum": 4, "lora_r": 32}
_TRAINER_DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj")


# --------------------------------------------------------------------------- #
# the key split: what the JOB fixes vs what the CARD gets to choose
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Family:
    """What a training job fixes before it knows which card it will run on.

    `eff_batch` and not `batch`/`grad_accum`: the split between micro-batch and
    accumulation is the box's to pick (that is what MODE=autotune does, under
    EXPECT_EFF_BATCH's fail-closed guard), and it moves tok/s a lot, so it is an
    OPERATING POINT. The product is what the optimizer sees and is pinned by the
    bundle. Note it includes world size — build the family with the card count
    you are ranking (`family_from_env(env, world_size=n)`).

    `max_seq` matches EXACTLY. It is a cap, not a length, and this module never
    interpolates between windows.
    """

    base_slug: str
    quant_mode: str
    max_seq: int
    eff_batch: int
    packing: str
    target_modules_class: str
    lora_r: int | None
    ce_chunk_matmul: str

    def key(self) -> tuple:
        return (self.base_slug, self.quant_mode, self.max_seq, self.eff_batch,
                self.packing, self.target_modules_class, self.lora_r,
                self.ce_chunk_matmul)

    def slug(self) -> str:
        return (f"{self.base_slug} {self.quant_mode} w{self.max_seq} "
                f"eb{self.eff_batch} pack={self.packing} "
                f"{self.target_modules_class}/r{self.lora_r} "
                f"ce={self.ce_chunk_matmul}")


# The operating point. `attn_impl` sits here and not in the family on purpose:
# ATTN_IMPL defaults to `auto` and RESOLVES ON THE BOX against the card's SM
# (flash_attention_2 is refused on sm_100/sm_120), so a submit cannot know it and
# it varies with the card — this module's definition of an operating point.
# `gc` is the grad-checkpointing FRACTION, which is the "more VRAM -> faster"
# lever itself: gc=none costs ~2.5x the peak and buys ~2x the tok/s.
_OpPoint = tuple


def _op_point(shape: dict) -> _OpPoint:
    gc = ("none" if shape.get("grad_checkpointing") is False
          else vram_facts.gc_flag_class(shape.get("grad_checkpointing_flag")))
    return (shape.get("batch"), shape.get("grad_accum"), gc,
            shape.get("attn_impl"), shape.get("sdpa_backends"))


def _op_point_str(op: _OpPoint) -> str:
    b, ga, gc, attn, backends = op
    s = f"b{b}xga{ga} gc={gc}"
    if attn:
        s += f" attn={attn}" + (f"/{backends}" if backends else "")
    return s


@dataclass(frozen=True)
class RateEstimate:
    tok_s: float
    tier: str                # "measured" | "provisional"
    n: int                   # anchors behind the median
    spread: float            # max/min of those anchors
    runs: tuple[str, ...]    # provenance run ids
    op_point: str            # e.g. "b1xga32 gc=full attn=sdpa/flashmeff"
    why: str                 # one-line derivation note


# --------------------------------------------------------------------------- #
# reading an anchor
# --------------------------------------------------------------------------- #
# Run ids are usually date-prefixed and not always: `20260816T043729-...`,
# `2026-08-22-mergeddemoa-...` and `ak1-t215-accept-20260820` all occur, and three
# anchors carry no date at all. Undated is a first-class state (provisional at
# best, excluded past the multi-GPU boundary), never a guessed date.
_DATE_PATTERNS = (re.compile(r"(20\d{2})(\d{2})(\d{2})T"),
                  re.compile(r"(20\d{2})-(\d{2})-(\d{2})"),
                  re.compile(r"(20\d{2})(\d{2})(\d{2})"))


def anchor_date(anchor: dict) -> str | None:
    """ISO date parsed out of the run id, or None. Sortable as a string."""
    run = str((anchor or {}).get("run") or "")
    for pat in _DATE_PATTERNS:
        m = pat.search(run)
        if m:
            y, mo, da = m.groups()
            if "01" <= mo <= "12" and "01" <= da <= "31":
                return f"{y}-{mo}-{da}"
    return None


def stack_fingerprint(anchor: dict) -> tuple | None:
    """(torch, flash_attn, transformers, trl) — the epoch key. None when the
    anchor predates schema 2 and records no hardware block at all."""
    hw = ((anchor or {}).get("telemetry") or {}).get("hardware") or {}
    fp = (hw.get("torch_version"), hw.get("flash_attn_version"),
          hw.get("transformers_version"), hw.get("trl_version"))
    return fp if any(fp) else None


def anchor_gpu(anchor: dict) -> str:
    """This module's card class for an anchor, via `gpu_rates.normalize_gpu_name`
    so anchor and vast offer meet in one vocabulary. "" for a heterogeneous box:
    a rate attributed to two different cards is not a rate for either."""
    hw = ((anchor or {}).get("telemetry") or {}).get("hardware") or {}
    names = {gpu_rates.normalize_gpu_name(n)
             for n in (hw.get("gpu_names") or []) if n}
    return names.pop() if len(names) == 1 else ""


def anchor_family(anchor: dict) -> Family | None:
    """None when the anchor carries no resolved base — `base_slug` is "" for 11
    harvested runs whose `shape.base` is a container path (`workspace/base`,
    `nudge/model`), and a family keyed on an unidentified model is a guess."""
    base = str((anchor or {}).get("base_slug") or "")
    shape = (anchor or {}).get("shape") or {}
    if not base or not shape.get("max_seq"):
        return None
    eff = shape.get("eff_batch")
    if not eff:
        eff = (int(shape.get("batch") or 1) * int(shape.get("grad_accum") or 1)
               * int(shape.get("world_size") or 1))
    return Family(
        base_slug=base,
        quant_mode=str(shape.get("quant_mode") or "unknown"),
        max_seq=int(shape["max_seq"]),
        eff_batch=int(eff),
        packing=str(shape.get("packing") or "unknown"),
        target_modules_class=vram_facts.target_modules_class(
            shape.get("target_modules")),
        lora_r=int(shape["lora_r"]) if shape.get("lora_r") else None,
        ce_chunk_matmul=str(shape.get("ce_chunk_matmul") or "unknown"),
    )


def anchor_rate(anchor: dict) -> float | None:
    """tok/s for an anchor: tokens-per-step over the p50 STEP TIME.

    Not the summary's own `throughput.tokens_per_second`, which is a segment
    average over everything the segment contained — model load, the ~165 s
    Triton compile of step 1, a checkpoint save. The two disagree by up to 39% on
    repeats that agree on p50 to 0.1% (ck2 `eager_gcon_r1/r2`: 5,557 vs 7,716
    reported, 12.500 vs 12.513 s p50), so the reported field carries warmup where
    the p50 does not. GPU_RATES.md's "Adding a rate" already specifies
    `throughput.step_time_p50_s` as the denominator; this is that division.

    Falls back to the reported field only when the triple is incomplete.
    """
    t = ((anchor or {}).get("telemetry") or {}).get("throughput") or {}
    n, p50, seen = (t.get("n_steps_timed"), t.get("step_time_p50_s"),
                    t.get("tokens_seen"))
    if all(isinstance(v, (int, float)) for v in (n, p50, seen)) and n and p50 > 0:
        return (seen / n) / p50
    v = t.get("tokens_per_second")
    return float(v) if isinstance(v, (int, float)) and v > 0 else None


def anchor_peak_per_card_gb(anchor: dict) -> float | None:
    """Per-CARD VRAM this anchor needs, or None when it cannot be established.

    `peak_vram_reserved_gb` is a MAX over cards, not a sum — verified against
    every one of the 150 anchors that carry both it and the per-GPU list, zero
    mismatches — so a multi-card anchor with an empty per-GPU list is still
    readable and nothing here ever divides by world size.

    Charged as max(allocated + RESERVED_HEADROOM_GB, reserved): the card has to
    hold the reserved pool, and `vram_facts`' headroom convention is what sizes
    the allocated figure. Returns None when neither is recorded, and a None makes
    the operating point unfittable rather than silently admitted.
    """
    m = (anchor or {}).get("measured") or {}

    def _per_card(whole, per):
        vals = [v for v in (m.get(per) or []) if isinstance(v, (int, float))]
        if vals:
            return max(vals)
        v = m.get(whole)
        return v if isinstance(v, (int, float)) else None

    alloc = _per_card("peak_vram_alloc_gb", "peak_vram_alloc_gb_per_gpu")
    reserved = _per_card("peak_vram_reserved_gb", "peak_vram_reserved_gb_per_gpu")
    cands = []
    if alloc is not None:
        cands.append(alloc + vram_facts.RESERVED_HEADROOM_GB)
    if reserved is not None:
        cands.append(reserved)
    return max(cands) if cands else None


def _world_size(anchor: dict) -> int:
    return int(((anchor or {}).get("shape") or {}).get("world_size") or 1)


def _past_multi_gpu_epoch(anchor: dict) -> bool:
    if _world_size(anchor) <= 1:
        return True
    d = anchor_date(anchor)
    return bool(d) and d >= MULTI_GPU_EPOCH


# --------------------------------------------------------------------------- #
# the anchor pool
# --------------------------------------------------------------------------- #
def usable_anchors(facts: dict | None = None) -> list:
    """Anchors admissible as rate evidence. Reads `vram_facts.json` LAZILY —
    nothing in this module touches the filesystem at import."""
    doc = facts if facts is not None else vram_facts.load_facts()
    out = []
    for a in (doc.get("anchors") or []):
        t = ((a.get("telemetry") or {}).get("throughput") or {})
        if not isinstance(t.get("n_steps_timed"), (int, float)):
            continue
        if t["n_steps_timed"] < MIN_TIMED_STEPS:
            continue
        if not (anchor_rate(a) and anchor_gpu(a) and anchor_family(a)):
            continue
        if not _past_multi_gpu_epoch(a):
            continue
        out.append(a)
    return out


# (gpu, world_size, family key) -> anchors, plus the epoch the tiers are read
# against. Memoized on the document's identity because `search` calls
# rate_for_offer once per candidate offer and rebuilding it was measured at
# 2.3 ms a lookup — ~1 s of pure re-filtering on a 500-offer board.
_INDEX_CACHE: dict = {}


def _index(facts: dict | None):
    doc = facts if facts is not None else vram_facts.load_facts()
    key = id(doc)
    hit = _INDEX_CACHE.get(key)
    if hit is not None and hit[0] is doc:
        return hit[1], hit[2]
    pool = usable_anchors(doc)
    cur = current_stack(pool)
    idx: dict = {}
    for a in pool:
        idx.setdefault((anchor_gpu(a), _world_size(a),
                        anchor_family(a).key()), []).append(a)
    _INDEX_CACHE.clear()               # one entry: the document in play
    _INDEX_CACHE[key] = (doc, idx, cur)
    return idx, cur


def current_stack(anchors) -> tuple | None:
    """The epoch every `measured`-tier anchor must match: the stack fingerprint
    of the NEWEST dated anchor. One rolling image means the newest run's stack is
    what a run submitted now would get; anything older is a floor.

    Tie-broken by the modal fingerprint ON that newest date, so a single stray
    anchor from a dev box cannot re-date the whole table by itself.
    """
    dated = [(anchor_date(a), a) for a in anchors if anchor_date(a)]
    if not dated:
        return None
    newest = max(d for d, _ in dated)
    fps = [stack_fingerprint(a) for d, a in dated if d == newest]
    fps = [f for f in fps if f]
    if not fps:
        return None
    return max(set(fps), key=fps.count)


def _tier(anchor: dict, cur: tuple | None) -> str:
    # Undated is provisional at best: recency is what ages a rate out, and an
    # anchor that cannot be placed in time cannot be shown to be current.
    if cur and anchor_date(anchor) and stack_fingerprint(anchor) == cur:
        return "measured"
    return "provisional"


# --------------------------------------------------------------------------- #
# the lookup
# --------------------------------------------------------------------------- #
def _estimate_op_point(anchors, cur, op, fitted_note) -> RateEstimate:
    """Median of the K most recent anchors at ONE operating point, at the best
    tier available. Tiers are never averaged together — a provisional anchor is a
    floor and a measured one is a reading, and their mean is neither."""
    for tier in TIERS:
        pool = [a for a in anchors if _tier(a, cur) == tier]
        if pool:
            break
    dated = [a for a in pool if anchor_date(a)]
    ranked = sorted(dated or pool,
                    key=lambda a: (anchor_date(a) or "", a.get("run") or ""),
                    reverse=True)[:K_RECENT]
    vals = [anchor_rate(a) for a in ranked]
    lo, hi = min(vals), max(vals)
    dates = sorted({anchor_date(a) for a in ranked if anchor_date(a)})
    span = (f"{dates[0]}..{dates[-1]}" if len(dates) > 1
            else (dates[0] if dates else "undated"))
    return RateEstimate(
        tok_s=round(statistics.median(vals), 1),
        tier=tier,
        n=len(vals),
        spread=round(hi / lo, 3) if lo > 0 else 0.0,
        runs=tuple(sorted({a.get("run", "") for a in ranked})),
        op_point=_op_point_str(op),
        why=(f"median of {len(vals)} most recent {tier} anchor(s) ({span}), "
             f"{len(anchors)} at this operating point{fitted_note}"),
    )


def rate_for_offer(family: Family, gpu_name: str, num_gpus: int = 1,
                   gpu_ram_gb: float | None = None,
                   *, facts: dict | None = None) -> RateEstimate | None:
    """tok/s for the WHOLE box, or None when this cell is unmeasured.

    Among anchors matching `(normalize_gpu_name(gpu_name), num_gpus, family)`,
    keeps the operating points whose measured per-card peak fits `gpu_ram_gb`,
    and returns the best of them ranked `(tier, tok_s)` — tier first, so a
    measured reading is never displaced by a faster provisional floor.

    `gpu_ram_gb` is the offer's PER-CARD VRAM. Passing None skips the fit filter
    and asks only "what has this card class done at this shape", which is the
    right question for a report and the wrong one for a purchase.

    None means UNMEASURED, which is an unknown box and not a bad one: rank it
    last, never as zero. Nothing here transfers a rate from another card.
    """
    if not isinstance(family, Family):
        return None
    if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus < 1:
        return None
    gpu = gpu_rates.normalize_gpu_name(gpu_name)
    if not gpu:
        return None

    idx, cur = _index(facts)
    fam_key = family.key()
    for key in (gpu,) + tuple(SKU_FALLBACK.get(gpu, ())):
        matched = idx.get((key, num_gpus, fam_key))
        if not matched:
            continue
        by_op = _by_op(matched)
        n_ops = len(by_op)
        if gpu_ram_gb is not None:
            budget = float(gpu_ram_gb)
            # EVERY anchor at the operating point must fit, not the median one:
            # peak is set by the longest row a corpus actually produced, the
            # within-group scatter is real (up to 6.3 GB), and the failure mode
            # of admitting the mean is an OOM on a box we just paid for. An
            # unreadable peak is unfittable, never silently admitted.
            by_op = {op: v for op, v in by_op.items()
                     if all((anchor_peak_per_card_gb(a) or float("inf")) <= budget
                            for a in v)}
        if not by_op:
            continue
        note = ("" if gpu_ram_gb is None
                else f"; {len(by_op)} of {n_ops} operating point(s) fit "
                     f"{float(gpu_ram_gb):.0f} GB/card")
        if key != gpu:
            note += f"; SKU floor: offer says {gpu!r}, priced as {key!r}"
        ests = [_estimate_op_point(v, cur, op, note) for op, v in by_op.items()]
        return max(ests, key=lambda e: (e.tier == "measured", e.tok_s))
    return None


def tokens_per_dollar(est: RateEstimate, dph: float) -> float | None:
    """Training tokens bought per USD at `dph` dollars/hour for the whole box.

    Steady-state, inheriting `gpu_rates.py`'s header: per-job fixed costs (boot,
    base pull, cold compile, upload) are excluded, so this over-states absolute
    tokens per dollar and is a RANKING quantity, not a budget.

    None — not 0.0 — when `est` is missing or `dph` is not positive. A zero would
    sort as "worst buy", which is a claim; None is the refusal the caller has to
    handle.
    """
    if not isinstance(est, RateEstimate):
        return None
    try:
        dph = float(dph)
    except (TypeError, ValueError):
        return None
    return est.tok_s * 3600.0 / dph if dph > 0 else None


def probe_hint(family: Family, gpu_name: str) -> str:
    """What to run to turn an unmeasured cell into an anchor.

    Every real run is a benchmark here — `harvest_vram.py` banks the shape and
    the throughput block off any `train_summary.json` — so the honest answer is
    usually "run the job on that card and re-harvest", not "rent a bench box".
    """
    gpu = gpu_rates.normalize_gpu_name(gpu_name) or "<card>"
    fam = family if isinstance(family, Family) else None
    bundle = ("gpu-rate-9b-w20480"
              if fam and fam.target_modules_class == "all-linear"
              else "fit-ladder")
    shape = (f"BASE_SLUG={fam.base_slug} MAX_SEQ={fam.max_seq} "
             f"QUANT={fam.quant_mode}" if fam else "the job's own env")
    return (f"no anchor for {gpu} at [{fam.slug() if fam else '?'}] — run "
            f"tools/witness/jobs/{bundle} on a {gpu} with {shape}, then "
            f"`python3 tools/vast/harvest_vram.py --write` to bank it")


# --------------------------------------------------------------------------- #
# census — what the anchor file actually supports, for docs and tests
# --------------------------------------------------------------------------- #
def rate_census(facts: dict | None = None) -> list:
    """One row per derivable (gpu, num_gpus, family) cell. Pure."""
    pool = usable_anchors(facts)
    cur = current_stack(pool)
    cells: dict = {}
    for a in pool:
        cells.setdefault(
            (anchor_gpu(a), _world_size(a), anchor_family(a)), []).append(a)
    rows = []
    for (gpu, ws, fam), v in cells.items():
        ops = _by_op(v)
        best = max((_estimate_op_point(g, cur, op, "") for op, g in ops.items()),
                   key=lambda e: (e.tier == "measured", e.tok_s))
        rows.append({"gpu": gpu, "num_gpus": ws, "family": fam,
                     "n_anchors": len(v), "n_op_points": len(ops),
                     "tier": best.tier, "tok_s": best.tok_s,
                     "op_point": best.op_point})
    rows.sort(key=lambda r: (-r["n_anchors"], r["gpu"]))
    return rows


def _by_op(anchors) -> dict:
    out: dict = {}
    for a in anchors:
        out.setdefault(_op_point(a.get("shape") or {}), []).append(a)
    return out


# --------------------------------------------------------------------------- #
# job env -> Family
# --------------------------------------------------------------------------- #
def _base_slug_from_assets(assets) -> str:
    """Mirrors jobmeta.base_slug_from_assets — most training bundles never set
    BASE_SLUG and name the model once, as the B2 prefix of their `base` asset."""
    for a in (assets or []):
        b2 = str((a or {}).get("b2") or "")
        if b2.startswith("base-models/"):
            return b2.split("/", 1)[1].strip("/").split("/")[0]
    return ""


def _window_ladder(env: dict) -> list:
    raw = str((env or {}).get("WINDOW_LADDER") or "").strip()
    out = []
    for part in raw.replace(" ", ",").split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return sorted(out)


def family_from_env(env: dict, *, world_size: int = 1,
                    assets=None) -> Family | None:
    """A `Family` from a jobs-v2 `env:` block (+ `assets:`), or None.

    NEVER RAISES. An env this cannot map is not an error — it is an eval, a
    generation sweep or a probe, and the caller degrades to price-only ranking
    exactly as it does today.

    `world_size` is the card count being ranked, because `eff_batch` is
    batch x grad_accum x world_size. Rank a 4-card offer with
    `family_from_env(env, world_size=4)` or the family names a job nobody ran.
    """
    try:
        env = {str(k).strip().upper(): v for k, v in dict(env or {}).items()}
        base = str(env.get("BASE_SLUG") or "").strip() \
            or _base_slug_from_assets(assets)
        max_seq = str(env.get("MAX_SEQ") or "").strip()
        if not max_seq:
            # A ladder bundle probes rungs largest-first on the box and falls
            # back, so the SMALLEST rung is the one it is guaranteed to run —
            # same choice jobmeta's VRAM gate makes, for the same reason.
            ladder = _window_ladder(env)
            max_seq = str(ladder[0]) if ladder else ""
        if not base or not max_seq:
            return None
        ws = int(world_size) if int(world_size) >= 1 else 1

        def _int(key, default):
            v = str(env.get(key, "")).strip()
            return int(v) if v.lstrip("-").isdigit() else default

        batch = _int("BATCH", _TRAINER_DEFAULTS["batch"])
        accum = _int("GRAD_ACCUM", _TRAINER_DEFAULTS["grad_accum"])
        eff = _int("EXPECT_EFF_BATCH", 0) or (batch * accum * ws)
        tm = str(env.get("TARGET_MODULES") or "").strip()
        targets = (tm if tm == "all-linear"
                   else ([x.strip() for x in tm.split(",") if x.strip()] if tm
                         else list(_TRAINER_DEFAULT_TARGETS)))
        return Family(
            base_slug=base,
            quant_mode=str(env.get("QUANT") or "bf16").strip() or "bf16",
            max_seq=int(max_seq),
            eff_batch=int(eff),
            packing=(str(env.get("PACKING") or "").strip()
                     or _TRAINER_DEFAULTS["packing"]),
            target_modules_class=vram_facts.target_modules_class(targets),
            lora_r=_int("LORA_R", _TRAINER_DEFAULTS["lora_r"]),
            ce_chunk_matmul=(str(env.get("CE_CHUNK_MATMUL") or "").strip()
                             or _TRAINER_DEFAULTS["ce_chunk_matmul"]),
        )
    except Exception:
        return None
