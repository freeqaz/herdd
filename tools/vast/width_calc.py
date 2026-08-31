#!/usr/bin/env python3
"""Size vLLM's ``--max-num-seqs`` from MEASURED boot lines, not from memory.

The doctrine this encodes lives in
``docs/plans/throughput/WIDTH_SIZING_LEVER_2026-08-22.md``. Two facts make the
arithmetic worth a tool instead of a paragraph:

1. vLLM prints everything needed to calibrate a model's per-sequence memory
   cost. ``Available KV cache memory: M GiB`` plus ``Maximum concurrency for L
   tokens per request: Cx`` gives one equation ``M / (a*L + b) = C``. Two boots
   at different ``--max-model-len`` solve for ``a`` (KV bytes per token) and
   ``b`` (per-sequence recurrent/state overhead). Both are properties of the
   MODEL, so a calibration taken on one card transfers to every other card.
2. That bound is the k=1, full-length, no-prefix-sharing worst case. A k>1
   fan-out under prefix caching occupies far less, so the width a real roster
   supports must be RESCALED from observed steady-state KV occupancy. Feeding
   the conservative bound to a k=20 lane leaves most of the pool unused.

Stdlib only, no vLLM import: it parses log text and does arithmetic.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass

GIB = 1024**3
KIB = 1024

# vLLM derives max_cudagraph_capture_size = min(max_num_seqs*2, 512) and steps
# the ladder by 8 above 32 (vllm/config/vllm.py:1795). Landing the width ON a
# step keeps the largest FULL graph equal to the width instead of one step under.
CAPTURE_STEP = 8

# Headroom against the KV pool. vLLM preempts (recompute) when it cannot grow a
# running sequence, so a width sized at 100% of the pool thrashes the moment the
# length distribution's tail arrives. 0.80 is the shipped default here; the
# 2026-08-21 cont8k leg ran a steady 94.1% with zero preemptions, so 0.80 is
# conservative rather than a measured cliff.
DEFAULT_TARGET_UTIL = 0.80


@dataclass(frozen=True)
class BootPoint:
    """One vLLM boot: available KV bytes, the window, and the printed concurrency."""

    kv_avail_bytes: float
    max_model_len: int
    max_concurrency: float

    @property
    def per_seq_bytes(self) -> float:
        return self.kv_avail_bytes / self.max_concurrency


@dataclass(frozen=True)
class Calibration:
    """Per-model memory cost: ``per_seq(L) = kv_bytes_per_token*L + state_bytes``."""

    kv_bytes_per_token: float
    state_bytes_per_seq: float

    def per_seq_bytes(self, seq_len: int) -> float:
        return self.kv_bytes_per_token * seq_len + self.state_bytes_per_seq

    def concurrency(self, kv_avail_bytes: float, seq_len: int) -> float:
        return kv_avail_bytes / self.per_seq_bytes(seq_len)


_RE_KV_AVAIL = re.compile(r"Available KV cache memory:\s*([0-9.]+)\s*GiB")
_RE_CONC = re.compile(
    r"Maximum concurrency for ([0-9,]+) tokens per request:\s*([0-9.]+)x"
)


def parse_boot_log(text: str) -> BootPoint:
    """Pull the one boot point out of a vLLM engine log (or any text holding it)."""
    kv = _RE_KV_AVAIL.search(text)
    conc = _RE_CONC.search(text)
    if not kv or not conc:
        missing = []
        if not kv:
            missing.append("'Available KV cache memory'")
        if not conc:
            missing.append("'Maximum concurrency for ... tokens per request'")
        raise ValueError("log has no " + " and no ".join(missing))
    return BootPoint(
        kv_avail_bytes=float(kv.group(1)) * GIB,
        max_model_len=int(conc.group(1).replace(",", "")),
        max_concurrency=float(conc.group(2)),
    )


def calibrate(points: list[BootPoint]) -> Calibration:
    """Solve a*L + b = per_seq over >=2 boots at DIFFERENT max_model_len.

    With exactly two points this is the line through them; with more it is the
    least-squares fit, which is what you want once rounding noise is in play.
    """
    if len(points) < 2:
        raise ValueError("calibration needs >= 2 boots at different --max-model-len")
    lens = {p.max_model_len for p in points}
    if len(lens) < 2:
        raise ValueError(
            "all boots share max_model_len=%d; vary the window, not the width"
            % points[0].max_model_len
        )
    n = len(points)
    sx = sum(p.max_model_len for p in points)
    sy = sum(p.per_seq_bytes for p in points)
    sxx = sum(p.max_model_len**2 for p in points)
    sxy = sum(p.max_model_len * p.per_seq_bytes for p in points)
    denom = n * sxx - sx * sx
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    return Calibration(kv_bytes_per_token=a, state_bytes_per_seq=b)


def block_ceiling(kv_avail_bytes: float, attn_block_size: int,
                  cal: Calibration) -> int:
    """The HARD boot-time width ceiling on a hybrid (Qwen3.5/3.6) model.

    vLLM: "max_num_seqs (64) exceeds available Mamba cache blocks (63). Each
    decode sequence requires one Mamba cache block, so CUDA graph capture cannot
    proceed." It is a ValueError at boot, not a degradation — which is why a
    hybrid CANNOT be given a deliberately generous knob the way a dense model
    can. The engine prints `attn_block_size` at boot ("Setting attention block
    size to N tokens to ensure attention page size >= mamba page size").

    Measured against the engine's own refusals on a Qwen3.5-9B / RTX 3090
    (2026-08-23): predicts 62 where vLLM said 63, and 56 where it said 58 — i.e.
    it under-reads by 1-2 blocks, which is the safe direction. Pinned in
    `test_width_calc.py`.
    """
    page = attn_block_size * cal.kv_bytes_per_token
    return int(kv_avail_bytes // page)


def round_to_capture(width: float) -> int:
    """Round to the NEAREST CUDA-graph capture step, never below 1.

    Nearest, not floor: `target_util` already carries the headroom, and flooring
    a bound of 15.95 to 8 would halve the width to buy safety that is already
    bought. Below one step the ladder is per-integer, so keep the exact value.
    """
    if width < CAPTURE_STEP:
        return max(1, int(width))
    return int(width / CAPTURE_STEP + 0.5) * CAPTURE_STEP


def width_from_length(
    kv_avail_bytes: float,
    seq_len: int,
    cal: Calibration,
    target_util: float = DEFAULT_TARGET_UTIL,
    attn_block_size: int | None = None,
) -> int:
    """Conservative width: every sequence holds ``seq_len`` tokens, nothing shared.

    This is the right bound for a k=1 lane. On a k>1 fan-out with prefix caching
    it under-reads the affordable width by roughly the fan-out factor — use
    :func:`width_from_observed` there.

    Pass ``attn_block_size`` on a HYBRID model to also apply :func:`block_ceiling`.
    Skipping it on a hybrid is how you author a width the engine refuses to boot.
    """
    w = kv_avail_bytes * target_util / cal.per_seq_bytes(seq_len)
    if attn_block_size is not None:
        w = min(w, block_ceiling(kv_avail_bytes, attn_block_size, cal))
    return round_to_capture(w)


def width_from_observed(
    width_now: int,
    kv_util_now: float,
    target_util: float = DEFAULT_TARGET_UTIL,
) -> int:
    """Rescale a width from the steady-state KV occupancy it actually produced.

    ``kv_util_now`` is the peak ``GPU KV cache usage: P%`` seen while the queue
    was non-empty, as a fraction. Requires that the knob was BINDING (running ==
    width with requests waiting); if the queue was empty the occupancy describes
    the client, not the card.
    """
    if not 0 < kv_util_now <= 1:
        raise ValueError("kv_util_now is a fraction in (0, 1]")
    return round_to_capture(width_now * target_util / kv_util_now)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _fmt_cal(cal: Calibration) -> str:
    return (
        f"kv_per_token = {cal.kv_bytes_per_token / KIB:.2f} KiB    "
        f"state_per_seq = {cal.state_bytes_per_seq / GIB:.3f} GiB"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="fit a model's (kv/token, state/seq) from boot logs")
    c.add_argument("logs", nargs="+", help="vLLM engine logs, >=2 at different --max-model-len")

    p = sub.add_parser("plan", help="conservative width for a length (k=1, nothing shared)")
    p.add_argument("--kv-avail-gib", type=float, required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--kv-per-token-kib", type=float, required=True)
    p.add_argument("--state-per-seq-gib", type=float, required=True)
    p.add_argument("--target-util", type=float, default=DEFAULT_TARGET_UTIL)
    p.add_argument("--attn-block-size", type=int, default=None,
                   help="HYBRID models: apply the hard mamba-block boot ceiling. "
                        "vLLM prints this at boot ('Setting attention block size "
                        "to N tokens'). Omitting it on a hybrid can author a "
                        "width the engine REFUSES to start at.")

    r = sub.add_parser("rescale", help="width from the occupancy a prior run produced")
    r.add_argument("--width-now", type=int, required=True)
    r.add_argument("--kv-util-now", type=float, required=True, help="fraction, e.g. 0.402")
    r.add_argument("--target-util", type=float, default=DEFAULT_TARGET_UTIL)

    ns = ap.parse_args(argv)

    if ns.cmd == "calibrate":
        pts = []
        for path in ns.logs:
            with open(path, encoding="utf-8", errors="replace") as fh:
                pts.append(parse_boot_log(fh.read()))
        cal = calibrate(pts)
        for pt in pts:
            print(
                f"  boot: L={pt.max_model_len:>7,}  KV={pt.kv_avail_bytes / GIB:6.2f} GiB"
                f"  conc={pt.max_concurrency:6.2f}x  ->  {pt.per_seq_bytes / GIB:.4f} GiB/seq"
            )
        print(_fmt_cal(cal))
        return 0

    if ns.cmd == "plan":
        cal = Calibration(ns.kv_per_token_kib * KIB, ns.state_per_seq_gib * GIB)
        kv = ns.kv_avail_gib * GIB
        w = width_from_length(kv, ns.seq_len, cal, ns.target_util, ns.attn_block_size)
        print(f"per_seq({ns.seq_len:,}) = {cal.per_seq_bytes(ns.seq_len) / GIB:.4f} GiB")
        if ns.attn_block_size:
            bc = block_ceiling(kv, ns.attn_block_size, cal)
            mem = round_to_capture(kv * ns.target_util / cal.per_seq_bytes(ns.seq_len))
            print(f"memory bound {mem}   mamba-block ceiling {bc}   "
                  f"BINDING: {'blocks' if bc < mem else 'memory'}")
        else:
            print("NOTE: no --attn-block-size, so the hybrid boot ceiling is NOT "
                  "applied. On Qwen3.5/3.6 that ceiling is a hard ValueError.")
        print(f"--max-num-seqs {w}   (conservative, k=1; rescale for a k>1 APC lane)")
        return 0

    w = width_from_observed(ns.width_now, ns.kv_util_now, ns.target_util)
    print(f"--max-num-seqs {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
