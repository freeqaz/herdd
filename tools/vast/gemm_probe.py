#!/usr/bin/env python3
"""gemm_probe.py — the boot-time, co-tenant-safe sibling of `gemm_ceiling.py`.

WHAT IT IS FOR
--------------
`tools/vast/mfu.py` divides achieved TFLOP/s by a measured dense-bf16 GEMM
ceiling. That ceiling has been measured on exactly two parts ever (an RTX 3090
and an RTX PRO 6000 **Max-Q**), so every roof-HFU we quote for anything else borrows a
number from the wrong SKU and `mfu.py` correctly stamps it PROVISIONAL.

Separately, `TRAINING_THROUGHPUT_REVIEW_2026-08-06.md` §7 and
`PERF_LEVERS_INVESTIGATION_2026-08-06.md` §1 rank a **host acceptance probe** as
the largest and cheapest lever found: measured host spread of **1.75×** (fit
probe) to **2.13×** (matched optimizer steps 21–28) in throughput *at the same
rental price*, on the same GPU model, with a byte-identical resolved config.
Nothing in the selection path measures compute — all three offer-query builders
order on price alone.

One 30-second measurement closes both. This module is that measurement, made
safe to run unattended at box boot.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
**It is observation only.** Nothing here rejects a box, re-rents, re-bids,
aborts a job, or influences offer selection. That half moves money and is held
for owner sign-off — and it *cannot be specified responsibly yet*, because we
have no distribution of host ceilings to set a threshold against. This probe is
what produces that distribution. See
`docs/plans/witness/perf/HOST_ACCEPTANCE_PROBE_2026-08-07.md` §5.

WHY IT IS A SEPARATE FILE FROM gemm_ceiling.py
----------------------------------------------
`gemm_ceiling.py` lives in the `perf-levers` JOB BUNDLE. Job bundles are
content-addressed: editing one is a different experiment, and it only ever runs
when someone rents a box to run a bench. This one rides the **jobd daemon
bundle** (`_job_attach_files()`), runs on every box we boot, and carries the
three things a boot-path program needs and a bench does not:

  1. **A hard wall-clock bound.** The GEMM runs in a CHILD process under
     `subprocess.run(timeout=...)`. A wedged CUDA call is not interruptible
     in-process — a signal handler will not unstick `cudaDeviceSynchronize` —
     so the only bounded option is a separate process the parent can SIGKILL.
     Timeout ⇒ record `skipped:timeout`, exit 0, boot continues.
  2. **A busy-GPU refusal.** It measures nothing if any visible card is doing
     work, or if jobd already has a job running, or if the card readings needed
     to *prove* idleness are unavailable. "Cannot prove idle" refuses, exactly
     as `metrics_probe.card_is_idle` refuses to prove idleness from a partial
     read. A probe that perturbs a co-tenant or an in-flight training step
     costs more than the number is worth.
  3. **A tiny VRAM footprint,** planned against free VRAM *before* allocating
     (`plan_shapes`), so it cannot OOM a card that something else is using.

It emits the **same JSON schema** as `gemm_ceiling.py` (device / capability /
sm_count / torch / cuda / shapes[] / ceiling_tflops), so
`mfu.py --ceiling-json <file>` consumes either without a code path of its own,
plus host-attribution fields (`power_limit_w`, `sm_clock_mhz`, `throttle`) that
turn "consistent with a lowered power cap" into a one-line read.

Verbatim from `gemm_ceiling.py`, and enforced here in code rather than by
convention: *a TFLOP/s figure with no device attached is not quotable.* Without
`torch.cuda.get_device_properties().name` the record's status is
`refused:no_device` and it carries no `ceiling_tflops` at all.

SHAPES — why a generic set at boot, and not each base's own
-----------------------------------------------------------
The ceiling is shape-dependent: `V7_PERF_LEVERS_2026-08-05` §2 measured
283.3 / 274.1 / 231.6 TFLOP/s across gemma-4's three own GEMM shapes, a 22%
spread, and the axis of that spread is the **aspect class** (square attention
projection vs wide-N gate/up vs wide-K down), not the absolute K.

At jobd boot **no base is known** — no job has been claimed, so there is no
model to take shapes from. So the default is one small fixed set covering the
three aspect classes `mfu.classify_gemm` already discriminates, and `mfu.py`
does the **FLOP-weighted harmonic mean over each model's own MAC mix**
(`mac_mix` + `harmonic_weighted`) — machinery that exists precisely so the
ceiling need not be measured at the model's shapes. The residual approximation
(this model's K within a class vs the probe's K) is RECORDED, not hidden:
records carry `shape_basis: "generic"`, `mfu.Ceiling` carries it through, and the
resulting roof-HFU says so in its note. ("roof-HFU", not MFU: see
`tools/vast/mfu.py`'s "WHAT THIS METRIC IS" block — relabelled 2026-08-09.)

When the base IS known — a bench cell, or any caller who ran
`mfu.py --model <base> --gemm-cmd` — pass those shapes with `--shape MxKxN` and
the record comes back `shape_basis: "model"`. One instrument, two shape regimes,
and the record always says which.

CLI
---
    # the boot lane (what jobd runs): guard, measure, write, never fail
    python3 gemm_probe.py --out /workspace/.gemm_probe.json --fields-out /tmp/f

    # would this box be probed right now, and if not why?
    python3 gemm_probe.py --check-only

    # a known base's own shapes (mfu.py --gemm-cmd prints them)
    python3 gemm_probe.py --shape 12288x3840x15360 --shape 12288x15360x3840 ...

Exit status is **0 on every outcome** unless `--strict`: this runs on the boot
path, and a probe that can fail a boot is worse than no probe.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

#: Bumped when the record schema or the default shape set changes, so a
#: cross-host comparison can refuse to mix two incompatible measurements.
PROBE_VERSION = 1

#: Default shapes: (M, K, N) for y[M,N] = x[M,K] @ w[K,N]. One per aspect class
#: (`mfu.classify_gemm`): square, wide-N, wide-K. M=8192 is large enough that the
#: tile count (>= 8192*4096/128^2 = 2048) buries a 188-SM part several waves deep,
#: and small enough that the largest allocation below is 448 MiB.
GENERIC_SHAPES = (
    (8192, 4096, 4096),      # attn_proj  — square
    (8192, 4096, 16384),     # mlp_up     — wide N (gate/up, and the lm_head's aspect)
    (8192, 16384, 4096),     # mlp_down   — wide K
)

#: Hard wall-clock ceiling on the child process, in seconds. Dominated by
#: `import torch` + CUDA context creation (5-15 s observed), not by the GEMMs
#: (~5 ms/iteration at these shapes). 90 s is ~0.4% of a 6 h rental in the worst
#: case and the boot proceeds regardless.
DEFAULT_DEADLINE_S = 90

#: Peak device memory the probe may allocate, in bytes. Shapes whose working set
#: exceeds it are SKIPPED (and recorded as skipped), never truncated.
DEFAULT_VRAM_BUDGET_B = 2 * 1024 ** 3

#: Busy thresholds. A card above either is doing someone's work and must not be
#: perturbed. Deliberately tight: an idle CUDA context on a fresh box holds well
#: under 512 MiB, and a *training* process holds tens of GB.
BUSY_UTIL_PCT = 10
BUSY_MEM_MB = 2048

_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}


# --------------------------------------------------------------------------- #
# guard — may this box be probed right now?
# --------------------------------------------------------------------------- #
def running_job_ids(state_dir):
    """jobd's live census: one `<jid>.running` per job it is running right now.

    Same file jobd hands entrypoints as `JOBD_STATE_DIR` for the sibling census
    (tools/witness/cpu_budget.py sizes its pool from it). Unreadable or absent
    directory ⇒ empty, because "no state dir" is the normal case on a box that is
    not running jobd at all, not evidence of a job.
    """
    if not state_dir:
        return []
    try:
        return sorted(os.path.basename(p)[: -len(".running")]
                      for p in glob.glob(os.path.join(state_dir, "*.running")))
    except OSError:
        return []


def busy_reason(gpus, *, running_jobs=(), util_pct=BUSY_UTIL_PCT,
                mem_mb=BUSY_MEM_MB):
    """None when it is safe to probe; otherwise a short reason slug.

    `gpus` is `metrics_probe.collect()["gpus"]` — a list of per-card dicts.

    UNREADABLE IS BUSY. If `util` or `mem_used_mb` is missing for a card we
    cannot prove that card is idle, and the probe refuses. This is the same rule
    `metrics_probe.card_is_idle` applies in the other direction ("an unreadable
    field means *cannot prove idle*"), and it is the conservative side: the cost
    of a false refusal is one missing datapoint, the cost of a false accept is a
    perturbed training step on a box someone is paying for.
    """
    if not gpus:
        return "no_gpu"
    if running_jobs:
        return "job_running:" + ",".join(sorted(running_jobs)[:3])
    for g in gpus:
        idx = g.get("idx")
        util, used = g.get("util"), g.get("mem_used_mb")
        if util is None or used is None:
            return f"unreadable_gpu:{idx}"
        if util > util_pct:
            return f"gpu_busy:gpu{idx}_util_{util}pct"
        if used > mem_mb:
            return f"gpu_busy:gpu{idx}_mem_{used}MB"
    return None


# --------------------------------------------------------------------------- #
# shape planning — never allocate more than the budget
# --------------------------------------------------------------------------- #
def shape_bytes(m, k, n, itemsize=2):
    """Working-set bytes for y[M,N] = x[M,K] @ w[K,N] (all three tensors live)."""
    return (m * k + k * n + m * n) * itemsize


def plan_shapes(shapes, *, budget_b=DEFAULT_VRAM_BUDGET_B, itemsize=2):
    """(kept, skipped) — skipped is [(shape, bytes)] for shapes over budget.

    Planned on the HOST before the child starts, so a shape that would not fit is
    never handed to torch at all. A probe that OOMs a card someone else is using
    is worse than a probe that reports one fewer aspect class.
    """
    kept, skipped = [], []
    for s in shapes:
        b = shape_bytes(*s, itemsize=itemsize)
        if b <= budget_b:
            kept.append(tuple(s))
        else:
            skipped.append((tuple(s), b))
    return kept, skipped


def parse_shape(text):
    """'8192x4096x16384' -> (8192, 4096, 16384). Raises ValueError otherwise."""
    parts = str(text).lower().split("x")
    if len(parts) != 3:
        raise ValueError(f"shape must be MxKxN, got {text!r}")
    vals = tuple(int(p) for p in parts)
    if any(v <= 0 for v in vals):
        raise ValueError(f"shape dimensions must be positive, got {text!r}")
    return vals


# --------------------------------------------------------------------------- #
# the measurement (child process)
# --------------------------------------------------------------------------- #
def _bench_in_process(shapes, *, dtype="bf16", warmup=3, iters=15):
    """Run the GEMMs. ONLY ever called inside the child — imports torch.

    Timing is CUDA-event based (the host clock does not see kernel time), and the
    FLOP count is the same `2*M*K*N` `gemm_ceiling.py` uses, so the two
    instruments' numbers are directly comparable.
    """
    import torch                                        # noqa: PLC0415
    if not torch.cuda.is_available():
        return None, "no_cuda_device"
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16,
          "fp32": torch.float32}[dtype]
    torch.backends.cuda.matmul.allow_tf32 = True         # what training gets
    rows = []
    for (m, k, n) in shapes:
        a = torch.randn(m, k, device="cuda", dtype=dt)
        b = torch.randn(k, n, device="cuda", dtype=dt)
        for _ in range(warmup):
            a @ b
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            a @ b
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
        del a, b
        torch.cuda.empty_cache()
        rows.append({"m": m, "k": k, "n": n, "ms": round(ms, 4),
                     "tflops": round(2.0 * m * k * n / (ms * 1e-3) / 1e12, 1)})
    props = torch.cuda.get_device_properties(0)
    return {
        "device": (props.name or "").strip(),
        "capability": f"sm_{props.major}{props.minor}",
        "sm_count": props.multi_processor_count,
        "vram_total_mb": int(props.total_memory // (1024 * 1024)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": dtype,
        "warmup": warmup,
        "iters": iters,
        "shapes": rows,
    }, None


def run_bench(shapes, *, dtype="bf16", warmup=3, iters=15,
              deadline_s=DEFAULT_DEADLINE_S, python=None):
    """Run `_bench_in_process` in a CHILD, bounded by `deadline_s`.

    Returns (result_dict, error_slug). The child is the whole point: a hung CUDA
    call cannot be interrupted from inside the process that made it, so the wall
    -clock bound has to be a process the parent can kill. `subprocess.run`'s
    timeout SIGKILLs the child, so a wedged driver costs `deadline_s` and nothing
    more — the boot continues either way.
    """
    argv = [python or sys.executable, os.path.abspath(__file__), "_bench",
            "--dtype", dtype, "--warmup", str(warmup), "--iters", str(iters)]
    for (m, k, n) in shapes:
        argv += ["--shape", f"{m}x{k}x{n}"]
    try:
        r = subprocess.run(argv, capture_output=True, text=True,
                           timeout=deadline_s)
    except subprocess.TimeoutExpired:
        return None, f"timeout_{deadline_s}s"
    except OSError as e:
        return None, f"spawn_failed:{type(e).__name__}"
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()
        return None, "bench_rc_%d:%s" % (r.returncode, tail[-1][:120] if tail else "")
    try:
        blob = json.loads(r.stdout)
    except ValueError:
        return None, "bench_unparseable_output"
    if blob.get("error"):
        return None, str(blob["error"])
    return blob, None


# --------------------------------------------------------------------------- #
# host attribution
# --------------------------------------------------------------------------- #
def _import_metrics_probe():
    """metrics_probe ships FLAT beside this file in the jobd bundle and nested in
    the repo. Absent ⇒ None, and the caller refuses to probe (it cannot prove the
    GPU is idle without it) rather than guessing."""
    for p in (_HERE, os.path.dirname(_HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import metrics_probe                             # noqa: PLC0415
        return metrics_probe
    except ImportError:
        return None


def host_attribution(gpus):
    """The fields that make a TFLOP/s number attributable after the box is gone.

    `PERF_LEVERS_INVESTIGATION` §2.5: box 46936034 ran 2.13× slow for a whole
    training window and the cause can never now be proven, because the heartbeat
    persisted `gpu_pwr` as a percentage of an unrecorded limit. A 300 W-capped
    card and a 600 W card both read "100%". So the absolute cap, the clock and
    the throttle bits ride with every ceiling this probe emits.
    """
    lims = [g["power_limit_w"] for g in gpus or () if g.get("power_limit_w")]
    clks = [g["sm_clock_mhz"] for g in gpus or () if g.get("sm_clock_mhz")]
    temps = [g["temp_c"] for g in gpus or () if g.get("temp_c") is not None]
    thr = sorted({t for g in gpus or () for t in (g.get("throttle") or [])
                  if t != "none"})
    out = {"gpu_count": len(gpus or ())}
    if lims:
        out["power_limit_w"] = round(min(lims))          # the slowest card paces
    if clks:
        out["sm_clock_mhz"] = min(clks)
    if temps:
        out["temp_c"] = max(temps)
    if thr:
        out["throttle"] = thr
    return out


def _env_ids():
    """Identity the box actually knows. `machine_id` is NOT among it on any lane
    today (vast does not inject it), so it is recorded ONLY when something put it
    in the environment and is otherwise absent — never invented. Resolving
    instance -> machine is `hostfacts.py ingest`'s job, laptop-side, where the
    mapping is authoritative. "Inherit, never invent."
    """
    out = {}
    iid = (os.environ.get("INSTANCE_ID") or os.environ.get("CONTAINER_ID") or "")
    if iid.strip():
        out["instance_id"] = iid.strip()
    for k in ("MACHINE_ID", "VAST_MACHINE_ID"):
        v = (os.environ.get(k) or "").strip()
        if v:
            out["machine_id"] = v
            break
    return out


# --------------------------------------------------------------------------- #
# the record
# --------------------------------------------------------------------------- #
def build_record(bench, *, attribution=None, shape_basis="generic",
                 skipped_shapes=(), elapsed_s=None, status=None, reason=""):
    """Assemble the on-disk / on-B2 record.

    Schema is a SUPERSET of `gemm_ceiling.py --json`, so
    `mfu.Ceiling.from_gemm_ceiling_json` (and therefore
    `mfu.py --ceiling-json`) reads it with no format branch.

    A bench with no device name yields `refused:no_device` and NO
    `ceiling_tflops`: *a TFLOP/s figure with no device attached is not quotable*
    (`gemm_ceiling.py`), and the cheapest way to keep that true is to make the
    unquotable number not exist.
    """
    rec = {
        "probe_version": PROBE_VERSION,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shape_basis": shape_basis,
        "status": status or ("ok" if bench else "failed"),
    }
    rec.update(_env_ids())
    if reason:
        rec["reason"] = reason
    if elapsed_s is not None:
        rec["elapsed_s"] = round(float(elapsed_s), 2)
    if skipped_shapes:
        rec["skipped_shapes"] = [{"m": s[0], "k": s[1], "n": s[2],
                                  "bytes": int(b)} for s, b in skipped_shapes]
    rec.update(attribution or {})
    if not bench:
        return rec
    rec.update(bench)
    rows = bench.get("shapes") or []
    if not (rec.get("device") or "").strip():
        rec["status"] = "refused:no_device"
        rec["reason"] = ("torch.cuda.get_device_properties().name was empty — a "
                         "TFLOP/s figure with no device attached is not quotable")
        return rec
    if rows:
        rec["ceiling_tflops"] = max(r["tflops"] for r in rows)
        rec["min_tflops"] = min(r["tflops"] for r in rows)
    return rec


# --------------------------------------------------------------------------- #
# heartbeat / box-event field rendering
# --------------------------------------------------------------------------- #
def _tag(s):
    """Field-safe token: jobd's K=V parser splits on '=', bash splits on
    whitespace, and the k:v heartbeat packer splits on ',' and ':'. Same
    reduction `metrics_probe._gpu_name_tag` applies, for the same reason: a
    device name here is an attribution key, not prose."""
    out = []
    for ch in str(s):
        out.append(ch if (ch.isalnum() or ch in "_.+-") else "_")
    return "".join(out).strip("_") or "unknown"


def render_fields(rec):
    """`k=v` lines for `jobd.sh emit_box gemm_probe ...` — one per line, no
    whitespace in any value."""
    pairs = [("status", _tag(rec.get("status", "unknown")))]
    for key, out in (("device", "gpu"), ("capability", "cap"),
                     ("shape_basis", "shape_basis")):
        if rec.get(key):
            pairs.append((out, _tag(rec[key])))
    for key in ("ceiling_tflops", "min_tflops", "power_limit_w", "sm_clock_mhz",
                "temp_c", "sm_count", "gpu_count", "elapsed_s", "probe_version"):
        if rec.get(key) is not None:
            pairs.append((key, rec[key]))
    if rec.get("throttle"):
        pairs.append(("throttle", "|".join(_tag(t) for t in rec["throttle"])))
    for r in rec.get("shapes") or []:
        pairs.append((f"tflops_{r['m']}x{r['k']}x{r['n']}", r["tflops"]))
    if rec.get("reason"):
        pairs.append(("reason", _tag(rec["reason"])[:120]))
    return "\n".join(f"{k}={v}" for k, v in pairs)


# --------------------------------------------------------------------------- #
# the whole probe
# --------------------------------------------------------------------------- #
def probe(*, shapes=None, dtype="bf16", warmup=3, iters=15,
          deadline_s=DEFAULT_DEADLINE_S, budget_b=DEFAULT_VRAM_BUDGET_B,
          state_dir=None, python=None, check_only=False, metrics=None):
    """Guard, plan, measure, record. NEVER raises — the caller is a boot path.

    `metrics` is an injected `metrics_probe.collect()`-shaped dict (tests, and
    any caller who already has a snapshot). Absent, it is collected here; if
    `metrics_probe` cannot be imported at all the probe REFUSES, because
    without it there is no way to establish that the GPU is idle.
    """
    t0 = time.monotonic()
    shape_basis = "generic" if shapes is None else "model"
    shapes = list(shapes if shapes is not None else GENERIC_SHAPES)

    if metrics is None:
        mp = _import_metrics_probe()
        if mp is None:
            return build_record(None, shape_basis=shape_basis,
                                status="skipped:no_metrics_probe",
                                reason="metrics_probe.py not importable — cannot "
                                       "establish that the GPU is idle")
        try:
            metrics = mp.collect(window=0.2)
        except Exception as e:                            # noqa: BLE001 boot path
            return build_record(None, shape_basis=shape_basis,
                                status="skipped:metrics_failed",
                                reason=f"{type(e).__name__}: {e}"[:200])

    gpus = metrics.get("gpus") or []
    attribution = host_attribution(gpus)
    jobs = running_job_ids(state_dir or os.environ.get("JOBD_STATE_DIR"))
    reason = busy_reason(gpus, running_jobs=jobs)
    if reason:
        return build_record(None, attribution=attribution,
                            shape_basis=shape_basis,
                            status=f"skipped:{reason.split(':')[0]}",
                            reason=reason,
                            elapsed_s=time.monotonic() - t0)
    if check_only:
        return build_record(None, attribution=attribution,
                            shape_basis=shape_basis, status="would_run",
                            elapsed_s=time.monotonic() - t0)

    kept, skipped = plan_shapes(shapes, budget_b=budget_b,
                                itemsize=_DTYPE_BYTES.get(dtype, 2))
    if not kept:
        return build_record(None, attribution=attribution,
                            shape_basis=shape_basis,
                            status="skipped:no_shape_fits_budget",
                            reason=f"every shape exceeds {budget_b} B",
                            skipped_shapes=skipped,
                            elapsed_s=time.monotonic() - t0)

    bench, err = run_bench(kept, dtype=dtype, warmup=warmup, iters=iters,
                           deadline_s=deadline_s, python=python)
    if err:
        status = ("skipped:timeout" if err.startswith("timeout")
                  else f"failed:{err.split(':')[0]}")
        return build_record(None, attribution=attribution,
                            shape_basis=shape_basis, status=status, reason=err,
                            skipped_shapes=skipped,
                            elapsed_s=time.monotonic() - t0)
    return build_record(bench, attribution=attribution, shape_basis=shape_basis,
                        skipped_shapes=skipped,
                        elapsed_s=time.monotonic() - t0)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cmd_bench(argv):
    """Hidden child entrypoint. Prints one JSON object; never a traceback, so the
    parent's failure slug is always a parsed field rather than a stderr guess."""
    ap = argparse.ArgumentParser(prog="gemm_probe _bench")
    ap.add_argument("--shape", action="append", default=[])
    ap.add_argument("--dtype", default="bf16", choices=sorted(_DTYPE_BYTES))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=15)
    a = ap.parse_args(argv)
    try:
        shapes = [parse_shape(s) for s in a.shape] or list(GENERIC_SHAPES)
        blob, err = _bench_in_process(shapes, dtype=a.dtype, warmup=a.warmup,
                                      iters=a.iters)
        print(json.dumps(blob if blob else {"error": err}))
    except Exception as e:                                # noqa: BLE001
        print(json.dumps({"error": f"{type(e).__name__}: {e}"[:200]}))
        return 1
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_bench":
        return _cmd_bench(argv[1:])

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", action="append", default=[],
                    help="MxKxN, repeatable. Given ⇒ shape_basis 'model'; "
                         "omitted ⇒ the generic three-aspect-class set. "
                         "`mfu.py --model <base> --gemm-cmd` prints a base's own.")
    ap.add_argument("--dtype", default="bf16", choices=sorted(_DTYPE_BYTES))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--deadline-s", type=float, default=DEFAULT_DEADLINE_S,
                    help=f"hard wall-clock bound on the child "
                         f"(default {DEFAULT_DEADLINE_S})")
    ap.add_argument("--vram-budget-mb", type=float,
                    default=DEFAULT_VRAM_BUDGET_B / 1024 ** 2)
    ap.add_argument("--state-dir", default=None,
                    help="jobd state dir; a *.running file there means a job is "
                         "live and the probe refuses (default $JOBD_STATE_DIR)")
    ap.add_argument("--python", default=None,
                    help="interpreter for the child (the one with torch — on a "
                         "box that is the venv named by "
                         "/workspace/.train_env_activate)")
    ap.add_argument("--check-only", action="store_true",
                    help="run the guard and report, measure nothing")
    ap.add_argument("--out", default="", help="write the record here")
    ap.add_argument("--fields-out", default="",
                    help="write `k=v` lines here for jobd's emit_box")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="exit nonzero when the probe did not produce a ceiling "
                         "(NEVER set this on the boot path)")
    a = ap.parse_args(argv)

    try:
        shapes = [parse_shape(s) for s in a.shape] or None
    except ValueError as e:
        print(f"!! {e}", file=sys.stderr)
        return 2

    rec = probe(shapes=shapes, dtype=a.dtype, warmup=a.warmup, iters=a.iters,
                deadline_s=a.deadline_s,
                budget_b=int(a.vram_budget_mb * 1024 ** 2),
                state_dir=a.state_dir, python=a.python,
                check_only=a.check_only)

    if not a.quiet:
        print(json.dumps(rec, indent=2))
    for path, text in ((a.out, json.dumps(rec, indent=2)),
                       (a.fields_out, render_fields(rec))):
        if not path:
            continue
        try:
            with open(path, "w") as fh:
                fh.write(text + "\n")
        except OSError as e:
            print(f"!! could not write {path}: {e}", file=sys.stderr)
    if a.strict and rec.get("ceiling_tflops") is None:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
