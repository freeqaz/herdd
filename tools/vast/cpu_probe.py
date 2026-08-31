#!/usr/bin/env python3
"""cpu_probe.py — what a box's CORES are worth, measured at boot.

WHAT IT IS FOR
--------------
`vastlib/market/offers.py:cpu_score` ranks CPU offers on `cores x GHz`. That
prior is blind to two things it cannot see by construction:

  1. **IPC.** A Broadwell Xeon and a Zen 3 EPYC at equal clock are not equal
     cores, so the score over-rates old silicon.
  2. **Whether the box scales at all.** A 256-core slice on a contended host,
     a thermal-limited part, or a noisy co-tenant all advertise 256 cores and
     deliver a fraction of them. Nothing in the selection path notices.

This measures both: a single-worker rate (the IPC proxy) and an all-core rate
at slice width, whose ratio is the scaling factor. Boot-time and box-scoped, so
every box we rent produces one without anyone asking it to.

WHY A BENCHMARK, WHEN `hostfacts.cpu_record` SAYS "HARVESTED, NOT BENCHMARKED"
------------------------------------------------------------------------------
That doctrine is right about fidelity and was wrong about coverage. Harvested
records are bundle-scoped: measured 2026-08-24, `drop_cpu_record` had two
callers, neither had ever run, and the store held **0 cpu records against 202
gemm** — the gemm figure being what a box-scoped probe collects over the same
period. Harvest-only produced no distribution at all.

There is also a narrower reason a fixed workload is the *right* instrument for
this particular question. `per_core_s` is defended as the cross-machine
comparand, and that only holds if the machines ran the same work. Real compiles
differ per job, so harvested rates compare across machines only by accident.
For IPC specifically, fixed work is not a compromise — it is the measurement.

So: both, and they never merge. The probe emits `units="pyops"` on every box;
harvested producers keep emitting their own units; `summarize_cpu` groups by
unit and refuses to average across them.

CO-TENANT SAFETY — this is not the CPU farm ruling 0a9f1926 relitigated
-----------------------------------------------------------------------
That ruling killed a sidecar that stole cores from a running job. This is the
opposite shape and carries the three properties that make `gemm_probe`
acceptable on a GPU box: it runs at boot **before the first claim**, it refuses
if anything is already running, and it is bounded by a hard deadline.

MEASURE THE SLICE, NOT THE HOST
-------------------------------
`nproc` and `/proc/stat` are not virtualised in a container and describe the
whole physical host. This has already produced two wrong instruments in this
lane: box 48293057 read 4.3 busy cores host-wide while its own cgroup had used
0.34, and a fit denominator taken from `nproc` predicted 0.39 where the truth
was 3.125. Width and busy-ness both come from the cgroup here, and the record
carries `width_source` so a consumer can tell a measured slice from a guess.

Stdlib only, and no torch: the boxes this matters most for have no GPU and no
train env.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

PROBE_VERSION = 1

#: Refuse if our own cgroup is already burning more than this many cores.
#: PROVISIONAL — set before any distribution exists, deliberately loose so boot
#: noise (an rclone pull, jobd itself) does not suppress the measurement. The
#: pre-probe level is recorded on every record precisely so this can be set from
#: data later rather than from taste.
BUSY_CORES = 1.0

#: Process-per-core costs a whole interpreter each, so full width on a 256-core
#: slice would be ~4 GB of RSS to learn something 64 workers already show. The
#: cap is recorded as `workers` and `scaling` is computed against it, never
#: against `cores` — a capped run must not read as a scaling failure.
MAX_WORKERS = 64

DEFAULT_DEADLINE_S = 90.0

#: Sized so a modern core takes ~1s. Measured 2026-08-25 on a 7950X at 10.2M
#: iterations/s, so 8M lands just under a second there and a few seconds on the
#: slowest thing we would rent. The window has to dominate two constants it
#: would otherwise measure instead: the ~10ms of fork setup per pool round, and
#: scheduler noise. An earlier 400k landed at 39ms and measured mostly those.
KERNEL_ITERS = 8_000_000
DEFAULT_ROUNDS = 3

COMPILE_REPS = 12
_COMPILERS = ("cc", "gcc", "clang")

#: How many copies of the body below make one TU. Sized by measurement, not
#: taste. At ONE copy, `cc` process startup was **41%** of the bench, so the
#: number was substantially a fork+exec benchmark and would flatter a machine
#: with cheap process creation over one with a fast compiler. Measured on a
#: 7950X, startup as a share of 12 compiles:
#:
#:     repeats   12x wall   startup
#:          12      0.46s       21%
#:          32      0.85s       11%
#:          64      1.47s        6%
#:
#: 64 buys 6% for about a second and a half of a 240s boot budget.
_C_REPEATS = 64

_C_PROLOGUE = """
#include <stddef.h>
typedef struct { double x, y, z; } v3;
"""

# A fixed TU: real parsing, inlining and optimisation, no headers beyond
# stddef, so it compiles identically everywhere a C compiler exists.
_C_BODY = """
static v3 add%(n)d(v3 a, v3 b){ v3 r; r.x=a.x+b.x; r.y=a.y+b.y; r.z=a.z+b.z; return r; }
static double dot%(n)d(v3 a, v3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
static unsigned long lcg%(n)d(unsigned long s){ return s*6364136223846793005UL + 1442695040888963407UL; }
double probe_kernel%(n)d(const double *in, size_t n, double *out){
    v3 acc = {0.0, 0.0, 0.0};
    unsigned long s = 12345UL;
    for (size_t i = 0; i + 3 < n; i += 3) {
        v3 p = { in[i], in[i+1], in[i+2] };
        s = lcg%(n)d(s);
        if (s & 1UL) { p.x = -p.x; }
        if (s & 2UL) { p.y = p.y * 0.5; }
        acc = add%(n)d(acc, p);
        out[i/3] = dot%(n)d(acc, p);
    }
    return dot%(n)d(acc, acc);
}
"""


def c_source(repeats=_C_REPEATS):
    """The TU, `repeats` distinct copies of the body. Distinct NAMES on purpose:
    identical ones would let the compiler do the work once."""
    return _C_PROLOGUE + "".join(_C_BODY % {"n": i} for i in range(repeats))


# --------------------------------------------------------------------------- #
# the slice: how wide are we, really
# --------------------------------------------------------------------------- #
def _read(path):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _cpuset_count(text):
    """`0-3,8` -> 5. cpuset ranges are inclusive on both ends."""
    n = 0
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            try:
                n += int(hi) - int(lo) + 1
            except ValueError:
                return 0
        else:
            n += 1
    return n


def cpu_width():
    """(cores, source) — the width of THIS container's slice.

    Quota first: a `cpu.max` of `400000 100000` is 4 cores no matter how many
    the host has, and it is the number that bounds us. cpuset next, then the
    affinity mask. `os.cpu_count()` is last and is the only source here that can
    describe the host rather than us — hence `width_source` on the record.
    """
    v2 = _read("/sys/fs/cgroup/cpu.max").split()
    if len(v2) == 2 and v2[0] != "max":
        try:
            q, p = float(v2[0]), float(v2[1])
            if q > 0 and p > 0:
                return q / p, "cgroup_v2_quota"
        except ValueError:
            pass

    q1 = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    p1 = _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    try:
        if q1 and p1 and float(q1) > 0 and float(p1) > 0:
            return float(q1) / float(p1), "cgroup_v1_quota"
    except ValueError:
        pass

    for path, src in (("/sys/fs/cgroup/cpuset.cpus.effective", "cpuset_v2"),
                      ("/sys/fs/cgroup/cpuset/cpuset.effective_cpus",
                       "cpuset_v1")):
        n = _cpuset_count(_read(path))
        if n:
            return float(n), src

    try:
        return float(len(os.sched_getaffinity(0))), "sched_affinity"
    except (AttributeError, OSError):
        pass
    return float(os.cpu_count() or 1), "os_cpu_count"


def cgroup_cpu_usage_s():
    """Cumulative CPU-seconds burned by THIS cgroup, or None.

    `/proc/stat` is deliberately not consulted: a container does not virtualise
    it, so it reports the host and reads busy on a box whose own slice is idle.
    """
    for line in _read("/sys/fs/cgroup/cpu.stat").splitlines():
        if line.startswith("usage_usec"):
            try:
                return float(line.split()[1]) / 1e6
            except (IndexError, ValueError):
                return None
    v1 = _read("/sys/fs/cgroup/cpuacct/cpuacct.usage")
    if v1:
        try:
            return float(v1) / 1e9
        except ValueError:
            return None
    return None


def busy_cores(dwell=0.4):
    """Cores currently in use by this cgroup, or None if unreadable."""
    a = cgroup_cpu_usage_s()
    if a is None:
        return None
    t0 = time.perf_counter()
    time.sleep(dwell)
    b = cgroup_cpu_usage_s()
    elapsed = time.perf_counter() - t0
    if b is None or elapsed <= 0:
        return None
    return max(0.0, (b - a) / elapsed)


# --------------------------------------------------------------------------- #
# guard — may this box be probed right now?
# --------------------------------------------------------------------------- #
try:                                                       # noqa: SIM105
    from gemm_probe import running_job_ids
except Exception:                                          # noqa: BLE001
    # Both probes ride the jobd bundle, but a box mid-rotation can carry one and
    # not the other. Neither may take the other down at import.
    def running_job_ids(state_dir):
        """One `<jid>.running` per job jobd is running right now."""
        if not state_dir:
            return []
        try:
            return sorted(os.path.basename(p)[: -len(".running")]
                          for p in glob.glob(os.path.join(state_dir,
                                                          "*.running")))
        except OSError:
            return []


def busy_reason(*, running_jobs=(), level=None, threshold=BUSY_CORES):
    """None when it is safe to probe; otherwise a short reason slug.

    Unlike `gemm_probe.busy_reason`, an UNREADABLE level does not refuse. There
    the unreadable case means a card we cannot prove idle; here it means a host
    whose cgroup files we cannot see, which is most non-container environments
    including every test runner. `running_jobs` is the definitive guard and does
    not depend on the cgroup being legible.
    """
    if running_jobs:
        return "job_running:" + ",".join(sorted(running_jobs)[:3])
    if level is not None and level > threshold:
        return f"cpu_busy:{level:.2f}_cores"
    return None


# --------------------------------------------------------------------------- #
# the kernel
# --------------------------------------------------------------------------- #
def kernel(iters=KERNEL_ITERS, seed=1):
    """A fixed integer/branch workload. Deterministic, allocation-free.

    Pure Python on purpose: it needs no toolchain, so it is the one measurement
    that runs on every box including a bare GPU trainer. Absolute values are
    interpreter-bound and meaningless; the ratios between machines are the
    product, and those are exactly comparable because the work is identical.
    """
    x = seed & 0xFFFFFFFF
    acc = 0
    for _ in range(iters):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        acc += x & 0xFF
        if x & 1:
            acc ^= x >> 7
    return acc


def _timed(iters, rounds):
    """Best-of-`rounds` wall time for `iters` kernel iterations.

    Best-of, not mean: we are after the machine's capability, and every source
    of error here (a scheduler preemption, a co-tenant spike) is one-directional
    and makes it look slower.
    """
    best = None
    for _ in range(rounds):
        t0 = time.perf_counter()
        kernel(iters)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best


def _pool_worker(args):
    iters, rounds = args
    return _timed(iters, rounds)


def bench_single(iters=KERNEL_ITERS, rounds=DEFAULT_ROUNDS):
    """{per_s, wall_s} for one worker."""
    wall = _timed(iters, rounds)
    return {"single_per_s": round(iters / wall, 2), "single_wall_s": round(wall, 4)}


def bench_allcore(workers, iters=KERNEL_ITERS, rounds=DEFAULT_ROUNDS):
    """{allcore_per_s, workers, wall_s} — `workers` copies of the kernel at once.

    Aggregate rate is total work over WALL time, so a machine that serialises
    the workers scores as if it had one core, which is the finding.

    Best-of-`rounds` over whole pool runs, matching `bench_single`'s protocol.
    The symmetry is load-bearing: `scaling` divides one by the other, and giving
    the single-worker leg best-of-3 while the pool got one shot would book every
    unlucky pool round as a machine that fails to scale.
    """
    workers = max(1, int(workers))
    if workers == 1:
        wall = _timed(iters, rounds)
        return {"allcore_per_s": round(iters / wall, 2), "workers": 1,
                "allcore_wall_s": round(wall, 4), "allcore_count": iters}

    import concurrent.futures as cf                        # noqa: PLC0415
    import multiprocessing as mp                           # noqa: PLC0415

    # fork keeps startup off the measurement; spawn re-imports this module per
    # worker, which on a 64-wide box is a large constant added to `wall`.
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()

    best = None
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        for _ in range(max(1, int(rounds))):
            t0 = time.perf_counter()
            list(ex.map(_pool_worker, [(iters, 1)] * workers))
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
    return {"allcore_per_s": round(workers * iters / best, 2),
            "workers": workers, "allcore_wall_s": round(best, 4),
            "allcore_count": workers * iters}


# --------------------------------------------------------------------------- #
# the optional compile bench
# --------------------------------------------------------------------------- #
def find_cc():
    for name in _COMPILERS:
        p = shutil.which(name)
        if p:
            return p
    return None


def bench_compile(reps=COMPILE_REPS, cc=None, deadline_s=45.0):
    """{count, wall_s, cc} for `reps` compiles of a fixed TU, or None.

    Closer to the work these boxes actually do than the Python kernel, and
    therefore the better number where it exists — but it needs a toolchain, so
    it can never be the universal one.
    """
    cc = cc or find_cc()
    if not cc:
        return None
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "probe_tu.c")
        with open(src, "w") as fh:
            fh.write(c_source())
        cmd = [cc, "-O2", "-c", "-o", os.path.join(d, "probe_tu.o"), src]
        try:
            first = subprocess.run(cmd, capture_output=True, timeout=deadline_s)
        except (OSError, subprocess.SubprocessError):
            return None
        if first.returncode != 0:
            return None
        t0 = time.perf_counter()
        done = 0
        for _ in range(reps):
            if time.perf_counter() - t0 > deadline_s:
                break
            try:
                r = subprocess.run(cmd, capture_output=True, timeout=deadline_s)
            except (OSError, subprocess.SubprocessError):
                break
            if r.returncode != 0:
                break
            done += 1
        wall = time.perf_counter() - t0
    if done < 1 or wall <= 0:
        return None
    return {"count": done, "wall_s": round(wall, 4),
            "cc": os.path.basename(cc)}


# --------------------------------------------------------------------------- #
# host attribution
# --------------------------------------------------------------------------- #
def cpu_name():
    """The `model name` line, or ''. Attribution only — never a capability claim."""
    for line in _read("/proc/cpuinfo").splitlines():
        if line.lower().startswith(("model name", "cpu model")):
            return line.split(":", 1)[-1].strip()
    return ""


def _env_ids():
    """Identity the box knows about itself. `machine_id` is NOT among it — vast
    injects no such variable — so it appears only if something put it in the
    environment. Resolving instance -> machine is `hostfacts.py ingest`'s job.
    "Inherit, never invent." """
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
def build_record(bench, *, cores=None, width_source=None, level=None,
                 elapsed_s=None, status=None, reason=""):
    """The `pyops` record: a `hostfacts.cpu_record` plus probe attribution.

    `scaling` is against `workers`, not `cores`: with MAX_WORKERS below slice
    width the two differ, and dividing by `cores` would report a capped run as
    a machine that fails to scale.
    """
    rec = {
        "probe_version": PROBE_VERSION,
        "kind": "cpu",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status or ("ok" if bench else "failed"),
    }
    rec.update(_env_ids())
    if cores:
        rec["cores"] = round(float(cores), 3)
    if width_source:
        rec["width_source"] = width_source
    if level is not None:
        rec["pre_probe_busy_cores"] = round(float(level), 3)
    if elapsed_s is not None:
        rec["elapsed_s"] = round(float(elapsed_s), 2)
    if reason:
        rec["reason"] = reason
    name = cpu_name()
    if name:
        rec["cpu_name"] = name
    if not bench:
        return rec
    rec.update(bench)
    single = bench.get("single_per_s")
    allcore = bench.get("allcore_per_s")
    workers = bench.get("workers") or 1
    if single and allcore and workers:
        rec["scaling"] = round(allcore / (single * workers), 4)
    return rec


def _tag(s):
    """Field-safe token: jobd's `K=V` parser splits on '=', bash on whitespace,
    the heartbeat packer on ',' and ':'."""
    out = [ch if (ch.isalnum() or ch in "_.+-") else "_" for ch in str(s)]
    return "".join(out).strip("_") or "unknown"


def render_fields(rec):
    """`k=v` lines for `jobd.sh emit_box cpu_probe ...` — no whitespace in a value."""
    pairs = [("status", _tag(rec.get("status", "unknown")))]
    if rec.get("cpu_name"):
        pairs.append(("cpu", _tag(rec["cpu_name"])))
    if rec.get("width_source"):
        pairs.append(("width_source", _tag(rec["width_source"])))
    for key in ("cores", "workers", "single_per_s", "allcore_per_s", "scaling",
                "pre_probe_busy_cores", "compile_per_s", "elapsed_s",
                "probe_version"):
        if rec.get(key) is not None:
            pairs.append((key, rec[key]))
    if rec.get("reason"):
        pairs.append(("reason", _tag(rec["reason"])[:120]))
    return "\n".join(f"{k}={v}" for k, v in pairs)


# --------------------------------------------------------------------------- #
# the whole probe
# --------------------------------------------------------------------------- #
def probe(*, iters=KERNEL_ITERS, rounds=DEFAULT_ROUNDS, max_workers=MAX_WORKERS,
          state_dir=None, force=False, with_compile=True,
          deadline_s=DEFAULT_DEADLINE_S):
    """(pyops_record, compile_bench|None). Never raises; refusals are records."""
    t0 = time.perf_counter()
    cores, width_source = cpu_width()
    level = busy_cores()
    jobs = running_job_ids(state_dir or os.environ.get("JOBD_STATE_DIR"))

    reason = None if force else busy_reason(running_jobs=jobs, level=level)
    if reason:
        return build_record(None, cores=cores, width_source=width_source,
                            level=level, status=f"skipped:{reason}",
                            elapsed_s=time.perf_counter() - t0), None

    try:
        bench = bench_single(iters, rounds)
        workers = max(1, min(int(cores), int(max_workers)))
        bench.update(bench_allcore(workers, iters, rounds))
    except Exception as e:                                 # noqa: BLE001
        return build_record(None, cores=cores, width_source=width_source,
                            level=level, status="failed",
                            reason=f"{type(e).__name__}: {e}",
                            elapsed_s=time.perf_counter() - t0), None

    comp = None
    if with_compile and (time.perf_counter() - t0) < deadline_s:
        try:
            comp = bench_compile()
        except Exception:                                  # noqa: BLE001
            comp = None                                    # never fatal
    if comp:
        bench["compile_per_s"] = round(comp["count"] / comp["wall_s"], 4)
        bench["compile_cc"] = comp["cc"]

    rec = build_record(bench, cores=cores, width_source=width_source,
                       level=level, elapsed_s=time.perf_counter() - t0)
    return rec, comp


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _import_hostfacts():
    """hostfacts.py sits beside us on a box and one dir up in the repo — the
    same seam jobd.sh uses to find the probes themselves."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (here, os.path.dirname(here)):
        if d not in sys.path:
            sys.path.insert(0, d)
    import hostfacts                                       # noqa: PLC0415
    return hostfacts


def _cmd_bench(a):
    rec, _ = probe(iters=a.iters, rounds=a.rounds, max_workers=a.max_workers,
                   state_dir=a.state_dir, force=a.force,
                   with_compile=not a.no_compile)
    if a.fields:
        print(render_fields(rec))
    else:
        print(json.dumps(rec, indent=2, sort_keys=True))
    return 0


def _cmd_drop(a):
    """Measure and hand the records to jobd's drop dir.

    Two separate records when a compiler exists: `pyops` and `compile_tu` are
    different units and `summarize_cpu` must never average them together.
    """
    hf = _import_hostfacts()
    rec, comp = probe(iters=a.iters, rounds=a.rounds, max_workers=a.max_workers,
                      state_dir=a.state_dir, force=a.force,
                      with_compile=not a.no_compile)
    written = []
    if rec.get("status") == "ok":
        extra = {k: v for k, v in rec.items()
                 if k in ("scaling", "workers", "single_per_s", "allcore_per_s",
                          "width_source", "pre_probe_busy_cores",
                          "probe_version", "machine_id")}
        written.append(hf.drop_cpu_record(
            units="pyops", count=rec["allcore_count"],
            wall_s=rec["allcore_wall_s"], cores=rec.get("cores"),
            cpu_name=rec.get("cpu_name"), workload="cpu_probe.kernel",
            instance_id=rec.get("instance_id"), directory=a.directory, **extra))
        if comp:
            # `cores=1`, NOT the slice width: `bench_compile` is a serial loop
            # of one subprocess at a time, so its rate is a SINGLE-THREAD rate.
            # Dividing it by the width made `per_core_s` mean
            # single-thread-over-width, which ranks narrow boxes best and read
            # as a 35x fleet spread that was mostly just width. Records written
            # before 2026-08-27 carry the width here; read their `per_s`, which
            # was always right.
            written.append(hf.drop_cpu_record(
                units="compile_tu", count=comp["count"], wall_s=comp["wall_s"],
                cores=1, cpu_name=rec.get("cpu_name"),
                workload=f"cpu_probe.compile:{comp['cc']}",
                instance_id=rec.get("instance_id"), directory=a.directory,
                width_source=rec.get("width_source"),
                probe_version=PROBE_VERSION))
    if a.fields:
        print(render_fields(rec))
    for p in written:
        print(f"dropped {p}", file=sys.stderr)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("bench", _cmd_bench), ("drop", _cmd_drop)):
        p = sub.add_parser(name)
        p.add_argument("--iters", type=int, default=KERNEL_ITERS)
        p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
        p.add_argument("--max-workers", type=int, default=MAX_WORKERS)
        p.add_argument("--state-dir", default=None)
        p.add_argument("--no-compile", action="store_true")
        p.add_argument("--force", action="store_true",
                       help="probe even if the box looks busy (skews the number)")
        p.add_argument("--fields", action="store_true",
                       help="emit jobd k=v lines instead of JSON")
        if name == "drop":
            p.add_argument("--directory", default=None)
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
