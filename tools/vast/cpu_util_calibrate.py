#!/usr/bin/env python3
"""Calibrate what vast's `cpu_util` field actually MEANS, by passive correlation.

The vast instance payload carries `cpu_util` alongside `gpu_util`, and the fleet
needs it to tell a busy CPU box from an idle one (`fleet/rows.workload_evidence`).
But the units are undocumented and two readings fit the data equally well:

  * **cores-busy** — a count, directly comparable to load average
  * **percent-of-box** — 0..100, needing a `cpu_cores_effective` multiply

The two differ by a factor of `cpu_cores_effective`, so boxes with DIFFERENT core
counts separate them: a 256-core and a 32-core box cannot both fit the wrong
hypothesis. This samples the live fleet and fits both.

WHY PASSIVE, AND NOT `stress-ng`
--------------------------------
The obvious calibration is a known N-core load. It is the wrong tool here: the
boxes that make this measurable are the ones running real work, and at least one
carries a frozen comparand (an A/B whose `--workers`/`--beam-depth` are terms in
the comparison). Adding load to a box mid-measurement corrupts the thing being
measured. So this reads only what is already there, and accepts the cost: it
samples the operating points that happen to occur, not the ones we would choose.

THE COMPARAND IS /proc/stat, NOT LOAD AVERAGE
---------------------------------------------
The first cut of this script fitted against load average and had to be thrown
away. Linux load counts tasks in UNINTERRUPTIBLE SLEEP as well as running ones,
so an I/O-heavy box reads far above its CPU use: the live vLLM serve box showed
load 4.38 against `cpu_util` 1.24, a 3.5x gap that is real I/O, not a unit
mismatch. Fitting a unit question against that comparand would have "discovered"
a coefficient that was mostly disk and network.

So busy-cores is measured directly, from two `/proc/stat` reads a few seconds
apart: `busy_cores = (1 - idle_delta/total_delta) * ncpu`. That is the same
quantity `cpu_util` would be under the cores-busy hypothesis, with no queueing
semantics in the way. Load average is still recorded, but only as context.

AND THE COMPARAND IS THE CGROUP, NOT /proc/stat
----------------------------------------------
The second cut was thrown away too, for a subtler reason. A container does not
virtualise `/proc/stat`, so it reports the WHOLE HOST — and these slices are 32
effective cores of a 256-core machine. Box 48293057 read a load average of 26
and `busy_cores` 4.3 host-wide while its own cgroup was using **0.34 cores**:
that load was other tenants. An early note here called that box's `cpu_util=0.0`
"telemetry absent under real load"; it was neither absent nor under load, and
the claim was an artifact of measuring the wrong thing. Usage now comes from the
cgroup counter, which is scoped to us.

WHAT IT STILL CANNOT TELL YOU
-----------------------------
It samples the operating points that happen to occur, not the ones we would
choose — and if every live box has the same `cpu_cores_effective` (all 32 on
2026-08-21) the fleet CANNOT separate the two hypotheses at all, because they
differ only by that factor. Expect AMBIGUOUS until the fleet spans core counts.

That is survivable because the consuming predicate does not need the units. The
costs are asymmetric: a false "busy" leaves a box unparked and wastes cents, a
false "idle" parks real work and destroys hours. So the threshold is set LOW and
over-inclusive on purpose, and the field is POSITIVE EVIDENCE ONLY — a 0.0 or
absent reading means "this signal says nothing", never "the box is idle".

USAGE
    cpu_util_calibrate.py sample --rounds 12 --interval 45 --out samples.jsonl
    cpu_util_calibrate.py fit samples.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from vastlib.core import api, config  # noqa: E402


def _live_boxes() -> list[dict]:
    """Instances the API reports running, with the CPU fields we are fitting."""
    d = api.request("GET", "v1/instances/")
    out = []
    for i in d.get("instances") or []:
        if (i.get("actual_status") or "").lower() != "running":
            continue
        out.append({
            "iid": str(i.get("id")),
            "label": i.get("label"),
            "cpu_util": i.get("cpu_util"),
            "gpu_util": i.get("gpu_util"),
            "cpu_cores": i.get("cpu_cores"),
            "cpu_cores_effective": i.get("cpu_cores_effective"),
            "cpu_name": i.get("cpu_name"),
        })
    return out


def _parse_usage(chunk: str) -> tuple[str, float] | None:
    """(cgroup_version, cumulative CPU-seconds) from a `v2 <usec>` / `v1 <ns>`
    line. Both are normalised to SECONDS so the delta is directly core-seconds
    per wall-second, i.e. busy cores."""
    for ln in chunk.splitlines():
        f = ln.split()
        if len(f) == 2 and f[0] in ("v1", "v2"):
            try:
                v = float(f[1])
            except ValueError:
                continue
            return f[0], (v / 1e6 if f[0] == "v2" else v / 1e9)
    return None


def _box_load(iid: str, dwell: float = 5.0, timeout: int = 60) -> dict | None:
    """Busy-cores measured on the box, plus the cgroup quota and load context.

    The quota matters and `nproc` does not: a vast box is a SLICE, and `nproc`
    reports the whole host. A 256-core reading against a 32-core slice would fit
    the wrong hypothesis by exactly the factor we are trying to resolve.
    """
    # OUR cgroup's CPU time, not /proc/stat. A container does not virtualise
    # /proc/stat, so it reports the WHOLE HOST — and these slices are 32 cores
    # of a 256-core box, so a host-wide reading is mostly other tenants. The
    # cgroup counter is scoped to us: cgroup v2 `cpu.stat: usage_usec`, v1
    # `cpuacct.usage` (ns). Measured 2026-08-21, this is the difference between
    # a comparand that answers the question and one that answers someone else's.
    usage = ("(cat /sys/fs/cgroup/cpu.stat 2>/dev/null | awk '/^usage_usec/{print \"v2\", $2}')"
             " || true; "
             "(cat /sys/fs/cgroup/cpuacct/cpuacct.usage 2>/dev/null | awk '{print \"v1\", $1}')"
             " || true")
    cmd = (f"cat /proc/loadavg; echo ---; {usage}; sleep {dwell}; echo ..; "
           f"{usage}; echo ---; "
           f"cat /sys/fs/cgroup/cpu.max 2>/dev/null || "
           f"cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us "
           f"/sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null; echo ---; nproc")
    try:
        p = subprocess.run(
            [sys.executable, os.path.join(_HERE, "herdd.py"), "ssh", iid, "--exec", cmd],
            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if p.returncode != 0:
        return None
    parts = [s.strip() for s in p.stdout.split("---")]
    if len(parts) < 4:
        return None
    try:
        la = parts[0].split()
        load1, load5 = float(la[0]), float(la[1])
    except (ValueError, IndexError):
        return None
    try:
        nproc = int(parts[3].split()[-1])
    except (ValueError, IndexError):
        return None
    before, _, after = parts[1].partition("..")
    a, b = _parse_usage(before), _parse_usage(after)
    if a is None or b is None:
        return None
    (src, ua), (_, ub) = a, b
    busy_cores = (ub - ua) / dwell           # both normalised to seconds
    if busy_cores < 0:
        return None                          # counter reset / rehost mid-sample
    quota = None
    qt = parts[2].split()
    try:
        if len(qt) >= 2 and qt[0] != "max":
            quota = float(qt[0]) / float(qt[1])
    except (ValueError, ZeroDivisionError):
        quota = None
    return {"busy_cores": round(busy_cores, 3), "cgroup": src, "load1": load1,
            "load5": load5, "quota_cores": quota, "nproc": nproc,
            "dwell_s": dwell}


def cmd_sample(a: argparse.Namespace) -> None:
    config.load_env()
    n = 0
    with open(a.out, "a", encoding="utf-8") as fh:
        for r in range(a.rounds):
            # The API read and the on-box read must bracket the same moment; the
            # API read is cheap so it goes first and is re-read per box below is
            # NOT done deliberately — one payload, many boxes, one timestamp.
            try:
                boxes = _live_boxes()
            except SystemExit:            # api.request exits on HTTP error
                print(f"round {r}: API read failed, skipping", file=sys.stderr)
                time.sleep(a.interval)
                continue
            ts = time.time()
            for b in boxes:
                ld = _box_load(b["iid"])
                if ld is None:
                    print(f"  {b['iid']}: unreachable, skipped", file=sys.stderr)
                    continue
                row = {"ts": ts, "round": r, **b, **ld}
                fh.write(json.dumps(row) + "\n")
                fh.flush()               # a killed sampler keeps what it got
                n += 1
                print(f"  {b['iid']:10} cpu_util={b['cpu_util']!s:>10} "
                      f"busy_cores={ld['busy_cores']:>7.2f} "
                      f"load1={ld['load1']:>7.2f} nproc={ld['nproc']} "
                      f"eff={b['cpu_cores_effective']}")
            print(f"round {r + 1}/{a.rounds}: {n} samples so far", flush=True)
            if r + 1 < a.rounds:
                time.sleep(a.interval)
    print(f"wrote {n} samples to {a.out}")


def cmd_fit(a: argparse.Namespace) -> None:
    rows = [json.loads(ln) for ln in open(a.samples, encoding="utf-8") if ln.strip()]
    if not rows:
        sys.exit("no samples")
    # Per box: does cpu_util track load DIRECTLY (cores-busy) or load/cores*100
    # (percent-of-box)? Report both residuals and let the numbers decide.
    by_box: dict[str, list[dict]] = {}
    for r in rows:
        by_box.setdefault(r["iid"], []).append(r)
    print(f"{'box':<10} {'label':<20} {'n':>3} {'ncpu':>5} "
          f"{'cpu_util':>9} {'busy_cores':>11} {'ratio':>8}  note")
    ratios = []
    for iid, rs in sorted(by_box.items()):
        rs = [r for r in rs if isinstance(r.get("busy_cores"), (int, float))]
        if not rs:
            continue
        # The SLICE, not the host. `busy_cores` comes from our cgroup, so the
        # percent hypothesis must be tested against the same denominator vast
        # prices us on — `cpu_cores_effective` (32 here) — and not `nproc`
        # (256, the whole machine). Testing against nproc predicts 0.39 where
        # the truth is 3.125, i.e. it refutes the correct hypothesis.
        ncpu = rs[-1].get("cpu_cores_effective") or rs[-1].get("nproc") or 0
        us = [r["cpu_util"] for r in rs if isinstance(r.get("cpu_util"), (int, float))]
        bs = [r["busy_cores"] for r in rs]
        if not us:
            continue
        mu, mb = sum(us) / len(us), sum(bs) / len(bs)
        note = ""
        # A box whose telemetry vast never populates is not an idle box: it is a
        # box with no signal, and including it would drag the fit toward zero.
        #
        # The test is not "the number is inconvenient": measured busy_cores of
        # `mb` predicts a NONZERO cpu_util under BOTH hypotheses (mb cores, or
        # mb/cores*100 percent), so an exact 0.0 is consistent with neither.
        # Corroborated independently on 48293057, 2026-08-21 — its gpu_util and
        # mem_util were null and disk_util -1 in the same payload, i.e. the
        # whole telemetry block was dead. Note that `cpu_util` renders an
        # unreported box as 0.0 where `gpu_util` says None, which is exactly why
        # the consuming predicate treats 0.0 as "no signal", never as "idle".
        if mu == 0.0 and mb > 0.2:
            note = "TELEMETRY UNREPORTED (0.0 against measured work)"
        ratio = (mu / mb) if mb > 0.25 else float("nan")
        print(f"{iid:<10} {str(rs[-1].get('label'))[:20]:<20} {len(rs):>3} "
              f"{ncpu:>5} {mu:>9.3f} {mb:>11.2f} {ratio:>8.3f}  {note}")
        if mb > 0.3 and not note:        # same floor as the divide guard above
            ratios.append((iid, ncpu, ratio))
    print()
    if len(ratios) < 2:
        print("VERDICT: inconclusive — need >=2 boxes with real CPU load and "
              "usable telemetry. Re-run when the fleet has them.")
        return
    # cores-busy => ratio ~1 regardless of ncpu; percent => ratio ~100/ncpu.
    print("hypothesis check (ratio = mean cpu_util / mean measured busy_cores):")
    for iid, ncpu, ratio in ratios:
        print(f"  {iid}: cores-busy predicts ~1.000 (got {ratio:.3f}); "
              f"percent-of-box predicts ~{100.0 / ncpu if ncpu else 0:.3f}")
    near1 = all(0.6 <= r <= 1.6 for _, _, r in ratios)
    pct = all(ncpu and abs(r - 100.0 / ncpu) < 0.4 * (100.0 / ncpu)
              for _, ncpu, r in ratios)
    # The two hypotheses differ by EXACTLY the core count, so a fleet whose
    # boxes all share one cannot tell "percent of a 32-core slice" from "cores
    # busy times 3.125" — or from any other constant that happens to equal
    # 100/32. Saying which one won, without saying that, would be a fit
    # reported as a fact.
    spread = {ncpu for _, ncpu, _ in ratios}
    print()
    if near1 and not pct:
        print("VERDICT: CORES-BUSY. cpu_util is a core count; divide by the "
              "core count for a fraction.")
    elif pct and not near1:
        print("VERDICT: PERCENT-OF-SLICE. cpu_util ~ busy_cores / "
              "cpu_cores_effective * 100; multiply by cores/100 for cores.")
    else:
        print("VERDICT: AMBIGUOUS on this evidence. Do NOT pick a coefficient. "
              "Use a predicate true under BOTH readings and say so at the "
              "threshold — see this module's docstring.")
    if len(spread) < 2:
        print(f"  PROVISIONAL: every box sampled has cpu_cores_effective="
              f"{spread.pop() if spread else '?'}. The hypotheses differ by "
              f"exactly that factor, so this fleet cannot separate them from a "
              f"coincidence. Re-run when the fleet spans core counts.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="collect paired API/on-box readings")
    s.add_argument("--rounds", type=int, default=12)
    s.add_argument("--interval", type=float, default=45.0)
    s.add_argument("--out", default="cpu_util_samples.jsonl")
    s.set_defaults(fn=cmd_sample)
    f = sub.add_parser("fit", help="report which hypothesis the samples support")
    f.add_argument("samples")
    f.set_defaults(fn=cmd_fit)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
