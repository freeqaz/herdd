#!/usr/bin/env python3
"""Host metrics probe — one-shot GPU/CPU/mem/net/disk snapshot for vast boxes.

Stdlib-only, zero deps, self-contained. Runs either ON the box (shipped over ssh
stdin by `herdd metrics`, or invoked from the baked tree by the box heartbeat
loops) or locally. Answers the question we could only answer by hand-running
`nvidia-smi` before: *is this box saturating the GPU, or is it capped by network
I/O / CPU contention / disk / a thermal-or-power throttle?*

Two output modes:

  snapshot [--json | --table]   full per-card block + host rollup + a one-line
                                advisory verdict (default --table)
  fields   [--window S]         one compact `k:v,k:v` line meant to be spliced
                                into a heartbeat event as a single `--field
                                host_metrics=<line>` (no '=' / no whitespace in
                                the value, so the K=V field parser stays happy)

Delta metrics (cpu %, net MB/s, disk MB/s) are sampled over --window seconds
(default 1.0): read the /proc counters, sleep, re-read, divide. The GPU block is
read once via `nvidia-smi --query-gpu` (+ an optional `dmon -s t` PCIe sample).
The nvidia-smi binary/argv is overridable via $METRICS_NVIDIA_SMI so tests can
inject a fake and a GPU-less box degrades cleanly (set it to /bin/false → the
`gpus` list is empty and an error is noted, never raised).

Nothing here raises on a missing card, an `[N/A]` field, or an absent /proc file
— a probe must never wedge the caller (a dying box still has to self-park).
"""
import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time

# --------------------------------------------------------------------------- #
# GPU — nvidia-smi --query-gpu
# --------------------------------------------------------------------------- #
# Column order of the query below; parse_nvidia_smi maps positionally.
GPU_QUERY_FIELDS = [
    "index", "name", "utilization.gpu", "utilization.memory",
    "memory.used", "memory.total", "power.draw", "power.limit",
    "temperature.gpu", "clocks.sm", "pstate", "clocks_throttle_reasons.active",
]

# clocks_throttle_reasons.active is a hex bitmask; each bit → a slug. Names per
# the NVML docs. gpu_idle/app_clocks/sync_boost are benign; sw_power (running at
# the power limit) is expected under load; sw_thermal/hw_* are the ones that
# actually mean "your card can't run as fast as it wants".
THROTTLE_BITS = [
    (0x0000000000000001, "gpu_idle"),
    (0x0000000000000002, "app_clocks"),
    (0x0000000000000004, "sw_power"),
    (0x0000000000000008, "hw_slowdown"),
    (0x0000000000000010, "sync_boost"),
    (0x0000000000000020, "sw_thermal"),
    (0x0000000000000040, "hw_thermal"),
    (0x0000000000000080, "hw_power_brake"),
    (0x0000000000000100, "display_clocks"),
]
# Throttle reasons that mean the GPU is being actively held below its target —
# worth surfacing in the verdict even at high util.
THROTTLE_CONCERNING = {"hw_slowdown", "sw_thermal", "hw_thermal", "hw_power_brake"}

# ---- the idle-card reporting artifact (Blackwell / newer drivers) ------------
# MEASURED 2026-07-30 (BOX_SATURATION_AUDIT §1.1 + §6 red flag 4): on a 4-card
# RTX PRO box, nvidia-smi reported clocks_event_reasons 0x8C —
# sw_power_cap|hw_slowdown|hw_power_brake — on EVERY card, INCLUDING one drawing
# 8 W of a 300 W limit at 0 % utilization for the whole run. Nothing was
# throttled; a card doing no work cannot be power-braked. Every jobd heartbeat
# therefore carried `thr:hw_power_brake|hw_slowdown` on a healthy box, which is
# exactly the field the next person diagnosing a slow box would chase.
# So: when a card is MEASURABLY IDLE, these two hardware bits are relabeled as an
# artifact rather than reported under their alarming names. Only these two — a
# thermal bit is never dismissed. Loaded cards are untouched, and if either
# reading needed to prove idleness is unavailable ([N/A]) the raw name stands.
THROTTLE_IDLE_ARTIFACT = {"hw_slowdown", "hw_power_brake"}
IDLE_UTIL_MAX_PCT = 5          # utilization.gpu at or below this = doing nothing
IDLE_POWER_MAX_FRAC = 0.15     # ...and drawing <=15% of its limit (measured 2.7%)


def decode_throttle(hexstr):
    """'0x0000000000000020' → ['sw_thermal']; benign/none → ['none']."""
    try:
        val = int(str(hexstr).strip(), 16)
    except (ValueError, AttributeError):
        return []
    active = [slug for bit, slug in THROTTLE_BITS if val & bit]
    # Drop gpu_idle from the active set — it's just "clocks are low because
    # there's no work", not a throttle in the sense we care about.
    active = [a for a in active if a != "gpu_idle"]
    return active or ["none"]


def card_is_idle(g):
    """Is this card measurably doing nothing? Requires BOTH readings (util and
    power vs limit) — an unreadable field means "cannot prove idle", so the
    caller keeps reporting whatever the driver claimed."""
    util = g.get("util")
    pwr, lim = g.get("power_w"), g.get("power_limit_w")
    if util is None or pwr is None or not lim:
        return False
    return util <= IDLE_UTIL_MAX_PCT and pwr <= IDLE_POWER_MAX_FRAC * lim


def split_concerning(gpus):
    """(real, artifact): concerning throttle slugs across `gpus`, with the
    idle-card hw_slowdown/hw_power_brake artifact separated out (see
    THROTTLE_IDLE_ARTIFACT). A slug seen for real on ANY card stays in `real` and
    is dropped from `artifact` — one genuinely throttled card is not excused by an
    idle sibling reporting the same bit."""
    real, artifact = set(), set()
    for g in gpus or ():
        hits = {t for t in (g.get("throttle") or []) if t in THROTTLE_CONCERNING}
        if not hits:
            continue
        art = hits & THROTTLE_IDLE_ARTIFACT if card_is_idle(g) else set()
        artifact |= art
        real |= (hits - art)
    return sorted(real), sorted(artifact - real)


def concerning_labels(gpus):
    """Throttle labels for a heartbeat/report line: the real ones by name, plus
    ONE `idle-card-artifact(...)` token when only-idle cards raised hw bits.
    Splice-safe (no '=', no whitespace)."""
    real, artifact = split_concerning(gpus)
    return real + ([f"idle-card-artifact({'|'.join(artifact)})"] if artifact else [])


def _num(tok, cast=float):
    """Parse a --nounits nvidia-smi token; '[N/A]'/'' → None."""
    tok = (tok or "").strip()
    if not tok or tok.startswith("[") or tok.lower() in ("n/a", "na"):
        return None
    try:
        return cast(tok)
    except ValueError:
        return None


def parse_nvidia_smi(csv_text):
    """CSV (noheader,nounits) rows of GPU_QUERY_FIELDS → list of gpu dicts."""
    gpus = []
    for line in (csv_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < len(GPU_QUERY_FIELDS):
            continue
        (idx, name, util, mutil, mused, mtot, pdraw, plim, temp, sclk,
         pstate, thr) = cols[:len(GPU_QUERY_FIELDS)]
        gpus.append({
            "idx": _num(idx, int),
            "name": name,
            "util": _num(util, int),
            "mem_util": _num(mutil, int),
            "mem_used_mb": _num(mused, int),
            "mem_total_mb": _num(mtot, int),
            "power_w": _num(pdraw),
            "power_limit_w": _num(plim),
            "temp_c": _num(temp, int),
            "sm_clock_mhz": _num(sclk, int),
            "pstate": pstate or None,
            "throttle": decode_throttle(thr),
        })
    return gpus


def parse_dmon_t(text):
    """`nvidia-smi dmon -s t` output → {gpu_idx: (rxpci_mbps, txpci_mbps)}.

    Rows look like `    0     51     14`; header/unit lines start with '#'.
    """
    out = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            out[int(parts[0])] = (int(float(parts[1])), int(float(parts[2])))
        except ValueError:
            continue
    return out


def _nvidia_smi_argv():
    """Base nvidia-smi argv, overridable via $METRICS_NVIDIA_SMI (tests / no-GPU
    boxes point it at a fake or /bin/false)."""
    override = os.environ.get("METRICS_NVIDIA_SMI")
    if override:
        return shlex.split(override)
    return ["nvidia-smi"]


def read_gpus(timeout=8):
    """Run nvidia-smi (query + optional dmon PCIe) → (gpus, error_or_None)."""
    base = _nvidia_smi_argv()
    query = base + [
        "--query-gpu=" + ",".join(GPU_QUERY_FIELDS),
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.run(query, capture_output=True, text=True,
                             timeout=timeout)
    except FileNotFoundError:
        return [], "nvidia-smi: not found"
    except subprocess.TimeoutExpired:
        return [], "nvidia-smi: timed out"
    except OSError as e:
        return [], f"nvidia-smi: {e}"
    if out.returncode != 0:
        return [], f"nvidia-smi rc={out.returncode}: {(out.stderr or '').strip()[:120]}"
    gpus = parse_nvidia_smi(out.stdout)
    # Best-effort PCIe throughput — a separate cheap dmon sample; never fatal.
    try:
        d = subprocess.run(base + ["dmon", "-s", "t", "-c", "1"],
                           capture_output=True, text=True, timeout=timeout)
        pci = parse_dmon_t(d.stdout) if d.returncode == 0 else {}
    except (OSError, subprocess.SubprocessError):
        pci = {}
    for g in gpus:
        rxtx = pci.get(g.get("idx"))
        if rxtx:
            g["pcie_rx_mbps"], g["pcie_tx_mbps"] = rxtx
    return gpus, None


# --------------------------------------------------------------------------- #
# /proc counters — cpu, mem, net, disk
# --------------------------------------------------------------------------- #
def _read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return ""


def parse_proc_stat(text):
    """First `cpu ` aggregate line → (busy_jiffies, total_jiffies)."""
    for line in (text or "").splitlines():
        if line.startswith("cpu "):
            vals = [int(x) for x in line.split()[1:] if x.isdigit()]
            if len(vals) < 4:
                break
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
            total = sum(vals)
            return total - idle, total
    return None


def cpu_busy_pct(before, after):
    """Two parse_proc_stat() tuples → CPU busy % over the interval."""
    if not before or not after:
        return None
    dbusy = after[0] - before[0]
    dtotal = after[1] - before[1]
    if dtotal <= 0:
        return None
    return round(100.0 * dbusy / dtotal, 1)


def parse_meminfo(text):
    out = {}
    for line in (text or "").splitlines():
        m = re.match(r"(\w+):\s+(\d+)", line)
        if m:
            out[m.group(1)] = int(m.group(2))  # kB
    return out


def mem_used_pct(meminfo):
    tot = meminfo.get("MemTotal")
    avail = meminfo.get("MemAvailable")
    if not tot:
        return None, None, None
    if avail is None:
        avail = meminfo.get("MemFree", 0)
    used_pct = round(100.0 * (tot - avail) / tot, 1)
    return used_pct, (tot - avail) // 1024, tot // 1024  # pct, used_mb, total_mb


def parse_net_dev(text):
    """/proc/net/dev → {iface: (rx_bytes, tx_bytes)}, lo excluded."""
    out = {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        iface, rest = line.split(":", 1)
        iface = iface.strip()
        if iface == "lo":
            continue
        cols = rest.split()
        if len(cols) < 9:
            continue
        try:
            out[iface] = (int(cols[0]), int(cols[8]))  # rx_bytes, tx_bytes
        except ValueError:
            continue
    return out


def net_rates(before, after, dt):
    """Two parse_net_dev() dicts + interval → (total_rx_mbps, total_tx_mbps,
    {iface: (rx_mbps, tx_mbps)}). MB/s = 1e6 bytes/s."""
    if dt <= 0:
        return 0.0, 0.0, {}
    per = {}
    trx = ttx = 0.0
    for iface, (rx1, tx1) in after.items():
        rx0, tx0 = before.get(iface, (rx1, tx1))
        rxr = max(0, rx1 - rx0) / 1e6 / dt
        txr = max(0, tx1 - tx0) / 1e6 / dt
        per[iface] = (round(rxr, 2), round(txr, 2))
        trx += rxr
        ttx += txr
    return round(trx, 2), round(ttx, 2), per


# whole disks only (not partitions) so read/write bytes aren't double-counted
_DISK_RE = re.compile(r"^(sd[a-z]+|nvme\d+n\d+|vd[a-z]+|xvd[a-z]+|hd[a-z]+)$")


def parse_diskstats(text):
    """/proc/diskstats → {dev: (sectors_read, sectors_written)} for whole disks."""
    out = {}
    for line in (text or "").splitlines():
        cols = line.split()
        if len(cols) < 10:
            continue
        name = cols[2]
        if not _DISK_RE.match(name):
            continue
        try:
            out[name] = (int(cols[5]), int(cols[9]))  # sectors r / w (512 B each)
        except ValueError:
            continue
    return out


def disk_rates(before, after, dt):
    """Two parse_diskstats() dicts + interval → (read_mbps, write_mbps)."""
    if dt <= 0:
        return 0.0, 0.0
    rd = wr = 0.0
    for dev, (r1, w1) in after.items():
        r0, w0 = before.get(dev, (r1, w1))
        rd += max(0, r1 - r0) * 512 / 1e6 / dt
        wr += max(0, w1 - w0) * 512 / 1e6 / dt
    return round(rd, 2), round(wr, 2)


# --------------------------------------------------------------------------- #
# collect + verdict
# --------------------------------------------------------------------------- #
def _snapshot_counters():
    return {
        "t": time.monotonic(),
        "stat": parse_proc_stat(_read("/proc/stat")),
        "net": parse_net_dev(_read("/proc/net/dev")),
        "disk": parse_diskstats(_read("/proc/diskstats")),
    }


def collect(window=1.0):
    """Full host snapshot. Delta metrics span `window` seconds."""
    errors = []
    c0 = _snapshot_counters()
    gpus, gerr = read_gpus()
    if gerr:
        errors.append(gerr)
    # Sleep out the remainder of the window (read_gpus already burned some of it).
    remaining = window - (time.monotonic() - c0["t"])
    if remaining > 0:
        time.sleep(remaining)
    c1 = _snapshot_counters()
    dt = c1["t"] - c0["t"]

    trx, ttx, per_iface = net_rates(c0["net"], c1["net"], dt)
    drd, dwr = disk_rates(c0["disk"], c1["disk"], dt)
    meminfo = parse_meminfo(_read("/proc/meminfo"))
    used_pct, used_mb, total_mb = mem_used_pct(meminfo)
    try:
        load1 = os.getloadavg()[0]
    except (OSError, AttributeError):
        load1 = None
    cores = os.cpu_count() or 1

    utils = [g["util"] for g in gpus if g.get("util") is not None]
    snap = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "window_s": round(dt, 2),
        "gpus": gpus,
        "gpu_count": len(gpus),
        "gpu_util_avg": round(sum(utils) / len(utils), 1) if utils else None,
        "cpu": {
            "busy_pct": cpu_busy_pct(c0["stat"], c1["stat"]),
            "load1": round(load1, 2) if load1 is not None else None,
            "cores": cores,
            "load_per_core": round(load1 / cores, 2) if load1 is not None else None,
        },
        "mem": {"used_pct": used_pct, "used_mb": used_mb, "total_mb": total_mb},
        "net": {"rx_mbps": trx, "tx_mbps": ttx, "per_iface": per_iface},
        "disk": {"read_mbps": drd, "write_mbps": dwr},
        "errors": errors,
    }
    snap["verdict"] = verdict(snap)
    return snap


# ---- utilization says "busy"; power says "how much silicon" -----------------
# MEASURED 2026-08-06/07, and this is why the verdict consults power at all.
# Two training lanes, same card class, indistinguishable by utilization:
#
#   v7 7B  (dec)      util  99-100%   power 86-88% of 600 W   roof-HFU ~41-51%
#   v9 gemma-4 12B    util  99.8%     power    67% of 600 W   roof-HFU ~30-38%
#
# `roof-HFU`, not MFU: executed FLOPs (grad-ckpt recompute billed) over a
# MEASURED dense-bf16 GEMM roof. Relabelled 2026-08-09, numbers unchanged; see
# `tools/vast/mfu.py`'s "WHAT THIS METRIC IS" block. Do not compare to published
# MFU figures.
#
# `utilization.gpu` is the fraction of sample intervals with >=1 kernel RESIDENT.
# It says nothing about how wide that kernel is, so a mask-bound or memory-bound
# kernel pins it at 100 while leaving a third of the power envelope unclaimed.
# The v9 lane held util 100 for 94.3% of a 300 s trace and never once crossed
# 500 W. Power is the cheapest proxy we have for how much silicon is actually
# lit, and reporting "saturated" without it is how a run at ~30% roof-HFU reads
# as healthy to the next person who looks.
POWER_SATURATED_FRAC = 0.80   # >= this: "saturated" is a fair description
POWER_NARROW_FRAC = 0.70      # <  this: busy, but demonstrably not compute-bound


def power_frac(gpus):
    """Mean power draw / power limit across cards reporting BOTH, else None.

    None means "cannot tell", NOT "low" — a driver that hides power.draw
    ([N/A], which this module already tolerates everywhere else) must not be
    read as a narrow kernel. Callers degrade to an explicit unverified verdict.
    """
    fr = []
    for g in gpus or []:
        d, lim = g.get("power_w"), g.get("power_limit_w")
        if isinstance(d, (int, float)) and isinstance(lim, (int, float)) and lim > 0:
            fr.append(d / lim)
    return (sum(fr) / len(fr)) if fr else None


def verdict(s):
    """Advisory one-liner: is the box GPU-bound, or held back by what? Never a
    gate — just a hint pointing at the likely bottleneck."""
    gpus = s.get("gpus") or []
    if not gpus:
        return "no GPU visible" + (f" ({s['errors'][0]})" if s.get("errors") else "")
    util = s.get("gpu_util_avg")
    if util is None:
        return "GPU present, utilization unavailable"
    # REAL throttles only: an idle card's hw_slowdown/hw_power_brake is a driver
    # reporting artifact (THROTTLE_IDLE_ARTIFACT) and must not colour the verdict
    # of a box whose other cards are fine.
    concerning, _artifact = split_concerning(gpus)
    powercap = any("sw_power" in g.get("throttle", []) for g in gpus)
    cpu = s.get("cpu", {})
    lpc = cpu.get("load_per_core") or 0
    net = s.get("net", {})
    net_mbps = (net.get("rx_mbps") or 0) + (net.get("tx_mbps") or 0)
    disk = s.get("disk", {})
    disk_mbps = (disk.get("read_mbps") or 0) + (disk.get("write_mbps") or 0)

    if util >= 85:
        if concerning:
            return f"GPU-bound (util {util:.0f}%) but THROTTLING: {','.join(concerning)}"
        if powercap:
            return f"GPU-bound (util {util:.0f}%, power-capped at limit — expected)"
        # High util alone does NOT establish saturation — see POWER_*_FRAC above.
        pf = power_frac(gpus)
        if pf is None:
            return (f"GPU-bound (util {util:.0f}%) — power draw unavailable, "
                    f"saturation UNVERIFIED")
        if pf < POWER_NARROW_FRAC:
            return (f"GPU busy (util {util:.0f}%) but drawing only {100 * pf:.0f}% "
                    f"of power limit — narrow/memory-bound kernels, NOT saturated")
        if pf < POWER_SATURATED_FRAC:
            return (f"GPU-bound (util {util:.0f}%, power {100 * pf:.0f}% of limit) "
                    f"— mostly saturated")
        return (f"GPU-bound (util {util:.0f}%, power {100 * pf:.0f}% of limit) "
                f"— saturated")
    if util < 8:
        if net_mbps >= 50:
            return f"GPU idle (util {util:.0f}%) — busy on network ({net_mbps:.0f} MB/s)"
        return f"GPU idle (util {util:.0f}%) — no GPU work running?"
    # Under-utilized while doing *something* — point at the likely culprit.
    if concerning:
        return f"GPU under-utilized (util {util:.0f}%) — throttling: {','.join(concerning)}"
    reasons = []
    if lpc >= 1.0:
        reasons.append(f"CPU-contended (load/core {lpc:.1f})")
    if net_mbps >= 50:
        reasons.append(f"high network I/O ({net_mbps:.0f} MB/s)")
    if disk_mbps >= 100:
        reasons.append(f"high disk I/O ({disk_mbps:.0f} MB/s)")
    if reasons:
        return f"GPU under-utilized (util {util:.0f}%) — likely {'; '.join(reasons)}"
    return (f"GPU under-utilized (util {util:.0f}%) — input/host-bound "
            f"(net/cpu/disk all low)")


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _fmt(v, spec="{}", dash="-"):
    return dash if v is None else spec.format(v)


def render_table(s):
    lines = []
    lines.append(f"host {s['host']}  @ {s['ts']}  (window {s['window_s']}s)")
    gpus = s.get("gpus") or []
    if gpus:
        lines.append("")
        lines.append(f"{'GPU':<3} {'util':>4} {'mem':>4} {'vram':>13} "
                     f"{'pwr':>11} {'temp':>4} {'clk':>6} {'pcie r/tx MB/s':>15}  throttle")
        for g in gpus:
            vram = (f"{_fmt(g.get('mem_used_mb'))}/"
                    f"{_fmt(g.get('mem_total_mb'))}MB")
            pwr = f"{_fmt(g.get('power_w'),'{:.0f}')}/{_fmt(g.get('power_limit_w'),'{:.0f}')}W"
            pcie = (f"{_fmt(g.get('pcie_rx_mbps'))}/{_fmt(g.get('pcie_tx_mbps'))}"
                    if g.get("pcie_rx_mbps") is not None else "-")
            # per-card: same idle-artifact relabel as the heartbeat line, so the
            # table never shows a bare `hw_power_brake` on a card at 8 W of 300 W
            art = (THROTTLE_IDLE_ARTIFACT if card_is_idle(g) else set())
            thr = ",".join(f"idle-artifact:{t}" if t in art else t
                           for t in g.get("throttle", []) if t != "none") or "-"
            lines.append(
                f"{_fmt(g.get('idx')):<3} {_fmt(g.get('util'),'{}')+'%':>4} "
                f"{_fmt(g.get('mem_util'),'{}')+'%':>4} {vram:>13} {pwr:>11} "
                f"{_fmt(g.get('temp_c'),'{}')+'C':>4} "
                f"{_fmt(g.get('sm_clock_mhz'),'{}'):>6} {pcie:>15}  {thr}")
    else:
        lines.append("(no GPU visible" +
                     (f" — {s['errors'][0]}" if s.get("errors") else "") + ")")
    cpu = s.get("cpu", {})
    lines.append("")
    lines.append(f"cpu  {_fmt(cpu.get('busy_pct'),'{:.0f}')}% busy   "
                 f"load1 {_fmt(cpu.get('load1'))} / {cpu.get('cores')} cores "
                 f"({_fmt(cpu.get('load_per_core'))}/core)")
    mem = s.get("mem", {})
    lines.append(f"mem  {_fmt(mem.get('used_pct'),'{:.0f}')}% "
                 f"({_fmt(mem.get('used_mb'))}/{_fmt(mem.get('total_mb'))} MB)")
    net = s.get("net", {})
    lines.append(f"net  rx {_fmt(net.get('rx_mbps'))} / tx {_fmt(net.get('tx_mbps'))} MB/s")
    disk = s.get("disk", {})
    lines.append(f"disk r {_fmt(disk.get('read_mbps'))} / w {_fmt(disk.get('write_mbps'))} MB/s")
    lines.append("")
    lines.append(f">> {s.get('verdict', '')}")
    return "\n".join(lines)


def _min_power_limit(gpus):
    """Binding power cap across the cards, in whole watts, or None.

    MIN rather than mean: the slowest card paces a DDP step, so the lowest cap
    is the one that shows up in throughput. When the cards disagree the value
    becomes "<min>-<max>" — a heterogeneous cap is itself worth seeing, and
    collapsing it to one number would hide exactly the case we care about."""
    lims = [g["power_limit_w"] for g in gpus if g.get("power_limit_w")]
    if not lims:
        return None
    lo, hi = round(min(lims)), round(max(lims))
    return lo if lo == hi else f"{lo}-{hi}"


# Field values may not contain '=' (the K=V parser splits on it), whitespace,
# ',' (the k:v separator) or ':' (the key separator). Everything else in a
# device name is dropped rather than substituted — a name is an attribution
# key, not prose, and a lossy-but-stable tag beats a quoted string here.
_NAME_STRIP = re.compile(r"[^A-Za-z0-9_.+-]+")


def _gpu_name_tag(gpus):
    """One filename-safe device tag, or None. Cards are assumed homogeneous —
    if they are not, the tag says so rather than picking a winner."""
    names = []
    for g in gpus:
        n = (g.get("name") or "").strip()
        if not n:
            continue
        # "NVIDIA " prefixes every card and distinguishes nothing.
        n = re.sub(r"^NVIDIA\s+", "", n)
        n = _NAME_STRIP.sub("_", n).strip("_")
        if n and n not in names:
            names.append(n)
    if not names:
        return None
    return names[0] if len(names) == 1 else "MIXED[" + "|".join(names) + "]"


def render_fields(s):
    """One compact `k:v,k:v` line for a heartbeat --field host_metrics=<line>.
    No '=' and no whitespace in the value (K=V field parser splits on '=')."""
    gpus = s.get("gpus") or []
    mem_utils = [g["mem_util"] for g in gpus if g.get("mem_util") is not None]
    temps = [g["temp_c"] for g in gpus if g.get("temp_c") is not None]
    pcts = []
    for g in gpus:
        if g.get("power_w") and g.get("power_limit_w"):
            pcts.append(100.0 * g["power_w"] / g["power_limit_w"])
    thr = concerning_labels(gpus)
    cpu = s.get("cpu", {})
    net = s.get("net", {})
    parts = [
        ("gpu_util", s.get("gpu_util_avg")),
        ("gpu_mem", round(sum(mem_utils) / len(mem_utils), 0) if mem_utils else None),
        ("gpu_pwr", round(sum(pcts) / len(pcts), 0) if pcts else None),
        # ABSOLUTE limit, not just gpu_pwr's percentage OF it. A host that has
        # lowered the cap reads 100% in gpu_pwr exactly like a healthy box at
        # full power — the two are indistinguishable without the denominator.
        # That is how box 46936034 ran 2.13x slow for a whole training window
        # with nothing in its heartbeats looking wrong
        # (docs/plans/witness/perf/PERF_LEVERS_INVESTIGATION_2026-08-06.md §2.4);
        # the probe sampled power.limit and then threw it away here.
        ("gpu_plim", _min_power_limit(gpus)),
        # Device name, so a throughput number can be attributed to a SKU after
        # the box is destroyed. Same reason: 46936034's "Workstation Edition"
        # vs its replacement's "Server Edition" was only recoverable because
        # someone had run nvidia-smi by hand before teardown.
        ("gpu", _gpu_name_tag(gpus)),
        ("gpu_temp", max(temps) if temps else None),
        ("cpu", cpu.get("busy_pct")),
        ("load", cpu.get("load_per_core")),
        ("net_rx", net.get("rx_mbps")),
        ("net_tx", net.get("tx_mbps")),
        ("disk_r", s.get("disk", {}).get("read_mbps")),
        ("disk_w", s.get("disk", {}).get("write_mbps")),
    ]
    kv = [f"{k}:{v:g}" if isinstance(v, float) else f"{k}:{v}"
          for k, v in parts if v is not None]
    if thr:
        kv.append("thr:" + "|".join(thr))
    return ",".join(kv)


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="metrics_probe", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    ps = sub.add_parser("snapshot", help="full host snapshot + verdict")
    ps.add_argument("--window", type=float, default=1.0,
                    help="delta sample window seconds (default 1.0)")
    ps.add_argument("--json", action="store_true", help="raw JSON (for agents)")
    ps.add_argument("--table", action="store_true", help="human table (default)")
    ps.add_argument("--count", type=int, default=1,
                    help="emit N snapshots (0 = forever) for --watch streaming")
    ps.add_argument("--interval", type=float, default=0.0,
                    help="seconds to pause between snapshots when --count != 1")

    pf = sub.add_parser("fields", help="compact k:v line for a heartbeat field")
    pf.add_argument("--window", type=float, default=1.0)

    a = ap.parse_args(argv)
    cmd = a.cmd or "snapshot"

    if cmd == "fields":
        window = getattr(a, "window", 1.0)
        print(render_fields(collect(window=window)))
        return 0

    # snapshot (default), possibly streamed
    count = getattr(a, "count", 1)
    interval = getattr(a, "interval", 0.0)
    as_json = getattr(a, "json", False)
    n = 0
    while count == 0 or n < count:
        s = collect(window=a.window)
        if as_json:
            print(json.dumps(s), flush=True)
        else:
            print(render_table(s), flush=True)
            if count != 1:
                print("", flush=True)
        n += 1
        if (count == 0 or n < count) and interval > 0:
            time.sleep(interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
