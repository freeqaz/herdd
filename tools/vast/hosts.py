#!/usr/bin/env python3
"""hosts.py — per-machine host scorecard: which vast hosts have we used, and
which were fast? Use it to pick warm, fast boxes instead of gambling on a cold
random offer.

WHY THIS EXISTS
  Vast has NO API that says "your docker image is cached on this host." The only
  vast-native "our data lives here" signal is a Volume (see --volumes), and we
  own none by default. But two things ARE true about a machine_id we've rented
  before:
    * its host almost certainly still has our docker image LAYERS cached, so a
      re-rent skips most of the multi-minute image pull (provisioning is the
      dominant, most-variable boot phase); and
    * we already MEASURED its B2 pull throughput last time.
  So we build the history ourselves. `herdd train` records machine_id + geo +
  advertised inet_down on the `launched` event; the box records measured MB/s on
  its `pull_throughput` event. hosts.py joins them by run_id and aggregates per
  machine_id.

THE OTHER HALF
  This file is the POSITIVE scorecard — measured throughput, warm caches, which
  host to prefer. `herdd fleet hosts` is the NEGATIVE one: fleetd's durable
  per-machine strike record (FLEETD_DESIGN.md §9), written by the boot
  watchdogs. `--search` reads it too, so a host the automatic lanes refuse is
  flagged BLOCKED here rather than recommended — two pickers disagreeing about
  the same machine is worse than either being wrong on its own.

USAGE
  hosts.py                       # scorecard: machines we've used, best MB/s first
  hosts.py --search 4090         # live offers for a GPU, KNOWN/fast ones flagged
  hosts.py --search 4090 --price 0.4
  hosts.py --search 5090 --min-effective-cores 56   # CPU-bound lane (opt-in)
  hosts.py --volumes             # vast Volumes we own (host-local persistent data)
  hosts.py --json                # machine-readable

  To pin a launch to a vetted machine:  herdd.py train --machine <id> ...
  (that flag already exists; --search prints the exact command for the best host.)

CREDENTIALS / DEGRADATION
  Same as boxstate.py: VASTAI_API_KEY + B2_* from the nearest .env; B2 read via
  the rclone [b2] remote. History starts from the first launch that recorded
  machine_id — older runs simply contribute nothing (shown as "no host data").

Stdlib only. Reuses herdd.py + boxstate.py helpers by import.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import boxstate                      # noqa: E402  (reuse its read-only B2 accessor)
import herdd                       # noqa: E402  (argparse main is __main__-guarded)

from vastlib.market import hostrep   # noqa: E402  (durable per-machine strikes)

load_env = herdd.load_env


# --------------------------------------------------------------------------- #
# gather: read the runmeta event log, build per-machine history
# --------------------------------------------------------------------------- #
# Event filenames are `<ts>-<actor>-<nonce>.json`; actor is `cli_*` for launcher
# events (the launched event) and `box_<iid>` for on-box events. A busy run has
# hundreds of supervisor/box heartbeats, so we DON'T cat them all: we cat only
# the `-cli` events (launched) and the first few `-box_` events (pull_throughput
# is emitted ~20-30s into boot, so it's among the earliest box events).
_BOX_SCAN_CAP = 14
# The jobs/eval lane's per-box event stream (jobs/nodes/<IID>/events/) is short
# (a few lifecycle events + one asset_throughput per staged asset), so cat it
# whole under a small cap. The event NAME lives inside the JSON body, not the
# ts-actor-nonce filename, so we cannot pre-filter by name — we read + inspect.
_NODE_SCAN_CAP = 60


def _cat_json(b2, run_id, name):
    ok, body = b2.cat(f"runs/{run_id}/events/{name}")
    if not ok:
        return None
    try:
        return json.loads(body)
    except Exception:
        return None


def scan_asset_throughput(b2, iid):
    """ONE bounded scan of a box's per-box stream jobs/nodes/<iid>/events/,
    returning BOTH signals the asset_throughput events carry:

      (max_mbps, {asset_name: bytes})

    `max_mbps` is the host-quality signal (see `_jobs_lane_mbps`). The byte
    totals are the DISK-SIZING signal: jobd's `stage_one_asset` measures each
    staged asset's real on-box size with `_dir_bytes` and has been emitting it
    all along (`onstart/jobd.sh:843`), but the only consumer of this stream
    filtered on `mbps` and read straight past `bytes`. Recovering it makes every
    box that ever staged an asset a retrospective calibration sample for the
    disk estimator — no rentals, no new events, data we already paid for.

    Both are best-effort: a missing stream, unparseable body, or absent field
    yields None / an empty dict, never an exception. The scan stays bounded by
    `_NODE_SCAN_CAP` (cost control), so on a box with very many assets the byte
    map is a subset, not a guaranteed total — callers summing it for a floor
    must treat it as a lower bound."""
    if not iid:
        return None, {}
    present, names = b2.lsf(f"jobs/nodes/{iid}/events/")
    if not present:
        return None, {}
    best = None
    by_asset = {}
    for n in sorted(names)[:_NODE_SCAN_CAP]:
        ok, body = b2.cat(f"jobs/nodes/{iid}/events/{n}")
        if not ok:
            continue
        try:
            ev = json.loads(body)
        except Exception:
            continue
        if ev.get("event") != "asset_throughput":
            continue
        if ev.get("mbps") is not None:
            try:
                m = float(ev.get("mbps"))
                if best is None or m > best:
                    best = m
            except (TypeError, ValueError):
                pass
        name, raw = ev.get("asset"), ev.get("bytes")
        if name and raw is not None:
            try:
                # keyed by asset name and MAXed, not summed: a re-pull after a
                # park/resume re-emits the same asset, and the cache dedupes it
                # on disk, so summing would double-count a box's real footprint.
                b = int(raw)
                if b > by_asset.get(str(name), -1):
                    by_asset[str(name)] = b
            except (TypeError, ValueError):
                pass
    return best, by_asset


def _jobs_lane_mbps(b2, iid):
    """Best-effort MEASURED B2 pull throughput for a jobs/eval box: the MAX MB/s
    seen in its asset_throughput stream (the largest asset gives the most
    reliable sustained rate; a tiny asset over fixed pull overhead
    under-reports), or None. Without this join every jobs/eval box — the most
    numerous fleet — contributes zero host-quality history (the train lane's
    pull_throughput heartbeat never fires there). Thin wrapper over
    `scan_asset_throughput` so the stream is walked once for both signals."""
    return scan_asset_throughput(b2, iid)[0]


def _run_launch_and_throughput(b2, run_id):
    """Return (launched_event, measured_mbps, killed_events) for a run, catting
    only the events that can carry them (bounded), not the whole heartbeat
    stream. killed_events is a list of boot_killed_slow runmeta records (a slow
    host the watcher condemned — each carries its OWN machine_id, which may
    differ from the run's final launched machine_id after a relaunch)."""
    present, names = b2.lsf(f"runs/{run_id}/events/")
    if not present:
        return None, None, []
    names = sorted(names)                          # ts-prefixed => chronological
    launched = None
    killed = []
    # launched AND boot_killed_slow are launcher-side runmeta events (cli actor).
    for n in (x for x in names if "-cli" in x):
        ev = _cat_json(b2, run_id, n)
        if not ev:
            continue
        e = ev.get("event")
        if e == "launched" and ev.get("machine_id"):
            launched = ev
        elif e == "boot_killed_slow" and ev.get("machine_id"):
            killed.append({
                "machine_id": str(ev.get("machine_id")),
                "mbps": ev.get("mbps"), "window_s": ev.get("window_s"),
                "phase": ev.get("phase"), "run_id": run_id, "ts": ev.get("ts"),
            })
    measured = None
    scanned = 0
    for n in (x for x in names if "-box_" in x):
        if scanned >= _BOX_SCAN_CAP:
            break
        scanned += 1
        ev = _cat_json(b2, run_id, n)
        if (ev and ev.get("event") == "heartbeat"
                and ev.get("phase") == "pull_throughput"
                and ev.get("mbps") is not None):
            measured = ev.get("mbps")
            break
    # jobs/eval lane: no train pull_throughput heartbeat exists, so bridge the
    # run to its box via the launched event's instance_id and join the box's
    # asset_throughput stream — how jobs boxes finally contribute host history.
    if measured is None and launched is not None:
        measured = _jobs_lane_mbps(b2, str(launched.get("instance_id") or ""))
    return launched, measured, killed


def gather_hosts(b2):
    """Walk all runs; return {machine_id: {...aggregate...}} keyed by str machine_id.

    Joins per run: the `launched` event carries machine_id/geo/inet_down/gpu/dph;
    the box `pull_throughput` event (event==heartbeat, phase==pull_throughput)
    carries the MEASURED mbps. Either may be absent on older runs."""
    present, runs = b2.lsf("runs/")
    hosts: dict[str, dict] = {}
    killed_map: dict[str, dict] = {}   # machine_id -> newest boot_killed_slow record
    run_ids = [r.rstrip("/") for r in sorted(runs)]
    runs_seen = len(run_ids)
    runs_with_machine = 0
    # rclone cat is one network round-trip per object; fan out across runs so the
    # wall time is one run's latency, not the sum. Order doesn't matter (we
    # aggregate by machine_id).
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=12) as ex:
        results = list(ex.map(
            lambda rid: (rid, *_run_launch_and_throughput(b2, rid)), run_ids))
    for run_id, launched, measured, killed in results:
        for k in killed:
            mid = k["machine_id"]
            prev = killed_map.get(mid)
            if prev is None or (k.get("ts") or "") >= (prev.get("ts") or ""):
                killed_map[mid] = k
        if not launched:
            continue
        runs_with_machine += 1
        mid = str(launched["machine_id"])
        h = hosts.setdefault(mid, {
            "machine_id": mid, "runs": [], "gpu": launched.get("gpu"),
            "geo": launched.get("geolocation") or launched.get("geo"),
            "inet_down": launched.get("inet_down"),
            "measured_mbps": [], "last_ts": None,
        })
        h["runs"].append(run_id)
        # keep the newest advertised metadata
        if launched.get("gpu"):
            h["gpu"] = launched["gpu"]
        if launched.get("geolocation") or launched.get("geo"):
            h["geo"] = launched.get("geolocation") or launched.get("geo")
        if launched.get("inet_down") not in (None, ""):
            h["inet_down"] = launched["inet_down"]
        ts = launched.get("ts")
        if ts and (h["last_ts"] is None or ts > h["last_ts"]):
            h["last_ts"] = ts
        if measured is not None:
            try:
                h["measured_mbps"].append(float(measured))
            except (TypeError, ValueError):
                pass
    # derive medians / sort key + fold in the boot_killed_slow verdicts (a SLOW/
    # KILLED host is a bad host and a natural exclude_machines seed).
    for h in hosts.values():
        h["n_runs"] = len(h["runs"])
        h["med_mbps"] = (round(statistics.median(h["measured_mbps"]), 1)
                         if h["measured_mbps"] else None)
        k = killed_map.get(h["machine_id"])
        h["killed"] = bool(k)
        if k:
            h["kill_mbps"] = k.get("mbps")
            h["kill_window_s"] = k.get("window_s")
            h["kill_phase"] = k.get("phase")
    # a machine condemned before it ever recorded a launched-based host row still
    # belongs on the scorecard as SLOW/KILLED (so it can be avoided next launch).
    for mid, k in killed_map.items():
        if mid not in hosts:
            hosts[mid] = {
                "machine_id": mid, "runs": [], "gpu": None, "geo": None,
                "inet_down": None, "measured_mbps": [], "last_ts": k.get("ts"),
                "n_runs": 0, "med_mbps": None, "killed": True,
                "kill_mbps": k.get("mbps"), "kill_window_s": k.get("window_s"),
                "kill_phase": k.get("phase"),
            }
    return hosts, {"runs_seen": runs_seen, "runs_with_machine": runs_with_machine,
                   "killed_machines": len(killed_map)}


# --------------------------------------------------------------------------- #
# vast API: live offers + owned volumes
# --------------------------------------------------------------------------- #
def _offer_query(gpu, max_price, limit=40):
    """Build the v0/bundles/ query body for a GPU ALIAS.

    `gpu` is whatever `--search` was handed — a friendly alias ('5090',
    'rtxpro6000', 'h100'), which is what this tool's USAGE block advertises.
    The vast API matches `gpu_name` EXACTLY against its own spelling
    ('RTX 5090'), so an alias MUST be expanded through herdd.normalize_gpu()
    before it goes on the wire.

    Skipping that expansion is what made `--search` useless for every GPU
    class: `gpu_name eq '5090'` matches nothing, and the empty result renders
    as "no rentable verified offers for gpu='5090'" — a message that reads like
    a genuinely empty market and is therefore silent about the bug. (Verified
    against the live API 2026-08-06: `eq '5090'` -> 0 offers, `eq 'RTX 5090'`
    -> a full page, with `verified`/`rentable`/price identical. So the alias
    was the ONLY defect; nothing else in the query was wrong.)

    One alias can expand to SEVERAL API names — `rtxpro6000` covers both the WS
    and the S SKU — which `eq` cannot express. Use `in` whenever the expansion
    is not exactly one name; the API honours it (both SKUs come back).
    """
    q = {
        "verified": {"eq": True}, "rentable": {"eq": True},
        "type": "ask", "order": [["dph_total", "asc"]], "limit": limit,
    }
    names = herdd.normalize_gpu([gpu]) if gpu else []
    if len(names) == 1:
        q["gpu_name"] = {"eq": names[0]}
    elif names:
        q["gpu_name"] = {"in": names}
    if max_price:
        q["dph_total"] = {"lte": float(max_price)}
    return q


def _search_offers(gpu, max_price, limit=40):
    """Return live rentable offers for a GPU (list of dicts). Best-effort."""
    q = _offer_query(gpu, max_price, limit)
    import urllib.parse
    ok, d, err = herdd.request_soft(
        "GET", "v0/bundles/?q=" + urllib.parse.quote(json.dumps(q)))
    if not ok or not isinstance(d, dict):
        return [], err
    return d.get("offers", []), None


def _owned_volumes():
    ok, d, err = herdd.request_soft("GET", "v0/volumes/")
    if not ok:
        return [], err
    vols = d.get("volumes", d) if isinstance(d, dict) else d
    return (vols or []), None


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _age(ts):
    if not ts:
        return "?"
    # ts is runmeta's colon-free ms UTC, e.g. 20260709T220606013Z
    try:
        import datetime
        t = datetime.datetime.strptime(ts[:15], "%Y%m%dT%H%M%S").replace(
            tzinfo=datetime.timezone.utc)
        d = (datetime.datetime.now(datetime.timezone.utc) - t).total_seconds()
        if d < 3600:
            return f"{int(d/60)}m ago"
        if d < 86400:
            return f"{int(d/3600)}h ago"
        return f"{int(d/86400)}d ago"
    except Exception:
        return ts[:8]


def scorecard(hosts, meta):
    if not hosts:
        return (f"no host history yet ({meta['runs_seen']} runs scanned, none "
                "recorded a machine_id).\nLaunches from `herdd train` going "
                "forward will populate this (it records machine_id on the "
                "`launched` event). Measured MB/s appears once a training box "
                "emits its pull_throughput event.")
    # killed hosts sink to the bottom (bad hosts), then by measured MB/s desc.
    rows = sorted(hosts.values(),
                  key=lambda h: (h.get("killed", False), h["med_mbps"] is None,
                                 -(h["med_mbps"] or 0), -h["n_runs"]))
    out = ["MACHINES WE'VE USED (image layers likely cached — prefer these):",
           f"  {'MACHINE':>8}  {'GPU':<12} {'GEO':<18} {'ADV Mb/s':>8} "
           f"{'MEAS MB/s':>9} {'RUNS':>4}  {'LAST':<8}  NOTE"]
    n_killed = 0
    for h in rows:
        meas = f"{h['med_mbps']:.1f}" if h["med_mbps"] is not None else "—"
        adv = f"{h['inet_down']:.0f}" if isinstance(h["inet_down"], (int, float)) else "?"
        note = ""
        if h.get("killed"):
            n_killed += 1
            km = h.get("kill_mbps")
            kw = h.get("kill_window_s")
            note = ("SLOW/KILLED"
                    + (f" ({km} MB/s" if km is not None else " (")
                    + (f"/{kw}s" if kw else "")
                    + (f", {h['kill_phase']}" if h.get("kill_phase") else "") + ")")
        out.append(f"  {h['machine_id']:>8}  {(h['gpu'] or '?'):<12} "
                   f"{(h['geo'] or '?'):<18.18} {adv:>8} {meas:>9} "
                   f"{h['n_runs']:>4}  {_age(h['last_ts']):<8}  {note}")
    out.append("")
    out.append("Pin a launch to one:  python3 tools/vast/herdd.py train --machine <MACHINE> ...")
    out.append("MEAS MB/s = median measured B2 pull (train pull_throughput OR jobs/eval "
               "asset_throughput event); '—' = no box has measured it yet.")
    if n_killed:
        out.append(f"SLOW/KILLED = the boot-health watcher condemned this host for a "
                   f"sustained-low pull ({n_killed} machine(s)); avoid on relaunch.")
    return "\n".join(out)


def filter_min_effective_cores(offers, minimum):
    """Drop offers whose EFFECTIVE core count is below `minimum` (and offers
    that do not say). Strictly opt-in — no default search path applies it.

    An offer's own `cpu_cores` is the WHOLE HOST's count; what you rent is
    `cpu_cores x gpu_frac` (see herdd.effective_cores). A row that publishes
    neither is dropped rather than kept: the flag is only ever passed when the
    core count is the thing being selected on, and silently admitting unknowns
    would defeat it."""
    if not minimum:
        return list(offers), 0
    kept = []
    for o in offers:
        eff = herdd.effective_cores(o)
        if eff is not None and eff >= float(minimum):
            kept.append(o)
    return kept, len(offers) - len(kept)


def search_view(hosts, gpu, max_price, min_effective_cores=None):
    offers, err = _search_offers(gpu, max_price)
    if err:
        return f"offer search failed: {err}"
    dropped = 0
    if offers and min_effective_cores:
        offers, dropped = filter_min_effective_cores(offers,
                                                     min_effective_cores)
        if not offers:
            return (f"no offer for gpu={gpu!r} at price<={max_price} has "
                    f">= {min_effective_cores:g} effective cores "
                    f"({dropped} offer(s) filtered out).\n"
                    f"Effective cores = cpu_cores x gpu_frac — the slice of the "
                    f"host you actually rent. Raise --price, drop the floor, or "
                    f"ask for more GPUs (a bigger slice carries more cores).")
    if not offers:
        # name the EXPANDED gpu_name(s) actually sent: an empty market and a
        # bad alias used to render identically, which is how the alias bug
        # (see _offer_query) stayed invisible. `--price` defaults to 0.5 and is
        # a real filter, so echo it as a suspect too.
        expanded = ", ".join(herdd.normalize_gpu([gpu])) if gpu else "(any)"
        return (f"no rentable verified offers for gpu={gpu!r} "
                f"(queried gpu_name: {expanded}) at price<={max_price}.\n"
                f"If that gpu_name looks wrong, the alias is unknown to "
                f"herdd.GPU_ALIASES and was passed through verbatim. "
                f"If it looks right, raise --price (default 0.5).")
    known = {h["machine_id"]: h for h in hosts.values()}
    # The NEGATIVE half of the same question, from fleetd's durable store
    # (FLEETD_DESIGN.md §9). This file's `killed` flag is one run's kill_mbps;
    # `rep` is every condemned boot across every session, decayed. Read here so
    # an operator picking by hand cannot land on a machine the automatic lanes
    # already refuse — the two pickers disagreeing is worse than either being
    # wrong. Fail-open: no store, no annotation.
    try:
        rep = hostrep.verdicts()
    except Exception:                 # noqa: BLE001 — advisory; never fail a search
        rep = {}
    # annotate + sort: blocked and killed hosts to the very bottom; then
    # known-with-measured first (by MB/s), then known, then new.
    def key(o):
        h = known.get(str(o.get("machine_id")))
        r = rep.get(str(o.get("machine_id"))) or {}
        killed = bool(h and h.get("killed")) or bool(r.get("blocked_reason"))
        meas = h["med_mbps"] if h and h["med_mbps"] is not None else -1
        return (killed, -1 if h else 0, -(meas if meas is not None else -1),
                o.get("dph_total", 9) * float(r.get("penalty") or 1.0))
    offers = sorted(offers, key=key)
    out = [f"LIVE OFFERS for {gpu} (price<={max_price}"
           + (f", eff cores>={min_effective_cores:g}"
              if min_effective_cores else "")
           + ") — KNOWN hosts first (warm image cache + measured MB/s):",
           f"  {'OFFER':>10} {'MACHINE':>8}  {'GPU':<10} {'GEO':<16} "
           f"{'ADV Mb/s':>8} {'EFF CORES':>9} {'$/hr':>7}  NOTE"]
    for o in offers[:25]:
        mid = str(o.get("machine_id"))
        h = known.get(mid)
        r = rep.get(mid) or {}
        if r.get("blocked_reason"):
            note = f"BLOCKED by fleetd — {r['blocked_reason']}"
        elif r.get("penalty", 1.0) > 1.0:
            note = (f"reputation x{r['penalty']:.2f} (score {r['score']:.2f}) — "
                    f"prefer a clean host up to "
                    f"{(r['penalty'] - 1) * 100:.0f}% dearer")
        elif h and h.get("killed"):
            km = h.get("kill_mbps")
            note = "SLOW/KILLED — avoid" + (f" (~{km} MB/s)" if km is not None else "")
        elif h and h["med_mbps"] is not None:
            note = f"KNOWN ~{h['med_mbps']:.0f} MB/s, {h['n_runs']} run(s)"
        elif h:
            note = f"KNOWN (used {h['n_runs']}x, no MB/s yet)"
        else:
            note = "new host"
        eff = herdd.effective_cores(o)
        out.append(f"  {o.get('id'):>10} {mid:>8}  "
                   f"{(o.get('gpu_name') or '?'):<10.10} "
                   f"{(o.get('geolocation') or '?'):<16.16} "
                   f"{(o.get('inet_down') or 0):>8.0f} "
                   f"{('?' if eff is None else format(eff, '.0f')):>9} "
                   f"{o.get('dph_total', 0):>7.3f}  {note}")
    best_known = next((o for o in offers
                       if known.get(str(o.get("machine_id")))
                       and known[str(o.get("machine_id"))]["med_mbps"] is not None
                       and not known[str(o.get("machine_id"))].get("killed")), None)
    out.append("")
    out.append("EFF CORES = cpu_cores x gpu_frac — the CPU slice this OFFER "
               "rents, not the host's advertised total (128/384-core 5090 "
               "hosts resolve to 16-55). Compute it per offer; a bigger GPU "
               "slice usually carries more cores. More is faster for the eval "
               "lane's CPU-bound scoring — guidance for PICKING, never config: "
               "the box sizes its own concurrency at runtime.")
    if dropped:
        out.append(f"({dropped} offer(s) hidden by "
                   f"--min-effective-cores {min_effective_cores:g}.)")
    if best_known:
        out.append(f"Fastest KNOWN host available: machine {best_known['machine_id']} "
                   f"(offer {best_known['id']}).")
        out.append(f"  python3 tools/vast/herdd.py train --machine {best_known['machine_id']} ...")
    else:
        out.append("No previously-measured host is currently offered — any pick is a "
                   "cold host. Advertised inet_down is an upper bound only (whole-"
                   "machine Ookla snapshot, verified to 100–500 Mb/s).")
    return "\n".join(out)


def volumes_view():
    vols, err = _owned_volumes()
    if err:
        return f"volume query failed: {err}"
    if not vols:
        return ("no vast Volumes owned. A Volume is host-LOCAL persistent storage: "
                "fill it once with the base weights on a fast host, and a later "
                "instance on that SAME machine reuses them (pull ~0). Constraints: "
                "same-machine rebooking only, and a Volume can't be filled directly "
                "from cloud storage. Create: vastai create volume ... (see vast docs).")
    out = ["VAST VOLUMES WE OWN (host-local persistent data — these machines are warm):"]
    for v in vols:
        out.append(f"  volume {v.get('id')} machine={v.get('machine_id')} "
                   f"{v.get('size', '?')}GB {v.get('label', '')}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="per-machine vast host scorecard")
    ap.add_argument("--search", metavar="GPU",
                    help="show live offers for this GPU, KNOWN/fast hosts flagged")
    ap.add_argument("--price", type=float, default=0.5,
                    help="max $/hr for --search (default 0.5)")
    ap.add_argument("--min-effective-cores", type=float, metavar="N",
                    help="OPT-IN --search filter: keep only offers with at "
                         "least N effective cores (cpu_cores x gpu_frac, the "
                         "slice you actually rent). No default; nothing else "
                         "applies it")
    ap.add_argument("--volumes", action="store_true",
                    help="list vast Volumes we own (host-local persistent data)")
    ap.add_argument("--json", action="store_true", help="machine-readable dump")
    a = ap.parse_args(argv)

    load_env()

    if a.volumes and not a.json:
        print(volumes_view())
        return 0

    b2 = boxstate.B2()
    if not b2.ok:
        print(f"B2 unreachable ({b2.reason}) — cannot read host history.",
              file=sys.stderr)
        # --search can still work off the API alone (no KNOWN annotations)
        if a.search:
            print(search_view({}, a.search, a.price, a.min_effective_cores))
            return 0
        return 2

    hosts, meta = gather_hosts(b2)

    if a.json:
        vols, _ = _owned_volumes()
        print(json.dumps({"hosts": list(hosts.values()), "meta": meta,
                          "volumes": vols}, indent=2, default=str))
        return 0

    if a.search:
        print(search_view(hosts, a.search, a.price, a.min_effective_cores))
        return 0

    print(scorecard(hosts, meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
