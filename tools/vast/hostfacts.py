#!/usr/bin/env python3
"""hostfacts.py — the GEMM ceiling of a MACHINE, not of a job.

WHY THIS FILE EXISTS AT ALL, AND NOT A HEARTBEAT FIELD
------------------------------------------------------
`memory: workload-state-stored-on-the-box` names one defect class behind four
separate vast/jobd bugs: *state that belongs to the WORKLOAD is stored on the
BOX*, and each instance is lost by a different rehost path. A host's dense-bf16
GEMM ceiling is the **mirror image** of that bug — a fact that belongs to the
MACHINE — and it has the same three wrong homes:

  * **Under `jobs/<job_id>/`** (results, heartbeats, the job event stream). This
    is the defect verbatim, only inverted. `job retarget` keeps the JOB_ID and
    moves it to a DIFFERENT box, so a ceiling filed under the job would be
    attributed to whichever machine happened to run first — and a fresh submit
    mints a new JOB_ID, so the same machine's ceiling would be re-measured and
    re-filed from scratch every time. Neither is queryable per host.
  * **In a heartbeat field only.** Heartbeats are per-job events folded into a
    job view. They answer "what was this box doing at 04:12" and cannot answer
    "what does machine 140799 bench at", which is the question that picks a box.
    We still emit one (see `render_fields` in `gemm_probe.py`) because it is free
    and it makes a live run diagnosable — but as a *mirror*, not the store.
  * **In `infra-metadata.db` only.** That file is gitignored and is a
    REBUILDABLE CACHE of the vast API (it has a `meta(fetched_at)` table and its
    `instances` rows are replaced wholesale on every refresh). Nothing in it
    survives the instance being destroyed, which is exactly when the fact
    becomes valuable. It gets a materialised view here; it is not the truth.

WHERE IT ACTUALLY LIVES — two tiers, and why
--------------------------------------------
    jobs/nodes/<IID>/hostfacts/<kind>-<ts>.json            written BY THE BOX
    hostfacts/by-machine/<MACHINE>/<kind>-<IID>-<ts>.json  written by `ingest`

`<kind>` is `gemm` (the dense-bf16 ceiling this file was built for) or `cpu`
(what a machine's CORES are worth, for compile/search work that never touches a
GPU — added 2026-08-21). They share every mechanism below and differ only in
what the record holds, so the kind is a filename token, not a second module.
The `cpu` figure is HARVESTED from work we already run rather than benchmarked:
those boxes already compile thousands of TUs, so the honest rate is the one the
real job produced, and no rental is spent measuring. It exists because the
selection-time prior (`market.offers.cpu_score`, GHz*cores) is blind to IPC and
over-rates old silicon; the correction is measurement, not an IPC table.

The box knows its **instance id** and does not know its **machine id** — vast
injects `CONTAINER_ID`/`INSTANCE_ID` into the container and nothing else. So the
box writes what it knows, keyed by what it knows. "Inherit, never invent."

The box tier is under `jobs/nodes/<IID>/` and **not** under a `hostfacts/` root
of its own for a hard reason: a B2 key carries a **single `namePrefix`**, and a
split box's write key is `namePrefix=jobs/` (`CREDENTIAL_LIFECYCLE.md` §2). A
write anywhere else 403s — that is verbatim the `B2_PUBLISH_KEY_SCOPE_FIX`
incident, where both v7 arms trained to completion and *then* failed to publish.
`jobs/nodes/<IID>/` is not a job namespace: `nodes` is the reserved segment jobd
already uses for the per-BOX lifecycle stream (`jobd_up`, `drained`,
`parked_self`, `JOBD_STATUS`), so the record is host-scoped inside a prefix the
box is actually allowed to write.

The instance -> machine mapping is authoritative only on the laptop (the vast
instances API), and it is **destroyed with the instance**: once a box is gone its
row leaves the API and the mapping is unrecoverable — the same trap
`COMPUTE_OPTIMAL_BOX_SELECTION` §5.1 hit with offer rows ("once a machine is
fully rented its offers leave the market, and the advertised `gpu_max_power`
becomes unrecoverable"). So resolution must be EAGER: `hostfacts.py ingest`
resolves live instances and **pins** a machine-keyed copy, which is thereafter
independent of the API. An unresolvable record is left where it is and stays
readable by instance — it degrades to less-queryable, never to lost.

This is the same join `hosts.py` already runs in production (launcher-side
`launched` events carry machine_id; box-side events carry measurements; joined by
run_id), with the resolution pinned instead of recomputed.

WHAT THIS MODULE DOES NOT DO
----------------------------
No re-rent, no bid. It reads, aggregates and prints.

It DOES now feed selection, which it did not until 2026-08-27. The
acceptance-POLICY half was held unbuilt because "a threshold needs a
distribution, and this is the thing that produces the distribution"
(`docs/plans/witness/perf/HOST_ACCEPTANCE_PROBE_2026-08-07.md` §5). The
distribution exists — 53 machines, 41 CPU models, fleet spread 7.07x — so
`calibrate` freezes it into `cpu_calibration.json` and `market.offers` ranks
and floors CPU-shaped offers on it. The boundary that remains: only offers we
have MEASURED can be refused, because an offer we have never seen cannot be
shown to be slow.

USAGE
    hostfacts.py list                     # per-machine scorecard, fleet median
    hostfacts.py list --local <dir>       # ...from a pulled copy, no network
    hostfacts.py show 140799              # every record for a machine/instance
    hostfacts.py ceiling --instance 46947265 --json > ceiling.json
        # then: mfu.py --model gemma-4-12b-text --tok-s 1430 \
        #             --device '<name>' --ceiling-json ceiling.json
    hostfacts.py ingest                   # resolve live instances -> by-machine/

Stdlib only. B2 goes through the user's rclone `[b2]` remote; every command
takes `--local <dir>` and runs entirely offline against a directory laid out
with the same key structure.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

PREFIX = "hostfacts"
#: The BOX tier. Inside `jobs/` because a split box's write key is
#: `namePrefix=jobs/` and a write anywhere else 403s; `nodes` is jobd's existing
#: per-BOX (not per-job) segment. See the module docstring.
NODES = "jobs/nodes"
NODE_LEAF = "hostfacts"
#: The PINNED tier, written laptop-side with a bucket-wide key.
BY_MACHINE = f"{PREFIX}/by-machine"


# --------------------------------------------------------------------------- #
# keys
# --------------------------------------------------------------------------- #
def _slug(s):
    """Key-safe token. Ids are digits in practice; anything else is reduced
    rather than rejected, so a hand-made record never fails to file."""
    out = "".join(c if (c.isalnum() or c in "_.-") else "_" for c in str(s))
    return out.strip("_") or "unknown"


# The fact KINDS a machine can carry, as the leading token of every filename.
#
# `gemm` is the dense-bf16 ceiling this file was built for. `cpu` is the same
# idea one hardware axis over: what a machine's CORES are worth, for the
# compile/search work that never touches a GPU. They share every mechanism —
# the two tiers, the eager instance->machine pin, "inherit, never invent" — and
# differ only in what the record contains, so the kind is a filename token
# rather than a second module.
#
# Why measured at all, when the offer market advertises cores and GHz:
# `market.offers.cpu_score` multiplies those into a ranking prior and is BLIND
# TO IPC, so it over-rates old silicon by a margin that grows with the age of
# the part. The correction is this file, not a hand-written IPC table.
KIND_GEMM = "gemm"
KIND_CPU = "cpu"
KINDS = (KIND_GEMM, KIND_CPU)


def instance_key(instance_id, ts, kind=KIND_GEMM):
    """`jobs/nodes/<IID>/hostfacts/<kind>-<ts>.json` — what the BOX writes.

    One immutable object per measurement (never a mutable `latest.json`): B2 has
    no compare-and-set, and a shared mutable object under concurrent writers is
    the hazard the runs/ event log is designed around (skill: vast-runs).
    """
    return (f"{NODES}/{_slug(instance_id)}/{NODE_LEAF}/"
            f"{_slug(kind)}-{_slug(ts)}.json")


def instance_prefix(instance_id):
    return f"{NODES}/{_slug(instance_id)}/{NODE_LEAF}"


def machine_key(machine_id, instance_id, ts, kind=KIND_GEMM):
    """`hostfacts/by-machine/<MACHINE>/<kind>-<IID>-<ts>.json` — the PINNED copy.

    Carries the instance id in the name so re-rentals of one machine accumulate
    side by side and a within-machine spread (throttle on one rental, not the
    next) stays visible instead of collapsing to one number.
    """
    return (f"{BY_MACHINE}/{_slug(machine_id)}/"
            f"{_slug(kind)}-{_slug(instance_id)}-{_slug(ts)}.json")


def _kind_from_key(key):
    """Leading filename token -> kind. Unknown tokens read as `gemm`: every
    record written before kinds existed is one, and a rename must never orphan
    a measurement that cost a rental to produce."""
    leaf = key.strip("/").split("/")[-1]
    tok = leaf.split("-", 1)[0]
    return tok if tok in KINDS else KIND_GEMM


def _instance_from_key(key):
    """`jobs/nodes/<IID>/hostfacts/<kind>-<ts>.json` -> `<IID>`."""
    p = key.strip("/").split("/")
    return p[2] if len(p) >= 5 and p[0] == "jobs" and p[1] == "nodes" else None


def _machine_from_key(key):
    """`hostfacts/by-machine/<MACHINE>/<kind>-...json` -> `<MACHINE>`."""
    p = key.strip("/").split("/")
    return p[2] if len(p) >= 4 and p[1] == "by-machine" else None


def _dup_suffix_from_key(key):
    """The `-N` `drop_record` appends to a same-second record, or `""`.

    `ts` is second-resolution and cpu_probe drops two records back to back, so
    the box already disambiguates them by filename. Nothing downstream carried
    that through: `ingest` re-derived the name from `rec["ts"]`, both halves of
    a pair landed on ONE key, and the second put overwrote the first. Carrying
    the suffix keeps the pin deterministic and order-independent — assigning a
    fresh one on collision would rename a record on every re-run.

    Only a trailing `-<digits>` counts, so a ts is never mistaken for one.
    """
    leaf = key.strip("/").split("/")[-1]
    if leaf.endswith(".json"):
        leaf = leaf[:-len(".json")]
    head, sep, tail = leaf.rpartition("-")
    return f"-{tail}" if (sep and head and tail.isdigit()) else ""


# --------------------------------------------------------------------------- #
# stores
# --------------------------------------------------------------------------- #
class LocalStore:
    """A directory laid out with the B2 key structure. Used by the tests, by
    `--local` against a pulled copy, and by the box when B2 is unreachable."""

    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.ok = True
        self.reason = None

    def _p(self, key):
        return os.path.join(self.root, key.replace("/", os.sep))

    def keys(self, prefix):
        base = self._p(prefix)
        out = []
        for dirpath, _dirs, files in os.walk(base):
            for f in files:
                if not f.endswith(".json"):
                    continue
                full = os.path.join(dirpath, f)
                out.append(os.path.relpath(full, self.root).replace(os.sep, "/"))
        return sorted(out)

    def dirs(self, prefix):
        try:
            return sorted(d for d in os.listdir(self._p(prefix))
                          if os.path.isdir(os.path.join(self._p(prefix), d)))
        except OSError:
            return []

    def get(self, key):
        try:
            with open(self._p(key)) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def put(self, key, blob):
        p = self._p(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            json.dump(blob, fh, indent=2)
        return True


class B2Store:
    """B2 via the user's rclone `[b2]` remote. `ok=False` degrades every command
    to a printed reason rather than a traceback — the same contract boxstate.B2
    has, and for the same reason (an agent diagnosing a slow run should get a
    sentence, not a stack)."""

    def __init__(self, bucket=None, runner=None):
        self.bucket = bucket or os.environ.get("B2_BUCKET")
        self._run = runner or self._rclone
        self.ok = bool(self.bucket)
        self.reason = None if self.ok else "B2_BUCKET not set (env or .env)"

    @staticmethod
    def _rclone(args):
        try:
            r = subprocess.run(["rclone", *args], capture_output=True, text=True)
            return r.returncode, r.stdout, r.stderr
        except FileNotFoundError:
            return 127, "", "rclone not found on PATH"

    def _p(self, key):
        return f"b2:{self.bucket}/{key.lstrip('/')}"

    def keys(self, prefix):
        if not self.ok:
            return []
        rc, out, _ = self._run(["lsf", "-R", "--files-only", self._p(prefix)])
        if rc != 0:
            return []
        return sorted(f"{prefix.rstrip('/')}/{ln.strip()}"
                      for ln in (out or "").splitlines()
                      if ln.strip().endswith(".json"))

    def dirs(self, prefix):
        """Immediate subdirectories only. On S3 that is one delimiter-scoped
        request — which is why the box tier is enumerated as
        `dirs(jobs/nodes) x keys(<iid>/hostfacts)` and never as a recursive
        listing of `jobs/nodes/`: that prefix also holds every box's whole
        lifecycle event stream, thousands of objects we do not want."""
        if not self.ok:
            return []
        rc, out, _ = self._run(["lsf", "--dirs-only", self._p(prefix)])
        if rc != 0:
            return []
        return sorted(ln.strip().rstrip("/") for ln in (out or "").splitlines()
                      if ln.strip())

    def get(self, key):
        if not self.ok:
            return None
        rc, out, _ = self._run(["cat", self._p(key)])
        # rclone cat on a MISSING S3 object exits 0 with empty stdout — rc alone
        # cannot tell present from absent (boxstate.B2.cat documents the same).
        if rc != 0 or not (out or "").strip():
            return None
        try:
            return json.loads(out)
        except ValueError:
            return None

    def put(self, key, blob):
        """`rclone rcat` streams a PUT (no HeadObject 403 flake) — the same
        primitive `_stage_jobd_bootstrap` uses for an immutable object."""
        if not self.ok:
            return False
        try:
            r = subprocess.run(["rclone", "rcat", self._p(key)],
                               input=json.dumps(blob, indent=2), text=True,
                               capture_output=True)
            return r.returncode == 0
        except FileNotFoundError:
            return False


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
#: Bounded concurrency for store reads. Every one is an `rclone` subprocess and
#: the enumeration is O(boxes ever rented) of them — serially that is minutes,
#: which is why this scorecard went 17 days without being run. Threads, not
#: processes: the work is entirely waiting on a child.
READ_WORKERS = max(1, int(os.environ.get("HOSTFACTS_READ_WORKERS") or 16))


def _map_reads(fn, items, workers=None):
    """`fn` over `items`, RESULTS IN ORDER, with bounded concurrency.

    In order because the caller's determinism comes from sorted keys, and a
    scorecard whose rows move between runs is one nobody can diff.
    """
    items = list(items)
    n = min(workers or READ_WORKERS, len(items))
    if n <= 1:
        return [fn(x) for x in items]
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(fn, items))


def node_keys(store):
    """Every box-written record key, without recursively listing `jobs/nodes/`
    (which also holds every box's lifecycle event stream)."""
    out = []
    for keys in _map_reads(lambda iid: store.keys(instance_prefix(iid)),
                           store.dirs(NODES)):
        out.extend(keys)
    return sorted(out)


def load_records(store):
    """Every record in the store, each tagged with the key it came from.

    The PINNED tier is read first so a pinned record wins the de-duplication
    below: it is the same measurement with a resolved machine attached.
    """
    seen, out = set(), []
    for keys in (store.keys(BY_MACHINE), node_keys(store)):
        for key, rec in zip(keys, _map_reads(store.get, keys)):
            if not isinstance(rec, dict):
                continue
            rec = dict(rec)
            rec.setdefault("instance_id", _instance_from_key(key) or "")
            m = _machine_from_key(key)
            if m:
                rec["machine_id"] = m
            rec["_key"] = key
            # (instance, ts) ALONE also ate the second measurement: cpu_probe
            # drops `pyops` and `compile_tu` back to back and `ts` is
            # second-resolution, so one of the pair was discarded here — 86
            # records, measured 2026-08-27. Kind and units are what tell two
            # measurements apart; a pinned copy and its node-tier original still
            # agree on all four, which is the shadowing this dedup is for.
            ident = (str(rec.get("instance_id") or ""), str(rec.get("ts") or ""),
                     kind_of(rec), str(rec.get("units") or ""))
            if ident in seen:
                continue
            seen.add(ident)
            out.append(rec)
    return out


def kind_of(rec):
    """Which fact this is. Absent reads as `gemm`: every record written before
    kinds existed is one, and each cost a rental to produce."""
    k = str((rec or {}).get("kind") or "").strip()
    return k if k in KINDS else KIND_GEMM


def of_kind(records, kind):
    """Records of one kind. The scorecards MUST filter through this rather than
    rely on a shape test — a cpu record has no `ceiling_tflops`, so the GEMM
    rollup would not crash on one, it would quietly count it as a host with
    zero quotable measurements and understate that machine."""
    return [r for r in (records or []) if kind_of(r) == kind]


def quotable(rec):
    """Does this record carry a ceiling anyone may quote?

    `gemm_ceiling.py`'s rule, enforced at read time as well as at write time:
    a TFLOP/s figure with no device attached is not quotable.
    """
    return bool(rec.get("ceiling_tflops")) and bool((rec.get("device") or "").strip())


def weighted_ceiling(rec, mac_mix):
    """FLOP-weighted harmonic mean of this record's per-class rates, or None.

    Delegates to `mfu.harmonic_weighted`/`mfu.classify_gemm` so there is ONE
    implementation of the weighting and one place the `lm_head -> mlp_up`
    substitution is stated.
    """
    import mfu                                            # noqa: PLC0415
    if not quotable(rec):
        return None
    try:
        return mfu.Ceiling.from_gemm_ceiling_json(rec, weights=mac_mix).tflops
    except mfu.DenominatorError:
        return None


def group_by_host(records):
    """{host_key: [records]} — machine id where known, `iid:<IID>` otherwise.

    An unresolved record is NOT dropped and NOT merged into a machine it might
    belong to: it groups under its own instance, so the scorecard shows both
    "machine 140799 benches at X" and "these three rentals were never resolved".
    """
    groups = {}
    for r in records:
        m = str(r.get("machine_id") or "").strip()
        key = m if m else f"iid:{r.get('instance_id') or '?'}"
        groups.setdefault(key, []).append(r)
    return groups


def summarize(records):
    """Per-host rollup + the fleet median, ready to print or serialise.

    The **fleet median is taken over HOSTS, not over records** — three rentals of
    one fast machine must not drag the median that every other host is compared
    against. `ratio` is host/median, so 1.00 is typical and 0.47 is the box that
    cost $1.47 and 2.5 h in PERF_LEVERS_INVESTIGATION §2.
    """
    hosts = []
    records = of_kind(records, KIND_GEMM)
    for key, recs in sorted(group_by_host(records).items()):
        ok = [r for r in recs if quotable(r)]
        best = max((r["ceiling_tflops"] for r in ok), default=None)
        devices = sorted({(r.get("device") or "").strip() for r in ok
                          if (r.get("device") or "").strip()})
        plims = sorted({r["power_limit_w"] for r in ok if r.get("power_limit_w")})
        hosts.append({
            "host": key,
            "resolved": not key.startswith("iid:"),
            "n_records": len(recs),
            "n_quotable": len(ok),
            "ceiling_tflops": best,
            "min_tflops": min((r["min_tflops"] for r in ok
                               if r.get("min_tflops") is not None), default=None),
            "devices": devices,
            "power_limit_w": plims,
            "instances": sorted({str(r.get("instance_id") or "?") for r in recs}),
            "last_ts": max((str(r.get("ts") or "") for r in recs), default=""),
            "statuses": sorted({str(r.get("status") or "?") for r in recs}),
        })
    ceilings = [h["ceiling_tflops"] for h in hosts if h["ceiling_tflops"]]
    median = statistics.median(ceilings) if ceilings else None
    for h in hosts:
        h["ratio_to_median"] = (round(h["ceiling_tflops"] / median, 3)
                                if median and h["ceiling_tflops"] else None)
    hosts.sort(key=lambda h: (-(h["ceiling_tflops"] or 0), h["host"]))
    return {"hosts": hosts, "fleet_median_tflops": median,
            "n_hosts": len(hosts), "n_hosts_quotable": len(ceilings),
            "spread": (round(max(ceilings) / min(ceilings), 2)
                       if len(ceilings) > 1 else None)}


# --------------------------------------------------------------------------- #
# cpu facts: what a machine's CORES are worth, measured
# --------------------------------------------------------------------------- #
def cpu_record(instance_id, ts, *, units, count, wall_s,
               cores=None, cpu_name=None, workload=None, **extra):
    """A `cpu` hostfact: `count` units of work in `wall_s` seconds.

    HARVESTED **OR** BENCHMARKED — amended 2026-08-25, see below.

    The original rule read: harvested, not benchmarked. A synthetic CPU
    benchmark would measure a machine on work we do not run; the boxes this
    exists for already compile thousands of translation units, so the honest
    throughput number is the one the real job produced. That also makes it free
    — no rental is spent on measurement — which is the same argument
    `ingest_measured_throughput.py` makes for reading banked artifacts rather
    than hand-maintaining a table.

    That is right about fidelity and was wrong about coverage. Harvested
    producers are bundle-scoped, so they only fire when their bundle runs:
    measured 2026-08-24 this store held **0 cpu records against 202 gemm**, the
    difference being that gemm's producer rides jobd and runs on every box.
    A rule that yields no distribution cannot inform a selection.

    There is also a narrower reason a fixed workload is the RIGHT instrument
    here rather than a compromise. `per_core_s` below is defended as the
    cross-machine comparand, and that only holds if the machines ran the same
    work — real compiles differ per job, so harvested rates compare across
    machines only by accident. For IPC specifically, fixed work is the
    measurement.

    So both are admitted and they never merge: `cpu_probe.py` emits
    `units="pyops"` on every box, harvested producers keep their own units, and
    `summarize_cpu` groups by unit and refuses to average across them.

    `units` names what was counted (e.g. `"tu"` for translation units) and is
    NOT optional: a bare rate with no unit is the thing that gets quoted years
    later against a different workload. `per_s` and `per_core_s` are derived
    here so every consumer divides the same way; `per_core_s` is the one that
    compares across machines, since a wide slow box and a narrow fast one are
    only comparable per core.

    Deliberately NOT computed: any accept/reject verdict. See this module's
    "WHAT THIS MODULE DOES NOT DO" — a threshold needs a distribution, and
    this is the thing that produces it.
    """
    if not units:
        raise ValueError("cpu_record: `units` is required — an unlabelled rate "
                         "is unquotable")
    if wall_s is None or float(wall_s) <= 0:
        raise ValueError(f"cpu_record: wall_s must be > 0 (got {wall_s!r})")
    count, wall_s = float(count), float(wall_s)
    rec = {"probe_version": 1, "kind": KIND_CPU, "ts": str(ts),
           "instance_id": str(instance_id), "units": str(units),
           "count": count, "wall_s": wall_s,
           "per_s": round(count / wall_s, 4)}
    if cores:
        rec["cores"] = float(cores)
        rec["per_core_s"] = round(count / wall_s / float(cores), 5)
    if cpu_name:
        rec["cpu_name"] = str(cpu_name)
    if workload:
        rec["workload"] = str(workload)
    rec.update(extra)
    return rec


# --------------------------------------------------------------------------- #
# the box-side DROP DIR — how a producer hands a record to jobd
# --------------------------------------------------------------------------- #
# A harvested fact cannot be written the way the GEMM ceiling is. That one is a
# BENCHMARK jobd runs itself at boot, so jobd owns the file and uploads it in
# the same function. A harvested rate is produced by the JOB, mid-run, in a
# process jobd only knows the pid of — and the job cannot upload it itself
# without every bundle re-implementing the B2 key convention, the scoped-key
# rule and the immutable-object pattern (three things that are one-line wrong
# in an obvious way and 403 or silently overwrite).
#
# So: the producer drops a file in a directory, and jobd drains it. The
# contract is only the DIRECTORY and the FILENAME — no imports across the
# seam, so a bundle that cannot import this module can still take part with a
# `json.dump` and a mkdir.
DROP_DIR_ENV = "JOBD_HOSTFACTS_DROP"
DEFAULT_DROP_DIR = "/workspace/hostfacts.d"


def drop_dir():
    """Where a producer leaves records for jobd to upload.

    Under `$JOBD_ROOT` (not `/tmp`) because the drain is asynchronous: a record
    written to a container-local tmp that a restart clears is a measurement
    that cost real work and vanished before anyone read it."""
    d = os.environ.get(DROP_DIR_ENV)
    if d:
        return d
    root = os.environ.get("JOBD_ROOT")
    return os.path.join(root, "hostfacts.d") if root else DEFAULT_DROP_DIR


def drop_record(rec, *, directory=None, ts=None):
    """Write `rec` into the drop dir as `<kind>-<ts>.json`. Returns the path.

    ATOMIC by tmp-then-rename: jobd's drain runs on a timer and would otherwise
    happily upload a half-written file, which is unrecoverable — the object is
    immutable once PUT and the corrupted record outlives the run that made it.

    The filename carries the kind because that is how `ingest` decides what a
    record IS (`_kind_from_key`), and the ts because one immutable object per
    measurement is the whole storage model."""
    d = directory or drop_dir()
    os.makedirs(d, exist_ok=True)
    kind = _slug(rec.get("kind") or KIND_CPU)
    stamp = _slug(ts or rec.get("ts") or "")
    path = os.path.join(d, f"{kind}-{stamp}.json")
    # `ts` is second-resolution and two measurements in one second is a normal
    # case: cpu_probe drops a `pyops` and a `compile_tu` record back to back.
    # Without this the second SILENTLY REPLACES the first — measured 2026-08-25,
    # where it destroyed the pyops record, i.e. the universal one. The suffix
    # goes after the stamp so `_kind_from_key` (leading token) is unaffected.
    n = 1
    while os.path.exists(path):
        n += 1
        path = os.path.join(d, f"{kind}-{stamp}-{n}.json")
    tmp = path + ".partial"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return path


def drop_cpu_record(*, units, count, wall_s, instance_id=None, ts=None,
                    directory=None, **kw):
    """`cpu_record` + `drop_record` — the whole producer-side API.

    Exists so a producer never hand-rolls the record shape: `cpu_record`
    derives `per_s`/`per_core_s` so every consumer divides the same way, and a
    bundle computing its own rate is how two definitions of "per core" end up
    in one scorecard.

    `instance_id` defaults to what the container was told about itself —
    "inherit, never invent". A box that cannot name itself still drops the
    record: jobd's drain re-keys on `$IID` anyway, so an `unknown` here is
    recoverable, while refusing to write would lose the measurement."""
    iid = instance_id or os.environ.get("INSTANCE_ID") \
        or os.environ.get("CONTAINER_ID") or "unknown"
    stamp = ts or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec = cpu_record(iid, stamp, units=units, count=count, wall_s=wall_s, **kw)
    return drop_record(rec, directory=directory, ts=stamp)


def summarize_cpu(records):
    """Per-host CPU throughput rollup, the `cpu` mirror of `summarize`.

    Rolls up PER UNIT: two records counting different things are two rows, never
    one averaged row. Mixing translation units with anything else would produce
    a number with no referent, which is the failure `cpu_record` refuses at the
    write side by making `units` mandatory.

    `per_core_s` is the cross-machine comparand — a wide slow box and a narrow
    fast one are only comparable per core — and the median is over HOSTS, for
    the same reason `summarize` gives: three rentals of one machine must not
    drag the figure every other host is measured against.
    """
    rows = []
    for key, recs in sorted(group_by_host(of_kind(records, KIND_CPU)).items()):
        by_units = {}
        for r in recs:
            by_units.setdefault(str(r.get("units") or "?"), []).append(r)
        for units, rs in sorted(by_units.items()):
            per_core = [r["per_core_s"] for r in rs if r.get("per_core_s")]
            rows.append({
                "host": key,
                "resolved": not key.startswith("iid:"),
                "units": units,
                "n_records": len(rs),
                "best_per_s": max((r["per_s"] for r in rs if r.get("per_s")),
                                  default=None),
                "best_per_core_s": max(per_core, default=None),
                "cpu_names": sorted({(r.get("cpu_name") or "").strip()
                                     for r in rs
                                     if (r.get("cpu_name") or "").strip()}),
            })
    # The rollup above keeps units apart per ROW; the fleet figures have to do
    # the same or they average a pyops rate against a merge_tensors one. That
    # is not hypothetical — the first real probe data read `spread 5.7e8x`.
    by_units = {}
    for u in sorted({r["units"] for r in rows}):
        vals = sorted(r["best_per_core_s"] for r in rows
                      if r["units"] == u and r["best_per_core_s"])
        if not vals:
            continue
        by_units[u] = {
            "fleet_median_per_core_s": statistics.median(vals),
            "spread": (round(max(vals) / min(vals), 2)
                       if len(vals) > 1 else None),
            "n_hosts": len({r["host"] for r in rows if r["units"] == u}),
        }
    # A cross-unit median has no referent, so it exists only when the fleet
    # speaks one unit. Mixed reads None — never a blended number a caller could
    # mistake for a comparand.
    only = next(iter(by_units)) if len(by_units) == 1 else None
    return {"hosts": rows, "n_hosts": len({r["host"] for r in rows}),
            "by_units": by_units,
            "fleet_median_per_core_s":
                by_units[only]["fleet_median_per_core_s"] if only else None,
            "spread": by_units[only]["spread"] if only else None}


# --------------------------------------------------------------------------- #
# ingest: resolve instance -> machine while the mapping still exists
# --------------------------------------------------------------------------- #
def ingest(store, resolve_machine, *, dry_run=False):
    """Pin every unresolved by-instance record to `by-machine/`.

    `resolve_machine(instance_id) -> machine_id | None`. Unresolvable records are
    LEFT ALONE (still readable by instance) — this is a promotion, never a move,
    so a failed resolution can never lose a measurement.

    Returns {"pinned": [...], "unresolved": [...], "already": n}.
    """
    pinned, unresolved, already = [], [], 0
    existing = set(store.keys(BY_MACHINE))
    keys = node_keys(store)
    for key, rec in zip(keys, _map_reads(store.get, keys)):
        if not isinstance(rec, dict):
            continue
        iid = str(rec.get("instance_id") or _instance_from_key(key) or "")
        ts = str(rec.get("ts") or "")
        mid = rec.get("machine_id") or resolve_machine(iid)
        if not mid:
            unresolved.append(key)
            continue
        # Kind rides across from the SOURCE key, so ingest stays kind-agnostic:
        # a cpu record pins as a cpu record without this function knowing what
        # is inside it.
        # The `-N` a same-second pair already carries on the box rides across
        # too, or both halves land on one key and the second put DESTROYS the
        # first (`_dup_suffix_from_key`). Absent a suffix the name is unchanged,
        # so the objects already pinned stay where they are.
        dest = machine_key(mid, iid, ts + _dup_suffix_from_key(key),
                           _kind_from_key(key))
        if dest in existing:
            already += 1
            continue
        out = dict(rec)
        out["machine_id"] = str(mid)
        out["instance_id"] = iid
        out["pinned_from"] = key
        if not dry_run:
            store.put(dest, out)
        # `existing` was a snapshot, so without this a second source mapping to
        # this name would overwrite what we just wrote rather than read as
        # already-pinned.
        existing.add(dest)
        pinned.append(dest)
    return {"pinned": pinned, "unresolved": unresolved, "already": already}


def vast_machine_resolver():
    """instance_id -> machine_id from the live vast API, resolved ONCE.

    Imported lazily: `herdd` wants credentials, and every other entry point in
    this module works offline. A resolver that cannot reach the API returns None
    for everything, which `ingest` treats as "leave it unresolved".
    """
    table = {}
    try:
        import herdd                                    # noqa: PLC0415
        herdd.load_env()
        # _instances_soft, not _instances: an API blip must leave records
        # unresolved, never sys.exit out of an ingest that already pinned some.
        for inst in (herdd._instances_soft() or []):
            iid, mid = inst.get("id"), inst.get("machine_id")
            if iid and mid:
                table[str(iid)] = str(mid)
    except Exception as e:                                # noqa: BLE001
        print(f"~~ vast API unreachable ({type(e).__name__}: {e}) — "
              f"records stay unresolved, nothing is lost", file=sys.stderr)
    return lambda iid: table.get(str(iid))


#: Mirrors `vastlib.core.machine_ledger`. Spelled here rather than imported
#: because this module is a Zone S flat leaf shipped in the jobd bundle and must
#: import bare-name under `python3 -P`. `test_machine_ledger.py` pins the two
#: equal so the duplication cannot drift.
LEDGER_PATH_ENV = "VAST_MACHINE_LEDGER_PATH"
LEDGER_STATE_ENV = "FLEETD_STATE_DIR"
LEDGER_DEFAULT_DIR = "~/.local/state/vast-fleetd"
LEDGER_FILENAME = "machine_ledger.json"


def ledger_path():
    """Where fleetd writes the instance -> machine ledger."""
    override = os.environ.get(LEDGER_PATH_ENV)
    if override:
        return os.path.expanduser(override)
    d = os.environ.get(LEDGER_STATE_ENV) or LEDGER_DEFAULT_DIR
    return os.path.join(os.path.expanduser(d), LEDGER_FILENAME)


def ledger_machine_resolver(path=None):
    """instance_id -> machine_id from fleetd's ledger, which does NOT expire.

    The live API answers only for boxes that still exist, so on its own it can
    never resolve a record written by a box that is gone — measured 2026-08-24
    at 3 resolvable of 202. fleetd sees `machine_id` on every instance every
    tick and writes it down, so this source survives the box.

    A CONFLICTED entry (one instance id that has named two machines) resolves to
    None: an ambiguous attribution is worse than an absent one, because the
    record would be filed under a machine that may not have produced it.
    """
    try:
        with open(path or ledger_path()) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return lambda iid: None
    if not isinstance(data, dict):
        return lambda iid: None

    def _resolve(iid):
        entry = data.get(str(iid))
        if not isinstance(entry, dict) or entry.get("conflicts"):
            return None
        mid = entry.get("machine_id")
        return str(mid) if mid else None
    return _resolve


def identity_machine_resolver(store):
    """instance_id -> machine_id from the `identity-*.json` the LAUNCHER wrote.

    `boxes.lifecycle.record_box_identity_soft` files one under the box's own
    `jobs/nodes/<IID>/` at rent time. This is the copy that survives the laptop:
    it needs no local state and no API, so a bare copy of the bucket resolves.

    Looked up per instance and only for records the earlier resolvers missed —
    listing every box's prefix up front would cost one call per box to answer a
    question most of them have already answered.
    """
    def _resolve(iid):
        try:
            keys = [k for k in store.keys(f"{NODES}/{_slug(iid)}/")
                    if "/identity-" in k]
        except Exception:                                 # noqa: BLE001
            return None
        for key in sorted(keys, reverse=True):            # newest first
            rec = store.get(key)
            if isinstance(rec, dict) and rec.get("machine_id"):
                return str(rec["machine_id"])
        return None
    return _resolve


def chained_resolver(*resolvers):
    """First non-None wins. Order is precedence, and the caller sets it.

    `ingest` already prefers a `machine_id` carried on the record itself; this
    composes what to try after that.
    """
    def _resolve(iid):
        for r in resolvers:
            if r is None:
                continue
            try:
                mid = r(iid)
            except Exception:                             # noqa: BLE001
                continue                                  # a dead source is not fatal
            if mid:
                return mid
        return None
    return _resolve


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render_table(summary):
    med = summary.get("fleet_median_tflops")
    lines = [f"{'host':<14} {'TFLOP/s':>8} {'vs med':>7} {'plim W':>7} "
             f"{'n':>3}  device / status"]
    for h in summary["hosts"]:
        tf = f"{h['ceiling_tflops']:.1f}" if h["ceiling_tflops"] else "-"
        ratio = f"{h['ratio_to_median']:.2f}x" if h["ratio_to_median"] else "-"
        plim = "/".join(str(int(p)) for p in h["power_limit_w"]) or "-"
        what = ", ".join(h["devices"]) or ", ".join(h["statuses"])
        lines.append(f"{h['host']:<14} {tf:>8} {ratio:>7} {plim:>7} "
                     f"{h['n_records']:>3}  {what}")
    lines.append("")
    if med:
        lines.append(f"fleet median {med:.1f} TFLOP/s over "
                     f"{summary['n_hosts_quotable']} host(s)"
                     + (f"; spread {summary['spread']}x" if summary["spread"]
                        else ""))
    else:
        lines.append("no quotable ceiling yet — every record is a skip, a "
                     "failure, or has no device name attached")
    lines.append("NOTE: observation only. Nothing selects, rejects or re-rents "
                 "on these numbers (HOST_ACCEPTANCE_PROBE_2026-08-07.md §5).")
    return "\n".join(lines)


def render_cpu_table(summary):
    """`render_table`'s cpu mirror. One row per (host, units) — see
    `summarize_cpu` for why units never collapse into one another."""
    by_units = summary.get("by_units") or {}
    lines = [f"{'host':<14} {'units':<14} {'per s':>10} {'per core/s':>11} "
             f"{'vs med':>7} {'n':>3}  cpu"]
    for h in summary["hosts"]:
        ps = f"{h['best_per_s']:.3g}" if h["best_per_s"] else "-"
        pcs = f"{h['best_per_core_s']:.4g}" if h["best_per_core_s"] else "-"
        # against this row's OWN unit median: the previous cross-unit ratio
        # printed 0.00x for every row of the smaller-numbered unit.
        umed = (by_units.get(h["units"]) or {}).get("fleet_median_per_core_s")
        ratio = (f"{h['best_per_core_s'] / umed:.2f}x"
                 if (umed and h["best_per_core_s"]) else "-")
        lines.append(f"{h['host']:<14} {h['units']:<14} {ps:>10} {pcs:>11} "
                     f"{ratio:>7} {h['n_records']:>3}  "
                     f"{', '.join(h['cpu_names']) or '-'}")
    lines.append("")
    if by_units:
        for u, s in by_units.items():
            lines.append(f"fleet median {s['fleet_median_per_core_s']:.4g} "
                         f"{u}/core/s over {s['n_hosts']} host(s)"
                         + (f"; spread {s['spread']}x" if s["spread"] else ""))
    else:
        lines.append("no cpu records yet — these are HARVESTED from real job "
                     "work, so a host appears once it has run some")
    # Was "observation only ... nothing selects or rejects on these numbers",
    # true from 2026-08-07 until the distribution existed. It does now, so the
    # line would be a false reassurance about a table that spends money.
    lines.append("NOTE: these numbers SELECT. `hostfacts.py calibrate --write` "
                 "freezes them into cpu_calibration.json (owner decision "
                 "2026-08-27). The two arms above are used for DIFFERENT "
                 f"questions and are not interchangeable: {CALIBRATION_ARMS[0]} "
                 "ranks CPU-shaped offers (all-core, so it may be multiplied by "
                 f"a slice width), while {CALIBRATION_ARMS[1]} -- a SERIAL "
                 "single-compile rate -- is what market.offers."
                 "CPU_PERF_FLOOR_RATIO multiplies to get the floor. Only "
                 "MEASURED offers can be refused; unmeasured ones rank last. "
                 "Re-run calibrate after banking new hosts.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# calibration: the scorecard, frozen into something the OFFER LANE can read
# --------------------------------------------------------------------------- #

#: Where `calibrate --write` puts the table. Tracked in git and read offline by
#: `market.offers`: a search must not wait on B2, and this file is the only
#: thing standing between a purchase decision and `cores x GHz`.
CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "cpu_calibration.json")

#: The default single-arm comparand, kept for `calibrate --units`. `pyops` is
#: the ALL-CORE arm, run on every box with no toolchain needed.
CALIBRATION_UNITS = "pyops"

#: Serial-by-construction units, whose `per_core_s` is not a per-core anything.
#: For these the table carries `per_s` — the rate as measured, on one thread.
SERIAL_UNITS = ("compile_tu",)

#: Both arms ship, because they answer different questions and each is right
#: for exactly one of them (2026-08-28, measured over 60 machines carrying
#: both):
#:
#:   pyops       all-core, so `per_core_s` already embeds the box's scaling
#:               losses. Multiply by the slice width and you get a THROUGHPUT
#:               estimate that has paid for contention once.
#:   compile_tu  a serial loop of real compiles, i.e. a direct single-compile
#:               LATENCY measurement of the workload our CPU lanes actually
#:               run. It cannot be scaled to a width — nothing measured it
#:               under contention.
#:
#: They rank the fleet differently (Spearman 0.673) and the disagreement is
#: systematic, not noise: median pyops/compile ratio is 0.68 on desktop/HEDT
#: parts against 1.16 on server parts, so pyops flatters low-clock many-core
#: silicon exactly where a serial compile suffers most. Collapsing them to one
#: number would hide that, so the table carries both and each consumer names
#: the arm it wants.
CALIBRATION_ARMS = ("pyops", "compile_tu")


def _rate_of(row):
    """The comparable rate for a summarize_cpu row.

    All-core units divide by the width and give a per-core rate. A SERIAL unit
    cannot: `bench_compile` runs one compile at a time, so dividing by the width
    would answer "single thread over 128 threads", which ranks narrow boxes best.
    Records written before 2026-08-27 carry the width for that arm; `per_s` is
    era-independent and was always what the loop measured.
    """
    if row["units"] in SERIAL_UNITS:
        return row["best_per_s"]
    return row["best_per_core_s"]


def calibration(records, units=CALIBRATION_UNITS, generated=None):
    """Measured CPU rates keyed the two ways an OFFER can be joined to them.

    A vast offer carries both `machine_id` and `cpu_name`, and the `cpu_name` is
    byte-identical to the string the probe banks — so this is a string join, not
    an inferred IPC table, and the tiers are exactly as strong as their names:

      by_machine  this exact machine, measured. No extrapolation.
      by_model    other machines of the same CPU model. Carries `n_machines`
                  and `spread`, because generalisation across a model is only
                  as good as that spread and the caller must be able to see it.

    There is deliberately no third, family tier. Within a generation the bands
    are tight (EPYC Rome measured 1.11x across five parts, Milan 1.24x) but the
    9xx4 line spans 2.58x, because dense Zen4c parts (9754, 9B14) share a family
    with high-frequency ones (9374F). A family estimate there would be worse
    than admitting ignorance, and an unmeasured offer is ranked last rather than
    guessed at.

    `by_machine` needs machine grain and so takes resolved hosts only; a model
    estimate does not need identity, so `by_model` takes every host that named
    its CPU. Rates come from `summarize_cpu` rather than a second derivation —
    one definition of "per core", per this module's own write-side rule.
    """
    s = summarize_cpu(records)
    rows = [r for r in s["hosts"]
            if r["units"] == units and _rate_of(r) and len(r["cpu_names"]) == 1]
    by_machine, by_model = {}, {}
    for r in rows:
        rate, name = _rate_of(r), r["cpu_names"][0]
        if r["resolved"]:
            by_machine[r["host"]] = {"rate": rate, "n_records": r["n_records"],
                                     "cpu_name": name}
        by_model.setdefault(name, []).append(rate)
    models = {}
    for name, vals in sorted(by_model.items()):
        vals = sorted(vals)
        models[name] = {
            "rate": statistics.median(vals),
            "n_machines": len(vals),
            "spread": round(vals[-1] / vals[0], 3) if len(vals) > 1 else None,
        }
    # Over the SAME rate the tiers use, not `summarize_cpu`'s per-core median —
    # for a serial unit those are different numbers and the floor reads this one.
    fleet_vals = sorted(_rate_of(r) for r in rows)
    return {
        "units": units,
        "rate_is": "per_s" if units in SERIAL_UNITS else "per_core_s",
        "generated": generated or time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime()),
        "n_machines": len(by_machine),
        "n_models": len(models),
        # The floor in `market.offers` is a RATIO of this, so it re-calibrates
        # itself if the kernel or the fleet changes rather than pinning a
        # constant to one probe version.
        "fleet_median": statistics.median(fleet_vals) if fleet_vals else None,
        "fleet_spread": (round(fleet_vals[-1] / fleet_vals[0], 3)
                         if len(fleet_vals) > 1 and fleet_vals[0] else None),
        "by_machine": dict(sorted(by_machine.items())),
        "by_model": models,
    }


def calibration_table(records, arms=CALIBRATION_ARMS, generated=None):
    """The tracked multi-arm table: one `calibration()` per arm, side by side.

    Arms are kept SEPARATE rather than blended. See `CALIBRATION_ARMS` for why
    — they measure different things, disagree systematically, and a consumer
    that cannot say which one it asked for is not making a measurement claim.

    An arm with no measured host is omitted rather than written empty, so
    `arms` reads as "what is actually measured" and a consumer's fallback path
    is exercised by a real absence instead of a zero-row table.
    """
    generated = generated or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    built = {}
    for units in arms:
        arm = calibration(records, units=units, generated=generated)
        if arm["by_machine"] or arm["by_model"]:
            built[units] = arm
    return {"schema": 2, "generated": generated, "arms": built}


def calibration_arm(blob, arm):
    """One arm's sub-table out of `blob`, or None.

    Accepts a schema-1 table (the single-arm shape, arms unnamed) and returns it
    for ANY requested arm: that table is `pyops`, and before this schema existed
    every consumer read it for every purpose. Reading it as the arm asked for is
    the same answer those consumers already got, so an old table degrades to old
    behaviour rather than to no behaviour.
    """
    if not isinstance(blob, dict):
        return None
    arms = blob.get("arms")
    if isinstance(arms, dict):
        hit = arms.get(arm)
        return hit if isinstance(hit, dict) and hit.get("by_machine") else None
    return blob if blob.get("by_machine") else None


def load_calibration(path=None):
    """The tracked table, or None. Never raises: an absent or corrupt table
    means "nothing is measured", which every caller already handles — it is the
    same state as an offer we have never seen.

    Returns the whole blob, either schema. Pick an arm with `calibration_arm`.
    """
    try:
        with open(path or CALIBRATION_PATH) as fh:
            blob = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(blob, dict):
        return None
    ok = blob.get("by_machine") or any(
        isinstance(v, dict) and v.get("by_machine")
        for v in (blob.get("arms") or {}).values())
    return blob if ok else None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _store(a):
    if a.local:
        return LocalStore(a.local)
    # B2Store reads B2_BUCKET from os.environ at construction, and its "not set
    # (env or .env)" message promises a .env consult that nothing performed —
    # so every non-interactive caller (the systemd ingest timer above all) got
    # a store that was ok=False for a credential sitting in .env all along.
    # Lazy and B2-only, to keep the offline entry points import-free.
    try:
        import herdd                                    # noqa: PLC0415
        herdd.load_env()
    except Exception:                                     # noqa: BLE001, S110
        pass                                              # explicit env still works
    return B2Store()


def _pick(records, ident):
    ident = str(ident)
    hit = [r for r in records
           if str(r.get("machine_id") or "") == ident
           or str(r.get("instance_id") or "") == ident]
    return sorted(hit, key=lambda r: str(r.get("ts") or ""))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", default="",
                    help="read/write a local directory with the same key "
                         "layout instead of B2 (offline)")
    sub = ap.add_subparsers(dest="cmd")

    pl = sub.add_parser("list", help="per-host scorecard + fleet median")
    pl.add_argument("--json", action="store_true")
    # Default gemm, so the command every runbook already names is unchanged.
    pl.add_argument("--kind", choices=list(KINDS), default=KIND_GEMM,
                    help="which fact to score: gemm (dense-bf16 ceiling) or "
                         "cpu (harvested work-rate). Separate scorecards, never "
                         "one table — they share no unit.")

    ps = sub.add_parser("show", help="every record for a machine or instance id")
    ps.add_argument("ident")

    pc = sub.add_parser("ceiling",
                        help="emit the newest quotable record for a host as a "
                             "gemm_ceiling-shaped blob (feed mfu.py "
                             "--ceiling-json)")
    pc.add_argument("--machine", default="")
    pc.add_argument("--instance", default="")
    pc.add_argument("--json", action="store_true")

    pi = sub.add_parser("ingest",
                        help="resolve live instances -> by-machine/ and pin")
    pi.add_argument("--dry-run", action="store_true")

    pk = sub.add_parser("calibrate",
                        help="freeze the cpu scorecard into the table the "
                             "offer lane reads (tools/vast/cpu_calibration.json)")
    pk.add_argument("--write", action="store_true",
                    help="write the table instead of printing it")
    pk.add_argument("--units", default="",
                    help="calibrate ONE arm and emit the single-arm shape, for "
                         "inspection (default: every arm in "
                         f"{'/'.join(CALIBRATION_ARMS)})")
    pk.add_argument("--path", default="",
                    help="destination (default: beside this script)")

    a = ap.parse_args(argv)
    cmd = a.cmd or "list"
    store = _store(a)
    if not getattr(store, "ok", True):
        print(f"!! host-facts store unavailable: {store.reason}", file=sys.stderr)
        return 2

    if cmd == "ingest":
        # Ledger FIRST: it is a written-down fact that outlives the box, while
        # the API answers only for boxes that still exist. Trying the API first
        # would still be correct but would make a live box's answer depend on
        # the network for a mapping already on disk.
        resolver = chained_resolver(ledger_machine_resolver(),
                                    identity_machine_resolver(store),
                                    vast_machine_resolver())
        res = ingest(store, resolver, dry_run=a.dry_run)
        print(json.dumps({"pinned": len(res["pinned"]),
                          "already_pinned": res["already"],
                          "unresolved": len(res["unresolved"]),
                          "dry_run": bool(a.dry_run)}, indent=2))
        if res["unresolved"]:
            print("~~ unresolved (not in the ledger and gone from the API — "
                  "still readable by instance):\n   "
                  + "\n   ".join(res["unresolved"][:10]), file=sys.stderr)
        return 0

    records = load_records(store)
    if cmd == "calibrate":
        if a.units:
            table = calibration(records, units=a.units)
            arms = {a.units: table} if table["n_machines"] else {}
        else:
            table = calibration_table(records)
            arms = table["arms"]
        if not arms:
            want = a.units or "/".join(CALIBRATION_ARMS)
            print(f"!! nothing measured in {want} — no table written",
                  file=sys.stderr)
            return 1
        blob = json.dumps(table, indent=1, sort_keys=True) + "\n"
        if not a.write:
            print(blob, end="")
            return 0
        dest = a.path or CALIBRATION_PATH
        with open(dest, "w") as fh:
            fh.write(blob)
        print(f"wrote {dest}:")
        for units, arm in sorted(arms.items()):
            print(f"  {units:<11} {arm['n_machines']:>3} machines, "
                  f"{arm['n_models']:>3} models, median "
                  f"{arm['fleet_median']:.4g} ({arm['rate_is']}), "
                  f"spread {arm['fleet_spread']}x")
        return 0
    if cmd == "list":
        if a.kind == KIND_CPU:
            s = summarize_cpu(records)
            print(json.dumps(s, indent=2) if a.json else render_cpu_table(s))
            return 0
        s = summarize(records)
        print(json.dumps(s, indent=2) if a.json else render_table(s))
        return 0
    if cmd == "show":
        hit = _pick(records, a.ident)
        if not hit:
            print(f"!! no host-facts record for {a.ident!r}", file=sys.stderr)
            return 1
        print(json.dumps(hit, indent=2))
        return 0
    if cmd == "ceiling":
        ident = a.machine or a.instance
        if not ident:
            ap.error("ceiling needs --machine or --instance")
        hit = [r for r in _pick(records, ident) if quotable(r)]
        if not hit:
            print(f"!! no quotable ceiling for {ident!r} — run gemm_probe.py on "
                  f"that box, or check `hostfacts.py show {ident}` for why every "
                  f"record was skipped", file=sys.stderr)
            return 1
        rec = hit[-1]
        blob = {k: v for k, v in rec.items() if not k.startswith("_")}
        print(json.dumps(blob, indent=2))
        return 0
    ap.error(f"unknown command {cmd!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
