"""vastlib.jobs.scan — fold MANY job event logs in a fixed number of rclone calls.

The defect this module exists to remove
---------------------------------------
`jobmeta.read_job` folds ONE job and shells out ONE `rclone copy` to do it. That
is the right shape box-side (jobd folds a single job and has no fleet to
amortize over) and the wrong shape for `job orphans` / `job ls`, which call it
once per queued ticket. Measured on the live queue 2026-08-17, 275 tickets:

    list_all_queued (1 rclone lsf)          0.59 s
    _present_iids_set (vast API)            0.37 s
    _live_iids_set (vast API)               0.38 s
    read_job x275 (275 rclone copy)       138.31 s   <-- 99.0% of the command
    fold + local file reads                 0.24 s
    TOTAL                                 139.68 s

Every one of those 275 copies transferred ZERO bytes — the cache already held
every body — so the 0.50 s mean was pure fixed cost: fork/exec, rclone config
parse, B2 authorize, one LIST round-trip. The work itself (reading ~19 k small
JSON files and folding them) was 0.24 s, i.e. 0.17% of the wall clock. The
client was the whole problem; see `docs/plans/vast-tooling-refactor-v2.md` and
the run notes for why the B2 layout was left alone.

What this does instead
----------------------
1. ONE `rclone lsf -R --filter-from …` scoped to the job ids asked for. That
   single listing yields every event key of every job, plus each job's
   `results.DONE.json` marker. 5.1 s for 275 jobs / 14 331 keys, and it scales
   with the QUEUE rather than with the bucket (an unfiltered recursive listing
   of `jobs/` walks 78 236 objects and costs 15.4 s, growing with every
   checkpoint anyone ever wrote).
2. ONE `rclone copy --files-from …` for exactly the bodies the local cache is
   missing — skipped entirely when it is missing none.
3. Local reads + `jobmeta.fold_events`, per job.

So: **2 rclone subprocesses for N jobs, and 1 when the cache is warm.** Not
2N. `test_vastlib_jobs_scan.py` asserts the call count against an injected
runner, which is why that property cannot rot.

THE FRESHNESS CONTRACT (read `jobmeta.read_job_fresh`'s docstring first)
------------------------------------------------------------------------
During the 2026-07-30 frontier launch the folded view ran minutes behind
reality — jobs that had already failed on-box still read `submitted live=False`,
which is indistinguishable from "still queued on a dead box". `read_job`'s
incremental cache was one of the two named contributors. `job orphans` exists to
tell "will never run" from "hasn't started yet", so a fast-but-stale fold here
would turn a correct verdict into a wrong one. The contract is therefore stated,
not assumed:

* **Nothing folded is ever cached.** No view, no status, no verdict, no TTL.
  Every call re-folds from bodies. There is no "is the cache stale" question to
  get wrong, because the only cached thing is an event BODY, and event objects
  are immutable and append-only — a body that exists is correct forever.
* **The key set is re-learned from B2 on every call**, by the listing in step 1.
  That listing is the freshness boundary, and it is the SAME boundary
  `read_job` has (its `rclone copy` also LISTs). It is in fact tighter: the
  per-job loop listed job #275 a hundred-odd seconds after job #1, so the old
  scan folded a queue smeared across two minutes of wall clock. This one takes
  a single snapshot.
* **The fold set is `listed ∪ cached`, never `listed` alone.** A key once
  observed is never un-observed, so the union can only ever be FRESHER than the
  listing — and it keeps parity with `read_job`, which folds whatever the cache
  directory holds. (It also means a body another process cached a moment ago is
  not dropped on the floor.)
* **A listed body that could not be fetched is an ERROR, never a silent gap.**
  Missing events fold to a younger status, and a younger status is exactly what
  mints a false orphan. Such a job comes back carrying `scan_error` and the
  caller renders it UNKNOWN rather than guessing.
* **This does NOT reach `read_job_fresh`'s guarantee** and does not try to.
  That call reads each key with a strongly-consistent `cat` and probes
  `results.DONE.json` directly; it stays the tool for adjudicating ONE job, and
  is unchanged. What this adds, free, is the `done_marker` KEY from the same
  listing — LIST-grade evidence (weaker than the `cat`) that a job finished
  even if no `done` event has surfaced. It is reported, never used to
  manufacture a verdict.

Zone boundary
-------------
`jobmeta.py` is a Zone S shipped flat leaf that jobd imports on-box, so it may
not grow a dependency on anything that does not ship with it. This module is
Zone P — operator-side only — and the entire fast path lives here. jobmeta's
per-job API is untouched apart from `event_cache_root`/`event_cache_dir`, which
are stdlib-only and exist so the cache LAYOUT has one definition instead of
three copies of the same `os.path.join`.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Mapping
from typing import Any, Callable

import jobmeta
import runmeta

# rclone's injectable-runner contract (runmeta's): (args, input=None) -> (rc, out, err).
Runner = Callable[..., tuple[int, str, str]]

# Concurrency for the scoped listing. rclone walks `<jid>/` and `<jid>/events/`
# as two directory listings per job and runs them `--checkers`-wide; measured
# 2026-08-17 over 275 queued jobs: 32 -> 5.72 s, 64 -> 5.05 s. `--fast-list` is
# deliberately NOT used — it forces one flat recursive listing of the whole
# `jobs/` prefix (78 236 objects, 15.4 s) and ignores the filter's ability to
# prune, so it is both slower today and unboundedly slower as checkpoints
# accumulate.
LIST_CHECKERS = 64
COPY_TRANSFERS = 16
COPY_CHECKERS = 32

DONE_MARKER = "results.DONE.json"

# Below this many jobs, fold them ONE AT A TIME with `jobmeta.read_job` instead.
#
# The bulk listing has a ~4 s FLOOR that does not depend on the job count: rclone
# must enumerate the directory entries under `jobs/` before it can decide which
# of them the filter lets it descend into, and that prefix holds 78 236 objects.
# Measured on the live bucket 2026-08-17 (bulk vs. the per-job loop, warm cache):
#
#     n=1    bulk 4.05 s   loop  1.07 s      n=8    bulk 3.91 s   loop  4.74 s
#     n=3    bulk 4.21 s   loop  2.16 s      n=20   bulk 4.09 s   loop 10.83 s
#     n=275  bulk 5.05 s   loop 138.31 s
#
# The crossover therefore sits in the 7-10 band (network jitter is a second of
# it — a live `--box` scan of 7 tickets re-measured at 5.06 s where the loop
# predicts ~3.9 s), and a blanket bulk path would have made
# `job orphans --box <IID>` (typically 1-13 tickets) 4x SLOWER while making the
# unfiltered scan 21x faster. 10 is the conservative side of that band: it costs
# at most ~1 s in the 7-9 range and protects every small scan.
# Re-measure before moving it — the floor tracks the object count
# under `jobs/`, which only grows.
BULK_MIN_JOBS = 10

# The same threshold for the box-lifecycle logs (`jobs/nodes/<iid>/events/`).
# Their prefix is far smaller than `jobs/`, so the listing floor is lower and
# the crossover is earlier; reusing one number keeps the two paths reasoning
# alike and is well inside the measured band either way.
BULK_MIN_BOXES = BULK_MIN_JOBS


class ScanError(RuntimeError):
    """The bulk listing failed. Raised rather than returned as an empty scan:
    an unreadable queue that reads as "no events" mints a false ORPHAN_UNCLAIMED
    on every ticket, which is a fleet-wide wrong answer, not a degraded one."""


def _runner(runner: Runner | None) -> Runner:
    return runner if runner is not None else runmeta._default_runner


def _filter_lines(job_ids: Iterable[str]) -> list[str]:
    """rclone filter rules that admit exactly the two key shapes we read.

    Every id has already been through `jobmeta.validate_job_id` (`JOB_ID_RE`,
    `[A-Za-z0-9._-]{1,64}`), so no id can carry a newline, a `/`, a `*` or a
    `{` — i.e. it cannot forge a rule or widen the glob. That validation is
    load-bearing here, not decoration."""
    out: list[str] = []
    for jid in job_ids:
        out.append(f"+ /{jid}/events/*.json")
        out.append(f"+ /{jid}/{DONE_MARKER}")
    out.append("- *")
    return out


def _parse_listing(text: str, wanted: set[str]) -> tuple[dict[str, set[str]], set[str]]:
    """`(event keys per job, job ids with a DONE marker)` from `lsf -R` output.

    Paths are relative to `jobs/`. Anything that is not one of the two expected
    shapes, or names a job we did not ask about, is dropped — a filter is a
    request, not a guarantee, and a stray key must not become a phantom event."""
    events: dict[str, set[str]] = {jid: set() for jid in wanted}
    done: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("/")
        if len(parts) == 3 and parts[1] == "events" and parts[2].endswith(".json"):
            if parts[0] in events:
                events[parts[0]].add(parts[2])
        elif len(parts) == 2 and parts[1] == DONE_MARKER:
            if parts[0] in wanted:
                done.add(parts[0])
    return events, done


def _cached_keys(job_id: str, cache_dir: str | None) -> set[str]:
    try:
        return {n for n in os.listdir(jobmeta.event_cache_dir(job_id, cache_dir))
                if n.endswith(".json")}
    except OSError:
        return set()


def list_event_keys(job_ids: Iterable[str], *, runner: Runner | None = None,
                    bucket: str | None = None,
                    ) -> tuple[dict[str, set[str]], set[str]]:
    """ONE `rclone lsf -R` over the given jobs. `(event keys, done-marker jobs)`.

    Raises `ScanError` on a nonzero rc. Returns empty structures — and makes NO
    rclone call — for an empty job list."""
    wanted = sorted({str(j) for j in job_ids})
    for jid in wanted:
        jobmeta.validate_job_id(jid)          # Zone S: JOB_ID_RE, see _filter_lines
    if not wanted:
        return {}, set()
    b = jobmeta._bucket(bucket)  # type: ignore[no-untyped-call]  # Zone S, untyped
    with tempfile.TemporaryDirectory(prefix="vast-jobscan-") as td:
        fpath = os.path.join(td, "filter.txt")
        with open(fpath, "w") as fh:
            fh.write("\n".join(_filter_lines(wanted)) + "\n")
        rc, out, err = _runner(runner)(
            ["lsf", "-R", "--files-only", "--checkers", str(LIST_CHECKERS),
             "--filter-from", fpath, f"b2:{b}/jobs/"])
    if rc != 0:
        raise ScanError(f"bulk job listing failed rc={rc}: {(err or '').strip()}")
    return _parse_listing(out or "", set(wanted))


def fetch_missing_bodies(missing: Mapping[str, Iterable[str]], *,
                         runner: Runner | None = None, bucket: str | None = None,
                         cache_dir: str | None = None) -> int:
    """ONE `rclone copy --files-from` for every missing event body, across all
    jobs. Returns the number of keys requested; makes no call for zero.

    The destination is `event_cache_root`, and the `--files-from` entries are
    `<job_id>/events/<key>.json` — the cache mirrors the B2 key space with
    `jobs/` stripped, so each body lands in exactly the directory `read_job`
    and `parked_lifecycle.job_ticket` already read from. That shared layout is
    the reason this fast path needs no cache of its own."""
    rels = [f"{jid}/events/{k}" for jid, keys in sorted(missing.items())
            for k in sorted(keys)]
    if not rels:
        return 0
    b = jobmeta._bucket(bucket)  # type: ignore[no-untyped-call]  # Zone S, untyped
    root = jobmeta.event_cache_root(cache_dir)
    os.makedirs(root, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vast-jobscan-") as td:
        fpath = os.path.join(td, "files-from.txt")
        with open(fpath, "w") as fh:
            fh.write("\n".join(rels) + "\n")
        rc, _out, err = _runner(runner)(
            ["copy", f"b2:{b}/jobs/", root, "--files-from", fpath,
             "--transfers", str(COPY_TRANSFERS), "--checkers", str(COPY_CHECKERS)])
    if rc != 0:
        raise ScanError(f"bulk event fetch failed rc={rc}: {(err or '').strip()}")
    return len(rels)


def _read_bodies(job_id: str, keys: Iterable[str],
                 cache_dir: str | None) -> tuple[list[bytes], list[str]]:
    d = jobmeta.event_cache_dir(job_id, cache_dir)
    bodies: list[bytes] = []
    unread: list[str] = []
    for name in sorted(keys):
        try:
            with open(os.path.join(d, name), "rb") as fh:
                bodies.append(fh.read())
        except OSError:
            unread.append(name)
    return bodies, unread


def _fold_one_by_one(job_ids: list[str], *, live_iids: Any,  # noqa: ANN401 — jobmeta's seam
                     runner: Runner | None, bucket: str | None,
                     cache_dir: str | None) -> dict[str, dict[str, Any]]:
    """The small-queue path: `jobmeta.read_job` per job (see `BULK_MIN_JOBS`).

    `done_marker` comes back **None, not False** — it is not probed here, and
    "unmeasured" must not render as "no marker". Probing it would cost one
    `rclone cat` per job, which is the fan-out this module exists to avoid."""
    views: dict[str, dict[str, Any]] = {}
    for jid in job_ids:
        kw: dict[str, Any] = {"live_iids": live_iids}
        if runner is not None:
            kw["runner"] = runner
        if bucket is not None:
            kw["bucket"] = bucket
        if cache_dir is not None:
            kw["cache_dir"] = cache_dir
        try:
            view: dict[str, Any] = jobmeta.read_job(jid, **kw)
        except Exception as e:                  # one bad log never hides the rest
            views[jid] = {"scan_error": str(e), "done_marker": None}
            continue
        view["done_marker"] = None
        views[jid] = view
    return views


def fold_many(job_ids: Iterable[str], *, live_iids: Any = (),  # noqa: ANN401 — jobmeta's seam
              runner: Runner | None = None, bucket: str | None = None,
              cache_dir: str | None = None) -> dict[str, dict[str, Any]]:
    """Fold every named job. `{job_id: view}`, same view shape as `read_job`
    plus `done_marker` (True/False from the listing, None when not probed) and,
    on failure, `scan_error` (str).

    Cost: 1 rclone subprocess warm, 2 cold, INDEPENDENT of len(job_ids) — above
    `BULK_MIN_JOBS`. At or below it the per-job path is measurably faster and is
    used instead; that threshold is the only reason this function has two
    branches, and the measurement behind it is at its definition.

    Freshness: see the module docstring — nothing folded is cached, the key set
    is re-listed on every call, and the fold set is `listed ∪ cached`.

    Raises `ScanError` if the listing or the bulk fetch fails as a whole. A
    per-job problem (a body that is listed but still unreadable after the
    fetch) is reported IN that job's view as `scan_error`, so one broken log
    never hides the other 274."""
    wanted = sorted({str(j) for j in job_ids})
    if not wanted:
        return {}
    # Up front, and for BOTH paths: a malformed id is a caller bug, not a row.
    # It is also what keeps an id from forging an rclone filter rule downstream
    # (`_filter_lines`), so it must not be reachable-only-on-the-bulk-branch.
    for jid in wanted:
        jobmeta.validate_job_id(jid)
    if len(wanted) < BULK_MIN_JOBS:
        return _fold_one_by_one(wanted, live_iids=live_iids, runner=runner,
                                bucket=bucket, cache_dir=cache_dir)
    listed, done = list_event_keys(wanted, runner=runner, bucket=bucket)

    cached = {jid: _cached_keys(jid, cache_dir) for jid in wanted}
    missing = {jid: listed.get(jid, set()) - cached[jid] for jid in wanted}
    fetch_missing_bodies({j: k for j, k in missing.items() if k},
                         runner=runner, bucket=bucket, cache_dir=cache_dir)

    views: dict[str, dict[str, Any]] = {}
    for jid in wanted:
        keys = listed.get(jid, set()) | cached[jid]
        bodies, unread = _read_bodies(jid, keys, cache_dir)
        view: dict[str, Any] = jobmeta.fold_events(bodies, live_iids)
        view["done_marker"] = jid in done
        if unread:
            # A LISTED body we still cannot read after the fetch. Folding around
            # it would report a younger status than the truth, and a younger
            # status is what mints a false orphan. Say so instead.
            view["scan_error"] = (f"{len(unread)} event body/bodies listed on B2 but "
                                  f"unreadable locally (first: {unread[0]})")
        views[jid] = view
    return views


# --------------------------------------------------------------------------- #
# The box-lifecycle logs — `job ls`'s OTHER per-item rclone spawn
# --------------------------------------------------------------------------- #
# `cmd_job_ls` calls `jobmeta.read_box` once per DISTINCT BOX in the queue, and
# `read_box` is `read_job`'s twin: one `rclone copy` of
# `jobs/nodes/<iid>/events/` per call. Measured 2026-08-17 on the live queue,
# 154 distinct boxes: 69.6 s in 155 subprocesses — which is why `job ls` still
# took 78.6 s after the job folds dropped from 139 s to 6.5 s. Same defect, same
# shape, same fix. The box roster grows with fleet HISTORY (every box that ever
# held a ticket stays in the queue listing), so this one only gets worse.
#
# Scoped to `jobs/nodes/` rather than `jobs/`: the nodes prefix is small, so the
# listing does not pay for the 78 236 objects under `jobs/`.

def _box_filter_lines(iids: Iterable[str]) -> list[str]:
    out = [f"+ /{iid}/events/*.json" for iid in iids]
    out.append("- *")
    return out


def _parse_box_listing(text: str, wanted: set[str]) -> dict[str, set[str]]:
    """`{iid: event keys}` from `lsf -R` output relative to `jobs/nodes/`."""
    events: dict[str, set[str]] = {iid: set() for iid in wanted}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("/")
        if len(parts) == 3 and parts[1] == "events" and parts[2].endswith(".json"):
            if parts[0] in events:
                events[parts[0]].add(parts[2])
    return events


def _box_cache_dir(iid: str, cache_dir: str | None) -> str:
    return os.path.join(jobmeta.event_cache_root(cache_dir), "nodes", iid, "events")


def _fold_boxes_one_by_one(iids: list[str], *, runner: Runner | None,
                           bucket: str | None, cache_dir: str | None,
                           ) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for iid in iids:
        kw: dict[str, Any] = {}
        if runner is not None:
            kw["runner"] = runner
        if bucket is not None:
            kw["bucket"] = bucket
        if cache_dir is not None:
            kw["cache_dir"] = cache_dir
        try:
            out[iid] = jobmeta.read_box(iid, **kw)
        except Exception:
            # Matches `cmd_job_ls`'s existing swallow: a box whose lifecycle log
            # is unreadable renders as "not parked", never as a crash. Unlike a
            # JOB fold, nothing here can mint an orphan verdict.
            out[iid] = {"parked": False, "drained_pending": False}
    return out


def fold_boxes(instance_ids: Iterable[str], *, runner: Runner | None = None,
               bucket: str | None = None, cache_dir: str | None = None,
               ) -> dict[str, dict[str, Any]]:
    """`{instance_id: box lifecycle view}` — `jobmeta.read_box` for many boxes in
    a fixed number of rclone calls. Same freshness contract as `fold_many`:
    nothing folded is cached, the key set is re-listed every call, and the fold
    set is `listed ∪ cached`.

    Never raises for one bad box; a wholesale listing failure falls back to the
    per-box path rather than reporting an empty fleet, because an unreadable
    lifecycle log downgrades to "not parked" and cannot mint a verdict."""
    wanted = sorted({str(i) for i in instance_ids})
    if not wanted:
        return {}
    if len(wanted) < BULK_MIN_BOXES:
        return _fold_boxes_one_by_one(wanted, runner=runner, bucket=bucket,
                                      cache_dir=cache_dir)
    b = jobmeta._bucket(bucket)  # type: ignore[no-untyped-call]  # Zone S, untyped
    root = os.path.join(jobmeta.event_cache_root(cache_dir), "nodes")
    try:
        with tempfile.TemporaryDirectory(prefix="vast-boxscan-") as td:
            fpath = os.path.join(td, "filter.txt")
            with open(fpath, "w") as fh:
                fh.write("\n".join(_box_filter_lines(wanted)) + "\n")
            rc, out, err = _runner(runner)(
                ["lsf", "-R", "--files-only", "--checkers", str(LIST_CHECKERS),
                 "--filter-from", fpath, f"b2:{b}/jobs/nodes/"])
        if rc != 0:
            raise ScanError(f"bulk box listing failed rc={rc}: {(err or '').strip()}")
        listed = _parse_box_listing(out or "", set(wanted))

        rels = []
        cached: dict[str, set[str]] = {}
        for iid in wanted:
            try:
                cached[iid] = {n for n in os.listdir(_box_cache_dir(iid, cache_dir))
                               if n.endswith(".json")}
            except OSError:
                cached[iid] = set()
            rels += [f"{iid}/events/{k}" for k in sorted(listed[iid] - cached[iid])]
        if rels:
            os.makedirs(root, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="vast-boxscan-") as td:
                fpath = os.path.join(td, "files-from.txt")
                with open(fpath, "w") as fh:
                    fh.write("\n".join(rels) + "\n")
                rc, _out, err = _runner(runner)(
                    ["copy", f"b2:{b}/jobs/nodes/", root, "--files-from", fpath,
                     "--transfers", str(COPY_TRANSFERS),
                     "--checkers", str(COPY_CHECKERS)])
            if rc != 0:
                raise ScanError(
                    f"bulk box fetch failed rc={rc}: {(err or '').strip()}")
    except ScanError:
        # Degrade to the shape that was there before rather than blanking the
        # fleet's park/drain state. Costs one rclone spawn per box; correct.
        return _fold_boxes_one_by_one(wanted, runner=runner, bucket=bucket,
                                      cache_dir=cache_dir)

    views: dict[str, dict[str, Any]] = {}
    for iid in wanted:
        d = _box_cache_dir(iid, cache_dir)
        bodies: list[bytes] = []
        for name in sorted(listed[iid] | cached[iid]):
            try:
                with open(os.path.join(d, name), "rb") as fh:
                    bodies.append(fh.read())
            except OSError:
                pass
        views[iid] = jobmeta.fold_box_events(bodies)
    return views
