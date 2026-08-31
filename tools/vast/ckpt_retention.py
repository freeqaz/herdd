#!/usr/bin/env python3
"""Retention sweep for INTERMEDIATE training checkpoints on B2.

The growth problem
------------------
Every jobs-v2 training job syncs its HF checkpoint tree to
`jobs/<JOB_ID>/checkpoints/` on a timer, and **nothing has ever removed any of
it.** One checkpoint is ~0.98 GB (`adapter_model.safetensors` 323 MB fp32,
`optimizer.pt` 646 MB, `tokenizer.json` 11 MB) and a long run leaves many. The
box-side half of this (`jobd.sh`: delete-after-sync, end-of-run scrub) bounds the
BOX disk; this module is the bucket-side half.

What this sweep may touch — and what it must never touch
--------------------------------------------------------
There are TWO prefixes with `checkpoints` in the name and they are completely
different things:

    checkpoints/<RUN_NAME>/         <- the PUBLISHED FINAL ADAPTER. The product
                                       of a run. `PUB_DEST` in every train
                                       bundle's run.sh. **NEVER SWEPT. NEVER
                                       ENUMERATED HERE.**
    jobs/<JOB_ID>/checkpoints/      <- mid-run resume state. THIS is the sweep's
                                       only subject.

`_assert_sweepable` fails CLOSED on every path before it can reach a delete
call, and `sweep()` refuses to run at all if any planned path fails it. That
check is not defence-in-depth decoration: a bug that pointed this tool one level
up would delete the entire model output of every run we have ever done, and B2
deletes are irreversible.

Five gates, all of which must pass before a job's intermediates are swept
------------------------------------------------------------------------
1. **TERMINAL and non-resumable.** `done` and `cancelled` only. A `failed` job
   is RESUMABLE — `herdd job requeue` re-opens it and seeds the new attempt
   from exactly this prefix (JOBS_DESIGN "Terminal-failed recovery") — so its
   checkpoints are live resume state, not garbage. `--include-failed` exists but
   is off by default and says so loudly.
2. **PUBLISHED.** `checkpoints/<RUN_NAME>/PUBLISHED.json` or
   `jobs/<JOB_ID>/results.DONE.json` must exist. No published artifact means the
   run's output may exist NOWHERE ELSE, and the intermediates are then the only
   copy of anything it produced.
3. **OLD, and READABLY so.** The newest object under the prefix is older than
   the window (default 7 days). An object whose `ModTime` we cannot parse makes
   the whole job `keep_unknown_age` — mapping an unreadable timestamp to epoch 0
   would make it look older than every window, which is the one unknown in this
   module that would have resolved to DELETE instead of KEEP.
4. **STATUS KNOWN.** A job whose event stream we could not read is KEPT.
   Unknown is not "probably fine" — it is the one state where we have no
   evidence at all.
5. **NOT BOX-PRUNED.** `jobs/<JOB_ID>/CHECKPOINTS_PRUNED.json` must be absent.
   Since 2026-08-05 jobd prunes checkpoint dirs off the BOX disk once they are
   verified on B2 (`CHECKPOINT_LIFECYCLE.md`). `checkpoints:` and `results:` are
   the same `out/**` glob in every training bundle, so `results/` used to hold a
   second complete copy of the grid — after pruning it holds only the steps that
   survived to the end, and **`jobs/<JOB_ID>/checkpoints/` is the SOLE copy of
   the rest.** Gate 2 would otherwise read a non-empty `results/` as proof of
   redundancy that no longer exists. Their doc asks a bucket-side sweep to
   compare step sets or refuse outright; we refuse outright, which is the
   version that cannot be subtly wrong.

   **`--sweep-box-pruned` opts in, and then a keep policy is MANDATORY.** The
   refusal is right about the facts and wrong as a permanent answer: measured
   2026-08-27, marked jobs held **476 of 615 GiB (77%)** of `jobs/`, so refusing
   them outright means the sweep can never reclaim the bulk of the bucket. For a
   marked job the question is not "is this redundant" (it is not) but "which
   steps do we keep" — which is what `--keep-first/--keep-last/--keep-stride`
   already answer. Opting in with an EMPTY policy is refused separately
   (`keep_box_pruned_no_policy`): that combination, and only that one, deletes a
   sole copy and leaves no skeleton behind.

The dose curve
--------------
Intermediate checkpoints are not only resume state; they are the input to the
dose-curve / echo-collapse analysis (v4's dose curve was U-shaped, so the EARLY
checkpoints are the interesting ones — memory `dose-curve-before-epochs-retrain`,
`v4-echo-collapse-overtraining`). A window that deletes them before that analysis
runs destroys real evidence, and re-creating it costs a full training run.
`--keep-first N` / `--keep-last N` (and `--keep-stride K`) preserve a dose-curve
skeleton at ~0.98 GB per retained step; `inventory` reports what each policy
costs so the window can be chosen with numbers instead of vibes.

It does not finish on the live bucket yet
-----------------------------------------
Measured 2026-08-27: `inventory` was killed at a 900 s timeout having printed
NOTHING, and a `plan` scoped to a SINGLE job with `--job` did not finish in
10 min either. So the cost is not the per-job probes — `--job` filters before
those run — it is `list_checkpoint_objects`, one rclone pass over ~100 k objects
under `jobs/**/checkpoints/`. Both subcommands print only at the very end, so a
timeout reads as silence; piping either through `tail` additionally swallows the
kill and exits 0.

Until that listing is paged or cached, drive `classify_job` directly with an
object list you already have (it is PURE and takes `CkptObj`s), or budget tens
of minutes. The gate arms in this module's tests are the fast path for checking
behaviour.

Safety posture
--------------
DRY-RUN IS THE DEFAULT. `--apply` is required to delete, every deletion is
logged with its byte count, and `sweep` refuses to proceed if it cannot read the
job status stream (gate 4) — "verification unavailable => do nothing".
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobmeta                                             # noqa: E402

# --- constants -------------------------------------------------------------- #
WINDOW_DAYS = 7.0        # owner's proposal ("maybe we retain for a window like
                         # 7 days"), a TUNABLE DEFAULT rather than a fixed
                         # constant — `inventory` reports alternatives.
KEEP_FIRST = 0           # dose-curve skeleton: oldest N steps kept
KEEP_LAST = 0            # newest N steps kept
KEEP_STRIDE = 0          # keep every Kth step (0 = off)

#: The ONLY prefix shape this module may delete under. Everything else — and
#: most emphatically `checkpoints/<RUN_NAME>/`, the published adapters — is out
#: of bounds.
SWEEPABLE_PREFIX = "jobs/"
SWEEPABLE_SEGMENT = "/checkpoints/"

#: Job statuses whose intermediates are safe to consider. `failed` is ABSENT on
#: purpose: `herdd job requeue` re-opens a terminal-failed job and seeds it
#: from this exact prefix, so those checkpoints are resume state.
SWEEPABLE_STATUSES = frozenset({"done", "cancelled"})
RESUMABLE_STATUSES = frozenset({"failed"})

Verdict = namedtuple("Verdict",
                     ["job_id", "action", "reason", "bytes", "paths",
                      "status", "steps", "kept_steps", "newest_ts"])
CkptObj = namedtuple("CkptObj", ["path", "size", "mtime"])

# The completion contract, third of three spellings (jobd.sh
# `_ckpt_names_complete`, train_proposer_lora.py `_CKPT_REQUIRED_FILES`; long
# form in CHECKPOINT_LIFECYCLE.md §"The completion contract"). Each tuple is a
# set of accepted spellings for one requirement.
CKPT_COMPLETE_SUFFIX = ".complete.json"
CKPT_REQUIRED_FILES = (
    ("trainer_state.json",),
    ("optimizer.pt", "optimizer.bin"),
    ("scheduler.pt",),
    ("adapter_model.safetensors", "adapter_model.bin", "model.safetensors",
     "pytorch_model.bin", "model.safetensors.index.json",
     "pytorch_model.bin.index.json"),
)


class UnsafePath(Exception):
    """A path that is not an intermediate-checkpoint object. Raised, never
    swallowed: reaching this means the caller's assumptions are wrong, and the
    correct response to that is to stop, not to skip one entry."""


# --- the safety check ------------------------------------------------------- #

def _assert_sweepable(path):
    """Fail CLOSED unless `path` is an object under `jobs/<JOB_ID>/checkpoints/`.

    Explicitly rejects, among everything else:
      * `checkpoints/<RUN_NAME>/...`  — the PUBLISHED FINAL ADAPTERS
      * `jobs/<JOB_ID>/results/...`   — the job's published output
      * `jobs/<JOB_ID>/events/...`    — the state channel the fold reads
      * anything with `..` in it, and anything that is a bare prefix rather than
        an object (a directory-shaped delete is how a whole tree disappears).
    """
    p = (path or "").lstrip("/")
    if not p or p.endswith("/"):
        raise UnsafePath(f"refusing a non-object path: {path!r}")
    if ".." in p.split("/") or "." in p.split("/"):
        # `.` cannot escape `jobs/` on its own, but it is the same
        # Python-normalizes-differently-than-the-backend class the `..` check
        # guards, and no real listing produces it.
        raise UnsafePath(f"refusing a path with a relative segment: {path!r}")
    if not p.startswith(SWEEPABLE_PREFIX):
        raise UnsafePath(
            f"refusing {path!r}: outside jobs/. `checkpoints/<RUN_NAME>/` at the "
            f"bucket top level holds the PUBLISHED FINAL ADAPTERS and is never "
            f"swept by this tool")
    parts = p.split("/")
    # jobs/<JOB_ID>/checkpoints/<...>/<object>
    if len(parts) < 4 or parts[2] != "checkpoints" or not parts[1]:
        raise UnsafePath(
            f"refusing {path!r}: only objects under jobs/<JOB_ID>/checkpoints/ "
            f"are intermediate training checkpoints")
    return p


def ckpt_key(path):
    """`(root, step)` for a checkpoint object, or `(None, None)`.

    The `checkpoint-<N>` segment is NOT at a fixed depth. Measured against the
    live bucket 2026-08-05: 22,577 of 22,898 objects under `jobs/*/checkpoints/`
    are `checkpoints/out/checkpoint-<N>/...` — jobd pushes the job's whole
    `work/` tree, so the training output dir sits in between, and a multi-arm
    bundle adds `arms/<name>/` as well. Keying on a fixed depth finds NOTHING on
    the real bucket, and a retention policy that thinks a job has zero steps
    would delete every object with no `--keep-*` protection at all.

    `root` is everything before the `checkpoint-<N>` segment, so steps from two
    layout roots (`arms/a/checkpoint-50` vs `arms/b/checkpoint-50`) are kept
    apart — the same per-root grouping jobd's own prune uses.
    """
    p = _assert_sweepable(path)
    parts = p.split("/")[3:]                # after jobs/<JOB_ID>/checkpoints/
    for i in range(len(parts) - 1, -1, -1):
        seg = parts[i]
        if seg.startswith("checkpoint-") and seg[len("checkpoint-"):].isdigit():
            return "/".join(parts[:i]), int(seg[len("checkpoint-"):])
    return None, None


def ckpt_step(path):
    """The step number of a checkpoint object, or None when it is not inside a
    `checkpoint-<N>` directory (e.g. a stray file directly under `checkpoints/`,
    which no keep policy protects and which is therefore swept with the rest)."""
    return ckpt_key(path)[1]


def marker_key(path):
    """`(root, step)` for a completion MARKER object, else `(None, None)`.

    jobd publishes `…/checkpoint-<N>.complete.json` as a SIBLING of
    `…/checkpoint-<N>/` (CHECKPOINT_LIFECYCLE.md §"The completion contract"), so
    `ckpt_key` — which looks for a path SEGMENT named `checkpoint-<N>` — reads it
    as keyless and the keep policy would delete a kept checkpoint's marker.
    """
    p = _assert_sweepable(path)
    parts = p.split("/")[3:]                # after jobs/<JOB_ID>/checkpoints/
    if not parts or not parts[-1].endswith(CKPT_COMPLETE_SUFFIX):
        return None, None
    stem = parts[-1][:-len(CKPT_COMPLETE_SUFFIX)]
    if not stem.startswith("checkpoint-") or not stem[len("checkpoint-"):].isdigit():
        return None, None
    return "/".join(parts[:-1]), int(stem[len("checkpoint-"):])


def any_key(path):
    """`ckpt_key` widened to count a sibling completion marker as belonging to
    its checkpoint, so the two are always kept or swept together."""
    k = ckpt_key(path)
    return k if k[1] is not None else marker_key(path)


def _names_complete(names):
    """The file-set half of the completion contract, over a set of BASE names."""
    return all(any(n in names or (n + ".bnb_skipped") in names for n in group)
               for group in CKPT_REQUIRED_FILES)


def incomplete_checkpoint_keys(paths):
    """The `(root, step)` keys this object list shows to be POSITIVELY incomplete.

    An interrupted multi-GB upload leaves a real `checkpoint-<N>/` prefix on B2
    that is not resumable state — it is debris. Letting it occupy a `--keep-last`
    slot is how a corruption evicts a good checkpoint, so the keep skeleton is
    computed over the complete steps only.

    SELF-CALIBRATING, and that is the safety property, not a nicety. A step is
    called incomplete only when a SIBLING under the same layout root proves the
    job writes the files this one lacks — either by carrying a completion marker
    or by holding the full set itself. So:

      * a checkpointer whose layout we do not model (no `optimizer.pt` anywhere,
        `SAVE_ONLY_MODEL`, a non-HF writer) has no complete sibling either, and
        NOTHING is withheld;
      * a caller who hands us a filtered or partial object list gets the same
        answer for the same reason;
      * the torn v16 shapes — a 2-object `checkpoint-96` beside eleven-object
        neighbours — are named exactly.

    An absolute file-set test would instead read a whole unfamiliar job as
    debris and strip its keep protection, which in a module whose next step is a
    delete is the wrong direction to be wrong in.
    """
    marked, members = set(), {}
    for path in paths:
        mk = marker_key(path)
        if mk[1] is not None:
            marked.add(mk)
            continue
        k = ckpt_key(path)
        if k[1] is None:
            continue
        members.setdefault(k, set()).add(path.rsplit("/", 1)[-1])
    judgeable = {root for (root, _step) in marked}
    judgeable |= {root for (root, _step), names in members.items()
                  if _names_complete(names)}
    return {k for k, names in members.items()
            if k[0] in judgeable and k not in marked and not _names_complete(names)}


def job_id_of(path):
    return _assert_sweepable(path).split("/")[1]


# --- policy (PURE) ---------------------------------------------------------- #

def steps_to_keep(steps, *, keep_first=KEEP_FIRST, keep_last=KEEP_LAST,
                  keep_stride=KEEP_STRIDE):
    """Which checkpoint steps survive the sweep. PURE.

    The dose curve wants a SKELETON, not the tail: v4's curve was U-shaped, so
    the early steps carry the signal. `keep_first` is therefore a first-class
    knob and not an afterthought behind `keep_last`.
    """
    ordered = sorted(set(int(s) for s in steps if s is not None))
    keep = set()
    if keep_first > 0:
        keep |= set(ordered[:keep_first])
    if keep_last > 0:
        keep |= set(ordered[-keep_last:])
    if keep_stride and keep_stride > 0:
        keep |= set(ordered[::int(keep_stride)])
    return keep


def keys_to_keep(keys, **kw):
    """`steps_to_keep` applied PER LAYOUT ROOT. PURE.

    `keys` is an iterable of `(root, step)`. A multi-arm bundle writes
    `arms/a/checkpoint-50` and `arms/b/checkpoint-50`; treating those as one
    series would let `--keep-last 1` protect arm b's newest step and delete arm
    a's, which is not what "keep the last checkpoint" means to anyone.
    """
    by_root = {}
    for root, step in keys:
        if step is None:
            continue
        by_root.setdefault(root, set()).add(step)
    out = set()
    for root, steps in by_root.items():
        out |= {(root, s) for s in steps_to_keep(steps, **kw)}
    return out


def classify_job(*, job_id, objects, status, published, now,
                 window_days=WINDOW_DAYS, include_failed=False,
                 keep_first=KEEP_FIRST, keep_last=KEEP_LAST,
                 keep_stride=KEEP_STRIDE, pruned=False,
                 sweep_box_pruned=False):
    """One job -> one `Verdict`. PURE, and every unknown resolves to KEEP.

    `status` is the folded job status (`jobmeta.fold_events`), or None when the
    event stream could not be read. `published` and `pruned` are True/False/None
    with the same convention — None means "could not check", which is a KEEP,
    not a pass.
    """
    objs = [o for o in (objects or []) if o.size >= 0]
    unaged = [o for o in objs if o.mtime is None]
    # `any_key`, not `ckpt_key`: a checkpoint's sibling completion marker keys to
    # the checkpoint, so the keep policy never protects one without the other.
    keys = {o.path: any_key(o.path) for o in objs}
    incomplete = incomplete_checkpoint_keys(o.path for o in objs)
    steps = sorted({k for k in keys.values() if k[1] is not None})
    newest = max((o.mtime for o in objs if o.mtime is not None), default=None)

    def keep(action, reason, kept=()):
        return Verdict(job_id, action, reason, 0, (), status, tuple(steps),
                       tuple(sorted(kept)), newest)

    if not objs:
        return keep("nothing", "no intermediate checkpoint objects")
    if status is None:
        return keep("keep_unknown_status",
                    "could not read this job's event stream — an unreadable "
                    "status is not evidence that the job ended, and this is the "
                    "one state where we have nothing to go on")
    if status in RESUMABLE_STATUSES and not include_failed:
        return keep("keep_resumable",
                    f"status={status}: a terminal-FAILED job is RESUMABLE — "
                    f"`herdd job requeue` re-opens it and seeds the retry from "
                    f"this exact prefix, so these are live resume state")
    if status not in SWEEPABLE_STATUSES and not (
            include_failed and status in RESUMABLE_STATUSES):
        return keep("keep_live",
                    f"status={status}: the job is running, queued or otherwise "
                    f"not terminal — its checkpoints ARE its resume state")
    if published is None:
        return keep("keep_unpublished",
                    "could not confirm a published artifact — treated as absent")
    if not published:
        return keep("keep_unpublished",
                    "no published adapter (checkpoints/<RUN_NAME>/PUBLISHED.json) "
                    "and no results.DONE.json — with no published artifact the "
                    "run's output may exist nowhere else, so its intermediates "
                    "are the only copy")
    if unaged:
        return keep("keep_unknown_age",
                    f"{len(unaged)} object(s) carry no readable ModTime — the "
                    f"age window is the only gate standing between a job "
                    f"written minutes ago and deletion, and an unreadable "
                    f"timestamp is not evidence of age")
    if pruned is None:
        return keep("keep_pruned_unknown",
                    "could not check for CHECKPOINTS_PRUNED.json — if jobd "
                    "pruned this job's checkpoints off the box disk, this "
                    "prefix is the SOLE copy of the pruned steps")
    if pruned and not sweep_box_pruned:
        return keep("keep_box_pruned",
                    "jobs/<JOB_ID>/CHECKPOINTS_PRUNED.json is present: jobd "
                    "pruned checkpoint dirs off the box before finalize globbed "
                    "out/**, so results/ is NOT a superset and this prefix holds "
                    "the only copy of the pruned steps "
                    "(CHECKPOINT_LIFECYCLE.md). --sweep-box-pruned opts in, and "
                    "requires an explicit keep policy")
    if pruned and not (keep_first or keep_last or keep_stride):
        # --sweep-box-pruned without a skeleton is the one combination that
        # deletes a SOLE copy and leaves nothing behind. The keep policy is what
        # makes the opt-in a retention decision rather than a data loss.
        return keep("keep_box_pruned_no_policy",
                    "--sweep-box-pruned given but the keep policy is empty: this "
                    "prefix is the ONLY copy of the pruned steps, so sweeping it "
                    "bare would delete the dose curve outright. Pass "
                    "--keep-first/--keep-last/--keep-stride")
    cutoff = now - float(window_days) * 86400.0
    if newest is None:
        return keep("keep_unknown_age",
                    "no object under this prefix carries a readable ModTime")
    if newest > cutoff:
        age_d = (now - newest) / 86400.0
        return keep("keep_young",
                    f"newest checkpoint object is {age_d:.1f}d old, inside the "
                    f"{window_days:g}d window")

    # A torn upload is debris, not a checkpoint, and must not consume a keep slot
    # — `--keep-last 1` protecting a 2-object `checkpoint-176/` while deleting the
    # complete `checkpoint-160/` behind it is a corruption evicting good state.
    # Only POSITIVELY incomplete steps are withheld (see incomplete_checkpoint_keys).
    kept = keys_to_keep([k for k in steps if k not in incomplete],
                        keep_first=keep_first, keep_last=keep_last,
                        keep_stride=keep_stride)
    doomed = [o for o in objs if keys[o.path] not in kept]
    if not doomed:
        return keep("nothing",
                    f"every step is retained by the keep policy "
                    f"(first={keep_first}, last={keep_last}, stride={keep_stride})",
                    kept)
    freed = sum(o.size for o in doomed)
    age_d = (now - newest) / 86400.0 if newest else 0.0
    kept_note = (f"; keeping steps {sorted(kept)} for the dose curve"
                 if kept else "")
    if incomplete:
        kept_note += (f"; {len(incomplete)} torn/incomplete step(s) "
                      f"{sorted(incomplete)} withheld from the keep policy")
    return Verdict(job_id, "sweep",
                   f"status={status}, published, newest object {age_d:.1f}d old "
                   f"(> {window_days:g}d) — {len(doomed)} object(s), "
                   f"{freed / 1e9:.2f} GB{kept_note}",
                   freed, tuple(o.path for o in doomed), status,
                   tuple(steps), tuple(sorted(kept)), newest)


def plan_summary(verdicts):
    """Roll a list of Verdicts into the numbers a human needs to decide."""
    out = {"jobs": len(verdicts), "sweep_jobs": 0, "sweep_objects": 0,
           "sweep_bytes": 0, "by_action": {}}
    for vd in verdicts:
        out["by_action"][vd.action] = out["by_action"].get(vd.action, 0) + 1
        if vd.action == "sweep":
            out["sweep_jobs"] += 1
            out["sweep_objects"] += len(vd.paths)
            out["sweep_bytes"] += vd.bytes
    return out


# --- B2 side (I/O, injectable) ---------------------------------------------- #

def _rclone(args, runner=None):
    """(rc, stdout, stderr). Never raises; rc=127 means rclone is absent."""
    if runner is not None:
        return runner(args)
    try:
        r = subprocess.run(["rclone", *args], capture_output=True, text=True)
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return 127, "", "rclone not found on PATH"


def _parse_ts(s):
    """rclone's RFC3339 ModTime -> epoch seconds. None when unparseable."""
    if not s:
        return None
    txt = str(s).replace("Z", "+00:00")
    if "." in txt:                      # trim sub-second to what fromisoformat takes
        head, _, tail = txt.partition(".")
        frac, sign, off = tail.partition("+") if "+" in tail else tail.partition("-")
        txt = f"{head}.{frac[:6]}{sign}{off}" if sign else f"{head}.{frac[:6]}"
    try:
        return datetime.datetime.fromisoformat(txt).timestamp()
    except ValueError:
        return None


def list_checkpoint_objects(bucket, *, runner=None):
    """Every object under `jobs/*/checkpoints/` -> `{job_id: [CkptObj]}`.

    Uses an `--include` filter anchored at `/*/checkpoints/**` so the listing
    itself can never walk `checkpoints/<RUN_NAME>/` — the published adapters are
    not merely excluded from deletion, they are never enumerated. Returns
    `(mapping, err)`; a non-None `err` means DO NOTHING.
    """
    rc, out, err = _rclone(["lsjson", "-R", "--files-only", "--fast-list",
                            "--no-mimetype",
                            "--include", "/*/checkpoints/**",
                            f"b2:{bucket}/jobs/"], runner=runner)
    if rc != 0:
        return {}, (err or f"rclone lsjson exited {rc}").strip()
    try:
        rows = json.loads(out or "[]")
    except json.JSONDecodeError as e:
        return {}, f"could not parse the rclone listing: {e}"
    jobs = {}
    for row in rows:
        rel = (row.get("Path") or "").lstrip("/")
        path = f"jobs/{rel}"
        try:
            jid = job_id_of(path)
        except UnsafePath:
            continue                     # not an intermediate checkpoint: ignore
        # KEEP a None mtime as None. `or 0.0` would map "we could not read the
        # age" to epoch 1970 — older than every window — so gate 3 would PASS
        # and the objects would be deleted. Every other unknown in this module
        # resolves to KEEP; this one would have resolved to DELETE, and it is the
        # only gate protecting a job whose checkpoints were written minutes ago.
        jobs.setdefault(jid, []).append(
            CkptObj(path, int(row.get("Size") or 0),
                    _parse_ts(row.get("ModTime"))))
    return jobs, None


def job_events(bucket, job_id, *, runner=None):
    """Raw event dicts for one job, via the canonical `jobmeta.read_job_events`.

    Reuses the tested reader rather than a second parser, but does NOT inherit
    its "no events == safe negative" contract: for a DELETION decision an
    unreadable stream and an empty one must not look alike, so callers here read
    `[]` as UNKNOWN.
    """
    r = runner or _rclone_runner(bucket)
    try:
        return jobmeta.read_job_events(job_id, runner=r, bucket=bucket)
    except Exception:                                      # noqa: BLE001
        return []


def _rclone_runner(bucket):
    """A `jobmeta`-shaped runner (`args -> (rc, out, err)`) over our `_rclone`."""
    del bucket                                             # jobmeta passes b2: paths
    return lambda args: _rclone(list(args))


def job_status(bucket, job_id, *, runner=None, events=None):
    """The folded status of one job, or None when we could not determine it.

    None is load-bearing: `classify_job` turns an unknown status into a KEEP. An
    EMPTY event list also yields None — for a deletion decision "the stream said
    nothing" is not evidence the job ended, it is evidence we do not know.

    `reopened` (a requeued job, `jobmeta.requeue_ticket`) is reported as its own
    status so a re-opened job can never read as terminal: requeue re-mints the
    ticket for a terminal-FAILED job and the retry resumes from exactly the
    prefix this tool deletes.
    """
    evs = events if events is not None else job_events(bucket, job_id,
                                                       runner=runner)
    if not evs:
        return None
    try:
        view = jobmeta.fold_events(evs)
    except Exception:                                      # noqa: BLE001
        return None
    if view.get("reopened"):
        return "reopened"
    return view.get("status")


def run_name_of(events):
    """Best-effort RUN_NAME from a job's events, or None.

    Only used to look for the STRONGEST publication witness
    (`checkpoints/<RUN_NAME>/PUBLISHED.json`). Absence downgrades the check, it
    never upgrades it — a job with no discoverable run name simply falls back to
    the per-job markers.
    """
    for ev in reversed(list(events or [])):
        if not isinstance(ev, dict):
            continue
        for key in ("run_name", "RUN_NAME"):
            v = ev.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        cfg = ev.get("config_echo")
        if isinstance(cfg, dict):
            v = cfg.get("run_name") or cfg.get("RUN_NAME")
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def is_published(bucket, job_id, *, run_name=None, runner=None):
    """True / False / None ("could not check") for a job's published artifact.

    Gate 2 of four. Two accepted witnesses:

      * `checkpoints/<RUN_NAME>/PUBLISHED.json` — the train bundles' `PUB_DEST`
        marker: the published FINAL ADAPTER really exists. Strongest, and only
        available when the run name is discoverable.
      * `jobs/<JOB_ID>/results.DONE.json` **and a non-empty `results/`** — jobd's
        written-last publish marker plus proof it covers something. The marker
        alone is NOT enough: jobd writes DONE even when publish-verify failed
        (a finished job must reach a terminal state), so a bare marker can sit
        over an empty results tree.

    Read-only `lsf`/`cat` probes only. This function never deletes, and it never
    lists recursively under `checkpoints/` — the published adapters are probed by
    exact key, never enumerated.
    """
    if run_name:
        rc, out, _ = _rclone(
            ["lsf", f"b2:{bucket}/checkpoints/{run_name}/PUBLISHED.json"],
            runner=runner)
        if rc == 0 and (out or "").strip():
            return True
    rc, out, err = _rclone(["lsf", f"b2:{bucket}/jobs/{job_id}/results.DONE.json"],
                           runner=runner)
    if rc != 0:
        low = (err or "").lower()
        return False if ("not found" in low or "no such" in low) else None
    if not (out or "").strip():
        return False
    rc2, out2, err2 = _rclone(["lsf", "-R", "--files-only",
                               f"b2:{bucket}/jobs/{job_id}/results/"],
                              runner=runner)
    if rc2 != 0:
        low = (err2 or "").lower()
        return False if ("not found" in low or "no such" in low) else None
    return bool((out2 or "").strip())


def is_box_pruned(bucket, job_id, *, runner=None):
    """True / False / None for `jobs/<JOB_ID>/CHECKPOINTS_PRUNED.json`.

    Gate 5. jobd writes this marker when it deleted checkpoint dirs off the BOX
    disk after verifying them on B2 (`CHECKPOINT_LIFECYCLE.md`, 2026-08-05).
    Because `checkpoints:` and `results:` are the same `out/**` glob, a pruned
    job's `results/` is NOT a second complete copy of the grid — so gate 2's
    "something was published" stops implying "these intermediates are
    redundant", and `jobs/<JOB_ID>/checkpoints/` becomes the sole copy of every
    pruned step. Refusing outright is deliberately blunter than comparing step
    sets: the blunt version cannot be subtly wrong.
    """
    rc, out, err = _rclone(
        ["lsf", f"b2:{bucket}/jobs/{job_id}/CHECKPOINTS_PRUNED.json"],
        runner=runner)
    if rc != 0:
        low = (err or "").lower()
        return False if ("not found" in low or "no such" in low) else None
    return bool((out or "").strip())


def delete_objects(bucket, paths, *, runner=None, log=print):
    """Delete intermediate-checkpoint objects. Returns (deleted, failed).

    EVERY path is re-checked with `_assert_sweepable` immediately before its
    delete call — not once at plan time, here, at the last possible moment. If
    any path fails the check NOTHING is deleted: a plan containing one unsafe
    path is a plan we do not understand, and partial execution of a plan we do
    not understand is the worst available option.
    """
    checked = [_assert_sweepable(p) for p in paths]        # raises => no deletes
    deleted, failed = [], []
    for p in checked:
        rc, _, err = _rclone(["deletefile", f"b2:{bucket}/{p}"], runner=runner)
        if rc == 0:
            deleted.append(p)
            log(f"   deleted b2:{bucket}/{p}")
        else:
            failed.append((p, (err or f"rc={rc}").strip()))
            log(f"   FAILED  b2:{bucket}/{p}: {(err or '').strip()}")
    return deleted, failed


# --- driver ----------------------------------------------------------------- #

def build_verdicts(bucket, *, now, window_days=WINDOW_DAYS, include_failed=False,
                   keep_first=KEEP_FIRST, keep_last=KEEP_LAST,
                   keep_stride=KEEP_STRIDE, runner=None, only=None,
                   status_of=None, published_of=None, pruned_of=None, jobs=None,
                   sweep_box_pruned=False):
    """Listing + per-job gates -> `(verdicts, err)`. `err` non-None => DO NOTHING.

    `status_of` / `published_of` / `pruned_of` / `jobs` are injection seams
    (tests, and `inventory`, which probes B2 once and then re-classifies each
    candidate window purely rather than re-listing four times).
    """
    err = None
    if jobs is None:
        jobs, err = list_checkpoint_objects(bucket, runner=runner)
    if err:
        return [], err
    evcache = {}

    def _default_status(jid):
        evcache[jid] = job_events(bucket, jid, runner=runner)
        return job_status(bucket, jid, events=evcache[jid])

    def _default_published(jid):
        return is_published(bucket, jid, runner=runner,
                            run_name=run_name_of(evcache.get(jid)))

    status_of = status_of or _default_status
    published_of = published_of or _default_published
    pruned_of = pruned_of or (lambda jid: is_box_pruned(bucket, jid,
                                                        runner=runner))
    out = []
    for jid in sorted(jobs):
        if only and jid not in only:
            continue
        objs = jobs[jid]
        st = status_of(jid)
        # Only probe publication/pruning for a job that could otherwise be swept
        # — each probe is a B2 round trip and the answer is irrelevant for a job
        # whose status already keeps it.
        pub = None
        pruned = None
        if st in SWEEPABLE_STATUSES or (include_failed and st in RESUMABLE_STATUSES):
            pub = published_of(jid)
            pruned = pruned_of(jid)
        out.append(classify_job(job_id=jid, objects=objs, status=st,
                                published=pub, pruned=pruned, now=now,
                                window_days=window_days,
                                include_failed=include_failed,
                                keep_first=keep_first, keep_last=keep_last,
                                keep_stride=keep_stride,
                                sweep_box_pruned=sweep_box_pruned))
    return out, None


def sweep(bucket, verdicts, *, apply=False, runner=None, log=print):
    """Execute (or preview) a plan. DRY-RUN unless `apply=True`.

    Refuses the whole plan if any path fails `_assert_sweepable` — see
    `delete_objects`.
    """
    doomed = [vd for vd in verdicts if vd.action == "sweep"]
    total = sum(vd.bytes for vd in doomed)
    n = sum(len(vd.paths) for vd in doomed)
    if not apply:
        log(f"[DRY-RUN] would delete {n} object(s), {total / 1e9:.2f} GB across "
            f"{len(doomed)} job(s). Re-run with --apply to execute.")
        for vd in doomed:
            log(f"   {vd.job_id}: {vd.reason}")
        return {"applied": False, "objects": n, "bytes": total,
                "jobs": len(doomed), "deleted": 0, "failed": 0}
    deleted, failed = [], []
    for vd in doomed:
        log(f">> sweeping {vd.job_id}: {vd.reason}")
        d, f = delete_objects(bucket, vd.paths, runner=runner, log=log)
        deleted += d
        failed += f
    log(f">> deleted {len(deleted)} object(s), {total / 1e9:.2f} GB; "
        f"{len(failed)} failure(s)")
    return {"applied": True, "objects": n, "bytes": total, "jobs": len(doomed),
            "deleted": len(deleted), "failed": len(failed)}


# --- inventory -------------------------------------------------------------- #

B2_STORAGE_USD_PER_GB_MONTH = 0.006     # Backblaze B2 list price, 2026-08.


def inventory(bucket, *, now, windows=(7.0, 14.0, 30.0, 60.0), runner=None,
              status_of=None, published_of=None, pruned_of=None, **kw):
    """What is there, how old it is, what it costs, and what each candidate
    window would reclaim. Read-only — this never deletes and never can."""
    jobs, err = list_checkpoint_objects(bucket, runner=runner)
    if err:
        return {"error": err}
    per_job, total, oldest, newest = {}, 0, None, None
    for jid, objs in jobs.items():
        b = sum(o.size for o in objs)
        aged = [o.mtime for o in objs if o.mtime is not None]
        mx = max(aged) if aged else None
        mn = min(aged) if aged else None
        per_job[jid] = {"bytes": b, "objects": len(objs),
                        "steps": sorted(f"{r}/checkpoint-{s}" if r
                                        else f"checkpoint-{s}"
                                        for r, s in {ckpt_key(o.path)
                                                     for o in objs}
                                        if s is not None),
                        "newest_ts": mx, "oldest_ts": mn,
                        "age_days": (round((now - mx) / 86400.0, 2)
                                     if mx is not None else None)}
        total += b
        if mn is not None:
            oldest = mn if oldest is None else min(oldest, mn)
        if mx is not None:
            newest = mx if newest is None else max(newest, mx)
    buckets = {"<7d": 0, "7-14d": 0, "14-30d": 0, "30-60d": 0, ">60d": 0,
               "unknown": 0}
    for j in per_job.values():
        d = j["age_days"]
        key = ("unknown" if d is None else
               "<7d" if d < 7 else "7-14d" if d < 14 else "14-30d" if d < 30
               else "30-60d" if d < 60 else ">60d")
        buckets[key] += j["bytes"]
    out = {"bucket": bucket, "total_bytes": total, "jobs": len(per_job),
           "objects": sum(j["objects"] for j in per_job.values()),
           "oldest_ts": oldest, "newest_ts": newest,
           "monthly_usd": round(total / 1e9 * B2_STORAGE_USD_PER_GB_MONTH, 2),
           "age_bytes": buckets, "per_job": per_job, "windows": {}}
    # Probe status/publication ONCE and memoize; the window comparison is then a
    # pure re-classification instead of four full passes over the bucket.
    evcache, stcache, pubcache = {}, {}, {}

    def _st(jid):
        if jid not in stcache:
            if status_of is not None:
                stcache[jid] = status_of(jid)
            else:
                evcache[jid] = job_events(bucket, jid, runner=runner)
                stcache[jid] = job_status(bucket, jid, events=evcache[jid])
        return stcache[jid]

    def _pub(jid):
        if jid not in pubcache:
            pubcache[jid] = (published_of(jid) if published_of is not None
                             else is_published(bucket, jid, runner=runner,
                                               run_name=run_name_of(
                                                   evcache.get(jid))))
        return pubcache[jid]

    prcache = {}

    def _pruned(jid):
        if jid not in prcache:
            prcache[jid] = (pruned_of(jid) if pruned_of is not None
                            else is_box_pruned(bucket, jid, runner=runner))
        return prcache[jid]

    for w in windows:
        vds, e = build_verdicts(bucket, now=now, window_days=w, runner=runner,
                                status_of=_st, published_of=_pub,
                                pruned_of=_pruned, jobs=jobs, **kw)
        out["windows"][str(w)] = ({"error": e} if e else plan_summary(vds))
    out["status_counts"] = {}
    for jid in per_job:
        key = str(_st(jid))
        out["status_counts"][key] = out["status_counts"].get(key, 0) + 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="ckpt_retention",
        description="Retention sweep for INTERMEDIATE training checkpoints "
                    "under jobs/<JOB_ID>/checkpoints/ on B2. NEVER touches "
                    "checkpoints/<RUN_NAME>/ (the published final adapters).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="dry-run is the default; --apply is required to delete anything.")
    ap.add_argument("cmd", choices=("inventory", "plan", "sweep"))
    ap.add_argument("--bucket", default=os.environ.get("B2_BUCKET"))
    ap.add_argument("--window-days", type=float, default=WINDOW_DAYS,
                    help=f"sweep only jobs whose newest checkpoint object is "
                         f"older than this (default {WINDOW_DAYS:g})")
    ap.add_argument("--keep-first", type=int, default=KEEP_FIRST, metavar="N",
                    help="retain the N OLDEST steps per job — the dose curve is "
                         "U-shaped, so the early steps carry the signal")
    ap.add_argument("--keep-last", type=int, default=KEEP_LAST, metavar="N")
    ap.add_argument("--keep-stride", type=int, default=KEEP_STRIDE, metavar="K",
                    help="retain every Kth step (a cheap dose-curve skeleton)")
    ap.add_argument("--include-failed", action="store_true",
                    help="ALSO sweep terminal-FAILED jobs. OFF by default: "
                         "`job requeue` re-opens a failed job and seeds the "
                         "retry from exactly this prefix")
    ap.add_argument("--sweep-box-pruned", action="store_true",
                    help="ALSO sweep jobs jobd pruned off the box disk "
                         "(CHECKPOINTS_PRUNED.json present). Their "
                         "checkpoints/ prefix is the SOLE copy of the pruned "
                         "steps, so this REQUIRES a keep policy — without one "
                         "the job is still kept. This is the gate that governs "
                         "the bulk of the bucket: every job since 2026-08-05 "
                         "carries the marker.")
    ap.add_argument("--job", action="append", dest="only", metavar="JOB_ID",
                    help="restrict to these job ids (repeatable)")
    ap.add_argument("--apply", action="store_true",
                    help="ACTUALLY DELETE. Irreversible.")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if not a.bucket:
        ap.error("no bucket: pass --bucket or set B2_BUCKET")
    import time
    now = time.time()
    if a.cmd == "inventory":
        rep = inventory(a.bucket, now=now, keep_first=a.keep_first,
                        keep_last=a.keep_last, keep_stride=a.keep_stride,
                        include_failed=a.include_failed)
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 1 if rep.get("error") else 0
    vds, err = build_verdicts(a.bucket, now=now, window_days=a.window_days,
                              include_failed=a.include_failed,
                              keep_first=a.keep_first, keep_last=a.keep_last,
                              keep_stride=a.keep_stride,
                              sweep_box_pruned=a.sweep_box_pruned,
                              only=set(a.only) if a.only else None)
    if err:
        print(f"!! could not build a plan ({err}) — DOING NOTHING", file=sys.stderr)
        return 1
    if a.json:
        print(json.dumps({"summary": plan_summary(vds),
                          "verdicts": [vd._asdict() for vd in vds]},
                         indent=2, sort_keys=True, default=list))
    else:
        for vd in vds:
            print(f"{vd.action:22s} {vd.job_id}  {vd.reason}")
        s = plan_summary(vds)
        print(f"-- {s['sweep_jobs']}/{s['jobs']} job(s) sweepable: "
              f"{s['sweep_objects']} object(s), {s['sweep_bytes'] / 1e9:.2f} GB")
    if a.cmd == "plan":
        return 0
    sweep(a.bucket, vds, apply=a.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
