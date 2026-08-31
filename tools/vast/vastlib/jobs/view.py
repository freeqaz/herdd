"""vastlib.jobs.view — READ a job: fold it, cache it, render it, list it.

Why this exists
---------------
Everything in this module answers "what is true about this job/queue right
now", and answers it from B2 plus at most one vast-API liveness read. Nothing
here writes a ticket, emits an event, or ssh's anywhere — that is
`jobs/control.py`, deliberately next door and deliberately separate, because
the read path is the one an operator runs a hundred times a day and the write
path is the one that can double-run a training job.

The three contracts this module is not allowed to "clean up"
------------------------------------------------------------
* **`_present_iids_set` is TRI-STATE**, and it is the highest-blast-radius
  contract in the jobs lane: a `set` of every instance id in the account, or
  `None` because the listing could not be READ, or `None` because we are in the
  LOCAL lane. Both `None`s flow into
  `parked_lifecycle.ticket_orphan_verdict(box_present=None)`, which mints NO
  verdict — so one API 500 says "unknowable" instead of "every box is
  destroyed". Typing it `set[str]` and coercing `None` to `set()` classifies the
  whole fleet's queue as ORPHANED, and `job orphans --resolve -y` then CANCELS
  those jobs. That is why it calls `api.request_soft` DIRECTLY rather than
  `lifecycle._instances_soft`, whose whole job is to flatten an API error into
  `[]`.
* **`_live_iids_set` returns STRINGS.** The vast API types an instance id as an
  `int`; every other spelling of a box id in the jobs lane is a `str` (a queue
  path segment, a ticket's `box` field, an event's `instance_id`, an `--box`
  argv). The `str()` is the fix for a real defect in which `job ls` reported
  every box in the account dead — including one actively training — because the
  membership tests compared `str` against `int`.
* **`_JOB_VIEW_STICKY` is not `jobmeta.TERMINAL`.** `failed` is re-openable by
  `job requeue`, so freezing it on disk is what turned a healthy run into a
  fleet-wide ZOMBIE_NO_JOBD alarm on 2026-08-07. The 22-line comment above the
  constant is that incident, and it ports with the constant.

The mutable global, and the one import form that keeps it working
-----------------------------------------------------------------
`_JOB_LOCAL` lives in `jobs/runlocal.py` and is flipped at runtime by
`_job_local_activate()`. The two functions here that would otherwise reach for
the vast API read it as **`runlocal._JOB_LOCAL`, at call time** (plan §8b). A
`from .runlocal import _JOB_LOCAL` binds `False` once at import and the local
lane silently starts hitting the real API — exactly the credential touch
`LOCAL_GPU_LANE.md` promises never to make. That read is also what makes
`monkeypatch.setattr(runlocal, "_JOB_LOCAL", True)` steer this module, which is
how three existing tests drive the local lane.

`view` and `runlocal` therefore import each other. The cycle is real and it is
resolved the same way: neither module touches the other's attributes at import
time, only inside function bodies. `test_vastlib_jobs_view.py` imports the pair
in both orders to prove it.

What is deliberately NOT here
-----------------------------
* **`_TQDM_RE`, `_tqdm_points`, `_step_delta_s`.** They live in `jobs/risk.py`
  (same ring, no DAG problem) because `_job_eta_s`/`_job_pct` cannot be ported
  without them, and a second copy of the bar regex is exactly the drift this
  refactor exists to remove. `_NUM_TOKENS_RE` stays HERE — `_job_progress` is
  its only consumer.
* **`_hms_secs`, `_age_str`, `_ts_to_epoch`.** `core/fmt.py` owns the pure
  render atoms; it explicitly excludes `_step_rate`/`_job_progress`/`_job_cell`/
  `_hb_age_s` as job-domain semantics, which is what routes them here.
* **Every B2 mutation.** `jobs/control.py` owns retarget/requeue/cancel/orphan
  resolution, and `jobs/submit.py` owns submission.
* **`require_local_gpu`.** One switch in one place (`core.config`).

Provenance: behavior-preserving move of 30 symbols from `tools/vast/herdd.py`
(plan §8 step 5, 2026-08-16), each carrying its `# moved-from:` marker.
ADD-ONLY: `herdd.py` keeps its live copies until step 6. Bodies are copied
verbatim; annotations were added, and the two mechanical changes strict typing
and the package layout forced are documented at their sites (`_REPO_ROOT`, and
the `_fold_fleet_jobs` seam wiring at the foot of the file).
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import joblocal

from vastlib.boxes import lifecycle, reap
from vastlib.core import api
from vastlib.jobs import risk, runlocal, scan
from vastlib.storage import b2

import bidpolicy
import jobmeta
import runmeta

#: One parsed tqdm bar, as `jobs/risk.py` produces it.
TqdmPoint = risk.TqdmPoint


# --------------------------------------------------------------------------- #
# THE REPO ROOT. The flat `herdd._repo_root()` was three `os.path.dirname`
# calls from `tools/vast/herdd.py`. From `tools/vast/vastlib/jobs/view.py`
# the SAME expression yields `tools/vast`, and nothing raises — `job pull` would
# quietly write into `tools/vast/out/jobs/<id>/` and `find_job_defs` would find
# zero bundles because `tools/vast/tools/witness/jobs` does not exist. A
# listing that finds nothing looks exactly like a repo with no bundles.
#
# So the depth is RECOMPUTED for this file's location (five dirnames) and
# hoisted to a module constant, for the same reason `boxes/ssh.py::_REPO_ROOT`
# and `core/config.py::_HERE` are constants: the depth is a property of where
# the module lives, not of the function, and a package that moves again should
# have to fix exactly one line.
#
# `test_vastlib_jobs_view.py::test_repo_root_matches_herdd_computation` pins
# it against the three-dirname expression applied to `herdd.py`'s own path,
# and `test_naive_file_arithmetic_here_would_be_wrong` proves the trap is real.
#
# DELIBERATELY MARKER-LESS (ruled 2026-08-16, wave 6a): this constant carried a
# `# moved-from: herdd._repo_root -> _REPO_ROOT` marker, which made
# `herdd._repo_root` a DUPLICATE original with two vastlib targets — the test
# migration cannot rewrite a name with two homes. `jobs/submit.py::_repo_root`
# owns that mapping: it is the FUNCTION the flat name was, with three
# `monkeypatch.setattr` sites that need a module attribute to replace. What
# lives here is a same-valued but distinct constant, recomputed for this file's
# depth — no rename claim on the flat symbol. Recorded in
# `gen_rename_table.py::KNOWN_MARKERLESS` so the omission reads as ruled.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


# --------------------------------------------------------------------------- #
# The vast-API liveness/presence reads — the ONLY two places in the jobs lane
# that ask the API rather than B2, and therefore the only two that have to know
# about the local lane.
# --------------------------------------------------------------------------- #

# moved-from: herdd._live_iids_set
def _live_iids_set() -> Any:  # noqa: ANN401 — set[str], or joblocal's own set
    """All vast instance ids currently live (any label) — the liveness injection
    for a job fold (a claimed/started job whose box is gone folds to `interrupted`)."""
    if runlocal._JOB_LOCAL:
        # LOCAL GPU LANE: there is no vast API to ask, and asking would be a
        # credential touch the lane promises never to make. jobd's own flock on
        # $JOBD_ROOT/.jobd.lock is the equivalent, and equally REAL: kill
        # `run-local` mid-job and the fold correctly says `interrupted`.
        return joblocal.live_boxes()
    out = set()
    for i in lifecycle._instances_soft():
        if (i.get("actual_status") or "").lower() in bidpolicy.LIVE_STATES:
            # STRINGS. The vast API types an instance id as an int; every OTHER
            # spelling of a box id in the jobs lane is a string (a queue path
            # segment `jobs/queue/<box>/`, a ticket's `box` field, an event's
            # `instance_id`, an `--box` argv). `fold_events` normalizes on the
            # way in, so the mismatch was invisible there and NOT invisible in
            # the direct membership tests: `job ls`'s per-box `live=` and `job
            # cancel --hard`'s live check compared str against int and were
            # therefore ALWAYS False — `job ls` reported every box in the
            # account as dead, including one actively running a training job.
            out.add(str(i.get("id")))
    return out


# moved-from: herdd._present_iids_set
def _present_iids_set() -> set[str] | None:
    """Every vast instance id in the ACCOUNT, any `actual_status` — or **None**
    when the listing could not be read.

    Presence is a different question from liveness and the only one that can
    answer "can this box ever claim its queue again?". A parked/stopped box is
    absent from `_live_iids_set` and still perfectly able to: `herdd start`
    re-runs onstart, jobd comes back, the tickets get claimed. Only absence from
    the account is permanent.

    The None is load-bearing. `_instances_soft` swallows every API error into an
    empty list, which is indistinguishable from an empty account — and reading
    that as "every box is destroyed" would classify the whole fleet's queue as
    orphaned on one 500. So this uses the soft request DIRECTLY and reports the
    failure as unknowable (parked_lifecycle.ticket_orphan_verdict's
    `box_present=None`). LOCAL lane: presence is meaningless (the machine is
    always there) -> None, so the local lane never mints an orphan verdict."""
    if runlocal._JOB_LOCAL:
        return None
    ok, d, _ = api.request_soft("GET", "v1/instances/")
    if not ok:
        return None
    inst = d.get("instances", d) if isinstance(d, dict) else d
    if not isinstance(inst, list):
        return None
    return {str(i.get("id")) for i in inst if isinstance(i, dict) and i.get("id") is not None}


# moved-from: herdd._box_lifecycle_soft
def _box_lifecycle_soft(iid: object) -> dict[str, Any]:
    """Fold the per-box lifecycle stream (jobs/nodes/<iid>/events/) — never
    raises (a read failure must not crash the supervise loop). Returns at least
    {parked, drained_pending}."""
    try:
        return jobmeta.read_box(str(iid))
    except Exception:
        return {"parked": False, "drained_pending": False, "park_reason": None}


# --------------------------------------------------------------------------- #
# job-view disk cache: what may be frozen, and what must never be
# --------------------------------------------------------------------------- #
# `_fold_fleet_jobs` caches a folded view forever, so the predicate below is a
# claim that the fold CANNOT CHANGE AGAIN. "Terminal" is not that claim.
# `jobmeta.fold_events` documents exactly ONE un-stick — `herdd job requeue`
# re-opens a terminal-FAILED job under the SAME JOB_ID by emitting a newer
# `resumed` — so `failed` is re-openable and must never be frozen. `done` and
# `cancelled` are unconditionally sticky and are the only two that may be.
#
# Freezing `failed` is what turned a healthy run into a fleet-wide zombie alarm
# on 2026-08-07: 20260806T212132-v9-gemma4-dec-train-8818 failed on 47041615 at
# 03:51:33 (insufficient_disk) and its view was frozen at 03:53 with
# last_heartbeat_ts=03:40:02. Two operator requeues later the job was running on
# 47045282 and heartbeating every 60 s — but `ls`/fleetd still read the frozen
# view, so `_fleet_jobd_hb_epoch` never saw a fresh fold heartbeat, fell back to
# the JOBD_STATUS marker, and raised ZOMBIE_NO_JOBD ("destroy + relaunch")
# against a box at step 142/156 of a 156-step run. Same family as the
# stale-`failed`-outranks-newer-`done` fold defect: a dead attempt's record
# outliving the attempt that replaced it.
#
# `_JOB_VIEW_CACHE_V` is a format stamp, not decoration: entries written under
# the old rule are still on disk, frozen at a `failed` that has since been
# requeued, and an unstamped body must be re-read rather than trusted.
# moved-from: herdd._JOB_VIEW_STICKY
_JOB_VIEW_STICKY = frozenset({"done", "cancelled"})
# moved-from: herdd._JOB_VIEW_CACHE_KEY
_JOB_VIEW_CACHE_KEY = "_cache_v"
# moved-from: herdd._JOB_VIEW_CACHE_V
_JOB_VIEW_CACHE_V = 2


# moved-from: herdd._job_view_cacheable
def _job_view_cacheable(view: Any) -> bool:  # noqa: ANN401 — folded view or None
    return bool(view) and view.get("status") in _JOB_VIEW_STICKY


# moved-from: herdd._fold_fleet_jobs
def _fold_fleet_jobs(live_iids: Any,  # noqa: ANN401 — set[str] | set[int], caller-shaped
                     prog: Any = None) -> dict[str, list[Any]]:  # noqa: ANN401 — Progress
    """box-id (str) -> list of folded jobd job views, best-effort. Each queued/
    terminal ticket is attached to BOTH its target box and the box it actually
    claimed (so a retargeted job still shows under the box it ran on), but at
    most ONCE per box — see the dedupe below. Empty on ANY b2/jobmeta failure —
    a read-only listing must never hard-fail.

    Cost control (one rclone spawn per job otherwise dominates ls wall-time): a
    fold that can never change again is cached at
    ~/.cache/vast-jobmeta/<job_id>/view.json and never re-read from B2;
    everything else folds in parallel. "Can never change again" is NOT the same
    as terminal — see _JOB_VIEW_STICKY."""
    out: dict[str, list[Any]] = {}
    try:
        b2._ensure_b2_remote()
        pairs = jobmeta.list_all_queued()
    except Exception:
        return out
    # XDG_CACHE_HOME, read here exactly as `boxes/reap.py` reads it for its two
    # ledgers: same name, same default, same precedence (`core/config.py`'s
    # env-read inventory records the site). Read at CALL time, not at import, so
    # a test that setenv's it is honoured.
    cache_root = os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "vast-jobmeta")

    def view_of(jid: str) -> Any:  # noqa: ANN401 — folded view or None
        try:
            vpath = os.path.join(cache_root, jid, "view.json")
            try:
                with open(vpath) as fh:
                    v = json.load(fh)
                if _job_view_cacheable(v) and v.get(_JOB_VIEW_CACHE_KEY) == \
                        _JOB_VIEW_CACHE_V:
                    return v
            except Exception:
                pass
            try:
                v = jobmeta.read_job(jid, live_iids=live_iids)
            except Exception:
                return None
            if _job_view_cacheable(v):
                try:
                    os.makedirs(os.path.dirname(vpath), exist_ok=True)
                    tmp = vpath + ".tmp"
                    with open(tmp, "w") as fh:
                        json.dump(dict(v, **{_JOB_VIEW_CACHE_KEY:
                                             _JOB_VIEW_CACHE_V}), fh)
                    os.replace(tmp, vpath)
                except Exception:
                    pass
            return v
        finally:
            if prog:
                prog.tick()

    jids = sorted({jid for _, jid in pairs})
    if prog:
        prog.add(len(jids))
    views = {}
    if jids:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(12, len(jids))) as ex:
            for jid, v in zip(jids, ex.map(view_of, jids)):
                if v is not None:
                    views[jid] = v
    # DEDUPE PER BOX, by job_id. A queue prefix keeps every ticket ever placed
    # under it, so a RETARGETED job is legitimately listed under both its
    # original target and its new one. The original's pair then attaches the
    # view to `instance_id` (the box it actually claimed) and the new target's
    # own pair attaches it there a second time — one ticket, two rows, which
    # reads as the box running the job twice. Seen live 2026-08-27: two gtp KD
    # tickets retargeted 48839596 -> 48852151, each rendered twice by `ls`
    # against exactly one trainer process apiece on the box.
    seen: dict[str, set[str]] = {}
    for box, jid in pairs:
        v = views.get(jid)
        if v is None:
            continue
        keys = {str(box)}
        if v.get("instance_id"):
            keys.add(str(v["instance_id"]))
        for k in keys:
            if jid in seen.setdefault(k, set()):
                continue
            seen[k].add(jid)
            out.setdefault(k, []).append(v)
    return out


# --------------------------------------------------------------------------- #
# The tqdm progress renderers. `_TQDM_RE` / `_tqdm_points` / `_step_delta_s`
# live in `jobs/risk.py` (ONE bar regex, ONE delta implementation); only the
# token regex and the three display functions are here.
# --------------------------------------------------------------------------- #

# HF trainer log-dict line carrying the cumulative token count:
#   `{'loss': ..., 'num_tokens': '4.363e+07', ...}`
# moved-from: herdd._NUM_TOKENS_RE
_NUM_TOKENS_RE = re.compile(r"['\"]num_tokens['\"]:\s*['\"]?([0-9.eE+]+)")


# moved-from: herdd._job_cell
def _job_cell(v: Mapping[str, Any]) -> str:
    """PURE-ish. One folded job view -> the `ls --minimal` jobs cell:
    `name:status`, extended with `:NN%:rate` and `:ckptN` when `_job_progress`
    can parse them. Shared by `_render_minimal` and the dash-cache `instances`
    projection so the dashboard's jobs string can never drift from the CLI's.

    Emits ONLY the parsed scalars — the raw container tail (`last_tail`) is
    never part of the string, and never reaches the dashboard cache."""
    s = f"{v.get('name') or v.get('job_id')}:{v.get('display_status')}"
    pg = _job_progress(v)
    if "pct" in pg:
        s += f":{pg['pct']}%" + (f":{pg['rate']}" if "rate" in pg else "")
    if "ckpt" in pg:
        s += f":ckpt{pg['ckpt']}"
    return s


# moved-from: herdd._step_rate
def _step_rate(pts: Sequence[TqdmPoint] | None) -> tuple[str, str] | None:
    """PURE. `(text, kind)` for the freshest step-rate figure in `pts`.

    **tqdm's own rate is not the step time, and on a resume it is not even
    close.** It is an aggregate over the whole attempt (a bias-corrected EMA of
    per-refresh `dn`/`dt`, tqdm's `smoothing=0.3` default), and on a
    checkpoint-resume the dataloader fast-forward enters it as one enormous `dn`
    over a short `dt` — HF's `ProgressCallback` advances the bar by
    `global_step - current_step` on the first real step after the skip, so
    twenty skipped steps land as a single 20-step "iteration". The figure then
    starts far below truth and climbs for the rest of the epoch.

    Observed on box 47021787, 2026-08-06, v9-gemma4-dec, ONE attempt with no
    change in the underlying step time:

        7.50 → 10.08 → 17.07 → 21.21 → 28.24 → … → 98.54 s/it

    while the consecutive-step delta over the same window was a flat ~101 s.
    Anyone reading the `ls` phase column early in that resume would have
    concluded the box was 13× faster than it was.
    `PERF_LEVERS_INVESTIGATION_2026-08-06.md` §2.2 measured the same effect on
    box 46947265 and drew the rule: *never quote tqdm s/it from a resumed run*.
    Its diagnosis of the mechanism as a running AVERAGE is off — the fixture in
    `test_jobprogress_rate.py` shows the bar tracking local per-step variation,
    which a cumulative average cannot do — but the rule is right and this is the
    fix for it.

    So: subtract the elapsed stamps of the last two bars and divide by the step
    gap. Fall back to tqdm's figure only when there is no second bar (a first
    heartbeat) or when the steps are too fast for tqdm's 1-second elapsed
    resolution — and LABEL the fallback `~…(avg)` so a reader can tell which
    number they are looking at."""
    if not pts:
        return None
    pct, step, total, el, rate, unit = pts[-1]
    dt = risk._step_delta_s(pts)   # ONE delta implementation (see _step_delta_s)
    if dt is not None:
        return (f"{dt:.0f}s/it" if dt >= 10 else f"{dt:.1f}s/it"), "delta"
    return f"~{rate:.1f}{unit}(avg)", "avg"


# moved-from: herdd._job_progress
def _job_progress(v: Mapping[str, Any]) -> dict[str, Any]:
    """Best-effort training progress from one folded job view — no extra
    network: the tqdm lines in the heartbeat log tail give completion % +
    step/total + step rate; the HF trainer's cumulative `num_tokens` over the
    tqdm elapsed time gives an average tokens/s; jobd checkpoint events give
    the checkpoint count. {} when nothing is parseable (non-training job, no
    heartbeat yet).

    `rate` is the CONSECUTIVE-STEP DELTA where two bars are available and
    tqdm's own aggregate (prefixed `~`, suffixed `(avg)`) where they are not —
    see `_step_rate` for why the difference matters. `rate_kind` carries the
    same distinction as a bare `delta`/`avg` token for programmatic readers."""
    out: dict[str, Any] = {}
    tail = v.get("last_tail") or ""
    pts = risk._tqdm_points(tail)
    if pts:
        pct, step, total, el, _rate, _unit = pts[-1]
        out.update(pct=pct, step=step, total=total)
        sr = _step_rate(pts)
        if sr:
            out["rate"], out["rate_kind"] = sr
        tk = None
        for tk in _NUM_TOKENS_RE.finditer(tail):
            pass                      # last match wins; `tk` leaks deliberately
        if tk and el > 0:
            out["toks"] = float(tk.group(1)) / el
    if v.get("n_checkpoints"):
        out["ckpt"] = v["n_checkpoints"]
    return out


# --------------------------------------------------------------------------- #
# One job: fold, render, watch, wait
# --------------------------------------------------------------------------- #

# moved-from: herdd._job_view
def _job_view(job_id: str) -> Any:  # noqa: ANN401 — jobmeta's folded view dict
    b2._ensure_b2_remote()
    try:
        return jobmeta.read_job(job_id, live_iids=_live_iids_set())
    except (jobmeta.JobmetaError, runmeta.RunmetaError) as e:
        sys.exit(f"error: {e}")


# moved-from: herdd._print_job_view
def _print_job_view(v: Mapping[str, Any]) -> None:
    print(f"== job {v['job_id']} ==")
    print(f"  status={v['display_status']} (fold={v['status']}) "
          f"live={v['live']} box={v['instance_id'] or v['target_box'] or '-'}")
    print(f"  name={v['name']} entrypoint={v['entrypoint']} "
          f"bundle={(v['bundle_sha256'] or '-')[:12]} events={v['n_events']} "
          f"parse_errors={v['parse_errors']}")
    if v.get("exp_id"):        # matrix-arm association (MATRIX_DESIGN.md)
        print(f"  experiment={v['exp_id']} arm={v.get('arm') or '-'} "
              f"(full matrix: jobmatrix.py status {v['exp_id']})")
    if v.get("reopened"):        # `job requeue` un-stuck it — say whose rc that was
        print(f"  RE-OPENED by requeue at {v.get('last_resumed_ts')} "
              f"(prior attempt: rc={v.get('prior_rc')} "
              f"reason={v.get('prior_fail_reason') or '-'})")
    if v["rc"] is not None:
        print(f"  rc={v['rc']} fail_reason={v['fail_reason'] or '-'}")
    elif v["display_status"] == "cancelled" and v.get("fail_reason"):
        print(f"  cancel_reason={v['fail_reason']}")
    if v["last_heartbeat_ts"]:
        print(f"  last_heartbeat={v['last_heartbeat_ts']}")
    if v.get("last_metrics"):
        print(f"  host_metrics={v['last_metrics']}")
    if v["results"]:
        print(f"  results={len(v['results'])} file(s)")
    print(f"  last_event={v['last_event']}@{v['last_event_ts']}")


# moved-from: herdd._job_view_fresh
def _job_view_fresh(job_id: str) -> Any:  # noqa: ANN401 — jobmeta's folded view dict
    b2._ensure_b2_remote()
    try:
        return jobmeta.read_job_fresh(job_id, live_iids=_live_iids_set())
    except (jobmeta.JobmetaError, runmeta.RunmetaError) as e:
        sys.exit(f"error: {e}")


# moved-from: herdd._print_fresh_notes
def _print_fresh_notes(v: Mapping[str, Any]) -> None:
    """The three things a folded view cannot say for itself. See
    jobmeta.read_job_fresh — and the 2026-07-30 launch postmortem §7a.

    The DONE-marker note is THREE notes, because a marker beside a non-terminal
    fold has two causes and they call for opposite actions: B2 LIST lag (the job
    finished, pull it) and a re-opened job whose marker belongs to the attempt
    that DIED (the job is running, pulling it hands over debris). Saying
    "the job FINISHED" for the second is a false terminal on a healthy run —
    measured live 2026-08-28, see `jobmeta.DONE_MARKER_CURRENT`."""
    if v.get("unclaimed"):
        print("  live=n/a — no claim/started/heartbeat event is visible, which says "
              "NOTHING about the box (use `herdd ls` for box liveness)")
    if v.get("done_marker") and v["status"] not in jobmeta.TERMINAL:
        _print_done_marker_note(v)
    print("  (--fresh: uncached, per-key event read — narrows the B2 LIST window, "
          "cannot eliminate it)")


def _print_done_marker_note(v: Mapping[str, Any]) -> None:
    """One note per `jobmeta.classify_done_marker` verdict. Absent verdict (an
    older fold, or a caller that filled `done_marker` by hand) reads as
    `current`, which is the pre-2026-08-28 wording unchanged."""
    verdict = v.get("done_marker_verdict") or jobmeta.DONE_MARKER_CURRENT
    rc = v.get("done_marker_rc")
    rc_s = f" rc={rc}" if rc is not None else ""
    box = v.get("done_marker_box")
    if verdict == jobmeta.DONE_MARKER_STALE:
        print(f"  !! results.DONE.json on B2 is from a PRIOR attempt — written "
              f"{v.get('done_marker_ts')}{rc_s}"
              + (f" by box {box}" if box else "")
              + f", BEFORE this job was re-opened at {v.get('reopened_at')}. It "
              f"says nothing about the attempt now running ({v['status']}).")
        print(f"     `job pull` would hand you the DEAD attempt's results/ tree. "
              f"The live attempt's durable state is under "
              f"jobs/{v.get('job_id')}/checkpoints/.")
    elif verdict == jobmeta.DONE_MARKER_UNKNOWN:
        print(f"  !! results.DONE.json EXISTS on B2{rc_s} and this job was RE-OPENED "
              f"at {v.get('reopened_at')}, but the marker could not be DATED — it "
              f"may belong to either attempt. Do NOT read it as finished; a "
              f"requeue does not clear the prior attempt's marker.")
        print(f"     adjudicate: rclone lsjson "
              f"b2:$B2_BUCKET/jobs/{v.get('job_id')}/results.DONE.json")
    else:
        print(f"  !! results.DONE.json EXISTS on B2 while the event fold still says "
              f"{v['status']} — B2 LIST lag: the job FINISHED. `job pull` will work.")


# moved-from: herdd.cmd_job_status
def cmd_job_status(a: Any) -> None:  # noqa: ANN401 — argparse.Namespace
    fresh = getattr(a, "fresh", False)
    view: Callable[[str], Any] = _job_view_fresh if fresh else _job_view
    if not a.watch:
        v = view(a.job_id)
        if a.json:
            print(json.dumps(v, indent=2)); return    # noqa: E702 — verbatim (plan §7.4)
        _print_job_view(v)
        if fresh:
            _print_fresh_notes(v)
        return
    # --watch: poll until terminal (runmeta-style live injection each pass)
    while True:
        v = view(a.job_id)
        if a.json:
            print(json.dumps(v))
        else:
            _print_job_view(v)
            if fresh:
                _print_fresh_notes(v)
        if v["status"] in jobmeta.TERMINAL:
            return
        time.sleep(a.interval)


# moved-from: herdd.cmd_job_wait
def cmd_job_wait(a: Any) -> None:  # noqa: ANN401 — argparse.Namespace
    """Block until a job reaches --until, or --timeout expires. A first-class
    replacement for hand-rolled `until herdd job status ... | grep` loops:
    `job status --watch` only ever blocks until TERMINAL and cannot gate on an
    intermediate state (e.g. 'running'). Exit codes: 0 = reached the state;
    2 = --until terminal but the job FAILED (so a shell `&&` chain stops on a
    bad outcome); 124 = timed out (coreutils convention); 1 = the state became
    unreachable (job went terminal as something other than what was asked)."""
    want = a.until.strip().lower()
    if want not in jobmeta.WAIT_STATES:
        sys.exit(f"error: --until {want!r} not one of {', '.join(jobmeta.WAIT_STATES)}")
    deadline = time.time() + a.timeout
    while True:
        v = _job_view(a.job_id)
        d = jobmeta.wait_decision(v, want)   # type: ignore[no-untyped-call]
        if d == "match":
            if a.json:
                print(json.dumps(v))
            else:
                print(f">> job {a.job_id} reached {want} (status={v['status']} "
                      f"display={v['display_status']} rc={v['rc']})")
            if want == "terminal" and v["status"] == "failed":
                sys.exit(2)     # reached terminal, but as a FAILURE — signal it
            return
        if d == "unreachable":
            sys.exit(f"!! job {a.job_id} is terminal ({v['display_status']}) — will "
                     f"never reach {want!r} (rc={v['rc']} "
                     f"reason={v['fail_reason'] or '-'})")
        if time.time() >= deadline:
            print(f"!! job {a.job_id} did not reach {want!r} within {a.timeout}s "
                  f"(now: {v['display_status']})", file=sys.stderr)
            sys.exit(124)
        time.sleep(a.interval)


# --------------------------------------------------------------------------- #
# Logs: presence is not provenance
# --------------------------------------------------------------------------- #

# moved-from: herdd._hb_age_s
def _hb_age_s(ts: object) -> float | None:
    """Seconds since a runmeta colon-free ms UTC stamp (20260806T170440975Z), or
    None if unparseable. Used to age-stamp a heartbeat tail."""
    if not ts:
        return None
    try:
        import datetime  # function-local, verbatim (plan §7.4)
        t = datetime.datetime.strptime(str(ts)[:15], "%Y%m%dT%H%M%S").replace(
            tzinfo=datetime.timezone.utc)
        return (datetime.datetime.now(datetime.timezone.utc) - t).total_seconds()
    except Exception:
        return None


# moved-from: herdd._job_log_provenance
def _job_log_provenance(v: Mapping[str, Any], job_id: str) -> list[str]:
    """The header every `job logs` dump MUST carry: WHICH attempt's bytes these
    are, WHICH box emitted them, and HOW OLD they are.

    Why this exists (2026-08-06, twice in one hour on job
    20260806T082213-v11-...-aff8): `job logs` renders prior-attempt bytes with no
    boundary and no age, so the reader cannot tell a live log from a dead one.
    It produced a FALSE FAILURE (a `ChildFailedError` grepped out of a previous
    attempt's log while the current attempt was healthy) and a FALSE RESUME
    CONFIRM (a match served from a box that had already been evicted). Same class
    as the dead-box resume line and the hollow-checkpoint listing: **presence is
    not provenance.**

    The sharpest case is the non-terminal branch, where the tail comes from the
    last heartbeat — which a retarget/requeue does NOT invalidate. So when the
    emitting box differs from the box the job is now aimed at, the bytes below
    PREDATE the move and say nothing about the current attempt. That is called
    out loudly rather than left for the reader to infer."""
    lines = []
    hb_box = str(v.get("instance_id") or "") or None
    tgt = str(v.get("target_box") or "") or None
    age = _hb_age_s(v.get("last_heartbeat_ts"))
    age_s = f"{int(age)}s ago" if age is not None else "age unknown"
    lines.append(f"== job {job_id} · status={v.get('display_status')} "
                 f"· job box={tgt or '?'} ==")
    if v.get("last_heartbeat_ts"):
        lines.append(f"-- tail from heartbeat @ {v['last_heartbeat_ts']} "
                     f"({age_s}), emitted by box {hb_box or '?'} --")
    if hb_box and tgt and hb_box != tgt:
        lines.append(
            f"!! PROVENANCE: these bytes were emitted by box {hb_box}, but the "
            f"job is now targeted at box {tgt}. They PREDATE the move and "
            f"describe a PRIOR attempt — a failure below may already be "
            f"resolved, and a success below is not evidence the current attempt "
            f"is running. Confirm against `job status {job_id} --fresh`.")
    if age is not None and age > 600:
        lines.append(f"!! STALE: newest heartbeat is {int(age // 60)} min old — "
                     f"the emitting box may be gone.")
    return lines


# moved-from: herdd.cmd_job_logs
def cmd_job_logs(a: Any) -> None:  # noqa: ANN401 — argparse.Namespace
    v = _job_view(a.job_id)
    if v["status"] in jobmeta.TERMINAL:
        # RAW `os.environ` read, ported verbatim. Every other bucket read in the
        # tree goes through the config/b2 layer; this one does not, and making
        # it do so is a behavior change (a different precedence for one env key)
        # that plan §7.4 puts out of scope for the move. Flagged, not fixed.
        bucket = os.environ.get("B2_BUCKET")
        # name the object and its size/mtime BEFORE dumping: a terminal log.txt
        # is whichever attempt finalized, which after a requeue is not
        # necessarily the attempt the reader has in mind.
        rc_m, meta = b2._rclone(["lsl", f"b2:{bucket}/jobs/{a.job_id}/log.txt"])
        stamp = meta.strip().splitlines()[0].strip() if rc_m == 0 and meta.strip() else ""
        rc, out = b2._rclone(["cat", f"b2:{bucket}/jobs/{a.job_id}/log.txt"])
        if rc == 0 and out:
            print(f"== job {a.job_id} · status={v['display_status']} · "
                  f"jobs/{a.job_id}/log.txt"
                  + (f" · {stamp}" if stamp else "")
                  + " ==")
            print("-- terminal log.txt: the bytes of whichever attempt FINALIZED. "
                  "After a requeue/retarget this may not be the attempt you mean; "
                  "per-attempt files live beside it in the results/ tree. --")
            sys.stdout.write(out)
        else:
            print(f"(no log.txt for {a.job_id}; status={v['display_status']})")
        return
    # still running: show the latest heartbeat tail, with full provenance
    for line in _job_log_provenance(v, a.job_id):
        print(line)
    print(v["last_tail"] or "(no heartbeat tail yet)")


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

# moved-from: herdd.cmd_job_pull
def cmd_job_pull(a: Any) -> None:  # noqa: ANN401 — argparse.Namespace
    dest = a.dest or os.path.join(_REPO_ROOT, "out", "jobs", a.job_id)
    b2._ensure_b2_remote()
    _pull_attempt_guard(a.job_id, allow_stale=getattr(a, "allow_stale", False))
    try:
        manifest = jobmeta.pull_results(a.job_id, dest)
    except (jobmeta.JobmetaError, runmeta.RunmetaError) as e:
        sys.exit(f"error: {e}")
    print(f">> pulled {len(manifest)} result file(s) -> {dest}")
    for f in manifest:
        print(f"   {f}")
    if not manifest:
        _job_pull_explain_empty(a.job_id)


def _pull_attempt_guard(job_id: str, *, allow_stale: bool = False) -> None:
    """Refuse to hand over a results/ tree that belongs to a DEAD attempt.

    `jobs/<id>/results/` and its DONE marker are mutable keys that a requeue does
    not clear, so on a re-opened job they hold whatever the FAILED attempt
    published — 57.7 KiB of scheduler/rng debris in the 2026-08-28 incident,
    against a live 2.1 GiB checkpoint under `checkpoints/`. A warning on `job
    status` is worth little if `pull` still ships the debris in silence and exit
    0, so the refusal lives here and `--allow-stale` is the deliberate override.

    FAILS OPEN by construction: a probe that raises leaves the pull exactly as it
    was before this guard existed. Only a POSITIVE `stale` verdict refuses; an
    undatable marker warns and proceeds, because refusing every pull on a B2
    listing hiccup is a worse failure than the one being prevented."""
    try:
        view = jobmeta.read_job(job_id, live_iids=_live_iids_set())
        reopened_at = view.get("reopened_at")
        if not reopened_at:
            return
        probe = jobmeta.probe_done_marker(job_id, reopened_at=reopened_at)
    except Exception:
        return
    if not probe.get("present"):
        return
    verdict, ts = probe.get("verdict"), probe.get("ts")
    rc_s = "" if probe.get("rc") is None else f" (rc={probe['rc']})"
    if verdict == jobmeta.DONE_MARKER_UNKNOWN:
        print(f"!! job {job_id} was RE-OPENED at {reopened_at} and its "
              f"results.DONE.json{rc_s} could not be dated — the tree below may "
              f"be the PRIOR attempt's. Check it against "
              f"`{os.path.basename(sys.argv[0])} job status {job_id} --fresh`.")
        return
    if verdict != jobmeta.DONE_MARKER_STALE or allow_stale:
        if verdict == jobmeta.DONE_MARKER_STALE:
            print(f"!! --allow-stale: pulling the PRIOR attempt's results/ "
                  f"(marker written {ts}{rc_s}, job re-opened {reopened_at}).")
        return
    prog = os.path.basename(sys.argv[0])
    sys.exit(
        f"!! REFUSING to pull {job_id}: its results/ tree belongs to a DEAD "
        f"attempt.\n"
        f"   results.DONE.json was written {ts}{rc_s}, BEFORE the job was "
        f"re-opened at {reopened_at} (fold={view.get('display_status')}).\n"
        f"   A requeue/retarget does not clear the prior attempt's marker or its "
        f"results/, so what is on B2 under results/ is that attempt's debris — "
        f"pulling it and calling the job complete is the failure this refusal "
        f"exists to stop.\n"
        f"   live state : rclone lsl b2:$B2_BUCKET/jobs/{job_id}/checkpoints/\n"
        f"   adjudicate : {prog} job status {job_id} --fresh\n"
        f"   pull anyway: {prog} job pull {job_id} --allow-stale")


# moved-from: herdd._job_pull_explain_empty
def _job_pull_explain_empty(job_id: str) -> None:
    """Explain a 0-file pull instead of leaving it to read as data loss.

    `pull_results` reads jobs/<id>/results/, which jobd writes exactly ONCE at
    finalize; a job that is still running (or interrupted mid-run) keeps its
    durable state under jobs/<id>/checkpoints/. So "pulled 0 result file(s)" +
    exit 0 is the CORRECT reading of a perfectly healthy job — and, unexplained,
    is indistinguishable from a durability bug. On 2026-08-03 it sent an operator
    hunting a phantom checkpoint hole on job 20260803T090655-frontier-wave-aac5
    (preempted mid-wave) while all 1090 generated rows sat safely in
    checkpoints/. Best-effort and fully guarded: a diagnostic must never change
    the pull's outcome or its exit status."""
    try:
        view = jobmeta.read_job(job_id, live_iids=_live_iids_set())
        status = view.get("display_status") or view.get("status") or "unknown"
    except Exception:
        status = "unknown"
    try:
        ckpts = jobmeta.list_checkpoints(job_id)
    except Exception:
        ckpts = []
    print(f"   (job status={status})")
    if status not in jobmeta.TERMINAL:
        print("   NOTE: results/ is written ONCE at finalize, so a non-terminal "
              "job pulls 0 files. This is expected, NOT lost work.")
    if ckpts:
        print(f"   {len(ckpts)} file(s) ARE durable under "
              f"jobs/{job_id}/checkpoints/ (jobd's mid-run sync; what a resume "
              f"pulls back). Inspect/fetch with rclone, e.g.:")
        print(f"     rclone lsl b2:$B2_BUCKET/jobs/{job_id}/checkpoints/")
    else:
        print(f"   checkpoints/ is ALSO empty — if this job was expected to "
              f"checkpoint, that IS a durability concern; check `job status "
              f"{job_id}` for checkpoint_sync_failed events.")


# --------------------------------------------------------------------------- #
# Bundle DEFINITIONS (the submittable side), not tickets
# --------------------------------------------------------------------------- #

# moved-from: herdd.JOB_DEF_HOMES
JOB_DEF_HOMES = ("tools/witness/jobs", "tools/vast/jobs", "tools/pipeline/jobs")


# moved-from: herdd.find_job_defs
def find_job_defs(repo_root: str | None = None,
                  ) -> list[tuple[str, Any, str | None]]:
    """Every job-bundle DEFINITION in the repo, as (path, cfg-or-None, err).

    `job ls` lists job TICKETS (queued work on a box). This is its
    definition-side twin: the bundles that exist to be submitted. They live in
    THREE homes — tools/{witness,vast,pipeline}/jobs/ — because each bundle is
    co-located with the package its build/vendoring scripts copy source-of-truth
    modules out of (sync_scorer_files.sh, run_local_k5.sh, prep_m2_bundle.py).
    Relocating them into one tree was considered and REJECTED: it would rewrite
    16 path anchors across 14 files in the repo's most active directory to buy
    nothing but a shorter glob. A registry costs one function.

    Sorted by path; a bundle whose config will not parse is reported with its
    error rather than dropped — an unlistable bundle is the interesting case."""
    root = repo_root or _REPO_ROOT
    out: list[tuple[str, Any, str | None]] = []
    for home in JOB_DEF_HOMES:
        d = os.path.join(root, home)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            bundle = os.path.join(d, name)
            if not os.path.isfile(os.path.join(bundle, "job-config.yaml")):
                continue
            rel = os.path.join(home, name)
            try:
                out.append((rel, jobmeta.load_job_config(bundle), None))
            except Exception as e:
                out.append((rel, None, str(e)))
    return out


# moved-from: herdd.find_job_def_strays
def find_job_def_strays(repo_root: str | None = None) -> list[str]:
    """Directories sitting in a jobs/ home with NO job-config.yaml.

    Reported rather than silently skipped. A jobs/ home is where a reader looks
    for submittable bundles, so a directory there that is not one is either a
    half-authored bundle or a script collection filed under a misleading name —
    both worth seeing. tools/witness/jobs/v3-gate-e is the standing example:
    a gen/serve/score script set with a RUNBOOK, no entrypoint and no config,
    which is why a `*/jobs/*/job-config.yaml` glob finds 42 bundles in 43
    directories (counted 2026-08-13)."""
    root = repo_root or _REPO_ROOT
    strays = []
    for home in JOB_DEF_HOMES:
        d = os.path.join(root, home)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isdir(p) and not name.startswith((".", "_")) \
                    and not os.path.isfile(os.path.join(p, "job-config.yaml")):
                strays.append(os.path.join(home, name))
    return strays


# moved-from: herdd._job_shape
def _job_shape(cfg: Mapping[str, Any] | None) -> str:
    """Launch-shape label for a bundle's job-config.yaml, from needs.gpus +
    env.MODE — the same rule JOBS_INDEX.md's Shape column documents. MODE
    unset is the fail-closed `pinned` default (AUTOTUNE_DESIGN.md), so an
    absent MODE reads as pinned, never as an unknown.

    This is bundle-level, not matrix-arm-level: a matrix.py can fan a bundle
    out into concurrent 1-GPU arms (JOBS_INDEX.md's "concurrent 1-GPU/arm"
    bucket, e.g. repair-lifter-train's base-model bakeoff) that this label
    cannot see, because find_job_defs() never parses matrix.py. What it does
    tell you: whether the job-config.yaml surface itself claims the whole box
    under DDP, one card pinned to world_size 1, or one card left to autotune
    (single-arm submit, so autotune only affects NUM_WORKERS sizing)."""
    needs = (cfg or {}).get("needs") or {}
    if not needs.get("gpu"):
        return "CPU"
    gpus = needs.get("gpus")
    mode = ((cfg or {}).get("env") or {}).get("MODE") or "pinned"
    if gpus == "all":
        return "whole-box DDP" if mode == "autotune" else "1-GPU pinned"
    return "1-GPU autotune" if mode == "autotune" else "1-GPU pinned"


# moved-from: herdd.cmd_job_defs
def cmd_job_defs(a: Any) -> None:  # noqa: ANN401 — argparse.Namespace
    """List job-bundle DEFINITIONS (the submittable bundles), not tickets."""
    defs = find_job_defs()
    strays = find_job_def_strays()
    if not defs:
        print("no job bundles found."); return          # noqa: E702 — verbatim (plan §7.4)
    if getattr(a, "json", False):
        print(json.dumps({
            "bundles": [{"path": p, "name": (c or {}).get("name"),
                         "entrypoint": (c or {}).get("entrypoint"),
                         "gpu": bool(((c or {}).get("needs") or {}).get("gpu")),
                         "shape": _job_shape(c),
                         "assets": len((c or {}).get("assets") or []),
                         "tracks": len(jobmeta.collect_tracked(c or {})),
                         "error": e}
                        for p, c, e in defs],
            "not_bundles": strays}, indent=2))
        return
    print(f"{'PATH':<44} {'NAME':<26} {'GPU':<4} {'SHAPE':<15} {'ASSETS':>6} {'TRACKS':>6}")
    for path, cfg, err in defs:
        if err:
            print(f"{path:<44} !! unreadable: {err}")
            continue
        gpu = "yes" if (cfg.get("needs") or {}).get("gpu") else "-"
        print(f"{path:<44} {cfg.get('name', '?'):<26} {gpu:<4} "
              f"{_job_shape(cfg):<15} "
              f"{len(cfg.get('assets') or []):>6} "
              f"{len(jobmeta.collect_tracked(cfg)):>6}")
    print(f"\n{len(defs)} job bundle(s) across {len(JOB_DEF_HOMES)} homes. "
          f"Purpose/liveness table: tools/vast/JOBS_INDEX.md")
    if strays:
        print(f"\nnot job bundles (no job-config.yaml, but filed under a jobs/ "
              f"home): {', '.join(strays)}")


# --------------------------------------------------------------------------- #
# The queue listing
# --------------------------------------------------------------------------- #

# moved-from: herdd.cmd_job_ls
def cmd_job_ls(a: Any) -> None:  # noqa: ANN401 — argparse.Namespace
    b2._ensure_b2_remote()
    try:
        if a.box:
            pairs = [(str(a.box), j) for j in jobmeta.list_queue(str(a.box))]
        else:
            pairs = jobmeta.list_all_queued()
    except jobmeta.QueueUnreadable as e:
        # "no queued jobs." from a FAILED listing is the operator-facing half of
        # the fleetd blind-queue defect: it reads as a clean, empty fleet.
        sys.exit(f"error: {e}")
    if not pairs:
        print("no queued jobs."); return                # noqa: E702 — verbatim (plan §7.4)
    import parked_lifecycle as _pl  # function-local, verbatim (plan §7.4)
    live = _live_iids_set()
    present = _present_iids_set()          # None == unreadable; see the helper
    # ONE bulk fold for the whole queue, not one `rclone copy` per ticket. This
    # listing carried the same defect `job orphans` did (275 tickets = 275
    # subprocesses = ~139 s); `scan.py`'s docstring holds the measurement and
    # the freshness contract.
    try:
        folds = scan.fold_many([j for _b, j in pairs], live_iids=live)
    except Exception as e:
        sys.exit(f"error: bulk job scan failed: {e}")
    # group by box so we can print each box's lifecycle (parked-after-drain etc.)
    by_box: dict[str, list[str]] = {}
    for box, jid in pairs:
        by_box.setdefault(box, []).append(jid)
    # ... and the SAME defect a second time, on the box side: `read_box` is
    # `read_job`'s twin (one `rclone copy` of jobs/nodes/<iid>/events/ per
    # call) and this loop runs it once per DISTINCT BOX. Measured 2026-08-17:
    # 154 boxes, 69.6 s — which is why `job ls` still took 78.6 s after the job
    # folds dropped from 139 s to 6.5 s. The roster grows with fleet HISTORY,
    # so it only gets worse. `fold_boxes` never raises: an unreadable lifecycle
    # log degrades to "not parked", exactly as the per-box try/except did.
    box_folds = scan.fold_boxes(by_box)
    prog = os.path.basename(sys.argv[0])
    n_stuck_orphans = 0
    for box in sorted(by_box):
        # box-lifecycle summary: did jobd self-park after the queue drained?
        bx = box_folds.get(box) or {"parked": False, "drained_pending": False}
        gone = present is not None and box not in present
        tag = f"live={box in live}"
        if gone:
            # THE HEADLINE. Not "live=False" (which a parked box also is, and
            # which this view got wrong for every box until 2026-08-02): the
            # instance does not exist in the account, so nothing it still holds
            # will ever be claimed.
            tag = "GONE — instance destroyed (no box will ever claim these)"
        elif bx.get("parked"):
            tag = (f"PARKED-SELF ({bx.get('park_reason') or '?'}): "
                   f"{bx.get('n_done') or 0} done, {bx.get('n_failed') or 0} failed "
                   f"@ {bx.get('parked_ts')} — resume: {prog} start {box}")
        elif bx.get("drained_pending"):
            tag = (f"DRAINED (no self-park key) — park it: {prog} stop {box}")
        print(f"== box {box}: {tag} ==")
        print(f"  {'JOB_ID':<34} {'STATUS':<12} {'EVENTS':>6}")
        n_terminal = 0
        for jid in by_box[box]:
            v = folds.get(jid)
            if v is None or v.get("scan_error"):
                err = (v or {}).get("scan_error", "no fold returned")
                print(f"  {jid:<34} (err: {err})"); continue   # noqa: E702 — verbatim
            verdict, _why = _pl.ticket_orphan_verdict(
                box_present=(None if present is None else box in present),
                job_status=v["status"])
            mark = ""
            if verdict in _pl.TICKET_ORPHANS_STUCK:
                mark = "  !! ORPHAN"
                n_stuck_orphans += 1
            elif verdict == _pl.TICKET_ORPHAN_TERMINAL:
                mark = "  (stale pointer)"
            print(f"  {jid:<34} {v['display_status']:<12} {v['n_events']:>6}{mark}")
            if v["status"] in jobmeta.TERMINAL:
                n_terminal += 1
        if n_terminal:
            print(f"  ({n_terminal} terminal — pull results: {prog} job pull <JOB_ID>)")
    if present is None:
        print("\n~~ instance listing unreadable — orphan detection skipped "
              "(a ticket on a destroyed box would look pending)")
    elif n_stuck_orphans:
        print(f"\n!! {n_stuck_orphans} ORPHANED ticket(s): the target box no longer "
              f"exists, so these will read as pending FOREVER.\n"
              f"   inspect: {prog} job orphans   |   resolve: "
              f"{prog} job orphans --resolve --reason '<why>' -y")


# --------------------------------------------------------------------------- #
# SEAM WIRING — new code, no `moved-from:` marker (README §2 rule 7).
#
# `boxes/reap.py` calls `_fold_fleet_jobs` from `cmd_reap`, but `boxes` sits
# BELOW `jobs` in the §5 DAG and may never import it — import-linter rejects the
# edge, which is why reap ships a placeholder rather than a copy. The placeholder
# is INJECTABLE, and this is the injection: importing `vastlib.jobs.view` binds
# it, and the composition roots (`cli.main`, `fleet.daemon`) import this module.
#
# The hook is a named forwarder rather than the function object itself, and the
# forwarder calls `_fold_fleet_jobs` by BARE NAME — which Python resolves
# through this module's globals at CALL time. So
# `monkeypatch.setattr(view, "_fold_fleet_jobs", ...)` still steers reap's call
# sites, exactly as plan §8b requires, where `reap._FOLD_FLEET_JOBS = ` the raw
# object would have frozen the patch out.
# --------------------------------------------------------------------------- #

def _reap_fold_hook(live_iids: Any,  # noqa: ANN401 — set[str] | set[int]
                    prog: Any = None) -> dict[str, list[Any]]:  # noqa: ANN401
    """`boxes.reap`'s bound view of `_fold_fleet_jobs` (see the block above)."""
    return _fold_fleet_jobs(live_iids, prog)


reap._FOLD_FLEET_JOBS = _reap_fold_hook


# --------------------------------------------------------------------------- #
# CPU compile-farm — advisory read-only status fold (AUTOMATION_PLAN §"Out-of-
# scope status channels": cmd_runs MAY fold FARM_STATUS as an advisory secondary
# column; NO writer is added to the append-only events store — the farm marker
# `farm/<FARM_RUN>/FARM_STATUS` stays its own coarse heartbeat, folded here for
# visibility only). The farm is default-ON on training boxes (onstart/train.sh
# block-2c) and its namespace id is FARM_RUN_ID, which defaults to RUN_ID — so a
# farm dir name maps 1:1 to a run row in the common case. A custom `--farm-run`
# namespace that matches no training run simply doesn't annotate a row.
#
# HOMED HERE, one ring below their caller: these four are the `runs` status
# fold, and `storage.json` refused them explicitly — "`_farm_status_by_run`,
# `_ckpt_steps_by_run`, `_train_summary_step` … runs/train status fold, NOT
# storage. They sit textually adjacent to the rclone seam and take
# `runner=_rclone_soft` as a DEFAULT ARG (bound at def time). They belong to the
# cmd_runs cluster (cli/runs.py or jobs/view.py)". `jobs/view.py` is the fold
# side of that pair and is where every other read-only run/job projection landed.
#
# DUPLICATE, RULED 2026-08-16 (wave 6a): `vastlib/cli/runs.py` landed its own
# copies of these four plus `_MAX_SUMMARY_READS` in the same wave — flagged by
# `gen_rename_table.py --check` under `duplicates_needing_rulings`, and the test
# migration cannot rewrite a name with two homes. Bodies were identical. THIS is
# the home, on `cli/__init__.py`'s own contract — "a command module parses,
# calls one function in the ring below, and renders. No policy, no I/O of its
# own" — and these three do rclone listings. `cli/runs.py` deleted its copies
# (and their markers) and calls `jobs_view._farm_status_by_run` /
# `._ckpt_steps_by_run` / `._train_summary_step`.
#
# THE `runner=` DEFAULTS ARE BOUND AT DEF TIME — ported verbatim, and that is a
# patch-point contract, not an accident: `monkeypatch.setattr(b2,
# "_rclone_soft", …)` AFTER import does NOT steer these three, exactly as
# patching `herdd._rclone_soft` did not steer them in the flat module. A test
# (or a `cli/runs.py` composition root) that means to redirect the transport
# passes `runner=` explicitly; the LAUNCHER binding is the only other lever.
# --------------------------------------------------------------------------- #


# moved-from: herdd._parse_farm_status
def _parse_farm_status(text: str | None) -> str | None:
    """FARM_STATUS content is `<WORD> <utc-ts>` (farm_worker.sh:farm_status);
    return the coarse word (RUNNING/DONE/FAILED, upper), or None if empty/absent.
    PURE."""
    tok = (text or "").split()
    return tok[0].upper() if tok else None


# moved-from: herdd._farm_status_by_run
def _farm_status_by_run(base: str, rids: Any,  # noqa: ANN401 — caller-shaped iterable of run ids
                        runner: Callable[..., Any] = b2._rclone_soft) -> dict[str, str]:
    """Best-effort {run_id -> farm status word} for the run_ids in `rids`.

    ONE `lsf farm/` gates the whole thing: if no farm namespaces exist the fold
    is empty at zero per-run cost. Only farm dirs that match a displayed run get
    a FARM_STATUS cat. Never raises (advisory column) — any rclone failure yields
    an empty/partial map and the rows fall back to '-'."""
    rc, out, _ = runner(["lsf", "--dirs-only", f"{base}/farm/"])
    if rc != 0 or not out:
        return {}
    farmed = {x.strip().rstrip("/") for x in out.splitlines() if x.strip()}
    want = farmed & set(rids)
    status = {}
    for rid in sorted(want):
        rc, st, _ = runner(["cat", f"{base}/farm/{rid}/FARM_STATUS"])
        if rc == 0:
            w = _parse_farm_status(st)
            if w:
                status[rid] = w
    return status


# moved-from: herdd._ckpt_steps_by_run
def _ckpt_steps_by_run(base: str,
                       runner: Callable[..., Any] = b2._rclone_soft) -> dict[str, dict[str, Any]]:
    """Best-effort {run_id -> {"step": max checkpoint step, "summaries": [...]}}
    read off the artifact tree.

    ONE recursive listing covers every run and every layout that has shipped:
      checkpoints/<rid>/checkpoint-<step>/                      (bare)
      checkpoints/<rid>/adapter/checkpoint-<step>/              (adapter)
      checkpoints/<rid>/arms/<arm>/checkpoint-<step>/           (ladder/multi-arm)
    The arm layout is why this is depth **4** and not 3: every ladder run
    (seq-ladder-01, seqdg-smokeB-01, modelzoo-reader-06, base-reader-nanbeige-02)
    keeps its checkpoints one level deeper, and a depth-3 scan sees the `arms/`
    directory and nothing below it. Depth 4 is still the bound that matters —
    the level below a checkpoint dir is model weights, thousands of objects per
    run.

    The same pass also collects `train_summary.json` paths (`--include` is a
    client-side filter, so asking for them costs no extra listing). They are the
    LAST step source: a run that finished, uploaded its adapter and had its
    intermediate checkpoint dirs pruned — or never wrote any — still records
    `global_steps` there, and for those runs it is the only surviving evidence
    that the run ever took a step.

    Training bundles that write checkpoints but never emit a `checkpoint` EVENT
    (most of them) are invisible to the fold, so this is the only step signal
    those runs have — the run detail page already derived steps this way while
    the list column sat blank. Advisory: any rclone failure yields {} and the
    rows fall back to the event fold alone."""
    rc, out, _ = runner(["lsf", "-R", "--max-depth", "4", "--fast-list",
                         "--include", "**train_summary.json",
                         f"{base}/checkpoints/"])
    if rc != 0 or not out:
        return {}
    found: dict[str, dict[str, Any]] = {}
    for line in out.splitlines():
        line = line.strip()
        if "/" not in line:
            continue
        parts = line.rstrip("/").split("/")
        if len(parts) < 2:
            continue                    # the run's own top-level directory
        rid, leaf = parts[0], parts[-1]
        m = re.match(r"^checkpoint-(\d+)$", leaf)
        if not m and leaf != "train_summary.json":
            continue                    # never mint an empty slot for a run
        slot = found.setdefault(rid, {"step": None, "summaries": []})
        if m:
            step = int(m.group(1))
            if slot["step"] is None or step > slot["step"]:
                slot["step"] = step
        else:
            slot["summaries"].append(f"{base}/checkpoints/{line}")
    return found


# A run with more arms than this resolved its step from checkpoint dirs anyway;
# the cap only stops a pathological tree becoming N network reads.
# moved-from: herdd._MAX_SUMMARY_READS
_MAX_SUMMARY_READS = 4


# moved-from: herdd._train_summary_step
def _train_summary_step(paths: Any,  # noqa: ANN401 — caller-shaped iterable of B2 paths
                        runner: Callable[..., Any] = b2._rclone_soft) -> int | None:
    """Max `global_steps` across `train_summary.json` paths, or None.

    The trainer writes this file at the end of every run. It is the last step
    source consulted — only for runs the checkpoint-dir scan could not resolve —
    so in practice it is at most one small GET for a handful of runs.
    `json.loads` accepts the bare `NaN` literals the trainer emits for a run
    with zero recorded loss points. Best-effort: unreadable or non-JSON
    summaries are skipped, never raised."""
    steps = []
    for p in list(paths)[:_MAX_SUMMARY_READS]:
        rc, out, _ = runner(["cat", p])
        if rc != 0 or not out:
            continue
        try:
            d = json.loads(out)
        except (ValueError, TypeError):
            continue
        if isinstance(d, dict):
            gs = runmeta._num(d.get("global_steps"))  # type: ignore[no-untyped-call]
            if gs is not None:
                steps.append(int(gs))
    return max(steps) if steps else None
