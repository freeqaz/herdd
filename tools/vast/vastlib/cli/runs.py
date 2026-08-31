"""`herdd runs` — the training-run ledger folded out of B2.

The list view is a fan-out: one `lsf` over `checkpoints/` and one over `runs/`
gives the run-id union, then every run's event fold (`runmeta.read_run`, an
independent B2 read) runs CONCURRENTLY. That concurrency is load-bearing, not an
optimization — the dashboard shells into this command, and serializing N network
round-trips is the difference between a page and a timeout. One bad run never
breaks the list: `_row` catches, warns on stderr, and returns None.

Where a run's STEP comes from, in order
---------------------------------------
1. the event fold's `latest_step`;
2. the artifact tree (`_ckpt_steps_by_run`) — ONE recursive listing at depth 4,
   which is depth 4 and not 3 because every ladder/multi-arm run keeps its
   checkpoints one level deeper under `arms/<arm>/`;
3. `train_summary.json` (`_train_summary_step`) — LAST, capped at
   `_MAX_SUMMARY_READS`, and only for a run neither of the first two resolved.
   For a run that finished, uploaded its adapter and had its intermediate
   checkpoint dirs pruned, this file is the only surviving evidence it ever
   took a step.

Most training bundles write checkpoints without ever emitting a `checkpoint`
EVENT, so (2) and (3) are not a fallback for exotic runs — they are the only
step signal a majority of runs have.

Advisory columns are advisory
-----------------------------
`FARM` (the CPU-farm heartbeat) and `SUPV` (supervisor liveness) never raise and
never change a run's status: one `lsf farm/` gates the farm fold, so it costs
nothing when unused, and any rclone failure yields a partial map whose rows
render `-`. `SUPV` prints `UNSUPERVISED` only for a NON-terminal run with a live
box — the case where nobody is watching something that is billing.

Costs marked `~` are DERIVED (dph x observed span), not billed. That prefix is
the whole contract of `cost_source`; a consumer that drops it reports an
estimate as an invoice.

What is deliberately NOT here
-----------------------------
* The event schema and the fold. `runmeta` owns both, and its
  `.claude/skills/vast-runs/SKILL.md` is the contract (append-only, one
  immutable object per event, liveness from vast and not from recency).
* The `--refresh-cache` sqlite writer's schema — `storage.dashcache`.
* Any mutation. `runs` reads B2 and the instance list; it can neither start,
  stop, nor bill anything.
* The three B2 folds themselves. DUPLICATE RULING (2026-08-16, wave 6a):
  `_parse_farm_status`, `_farm_status_by_run`, `_ckpt_steps_by_run`,
  `_train_summary_step` and `_MAX_SUMMARY_READS` are homed in `jobs/view.py` —
  the fold side of the pair, one ring below, and the ring a command module is
  supposed to call into rather than duplicate (`cli/__init__.py`: "no policy, no
  I/O of its own"; these do rclone listings). This module imports them. Same for
  `_ts_age_s`, homed in `core/fmt.py` because `boxstate.py` reaches it as
  `herdd._ts_age_s` and the thin launcher must re-export it from below `cli`.
  Note the ported `runner=b2._rclone_soft` default binds at DEF TIME in
  `jobs/view.py`; patch `jobs_view` or pass `runner=` — patching `b2` after
  import does not steer these, exactly as in the flat module.

Provenance: moved from `tools/vast/herdd.py` (`cmd_runs`, `gather_run_rows`,
`_supervision_state`, `SUPERVISOR_STALE_S`, parser block in `main()`), plan §8
step 6, 2026-08-16, behavior-preserving.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from typing import Any

from vastlib.boxes import lifecycle
from vastlib.cli import _args, _docs
from vastlib.core import fmt, models
from vastlib.jobs import view as jobs_view
from vastlib.launch import spec
from vastlib.storage import b2, dashcache

import runmeta

# moved-from: herdd.SUPERVISOR_STALE_S
SUPERVISOR_STALE_S = 600     # cmd_runs: no supervisor heartbeat within this -> UNSUPERVISED


# moved-from: herdd.gather_run_rows
def gather_run_rows(base: str) -> list[dict[str, Any]]:
    """Union of run-ids from checkpoints/ AND runs/, each folded into one list
    row. The per-run folds (each an independent B2 read via runmeta.read_run)
    fan out CONCURRENTLY -- this is the hot path the dashboard shells into, so it
    must not serialize N network round-trips. Deterministic order: sorted rids,
    Nones (skipped runs) dropped. One bad run never breaks the list."""
    _, ck_out = b2._rclone(["lsf", "--dirs-only", f"{base}/checkpoints/"])
    _, rn_out = b2._rclone(["lsf", "--dirs-only", f"{base}/runs/"])   # may not exist yet
    rids: Any = set()   # a set while collecting, a sorted list below (verbatim)
    for out in (ck_out, rn_out):
        for x in (out or "").splitlines():
            x = x.strip().rstrip("/")
            if x and not x.startswith("_"):        # exclude runs/_selftest/, etc.
                rids.add(x)
    rids = sorted(rids)
    if not rids:
        return []

    # ONE instances call -> run_id -> {live iids}
    live_by_run: dict[Any, set[Any]] = {}
    for i in lifecycle.live_run_instances(None, instances=lifecycle._instances_soft()):
        live_by_run.setdefault(models._instance_run_label(i), set()).add(i.get("id"))

    # advisory CPU-farm heartbeat (one lsf gates it; empty at zero cost if unused)
    farm_by_run = jobs_view._farm_status_by_run(base, rids)
    # ONE recursive listing -> run_id -> {max checkpoint step, summary paths}
    # (artifact-derived fallback for runs whose bundle never emitted a
    # `checkpoint` event)
    ckpt_arts = jobs_view._ckpt_steps_by_run(base)

    def _row(rid: str) -> dict[str, Any] | None:
        live_iids = live_by_run.get(rid, set())
        try:
            view = runmeta.read_run(rid, live_iids=live_iids)
        except Exception as e:                     # one bad run never breaks the list
            print(f"warn: skipping {rid}: {e}", file=sys.stderr)
            return None
        instance_live = bool(live_iids) or bool(view.get("live"))
        # I4: only pay for a STATUS read when the fold is non-terminal AND the box
        # is gone (the exact window final_status can infer a terminal from STATUS).
        status_marker = None
        if view.get("n_events") and view["status"] not in ("done", "failed") \
                and not instance_live:
            _, status_marker = b2._rclone(["cat", f"{base}/checkpoints/{rid}/STATUS"])
            status_marker = status_marker or None   # "" (no file/empty) -> None
        try:
            fs = runmeta.final_status(view, status_marker=status_marker,
                                      instance_live=instance_live)
        except IndexError:
            # runmeta.final_status does (status_marker or "").split(None,1)[0]
            # unconditionally whenever instance_live is False, which raises
            # for a falsy/whitespace-only status_marker (its own default!) —
            # degrade to the fold's own status rather than crash the list.
            s = view.get("status")
            fs = {"status": s, "terminal": s in ("done", "failed"),
                  "display": s if s in ("done", "failed")
                  else ("running" if instance_live else "stopped")}
        sup_flag = "-"
        if not fs["terminal"] and bool(live_iids):
            if _supervision_state(rid, SUPERVISOR_STALE_S) in (
                    "none", "exited", "unsupervised"):
                sup_flag = "UNSUPERVISED"
        # step: event fold OR the artifact tree, whichever is further along;
        # then train_summary.json, but only for a run neither could resolve
        # (it costs a GET, and a checkpoint dir is the stronger evidence).
        step = view.get("latest_step")
        art = ckpt_arts.get(rid) or {}
        ck = art.get("step")
        if ck is not None and (step is None or ck > step):
            step = ck
        if step is None and art.get("summaries"):
            step = jobs_view._train_summary_step(art["summaries"])
        return {
            "run": rid,
            "status": fs["display"],
            "terminal": fs["terminal"],
            "gpu": view.get("gpu"),
            "dph": view.get("dph"),
            "latest_step": step,
            "cost_usd": view.get("cost_usd"),
            # "event" | "derived" | None — see runmeta.fold_events. Consumers
            # MUST render "derived" as an approximation, never as a billed figure.
            "cost_source": view.get("cost_source"),
            "relaunch_count": view.get("relaunch_count", 0),
            "instance_id": view.get("instance_id"),
            "live": bool(live_iids),
            "n_events": view.get("n_events", 0),
            "parse_errors": view.get("parse_errors", 0),
            "supervised": sup_flag,
            "farm": farm_by_run.get(rid),   # advisory CPU-farm heartbeat, or None
            # fold already computes these; project them so the list can sort/date
            "started_at": view.get("started_at"),
            "ended_at": view.get("ended_at"),
            "last_event_ts": view.get("last_event_ts"),
        }

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(16, len(rids))) as ex:
        return [r for r in ex.map(_row, rids) if r is not None]
# moved-from: herdd._supervision_state
def _supervision_state(run_id: object, stale_after_s: float) -> str:
    """Classify supervisor liveness from supervisor events:
    'none' (no supervisor ever) | 'exited' (latest is supervisor_exiting) |
    'unsupervised' (newest heartbeat/start older than stale_after_s) |
    'supervised' (fresh)."""
    sup = [e for e in spec._raw_events_soft(run_id)
           if e.get("event") in ("supervisor_started", "heartbeat",
                                  "supervised", "supervisor_exiting")]
    if not sup:
        return "none"
    last = sup[-1]                                    # _raw_events_soft is (ts,nonce)-sorted
    if last.get("event") == "supervisor_exiting":
        return "exited"
    age = fmt._ts_age_s(last.get("ts"))
    if age is not None and age > stale_after_s:
        return "unsupervised"
    return "supervised"
# moved-from: herdd.cmd_runs
def run(a: argparse.Namespace) -> None:
    """List training runs from b2:$B2_BUCKET/checkpoints/<RUN_ID> and their status.

    Status comes from each run's STATUS marker (written by onstart/train.sh and
    run_durable.sh: RUNNING/DONE/FAILED + a UTC timestamp). An artifacts/<id>/
    dir means the final model was published (success)."""
    bucket = os.environ.get("B2_BUCKET")
    if not bucket:
        sys.exit("error: B2_BUCKET not set (env or .env)")
    b2._ensure_b2_remote()
    base = f"b2:{bucket}"

    if a.run:  # detail view of one run
        live = {i.get("id") for i in lifecycle.live_run_instances(a.run,
                                                         instances=lifecycle._instances_soft())}
        try:
            view = runmeta.read_run(a.run, live_iids=live)
        except runmeta.RunmetaError as e:
            sys.exit(f"error: {e}")
        if a.json:
            print(json.dumps(view, indent=2)); return  # noqa: E702 — verbatim body (plan §7.4)
        print(f"== run {a.run} ==")
        _c = fmt.dollars(view['cost_usd']) if view['cost_usd'] is not None else '-'
        if view.get('cost_source') == 'derived':
            _c = '~' + _c              # dph x observed span, not a billed figure
        print(f"  status={view['display_status']} step={view['latest_step']} "
              f"cost={_c} "
              f"relaunch={view['relaunch_count']} gpu={view['gpu']} "
              f"iid={view['instance_id']} live={view['live']} "
              f"events={view['n_events']} parse_errors={view['parse_errors']}")
        # last_event surfaces preempted/rescued/bid_raised (SPOT_DESIGN §3.2/3.3)
        # directly — none of them change display_status above by design (rescued/
        # bid_raised are fold-neutral; preempted only flips display_status via
        # preempted_pending until something confirms otherwise), so without this
        # line a rescued or bid-defended run's B2 history is invisible here.
        if view["last_event"]:
            print(f"  last_event={view['last_event']} @ {view['last_event_ts']}")
        for area in ("checkpoints", "artifacts"):
            print(f"== {area}/{a.run} ==")
            _, out = b2._rclone(["lsl", f"{base}/{area}/{a.run}"])
            print((out or "").rstrip() or "  (none)")
        _, st = b2._rclone(["cat", f"{base}/checkpoints/{a.run}/STATUS"])
        print(f"== STATUS ==\n  {(st or '').strip() or '(none)'}")
        _, fs = b2._rclone(["cat", f"{base}/farm/{a.run}/FARM_STATUS"])
        print(f"== FARM_STATUS ==\n  {(fs or '').strip() or '(none — CPU farm idle/unqueued)'}")
        return

    rows = gather_run_rows(base)

    # dashboard SWR writer: fold the list and upsert it into infra-metadata.db
    # instead of printing (concurrency already happened inside gather_run_rows).
    if getattr(a, "refresh_cache", False):
        db = dashcache._infra_cache_db(a)
        dashcache._infra_cache_write(rows, db)
        print(f"cached {len(rows)} run(s) -> {db}", file=sys.stderr)
        return

    if not rows:
        print(f"no runs under {base}/runs/ or {base}/checkpoints/"); return  # noqa: E702 — verbatim body (plan §7.4)

    if a.json:
        print(json.dumps(rows, indent=2)); return  # noqa: E702 — verbatim body (plan §7.4)
    print(f"  {'RUN_ID':<26} {'STATUS':<16} {'UPDATED':<12} {'GPU':<14} "
          f"{'STEP':>8} {'COST':>9} {'RLNCH':>5}  {'FARM':<8} {'SUPV':<12}")
    for r in rows:
        step = r["latest_step"] if r["latest_step"] is not None else "-"
        cost = fmt.dollars(r["cost_usd"]) if r["cost_usd"] is not None else "-"
        if r.get("cost_source") == "derived":
            cost = "~" + cost          # dph x observed span, not a billed figure
        updated = fmt._fmt_run_ts(r.get("last_event_ts") or r.get("ended_at")
                              or r.get("started_at"))
        print(f"  {r['run']:<26} {r['status'][:16]:<16} {updated:<12} "
              f"{(r['gpu'] or '-'):<14} {str(step):>8} {cost:>9} "
              f"{r['relaunch_count']:>5}  {(r.get('farm') or '-'):<8} "
              f"{r.get('supervised', '-'):<12}")


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    prn = add_cmd(sub, "runs", "list training runs + status from B2 (checkpoints/<RUN_ID>)",
                  _docs.DOC_SKILL_RUNS, _docs.DOC_TRAINING,
                  "NOTE: liveness comes from `herdd ls`, not a trailing 'running' event")
    prn.add_argument("--run", help="detail view of one RUN_ID (list checkpoints + artifacts)")
    prn.add_argument("--json", action="store_true")
    prn.add_argument("--refresh-cache", action="store_true",
                     help="fold the run list and UPSERT it into the infra-metadata "
                          "sqlite cache instead of printing (dashboard SWR writer)")
    prn.add_argument("--cache-db", metavar="PATH",
                     help="override the infra-metadata.db path "
                          "(default: tools/vast/infra-metadata.db, env INFRA_METADATA_DB)")
    prn.set_defaults(func=run)
    return prn
