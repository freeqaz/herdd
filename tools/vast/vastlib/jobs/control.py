"""vastlib.jobs.control — MOVE, RE-OPEN, KILL and SWEEP a job's ticket.

Why this exists
---------------
Every function here writes to B2, and each write can double-run a training job
or end one that was still working. `jobs/view.py` next door is the read half,
and the split is deliberate: the read path runs a hundred times a day, the write
path is a recovery lane with a refusal ladder in front of every mutation.

The four commands are one mechanism seen from four directions —
`retarget` moves a ticket, `requeue` re-opens a terminal-failed job onto a
different box under the SAME job id, `cancel` makes a job terminal and
non-resumable, and `orphans --resolve` is a cancel whose reason was composed
from evidence rather than typed by a human.

The contracts this module is not allowed to "clean up"
-------------------------------------------------------
* **`_job_cancel_writes`' ORDER.** CANCEL marker -> terminal `cancelled` event
  -> ticket delete. Each step is independently correct only in that order: a
  running jobd must be able to SEE the marker, and the ticket delete cannot stop
  a job that is already running. It is ONE copy, shared by `cmd_job_cancel` and
  `cmd_job_orphans --resolve`; keeping it one copy is the point. `**extra`
  (`orphan=` / `orphan_box=`) rides onto the event because the fold tolerates
  unknown keys by contract.
* **`cmd_job_orphans`' exit codes**: 1 = the instance listing was UNREADABLE (no
  verdict minted), 2 = stuck orphans found without `--resolve`, 0 = clean. They
  exist so `&&` chains can branch, and the 1 is not an error code — it is the
  tri-state of `view._present_iids_set` reaching the shell.
* **`_job_cancel_kill_script` ships as a base64'd FILE.** A bare
  `pkill -f <jid>` over `ssh --exec` matches its OWN wrapper cmdline and can
  kill the session — the same failure class as this box's standing wait-loop
  rule. Inlining it back re-arms the footgun.
* **`stale` membership is EVIDENCE, never assumption** in `cmd_job_retarget`: a
  box is in the delete set because we read its ticket or the queue listing named
  it. Deleting a key that was never there just prints a scary rclone failure at
  the operator.
* **No `--allow-bundle-drift` on requeue.** A changed bundle is a DIFFERENT
  experiment; letting it inherit the failed job's event log and checkpoints
  would silently mix two runs under one id.

`sys.exit` inside library code
------------------------------
There are forty-odd `sys.exit` sites here, several inside non-`cmd_` helpers
(`_retarget_reconstruct`, `_retarget_drop_stale`). That is the CLI error
contract of the code being moved, and `_ssh_kill_job` already catches
`SystemExit` from `lifecycle._get_instance` — proof the pattern leaks. Porting
them verbatim is correct for behavior preservation; converting them to
exceptions is a separate change and not this one (plan §7.4).

What is deliberately NOT here
-----------------------------
* The reads. `_live_iids_set`, `_job_view`, `_print_job_view` and the fold cache
  are `jobs/view.py`'s, called module-attribute-style so a patch on `view`
  steers these commands.
* `_apply_env_overrides` — `jobs/submit.py` owns it; `cmd_job_requeue` calls
  DOWN into the same implementation `job submit` uses rather than forking the
  env-pin grammar.
* `jobmeta`'s B2 primitives (`write_ticket`, `delete_ticket`, `emit_event`,
  `write_cancel_marker`, `requeue_ticket`). Zone S, bare-name imported,
  untouched — `jobmeta.py` documents itself as the non-exiting core of these
  commands.

Provenance: behavior-preserving move of 13 symbols from `tools/vast/herdd.py`
(plan §8 step 5, 2026-08-16), each carrying its `# moved-from:` marker.
ADD-ONLY: `herdd.py` keeps its live copies until step 6.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
import subprocess
import sys
from typing import Any, Mapping

from vastlib.boxes import lifecycle, ssh
from vastlib.fleet import client as fleet_client
from vastlib.jobs import scan, submit, view
from vastlib.storage import b2

import jobmeta

# --------------------------------------------------------------------------- #
# retarget — move a ticket to another box's queue
# --------------------------------------------------------------------------- #

# moved-from: herdd._retarget_queued_boxes
def _retarget_queued_boxes(jid: str) -> list[str]:
    """Every box whose queue currently holds a ticket for JID (usually 0 or 1).

    WHY THE WHOLE-QUEUE SCAN AND NOT JUST `--from` (task #75). jobd NEVER deletes
    a queue ticket — it only `cat`/`lsf`/`copyto`s jobs/queue/<IID>/ (jobd.sh) —
    so "the ticket jobd consumed" is not a thing that happens. What moves a
    ticket behind the operator's back is fleetd: `_retarget_pending_tickets`
    rewrites `box` and deletes the old pointer during an eviction/pull
    replacement. So the ordinary failure is a ticket sitting under a box the
    operator has not heard of yet, and `--from` naming the old one.

    `jobmeta.list_all_queued` used to answer [] both for "nothing queued" and for
    a FAILED listing; it now raises, and this refuses rather than let an
    unreadable queue be read as proof of absence — downstream that would delete
    or rebuild the wrong pointer."""
    try:
        return sorted({b for b, j in jobmeta.list_all_queued() if j == jid})
    except jobmeta.QueueUnreadable as e:
        sys.exit(f"error: {e}\n"
                 f"       Refusing to retarget on an unreadable queue: "
                 f"'not queued anywhere' is a conclusion this listing cannot "
                 f"support. Fix B2 access and re-run.")


# moved-from: herdd._vram_advisory
def _vram_advisory(cfg: Any, *, where: str) -> None:  # noqa: ANN401 — job config or None
    """Print the VRAM sizing verdict WITHOUT its refusal.

    The recovery paths need the finding but must never be blocked by it.
    `requeue` refuses any bundle edit outright and `retarget` copies the ticket
    verbatim, so the operator has nothing left to change except abandoning the
    recovery — and `retarget` is what fleetd drives on an eviction, where a
    refusal would turn an automatic rescue into a lost job.

    It is still worth printing, because the thing that moves under a fixed
    bundle is the FACTS TABLE: an anchor minted since the original submit can
    show a floor that was merely unmeasured then to be known-wrong now. That is
    real information about a job about to re-run; it is just not a reason to
    stop the rescue."""
    try:
        lines, _refuse = jobmeta.vram_gate_report(  # type: ignore[no-untyped-call]
            jobmeta.vram_gate_findings(cfg))
    except Exception:
        return                       # advice must never break a recovery
    for ln in lines:
        print(ln.replace("!! vram:", f"!! vram ({where}):"), file=sys.stderr)


# moved-from: herdd._retarget_reconstruct
def _retarget_reconstruct(a: Any,  # noqa: ANN401 — argparse.Namespace
                          view_: Mapping[str, Any], jid: str, new_box: str,
                          old_box: str, queued_boxes: list[str]) -> Any:  # noqa: ANN401
    """`--reconstruct`: mint a ticket for a job whose queue pointer is GONE.

    The 2026-08-08 recovery, promoted from a hand-run `jobmeta.make_ticket`
    snippet (V10_SPOT_PROVISIONING §6). Everything the ticket needs survives in
    the event log + the bundle object: `bundle_sha256` off the folded `submitted`
    event, and the config out of the bundle itself. `retargeted_from` is stamped
    exactly as a real retarget stamps it, which is what makes the new box's jobd
    pull jobs/<JOB_ID>/checkpoints/ back instead of restarting (jobd.sh
    run_job_body / HANDOFF_DESIGN §4).

    Fail-closed on the double-run hazard: if the queue scan named ANY box for
    this job, a pointer still exists somewhere and minting a second one would
    queue the job twice. That is reachable even here — the scan lists, the
    per-box `cat` can still fail or race — so it is re-checked rather than
    assumed from control flow, and a LIVE box among them is named first because
    that is the case that actually double-runs.

    Returns the ticket dict (never writes; the caller owns B2 mutation).

    (The `view` parameter is spelled `view_` here only because `view` is this
    module's name for `jobs/view.py`; nothing else about it changed.)"""
    if queued_boxes:
        live = view._live_iids_set()
        hot = [b for b in queued_boxes if b in live]
        where = ", ".join(queued_boxes)
        sys.exit(
            f"error: refusing to --reconstruct {jid} — a queue ticket still "
            f"exists (box {where}"
            + (f"; {', '.join(hot)} is LIVE and would claim it" if hot else "")
            + f"). Retarget it instead: --from {queued_boxes[0]}")
    sha = view_.get("bundle_sha256")
    if not sha:
        sys.exit(f"error: {jid} has no `submitted` event carrying a bundle_sha256 "
                 f"— there is nothing to reconstruct a ticket from")
    if not jobmeta.bundle_exists(sha):
        sys.exit(f"error: bundle {sha[:12]}… is not on B2 (lifecycle-expired or "
                 f"never landed) — cannot reconstruct {jid}'s ticket. Re-submit "
                 f"the bundle under a new JOB_ID with `job submit`")
    staging = os.path.join(view._REPO_ROOT, "out", "jobs", "_reconstruct", sha)
    blob = staging + ".tar.zst"
    os.makedirs(os.path.dirname(staging), exist_ok=True)
    ok, err = jobmeta.download_bundle(sha, blob)  # type: ignore[no-untyped-call]
    if not ok:
        sys.exit(f"error: bundle download failed: {err}")
    try:
        jobmeta.extract_bundle(blob, staging, expect_sha=sha)
        raw = jobmeta.load_job_config(staging)
        # `materialized=True`: this tree came out of a TAR, so any `includes:`
        # are already files in it. Validating it as an authoring tree refused
        # every migrated bundle here — a recovery path that only failed once
        # the ticket was already lost.
        cfg, warnings = jobmeta.validate_job_config(raw, staging,
                                                    materialized=True)
    except jobmeta.JobmetaError as e:
        sys.exit(f"error: {e}")
    for w in warnings:
        print(f"warn: {w}", file=sys.stderr)
    print(f">> ticket RECONSTRUCTED from bundle {sha[:12]}… (no queue pointer "
          f"survived anywhere)")
    # the five `f` prefixes below carry no placeholder and never did — operator
    # text kept byte-identical to herdd's (plan §7.4), so the F541s are waived
    # rather than "fixed" (same call as `jobs/submit.py`'s).
    print(f"!! submit-time `--env` pins are NOT in the bundle — env overrides are "  # noqa: F541
          f"ticket-side, and the ticket is what was lost. If the original submit "  # noqa: F541
          f"used any, this re-run DIFFERS from it "                                 # noqa: F541
          f"(the original values are usually in the checkpointed "                  # noqa: F541
          f"results/input-manifest.json)")                                          # noqa: F541
    ticket = jobmeta.make_ticket(jid, sha, lifecycle._cli_actor(), cfg, new_box)
    if old_box:
        ticket["retargeted_from"] = old_box
    return ticket


def _retarget_poison_refusal(v: Mapping[str, Any], new_box: str,
                             jid: str) -> str | None:
    """The named reason `job retarget` refuses to move JID onto `new_box`, or
    None when the target is clean. Pure (takes a fold view) so the policy is
    testable without B2 — same shape as `_requeue_refusal`.

    A box whose jobd went terminal on this JOB_ID holds a local
    `$STATE_DIR/<JOB_ID>.terminal` breadcrumb and skips the ticket before any B2
    read, so the move would be a silent forever-noop. `cmd_job_requeue` has had
    this gate since it was written (its GATE 3); retarget lacked it, which is how
    a single spurious claim-time `failed` turned into two sweep arms queued on
    the one box guaranteed to ignore them (JOB_RETARGET_RACE_2026-08-20.md).

    Evidence, not assumption: `terminal_boxes` is folded from the event stream's
    own attribution, and it is one-directional — membership proves the box
    latched, absence proves nothing (jobd also latches on paths that emit no
    event: remote-done, the restart and disk caps)."""
    if str(new_box) not in set(v.get("terminal_boxes") or ()):
        return None
    return (f"refusing to retarget {jid} onto {new_box} — that box's jobd already "
            f"went terminal on this JOB_ID, so its local terminal cache "
            f"($STATE_DIR/{jid}.terminal) makes it skip the ticket forever, "
            f"silently. Retarget onto a DIFFERENT box, or delete that breadcrumb "
            f"on {new_box} first")


# moved-from: herdd.cmd_job_retarget
def cmd_job_retarget(a: Any) -> None:  # noqa: ANN401 — argparse.Namespace
    """Move a queued/interrupted job's ticket to ANOTHER box's queue: write the
    ticket (same JOB_ID — the event log continues) under the new box, DELETE the
    old queue pointer (without it the old box would double-run the job the moment
    it resumes), emit `retargeted`. The new box's jobd claims it fresh; a
    checkpointing job pulls its synced state back from jobs/<JOB_ID>/results/, so
    training arms continue rather than restart.

    A MISSING ticket at `--from` used to be a hard exit (task #75), which is how
    the 2026-08-08 night ended with "no CLI path moves an interrupted job off a
    dead box" and a hand-run `jobmeta.make_ticket`. Two answers, in order:

      1. SCAN THE QUEUE. jobd never deletes tickets; fleetd's eviction
         replacement moves them. So the common shape is a live ticket under a box
         the operator hasn't seen — already at `--box` (idempotent success), or
         at a third box that simply becomes the effective source.
      2. `--reconstruct` (opt-in) when it is genuinely gone anywhere: rebuild the
         ticket from the `submitted` event's bundle. Opt-in because it cannot
         recover submit-time `--env` pins."""
    b2._ensure_b2_remote()
    jid = a.job_id
    new_box = str(a.box)
    v = jobmeta.read_job(jid, live_iids=view._live_iids_set())
    if v["status"] in jobmeta.TERMINAL:
        sys.exit(f"error: {jid} is already terminal ({v['status']}) — nothing to retarget")
    old_box = str(a.from_box or v.get("target_box") or v.get("instance_id") or "")
    if not old_box:
        sys.exit("error: cannot determine the source box (pass --from)")
    if old_box == new_box:
        sys.exit("error: source and target box are the same")
    if v["display_status"] == "running":
        # Name the box the job is ACTUALLY running on (the folded view's
        # target_box/instance_id), never `old_box` alone — `old_box` takes
        # `a.from_box` first, so a WRONG `--from` (stale, or simply a typo) on a
        # job running elsewhere used to print "is RUNNING on live box <the
        # wrong id>", asserting liveness of a box the fold never named. The
        # refusal itself is correct (never let a running job get double-queued
        # under a second box); only the box name in the message was wrong.
        true_box = str(v.get("target_box") or v.get("instance_id") or old_box)
        stale_note = ""
        if a.from_box and str(a.from_box) != true_box:
            stale_note = f" (--from {old_box} is stale — the job is on {true_box})"
        sys.exit(f"error: {jid} is RUNNING on live box {true_box}{stale_note} — "
                 f"retargeting would double-run it (park/verify the box first)")
    poison = _retarget_poison_refusal(v, new_box, jid)
    if poison:
        sys.exit(f"error: {poison}")
    # `stale` = every box PROVEN to hold a pointer that must go once the move
    # lands. Membership is evidence-based (a ticket we read, or a box the queue
    # listing named), never assumption: deleting a key that was never there just
    # prints a scary rclone failure at the operator.
    ticket = jobmeta.read_ticket(old_box, jid)  # type: ignore[no-untyped-call]
    if ticket is not None and not getattr(a, "stale_ok", False):
        is_stale, why = jobmeta.ticket_staleness(ticket)  # type: ignore[no-untyped-call]
        if is_stale:
            sys.exit(
                f"error: {jid} is STALE — {why}.\n"
                f"  A retarget re-runs the ticket's FROZEN bytes: bundle "
                f"{str(ticket.get('bundle_sha256') or '?')[:12]}, submitted "
                f"{ticket.get('submitted_ts')}. Repo-side fixes to that bundle "
                f"since then are NOT picked up — measured 2026-08-24, when "
                f"retargeting four screen-v1 arms re-ran a superseded fla gate "
                f"and re-emitted provenance that had already been corrected.\n"
                f"  Resubmit from the current bundle (preferred), or "
                f"--stale-ok to move these exact bytes anyway, or retire it: "
                f"herdd.py job dlq add {jid} --reason '<why>'")
    stale = {old_box} if ticket is not None else set()
    if ticket is None:
        queued = _retarget_queued_boxes(jid)
        stale.update(b for b in queued if b != new_box)
        if new_box in queued:
            # IDEMPOTENT SUCCESS. Somebody (usually fleetd) already put the
            # ticket where we were asked to put it; saying "no ticket at
            # jobs/queue/<old>/" and exiting 1 is a lie about the outcome.
            print(f">> {jid} is ALREADY queued at {new_box} — nothing to move")
            _retarget_drop_stale(a, jid, new_box, stale)
            return
        others = [b for b in queued if b != new_box]
        if len(others) == 1:
            old_box = others[0]
            print(f">> no ticket at jobs/queue/{a.from_box or old_box}/{jid}.json — "
                  f"the ticket is at {old_box} (fleetd moves tickets on eviction "
                  f"replacement; jobd never deletes them). Using {old_box} as the "
                  f"source")
            ticket = jobmeta.read_ticket(old_box, jid)  # type: ignore[no-untyped-call]
        elif len(others) > 1:
            sys.exit(f"error: {jid} is queued on MORE THAN ONE box "
                     f"({', '.join(others)}) — that is a double-run in waiting. "
                     f"Delete the wrong pointer(s) by hand, then retarget")
    if ticket is None:
        if not getattr(a, "reconstruct", False):
            sys.exit(
                f"error: no ticket for {jid} anywhere under jobs/queue/ "
                f"(checked {old_box} and the whole queue). jobd does not delete "
                f"tickets, so this was moved or cleaned up out of band. Rebuild it "
                f"from the submitted bundle with --reconstruct (submit-time --env "
                f"pins are NOT recoverable), or `job submit` a new JOB_ID")
        ticket = _retarget_reconstruct(a, v, jid, new_box, old_box,
                                       _retarget_queued_boxes(jid))
    else:
        ticket = dict(ticket)
        ticket["box"] = new_box
        ticket["retargeted_from"] = old_box
    _vram_advisory(ticket.get("config"), where="retarget")
    stale.discard(new_box)
    if a.dry_run:
        print(f"[dry-run] would write jobs/queue/{new_box}/{jid}.json, delete "
              + ", ".join(f"jobs/queue/{b}/{jid}.json" for b in sorted(stale))
              + ", emit `retargeted`")
        return
    ok, key, err = jobmeta.write_ticket(ticket)  # type: ignore[no-untyped-call]
    if not ok:
        sys.exit(f"error: ticket write failed: {err}")
    _retarget_drop_stale(a, jid, new_box, stale)
    jobmeta.emit_event(jid, "retargeted", box=new_box, from_box=old_box)
    # A standing watch on the destination re-arms on a TICKET, and this is one.
    # Its own poll cannot see this: it reads nothing while the box is parked and
    # `unknown` when the B2 listing blips (2026-08-27 — a retarget onto a drained
    # box left it evicted with no rescue and no replacement).
    fleet_client.fleet_ticket_placed(new_box, jid, source="job retarget")
    print(f">> {jid}: {old_box} -> {new_box} ({key})")
    print(f">> the new box must run jobd: `herdd job attach {new_box}` if it is not")
    # This is the moment the ladder becomes safe to arm on the NEW box: the
    # ticket is non-terminal and in its queue, so a `jobs` watch cannot read the
    # queue as drained and park what was just rented. Nothing on the retarget
    # path registers a watch, and a `launch --jobs` box carries a BARE one, so
    # without this line the box rides out its bid undefended (2026-08-25). It is
    # silent where a spend-capable watch already covers the box — the wake above
    # has re-armed it and telling an operator to re-register it invites a second
    # cap on the same money.
    fleet_client.print_jobs_ticket_hint(new_box)


# moved-from: herdd._retarget_drop_stale
def _retarget_drop_stale(a: Any,  # noqa: ANN401 — argparse.Namespace
                         jid: str, new_box: str, boxes: Any) -> None:  # noqa: ANN401
    """Delete every leftover queue pointer for JID outside `new_box`.

    Belt-and-suspenders, mirroring `cmd_job_orphans`' pre-write box check: the
    ONE box this must never touch is the target's, so assert it rather than trust
    the caller's set arithmetic — a future refactor of the scan cannot then
    quietly delete the ticket it just wrote."""
    for b in sorted(boxes):
        if b == new_box:
            sys.exit(f"error: refusing to delete jobs/queue/{b}/{jid}.json — that "
                     f"is the retarget TARGET (internal inconsistency)")
        if getattr(a, "dry_run", False):
            print(f"[dry-run]   delete jobs/queue/{b}/{jid}.json")
            continue
        ok, err = jobmeta.delete_ticket(b, jid)  # type: ignore[no-untyped-call]
        if not ok:
            print(f"!! old ticket delete failed ({err}) — if box {b} resumes, "
                  f"it may double-run {jid}; delete jobs/queue/{b}/{jid}.json "
                  f"by hand")


# --------------------------------------------------------------------------- #
# requeue — re-open a TERMINAL-FAILED job under the same JOB_ID
# --------------------------------------------------------------------------- #

# moved-from: herdd._requeue_refusal
def _requeue_refusal(v: Mapping[str, Any]) -> str | None:
    """The named reason `job requeue` refuses a job, or None when it is eligible.
    Pure (takes a fold view) so the policy is testable without B2.

    ELIGIBLE = folded status is exactly `failed`. Everything else is refused with
    the reason spelled out, because each has a different correct next step."""
    st, disp = v.get("status"), v.get("display_status")
    if v.get("n_events", 0) == 0:
        return ("no events for this JOB_ID — check the id (`job ls`). requeue "
                "re-opens an EXISTING job; it never invents one")
    if st == "done":
        return ("the job is `done` (the entrypoint reached rc=0 and published "
                "results) — `done` is STICKY and never re-opens. Run it again "
                "with `job submit` (a new JOB_ID), or `job pull` the results")
    if st == "cancelled":
        return ("the job is `cancelled` — an operator's explicit never-revive "
                "verdict, STICKY by design (JOBS_DESIGN 'Cancel'). Re-run it "
                "with `job submit` (a new JOB_ID)")
    if v.get("reopened"):
        return (f"already re-opened by an earlier requeue (now {disp}, resumed at "
                f"{v.get('last_resumed_ts')}) — let it run, or `job cancel` it "
                f"first. Requeueing a live job would double-run it")
    if st in ("claimed", "started"):
        return (f"the job is {disp} on box {v.get('instance_id')} — requeue is for "
                f"TERMINAL-FAILED jobs only. Use `job retarget` to move an "
                f"interrupted job, or `job cancel` a running one")
    if st == "submitted":
        return (f"the job is still QUEUED on box {v.get('target_box')} (never "
                f"claimed) — nothing failed. Use `job retarget` to move the ticket")
    if st != "failed":
        return f"status is `{st}` (display {disp}), not `failed`"
    return None


# moved-from: herdd.cmd_job_requeue
def cmd_job_requeue(a: Any) -> None:  # noqa: ANN401 — argparse.Namespace
    """Re-open a TERMINAL-FAILED job onto another box, in ONE command.

    The recovery this replaces (executed by hand 2026-07-30 for two infra-killed
    waves): `job submit` the bundle onto a dead-letter box to get a non-terminal
    ticket, `rclone copy` the old job's checkpoints/ under the NEW job id, then
    `job retarget` onto a live box. Three commands, a JOB_ID split that has to be
    disclosed in the readout, and an env-pin archaeology step.

    requeue keeps the SAME JOB_ID, so there is no checkpoint copy (the prefix is
    already jobs/<JOB_ID>/checkpoints/) and no split to disclose. It is for
    INFRA-killed jobs — a box-level cause (wrong driver, a tree flock two waves
    deep, a dead mount), not a code bug: the bundle is required to be
    BYTE-IDENTICAL to the one that failed, so a requeue can only ever re-run the
    same experiment. There is deliberately NO --allow-bundle-drift: a changed
    bundle is a DIFFERENT experiment, and letting it inherit the failed job's
    event log + checkpoints would silently mix two runs under one id. Edit the
    bundle => `job submit` (new JOB_ID).

    Fail-closed at three gates: the status must fold to `failed`, the recomputed
    bundle sha must equal the `submitted` event's, and the target box must not be
    the one that failed it."""
    b2._ensure_b2_remote()
    jid = a.job_id
    try:
        jobmeta.validate_job_id(jid)
    except jobmeta.JobmetaError as e:
        sys.exit(f"error: {e}")
    src = os.path.abspath(a.bundle)
    if not os.path.isdir(src):
        sys.exit(f"error: --bundle is not a directory: {a.bundle}")
    new_box = str(a.box)

    # GATE 1 — status. `--fresh` semantics deliberately: a cached fold that is a
    # few minutes behind is exactly how you requeue a job that is still running.
    v = jobmeta.read_job_fresh(jid, live_iids=view._live_iids_set())
    why = _requeue_refusal(v)
    if why:
        sys.exit(f"error: refusing to requeue {jid}: {why}")

    old_box = str(a.from_box or v.get("instance_id") or v.get("target_box") or "")
    if not old_box:
        sys.exit("error: cannot determine the box the job failed on (pass --from)")
    # GATE 3 — a box that already ran this job carries a LOCAL terminal breadcrumb
    # ($STATE_DIR/<JOB_ID>.terminal) that its jobd checks before any B2 read, so a
    # ticket requeued back onto it is skipped forever, silently. Refuse instead.
    if old_box == new_box:
        sys.exit(f"error: refusing to requeue {jid} onto {new_box} — that is the box "
                 f"it failed on, and its jobd's local terminal cache would skip the "
                 f"ticket forever. Requeue onto a DIFFERENT box")

    # GATE 2 — bundle identity. Same content address `job submit` computes (sha256
    # of the deterministic tar), compared against the `submitted` event's record.
    want_sha = v.get("bundle_sha256")
    if not want_sha:
        sys.exit(f"error: {jid} has no `submitted` event carrying a bundle_sha256 "
                 f"— cannot verify the bundle; nothing to reconstruct a ticket from")
    try:
        sha = jobmeta.bundle_sha256(src)
    except jobmeta.JobmetaError as e:
        sys.exit(f"error: {e}")
    if sha != want_sha:
        sys.exit(
            f"error: bundle DRIFT — {src}\n"
            f"       recomputed {sha}\n"
            f"       submitted  {want_sha}\n"
            f"       A requeue re-runs the SAME experiment under the SAME JOB_ID; a "
            f"changed bundle is a different one. Check out the bundle as it was, or "
            f"`job submit` it fresh (new JOB_ID).\n"
            f"       NB the address covers the bundle's `includes:` too, so a "
            f"bundle folder that git says is UNCHANGED still drifts when "
            f"tools/vast/jobcommon/ moves — check that out at the same revision.")

    # ticket config: prefer the ORIGINAL ticket if its queue pointer survived —
    # that carries the exact `--env` pins the first submit folded in, which the
    # bundle folder does NOT contain (env overrides are ticket-side; the bundle
    # sha is invariant under them). Otherwise rebuild from the bundle and say so,
    # loudly, because silently dropping an env pin is how a "same" re-run differs.
    src_ticket = jobmeta.read_ticket(old_box, jid)  # type: ignore[no-untyped-call]
    if src_ticket and src_ticket.get("config"):
        cfg = src_ticket["config"]
        print(f">> ticket config: reused from the surviving queue ticket "
              f"jobs/queue/{old_box}/{jid}.json (env pins preserved)")
        if getattr(a, "env", None) or getattr(a, "artifact", None):
            flag = "--env" if getattr(a, "env", None) else "--artifact"
            sys.exit(f"error: {flag} is only for the REBUILT-from-bundle path; this "
                     "job's original ticket survived and carries its own env. "
                     f"Drop {flag} (or delete the stale ticket if you mean to change it)")
    else:
        try:
            raw = jobmeta.load_job_config(src)
            art_keys = submit._apply_artifact_env(raw, getattr(a, "artifact", None))
            env_keys = submit._apply_env_overrides(raw, getattr(a, "env", None))
            cfg, warnings = jobmeta.validate_job_config(raw, src)
        except jobmeta.JobmetaError as e:
            sys.exit(f"error: {e}")
        for w in warnings:
            print(f"warn: {w}", file=sys.stderr)
        print(f">> ticket config: REBUILT from {src} (no surviving queue ticket at "
              f"jobs/queue/{old_box}/{jid}.json)")
        # five more placeholder-free `f` prefixes; see `_retarget_reconstruct`.
        print(f"!! submit-time `--env` pins are NOT in the bundle — if the original "  # noqa: F541
              f"submit used any, pass them again with --env K=V or this re-run "      # noqa: F541
              f"differs from the one that failed "                                    # noqa: F541
              f"(the original values are usually in the checkpointed "                # noqa: F541
              f"results/input-manifest.json)")                                        # noqa: F541
        if art_keys:
            print(f">> artifact env (from the modelkit registry): "
                  f"{', '.join(sorted(art_keys))}")
        if env_keys:
            print(f">> env override (requeue-time): {', '.join(env_keys)}")

    _vram_advisory(cfg, where="requeue")
    print(f">> requeue {jid}: {old_box} -> {new_box} "
          f"(prior rc={v.get('prior_rc') if v.get('reopened') else v.get('rc')} "
          f"reason={v.get('fail_reason') or v.get('prior_fail_reason') or '-'})")
    print(f">> bundle {sha[:12]}… VERIFIED identical to the submitted bundle")
    print(f">> checkpoints: same JOB_ID -> jobs/{jid}/checkpoints/ is pulled back by "
          f"the new box's jobd (`retargeted_from` on the ticket) — no copy needed")

    if a.dry_run:
        print(f"[dry-run] would write jobs/queue/{new_box}/{jid}.json "
              f"(retargeted_from={old_box}, {jobmeta.REQUEUE_TICKET_MARK}) and emit "
              f"`resumed` (kind=requeue) — no B2 mutations")
        return

    # the bundle object should already be on B2 from the original submit; a
    # re-upload is cheap insurance against a lifecycle-expired object (the box
    # would otherwise die at "bundle download failed").
    if not jobmeta.bundle_exists(sha):
        print(">> bundle object MISSING on B2 (expired/never landed) — re-uploading")
        staging = os.path.join(view._REPO_ROOT, "out", "jobs", "_bundles")
        tmp_out = os.path.join(staging, "pending.tar.zst")
        info = jobmeta.write_bundle(src, tmp_out)
        final_out = os.path.join(staging, f"{info['sha256']}.tar.zst")
        os.replace(tmp_out, final_out)
        ok, err = jobmeta.upload_bundle(final_out, sha)  # type: ignore[no-untyped-call]
        if not ok:
            sys.exit(f"error: bundle re-upload failed: {err}")

    try:
        res = jobmeta.requeue_ticket(jid, new_box, cfg, sha, old_box=old_box,
                                     attempt=(v.get("attempts") or 0) + 1,
                                     actor=lifecycle._cli_actor())
    except jobmeta.JobmetaError as e:
        sys.exit(f"error: {e}")
    print(f">> ticket: {res['key']}")
    # Same seam as retarget: a live ticket now exists on `new_box`, so a dormant
    # standing watch there re-arms instead of waiting on its own queue poll.
    fleet_client.fleet_ticket_placed(new_box, jid, source="job requeue")
    print(f">> emitted `resumed` (kind=requeue, retargeted_from={old_box}) — the "
          f"fold re-opens: `failed` is no longer sticky for {jid}")
    if res.get("old_ticket_deleted") is False:
        print(f"!! stale ticket delete failed ({res.get('delete_err')}) — if box "
              f"{old_box} resumes it may double-run {jid}; delete "
              f"jobs/queue/{old_box}/{jid}.json by hand")
    prog = os.path.basename(sys.argv[0])
    print(f">> the target box must run a jobd that HONOURS the requeue mark "
          f"(2026-07-31+): a failed run publishes results.DONE.json before it "
          f"emits `failed`, and an older jobd skips any ticket whose DONE marker "
          f"exists. Re-attach to be sure: {prog} job attach {new_box}")
    print(f">>   status : {prog} job status {jid} --fresh")
    print(f">>   logs   : {prog} job logs {jid}")
    # Same seam as retarget: the ticket now exists on the new box, which is
    # both the earliest and the last convenient moment to arm the ladder.
    fleet_client.print_jobs_ticket_hint(new_box)


# --------------------------------------------------------------------------- #
# cancel — terminal + non-resumable, in three writes
# --------------------------------------------------------------------------- #

# moved-from: herdd._job_cancel_kill_script
def _job_cancel_kill_script(jid: object) -> str:
    """A base64-shipped remote kill script for `job cancel --hard`. Ships as a
    FILE run via `bash /tmp/…` so the remote cmdline never contains the JOB_ID —
    a bare `pkill -f <jid>` over ssh --exec would match its OWN wrapper cmdline
    and can kill the session (the pkill-self-match footgun). Best-effort: prefers
    the recorded runner pid tree, falls back to a workdir-path pkill."""
    return (
        "#!/usr/bin/env bash\n"
        "set -u\n"
        f"JID={shlex.quote(str(jid))}\n"
        'RF="/workspace/jobs/$JID/.running"\n'
        'kt(){ local p="$1" c; for c in $(pgrep -P "$p" 2>/dev/null || '
        'ps -o pid= --ppid "$p" 2>/dev/null); do kt "$c"; done; '
        'kill -KILL "$p" 2>/dev/null || true; }\n'
        'if [ -f "$RF" ]; then read -r pid _ < "$RF"; '
        '[ -n "${pid:-}" ] && kt "$pid"; fi\n'
        'pkill -KILL -f "/workspace/jobs/$JID/" 2>/dev/null || true\n'
        'rm -f "$RF" 2>/dev/null || true\n'
        'echo "job cancel --hard: killed process tree for $JID"\n')


# moved-from: herdd._ssh_kill_job
def _ssh_kill_job(iid: object, jid: object) -> None:
    """Belt-and-suspenders remote kill for `job cancel --hard`: ssh to the box and
    run the base64-shipped kill script. Best-effort — a failure here is non-fatal
    (the B2-side cancel already made the job terminal + non-resumable)."""
    try:
        i = lifecycle._get_instance(iid)
    except SystemExit:
        print(f"!! --hard: instance {iid} not found; skipped the ssh kill "
              f"(B2-side cancel still applied)")
        return
    host, port, _ = ssh._pick_ssh_endpoint(i)
    if not (host and port):
        print(f"!! --hard: no ssh endpoint for {iid} "
              f"(status={i.get('actual_status')}); skipped the ssh kill")
        return
    b64 = base64.b64encode(_job_cancel_kill_script(jid).encode()).decode("ascii")
    remote = (f"echo {b64} | base64 -d > /tmp/jobcancel_kill.sh && "
              f"bash /tmp/jobcancel_kill.sh; rm -f /tmp/jobcancel_kill.sh")
    sshcmd = ["ssh", "-p", str(port), f"root@{host}",
              "-o", "StrictHostKeyChecking=accept-new", "-o", "LogLevel=ERROR", remote]
    r = subprocess.run(sshcmd, capture_output=True, text=True)
    if r.returncode == 0:
        print(f">> --hard: {(r.stdout or '').strip() or 'remote kill ran'}")
    else:
        print(f"!! --hard: remote kill exited {r.returncode} "
              f"({(r.stderr or '').strip()}) — B2-side cancel still applied")


# moved-from: herdd._job_cancel_writes
def _job_cancel_writes(jid: str, box: str | None, *, reason: str, actor: str,
                       # `Any` is the CONTRACT here, not a gap: `**extra` rides
                       # straight onto the B2 event, whose schema tolerates
                       # unknown keys by design (§4 frozen). Narrowing it to
                       # `str` would refuse a future numeric/boolean evidence
                       # field that the fold would happily carry.
                       **extra: Any) -> list[str]:  # noqa: ANN401
    """The three B2 writes that make a job terminal + NON-resumable, in the order
    that keeps each one independently correct. ONE copy, shared by `job cancel`
    and `job orphans --resolve` — an orphan resolution is an ordinary cancel
    whose reason was composed from evidence, not a parallel mechanism.

    `extra` rides onto the `cancelled` event as additional fields (the fold
    tolerates unknown keys by contract), which is how the orphan lane records
    its machine-checked evidence next to the operator's note.

    Returns a list of warning lines for the caller to print; raises nothing."""
    warn = []
    # 1) CANCEL marker FIRST: a running box's jobd must be able to see it and kill
    #    the entrypoint (the ticket-delete below cannot stop an already-running job).
    ok, err = jobmeta.write_cancel_marker(  # type: ignore[no-untyped-call]
        jid, actor=actor, reason=reason)
    if not ok:
        warn.append(f"!! CANCEL marker write failed ({err}) — a running box may not "
                    f"stop; use --hard or park the box")
    # 2) terminal `cancelled` event: folds the job non-resumable even if the box
    #    never runs again (unreachable/parked/already-interrupted/destroyed).
    jobmeta.emit_event(jid, "cancelled", actor=actor, reason=reason,
                       box=box or None, **extra)
    # 3) delete the queue ticket so `job ls` drops it and NO box reclaims/resumes it.
    if box:
        ok, err = jobmeta.delete_ticket(box, jid)  # type: ignore[no-untyped-call]
        if not ok:
            warn.append(f"!! ticket delete failed ({err}) — if box {box} resumes it "
                        f"should honor the CANCEL marker, but delete "
                        f"jobs/queue/{box}/{jid}.json by hand")
    return warn


# moved-from: herdd.cmd_job_cancel
def cmd_job_cancel(a: Any) -> None:  # noqa: ANN401 — argparse.Namespace
    """Cancel a job into a TERMINAL, NON-resumable `cancelled` state (JOBS_DESIGN).

    Unlike `interrupted` (which jobd resumes on the next box boot), a cancelled
    job stays dead: (1) a terminal `cancelled` event folds the job to `cancelled`;
    (2) the queue ticket is DELETED so `job ls` drops it and no box ever reclaims/
    resumes it; (3) a CANCEL marker tells a box that is RUNNING the job right now
    to kill the entrypoint's process tree and record its own terminal `cancelled`.
    Steps (1)+(2) are correct even if the box is unreachable/parked. `--hard`
    additionally ssh's to a live box and kills the process tree directly.
    Idempotent: cancelling an already-terminal job is a clean no-op."""
    b2._ensure_b2_remote()
    jid = a.job_id
    try:
        jobmeta.validate_job_id(jid)
    except jobmeta.JobmetaError as e:
        sys.exit(f"error: {e}")
    v = jobmeta.read_job(jid, live_iids=view._live_iids_set())

    # resolve the owning box: explicit flag, else the event fold, else scan the
    # queues for the ticket (a still-queued, never-claimed job).
    box = str(a.box or v.get("instance_id") or v.get("target_box") or "")
    if not box:
        # An unreadable queue must not become "unknown job — nothing to cancel".
        try:
            queued = jobmeta.list_all_queued()
        except jobmeta.QueueUnreadable as e:
            sys.exit(f"error: {e}\n"
                     f"       Cannot resolve the box for {jid}; pass --box to "
                     f"cancel without the queue listing.")
        for b, j in queued:
            if j == jid:
                box = str(b); break                # noqa: E702 — verbatim (plan §7.4)

    # unknown job: no events AND no ticket anywhere -> nothing to cancel.
    if v["n_events"] == 0 and not box:
        sys.exit(f"error: unknown job {jid} (no events, no queue ticket) — "
                 f"check the JOB_ID (`{os.path.basename(sys.argv[0])} job ls`)")

    if v["status"] in jobmeta.TERMINAL:
        print(f">> {jid} already terminal ({v['status']}) — nothing to cancel")
        return

    reason = a.reason or "cancelled by operator"
    was_running = v["display_status"] == "running"
    if a.dry_run:
        print(f"[dry-run] would cancel {jid} (fold={v['display_status']}, box={box or '?'}):")
        print(f"[dry-run]   write marker jobs/{jid}/CANCEL")
        print(f"[dry-run]   emit terminal `cancelled` event (reason: {reason})")
        if box:
            print(f"[dry-run]   delete ticket jobs/queue/{box}/{jid}.json")
        if a.hard and box:
            print(f"[dry-run]   --hard: ssh {box} + kill the job's process tree")
        return

    actor = lifecycle._cli_actor()
    for w in _job_cancel_writes(jid, box, reason=reason, actor=actor):
        print(w)
    print(f">> {jid}: cancelled (terminal, non-resumable){' box=' + box if box else ''}")
    # 4) --hard: kill the on-box process tree now (belt-and-suspenders for when the
    #    cooperative watch is too slow or jobd is dead but the box is reachable).
    if a.hard:
        if box and box in view._live_iids_set():
            _ssh_kill_job(box, jid)
        elif box:
            print(f"!! --hard: box {box} is not live — skipped the ssh kill "
                  f"(cooperative CANCEL marker still applies if it resumes)")
        else:
            print("!! --hard: no owning box known — skipped the ssh kill")
    if was_running:
        print(">> the running box's jobd kills the entrypoint on its next "
              "CANCEL-poll (~15s default); confirm: "
              f"{os.path.basename(sys.argv[0])} job status {jid}")


# --------------------------------------------------------------------------- #
# flush — ask a RUNNING box to checkpoint now (the opposite of cancel)
# --------------------------------------------------------------------------- #

def cmd_job_flush(a: Any) -> None:  # noqa: ANN401 — argparse.Namespace
    """Ask the box running a job to fire ONE unfiltered checkpoint sync now.

    Writes jobs/<id>/CHECKPOINT_NOW. jobd sees it on the same poll that watches
    CANCEL, ships the whole declared checkpoint glob with no --min-age, and
    deletes the marker. The entrypoint is never signalled — this is a flush, not
    a stop, and nothing about the job's state changes.

    It CANNOT rescue an eviction: vast delivers no SIGTERM on a spot reclaim and
    the warning budget is single-digit seconds, far short of one marker poll. The
    use is a pre-park or pre-handoff flush, where you decide when to stop."""
    b2._ensure_b2_remote()
    jid = a.job_id
    try:
        jobmeta.validate_job_id(jid)
    except jobmeta.JobmetaError as e:
        sys.exit(f"error: {e}")
    v = jobmeta.read_job(jid, live_iids=view._live_iids_set())
    if v["n_events"] == 0:
        sys.exit(f"error: unknown job {jid} (no events) — check the JOB_ID "
                 f"(`{os.path.basename(sys.argv[0])} job ls`)")
    # A terminal job has no running entrypoint, so nothing would ever consume the
    # marker — refuse rather than leave litter on B2.
    if v["status"] in jobmeta.TERMINAL:
        sys.exit(f"error: {jid} is terminal ({v['status']}) — nothing is running "
                 f"to flush")
    if v["display_status"] != "running":
        print(f"!! {jid} folds as {v['display_status']}, not running — the marker "
              f"will sit until a box actually runs the job")
    if a.dry_run:
        print(f"[dry-run] would flush {jid} (fold={v['display_status']}):")
        print(f"[dry-run]   write marker jobs/{jid}/CHECKPOINT_NOW")
        return
    if jobmeta.has_checkpoint_now_marker(jid):
        print(f">> {jid}: a CHECKPOINT_NOW marker is already pending (not yet "
              f"consumed) — overwriting it; still at most one flush")
    ok, err = jobmeta.write_checkpoint_now_marker(  # type: ignore[no-untyped-call]
        jid, actor=lifecycle._cli_actor(), reason=a.reason or None)
    if not ok:
        sys.exit(f"error: CHECKPOINT_NOW write failed ({err}) — no flush requested")
    print(f">> {jid}: flush requested; the box's jobd ships the whole checkpoint "
          f"glob on its next marker poll (~15s) + watch tick (~5s)")
    print(">> this does NOT stop the job, and it cannot rescue an eviction — "
          "use it before a park or a handoff")


# --------------------------------------------------------------------------- #
# orphans — tickets whose target box no longer exists
# --------------------------------------------------------------------------- #

# moved-from: herdd._job_orphan_scan
def _job_orphan_scan(box_filter: object = None, job_filter: object = None,
                     ) -> tuple[list[dict[str, Any]], set[str] | None]:
    """Impure scan of the whole queue: `(rows, present)`.

    Each row: {box, job_id, status, display_status, n_events, verdict, why,
    done_marker}.
    `present` is `_present_iids_set()`'s three-valued answer, passed back so the
    caller can say "unreadable" rather than "clean" when the API failed.

    The folds come from `scan.fold_many` — ONE bulk listing plus at most one
    bulk fetch — not from a `read_job` per ticket. Measured on the live queue
    2026-08-17: 275 tickets took 139.7 s the old way, of which 138.3 s was 275
    `rclone copy` subprocesses that transferred nothing. The fold is not cached
    and the key set is re-listed on every call; `scan.py`'s docstring states the
    freshness contract and says why a TTL cache would re-create the 2026-07-30
    "already failed, still reads as submitted" incident this command exists to
    detect."""
    import parked_lifecycle as _pl  # function-local, verbatim (plan §7.4)
    try:
        if box_filter:
            pairs = [(str(box_filter), j)
                     for j in jobmeta.list_queue(str(box_filter))]
        else:
            pairs = jobmeta.list_all_queued()
    except jobmeta.QueueUnreadable as e:
        # Same rule as the bulk-fold guard below: do NOT degrade to "no tickets",
        # which reports a fleet-wide clean bill of health from a failed listing.
        sys.exit(f"error: {e}")
    if job_filter:
        pairs = [(b, j) for b, j in pairs if j == job_filter]
    present = view._present_iids_set()
    live = view._live_iids_set()
    try:
        folds = scan.fold_many([j for _b, j in pairs], live_iids=live)
    except Exception as e:
        # The whole listing failed. Do NOT degrade to "no events" — that folds
        # every ticket to unclaimed and mints a fleet-wide false orphan report.
        sys.exit(f"error: bulk job scan failed: {e}")
    rows = []
    for box, jid in pairs:
        v = folds.get(jid)
        if v is None or v.get("scan_error"):         # one bad log never hides the rest
            why = (v or {}).get("scan_error", "no fold returned")
            rows.append({"box": box, "job_id": jid, "status": "unknown",
                         "display_status": "unknown", "n_events": 0,
                         "done_marker": (v or {}).get("done_marker"),
                         "verdict": _pl.TICKET_UNKNOWN, "why": f"unreadable: {why}"})
            continue
        verdict, why = _pl.ticket_orphan_verdict(
            box_present=(None if present is None else box in present),
            job_status=v["status"])
        rows.append({"box": box, "job_id": jid, "status": v["status"],
                     "display_status": v["display_status"],
                     "n_events": v["n_events"],
                     # LIST-grade evidence that the job finished, free from the
                     # same listing. Reported, never used to mint a verdict —
                     # `read_job_fresh`'s `cat` is the adjudicating probe.
                     # TRI-STATE: None means NOT PROBED (the small-queue path
                     # does not list), which is not the same as "no marker".
                     # And it is evidence about SOME attempt, not this one: the
                     # marker survives a requeue, so `reopened_at` rides beside
                     # it as the boundary that dates it (jobmeta's
                     # `classify_done_marker`). A marker on a job with a
                     # `reopened_at` proves nothing on its own.
                     "done_marker": v.get("done_marker"),
                     "reopened_at": v.get("reopened_at"),
                     "verdict": verdict, "why": why})
    return rows, present


# moved-from: herdd.cmd_job_orphans
def cmd_job_orphans(a: Any) -> None:  # noqa: ANN401 — argparse.Namespace
    """Report — and, with `--resolve`, close out — queue tickets whose target box
    no longer exists (`parked_lifecycle.ticket_orphan_verdict`).

    WHY THIS IS A SEPARATE COMMAND AND NOT AUTOMATIC. The ORDERLY box death
    already moves its queue: the boot-pull watchdog retargets every pending
    ticket onto a replacement BEFORE destroying the condemned box. This lane is
    the disorderly remainder — a manual destroy, the idle reaper, an eviction
    nothing rescued — where the operator's intent is exactly what is missing
    from the record, and an automatic sweeper would have to guess it. Guessing
    "cancel" races a human who meant to `job retarget`; guessing "retarget"
    spends money. So detection is automatic and loud (`job ls` screams, this
    command explains) and RESOLUTION is explicit, one operator, with a recorded
    reason.

    Resolution IS a cancel — `_job_cancel_writes`, the same three writes as
    `job cancel` — because the state jobs-v2 already has for "terminal and never
    revive" is exactly right for an orphan; nothing about the state model was
    missing. What was missing is that nobody could SEE the orphan. The event
    additionally carries `orphan=<verdict>` and `orphan_box=<iid>`, so the log
    distinguishes "an operator killed this job" from "this job's box died under
    it and the operator swept the corpse"."""
    import parked_lifecycle as _pl  # function-local, verbatim (plan §7.4)
    b2._ensure_b2_remote()
    prog = os.path.basename(sys.argv[0])
    rows, present = _job_orphan_scan(a.box, a.job)

    if getattr(a, "json", False):
        print(json.dumps({"listing_readable": present is not None, "rows": rows},
                         indent=2))
    if present is None:
        msg = ("!! instance listing unreadable — cannot tell a destroyed box from a "
               "parked one, so NO orphan verdict was minted (a ticket on a dead box "
               "looks pending). Retry when the vast API answers.")
        if not getattr(a, "json", False):
            print(msg)
        sys.exit(1)
    if not rows:
        if not getattr(a, "json", False):
            print("no queued tickets.")
        return

    stuck = [r for r in rows if r["verdict"] in _pl.TICKET_ORPHANS_STUCK]
    stale = [r for r in rows if r["verdict"] == _pl.TICKET_ORPHAN_TERMINAL]

    if not getattr(a, "json", False):
        print(f"  {'BOX':<12} {'JOB_ID':<40} {'STATUS':<12} VERDICT")
        for r in sorted(rows, key=lambda r: (r["box"], r["job_id"])):
            if r["verdict"] == _pl.TICKET_OK and not a.all:
                continue
            print(f"  {r['box']:<12} {r['job_id']:<40} {r['status']:<12} {r['verdict']}")
        if stale:
            print(f"\n>> {len(stale)} stale pointer(s): the box is gone but the job "
                  f"already reached a terminal state — results/events are on B2 and "
                  f"nothing is stuck. Left in place ON PURPOSE: the ticket is what "
                  f"keeps the box's history visible in `{prog} job ls`.")
        if not stuck:
            print("\n>> no stuck orphans.")
            return
        print(f"\n!! {len(stuck)} STUCK orphan(s) — the target box does not exist, so "
              f"these never run and never end:")
        for r in stuck:
            print(f"   {r['job_id']}: {r['why']}")

    if not a.resolve:
        if stuck and not getattr(a, "json", False):
            print(f"\n   resolve (records WHY, and keeps the frozen config in "
                  f"the DLQ):\n"
                  f"     {prog} job orphans --resolve --reason '<why>' -y\n"
                  f"   an ORPHAN_INTERRUPTED job MAY have checkpoints — CHECK "
                  f"before moving it (`b2x ls jobs/<JOB_ID>/checkpoints/`) "
                  f"rather than assuming: measured on this bucket 2026-08-26, "
                  f"four of six interrupted orphans held nothing at all.\n"
                  f"   If it does hold work: `{prog} job retarget <JOB_ID> "
                  f"--box <LIVE>` — but note that re-runs the ticket's FROZEN "
                  f"config and bundle, not today's, and is refused past "
                  f"{jobmeta.STALE_TICKET_DAYS:g}d without --stale-ok. "
                  f"Resubmitting from the current bundle is usually right.")
        sys.exit(2 if stuck else 0)

    # --- resolve ------------------------------------------------------------
    note = (a.reason or "").strip()
    if not note:
        sys.exit("error: --resolve requires --reason: the point of resolving an "
                 "orphan through this path (rather than deleting the ticket) is "
                 "that the event log records WHY it was abandoned — e.g. "
                 "--reason 'superseded by the resubmitted arms under new JOB_IDs'")
    sel = [r for r in stuck
           if r["verdict"] == _pl.TICKET_ORPHAN_UNCLAIMED or a.include_interrupted]
    skipped = [r for r in stuck if r not in sel]
    for r in skipped:
        print(f">> SKIP {r['job_id']}: {r['verdict']} — the job ran and may have "
              f"checkpoints; move it ({prog} job retarget {r['job_id']} --box <LIVE>) "
              f"or pass --include-interrupted to cancel it anyway")
    if not sel:
        print(">> nothing to resolve."); return      # noqa: E702 — verbatim (plan §7.4)

    for r in sel:
        # Belt-and-suspenders: only a ticket whose box is provably ABSENT is ever
        # written to by this lane. `stuck` already guarantees it; re-assert here
        # so a future refactor of the filters cannot quietly widen the blast
        # radius onto a live box's queue.
        if r["box"] in present:
            sys.exit(f"error: refusing to resolve {r['job_id']} — box {r['box']} "
                     f"exists (internal inconsistency)")
    if not a.dry_run and not a.yes:
        sys.exit(f"error: --resolve writes to B2 ({len(sel)} ticket(s)); re-run "
                 f"with -y (or --dry-run to preview)")
    for r in sel:
        reason = (f"orphaned ticket: target box {r['box']} no longer exists in the "
                  f"vast account ({r['why']}); {note}")
        if a.dry_run:
            print(f"[dry-run] would resolve {r['job_id']} (box {r['box']}, "
                  f"{r['verdict']}):")
            print(f"[dry-run]   copy ticket -> {jobmeta.dlq_key(r['box'], r['job_id'])}")
            print(f"[dry-run]   write marker jobs/{r['job_id']}/CANCEL")
            print(f"[dry-run]   emit terminal `cancelled` (reason: {reason})")
            print(f"[dry-run]   delete ticket jobs/queue/{r['box']}/{r['job_id']}.json")
            continue
        # DLQ FIRST, and only then the cancel writes — the last of which deletes
        # the ticket. The ticket is the only place the frozen `config` lives
        # (the submitted event carries bundle_sha256/entrypoint/timeout_s, not
        # the env), so resolving an orphan used to be the one operation that
        # could destroy the record of what the job would have run. A DLQ write
        # that fails is reported and does NOT stop the cancel: ending a job that
        # can never run again is still the right outcome, but the operator must
        # know the frozen config went with it.
        try:
            _tk = jobmeta.read_ticket(r["box"], r["job_id"])  # type: ignore[no-untyped-call]
            if _tk is None:
                _ok, _key, _err = False, "", "no ticket to preserve"
            else:
                _ok, _key, _err = jobmeta.write_dlq_entry(  # type: ignore[no-untyped-call]
                    _tk, reason=reason, actor=lifecycle._cli_actor(),
                    verdict=r["verdict"])
        except Exception as _e:      # noqa: BLE001 — never block the cancel
            _ok, _key, _err = False, "", str(_e)
        if _ok:
            print(f">> retired to {_key} (frozen config preserved)")
        elif _err != "no ticket to preserve":
            print(f"!! DLQ write FAILED for {r['job_id']} ({_err}) — "
                  f"cancelling anyway, but the frozen config is NOT kept")
        for w in _job_cancel_writes(r["job_id"], r["box"], reason=reason,
                                    actor=lifecycle._cli_actor(),
                                    orphan=r["verdict"], orphan_box=r["box"]):
            print(w)
        print(f">> {r['job_id']}: cancelled as {r['verdict']} (terminal, "
              f"non-resumable) box={r['box']}")
    if not a.dry_run:
        print(f">> resolved {len(sel)} orphan(s). Confirm: {prog} job ls")
