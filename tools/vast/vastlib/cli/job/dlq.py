"""`herdd job dlq` — the dead-letter queue for job tickets.

WHY A DLQ AND NOT JUST `job cancel`. Cancelling deletes the queue pointer, and
the pointer is the only place a job's frozen `config` is recorded: the
`submitted` event carries `bundle_sha256`, `entrypoint` and `timeout_s`, but not
the resolved env. So `job cancel` on a ticket that never ran destroys the only
evidence of what it would have run with. The DLQ is a MOVE — same body, plus a
`dead_letter` block naming who retired it and why — under a prefix nothing
claims from, with `restore` as the deliberate way back.

The three verbs:

  add <JOB_ID> --reason '<why>'    retire a queue pointer (does NOT end the job)
  ls [--box IID]                   what has been retired, and why
  restore <JOB_ID> --box <LIVE>    put it back on a live box's queue

`add` deliberately does not emit a terminal `cancelled` event: retiring a
POINTER is not ending a JOB, and `cancelled` is unconditionally sticky in the
fold, so conflating the two would kill a job still wanted elsewhere. When you
mean "this job is over", use `job cancel` — or `job orphans --resolve`, which
does both.
"""
from __future__ import annotations

import argparse
import sys

from vastlib.cli import _args, _docs
from vastlib.storage import b2

import jobmeta


def _actor() -> str:
    import socket
    return f"cli:{socket.gethostname()}"


def cmd_add(a: argparse.Namespace) -> None:
    b2._ensure_b2_remote()
    jid = a.job_id
    box = str(a.box) if a.box else ""
    if not box:
        # Find the ticket without making the operator hunt for the box id.
        try:
            hits = [(bx, j) for bx, j in jobmeta.list_all_queued() if j == jid]
        except jobmeta.QueueUnreadable as e:
            sys.exit(f"error: {e}")
        if not hits:
            sys.exit(f"error: no queue ticket for {jid} — nothing to retire "
                     f"(it may already be dead-lettered: job dlq ls)")
        if len(hits) > 1:
            boxes = ", ".join(sorted(b for b, _ in hits))
            sys.exit(f"error: {jid} has tickets on several boxes ({boxes}) — "
                     f"pass --box to say which")
        box = hits[0][0]

    if a.dry_run:
        tk = jobmeta.read_ticket(box, jid)  # type: ignore[no-untyped-call]
        if tk is None:
            sys.exit(f"error: no ticket at jobs/queue/{box}/{jid}.json")
        age = jobmeta.ticket_age_days(tk)  # type: ignore[no-untyped-call]
        print(f"DRY-RUN would retire {jid}")
        print(f"  from   jobs/queue/{box}/{jid}.json")
        print(f"  to     {jobmeta.dlq_key(box, jid)}")
        print(f"  age    {'unknown' if age is None else f'{age:.1f}d'}")
        print(f"  bundle {str(tk.get('bundle_sha256') or '?')[:12]}")
        print(f"  reason {a.reason}")
        return

    res = jobmeta.dead_letter_ticket(box, jid, reason=a.reason,
                                     actor=_actor(), verdict=a.verdict)
    st = res["status"]
    if st == "dead_lettered":
        print(f"retired {jid} -> {res['key']}")
        if not res["ticket_deleted"]:
            print(f"!! the queue pointer survived the move ({res['delete_err']}) "
                  f"— delete jobs/queue/{box}/{jid}.json by hand, or the box "
                  f"can still claim it")
        return
    if st == "already_dead_lettered":
        print(f"{jid} is already in the DLQ — nothing to do")
        return
    if st == "no_ticket":
        sys.exit(f"error: no ticket at jobs/queue/{box}/{jid}.json")
    sys.exit(f"error: dead-letter write FAILED for {jid} ({res.get('err')}) — "
             f"the queue pointer was left in place on purpose; nothing was lost")


def cmd_ls(a: argparse.Namespace) -> None:
    b2._ensure_b2_remote()
    try:
        pairs = jobmeta.list_dlq(a.box)  # type: ignore[no-untyped-call]
    except jobmeta.QueueUnreadable as e:
        sys.exit(f"error: {e}")
    if not pairs:
        print("dead-letter queue is empty")
        return
    print(f"  {'JOB_ID':<46} {'BOX':<10} {'AGE@RETIRE':<11} REASON")
    for box, jid in pairs:
        entry = jobmeta.read_dlq_entry(box, jid) or {}  # type: ignore[no-untyped-call]
        dl = entry.get(jobmeta.DEAD_LETTER_MARK) or {}
        age = dl.get("age_days_at_retirement")
        age_s = "?" if age is None else f"{age:.1f}d"
        print(f"  {jid[:46]:<46} {box:<10} {age_s:<11} {dl.get('reason') or ''}")
    print(f"\n  {len(pairs)} retired ticket(s). The frozen config is preserved "
          f"in each entry.\n  restore one onto a live box: "
          f"herdd.py job dlq restore <JOB_ID> --box <LIVE_IID>")


def cmd_restore(a: argparse.Namespace) -> None:
    b2._ensure_b2_remote()
    jid = a.job_id
    try:
        pairs = [(bx, j) for bx, j in jobmeta.list_dlq() if j == jid]  # type: ignore[no-untyped-call]
    except jobmeta.QueueUnreadable as e:
        sys.exit(f"error: {e}")
    if not pairs:
        sys.exit(f"error: {jid} is not in the dead-letter queue")
    src_box = pairs[0][0]
    entry = jobmeta.read_dlq_entry(src_box, jid) or {}  # type: ignore[no-untyped-call]
    age = jobmeta.ticket_age_days(entry)  # type: ignore[no-untyped-call]
    if age is not None and age > jobmeta.STALE_TICKET_DAYS and not a.stale_ok:
        sys.exit(
            f"error: {jid} is {age:.1f}d old — restoring runs the bytes it was "
            f"frozen with at submit (bundle "
            f"{str(entry.get('bundle_sha256') or '?')[:12]}), not today's "
            f"bundle. Resubmit from the current bundle, or --stale-ok.")
    res = jobmeta.restore_dlq_entry(jid, src_box, str(a.box), actor=_actor())
    if res["status"] != "restored":
        sys.exit(f"error: restore failed for {jid}: {res}")
    print(f"restored {jid} -> jobs/queue/{res['box']}/{jid}.json")
    print("  the box's jobd will claim it on its next poll")


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    p = sub.add_parser("dlq", help="dead-letter queue: retire a job ticket "
                       "without destroying its frozen config, list what was "
                       "retired, restore one deliberately",
                       epilog=_args._docs_epilog(_docs.DOC_JOBS),
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    dsub = p.add_subparsers(dest="dlqcmd", required=True)

    pa = dsub.add_parser("add", help="retire a queue ticket into the DLQ "
                         "(does NOT emit a terminal event — use job cancel to "
                         "end the job itself)")
    pa.add_argument("job_id")
    pa.add_argument("--reason", required=True,
                    help="why this ticket is dead weight. Recorded in the "
                         "entry; a future reader has only this")
    pa.add_argument("--box", default=None,
                    help="source box (default: found by scanning the queue)")
    pa.add_argument("--verdict", default=None,
                    help="orphan verdict, when retiring from a scan")
    pa.add_argument("--dry-run", dest="dry_run", action="store_true")
    pa.set_defaults(jobfunc=cmd_add)

    pl = dsub.add_parser("ls", help="list retired tickets and why")
    pl.add_argument("--box", default=None, help="only this box's retirements")
    pl.set_defaults(jobfunc=cmd_ls)

    pr = dsub.add_parser("restore", help="put a retired ticket back on a live "
                         "box's queue")
    pr.add_argument("job_id")
    pr.add_argument("--box", required=True, help="LIVE target instance id")
    pr.add_argument("--stale-ok", dest="stale_ok", action="store_true",
                    help="restore even if the ticket is past the staleness "
                         "bound (it runs its frozen bytes, not today's)")
    pr.set_defaults(jobfunc=cmd_restore)
    return p
