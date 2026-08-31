"""vastlib.cli.job.retarget — `herdd job retarget`: move a ticket to another box.

Why this module exists
----------------------
Retarget keeps the SAME `JOB_ID`, so checkpoints continue — that is the whole
point, and it is why `--from` tolerates a stale value (the whole queue is
scanned when no ticket is where the fold says it is, because fleetd moves
tickets on eviction replacement). `--reconstruct` is the last-resort rung: mint
a pointer from the submitted bundle when none survives anywhere, refusing while
any ticket still exists, and it cannot recover submit-time `--env` pins.

What is deliberately NOT here
-----------------------------
* The handler and the queue arithmetic. `cmd_job_retarget`,
  `_retarget_queued_boxes`, `_retarget_drop_stale` and `_retarget_reconstruct`
  landed in `vastlib.jobs.control` at plan §8 step 5; this module is
  argparse-only and reaches the handler by module attribute.
* `--local`. Moving work between rented machines is a box concept, so
  `retarget` is deliberately absent from `_JOB_LOCAL_SUBCOMMANDS`.

Provenance: verbatim move of the `main()`-inline `pjr` parser block from
`tools/vast/herdd.py`, plan §8 step 6, 2026-08-16.
"""

from __future__ import annotations

import argparse

from vastlib.cli import _args, _docs
from vastlib.jobs import control

import jobmeta


def run(a: argparse.Namespace) -> None:
    """§5 contract; body in `vastlib.jobs.control`, by module attribute."""
    control.cmd_job_retarget(a)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Verbatim from `main()`: same help strings, same flag order, same dests."""
    pjr = sub.add_parser("retarget", help="move an interrupted/queued job's ticket "
                         "to another box (same JOB_ID; old ticket deleted)",
                         epilog=_args._docs_epilog(_docs.DOC_JOBS),
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    pjr.add_argument("job_id")
    pjr.add_argument("--box", required=True, help="new target instance id")
    pjr.add_argument("--from", dest="from_box", default=None,
                     help="source box (default: from the job's event fold). A "
                          "stale value is tolerated: the whole queue is scanned "
                          "when no ticket is there, since fleetd moves tickets on "
                          "eviction replacement")
    pjr.add_argument("--reconstruct", action="store_true",
                     help="when NO queue pointer survives anywhere, mint one from "
                          "the submitted bundle (same JOB_ID, checkpoints "
                          "continue). Refuses while any ticket still exists. "
                          "Submit-time --env pins are NOT recoverable")
    pjr.add_argument("--stale-ok", dest="stale_ok", action="store_true",
                     help=f"move a ticket older than "
                          f"{jobmeta.STALE_TICKET_DAYS:g}d anyway. A retarget "
                          f"re-runs the ticket's FROZEN config and bundle, not "
                          f"today's — repo-side fixes since submit are not "
                          f"picked up. Prefer resubmitting from the current "
                          f"bundle")
    pjr.add_argument("--dry-run", dest="dry_run", action="store_true")
    pjr.set_defaults(jobfunc=run)
    return pjr
