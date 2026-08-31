"""vastlib.cli.job.cancel — `herdd job cancel`: terminal, NON-resumable.

Why this module exists
----------------------
Cancel is the one job verb whose result cannot be undone by another job verb:
the ticket is deleted and the state is terminal and non-resumable, so a
mistaken cancel is a requeue-from-scratch. The cooperative CANCEL marker is the
default; `--hard` additionally ssh's to a live box and kills the process tree
(belt-and-suspenders, not the primary mechanism).

What is deliberately NOT here
-----------------------------
* The handler. `cmd_job_cancel`, `_job_cancel_writes` and `_ssh_kill_job`
  landed in `vastlib.jobs.control` at plan §8 step 5; this module is
  argparse-only and reaches the handler by module attribute.
* `--local`, added by the group's post-hoc loop (see `cli/job/__init__.py`) —
  cancel IS in `_JOB_LOCAL_SUBCOMMANDS`.

Provenance: verbatim move of the `main()`-inline `pjc` parser block from
`tools/vast/herdd.py`, plan §8 step 6, 2026-08-16.
"""

from __future__ import annotations

import argparse

from vastlib.cli import _args, _docs
from vastlib.jobs import control


def run(a: argparse.Namespace) -> None:
    """§5 contract; body in `vastlib.jobs.control`, by module attribute."""
    control.cmd_job_cancel(a)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Verbatim from `main()`: same help strings, same flag order, same dests."""
    pjc = sub.add_parser("cancel", help="cancel a job into a terminal, "
                         "NON-resumable `cancelled` state (deletes the ticket + "
                         "kills a running box's entrypoint)",
                         epilog=_args._docs_epilog(_docs.DOC_JOBS),
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    pjc.add_argument("job_id")
    pjc.add_argument("--box", default=None,
                     help="owning instance id (default: from the event fold / queue scan)")
    pjc.add_argument("--reason", default=None, help="cancel reason (recorded on the event)")
    pjc.add_argument("--hard", action="store_true",
                     help="also ssh to a live box and kill the job's process tree now "
                          "(belt-and-suspenders; the cooperative CANCEL marker is default)")
    pjc.add_argument("--dry-run", dest="dry_run", action="store_true",
                     help="show the plan; NO B2 mutations, no ssh")
    pjc.set_defaults(jobfunc=run)
    return pjc
