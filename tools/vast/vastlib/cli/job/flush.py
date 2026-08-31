"""vastlib.cli.job.flush — `herdd job flush`: checkpoint now, keep running.

Why this module exists
----------------------
`cancel`'s inverse. Both write a marker under `jobs/<id>/` that a running box's
jobd sees on the SAME poll, but where CANCEL means "kill the entrypoint tree and
go terminal", CHECKPOINT_NOW means "ship the checkpoint glob and carry on". The
job's state is untouched, so unlike cancel this verb is safely repeatable.

The one thing it is not
-----------------------
An eviction rescue. vast delivers no SIGTERM on a spot reclaim and the warning
budget is single-digit seconds, so a trigger that has to be NOTICED over B2
cannot run inside it. This is for a flush you schedule — before a park, before a
handoff.

What is deliberately NOT here
-----------------------------
* The handler. `cmd_job_flush` lives in `vastlib.jobs.control` beside
  `cmd_job_cancel`; this module is argparse-only and reaches it by module
  attribute.
* `--local`: the local lane has no jobd poll loop to see the marker, so flush is
  not in `_JOB_LOCAL_SUBCOMMANDS`.
"""

from __future__ import annotations

import argparse

from vastlib.cli import _args, _docs
from vastlib.jobs import control


def run(a: argparse.Namespace) -> None:
    """§5 contract; body in `vastlib.jobs.control`, by module attribute."""
    control.cmd_job_flush(a)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    pjf = sub.add_parser("flush", help="ask the running box to checkpoint NOW "
                         "(one unfiltered sync; does NOT stop the job, and "
                         "cannot rescue an eviction)",
                         epilog=_args._docs_epilog(_docs.DOC_JOBS),
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    pjf.add_argument("job_id")
    pjf.add_argument("--reason", default=None,
                     help="why (recorded on the marker)")
    pjf.add_argument("--dry-run", dest="dry_run", action="store_true",
                     help="show the plan; NO B2 mutations")
    pjf.set_defaults(jobfunc=run)
    return pjf
