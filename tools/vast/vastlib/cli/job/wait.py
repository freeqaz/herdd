"""vastlib.cli.job.wait — `herdd job wait`: block until a job reaches a state.

Why this module exists
----------------------
`wait` exists to kill `status | grep` polling loops in scripts and agent
sessions, so its contract is an EXIT CODE contract: 124 on timeout, like
`timeout(1)`. The default `--until terminal` means any of done/failed/
cancelled. Both facts live in the help text and both are load-bearing for the
callers that branch on `$?`.

What is deliberately NOT here
-----------------------------
* The handler. `cmd_job_wait` landed in `vastlib.jobs.view` at plan §8 step 5;
  this module is argparse-only and reaches it by module attribute.
* `--local`, added by the group's post-hoc loop (see `cli/job/__init__.py`).

Provenance: verbatim move of the `main()`-inline `pjw` parser block from
`tools/vast/herdd.py`, plan §8 step 6, 2026-08-16.
"""

from __future__ import annotations

import argparse

from vastlib.cli import _args, _docs
from vastlib.jobs import view


def run(a: argparse.Namespace) -> None:
    """§5 contract; body in `vastlib.jobs.view`, by module attribute."""
    view.cmd_job_wait(a)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Verbatim from `main()`: same help strings, same flag order, same dests."""
    pjw = sub.add_parser("wait", help="block until a job reaches a state "
                         "(--until), with a timeout — replaces status|grep loops",
                         epilog=_args._docs_epilog(_docs.DOC_JOBS),
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    pjw.add_argument("job_id")
    pjw.add_argument("--until", default="terminal",
                     help="target state (default 'terminal' = any of "
                          "done/failed/cancelled); or a specific done|failed|"
                          "cancelled|running|interrupted|queued")
    pjw.add_argument("--timeout", type=int, default=1800,
                     help="max seconds to wait (default 1800; exit 124 on timeout)")
    pjw.add_argument("--interval", type=int, default=15, help="poll seconds (default 15)")
    pjw.add_argument("--json", action="store_true",
                     help="print the final job view as JSON on match")
    pjw.set_defaults(jobfunc=run)
    return pjw
