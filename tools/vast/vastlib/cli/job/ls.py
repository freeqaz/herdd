"""vastlib.cli.job.ls — `herdd job ls`: queued tickets + their folded statuses.

Why this module exists
----------------------
`ls` is the TICKET side of the job surface; `job defs` is the DEFINITION side.
Keeping the two as separate commands (rather than one flagged verb) is what
makes "the bundle exists but no ticket does" a readable state instead of an
empty table.

What is deliberately NOT here
-----------------------------
* The handler. `cmd_job_ls` landed in `vastlib.jobs.view` at plan §8 step 5;
  this module is argparse-only and reaches it by module attribute.
* `--local`, added by the group's post-hoc loop (see `cli/job/__init__.py`).

Provenance: verbatim move of the `main()`-inline `pjls` parser block from
`tools/vast/herdd.py`, plan §8 step 6, 2026-08-16.
"""

from __future__ import annotations

import argparse

from vastlib.jobs import view


def run(a: argparse.Namespace) -> None:
    """§5 contract; body in `vastlib.jobs.view`, by module attribute."""
    view.cmd_job_ls(a)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Verbatim from `main()`: same help strings, same flag order, same dests."""
    pjls = sub.add_parser("ls", help="queued tickets + folded statuses")
    pjls.add_argument("--box", default=None, help="restrict to one instance id")
    pjls.set_defaults(jobfunc=run)
    return pjls
