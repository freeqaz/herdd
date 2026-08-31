"""vastlib.cli.job.logs — `herdd job logs`: the terminal log, or the live heartbeat tail.

Why this module exists
----------------------
One flag-less verb whose behavior forks on the job's state: a terminal job has
a `log.txt` to print, a running one has only the newest heartbeat's tail. The
handler prints WHICH of the two it read (`_job_log_provenance`), because
mistaking a stale heartbeat for the final log is how a failure gets missed.

What is deliberately NOT here
-----------------------------
* The handler. `cmd_job_logs` landed in `vastlib.jobs.view` at plan §8 step 5;
  this module is argparse-only and reaches it by module attribute.
* `--local`, added by the group's post-hoc loop (see `cli/job/__init__.py`).

Provenance: verbatim move of the `main()`-inline `pjl` parser block from
`tools/vast/herdd.py`, plan §8 step 6, 2026-08-16.
"""

from __future__ import annotations

import argparse

from vastlib.jobs import view


def run(a: argparse.Namespace) -> None:
    """§5 contract; body in `vastlib.jobs.view`, by module attribute."""
    view.cmd_job_logs(a)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Verbatim from `main()`: same help strings, same flag order, same dests."""
    pjl = sub.add_parser("logs", help="log.txt if terminal, else latest heartbeat tail")
    pjl.add_argument("job_id")
    pjl.set_defaults(jobfunc=run)
    return pjl
