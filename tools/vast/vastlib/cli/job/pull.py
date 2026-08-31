"""vastlib.cli.job.pull — `herdd job pull`: download a job's results/ tree.

Why this module exists
----------------------
`pull` is the only job verb that writes to the workstation, and its default
destination — `out/jobs/<JOB_ID>/` under the repo root — is the reason results
do not land in `~/tmp` and go missing. The positional `dest` is optional and
stays optional.

What is deliberately NOT here
-----------------------------
* The handler. `cmd_job_pull` landed in `vastlib.jobs.view` at plan §8 step 5;
  this module is argparse-only and reaches it by module attribute.
* `--local`, added by the group's post-hoc loop (see `cli/job/__init__.py`).

Provenance: verbatim move of the `main()`-inline `pjp` parser block from
`tools/vast/herdd.py`, plan §8 step 6, 2026-08-16. `--allow-stale` was added
2026-08-28 for `view._pull_attempt_guard` and is the one flag that is not from
the original block.
"""

from __future__ import annotations

import argparse

from vastlib.jobs import view


def run(a: argparse.Namespace) -> None:
    """§5 contract; body in `vastlib.jobs.view`, by module attribute."""
    view.cmd_job_pull(a)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Verbatim from `main()`: same help strings, same flag order, same dests."""
    pjp = sub.add_parser("pull", help="download results/ -> out/jobs/<JOB_ID>/")
    pjp.add_argument("job_id")
    pjp.add_argument("dest", nargs="?", default=None)
    pjp.add_argument("--allow-stale", action="store_true",
                     help="pull even when results/ belongs to an attempt that "
                          "PREDATES the job's requeue (refused by default — the "
                          "tree is the dead attempt's, not the running one's)")
    pjp.set_defaults(jobfunc=run)
    return pjp
