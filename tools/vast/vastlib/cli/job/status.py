"""vastlib.cli.job.status — `herdd job status`: fold a job's event log.

Why this module exists
----------------------
`status` is the read every other job command's operator instinct starts from,
and its one subtlety is `--fresh`: the default read is cached and `--fast-list`
based, which is why a fold can say `submitted live=False` minutes after the box
actually moved. `--fresh` bypasses both, probes the strongly-consistent
`results.DONE.json`, and renders `live=n/a` while unclaimed. That distinction
lives entirely in the help text, so the text is the interface.

What is deliberately NOT here
-----------------------------
* The handler. `cmd_job_status` landed in `vastlib.jobs.view` at plan §8
  step 5; this module is argparse-only and reaches it by module attribute.
* `--local`, added by the group's post-hoc loop (see `cli/job/__init__.py`).

Provenance: verbatim move of the `main()`-inline `pjst` parser block from
`tools/vast/herdd.py`, plan §8 step 6, 2026-08-16.
"""

from __future__ import annotations

import argparse

from vastlib.jobs import view


def run(a: argparse.Namespace) -> None:
    """§5 contract; body in `vastlib.jobs.view`, by module attribute."""
    view.cmd_job_status(a)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Verbatim from `main()`: same help strings, same flag order, same dests."""
    pjst = sub.add_parser("status", help="fold a job's event log (+ live IID check)")
    pjst.add_argument("job_id")
    pjst.add_argument("--watch", action="store_true", help="poll until terminal")
    pjst.add_argument("--interval", type=int, default=15, help="--watch poll seconds")
    pjst.add_argument("--json", action="store_true")
    pjst.add_argument("--fresh", action="store_true",
                      help="bypass the local event cache AND --fast-list (per-key "
                           "reads), probe the strongly-consistent results.DONE.json, "
                           "and render live=n/a while unclaimed — for when a fold "
                           "says `submitted live=False` minutes after the box moved")
    pjst.set_defaults(jobfunc=run)
    return pjst
