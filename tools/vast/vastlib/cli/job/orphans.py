"""vastlib.cli.job.orphans — `herdd job orphans`: tickets whose target box is gone.

Why this module exists
----------------------
An orphan is a ticket that will pend forever because the box it names no longer
exists. `--resolve` is the only destructive path, and it is triple-gated:
`--reason` (recorded on the `cancelled` event next to the machine-checked
evidence), `-y`, and `--include-interrupted` before it will touch a ticket that
was CLAIMED before the box died — those may have checkpoints, and the honest
move for them is `job retarget` / `job requeue`, not cancellation. All three
gates are argparse-level, which is why they are stated here.

What is deliberately NOT here
-----------------------------
* The handler and the scan. `cmd_job_orphans` and `_job_orphan_scan` landed in
  `vastlib.jobs.control` at plan §8 step 5; this module is argparse-only and
  reaches the handler by module attribute.
* `--local`. `orphans` is not in `_JOB_LOCAL_SUBCOMMANDS`: a dead box is a
  box concept.

Provenance: verbatim move of the `main()`-inline `pjo` parser block from
`tools/vast/herdd.py`, plan §8 step 6, 2026-08-16.
"""

from __future__ import annotations

import argparse

from vastlib.cli import _args, _docs
from vastlib.jobs import control


def run(a: argparse.Namespace) -> None:
    """§5 contract; body in `vastlib.jobs.control`, by module attribute."""
    control.cmd_job_orphans(a)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Verbatim from `main()`: same help strings, same flag order, same dests."""
    pjo = sub.add_parser("orphans", help="tickets whose target box no longer "
                         "exists (pending forever); --resolve cancels them",
                         epilog=_args._docs_epilog(_docs.DOC_JOBS),
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    pjo.add_argument("--box", default=None, help="restrict to one instance id")
    pjo.add_argument("--job", default=None, metavar="JOB_ID",
                     help="restrict to one JOB_ID")
    pjo.add_argument("--all", action="store_true",
                     help="also list healthy (OK) tickets")
    pjo.add_argument("--json", action="store_true")
    pjo.add_argument("--resolve", action="store_true",
                     help="cancel the stuck orphans (terminal, non-resumable) — "
                          "requires --reason and -y")
    pjo.add_argument("--reason", default=None,
                     help="WHY these are being abandoned; recorded on the "
                          "`cancelled` event alongside the machine-checked "
                          "evidence (required with --resolve)")
    pjo.add_argument("--include-interrupted", dest="include_interrupted",
                     action="store_true",
                     help="also cancel orphans that were CLAIMED before the box "
                          "died — they may have checkpoints; prefer `job "
                          "retarget`/`job requeue` to move the work instead")
    pjo.add_argument("--dry-run", dest="dry_run", action="store_true",
                     help="show the plan; NO B2 mutations")
    pjo.add_argument("-y", "--yes", action="store_true")
    pjo.set_defaults(jobfunc=run)
    return pjo
