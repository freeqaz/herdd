"""vastlib.cli.job.defs — `herdd job defs`: bundle DEFINITIONS across all three homes.

Why this module exists
----------------------
Job bundles live in three places by design, and a definition that drifts into a
fourth is invisible to `job ls` (which lists tickets, not definitions). `defs`
is the definition-side twin: it walks `JOB_DEF_HOMES` and reports strays.

What is deliberately NOT here
-----------------------------
* The handler and the home roster. `cmd_job_defs`, `JOB_DEF_HOMES`,
  `find_job_defs` and `find_job_def_strays` landed in `vastlib.jobs.view` at
  plan §8 step 5; this module is argparse-only and reaches the handler by
  module attribute.
* `--local`. `defs` is not in `_JOB_LOCAL_SUBCOMMANDS` — definitions are repo
  state, not bucket state.

Provenance: verbatim move of the `main()`-inline `pjd` parser block from
`tools/vast/herdd.py`, plan §8 step 6, 2026-08-16.
"""

from __future__ import annotations

import argparse

from vastlib.cli import _args, _docs
from vastlib.jobs import view


def run(a: argparse.Namespace) -> None:
    """§5 contract; body in `vastlib.jobs.view`, by module attribute."""
    view.cmd_job_defs(a)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Verbatim from `main()`: same help strings, same flag order, same dests."""
    pjd = sub.add_parser("defs", help="job bundle DEFINITIONS across all three "
                         "homes (the definition-side twin of `job ls`, which "
                         "lists tickets)",
                         epilog=_args._docs_epilog(_docs.DOC_JOBS),
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    pjd.add_argument("--json", action="store_true")
    pjd.set_defaults(jobfunc=run)
    return pjd
