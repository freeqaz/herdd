"""vastlib.cli.job.requeue — `herdd job requeue`: re-open a terminal-FAILED job elsewhere.

Why this module exists
----------------------
Requeue is retarget's sibling for a job that already reached a terminal failed
state: same `JOB_ID`, same bundle, checkpoints continue. Two of its flags are
refusals rather than conveniences — `--bundle` must hash IDENTICALLY to the one
submitted (verified, with no drift override), and `--box` must not be the box
the job failed on. `--env` re-applies submit-time pins ONLY when the original
ticket is gone and the config has to be rebuilt.

What is deliberately NOT here
-----------------------------
* The handler and the refusal logic. `cmd_job_requeue`, `_requeue_refusal` and
  `_vram_advisory` landed in `vastlib.jobs.control` at plan §8 step 5; this
  module is argparse-only and reaches the handler by module attribute.
* `--local`. Re-opening work on another rented machine is a box concept, so
  `requeue` is deliberately absent from `_JOB_LOCAL_SUBCOMMANDS`.

Provenance: verbatim move of the `main()`-inline `pjq` parser block from
`tools/vast/herdd.py`, plan §8 step 6, 2026-08-16.
"""

from __future__ import annotations

import argparse

from vastlib.cli import _args, _docs
from vastlib.jobs import control


def run(a: argparse.Namespace) -> None:
    """§5 contract; body in `vastlib.jobs.control`, by module attribute."""
    control.cmd_job_requeue(a)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Verbatim from `main()`: same help strings, same flag order, same dests."""
    pjq = sub.add_parser("requeue", help="re-open a TERMINAL-FAILED job on another "
                         "box (same JOB_ID, same bundle, checkpoints continue)",
                         epilog=_args._docs_epilog(_docs.DOC_JOBS),
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    pjq.add_argument("job_id")
    pjq.add_argument("--box", required=True, help="target instance id (must NOT be "
                     "the box the job failed on)")
    pjq.add_argument("--bundle", required=True, metavar="DIR",
                     help="the job bundle directory — must hash IDENTICALLY to the "
                          "one submitted (verified; no drift override)")
    pjq.add_argument("--from", dest="from_box", default=None,
                     help="the box it failed on (default: from the job's event fold)")
    pjq.add_argument("--env", action="append", default=None, metavar="K=V",
                     help="re-apply a submit-time env pin (ONLY when the original "
                          "queue ticket is gone and the config is rebuilt)")
    pjq.add_argument("--artifact", action="append", default=None,
                     metavar="PREFIX=SLUG",
                     help="re-apply a submit-time modelkit-registry artifact pin "
                          "(same rebuilt-config-only rule as --env)")
    pjq.add_argument("--dry-run", dest="dry_run", action="store_true")
    pjq.set_defaults(jobfunc=run)
    return pjq
