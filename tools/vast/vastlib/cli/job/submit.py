"""vastlib.cli.job.submit — `herdd job submit`: bundle a folder, queue it on a box.

Why this module exists
----------------------
`submit` is the widest flag block in the group (14 flags, seven of them
override switches) and every one of those flags is a REFUSAL being waived:
`--allow-stale-assets`, `--strict-assets`, `--no-asset-check`,
`--allow-vram-drift`, `--allow-unscoped-writes`, `--require-box-eval-pin`. The
help text is where an operator learns what each waiver actually costs, which
is why it is long and why it is byte-frozen.

What is deliberately NOT here
-----------------------------
* The handler. `cmd_job_submit` landed in `vastlib.jobs.submit` at plan §8
  step 5 (it is called by `cmd_job_run_local` too, so it could not wait for
  `cli/`). This module is argparse-only and reaches it by module attribute.
* `--local`. It is added by the group's post-hoc loop over
  `runlocal._JOB_LOCAL_SUBCOMMANDS`, AFTER every subcommand is built, so it is
  the last option on this help page. Adding it here would move it.

Provenance: verbatim move of the `main()`-inline `pjs` parser block from
`tools/vast/herdd.py`, plan §8 step 6, 2026-08-16.
"""

from __future__ import annotations

import argparse

from vastlib.cli import _args, _docs
from vastlib.jobs import submit as jobs_submit


def run(a: argparse.Namespace) -> None:
    """The §5 command-module contract. The body is `vastlib.jobs.submit.
    cmd_job_submit`, reached by MODULE ATTRIBUTE so a test that patches it
    still steers this dispatch (plan §8 porting mechanic (b))."""
    jobs_submit.cmd_job_submit(a)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Verbatim from `main()`: same help strings, same flag order, same dests."""
    pjs = sub.add_parser("submit", help="bundle a folder + queue it on a box",
                         epilog=_args._docs_epilog(_docs.DOC_JOBS),
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    pjs.add_argument("dir", help="job folder (must contain job-config.yaml or .json)")
    pjs.add_argument("--box", default=None, help="target instance id (queue prefix); "
                     "omit it and pass --local to queue onto this machine instead")
    pjs.add_argument("--name", default=None, help="override job-config name (slug)")
    pjs.add_argument("--timeout", type=int, default=None, help="override timeout_s")
    pjs.add_argument("--env", action="append", default=None, metavar="K=V",
                     help="override/add ONE job-config `env:` entry (repeatable). "
                          "The yaml `env:` block stays the source of truth — this "
                          "is for per-submit pins (TARGET=dc3, K=20, a sha) without "
                          "editing the bundle. Values are never echoed")
    pjs.add_argument("--artifact", action="append", default=None,
                     metavar="PREFIX=SLUG",
                     help="export one modelkit registry artifact as env "
                          "(repeatable): PREFIX_B2 plus its identity and serve "
                          "facts, so a `${PREFIX_B2}` asset prefix resolves from "
                          "the committed registry instead of a typed B2 path. "
                          "A raw --env PREFIX_B2=... still wins and bypasses it")
    pjs.add_argument("--dry-run", dest="dry_run", action="store_true",
                     help="validate + bundle + hash + dedupe-check; NO B2 mutations")
    pjs.add_argument("--strict-assets", dest="strict_assets", action="store_true",
                     help="turn the HEURISTIC (runset sentinel) B2-staleness warning "
                          "into a refusal too. A `tracks:`-declared mismatch always "
                          "refuses, with or without this flag")
    pjs.add_argument("--allow-stale-assets", dest="allow_stale_assets",
                     action="store_true",
                     help="submit anyway when a `tracks:`-declared staged object "
                          "differs from the repo file it mirrors — i.e. run the "
                          "bytes currently on B2 on purpose (still printed loudly)")
    pjs.add_argument("--no-asset-check", dest="no_asset_check", action="store_true",
                     help="skip the B2-staleness preflight entirely")
    pjs.add_argument("--allow-vram-drift", dest="allow_vram_drift",
                     action="store_true",
                     help="submit anyway when needs.gpu_ram_gb is below a peak "
                          "this shape has already been MEASURED to reach "
                          "(tools/vast/VRAM_SIZING.md). The honest use is a "
                          "shape that genuinely changed since the last anchor")
    pjs.add_argument("--allow-unscoped-writes", dest="allow_unscoped_writes",
                     action="store_true",
                     help="submit anyway when the bundle writes a B2 prefix no "
                          "shipped box key is scoped to. The honest use is a "
                          "SINGLE-KEY box (no minter configured), where [b2] is "
                          "bucket-wide read-write; otherwise the box 403s after "
                          "doing the work")
    pjs.add_argument("--require-box-eval-pin", dest="require_box_eval_pin",
                     action="store_true",
                     help="[needs.venv: eval] REFUSE unless the target box's "
                          "launch env carries EVAL_ENV_VER, instead of noting it. "
                          "For a caller that just rented the box and injected the "
                          "pin (launch_jobs_box.sh): the box is cold, so a missing "
                          "pin means the injection did not land and jobd is about "
                          "to fetch eval-env/LATEST. Not for a resubmit onto a "
                          "warm box, where the fetch has already happened")
    pjs.set_defaults(jobfunc=run)
    return pjs
