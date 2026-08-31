"""vastlib.cli.job.run_local — `herdd job run-local`: the same bundle, this machine's GPUs.

Why this module exists
----------------------
`run-local` is the local-GPU lane's entry point (tools/vast/LOCAL_GPU_LANE.md):
same config, same jobd, same results/checkpoint semantics as a rented box, with
B2 swapped for a local-dir bucket. Two of its flags exist because the local
machine is not a rented one — `--gpus` is a real device allowance (not a fake
count) and `--asset NAME=DIR` symlinks an existing directory into the cache so
a multi-GB pull never happens. `--force` is the only override, and it never
kills anything: it starts anyway over busy cards.

Note the module name: the command is `run-local`, the module is `run_local`
(dashes fold to underscores — the `cli/` naming convention).

What is deliberately NOT here
-----------------------------
* The handler and the whole lane. `cmd_job_run_local`, `_run_local_preflight`,
  `_run_local_asset_warnings` and `_job_local_activate` landed in
  `vastlib.jobs.runlocal` at plan §8 step 5; this module is argparse-only and
  reaches the handler by module attribute.
* `--local`. This command IS the local lane; the flag would be redundant, and
  `run-local` is absent from `_JOB_LOCAL_SUBCOMMANDS` for that reason.
* The GPU authorization. `allow_local_gpu` is a `core.config` decision
  (owner ruling 2026-08-11, one switch in one place), checked inside the
  preflight before any directory is created — not an argparse gate.

Provenance: verbatim move of the `main()`-inline `pjrl` parser block from
`tools/vast/herdd.py`, plan §8 step 6, 2026-08-16.
"""

from __future__ import annotations

import argparse

from vastlib.cli import _args, _docs
from vastlib.jobs import runlocal


def run(a: argparse.Namespace) -> None:
    """§5 contract; body in `vastlib.jobs.runlocal`, by module attribute."""
    runlocal.cmd_job_run_local(a)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Verbatim from `main()`: same help strings, same flag order, same dests."""
    pjrl = sub.add_parser(
        "run-local", help="run a bundle on THIS machine's GPUs — same config, "
        "same jobd, same results/checkpoint semantics as a rented box",
        epilog=_args._docs_epilog(_docs.DOC_JOBS),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    pjrl.add_argument("dir", nargs="?", default=None,
                      help="job folder to submit first; omit to just drain "
                           "whatever is already queued locally (this is also how a "
                           "resume works — same JOB_ID, checkpoint pulled back)")
    pjrl.add_argument("--gpus", default=None, metavar="CSV",
                      help="device indices jobd may schedule onto (default: all). "
                           "Real indices and real VRAM — an allowance, not a fake")
    pjrl.add_argument("--asset", action="append", default=None, metavar="NAME=DIR",
                      help="point an `assets:` entry at an existing LOCAL directory "
                           "(symlinked into the cache — NO multi-GB copy, no B2 "
                           "pull). Repeatable; remembered for later runs")
    pjrl.add_argument("--root", default=None, metavar="DIR",
                      help="JOBD_ROOT (default <JOBLOCAL_HOME>/workspace). Pass "
                           "/workspace for byte-identical absolute `dest:` behavior")
    pjrl.add_argument("--watch", action="store_true",
                      help="stay up and keep draining the local queue instead of "
                           "exiting once it is empty")
    pjrl.add_argument("--cpu-slots", dest="cpu_slots", type=int, default=2,
                      help="max concurrent gpus=0 jobs (default 2)")
    pjrl.add_argument("--force", action="store_true",
                      help="start even though a card we would use is already busy "
                           "(we never kill anything either way)")
    pjrl.add_argument("--name", default=None, help="override job-config name (slug)")
    pjrl.add_argument("--timeout", type=int, default=None, help="override timeout_s")
    pjrl.add_argument("--env", action="append", default=None, metavar="K=V",
                      help="override/add ONE job-config `env:` entry (repeatable)")
    pjrl.add_argument("--artifact", action="append", default=None,
                      metavar="PREFIX=SLUG",
                      help="export one modelkit registry artifact as env "
                           "(repeatable) — see `job submit --artifact`")
    pjrl.add_argument("--dry-run", dest="dry_run", action="store_true",
                      help="preflight (GPUs, assets, dest warnings) and stop")
    pjrl.set_defaults(jobfunc=run)
    return pjrl
