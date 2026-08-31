"""vastlib.cli.job — the `herdd job` group: parser tree + `jobfunc` dispatch.

Why this subpackage exists
--------------------------
`job` is the largest of the four nested dispatchers in the flat CLI (job,
fleet, workflow, notify — cli-surface.json hazard H2; plan §5 undercounts at
three): 14 subcommands built by a 330-line inline block in `main()`. The group
shape is preserved verbatim — the top-level parser sets `func=cmd_job`, each
subparser sets `jobfunc=<the subcommand module's handler>`, and dispatch stays
`a.func(a)` at the seam with `a.jobfunc(a)` one level down.

`cmd_job` is NOT a pure dispatcher and that is load-bearing: it calls
`_job_local_activate()` when `a.local` is set, BEFORE handing off, which is
what points the whole process at a local-dir bucket instead of B2. Both it and
`_JOB_LOCAL_SUBCOMMANDS` were already ported (plan §8 step 5) into
`vastlib.jobs.runlocal`, so this module imports them rather than re-declaring
them — one `# moved-from:` marker per symbol, in one place.

The `--local` post-hoc loop
---------------------------
The seven box-agnostic subcommands get `--local` from a LOOP that runs after
they are all built, not from a flag in each block. That ordering is printed
output: `--local` is the LAST option on every one of those seven help pages,
and `run-local` / `supervise` are built after the loop, so the flat file's
insertion order is reproduced exactly here. The loop is also the reason the
flag cannot drift: a new subcommand cannot quietly acquire a half-wired local
mode, because the roster is `runlocal._JOB_LOCAL_SUBCOMMANDS` and nothing else.

What is deliberately NOT here
-----------------------------
* **Command bodies that already landed below.** Twelve of the fourteen handlers
  are in `vastlib.jobs.{submit,view,control,runlocal}` (plan §8 step 5); those
  command modules are argparse-only and delegate. Only `attach` and
  `supervise` carry their bodies, because nothing below `cli` owns them.
* **`_docs_epilog` / `_add_cmd`.** The `add_parser` factory is INJECTED — the
  shape `add_fleet_parser` / `add_notify_parser` already use, and the one
  cli-surface.json names as the reference for plan §5's registry loop — so the
  group never reaches into the composition root.
* **Typed `Args` dataclasses.** The handler bodies are ported VERBATIM in this
  wave and read off the bare `Namespace`, because the acceptance bar for step 6
  is a byte-identical help tree plus behavior-preserving handlers. Lifting the
  namespaces is a follow-on, not a port.

Provenance: verbatim move from `tools/vast/herdd.py` (plan §8 step 6,
2026-08-16) — the `main()`-inline `job` parser block (`pj`..`pjsv`), plus
`cmd_job_attach` and `cmd_job_supervise`. Step 6 is ADD-ONLY at this commit:
`herdd.py` keeps its own copies and the CLI-surface byte diff is what proves
the two parser trees equal while both are alive.
"""

from __future__ import annotations

import argparse
from typing import Callable

from vastlib.cli import _docs
from vastlib.cli.job import (
    attach,
    cancel,
    defs,
    dlq,
    flush,
    logs,
    ls,
    orphans,
    pull,
    requeue,
    retarget,
    run_local,
    status,
    submit,
    supervise,
    wait,
)
from vastlib.jobs import runlocal

# The `_add_cmd(sub, name, help_, *docs, aliases=())` factory, injected by the
# composition root so this group never imports it. Typed loosely on purpose:
# the precise call shape is the `Protocol` next to the factory in
# `cli/_args.py`, and a second declaration of it per group subpackage is the
# duplication the injection exists to avoid.
AddCmd = Callable[..., argparse.ArgumentParser]


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: AddCmd) -> argparse.ArgumentParser:
    """Build the whole `job` group onto `sub` and return its parser.

    Verbatim from the `main()`-inline block: same help strings, same subcommand
    ORDER (argparse prints them in insertion order, so the order is printed
    output), same `dest="jobcmd"`, same `required=True`, the same `func` /
    `jobfunc` defaults — and the same PLACEMENT of the `--local` loop, between
    `cancel` and `run-local`.
    """
    # --- job: B2-mediated job submission (JOBS_DESIGN.md) ---------------------
    pj = add_cmd(sub, "job", "submit/track batch jobs to a box via B2 (jobd)",
                 _docs.DOC_JOBS, _docs.DOC_EVALS, _docs.DOC_SKILL)
    jsub = pj.add_subparsers(dest="jobcmd", required=True)
    pj.set_defaults(func=runlocal.cmd_job)

    pjs = submit.add_parser(jsub)
    pjst = status.add_parser(jsub)
    pjw = wait.add_parser(jsub)
    pjl = logs.add_parser(jsub)
    pjp = pull.add_parser(jsub)
    pjls = ls.add_parser(jsub)
    defs.add_parser(jsub)
    orphans.add_parser(jsub)
    dlq.add_parser(jsub)
    attach.add_parser(jsub)
    retarget.add_parser(jsub)
    requeue.add_parser(jsub)
    pjc = cancel.add_parser(jsub)
    # Next to `cancel`: same marker mechanism, same poll, opposite intent.
    flush.add_parser(jsub)

    # --- LOCAL GPU LANE (tools/vast/LOCAL_GPU_LANE.md) -----------------------
    # `--local` on the box-agnostic subcommands points the SAME code at a
    # local-dir bucket instead of B2. Added in one loop so a new subcommand
    # cannot quietly acquire a half-wired local mode.
    _local_parsers = {"submit": pjs, "status": pjst, "wait": pjw, "logs": pjl,
                      "pull": pjp, "ls": pjls, "cancel": pjc}
    for _name in runlocal._JOB_LOCAL_SUBCOMMANDS:
        _local_parsers[_name].add_argument(
            "--local", action="store_true",
            help="operate on the LOCAL job bucket (this machine's GPUs) instead of "
                 "B2 — never touches the vast API or a credential. "
                 "See tools/vast/LOCAL_GPU_LANE.md")

    run_local.add_parser(jsub)
    supervise.add_parser(jsub)
    return pj
