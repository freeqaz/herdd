"""vastlib.cli.workflow — the `herdd workflow` group: parser tree + `wffunc` dispatch.

Why this subpackage exists
--------------------------
`workflow` is one of the four nested dispatchers in the flat CLI (job, fleet,
workflow, notify — cli-surface.json hazard H2; plan §5 undercounts at three).
Its shape is the group pattern: the top-level parser sets `func=<this run>`,
each subparser sets `wffunc=<its module's run>`, and the group handler does
nothing but call `a.wffunc(a)`. That two-level indirection is preserved
verbatim — `a.func(a)` at the seam, `a.wffunc(a)` one level down — because the
dispatch shape is a frozen contract (plan §5) and because every one of these
handlers is reachable only through it.

This layer is I/O-thin BY DESIGN, and that is the property to protect when
editing it: every handler delegates the actual decision to a
`workflows.ctl.*_workflow()` wrapper (each returns `(exit_code, json_dict)`
using the frozen `EXIT_*` constants) and only handles argparse plumbing,
printing, and `sys.exit(rc)`. No pattern/box/launch logic lives here — a
plan/run that needs a box only ever surfaces `workflows.ctl`'s `need_box`
action as status text (M3 seam).

Layering note (the whole reason `workflows/` exists)
----------------------------------------------------
`herdd.py` and `workflowctl.py` import each other at module top level. The
package breaks that by direction: `vastlib.workflows` sits one ring BELOW
`cli`, consumes `core`/`boxes`/`jobs`/`launch`, and never imports upward. So
this subpackage may import `workflows.ctl`; nothing in `workflows` may import
back. import-linter enforces it.

One name is shadowed on purpose
-------------------------------
`run` is both a subcommand module (`workflow run`) and, per the `cli/`
convention, the group's own handler. The module is therefore imported as
`_run_mod` and the handler keeps the conventional name. After this package
finishes importing, the attribute `vastlib.cli.workflow.run` is the HANDLER;
the module is reachable as `sys.modules["vastlib.cli.workflow.run"]` or by
`from vastlib.cli.workflow import run as ...` inside a fresh import. No other
group has this collision (`job run-local` lands in `run_local.py`).

What is deliberately NOT here
-----------------------------
* The reconcile loop, spec validation, event folding, the box seams — all of
  that is `vastlib/workflows/ctl.py`. If a change to a handler body needs more
  than "print this differently", it belongs one ring down.
* `_docs_epilog` / `_add_cmd`. The factory is INJECTED (`_add_cmd_fn`), the
  shape `add_fleet_parser` / `add_notify_parser` already use and the one
  cli-surface.json names as the reference for the registry loop, so the group
  never reaches into the composition root.
* Typed `Args` dataclasses (plan §5's `_args.py` pattern). The handler bodies
  are ported VERBATIM in this wave — they read `a.path` / `a.json` off the
  bare `Namespace` exactly as before — because the acceptance bar for step 6
  is a byte-identical help tree and behavior-preserving handlers. Lifting the
  namespaces into dataclasses is a follow-on, not a port.

Provenance: verbatim move from `tools/vast/herdd.py` (plan §8 step 6,
2026-08-16) — `cmd_workflow` plus the `main()`-inline parser block
(:21276-21351, the only group built inline rather than by an `add_*_parser`
function). Each ported symbol carries its `# moved-from:` marker. Step 6 is
ADD-ONLY at this commit: `herdd.py` keeps its own copies and the CLI-surface
byte diff is what proves the two parser trees equal while both are alive.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from vastlib.cli import _docs
from vastlib.cli.workflow import cancel, logs, plan, pull, resume, status
from vastlib.cli.workflow import run as _run_mod

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vastlib.cli._args import AddCmd


# moved-from: herdd.cmd_workflow -> run
def run(a: argparse.Namespace) -> None:
    """Dispatch `herdd workflow <action>`."""
    a.wffunc(a)


def add_parser(sub: object, _add_cmd_fn: AddCmd) -> argparse.ArgumentParser:
    """`herdd workflow <sub>` — plan/run/track a multi-stage pipeline.

    New code (the flat file builds this group INLINE in `main()`, unlike
    `fleet`/`notify` which already have builder functions), but every string,
    default and ordering decision inside it is verbatim. `DOC_WORKFLOW` was a
    `main()` local; it now sits with the other doc pointers in `cli/_docs.py`
    (cli-surface.json MED-H7).
    """
    pw = _add_cmd_fn(sub, "workflow", "plan/run/track a multi-stage workflow (workflowctl)",
                     _docs.DOC_WORKFLOW, _docs.DOC_JOBS, _docs.DOC_SKILL)
    wsub = pw.add_subparsers(dest="wfcmd", required=True)
    pw.set_defaults(func=run)

    # Order is the help page. Each module owns its own flag block and its
    # `p.set_defaults(wffunc=...)`; this list is the only place the sequence
    # lives, and the CLI-surface diff reads it as the subcommand order.
    plan.add_parser(wsub)
    _run_mod.add_parser(wsub)
    status.add_parser(wsub)
    logs.add_parser(wsub)
    pull.add_parser(wsub)
    cancel.add_parser(wsub)
    resume.add_parser(wsub)
    return pw
