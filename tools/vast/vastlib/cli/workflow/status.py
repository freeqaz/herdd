"""`herdd workflow status` — fold the event log, render the stage table.

Why this module exists
----------------------
`status` is the read path, and the one handler that talks to BOTH workflow
modules: `workflows.ctl.status_workflow` folds the event log into a view (and
derives the exit code from terminality), while `workflows.meta.
format_status_table` renders it. The split is deliberate — the fold is pure,
the renderer is pure, and the only impure part (`status_extras`, which reads
live figures for the table) is wrapped in a blanket `except` so a missing
credential degrades the table instead of failing the command.

What is deliberately NOT here
-----------------------------
* A second rendering. `--json` prints the folded view verbatim; the human
  table has exactly one implementation, in `workflows.meta`.
* Any retry or refresh loop. `status` is a single read by contract.

Provenance: moved from `tools/vast/herdd.py::cmd_workflow_status` plus its
`main()`-inline parser block, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from vastlib.storage import b2
from vastlib.workflows import ctl as workflowctl
from vastlib.workflows import meta as workflowmeta


# moved-from: herdd.cmd_workflow_status -> run
def run(a: argparse.Namespace) -> None:
    b2._ensure_b2_remote()
    rc, v = workflowctl.status_workflow(a.wf_id)
    if a.json:
        print(json.dumps(v, indent=2))
    else:
        # `extras` is annotated (the one addition to an otherwise verbatim
        # body): mypy fixes a local's type at its FIRST assignment, so without
        # it the `None` fallback is an error rather than the designed
        # degrade-to-a-thinner-table path.
        extras: dict[str, Any] | None
        try:
            extras = workflowctl.status_extras(a.wf_id)
        except Exception:
            extras = None
        print(workflowmeta.format_status_table(v, extras=extras))
    sys.exit(rc)


def add_parser(wsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = wsub.add_parser("status", help="fold a workflow's event log")
    p.add_argument("wf_id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(wffunc=run)
