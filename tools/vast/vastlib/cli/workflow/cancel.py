"""`herdd workflow cancel` — a cooperative, TERMINAL, non-resumable cancel.

Why this module exists
----------------------
Cancel is the one verb whose result cannot be undone: it writes a CANCEL
marker (the same discipline `job cancel` uses, one level up — at the workflow
rather than the single-job grain) and emits a terminal `workflow_cancelled`
event, after which `resume` will not reattach. The handler therefore prints the
"(terminal, non-resumable)" wording on success — that phrasing is the whole
warning, and it is why the command is not spelled `stop`.

`--reason` is recorded ON the event, so the post-mortem answer to "why did this
pipeline stop" lives in the same log as everything else that happened to it.

What is deliberately NOT here
-----------------------------
* Any box teardown. Cancelling a workflow is a control-plane decision;
  reclaiming boxes is the reconciler's and `fleetd`'s business.
* A `--force`. There is no non-cooperative cancel at this grain: the marker is
  read by the controller, which exits at its next tick.

Provenance: moved from `tools/vast/herdd.py::cmd_workflow_cancel` plus its
`main()`-inline parser block, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import sys

from vastlib.boxes import lifecycle
from vastlib.storage import b2
from vastlib.workflows import ctl as workflowctl


# moved-from: herdd.cmd_workflow_cancel -> run
def run(a: argparse.Namespace) -> None:
    """Cooperative, TERMINAL, non-resumable cancel — mirrors `job cancel`'s
    CANCEL-marker discipline but at the workflow (not single-job) level."""
    b2._ensure_b2_remote()
    rc, result = workflowctl.cancel_workflow(a.wf_id, actor=lifecycle._cli_actor(),
                                             reason=a.reason)
    if rc == workflowctl.EXIT_INVALID:
        print(f"error: {result.get('error')}", file=sys.stderr)
    else:
        print(f">> workflow {a.wf_id}: cancelled (terminal, non-resumable)")
    sys.exit(rc)


def add_parser(wsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = wsub.add_parser("cancel", help="cancel a workflow into a terminal, "
                        "NON-resumable `cancelled` state")
    p.add_argument("wf_id")
    p.add_argument("--reason", default=None, help="cancel reason (recorded on the event)")
    p.set_defaults(wffunc=run)
