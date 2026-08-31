"""`herdd workflow logs` — print a workflow's folded event log, one JSON object per line.

Why this module exists
----------------------
`status` answers "where is it"; `logs` answers "what happened", and it answers
it in the raw event schema rather than a rendering, because those events are
the B2-mediated contract every other reader (the dashboard, `workflowmeta`'s
fold, a post-mortem) also reads. The `--stage` filter is applied HERE, on the
already-folded list, rather than pushed into the read: the fold's ordering
(`ts`, then `nonce`) is what makes the output deterministic, and filtering
after it keeps a stage's lines in the same order they appear globally.

An empty result prints a `(no events ...)` line rather than nothing, and the
exit code still comes from the fold — "no events" is a legitimate state for a
just-planned workflow, not an error.

What is deliberately NOT here
-----------------------------
* Any parsing or summarising of the events. That is `workflows.meta`'s job and
  `status`'s output; a `logs` that reformatted would give two answers to the
  same question.
* Log FILES from the box. This is the workflow event log, not a job's stdout —
  `job logs` is the other lane.

Provenance: moved from `tools/vast/herdd.py::cmd_workflow_logs` plus its
`main()`-inline parser block, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import json
import sys

from vastlib.storage import b2
from vastlib.workflows import ctl as workflowctl


# moved-from: herdd.cmd_workflow_logs -> run
def run(a: argparse.Namespace) -> None:
    b2._ensure_b2_remote()
    rc, result = workflowctl.logs_workflow(a.wf_id)
    events = result.get("events", [])
    if a.stage:
        events = [e for e in events if e.get("stage") == a.stage]
    if not events:
        print(f"(no events for workflow {a.wf_id}"
              f"{' stage=' + a.stage if a.stage else ''})")
    for e in events:
        print(json.dumps(e))
    sys.exit(rc)


def add_parser(wsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = wsub.add_parser("logs", help="print a workflow's folded event log")
    p.add_argument("wf_id")
    p.add_argument("--stage", default=None, help="restrict to one stage's events")
    p.set_defaults(wffunc=run)
