"""`herdd workflow resume` — reattach a controller to an EXISTING wf_id.

Why this module exists
----------------------
`resume` is `run` minus the plan: the spec is read from B2 by id, not loaded
from a local `.py` path, so a workflow can be picked up on a different machine
or after the session that started it is gone. Two details are the reason it is
its own command rather than a flag on `run`:

* **`--takeover` defaults True.** A resume is, by definition, expected to
  replace whatever controller last held this workflow. The default lives in
  `workflows.ctl.resume_workflow`; the CLI still exposes the flag explicitly
  rather than hardcoding it, so the help page states the behavior.
* **The re-exec argv carries the wf_id, not a path.** `--detach` builds
  `<python> tools/vast/herdd.py workflow resume <wf_id> [--takeover]` —
  the Zone E entry-script path (plan §3), anchored in `_entry.py`.

What is deliberately NOT here
-----------------------------
* The credential rotation a resume performs before any paid work continues,
  and the controller claim's staleness arithmetic — `workflows.ctl.
  resume_workflow` owns both.
* A resume of a CANCELLED workflow. `cancel` is terminal and non-resumable by
  design; the refusal comes from the fold, not from a check here.

Provenance: moved from `tools/vast/herdd.py::cmd_workflow_resume` plus its
`main()`-inline parser block, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import json
import sys

from vastlib.boxes import lifecycle
from vastlib.cli.workflow import _entry
from vastlib.storage import b2
from vastlib.workflows import ctl as workflowctl


# moved-from: herdd.cmd_workflow_resume -> run
def run(a: argparse.Namespace) -> None:
    """Reattach a controller to an EXISTING wf_id (spec read from B2, not a
    local path) — `--takeover` defaults True in workflowctl.resume_workflow
    since a resume is, by definition, expected to replace whatever
    controller last held this workflow; the CLI still exposes the flag
    explicitly rather than hardcoding it."""
    b2._ensure_b2_remote()
    argv = [sys.executable, _entry.HERDD_SCRIPT, "workflow", "resume", a.wf_id]
    if a.takeover:
        argv.append("--takeover")
    actor = lifecycle._cli_actor()
    try:
        rc, result = workflowctl.resume_workflow(
            a.wf_id, actor=actor, detach=a.detach, takeover=a.takeover, argv=argv,
            detached_controller=getattr(a, "detached_controller", False),
            controller_deps=lambda wf, wf_id: workflowctl.build_live_controller_deps(
                wf, wf_id, actor=actor))
    except workflowctl.DetachUnavailable as e:
        print(str(e), file=sys.stderr)
        sys.exit(workflowctl.EXIT_INVALID)
    except workflowctl.WorkflowCtlError as e:
        # raised (not returned) here -> claim_controller refused the
        # (re)claim, i.e. a live controller is still heartbeating.
        if a.json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"error: {e}", file=sys.stderr)
        sys.exit(workflowctl.EXIT_CREDENTIAL)
    if a.json:
        print(json.dumps(result, indent=2))
    elif a.detach:
        print(f">> workflow {result.get('wf_id')} detached")
    else:
        v = result.get("view") or {}
        print(f">> workflow {result.get('wf_id')} rc={rc} status={v.get('status')}")
    sys.exit(rc)


def add_parser(wsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = wsub.add_parser("resume", help="reattach a controller to an existing wf_id "
                        "(spec read from B2, not a local path)")
    p.add_argument("wf_id")
    p.add_argument("--takeover", action="store_true", default=True,
                   help="claim the controller role even if a live one is recorded "
                        "(default: True — a resume is expected to replace whatever "
                        "controller last held this workflow)")
    p.add_argument("--detach", action="store_true",
                   help="hand off to `systemd-run --user` instead of blocking this shell")
    p.add_argument("--detached-controller", action="store_true",
                   help="INTERNAL (appended by --detach to the systemd re-exec): "
                        "this process IS the detached controller — a terminal "
                        "workflow exits 0 so Restart=on-failure doesn't "
                        "flap-restart the unit forever")
    p.add_argument("--json", action="store_true")
    p.set_defaults(wffunc=run)
