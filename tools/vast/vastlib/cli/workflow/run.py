"""`herdd workflow run` — plan, claim the controller, reconcile until terminal.

Why this module exists
----------------------
`run` is the workflow verb that spends money, and the one with a second process
in it. Two things here are contracts rather than conveniences:

1. **The re-exec argv.** `--detach` hands the controller to `systemd-run
   --user` (`Restart=on-failure`), and the argv it re-execs is the exact
   FOREGROUND command this same invocation would have run —
   `<python> tools/vast/herdd.py workflow run <path> [--takeover]`. That
   entry-script path is Zone E frozen (plan §3); the literal is ported
   unchanged and its anchor lives in `_entry.py`, never rebuilt from this
   module's `__file__`.
2. **No hidden fallback.** If `systemd-run` is unavailable, `spawn_detached`
   raises `DetachUnavailable` whose message IS that foreground command; this
   handler prints it and exits `EXIT_INVALID`. It never degrades to a `nohup`
   the operator cannot find later.

The `--takeover` help interpolates `POLL_INTERVAL_S * HEARTBEAT_STALE_MULT` —
the staleness window a refused takeover reports. That multiplication runs at
PARSER-BUILD time, i.e. on every `herdd` invocation of every command, which
is why both constants are read from `workflows.ctl` (one ring down, already
imported here) instead of being re-literalled: a drift between the printed
number and the number `controller_is_stale` actually uses would be a lie in the
help text that nothing else would catch.

What is deliberately NOT here
-----------------------------
* The reconcile loop, the controller claim, the systemd unit construction —
  `workflows.ctl.run_workflow` / `run_controller` / `spawn_detached`.
* `--detached-controller`'s semantics. The flag only marks "this process IS
  the detached controller"; what that changes (a terminal workflow exits 0 so
  `Restart=on-failure` cannot flap-restart the unit forever) is decided one
  ring down.

Provenance: moved from `tools/vast/herdd.py::cmd_workflow_run` plus its
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


# moved-from: herdd.cmd_workflow_run -> run
def run(a: argparse.Namespace) -> None:
    """Foreground reconcile loop for a fresh workflow (plan + claim controller
    + reconcile-until-terminal), or (`--detach`) hand it to `systemd-run
    --user` and return immediately. `--detach` NEVER falls back to a hidden
    nohup: if systemd-run is unavailable, print the exact foreground command
    the operator must run instead and exit EXIT_INVALID."""
    b2._ensure_b2_remote()
    argv = [sys.executable, _entry.HERDD_SCRIPT, "workflow", "run", a.path]
    if a.takeover:
        argv.append("--takeover")
    actor = lifecycle._cli_actor()
    try:
        rc, result = workflowctl.run_workflow(
            a.path, wf_id=a.wf_id, actor=actor, detach=a.detach,
            takeover=a.takeover, argv=argv,
            detached_controller=getattr(a, "detached_controller", False),
            controller_deps=lambda wf, wf_id: workflowctl.build_live_controller_deps(
                wf, wf_id, actor=actor))
    except workflowctl.DetachUnavailable as e:
        print(str(e), file=sys.stderr)
        sys.exit(workflowctl.EXIT_INVALID)
    except workflowctl.WorkflowCtlError as e:
        # a raised (not returned) WorkflowCtlError here can only come from
        # run_controller's claim_controller — i.e. a refused 2nd live
        # controller (spec-validation errors are already folded into the
        # (rc, dict) return above).
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
        if rc != workflowctl.EXIT_OK and result.get("error"):
            # without this a config/spec failure is a bare `rc=1` (found live
            # 2026-07-15: the --detach crash-loop was undiagnosable from the
            # unit journal because the real error was swallowed here)
            print(f"error: {result['error']}", file=sys.stderr)
    sys.exit(rc)


def add_parser(wsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = wsub.add_parser("run", help="foreground reconcile loop for a fresh workflow "
                        "(plan + claim controller + reconcile-until-terminal)")
    p.add_argument("path", help="path to a WORKFLOW module (.py defining `WORKFLOW`)")
    p.add_argument("--detach", action="store_true",
                   help="hand off to `systemd-run --user` (Restart=on-failure) "
                        "instead of blocking this shell")
    p.add_argument("--takeover", action="store_true",
                   help="claim the controller role even if a live one is recorded "
                        "(refused unless its heartbeat is older than "
                        f"{workflowctl.POLL_INTERVAL_S * workflowctl.HEARTBEAT_STALE_MULT}s)")
    p.add_argument("--wf-id", default=None,
                   help="drive this already-planned workflow id instead of "
                        "minting one (--detach appends it to the child re-exec "
                        "so the printed id IS the id the detached controller "
                        "drives)")
    p.add_argument("--detached-controller", action="store_true",
                   help="INTERNAL (appended by --detach to the systemd re-exec): "
                        "this process IS the detached controller — a terminal "
                        "workflow exits 0 so Restart=on-failure doesn't "
                        "flap-restart the unit forever")
    p.add_argument("--json", action="store_true")
    p.set_defaults(wffunc=run)
