"""`herdd debug <action> <RUN_ID>` — drive the post-FAILED SSH debug-hold.

A thin wrapper over `tools/vast/debug_box.sh`, and thin ON PURPOSE: that script
is the single source of truth for the B2 STOP/EXTEND marker protocol, and the
box-side supervisor reads the same markers. Two implementations of a two-sided
protocol is how a hold gets extended on one side and torn down on the other.

Why the argument is a RUN_ID and not an instance id
---------------------------------------------------
The debug-hold exists precisely for a box that has already failed, and a failed
box may have been destroyed and relaunched under a new instance id — sometimes
several times. `RUN_ID` is the durable name that survives that, which is why the
markers are keyed on it and why `ssh` is an ACTION here (it prints how to find
the RUN_ID) rather than a shell: opening an actual shell is
`herdd ls` then `herdd ssh <id>`.

`sys.exit(r.returncode)` — not a print
--------------------------------------
The script's exit code is the answer (a missing marker exits non-zero), so it
propagates verbatim and this command can gate a shell `&&` chain.

What is deliberately NOT here
-----------------------------
* The marker logic. Reading, writing or interpreting a STATUS/STOP/EXTEND
  object is `debug_box.sh` + the box side; see `DEBUG_BOX.md`.
* Any B2 credential handling — the script mints its own.

Provenance: moved from `tools/vast/herdd.py` (`cmd_debug`, parser block in
`main()`), plan §8 step 6, 2026-08-16, behavior-preserving. The one mechanical
change: the script is resolved from `_TOOLS_VAST_DIR` below rather than from
this module's own `__file__`, which is three directories deeper.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from vastlib.cli import _args, _docs

# `tools/vast/` — three dirnames up from `tools/vast/vastlib/cli/`. The flat
# `cmd_debug` spelled this `os.path.dirname(os.path.abspath(__file__))` inside
# the function; keeping the depth in one named constant is the same treatment
# `cli/_runsets.py::_HERE` and `cli/workflow/_entry.py::_TOOLS_VAST_DIR` got,
# and for the same reason: a wrong depth here is a silent "script not found".
_TOOLS_VAST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# moved-from: herdd.cmd_debug
def run(a: argparse.Namespace) -> None:
    """Debug a crashed training box that is in its SSH debug-hold window.
    Thin wrapper over tools/vast/debug_box.sh (the single source of truth for the
    B2 STOP/EXTEND marker logic — keyed on RUN_ID, so no ephemeral instance id is
    needed). status = read STATUS marker; stop = tear down now (~20s);
    extend = add another FAIL_HOLD_MINUTES; ssh = hint how to find the RUN_ID.
    To open an actual shell use 'herdd ls' then 'herdd ssh <id>'."""
    script = os.path.join(_TOOLS_VAST_DIR, "debug_box.sh")
    r = subprocess.run([script, a.action, a.run_id])
    sys.exit(r.returncode)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    pdb = add_cmd(sub, "debug",
                  "debug a crashed training box (SSH debug-hold): status|stop|extend by RUN_ID",
                  _docs.DOC_DEBUG, _docs.DOC_TRAINING)
    pdb.add_argument("action", choices=["status", "stop", "extend", "ssh"])
    pdb.add_argument("run_id", help="RUN_ID the box was launched with")
    pdb.set_defaults(func=run)
    return pdb
