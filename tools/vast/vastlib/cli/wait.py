"""`herdd wait <id>` — block until an instance reaches a state.

An argparse-only shim: the body landed in `boxes.lifecycle.cmd_wait` at plan
step 3, so this module is the parser block and nothing else.

Two details the four bare flags do not advertise:

* **`--state running` prints the ssh endpoint on arrival.** That is the whole
  ergonomics of the command in a launch script — the endpoint changes on every
  resume, so `wait` is where you learn the current one rather than re-running
  `show`.
* **A timeout is a non-zero exit, not a shrug.** `_wait` exits with the last
  observed state on expiry (default 600 s), so the caller can tell "reached it"
  from "gave up" — the distinction the box-babysitting rules turn on.

What is deliberately NOT here
-----------------------------
* Boot judgement. Deciding a slow box is STARVED rather than merely slow is
  `herdd guard` and `fleetd`; `wait` polls a state and says what it saw.

Provenance: parser block moved from `tools/vast/herdd.py` `main()`, plan §8
step 6, 2026-08-16, behavior-preserving. Body: `boxes/lifecycle.py`.
"""

from __future__ import annotations

import argparse

from vastlib.boxes import lifecycle
from vastlib.cli import _args, _docs


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    pw = add_cmd(sub, "wait", "poll until an instance reaches a state", _docs.DOC_README)
    pw.add_argument("id", type=int)
    pw.add_argument("--state", default="running")
    pw.add_argument("--timeout", type=int, default=600)
    pw.set_defaults(func=lifecycle.cmd_wait)
    return pw
