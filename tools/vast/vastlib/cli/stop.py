"""`herdd stop` (alias `park`) — end GPU billing, keep the disk.

An argparse-only shim: the body landed in `boxes.lifecycle.cmd_stop` at plan
step 3, so this module is the parser block and nothing else.

Two details that are load-bearing and easy to lose in a port:

* **`aliases=("park",)`.** `stop` and `start` are the only two aliased commands
  in the whole 69-node surface. argparse renders the pair as `stop (park)` in
  the top-level listing, so dropping the alias changes printed help AND 404s
  `herdd park`, which appears in runbooks and in agent muscle memory
  (cli-surface.json hazard H11).
* **Park is a pause WITHIN a session, not warm storage.** The help says the
  disk persists and keeps billing; the reaper destroys a stopped box idle past
  2h unless it carries a `keep` token. Both halves have to stay in the text.

Provenance: parser block moved from `tools/vast/herdd.py` `main()`, plan §8
step 6, 2026-08-16, behavior-preserving. Body: `boxes/lifecycle.py`.
"""

from __future__ import annotations

import argparse

from vastlib.boxes import lifecycle
from vastlib.cli import _args, _docs


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    pst = add_cmd(sub, "stop",
                  "stop/park instance(s): GPU billing ends, disk persists on the "
                  "machine (still bills storage) — resume later with `start`",
                  _docs.DOC_README, _docs.DOC_EVALS,
                  "NOTE: park (stop) beats destroy for eval iteration — image + "
                  "weights stay on disk, so `start` is back serving in minutes",
                  aliases=("park",))
    pst.add_argument("id", type=int, nargs="+")
    pst.add_argument("--wait", type=int, default=0, metavar="SECS",
                     help="wait up to N s for each instance to reach stopped")
    pst.set_defaults(func=lifecycle.cmd_stop)
    return pst
