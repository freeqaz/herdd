"""`herdd start` (alias `resume`) — bring a stopped box back on its own machine.

An argparse-only shim: the body landed in `boxes.lifecycle.cmd_start` at plan
step 3, so this module is the parser block and nothing else.

Three details the help text is carrying, all of them earned:

* **`aliases=("resume",)`** — with `stop (park)`, the only aliases in the
  surface (cli-surface.json hazard H11).
* **A start can be REFUSED.** The box's disk is still on that machine, but its
  GPUs may now be rented by someone else; vast answers a start with a busy
  error rather than queuing it. `--retry` polls that refusal; destroy +
  relaunch is the fallback, and it loses the disk.
* **`--no-reattach`** opts out of the auto-reattach that rotates a resumed jobs
  box's B2 key. The rotation is the default because a resumed box's old key may
  have expired while it sat parked, and a jobs box with a dead key looks
  exactly like a hung one.

Provenance: parser block moved from `tools/vast/herdd.py` `main()`, plan §8
step 6, 2026-08-16, behavior-preserving. Body: `boxes/lifecycle.py`.
"""

from __future__ import annotations

import argparse

from vastlib.boxes import lifecycle
from vastlib.cli import _args, _docs


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    psr = add_cmd(sub, "start",
                  "start/resume stopped instance(s) on their original machine "
                  "(disk + caches intact; ssh endpoint changes)",
                  _docs.DOC_README, _docs.DOC_EVALS,
                  "NOTE: a start is REFUSED while another renter holds the GPUs — "
                  "--retry keeps asking; destroy + relaunch is the fallback",
                  aliases=("resume",))
    psr.add_argument("id", type=int, nargs="+")
    psr.add_argument("--wait", type=int, default=0, metavar="SECS",
                     help="wait up to N s for running, then print the fresh ssh endpoint")
    psr.add_argument("--retry", type=int, default=0, metavar="SECS",
                     help="keep retrying a GPU-busy refusal for up to N s (poll ~20s)")
    psr.add_argument("--no-reattach", action="store_true",
                     help="skip the auto-reattach that rotates a resumed jobs "
                          "box's B2 key (requires --wait; jobs boxes only)")
    psr.set_defaults(func=lifecycle.cmd_start)
    return psr
