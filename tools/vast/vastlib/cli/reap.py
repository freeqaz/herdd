"""`herdd reap` — the scheduled auto-teardown: idle stopped boxes + live zombies.

An argparse-only shim: the body landed in `boxes.reap.cmd_reap` at plan step 3
(with the idle ledger, the keep-retention grammar, the zombie confirm map and
the durability advisory), so this module is the parser block and nothing else.

Why the help text is this long, and why it may not be trimmed
-------------------------------------------------------------
This command runs UNATTENDED — a systemd user timer fires it every 15 minutes
(`reaper_install.sh`) — and it destroys things. Its `--help` is the only place
the operator learns the kill switches (`HERDD_REAP=0`,
`HERDD_REAP_ZOMBIE=0`), the preview-vs-execute contract (bare `reap` previews
and exits 2 when it WOULD act; `-y` executes), and the graded live lane:
DESTROY only the provably-workless, PARK the measured-dead without that proof,
alarm otherwise, and never on a single observation — the verdict has to persist
`REAP_ZOMBIE_CONFIRM_S` (900 s) with no pull/heartbeat progress.

`--idle-hours` prints "default: env HERDD_REAP_IDLE_H, else 2" rather than a
number, exactly as the flat parser did: the real default is resolved inside the
body from `boxes.reap.REAP_IDLE_H_DEFAULT`, and `default=None` here is what lets
the body tell "operator passed 2" from "operator passed nothing".

Provenance: parser block moved from `tools/vast/herdd.py` `main()`, plan §8
step 6, 2026-08-16, behavior-preserving. Body: `boxes/reap.py`.
"""

from __future__ import annotations

import argparse

from vastlib.boxes import reap as reap_mod
from vastlib.cli import _args, _docs


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    prp = add_cmd(sub, "reap",
                  "destroy stopped boxes idle past a threshold (default 2h) "
                  "AND sweep live zombie boxes (dead boot / dead jobd) — "
                  "the auto-teardown; a `keep` label token opts a box out",
                  _docs.DOC_README,
                  "NOTE: scheduled by tools/vast/reaper_install.sh (systemd "
                  "user timer, 15-min cadence); HERDD_REAP=0 disables "
                  "globally; bare `reap` previews (exit 2 when it would "
                  "act), `-y` executes. Live zombie lane: DESTROY only the "
                  "provably-workless (jobs box, jobd never stamped), PARK "
                  "measured-dead boxes without that proof (2h idle fuse "
                  "then follows), alarm-only otherwise; action requires the "
                  "verdict to persist REAP_ZOMBIE_CONFIRM_S (900s) with no "
                  "pull/heartbeat progress. HERDD_REAP_ZOMBIE=0 disables "
                  "the live lane")
    prp.add_argument("--idle-hours", type=float, default=None, metavar="H",
                     help="idle threshold in hours (default: env "
                          "HERDD_REAP_IDLE_H, else 2)")
    prp.add_argument("-y", "--yes", action="store_true",
                     help="actually destroy (default: preview only)")
    prp.add_argument("--json", action="store_true",
                     help="print the classification as JSON (never destroys)")
    prp.set_defaults(func=reap_mod.cmd_reap)
    return prp
