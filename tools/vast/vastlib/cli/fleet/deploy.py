"""`herdd fleet deploy` — THE deploy path: move the release checkout, then PROVE it.

Why this module exists
----------------------
`fleet restart` re-executes the release checkout; it does not update it. Only
deploy moves that checkout to a known revision, re-points the unit, restarts,
and then VERIFIES the live rev — which is the step that makes this a separate
command rather than a flag on restart.

Like `install`, it shells out to `fleetd.py deploy` (see `fleet.deploy
.cmd_deploy`, which owns the audit, the dependency install and the
verification) instead of importing it: the deploy runs the RELEASE checkout's
code, not this working tree's.

Provenance: moved from `tools/vast/herdd.py::cmd_fleet_deploy`, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from vastlib.fleet import deploy as fleet_deploy


# moved-from: herdd.cmd_fleet_deploy -> run
def run(a: argparse.Namespace) -> None:
    """THE deploy path: move the release checkout to a known revision and prove
    the daemon came back on it. See fleetd.cmd_deploy — the verification at the
    end is why this exists rather than `restart`."""
    argv = [sys.executable, fleet_deploy._fleetd_script(), "deploy"]
    for flag, val in (("--checkout", a.checkout), ("--ref", a.ref),
                      ("--python", a.python)):
        if val:
            argv += [flag, val]
    if a.no_restart:
        argv.append("--no-restart")
    if a.force:
        argv.append("--force")
    sys.exit(subprocess.call(argv))


def add_parser(fsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = fsub.add_parser("deploy", help="update the RELEASE checkout to a known "
                                       "revision, re-point the unit, restart, "
                                       "and VERIFY the live rev")
    p.add_argument("--checkout", default=None,
                   help="release checkout (default $FLEETD_CHECKOUT or "
                        "~/.local/share/vast-fleetd/checkout); cloned if absent")
    p.add_argument("--ref", default=None,
                   help="revision (default: local/main — the LOCAL repo's main, "
                        "fetched into the release checkout; origin/main only "
                        "when that fetch fails)")
    p.add_argument("--python", default=None, help="interpreter to bake")
    p.add_argument("--no-restart", action="store_true",
                   help="write the unit but do not restart")
    p.add_argument("--force", action="store_true",
                   help="install even if the release audit fails")
    p.set_defaults(fleetfunc=run)
