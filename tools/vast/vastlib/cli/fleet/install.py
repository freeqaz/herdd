"""`herdd fleet install` — generate + enable the systemd USER unit.

Why this module exists
----------------------
The unit is written AT INSTALL TIME on the operator's machine, so absolute
paths never enter git. That generation belongs to `fleetd.py install-unit`, and
this command SHELLS OUT to it (same interpreter, `deploy._fleetd_script()` for
the path) rather than importing it: there is one unit renderer, and it is the
daemon's.

Provenance: moved from `tools/vast/herdd.py::cmd_fleet_install`, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from vastlib.fleet import deploy as fleet_deploy


# moved-from: herdd.cmd_fleet_install -> run
def run(a: argparse.Namespace) -> None:
    """Generate + enable the systemd USER unit. The unit is written AT INSTALL
    TIME on the operator's machine (absolute paths never enter git)."""
    rc = subprocess.call([sys.executable, fleet_deploy._fleetd_script(), "install-unit"]
                         + (["--no-enable"] if a.no_enable else []))
    sys.exit(rc)


def add_parser(fsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = fsub.add_parser("install", help="generate + enable the systemd user unit")
    p.add_argument("--no-enable", action="store_true",
                   help="write the unit but do not enable/start it")
    p.set_defaults(fleetfunc=run)
