"""`herdd fleet uninstall` — disable + remove the systemd user unit.

Why this module exists
----------------------
The closing line is the reason this is not a one-liner: nothing is babysitting
the fleet after an uninstall, and a still-running box with no daemon is the
orphaned-billing shape. The command therefore ends by telling the operator to
park or re-watch whatever is still up.

Provenance: moved from `tools/vast/herdd.py::cmd_fleet_uninstall`, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import os
import subprocess

from vastlib.fleet import client


# moved-from: herdd.cmd_fleet_uninstall -> run
def run(a: argparse.Namespace) -> None:
    unit = os.path.expanduser(f"~/.config/systemd/user/{client.FLEET_UNIT_NAME}")
    subprocess.call(["systemctl", "--user", "disable", "--now",
                     client.FLEET_UNIT_NAME])
    if os.path.exists(unit):
        os.remove(unit)
        print(f"removed {unit}")
    subprocess.call(["systemctl", "--user", "daemon-reload"])
    print("fleetd uninstalled — NOTE: nothing is babysitting the fleet now; "
          "park or re-watch anything still running (herdd ls)")


def add_parser(fsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = fsub.add_parser("uninstall", help="disable + remove the systemd user unit")
    p.set_defaults(fleetfunc=run)
