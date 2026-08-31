"""`herdd fleet unwatch` — stop managing a box, which does NOT stop the box.

Why this module exists
----------------------
The confirmation line is the whole point: unwatching leaves the box RUNNING and
billing, and the unwatched-box safety net re-applies after the grace window.
Saying so at the moment of the call is what stops "unwatch" from being read as
"amnesty".

Provenance: moved from `tools/vast/herdd.py::cmd_fleet_unwatch`, plan §8 step 6.
"""

from __future__ import annotations

import argparse

from vastlib.fleet import client


# moved-from: herdd.cmd_fleet_unwatch -> run
def run(a: argparse.Namespace) -> None:
    data = client._fleet_call_or_die("unwatch", target=str(a.target),
                                     requester=client._fleet_requester())
    print(f"unwatched {data.get('target')} — box left RUNNING; the unwatched-box "
          f"safety net re-applies after the grace window (unwatch is not amnesty)")


def add_parser(fsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = fsub.add_parser("unwatch", help="stop managing a box (it keeps running)")
    p.add_argument("target")
    p.set_defaults(fleetfunc=run)
