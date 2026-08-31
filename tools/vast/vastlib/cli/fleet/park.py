"""`herdd fleet park` — ask the daemon to park a box now.

Why this module exists
----------------------
Parking is a REQUEST, not an action taken here: the daemon executes it and
journals it, so there is exactly one actor on the box and exactly one record of
why it stopped. The printed line says so, because a CLI that reported "parked"
before the daemon acted would be the second actor.

Provenance: moved from `tools/vast/herdd.py::cmd_fleet_park`, plan §8 step 6.
"""

from __future__ import annotations

import argparse

from vastlib.fleet import client


# moved-from: herdd.cmd_fleet_park -> run
def run(a: argparse.Namespace) -> None:
    data = client._fleet_call_or_die("park", target=str(a.target),
                                     requester=client._fleet_requester(),
                                     reason=a.reason)
    print(f"park requested for {data.get('target')} — daemon executes + journals it")


def add_parser(fsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = fsub.add_parser("park", help="ask the daemon to park a box now")
    p.add_argument("target")
    p.add_argument("--reason", default="")
    p.set_defaults(fleetfunc=run)
