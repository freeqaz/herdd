"""`herdd fleet resume` — ask the daemon to resume a parked box.

Why this module exists
----------------------
The mirror of `park`, and a request for the same reason: the daemon executes
and journals it. Resuming through the daemon is also what keeps the watch and
its ceiling attached to the box that comes back — a hand-rolled `herdd start`
returns a box nobody is supervising.

Provenance: moved from `tools/vast/herdd.py::cmd_fleet_resume`, plan §8 step 6.
"""

from __future__ import annotations

import argparse

from vastlib.fleet import client


# moved-from: herdd.cmd_fleet_resume -> run
def run(a: argparse.Namespace) -> None:
    data = client._fleet_call_or_die("resume", target=str(a.target),
                                     requester=client._fleet_requester(),
                                     reason=a.reason)
    print(f"resume requested for {data.get('target')} — daemon executes + journals it")


def add_parser(fsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = fsub.add_parser("resume", help="ask the daemon to resume a parked box")
    p.add_argument("target")
    p.add_argument("--reason", default="")
    p.set_defaults(fleetfunc=run)
