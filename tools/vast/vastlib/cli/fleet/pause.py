"""`herdd fleet pause` — suspend supervision for a BOUNDED window.

Why this module exists
----------------------
This is the interruption-drill primitive. The window is required (`--for`) and
auto-expires precisely so a crashed agent cannot leave a box unsupervised
forever; `--for 0` is the explicit clear. Both outcomes print differently
because "paused until X" and "supervision resumes now" are the two facts an
operator needs to be able to tell apart at a glance.

Provenance: moved from `tools/vast/herdd.py::cmd_fleet_pause`, plan §8 step 6.
"""

from __future__ import annotations

import argparse

from vastlib.fleet import client


# moved-from: herdd.cmd_fleet_pause -> run
def run(a: argparse.Namespace) -> None:
    data = client._fleet_call_or_die("pause", target=str(a.target), seconds=a.seconds,
                                     reason=a.reason,
                                     requester=client._fleet_requester())
    if a.seconds <= 0:
        print(f"pause cleared for {data.get('target')} — supervision resumes now")
    else:
        print(f"paused {data.get('target')} for {a.seconds}s "
              f"(auto-expires at {data.get('until_iso')}; a crashed agent cannot "
              f"leave it paused forever)")


def add_parser(fsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = fsub.add_parser("pause", help="suspend supervision for a bounded window "
                                      "(the interruption-drill primitive)")
    p.add_argument("target")
    p.add_argument("--for", dest="seconds", type=int, required=True, metavar="SECS",
                   help="auto-expiring pause window; 0 clears the pause")
    p.add_argument("--reason", default="")
    p.set_defaults(fleetfunc=run)
