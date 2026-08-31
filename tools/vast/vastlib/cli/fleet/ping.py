"""`herdd fleet ping` — daemon liveness, version and tick age.

Why this module exists
----------------------
The first question about a daemon is whether it is running; the second, which
is the one that bites, is whether it is running THE CODE YOU ARE READING. The
version-skew line exists because `fleet restart` does not close that gap: the
daemon runs the RELEASE checkout, which a restart re-executes rather than
updates. Only `fleet deploy` moves it.

What is deliberately NOT here
-----------------------------
* The socket call. `client.fleet_request` owns the transport and its error
  taxonomy; this module only decides what a failure PRINTS (a one-line "DOWN"
  and exit 1 — `ping` is the one command that must not raise when the daemon
  is absent, since absence is the answer it exists to report).

Provenance: moved from `tools/vast/herdd.py::cmd_fleet_ping`, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import sys

from vastlib.fleet import client


# moved-from: herdd.cmd_fleet_ping -> run
def run(a: argparse.Namespace) -> None:
    ok, data, err = client.fleet_request("ping", _timeout=5)
    if not ok:
        print(f"fleetd: DOWN ({err})")
        sys.exit(1)
    print(f"fleetd: up  version={data.get('version')} rev={data.get('rev')} "
          f"pid={data.get('pid')} tick_age={data.get('tick_age_s')}s "
          f"watches={data.get('watches')} dry_run={data.get('dry_run')}")
    mine = client._git_rev_short()
    if mine and data.get("rev") and not client.rev_matches(mine, data["rev"]):
        # `fleet restart` alone does NOT close this: the daemon runs the RELEASE
        # checkout, which a restart re-executes rather than updates. `deploy`
        # moves the release checkout to origin/main and then restarts.
        print(f"!! version skew: this checkout is {mine}, the daemon runs "
              f"{data['rev']} — deploy main: herdd fleet deploy")


def add_parser(fsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = fsub.add_parser("ping", help="daemon liveness + version + tick age")
    p.set_defaults(fleetfunc=run)
