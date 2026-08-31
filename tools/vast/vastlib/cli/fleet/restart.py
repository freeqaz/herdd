"""`herdd fleet restart` — restart the daemon, REFUSING while a recovery is in flight.

Why this module exists
----------------------
2026-08-08 23:24:37Z: a redeploy landed two minutes after a human destroyed box
47214941, mid-chain. The restarted daemon reconciled the stale watch and ran its
OWN recovery — condemn, launch, retarget, destroy — duplicating work already in
progress: ~$0.9 of duplicated recovery and two actors on one job. The refusal is
that incident's fix, and `--force` is the escape hatch that names it.

The check is cheap by construction: `client.fleet_recoveries_in_flight()` reads
`state.json` DIRECTLY, no socket, because the moment an operator types `fleet
restart` is very often the moment the daemon is not answering.

Provenance: moved from `tools/vast/herdd.py::cmd_fleet_restart`, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from vastlib.fleet import client


# moved-from: herdd.cmd_fleet_restart -> run
def run(a: argparse.Namespace) -> None:
    """Restart the daemon — REFUSING while a recovery chain is in flight.

    2026-08-08 23:24:37Z: a redeploy landed two minutes after a human destroyed
    box 47214941, mid-chain. The restarted daemon reconciled the stale watch and
    ran its OWN recovery — condemn, launch, retarget, destroy — duplicating work
    already in progress: ~$0.9 of duplicated recovery and two actors on one job.

    The state file already knew. The ladder's per-eviction-cycle counters are
    durable precisely so a restart cannot forget them, so the same file read one
    moment earlier answers "is something mid-recovery". Cheap by construction: one
    file read, no API call, no daemon round trip."""
    inflight = ([] if getattr(a, "force", False)
                else client.fleet_recoveries_in_flight())
    if inflight:
        print("!! REFUSING to restart fleetd: a recovery chain is in flight, and "
              "a restart re-initialises the ladder that is driving it (2026-08-08 "
              "23:24Z: a redeploy mid-chain produced a DUPLICATE recovery — two "
              "actors on one job).")
        for r in inflight:
            print(f"   {str(r.get('iid') or '?'):<12}{str(r.get('kind')):<18}"
                  f"{r.get('detail')}"
                  + (f"   [watch {r['target']}]"
                     if str(r.get("target")) != str(r.get("iid")) else ""))
        print("\n   Wait for it to resolve (`herdd fleet status`, "
              "`herdd fleet log -f`), or restart anyway with "
              "`herdd fleet restart --force`.")
        sys.exit(2)
    rc = subprocess.call(["systemctl", "--user", "restart", client.FLEET_UNIT_NAME])
    if rc == 0:
        print(f"restarted {client.FLEET_UNIT_NAME} "
              f"(state.json + journal survive a restart)")
    sys.exit(rc)


def add_parser(fsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = fsub.add_parser("restart", help="systemctl --user restart the daemon "
                                        "(refuses mid-recovery; --force overrides)")
    p.add_argument("--force", action="store_true",
                   help="restart even while a recovery chain is in flight "
                        "(re-bid ladder, replacement, resume-in-place, queued "
                        "destroy). A restart mid-chain produced a DUPLICATE "
                        "recovery on 2026-08-08")
    p.set_defaults(fleetfunc=run)
