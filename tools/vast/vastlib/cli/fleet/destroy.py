"""`herdd fleet destroy` — THE destroy path: explicit, journaled, daemon-executed.

Why this module exists
----------------------
fleetd never destroys on its own, so a destroy is always an operator decision
and always carries `--yes`. `--when` defers it until a condition holds
(re-checked every tick, must hold twice, executed at most once) and the
results-check guard holds a box whose job results never reached B2 — which is
why turning that guard off needs its own flag rather than being implied by
`--yes`.

Provenance: moved from `tools/vast/herdd.py::cmd_fleet_destroy`, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import sys

from vastlib.fleet import client


# moved-from: herdd.cmd_fleet_destroy -> run
def run(a: argparse.Namespace) -> None:
    if not a.yes:
        sys.exit("error: destroy needs --yes (fleetd never destroys on its own)")
    data = client._fleet_call_or_die("destroy", target=str(a.target), when=a.when,
                                     reason=a.reason, yes=True,
                                     results_check=not a.no_results_check,
                                     requester=client._fleet_requester())
    print(f"destroy queued for {data.get('target')} when={data.get('when')} "
          f"(re-checked every tick, executed at most once)")


def add_parser(fsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = fsub.add_parser("destroy", help="THE destroy path: explicit, journaled, "
                                        "executed by the daemon")
    p.add_argument("target")
    p.add_argument("--yes", action="store_true", required=False)
    p.add_argument("--when", default="now", choices=["now", "drained", "parked"],
                   help="defer until the condition holds (re-checked every tick, "
                        "must hold twice, executed at most once)")
    p.add_argument("--no-results-check", dest="no_results_check",
                   action="store_true",
                   help="skip the 'job results are published to B2' guard on a "
                        "deferred destroy (a box with unpublished results is "
                        "held + alarmed otherwise)")
    p.add_argument("--reason", default="")
    p.set_defaults(fleetfunc=run)
