"""`herdd fleet ack` — clear a LATCHED alarm.

Why this module exists
----------------------
Derived alarms clear themselves the moment the condition stops holding; latched
ones cannot, because the evidence that fired them was consumed. Acking is
therefore an operator statement ("seen, handled"), which is why it carries a
`requester` — the daemon journals WHO cleared it.

What is deliberately NOT here
-----------------------------
* The alarm keys. They are the daemon's, printed by `fleet status --json`; this
  command validates only that ONE of key/--all was given, because acking
  nothing silently is the failure that makes an operator think an alarm was
  handled.

Provenance: moved from `tools/vast/herdd.py::cmd_fleet_ack`, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import sys

from vastlib.fleet import client


# moved-from: herdd.cmd_fleet_ack -> run
def run(a: argparse.Namespace) -> None:
    if not a.all and not a.key:
        sys.exit("error: fleet ack needs an alarm KEY (see `fleet status --json`) "
                 "or --all")
    data = client._fleet_call_or_die("ack", key=a.key, all=bool(a.all),
                                     requester=client._fleet_requester())
    cleared = data.get("cleared") or []
    print(f"acked {len(cleared)} latched alarm(s): {', '.join(cleared) or 'none'}")


def add_parser(fsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = fsub.add_parser("ack", help="clear a LATCHED alarm you have seen "
                                    "(derived alarms clear themselves)")
    p.add_argument("key", nargs="?", help="alarm key from `fleet status --json`")
    p.add_argument("--all", action="store_true", help="clear every latched alarm")
    p.set_defaults(fleetfunc=run)
