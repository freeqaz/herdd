"""`herdd fleet report` — the fleet review as a command.

Why this module exists
----------------------
FLEET_REVIEW_2026-08-14 item 6. Both productive reviews of the 2026-08-10 ->
08-14 window were hand-written mining loops over the journal, and the same four
aggregates paid for themselves twice — a prior-bid echo 12 s from the window
edge (which is why `BID_SELF_FLOOR_LAG_S` is 3600) and 158 identical refusal
events announcing 2 facts (which is why both refusal sites now journal on
reason change). A loop that only runs when someone thinks to write it is not
instrumentation.

What is deliberately NOT here
-----------------------------
* The aggregates, and the flags that select them. `fleet_report.py` is a Zone-S
  -adjacent flat leaf with its own standalone CLI; it owns `add_args(p)` and
  `run(a)` so that `herdd fleet report` and `python3 fleet_report.py` take
  the SAME flags and produce the same tables. This module is the eleven lines
  that hang it off the `fleet` group.
* Any socket call. Like `fleet log`, it reads the append-only journal directly
  — post-mortem is exactly when the daemon is down.

Provenance: moved from `tools/vast/herdd.py::cmd_fleet_report` and its parser
block, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import sys

import fleet_report

from vastlib.cli import _args
from vastlib.cli._docs import DOC_AUTOBID, DOC_FLEET_REVIEW, DOC_FLEETD


# moved-from: herdd.cmd_fleet_report -> run
def run(a: argparse.Namespace) -> None:
    """The fleet review as a command (FLEET_REVIEW_2026-08-14 item 6).

    Both productive reviews of the 2026-08-10 -> 08-14 window were hand-written
    mining loops over the journal, and the same four aggregates paid for
    themselves twice — a prior-bid echo 12 s from the window edge (which is why
    `BID_SELF_FLOOR_LAG_S` is 3600), and 158 identical refusal events announcing
    2 facts (which is why both refusal sites now journal on reason change). A
    loop that only runs when someone thinks to write it is not instrumentation.

    Reads the append-only journal DIRECTLY — like `fleet log`, and for the same
    reason: post-mortem is exactly when the daemon is down. No API, no socket."""
    sys.exit(fleet_report.run(a))  # type: ignore[no-untyped-call]


def add_parser(fsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = fsub.add_parser(
        "report", help="journal aggregates: self-floor ages, refusal episodes, "
                       "eviction outcomes, watch lifecycle",
        epilog=_args._docs_epilog(DOC_FLEET_REVIEW, DOC_FLEETD, DOC_AUTOBID),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    fleet_report.add_args(p)  # type: ignore[no-untyped-call]
    p.set_defaults(fleetfunc=run)
