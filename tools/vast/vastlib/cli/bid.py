"""`herdd bid <id> --price` — change a live bid instance's standing bid in place.

The primitive the autobid ladder is built on, exposed as a manual escape hatch.
Raising a stopped/outbid box's bid back above the market `min_bid` is what makes
vast auto-resume it, which is why the help says "running OR stopped".

Three things the body does that are policy, not plumbing:

* **The [0.001, 32] $/hr clamp** is a fat-finger fuse, not an API limit. A
  mistyped `--price 320` on an 8×H100 box is a real amount of money.
* **`--dry-run` prints the would-be PUT** and changes nothing — the same
  preview-then-execute shape `reap` and `guard --fix` use.
* **It emits NOTHING to the run log.** `supervise` accounts for its own bid
  moves; a manual bid on an arbitrary box has no run to attribute to, and
  inventing one would corrupt the cost ledger.

Provenance: moved from `tools/vast/herdd.py` (`cmd_bid`, parser block in
`main()`), plan §8 step 6, 2026-08-16, behavior-preserving.
"""

from __future__ import annotations

import argparse
import sys

from vastlib.boxes import lifecycle
from vastlib.cli import _args, _docs
from vastlib.core import fmt


# moved-from: herdd.cmd_bid
def run(a: argparse.Namespace) -> None:
    """Change the standing bid $/hr on an existing bid instance in place (the
    primitive `supervise` uses for defend/rescue, and a manual escape hatch).
    Emits NOTHING to the run log — usable on any box, run or not."""
    if not (0.001 <= a.price <= 32):
        sys.exit("error: bid price must be in [0.001, 32] $/hr")
    if a.dry_run:
        print(f"[dry-run] would PUT bid_price/{a.id}/ price={a.price}")
        return
    ok, err = lifecycle.set_bid(a.id, a.price)
    if not ok:
        sys.exit(f"error: could not set bid on {a.id}: {err}")
    print(f"bid set on {a.id}: {fmt.dollars(a.price)}/hr "
          f"(a stopped-because-outbid box auto-resumes once the bid regains priority)")


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    pbid = add_cmd(sub, "bid",
                   "change the standing bid $/hr on an existing bid instance "
                   "(running OR stopped) in place — raising a stopped/outbid box "
                   "above market min_bid auto-resumes it",
                   _docs.DOC_SUPERVISE, _docs.DOC_README)
    pbid.add_argument("id", type=int)
    pbid.add_argument("--price", type=float, required=True,
                      help="new bid $/hr (0.001-32)")
    pbid.add_argument("--dry-run", action="store_true",
                      help="print the would-be PUT, change nothing")
    pbid.set_defaults(func=run)
    return pbid
