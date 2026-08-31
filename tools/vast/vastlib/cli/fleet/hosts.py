"""`herdd fleet hosts` — the durable host-reputation store, read and edited.

Why this module exists
----------------------
`vastlib.market.hostrep` silently reorders and filters every automatic offer
pick. A policy that changes which box gets rented and cannot be inspected is an
unfalsifiable one — the operator's first question on any surprising rental is
"did reputation do that?", and there has to be a command that answers it.

So this prints the whole evidence base, not a verdict: per machine the decayed
score, the multiplier it becomes, how many strikes and — the column that matters
— across how many DISTINCT DAYS, because that is the term that turns a bad hour
into a blocked host.

Why it lives under `fleet` and not next to `hosts.py`
-----------------------------------------------------
`tools/vast/hosts.py` is the POSITIVE scorecard: measured pull throughput joined
out of the B2 run log, an operator's tool for choosing a warm host. This is the
NEGATIVE one, it is written by fleetd during supervision, and its file sits in
the fleetd state dir. Same subject, opposite sign, different writer.

What is deliberately NOT here
-----------------------------
* Any socket round trip. Unlike every other `fleet` subcommand this reads and
  writes the JSON file directly, on purpose: post-mortem — "why did it rent that
  host again" — is exactly when the daemon may be down, and the store is a plain
  file with an atomic writer precisely so a second process can edit it.
* Any scoring. The arithmetic is `hostrep`'s and is documented there.

Added 2026-08-20 with the durable-reputation layer (owner directive: "a cheap
host that doesn't work is not worth using for us").
"""

from __future__ import annotations

import argparse
import json
import time

from vastlib.core import config, fmt
from vastlib.market import hostrep


def _age(ts: object, now: float) -> str:
    try:
        return fmt._fmt_age(now - float(ts))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "-"


def run(a: argparse.Namespace) -> None:
    """List the store, or apply one operator edit to it."""
    now = time.time()
    path = hostrep.store_path()

    if getattr(a, "block", None) is not None:
        ok = hostrep.hold(a.block, days=float(a.days), reason=a.reason or "")
        print(f"{'held' if ok else 'FAILED to hold'} machine {a.block} for "
              f"{a.days:g}d — {a.reason or 'no reason given'}")
        return
    if getattr(a, "allow", None) is not None:
        ok = hostrep.release(a.allow)
        print(f"{'released' if ok else 'FAILED to release'} machine {a.allow} "
              f"(hold and block cleared; strike history kept, so its score "
              f"still decays from real evidence)")
        return
    if getattr(a, "forget", None) is not None:
        ok = hostrep.forget(a.forget)
        print(f"{'forgot' if ok else 'FAILED to forget'} machine {a.forget} "
              f"— all evidence dropped")
        return
    if getattr(a, "prune", False):
        n = hostrep.prune(older_than_d=float(a.older_than))
        print(f"pruned {n} machine record(s) with no evidence newer than "
              f"{a.older_than:g}d")
        return

    rows = hostrep.summary(now)
    if getattr(a, "json", False):
        print(json.dumps({"path": path, "enabled": hostrep.enabled(),
                          "block_score": config._boot_knob("HOSTREP_BLOCK_SCORE"),
                          "machines": rows}, indent=2, sort_keys=True))
        return
    if not hostrep.enabled():
        print(f"!! host reputation is DISABLED ({hostrep.DISABLE_ENV} is set) — "
              f"picks are pure cheapest-first; the rows below are inert")
    if not rows:
        print(f"no host reputation recorded yet ({path})")
        return
    print(f"{'MACHINE':<10}{'SCORE':<8}{'x PRICE':<9}{'STRIKES':<9}{'DAYS':<6}"
          f"{'LAST BAD':<10}{'LAST OK':<10}{'KINDS':<26}STATUS")
    for r in rows:
        status = r["blocked_reason"] and f"BLOCKED — {r['blocked_reason']}" or ""
        kinds = ",".join(r["kinds"])
        print(f"{r['machine_id']:<10}{r['score']:<8.2f}"
              f"{'x' + format(r['penalty'], '.2f'):<9}{r['strikes']:<9}"
              f"{r['distinct_days']:<6}{_age(r['last_strike_ts'], now):<10}"
              f"{_age(r['last_ok_ts'], now):<10}{kinds[:25]:<26}{status}")
    print(f"\nstore: {path}")
    print("SCORE decays (half-life "
          f"{config._boot_knob('HOSTREP_HALF_LIFE_D'):g}d) and is multiplied by "
          f"1 + {config._boot_knob('HOSTREP_RECURRENCE_BONUS'):g}x(DAYS-1) — "
          "failures spread across days weigh far more than a bad hour.")
    print("x PRICE is a RANKING multiplier only. It never changes a bid, a "
          "ceiling or a budget: a host at x1.35 loses to any clean host up to "
          "35% dearer, and still wins if it is the only affordable one.")


def add_parser(fsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = fsub.add_parser("hosts", help="durable host reputation: which machines "
                                      "we have evidence against, and why")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--block", metavar="MACHINE_ID", type=int,
                   help="operator hold: exclude this machine from every "
                        "automatic pick for --days regardless of its score")
    g.add_argument("--allow", metavar="MACHINE_ID", type=int,
                   help="lift a hold AND any earned block cooldown, keeping the "
                        "strike history — the 'I know why it failed and it is "
                        "fixed' verb")
    g.add_argument("--forget", metavar="MACHINE_ID", type=int,
                   help="drop the machine's record entirely — for when OUR bug "
                        "(a bad image, a broken onstart) charged strikes to "
                        "hosts that did nothing wrong")
    g.add_argument("--prune", action="store_true",
                   help="drop strikes older than --older-than and any machine "
                        "left with no evidence (housekeeping only: the score "
                        "already treats a 90d strike as worth 0.01)")
    p.add_argument("--days", type=float, default=7.0,
                   help="hold length for --block (default: 7)")
    p.add_argument("--reason", default=None,
                   help="why, recorded with a --block and printed in the listing")
    p.add_argument("--older-than", type=float, default=90.0, metavar="DAYS",
                   help="age cutoff for --prune (default: 90)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fleetfunc=run)
