"""`herdd fleet spend` — accrued spend, and the divergence from an upper bound.

Why this module exists
----------------------
The whole `--reconcile` half is a labelling problem, not an arithmetic one: it
compares fleetd's accrual against `dph_total x age`, which is an UPPER BOUND
(vast bills no GPU while `loading` and exposes no loading->running timestamp),
and every place it is printed says so. A divergence reads as "this much of the
box's billed life fleetd never watched" — never as an amount owed.

What is deliberately NOT here
-----------------------------
* The estimate. The daemon computes `reconcile`, including the `present` /
  `watched` flags whose ORDER matters in the note column ("gone" outranks
  "unwatched": a destroyed box has no live anchor to estimate against, so
  calling its accrual invisible would be wrong).
* `_fmt_age` — shared with `fleet status`, so it lives in `core.fmt`.

Provenance: moved from `tools/vast/herdd.py::cmd_fleet_spend`, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import json

from vastlib.core import fmt
from vastlib.fleet import client


# moved-from: herdd.cmd_fleet_spend -> run
def run(a: argparse.Namespace) -> None:
    """Accrued spend, and with --reconcile the divergence from an independent
    estimate (recalibration 2026-08-09, item E).

    The 2026-08-08 night's watch accounting saw ~$4.09 of a ~$5.66 invoice. The
    gap was structural: fleetd accrues from WATCH ADOPTION and a box bills from
    `start_date`, so every boot/loading window before a `fleet watch` lands is
    invisible — and an understudy nobody watched contributes its whole bill.

    There is no per-instance invoice in the vast API (the invoice lives on the
    account and cannot be attributed back to a box), so --reconcile compares
    against `dph_total x age` instead. That is an UPPER BOUND, and it is labelled
    one everywhere it is printed: vast does not bill GPU while `loading` and the
    API exposes no loading->running timestamp, so a slow image pull inflates it.
    Read a divergence as "this much of the box's billed life fleetd never
    watched" — never as an amount owed."""
    reconcile = bool(getattr(a, "reconcile", False))
    data = client._fleet_call_or_die("spend", since=a.since, reconcile=reconcile)
    if getattr(a, "json", False):
        print(json.dumps(data, indent=2, sort_keys=True))
        return
    print(f"fleet spend since {data.get('since') or 'daemon start'}: "
          f"{fmt.dollars(data.get('total_usd') or 0)}")
    for iid, usd in sorted((data.get("by_box") or {}).items()):
        print(f"  {iid:<12}{fmt.dollars(usd)}")
    if not reconcile:
        return
    rows = data.get("reconcile")
    if rows is None:
        print(f"\nreconcile UNAVAILABLE: {data.get('reconcile_error') or 'unknown'}")
        return
    if not rows:
        print("\nreconcile: no boxes to reconcile")
        return
    print(f"\n{'BOX':<12}{'ACCRUED':<10}{'UPPER BOUND':<13}{'DIVERGENCE':<18}"
          f"{'AGE':<8}{'PRE-WATCH':<10}NOTE")
    tot_a = tot_u = 0.0
    for r in rows:
        ub, div = r.get("upper_bound_usd"), r.get("divergence_usd")
        tot_a += r.get("accrued_usd") or 0.0
        tot_u += ub or 0.0
        # "gone" outranks "unwatched": a destroyed box has no live anchor to
        # estimate against, so calling its accrual invisible would be wrong.
        note = ("gone from the listing; accrual is final and unestimatable"
                if not r.get("present")
                else "UNWATCHED — its whole bill is invisible to fleetd"
                if not r.get("watched")
                else "")
        head = r.get("unwatched_head_s")
        pct = r.get("divergence_pct")
        div_s = "?" if div is None else (
            fmt.dollars(div) + (f" ({pct:g}%)" if pct is not None else ""))
        print(f"{r['iid']:<12}{fmt.dollars(r.get('accrued_usd') or 0):<10}"
              f"{(fmt.dollars(ub) if ub is not None else '?'):<13}"
              f"{div_s:<18}"
              f"{(fmt._fmt_age(r.get('age_s')) if r.get('age_s') is not None else '?'):<8}"
              f"{(fmt._fmt_age(head) if head is not None else '-'):<10}{note}")
    print(f"{'TOTAL':<12}{fmt.dollars(round(tot_a, 4)):<10}"
          f"{fmt.dollars(round(tot_u, 4)):<13}"
          f"{fmt.dollars(round(tot_u - tot_a, 4)):<18}"
          f"(estimable boxes only)")
    print(f"\n{data.get('reconcile_basis')}")


def add_parser(fsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = fsub.add_parser("spend", help="fleet-wide $ accounting from the journal")
    p.add_argument("--since", default=None, metavar="ISO8601")
    p.add_argument("--reconcile", action="store_true",
                   help="per box, compare fleetd's accrued spend against "
                        "dph_total x (now - start_date) and print the "
                        "divergence — the boot/loading windows and unwatched "
                        "boxes that made the 2026-08-08 accounting see $4.09 of "
                        "a $5.66 invoice. An UPPER BOUND, not the bill: vast "
                        "exposes no per-instance invoice")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fleetfunc=run)
