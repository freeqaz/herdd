"""`herdd fleet status` — the fleet table, its footnotes, and the alarm block.

Why this module exists
----------------------
Everything printed below a row exists because a shape of quiet overspend once
rendered identically to a healthy box: a PROVISIONAL auto-adopt cap nobody
chose, a ceiling that outlived the watch that armed it, an explicitly uncapped
watch, and a RETAINED evicted box still billing allocated disk. Each is
printed unconditionally — an undisclosed retained box is the orphaned-billing
shape fleetd exists to prevent, so it must not sit behind a flag. The one
footnote that graduated into the table is the replaced-box mapping (a watch
whose ladder replaced its box is filed under the original id): it is the WATCH
column now, because agents grep rows and lose prose printed below them.

What is deliberately NOT here
-----------------------------
* The alarm SEMANTICS. Whether an alarm is derived (true right now) or LATCHED
  (its evidence was consumed when it fired) is the daemon's classification;
  `_print_fleet_alarms` only renders the distinction, and falls back to the
  string list when talking to an older daemon that has no `alarm_records`.
* `_fmt_age`. It is shared with `fleet spend` and lives in `core.fmt`.

Provenance: moved from `tools/vast/herdd.py::cmd_fleet_status` and
`::_print_fleet_alarms`, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from vastlib.core import fmt
from vastlib.fleet import client

# FLEET_REVIEW_2026-08-20 item 4: measured 121 of a status listing's ceiling
# rows are `durable ceilings with no live watch` for boxes long gone (91 of 99
# on 2026-08-18) — the two real signals buried under tombstones. The review's
# stated predicate is "last=instance_gone past ~7 days", but `ceiling_rows()`
# (vastlib/fleet/rows.py) puts no timestamp on a ceiling row — verified live
# 2026-08-20, `fleet status --json` carries none — and this module may only
# touch CLI display code, not the daemon/rows payload that would need to grow
# one. So the age test collapses to the fallback the review anticipates for
# that case: every row whose `last_verdict` is this value is gone for good
# (the box left the listing), so ALL of them collapse by default regardless of
# how long ago, not just the ones past 7 days.
AGED_CEILING_VERDICT = "instance_gone"


# moved-from: herdd.cmd_fleet_status -> run
def run(a: argparse.Namespace) -> None:
    data = client._fleet_call_or_die("status")
    if getattr(a, "json", False):
        print(json.dumps(data, indent=2, sort_keys=True))
        return
    rows = data.get("rows") or []
    print(f"fleetd tick_age={data.get('tick_age_s')}s  dry_run={data.get('dry_run')}  "
          f"spend_total={fmt.dollars(data.get('spend_total_usd') or 0)}")
    print(f"{'BOX':<12}{'PROFILE':<9}{'STATE':<12}{'SPENT':<9}{'BUDGET':<9}"
          f"{'LEFT':<9}{'PAUSED':<9}{'WATCH':<12}LAST ACTION")
    for r in rows:
        paused = (f"{int(r.get('pause_left_s') or 0)}s" if r.get("paused") else "-")
        # SPENT is the CEILING's cumulative spend where one exists: the watch's
        # own counter reads $0.00 on a box that just inherited a ceiling with
        # $4.90 already drawn, which is the reading that made a lapse invisible.
        spent = r.get("ceiling_spend_usd")
        if spent is None:
            spent = r.get("spend_usd") or 0
        left = r.get("remaining_usd")
        # WATCH is the key the watch is FILED under when it differs from the
        # billing box — a ladder that replaced its box leaves the watch on the
        # original id (or a run: key), and either id addresses it. A column,
        # not a footnote: agents grep rows and lose prose printed below them.
        key, iid = str(r.get("target") or ""), str(r.get("iid") or "")
        watch = key if (key and iid and key != iid) else "-"
        print(f"{str(r.get('iid') or r.get('target')):<12}"
              f"{str(r.get('profile') or '-'):<9}"
              f"{str(r.get('state') or '-'):<12}"
              f"{fmt.dollars(spent):<9}"
              f"{(fmt.dollars(r['budget_usd']) if r.get('budget_usd') is not None else 'NONE'):<9}"
              f"{(fmt.dollars(left) if left is not None else '-'):<9}"
              f"{paused:<9}{watch:<12}{r.get('last_action') or '-'}")
    # A cap nobody chose, and a cap the operator's own watch left behind. Both
    # are things `fleet status` used to render identically to a real one.
    for r in rows:
        if r.get("ceiling_source") == "default":
            print(f"** {r.get('iid') or r.get('target')} is running under the "
                  f"PROVISIONAL auto-adopt cap "
                  f"{fmt.dollars(data.get('adopt_default_budget_usd') or 0)} — nobody "
                  f"chose that figure; `fleet watch <IID> --profile <P> "
                  f"--budget <USD>`")
        elif r.get("ceiling_source") == "inherited" and r.get("adopted"):
            print(f"** {r.get('iid') or r.get('target')}: an armed watch LAPSED "
                  f"here; the ceiling survived ({fmt.dollars(r.get('remaining_usd') or 0)} "
                  f"left) but the bid rescue/replacement ladder did NOT — "
                  f"re-register a real `fleet watch`")
        elif r.get("budget_usd") is None and r.get("profile") != "-":
            print(f"** {r.get('iid') or r.get('target')} has NO budget cap "
                  f"(explicit uncapped watch) — it bills without a ceiling")
    # Durable ceilings with no live watch: the headroom a re-arm inherits. The
    # confirmed-gone ones are noise in the default view (item 4) — collapse
    # them to a count unless --all was passed.
    orphan = [c for c in (data.get("ceilings") or []) if not c.get("live_boxes")]
    show_all = getattr(a, "all", False)
    aged = ([] if show_all else
            [c for c in orphan if c.get("last_verdict") == AGED_CEILING_VERDICT])
    aged_ids = {c.get("ceiling_id") for c in aged}
    visible = orphan if show_all else [
        c for c in orphan if c.get("ceiling_id") not in aged_ids]
    if visible:
        print("durable ceilings with no live watch (headroom a re-arm inherits):")
        for c in visible:
            print(f"   {c['ceiling_id']:<12}cap {fmt.dollars(c['cap_usd'])}  "
                  f"spent {fmt.dollars(c['spend_usd'])}  "
                  f"left {fmt.dollars(c['remaining_usd'])}  "
                  f"src={c.get('source')} epochs={c.get('epochs')} "
                  f"last={c.get('last_verdict')}")
    if aged:
        print(f"+ {len(aged)} aged ceilings for boxes gone "
              f"(last={AGED_CEILING_VERDICT}; --all to list)")
    # The replaced-box mapping ("** <iid> is the CURRENT box of watch <key>")
    # moved into the table as the WATCH column — same fact, greppable per row.
    # Boxes nobody chose to keep, still billing ALLOCATED disk: the evicted
    # primaries held for salvage. Printed unconditionally (never behind a flag)
    # — an undisclosed retained box is the orphaned-billing shape fleetd exists
    # to prevent.
    for r in data.get("retained") or []:
        left = fmt._fmt_age(r.get("left_s")) if r.get("left_s") is not None else "?"
        cost = (f"~{fmt.dollars(r.get('est_cost_usd') or 0)}"
                + (f"-{fmt.dollars(r['est_cost_hi_usd'])}"
                   if r.get("est_cost_hi_usd") not in (None, r.get("est_cost_usd"))
                   else ""))
        state = ("EXPIRED (awaiting reap)" if r.get("status") == "expired"
                 else f"{left} left")
        print(f"** RETAINED {r.get('iid')}: evicted ({r.get('eviction_class')}), "
              f"held for salvage — {state}, {cost} storage, replaced by "
              f"{r.get('replacement_iid')}"
              + ("" if r.get("keep_labeled")
                 else "  [NO keep label — reap may take it at the 2h idle mark]"))
    _print_fleet_alarms(data)


# moved-from: herdd._print_fleet_alarms
def _print_fleet_alarms(data: Any) -> None:  # noqa: ANN401 — daemon payload
    """Alarm block. Every line here is either a condition that is TRUE RIGHT NOW
    (derived — it disappears from the next `fleet status` the moment you fix it)
    or one explicitly LATCHED because its evidence was consumed when it fired.
    The distinction is printed, so the block never has to be taken on faith."""
    recs = data.get("alarm_records")
    if recs is None:                                  # older daemon: strings only
        for al in data.get("alarms") or []:
            print(f"!! {al}")
        return
    for r in recs:
        age = fmt._fmt_age(r.get("age_s"))
        if r.get("sticky"):
            print(f"!! [LATCHED {age} ago] {r.get('msg')}")
            print(f"     (ack when handled: herdd fleet ack {r.get('key')})")
        else:
            print(f"!! {r.get('msg')}  [{age}]")


def add_parser(fsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = fsub.add_parser("status", help="fleet table: profile, paused, budget, alarms")
    p.add_argument("--json", action="store_true")
    p.add_argument("--all", action="store_true",
                   help="show every durable-ceiling row, including ones for "
                        "boxes long gone (default: collapsed to a count)")
    p.set_defaults(fleetfunc=run)
