"""`herdd guard` — the fleet zombie sweep, and the graded remedy behind `--fix`.

Classify every box's boot health, print the verdict table, and — only when asked
by name — apply the remedy. The three-step ladder is the whole design:
`guard` prints, `guard --fix` PREVIEWS the plan, `guard --fix -y` executes.

Why the remedy is GRADED (2026-08-03), and why that is not negotiable
---------------------------------------------------------------------
`guard --fix -y` was the last ungraded destroyer in the tool: it destroyed every
`ZOMBIE_*` verdict on the spot, while the reap lane already refused to destroy a
GPU-unbilled box without a no-progress confirmation. That is the command that
killed box 46682313 mid-pull — 90 seconds after its co-resident twin on the same
image cleared the identical verdict to OK. The loop was closed, too: fleetd's
alarm text for that verdict literally offered `fix: herdd guard --fix`, so the
control plane diagnosed a false positive and then handed over the irreversible
remedy.

So `--fix` now routes through `parked_lifecycle.zombie_action`, the same policy
the automatic reaper uses: DESTROY only a RUNNING, billing, provably-workless
box; PARK a GPU-unbilled loading stall (recoverable — `start` brings it back and
the idle reaper finishes it in 2 h); HOLD everything weaker. `--force` restores
the old behaviour and is kept deliberately as a human escape hatch that has to
be asked for by name.

`confirmed=True` is passed on purpose: a human running `guard` IS the
confirmation step — they are looking at the box right now — which is why guard
stays the fast lever for the expensive running-but-dead shapes. What it can no
longer do is destroy something the policy says to park.

Verdict membership is asked, not re-derived
-------------------------------------------
The flat code tested `h["verdict"] in _GUARD_ZOMBIE_VERDICTS`. Those two
frozensets and the short-tag dict collapsed into `boxes.health.GuardVerdict` at
plan step 3, so this module asks the lattice (`verdict_is_zombie`,
`verdict_is_advisory`) instead of holding its own copy of the membership. Same
answers, one definition — and an unknown or absent verdict is neither, exactly
as the set test behaved.

ADVISORY verdicts (`STALE_IMAGE`, `LOADING_SLOW`) are printed, counted, and
NEVER acted on: they are healthy boxes that are slow or running old code.

Exit codes are a cron contract: 2 when zombies are present or a preview
declined to act, 0 on a clean fleet or a fully successful fix, and a message on
stderr + non-zero when a sweep partially failed (those boxes are still billing).

What is deliberately NOT here
-----------------------------
* The classification itself — `boxes.health.gather_fleet_health` and the
  `GuardVerdict` lattice.
* The destroy/park primitives — `boxes.lifecycle`, including the credential
  revoke that rides with a destroy.
* The zombie POLICY — `parked_lifecycle.zombie_action`, reached through
  `boxes.reap._guard_fix_plan`, which imports it function-locally exactly as the
  flat body did (the sibling imports `herdd`, so a module-level import would
  be a cycle).
* `_guard_evidence_bits` / `_guard_fix_plan` themselves. DUPLICATE RULING
  (2026-08-16, wave 6a): both are homed in `boxes/reap.py` and this module
  IMPORTS them. `health.py` routes the fix plan there by name, and a graded
  destroy is policy, not argparse — so the copies that landed here in the same
  wave are gone and the rename table has one target per name.

Provenance: moved from `tools/vast/herdd.py` (`cmd_guard`, parser block in
`main()`), plan §8 step 6, 2026-08-16, behavior-preserving.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Mapping

from vastlib.boxes import health as boxes_health
from vastlib.boxes import lifecycle
from vastlib.boxes import reap as boxes_reap
from vastlib.cli import _args, _docs
from vastlib.core import fmt
from vastlib.fleet import client as fleet_client
from vastlib.jobs import view as jobs_view

import bidpolicy


# moved-from: herdd.cmd_guard
def run(a: argparse.Namespace) -> None:
    """Durable fleet zombie-sweep: classify every box's boot health and, with
    `--fix -y`, apply the GRADED remedy (`parked_lifecycle.zombie_action`) to
    the boxes in a ZOMBIE_* shape — DESTROY only where the policy licenses it
    (a RUNNING box, billing full GPU, provably workless), PARK where death is
    measured but the phase is GPU-unbilled or worklessness is unproven, and
    nothing at all where the evidence is weaker. `guard` alone prints the
    verdict table; `guard --fix` previews the plan; `guard --fix -y` executes.
    `--force` overrides the grading and destroys every zombie (the pre-
    2026-08-03 behavior, kept as a deliberate human escape hatch — it is the
    behavior that destroyed 46682313 mid-pull, so it must be asked for by
    name). Exits nonzero when zombies are present (cron/scripting), 0 on a
    clean fleet or a fully successful fix. Never touches OK/BOOTING boxes, and
    never touches ADVISORY verdicts (STALE_IMAGE, LOADING_SLOW)."""
    pal = fmt._Pal(fmt._color_on())
    now = time.time()
    ins = lifecycle._instances()
    live = [i.get("id") for i in ins
            if (i.get("actual_status") or "").lower() in bidpolicy.LIVE_STATES]
    try:
        jobs_by_box = jobs_view._fold_fleet_jobs(set(live))
    except Exception:
        jobs_by_box = {}
    health = boxes_health.gather_fleet_health(ins, jobs_by_box, now=now)
    rows = [health[str(i.get("id"))] for i in ins
            if str(i.get("id")) in health]
    zombies = [h for h in rows if boxes_health.verdict_is_zombie(h.get("verdict"))]
    # ADVISORY rows are reported and exit 0 — they are healthy boxes running old
    # code, and `--fix` must never reach them (velvet P1 is alarm-only).
    advisory = [h for h in rows if boxes_health.verdict_is_advisory(h.get("verdict"))]

    if getattr(a, "json", False):
        print(json.dumps({"health": rows,
                          "zombies": [h.get("iid") for h in zombies],
                          "advisory": [h.get("iid") for h in advisory]},
                         indent=2))
        sys.exit(2 if zombies else 0)

    print(f"== herdd guard — {len(rows)} box(es), {len(zombies)} zombie(s)"
          + (f", {len(advisory)} advisory" if advisory else "") + " ==")
    # zombies first (oldest first), then advisory, then the healthy rows.
    def _rank(h: Any) -> int:  # noqa: ANN401 — a BoxHealth row dict
        v = h.get("verdict")
        return (0 if boxes_health.verdict_is_zombie(v)
                else 1 if boxes_health.verdict_is_advisory(v) else 2)

    order = sorted(rows, key=lambda h: (_rank(h), -(h.get("age_s") or 0)))
    for h in order:
        z = boxes_health.verdict_is_zombie(h.get("verdict"))
        adv = boxes_health.verdict_is_advisory(h.get("verdict"))
        tag = h.get("verdict")
        mark = "!!" if z else ("~~" if adv else "  ")
        line = f"  {mark} {h.get('iid')}  {tag}: {h.get('reason')}"
        ev = boxes_reap._guard_evidence_bits(h.get("evidence") or {})
        if ev:
            line += f"  [{ev}]"
        print(pal.red(line) if z else (pal.yellow(line) if adv else line))

    if not zombies:
        print("no zombies — fleet boot-healthy."
              + (f"  ({len(advisory)} advisory: see ~~ above — `guard --fix` "
                 f"does NOT touch these)" if advisory else ""))
        return

    ids = [h.get("iid") for h in zombies]
    if not getattr(a, "fix", False):
        print(f"\n{len(ids)} zombie(s): {ids}")
        print("re-run `herdd guard --fix` to preview the graded remedy "
              "(destroy where licensed, park otherwise), then `--fix -y` to "
              "execute.")
        sys.exit(2)

    by_iid = {str(i.get("id")): i for i in ins}
    force = bool(getattr(a, "force", False))
    # Annotated to `boxes_reap._guard_fix_plan`'s own return type: the `--force`
    # arm below builds the literal 3-tuples the graded plan would have, and
    # inferring from it alone narrows the row to `dict` and the action to `str`.
    plan: list[tuple[Mapping[str, Any], Any, Any]]
    if force:
        plan = [(h, "destroy", "--force: grading overridden by the operator")
                for h in zombies]
    else:
        plan = boxes_reap._guard_fix_plan(zombies, by_iid)
    destroy_ids = [h.get("iid") for h, act, _ in plan if act == "destroy"]
    park_ids = [h.get("iid") for h, act, _ in plan if act == "park"]
    held = [(h, why) for h, act, why in plan if act not in ("destroy", "park")]

    print(f"\n-- graded plan: {len(destroy_ids)} destroy, {len(park_ids)} park,"
          f" {len(held)} held --")
    # `zrow`, not `h`: the verdict-table loop above already bound `h` to a
    # `dict` row, and a plan row is the wider `Mapping` the reap policy returns.
    for zrow, act, why in plan:
        mark = "!!" if act != "alarm" else "  "
        print((pal.red if act != "alarm" else str)(
            f"  {mark} {zrow.get('iid')}  {zrow.get('verdict')} -> "
            f"{act.upper() if act != 'alarm' else 'HELD'} ({why})"))
    if not (destroy_ids or park_ids):
        print("nothing licensed — alarm only. A GPU-unbilled `loading` box is "
              "never destroyed on a timer (2026-08-03): park it by hand with "
              "`herdd stop <id>` if you need the schedule back, or "
              "`herdd guard --fix -y --force` to override.")
        sys.exit(2)
    if not getattr(a, "yes", False):
        print(f"\n[dry-run] guard --fix WOULD DESTROY {destroy_ids} and PARK "
              f"{park_ids}. Re-run `herdd guard --fix -y` to execute.")
        sys.exit(2)

    # execute the graded plan. PARK, not destroy, is the licensed remedy for
    # every GPU-unbilled loading stall — it stops the schedule bleed, keeps the
    # disk, and lands the box in the reaper's 2h idle fuse, so every step to an
    # irreversible outcome stays recoverable (`herdd start`, or a keep label).
    label_ins = lifecycle._instances_soft()
    failed: list[Any] = []
    if destroy_ids:
        failed += lifecycle._destroy_and_revoke(destroy_ids, label_ins,
                                      "guard_zombie_destroy", noun="zombie ")
    for iid in park_ids:
        lifecycle._emit_stopping_intent(iid, "guard_zombie_park", instances=label_ins)
        try:
            fleet_client.fleet_operator_intent(iid, "stop", reason="guard_zombie_park")
        except Exception:
            pass
        ok, err = lifecycle.stop_box(iid)
        if ok:
            print(f"parked zombie {iid} (recoverable: `herdd start {iid}`; "
                  f"the idle reaper finishes it in 2h unless kept/resumed)")
        else:
            print(f"FAILED to park zombie {iid}: {err}", file=sys.stderr)
            failed.append(iid)
    if failed:
        sys.exit(f"error: could not sweep {failed} — still billing, retry!")
    print(f"swept {len(destroy_ids)} destroyed + {len(park_ids)} parked.")


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    pg = add_cmd(sub, "guard",
                 "fleet zombie-sweep: flag (and, with --fix -y, destroy) boxes "
                 "stuck in a dead-boot or dead-jobd shape",
                 _docs.DOC_README,
                 "the same classification `ls` screams — run it any session; "
                 "exits nonzero when zombies exist (cron-friendly)")
    pg.epilog = (
        "verdicts (boot phases: loading = image pull/standup, GPU UNBILLED;\n"
        "env-setup = running + onstart/jobd bootstrap, BILLED full GPU):\n"
        "  LOADING_SLOW           ADVISORY: past the loading deadline but the\n"
        "                         pull is STILL ADVANCING in status_msg, and\n"
        "                         inside GUARD_LOADING_HARD_S (3600s). A slow\n"
        "                         host, not a dead boot — --fix never acts\n"
        "  ZOMBIE_LOADING_STALL   loading/created past GUARD_LOADING_DEADLINE_S\n"
        "                         (default 1500s) with the pull inert, or past\n"
        "                         the 3600s hard bound — burns schedule, not\n"
        "                         GPU dollars, so its remedy is PARK not destroy\n"
        "  ZOMBIE_NO_JOBD         running jobs-box, JOBD_STATUS heartbeat stale\n"
        "                         past GUARD_JOBD_STALE_S (600s), or never\n"
        "                         stamped past GUARD_ENVSETUP_DEADLINE_S (900s\n"
        "                         from launch) — env-setup dead/overlong while\n"
        "                         billing full GPU\n"
        "  ZOMBIE_TICKET_UNCLAIMED submitted ticket unclaimed past\n"
        "                         GUARD_TICKET_DEADLINE_S (1500s)\n"
        "  ZOMBIE_PYHALF          the box's OWN beacon says pyhalf=broken —\n"
        "                         jobd.py cannot import its modules, so it can\n"
        "                         claim nothing and emit nothing. A BUNDLE\n"
        "                         fault: --fix HOLDS it (a relaunch reproduces\n"
        "                         it); the box self-parks at 300s and fleetd\n"
        "                         parks it at 600s\n"
        "  --fix applies the GRADED remedy (parked_lifecycle.zombie_action):\n"
        "  DESTROY only a RUNNING, billing, provably-workless box; PARK a\n"
        "  GPU-unbilled loading stall (recoverable — `start` brings it back,\n"
        "  the idle reaper finishes it in 2h); hold everything weaker.\n"
        "  Bare `--fix` previews, `--fix -y` executes, `--force` overrides the\n"
        "  grading and destroys every zombie. Knobs are env-overridable.\n"
        "  Before destroying an env-setup zombie by hand, check `reap` — its\n"
        "  confirm lane shows whether download/disk progress says the install\n"
        "  is actually still moving.\n\n"
        + (pg.epilog or ""))
    pg.add_argument("--fix", action="store_true",
                    help="apply the graded remedy to ZOMBIE_* boxes "
                         "(preview unless -y)")
    pg.add_argument("-y", "--yes", action="store_true",
                    help="with --fix: actually execute the plan")
    pg.add_argument("--force", action="store_true",
                    help="with --fix: destroy EVERY zombie, overriding the "
                         "grading — including GPU-unbilled boxes still pulling "
                         "(this is what destroyed 46682313 mid-pull)")
    pg.add_argument("--json", action="store_true")
    pg.set_defaults(func=run)
    return pg
