"""`herdd supervise` — babysit one RUN_ID: observe, relaunch, stop on a cap.

The 40-flag parser plus the eighteen-line driver that is all this command still
is. The policy landed at plan step 4 as `supervise/run_lane.py`
(`supervise_init` / `supervise_tick` / `supervise_finalize`), and the fleetd
`run` profile calls those same three — so there is exactly one copy of the
eviction/bid/handoff semantics and this module is the LEGACY INLINE LOOP that
drives it when no daemon answers.

The four things this driver still owns
--------------------------------------
1. **`--budget` is enforced POST-PARSE, not by argparse.** It is the hard spend
   cap and stays mandatory on any live run; only `--dry-run` (which never
   spends) is exempt. argparse cannot express "required unless another flag",
   and the check is written to mirror `cmd_job_supervise` exactly so the two
   supervision lanes exempt the same thing.
2. **fleetd delegation comes FIRST** (FLEETD_DESIGN §6). A `supervise` PROCESS
   is precisely the dangling-babysitter shape the daemon exists to abolish, so
   when the socket answers, the watch is registered and this returns. The loop
   below is the fallback for remote/CI/no-daemon use and for `--no-fleet`.
3. **`KeyboardInterrupt` is a TERMINAL ACTION, not a traceback.** Ctrl-C
   becomes `stop_fatal/operator_interrupt` and falls through to
   `supervise_finalize`, which is what parks the box and emits the final cost
   event. A port that let the exception escape would leave a live box billing
   with no `supervisor_exiting` in its log.
4. **`a.interval` sleeps between ticks** — the tick itself is pure of the
   clock, which is what lets fleetd run it on the daemon's cadence instead.

Why `--image` reads `cli/launch.default_image()`
------------------------------------------------
`supervise` and `launch` are the two parsers that bake `herdd.yaml`'s
`default_image` into a flag default. The flat prologue read it once and closed
over it; the registry loop has one two-argument call shape, so the read lives
next to the help text that quotes it (`cli/launch.default_image`, convention 3).
Here it is the RELAUNCH fallback: an eviction relaunch whose `launched` event
carries no image would otherwise resurrect the run on a different env, so the
resolution refuses rather than falling back to stock.

The mutually-exclusive group is load-bearing and INVISIBLE in `--help`
----------------------------------------------------------------------
`--strict-ceiling` / `--handoff` / `--no-handoff` are the three answers to
"we are over the 0.50x-on-demand preferred ceiling" (HANDOFF_DESIGN §1/§6/§8),
and they must stay ONE `add_mutually_exclusive_group()`. argparse renders a
group and three loose flags identically, so losing it is a silent behaviour
change — `--handoff --no-handoff` starts being accepted (cli-surface.json
hazard H6, pinned by `test_vastlib_cli_main.py` and the fixture diff).

What is deliberately NOT here
-----------------------------
* The tick. `supervise/run_lane.py`, with `_observe`, the cost accrual, the bid
  ladder and the self-floor guard.
* The delegation transport. `fleet/client.fleet_delegate_supervise`.
* `--follow`'s journal tail — also `fleet/client`, reached through the
  delegation return.

Provenance: moved from `tools/vast/herdd.py` (`cmd_supervise`, parser block in
`main()`), plan §8 step 6, 2026-08-16, behavior-preserving.
"""

from __future__ import annotations

import argparse
import sys
import time

from vastlib.cli import _args, _compose, _docs, launch, search
from vastlib.core import config
from vastlib.fleet import client as fleet_client
from vastlib.launch import spec
from vastlib.supervise import run_lane

import bidpolicy
import runmeta


# moved-from: herdd.cmd_supervise
def run(a: argparse.Namespace) -> None:
    # This command reaches `_do_launch` INDIRECTLY — the run lane's eviction
    # relaunch and handoff rungs go through `supervise.replacement._relaunch`,
    # which calls `launch._do_launch` — and it is a long-lived loop, so a seam
    # that raises would surface hours in, on the tick that was replacing an
    # evicted box. Bind first. See `cli/_compose.py` for why this cannot live in
    # `launch/launch.py`, and why it is a call-time bind.
    _compose.bind()
    run_id = runmeta.validate_run_id(a.run_id)
    # F8: --budget is the hard spend cap and stays MANDATORY on any live run; only a
    # --dry-run (never spends) is exempt. Enforced post-parse (argparse no longer
    # marks it required) so the exemption mirrors cmd_job_supervise exactly.
    if not getattr(a, "dry_run", False) and a.budget is None:
        sys.exit("error: --budget USD is required (hard spend cap; --dry-run exempt)")
    # fleetd compat (FLEETD_DESIGN §6): a `supervise` PROCESS is the dangling-
    # babysitter shape the daemon exists to abolish. When the socket answers,
    # register the watch and return; the inline loop below stays the fallback
    # for remote/CI/no-daemon use (and for --no-fleet).
    if fleet_client.fleet_delegate_supervise(a, run_id):
        return
    st, hf, handoff_on = run_lane.supervise_init(a)
    act = bidpolicy.Action("noop", "init")
    try:
        while True:
            done = run_lane.supervise_tick(st, a, hf, handoff_on)
            if done is not None:
                act = done
                break
            time.sleep(a.interval)
    except KeyboardInterrupt:
        act = bidpolicy.Action("stop_fatal", "operator_interrupt")
    run_lane.supervise_finalize(st, a, act, hf, handoff_on)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    psup = add_cmd(sub, "supervise",
                   "babysit one RUN_ID: observe, relaunch on eviction, stop on "
                   "terminal/operator-intent/budget",
                   _docs.DOC_SUPERVISE, _docs.DOC_SKILL_RUNS)
    psup.add_argument("run_id")
    search.add_search_filters(psup)                   # relaunch reuses the offer search
    psup.add_argument("--image", default=launch.default_image(),
                      help="fallback image if the launched event lacks one "
                           "(default from herdd.yaml's default_image, else "
                           f"{spec._EXPECTED_DEFAULT_IMAGE!r}). No stock fallback — "
                           f"a relaunch with no image REFUSES rather than "
                           f"resurrecting the run on a different env)")
    psup.add_argument("--disk", type=int,
                      default=config.default_disk_gb("supervise"),
                      help="container disk GB (herdd.yaml default_disk)")
    psup.add_argument("--onstart", default=None, help="file path or inline string")
    psup.add_argument("--env", action="append", default=None, help="KEY=VALUE (repeatable)")
    psup.add_argument("--runtype", default="ssh_direct",
                      choices=["ssh", "ssh_direct", "ssh_proxy", "jupyter"])
    psup.add_argument("--interval", type=int, default=45,
                      help="seconds between polls (default 45)")
    psup.add_argument("--budget", type=float, default=None,
                      help="HARD stop: cumulative spend cap in USD (required unless "
                           "--dry-run; no default) — cmd_supervise errors if it is "
                           "missing on a live run")
    psup.add_argument("--wall-budget", dest="wall_budget", type=float,
                      default=48 * 3600, metavar="SECS",
                      help="HARD stop: wall-clock budget in seconds (default 48h)")
    psup.add_argument("--max-relaunch", dest="max_relaunch", type=int, default=3,
                      help="HARD stop: max eviction relaunches (default 3)")
    psup.add_argument("--max-bid", dest="max_bid", type=float, default=None,
                      help="max resume/defend bid $/hr — the single cap on both bid "
                           "movements (default: auto, just under on-demand; "
                           f"--strict-ceiling caps at {bidpolicy.BID_CEILING_ONDEMAND_FRAC:g}x on-demand)")  # noqa: E501 — verbatim parser block (plan §7.4)
    # Over the 0.50x-on-demand preferred ceiling, three mutually-exclusive answers
    # (HANDOFF_DESIGN §1/§6/§8): DEFAULT = handoff (get-and-hold + migrate to a
    # cheaper box), --no-handoff (get-and-hold only, keep the expensive box),
    # --strict-ceiling (hard-terminate above the line). Promoted to default
    # 2026-07-15 after the run-lane live validation (handoff-canary-4).
    psup_ceil = psup.add_mutually_exclusive_group()
    psup_ceil.add_argument("--strict-ceiling", dest="strict_ceiling", action="store_true",
                      help=f"hard-cap the default bid ceiling at {bidpolicy.BID_CEILING_ONDEMAND_FRAC:g}x "  # noqa: E501 — verbatim parser block (plan §7.4)
                           "on-demand and let the box terminate above it (else get-and-hold: "
                           "pay up to just under on-demand). Ignored under --max-bid")
    psup_ceil.add_argument("--handoff", dest="handoff", action="store_true",
                      help="over the 0.50x-on-demand ceiling, get-and-hold AND migrate the run "
                           "to a cheaper box (launch understudy -> resume from checkpoint -> "
                           "drop the primary). DEFAULT as of 2026-07-15; flag kept for explicitness")  # noqa: E501 — verbatim parser block (plan §7.4)
    psup_ceil.add_argument("--no-handoff", dest="handoff", action="store_false",
                      help="disable migration: get-and-hold the over-ceiling box without spinning "  # noqa: E501 — verbatim parser block (plan §7.4)
                           "up a cheaper understudy (the escape hatch after the default flip)")
    psup.set_defaults(handoff=True)
    psup.add_argument("--defend-at", dest="defend_at", type=float, default=None,
                      help="proactive raise threshold, x last_bid (default "
                           f"{bidpolicy.DEFEND_AT:g} — raise once market_min_bid reaches "
                           "this fraction of our standing bid)")
    psup.add_argument("--rescue-wait", dest="rescue_wait", type=int, default=900,
                      metavar="SECS",
                      help="after a rescue bid raises an outbid box, wait up to this "
                           "many seconds for vast to auto-resume it before falling "
                           "back to destroy+relaunch (default 900)")
    psup.add_argument("--price", type=float, default=None,
                      help="original bid $/hr; seeds --max-bid if events lack it")
    psup.add_argument("--dry-run", action="store_true",
                      help="log-only: never destroy/PUT, still emit events")
    psup.add_argument("--follow", action="store_true",
                      help="with fleetd: block like the legacy loop did — tail "
                           "the daemon journal for this watch until it finishes "
                           "(mirrors the old foreground exit codes)")
    psup.add_argument("--no-fleet", dest="no_fleet", action="store_true",
                      help="run the LEGACY inline babysitter loop even when fleetd "
                           "is up (remote/CI use). Default: hand the watch to the "
                           "daemon and exit — FLEETD_DESIGN §6")
    psup.add_argument("--boot-health", dest="boot_health", action="store_true",
                      help="opt-in boot-throughput watchdog: while the box is "
                           "pre-running, sample the docker image-pull rate; on a "
                           f"sustained-slow host (< {int(config._BOOT_KNOB_DEFAULTS['BOOT_MIN_MBPS'])} "  # noqa: E501 — verbatim parser block (plan §7.4)
                           f"MB/s over {config._BOOT_KNOB_DEFAULTS['BOOT_MBPS_WINDOW_S']}s) emit "  # noqa: E501 — verbatim parser block (plan §7.4)
                           "boot_killed_slow, destroy, and relaunch on a DIFFERENT "
                           "machine (counts against --max-relaunch). Composes with "
                           "the box self-park deadline. Knobs: BOOT_MIN_MBPS / "
                           "BOOT_MBPS_WINDOW_S (env or herdd.yaml)")
    psup.add_argument("--no-boot-sla", dest="boot_sla", action="store_false",
                      default=True,
                      help="disable the DEFAULT come-online boot SLA (owner "
                           f"directive 2026-08-03; BOOT_SLA_S="
                           f"{int(config._BOOT_KNOB_DEFAULTS['BOOT_SLA_S'])}s): a box "
                           "not `running` by the deadline is destroyed and "
                           "relaunched on a different machine (counts against "
                           "--max-relaunch; deadline widens after "
                           f"{config._BOOT_KNOB_DEFAULTS['BOOT_SLA_MAX_KILLS']} kills)")
    psup.set_defaults(func=run)
    return psup
