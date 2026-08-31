"""vastlib.cli.job.supervise — `herdd job supervise`: babysit a JOB box's bid.

Why this module exists
----------------------
This is the LEGACY INLINE driver of the jobs supervision lane, kept for
remote/CI/no-daemon use. The per-tick ladder itself lives in
`supervise.job_lane.job_supervise_tick`, which is the same code fleetd's `jobs`
profile runs (FLEETD_DESIGN §3) — so this module is a loop, a delegation check
and a print, and nothing else. Four properties of the 30-line body are the
whole reason it is not "just a while loop":

1. **`--budget` is required unless `--dry-run`.** A hard spend cap is the only
   thing standing between an unattended babysitter and an open-ended invoice.
2. **fleetd delegation happens BEFORE any work** (FLEETD_DESIGN §6): a
   `job supervise` PROCESS is exactly the babysitter-tied-to-an-agent-shell
   shape that leaves dangling billing instances, so when the daemon is up the
   watch is handed to it and this function returns.
3. **`handoff_can_complete = True`.** This driver CAN complete a handoff — it
   keeps ticking the same ladder until a verdict, so the tick after
   `drain_primary` always happens (defect #61 was fleetd's early return, not
   this loop's).
4. **The `finally` reaps a pre-cutover understudy on EVERY exit path** —
   self-park, operator-park, drained, budget, and the `sys.exit(3)`
   unrecoverable branch (SystemExit runs `finally` too). Phase-guarded and
   idempotent, so a promoted post-cutover understudy is left standing.

The mutually-exclusive ceiling group
------------------------------------
`--strict-ceiling | --handoff | --no-handoff` are three answers to ONE
question, and argparse's exclusivity is invisible to `--help` (manifest hazard
H6): porting them as plain flags renders identically and silently stops the
pair being rejected. The group object is carried verbatim. Its default is not
argparse's either — `set_defaults(handoff=jobs_handoff_enabled())` is the
SAFE-OFF switch for this lane (`vastconf.JOBS_HANDOFF_UNSAFE_KEY`, one switch
in one place, after the 2026-08-08 incident); an explicit `--handoff` on the
command line still wins, because this is a default, not a prohibition.

What is deliberately NOT here
-----------------------------
* The tick ladder, the handoff accrual, the replacement rungs, the salvage
  drivers — `vastlib.supervise.*`. If a change here needs more than "print
  this differently", it belongs one ring down.
* `_serve_self_park_soft`. cli-surface.json attributes it to this command by
  transitive closure, but its only caller is `supervise.job_lane`, which sits
  BELOW `cli` and cannot import upward. It stays the raising seam it is today;
  its real home is the serve lane, not this module.
* `--local`. Babysitting a spot bid is a box concept, so `supervise` is
  deliberately absent from `_JOB_LOCAL_SUBCOMMANDS`.

Provenance: verbatim move of `tools/vast/herdd.py::cmd_job_supervise` plus
its `main()`-inline `pjsv` parser block, plan §8 step 6, 2026-08-16. Callees
resolved to their vastlib homes by module attribute; the bid/replacement
constants interpolated into help text come from Zone S `bidpolicy`, the same
objects the flat parser reads.
"""

from __future__ import annotations

import argparse
import sys
import time

from vastlib.cli import _args, _docs
from vastlib.core import config
from vastlib.fleet import client
from vastlib.storage import b2
from vastlib.supervise import handoff, job_lane

import bidpolicy


# moved-from: herdd.cmd_job_supervise
def cmd_job_supervise(a: argparse.Namespace) -> None:
    """Supervise a JOB BOX the way `herdd supervise` babysits a training run —
    the jobs/spot unification seam. Reuses the SAME spot primitives (_bid_action
    defend/rescue, _put_bid_soft, _market_min_bid_soft, NOT_LIVE_DEBOUNCE):

      while the queue has non-terminal jobs:
        live     -> accrue spend against --budget; DEFEND the bid when the market
                    climbs (raise to BID_TARGET_MULT x floor, capped by --max-bid); re-attach
                    jobd after every not-live->live transition (an attach-started
                    daemon does not survive a resume by itself).
        stopped  -> classify THREE ways (SPOT_DESIGN §3.6, box-event stream
                    FIRST): jobd self-park on drain (parked_self -> SUCCESS exit),
                    market outbid (bid box, no self-park -> RESCUE), or operator
                    park (on-demand + intended=stopped -> clean exit). An
                    ambiguous bid-box stop defaults to rescue, never to giving up.
        not live -> debounce, then RESCUE: raise the bid on the stopped box so
                    vast auto-resumes it (jobd then RESUMES the interrupted jobs —
                    box-side v2). If the rescue stalls past --rescue-wait or the
                    box is gone: print the exact `job retarget` commands and exit.
      queue drained -> PARK the box (teardown default; --keep to leave running).

    What this does NOT do: relaunch a fresh box by itself (a job box has no
    launch spec) — that is the retarget path, one command per pending job.

    The per-tick ladder lives in `job_supervise_tick` so fleetd's `jobs` profile
    runs this exact code from the daemon (FLEETD_DESIGN §3); this function is the
    legacy inline driver kept for remote/CI/no-daemon use."""
    if not a.dry_run and a.budget is None:
        sys.exit("error: --budget USD is required (hard spend cap; --dry-run exempt)")
    # fleetd compat (FLEETD_DESIGN §6): hand the watch to the daemon when it is
    # up — a `job supervise` PROCESS is exactly the babysitter-tied-to-an-agent-
    # shell shape that leaves dangling billing instances.
    if client.fleet_delegate_job_supervise(a):
        return
    b2._ensure_b2_remote()
    # This driver CAN complete a handoff: it is a `while True` loop that keeps
    # ticking the same ladder until a verdict, so the tick after `drain_primary`
    # always happens and `complete` promotes the understudy into `jc` (defect
    # #61 was fleetd's early return, not this loop's).
    a.handoff_can_complete = True
    jc, hf = job_lane.job_supervise_init(a)
    _mbdesc = (f"${a.max_bid}" if a.max_bid is not None
               else (f"auto ({bidpolicy.BID_CEILING_ONDEMAND_FRAC:g}x on-demand, strict)"
                     if getattr(a, "strict_ceiling", False)
                     else "auto (just under on-demand; preferred "
                          f"{bidpolicy.BID_CEILING_ONDEMAND_FRAC:g}x on-demand)"))
    print(f">> supervising job box {jc['iid']} (budget={a.budget} max_bid={_mbdesc} "
          f"poll={job_lane.JOB_SUP_POLL_S}s dry_run={a.dry_run})")
    try:
        while True:
            verdict = job_lane.job_supervise_tick(jc, hf)
            if verdict == "unrecoverable":
                sys.exit(3)
            if verdict is not None:
                return
            time.sleep(job_lane.JOB_SUP_POLL_S if not a.dry_run
                       else min(job_lane.JOB_SUP_POLL_S, 5))
    finally:
        if jc["handoff_on"]:
            # F3: reap a mid-flight PRE-cutover understudy on EVERY exit path
            # (self-park / operator-park / queue-drained / queue-empty / budget /
            # the sys.exit(3) unrecoverable branch — SystemExit runs finally too),
            # never just the budget branch. Phase-guarded + idempotent: it only
            # reaps _HANDOFF_PRE_CUTOVER phases, so a post-cutover understudy (now
            # the canonical box promoted into jc) is left standing.
            handoff._job_handoff_reap_on_exit(jc, hf)


# The §5 command-module contract: `add_parser(sub)` + `run(a)`. `run` is an
# alias, not a second function — the parser binds the handler by identity at
# build time, so the name the dispatcher stores stays `cmd_job_supervise`.
run = cmd_job_supervise


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Verbatim from `main()`: same help strings, same flag order, same dests,
    the same mutually-exclusive ceiling group, and the same salvage block."""
    pjsv = sub.add_parser("supervise", help="babysit a job box like `supervise` "
                          "babysits a run: defend/rescue the spot bid, re-attach "
                          "jobd on resume, park when the queue drains",
                          epilog=_args._docs_epilog(_docs.DOC_JOBS, _docs.DOC_SUPERVISE),
                          formatter_class=argparse.RawDescriptionHelpFormatter)
    pjsv.add_argument("id", type=int, help="instance id of the job box")
    pjsv.add_argument("--budget", type=float, default=None, metavar="USD",
                      help="hard spend cap (required; box is parked at the cap)")
    pjsv.add_argument("--wall-budget", dest="wall_budget", type=float,
                      default=None, metavar="SECS",
                      help="expected remaining wall clock in seconds — feeds the "
                           "handoff amortization horizon (§2.3: a nearly-done job "
                           "should not migrate; default: a 24h horizon). NOT a "
                           "hard stop on this lane (unlike run supervise)")
    pjsv.add_argument("--max-bid", dest="max_bid", type=float, default=None,
                      help="hard defend/rescue/decay bid ceiling $/hr (default: auto, "
                           "just under on-demand [get-and-hold]; on-demand unreadable -> "
                           f"{bidpolicy.BID_MAX_MULT:g}x rolling-median floor fallback)")
    # Three mutually-exclusive answers to the over-ceiling question, DEFAULT =
    # handoff (HANDOFF_DESIGN §1/§9 T7; promoted to default 2026-07-15), exactly
    # like the run lane.
    pjsv_ceil = pjsv.add_mutually_exclusive_group()
    pjsv_ceil.add_argument("--strict-ceiling", dest="strict_ceiling", action="store_true",
                      help=f"hard-cap the default ceiling at "
                           f"{bidpolicy.BID_CEILING_ONDEMAND_FRAC:g}x "
                           "on-demand and let the box terminate above it. Ignored under --max-bid")
    pjsv_ceil.add_argument("--handoff", dest="handoff", action="store_true",
                      help=f"over the {bidpolicy.BID_CEILING_ONDEMAND_FRAC:g}x-on-demand "
                           "ceiling, keep "
                           "get-and-hold bidding AND migrate the job(s) to a cheaper box "
                           "(pre-warm a `launch --jobs` understudy -> retarget the tickets -> "
                           "drop the primary). OFF by default since the 2026-08-08 incident "
                           "(HANDOFF_DESIGN §11); this asks for it anyway on THIS run, and the "
                           "arm/fence preconditions still apply")
    pjsv_ceil.add_argument("--no-handoff", dest="handoff", action="store_false",
                      help="disable migration: get-and-hold the over-ceiling box, keep the "
                           "tickets on it (the default since 2026-08-08)")
    # SAFE-OFF on the jobs lane, one switch in one place (see
    # vastconf.JOBS_HANDOFF_UNSAFE_KEY). The run lane's default above is
    # deliberately untouched. An explicit `--handoff` on the command line still
    # wins — this is the DEFAULT, not a prohibition.
    pjsv.set_defaults(handoff=config.jobs_handoff_enabled())
    pjsv.add_argument("--rescue-wait", dest="rescue_wait", type=int, default=None,
                      metavar="SECS", help=f"auto-resume stall cap before giving up "
                      f"(default {job_lane.JOB_SUP_RESCUE_WAIT_S})")
    pjsv.add_argument("--max-replacements", dest="max_replacements", type=int,
                      default=None, metavar="N",
                      help=f"cap on AUTOMATIC replacement rentals after an "
                           f"eviction (default {bidpolicy.MAX_REPLACEMENTS}; 0 disables "
                           f"auto-replacement). Each replacement respects the "
                           f"--budget remainder and a price ceiling derived "
                           f"from the original launch")
    pjsv.add_argument("--replace-ceiling-mult", dest="replace_ceiling_mult",
                      type=float, default=None, metavar="X",
                      help=f"a replacement may cost at most this x the ORIGINAL "
                           f"launch price (default {bidpolicy.REPLACE_CEILING_MULT:g})")
    pjsv.add_argument("--replacement-verified", dest="replacement_verified",
                      choices=("0", "1"), default=None,
                      help="restrict AUTOMATIC replacement rentals to "
                           "vast-verified hosts (default 1; 0 widens the "
                           "candidate class to unverified hosts). Also "
                           "JOB_REPLACEMENT_VERIFIED in the env or herdd.yaml")
    pjsv.add_argument("--replacement-retention-hours",
                      dest="replacement_retention_hours", type=float,
                      default=None, metavar="H",
                      help=f"hold the EVICTED box this long after replacing it, "
                           f"so state that never reached B2 can be salvaged "
                           f"(default {bidpolicy.REPLACEMENT_RETENTION_H:g}h, "
                           f"~$0.27-$0.58 "
                           f"of allocated-disk storage; 0 destroys it at once). "
                           f"The box carries a self-expiring `keep:` label, so "
                           f"`herdd reap` reclaims it when the window closes")
    _args._add_salvage_args(pjsv)
    pjsv.add_argument("--keep", action="store_true",
                      help="do NOT park the box when the queue drains")
    pjsv.add_argument("--dry-run", dest="dry_run", action="store_true",
                      help="log-only: no bid PUTs, no attach, no park")
    pjsv.add_argument("--follow", action="store_true",
                      help="with fleetd: block until the watch finishes (mirrors "
                           "the legacy foreground behavior + exit code 3)")
    pjsv.add_argument("--no-fleet", dest="no_fleet", action="store_true",
                      help="run the LEGACY inline babysitter loop even when fleetd "
                           "is up (remote/CI use) — FLEETD_DESIGN §6")
    pjsv.set_defaults(jobfunc=cmd_job_supervise)
    return pjsv
