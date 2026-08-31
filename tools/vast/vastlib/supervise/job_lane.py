"""vastlib.supervise.job_lane — the JOBS lane: one tick of `job supervise`.

Why this module exists
----------------------
`job_supervise_tick` is the money path for every rented job box. It is called
by TWO drivers — the legacy inline `cmd_job_supervise` loop and fleetd's
`jobs`/`serve` profiles (FLEETD_DESIGN §3/§4) — and that is the point: one copy
of the bid ladder, so a policy fix cannot land in one driver and miss the
other. The tick owns the whole per-poll decision tree: instance fetch, spend
accrual, stop classification (self-park / operator park / eviction), boot
watchdogs, the jobd queue read, the self-floor-guarded bid ladder, the handoff,
the rescue/re-bid/replacement rungs and the retention sweep. It returns `None`
to keep supervising or one of `JOB_SUP_VERDICTS`; those strings are the control
contract both drivers branch on (`cmd_job_supervise` exits 3 on
`unrecoverable`), so they are a wire format, not an implementation detail.

Three things about this file are load-bearing and are NOT stylistic:

* **The mirror stays a mirror.** The run lane (`run_lane.py`) has a twin of
  almost everything here and the six divergences between them are pinned
  deliberately (plan §5 NOTE, v1 §7, FLEET_REVIEW_2026-08-14 item 1): the
  strict `live and is_bid` tenancy gate, the `jobs_bid_*` event names, the
  pop-vs-assign latch clear, `floor_samples`, the `last_bid_put` key name, and
  the sticky-on-demand clear at box swap. Unifying any of them is a money-path
  change and its own owner-called work, not a side effect of moving files.
* **`serve_mode` is a second personality inside ONE function.** Eight branch
  sites in the tick plus the `job_supervise_init` flag. Splitting the tick by
  lane would fork a third copy of the ladder — the exact defect the single-copy
  design exists to prevent. It stays one function.
* **Late binding is the contract.** Every cross-module call below is written in
  module-attribute form (`journal._job_ladder_journal(...)`,
  `pricing._market_ondemand_soft(...)`, `risk._ckpt_watchdog_alarm(...)`), so a
  `monkeypatch.setattr` on the owning module still steers this tick. A
  `from … import fn` here would bind at import time, the patch would miss, and
  the test would go vacuously green (plan §10).

Tri-state discipline
--------------------
The five risk metrics this tick publishes onto `jc` — `work_at_risk_h`,
`running_unresumable`, `min_running_eta_s`, `ckpt_stale`, `remaining_wall_h`
(plus `timeout_ceiling_h`) — are read BY KEY by `bidpolicy` (Zone S, no import
edge). `None` means UNKNOWN and must stay `None`: substituting `0.0` turns a
migration REFUSAL into a migration (defect #67). Nothing here may grow an
`or 0.0`.

What is deliberately NOT here
-----------------------------
* **No loop and no sleep.** `time.sleep(JOB_SUP_POLL_S)` belongs to the caller
  (`cmd_job_supervise`, moving to `cli/` at step 6) and fleetd's scheduler. The
  tick is one poll and returns.
* **No bid state machine.** The per-tick bookkeeping (`reconcile_standing_bid`,
  `self_floor_guard`, `box_swap_reset`) is `ladder_core`, one copy shared with
  the run lane; the pure decisions (`_bid_action`, `classify_eviction`,
  `_handoff_trigger`, `resume_in_place`, `bid_decision`) are `bidpolicy`
  (Zone S). This module is the I/O and the journalling around them.
* **No typed `jc` yet.** `supervise/state.py` owns the jc/hf TYPES; this module
  owns the CONSTRUCTOR (`job_supervise_init`) — plan §5, and the overlap noted
  in both port manifests. The annotations here stay `MutableMapping[str, Any]`
  until state.py is wired in post-wave, deliberately: `jc` holds a live
  `argparse.Namespace` and a `set`, its key names and default types are a
  PERSISTENCE contract (fleetd `_replacement_state_restore` /
  `REPLACEMENT_STATE_KEYS`, state.json schema frozen by plan §4), and the
  set↔list plus stringified-machine-id asymmetries are fleetd's round trip to
  own, not this module's to smooth over.

Provenance: verbatim-with-types move of the jobs-lane cluster from
`tools/vast/herdd.py` (plan §8 step 4, ADD-ONLY — `herdd.py` keeps its live
copies until step 6), plus `classify_job_box_stop` / `_job_primary_evicted`,
which `jobs/risk.py` explicitly deferred here. Behavior-preserving; the only
edits are annotations and module-attribute qualification of cross-module calls.

RE-PORTED 2026-08-16 (plan §8 no-freeze drift duty) over peer commits
830579df / 73c44cb3 / 8b984898 / d5b0b773 — the notify-S2b slice. It changed
`_job_announce_eviction` and `job_supervise_tick` here and added the twelve
`_job_notify_*` symbols below; the deltas were applied verbatim, per symbol,
off an AST diff of `herdd.py` between the port base and that HEAD. Three
symbols the drift brief listed as stale measured BYTE-IDENTICAL and were left
alone (`_job_market_read` here, `_rebid_knob` / `_job_excluded_machines` in
`replacement.py`) — the new code had merely landed adjacent to them.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Callable, Mapping, MutableMapping

import disksize
import ladder_core

# `notify.py` is a flat sibling like `ladder_core` and is NOT on the Zone S
# roster (it never ships to a box), so isort files it here rather than in the
# trailing local-folder group with `bidpolicy` / `jobmeta`.
import notify

from vastlib.boxes import health, lifecycle
from vastlib.core import models
from vastlib.jobs import risk
from vastlib.jobs import view as jobs_view
from vastlib.market import offers as market_offers
from vastlib.market import pricing
from vastlib.supervise import handoff, journal, replacement, retention

import bidpolicy
import jobmeta

# --------------------------------------------------------------------------- #
# CROSS-RING / UNCLAIMED SEAMS — new code, no `moved-from:` marker (README §2
# rule 7). ONE name the verbatim bodies below call whose real definition no
# step-4 manifest claims, so porting it here would plant a second copy of an
# effectful driver a sibling module is about to own. Declaring it as a raising
# seam keeps the bodies verbatim, keeps the patch idiom at a stable attribute
# path, and makes the missing wiring loud instead of silent. The raise is the
# reminder; a test drives the tick by stubbing this module attribute.
#
# THREE names left this list and none of them left by a rebind — every one is
# now called in module-attribute form, so a `monkeypatch.setattr` on the OWNING
# module steers this tick:
#
#   * the three jobs-lane handoff drivers (`_job_handoff_reconcile`,
#     `_job_handoff_progress_warn`, `_job_handoff_tick`) -> `handoff.*`;
#   * `_sticky_on_demand` -> `pricing._sticky_on_demand` (step 6 leftovers). It
#     was never this module's to own: the RUN lane calls it too, and two copies
#     of the clamp is exactly the drift `market/pricing.py` exists to prevent;
#   * `_serve_self_park_soft` -> `replacement._serve_self_park_soft` (step 6
#     leftovers), with the rest of the serve cluster. Attribute call, NOT an
#     assignment rebind: `replacement` imports this module and this module
#     imports `replacement`, so a module-level alias here is an AttributeError
#     on whichever import order starts with `replacement`.
# --------------------------------------------------------------------------- #

_SEAM_HINT = ("not ported yet — rebind this module attribute when its cluster "
              "lands, or stub it in your test with monkeypatch.setattr")



def _box_lifecycle_soft(iid: object) -> dict[str, Any]:
    """The per-box lifecycle fold — `vastlib.jobs.view._box_lifecycle_soft`.

    Folds `jobs/nodes/<iid>/events/` to at least `{parked, drained_pending}` and
    never raises. The stop classification reads it FIRST, so a self-park is
    scored as success rather than as a loss — a raising stub here meant every
    `job supervise` tick on a non-serve jobs box raised, and the misread the
    fold prevents is exactly a parked box getting rescue-resumed. (Was a
    step-5-named seam; its body landed in jobs/view and the stub note went
    stale — closed by the 6f verifier's census, same one-line forwarder
    handoff.py uses for this name, same downward supervise->jobs edge.)"""
    return jobs_view._box_lifecycle_soft(iid)


# --------------------------------------------------------------------------- #
# Jobs-lane loop constants + the terminal verdict set
# --------------------------------------------------------------------------- #

# moved-from: herdd.JOB_SUP_POLL_S
JOB_SUP_POLL_S = 60          # job-box supervise tick (also the spend-accrual dt)
# moved-from: herdd.JOB_SUP_RESCUE_WAIT_S
JOB_SUP_RESCUE_WAIT_S = 900  # auto-resume stall cap after a rescue bid (SPOT_DESIGN)

# Verdicts a jobs-lane tick can end on (None = keep supervising).
# moved-from: herdd.JOB_SUP_VERDICTS
JOB_SUP_VERDICTS = ("self_parked", "operator_park", "budget", "drained",
                    "queue_empty", "unrecoverable",
                    # serve lane, boot-SLA breach: the box was destroyed and
                    # launch_serve.sh re-fired on a different host — terminal
                    # for THIS watch (the relaunch registered its successor).
                    "sla_relaunched",
                    # serve lane: the box VERIFIED an identity that is not the
                    # one its watch was registered for. Terminal for the ladder
                    # (parked, never rescued) but NOT for the watch under
                    # fleetd — the alarm has to keep burning where an operator
                    # can still act on it.
                    "identity_mismatch")


# --------------------------------------------------------------------------- #
# Pure classifiers (deferred here by jobs/risk.json: a contiguous cut would
# have swallowed them, and `classify_job_box_stop` is a documented public seam
# for parked_lifecycle.py — the name must survive the move)
# --------------------------------------------------------------------------- #

# moved-from: herdd.classify_job_box_stop
def classify_job_box_stop(*, present: object, live: object, is_bid: object,
                          intended_status: str | None,
                          box_parked: object, box_drained: object,
                          stop_intent: bool = False,
                          claimed_work: bool = False) -> str | None:
    """PURE three-way classification of a job box that is stopped / not running
    (SPOT_DESIGN §3.5; v2.1 added the third way a box reaches `stopped`).

      "self_parked"  -> jobd self-parked on queue drain (a `parked_self`/`drained`
                        box-event exists). EXPECTED success — supervise exits 0.
      "operator_park"-> a human/tool ran `herdd stop`. Clean exit.
      None           -> no self-park event explains the stop: FALL THROUGH to the
                        eviction/rescue path (a bid box stopped without a self-park
                        event is treated as OUTBID and rescued within budget, never
                        abandoned).

    The box-event stream is consulted FIRST: a self-park is a success signal, not
    a loss. Defaulting an ambiguous stop to "operator park" is the UNSAFE
    direction (it abandons a rescuable outbid box, the exact 2026-07-11
    bakeoff-05 regression), so anything unexplained falls through to rescue.

    **`intended_status` is not evidence of intent** (incident 2026-08-08, task
    #74). Vast reported `intended_status: stopped` for box 47214941 while it was
    being competitively displaced — the machine's min_bid rose $2.55 -> $2.81 and
    the offer went `avail: no` — so the field says "this box is meant to be
    stopped *now*", not "somebody asked for it to be stopped". The only reliable
    record of an ASK is the intent journal: `herdd stop` / `destroy` / `guard`
    / `reap` all call `fleet_operator_intent` BEFORE the vast PUT, fleetd stores
    it in `state["intents"]`, and a watch with a live intent goes dormant without
    ever reaching this ladder. So under the daemon, arriving here at all IS the
    negative evidence — `stop_intent` carries it explicitly for the inline CLI
    and for tests.

      * `stop_intent`  — a journaled operator/fleetd stop or destroy exists for
                         this box. Outranks everything below; a genuine `fleet
                         park` / `stop` / reap stays an operator park.
      * `claimed_work` — this watch holds at least one NON-TERMINAL ticket. A box
                         that stopped out from under live work, with nobody
                         having asked for it, is an eviction whatever its rental
                         type: an ON-DEMAND box cannot be outbid, but it can be
                         reclaimed by the host or torn down out-of-band, and
                         "give up silently" is the wrong answer to that too.

    The pre-2026-08-09 rule (an on-demand box with intended_status==stopped is
    unconditionally an operator park) survives for an IDLE on-demand box, which
    is the case it was actually written for."""
    if box_parked or box_drained:
        return "self_parked"
    intended_stopped = bool(present) and (intended_status or "").lower() == "stopped"
    if intended_stopped and stop_intent:
        return "operator_park"
    if intended_stopped and not is_bid and not claimed_work:
        return "operator_park"
    return None


# moved-from: herdd._job_primary_evicted
def _job_primary_evicted(present: object, live: object,
                         not_live_streak: int) -> bool:
    """PURE. Debounced primary-eviction verdict for the jobs-lane handoff tick,
    mirroring the run lane's NOT_LIVE_DEBOUNCE'd poll verdict (2026-07-18 review
    S2: the raw per-tick `(present and not live) or (not present)` let a single
    not-live blip — a resume flap, a transient instances-API miss that drops the
    whole listing — reap a warming understudy into a 1800s cooldown, or worse
    fast-CUTOVER off a still-live primary, leaking it un-fenced). `not_live_streak`
    is the loop's counter BEFORE this tick's increment, so `streak+1` matches the
    rescue trigger's timing exactly: handoff and rescue see the eviction on the
    same tick."""
    gone = (present and not live) or (not present)
    return bool(gone and (not_live_streak + 1) >= bidpolicy.NOT_LIVE_DEBOUNCE)


# --------------------------------------------------------------------------- #
# Per-tick context construction
# --------------------------------------------------------------------------- #

# moved-from: herdd.job_supervise_init
def job_supervise_init(a: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the jobs-lane per-tick context (`jc`, the analogue of the run lane's
    `st`) + its handoff state. Shared by the legacy inline loop and the fleetd
    `jobs` profile (FLEETD_DESIGN §3/§4) so both run the SAME ladder from one
    copy of the code. `iid` may be REASSIGNED to the understudy after a completed
    handoff, so whoever ticks keeps supervising the survivor."""
    now = time.time()
    jc: dict[str, Any] = {
          "a": a, "iid": str(a.id), "dry_run": a.dry_run,
          "budget_usd": a.budget, "spend_usd": 0.0, "instances": [],
          "pending_jobs": [], "running_jobs": [],
          # loop-carried accumulators (locals in the pre-fleetd inline loop)
          "last_bid": None, "max_bid": a.max_bid, "first_seen_dph": None,
          "floor_samples": [], "decay_streak": 0, "not_live": 0,
          "was_live": None, "rescue_deadline": None, "last_bid_put": 0.0,
          "t_prev": now, "t0": now, "pref_alarmed": False, "reconciled": False,
          # automatic eviction replacement (owner directive 2026-08-05). fleetd
          # seeds these from the durable watch record on every daemon start, so
          # a restart cannot reset the replacement counter and hand the ladder a
          # fresh budget of autonomous rentals — see Fleet._init_runtime.
          "replacements": 0, "replacement_history": [],
          "replacement_refused": None, "launch_dph_anchor": None,
          # ...and the disk the job was LAUNCHED at, which is the same class of
          # state: a bound the rehost sizing must not re-derive from whichever
          # box it is currently holding (task #69).
          "launch_disk_gb": None,
          # ...and the sm ARCHITECTURE allowlist the launch declared, same class
          # again: which silicon the workload runs on is not a fact about
          # whichever box currently holds it (2026-08-17/18, two runs voided by
          # an sm_120 rehost). Empty = unconstrained.
          "launch_cc_allow": [],
          # ...and the launch ENV pin (EVAL_ENV_VER), same class of state again:
          # which instrument the job grades on is a property of the WORKLOAD, and
          # the replacement lane cannot read it off an evicted box.
          "launch_env_pin": {},
          "evicted_machines": set(),
          # ...and when + why each was excluded, so the exclusion can EXPIRE by
          # class (EVICTED_EXCLUSION_TTL_S). Keyed by str(machine_id).
          "evicted_machine_ts": {},
          # handoff is DEFAULT over the ceiling (2026-07-15 flip); --strict-ceiling
          # wins (terminate above the line), --no-handoff sets a.handoff False.
          # serve lane: handoff is jobd-shaped (retarget tickets onto the
          # understudy) and has no serve analog, so serve_mode forces it OFF —
          # over the ceiling a serve watch gets the get-and-hold alarm only.
          "handoff_on": (getattr(a, "handoff", True)
                         and not getattr(a, "strict_ceiling", False)
                         and not getattr(a, "serve_mode", False)),
          # Can WHOEVER is driving this ladder carry a migration all the way to
          # `complete`? Defect #61: under fleetd it could not — a non-`run` watch
          # ends at `inst is None` two ticks after the primary is destroyed, one
          # tick before handoff_poll returns `complete`, so the understudy
          # inherited no watch and no budget cap and the stray sweep adopted it
          # as an uncapped `bare` box. FAIL CLOSED: a driver has to say so, and
          # the default is that it cannot. (fleetd says so via its policy since
          # the carryover fix; cmd_job_supervise says so below.)
          "handoff_can_complete": bool(getattr(a, "handoff_can_complete", False)),
          # The named escape hatch. Skips the ARM preconditions ONLY — never the
          # fence rails. See HANDOFF_DESIGN §11 and vastconf.JOBS_HANDOFF_UNSAFE_KEY.
          "handoff_unsafe_override": bool(
              getattr(a, "handoff_unsafe_ignore_preconditions", False)),
          # fleetd `serve` profile: same bid ladder, no jobd queue semantics —
          # see the serve_mode branches in job_supervise_tick.
          "serve_mode": bool(getattr(a, "serve_mode", False)),
          # WHAT this box is supposed to be serving (P3). Both None on every
          # watch that did not ask for the check, which is what makes the
          # identity tick a no-op for a legacy serve watch rather than a new
          # code path it has to survive.
          "model_artifact": getattr(a, "artifact", None),
          "expect_ident": getattr(a, "expect_ident", None)}
    return jc, handoff._init_job_handoff_state()


# moved-from: herdd._job_sup_inst
def _job_sup_inst(jc: MutableMapping[str, Any], iid: str) -> dict[str, Any] | None:
    """This tick's instance body for `iid` from the single per-tick fetch."""
    for i in jc.get("instances") or []:
        if str(i.get("id")) == iid:
            return i  # type: ignore[no-any-return]
    return None


# moved-from: herdd._job_sup_reattach
def _job_sup_reattach(jc: MutableMapping[str, Any], iid: str) -> None:
    """Re-attach jobd after a not-live -> live transition (an attach-started
    daemon does not survive a resume)."""
    if jc["a"].dry_run:
        print(f"[dry-run] would re-attach jobd to {iid}")
        return
    try:
        lifecycle.cmd_job_attach(argparse.Namespace(id=int(iid), dry_run=False))
    except SystemExit as e:                        # box mid-boot etc — retry next tick
        print(f"!! jobd re-attach failed ({e}) — will retry next tick")
    except Exception as e:                         # ssh refusal/timeout must NOT kill
        # the babysitter (2026-07-11 box 44514902: pubkey rejected -> unhandled
        # CalledProcessError killed supervise mid-rescue). Best-effort anyway: a
        # --jobs box re-runs its boot stanza on every resume, so jobd revives
        # without the ssh push.
        print(f"!! jobd re-attach failed ({type(e).__name__}: {e}) — box onstart "
              f"revives jobd on resume; will retry next tick")


# --------------------------------------------------------------------------- #
# Eviction-ladder rungs owned by this lane
# --------------------------------------------------------------------------- #

# moved-from: herdd._job_resume_in_place
def _job_resume_in_place(jc: MutableMapping[str, Any], a: argparse.Namespace,
                         iid: str, market: float | None, listed: bool | None,
                         is_bid: object, now: float) -> bool:
    """RUNG ZERO of the eviction ladder: `start` the box we already rent.

    Returns True when a start was issued and the caller should keep watching;
    False on any refusal, in which case the bid rescue / re-bid / replacement
    rungs below run exactly as they did before.

    Box 47226953, 2026-08-09 01:31Z, is the case. A budgeted jobs watch with a
    live claimed ticket, and the box stopped with NO price cause whatsoever —
    `min_bid` $0.3333 against our standing $0.667, `avail: yes`. The ladder as
    written classified that `ondemand_displaced` (purely because an on-demand
    price existed), which the re-bid ladder refuses by name, so the next rung
    was renting a cold replacement. That would have thrown away 59 GB of already
    pulled weights from a 104 GiB base+merged pull and paid ~12-15 min of setup,
    to replace a box that a `start` recovered in ~40 s.

    `bidpolicy.resume_in_place` owns the decision; everything here is I/O and
    journalling. The start is the SAME PUT `herdd start` issues, and it is
    allowed to fail: vast refuses while another renter holds the GPUs
    (`_start_busy`), and that refusal is itself information — it says the chunk
    is gone and the bid rungs are the right answer after all."""
    if jc.get("serve_mode"):
        # the serve lane's own boot-SLA relaunch spec owns its recovery, and it
        # has no queue whose loss makes a replacement expensive.
        return False
    if jc.get("rescue_deadline") is not None and now <= jc["rescue_deadline"]:
        # a rescue is already in flight — our own start (a start takes tens of
        # seconds) or a rescue bid. Re-issuing every 45 s would be churn, and the
        # deadline is also what stops `dead` from renting out from under it.
        return False
    dec = bidpolicy.resume_in_place(  # type: ignore[no-untyped-call]
        present=bool(_job_sup_inst(jc, iid)), is_bid=bool(is_bid),
        market_min_bid=market, last_bid=models._num_dph(jc.get("last_bid")),
        market_listed=listed,
        tries_used=int(jc.get("resume_tries", 0) or 0),
        max_tries=replacement._rebid_knob(jc, "resume_max_tries",
                                          bidpolicy.RESUME_MAX_TRIES),
        budget_usd=getattr(a, "budget", None),
        spend_usd=jc.get("spend_usd", 0.0))
    if dec.action != "start":
        if jc.get("resume_refused") != dec.reason:
            jc["resume_refused"] = dec.reason
            print(f".. resume-in-place declined: {dec.reason}")
        return False
    jc["resume_refused"] = None
    if getattr(a, "dry_run", False) or jc.get("dry_run"):
        print(f"[dry-run] would `start` {iid} in place ({dec.reason})")
        return False
    jc["resume_tries"] = int(jc.get("resume_tries", 0) or 0) + 1
    ok, err = lifecycle._put_state_soft(iid, "running")
    print(f">> RESUME-IN-PLACE {iid}: {'start issued' if ok else f'REFUSED ({err})'}"
          f" — {dec.reason}")
    journal._job_ladder_journal(
        jc, "jobs_box_resumed" if ok else "jobs_box_resume_failed", iid=str(iid),
        ok=bool(ok), error=None if ok else err, reason=dec.reason,
        standing_bid=models._num_dph(jc.get("last_bid")), market_min_bid=market,
        market_listed=listed, tries_used=jc["resume_tries"],
        budget_usd=getattr(a, "budget", None),
        spend_usd=round(jc.get("spend_usd", 0.0), 4),
        note="rung ZERO: start the warm box we already rent, before any rung "
             "that spends. A replacement pays a measured 11m35s of setup and "
             "re-pulls every weight; this keeps the disk."
             if ok else
             "vast refused the start — usually another renter holds the GPUs, "
             "which is itself evidence the chunk is gone; the bid rungs run next")
    journal._job_handoff_emit(jc, "box_resume_in_place", ok=bool(ok),
                              reason=dec.reason, error=None if ok else err)
    if not ok:
        return False
    # Give the box the same grace the bid rescue gets before the ladder decides
    # it is dead — a start takes tens of seconds and the next poll must not read
    # the not-yet-running box as a stalled rescue.
    jc["rescue_deadline"] = now + (getattr(a, "rescue_wait", None)
                                   or JOB_SUP_RESCUE_WAIT_S)
    return True


# moved-from: herdd._job_market_read
def _job_market_read(jc: MutableMapping[str, Any],
                     inst: Mapping[str, Any] | None) -> models.MarketRead:
    """THE market read for this tick — one `MarketRead`, memoized on `jc`.

    An eviction tick used to issue THREE independent offers queries against the
    same machine: the per-tick floor (`_market_min_bid_soft`), the listed probe
    on the not-live path (`_market_bid_listed_soft`, which computes a `min_bid`
    and throws it away), and the announcement's own `_market_min_bid_read`.
    Three reads of a moving market can disagree, and the disagreement is not
    cosmetic — it reaches a money decision: `resume_in_place(market_min_bid=None,
    market_listed=True, ...)` SKIPS, so the cheapest and most reversible rung on
    the ladder (start the box we are already renting, warm disk, costs nothing if
    it fails) was declined on a tick where a concurrent read had the floor.

    One read, three consumers, one consistent view: `listed=True` now always
    comes with the `min_bid` that proved it.

    Memoized on (machine, chunk, `jc["now"]`) rather than computed once at the
    top of the tick, deliberately: the boot watchdogs and the replacement rungs
    can move the watch to a DIFFERENT box mid-tick, and a floor read for the old
    machine must not be handed to the new one. A machine change re-reads; the
    next tick re-reads.

    `_market_min_bid_soft`'s two-state contract is untouched — the decay path
    (defect D5, handoff-canary-3) still calls it directly."""
    mid = (inst or {}).get("machine_id")
    g = (inst or {}).get("num_gpus")
    key = (str(mid), g, jc.get("now"))
    hit = jc.get("_market_read")
    if hit and hit[0] == key:
        return hit[1]  # type: ignore[no-any-return]
    r = (pricing._market_min_bid_read(mid, g) if mid
         else models.MarketRead(False, False, None))
    jc["_market_read"] = (key, r)
    return r


# --------------------------------------------------------------------------- #
# S2b: vast's own record of the displacement, as EVIDENCE (NOTIFY_DESIGN §6.3)
#
# HOMED HERE, not in `replacement.py`: eleven of the twelve functions below are
# private helpers of `_job_notify_try_match` / `_job_notify_rescue_min_bid`,
# whose only callers are `_job_announce_eviction` and `job_supervise_tick` —
# both in this file. The one exception is `_job_notify_box_swap_reset`, which
# `replacement._job_pull_condemn` and `replacement._job_eviction_replace` call
# as `job_lane._job_notify_box_swap_reset(jc)`; that direction already exists
# (this module calls `replacement._job_rebid_ladder` / `_job_eviction_replace`
# the same way), and the module-attribute form is what keeps a
# `monkeypatch.setattr` on either side steering the other.
# --------------------------------------------------------------------------- #

#: How many consumed `event_id`s the ladder remembers. Small on purpose: it
#: exists to stop ONE row from labelling TWO eviction cycles of the same box,
#: and rows age out of the freshness window long before this bound binds.
# moved-from: herdd.NOTIFY_CONSUMED_MAX
NOTIFY_CONSUMED_MAX = 16


# moved-from: herdd._job_notify_match
def _job_notify_match(jc: MutableMapping[str, Any],
                      iid: object) -> dict[str, Any] | None:
    """The outbid row matched to THIS box's CURRENT eviction cycle, or None.

    Two halves, and the split is the whole design:

      * a LATCH (`notify_matched`), keyed by box id, so every tick of one
        eviction cycle reads the same row — the `evicted_announced` discipline,
        for the same reason (seventeen not-live ticks are one eviction, so they
        are also one match and one journal row). Cleared on return-to-live and
        on every box swap, beside the announce latch it shadows;
      * the CONSUMED set (`notify_consumed_ids`), which deliberately does NOT
        clear on return-to-live. The freshness window cannot see the case where
        a box is evicted, rescued, and evicted again inside fifteen minutes; the
        consumed set can, and cycle 2 priced off cycle 1's row would be a wrong
        number on a money-moving rung.

    `notify_rows` is fed by the driver (fleetd's tick, gated — see
    `fleetd.notify_policy_enabled`). The inline `job supervise` CLI has no inbox
    poll and never sets it, so this returns None there and the ladder is exactly
    its pre-S2b self. That is D2 in one line: no rows, no difference."""
    cur = jc.get("notify_matched")
    if isinstance(cur, dict) and str(cur.get("iid")) == str(iid):
        return cur
    rows = jc.get("notify_rows")
    matched: dict[str, Any] | None = notify.match_outbid(  # type: ignore[no-untyped-call]
        rows if isinstance(rows, list) else [], iid, jc.get("now"),
        exclude_ids=_job_notify_consumed_ids(jc))
    return matched


# moved-from: herdd._job_notify_consumed_ids
def _job_notify_consumed_ids(jc: MutableMapping[str, Any]) -> list[str]:
    """The consumed set, defensively. It is DURABLE state (`fleetd.
    REPLACEMENT_STATE_KEYS`), which means a hand-edited or schema-drifted
    `state.json` can hand back any JSON shape at all — and both readers iterate
    it (review round 1, 2-9). A non-iterable there used to raise inside the tick,
    which fleetd catches per-watch as `watch_error`: ONE box wedged forever,
    never rescued, because a string was an int. `notify_matched` has been
    isinstance-guarded since it was written; this is the same guard for its
    sibling, and it degrades to "less memory", never to a dead watch."""
    ids = jc.get("notify_consumed_ids")
    if not isinstance(ids, (list, tuple, set)):
        return []
    return [str(i) for i in ids]


# moved-from: herdd._job_notify_mark_consumed
def _job_notify_mark_consumed(jc: MutableMapping[str, Any],
                              event_id: object) -> None:
    """Add one `event_id` to the bounded consumed set."""
    if event_id is None:
        return
    ids = _job_notify_consumed_ids(jc)
    if str(event_id) not in ids:
        ids.append(str(event_id))
    jc["notify_consumed_ids"] = ids[-NOTIFY_CONSUMED_MAX:]


# moved-from: herdd._job_notify_consume
def _job_notify_consume(jc: MutableMapping[str, Any],
                        ev: Mapping[str, Any]) -> None:
    """Latch a matched row to this eviction cycle and mark it consumed."""
    jc["notify_matched"] = dict(ev)
    _job_notify_mark_consumed(jc, ev.get("event_id"))


# moved-from: herdd._job_notify_sweep
def _job_notify_sweep(jc: MutableMapping[str, Any], iid: object) -> None:
    """Consume every OTHER fresh in-window row for this box (review round 1,
    2-2).

    The latch takes one row per eviction cycle, and before this sweep the rest
    of the cycle's rows stayed matchable for the full freshness window — i.e.
    straight into the NEXT cycle. That is reachable and it moves money: our own
    rescue raise is PUT against a stopped instance, and if THAT raise is outbid
    before the box resumes, vast mints a second row mid-cycle. Cycle 2 then
    matched cycle 1's leftover and priced its rescue off a row describing
    neither cycle (probed on 47833510's real prices: a $1.74 quote).

    So the rule is the cycle, not the row: every row this box minted inside the
    window belongs to the cycle we are living through, and the cycle spends them
    all. Run AFTER the match, never before — sweeping first would consume the
    row we are about to match. Run on the latched path too, so a row that lands
    three ticks into a cycle is still that cycle's."""
    rows = jc.get("notify_rows")
    for eid in notify.fresh_outbid_ids(  # type: ignore[no-untyped-call]
            rows if isinstance(rows, list) else [], iid, jc.get("now")):
        _job_notify_mark_consumed(jc, eid)


def _job_evicted_latch_reset(jc: MutableMapping[str, Any]) -> None:
    """Retire the announced class + its clock, wherever `evicted_announced` is
    retired. Separate helper rather than two more `pop`s at each of the four
    seams, because a cycle that keeps its clock across a return-to-live or a box
    swap would let the host-stop escalation date the NEXT eviction from the
    PREVIOUS one and rent a replacement on its first not-live tick."""
    jc.pop("evicted_class", None)
    jc.pop("evicted_since", None)


# moved-from: herdd._job_notify_cycle_reset
def _job_notify_cycle_reset(jc: MutableMapping[str, Any]) -> None:
    """Clear the per-eviction-cycle notify latches. NOT the consumed set: that
    one is what makes the NEXT cycle refuse a row this cycle already used."""
    jc.pop("notify_matched", None)
    jc.pop("notify_quote_said", None)


# moved-from: herdd._job_notify_box_swap_reset
def _job_notify_box_swap_reset(jc: MutableMapping[str, Any]) -> None:
    """A swap (replacement / rehost / handoff promotion) retires the consumed
    set too: it is keyed to a box we no longer hold, and the new box has a NEW
    instance id, so nothing it contains can ever match again. Cleared rather
    than carried for the same reason `box_swap_reset` clears the echo window —
    state about a contract we no longer hold is a liability, not a memory."""
    _job_notify_cycle_reset(jc)
    jc.pop("notify_consumed_ids", None)


# moved-from: herdd._job_notify_latched
def _job_notify_latched(jc: MutableMapping[str, Any],
                        iid: object) -> dict[str, Any] | None:
    """The row already MATCHED, consumed and journaled for this box's current
    eviction cycle — never a fresh match.

    Every later rung reads through here rather than re-matching, so no rung can
    classify off a row whose `class_without_notify`/`class_with_notify` pair was
    not journaled first. A row that changed a verdict silently is the one
    outcome this slice's whole calibration story cannot survive."""
    ev = jc.get("notify_matched")
    if isinstance(ev, dict) and str(ev.get("iid")) == str(iid):
        return ev
    return None


# moved-from: herdd._job_notify_rescue_min_bid
def _job_notify_rescue_min_bid(jc: MutableMapping[str, Any], iid: object,
                               on_demand: float | None = None) -> float | None:
    """The `new_min_bid` the rescue rung may price off this tick, or None.

    Only from a latched row for THIS box, and only when the row's displacing
    price is above our own bid AND below this machine's on-demand rate (§6.1,
    both halves — the second restored by review round 1, F1). What the rails
    then do with it is `bidpolicy._bid_action`'s business and
    `notify_rescue_bound`'s ceiling's.

    `on_demand` defaults to the box's STICKY on-demand read (`on_demand_last`,
    written by `_sticky_on_demand` earlier in the same tick), because a failed
    probe must not silently unclamp the predicate — that is the whole lesson of
    the four boxes that bid over on-demand. None only where this box has never
    had a successful on-demand read, which is the documented degradation."""
    ev = _job_notify_latched(jc, iid)
    od = on_demand if on_demand is not None else jc.get("on_demand_last")
    if ev is None or not bidpolicy.notify_outbid_supported(  # type: ignore[no-untyped-call]
            ev, on_demand=od):
        return None
    price: float | None = ev.get("new_min_bid")
    return price


# moved-from: herdd._job_notify_try_match
def _job_notify_try_match(jc: MutableMapping[str, Any], iid: object,
                          classify: Callable[[Mapping[str, Any] | None], Any],
                          *, machine_id: object,
                          listing_floor: float | None, market_listed: bool | None,
                          match_path: str,
                          floor_source: str) -> dict[str, Any] | None:
    """Match a row to this eviction cycle if one is there, and say so.

    ONE consume, ONE `notify_outbid_matched`, ONE floor check per cycle — the
    latch does that — but the ATTEMPT runs on every not-live tick, not only on
    the tick that announced the eviction. The two events race in the field: the
    2026-08-16 case had vast's row seventeen seconds ahead of our own read, and
    nothing guarantees that order. A match that only ever ran at announce time
    would silently lose every row that lands a tick late, and "silently lost
    evidence" is the failure mode this whole channel exists to end.

    `classify` is a one-argument callable (row -> class) so the caller supplies
    the tick's own evidence; the without/with pair is computed HERE so it can
    never be assembled from two different ticks' reads.

    `match_path` / `floor_source` label WHICH of the two call sites produced the
    row (review round 1, 2-3). They are not cosmetic: the announce path passes
    the RAW `_job_market_read` floor and the late path passes the self-floor-
    GUARDED one, and the guard collapses a suppressed echo to None. Both are the
    right read for their own caller and neither was changed here — but §6.5's
    calibration dataset was silently mixing two different quantities under one
    field name, which would have been discovered as a bias in the answer rather
    than as a bug in the record. Now the row says which it is.

    Returns the matched evidence, or None."""
    if _job_notify_latched(jc, iid) is not None:
        # A row that lands three ticks into a cycle is still THIS cycle's, and
        # the cycle spends it whether or not it is the row we latched (2-2).
        _job_notify_sweep(jc, iid)
        latched: dict[str, Any] = jc["notify_matched"]
        return latched
    ev = _job_notify_match(jc, iid)
    if ev is None:
        return None
    _job_notify_consume(jc, ev)
    _job_notify_sweep(jc, iid)
    without, with_ = classify(None), classify(ev)
    journal._job_ladder_journal(
        jc, notify.MATCHED_EVENT, iid=str(iid),
        event_id=ev.get("event_id"), your_bid=ev.get("your_bid"),
        new_min_bid=ev.get("new_min_bid"), created_at=ev.get("created_at"),
        class_without_notify=without, class_with_notify=with_,
        match_path=match_path, floor_source=floor_source,
        note="vast's own outbid record, matched to this box and this eviction "
             "cycle by instance id inside the freshness window. EVIDENCE (D2): "
             "it refines the class, it never outranks the on-demand "
             "discriminator or the risen-floor arm")
    # §6.3: vast's record of OUR standing bid, against ours. Journal-only,
    # permanently — belief reconciliation has exactly one writer
    # (`ladder_core.reconcile_standing_bid`, off `dph_base`), and a second one
    # re-opens the stranded-stale-belief class the 2026-08-10 review closed.
    # This row is how we would FIND OUT that our belief drifted; it is not how
    # we would fix it.
    believed, theirs = models._num_dph(jc.get("last_bid")), ev.get("your_bid")
    if (believed is not None and theirs is not None
            and abs(believed - theirs) > bidpolicy.BID_MIN_STEP):
        journal._job_ladder_journal(
            jc, notify.BID_MISMATCH_EVENT, iid=str(iid),
            event_id=ev.get("event_id"), believed_bid=believed,
            vast_your_bid=theirs, delta=round(theirs - believed, 6),
            note="vast's record of our standing bid disagrees with the lane's "
                 "belief by more than one grid step. JOURNAL ONLY — never an "
                 "input to belief reconciliation")
    # §6.5: the echo-guard calibration read. The listing floor at the stop
    # beside the one floor read that structurally CANNOT be our own bid echoing
    # back. No behaviour, by design: this accumulates the dataset FLEET_REVIEW
    # item 3 could otherwise only buy at $0.10 a sample.
    journal._job_ladder_journal(
        jc, notify.FLOOR_CHECK_EVENT, iid=str(iid), machine_id=machine_id,
        listing_floor_at_stop=listing_floor, market_listed=market_listed,
        new_min_bid=ev.get("new_min_bid"),
        standing_bid=models._num_dph(jc.get("last_bid")),
        match_path=match_path, floor_source=floor_source,
        note="offers-listing floor at the stop vs the authoritative displacing "
             "price. Journal-only calibration of the min_bid echo guard; "
             "changes no guard and no bid. `floor_source` says whether that "
             "floor is the RAW market read (announce path) or the self-floor-"
             "guarded one (late path) — two different quantities, and the "
             "calibration is only scoreable if the row says which")
    return ev


# moved-from: herdd._job_notify_quote_journal
def _job_notify_quote_journal(jc: MutableMapping[str, Any], iid: object,
                              s: Mapping[str, Any],
                              act: Any,  # noqa: ANN401 — bidpolicy.Action
                              market: float | None) -> None:
    """Journal the notification-sourced rescue quote, once per eviction cycle.

    Emitted whenever a matched row PROPOSED a price to the rescue rung — not
    only when a bid went out. The refusal case is the more informative half:
    `emitted: null` means `_bid_target` looked at the displacing price and said
    no (over the hard ceiling, no survival cushion, or already below our
    standing bid), which is the behaviour §6.4 promises and the one an
    adversarial reader should be able to CHECK in the field rather than take on
    faith.

    Journal-only: it records the decision, it does not make it.

    Since review round 1 (M3) the row also carries the BOUNDS: the ceiling the
    quote was held to, the launch anchor that ceiling was derived from, the
    budget left, and — when there was no price — the refusal text naming which
    line fired. Without those the field record could not score whether the bound
    ever bound, which is the same defect §6.5 exists to avoid on the floor
    side."""
    nmb = s.get("notify_min_bid")
    if nmb is None or jc.get("notify_quote_said"):
        return
    jc["notify_quote_said"] = True
    emitted = None
    if act is not None and act.kind == "rescue_bid":
        try:
            emitted = float(act.reason.split(":", 1)[1])
        except (TypeError, ValueError, IndexError):
            emitted = None
    bound = bidpolicy.notify_rescue_bound(s)  # type: ignore[no-untyped-call]
    journal._job_ladder_journal(
        jc, notify.RESCUE_QUOTE_EVENT, iid=str(iid), new_min_bid=nmb,
        market_floor=market, standing_bid=models._num_dph(jc.get("last_bid")),
        proposed_floor=bidpolicy.notify_rescue_floor(nmb, market),  # type: ignore[no-untyped-call]
        # `row_raised` False = the visible market floor already dominated the
        # displacing price, so the row bounded nothing and the rescue is its
        # pre-S2b self (the byte-identity boundary, stated as a field).
        row_raised=bound.floor is not None, ceiling=bound.ceiling,
        launch_dph_anchor=bound.anchor, budget_left=bound.budget_left,
        quoted=bound.price, refused=bound.refusal,
        max_bid=jc.get("max_bid"), emitted=emitted,
        rescue_attempted=jc.get("rescue_deadline") is not None,
        note="the rescue rung priced off vast's own displacing price, under "
             "the SAME ceiling the re-bid rung obeys (min of the "
             "rebid_ceiling_mult x launch anchor, max_bid, the job-aware "
             "defense ceiling, and the on-demand fraction) and the same "
             "min-runtime affordability floor; emitted=null means a rail or a "
             "bound REFUSED it, and the answer to that is escalation, never a "
             "bigger number")


#: How many of this box's recent floor reads corroborate a risen floor at
#: classification time (`bidpolicy.floor_rise_corroborated`). A SAMPLE count, not
#: a dwell: it decides how many other observations get a vote, so a faster tick
#: makes it a tighter window and a slower one a wider — and wider only ever makes
#: `outbid` MORE reachable, i.e. it degrades toward the pre-corroboration answer.
#: Eight covers the 2026-08-26 03:04:06Z spike (its neighbours were ~50 s either
#: side) at both the 45 s and the 15 s tick.
_EVICTION_FLOOR_SAMPLES_N = 8


def _job_floor_corroboration(jc: Mapping[str, Any]) -> tuple[float, ...]:
    """This box's recent market-floor reads, newest last — the second opinion the
    risen-floor arm needs before it may call an eviction `outbid`. Suppressed
    (self-floor) reads never enter `floor_samples`, so every entry is a read we
    believed was the market."""
    return tuple(jc.get("floor_samples") or ())[-_EVICTION_FLOOR_SAMPLES_N:]


# moved-from: herdd._job_announce_eviction
def _job_announce_eviction(jc: MutableMapping[str, Any], iid: str,
                           inst: Mapping[str, Any] | None, *, is_bid: object,
                           present: bool, astat: str | None,
                           intended_status: str | None, claimed_work: bool,
                           budget: float | None) -> str | None:
    """Say ONCE, in the journal, that this box has been taken — at the moment we
    decide it, not fourteen minutes later when the ladder runs out of moves.

    THE GAP THIS CLOSES (incident 2026-08-08, task #74). Box 47214941 was priced
    off its H200 at 23:03:12Z: the machine's min_bid rose past our standing $2.55
    and the offer went unavailable. The ladder DID fall through to the rescue
    path on that very tick — it printed `treating as OUTBID` seventeen times —
    but every one of those was a bare `print()`. `herdd fleet log` showed
    nothing but `tick` events until 23:17:16Z, when the re-bid ladder finally
    exhausted its rungs and `jobs_replaced` landed with `eviction_class:
    unknown`. For fourteen minutes the operator's documented observation surface
    said the watch was healthy, so the eviction was found by hand, by polling job
    status, and the hand-rescue then collided with fleetd's own.

    So this is deliberately NOT hung on the ladder's outcome. It fires on the
    classification, carries the evidence the classification was made on, and
    latches per box so seventeen ticks are one event.

    The market probe is the EVIDENCE-PRESERVING one (`_market_min_bid_read`, via
    the per-tick `_job_market_read`): "vast answered and this machine lists no
    rentable bid offer" is the only positive signal an outbid emits, and
    collapsing it into `None` is why `classify_eviction` had never returned
    `outbid` in production (D7)."""
    if str(jc.get("evicted_announced") or "") == str(iid):
        return None                 # already said, this eviction cycle
    jc["evicted_announced"] = str(iid)
    # An eviction ENDS the self-floor suppression episode: the market just
    # spoke (a competing bid displaced us), so "continuous suppression" is
    # over by definition — and the guard is tenant-gated, so nothing else
    # clears the clock while the box sits stopped. Without this, 47398836's
    # floor-blind alarm (2026-08-10 21:13:42) fired ONE TICK after
    # rescue_recovered, from a `self_floor_since` that had frozen across a
    # 67-minute stopped gap: the "30 min continuous" mostly measured a box
    # we did not hold.
    pricing._self_floor_reset(jc)
    mkt = _job_market_read(jc, inst)
    listed = mkt.listed if mkt.ok else None
    on_demand = pricing._sticky_on_demand(
        jc, pricing._market_ondemand_soft((inst or {}).get("machine_id"),
                                          (inst or {}).get("num_gpus")) if inst else None)
    def _classify(notify_row: Mapping[str, Any] | None,
                  ) -> Any:  # noqa: ANN401 — Zone S classify_eviction is untyped
        return bidpolicy.classify_eviction(  # type: ignore[no-untyped-call]
            present=present, actual_status=astat, market_min_bid=mkt.min_bid,
            on_demand=on_demand, last_bid=models._num_dph(jc.get("last_bid")),
            market_listed=listed,
            # An ON-DEMAND box cannot be displaced by an on-demand renter, and a
            # stale `last_bid` left by a previous box made it read that way on
            # 2026-08-16 (`ondemand_displaced` with `is_bid: false`).
            is_bid=bool(is_bid), notify=notify_row,
            # A risen floor is ONE sample of a machine whose other chunks price
            # separately; the recent reads say whether anything else saw a floor
            # above our bid. Uncorroborated => `host_stop`, the conservative
            # class (2026-08-26 03:04:06Z: $0.407 against our $0.24, with $0.20
            # read 54 s later and the box back two minutes after that).
            floor_samples=_job_floor_corroboration(jc),
            # An `insufficient_credit` refusal already recorded on this watch
            # outranks every market arm. It exists only once a priced call has
            # been refused, so the FIRST classification of a credit stop can
            # still read `outbid` and a later tick corrects it.
            account_credit_ok=bidpolicy.credit_ok_from_error(  # type: ignore[no-untyped-call]
                jc.get("last_error")))
    ecls = _classify(None)
    # S2b (NOTIFY_DESIGN §6.3). BOTH classifications are journaled, because the
    # field question this slice exists to answer is not "what is the class" but
    # "how often does vast's own record disagree with what we inferred" — and a
    # log that carries only the answer we adopted can never be used to decide
    # whether §6.2 earned its precedence.
    _ev = _job_notify_try_match(jc, iid, _classify,
                                machine_id=(inst or {}).get("machine_id"),
                                listing_floor=mkt.min_bid, market_listed=listed,
                                match_path="announce", floor_source="raw")
    if _ev is not None:
        ecls = _classify(_ev)
    fields = dict(
        eviction_class=ecls,
        machine_id=(inst or {}).get("machine_id"),
        actual_status=astat, intended_status=intended_status,
        is_bid=bool(is_bid), present=bool(present),
        standing_bid=models._num_dph(jc.get("last_bid")),
        market_min_bid=mkt.min_bid,
        # The two halves of the old `None`, kept apart on purpose.
        market_read_ok=mkt.ok, market_listed=listed,
        on_demand=on_demand, max_bid=jc.get("max_bid"),
        # the learn record's entry anchor + the defense's market reference
        # (AUTOBID_DESIGN "Next iteration" §4) — None on pre-2026-08-09 boxes
        entry_floor=models._num_dph(jc.get("entry_floor")),
        p_alt=models._num_dph(jc.get("p_alt")), p_alt_ts=jc.get("p_alt_ts"),
        claimed_work=bool(claimed_work),
        pending_jobs=[v.get("job_id") for v in (jc.get("pending_views") or [])],
        budget_usd=budget, spend_usd=round(jc.get("spend_usd", 0.0), 4),
        note="box STOPPED with no self-park event and no journaled stop intent "
             "— classified EVICTION. The bid rescue / re-bid ladder / "
             "replacement rungs run next, within --budget; this event is the "
             "eviction itself, not their outcome.")
    if _ev is not None:
        # Present ONLY when a row was matched. An unconditional `None` here
        # would change the shape of every eviction event ever emitted on a
        # fleet with no notifications, which is precisely the boundary S2b
        # promised not to cross.
        fields["notify_event_id"] = _ev.get("event_id")
    # The cycle's CLASS and its CLOCK, latched beside the announce latch and
    # durable with it. The class so the host-recovery escalation below reads the
    # same verdict the journal published (the S2b rule: the ladder and the
    # announcement never disagree about why this box stopped); the clock because
    # the escalation is a wall-time bound on the whole cycle, and every other
    # deadline the ladder owns is armed by whichever rung happened to fire.
    jc["evicted_class"] = ecls
    jc["evicted_since"] = jc.get("now")
    journal._job_ladder_journal(jc, "jobs_box_evicted", iid=str(iid), **fields)
    journal._job_handoff_emit(jc, "box_evicted", **fields)
    print(f"!! EVICTION {iid}: class={ecls} standing_bid="
          f"${fields['standing_bid']} market_min_bid=${mkt.min_bid} "
          f"(offers read_ok={mkt.ok} listed={listed}) "
          f"on_demand=${on_demand} pending={len(fields['pending_jobs'])}")
    return ecls  # type: ignore[no-any-return]


# --------------------------------------------------------------------------- #
# The lane's observation surface for the shared self-floor guard
# --------------------------------------------------------------------------- #

# moved-from: herdd._JobLaneFloorHooks
class _JobLaneFloorHooks(ladder_core.LaneHooks):
    """The JOBS lane's observation surface for the shared self-floor guard.

    Same state machine as the run lane's `_RunLaneFloorHooks`, a different
    place to say it: `_job_ladder_journal` writes into the BOX's own event log
    (`jobs/nodes/<IID>/events/`, the jobs lane's identity) where the run lane's
    `_sup_emit` writes into the RUN's. The event NAMES differ for the same
    reason — `jobs_bid_self_floor` / `jobs_bid_floor_blind` — and both are what
    `fleet log` and the journal analytics already key on, so they are a wire
    format, not a style choice.

    Constructed per tick with the box id and its instance body, because those
    are the two journal fields the state machine has no reason to know about.
    `_job_ladder_journal` is resolved as a module global inside the method
    bodies, so a test that monkeypatches it still lands."""

    def __init__(self, iid: str, inst: Any) -> None:  # noqa: ANN401 — raw vast body
        self.iid = iid
        self.inst = inst

    def scaled_read(self, jc: MutableMapping[str, Any],
                    market: float | None) -> None:
        print(f".. offers list no exact-count chunk while we are the live "
              f"tenant (rescaled floor ${market}) — listing mid-flap; "
              f"treating as a failed read, no bid moves")

    def self_floor(self, jc: MutableMapping[str, Any], *,
                   market_min_bid: float | None,
                   match: Any,  # noqa: ANN401 — bidpolicy match record (kind/price/age_s)
                   surviving_floor: float | None, visible: bool) -> None:
        _which = ("our own standing bid" if match.kind == "standing" else
                  f"a bid we held {match.age_s:.0f}s ago (${match.price})")
        _surv_note = (f"; sibling floor ${surviving_floor} stays the market"
                      if visible else "; holding the bid (no defend this "
                      "tick)")
        print(f".. market floor ${market_min_bid} == {_which} — that is the "
              f"price to displace OURSELVES, not a competing bidder"
              f"{_surv_note}")
        journal._job_ladder_journal(
            jc, "jobs_bid_self_floor", market_min_bid=market_min_bid,
            standing_bid=models._num_dph(jc.get("last_bid")),
            machine_id=self.inst.get("machine_id"),
            # WHICH bid the echo matched, and how stale it is. The `prior`
            # kind is the 2026-08-09 lag-window fix; counting it in the
            # journal is how we learn the real echo duration in the field.
            matched=match.kind, matched_bid=match.price,
            matched_age_s=(None if match.age_s is None
                           else round(match.age_s, 1)),
            surviving_floor=surviving_floor,
            note="offers read returned a bid of OURS as the chunk floor "
                 "(price-to-displace-the-tenant). The matching row is "
                 "suppressed; surviving_floor (a sibling chunk) stays the "
                 "market when present — defending against our own row is "
                 "a ~BID_TARGET_MULT-per-poll ratchet toward max_bid.")

    def floor_blind(self, jc: MutableMapping[str, Any], *,
                    since_s: float) -> None:
        _mins = since_s / 60.0
        print(f"!! floor-blind {_mins:.0f} min: every offers read has "
              f"matched our own bid series — no market signal; bid "
              f"${jc.get('last_bid')} is held, not decaying")
        journal._job_ladder_journal(
            jc, "jobs_bid_floor_blind", iid=str(self.iid),
            since_s=round(since_s, 1),
            standing_bid=models._num_dph(jc.get("last_bid")),
            machine_id=(self.inst or {}).get("machine_id"),
            note="continuous self-floor suppression past "
                 "BID_SELF_FLOOR_SUSTAINED_S: all listed chunks on this "
                 "machine appear to be ours, so there is no market floor "
                 "to defend against OR decay toward — the standing bid is "
                 "frozen where the ladder last put it")

    def episode_end(self, jc: MutableMapping[str, Any], *,
                    market: float | None) -> None:
        print(f".. market floor ${market} is a real competing read again "
              f"(no longer any bid of ours within the echo window)")


# --------------------------------------------------------------------------- #
# THE TICK
# --------------------------------------------------------------------------- #

# moved-from: herdd.job_supervise_tick
def job_supervise_tick(jc: MutableMapping[str, Any],
                       hf: MutableMapping[str, Any]) -> str | None:
    """ONE `job supervise` tick — the extracted body of cmd_job_supervise's
    while-loop, called by both the legacy inline loop and fleetd's `jobs`
    profile. Policy is unchanged (SPOT_DESIGN §3.2/§3.5/§3.6 + HANDOFF_DESIGN
    §9 T7); the only thing that left the body is the trailing
    `time.sleep(JOB_SUP_POLL_S)`, which the caller now owns.

    Returns None to keep supervising, else a JOB_SUP_VERDICTS string."""
    a = jc["a"]
    now = time.time()
    dt, jc["t_prev"] = now - jc["t_prev"], now
    jc["instances"] = lifecycle._instances_soft()    # ONE fetch/tick: inst + handoff
    iid = jc["iid"]                                  # may have moved to the understudy
    if jc["handoff_on"] and not jc["reconciled"]:    # adopt a crashed-mid-flight twin
        jc["now"] = now
        handoff._job_handoff_reconcile(jc, hf)
        jc["reconciled"] = True
    inst = _job_sup_inst(jc, iid)
    present = inst is not None
    astat = (inst.get("actual_status") or "").lower() if inst else None
    live = present and astat in bidpolicy.LIVE_STATES
    dph = models._num_dph(inst.get("dph_total")) if inst else None
    is_bid = bool(inst.get("is_bid")) if inst else False
    if live and dph:
        jc["spend_usd"] += dph * dt / 3600.0
    # The BID anchors are the standing bid (`dph_base`), not the billed total
    # (`dph_base` + storage) — see `_instance_standing_bid` for why the storage
    # sliver is load-bearing. `dph` stays the fallback so a body without
    # `dph_base` behaves exactly as before.
    #
    # Seed + echo-record + RECONCILE (review 2026-08-10, M3: the standing bid
    # can move without a successful PUT of ours — an out-of-band
    # `herdd bid --price`, a PUT vast applied but answered 5xx/timeout, a
    # handoff pin — and `last_bid` drives defend_at, the rebid ladder and the
    # guard's standing arm) are the shared `ladder_core.reconcile_standing_bid`,
    # the same call the run lane's `_observe` makes. This lane's two facts: the
    # rate-limit clock is named `last_bid_put` here (`last_bid_put_ts` there —
    # both persisted names, D4), and only this lane warns when a bid box's body
    # carries no `dph_base` at all (D3).
    _true_bid = models._instance_standing_bid(inst) if inst else None

    def _dph_base_missing() -> None:
        if inst and not jc.get("dph_base_missing_said"):
            jc["dph_base_missing_said"] = True
            print(f"!! bid box {iid} reports no dph_base — falling back to "
                  f"dph_total for the bid anchors; the self-floor guard runs "
                  f"one storage sliver off until the body carries dph_base")

    ladder_core.reconcile_standing_bid(  # type: ignore[no-untyped-call]
        jc, is_bid=is_bid, true_bid=_true_bid, dph=dph,
        machine_id=(inst or {}).get("machine_id"), now=now,
        put_ts_key="last_bid_put",
        on_missing_base=_dph_base_missing,
        on_reconcile=lambda old, new: print(
            f".. standing bid reconciled from the box: ${old} "
            f"(lane belief) -> ${new} (observed dph_base)"))
    jc["now"] = now
    if dph and jc.get("launch_dph_anchor") is None:
        # IMMUTABLE price anchor for the automatic-replacement ceiling (owner
        # directive 2026-08-05: "bid/on-demand price ceilings derived from the
        # original launch"). Unlike `first_seen_dph` — which the pull/eviction
        # replacement paths deliberately RE-anchor onto each new box so the bid
        # ladder tracks the current market — this is written once, on the first
        # priced observation of the ORIGINAL box, and never again: if each
        # replacement re-anchored the ceiling, three of them at 2x would license
        # an 8x box. Instance `dph_total` is the rate actually billed (it is the
        # same field the spend accrual above reads), bid or on-demand alike.
        jc["launch_dph_anchor"] = dph
    if inst is not None and jc.get("launch_disk_gb") is None:
        # IMMUTABLE DISK anchor, same rule and the same reason as the price
        # anchor above (task #69, 2026-08-08): written once, on the first
        # observation of the ORIGINAL box, and never rewritten. The rehost
        # sizing used to floor at whatever box the watch was holding, so one
        # under-sized hop propagated for the rest of the chain — driftr3 went
        # 110 -> 110 -> 60 GB and the job died on its own disk guard. A launch
        # `--disk` is a statement about the WORKLOAD, so it belongs on the watch
        # (durable: fleetd.REPLACEMENT_STATE_KEYS), not on the box.
        # Direction matters: this is a term in a max(), so re-observing a bigger
        # box still grows the replacement — the anchor can only ever floor it.
        #
        # The REQUEST outranks the allocation. `disk_space` is what vast
        # delivered, and a host with less to give hands back a smaller container
        # rather than refusing the rental — so anchoring on it makes the "launch
        # `--disk`" claim above false exactly when it matters, and every hop
        # after inherits the shortfall. `LAUNCH_DISK_GB` is the number the launch
        # asked for; boxes that predate the stamp simply fall through to the
        # allocation, which is what this line always did.
        _a_disk, _ = models._disk_gb(inst)
        _want, _short = disksize.launch_disk_gb_from_env(  # type: ignore[no-untyped-call]
            models._instance_env(inst), _a_disk)
        if _short and not jc.get("disk_shortfall_said"):
            jc["disk_shortfall_said"] = True
            print(f"!! box {iid} was launched --disk {_want:g}G but vast "
                  f"allocated {_a_disk:g}G ({_short:g}G short) — the host had "
                  f"less to give and did not refuse. Anything this job stages "
                  f"above {_a_disk:g}G will fail on the box; a replacement will "
                  f"be sized at the {_want:g}G that was asked for, not at what "
                  f"this host delivered.")
        if _want or _a_disk:
            jc["launch_disk_gb"] = max(_want or 0.0, _a_disk or 0.0)
    if inst is not None and not jc.get("launch_cc_allow"):
        # IMMUTABLE ARCH anchor, fourth of the same family and read off the same
        # channel: the sm allowlist the launch declared (`--cc-allow` ->
        # LAUNCH_CC_ALLOW). The replacement lane was architecture-BLIND, and
        # twice in two days a rehost that honoured the VRAM floor landed on an
        # sm_120 RTX PRO 6000 whose flash_attn has no kernel image (2026-08-17,
        # 2026-08-18). Durable (fleetd.REPLACEMENT_STATE_KEYS) because an
        # EVICTED primary is gone from the snapshot exactly when the constraint
        # is needed. No stamp = no constraint: boxes launched before this, and
        # every workload that never declared one, behave as they always did.
        _cc = market_offers.parse_cc_allow(
            models._instance_env(inst).get(market_offers.LAUNCH_CC_ALLOW_ENV))
        if _cc:
            jc["launch_cc_allow"] = list(_cc)
    if inst is not None and not jc.get("launch_env_pin"):
        # IMMUTABLE ENV anchor, third of the same family. The replacement lane
        # inherits EVAL_ENV_VER off the box it replaces, but reads it through
        # `_job_primary_shape(jctx, None)` — which on an EVICTED primary (gone
        # from the tick snapshot, i.e. the case the lane exists for) yields {},
        # so the pin silently vanished and jobd fell back to eval-env/LATEST.
        _pin = replacement.launch_env_pin_from(inst)
        if _pin:
            jc["launch_env_pin"] = _pin
    if inst is not None and jc.get("entry_floor") is None:
        # PRE-RENT market floor, seeded once from the ENTRY_FLOOR the launch
        # stamped into the box env (AUTOBID_DESIGN "Next iteration" §1). It is
        # the last market read on this machine not contaminated by our own
        # tenancy (#73), and the entry anchor of the learn record
        # {entry_floor, bid, hold_time, evicted}. Durable via
        # fleetd.REPLACEMENT_STATE_KEYS, like the two anchors above; boxes
        # launched before 2026-08-09 (or hand-priced) simply never have one.
        _ef = models._num_dph(models._instance_env(inst).get("ENTRY_FLOOR"))
        if _ef is not None and _ef > 0:
            jc["entry_floor"] = _ef
    try:
        # Follow every box this watch RETAINED after an eviction to a terminal
        # outcome (retained -> expired -> reaped/destroyed, or retention_lost).
        # Uses this tick's single instance fetch; never kills the babysitter.
        retention._job_retention_sweep(jc, now)
    except Exception as e:
        print(f"!! retention sweep errored ({type(e).__name__}: {e}) — "
              f"retained boxes still self-expire via their keep label")

    # WHAT is this box serving? Ahead of EVERY rung below, because each of them
    # (stop-classify, bid rescue, boot-SLA relaunch, replacement) exists to put
    # the endpoint back in service, and putting the WRONG weights back in
    # service is worse than leaving them down. No-op unless the watch carries
    # an `--expect-ident` pin. See replacement._serve_identity_tick.
    if jc.get("serve_mode"):
        sv = replacement._serve_identity_tick(jc, inst, now)
        if sv is not None:
            return sv

    # while the handoff fence is open (CUTOVER/DRAINING) the primary is
    # DELIBERATELY parked and its queue is being emptied by the retarget — skip
    # the stop-classify + queue-drain exits so they don't misread the retirement
    # as an eviction/operator-park or exit the supervisor mid-migration (the
    # handoff tick below drives the primary; fence uses LAST tick's phase).
    fenced_entry = jc["handoff_on"] and hf["phase"] in handoff._HANDOFF_FENCE_OPEN

    # A stopped / not-running box is a THREE-WAY decision (v2.1 added a third
    # way to reach `stopped` — jobd self-park). Consult the box-event stream
    # FIRST: a parked_self/drained is SUCCESS, not a loss. Only fall to the
    # operator-vs-eviction call when nothing explains the stop, and there
    # never abandon a rescuable BID box (SPOT_DESIGN §3.5; the 2026-07-11
    # bakeoff-05 regression was an OUTBID box misread as an operator park).
    intended_status = (inst.get("intended_status") or "").lower() if inst else None
    if not fenced_entry and ((present and not live) or intended_status == "stopped"):
        # serve lane: the self-park signal is the SERVE_STATUS marker (the
        # MAX_HOURS watchdog writes SELF_PARKED before its API stop), not a jobd
        # box event — without this read, every watchdog park on a BID serve box
        # would be misread as OUTBID and rescue-resumed forever.
        if jc.get("serve_mode"):
            if replacement._serve_self_park_soft(models._instance_serve_label(inst)):
                print(f">> serve box parked ITSELF (SERVE_STATUS self-park, "
                      f"MAX_HOURS watchdog) — supervise done (resume: "
                      f"{os.path.basename(sys.argv[0])} start {iid})")
                return "self_parked"
            bx = {"parked": False, "drained_pending": False}
        else:
            bx = _box_lifecycle_soft(iid)
        # LAST tick's non-terminal tickets — the same "use the previous tick's
        # view" rule `fenced_entry` above already runs on. This tick's queue read
        # happens further down (it needs the live/not-live verdict), and hoisting
        # it here would put a B2 round trip in front of the stop classification
        # for every tick of every box.
        claimed_work = bool(jc.get("pending_views"))
        decision = classify_job_box_stop(
            present=present, live=live, is_bid=is_bid,
            intended_status=intended_status,
            box_parked=bx.get("parked"), box_drained=bx.get("drained_pending"),
            # Under fleetd an operator/fleetd stop makes the watch dormant before
            # this ladder is ever called, so reaching here is itself the evidence
            # that nobody asked for this stop. `jc["stop_intent"]` lets a driver
            # that does NOT gate upstream (the inline CLI, tests) say so anyway.
            stop_intent=bool(jc.get("stop_intent")),
            claimed_work=claimed_work)
        if decision == "self_parked":
            print(f">> box self-parked after queue drain "
                  f"(parked_self, reason={bx.get('park_reason') or '?'}) — "
                  f"supervise done (results on B2; resume: "
                  f"{os.path.basename(sys.argv[0])} start {iid})")
            return "self_parked"
        if decision == "operator_park":
            print(">> box parked by an operator (journaled stop intent, or an "
                  "idle on-demand box) — stopping supervise (not an eviction)")
            return "operator_park"
        # else (None): the box stopped, nothing explains it, and nobody asked —
        # that is an EVICTION. The not-live rescue path below defends within
        # --budget; what happens HERE is saying so, once, in the journal.
        if intended_status == "stopped":
            print(">> box shows stopped and NOBODY ASKED (no self-park event, no "
                  "journaled stop intent) — EVICTION, not an operator park; "
                  "`intended_status: stopped` is vast describing the box, not a "
                  "record of intent")
        # Only once the box is genuinely DOWN. `intended_status == stopped` can
        # lead `actual_status` by a tick (vast is describing where the box is
        # headed), and `classify_eviction` correctly answers `unknown` for a box
        # that is still live — latching that would burn the announcement on the
        # weakest possible class. Costs one poll (~45 s) against the 14 minutes
        # this exists to close.
        if not live:
            _job_announce_eviction(jc, iid, inst, is_bid=is_bid, present=present,
                                   astat=astat, intended_status=intended_status,
                                   claimed_work=claimed_work, budget=a.budget)
    if a.budget is not None and jc["spend_usd"] >= a.budget:
        print(f">> BUDGET reached (${jc['spend_usd']:.2f} >= ${a.budget}) — parking {iid}")
        if not a.dry_run:
            lifecycle._stop_instance_soft(iid)
        return "budget"                              # caller reaps a pre-cutover twin

    # Boot watchdogs (owner directives 2026-08-02 + 2026-08-03). Jobs lane,
    # pre-`running` (GPU-unbilled pull phase): a sustained-slow aggregate pull
    # rate or a blown BOOT_PULL_TIMEOUT_S condemns the HOST — terminate,
    # reschedule the queue on a fresh box (excluding failed machines), keep
    # supervising the replacement. Jobs lane, `running` without a JOBD_STATUS
    # stamp: the come-online boot SLA (BOOT_SLA_S) condemns the same way —
    # this phase bills full GPU, so a dead env-setup is the EXPENSIVE shape.
    # Serve lane (2026-08-03; previously excluded because the launch shape was
    # not reconstructible here): launch_serve.sh now saves a relaunch spec, so
    # the serve SLA covers both phases — breach destroys the box and re-fires
    # the spec on a different host (the 46682177 39-minute-pull incident).
    # The handoff fence skip mirrors the stop-classify above: mid-migration
    # the primary's state is deliberate, not evidence.
    if not fenced_entry and present and astat in health._BOOT_LOADING_STATES:
        jc["boot_loading_iid"] = jc.get("iid")   # arms the come-online SLA
        if jc.get("serve_mode"):
            sv = replacement._serve_boot_sla_tick(jc, inst, now)  # type: ignore[arg-type]
            if sv is not None:
                return replacement._serve_boot_sla_condemn(jc, inst)  # type: ignore[arg-type]
        else:
            pv = replacement._job_pull_watchdog_tick(jc, inst, now)  # type: ignore[arg-type]
            if pv is not None:
                return replacement._job_pull_condemn(jc, inst, pv)  # type: ignore[func-returns-value, arg-type]
    elif astat == "running":
        # healthy pull: retire the sampler so a later resume starts fresh
        jc.pop("pull_sampler", None)
        jc.pop("pull_sampler_iid", None)
        # Stamp the RUNNING transition. _job_boot_sla_tick clocks the env-setup
        # budget from here rather than from box creation — see the comment
        # there for why sharing one clock with the pull is a live defect.
        if jc.get("boot_running_iid") != jc.get("iid"):
            jc["boot_running_iid"] = jc.get("iid")
            jc["boot_running_since"] = now
        if not fenced_entry and present:
            if jc.get("serve_mode"):
                sv = replacement._serve_boot_sla_tick(jc, inst, now)  # type: ignore[arg-type]
                if sv is not None:
                    return replacement._serve_boot_sla_condemn(jc, inst)  # type: ignore[arg-type]
            else:
                pv = replacement._job_boot_sla_tick(jc, inst, now)  # type: ignore[arg-type]
                if pv is not None:
                    return replacement._job_pull_condemn(jc, inst, pv)  # type: ignore[func-returns-value, arg-type]

    # queue state: drained == every ticket terminal. The serve lane has NO
    # queue — the box itself is the workload — so the two queue exits that
    # would return before the bid ladder are skipped and the ladder below
    # defends/rescues the endpoint. Lifecycle exits still come from the box:
    # MAX_HOURS watchdog -> self_parked above, budget -> budget, operator
    # park -> fleetd operator-intent (dormant) or operator_park above.
    if jc.get("serve_mode"):
        # no queue -> no horizon
        jids, views, pending = [], [], []  # type: ignore[var-annotated]
    else:
        # An UNREADABLE queue is not an empty one. Both exits below stop
        # defending the box, so neither may fire on a listing we could not read
        # — that is how box 48392137 was left evicted with a live ticket while
        # every tick printed "queue_empty" (2026-08-22).
        unreadable = None
        try:
            jids = jobmeta.list_queue(iid)
        except jobmeta.QueueUnreadable as e:
            unreadable, jids = str(e), []
        views = []
        for j in jids:
            try:
                views.append(jobmeta.read_job(j, live_iids={iid} if live else set()))
            except Exception as e:                   # one bad job never kills the loop
                print(f"!! read_job({j}) failed: {e}")
        pending = [v for v in views if v["status"] not in jobmeta.TERMINAL]
        if unreadable:
            print(f"!! QUEUE UNREADABLE for {iid}: {unreadable}")
            print(f"!!   Treating it as UNKNOWN, not empty: the drain and empty "
                  f"exits are suppressed and this box stays defended. Check "
                  f"`rclone lsf b2:$B2_BUCKET/jobs/queue/{iid}/` and the [b2] "
                  f"remote in ~/.config/rclone/rclone.conf.")
        if not unreadable and not fenced_entry and jids and not pending:
            print(f">> queue drained ({len(jids)} jobs terminal) — "
                  + ("leaving box running (--keep)" if a.keep else f"parking {iid}"))
            if not a.keep and not a.dry_run:
                lifecycle._stop_instance_soft(iid)
            return "drained"
        if not unreadable and not fenced_entry and not jids:
            print(f">> queue empty for {iid} — nothing to supervise; exiting "
                  f"(submit first, then supervise)")
            return "queue_empty"

    # Keep the folded VIEWS, not just the id list. The handoff block below prices
    # its migration against how much work is actually left on this box, and only
    # the views carry that (timeout_s + the attempt start) — discarding them here
    # is what left the amortization gate with nothing to read and a fabricated
    # 24-hour default in its place (defect #63; _jobs_remaining_wall_h).
    jc["pending_views"] = pending

    # missed-checkpoint watchdog: a RUNNING job that should be checkpointing but
    # has gone dark (dead key -> silent sync freeze, the J1 incident mechanism)
    # gets a loud per-poll alarm. Fires on pure SILENCE and on a box-side
    # `checkpoint_sync_failed` signal (SPOT_DESIGN §3.7). Never money-moving.
    for v in pending:
        alarm = risk._ckpt_watchdog_alarm(v, now)
        if alarm:
            print(f"!! CKPT-STALL {alarm}")

    # near-done / no-checkpoint advisory (HANDOFF_WARN_PCT). Deliberately OUTSIDE
    # the `handoff_on` block below: the no-checkpoint half of it is an EVICTION
    # exposure, and spot delivers no signal, so it is exactly the box a watch
    # with handoff disabled still needs to hear about. Never gates anything.
    handoff._job_handoff_progress_warn(jc, hf)

    # ONE market read per tick, shared with the eviction announcement above and
    # the listed probe below (`_job_market_read`) — three disagreeing reads of a
    # moving market used to reach `resume_in_place` as `min_bid=None,
    # listed=True` and skip the cheapest rung on the ladder.
    _mr = _job_market_read(jc, inst) if inst else models.MarketRead(False, False, None)
    market = _mr.min_bid
    # THE SELF-REFERENTIAL FLOOR (task #73). On a chunk we are the live tenant
    # of, vast's `min_bid` is the price to displace the current tenant — us — so
    # this read can hand back our own last PUT labelled "the market". Multiply it
    # and the defend ladder chases itself: 2.697 -> 2.818 -> 3.100 -> 3.410 in
    # five minutes on 47214941 (1.10x, the survival cushion) and 1.338 -> 2.676
    # -> 2.944 in six on 47218938 (2.00x, BID_TARGET_MULT), both on machines
    # whose true floor was ~$1.33.
    #
    # The state machine — row-level suppression (F3), the rescaled-while-tenant
    # refusal (F8), the (value, kind) dedup latch, the floor-blind clock and the
    # episode end (L6) — is `ladder_core.self_floor_guard`, ONE copy shared with
    # the run lane's `_self_floor_guard` (2026-08-14, FLEET_REVIEW item 1). What
    # stays here is what is genuinely this lane's:
    #
    #   * the TENANCY gate is strict `live and is_bid`. The run lane's tolerates
    #     a running->exited->running flap (intended_status still `running`,
    #     2026-08-10 #3) because it has no rung that consumes the floor while
    #     not-live; this lane HAS one (resume-in-place) and instead refuses the
    #     rescue RAISE further down. Divergence D1 in AUTOBID_DESIGN.md §"One
    #     core, two lanes" — intentional;
    #   * the OBSERVATION surface: `jobs_bid_self_floor` / `jobs_bid_floor_blind`
    #     into the BOX's log via `_job_ladder_journal`, where the run lane emits
    #     `bid_self_floor` / `bid_floor_blind` into the RUN's;
    #   * the latch clear shape: this lane POPS `self_floor_at` where the run
    #     lane assigns None (both read back None; D5);
    #   * `floor_samples` — the run lane folds the floor in
    #     `_refresh_default_ceiling` instead, so the append below has no twin.
    #
    # Suppressive only. It can lower no rail and raise no ceiling; the cushion,
    # the cost cap and the on-demand clamps in `_bid_target` are untouched. A
    # suppressed read is a FAILED read, so it neither moves the bid nor pollutes
    # `floor_samples`, whose median is the fallback `max_bid`.
    market = ladder_core.self_floor_guard(  # type: ignore[no-untyped-call]
        jc, market, tenant=bool(live and is_bid),
        floors=(_mr.floors if _mr.ok else None), scaled=_mr.scaled,
        machine_id=(inst or {}).get("machine_id"), now=now,
        hooks=_JobLaneFloorHooks(iid, inst), clear_latch_by_pop=True)
    if market:
        jc["floor_samples"].append(market)
    # on-demand price (ceiling anchor); a FAILED read falls back to the last one
    # we saw for this box rather than to None, which would unclamp the bid — see
    # _sticky_on_demand for the four boxes that bid over on-demand without it.
    on_demand = pricing._sticky_on_demand(
        jc, pricing._market_ondemand_soft(inst.get("machine_id"),
                                          inst.get("num_gpus")) if inst else None)
    # Replacement-market read (p_alt), refreshed on its own cadence for BID
    # boxes — live or stopped alike, so the one-shot defense prices off a read
    # at most P_ALT_POLL_S old at the moment of an eviction. Immune to #73 by
    # construction: it queries every machine EXCEPT this one.
    if is_bid and inst is not None:
        try:
            replacement._job_palt_poll(jc, now, own_machine=inst.get("machine_id"))
        except Exception as e:                    # a market read never kills a tick
            print(f".. p_alt poll errored ({type(e).__name__}: {e}) — "
                  f"the defense falls back to the plain ladder")
    # DEFAULT cap is on-demand-anchored (AUTOBID_DESIGN): get-and-hold pays up
    # to just under on-demand; --strict-ceiling caps at 0.50x on-demand. The
    # median-floor (2026-07-12 ratchet fix) is the fallback when on-demand is
    # unreadable. An explicit --max-bid overrides both.
    if a.max_bid is None:
        jc["max_bid"] = bidpolicy._default_max_bid(  # type: ignore[no-untyped-call]
            jc["floor_samples"], jc["first_seen_dph"], on_demand=on_demand,
            strict_ceiling=getattr(a, "strict_ceiling", False))
    s = bidpolicy.mk_poll_state(present=present, actual_status=astat,  # type: ignore[no-untyped-call]
                                market_min_bid=market, last_bid=jc["last_bid"],
                                max_bid=jc["max_bid"],
                                last_bid_put_ts=jc["last_bid_put"],
                                decay_streak=jc["decay_streak"],
                                rescue_attempted=jc["rescue_deadline"] is not None,
                                now=now, on_demand=on_demand,
                                # S2b review round 1 (M3): the anchor + the
                                # budget the NOTIFICATION-priced rescue quote is
                                # bounded by. Read by
                                # `bidpolicy.notify_rescue_bound` and by nothing
                                # else on this state — with `notify_min_bid`
                                # None they are inert, which is what keeps the
                                # pre-S2b rescue byte-identical.
                                launch_dph_anchor=models._num_dph(
                                    jc.get("launch_dph_anchor")),
                                budget_usd=a.budget,
                                spend_usd=jc.get("spend_usd", 0.0),
                                # ...and the other two bounds the re-bid rung
                                # answers to (review round 2). The rescue PUT
                                # runs BEFORE `_job_rebid_ladder` on this tick,
                                # so a bound only that rung applies is a bound
                                # the row walks past: with a live job-aware
                                # defense the two ceilings measured $0.606 vs
                                # $2.25. Same knob, same `defense_ceiling`, same
                                # inputs (`_job_defense_inputs`) — read by the
                                # notification-priced quote and by nothing else
                                # on this state.
                                rebid_ceiling_mult=replacement._rebid_knob(
                                    jc, "rebid_ceiling_mult",
                                    bidpolicy.REBID_CEILING_MULT),
                                defense_cap=replacement._job_defense_cap(
                                    jc, now),
                                # S2b (§6.4): the displacing price from a
                                # MATCHED outbid row, or None. Reaches one arm —
                                # the rescue quote — and only ever raises the
                                # floor it prices off; every rail
                                # (`_bid_target`'s preference, cost cap,
                                # survival cushion and the
                                # BID_CEILING_ONDEMAND_FRAC clamp) binds after
                                # it, unchanged. None on every box with no
                                # matched row, which is every box until the
                                # driver feeds `notify_rows`.
                                notify_min_bid=_job_notify_rescue_min_bid(
                                    jc, iid, on_demand=on_demand))
    # Assigned rather than passed: `RunState` pins `mk_poll_state`'s key set, so
    # Zone S reads these three with `.get()` and degrades to its old behaviour
    # when a lane (or an older state.json) has none of them.
    #   * bid_history + machine_id -> the decay hysteresis (`_recent_raise_hold`):
    #     a price a rescue/re-bid rung just paid may not be decayed away inside
    #     REBID_WAIT_S. The series is the one the self-floor guard already keeps.
    #   * decay_streak_since -> the decay dwell as a DURATION (BID_DECAY_S), not
    #     a poll count that a shorter tick silently re-tunes.
    s["machine_id"] = (inst or {}).get("machine_id")
    s["bid_history"] = pricing._bid_history_for(jc, s["machine_id"])
    s["decay_streak_since"] = jc.get("decay_streak_since")
    # preferred-ceiling / handoff trigger (get-and-hold only). Upgraded from a
    # bare print to a real box-lifecycle event (T7): the run lane emits
    # `bid_over_preferred_ceiling`, so the jobs lane should too — telemetry the
    # dwell/arm trigger and promotion metrics read (keyed on the box, the jobs
    # lane's identity, via jobs/nodes/<IID>/events/).
    # The ALARM is `_preferred_ceiling_alarm` (bid over the preferred line) and
    # the TRIGGER is `_handoff_trigger` (bid over what the bid POLICY would put
    # right now). They were the same test until 2026-08-08 22:17Z, when the
    # survival cushion walked our own bid 2.697 -> 3.410 on a tight machine and
    # the handoff read the policy's own target as excess — see `_handoff_trigger`.
    over_pref, pref = bidpolicy._preferred_ceiling_alarm(s)  # type: ignore[no-untyped-call]
    trigger_on, _pref2, policy_target, trigger_why = bidpolicy._handoff_trigger(s)  # type: ignore[no-untyped-call]
    if over_pref and not jc["pref_alarmed"]:
        if not jc["handoff_on"]:
            _tail = "; handoff (opt-in) would migrate to a cheaper box"
        elif trigger_on:
            _tail = "; handoff ARMED path migrating to a cheaper box"
        else:
            _tail = (f"; handoff will NOT arm ({trigger_why}"
                     + (f", policy target ${policy_target}" if policy_target
                        else "") + ")")
        print(f">> bid ${jc['last_bid']} over preferred ceiling ${pref} "
              f"({bidpolicy.BID_CEILING_ONDEMAND_FRAC:g}x on-demand ${on_demand}) — "
              f"get-and-hold{_tail}")
        try:
            jobmeta.emit_box_event(iid, "bid_over_preferred_ceiling",
                                   actor="job-supervise", bid=jc["last_bid"],
                                   preferred=pref, on_demand=on_demand)
        except Exception as e:                    # a failed emit must never kill the loop
            print(f"!! bid_over_preferred_ceiling emit failed ({e})")
        jc["pref_alarmed"] = True
    elif not over_pref:
        jc["pref_alarmed"] = False

    # --- the HARD-ceiling escalation (recalibration 2026-08-09, item A) ------ #
    # `bid_decision` refuses to price a machine whose survival cushion does not
    # fit under BID_CEILING_ONDEMAND_FRAC x on-demand. On a LIVE box that is a
    # silent no-op — `_bid_action` sees a None target and makes no move, which is
    # exactly right (HOLD the standing bid; never raise into a dominated price)
    # and exactly the shape #78 was filed about: a money decision nobody was told
    # about. So say it, once per condition, on the queue fleetd drains.
    #
    # This is a DECISION, not an alarm: the box is structurally unsafe to hold on
    # spot, and the rungs that answer that (replacement / on-demand) live below
    # the eviction branch. Journaling it here is what lets an operator act BEFORE
    # the eviction rather than read about it fourteen minutes after.
    # `market` and not a jc key: this is the SELF-FLOOR-GUARDED floor (a read
    # that came back as our own standing bid is None by here), so the escalation
    # can never be triggered by the #73 ratchet's phantom floor.
    _ceil_dec = bidpolicy.bid_decision(market, jc["max_bid"], on_demand)  # type: ignore[no-untyped-call]
    if live and _ceil_dec.escalate and not jc.get("ceiling_escalated"):
        print(f"!! bid policy will NOT defend {iid}: {_ceil_dec.reason}")
        journal._job_ladder_journal(
            jc, "jobs_bid_over_ceiling", iid=str(iid),
            standing_bid=models._num_dph(jc.get("last_bid")),
            market_min_bid=market, on_demand=on_demand,
            ceiling=_ceil_dec.ceiling, reason=_ceil_dec.reason,
            note="the standing bid is HELD and no raise/decay will be issued: "
                 "surviving this machine's floor costs more than the hard "
                 "ceiling allows, so the answer is a different box (the "
                 "replacement / on-demand rung), not a higher bid")
        jc["ceiling_escalated"] = True
    elif not _ceil_dec.escalate and market is not None:
        # `market is None` (suppressed echo or failed read) makes bid_decision
        # trivially non-escalating — clearing the latch there re-emitted
        # jobs_bid_over_ceiling once per suppress/real poll pair on a
        # structurally-unsafe machine (review 2026-08-10, L6). Only a real
        # read that genuinely fits under the ceiling ends the condition.
        jc["ceiling_escalated"] = False

    # --- jobs-lane handoff (T7): a SEPARATE decision on the understudy, run
    # AFTER the primary's own bid machinery, mirroring the run lane. While its
    # two-writer fence is open (CUTOVER/DRAINING) `fenced` suppresses the
    # primary's rescue/queue-exit churn so the untouched ladder can't fight the
    # deliberate retirement of the primary. Budget/park stops still win.
    fenced = jc["handoff_on"] and hf["phase"] in handoff._HANDOFF_FENCE_OPEN
    if jc["handoff_on"]:
        wb = getattr(a, "wall_budget", None)
        jc.update(now=now, dt=dt, iid=iid,
                  last_bid=jc["last_bid"], on_demand=on_demand, dph=dph,
                  # this tick's floor, so an unfence can recompute the policy
                  # target when the pre-fence bid was never recorded
                  market_min_bid=market,
                  budget_usd=a.budget,
                  # MEASURED horizon. Defect #63 replaced a fabricated 24 h with
                  # the queue's remaining `timeout_s`; defect #67 is that
                  # `timeout_s` is a HANG DETECTOR, not a work estimate — the
                  # 2026-08-08 22:17Z incident priced a migration against
                  # `remaining_wall_h: 9.904` (36000 s of hang ceiling minus
                  # 345 s elapsed) when the real work left was 1-2 h, inflating
                  # the projected saving ~5x. The horizon is now the progress
                  # ETA, capped by that ceiling and by the --wall-budget
                  # remainder; UNKNOWN (None) refuses rather than assuming the
                  # maximum. `wb` is almost never set under fleetd
                  # (JOBS_POLICY_DEFAULTS seeds wall_budget=None).
                  remaining_wall_h=risk._jobs_work_horizon_h(
                      jc.get("pending_views") or [], now,
                      wall_remaining_h=(max(0.0, (wb - (now - jc["t0"])) / 3600.0)
                                        if wb else None)),
                  # ...and the timeout ceiling itself, kept for the journal so a
                  # deferral can say WHICH of the two bounds it refused on.
                  timeout_ceiling_h=risk._jobs_remaining_wall_h(
                      jc.get("pending_views") or [], now,
                      wall_remaining_h=(max(0.0, (wb - (now - jc["t0"])) / 3600.0)
                                        if wb else None)),
                  # what the migration would DISCARD, and whether it may open the
                  # fence over it at all (tasks #62/#67).
                  work_at_risk_h=risk._jobs_work_at_risk_h(
                      jc.get("pending_views") or [], now),
                  running_unresumable=risk._jobs_unresumable_running(
                      jc.get("pending_views") or []),
                  min_running_eta_s=risk._jobs_min_running_eta_s(
                      jc.get("pending_views") or [], now),
                  ckpt_stale=risk._jobs_ckpt_stale(jc.get("pending_views") or [], now),
                  _over_pref=trigger_on,
                  pending_jobs=[v["job_id"] for v in pending],
                  running_jobs=[v["job_id"] for v in pending
                                if v["display_status"] == "running"],
                  primary_evicted=_job_primary_evicted(present, live,
                                                       jc["not_live"]))
        handoff._job_handoff_tick(jc, hf)
        fenced = hf["phase"] in handoff._HANDOFF_FENCE_OPEN
        moved_iid = jc.pop("_handoff_completed_iid", None)
        if moved_iid is not None:                 # migrated: supervise the survivor
            iid = str(moved_iid)
            jc["iid"] = iid
            # dph_base or NOTHING (review 2026-08-10, H1): the popped
            # `_handoff_completed_dph` is the understudy's observed dph_total
            # (bid + storage) — writing it into last_bid put the belief one
            # storage sliver above every echo, so the standing arm never
            # matched and the covering history entry aged out at lag_s, after
            # which the ladder defended 1.2x against its own echo. The run
            # lane's _handoff_complete had this fix; this lane did not. A None
            # is fail-closed: moves stay disabled until the reconcile path
            # seeds from the next body that carries dph_base.
            jc.pop("_handoff_completed_dph", None)
            _uinst = _job_sup_inst(jc, iid)
            _ubid = (models._instance_standing_bid(_uinst)
                     if (_uinst or {}).get("is_bid") else None)
            jc["last_bid"] = _ubid
            jc["first_seen_dph"] = _ubid
            jc["floor_samples"] = []
            jc["not_live"], jc["rescue_deadline"], jc["was_live"] = 0, None, None
            jc["pref_alarmed"] = False
            # per-box latches and echoes do not survive a box swap (L7/L8/#4)
            jc["decay_streak"] = 0
            jc["decay_streak_since"] = None
            jc["rebid_rungs"] = 0
            jc["rebid_refused"] = None
            jc["ceiling_escalated"] = False
            # the shared box-swap seam (episode latch + echo window + the
            # per-MACHINE sticky on-demand clamp). NOTE the run lane's twin,
            # `_handoff_complete`, does NOT clear the sticky clamp — divergence
            # D2 in AUTOBID_DESIGN.md §"One core, two lanes", an unfixed parity
            # gap on THAT side, preserved rather than repaired here.
            ladder_core.box_swap_reset(jc)  # type: ignore[no-untyped-call]
            jc.pop("evicted_announced", None)
            _job_evicted_latch_reset(jc)
            _job_notify_box_swap_reset(jc)
            return None

    # advance BEFORE _bid_action reads it. BOTH halves: the count is what the
    # persisted state has always carried, `since` is what makes the dwell a
    # duration — three polls is 90 s at a 45 s tick and 30 s at a 15 s one, and
    # only one of those is the number that was ratified.
    jc["decay_streak"], jc["decay_streak_since"] = \
        bidpolicy.next_decay_state(s)  # type: ignore[no-untyped-call]
    s["decay_streak"] = jc["decay_streak"]
    s["decay_streak_since"] = jc["decay_streak_since"]
    if live and not fenced:
        if jc["was_live"] is False:
            if jc.get("serve_mode"):
                # onstart re-runs serve_vllm.sh on resume, so the endpoint
                # revives itself — but the mapped ports CHANGE across a park.
                print(">> box came back — serve revives via onstart; ports "
                      "changed, re-tunnel + re-check serve_ready")
            else:
                print(">> box came back — re-attaching jobd (attach-started "
                      "daemons do not survive a resume)")
                _job_sup_reattach(jc, iid)
        jc["not_live"] = 0
        jc["rescue_deadline"] = None
        # the box is back: the re-bid ladder's rungs are per EVICTION CYCLE, so a
        # rescued box starts the next cycle with a full ladder (the counter that
        # must NOT reset is `replacements` — that one bounds rentals per watch).
        jc["rebid_rungs"] = 0
        jc["rebid_refused"] = None
        jc["resume_tries"] = 0                    # same rule for the resume rung
        # ...and the notification latch, on exactly the same rule. The CONSUMED
        # set deliberately survives: the next displacement of this box is a new
        # cycle, and the row that explained this one must never explain that one
        # (instance 47833510 was evicted twice in one night).
        _job_notify_cycle_reset(jc)
        # ...and so is the eviction announcement: the NEXT time this box is
        # taken is a new event, not a duplicate of the one we already logged.
        _job_evicted_latch_reset(jc)
        if jc.pop("evicted_announced", None) is not None:
            journal._job_ladder_journal(
                jc, "jobs_box_eviction_survived", iid=str(iid),
                standing_bid=models._num_dph(jc.get("last_bid")),
                note="the box we journaled as EVICTED is live again — the bid "
                     "rescue / re-bid ladder won it back; no replacement was "
                     "rented")
        act = bidpolicy._bid_action(s)  # type: ignore[no-untyped-call]
        if act and act.kind in ("raise_bid", "lower_bid"):
            target = float(act.reason.split(":", 1)[1])
            phase = "defending" if act.kind == "raise_bid" else "decaying"
            # name BOTH numbers + the floor; never mislabel our own bid as market
            print(f">> {phase} bid ${jc['last_bid']} -> ${target}, floor ${market}")
            if not a.dry_run:
                _ok_put = lifecycle._put_bid_soft(iid, target)[0]
                jc["last_bid_put"] = now      # ANY real PUT starts the rate-
                                              # limit clock (run-lane rule: a
                                              # 429 must not be retried hot)
                if _ok_put:
                    jc["last_bid"] = target
                    if act.kind == "lower_bid":
                        jc["decay_streak"] = 0    # PUT issued -> restart the run
                        jc["decay_streak_since"] = None
        n_run = sum(1 for v in pending if v["display_status"] == "running")
        print(f".. live: {len(pending)} pending ({n_run} running), "
              f"spend=${jc['spend_usd']:.2f}, our_bid=${jc['last_bid']}"
              + (f", floor=${market}" if market else "")
              + (f", max=${jc['max_bid']}" if jc["max_bid"] else ""))
    elif fenced:
        # primary deliberately parked for the handoff fence — skip its own
        # rescue/eviction ladder so it can't fight the retirement (the handoff
        # tick above drives the primary now).
        print(f".. handoff {hf['phase']}: primary {iid} fenced (parked), "
              f"understudy {hf.get('understudy_iid')}")
    else:
        jc["not_live"] += 1
        print(f".. NOT live (streak {jc['not_live']}, status={astat}, present={present})")
        # Is this machine still purchasable at all? Tri-state, and BOTH of the
        # rungs below need it: `False` is the only positive signal an outbid
        # emits (defect D7), and `True` with our bid still clearing the floor is
        # the positive signal that nobody took the box.
        #
        # Read off the SAME `MarketRead` the floor above came from. It used to be
        # its own `_market_bid_listed_soft` query — which computes a `min_bid`
        # and discards it — so `listed=True` could arrive next to a `market=None`
        # from a different instant, and `resume_in_place` refuses that pair.
        _r = _job_market_read(jc, inst) if inst else models.MarketRead(False, False, None)
        listed_now = _r.listed if _r.ok else None
        # S2b (§6.3): the row may land AFTER we noticed the stop — the two
        # observations race, and the 2026-08-16 case only happened to have
        # vast's row seventeen seconds early. So the match is retried on every
        # not-live tick of the cycle, not only on the tick that announced it.
        # A no-op once latched, and a no-op forever with no rows.
        if str(jc.get("evicted_announced") or "") == str(iid):
            _late = _job_notify_try_match(
                jc, iid,
                lambda row: bidpolicy.classify_eviction(  # type: ignore[no-untyped-call]
                    present=present, actual_status=astat, market_min_bid=market,
                    on_demand=on_demand,
                    last_bid=models._num_dph(jc.get("last_bid")),
                    market_listed=listed_now, is_bid=bool(is_bid), notify=row,
                    # same corroboration the announce path uses — the two must
                    # not disagree about why this box stopped
                    floor_samples=_job_floor_corroboration(jc),
                    account_credit_ok=bidpolicy.credit_ok_from_error(  # type: ignore[no-untyped-call]
                        jc.get("last_error"))),
                machine_id=(inst or {}).get("machine_id"),
                listing_floor=market, market_listed=listed_now,
                match_path="late", floor_source="guarded")
            if _late is not None:
                # the poll state was built before the match existed; the rescue
                # arm is the only thing that reads this, and it runs below.
                s["notify_min_bid"] = _job_notify_rescue_min_bid(
                    jc, iid, on_demand=on_demand)
        if jc["not_live"] >= bidpolicy.NOT_LIVE_DEBOUNCE:
            # RUNG ZERO — resume in place, before any rung that spends. Cheapest
            # and most reversible move on the ladder: it starts the box we are
            # already renting, keeps its warm disk, and costs nothing if it
            # fails. Runs at the debounce point rather than at the rescue
            # deadline for exactly that reason. `bidpolicy.resume_in_place`
            # refuses whenever a start could not be legal (displaced, outbid, no
            # market read, budget consumed), so the bid rungs below are reached
            # unchanged in every case they used to own.
            if _job_resume_in_place(jc, a, iid, market, listed_now, is_bid, now):
                jc["was_live"] = False
                return None                       # start issued; box may return
            act = bidpolicy._bid_action(s)  # type: ignore[no-untyped-call]
            if (act and act.kind == "rescue_bid"
                    and (inst or {}).get("intended_status") == "running"):
                # exited FLAP, not a displacement (review 2026-08-10, #3): a
                # healthy box can transiently report exited with intended
                # still `running` — the chunk is still ours, and a floor that
                # matches our own bid series there is the echo, not a
                # competitor. A rescue raise would outbid OURSELVES at 1.2x
                # per flap. Only the raise is refused; resume-in-place above
                # and the relaunch/replacement rungs below keep the raw read.
                _flap_self = bidpolicy.market_floor_self_match(  # type: ignore[no-untyped-call]
                    market, jc.get("last_bid"),
                    bid_history=pricing._bid_history_for(
                        jc, (inst or {}).get("machine_id")), now=now)
                if _flap_self is not None:
                    print(f".. exited flap on a chunk still ours (intended="
                          f"running), floor ${market} == our own "
                          f"{_flap_self.kind} bid — refusing the rescue raise")
                    act = None
            # S2b (§6.4): say what the notification proposed and what the rails
            # actually emitted — including `emitted: null`, which is the refusal
            # that proves the rails still bind. Once per eviction cycle, beside
            # the match that produced it, so `fleet report` can score
            # quote-vs-outcome without inferring either half.
            _job_notify_quote_journal(jc, iid, s, act, market)
            if act and act.kind == "rescue_bid" and jc["rescue_deadline"] is None:
                target = float(act.reason.split(":", 1)[1])
                print(f">> outbid — RESCUE: bid ${jc['last_bid']} -> ${target}, "
                      f"floor ${market} (box auto-resumes; jobd resumes the jobs)")
                _ok_put = True
                if not a.dry_run:
                    _ok_put = lifecycle._put_bid_soft(iid, target)[0]
                    jc["last_bid_put"] = now  # rate-limit clock on ANY real PUT
                    if _ok_put:
                        jc["last_bid"] = target
                # Arm the rescue deadline only when the PUT LANDED (review
                # 2026-08-10, M4): arming on a failed PUT burned the eviction
                # cycle's single rescue on a transient 429/5xx — a
                # JOB_SUP_RESCUE_WAIT_S stall waiting on a bid vast never saw,
                # then the re-bid ladder, then a replacement rental. A failed
                # PUT now retries next tick; three consecutive failures arm the
                # deadline anyway so a dead bid API cannot spin the rung
                # forever and the ladder advances to the rungs that can act.
                if _ok_put:
                    jc.pop("rescue_put_failures", None)
                    jc["rescue_deadline"] = now + (a.rescue_wait or JOB_SUP_RESCUE_WAIT_S)
                else:
                    _nf = int(jc.get("rescue_put_failures") or 0) + 1
                    jc["rescue_put_failures"] = _nf
                    print(f"!! rescue bid PUT failed ({_nf}/3) — retrying "
                          f"next tick")
                    if _nf >= 3:
                        jc.pop("rescue_put_failures", None)
                        jc["rescue_deadline"] = now + (a.rescue_wait
                                                       or JOB_SUP_RESCUE_WAIT_S)
            # THE WAITING IS OVER (owner directive 2026-08-28: "a job that hits
            # this case automatically moves to a new host. we don't want to
            # block on this case"). Every deadline above is armed by a rung that
            # BOUGHT something — a rescue raise (900 s), a re-bid rung (300 s),
            # a rung-zero `start` (another 900 s, up to RESUME_MAX_TRIES times)
            # — and each is sized for the mechanism it paid into. None of them
            # is sized for a HOST that stopped our box: no price shortens that
            # wait, so on a `host_stop` those deadlines are dead time that a
            # claimed queue spends parked. Box 48996785, 2026-08-28 09:59Z, is
            # the case: `host_stop`, a training job 38% in with its checkpoint
            # on B2, $4.84 of $5.00 left, and a 900 s rescue deadline armed on
            # the second not-live tick. The replacement rung this ladder already
            # has was fifteen minutes away and nothing in the journal said so.
            #
            # So the host gets a BOUNDED wait, not the bid ladder's, and then
            # the ladder escalates. Read off the announced class, never
            # re-classified here, and refused with no claimed work — a
            # replacement with nothing to retarget is spend for no progress.
            escalated = bidpolicy.host_stop_escalation(  # type: ignore[no-untyped-call]
                eviction_class=jc.get("evicted_class"),
                claimed_work=bool(pending), evicted_since=jc.get("evicted_since"),
                now=now, not_live=jc["not_live"],
                escalate_after_s=replacement._rebid_knob(
                    jc, "host_stop_escalate_s", bidpolicy.HOST_STOP_ESCALATE_S))
            dead = (not present) or (
                jc["rescue_deadline"] is not None and now > jc["rescue_deadline"]) or (
                jc["rescue_deadline"] is None and act is None
                and jc["not_live"] >= 2 * bidpolicy.NOT_LIVE_DEBOUNCE) or (
                escalated is not None)
            if dead:
                why = ("box gone (host death/destroyed)" if not present
                       else "rescue stalled past deadline"
                       if jc["rescue_deadline"] and now > jc["rescue_deadline"]
                       else f"host has not returned the box in {escalated:.0f}s"
                       if escalated is not None
                       else "no rescue possible (on-demand box or no market read)")
                # RUNG BELOW THE BID (owner directive 2026-08-05): the bid
                # ladder is out of moves, so rent a different box rather than
                # hand the operator a retarget checklist. Refusals fall straight
                # through to the pre-existing `unrecoverable` text below, so the
                # manual path is never worse than it was — it is only no longer
                # the FIRST resort. serve_mode is excluded: a serve box's
                # replacement is `launch_serve.sh`'s own SLA relaunch spec
                # (there is no queue to retarget), which already exists.
                if not jc.get("serve_mode"):
                    def _classify_dead(notify_row: Mapping[str, Any] | None,
                                       ) -> Any:  # noqa: ANN401 — Zone S, untyped
                        return bidpolicy.classify_eviction(  # type: ignore[no-untyped-call]
                            present=present, actual_status=astat,
                            market_min_bid=market, on_demand=on_demand,
                            last_bid=jc.get("last_bid"),
                            # D7: "vast answered and this machine lists no
                            # rentable bid offer" is EVIDENCE of displacement,
                            # and it is the only observable an outbid emits.
                            # `None` when the read itself failed — ignorance
                            # stays `unknown`. Probed HERE rather than folded
                            # into the per-tick floor read: this branch runs on
                            # an eviction, not on every poll of every box, and
                            # `_market_min_bid_soft`'s two-state contract stays
                            # exactly as every other caller expects it.
                            market_listed=listed_now,
                            # non-bid box => the ondemand-displacement arm is
                            # unreachable (2026-08-16); see classify_eviction.
                            is_bid=bool(is_bid), notify=notify_row,
                            # the class the LADDER acts on, so it gets the same
                            # corroboration the announcement did: `outbid` buys
                            # re-bid rungs, and buying them on one unconfirmed
                            # floor read is money spent on a stopping host.
                            floor_samples=_job_floor_corroboration(jc),
                            # Insolvency outranks the market arms — the rescue
                            # ladder must not price rungs against a stop no
                            # price can undo.
                            account_credit_ok=bidpolicy.credit_ok_from_error(  # type: ignore[no-untyped-call]
                                jc.get("last_error")))
                    # BARE and REFINED, off ONE set of reads, so the pair can
                    # never be assembled from two different ticks. `ecls` is what
                    # the ladder ACTS on — S2b: the SAME row the eviction event
                    # was classified with (latched, never re-matched here), so
                    # the ladder and the announcement cannot disagree about why
                    # this box stopped. It can only move a verdict TOWARD
                    # `outbid`, which makes the expensive on-demand replacement
                    # rung LESS reachable and the (bounded, rail-clamped) re-bid
                    # rungs more.
                    #
                    # `_ecls_bare` is the pre-S2b answer at these same reads, and
                    # it owns the one consumer a row must not touch: the evicted-
                    # MACHINE exclusion TTL (review round 1, F2/M2). See
                    # `replacement._job_eviction_replace`'s `exclusion_class`.
                    _ecls_bare = _classify_dead(None)
                    ecls = _classify_dead(_job_notify_latched(jc, iid))
                    if escalated is not None:
                        # SAY IT, in the journal, with the arithmetic. The
                        # `jobs_box_evicted` note promises "the bid rescue /
                        # re-bid ladder / replacement rungs run next" and on
                        # 2026-08-28 the operator watched eleven minutes of bare
                        # `tick` and could not tell a wait from a wedge. This is
                        # the one decision that turns the former into the
                        # latter's remedy, so it gets its own event type rather
                        # than a `why` string inside somebody else's.
                        journal._job_ladder_journal(
                            jc, "jobs_host_stop_escalated", iid=str(iid),
                            eviction_class=jc.get("evicted_class"),
                            ladder_class=ecls, waited_s=round(escalated, 1),
                            escalate_after_s=replacement._rebid_knob(
                                jc, "host_stop_escalate_s",
                                bidpolicy.HOST_STOP_ESCALATE_S),
                            rescue_deadline=jc.get("rescue_deadline"),
                            not_live=jc["not_live"],
                            resume_tries=int(jc.get("resume_tries", 0) or 0),
                            standing_bid=models._num_dph(jc.get("last_bid")),
                            market_min_bid=market, market_listed=listed_now,
                            pending_jobs=[v["job_id"] for v in pending],
                            budget_usd=a.budget,
                            spend_usd=round(jc.get("spend_usd", 0.0), 4),
                            note="the host has not brought the box back inside "
                                 "the bounded wait, and no price can make it — "
                                 "skipping the re-bid rungs (they buy a WARM box "
                                 "back from a competitor, and a host_stop has "
                                 "none) and going straight to the replacement "
                                 "rung, which still refuses on budget")
                        journal._job_handoff_emit(
                            jc, "host_stop_escalated", waited_s=round(escalated, 1),
                            eviction_class=jc.get("evicted_class"))
                        print(f">> HOST-STOP ESCALATION {iid}: no return in "
                              f"{escalated:.0f}s with {len(pending)} claimed "
                              f"job(s) — replacing rather than waiting")
                    # RUNG BETWEEN the single rescue and a replacement (autobid
                    # audit 2026-08-08): while a legal winning bid still exists
                    # under the ceiling, keep bidding on the box we already have.
                    # It holds its rehydrated env, base model, dataset and newest
                    # checkpoint; a replacement pays a measured 11m35s of setup on
                    # a cold disk. Bounded to one replacement's worth of wall time
                    # (REBID_MAX_RUNGS x REBID_WAIT_S) and to the same ceiling the
                    # replacement rung is allowed to spend, so preferring it can
                    # never cost more than the thing it avoids. Refusals fall
                    # straight through to the replacement ladder below, unchanged.
                    #
                    # SKIPPED on a host-stop escalation, and only there. This
                    # rung buys the warm box back FROM A COMPETITOR; a
                    # `host_stop` is the class that says there isn't one — our
                    # standing bid already clears the floor, or the rise that
                    # looked like a competitor failed corroboration. So a rung
                    # here would raise our bid against nobody and, worse, arm
                    # another `rebid_wait_s` (300 s) x up to REBID_MAX_RUNGS,
                    # re-opening the exact stall the escalation just ended.
                    # Every path that reaches `dead` any other way keeps the
                    # rung untouched.
                    if present and escalated is None:
                        try:
                            if replacement._job_rebid_ladder(jc, a, iid, market,
                                                             on_demand, ecls, now):
                                jc["was_live"] = False
                                return None       # box kept; re-bid in flight
                        except Exception as e:    # never kill the babysitter
                            print(f"!! re-bid ladder errored "
                                  f"({type(e).__name__}: {e}) — falling through "
                                  f"to the replacement ladder")
                    try:
                        if replacement._job_eviction_replace(
                                jc, hf, ecls, why,
                                exclusion_class=_ecls_bare):
                            jc["was_live"] = False
                            return None          # supervising the replacement
                    except Exception as e:       # never kill the babysitter
                        jc["replacement_refused"] = f"{type(e).__name__}: {e}"
                        print(f"!! eviction replacement errored "
                              f"({type(e).__name__}: {e}) — falling through to "
                              f"the manual retarget instructions")
                if jc.get("serve_mode"):
                    print(f"!! {iid} unrecoverable: {why}. Relaunch the serve "
                          f"(tools/vast/launch_serve.sh — the B2-staged model "
                          f"re-pulls; same SERVE_ID resumes the marker) and "
                          f"destroy this box.")
                else:
                    _ref = jc.get("replacement_refused")
                    _rbf = jc.get("rebid_refused")
                    print(f"!! {iid} unrecoverable: {why}."
                          + (f" Re-bid ladder REFUSED: {_rbf}." if _rbf else "")
                          + (f" Automatic replacement REFUSED: {_ref}." if _ref
                             else "")
                          + " Pending jobs + the exact migration commands:")
                    for v in pending:
                        print(f"     {os.path.basename(sys.argv[0])} job retarget "
                              f"{v['job_id']} --from {iid} --box <NEW_IID>")
                    print("   (rent a box, `job attach <NEW_IID>`, then retarget — "
                          "checkpointing jobs continue from their synced state)")
                return "unrecoverable"
    jc["was_live"] = live
    return None
