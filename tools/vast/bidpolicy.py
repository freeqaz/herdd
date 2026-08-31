#!/usr/bin/env python3
"""bidpolicy — the PURE bid-defense and handoff decision cores for spot boxes.

Owns the two I/O-free state machines that decide what a supervised interruptible
box should do next, and nothing else:

  * the **bid ladder** (`poll` + `mk_poll_state` + `_bid_action` / `_bid_target`
    / the decay helpers / the ceiling helpers) — one `Action` per tick from the
    six-row precedence taxonomy, money-moving rows reachable only after every
    guard clears. Policy spec: SPOT_DESIGN, AUTOBID_DESIGN.
  * the **handoff (migration) machine** (`handoff_poll` + `mk_handoff_state` +
    the headroom/candidate filters) — one `HandoffAction` per tick over the
    understudy lifecycle and the two-writer fence. Policy spec: HANDOFF_DESIGN.

Both supervise lanes (the run lane and the jobs lane) drive these SAME pure
functions, and `fleetd` reaches them through the drivers it imports from
`herdd`; the impure drivers — market reads, B2 markers, launches, bid PUTs —
stay in `herdd.py` on purpose. The `BID_*` / `HANDOFF_*` / `DEFEND_AT` /
`NOT_LIVE_DEBOUNCE` constants below are owner-ratified policy numbers, several of
them tuned by live incidents; their comment blocks carry that provenance and are
part of the payload, not decoration.

Leaf module: no I/O of any kind (no HTTP, no subprocess, no environment, no
filesystem, no clock — `now` is always passed IN via state), and it does not
import `herdd` — so it is importable and testable without the 10k-line CLI and
there is no import cycle. Its ONE sibling import is `runmeta`, itself a
stdlib-only leaf that imports nothing from tools/vast: `poll` folds run status
through the pure `runmeta.final_status`, exactly as it did inside herdd (same
house pattern as `jobmeta`/`workflowmeta`, which import `runmeta` the same way).

Provenance: extracted from herdd.py 2026-07-30, increment I2 of
docs/plans/vast-tooling-refactor.md; behavior-preserving (the whole block moved
verbatim, comments included). `herdd.py` re-imports every name below into its
own namespace, so `herdd.<name>` stays a valid reference — and a valid
monkeypatch target — for every existing consumer and test.
"""
from __future__ import annotations

import math
import os
import statistics
import sys
from collections import namedtuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import runmeta  # noqa: E402  (pure fold helpers only; runmeta never imports herdd)


# Shared box-state vocabulary. Chased here with the predicates that read it
# (_decay_candidate / _bid_action / _preferred_ceiling_alarm / poll) so there is
# ONE definition rather than a per-module mirror; `herdd` re-imports it, so
# `herdd.LIVE_STATES` (fleetd, workflowctl, the ls/guard paths) is unchanged.
LIVE_STATES = {"running", "loading", "created"}   # non-terminal actual_status


# --------------------------------------------------------------------------- #
# supervise — pure step core (S1)
# --------------------------------------------------------------------------- #
Action = namedtuple("Action", ["kind", "reason"])
NOT_LIVE_DEBOUNCE = 2        # not-live must persist >= this many polls (a blip != eviction)
DEFEND_AT = 0.9             # default proactive-raise threshold (x last_bid); a run's
                            # spot: defend_at (SPOT_DESIGN §3.4) overrides via state
BID_MIN_STEP = 0.01        # skip a bid raise smaller than a cent (SPOT_DESIGN §3.2)
BID_RATE_LIMIT_S = 60      # min seconds between bid PUTs (vast 429 guard)
NOMINAL_TICK_S = 45.0      # the supervise/fleetd tick the dwell POLL COUNTS below were
                           # tuned at. Not a knob and not read as one: it exists so the
                           # durations that replaced those counts show their derivation,
                           # and so a shortened tick cannot silently re-tune a policy
                           # number (at a 15 s tick a count of 3 spans 30 s, a third of
                           # the 90 s it was ratified as).
BID_DECAY_S = (3 - 1) * NOMINAL_TICK_S   # 90 s. How long the floor must stay receded before
                           # we LOWER a standing bid (SPOT_DESIGN §3.2; kills the one-way
                           # ratchet that burned J1 — a transient floor spike no longer
                           # sticks for the run). ONE non-candidate poll still resets the
                           # run: this is a dwell, not a budget.
                           # (3 - 1) because a streak of N CONSECUTIVE polls spans N-1
                           # intervals: the decision is taken ON the third observation,
                           # 90 s after the first, which is exactly what the count did.
BID_DECAY_POLLS = 3        # the same dwell as a COUNT — the fallback for a state with no
                           # `decay_streak_since` timestamp (a state file written by a
                           # daemon that predates the key, or a lane that never records
                           # it). Kept in step with BID_DECAY_S at NOMINAL_TICK_S.
BID_MAX_MULT = 3.0         # FALLBACK-ONLY cap (on-demand price unreadable): this x a
                           # rolling median of the observed floor (NOT 1.25x a possibly-
                           # spiked first-seen dph). Superseded by the on-demand anchor
                           # below whenever the on-demand price is known.
BID_TARGET_MULT = 1.20     # standing bid = this x the live spot floor. History:
                           # 1.5 -> 1.20 -> 2.00 (2026-08-08 displacement audit) ->
                           # 1.20 again (OWNER ruling 2026-08-09, recalibration
                           # decision 3: "pay near the market, not a fraction of
                           # on-demand"). Because the target is
                           # min(mult x floor, cost cap), this multiple now prices
                           # the bid across the ENTIRE measured floor distribution
                           # (floor/od 0.36-0.53 -> bids at 0.43-0.64 x od), and
                           # `BID_TARGET_ONDEMAND_FRAC` is a genuine cap that only
                           # binds on tight machines (floor/od > ~0.54). Margin over
                           # the floor is paid every hour (vast charges the bid), so
                           # it is INSURANCE against displacement; the margin-vs-
                           # eviction-rate curve is unmeasured until ~20 eviction
                           # records carry bid/floor/od (recording landed
                           # 2026-08-09) — re-tune THIS number from that data, per
                           # AUTOBID_DESIGN "Next iteration: replacement-cost-aware
                           # bid controller". Keep it strictly above
                           # BID_MIN_CUSHION_MULT (1.10) so the defend/decay logic
                           # has room between the survival floor and the target.
BID_TARGET_ONDEMAND_FRAC = 0.65   # COST cap on the standing bid = this x the on-demand
                           # price. Since the 2026-08-09 return to a 1.20x multiple
                           # this is a genuine CAP, not the price-setter: it binds
                           # only when floor/od > ~0.54 (1.2 x floor would exceed
                           # it), i.e. on the expensive tail. Grounded in
                           # spot_breakeven with the MEASURED v11 setup overhead
                           # (11m35s = 0.193 h): spot beats on-demand once a box lives
                           # longer than overhead/(1 - bid/on_demand) --
                           #   bid/od 0.54 (the old 1.20x floor) -> 25 min
                           #   bid/od 0.65 (this)                 -> 33 min
                           #   bid/od 0.70                        -> 39 min
                           # so the whole cost of moving from a 1.20x cushion to a
                           # ~1.4x one is EIGHT MINUTES of required box lifetime. The
                           # owner band was 55-70%; 0.65 sits at the top of the range
                           # where the marginal lifetime cost is still under 10 min.
BID_TARGET_MULT_UNPRICED = 1.20   # the preference multiple used when the on-demand
                           # price is UNKNOWN. Numerically equal to BID_TARGET_MULT
                           # since the 2026-08-09 ruling, but kept as a SEPARATE
                           # constant: an aggressive multiple is only ever safe when
                           # BID_TARGET_ONDEMAND_FRAC can cap it, and with no
                           # on-demand read there is no cap — so if the priced
                           # multiple is ever raised again, this one must NOT follow
                           # it (2x a floor at 60% of an unreadable on-demand bids
                           # 1.2x on-demand: provably pure waste, since on-demand
                           # outranks every bid at any price). An unknown market is
                           # never a licence to spend more. The survival cushion
                           # still applies; it is 1.10x, so this only ever binds
                           # upward. Measured consequence of NOT having this rule:
                           # four boxes carried standing bids ABOVE their machine's
                           # on-demand price (44962074 and 44965461 at 1.41x od,
                           # 46177923 at 1.50x, 47018759 at 1.03x) — see
                           # AUTOBID_AUDIT_2026-08-08.md §"bidding above on-demand".
BID_MIN_CUSHION_MULT = 1.10   # SURVIVAL floor: a bid may never land within this multiple
                           # of the market floor WHEN ON-DEMAND LEAVES ROOM FOR IT. This
                           # is the v7-eviction-1 / v11 defect promoted to an invariant:
                           # `min(BID_TARGET_MULT x floor, on_demand - EPS)` silently
                           # emits a bid a tenth of a cent over the floor whenever the
                           # on-demand reference sits just above the floor, which makes
                           # us the FIRST box any competing bidder displaces. Measured
                           # instances: $0.747 over a $0.746 floor (46848347, evicted
                           # into a hand-rescue), $1.043 over a $1.040 floor (v11 resume
                           # box), $1.071 over a $1.0667 floor (understudy 46909754,
                           # dead in 45 min), $0.401 over a $0.400 floor (46880245,
                           # outbid before it finished booting). When on-demand does NOT
                           # leave room (floor within ~10% of on-demand) the cushion is
                           # UNSATISFIABLE and the bid takes the whole remaining room
                           # (on_demand - EPS) instead: there the only deterrence left is
                           # "a competitor must pay more than on-demand to take this",
                           # which is exactly what a bid at on_demand - EPS asserts.
                           # Kept EQUAL to REPLACE_MIN_CUSHION so the launch bid and the
                           # replacement ladder's "thin cushion" refusal use ONE number.
BID_CEILING_ONDEMAND_FRAC = 0.75   # the HARD ceiling on every emitted bid = this x
                           # the on-demand price (`effective_bid_ceiling`).
                           # **Promoted from advisory to hard 2026-08-09** (the bid
                           # recalibration, item A). It was the "preferred" ceiling:
                           # advisory under get-and-hold (it only fed the handoff
                           # trigger, AUTOBID_DESIGN Phase 2) and the hard cap only
                           # under --strict-ceiling. That left NOTHING under the
                           # survival cushion except `on_demand - EPS`, and on
                           # 47214941 the emitted bid crossed even this line
                           # ($3.410 against 0.75 x $3.876 = $2.907). No path may
                           # now emit above it; a policy price that does not fit is
                           # an ESCALATION, not a bigger bid (`bid_decision` rail 4).
                           # Raised 0.50 -> 0.75 on 2026-08-08 in lockstep with the
                           # cost cap above: a ceiling BELOW the standing-bid target
                           # would make every freshly launched box breach it on its
                           # first tick (latching `bid_over_preferred_ceiling`
                           # fleet-wide) and would dead-arm handoff, whose candidate
                           # filter requires a candidate target at or under this
                           # line. It must stay strictly above
                           # BID_TARGET_ONDEMAND_FRAC. The 0.75-vs-0.50 choice is
                           # OWNER decision 1 in AUTOBID_RECALIBRATION_2026-08-09.md
                           # — lower-stakes now that the clamp is hard, because the
                           # frac no longer decides whether a runaway bid is
                           # possible, only where the escalation line sits.
BID_FALLBACK_DPH_MULT = 1.25   # deepest fallback ceiling (no floor samples, no on-demand):
                           # this x the first-seen dph
BID_ONDEMAND_EPS = 0.001   # gap kept below on-demand for the "never reach on-demand" clamp
                           # = one 3-decimal rounding unit, NOT a full cent (BID_MIN_STEP):
                           # on a cheap GPU the floor sits <1c under on-demand, so subtracting
                           # a whole cent would clamp the bid BELOW the floor -> stuck

# --- the SELF-REFERENTIAL floor (incident 2026-08-08, task #73) -------------- #
# On a machine chunk we are already the tenant of, vast lists the chunk's
# `min_bid` as THE PRICE TO DISPLACE THE CURRENT TENANT — i.e. our own standing
# bid. Feeding that number back into `_bid_target` makes the defend ladder chase
# itself: every poll reads its own last PUT as "the market", multiplies, and PUTs
# again. Measured twice on 2026-08-08:
#
#   22:10-22:14  47214941  2.697 -> 2.818 -> 3.100 -> 3.410  (1.10x per poll,
#                the survival cushion, on a machine whose true floor was 1.3158)
#   23:22-23:28  47218938  1.338 -> 2.676 -> 2.944            (2.00x per poll
#                after BID_TARGET_MULT went 1.20 -> 2.00; six minutes, $1.34 ->
#                $2.94/hr on a machine whose instance body reported min_bid
#                1.333333 throughout)
#
# The discriminator is EXACT EQUALITY, and it has to be: a floor strictly ABOVE
# our standing bid is the genuine outbid signal (`classify_eviction`) and must
# keep firing. A read that lands ON our own bid, on a box we are LIVE on, is us.
BID_SELF_FLOOR_EPS = 0.0005   # |floor - our standing bid| <= this => the read is
                           # OURSELVES, not the market. Half a tenth of a cent:
                           # tighter than BID_ONDEMAND_EPS (0.001, one price-grid
                           # step) so a competitor one grid step above us is still
                           # seen as a competitor, and wide enough to absorb the
                           # float noise in a price that round-trips through JSON.
                           #
                           # NEVER TIGHTEN. Measured 2026-08-10 (probe v2,
                           # machine 52305): vast stores the standing bid at 4
                           # decimals (`dph_base` reads back 0.0336) but echoes
                           # the rented chunk's min_bid QUANTIZED TO 3 (0.034).
                           # 0.0005 is exactly the rounding radius of that
                           # quantization — zero slack. The ladder's own rails
                           # all `round(x, 3)` before a PUT, so ladder bids echo
                           # bit-exact; the eps is what carries the 4-decimal
                           # prices humans and probes PUT.
BID_SELF_FLOOR_LAG_S = 3600 # ...and how far BACK the same equality still
                           # means ourselves. MEASURED 2026-08-09 (see
                           # AUTOBID_DESIGN "The echo has a lag window"): the
                           # chunk's `min_bid` does not track our bid moves —
                           # on box 47297871 it stayed pinned at our FIRST bid
                           # ($0.016) while the standing bid walked to $0.0421
                           # over three PUTs and several minutes. So the
                           # exact-equality test against the CURRENT bid, which
                           # is all the 2026-08-08 guard had, goes blind the
                           # instant the bid moves — and stays blind for the
                           # whole window, which is exactly when the defend
                           # ladder is active.
                           #
                           # Probe v2 (2026-08-10, machine 52305, full-row
                           # attribution) measured the echo's actual staleness:
                           # 16-47 s and 64-95 s after a RAISE, 143-222 s after
                           # a LOWER. The window was 900 s (~4x that probe) —
                           # until field data said otherwise. WIDENED to 3600 s
                           # 2026-08-14 on 4 days of deployed ts_last-semantics
                           # journal data: box 47511739 (machine 56779,
                           # 2026-08-12) kept echoing a RAISE-direction prior
                           # bid ($0.832 after moving to $0.902) at ages 43 s,
                           # 450 s, 695 s and 887.6 s — 12 s from the window
                           # edge, ~10x the probe's raise-direction figure. And
                           # the 887.6 s max is CENSORED at the window: an echo
                           # older than lag_s reads as market and never
                           # journals as a match, so the observed max sits at
                           # the boundary by construction and the true tail is
                           # unknowable from suppression data. A margin over a
                           # censored max must be a multiple, not 1.4%.
                           #
                           # One hour is deliberately longer than any observed
                           # echo: the cost of being wrong in each direction is
                           # asymmetric. Too SHORT re-opens a self-ratchet that
                           # has already cost real money twice (47214941,
                           # 47218938). Too LONG only suppresses a competitor
                           # who bids, to within half a tenth of a cent, a price
                           # WE held on THIS machine in the last hour —
                           # and the penalty for that is EVERY defend poll the
                           # match keeps covering (up to the whole window for a
                           # prior price; indefinitely for one equal to the
                           # standing bid — wording corrected 2026-08-10, F4),
                           # during which the standing bid is held, not lost
                           # (the rescue/re-bid ladder still owns a real
                           # eviction, and it reads the floor on a STOPPED box
                           # where this guard does not apply at all). The
                           # sustained-suppression alarm below is the
                           # compensating control for that held-blind state —
                           # and at 1800 s it now fires MID-window, before a
                           # prior-price entry can age out, never after.
BID_SELF_FLOOR_SUSTAINED_S = 1800
                           # continuous suppression longer than this journals a
                           # floor-blind alarm (review 2026-08-10, #6): on a
                           # machine whose every listed chunk is ours there is
                           # NO observable market, so defend AND decay are both
                           # off — correct, but it must not be silent, because
                           # an already-ratcheted bid stays frozen high with
                           # nothing to decay against until an operator looks.

SelfFloor = namedtuple("SelfFloor", ["kind", "price", "age_s"])

# --- handoff (Phase 2 of auto-bid; HANDOFF_DESIGN §10 owner rulings) --------- #
HANDOFF_DWELL_S = (5 - 1) * NOMINAL_TICK_S   # 180 s. ARM only after the bid has been over
                           # the preferred ceiling CONTINUOUSLY for this long (a flap
                           # guard: stricter sibling of BID_DECAY_S). A single non-over
                           # poll resets the dwell (driver-side). `5 - 1` for the same
                           # reason as BID_DECAY_S: five consecutive polls span four
                           # intervals, and the ARM lands ON the fifth.
HANDOFF_DWELL_POLLS = 5    # the same dwell as a COUNT — the fallback when the driver
                           # records no `over_ceiling_since` timestamp. Kept in step with
                           # HANDOFF_DWELL_S at NOMINAL_TICK_S.
HANDOFF_DEADLINE_S = 1800  # cap on the 2x-box window (30 min; top of the launch->resumed
                           # range). Any pre-CUTOVER phase still open at the deadline aborts
                           # (reap understudy, stay on primary): double-bill bounded per attempt.
HANDOFF_DRAIN_DEADLINE_S = 1800  # POST-cutover escape hatch. DRAINING waits for
                           # `understudy_producing` and nothing else, so a
                           # migration that cut over and THEN lost its understudy
                           # had no exit at all: precedence 2 only aborts phases
                           # still open PRE-cutover, and _handoff_stall_alarm
                           # explicitly "does NOT force a transition" — it alarms
                           # once and latches. Observed live 2026-08-05: the
                           # perf-levers handoff (46864225 -> 46864611) cut over
                           # having correctly refused to move a RUNNING job
                           # ("retargeted 0 job(s)" — the two-writer fence doing
                           # its job), then the understudy died, and the handoff
                           # sat in DRAINING with no path out. The cost is not the
                           # primary (it is unharmed and its job runs) but a dead
                           # understudy nobody reaps and a latched alarm that
                           # hides real ones.
HANDOFF_FENCE_TIMEOUT_S = 900  # CUTOVER escape hatch: once the primary is fenced (parked) we
                           # wait for its train.sh trap to emit `final_flush` before the
                           # understudy write-enables. That flush is seconds-to-minutes when
                           # vast delivers SIGTERM on the park; 15 min with no flush means it
                           # was NOT delivered (SIGKILL park), or the primary already terminated
                           # on its own (its trap emitted a terminal, never a final_flush — see
                           # onstart/preempt_trap.sh's RC-set branch). Proceeding from the last
                           # SYNCED checkpoint then loses <= one checkpoint interval, which the
                           # spot doctrine already tolerates (SPOT_DESIGN §3.3: the 180s periodic
                           # push is the primary defense; the flush fence only narrows the loss
                           # window). Safe because the parked primary is bid-pinned (0.001) +
                           # epoch-fenced and so cannot become a second writer.
HANDOFF_COOLDOWN_S = 1800  # quiet window after ANY attempt (success or abort) before another
                           # ARM — repeated breaches mean the market moved globally, not locally.
HANDOFF_MAX = 2            # hard cap on handoffs per run; after it, fall back to get-and-hold
                           # (handoff is a voluntary cost move, capped separately from
                           # max_relaunch which guards runaway EVICTION relaunches).
HANDOFF_WINDOW_H = HANDOFF_DEADLINE_S / 3600.0   # the amortization horizon (hours): the
                           # worst-case 2x-box window used BOTH for the headroom refusal gate
                           # (projected_2x_cost) and the conservative candidate-amortization
                           # inequality (migrate only when the understudy amortizes this window).
# --- T4b: producer side of T6's box-side guards (onstart/train.sh) ------------ #
HANDOFF_TTL_MARGIN_S = 900  # slack added to HANDOFF_DEADLINE_S for the understudy dead-man
                           # deadline. The understudy's watchdog clock starts at ITS boot, and
                           # boot itself (image pull + checkpoint resume) can eat much of the
                           # deadline window before the driver even reaches CUTOVER; the margin
                           # keeps a HEALTHY handoff (supervisor alive, promotion marker on the
                           # way) from tripping the dead-man early. It is a backstop for a DEAD
                           # supervisor, so err generous.
HANDOFF_TTL_S = HANDOFF_DEADLINE_S + HANDOFF_TTL_MARGIN_S   # understudy launch-env HANDOFF_TTL_S
                           # (T6 train.sh dead-man: self-park if no runs/<ID>/handoff/promoted
                           # marker appears by TTL — the supervisor probably died mid-handoff).
HANDOFF_PARK_BID = 0.001   # the bid we PIN the fenced (parked) PRIMARY to (cmd_bid's [0.001,32]
                           # minimum): guaranteed BELOW any live market floor, so vast cannot
                           # auto-resume the retired primary during the fence->drain window (the
                           # box-44566398 stuck-bid auto-resume leak) and race the understudy's
                           # checkpoint writes. STRONGER than "just below current floor": the
                           # floor can DROP, and a floor-relative pin would then auto-resume. The
                           # epoch guard alone can't stop the parked primary — its launch env
                           # carries NO HANDOFF_EPOCH, so T6's fail-safe (unset => not stale)
                           # lets it push; the bid-pin is the belt the epoch guard needs (§4).

# --- work-awareness rails (tasks #62/#67, incident 2026-08-08 22:17Z) -------- #
# A handoff is a VOLUNTARY cost optimisation, so every one of these refuses in
# the direction of NOT moving: the cost of a refusal is a missed saving, the cost
# of a wrong move is the workload. See HANDOFF_DESIGN §11.
HANDOFF_FENCE_HOLD_ETA_S = 1200  # HOLD the fence while a RUNNING job is estimated to
                           # be within this of finishing. ARM is not enough of a check on its
                           # own: the 2026-08-08 22:17Z incident armed at 22:17:41 and fenced at
                           # 22:21:17, and the cell became ~90 s-from-done inside that gap. The
                           # hold is bounded FOR FREE by HANDOFF_DEADLINE_S (SYNCED is a
                           # pre-CUTOVER phase, so precedence 2 aborts the attempt) — no new
                           # timer, no new half-open state.
HANDOFF_CKPT_FRESH_MULT = 1.5   # a RUNNING job that declares `checkpoint_s` must have synced
                           # within this multiple of its own interval before we are allowed to
                           # park the box under it. Older than that and the "resumable" claim is
                           # unproven in the only way that matters — what is ON B2.
HANDOFF_WARN_PCT = 80      # advisory only: a job past this completion percentage gets a
                           # journal line before any migration decision. Percent is the right
                           # unit for a WARNING and the wrong one for a block (90% of a 24 h job
                           # is still 2.4 h, where migrating genuinely pays) — the blocking
                           # units are seconds (above) and dollars (the amortization).
HANDOFF_FENCE_UNWIND_S = HANDOFF_DEADLINE_S   # HARD bound on an OPEN two-writer fence that
                           # never commits its cutover. CUTOVER's only exits were `resume_understudy`
                           # (post-flush or at HANDOFF_FENCE_TIMEOUT_S) and a `retarget_incomplete`
                           # latch that returns and stays put — so a cutover whose ticket delete
                           # failed left the primary PARKED and PINNED AT HANDOFF_PARK_BID with no
                           # path back, which is the first-eviction-target livelock the autobid
                           # audit had just finished removing everywhere else. Past this the fence
                           # UNWINDS (abort_unfence: tickets back, bid restored, box resumed).


def _actor_is_cli(actor):
    """A `stopping` event whose actor is cli:<host> == operator intent (never relaunch)."""
    return isinstance(actor, str) and actor.startswith("cli:")


def _spend_time_exceeded(s):
    """The spend + wall-clock HARD caps (design §4). PURE. These accrue WHILE the
    box is live, so poll() must check them BEFORE the live->noop short-circuit —
    otherwise a never-evicted box bills past --budget / --wall-budget forever
    (the caps' whole purpose). Returns the breached-cap name or None."""
    budget = s.get("budget_usd")
    if budget is not None and s.get("spend_usd", 0.0) >= budget:
        return "budget"
    wall = s.get("wall_budget_s")
    if wall is not None and s.get("wall_clock_s", 0.0) >= wall:
        return "wall_budget"
    return None


def _guardrail_exceeded(s):
    """All hard-stop guardrails (design §4), fixed order for a deterministic exit
    reason. Returns the breached-guard name or None. PURE. max_relaunch only
    gates a re-issue; the spend/time caps are ALSO enforced earlier in poll()
    via _spend_time_exceeded so a continuously-live box can't outrun them."""
    if s.get("relaunch_count", 0) >= s.get("max_relaunch", 3):
        return "max_relaunch"
    return _spend_time_exceeded(s)


def _evict_reason(s):
    """Eviction reason at step 6a: box-reported reason if any, else infer from
    presence (listed-but-stopped == outbid; gone == host_death). PURE."""
    r = s.get("stopping_reason")
    if r:
        return r
    return "outbid" if s.get("present") else "host_death"


def _underbid_parked(s):
    """PURE. True when an intended_status=stopped park is explained by the
    standing bid sitting BELOW the live market floor — vast's own underbid park,
    not operator intent (defect D6, live 2026-07-15: a decay PUT under the floor
    parked the box intended=stopped within 47s and rule 2a exited the supervisor
    `operator_stop`). Strictly below (epsilon for float noise): a bid AT the
    floor never masks a real operator park. Unknown bid or floor -> False
    (fail toward the conservative operator-intent read).

    Reads the RAW floor (`market_min_bid_raw`) when the state carries one: the
    self-floor guard collapses a suppressed echo to the same None a failed read
    produces, and a park explained by a floor that matches a PRIOR bid of ours
    is still an underbid park — decay lowered us, vast parked us, and the floor
    echoing the bid we decayed FROM is affirmative evidence of exactly that
    (review 2026-08-10, #1). This is a diagnostic, not a price input; no bid
    move is ever priced off the raw value."""
    lb = s.get("last_bid")
    mmb = s.get("market_min_bid_raw")
    if mmb is None:
        mmb = s.get("market_min_bid")
    if lb is None or mmb is None:
        return False
    return lb < mmb - 1e-9


def mk_poll_state(*, view=None, present=False, actual_status=None,
                  intended_status=None, status_marker=None,
                  stopping_actor=None, stopping_reason=None,
                  not_live_streak=0, backoff_ready=False,
                  relaunch_count=0, spend_usd=0.0, wall_clock_s=0.0,
                  max_relaunch=3, budget_usd=None, wall_budget_s=48 * 3600,
                  max_bid=None, last_bid=None, market_min_bid=None,
                  last_bid_put_ts=0.0, rescue_attempted=False, now=0.0,
                  defend_at=None, decay_streak=0, on_demand=None,
                  handoff_fenced=False, notify_min_bid=None,
                  launch_dph_anchor=None, rebid_ceiling_mult=None,
                  defense_cap=None):
    """Build the pure-relevant state dict (the portable lane hand-builds these).

    `notify_min_bid` (S2b) is the `new_min_bid` of an outbid notification the
    driver has MATCHED to this box's current eviction cycle, or None. It reaches
    exactly one arm — the rescue quote in `_bid_action` — and only ever raises
    the floor that arm prices off; see `notify_rescue_floor`.

    `launch_dph_anchor` (S2b review round 1, M3) is the immutable launch price
    the escalating rungs derive their ceiling from. It is read by exactly one
    arm — the NOTIFICATION-priced half of the rescue quote — so that the one
    spend rung a row can newly reach is held to the same
    `REBID_CEILING_MULT x anchor` line `rebid_ladder` is held to. None keeps the
    pre-S2b rescue exactly as it was, and `notify_min_bid=None` never reads it
    at all.

    `rebid_ceiling_mult` / `defense_cap` (review round 2) are the other two
    bounds `rebid_ladder` is held to and the first cut of that fix was not: the
    per-watch `rebid_ceiling_mult` KNOB rather than the module default, and the
    job-aware `defense_ceiling`. Both are read by the same one arm. They matter
    because the rescue PUT runs BEFORE the re-bid rung on the same tick, so a
    bound only the later rung applies is a bound the row can walk straight
    past — measured at 11 of 18 sampled states with a fresh `p_alt`. None each
    = "no knob / no live defense", which is the shipped default everywhere.

    NOT built here, read with `.get()` where the lane already carries them (the
    run lane's `st` IS its persistent state; the jobs lane assigns them onto the
    per-tick dict): `bid_history` + `machine_id` — the recorded standing-bid
    series and the chunk it was observed on, read by the decay hysteresis
    (`_recent_raise_hold`) alone. Absent => the pre-hysteresis decay exactly.
    They are deliberately not factory keys: `RunState`/`HandoffSnapshot` pin
    this factory's key set, and a guard that must degrade to "no hysteresis" on
    an old state file has no business inventing one."""
    return {
        "view": view or {}, "present": present, "actual_status": actual_status,
        "intended_status": intended_status, "status_marker": status_marker,
        "stopping_actor": stopping_actor, "stopping_reason": stopping_reason,
        "not_live_streak": not_live_streak, "backoff_ready": backoff_ready,
        "relaunch_count": relaunch_count, "spend_usd": spend_usd,
        "wall_clock_s": wall_clock_s, "max_relaunch": max_relaunch,
        "budget_usd": budget_usd, "wall_budget_s": wall_budget_s,
        "max_bid": max_bid, "last_bid": last_bid,
        "market_min_bid": market_min_bid, "last_bid_put_ts": last_bid_put_ts,
        "rescue_attempted": rescue_attempted, "now": now,
        "defend_at": defend_at, "decay_streak": decay_streak,
        "on_demand": on_demand, "handoff_fenced": handoff_fenced,
        "notify_min_bid": notify_min_bid,
        "launch_dph_anchor": launch_dph_anchor,
        "rebid_ceiling_mult": rebid_ceiling_mult, "defense_cap": defense_cap,
    }


BidTarget = namedtuple("BidTarget", ["price", "reason", "ceiling", "escalate"])
_CEIL_EPS = 1e-9           # float-noise slack on every ceiling comparison; the
                           # prices themselves live on a $0.001 grid, so this can
                           # never absorb a real price difference


def effective_bid_ceiling(on_demand, max_bid=None, *, ceiling_frac=None):
    """PURE. The HARD ceiling on any bid this system may emit, from any path
    (recalibration 2026-08-09, item A).

      * on-demand KNOWN  -> `ceiling_frac x on_demand` (BID_CEILING_ONDEMAND_FRAC),
        further clamped under `on_demand - BID_ONDEMAND_EPS`.
      * on-demand UNKNOWN -> the fallback `max_bid` (None when there is none —
        an unpriced market has no derivable ceiling, and the callers treat that
        as "no ceiling rail", not as "any price").

    `max_bid` is deliberately NOT folded in when on-demand is known. It is a
    separate rail with different semantics: breaching `max_bid` is an
    AFFORDABILITY refusal (clamp, then D7's "unwinnable floor"), while breaching
    this ceiling is a STRUCTURAL verdict about the machine. Conflating them would
    reclassify every ordinary `--max-bid` clamp as "this box is unsafe to hold"
    and would turn the re-bid ladder's headroom-exhausted terminal — the one that
    hands control to the replacement rung with its arithmetic — into a generic
    escalation string.

    Before 2026-08-09 this line existed only as an ADVISORY (`_preferred_ceiling`,
    the handoff trigger) while the emitted bid was capped at `on_demand - EPS`.
    That ordering is what let 47214941's survival cushion walk a standing bid to
    $3.410 on a machine whose preferred ceiling was 0.75 x $3.876 = $2.907: the
    cushion RAISES and outranked every cost rail, so the only thing under it was
    the on-demand clamp. The advisory line is now the hard one, and a bid the
    policy cannot fit under it is an ESCALATION (rent elsewhere / go on-demand),
    never a bigger number — see `bid_decision`."""
    ceiling_frac = (BID_CEILING_ONDEMAND_FRAC if ceiling_frac is None
                    else ceiling_frac)
    od = on_demand if (on_demand and on_demand > 0) else None
    if od is None:
        return None if max_bid is None else float(max_bid)
    return min(round(ceiling_frac * od, 3), round(od - BID_ONDEMAND_EPS, 3))


def bid_decision(market_min_bid, max_bid, on_demand=None, *,
                 mult=None, ondemand_frac=None, cushion_mult=None,
                 ondemand_cap=None, ceiling_frac=None):
    """The standing bid a move aims for, WITH the arithmetic that produced it.
    Returns `BidTarget(price, reason, ceiling, escalate)`; `_bid_target` is the
    price-only alias every legacy caller still uses.

    FIVE rails, applied in this order — the order IS the policy, and both the
    2026-08-08 audit and the 2026-08-09 recalibration exist because it was short
    a rail:

      1. PREFERENCE — `mult` x the live market floor (BID_TARGET_MULT).
      2. COST CAP   — `ondemand_frac` x on-demand (BID_TARGET_ONDEMAND_FRAC), plus
         any caller-supplied `ondemand_cap` (the spot-vs-on-demand breakeven price;
         see `spot_breakeven`). Lowers the preference; never raises it.
      3. SURVIVAL CUSHION — the result may never sit within `cushion_mult` of the
         floor (BID_MIN_CUSHION_MULT). This rail RAISES, and it outranks the cost
         cap: a bid that cannot survive is not cheap, it is a 12-15 minute setup
         bill for nothing.
      4. **HARD CEILING** (`effective_bid_ceiling`) — and this one does not
         clamp-and-continue when the CUSHION is what breached it. See below.
      5. HARD RAILS — strictly below on-demand (bidding >= on-demand is pure
         waste: on-demand outranks every bid) and at or under `max_bid`
         (SPOT_DESIGN §3.2 / invariant §5.4 — the single cap on defense AND
         rescue, binding from every path).

    **Rail 4, the recalibration (2026-08-09).** Rails 2 and 3 compose badly by
    construction: the cushion is a SURVIVAL rail and deliberately outranks the
    COST rail, so on a machine whose floor is a large fraction of on-demand the
    cushion is the only thing setting the price and nothing but `on_demand - EPS`
    was under it. Measured on 47214941 (2026-08-08): the defend ladder walked
    2.697 -> 2.818 -> 3.100 -> 3.410, each step exactly 1.10 x "the floor" it had
    been handed, against a 0.75 x $3.876 = **$2.907** preferred ceiling. (The
    floor there was our own bid read back — fixed separately by the self-floor
    guard, #73 — but the precedence defect is real on GENUINE floors too, and
    that is what this rail closes.)

    The correct answer to "survival costs more than the ceiling allows" is not a
    higher bid. It is that **this machine is structurally unsafe to hold on
    spot**, which is an escalation:

      * a LIVE box being defended -> no bid move at all (`price=None` disables
        `_bid_action`'s raise/decay/rescue), so the standing bid is HELD while
        the escalation is journaled;
      * a RESCUE / re-bid ladder -> the rung refuses and the ladder falls through
        to `replacement_decision`, whose on-demand rung is the rational answer;
      * `replacement_decision` itself -> `spot_price` is None, `_viable` refuses
        with "no price", and the on-demand rung is taken.

    Paying near on-demand for a preemptible box is strictly dominated (on-demand
    outranks every bid at any price — SPOT_DESIGN #6), so `escalate` is a strictly
    better outcome than the bid it replaces, never a lost opportunity. It also
    lines up with `spot_breakeven`: at the measured 0.193 h setup, a bid at
    0.75 x on-demand needs a **46-minute** box life just to break even, and one at
    `on_demand - EPS` never breaks even at any lifetime.

    This REPLACES the old "cushion collapses onto `on_demand - EPS` — take all the
    room there is, since at that price no rational bidder outbids us" branch. That
    argument was only ever about *deterrence* and ignored the price: a bid nobody
    outbids, at 99.9% of on-demand, is an on-demand box bought through the
    preemptible queue.

    `price` is None (and `escalate` False) when there is no market read, which
    disables both bid actions. An uncapped legacy run (max_bid None) with no
    on-demand read has no ceiling and still gets a finite target.

    Why rail 3 is not optional (AUTOBID_AUDIT_2026-08-08.md §2): with only rails
    1/2/5 the formula is `min(mult x floor, on_demand - EPS)`, and whenever the
    on-demand reference sits just above the floor the min picks the SECOND term
    and hands back a bid a rounding unit over the floor — the lowest-priority bid
    the market can hold. It has happened at least four measured times, and the
    "20% cushion" printed in the launch banner was $0.001 in fact.

    Razor-thin floors (defect D7, found pre-spend 2026-07-15): when the machine's
    floor sits within BID_ONDEMAND_EPS of on-demand, the od clamp lands the target
    BELOW the floor — a KNOWN-losing bid that vast answers with an underbid park
    (the D6 incident shape, but emitted deliberately every tick). Never emit one:
    raise the target back to the floor (ceil to the $0.001 grid) while it still
    sits strictly under on-demand AND under the ceiling; when the floor can't be
    afforded (over max_bid) or is at/over on-demand, `price` is None — bid moves
    disable and the normal eviction / relaunch ladder owns the box, exactly like a
    failed market read."""
    if market_min_bid is None:
        return BidTarget(None, "no_market_read: no floor this tick, so no bid "
                               "move is priced", None, False)
    ondemand_frac = (BID_TARGET_ONDEMAND_FRAC if ondemand_frac is None
                     else ondemand_frac)
    cushion_mult = BID_MIN_CUSHION_MULT if cushion_mult is None else cushion_mult
    od = on_demand if (on_demand and on_demand > 0) else None
    if mult is None:
        # No on-demand read => no cost cap => the aggressive preference is
        # unbounded above. Fall back to the conservative multiple (see
        # BID_TARGET_MULT_UNPRICED); the survival cushion still applies.
        mult = BID_TARGET_MULT if od is not None else BID_TARGET_MULT_UNPRICED
    hi = None if od is None else round(od - BID_ONDEMAND_EPS, 3)
    ceiling = effective_bid_ceiling(od, max_bid, ceiling_frac=ceiling_frac)

    t = round(mult * market_min_bid, 3)                         # 1 preference
    if od is not None:                                          # 2 cost caps
        t = min(t, round(ondemand_frac * od, 3))
    if ondemand_cap and ondemand_cap > 0:
        t = min(t, round(ondemand_cap, 3))
    cushion = round(cushion_mult * market_min_bid, 3)           # 3 survival rail
    reason = f"target:{t}"
    if ceiling is not None and cushion > ceiling + _CEIL_EPS:   # 4 HARD ceiling
        return BidTarget(
            None,
            f"escalate_over_ceiling: surviving this floor (${market_min_bid}) "
            f"needs ${cushion:.3f} ({cushion_mult:g}x) but the hard ceiling is "
            f"${ceiling:.3f}"
            + (f" ({BID_CEILING_ONDEMAND_FRAC:g}x on-demand ${od})" if od else "")
            + " — this machine is structurally unsafe to hold on spot; HOLD the "
              "standing bid and escalate (replacement / on-demand rung)",
            ceiling, True)
    if ceiling is not None and t > ceiling + _CEIL_EPS:
        t = ceiling
        reason = (f"ceiling_capped:{t} (the ${ceiling:.3f} hard ceiling, not the "
                  f"preference/cost rails, set this price)")
    t = max(t, cushion)
    if hi is not None:                                          # 5 hard rails
        t = min(t, hi)
    if max_bid is not None:
        t = min(t, max_bid)
    if ceiling is not None:
        t = min(t, ceiling)                                     # backstop
    if t < market_min_bid:                                     # D7 raise-to-floor
        floor_t = math.ceil(market_min_bid * 1000 - 1e-9) / 1000
        if ceiling is not None and floor_t > ceiling + _CEIL_EPS:
            return BidTarget(
                None,
                f"escalate_over_ceiling: the FLOOR itself (${floor_t:.3f}) is "
                f"above the ${ceiling:.3f} hard ceiling — no legal bid takes "
                f"this machine, escalate rather than raise", ceiling, True)
        affordable = max_bid is None or floor_t <= max_bid
        under_od = od is None or floor_t < od
        if not (affordable and under_od):
            return BidTarget(None,
                             f"unwinnable_floor: ${floor_t:.3f} does not fit "
                             f"under max_bid ${max_bid} / on-demand ${od}",
                             ceiling, False)                    # unwinnable floor
        t = floor_t
        reason = f"raised_to_floor:{t}"
    # ABSOLUTE rail, independent of every frac: a bid at or over on-demand is
    # strictly dominated, so it is escalation-worthy even if some caller-supplied
    # `ceiling_frac` would have permitted it. Unreachable through the rails above;
    # kept as a backstop because it is the one invariant that must never lapse.
    if od is not None and t >= od:
        return BidTarget(None,
                         f"escalate_over_ceiling: ${t:.3f} is at or above "
                         f"on-demand ${od} — on-demand outranks every bid at any "
                         f"price, so this is pure waste", ceiling, True)
    return BidTarget(t, reason, ceiling, False)


def _bid_target(market_min_bid, max_bid, on_demand=None, *,
                mult=None, ondemand_frac=None, cushion_mult=None,
                ondemand_cap=None, ceiling_frac=None):
    """Price-only alias for `bid_decision` — the signature every caller and test
    has used since the extraction. `None` covers all three no-bid outcomes (no
    market read, unwinnable floor, over-ceiling escalation); reach for
    `bid_decision` when the journal needs to say WHICH."""
    return bid_decision(market_min_bid, max_bid, on_demand, mult=mult,
                        ondemand_frac=ondemand_frac, cushion_mult=cushion_mult,
                        ondemand_cap=ondemand_cap,
                        ceiling_frac=ceiling_frac).price


def _decay_candidate(s):
    """PURE: True when a LIVE box's standing bid sits more than one step ABOVE the
    current target (BID_TARGET_MULT x floor) — i.e. the market floor has receded and
    our bid is now stale-high. Independent of the rate limit / streak (those gate the
    PUT, not the candidacy). The streak of consecutive candidate polls is what
    licenses the downward PUT (SPOT_DESIGN §3.2 bid-decay)."""
    if not (bool(s.get("present")) and s.get("actual_status") in LIVE_STATES):
        return False
    last_bid = s.get("last_bid")
    target = _bid_target(s.get("market_min_bid"), s.get("max_bid"), s.get("on_demand"))
    if last_bid is None or target is None:
        return False
    return target + BID_MIN_STEP < last_bid


DecayStreak = namedtuple("DecayStreak", ["streak", "since"])


def next_decay_state(s):
    """PURE: the decay-candidate run AFTER this poll, as `(streak, since)` —
    the count prev+1 and the timestamp the run STARTED, both reset the instant
    the floor stops being receded (a single non-candidate poll breaks the run,
    so a brief dip below our bid can't trigger a premature lower).

    The timestamp is what makes the dwell a DURATION rather than a poll count:
    `BID_DECAY_S` means the same thing at a 45 s tick and at a 15 s one, where a
    count means three times less waiting. A driver that only stores the count
    keeps the legacy behaviour — see `_decay_dwell_satisfied`."""
    if not _decay_candidate(s):
        return DecayStreak(0, None)
    prev = s.get("decay_streak", 0)
    since = s.get("decay_streak_since")
    if not prev or since is None:
        since = s.get("now")
    return DecayStreak(prev + 1, since)


def _next_decay_streak(s):
    """PURE: the count half of `next_decay_state` — the signature both lanes
    assign from today."""
    return next_decay_state(s).streak


def _decay_dwell_satisfied(s):
    """PURE. Has the floor stayed receded long enough to license a LOWER?

    Time-based whenever the state carries `decay_streak_since` (the start of the
    current candidate run); the legacy poll COUNT otherwise, so a state written
    by a daemon that predates the key, or a lane that never records it, behaves
    exactly as it did. Either way one non-candidate poll clears the run — the
    reset, not the threshold, is what killed the J1 ratchet."""
    streak = s.get("decay_streak", 0)
    if not streak:
        return False
    since, now = s.get("decay_streak_since"), s.get("now")
    if since is not None and now is not None:
        return float(now) - float(since) >= BID_DECAY_S
    return streak >= BID_DECAY_POLLS


def _recent_raise_hold(s, *, window_s=None):
    """PURE. The price a RAISE paid inside the last `window_s` (default
    `REBID_WAIT_S`), or None — the floor below which the decay arm may not move
    the standing bid yet.

    A rescue / re-bid rung buys the WARM box by paying more for it; the decay arm
    reads the same receded floor a few polls later and hands the money straight
    back, and the box is displaced at the price the rung had just outgrown. The
    rung's own `REBID_WAIT_S` is how long it is given to work, so that is how long
    it owns the price.

    Evidence is `bid_history` — ladder_core's `[ts_first, price, machine_id,
    ts_last]` entries, recorded per tick from the OBSERVED standing bid for the
    self-floor guard, so every rung is already in it without a bookkeeping
    obligation. A raise is a price strictly above the entry before it; `ts_first`
    is when it was first seen standing. No history (or no clock) => None, i.e.
    the pre-hysteresis behaviour. One-directional by construction: the caller may
    only refuse a decay with this, never raise a bid or block a defend."""
    hist = s.get("bid_history") or ()
    now = s.get("now")
    if now is None or not hist:
        return None
    window = REBID_WAIT_S if window_s is None else window_s
    mid = s.get("machine_id")
    seen = []
    for e in hist:
        try:
            ts, price = float(e[0]), float(e[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if price <= 0:
            continue
        # a replacement box inherits the watch state but none of the old chunk's
        # prices; a falsy machine_id on either side matches nothing
        if mid is not None:
            try:
                if str(e[2]) != str(mid):
                    continue
            except (IndexError, KeyError):
                continue
        seen.append((ts, price))
    seen.sort()
    hold = None
    for (_prev_ts, prev_price), (ts, price) in zip(seen, seen[1:]):
        if price > prev_price + _CEIL_EPS and float(now) - ts < window:
            hold = price if hold is None else max(hold, price)
    return hold


def _bid_action(s):
    """PURE bid-movement sub-decision (SPOT_DESIGN §3.2). Returns a `raise_bid`
    (proactive defend, box live), a `lower_bid` (proactive DECAY, box live — floor
    receded continuously for BID_DECAY_S), or a `rescue_bid` (reactive, box
    outbid/stopped) Action, or None. Money-moving, so poll() reaches here only after
    the terminal/operator/spend-cap guards. Disabled when the market read failed
    (market_min_bid None) or the standing bid is unknown (last_bid None — a legacy
    run with no captured bid relaunches instead). Never targets over max_bid; the
    1-cent floor and the 60s rate-limit gate every move; a rescue never re-fires
    within one eviction cycle (rescue_attempted), so a timed-out rescue-wait falls
    through to relaunch. defend_at (state, default DEFEND_AT) is the run's spot:
    defend_at override (SPOT_DESIGN §3.4).

    The decay arm additionally answers to `_recent_raise_hold`: a price a rung
    paid inside the last REBID_WAIT_S may not be given back yet. That guard can
    only SUPPRESS a decay — the raise and rescue arms never read it.

    `notify_min_bid` (S2b, NOTIFY_DESIGN §6.4) touches ONE arm — the rescue
    quote — and only ever RAISES the floor that arm prices off. It exists
    because a just-taken machine typically lists nothing, so `market_min_bid` is
    None at exactly the moment the rescue needs a number, while the notification
    carries the price that actually won. The quote goes through
    `notify_rescue_bound`, which puts it under the SAME ceiling and the SAME
    affordability floor `rebid_ladder`'s rungs answer to and then through
    `_bid_target`, so a row cannot buy a bid the rails would have refused; when
    they refuse, the answer stays escalation and not a bigger number. Nothing
    else moves: the defend and decay arms never see it, and the one-shot
    `rescue_attempted` latch and the rate limit gate it exactly as they gate
    every other rung. With no row, every line below is the pre-S2b function.

    Note the seam precisely (review round 1, M3): the bound applies to the quote
    a NOTIFICATION raised, and to nothing else. A rescue priced off a readable
    market floor — with or without a row that the floor already dominates — is
    byte-for-byte its pre-S2b self, anchor-free ceiling and all. That hole
    PREDATES S2b and closing it is an owner question, not a review finding; see
    NOTIFY_DESIGN §6.7."""
    mmb = s.get("market_min_bid")
    target = _bid_target(mmb, s.get("max_bid"), s.get("on_demand"))
    nrescue = notify_rescue_bound(s)
    rescue_target = target if nrescue.floor is None else nrescue.price
    if target is None and rescue_target is None:
        return None
    last_bid = s.get("last_bid")
    if last_bid is None:
        return None
    if s.get("now", 0.0) - s.get("last_bid_put_ts", 0.0) < BID_RATE_LIMIT_S:
        return None                                   # vast rate-limit guard
    if bool(s.get("present")) and s.get("actual_status") in LIVE_STATES:
        if target is None:
            return None       # live box, no market read: the pre-S2b answer.
                              # The notify floor is a RESCUE input only, and a
                              # live box is not in a rescue.
        defend_at = s.get("defend_at")
        defend_at = DEFEND_AT if defend_at is None else defend_at
        if mmb >= defend_at * last_bid and target - last_bid >= BID_MIN_STEP:
            return Action("raise_bid", f"defend:{target}")
        # bid decay: the floor receded and stayed down for BID_DECAY_S ->
        # LOWER the standing bid to BID_TARGET_MULT x the current floor (ratchet fix).
        if _decay_dwell_satisfied(s) and target + BID_MIN_STEP < last_bid:
            hold = _recent_raise_hold(s)
            if hold is not None and target < hold - _CEIL_EPS:
                return None      # hysteresis: a rung just paid this price to keep
                                 # the box; decaying under it re-opens the eviction
            return Action("lower_bid", f"decay:{target}")
        return None
    if (s.get("present") and not s.get("rescue_attempted")
            and rescue_target is not None and rescue_target > last_bid):
        return Action("rescue_bid", f"rescue:{rescue_target}")
    return None


def _default_max_bid(floor_samples, first_seen_dph, on_demand=None,
                     strict_ceiling=False, ondemand_frac=BID_CEILING_ONDEMAND_FRAC,
                     mult=BID_MAX_MULT):
    """PURE. The DEFAULT defend/rescue bid ceiling when the operator gave no
    --max-bid (SPOT_DESIGN §3.2; AUTOBID_DESIGN).

    PRIMARY — on-demand-anchored (ratchet-proof by construction: the on-demand
    LIST price does not spike like the floor):
      * default "get-and-hold" policy -> the hard cap is just below on-demand
        (`on_demand - BID_ONDEMAND_EPS`); we will pay up to (never reaching) on-demand
        to hold the box. `ondemand_frac x on_demand` is the ADVISORY "preferred
        ceiling" (the handoff trigger), NOT the hard cap.
      * --strict-ceiling policy -> the hard cap IS `ondemand_frac x on_demand`
        (the box is allowed to terminate above it).

    FALLBACK (on-demand unreadable) — preserve the 2026-07-12 J1 anti-ratchet
    exactly: a rolling MEDIAN of the observed floor x `mult` (median so a single
    transient floor spike can't anchor the cap high; the old 1.25x-first-seen-dph
    default latched a spiked bid for the whole run). first-seen dph is the deepest
    fallback, used until any floor read exists. None when nothing is available
    (leaves both bid moves disabled rather than guessing)."""
    if on_demand and on_demand > 0:
        if strict_ceiling:
            return round(ondemand_frac * on_demand, 3)
        return round(on_demand - BID_ONDEMAND_EPS, 3)
    fs = [f for f in (floor_samples or []) if isinstance(f, (int, float)) and f > 0]
    if fs:
        return round(mult * statistics.median(fs), 3)
    if first_seen_dph:
        return round(BID_FALLBACK_DPH_MULT * first_seen_dph, 3)
    return None


# --------------------------------------------------------------------------- #
# eviction classification + automatic-replacement decision (owner directive
# 2026-08-05, after the v7 training run needed TWO hand-rescues in one night —
# docs/plans/witness/g2_push/V7_TRAIN_RUN_2026-08-05.md).
#
# The bid ladder above can only move MONEY on the box it already has. Two of the
# three ways a spot box dies cannot be answered that way at all:
#
#   * ON-DEMAND DISPLACEMENT — an on-demand claim outranks EVERY interruptible
#     bid. v7 eviction 2 lost a box at a $1.05 bid whose on-demand rate was
#     $1.0017: we were bidding ABOVE on-demand and were still displaced. No
#     price wins that; only a different box does.
#   * HOST FAILURE — the instance left the listing entirely. Nothing to bid on.
#
# and the third (a genuine OUTBID) is only winnable while the post-spike floor
# still sits under the machine's on-demand price. v7 eviction 1 failed that test
# too: the floor rose $0.7599 -> $0.9099 on a machine whose on-demand list price
# was ~$0.748, so `_bid_target` correctly returned None ("unwinnable floor") and
# the ladder went straight to `unrecoverable`.
#
# So the ladder needs a rung BELOW the bid: rent a replacement. That is
# autonomous spend, so every gate here is a REFUSAL by default — no budget cap,
# no launch-price anchor, or no readable market means STOP, never "spend and
# hope". The impure half (offer search, launch, ticket retarget, destroy) lives
# in herdd; this decides only whether and at what price.
# --------------------------------------------------------------------------- #
EVICTION_OUTBID = "outbid"                 # bid box stopped, market floor > our bid
EVICTION_ONDEMAND = "ondemand_displaced"   # an on-demand renter claimed the machine
EVICTION_HOST_FAILURE = "host_failure"     # instance gone from the API listing
EVICTION_HOST_STOP = "host_stop"           # present, chunk still LISTED and rentable,
                                # our standing bid still clears the live floor — so
                                # nobody took this box, it simply stopped. Added
                                # 2026-08-09 from box 47226953: min_bid $0.3333
                                # against our standing $0.667, `avail: yes`, and the
                                # box `exited` anyway under a live claimed ticket.
                                # The pre-existing code answered `ondemand_displaced`
                                # there, purely because an on-demand price existed —
                                # which routes straight past the re-bid ladder
                                # ("no rung can win this back") to renting a cold
                                # replacement, when the warm box was one `start`
                                # away. See `resume_in_place`.
EVICTION_NO_CREDIT = "no_credit"           # the ACCOUNT cannot pay: vast stopped the
                                # box and every relaunch returns HTTP 400
                                # `insufficient_credit`. Added 2026-08-25 after a
                                # fleet-wide stop (7 boxes, 4 card classes, 3
                                # lanes, minutes apart) was journaled `outbid` —
                                # the on-demand boxes had no bid to lose, and the
                                # alarm sent the operator to the bid ladder
                                # ("no standing bid to raise") instead of the
                                # billing page. No bid, resume or replacement can
                                # undo it; only a top-up can.
EVICTION_UNKNOWN = "unknown"               # stopped, but no market read to judge by

# Marker vast returns on every priced call once the account is empty.
CREDIT_EXHAUSTED_MARKER = "insufficient_credit"


def credit_ok_from_error(err):
    """PURE tri-state. Read an account-credit verdict off a driver error string.

    `False` ONLY on vast's own `insufficient_credit` marker; `None` for anything
    else, including no error at all. Never `True`: the absence of that marker is
    not evidence the account is funded, and a classifier that treated it as such
    would assert solvency it never observed."""
    return False if (err and CREDIT_EXHAUSTED_MARKER in str(err)) else None

# Replacement-policy defaults (all overridable per watch — see
# herdd.job_supervise_init / `herdd fleet watch --max-replacements`).
MAX_REPLACEMENTS = 3            # autonomous rentals per watch, then STOP + alarm
REPLACE_CEILING_MULT = 2.0      # price ceiling = this x the ORIGINAL launch dph.
                                # 2x covers the observed contested-market spread
                                # (v7: $0.76 spot -> $2.21 on-demand was 2.9x and
                                # is deliberately NOT auto-approved) while making
                                # a runaway 10x market impossible to buy into.
REPLACE_ESCALATION_CAP_MULT = 4.0   # ABSOLUTE cap on a market-re-derived
                                # replacement ceiling, as a multiple of the
                                # ORIGINAL launch anchor. REPLACE_CEILING_MULT
                                # bounds the ceiling against a market that has
                                # not moved; this one bounds how far live market
                                # evidence may push it when the market HAS moved
                                # (2026-08-24 wedge: the $0.387 ceiling sat 3.4%
                                # under the only qualifying offer and the lane
                                # looped 36 ticks). 4x is what the operator
                                # re-armed by hand that morning; past it the
                                # answer is a human, not a bigger number.
REPLACE_MIN_RUNTIME_H = 0.25    # refuse a rental the remaining budget can only
                                # afford for less than this (15 min) — renting a
                                # box that budget-parks before it finishes booting
                                # spends money for zero progress.
REPLACE_MIN_CUSHION = BID_MIN_CUSHION_MULT   # a spot replacement's bid must clear
                                # the floor by at least this multiple. BOUND to the
                                # launch-side survival rail (2026-08-08) rather than
                                # repeated as a second 1.10: they are the same
                                # policy seen from two sides, and a drift between
                                # them would let the ladder rent a bid the target
                                # function is not allowed to place. Since
                                # `_bid_target` now RAISES to the cushion whenever
                                # on-demand leaves room, `thin` fires only in the
                                # genuinely-tight case (floor within ~10% of
                                # on-demand) — which is the correct trigger for the
                                # on-demand rung. Historically it fired on the
                                # v7-eviction-1 shape: a $0.747 bid against a
                                # $0.746 floor, "cushioned" 1.2x on paper and by
                                # $0.001 in fact.
SPOT_SETUP_H = 0.193            # boot -> productive overhead per spot cycle, in
                                # hours. MEASURED 11m35s on the v11 eval lane
                                # 2026-08-06 (venv, base pull, merge). It is the
                                # `setup_h` term in `spot_breakeven` and the unit
                                # the re-bid ladder's wall budget is denominated
                                # in. A lane with a materially different boot
                                # cost should override it per watch
                                # (JOB_SPOT_SETUP_H) rather than have this
                                # number quietly stand in for it.
SPOT_PRIOR_BAND_TOP_FRAC = 0.70   # top of the owner-ratified 55-70% bid/on-demand
                                # band (AUTOBID_AUDIT §4b, §9 question 1).
SPOT_PRIOR_LIFETIME_H = round(SPOT_SETUP_H / (1.0 - SPOT_PRIOR_BAND_TOP_FRAC), 3)
                                # = 0.643 h (38.6 min). The assumed spot lifetime
                                # used by `replacement_decision`'s breakeven rung
                                # when this lane has NO observed one — i.e. on the
                                # FIRST eviction, which is the only eviction most
                                # watches ever have.
                                #
                                # Why a prior at all (recalibration 2026-08-09,
                                # item B): `_job_observed_lifetime_h` is built from
                                # DEAD BID REPLACEMENTS, so it is None until the
                                # ladder has already rented and lost a spot box.
                                # The livelock trigger therefore could not fire at
                                # the moment it was most needed — the 2026-08-08
                                # NO-GO fired in a human's head instead.
                                #
                                # Why THIS number, and not the measured median.
                                # The realised spot lifetimes we have recorded are
                                # brutally short: 46935445 <1 min, 46934302 11 min,
                                # the v11 chat arm 11-13 min x4, the v11 resume box
                                # 13.8 min, 46880245 outbid before it finished
                                # booting, 46909754 45 min, 47214941 54 min, and
                                # ONE multi-hour survival (46936034, 2h29m). The
                                # median of those eleven is ~0.20 h — at or below
                                # SPOT_SETUP_H (0.193 h), which `spot_breakeven`
                                # scores as the LIVELOCK shape at any price ratio.
                                # A prior set from the median would therefore route
                                # every first eviction to the on-demand rung, which
                                # is a fleet-wide spend decision (1.5-3x per hour)
                                # and not ours to make from a prior.
                                #
                                # So the prior is derived from what the POLICY
                                # already asserts rather than from the observed
                                # tail. Ratifying a 55-70% bid/on-demand band is a
                                # statement that a box is expected to live long
                                # enough for that band to pay: `L_min =
                                # setup / (1 - b/od)`, so the band's top implies
                                # 0.193/0.30 = 0.643 h. Using a LONGER prior would
                                # assume more than the policy asserts; a SHORTER
                                # one would escalate machines the ratified cost cap
                                # has already declared acceptable.
                                #
                                # The consequence is the clean statement of the
                                # rule: the prior-based trigger fires exactly when
                                # the spot bid we would place exceeds the top of the
                                # ratified band (1 - 0.193/0.643 = 0.6998 ~= 0.70).
                                # With item A's hard ceiling holding every bid at or
                                # under 0.75 x on-demand, its live domain is the
                                # narrow band 0.70-0.75 x on-demand — exactly the
                                # tight machines where the SURVIVAL CUSHION has
                                # raised the bid above the cost cap. Below the band
                                # it never fires; above 0.75 item A has already
                                # escalated. The two rails are complements, not
                                # overlaps.
                                #
                                # `observed_lifetime_h` ALWAYS wins when it exists.
                                # 0 or None disables the prior and restores the
                                # pre-2026-08-09 "cannot fire without history".
SPOT_FASTDEATH_S = 900          # a replacement that dies inside this window counts
                                # as a "fast death" (the market is hostile, not
                                # unlucky)
MAX_SPOT_FASTDEATHS = 2         # this many fast deaths flips the ladder to
                                # on-demand for the remaining replacements

# --- retention of the LOST box (owner directive 2026-08-05) ----------------- #
# "don't have fleetd destroy the box immediately please. we should eat a few
# hours of parked host time just in case we have bugs that lose data. would be
# helpful to get confidence on first and it's quite cheap to park for a few
# hours."
#
# The lost box's local disk can hold state that never reached B2: checkpoint
# sync is PERIODIC, so up to one interval of training progress plus any
# non-checkpoint artifact (logs, eval outputs, a half-written adapter) may exist
# only there. Until automatic replacement has earned trust, that disk is
# evidence and a recovery source, so the replacement flow RETAINS it for a
# bounded window instead of destroying it.
REPLACEMENT_RETENTION_H = 3.0   # hours to hold the evicted box. 0 = destroy
                                # immediately (the pre-retention behavior).
RETENTION_BACKSTOP_GRACE_H = 1.0  # past deadline + this, the supervising ladder
                                # destroys the box itself if `herdd reap` has
                                # not (see herdd._job_retention_sweep). The
                                # PRIMARY expiry mechanism is the self-expiring
                                # keep label + the reaper timer; this only
                                # covers a workstation whose reap timer is off.
# Measured 2026-07-30 across the fleet: a stopped box bills $2.13-$4.62/day for
# its ALLOCATED disk. Used only when the instance body carries no
# `storage_total_cost` — a real per-box number always wins.
STORAGE_DPD_OBSERVED = (2.13, 4.62)
# The bid we PIN a RETAINED box to, and the reason retention may quote a
# storage-only price at all. Same value and same physics as HANDOFF_PARK_BID
# (cmd_bid's [0.001, 32] minimum, guaranteed below any live market floor); named
# separately because the two callers are unrelated lanes and a future change to
# one must not silently move the other.
#
# INCIDENT 2026-08-16, box 47833510 (session doc:
# docs/plans/vast-fleet-sessions/2026-08-16-replacement-probe-tpd/SESSION.md).
# The eviction ladder retained an evicted SPOT box, left its standing bid at the
# $1.20 the rescue rung had just raised it to, and left the two `start` calls
# vast had answered "state change queued" outstanding. ~65 min later machine
# 34985 had capacity again and vast honoured the queued start: the box came back
# RUNNING with no jobs, no watch and no alarm, billing the GPU rate ($0.8407/hr,
# ~20x the storage-only figure retention had disclosed). Retention's whole cost
# story assumes a STOPPED box, so the retain path now makes that true — it
# `stop`s the box (cancelling any queued start) and pins the bid here.
RETENTION_PARK_BID = HANDOFF_PARK_BID

Replacement = namedtuple("Replacement",
                         ["action", "rental", "price", "reason", "ceiling",
                          "budget_left",
                          # Which lifetime the breakeven rung was scored on:
                          # "observed" (this lane's own dead bid replacements),
                          # "prior" (SPOT_PRIOR_LIFETIME_H — no history yet), or
                          # None (the rung never ran). Appended with a default so
                          # every existing positional construction and every
                          # `dec.reason`-style consumer is untouched; a decision
                          # that cannot say whether it ran on evidence or on an
                          # assumption is not an auditable one.
                          "lifetime_basis"])
Replacement.__new__.__defaults__ = (None,)
Retention = namedtuple("Retention",
                       ["action", "deadline_ts", "cost_usd", "cost_hi_usd",
                        "reason"])


def market_floor_self_match(market_min_bid, standing_bid, *, bid_history=(),
                            now=None, lag_s=BID_SELF_FLOOR_LAG_S,
                            eps=BID_SELF_FLOOR_EPS):
    """PURE. Is this "market floor" actually OUR OWN bid read back — the one we
    hold now, or one we held recently enough that the chunk is still echoing it?

    Returns `SelfFloor(kind, price, age_s)` with kind in {"standing", "prior"},
    or None when the read is a genuine market observation.

    See BID_SELF_FLOOR_EPS for the incident and the arithmetic, and
    BID_SELF_FLOOR_LAG_S for why the CURRENT bid is not enough. In one line:
    the chunk's `min_bid` does not follow our bid moves, so from the moment a
    defend/decay/rescue rung moves the price, "floor == our standing bid" is
    false while "floor == a bid we were paying two minutes ago" is true — and
    the tick treats that stale echo as a competing bidder. Two live shapes
    ratchet through the gap, both off our own money:

      * DECAY then DEFEND. The floor recedes, decay lowers us to L, the next
        read echoes the OLD bid B > L, `mmb >= defend_at x last_bid` passes
        trivially, and the defend target is `mult x B` — a raise ABOVE the bid
        the decay just left, priced entirely off ourselves.
      * ANY SUB-11% RAISE. After a raise from B to B', the stale echo B still
        satisfies `B >= 0.9 x B'` whenever `B'/B < 1/defend_at` — which the
        1.10 survival cushion always is — so the defend re-fires on the echo
        every poll until the window closes.

    `bid_history` is the recent standing-bid series for THIS machine, newest or
    oldest first (order is irrelevant), as `(ts, price)` or `(ts, price, ...)`
    tuples; the caller is responsible for only passing entries observed on the
    machine currently held (a replacement box inherits no echoes). Entries
    older than `lag_s` are ignored, so a caller that never trims still gets the
    right answer. `now=None` disables the age test and matches on price alone —
    the conservative reading, used where no clock is threaded through.

    Callers must additionally gate on *being the tenant* — LIVE and `is_bid` —
    because the same equality on a STOPPED box means the opposite thing
    (somebody else now holds the chunk at a price that happens to match what we
    were paying, which is a real market read and the rescue ladder's whole
    input).

    Deliberately EXACT (within `eps`), never `>=`: a floor strictly above our
    bid is the genuine competing-bidder signal that `classify_eviction` keys on,
    and swallowing it would trade a money bug for an availability bug.

    The comparisons carry 1e-9 of absolute slop on top of `eps` because the
    worst case sits exactly ON the eps: the echo is 3-decimal-quantized (probe
    v2, 2026-08-10) and a 4-decimal bid at the half-grid point (0.0335 vs its
    echo 0.034) differs by 0.0005 in decimal but by 0.0005000000000000004 in
    binary — without the slop the guard is blind to it by 4e-19 of float
    noise. 1e-9 is orders of magnitude below any price granularity."""
    try:
        mmb = float(market_min_bid)
    except (TypeError, ValueError):
        return None
    eps = float(eps) + 1e-9
    try:
        if abs(mmb - float(standing_bid)) <= eps:
            return SelfFloor("standing", float(standing_bid), 0.0)
    except (TypeError, ValueError):
        pass
    best = None
    for e in bid_history or ():
        try:
            ts, price = float(e[0]), float(e[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if abs(mmb - price) > eps:
            continue
        # The echo window runs from when the price STOPPED being our standing
        # bid — entry field 3 (`ts_last`, refreshed each tick the price still
        # stands) when present; a legacy 3-field entry falls back to its only
        # ts. A negative age (wall clock stepped backwards under a persisted
        # ts) clamps to 0 and MATCHES: a future-dated entry is still a bid of
        # ours, and failing open here re-arms the self-echo ratchet.
        try:
            ts_last = float(e[3])
        except (TypeError, ValueError, IndexError, KeyError):
            ts_last = ts
        age = None if now is None else float(now) - ts_last
        if age is not None and age < 0:
            age = 0.0
        if age is not None and age > lag_s:
            continue                      # outside the echo window
        # newest match wins: it is the one that best explains the echo
        if best is None or (age is not None and best.age_s is not None
                            and age < best.age_s):
            best = SelfFloor("prior", price, age)
    return best


def market_floor_is_self(market_min_bid, standing_bid, *,
                         bid_history=(), now=None, lag_s=BID_SELF_FLOOR_LAG_S,
                         eps=BID_SELF_FLOOR_EPS):
    """PURE, boolean face of `market_floor_self_match` — kept because every
    pre-2026-08-09 caller and test asks the yes/no question. With no
    `bid_history` it is exactly the original current-bid-only test."""
    return market_floor_self_match(market_min_bid, standing_bid,
                                   bid_history=bid_history, now=now,
                                   lag_s=lag_s, eps=eps) is not None


def notify_price(value):
    """PURE. A price off the notification feed, or None.

    THE ONE NORMALIZATION SEAM for notification-carried money (S2b review round
    1, F6/m4). The inbox is a hidden endpoint written by a service we do not
    control and parsed through `json.loads`, which happily mints `inf` from the
    string `"1e309"` and `nan` from `NaN` — both of which sail through a bare
    `float()` and then through every `>` comparison the classifier and the
    rescue quote are built out of. A price is usable here only if it parses to a
    FINITE float STRICTLY ABOVE zero:

      * `0.0` — a zeroed `your_bid` would disable the only conflation guard the
        row carries (`new_min_bid > your_bid`), turning every displacement into
        a supported outbid;
      * negative — not a price;
      * `inf` / `"1e309"` — the m4 `rescue:inf` cell: an unbounded proposed
        floor, which is exactly the number no rail can clamp meaningfully;
      * `nan` — every comparison against it is False, so it fails closed
        already, but it must fail closed by RULE and not by accident.

    Everything downstream (`notify_outbid_supported`, `notify_rescue_floor`)
    reads its prices through here, so "what counts as a price" is one function
    and not four spellings of `float()`."""
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p) or p <= 0:
        return None
    return p


def notify_outbid_supported(notify, on_demand=None, *, last_bid=None):
    """PURE. Does this matched notification row support `EVICTION_OUTBID` in
    OUR vocabulary? (NOTIFY_DESIGN §6.1 — the one semantic trap.)

    Vast's `outbid` notification fires on any bid-instance displacement. Our
    class vocabulary is narrower on purpose, because ON-DEMAND DISPLACEMENT is
    the class no bid can undo and it gates the expensive replacement rung. The
    captured feed holds instance 47840057 displaced with `your_bid 0.16,
    new_min_bid 0.15` — a price BELOW our own bid, which is what an on-demand
    taker or a host action looks like from here. So the row-internal test is
    the one the price can carry: a displacing price ABOVE our bid is a genuine
    higher bidder; anything else is displacement of UNKNOWN class, and the
    caller must not mint a class it cannot defend.

    BOTH halves of §6.1, since the review (round 1, F1/m5). The `< on_demand`
    clause was specified and then dropped in the first cut, and it is not
    cosmetic: a displacing price at or ABOVE the machine's on-demand rate is not
    a genuine higher BIDDER at all. Nobody rationally bids past on-demand — the
    on-demand claim outranks every interruptible bid at any price (SPOT_DESIGN
    #6), which is the whole reason `_bid_target` clamps under it and
    `effective_bid_ceiling` exists. So such a row is displacement of unknown
    class, exactly like a below-bid one, and it must not mint `outbid`: that
    class shortens the evicted-machine exclusion TTL (`EVICTED_TTL_CLASSES`) on
    a machine an on-demand taker may be sitting on. `on_demand=None` is the
    documented degradation — a caller with no on-demand read gets the
    row-internal test alone, which is what shipped.

    `last_bid` (2026-08-26) is the THIRD clause, and it is the one the field
    asked for: the displacing price must also beat the bid WE hold, not merely
    the `your_bid` the row remembers. The row records the bid vast saw at the
    moment of displacement, so after any rung raises us it is stale by
    construction — measured over one month of journal, 15 matched outbid rows
    carry a `new_min_bid` BELOW our standing bid at that instant (e.g. a row
    quoting $0.48 -> $0.60 against a bid of $0.724). Each of those minted
    `outbid` off the row-internal test alone, which is how an eviction gets
    labelled `outbid` at a market floor arithmetically below our own bid.
    Nobody paid more than we bid => not an outbid in our vocabulary. `None`
    (the caller holds no bid, or does not say) keeps the two-clause predicate
    every pre-2026-08-26 caller had.

    **What this predicate STILL does NOT do**: reconcile the row's `your_bid`
    against our belief. That comparison is the stale-belief class §6.3 closed
    (two writers of one number); this clause tests the row's PRICE against ours,
    which is the same question the `rescue_target > last_bid` money refusal
    already asks — see NOTIFY_DESIGN §6.7.

    Missing or unreadable prices are False, not True: this predicate exists to
    ADD a verdict, and a verdict added from an absent number is the shape the
    whole notification channel was supposed to end."""
    if not isinstance(notify, dict):
        return False
    yb = notify_price(notify.get("your_bid"))
    nmb = notify_price(notify.get("new_min_bid"))
    if yb is None or nmb is None:
        return False
    if nmb <= yb:
        return False
    lb = notify_price(last_bid)
    if lb is not None and nmb <= lb:
        return False
    od = notify_price(on_demand)
    return od is None or nmb < od


def notify_rescue_floor(notify_min_bid, market_min_bid=None):
    """PURE. The floor a matched outbid row licenses the RESCUE rung to price
    off (NOTIFY_DESIGN §6.4): the displacing price plus one CENT, and never
    BELOW the floor the market read already gave us.

    Returns None when there is no row — which is what makes every pre-S2b call
    of `_bid_action` byte-identical: the rescue arm then prices off exactly the
    market floor it always did. Also None for any price `notify_price` refuses
    (zero, negative, infinite, unparseable), because an unbounded proposed floor
    is not a conservative input to a clamp — it is the m4 `rescue:inf` cell.

    "One grid step" is BID_MIN_STEP — a CENT, ten units of the $0.001 price grid
    (m6). Deliberate and conservative: the quote opens a cent above the price
    that actually won rather than a rounding unit above it, which is the same
    minimum-material-move constant every other rung uses.

    This is a PROPOSED price, not an emitted one. It is handed to `_bid_target`
    like any other floor, so the preference, the cost cap, the survival cushion
    and the `BID_CEILING_ONDEMAND_FRAC` hard clamp all bind untouched. A
    notification may propose a price; it may never widen what the rails emit."""
    nmb = notify_price(notify_min_bid)
    if nmb is None:
        return None
    mmb = notify_price(market_min_bid) or 0.0
    return max(mmb, nmb + BID_MIN_STEP)


def floor_rise_corroborated(market_min_bid, last_bid, *, floor_samples=(),
                            notify=None):
    """PURE, tri-state. Is a floor read ABOVE our standing bid backed by a SECOND
    observation? True / False / None (nothing to corroborate WITH).

    A single offers read is one sample of a multi-chunk machine, and a sibling
    chunk's price lands in it. Measured 2026-08-26 03:04:06Z: an eviction
    recorded `market_min_bid $0.407` against our $0.24 and was classified
    `outbid`, while the read 54 s later said $0.20 and the box came back at
    03:06:11 — nobody had taken it. On that machine the self-floor guard's
    `surviving_floor` path fired 46 times that night, which is exactly the shape
    that puts another chunk's price in this slot.

    Corroboration is any OTHER observation that also sits above our bid: one
    occurrence of the read under test is removed from `floor_samples` first, so
    a lone spike cannot vouch for itself, and a genuinely risen floor read twice
    still corroborates. A matched notification's displacing price can only
    CORROBORATE — a below-our-bid row is displacement of unknown class (§6.1) and
    says nothing about the listing, so it must never refute a floor read.

    `False` is a REFUSAL to mint the class, never evidence of the opposite;
    `None` (no other observations at all) leaves the caller's answer alone, so
    every caller that passes no samples is bit-identical to before."""
    lb = notify_price(last_bid)
    if lb is None:
        return None
    nmb = notify_price((notify or {}).get("new_min_bid")
                       if isinstance(notify, dict) else None)
    if nmb is not None and nmb > lb:
        return True                          # a price somebody actually paid
    others = [p for p in (notify_price(x) for x in (floor_samples or ()))
              if p is not None]
    mmb = notify_price(market_min_bid)
    if mmb is not None:                      # drop ONE copy of the sample itself
        for i, p in enumerate(others):
            if abs(p - mmb) <= _CEIL_EPS:
                others.pop(i)
                break
    if not others:
        return None
    return any(p > lb for p in others)


def classify_eviction(*, present, actual_status=None, market_min_bid=None,
                      on_demand=None, last_bid=None, market_listed=None,
                      is_bid=None, notify=None, account_credit_ok=None,
                      floor_samples=()):
    """PURE. Why did this supervised bid box stop? Returns one of the
    `EVICTION_*` constants. Callers reach here only AFTER the terminal/operator
    guards (self-park on drain and an operator stop are classified upstream by
    the box-event stream, SPOT_DESIGN §3.6 — this function never sees them).

    The discriminator that matters is ON-DEMAND DISPLACEMENT, because it is the
    one class no bid can undo: if our standing bid was already at or above the
    machine's on-demand price and we still lost the box, the winner was an
    on-demand renter, whose claim outranks every interruptible bid regardless of
    price. Reporting that as "outbid" is what made the v7 run look like a
    bidding problem when it was a supply problem.

    A missing market read is `unknown`, never `outbid` — the replacement
    decision treats the two identically today, but an eviction event that
    silently asserts a cause it could not observe is exactly the log line that
    sends the next postmortem the wrong way.

    `market_listed` (added 2026-08-09, defect D7 / incident task #74) splits
    that "missing read" in two, which is the whole reason this function had
    **never once returned `outbid` in production** (AUTOBID_AUDIT_2026-08-08 §4:
    0 of 15 decisions, while 46880245 / 46909754 / 46848347 / 47214941 were all
    demonstrably outbid). `market_min_bid` comes from a `rentable: {eq: True}`
    bid-offer query on our machine, and a machine that has just been TAKEN lists
    none — so the read is `None` at exactly the moment `mmb > lb` would have
    been true, and the `mmb is None` arm below fired instead. Tri-state:

      * `True`  — the offers API answered and this machine still lists rentable
                  bid offers (so `market_min_bid` is a real number);
      * `False` — the offers API answered and this machine lists NONE. That is
                  EVIDENCE of displacement, not ignorance: the chunk we held is
                  no longer purchasable, which is what being outbid looks like
                  from outside;
      * `None`  — nobody asked, or the read failed. Ignorance, and it still maps
                  to `unknown`. Rule 1 of SPOT_DESIGN §5 (transient != eviction)
                  is preserved by construction: this argument can only be
                  `False` on a request that SUCCEEDED.

    `is_bid` (added 2026-08-16, replacement-probe incident) makes the
    ONDEMAND-DISPLACEMENT arm unreachable on a box that is not a bid rental.
    An on-demand renter cannot displace an on-demand renter: the claim we hold
    is the same class as the one that would have to outrank it. On 2026-08-16
    the ladder nevertheless journaled `ondemand_displaced` with `is_bid: false`
    — off a STALE `last_bid` left on the watch by a previous bid box — and that
    class is the single strongest input to `replacement_decision`'s expensive
    rung ("the eviction was an on-demand claim, which outranks any bid"), so a
    misfire here buys an on-demand replacement on evidence that does not exist.
    Tri-state, and only `False` gates: `None` means nobody said, and every
    pre-2026-08-16 caller keeps its exact behaviour.

    `notify` (added 2026-08-16, NOTIFY_DESIGN S2b) is vast's OWN record that
    this instance was displaced: `{your_bid, new_min_bid, created_at}`, already
    matched to this box and this eviction cycle by the driver (§6.3 — matching
    is not this function's job, and a row matched by machine or by nothing would
    be worse than no row at all). `None` for every pre-S2b caller, and every
    `None` path through this function is the pre-S2b function exactly.

    WHERE IT SITS AND WHY. Below the risen-floor arm and below the on-demand
    arm, above every market-LISTING arm. It cannot reorder the two arms above
    it, because vast's `outbid` conflates displacement classes (see
    `notify_outbid_supported`) and the on-demand discriminator is the one class
    no bid can undo. It outranks the arms below it because those are inferences
    drawn from a listing — "the machine stopped listing" and "the machine still
    lists below our bid" — and this is the control-plane record of the
    displacement itself. That precedence is the 2026-08-16 06:24Z case: the
    inbox said outbid at $1.00 against our $0.45, and seventeen seconds later
    the offers read said `listed=True, min_bid=0.14` and this function returned
    `host_stop`. Only one of those two numbers is a price somebody paid.

    **NOBODY PAID MORE THAN WE BID => NOT AN OUTBID** (owner, 2026-08-26). Every
    arm that can return `outbid` now answers to a price above our own: the
    risen-floor arm always did, the notify arm does since `last_bid` gates
    `notify_outbid_supported`, and the listing arm has no price at all. The class
    steers the recovery ladder — `outbid` buys re-bid rungs, which is money spent
    on a host that is stopping boxes — so a class minted against the arithmetic
    is worse than no class.

    `floor_samples` (same date) are the machine's other recent floor reads, for
    the risen-floor arm's corroboration test; empty is the documented default and
    changes nothing. See `floor_rise_corroborated`."""
    if not present:
        return EVICTION_HOST_FAILURE
    if (actual_status or "").lower() in LIVE_STATES:
        return EVICTION_UNKNOWN            # not actually down; caller debounces
    # ACCOUNT INSOLVENCY outranks every arm below, because those all infer a
    # cause from the MARKET and this is a control-plane fact about us: vast
    # stops instances an empty account cannot pay for, whatever the floor is
    # doing. Its market shadow is indistinguishable from displacement (the
    # machine stops listing our chunk either way), so the arms below answer
    # `outbid` — which is how a 2026-08-25 fleet-wide stop sent the operator to
    # the bid ladder instead of the billing page. Tri-state, and only an
    # OBSERVED refusal gates: `None` (nobody asked) leaves every pre-2026-08-25
    # caller bit-identical, and `True` is never asserted by
    # `credit_ok_from_error` because not seeing the marker is not solvency.
    if account_credit_ok is False:
        return EVICTION_NO_CREDIT
    od = on_demand if (on_demand and on_demand > 0) else None
    lb = last_bid if (last_bid and last_bid > 0) else None
    mmb = market_min_bid if (market_min_bid and market_min_bid > 0) else None
    # An ON-DEMAND box has no bid to lose and cannot be displaced by a renter of
    # its own class, so both arms that would return EVICTION_ONDEMAND are shut
    # off for it (see `is_bid` in the docstring).
    od_claim_possible = is_bid is not False
    # ORDER MATTERS. A RISEN FLOOR is direct evidence of another bidder, so it is
    # tested first. On v7 eviction 1 the floor went $0.7599 -> $0.9099 while our
    # bid sat on the on-demand clamp ($0.747 against that machine's $0.748
    # on-demand price) — the "our bid >= on-demand" test below is also true
    # there, and checking it first would have labelled a plain outbid an
    # on-demand claim and skipped straight to the expensive rung.
    if mmb is not None and lb is not None and mmb > lb:
        # ...but ONE sample of a multi-chunk machine can be a sibling chunk's
        # price. When other observations exist and every one of them sits at or
        # under our bid, the rise is uncorroborated and the conservative class
        # wins: both classes carry a comparable exclusion TTL, and calling a host
        # event a market event re-lands the fleet on a host that is stopping
        # boxes. No samples => nothing to corroborate with => unchanged.
        if floor_rise_corroborated(mmb, lb, floor_samples=floor_samples,
                                   notify=notify) is False:
            return EVICTION_HOST_STOP
        return EVICTION_OUTBID
    # Floor at/below our standing bid and we lost the box anyway. With our bid
    # already at or above the machine's on-demand price, the only renter who can
    # take it is an on-demand one — v7 eviction 2 exactly ($1.05 bid on a box
    # whose on-demand rate was $1.0017). No price wins that back.
    if (od_claim_possible and od is not None and lb is not None
            and lb >= od - BID_ONDEMAND_EPS):
        return EVICTION_ONDEMAND
    # VAST'S OWN RECORD of the displacement, at a price above our bid and below
    # this machine's on-demand rate (S2b; the on-demand half restored by review
    # round 1, F1). Strictly stronger than every listing-derived arm below,
    # strictly weaker than the two above — see the docstring. A row whose
    # displacing price is at or below our own bid, or at or above on-demand,
    # falls through untouched: it still is not `host_stop` evidence in our
    # vocabulary, but it is not an outbid either, and the arms below give it the
    # answer they always gave.
    # `last_bid` gates it since 2026-08-26: a row whose displacing price does not
    # beat the bid WE hold cannot mint `outbid` (the row's `your_bid` goes stale
    # the moment a rung raises us — see `notify_outbid_supported`). Withheld on a
    # non-bid box, where the standing bid is a stale leftover of a previous
    # rental and not a price we are paying.
    if notify_outbid_supported(notify, on_demand=od,
                               last_bid=lb if is_bid is not False else None):
        return EVICTION_OUTBID
    # The offers API ANSWERED and this machine lists no rentable bid offer at
    # all. Ordered after the on-demand test on purpose: an on-demand renter also
    # empties the bid listing, and "no bid can win this back" is the more
    # actionable of the two classes, so the stronger evidence keeps precedence.
    if mmb is None and market_listed is False:
        return EVICTION_OUTBID
    if mmb is None:
        return EVICTION_UNKNOWN
    # POSITIVE evidence that nobody took the box: the chunk is still listed and
    # rentable, and our standing bid still clears its floor. An on-demand renter
    # sitting on our GPUs does not leave them purchasable as spot.
    if market_listed is True and lb is not None and lb >= mmb:
        return EVICTION_HOST_STOP
    return (EVICTION_ONDEMAND if (od is not None and od_claim_possible)
            else EVICTION_UNKNOWN)


RESUME_MAX_TRIES = 2        # `start` attempts per eviction cycle before the ladder
                            # moves on to the bid rungs. Two, not one: vast refuses a
                            # start while another renter holds the GPUs, and that is
                            # a transient the next poll often clears. Reset on any
                            # return to live, like the re-bid rungs.

Resume = namedtuple("Resume", ["action", "reason"])


def resume_in_place(*, present, is_bid, market_min_bid=None, last_bid=None,
                    market_listed=None, tries_used=0,
                    max_tries=RESUME_MAX_TRIES, budget_usd=None,
                    spend_usd=0.0):
    """PURE. Should the ladder just `start` this stopped box, before it spends?

    Returns `Resume(action, reason)` with action "start" or "skip"; every refusal
    carries its arithmetic, same contract as `replacement_decision`.

    **Why this rung exists, and why it goes FIRST** (box 47226953, 2026-08-09).
    A budgeted jobs watch held a live claimed ticket on an RTX PRO 6000 that
    stopped on its own at ~01:31Z. There was no price cause at all: `min_bid`
    $0.3333 against our standing $0.667, `avail: yes`. The documented recovery —
    rent a replacement, retarget, destroy the husk — would have been badly wrong.
    The stopped box still held 59 GB of pulled weights from a 104 GiB
    base+merged pull, and our bid was already winning, so `herdd start`
    recovered it in ~40 s at zero re-pull cost against a replacement's measured
    11m35s of setup plus the whole pull again.

    So the ladder is now: **resume in place -> re-bid -> replace**, cheapest and
    most reversible first. A `start` spends nothing beyond what the watch already
    authorised (it is the box we are already renting) and is undone by the box
    simply not coming back, which is why it can run at the debounce point
    instead of waiting for the rescue deadline.

    Refusals, in order:

      1. not `present` — a box that has left the listing cannot be started; that
         is `EVICTION_HOST_FAILURE` and the replacement rung's job.
      2. `tries_used >= max_tries`.
      3. budget already consumed — a resume restarts the meter, so it is gated on
         the same cap every other rung is.
      4. **bid box, chunk NOT listed** (`market_listed is False`) — we were
         displaced; the machine is not purchasable and a start will not stick.
      5. **bid box, floor above our standing bid** — we are outbid, and the
         answer to that is a higher bid, not a start that vast will refuse.
      6. **bid box with no usable market read** — cannot tell whether a start is
         legal, so leave it to the bid ladder. Ignorance never licenses a move
         here; it only declines to make one.

    An ON-DEMAND box skips 4-6 entirely: it has no bid to lose and cannot be
    outbid, so a start is always the right first move."""
    if not present:
        return Resume("skip", "box is not in the listing — nothing to start")
    if tries_used >= max_tries:
        return Resume("skip", f"resume already attempted {tries_used}/{max_tries} "
                              f"times this eviction cycle")
    if budget_usd is not None and spend_usd >= budget_usd:
        return Resume("skip", f"budget consumed (${spend_usd:.4f} >= "
                              f"${budget_usd}) — a resume restarts the meter")
    if is_bid:
        if market_listed is False:
            return Resume("skip", "the machine lists no rentable bid offer — we "
                                  "were displaced; a start will not stick")
        mmb = market_min_bid if (market_min_bid and market_min_bid > 0) else None
        lb = last_bid if (last_bid and last_bid > 0) else None
        if mmb is None or lb is None:
            return Resume("skip", "no usable market read (floor "
                                  f"{market_min_bid!r} vs bid {last_bid!r}) — "
                                  "leaving it to the bid ladder")
        if lb < mmb:
            return Resume("skip", f"standing bid ${lb} is BELOW the live floor "
                                  f"${mmb} — raise the bid, do not start")
        return Resume("start", f"standing bid ${lb} still clears the live floor "
                               f"${mmb} and the chunk is still rentable — nobody "
                               f"took this box, it stopped")
    return Resume("start", "on-demand box: no bid to lose and it cannot be "
                           "outbid, so a start is the cheapest recovery")


#: Eviction classes whose recovery depends on somebody OTHER than the market:
#: the host bringing the chunk back, or our own `start` landing. No price we can
#: pay shortens the wait, so the ladder must not sit on a bid-shaped deadline in
#: one of these.
#:
#: `host_stop` is the whole set today, and deliberately so. `outbid` is a market
#: state the re-bid rungs can actually buy out of, and they carry their own
#: bound (REBID_MAX_RUNGS x REBID_WAIT_S). `ondemand_displaced` / `host_failure`
#: / `no_credit` already route past the bid rungs by name. `unknown` reaches
#: `dead` on the pre-existing `act is None and not_live >= 2 x debounce` term
#: within a couple of polls, which is faster than any wait added here.
HOST_RECOVERY_CLASSES = (EVICTION_HOST_STOP,)

HOST_STOP_ESCALATE_S = 240
#: How long a `host_stop` with claimed work may hold the ladder before it stops
#: waiting for the host and escalates to the replacement rung.
#:
#: WHAT THIS TRADES. A warm-disk recovery is worth waiting for — the box we
#: already rent keeps its rehydrated env, base model, dataset and newest
#: checkpoint, where a replacement pays a MEASURED 11m35s of setup on a cold
#: disk plus the whole pull again (see `resume_in_place`). So the wait is sized
#: to give the warm rungs their real chance and not one poll more:
#: rung ZERO fires at NOT_LIVE_DEBOUNCE polls (60-90 s at the 30-45 s tick this
#: fleet runs) and a `start` was measured landing in ~40 s on box 47226953, so
#: 240 s leaves >= 2 min of slack after the cheapest rung has been tried and has
#: visibly failed to bring the box back.
#:
#: It is NOT sized to JOB_SUP_RESCUE_WAIT_S (900 s), and that is the fix. That
#: number is how long vast's bid re-auction is given to resume a box somebody
#: outbid us for — a different mechanism with a different clock. Arming it on a
#: `host_stop` parked box 48996785 for fifteen minutes on 2026-08-28 with a
#: 38%-complete training job and $4.84 of budget left, while the ladder had a
#: replacement rung it never reached.


def host_stop_escalation(*, eviction_class, claimed_work, evicted_since, now,
                         not_live=0, debounce=NOT_LIVE_DEBOUNCE,
                         escalate_after_s=HOST_STOP_ESCALATE_S,
                         classes=HOST_RECOVERY_CLASSES):
    """PURE. Has a host-recovery eviction waited long enough to be replaced?

    Returns the seconds waited (a float, always > 0) when the ladder must stop
    waiting and escalate to the replacement rung, else None. Every gate is a
    refusal by default, in the order they are checked:

      1. the cycle's class is not one the host owns (`HOST_RECOVERY_CLASSES`) —
         somebody else's rung is the right answer and it has its own bound;
      2. **no claimed work** — with nothing to retarget a replacement buys
         nothing and spends real money, so an idle watch waits as long as the
         host likes;
      3. no `evicted_since` — a cycle that never journaled its start has no
         clock, and ignorance never licenses a rental;
      4. the box has not been down for `2 x debounce` polls — the same
         "we have actually seen it stay down" bar the pre-existing `dead` term
         uses, so a flap cannot escalate;
      5. `escalate_after_s` has not elapsed since the eviction.

    `escalate_after_s <= 0` disables the escalation entirely (an operator
    opt-out, and the shape every pre-2026-08-28 caller had).

    This never CHOOSES a replacement — `replacement_decision` still owns whether
    one is affordable, and refuses on an exhausted budget exactly as before. All
    this decides is that the waiting is over."""
    if eviction_class not in tuple(classes or ()):
        return None
    if not claimed_work:
        return None
    if evicted_since is None or now is None:
        return None
    if not escalate_after_s or escalate_after_s <= 0:
        return None
    if int(not_live or 0) < 2 * int(debounce or 0):
        return None
    try:
        waited = float(now) - float(evicted_since)
    except (TypeError, ValueError):
        return None
    return waited if waited >= float(escalate_after_s) else None


def bid_can_win(market_min_bid, on_demand, max_bid=None):
    """PURE. Could ANY bid we are allowed to place win this machine back?
    `_bid_target` already encodes the answer (None = unwinnable floor: the
    post-spike floor sits at/above on-demand, or above our ceiling), so this is
    a named alias for the eviction path rather than a second copy of the rule."""
    return _bid_target(market_min_bid, max_bid, on_demand) is not None


#: What an autonomous replacement may be charged, and every bound that shaped
#: it. `price is None` IS the refusal (no anchor). `source` names the market
#: evidence that moved it, or None when nothing did.
ReplacementCeiling = namedtuple("ReplacementCeiling",
                                ["price", "base", "market_ref", "source",
                                 "bound", "escalated"])


def replacement_ceiling(*, launch_dph_anchor, ceiling_mult=REPLACE_CEILING_MULT,
                        market_floor=None, p_alt=None, on_demand=None,
                        budget_left=None, horizon_h=None,
                        escalation_cap_mult=REPLACE_ESCALATION_CAP_MULT,
                        cushion_mult=REPLACE_MIN_CUSHION,
                        min_runtime_h=REPLACE_MIN_RUNTIME_H):
    """PURE. The price ceiling on ANY autonomous replacement rental, re-derived
    against live market evidence. Returns a `ReplacementCeiling`; `price is
    None` only when no launch anchor was ever observed, which stays a refusal.

    The base is unchanged and remains the rule whenever the market is under it:
    `ceiling_mult x launch_dph_anchor`, the ORIGINAL launch price
    (FLEETD_DESIGN "anchored once ... never re-anchored on a swap").

    **What this adds, and why the anti-compounding property survives.** The
    anchor's immutability was specified against RATCHETING — three replacements
    at 2x each licensing an 8x box — and it is silent about the other direction:
    an anchor that is simply BELOW what the market now charges. On 2026-08-24 a
    $0.1933 anchor gave a $0.387 ceiling, the only qualifying offer billed
    $0.4000, and the pull-reschedule lane refused 36 consecutive ticks over a
    3.4% gap while the daemon held a $0.5333 `p_alt` read it never consulted.
    So a re-derivation is admitted, and it is safe against the original argument
    because **every escalation is computed from the SAME original anchor** —
    never from the previous escalated ceiling — and capped at
    `escalation_cap_mult x anchor`. N swaps cannot compound; the ceiling is a
    function of (anchor, market), not of its own history.

    Escalation is EVIDENCE-GATED and TIGHTEST-FIRST. `market_floor` (a price we
    actually saw a qualifying offer bill — the refusal itself is the reading)
    outranks nothing and undercuts nothing: the reference is the MINIMUM of the
    live evidence, because the ceiling only has to clear the cheapest box that
    would unblock the lane. With no evidence above the base, or none at all,
    the base stands and the refusal is unchanged.

    Bounds on the escalated value, all of which only ever TIGHTEN:

      * `cushion_mult x market_ref` — enough to place a survivable bid over the
        floor we read, not enough to buy a different class of box;
      * `escalation_cap_mult x anchor` — the absolute bound above;
      * `effective_bid_ceiling(on_demand)` — the hard 0.75x on-demand line that
        binds every emitted bid from every path;
      * `budget_left / horizon_h` — what the remaining budget sustains for the
        queue's projected runtime (`min_runtime_h` when no projection exists).

    The result is `max(base, min(bounds))`: re-derivation may raise a ceiling
    the market has outrun, never lower one the operator already authorised."""
    anchor = (launch_dph_anchor
              if (launch_dph_anchor and launch_dph_anchor > 0) else None)
    if anchor is None:
        return ReplacementCeiling(None, None, None, None,
                                  "no launch price anchor observed", False)
    base = round(ceiling_mult * anchor, 3)
    # Evidence, tightest first. A refused offer's realized price is the strongest
    # reading there is — we know a box exists at exactly that number — but a
    # fresh class-wide `p_alt` is what remains when the ceiling filtered the
    # search down to nothing, which is 29 of the 36 refusals in the incident.
    ev = [("market_floor", market_floor), ("p_alt", p_alt)]
    ev = [(k, float(v)) for k, v in ev if v is not None and float(v) > 0]
    over = [(k, v) for k, v in ev if v > base + 1e-9]
    if not over:
        why = ("market under the base ceiling" if ev
               else "no live market evidence")
        return ReplacementCeiling(base, base, None, None, why, False)
    source, market_ref = min(over, key=lambda kv: kv[1])
    want = round(cushion_mult * market_ref, 3)
    bounds = [(want, f"{cushion_mult:g}x the ${market_ref:.4f} {source} read"),
              (round(escalation_cap_mult * anchor, 3),
               f"{escalation_cap_mult:g}x the ${anchor:.4f} launch anchor "
               f"(absolute escalation cap)")]
    od_ceiling = effective_bid_ceiling(on_demand if (on_demand and on_demand > 0)
                                       else None)
    if od_ceiling is not None:
        bounds.append((od_ceiling, f"${od_ceiling:.3f} = "
                                   f"{BID_CEILING_ONDEMAND_FRAC:g}x on-demand "
                                   f"${float(on_demand):.4f} (hard)"))
    horizon = (float(horizon_h) if (horizon_h and float(horizon_h) > 0)
               else min_runtime_h)
    if budget_left is not None and horizon > 0:
        bounds.append((round(float(budget_left) / horizon, 3),
                       f"${float(budget_left):.2f} of budget over a "
                       f"{horizon:.2f}h horizon"))
    price, bound = min(bounds, key=lambda pb: pb[0])
    if price <= base + 1e-9:
        # Every escalation route is bounded at or under the base: the market has
        # moved but nothing we are allowed to spend follows it there. REFUSE at
        # the base — that is the whole point of a cost bound.
        return ReplacementCeiling(base, base, market_ref, source,
                                  f"escalation refused — bound by {bound}", False)
    return ReplacementCeiling(price, base, market_ref, source, bound, True)


def replacement_decision(*, eviction_class, replacements_used, budget_usd,
                         spend_usd, launch_dph_anchor, offer_min_bid=None,
                         offer_ondemand=None, spot_ondemand=None, fast_deaths=0,
                         max_replacements=MAX_REPLACEMENTS,
                         ceiling_mult=REPLACE_CEILING_MULT,
                         min_runtime_h=REPLACE_MIN_RUNTIME_H,
                         max_fast_deaths=MAX_SPOT_FASTDEATHS,
                         min_cushion=REPLACE_MIN_CUSHION,
                         observed_lifetime_h=None, setup_h=None,
                         ckpt_interval_h=0.0,
                         prior_lifetime_h=SPOT_PRIOR_LIFETIME_H,
                         ceiling=None, ceiling_basis=None):
    """PURE. Should the ladder rent a replacement box, and on what market?

    Returns a `Replacement(action, rental, price, reason, ceiling, budget_left)`
    where `action` is "rent" or "stop". Every refusal carries the arithmetic in
    `reason` so the journal line IS the audit trail — a rental decision nobody
    can reconstruct afterwards is not a bounded one.

    Spend bounds, in the order they are checked (each a hard refusal):

      1. **A budget cap is mandatory.** `budget_usd is None` -> stop. An
         uncapped watch may not spend autonomously, full stop; this is the one
         gate that cannot be widened by a knob.
      2. **Replacement count cap** (`max_replacements`, default 3). Prevents the
         crashloop-rental shape where a bad image or a hostile market burns the
         whole budget one boot at a time.
      3. **Budget remainder** must cover `min_runtime_h` at the candidate price.
         Renting a box you can afford for four minutes spends money for zero
         progress and then budget-parks.
      4. **Price ceiling derived from the ORIGINAL launch** —
         `ceiling_mult x launch_dph_anchor`. No anchor (never observed a price)
         -> stop: an unknown ceiling is not a licence to spend.

    `ceiling` / `ceiling_basis` (2026-08-24) let the CALLER hand in a ceiling it
    already derived through `replacement_ceiling`, which is how a market
    re-pricing reaches this rung. Passing it also collapses a real drift hazard:
    the eviction lane derived the ceiling twice — once for the offer search's
    `max_dph`, once here — and the two agreed only because the formula was
    copied. `None` re-derives the base exactly as every pre-2026-08-24 caller
    got it, so the omitted case is byte-identical. `ceiling_basis` is the phrase
    the refusal arithmetic names it by; without one an escalated ceiling would
    still be reported as "2x the launch price", which it no longer is.

    Rung choice (justified in FLEETD_DESIGN §"Automatic eviction replacement"):
    spot first, because it is what the operator authorised and it is 2-3x
    cheaper. It flips to on-demand when spot is STRUCTURALLY unsafe rather than
    merely unlucky:

      * the eviction was an on-demand displacement (a spot replacement on that
        market is buying the same loss again),
      * `fast_deaths >= max_fast_deaths` (two replacements already died inside
        SPOT_FASTDEATH_S),
      * the spread is INVERTED — the winnable bid price is at or above the
        candidate's on-demand rate, or
      * the CUSHION is thin — the winnable bid clears the floor by less than
        `min_cushion`, because the on-demand clamp compressed it. That is the v7
        eviction-1 shape ($0.747 over a $0.746 floor) and re-renting into it buys
        the same eviction again, or
      * spot LOSES PER USEFUL HOUR at this lane's expected lifetime
        (`spot_breakeven`, wired 2026-08-08). The first four triggers compare
        prices; only this one can see the v11 livelock, where the prices were
        fine and the lane still banked zero rows per cycle because realised
        lifetimes (11-13 min) sat at the setup cost (11m35s). Needs `setup_h`
        plus a lifetime — `observed_lifetime_h` (this lane's own inter-eviction
        history) when it exists, else `prior_lifetime_h`
        (SPOT_PRIOR_LIFETIME_H, recalibration 2026-08-09 item B), because
        `observed_lifetime_h` is None until a bid replacement has already died
        and the FIRST eviction is the only eviction most watches ever have.
        `Replacement.lifetime_basis` says which was used, and the reason string
        names it: an escalation made on an assumption must never read like one
        made on evidence.

    Either rung may be refused on price/budget; if the preferred rung is refused
    the other is tried before giving up, so a cheap on-demand box is never
    passed over for an unaffordable spot one.

    `offer_ondemand` — READ THIS BEFORE WIRING A CALLER. It is the cheapest
    ON-DEMAND price in class, read from an ON-DEMAND market query and
    deliberately NOT ceiling-filtered: it is both the clamp reference for the
    bid AND the price the on-demand rung would pay, so pushing the ceiling into
    that probe hides the one number the ceiling check needs. Feeding it
    un-ceilinged is what turns "the cheapest on-demand box costs $3.47, over the
    $2.164 ceiling" into a LOUD refusal instead of an unpriced rental.

    `offer_ondemand=None` means UNKNOWN (no on-demand market in class, or the
    probe failed) and is never a licence to guess. An unknown on-demand price:
    does NOT clamp the bid (the spot target keeps its full BID_TARGET_MULT
    cushion), does NOT make a cushion "thin", is never "inverted", and can never
    be rented — if some other signal still prefers the on-demand rung the ladder
    falls back to spot or refuses, naming the missing quote.

    `spot_ondemand` (added 2026-08-16, replacement-probe incident) is the
    SPOT CANDIDATE'S OWN machine on-demand rate, and it exists because
    `offer_ondemand` is asked to be two different numbers at once. The
    survival/ceiling rails inside `bid_decision` are a statement about ONE
    MACHINE — "is a preemptible claim on THIS box worth having, given what
    buying THIS box outright costs" — while `offer_ondemand` is the cheapest
    on-demand price ANYWHERE in class, which is the right number for the
    on-demand rung and the wrong one for that rail. With a widened candidate
    class the two diverge hard (a $0.40 H200 spot chunk whose own machine lists
    at $3.34, against a $0.55 on-demand box of a different class), and clamping
    the bid against the foreign price vetoes a structurally-safe candidate.
    So: `spot_ondemand` (when given and positive) is the reference for
    `bid_decision` and for `thin`; `offer_ondemand` stays the reference for
    `inverted`, for `spot_breakeven`, and for the on-demand rung's own price.
    `None` = not supplied, and everything falls back to `offer_ondemand`
    exactly as it did before — the same number in both roles.

    NEVER pass a BID offer's `dph_total` to EITHER of them (defect, 2026-08-05
    — doc 50,
    `docs/plans/reverse-compilation/data/training-methods-review/
    50_ANALYSIS_FLEETD_ONDEMAND_ESCALATION_2026-08-05.md`). On a bid-type vast
    offer that field is the CURRENT INTERRUPTIBLE price (~min_bid + 0.5%), not
    the machine's on-demand rate, so it reads as an on-demand price sitting a
    tenth of a cent over the floor: the clamp eats the entire cushion, `thin`
    fires, and the ladder takes the expensive rung on arithmetic that describes
    no real offer. One `or spot_offer["dph_total"]` fallback in the caller cost
    a $3.4741/hr on-demand rental against a $2.164 ceiling that same night."""
    if budget_usd is None:
        return Replacement("stop", None, None,
                           "no budget cap on this watch — autonomous spend is "
                           "refused without one", None, None)
    budget_left = round(float(budget_usd) - float(spend_usd or 0.0), 4)
    if replacements_used >= max_replacements:
        return Replacement("stop", None, None,
                           f"replacement cap reached ({replacements_used}/"
                           f"{max_replacements}) — not re-renting in a loop",
                           None, budget_left)
    if budget_left <= 0:
        return Replacement("stop", None, None,
                           f"budget exhausted (${spend_usd:.2f} of "
                           f"${budget_usd:.2f})", None, budget_left)
    anchor = launch_dph_anchor if (launch_dph_anchor and launch_dph_anchor > 0) else None
    if anchor is None:
        return Replacement("stop", None, None,
                           "no launch price anchor observed — cannot derive a "
                           "price ceiling, refusing to spend", None, budget_left)
    if ceiling is None:
        ceiling = round(ceiling_mult * anchor, 3)
    ceiling_basis = ceiling_basis or (f"{ceiling_mult:g}x the ${anchor:.4f} "
                                      f"launch price")

    # `_bid_target(floor, max_bid=None, on_demand)` IS herdd's `_auto_bid_price`
    # arithmetic (1.20x floor, hard-clamped under on-demand, D7 raise-to-floor,
    # None when the floor is unwinnable). Calling it here keeps this module a
    # leaf — and keeps the launch price and the replacement price on one rule.
    # UNKNOWN on-demand is a first-class state, not a zero and not a guess (see
    # the docstring): every use of the number below is gated on `od_known`.
    od_known = bool(offer_ondemand and offer_ondemand > 0)
    od_price = offer_ondemand if od_known else None
    # The ceiling is handed to `_bid_target` as the max_bid CAP rather than used
    # only as an after-the-fact filter (2026-08-08). With the cushion rail a
    # cushioned bid can now land over a tight ceiling, and refusing there would
    # trade a rentable box for no box at all; clamping instead keeps the rental
    # and lets `thin` below decide whether what fits under the ceiling is still a
    # survivable bid. If even the FLOOR is over the ceiling, `_bid_target`'s D7
    # branch returns None and `_viable` refuses with "no price", as before.
    # The clamp/survival reference for the BID is the candidate's OWN machine
    # on-demand rate when the caller measured one (see `spot_ondemand`); it
    # falls back to the class price, which is what every pre-2026-08-16 caller
    # passes and the only number that existed before.
    clamp_od = spot_ondemand if (spot_ondemand and spot_ondemand > 0) else od_price
    spot_dec = bid_decision(offer_min_bid, ceiling, clamp_od)
    spot_price = spot_dec.price

    def _viable(price, why_none="no price"):
        if price is None:
            # Name WHICH no-price this is. Since 2026-08-09 the commonest one on
            # the SPOT rung is the hard-ceiling ESCALATION (the candidate's floor
            # sits so close to its on-demand rate that no survivable bid fits
            # under BID_CEILING_ONDEMAND_FRAC x on-demand) — and that is precisely
            # the evidence for taking the on-demand rung, so it must not be
            # journaled as a bare "no price".
            return False, why_none
        if price > ceiling:
            return False, (f"${price:.4f} over the ${ceiling:.3f} ceiling "
                           f"({ceiling_basis})")
        if budget_left / price < min_runtime_h:
            return False, (f"${budget_left:.2f} left buys only "
                           f"{budget_left / price:.2f}h at ${price:.4f}/hr "
                           f"(< {min_runtime_h:g}h floor)")
        return True, None

    spot_ok, spot_why = _viable(spot_price, spot_dec.reason or "no price")
    od_ok, od_why = _viable(od_price)
    if not od_known:
        od_why = ("no on-demand price known — the on-demand market probe "
                  "returned nothing, and an unknown on-demand price neither "
                  "clamps the bid nor licenses the on-demand rung")

    inverted = (spot_price is not None and od_price is not None
                and spot_price >= od_price)
    floor = offer_min_bid if (offer_min_bid and offer_min_bid > 0) else None
    # `thin` names ONE mechanism: the on-demand clamp compressed the 1.20x
    # target down onto the floor (the v7 eviction-1 shape). With NO on-demand
    # price there is no clamp, so the cushion is BID_TARGET_MULT by
    # construction and this must not fire — an unknown on-demand price
    # masquerading as a thin cushion is precisely how the 2026-08-05 ladder
    # escalated to a rung it had no quote for.
    #
    # The comparison is against the ROUNDED cushion (2026-08-09). `spot_price`
    # comes off `bid_decision`, which quantises to the $0.001 grid; the raw
    # product does not. At floor $1.30 that is `1.43 < 1.4300000000000002` —
    # True — so `thin` fired on a bid that IS exactly the cushion, on nothing but
    # float noise. `_CEIL_EPS` is nine orders of magnitude below the price grid,
    # so it can absorb the noise and nothing else.
    #
    # Gated on `clamp_od`, not `od_known`: `thin` describes what the clamp that
    # ACTUALLY APPLIED did to the cushion, and since 2026-08-16 that clamp may
    # be the candidate's own machine price rather than the class one.
    thin = (clamp_od is not None and spot_price is not None and floor is not None
            and spot_price < round(min_cushion * floor, 3) - _CEIL_EPS)
    # LIVELOCK trigger (2026-08-08 — the wiring `spot_breakeven` was written for
    # and named in its own docstring). `thin`/`inverted` compare PRICES; this one
    # compares COST PER USEFUL HOUR, which is the only comparison that can see
    # the shape that actually burned the v11 chat arm: four evictions, 11-13 min
    # realised lifetimes against an 11m35s setup, and a full boot that moved the
    # banked-row count 40 -> 40. Nothing about those prices was wrong. Spot was
    # simply not cheaper per useful hour at that lifetime, and no bid fixes that.
    #
    # The lifetime it scores against is OBSERVED where this lane has one
    # (`observed_lifetime_h`, the median of its own dead bid replacements) and the
    # PRIOR otherwise (SPOT_PRIOR_LIFETIME_H — recalibration 2026-08-09, item B).
    # Before the prior the rung was dead on the FIRST eviction, which for most
    # watches is the only one they ever have: `_job_observed_lifetime_h` is built
    # from replacements that have already died, so the trigger could not fire
    # until the ladder had rented and lost a spot box. On 2026-08-08 that decision
    # had to be made by a human instead.
    #
    # `setup_h` is still mandatory — it is a MEASURED lane property, not an
    # assumption, and without it there is no arithmetic at all.
    lifetime_h, lifetime_basis = observed_lifetime_h, "observed"
    if lifetime_h is None:
        lifetime_h = prior_lifetime_h if (prior_lifetime_h
                                          and prior_lifetime_h > 0) else None
        lifetime_basis = "prior" if lifetime_h is not None else None
    livelock = False
    if (od_known and spot_price is not None
            and lifetime_h is not None and setup_h is not None):
        cheaper, _spot_cost, _od_cost = spot_breakeven(
            spot_dph=spot_price, ondemand_dph=od_price, setup_h=setup_h,
            expected_lifetime_h=lifetime_h,
            ckpt_interval_h=ckpt_interval_h or 0.0)
        livelock = cheaper is False
    else:
        lifetime_basis = None                 # the rung did not run
    prefer_od = (eviction_class == EVICTION_ONDEMAND
                 or fast_deaths >= max_fast_deaths
                 or inverted or thin or livelock)
    # Hoisted out of the f-string below: a multi-line expression inside a
    # replacement field is PEP 701 (3.12+), and this module is imported by the
    # BOX-side python half, which runs on whatever the image ships (3.11 on
    # stock pytorch images). See test_box_python_floor.py.
    _basis_note = ("measured from its own dead replacements"
                   if lifetime_basis == "observed" else
                   "ASSUMED — no replacement of this lane has died yet, so "
                   "this is the SPOT_PRIOR_LIFETIME_H prior, not evidence")
    if prefer_od:
        why = ("the eviction was an on-demand claim, which outranks any bid"
               if eviction_class == EVICTION_ONDEMAND
               else f"{fast_deaths} spot replacement(s) died inside "
                    f"{SPOT_FASTDEATH_S}s" if fast_deaths >= max_fast_deaths
               else f"spot/on-demand spread INVERTED (bid ${spot_price} >= "
                    f"on-demand ${od_price})" if inverted
               else f"spot cushion too thin (bid ${spot_price} is only "
                    f"{spot_price / floor:.3f}x the ${floor} floor, "
                    f"< {min_cushion:g}x — the on-demand clamp ate it)" if thin
               else f"spot LOSES per useful hour at this lane's "
                    f"{'OBSERVED' if lifetime_basis == 'observed' else 'ASSUMED'}"
                    f" lifetime ({lifetime_h:.2f}h "
                    f"{_basis_note}"
                    f", against {setup_h:.2f}h of setup): ${spot_price:.4f}/hr "
                    f"of spot costs more per productive hour than "
                    f"${od_price:.4f}/hr of on-demand")
        if od_ok:
            return Replacement("rent", "ondemand", od_price,
                               f"on-demand rung: {why}; ${od_price:.4f}/hr "
                               f"within the ${ceiling:.3f} ceiling, "
                               f"${budget_left:.2f} left", ceiling, budget_left,
                               lifetime_basis)
        if spot_ok and not inverted:
            return Replacement("rent", "bid", spot_price,
                               f"on-demand preferred ({why}) but refused "
                               f"({od_why}); falling back to spot at "
                               f"${spot_price:.4f}/hr", ceiling, budget_left,
                               lifetime_basis)
        return Replacement("stop", None, None,
                           f"on-demand rung preferred ({why}) but refused "
                           f"({od_why}); spot refused too ({spot_why or 'inverted spread'})",
                           ceiling, budget_left, lifetime_basis)
    if spot_ok:
        return Replacement("rent", "bid", spot_price,
                           f"spot rung: ${spot_price:.4f}/hr within the "
                           f"${ceiling:.3f} ceiling, ${budget_left:.2f} left",
                           ceiling, budget_left, lifetime_basis)
    if od_ok:
        return Replacement("rent", "ondemand", od_price,
                           f"spot refused ({spot_why}); on-demand "
                           f"${od_price:.4f}/hr fits the ${ceiling:.3f} ceiling",
                           ceiling, budget_left, lifetime_basis)
    return Replacement("stop", None, None,
                       f"no affordable replacement: spot {spot_why}, "
                       f"on-demand {od_why}", ceiling, budget_left)


# --------------------------------------------------------------------------- #
# re-bid ladder on outbid (autobid displacement audit, 2026-08-08)
#
# `_bid_action`'s `rescue_bid` fires exactly ONCE per eviction cycle
# (`rescue_attempted` / the driver's `rescue_deadline is None` gate) and aims at
# the ordinary standing target. That is the right first move and the wrong last
# one: if the post-spike floor keeps climbing, or the single rescue was priced
# off a floor read that was already stale, there is no second attempt and the
# ladder falls straight through to `unrecoverable` -> rent a replacement.
#
# A replacement costs a MEASURED 11m35s of setup (v11 eval lane) plus the loss of
# whatever never reached B2, and it starts on a cold disk. A re-bid costs the
# price delta on a box that still holds its rehydrated env, its base model, its
# dataset and its newest checkpoint. So while a legal winning bid exists under
# the ceiling the ladder should keep bidding, and only then rent.
#
# The ceiling is deliberately the SAME multiple of the launch anchor that
# `replacement_decision` is allowed to spend (REPLACE_CEILING_MULT): money we
# would authorise for a cold replacement box we should certainly authorise for
# the warm one we already own. Above that line both rungs stop and the operator
# is told, with the arithmetic, which bound fired.
# --------------------------------------------------------------------------- #
REBID_STEP = 0.25          # each rung raises the standing bid by this fraction of
                           # itself. 25% is well clear of BID_MIN_STEP at every
                           # price we rent at, and clears a typical observed floor
                           # spike in one move: v7 eviction 1's floor went
                           # $0.7599 -> $0.9099 (+19.7%), q6's anchor-to-decision
                           # drift was $1.0819 -> $1.60 over a whole session.
REBID_MAX_RUNGS = 3        # rungs per eviction cycle, then refuse + alarm. With
                           # REBID_WAIT_S below this is a 15-minute total budget —
                           # deliberately ONE replacement's setup cost (11m35s
                           # measured, plus slack): the ladder may spend at most as
                           # much wall time saving the warm box as replacing it
                           # would have cost, and not a minute more.
REBID_WAIT_S = 300         # how long a rung is given to auto-resume the box before
                           # the next rung. The box is STOPPED through this window,
                           # so it bills allocated storage only (~$0.09-$0.19/h),
                           # not GPU — the cost of waiting is wall time, which is
                           # exactly what REBID_MAX_RUNGS bounds. ALSO the decay
                           # hysteresis window (`_recent_raise_hold`): as long as a
                           # rung owns the price it just paid, decay may not give
                           # it back.
REBID_CEILING_MULT = REPLACE_CEILING_MULT   # ladder ceiling = this x the ORIGINAL
                           # launch dph. Bound to the replacement ceiling on
                           # purpose (see the block comment); if the two ever need
                           # to differ, the rebid one should be the HIGHER of the
                           # two, never the lower.

Rebid = namedtuple("Rebid", ["action", "price", "reason", "ceiling", "rungs_left"])


def rebid_ceiling(*, launch_dph_anchor, max_bid, on_demand,
                  ceiling_mult=REBID_CEILING_MULT, extra_bounds=()):
    """PURE. The price ceiling an ESCALATING bid rung is held to, or None when
    no ceiling is derivable at all.

    Extracted (S2b review round 1, M3) so the two rungs that escalate a standing
    bid on an evicted box — `rebid_ladder`'s rungs and the NOTIFICATION-priced
    half of `_bid_action`'s rescue quote — cannot drift apart. Before the
    extraction they had drifted the whole way: the re-bid rung was bounded by
    `min(REBID_CEILING_MULT x launch anchor, max_bid, 0.75 x on-demand)` while
    the notification-priced rescue was bounded by `max_bid` and the on-demand
    clamp alone. On the 2026-08-16 field box (anchor $0.45, on-demand $3.00,
    derived max_bid $2.999) that was $0.900 against $2.999 — and the row quoted
    $1.212, 1.35x the ceiling the very next rung would have refused.

    The bound set, in the order `rebid_ladder` has always built it:

      * `ceiling_mult x launch_dph_anchor` — the IMMUTABLE launch price, so
        three replacements at 2x cannot license an 8x box;
      * `max_bid` — the operator's cap, or the on-demand-anchored default;
      * `extra_bounds` — the caller's own tighter lines (the job-aware
        `defense_ceiling`; it only ever TIGHTENS);
      * and then the HARD on-demand ceiling (`effective_bid_ceiling`), which
        binds on every emitted bid from every path.

    **None when the bound set is empty** — no anchor and no `max_bid`. That is
    `rebid_ladder`'s refusal 5 verbatim and it is the rule, not an omission: an
    unknown ceiling is not a licence to spend. The on-demand clamp alone does
    not rescue that case, because it says only "below on-demand", which on an
    expensive machine is not a bound anybody authorised."""
    bounds = []
    anchor = (launch_dph_anchor
              if (launch_dph_anchor and launch_dph_anchor > 0) else None)
    if anchor is not None:
        bounds.append(round(ceiling_mult * anchor, 3))
    if max_bid is not None:
        bounds.append(float(max_bid))
    bounds.extend(b for b in (extra_bounds or ()) if b is not None)
    if not bounds:
        return None
    ceiling = min(bounds)
    od = on_demand if (on_demand and on_demand > 0) else None
    od_ceiling = effective_bid_ceiling(od)
    if od_ceiling is not None:
        ceiling = min(ceiling, od_ceiling)
    return ceiling


#: What the notification-priced rescue quote costs, and every bound that shaped
#: it. `price` None IS the refusal; `refusal` says which line fired.
NotifyRescue = namedtuple("NotifyRescue",
                          ["price", "floor", "ceiling", "anchor",
                           "budget_left", "refusal"])


def notify_rescue_bound(s):
    """PURE. The rescue quote a MATCHED outbid row licenses on this poll state,
    with the arithmetic that produced it (S2b review round 1, M3).

    Returns a `NotifyRescue`. `floor is None` means the row proposed nothing at
    this state — no row, an unusable price, or a displacing price the market
    floor we can already SEE already covers — and in that case the caller must
    take its pre-S2b path untouched. That is the byte-identity boundary: this
    function only ever describes the quote a NOTIFICATION raised.

    Three bounds, and all three are refusals rather than smaller numbers:

      1. **the ceiling** — `rebid_ceiling`, the same construction `rebid_ladder`
         uses, with the SAME FOUR INPUTS: launch anchor, `max_bid`, the
         per-watch `rebid_ceiling_mult` knob, and the job-aware `defense_cap`.
         The last two were missing from round 1's fix and round 2 measured what
         that cost: with a live defense (anchor $1.20, p_alt $0.60, 20 h
         remaining) the re-bid rung's ceiling is $0.606 and the rescue's was
         $2.25, and since the rescue PUT runs FIRST on the tick, the defense
         controller never got to bind it — 11 of 18 sampled states, plus 31 of
         64 at a tightened `JOB_REBID_CEILING_MULT`. A bound that only the later
         rung applies is not a bound. No derivable ceiling (no launch anchor,
         no max_bid) refuses the quote outright; before the review that state
         emitted `rescue:inf` off an `inf` row and `rescue:118.812` off a $99
         one, because the only rail left was a clamp with nothing in it.
      2. **the rails** — `_bid_target(floor, ceiling, on_demand)`, exactly as
         before: preference, cost cap, survival cushion, hard on-demand clamp,
         D7 raise-to-floor. A refusal here is an escalation, never a bigger
         number.
      3. **affordability** — the remaining budget must still buy
         `REPLACE_MIN_RUNTIME_H` at the quoted price. `rebid_ladder` has had
         this check since it was written (refusal 6); the rescue rung never did,
         so a --budget 5.00 watch with $4.90 spent would raise its bid to a
         price the residual budget budget-parks the box at inside five minutes.
         Uncapped watches (`budget_usd` None) are unaffected, same rule as the
         re-bid ladder: a rescue moves the price of a meter already running."""
    floor = notify_rescue_floor(s.get("notify_min_bid"), s.get("market_min_bid"))
    mmb = notify_price(s.get("market_min_bid"))
    anchor = s.get("launch_dph_anchor")
    if floor is None or (mmb is not None and floor <= mmb + _CEIL_EPS):
        # No row, or a row the visible market floor already dominates. The
        # notification raised nothing, so it bounds nothing: the caller's
        # pre-S2b rescue owns this state exactly as it always did.
        return NotifyRescue(None, None, None, anchor, None, None)
    mult = s.get("rebid_ceiling_mult")
    ceiling = rebid_ceiling(
        launch_dph_anchor=anchor, max_bid=s.get("max_bid"),
        on_demand=s.get("on_demand"),
        ceiling_mult=REBID_CEILING_MULT if mult is None else mult,
        extra_bounds=(s.get("defense_cap"),))
    budget = s.get("budget_usd")
    budget_left = (None if budget is None
                   else round(float(budget) - float(s.get("spend_usd") or 0.0), 4))
    if ceiling is None:
        return NotifyRescue(
            None, floor, None, anchor, budget_left,
            "no launch price anchor and no --max-bid — cannot derive a rescue "
            "ceiling, and an unknown ceiling is not a licence to spend "
            "(rebid_ladder refusal 5, same rule)")
    price = _bid_target(floor, ceiling, s.get("on_demand"))
    if price is None:
        return NotifyRescue(
            None, floor, ceiling, anchor, budget_left,
            f"the rails refused ${floor:.3f} under a ${ceiling:.3f} ceiling — "
            f"escalation, not a bigger number")
    if budget_left is not None:
        if budget_left <= 0:
            return NotifyRescue(None, floor, ceiling, anchor, budget_left,
                                "budget exhausted")
        if budget_left / price < REPLACE_MIN_RUNTIME_H:
            return NotifyRescue(
                None, floor, ceiling, anchor, budget_left,
                f"${budget_left:.2f} left buys only {budget_left / price:.2f}h "
                f"at ${price:.4f}/hr (< {REPLACE_MIN_RUNTIME_H:g}h floor) — "
                f"raising the bid would budget-park the box before it produced "
                f"anything")
    return NotifyRescue(price, floor, ceiling, anchor, budget_left, None)


def rebid_ladder(*, last_bid, market_min_bid, on_demand, max_bid,
                 rungs_used, launch_dph_anchor, eviction_class=None,
                 budget_usd=None, spend_usd=0.0,
                 step=REBID_STEP, max_rungs=REBID_MAX_RUNGS,
                 ceiling_mult=REBID_CEILING_MULT,
                 min_runtime_h=REPLACE_MIN_RUNTIME_H,
                 p_alt=None, remaining_h=None, ckpt_interval_h=0.0,
                 setup_h=SPOT_SETUP_H, defend=None, prior_runtime_h=None):
    """PURE. Should we raise the standing bid on an outbid box one more rung, and
    to what price? Returns `Rebid(action, price, reason, ceiling, rungs_left)`
    with `action` in {"rebid", "stop"}. Every refusal carries its arithmetic in
    `reason`, so the journal line IS the audit trail.

    Reached only AFTER `_bid_action`'s single `rescue_bid` has been tried and has
    not brought the box back — this is the escalation the ladder never had.

    Refusals, in check order (each a hard stop):

      1. `max_rungs <= 0` — the ladder is disabled by config.
      2. **ON-DEMAND DISPLACEMENT** — an on-demand claim outranks every
         interruptible bid at any price (v7 eviction 2: a $1.05 bid lost to a
         $1.0017 on-demand rate). Bidding into that is pure waste, and it is the
         single most common class we actually observe (5 of 15 auto-replacements
         explicitly, most of the 10 `unknown` ones by inference).
      3. No standing bid (`last_bid` None) — a legacy run with nothing to raise.
      4. Rungs exhausted.
      5. No ceiling derivable (no launch anchor AND no max_bid) — an unknown
         ceiling is not a licence to spend, same rule as `replacement_decision`.
      6. Budget, when a cap exists, must still buy `min_runtime_h` at the new
         price. Unlike a replacement rental this does NOT require a cap to exist:
         a rebid moves the price of a meter that is already running and already
         bounded by the watch's hard spend stop, whereas a rental starts a second
         meter. Requiring a cap here would silently disarm rescue on every
         uncapped watch — a regression, not a safeguard.
      7. The rung must be a MATERIAL raise (>= BID_MIN_STEP over the standing
         bid) after clamping to the ceiling; if the ceiling has no room left,
         that is the "refuse + alarm" terminal.
      8. The rung must be able to WIN — `_bid_target` None (floor at/over
         on-demand, or over the ceiling) means no legal bid takes this machine
         back and only a different box will.

    The rung price is `max(last_bid x (1 + step), the ordinary cushioned target
    for the CURRENT floor)`: at least one step up, and at least whatever the
    market now demands. It is then clamped to the ceiling, to `max_bid`, and to
    strictly-under-on-demand.

    **Job-aware ONE-SHOT defense (AUTOBID_DESIGN "Next iteration", owner
    2026-08-09).** When the caller supplies `p_alt` — the best qualifying
    REPLACEMENT offer's $/hr, read from the offers market, never from this
    machine's own floor — the ladder changes shape in two ways:

      * `defense_ceiling(p_alt, remaining_h, ...)` = `p_alt x (1 + (S+L)/R)`
        joins the bound set. It only TIGHTENS: every pre-existing rail (launch
        anchor, max_bid, the hard on-demand ceiling) still binds. A defense
        the job-aware ceiling refuses is the market saying "replacing is
        rationally cheaper than holding", and the refusal falls through to the
        replacement rung exactly like any other ceiling stop.
      * the ladder collapses to ONE rung (`max_rungs` clamped to 1): a single
        meaningful re-bid at the cushioned market target, not a +25% walk. An
        incremental chase against another autobidder is the 2026-08-08
        ratchet with a live counterparty; one priced shot, then replacement.

    `p_alt` also stands in as the ceiling anchor when BOTH `launch_dph_anchor`
    and `max_bid` are missing — the replacement market is a real ceiling
    derivation, unlike the "no anchor at all" refusal it replaces on that path.
    `p_alt=None` (unread replacement market) preserves the pre-2026-08-09
    ladder exactly.

    **`defend` / `prior_runtime_h`** (FLEET_REVIEW_2026-08-14 item 7) reach
    `defense_ceiling` untouched — this function reads no config, the caller
    passes the hint in. `defend="cheap"` zeroes the lost-work term, and when
    that CHANGES the ceiling the reason string says so with both prices: the
    ladder is allowed to defend disposable work less hard, but not silently."""
    lost_h, defend_mode = lost_work_hours(defend=defend,
                                          ckpt_interval_h=ckpt_interval_h,
                                          prior_runtime_h=prior_runtime_h)
    defend_explicit = (isinstance(defend, str)
                       and defend.strip().lower() in DEFEND_MODES)
    defense_cap, defense_basis = defense_ceiling(
        p_alt=p_alt, remaining_h=remaining_h, setup_h=setup_h,
        ckpt_interval_h=ckpt_interval_h, defend=defend,
        prior_runtime_h=prior_runtime_h)
    # Journal the hint's EFFECT, not its value. A cheap defend that changed
    # nothing is noise; a cheap defend that capped a ceiling `dear` would have
    # allowed is the entire feature, and the $2.755 chase is why it has to be
    # legible in the refusal line rather than inferable from the constants.
    defend_note = ""
    if defense_cap is not None and defend_mode == DEFEND_CHEAP:
        dear_h, _dear_mode = lost_work_hours(
            defend=DEFEND_DEAR, ckpt_interval_h=ckpt_interval_h,
            prior_runtime_h=prior_runtime_h)
        dear_cap, _dear_basis = defense_ceiling(
            p_alt=p_alt, remaining_h=remaining_h, setup_h=setup_h,
            ckpt_interval_h=ckpt_interval_h, defend=DEFEND_DEAR,
            prior_runtime_h=prior_runtime_h)
        if dear_h > 0 and dear_cap is not None and dear_cap > defense_cap:
            defend_note = (
                f" [defend={DEFEND_CHEAP}"
                + ("" if defend_explicit else " (derived: no checkpoint_s)")
                + f": {dear_h:.2f}h of un-checkpointed work is cheaper to "
                  f"RE-RUN than to defend, so the lost-work term is 0 and the "
                  f"job-aware ceiling is ${defense_cap:.3f} where "
                  f"defend={DEFEND_DEAR} would have allowed ${dear_cap:.3f}]")
    if defense_cap is not None:
        max_rungs = min(int(max_rungs), 1)    # one-shot: one priced defense,
                                              # never a bidding war
    rungs_left = max(0, int(max_rungs) - int(rungs_used or 0))
    if max_rungs <= 0:
        return Rebid("stop", None, "re-bid ladder disabled (max_rungs=0)",
                     None, 0)
    if eviction_class == EVICTION_ONDEMAND:
        return Rebid("stop", None,
                     "the eviction was an on-demand claim, which outranks every "
                     "interruptible bid at any price — no rung can win this box "
                     "back", None, rungs_left)
    if last_bid is None or last_bid <= 0:
        return Rebid("stop", None,
                     "no standing bid to raise (legacy run with no captured bid)",
                     None, rungs_left)
    if rungs_left <= 0:
        return Rebid("stop", None,
                     (f"one-shot job-aware defense already spent "
                      f"({rungs_used} rung used) — one priced re-bid per "
                      f"eviction cycle, never a bidding war; the replacement "
                      f"rung answers from here"
                      if defense_cap is not None else
                      f"re-bid ladder exhausted ({rungs_used}/{max_rungs} rungs) — "
                      f"the market is taking this machine faster than we can pay "
                      f"for it"), None, 0)

    od = on_demand if (on_demand and on_demand > 0) else None
    anchor = launch_dph_anchor if (launch_dph_anchor and launch_dph_anchor > 0) else None
    # The bound set, and the HARD on-demand ceiling under it, are `rebid_ceiling`
    # — extracted so the notification-priced rescue quote is held to the SAME
    # line this rung is (review round 1, M3). None here is the old empty-`bounds`
    # refusal, unchanged: an unknown ceiling is not a licence to spend.
    ceiling = rebid_ceiling(launch_dph_anchor=anchor, max_bid=max_bid,
                            on_demand=od, ceiling_mult=ceiling_mult,
                            extra_bounds=(defense_cap,))
    if ceiling is None:
        return Rebid("stop", None,
                     "no launch price anchor, no --max-bid and no replacement-"
                     "market read — cannot derive a re-bid ceiling, refusing "
                     "to raise", None, rungs_left)
    # ...still named locally, because the refusal text below reports WHICH bound
    # was binding and the on-demand one has its own sentence.
    od_ceiling = effective_bid_ceiling(od)

    dec = bid_decision(market_min_bid, ceiling, od)
    target = dec.price
    step_price = round(last_bid * (1.0 + step), 3)
    nxt = step_price if target is None else max(step_price, target)
    nxt = round(min(nxt, ceiling), 3)

    if market_min_bid and market_min_bid > 0 and target is None:
        return Rebid("stop", None,
                     (dec.reason if dec.escalate else
                      f"no winnable bid on this machine: floor ${market_min_bid} "
                      f"against on-demand ${od} and a ${ceiling:.3f} ceiling")
                     + " — only a different box wins this back",
                     ceiling, rungs_left)
    if nxt - last_bid < BID_MIN_STEP:
        binding = ""
        if defense_cap is not None and abs(ceiling - defense_cap) <= _CEIL_EPS:
            binding = (f" — the binding bound is the JOB-AWARE defense ceiling "
                       f"(${defense_cap:.3f} = p_alt ${p_alt} x (1 + "
                       f"({setup_h:g}h setup + {lost_h:g}h lost "
                       f"work/{defend_mode}) / {defense_basis} runtime)): "
                       f"holding this box costs more than replacing it, so the "
                       f"replacement rung IS the rational defense") + defend_note
        elif od_ceiling is not None and ceiling <= od_ceiling + _CEIL_EPS:
            binding = (f" — the binding bound is the HARD on-demand ceiling "
                       f"({BID_CEILING_ONDEMAND_FRAC:g}x ${od}), so stop "
                       f"climbing and fall through to the replacement rung")
        return Rebid("stop", None,
                     f"re-bid ceiling ${ceiling:.3f} reached at rung "
                     f"{rungs_used}/{max_rungs}: the next rung would be "
                     f"${step_price:.4f} (+{step:.0%} on ${last_bid}) and only "
                     f"${max(0.0, ceiling - last_bid):.4f} of headroom remains"
                     + binding,
                     ceiling, rungs_left)
    if budget_usd is not None:
        budget_left = round(float(budget_usd) - float(spend_usd or 0.0), 4)
        if budget_left <= 0:
            return Rebid("stop", None,
                         f"budget exhausted (${spend_usd:.2f} of "
                         f"${budget_usd:.2f})", ceiling, rungs_left)
        if budget_left / nxt < min_runtime_h:
            return Rebid("stop", None,
                         f"${budget_left:.2f} left buys only "
                         f"{budget_left / nxt:.2f}h at ${nxt:.4f}/hr "
                         f"(< {min_runtime_h:g}h floor) — raising the bid would "
                         f"budget-park the box before it produced anything",
                         ceiling, rungs_left)
    return Rebid("rebid", nxt,
                 f"rung {int(rungs_used or 0) + 1}/{max_rungs}: ${last_bid} -> "
                 f"${nxt:.4f} (floor ${market_min_bid}, ceiling ${ceiling:.3f}"
                 + (f", job-aware defense cap ${defense_cap:.3f} on p_alt "
                    f"${p_alt}/{defense_basis} runtime/{defend_mode} work"
                    if defense_cap is not None else "")
                 + f") — keeping the WARM box rather than paying a "
                 f"replacement's setup" + defend_note, ceiling, rungs_left - 1)


def spot_breakeven(*, spot_dph, ondemand_dph, setup_h, expected_lifetime_h,
                   ckpt_interval_h=0.0):
    """PURE. Is spot actually cheaper than on-demand PER USEFUL HOUR, given
    what an eviction cycle costs on this lane?

    WIRED 2026-08-08 (autobid audit) as `replacement_decision`'s fifth
    structural prefer-on-demand trigger, fed this lane's OBSERVED inter-eviction
    lifetime from `replacement_history` and `SPOT_SETUP_H` — exactly the wiring
    point this docstring proposed while it was advisory. It is deliberately NOT
    wired into the LAUNCH bid: a fresh box has no observed lifetime, and the
    breakeven PRICE it would imply (on-demand x (1 - overhead/L), ~0.90 x
    on-demand at a 2 h lifetime) never binds against the 0.65 x on-demand cost
    cap `_bid_target` already applies. Wiring it there would have been a no-op
    dressed as a rail.

    The model: every spot lifetime L delivers (L - overhead) useful hours for
    a cost of spot_dph x L, where overhead = setup_h (boot, base pull, verify,
    merge — measured 11m35s = 0.193 h on the v11 eval lane, 2026-08-06) plus
    ckpt_interval_h / 2 (the expected re-done work since the last checkpoint;
    0 for lanes that bank results incrementally). On-demand delivers useful
    hours at list price with no cycle tax. So spot wins iff

        spot_dph x L / (L - overhead)  <  ondemand_dph
        ==  spot_dph / ondemand_dph  <  1 - overhead / L

    — the floated threshold rule, which this codifies with the checkpoint term
    added. L <= overhead is the LIVELOCK shape (the 27B lane, and the v11 eval
    cycle that moved the banked-row count 40 -> 40 for a full boot): spot's
    cost per useful hour is infinite and no price ratio rescues it.

    Returns (spot_is_cheaper, spot_cost_per_useful_h, od_cost_per_useful_h);
    (None, None, None) when any required input is missing/invalid — an unknown
    market is never a licence to conclude either way. Real-number anchors:
    v11 eval (L ~= 12 min, setup 11m35s) -> spot loses at ANY ratio; a calm
    24 h lifetime at the same setup -> spot wins whenever it is priced under
    ~99% of on-demand."""
    try:
        s, od = float(spot_dph), float(ondemand_dph)
        L = float(expected_lifetime_h)
        overhead = float(setup_h) + float(ckpt_interval_h) / 2.0
    except (TypeError, ValueError):
        return None, None, None
    if s <= 0 or od <= 0 or L <= 0 or overhead < 0:
        return None, None, None
    if L <= overhead:                       # livelock: nothing useful per cycle
        return False, math.inf, od
    spot_cost = s * L / (L - overhead)
    return spot_cost < od, round(spot_cost, 4), od


# --- how much the accumulated work is worth defending -------------------------
#
# FLEET_REVIEW_2026-08-14 item 7. The job-aware ceiling prices what LOSING this
# box costs the job, and until now the only lost-work input was
# `ckpt_interval_h / 2` — right for a training run, silent for everything else.
# Bench bundles are deliberately checkpoint-free ("nothing worth resuming"), so
# a config had no way to distinguish "un-checkpointed because the work is
# precious and we were sloppy" from "un-checkpointed because re-running it is
# cheaper than storing it". The ladder had to guess, and on 2026-08-14 it chased
# box 47694876 from $0.896 to $2.755/hr (~2x the replacement price) defending a
# w8 bench bundle that was ~100% done and disposable.
#
# `defend:` in job-config.yaml is that missing sentence, one of:
#
#   dear   — losing accumulated work costs real money; price it into the
#            ceiling. Training-shaped: resumable, expensive to redo.
#   cheap  — the wall time already spent is NOT worth defending; the lost-work
#            term collapses to zero and the ceiling keeps only the setup term
#            (a replacement still eats S, whatever the work is worth).
#
# Derivation when the key is absent — `checkpoint_s` present => dear, absent =>
# cheap — is chosen so every EXISTING config keeps its current numbers: an
# un-checkpointed job contributed L=0 before this landed and contributes L=0
# after it. The new pricing (`prior_runtime_h`, the accumulated wall time an
# un-checkpointed job loses in full) is reachable ONLY by writing
# `defend: dear` on a job with no `checkpoint_s`, i.e. by asking for it.
DEFEND_DEAR = "dear"
DEFEND_CHEAP = "cheap"
DEFEND_MODES = (DEFEND_DEAR, DEFEND_CHEAP)


def resolve_defend(defend=None, *, ckpt_interval_h=0.0):
    """PURE. The defend mode actually in force. An explicit, valid hint always
    wins; otherwise it is derived from whether the job checkpoints at all.

    An unrecognised string is treated as absent rather than raising — this runs
    inside a money-moving tick, and jobmeta validates the key at SUBMIT time
    (where a typo is an error the author sees), so the ladder's job is to keep
    deciding, not to crash on a field it did not understand."""
    if isinstance(defend, str) and defend.strip().lower() in DEFEND_MODES:
        return defend.strip().lower()
    try:
        ckpt = float(ckpt_interval_h or 0.0)
    except (TypeError, ValueError):
        ckpt = 0.0
    return DEFEND_DEAR if ckpt > 0 else DEFEND_CHEAP


def lost_work_hours(*, defend=None, ckpt_interval_h=0.0, prior_runtime_h=None):
    """PURE. `L` — the hours of work an eviction destroys, as the ceiling should
    price them. Returns `(hours, mode)` with mode in DEFEND_MODES.

      * cheap                     -> 0.0. Disposable work is defended at
                                     disposable prices.
      * dear, checkpoints         -> ckpt_interval_h / 2, the expected work
                                     since the last sync (the `spot_breakeven`
                                     convention, unchanged).
      * dear, no checkpoints      -> `prior_runtime_h`, the WHOLE accumulated
                                     wall time: with nothing synced there is no
                                     "since the last checkpoint", the run
                                     restarts from zero. Unknown/absent -> 0.0,
                                     which is what this case priced before.

    Never negative and never NaN-propagating: a garbage input reads as 0.0, so a
    malformed field can only ever make the ceiling TIGHTER."""
    mode = resolve_defend(defend, ckpt_interval_h=ckpt_interval_h)
    if mode == DEFEND_CHEAP:
        return 0.0, mode
    try:
        ckpt = float(ckpt_interval_h or 0.0)
    except (TypeError, ValueError):
        ckpt = 0.0
    if ckpt > 0:
        return ckpt / 2.0, mode
    try:
        prior = float(prior_runtime_h or 0.0)
    except (TypeError, ValueError):
        prior = 0.0
    return max(0.0, prior), mode


def defense_ceiling(*, p_alt, remaining_h=None, setup_h=SPOT_SETUP_H,
                    ckpt_interval_h=0.0, defend=None, prior_runtime_h=None):
    """PURE. The job-aware ceiling on DEFENDING a held spot box, priced by what
    losing it costs THIS job right now (AUTOBID_DESIGN "Next iteration", owner
    direction 2026-08-09):

        B_max = p_alt x (1 + (S + L) / R)

    `p_alt` is the best qualifying REPLACEMENT offer's expected $/hr — the one
    market signal that stays real for the whole life of a held box. A held
    machine's own `min_bid` is the price to displace US (#73's mirror) and is
    never an input here. `S` is the measured setup a replacement eats
    (SPOT_SETUP_H), `R` the remaining runtime, and `L` the lost work —
    `lost_work_hours(defend=..., ckpt_interval_h=..., prior_runtime_h=...)`,
    which is `ckpt_interval_h / 2` for a checkpointing job (the `spot_breakeven`
    convention), the whole accumulated wall time for an explicitly `dear`
    un-checkpointed one, and ZERO whenever the job says `defend: cheap`
    (FLEET_REVIEW_2026-08-14 item 7). A cheap defend therefore collapses this to
    the setup-only ceiling `p_alt x (1 + S/R)`: a replacement still eats the
    boot, but the wall time it discards is not billed to the defense.

    The shape this buys: with hours to go, B_max sits barely above the
    replacement price — moving is nearly as good as holding (R=4h at
    p_alt=$0.45 -> $0.477). Near completion it RISES — a few extra cents for
    the last half hour beats eating another 11.6-minute setup (R=0.5h ->
    $0.669). That is the inverse of a static multiple, which defends every box
    equally hard at every point in its life.

    `remaining_h` None or <= 0 falls back to SPOT_PRIOR_LIFETIME_H — the same
    "what the policy already asserts" prior `replacement_decision`'s breakeven
    rung uses when a lane has no observed lifetime — with basis "prior" in the
    return, so the journal can say whether the number ran on evidence or on an
    assumption (p_alt x ~1.30 at the shipped constants).

    Returns `(price, basis)` with basis in {"remaining", "prior"}; `(None,
    None)` when `p_alt` is missing/invalid — an unreadable replacement market
    is never a licence to defend at any price. This ceiling only ever TIGHTENS
    a caller's bound set: the hard on-demand ceiling, `max_bid`, and the
    strictly-below-on-demand rail all still bind downstream of it."""
    try:
        pa = float(p_alt)
    except (TypeError, ValueError):
        return None, None
    if pa <= 0:
        return None, None
    try:
        r = float(remaining_h)
    except (TypeError, ValueError):
        r = 0.0
    basis = "remaining"
    if r <= 0:
        r, basis = SPOT_PRIOR_LIFETIME_H, "prior"
    lost_h, _mode = lost_work_hours(defend=defend,
                                    ckpt_interval_h=ckpt_interval_h,
                                    prior_runtime_h=prior_runtime_h)
    try:
        overhead = float(setup_h) + lost_h
    except (TypeError, ValueError):
        return None, None
    if overhead < 0:
        return None, None
    return round(pa * (1.0 + overhead / r), 3), basis


def retention_plan(*, retention_h, present, now, storage_day_usd=None,
                   backstop_grace_h=RETENTION_BACKSTOP_GRACE_H):
    """PURE. What happens to the box we were just evicted from, once the
    replacement is renting and the queue has moved?

    Returns `Retention(action, deadline_ts, cost_usd, cost_hi_usd, reason)`:

      * `already_gone`  — the instance is no longer in the listing (host failure,
        or a spot host that reclaimed the stopped bid instance and its disk).
        Nothing to retain and nothing to destroy. This is NOT an error: on an
        evicted SPOT box retention is best-effort by construction (verified
        incident, box 44612403 — `herdd start` on a parked bid instance
        answered HTTP 404 `no_such_instance` minutes later).
      * `destroy`       — `retention_h <= 0`: the pre-2026-08-05 behavior,
        destroy now.
      * `retain`        — hold the box until `deadline_ts` so its unsynced local
        state can be salvaged. `cost_usd`/`cost_hi_usd` are the estimated
        storage bill for the window (a single number when the instance reported
        `storage_total_cost`, the measured $2.13-$4.62/day range otherwise).

    The cost is returned rather than compared against anything: retention is an
    owner decision, and this function's job is to make it DISCLOSED, not to
    second-guess it.

    **THE RETAIN COST IS CONDITIONAL AND THE CONDITION IS ENFORCED ELSEWHERE.**
    `cost_usd` prices ALLOCATED DISK ONLY, which is the true bill for a STOPPED
    box and roughly 1/20th of the truth for a running one (measured 2026-08-16:
    $0.0407/hr disclosed vs $0.8407/hr actually billed on box 47833510). A
    retained SPOT box does not stay stopped on its own — vast honours a queued
    start and auto-resumes a bid instance whose standing bid clears the floor —
    so `herdd._job_quiesce_box` stops it and pins its bid to
    `RETENTION_PARK_BID` at retain time, and `_job_retention_sweep` re-checks it
    every tick. Change one and this number becomes a lie; `retention_live_cost`
    below is what prices the lie when it happens."""
    if not present:
        return Retention("already_gone", None, 0.0, 0.0,
                         "the lost box is no longer in the instance listing "
                         "(host failure, or a spot host reclaimed it) — nothing "
                         "to retain, nothing to destroy")
    try:
        hours = float(retention_h)
    except (TypeError, ValueError):
        hours = REPLACEMENT_RETENTION_H
    if hours <= 0:
        return Retention("destroy", None, 0.0, 0.0,
                         "retention disabled (--replacement-retention-hours 0) "
                         "— destroying the lost box immediately")
    lo_dpd, hi_dpd = STORAGE_DPD_OBSERVED
    if storage_day_usd and storage_day_usd > 0:
        lo_dpd = hi_dpd = float(storage_day_usd)
    lo, hi = round(lo_dpd * hours / 24.0, 4), round(hi_dpd * hours / 24.0, 4)
    cost = (f"~${lo:.2f}" if lo == hi else f"~${lo:.2f}-${hi:.2f}")
    return Retention("retain", now + hours * 3600.0, lo, hi,
                     f"holding the lost box {hours:g}h for salvage of state "
                     f"that never reached B2 ({cost} of allocated-disk storage, "
                     f"which is the bill only while the box stays STOPPED)")


def retention_live_cost(dph, held_s, storage_day_usd=None):
    """PURE. What a RUNNING retained box has cost so far, and the multiple by
    which that overshoots the storage-only figure `retention_plan` disclosed.

    Returns `(usd, multiple)`; `(None, None)` when `dph` is unreadable — a box we
    cannot price is reported as unpriced, never as $0.00. `multiple` is None when
    no storage rate is known to compare against.

    Exists so the resurrection alarm quotes MONEY. "A retained box is running"
    reads as bookkeeping; "$0.14 spent, 20.6x the disclosed rate, and climbing"
    is the sentence that gets the box destroyed."""
    try:
        rate = float(dph)
        secs = float(held_s)
    except (TypeError, ValueError):
        return None, None
    if rate <= 0 or secs < 0:
        return None, None
    usd = round(rate * secs / 3600.0, 4)
    lo_dpd, _hi = STORAGE_DPD_OBSERVED
    if storage_day_usd and storage_day_usd > 0:
        lo_dpd = float(storage_day_usd)
    storage_hourly = lo_dpd / 24.0
    mult = round(rate / storage_hourly, 1) if storage_hourly > 0 else None
    return usd, mult


def _preferred_ceiling(on_demand, ondemand_frac=BID_CEILING_ONDEMAND_FRAC):
    """PURE. The "preferred" bid ceiling = `ondemand_frac x on_demand` (0.75 x
    on-demand). Historically ADVISORY under get-and-hold — a standing bid above it
    was legal and only signalled the handoff trigger (AUTOBID_DESIGN Phase 2).

    Since 2026-08-09 it is the SAME line `effective_bid_ceiling` enforces as a HARD
    clamp on every emitted bid, so a bid above it can now only come from before the
    clamp existed, from `herdd bid --price` by hand, or from a shrinking
    on-demand price. The alarm (`_preferred_ceiling_alarm`) is kept for exactly
    those cases. None when on-demand is unknown."""
    if on_demand and on_demand > 0:
        return round(ondemand_frac * on_demand, 3)
    return None


def _refresh_default_ceiling(st):
    """PURE-on-state: fold this tick's market floor into st['floor_samples'] and,
    unless the operator pinned --max-bid (st['explicit_max_bid']), recompute the
    default ceiling st['max_bid'] from the live on-demand price (get-and-hold, or
    strict-ceiling per st['strict_ceiling']) with the median-floor fallback. No
    I/O. Returns st."""
    mmb = st.get("market_min_bid")
    if isinstance(mmb, (int, float)) and mmb > 0:
        st.setdefault("floor_samples", []).append(mmb)
    if not st.get("explicit_max_bid"):
        st["max_bid"] = _default_max_bid(
            st.get("floor_samples", []), st.get("first_seen_dph"),
            on_demand=st.get("on_demand"), strict_ceiling=st.get("strict_ceiling", False))
    return st


def _preferred_ceiling_alarm(st):
    """PURE: (over: bool, pref: float|None). True when a LIVE box's standing bid
    exceeds the advisory preferred ceiling (BID_CEILING_ONDEMAND_FRAC x on-demand)
    — the get-and-hold ADVISORY (AUTOBID_DESIGN Phase 2). Silent under
    --strict-ceiling (there the preferred line IS the hard cap, so the bid can't
    exceed it).

    This is the ALARM, and since 2026-08-08 it is no longer the handoff trigger
    on its own — see `_handoff_trigger`. Telling an operator "your bid is over
    the preferred line" stays true and worth saying even when the bid is exactly
    what the bid policy just decided to pay."""
    if st.get("strict_ceiling"):
        return False, None
    pref = _preferred_ceiling(st.get("on_demand"))
    last_bid = st.get("last_bid")
    live = bool(st.get("present")) and st.get("actual_status") in LIVE_STATES
    over = bool(live and pref is not None and isinstance(last_bid, (int, float))
                and last_bid > pref + BID_MIN_STEP)
    return over, pref


def _handoff_trigger(st):
    """PURE: `(armed, pref, policy_target, reason)` — the dwell input for the
    ECONOMIC handoff, and the answer to the trigger-domain question the
    2026-08-08 22:17Z incident asked.

    A handoff means "our bid is higher than this workload should be paying, so
    move it". The old test for that was `bid > preferred_ceiling` — a raw
    bid/on-demand ratio. The 2026-08-08 autobid work
    (AUTOBID_AUDIT_2026-08-08.md) made that test wrong by making the bid POLICY
    legitimately occupy the same region: `_bid_target` applies a SURVIVAL CUSHION
    (BID_MIN_CUSHION_MULT x floor) that OUTRANKS the 0.65 x on-demand cost cap,
    because a bid that cannot survive is not cheap — it is a 12-15 minute setup
    bill for nothing. On a tight machine that cushion sits ABOVE the
    0.75 x on-demand preferred line, so the very bid the policy computed tripped
    the ceiling alarm and, five polls later, armed a migration.

    Live numbers from the incident, all from the fleetd log: our defend ladder
    walked the bid 2.697 -> 2.818 -> 3.100 -> 3.410, each step exactly
    BID_MIN_CUSHION_MULT x the floor it had just been handed, while the preferred
    ceiling stood at 0.75 x 3.876 = $2.907. Every one of those bids was the
    policy target for its tick. The handoff read the last one as "$3.41 over
    ceiling" and rented a second box.

    So the trigger is now BOTH tests, and the second one is the binding one: the
    bid must exceed the preferred ceiling AND exceed what the bid policy would
    put right now. A bid AT or BELOW the policy target can never by itself arm a
    handoff — if the policy says this box costs what it costs, the answer is
    get-and-hold (or the ordinary decay ladder when the floor recedes), never a
    voluntary second rental. When the target is unreadable (no market floor this
    tick) we cannot tell the two apart, so we do not arm: same fail-closed
    direction as every other handoff gate.

    `reason` names which rail is holding: "under_pref", "at_policy_target",
    "no_policy_target", or "over_policy_target" when it is armed."""
    over, pref = _preferred_ceiling_alarm(st)
    if not over:
        return False, pref, None, "under_pref"
    target = _bid_target(st.get("market_min_bid"), st.get("max_bid"),
                         st.get("on_demand"))
    if target is None:
        return False, pref, None, "no_policy_target"
    last_bid = st.get("last_bid")
    if not isinstance(last_bid, (int, float)) or last_bid <= target + BID_MIN_STEP:
        return False, pref, target, "at_policy_target"
    return True, pref, target, "over_policy_target"


def poll(s) -> Action:
    """PURE step function. One Action per tick from the §2 six-row taxonomy;
    precedence is the invariant (money-spending `relaunch` reachable only after
    every earlier guard clears). Never mutates its input. Only I/O-free helper
    used: runmeta.final_status (pure)."""
    view = s.get("view") or {}
    live = bool(s.get("present")) and s.get("actual_status") in LIVE_STATES
    fs = runmeta.final_status(view, status_marker=s.get("status_marker"),
                              instance_live=live)
    if fs["terminal"]:                                              # 1
        return Action("stop_terminal", f"terminal:{fs['status']}")
    # 2a/2b operator-intent rows are SUSPENDED while a handoff fence is open
    # (CUTOVER/DRAINING): the fence itself parks the primary (intended=stopped)
    # and the drain destroys it via this same CLI — reading either as operator
    # intent made the supervisor exit 49s after its own fence (live canary
    # handoff-canary-2, 2026-07-15). The driver's `fenced` gates suppress every
    # downstream money move those rows guarded, and a REAL operator park during
    # the short fence window is indistinguishable from ours (the primary is
    # already parked) — intent on the promoted box is honored again the tick
    # after the handoff resolves.
    #
    # 2a also carves out the UNDERBID PARK (defect D6, live canary
    # handoff-canary-3, 2026-07-15): when the standing bid sits below the live
    # market floor, vast itself parks the box WITH intended_status=stopped —
    # observed <47s after our own decay PUT dropped under the (D5-misread)
    # floor. That is a market/self-inflicted park, not operator intent; falling
    # through lands on the normal not-live rows, where _bid_action's rescue
    # re-raises the bid (with the D5 chunk-matched floor) and vast auto-resumes
    # the box. BUT the carve-out must not swallow a REAL operator park that
    # happens to coincide with an underbid: a `herdd stop` emits a cli-actor
    # `stopping` intent (non-terminal in the fold, so nothing else ends the run
    # while the box is present+stopped — 2026-07-18 review P1: rescue was
    # re-raising the bid and resuming the box against the operator), so a live
    # cli `stopping` intent wins over the underbid read. Vast's own park writes
    # no such event, and _last_stopping_actor clears the intent on any
    # resumed/launched/relaunched — the D6 rescue path is untouched.
    if s.get("intended_status") == "stopped" \
            and not s.get("handoff_fenced") \
            and (_actor_is_cli(s.get("stopping_actor"))
                 or not _underbid_parked(s)):                      # 2a
        return Action("stop_terminal", "operator_stop")
    if not s.get("present") and _actor_is_cli(s.get("stopping_actor")) \
            and not s.get("handoff_fenced"):                       # 2b
        return Action("stop_terminal", "operator_destroy")
    cap = _spend_time_exceeded(s)                                   # 2c HARD spend/
    if cap:                                                         # time caps fire
        return Action("stop_budget", cap)                          # EVEN while live
    if live:                                                        # 3
        bid = _bid_action(s)                          # money-moving: defend a live box
        return bid or Action("noop", "live")          # AFTER spend caps, BEFORE noop
    if s.get("not_live_streak", 0) < NOT_LIVE_DEBOUNCE:            # 4
        return Action("noop", "debounce_not_live")
    gr = _guardrail_exceeded(s)                                     # 5 (before re-issue)
    if gr:
        return Action("stop_budget", gr)
    bid = _bid_action(s)                              # money-moving: rescue an outbid
    if bid is not None:                               # box in place BEFORE any destroy
        return bid
    if view.get("status") != "evicted":                            # 6a
        return Action("emit_evicted", _evict_reason(s))
    if not s.get("backoff_ready"):                                 # 6b
        return Action("noop", "backoff")
    return Action("relaunch", "resume_after_evicted")              # 6c


# --------------------------------------------------------------------------- #
# handoff — pure decision core (T2; HANDOFF_DESIGN §6)
#
# A SEPARATE pure function, NOT a seventh poll() row: poll()'s six-row
# precedence is the money-safety invariant and its state is single-instance by
# contract; handoff is the lifecycle of a SECOND instance (the understudy).
# The driver runs poll() for the primary as today, THEN handoff_poll() for the
# migration. Both pure functions: no I/O, no mutation of input, `now` passed IN.
# --------------------------------------------------------------------------- #
HandoffAction = namedtuple("HandoffAction", ["kind", "reason"])

# pre-CUTOVER phases: the primary is still the sole writer and the understudy
# (if any) can be reaped without data loss. The deadline abort and the primary-
# evicted fast-cutover/abort both key off this set.
_HANDOFF_PRE_CUTOVER = ("ARMED", "LAUNCHING", "WARMING", "SYNCED")


def mk_handoff_state(*, phase="IDLE", over_ceiling_streak=0,
                     primary_iid=None, primary_bid=None, primary_on_demand=None,
                     primary_dph=None, primary_evicted=False, primary_gone=False,
                     understudy_iid=None, understudy_dph=None,
                     understudy_on_demand=None, understudy_status=None,
                     understudy_live_since=None, understudy_producing=False,
                     understudy_gone=False, drain_ts=None,
                     candidate_min_bid=None, candidate_on_demand=None,
                     remaining_wall_h=0.0, final_flush_seen=False, fence_ts=None,
                     ckpt_pulled_epoch=None, handoff_started_ts=None,
                     handoff_spend_usd=0.0, handoffs_done=0, cooldown_until=0.0,
                     budget_usd=None, spend_usd=0.0, now=0.0,
                     driver_can_complete=True, work_at_risk_h=0.0,
                     running_unresumable=0, min_running_eta_s=None,
                     ckpt_stale=False, unsafe_override=False):
    """Build the pure-relevant handoff sub-state (mirror of mk_poll_state).

    Held separately from the poll() state: `phase`
    (IDLE|ARMED|LAUNCHING|WARMING|SYNCED|CUTOVER|DRAINING|DONE|ABORT), the dwell
    counter, the primary/understudy identities and liveness, the fence flags
    (`final_flush_seen`, `ckpt_pulled_epoch`, `understudy_producing`,
    `understudy_gone`, `primary_gone`, `drain_ts` — when the post-cutover
    DRAINING clock started), the 2x-window accounting (`handoff_started_ts`,
    `handoff_spend_usd`, budget copies), the guardrail counters (`handoffs_done`,
    `cooldown_until`), the candidate market read (`candidate_min_bid`,
    `candidate_on_demand`, `remaining_wall_h`), and `now`.

    The WORK-AWARENESS fields (2026-08-08, tasks #62/#67) are the driver's
    reading of what this box is actually DOING, and they are what turns a purely
    financial decision into one that can see the workload it is about to move:

      * `driver_can_complete` — the DRIVER asserts it can carry a migration all
        the way to `complete`. Defaults True so the run lane (whose watch follows
        its box by LABEL and therefore survives the primary's destroy) is
        unchanged; the jobs lane passes what it actually knows. A feature that
        cannot finish must not be allowed to start (defect #61).
      * `work_at_risk_h` — hours of compute a migration would DISCARD (time since
        the last checkpoint, or since the attempt started when there is no
        checkpoint at all). Priced into the amortization; not a veto.
      * `running_unresumable` — how many RUNNING jobs have no checkpoint to
        resume from. Blocks the fence: parking a box under a job that cannot
        resume does not migrate the work, it deletes it (defect #62).
      * `min_running_eta_s` — the tightest estimated seconds-to-finish across the
        running jobs, or None when no progress signal exists. Tri-state on
        purpose: None means "unknown", never 0 and never infinite.
      * `ckpt_stale` — a running job that declares `checkpoint_s` has not synced
        within HANDOFF_CKPT_FRESH_MULT of it.
      * `unsafe_override` — the named escape hatch (HANDOFF_DESIGN §11). Skips
        the ARM preconditions ONLY; the fence rails still bind."""
    return {
        "phase": phase, "over_ceiling_streak": over_ceiling_streak,
        "primary_iid": primary_iid, "primary_bid": primary_bid,
        "primary_on_demand": primary_on_demand, "primary_dph": primary_dph,
        "primary_evicted": primary_evicted, "primary_gone": primary_gone,
        "understudy_iid": understudy_iid, "understudy_dph": understudy_dph,
        "understudy_on_demand": understudy_on_demand,
        "understudy_status": understudy_status,
        "understudy_live_since": understudy_live_since,
        "understudy_producing": understudy_producing,
        "understudy_gone": understudy_gone, "drain_ts": drain_ts,
        "candidate_min_bid": candidate_min_bid,
        "candidate_on_demand": candidate_on_demand,
        "remaining_wall_h": remaining_wall_h,
        "final_flush_seen": final_flush_seen, "fence_ts": fence_ts,
        "ckpt_pulled_epoch": ckpt_pulled_epoch,
        "handoff_started_ts": handoff_started_ts,
        "handoff_spend_usd": handoff_spend_usd,
        "handoffs_done": handoffs_done, "cooldown_until": cooldown_until,
        "budget_usd": budget_usd, "spend_usd": spend_usd, "now": now,
        "driver_can_complete": driver_can_complete,
        "work_at_risk_h": work_at_risk_h,
        "running_unresumable": running_unresumable,
        "min_running_eta_s": min_running_eta_s,
        "ckpt_stale": ckpt_stale, "unsafe_override": unsafe_override,
    }


def _handoff_headroom_ok(budget_usd, spend_usd, primary_dph,
                         candidate_target_bid, margin=0.0):
    """PURE. The refusal gate before any ARM (HANDOFF_DESIGN §1/§3): require enough
    --budget headroom to cover the worst-case 2x-box window. With
    `projected_2x_cost = (primary_dph + candidate_target_bid) x HANDOFF_WINDOW_H`,
    require `budget_usd - spend_usd > projected_2x_cost + margin`. A missing
    candidate market read (candidate_target_bid None) or an unknown primary dph
    can't be bounded -> refuse. budget_usd None (no --budget cap) -> unbounded
    headroom -> OK."""
    if candidate_target_bid is None or primary_dph is None:
        return False
    if budget_usd is None:
        return True
    projected_2x_cost = (primary_dph + candidate_target_bid) * HANDOFF_WINDOW_H
    return (budget_usd - (spend_usd or 0.0)) > projected_2x_cost + margin


def _handoff_candidate_ok(primary_dph, candidate_min_bid, candidate_on_demand,
                          remaining_wall_h, primary_on_demand,
                          work_at_risk_h=0.0):
    """PURE. The "cheaper-enough" / amortization filter (HANDOFF_DESIGN §2.3). A
    candidate offer qualifies only if BOTH hold:

      (1) its get-and-hold target bid (`_bid_target`: 1.2x the candidate floor,
          clamped under the CANDIDATE's own on-demand) sits at/under the
          PRIMARY's preferred ceiling (0.50 x primary on-demand) — the migration
          must land us genuinely under the line we are escaping. NOT the
          candidate's OWN 0.5x-od line: live 2026-07-15, every idle offer
          market-wide priced min_bid ~= 0.98 x its own dph_total (floors track
          on-demand across 3090/4090/5090/A100, top-200 by floor), so the own-od
          form was unsatisfiable ANYWHERE and dead-armed the feature; the owner
          plan's "cheaper box under 50% of on-demand" anchors to the box being
          escaped. Churning onto a hot-but-cheap box is bounded by
          HANDOFF_COOLDOWN_S + HANDOFF_MAX, and (2) still demands real savings;
      (2) conservative amortization: projected_savings > migration_overhead, with
            projected_savings  = (primary_dph - candidate_target) x remaining_wall_h
            migration_overhead = projected_2x_cost + work_at_risk_cost
                               = (primary_dph + candidate_target) x HANDOFF_WINDOW_H
                               + primary_dph x work_at_risk_h
          — short-remaining runs never hand off.

    `work_at_risk_h` (task #67, 2026-08-08) is the WORK the migration would throw
    away: hours since the running job's last checkpoint, or hours since its
    attempt began when it has never checkpointed. Redoing that work costs
    `primary_dph` an hour on whatever box redoes it, so it belongs in the
    overhead, priced in the same dollars as everything else. This is deliberately
    NOT a hard "no checkpoint => never migrate" rule: restarting one hour of work
    to save $4/hr for the next ten is a correct trade, and the honest inequality
    gets it right on its own. It defaults to 0.0 so a caller that cannot measure
    it is scored exactly as before — an UNDER-statement of overhead, which is why
    the fence-side rails (`_handoff_fence_hold`) are the ones that fail closed.
    Do NOT add the understudy's asset re-pull as a separate term: it happens
    inside the 2x-box window HANDOFF_WINDOW_H already prices (double count).

    Any missing input -> refuse (cost can't be bounded), `remaining_wall_h`
    INCLUDED and named explicitly. That last one was the 2026-08-08 hole: the
    contract held inside this function, but the callers defaulted the horizon to
    a flat 24.0 hours whenever no --wall-budget was set — which under fleetd is
    always (JOBS_POLICY_DEFAULTS seeds wall_budget=None) — so the gate was fed an
    invented number instead of a missing one and armed a voluntary migration off
    a healthy box ~90 s from the end of a cell. The callers now measure it
    (herdd._jobs_remaining_wall_h) or pass None, and None lands here. Refusing
    is the safe direction: a handoff is a VOLUNTARY cost optimisation, so a
    refusal costs a missed saving, never the workload.

    GPU/VRAM/CUDA/disk FIT is a driver-side search filter (build_search_query),
    not part of this pure test."""
    if None in (primary_dph, candidate_min_bid, candidate_on_demand,
                primary_on_demand, remaining_wall_h):
        return False
    target = _bid_target(candidate_min_bid, None, candidate_on_demand)
    if target is None:
        return False
    pref = _preferred_ceiling(primary_on_demand)
    if pref is None or target > pref:                       # (1)
        return False
    projected_savings = (primary_dph - target) * (remaining_wall_h or 0.0)
    migration_overhead = ((primary_dph + target) * HANDOFF_WINDOW_H
                          + primary_dph * (work_at_risk_h or 0.0))
    return projected_savings > migration_overhead           # (2)


def _handoff_dwell_satisfied(hs, now=None):
    """PURE. Has the bid been over the preferred ceiling long enough to ARM?

    The time-based twin of `_decay_dwell_satisfied`, and for the same reason: a
    dwell expressed in POLLS is silently re-tuned by a shorter tick, and this one
    guards a VOLUNTARY second rental. `over_ceiling_since` (the driver's start of
    the current over-ceiling run) makes it `HANDOFF_DWELL_S`; without it the
    legacy `HANDOFF_DWELL_POLLS` count still applies, so a state file written
    before the key arms exactly as it always did."""
    streak = hs.get("over_ceiling_streak", 0)
    if not streak:
        return False
    since = hs.get("over_ceiling_since")
    now = hs.get("now", 0.0) if now is None else now
    if since is not None and now is not None:
        return float(now) - float(since) >= HANDOFF_DWELL_S
    return streak >= HANDOFF_DWELL_POLLS


def _handoff_arm_refusal(hs):
    """PURE. The WORK-side preconditions on ARM, checked before a single dollar
    of market read matters. Returns the refusal reason, or None to proceed.

    Two rows, both of them lessons from the 2026-08-08 22:17Z incident:

      1. `driver_can_complete` — defect #61. Under fleetd a jobs watch ends at
         `inst is None` two ticks after the primary is destroyed, which is one
         tick BEFORE `handoff_poll` can return `complete`; the understudy was
         left with no watch and no budget cap, and the stray sweep adopted it as
         an uncapped `bare` box. A migration that cannot finish must not start,
         so the driver has to SAY it can finish and the default is that it cannot.
      2. `running_unresumable` — defect #62. The fence parks the primary out from
         under whatever is running on it. If a RUNNING job has no checkpoint,
         that is not a migration, it is a deletion with extra steps (the incident
         discarded a cell at `n_checkpoints: 0`). Refuse at ARM as well as at the
         fence: arming rents a second box, and paying for one to reach a fence we
         already know we must not open is pure waste.

    `unsafe_override` (HANDOFF_DESIGN §11) skips BOTH — and nothing else."""
    if hs.get("unsafe_override"):
        return None
    if not hs.get("driver_can_complete", True):
        return "driver_cannot_complete"
    if (hs.get("running_unresumable") or 0) > 0:
        return "unresumable_running_job"
    return None


def _handoff_fence_hold(hs):
    """PURE. Reasons to HOLD at SYNCED instead of opening the two-writer fence —
    the last check before the primary is parked and bid-pinned, and the only one
    that sees the state of the work at the moment of the park rather than at ARM.

    ARM is not a sufficient check on its own: the incident armed at 22:17:41 and
    fenced at 22:21:17, and the running cell became ~90 s-from-done inside those
    216 seconds. Rows:

      * `running_unresumable` — a RUNNING job with no checkpoint cannot survive
         the park at all (defect #62). Hard: no amount of saving justifies it.
      * `min_running_eta_s` under HANDOFF_FENCE_HOLD_ETA_S — the job is about to
         finish; hold, and the migration re-tests next tick against a queue that
         no longer has it. Tri-state: None (no estimate) never holds, because
         "unknown" is not "close" — the ARM-side horizon gate is what refuses on
         an unknown, and it does it before a second box is ever rented.
      * `ckpt_stale` — the job says it checkpoints every `checkpoint_s` and has
         not done so within HANDOFF_CKPT_FRESH_MULT of that. Its "resumable"
         claim is stale, so treat it as unproven.

    A hold is NOT a new half-open state and needs no new timer: SYNCED is a
    pre-CUTOVER phase, so `handoff_poll`'s precedence 2 aborts the whole attempt
    at HANDOFF_DEADLINE_S and reaps the understudy. `unsafe_override` does NOT
    skip these — the override buys the old ARM economics, never a fence over
    work that cannot come back."""
    if (hs.get("running_unresumable") or 0) > 0:
        return "no_resumable_checkpoint"
    eta = hs.get("min_running_eta_s")
    if isinstance(eta, (int, float)) and not isinstance(eta, bool) \
            and eta < HANDOFF_FENCE_HOLD_ETA_S:
        return f"eta_{int(eta)}s"
    if hs.get("ckpt_stale"):
        return "checkpoint_stale"
    return None


def _handoff_candidate_target(hs):
    """PURE: the understudy's would-be standing bid, from its captured market floor
    and on-demand — used for the headroom projection at ARM. None with no read."""
    return _bid_target(hs.get("candidate_min_bid"), None,
                       hs.get("candidate_on_demand"))


def handoff_poll(hs) -> HandoffAction:
    """PURE step function for the migration state machine (HANDOFF_DESIGN §6). One
    HandoffAction per tick; NEVER mutates its input; `now` comes IN via state.

    Precedence: abort/deadline first (primary-evicted rescue, then the 2x-window
    deadline), then the phase advance, then noop. Kinds: `noop`, `arm`,
    `launch_understudy`, `warm_wait`, `mark_synced`, `fence_primary`,
    `resume_understudy`, `drain_primary`, `complete`, `abort_reap`,
    `abort_unfence`.

    The primary stays governed by the untouched poll() throughout phases 1-5
    (defend/rescue/decay/relaunch keep firing there); handoff_poll only drives the
    SECOND instance and the fence between them."""
    phase = hs.get("phase", "IDLE")
    now = hs.get("now", 0.0)

    # --- precedence 1: primary evicted mid-handoff (§5). If the understudy is
    # already SYNCED, fast-CUTOVER to it (skip relaunch); else reap it and let
    # the normal poll() relaunch ladder run on the primary. Beats the deadline:
    # a synced understudy with a dead primary must be promoted, never aborted
    # (that would leave zero boxes — invariant I2).
    if hs.get("primary_evicted") and phase in _HANDOFF_PRE_CUTOVER:
        if phase == "SYNCED":
            return HandoffAction("resume_understudy", "fast_cutover")
        return HandoffAction("abort_reap", "primary_evicted")

    # --- precedence 2: the 2x-box window deadline. Any pre-CUTOVER phase still
    # open at HANDOFF_DEADLINE_S aborts (reap understudy, stay on primary).
    started = hs.get("handoff_started_ts")
    if phase in _HANDOFF_PRE_CUTOVER and started is not None \
            and now - started >= HANDOFF_DEADLINE_S:
        return HandoffAction("abort_reap", "deadline")

    # --- precedence 3: POST-cutover stall. DRAINING's only exit is
    # `understudy_producing`, so an understudy that dies (or never produces)
    # after the cutover wedges the migration permanently — precedence 2 covers
    # only PRE-cutover phases and the stall alarm deliberately forces nothing.
    #
    # This does NOT weaken the byte-safety invariant. That invariant is "never
    # DESTROY the primary without understudy proof-of-life", and the action here
    # is the opposite of destroying it: give the primary back. A dead understudy
    # cannot be a second writer, so unfencing is safe by construction.
    if phase == "DRAINING" and not hs.get("understudy_producing"):
        dead = bool(hs.get("understudy_gone")) or (
            hs.get("understudy_status") is not None
            and hs.get("understudy_status") not in LIVE_STATES)
        drain_ts = hs.get("drain_ts")
        late = (drain_ts is not None
                and now - drain_ts >= HANDOFF_DRAIN_DEADLINE_S)
        if dead or late:
            why = "understudy_died_draining" if dead else "drain_deadline"
            # With no primary left there is nothing to unfence — reap the
            # understudy and let the ordinary relaunch/replacement ladder run.
            kind = "abort_reap" if hs.get("primary_gone") else "abort_unfence"
            return HandoffAction(kind, why)

    # --- precedence 4: an OPEN fence that never commits its cutover (2026-08-08,
    # task #62). CUTOVER's exits were `resume_understudy` (post-flush, or at
    # HANDOFF_FENCE_TIMEOUT_S) and — when the retarget's old-ticket delete failed
    # — a `retarget_incomplete` latch that deliberately returns and stays put. So
    # the phase had a path IN with no path OUT, and while it sat there the
    # primary was parked with its standing bid pinned to HANDOFF_PARK_BID
    # ($0.001): the exact first-eviction-target configuration the same-day
    # autobid work removed everywhere else, held open indefinitely on a box we
    # intend to keep. Past HANDOFF_FENCE_UNWIND_S the fence UNWINDS — tickets
    # back, bid restored, primary resumed — so the pin can never outlive the
    # fence window. Ordered AFTER the flush timeout's phase branch would be too
    # late (that branch returns), so it lives here, and its own timer is longer
    # than the flush timeout by construction: unwind only fires once the normal
    # cutover has had its full chance and something else is wedged.
    if phase == "CUTOVER":
        fence_ts = hs.get("fence_ts")
        if fence_ts is not None and now - fence_ts >= HANDOFF_FENCE_UNWIND_S:
            return HandoffAction("abort_unfence", "fence_unwind")

    # --- phase advance
    if phase == "IDLE":
        if now < hs.get("cooldown_until", 0.0):
            return HandoffAction("noop", "cooldown")
        if hs.get("handoffs_done", 0) >= HANDOFF_MAX:
            return HandoffAction("noop", "max_handoffs")
        streak = hs.get("over_ceiling_streak", 0)
        if not _handoff_dwell_satisfied(hs, now):
            return HandoffAction("noop", "dwell" if streak else "idle")
        refusal = _handoff_arm_refusal(hs)        # WORK-side gate, before the money one
        if refusal:
            return HandoffAction("noop", f"precondition:{refusal}")
        target = _handoff_candidate_target(hs)
        if not _handoff_headroom_ok(hs.get("budget_usd"), hs.get("spend_usd", 0.0),
                                    hs.get("primary_dph"), target):
            return HandoffAction("noop", "headroom")
        if not _handoff_candidate_ok(hs.get("primary_dph"),
                                     hs.get("candidate_min_bid"),
                                     hs.get("candidate_on_demand"),
                                     hs.get("remaining_wall_h", 0.0),
                                     hs.get("primary_on_demand"),
                                     hs.get("work_at_risk_h", 0.0)):
            return HandoffAction("noop", "candidate_reject")
        return HandoffAction("arm", "dwell_satisfied")

    if phase == "ARMED":
        return HandoffAction("launch_understudy", "armed")

    if phase in ("LAUNCHING", "WARMING"):
        if hs.get("ckpt_pulled_epoch") is not None:
            return HandoffAction("mark_synced", "checkpoint_pulled")
        return HandoffAction("warm_wait", "booting")

    if phase == "SYNCED":
        # understudy staged; park the primary and open the two-writer fence —
        # but only if the work on the primary can survive the park. The fence is
        # the irreversible step (it interrupts whatever is RUNNING), so the state
        # of the work is re-read HERE and not trusted from ARM (defects #62/#67).
        # A hold is bounded for free by precedence 2 above.
        hold = _handoff_fence_hold(hs)
        if hold:
            return HandoffAction("noop", f"fence_hold:{hold}")
        return HandoffAction("fence_primary", "synced")

    if phase == "CUTOVER":
        # fence invariant: the understudy becomes a WRITER only AFTER the
        # primary's final flush is observed (else it resumes from a torn/older base).
        if not hs.get("final_flush_seen"):
            # CUTOVER escape hatch: if the fence has stood for HANDOFF_FENCE_TIMEOUT_S
            # with no flush, SIGTERM was never delivered (SIGKILL park) or the primary
            # already terminated (its trap emitted a terminal, not a final_flush).
            # Proceed from the last SYNCED checkpoint (<= one interval lost; spot
            # doctrine tolerates it) — the parked primary is bid-pinned + epoch-fenced
            # so it cannot become a second writer. fence_ts None (never fenced) noops.
            fence_ts = hs.get("fence_ts")
            if fence_ts is not None and now - fence_ts >= HANDOFF_FENCE_TIMEOUT_S:
                return HandoffAction("resume_understudy", "flush_timeout")
            return HandoffAction("noop", "await_flush")
        return HandoffAction("resume_understudy", "post_flush")

    if phase == "DRAINING":
        # destroy the primary only after the understudy is CONFIRMED producing.
        if not hs.get("understudy_producing"):
            return HandoffAction("noop", "await_understudy_ckpt")
        if not hs.get("primary_gone"):
            return HandoffAction("drain_primary", "understudy_producing")
        return HandoffAction("complete", "drained")

    return HandoffAction("noop", "idle")          # DONE / ABORT / unknown
