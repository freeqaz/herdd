#!/usr/bin/env python3
"""ladder_core — the bid ladder's STATE-TRANSITION layer, ONE copy, TWO lanes.

Where this sits in the stack
----------------------------
    bidpolicy.py     PURE decisions.   "given these numbers, what should happen"
    ladder_core.py   STATE TRANSITIONS. "given that, what does the lane's dict
                     now hold" — the per-tick bookkeeping that has to happen
                     identically on both lanes: the standing-bid echo window,
                     the self-referential-floor guard's episode latches, the
                     standing-bid reconcile, and the box-swap seam resets.
    herdd.py       I/O + lane glue.  offers reads, bid PUTs, launches,
                     queue reads, journals, watch verdicts.

Why it exists (the defect it kills)
-----------------------------------
The two supervise lanes — the run/serve lane (`_observe` / `_self_floor_guard` /
`_relaunch` / `_handoff_complete`) and the jobs lane (`job_supervise_tick` /
`_job_announce_eviction` / `_job_pull_condemn` / `_job_eviction_replace`) — drove
the SAME pure `bidpolicy` functions through TWO hand-written copies of the
bookkeeping around them. Every review cycle then found a fix that had landed in
one copy and needed manual mirroring into the other:

  * the `last_bid` reconcile-from-`dph_base` path (review 2026-08-10, F2/M3) —
    written twice, a month apart;
  * the self-floor episode reset on eviction — fixed once per lane, merge
    `168d2d0f`, after a frozen `self_floor_since` faked a "continuous" 30-minute
    floor-blind alarm across a 67-minute stopped gap;
  * the box-swap seam resets (L7/L8/#4/H1/H2) — five seams, five hand-copies.

Three of the nine 2026-08-10 review defects and two of the three 2026-08-14
defects were lane-parity gaps. `FLEET_REVIEW_2026-08-14.md` item 1 calls that
"the single biggest standing defect source in the bid code". This module is the
answer: a fix here is a fix, not a fix plus a twin to remember.

What is DELIBERATELY not here
-----------------------------
Lane-specific glue stays in `herdd.py`, because it is genuinely different work
and not a twin: the offers/instances I/O, the journal and event emitters (the
run lane emits `bid_self_floor` into the RUN's event log via `_sup_emit`; the
jobs lane journals `jobs_bid_self_floor` into the BOX's log via
`_job_ladder_journal`), the queue reads, the watch verdicts, and the
retarget-vs-relaunch shape of a replacement. Those reach this module through the
`LaneHooks` seam below, so the STATE machine is shared while the OBSERVATION
surface stays each lane's own.

Also deliberately not here: anything that decides. `bidpolicy` stays the sole
decision layer — this module reads its verdicts and records their consequences.

The two lanes' state dicts (`st` for the run lane, `jc` for the jobs lane) are
passed in as plain dicts and mutated in place. Several of their keys are
PERSISTED by fleetd (`REPLACEMENT_STATE_KEYS` / `RUN_STATE_KEYS`), so key names
here are a wire format: `bid_history`, `self_floor_at`, `self_floor_since`,
`self_floor_sustained_said`, `last_bid`, `first_seen_dph`, `evicted_announced`.
Renaming one silently drops durable state across a daemon restart.

Leaf module: stdlib only, one sibling import (`bidpolicy`, itself a pure leaf).
It does NOT import `herdd` — no cycle, and the core is testable without the
20k-line CLI (`test_ladder_core.py`).
"""

from __future__ import annotations

import bidpolicy


# --------------------------------------------------------------------------- #
# small shared helpers
# --------------------------------------------------------------------------- #
def num_dph(x):
    """A price as a float, or None for anything that isn't one. The lanes' one
    coercion for every dollar figure that arrives from JSON, argparse or a vast
    instance body."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


#: entries kept per box in `bid_history`. Must stay >= lag_s /
#: BID_RATE_LIMIT_S (3600/60 = 60 distinct prices in a window, worst case) or
#: the cap silently evicts in-window entries during exactly the aggressive
#: ladder activity the guard exists for; the margin covers restarts re-noting
#: mid-window. At the old 900 s window 24 was never binding (900/60 = 15).
BID_HISTORY_MAX = 72


def hist_field(entry, idx):
    """One field of a `bid_history` entry, tolerating both shapes it exists in:
    a tuple in memory, and a LIST after the watch record round-trips through
    fleetd's state.json (JSON has no tuples). Missing index -> None."""
    try:
        return entry[idx]
    except (TypeError, IndexError, KeyError):
        return None


def note_standing_bid(ctx, bid_now, machine_id, now,
                      lag_s=bidpolicy.BID_SELF_FLOOR_LAG_S):
    """Record what this box's standing bid IS, so the self-floor guard can still
    recognise it after it changes (`bidpolicy.market_floor_self_match`).

    Instrumenting the OBSERVATION rather than our seven `_put_bid_soft` call
    sites is deliberate, and it is the whole reason this is one function and not
    a bookkeeping obligation on every rung: the echo is a price the chunk was
    recently AT, whatever moved it. A defend raise, a ladder rung, a handoff
    park pin, a `herdd bid --price` typed by a human, a restore after a fenced
    cutover — all of them land here on the next tick, and none of them has to
    remember to.

    Entries are `[ts_first, price, machine_id, ts_last]`. The machine travels
    with the price because a replacement box inherits NO echoes: same watch,
    same `jc`, a different chunk whose floor has never been ours. Consecutive
    equal prices collapse into one entry whose `ts_last` refreshes every tick
    the price is still standing — the echo window and the prune BOTH run from
    `ts_last`, because the echo outlives the moment a price STOPS being our
    bid, not the moment it started being it. (The original oldest-ts-only
    shape measured the window from first sighting, so a bid that sat still for
    >=`lag_s` was pruned the instant it moved and its echo re-armed the very
    ratchet this guard exists to stop.) `ts_first` survives for telemetry.
    Trimmed to `lag_s` and `BID_HISTORY_MAX`."""
    if bid_now is None or not bid_now or bid_now <= 0:
        return
    if not machine_id:
        return    # a falsy key would collide across boxes: prices recorded on
                  # an old box under None could suppress a floor on a NEW box
                  # whose read also lacks machine_id (review 2026-08-10, F7)
    now = float(now)
    hist = list(ctx.get("bid_history") or [])
    if hist:
        prev_price = num_dph(hist_field(hist[-1], 1))
        if prev_price is not None \
                and abs(prev_price - float(bid_now)) <= bidpolicy.BID_SELF_FLOOR_EPS \
                and str(hist_field(hist[-1], 2)) == str(machine_id):
            e = list(hist[-1])              # unchanged price: refresh ts_last
            while len(e) < 4:
                e.append(e[0])              # legacy 3-field: ts_last = ts_first
            e[3] = now
            hist[-1] = e
            ctx["bid_history"] = hist
            return
    hist.append([now, float(bid_now), machine_id, now])
    cutoff = now - float(lag_s)
    ctx["bid_history"] = [e for e in hist
                          if (num_dph(hist_field(e, 3))
                              or num_dph(hist_field(e, 0)) or 0.0) >= cutoff
                          ][-BID_HISTORY_MAX:]


def bid_history_for(ctx, machine_id):
    """The recorded standing-bid series for THIS machine only — the argument
    `market_floor_self_match` expects. A retarget/replacement leaves the old
    machine's entries in place (they cost nothing and expire on their own) but
    they can never match, because a floor read is per machine. A falsy
    machine_id matches NOTHING — the recorder refuses the same key, and a
    None-for-None coincidence across a box change is exactly the collision
    that rule exists to prevent (review 2026-08-10, F7)."""
    if not machine_id:
        return []
    return [e for e in (ctx.get("bid_history") or [])
            if str(hist_field(e, 2)) == str(machine_id)]


# --------------------------------------------------------------------------- #
# seam resets — the episode latch, and the box swap
# --------------------------------------------------------------------------- #
def self_floor_reset(ctx):
    """Clear the self-floor guard's per-box state — the episode latch, the
    sustained-suppression clock and its said-once flag. Every box-swap seam
    (relaunch, eviction replacement, pull reschedule, handoff promotion) calls
    this, and so does every EVICTION announcement: a suppression episode is a
    property of a chunk we no longer hold.

    The eviction call is the one that was fixed twice, once per lane (merge
    `168d2d0f`): without it 47398836's floor-blind alarm fired ONE TICK after
    `rescue_recovered`, off a `self_floor_since` that had frozen across a
    67-minute stopped gap, so the "30 min continuous" mostly measured a box we
    did not hold."""
    ctx.pop("self_floor_at", None)
    ctx.pop("self_floor_since", None)
    ctx.pop("self_floor_sustained_said", None)


def box_swap_reset(ctx, *, reset_sticky_on_demand=True):
    """The state that CANNOT survive a box swap, on either lane.

    A swap is a relaunch, an eviction replacement, a pull reschedule or a
    handoff promotion — different rungs, same fact: the watch now points at a
    different contract, and possibly a different machine.

      * the self-floor EPISODE (`self_floor_reset`) — a suppression episode is a
        property of a chunk we no longer hold;
      * the echo window (`bid_history`) — a swap can land on the SAME machine we
        just left, where the old entries would suppress a genuine competitor
        floor at a price we recently held (review 2026-08-10, #4). Cleared, not
        filtered: the entries' machine key makes a DIFFERENT machine harmless
        already, so the only case that matters is the same-machine one;
      * the sticky on-demand clamp (`on_demand_last`) — it is per MACHINE, so
        carrying it over clamps the new box's rails against the old one's price.

    `reset_sticky_on_demand=False` exists for ONE caller — the run lane's
    `_handoff_complete`, which historically did not clear it. That is an
    UNFIXED lane-parity gap, recorded rather than silently repaired here: see
    AUTOBID_DESIGN.md §"One core, two lanes" divergence D2, and the pinning test
    in `test_ladder_core.py`, which fixes today's behavior so the repair, when
    it is made, is a deliberate one-line change with a test that flips.

    Everything else a seam clears is lane-specific and stays at the call site:
    the jobs lane's `floor_samples` / `decay_streak` / `rebid_rungs` /
    `rebid_refused` / `pref_alarmed` / `ceiling_escalated` / `evicted_announced`,
    the run lane's `evicted_pending` / `backoff_deadline` / `relaunch_count`."""
    self_floor_reset(ctx)
    ctx["bid_history"] = []
    if reset_sticky_on_demand:
        ctx.pop("on_demand_last", None)


# --------------------------------------------------------------------------- #
# the standing-bid belief: seed, echo-record, reconcile
# --------------------------------------------------------------------------- #
def reconcile_standing_bid(ctx, *, is_bid, true_bid, dph, machine_id, now,
                           put_ts_key, on_missing_base=None, on_reconcile=None):
    """Fold this tick's instance body into the lane's belief about its own
    standing bid, and record the price into the echo window. Returns `bid_now`,
    the anchor the caller's own seeding used to use.

    THE BID ANCHOR IS `dph_base`, NOT `dph_total` (= bid + storage) — see
    `_instance_standing_bid`. Seeding from the total puts the belief one storage
    sliver ABOVE the number vast reports back as the chunk's `min_bid`, and
    `market_floor_self_match`'s standing arm is an exact-equality test by
    design, so the guard could not recognise our own bid. `dph` stays the
    fallback so a body without `dph_base` behaves exactly as before.

    Four things happen, in this order, exactly as both lanes did them:

      1. `note_standing_bid` — every price this chunk's bid has taken, for the
         echo window. Bid boxes only.
      2. SEED `last_bid` when the lane has no belief yet.
      3. RECONCILE the belief to the box otherwise (review 2026-08-10, F2/M3):
         the standing bid can move without a successful PUT of ours — an
         out-of-band `herdd bid --price`, a PUT vast applied but answered
         5xx/timeout, a handoff pin, a daemon restart. `last_bid` drives
         `defend_at`, the rebid ladder and the guard's standing arm, and the
         history entry covering the TRUE price silently ages out of the echo
         window while that price still stands. The rate-limit window
         (`put_ts_key`) keeps us from racing our own in-flight PUT.
      4. seed `first_seen_dph`, the pre-floor ceiling anchor.

    `put_ts_key` is the lane's name for the rate-limit clock — `last_bid_put_ts`
    on the run lane, `last_bid_put` on the jobs lane. Both are persisted names
    and neither is renamed here (divergence D4).

    `on_missing_base()` fires when a BID box's body carries no `dph_base` at
    all; only the jobs lane has ever had that warning (divergence D3), and the
    once-per-watch latch stays at that call site. `on_reconcile(old, new)`
    prints the reconcile line."""
    bid_now = true_bid or dph
    if is_bid:
        note_standing_bid(ctx, bid_now, machine_id, now)
        if true_bid is None and on_missing_base is not None:
            on_missing_base()
    if ctx.get("last_bid") is None and is_bid and bid_now:
        ctx["last_bid"] = bid_now
    elif (is_bid and true_bid
          and now - (ctx.get(put_ts_key) or 0.0) > bidpolicy.BID_RATE_LIMIT_S
          and abs((num_dph(ctx.get("last_bid")) or 0.0) - true_bid) > 1e-9):
        if on_reconcile is not None:
            on_reconcile(ctx.get("last_bid"), true_bid)
        ctx["last_bid"] = true_bid
    if is_bid and bid_now and ctx.get("first_seen_dph") is None:
        ctx["first_seen_dph"] = bid_now               # pre-floor ceiling anchor
    return bid_now


# --------------------------------------------------------------------------- #
# the self-referential floor guard (task #73)
# --------------------------------------------------------------------------- #
class LaneHooks:
    """The lane's OBSERVATION surface for `self_floor_guard` — every side effect
    the guard has that is not a state transition.

    Defaults are silent no-ops, so a test (or a future third lane) can drive the
    pure state machine with nothing wired. The two shipped implementations live
    in `herdd.py` next to their emitters, which is what keeps
    `monkeypatch.setattr(herdd, "_sup_emit", ...)` working: the hook body
    resolves the emitter as a module global at CALL time.

    Every method is called at most once per tick, and only from the branch its
    name describes."""

    def scaled_read(self, ctx, market):
        """First tick of a rescaled-while-tenant read (the listing is mid-flap).
        Latched by the guard; the hook only says it."""

    def self_floor(self, ctx, *, market_min_bid, match, surviving_floor,
                   visible):
        """A row of the offers read was our own bid. Fires once per (value,
        kind) — the dedup key — never once per tick."""

    def floor_blind(self, ctx, *, since_s):
        """Every read for BID_SELF_FLOOR_SUSTAINED_S has been our own echo:
        there is no observable market on this machine, so defend AND decay are
        both parked and the standing bid is frozen where the ladder left it.
        Once per episode."""

    def episode_end(self, ctx, *, market):
        """A REAL competing read arrived while we are still the tenant — the
        suppression episode is over."""


def self_floor_guard(ctx, market, *, tenant, floors=None, scaled=False,
                     machine_id=None, now=None, hooks=None,
                     clear_latch_by_pop=False):
    """THE SELF-REFERENTIAL FLOOR (task #73), one copy for both lanes. Returns
    the floor to use this tick — `market`, a surviving sibling row, or None when
    every offer on this machine is our own standing bid read back.

    On a chunk we are the live tenant of, vast's `min_bid` is the price to
    displace the current tenant — us — so the offers read can hand back our own
    last PUT labelled "the market". Multiply it and the defend ladder chases
    itself: 2.697 -> 2.818 -> 3.100 -> 3.410 in five minutes on 47214941
    (1.10x, the survival cushion) and 1.338 -> 2.676 -> 2.944 in six on
    47218938 (2.00x, BID_TARGET_MULT), both on machines whose true floor was
    ~$1.33 (FLEETD_INCIDENT_2026-08-08).

    `tenant` is the caller's tenancy gate, NOT a liveness gate, and the two
    lanes compute it differently ON PURPOSE — see divergence D1 in
    AUTOBID_DESIGN.md §"One core, two lanes". On a STOPPED box the same equality
    means somebody ELSE holds the chunk at a price that happens to match what we
    were paying, which is a real market read and the rescue ladder's whole
    input.

    Treated as a FAILED read rather than as a floor — that is what it is, "the
    only offer on this machine is ourselves" — so it neither moves the bid nor
    reaches `floor_samples` (whose median is the fallback `max_bid`, and would
    otherwise ratchet the ceiling along with the bid).

    Suppressive only. It can lower no rail and raise no ceiling; the cushion,
    the cost cap and the on-demand clamps in `_bid_target` are untouched.

    ROW-level when `floors` is provided (review 2026-08-10, F3): the machine's
    offers can contain BOTH our rented chunk (min_bid = our own echo) and a free
    sibling chunk (a genuine floor). Filtering the collapsed min() threw the
    sibling away with the echo — a genuine floor that rose ABOVE our bid stayed
    invisible for as long as our lower echo was listed, unboundedly for a
    standing match. Now only the matching rows are dropped; the minimum of the
    surviving rows is returned as the market.

    `scaled=True` (F8) means no offer matched our exact GPU count and the number
    is a different chunk size's price stretched to ours — while we are the live
    tenant our own chunk IS a listing at our count, so its absence marks the
    listing mid-flap (a measured transient) and the read is treated as failed.

    `clear_latch_by_pop` picks how the episode-end branch clears the dedup
    latch: the run lane assigns `self_floor_at = None`, the jobs lane pops the
    key. Both read back as None; the flag exists only so neither lane's
    persisted state.json shape changes under the refactor (divergence D5)."""
    hooks = hooks or LaneHooks()
    now = now if now is not None else ctx.get("now")
    hist = bid_history_for(ctx, machine_id)
    rows = [f for f in (floors or ()) if f is not None]
    if not rows and market is not None:
        rows = [market]

    # (1) a rescaled floor while we are the tenant is a mid-flap listing, not a
    #     market: fail the read outright, before any matching is attempted.
    if tenant and scaled and rows:
        if not ctx.get("_scaled_floor_said"):
            ctx["_scaled_floor_said"] = True
            hooks.scaled_read(ctx, market)
        return None
    ctx.pop("_scaled_floor_said", None)

    # (2) row-level match: which listed floors are echoes of our own bid
    self_match, self_val, visible = None, None, []
    if tenant:
        for f in rows:
            m = bidpolicy.market_floor_self_match(
                f, ctx.get("last_bid"), bid_history=hist, now=now)
            if m is not None:
                if self_match is None:
                    self_match, self_val = m, f
            else:
                visible.append(f)

    if self_match is not None:
        # Dedup on (value, kind), not value alone: on a machine whose every
        # listed chunk is ours (no free sibling to flap the read), a whole echo
        # episode holds one market value, and a value-only key would journal
        # only its FIRST match — the standing->prior transition after a bid
        # move, the one carrying a real `matched_age_s`, would be silent. The
        # field distribution of ages is how the window gets sized by
        # measurement (it already re-sized it once: 900 s -> 3600 s,
        # 2026-08-14, on a censored-at-the-boundary 887.6 s field max). The key
        # is a LIST, not a tuple, so it stays well-defined if this state ever
        # persists through state.json (json returns tuples as lists, and
        # list != tuple).
        key = [self_val, self_match.kind]
        if ctx.get("self_floor_at") != key:
            ctx["self_floor_at"] = key
            hooks.self_floor(ctx, market_min_bid=self_val, match=self_match,
                             surviving_floor=(min(visible) if visible else None),
                             visible=visible)
        if visible:
            # our own row is dropped; the machine's other chunks are still a
            # real market — defend/decay stay armed against THEM, and the
            # floor-blind clock does not run (there IS a signal)
            ctx.pop("self_floor_since", None)
            ctx.pop("self_floor_sustained_said", None)
            return min(visible)
        if ctx.get("self_floor_since") is None:
            ctx["self_floor_since"] = now
        elif (now - ctx["self_floor_since"] > bidpolicy.BID_SELF_FLOOR_SUSTAINED_S
              and not ctx.get("self_floor_sustained_said")):
            # every read for BID_SELF_FLOOR_SUSTAINED_S has been our own echo:
            # there is NO observable market on this machine (all listed chunks
            # are ours), so defend AND decay are both parked. Correct, but not
            # silently — an already-ratcheted bid stays frozen high with
            # nothing to decay against (review 2026-08-10, #6).
            ctx["self_floor_sustained_said"] = True
            hooks.floor_blind(ctx, since_s=now - ctx["self_floor_since"])
        return None

    # (3) The episode ends only on a REAL competing read while we are still the
    #     tenant. `self_match is None` also covers a FAILED offers read and
    #     every not-tenant tick — clearing the latch there printed "$None is a
    #     real competing read" and re-journaled a phantom episode start into the
    #     matched_age_s distribution on the next match (review 2026-08-10,
    #     #8/L6).
    if market is not None and tenant and ctx.get("self_floor_at") is not None:
        if clear_latch_by_pop:
            ctx.pop("self_floor_at", None)
        else:
            ctx["self_floor_at"] = None
        ctx.pop("self_floor_since", None)
        ctx.pop("self_floor_sustained_said", None)
        hooks.episode_end(ctx, market=market)
    return market
