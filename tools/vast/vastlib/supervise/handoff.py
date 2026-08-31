"""vastlib.supervise.handoff — the migration ladder: retire one box onto another.

Why this module exists
----------------------
A handoff is the only thing this system does that deliberately runs **two boxes
at once** and then destroys one of them. Everything here exists to keep that
window bounded and to keep exactly one writer on the checkpoint stream at a
time. The cluster was scattered across four regions of `herdd.py` (8234-9236
run lane, 12844-12935 jobs constants + per-job B2/signals, 16228-17020 jobs
lane) and read as three unrelated groups of `_handoff_*` names; collected here
the shape is one ladder, mirrored twice:

    IDLE -> ARMED -> LAUNCHING -> WARMING -> SYNCED -> CUTOVER -> DRAINING -> DONE
                                                    \\-> ABORT (reap | unfence)

The **pure** step function is `bidpolicy.handoff_poll` (Zone S) and it stays
there. This module is the observation half (what is true right now) and the
effect half (execute one `HandoffAction`), with nothing in between: every
decision is `handoff_poll`'s, every snapshot is `bidpolicy.mk_handoff_state`'s
dict, and no gate is re-implemented here.

Two lanes, deliberately NOT unified
-----------------------------------
The run lane (`_handoff_*`, keyed on `run_id`, moves a training run) and the
jobs lane (`_job_handoff_*` / `_handoff_job_*`, keyed on the primary box iid,
moves a ticket queue) are mirrors. Plan §5's NOTE and v1 §7 pin **six**
divergences as deliberate; a money-path unification is its own owner-called
change, not a side effect of moving files. The ones that live in this file:

* **the double-bill id guard.** `_handoff_accrue` compares
  `st["instance_id"] == hf["understudy_iid"]` RAW; `_handoff_job_accrue`
  compares `str(...) == str(...)`. The jobs lane spells box ids as strings
  everywhere (queue path segment, ticket `box` field, event `instance_id`,
  `--box` argv) and the run lane does not.
* **six work-awareness kwargs.** `_handoff_job_build_state` passes
  `driver_can_complete` / `work_at_risk_h` / `running_unresumable` /
  `min_running_eta_s` / `ckpt_stale` / `unsafe_override`;
  `_handoff_build_state` passes none of them and takes
  `mk_handoff_state`'s defaults, where `driver_can_complete` defaults **True**.
  The jobs lane fails CLOSED on that key by design (defect #61).
* **the primary-iid pin in the tick.** The run lane tests
  `st["instance_id"] not in (None, hf["understudy_iid"])`; the jobs lane tests
  `str(jctx["iid"]) not in (None, "None", str(hf["understudy_iid"]))`.
* **journals.** The jobs lane writes an in-memory decision queue
  (`journal._job_handoff_journal`) that fleetd drains into `fleet log`; the run
  lane has no such queue and only emits to B2. Every jobs-lane phase is
  journalled and no run-lane phase is.
* **cutover mechanics.** The run lane RELABELS the understudy (`run:<ID>`); the
  jobs lane RETARGETS each ticket and then relabels. Only the jobs lane can end
  a cutover INCOMPLETE (`retarget_incomplete`), and only it suppresses
  `understudy_producing` while that latch is set.
* **the reap-on-exit contract.** The run lane reaps only PRE-cutover
  (`_HANDOFF_PRE_CUTOVER`); the jobs lane also unwinds an OPEN fence at CUTOVER
  (retarget-back + unfence) and reaps `_HANDOFF_PRE_CUTOVER + ("CUTOVER",)`.

Keep the mirror a mirror. Reading one lane's function next to the other's is
the point of the file order below.

Frozen contracts this module is on the wrong side of
----------------------------------------------------
* **The 35-key `hs` dict.** `bidpolicy.mk_handoff_state` takes 35 keyword-only
  params and returns a plain dict with the same 35 keys;
  `handoff_poll` / `_handoff_fence_hold` / `_handoff_candidate_ok` read it by
  KEY, in Zone S, which this branch does not port. Both builders here must keep
  emitting exactly those keys, by keyword, as a dict. A typed dataclass on this
  side would have to serialize to that dict AT the call, never replace it.
* **`hf` is read across a process boundary.** `fleetd.Fleet._handoff_in_flight`
  does `rt["hf"].get("phase")` / `.get("understudy_iid")` — the defect-#61
  keep-alive predicate. Those two string keys are frozen for as long as
  `fleetd.py` is unported (plan §8 step 5). The whole `_init_*_handoff_state`
  key set is treated as frozen for the same reason: `stall_alarmed`,
  `retarget_incomplete` and `prefence_bid` are written by drivers in OTHER
  modules (`replacement`, `run_lane`, `job_lane`) that do not own the schema.
* **`hf` is mutated IN PLACE and callers hold the reference across ticks.**
  `_handoff_reset` / `_job_handoff_reset` do `hf.clear()` + `hf.update(fresh)`
  precisely so a caller (and fleetd's `rt["hf"]`) keeps seeing the same object.
  Every signature here therefore takes `MutableMapping[str, Any]`, never a
  slotted dataclass: `.clear()` / `.update()` / `.setdefault()` are load-bearing
  and a "cleaner" rebind would silently stop resetting for every existing
  holder.
* **B2 marker paths.** `runs/<RUN_ID>/handoff/<epoch>.json|promoted` and
  `jobs/<JOB_ID>/handoff/<epoch>.json|promoted` are a wire contract with the
  box-side guards `onstart/train.sh:286` and `onstart/jobd.sh:221`
  (`_handoff_epoch_stale`), which are Zone S and never ported. The producer
  side is `_handoff_b2_write` / `_handoff_job_b2_write` below.

What is deliberately NOT here
-----------------------------
* **No decisions.** `handoff_poll`, `_handoff_candidate_ok`,
  `_handoff_fence_hold`, `_handoff_headroom_ok`, `_handoff_arm_refusal`,
  `_handoff_candidate_target`, `_handoff_trigger` and every `HANDOFF_*`
  threshold stay in `bidpolicy.py`. This module reads them; it never second-
  guesses them and never re-derives one of their numbers.
* **No launch bodies, no offer ranking.** `_handoff_understudy_body`,
  `_handoff_pick_offer`, `_job_understudy_offer` and `_launch_job_understudy`
  are `supervise.replacement`'s. That module imports THIS one (for
  `_handoff_primary_dph`), so the four call sites below import it
  function-locally — the one direction that would otherwise be a cycle. They
  are still module-attribute calls (`replacement._launch_job_understudy(...)`),
  so the patch idiom survives.
* **No tick loop.** `supervise_tick` / `job_supervise_tick` own the poll
  machinery, the budget guards and the ORDER; `_handoff_tick` /
  `_job_handoff_tick` are the one step they call afterwards. Money-moving
  handoff steps sit AFTER poll()'s spend/budget guards by construction.
* **No emitters.** `journal._sup_emit` / `journal._job_handoff_emit` /
  `journal._job_handoff_journal` came out first (plan §8 step 2) and are called
  in module-attribute form so all 28+ `monkeypatch.setattr` sites keep steering.
* **No stall FORCING.** `_handoff_stall_alarm` alarms once and latches. The
  primary destroy stays gated on understudy proof-of-life — the byte-safety
  invariant — so a wedged migration is a loud alert, never a timer-driven
  teardown.

Provenance: moved from `tools/vast/herdd.py` at rev `a1f2c8a5`
(plan §8 step 4, 2026-08-16), behavior-preserving — bodies verbatim,
annotations added. Manifest: `tools/vast/.port_manifests/sup-handoff.json`,
plus the 24 effectful drivers that manifest flagged as claimed by NO manifest
(integrator ruling: half-migrating the knot is worse than the "pure only" cut
plan §5 describes).
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import time
from typing import Any, Callable, Iterable, MutableMapping, Sequence

import ladder_core

from vastlib.boxes import health, lifecycle, ssh
from vastlib.core import api, fmt, labels, models
from vastlib.jobs import control as jobs_control
from vastlib.jobs import view as jobs_view
from vastlib.launch import launch as launchmod
from vastlib.launch import spec as launch_spec
from vastlib.market import pricing
from vastlib.storage import b2
from vastlib.supervise import journal

# EAGER, and it must stay eager: `_HANDOFF_REFUSAL_NOTES` below interpolates
# `bidpolicy.HANDOFF_CKPT_FRESH_MULT` into an f-string at MODULE-EXEC time.
# Making this import lazy changes the text of a shipped operator message.
import bidpolicy
import jobmeta

# --------------------------------------------------------------------------- #
# CROSS-RING SEAM — new code, no `moved-from:` marker (`vastlib/README.md` §2
# rule 7). `_confirm_gone` is a symbol this cluster CALLS whose real home is
# another ring. At step 4 it RAISED, because a second implementation of a
# destroy-confirmation probe is exactly the fork this refactor exists to kill
# and a raise is a loud failure at the one moment it matters (step 6, when
# `herdd.py` starts calling this module) instead of a silent divergence.
#
# It is CLOSED at step 6d: the home landed (`boxes/lifecycle.py`), so the stub
# became the one-line forwarder below rather than a copy. Zero raising seams
# remain in this module.
# --------------------------------------------------------------------------- #

# The forwarder, and why it is one and not a rebind. The destroy-confirmation probe belongs
# beside `lifecycle._destroy_soft` — `core/result.py` documents the pair — and
# that is where the one body lives (ported step 3, `moved-from:` marker there).
# This module keeps the ATTRIBUTE because 7 `monkeypatch.setattr` sites steer
# the name at the supervise seam and would not be seen through an inline
# `lifecycle._confirm_gone(...)` call; `replacement.py` closes its twin stub the
# same way. Two attributes, one body, and `lifecycle._confirm_gone` still
# resolves at CALL time so a patch THERE also steers these four call sites.
def _confirm_gone(iid: object, tries: int = 6) -> bool:
    """True once vast no longer reports the instance present (destroy confirmed).
    Treats {"instances": null}/None (HTTP 200 for a gone box) and HTTP 404 as
    gone. Enforces destroy-husk-before-relaunch (never launch a twin over a live
    husk)."""
    return lifecycle._confirm_gone(iid, tries)


# --------------------------------------------------------------------------- #
# BOUND SEAMS (step 5, 2026-08-16). `supervise` and `jobs` are SIBLINGS in the
# §5 DAG (`supervise : jobs : fleet : workflows`, `:` = non-independent), so this
# ring may import that one — the raising stubs these three replace were only ever
# waiting for the module to exist.
#
# They stay as FUNCTIONS rather than `_live_iids_set = jobs_view._live_iids_set`
# aliases for two reasons: `test_vastlib_supervise_handoff.py` patches
# `handoff._box_lifecycle_soft` (an alias would still be patchable, but a patch
# on `jobs.view` would not steer this module), and the forwarder is the
# module-attribute call form plan §8b requires in BOTH directions.
# --------------------------------------------------------------------------- #

def _live_iids_set() -> Any:  # noqa: ANN401 — set[str] | joblocal's own set type
    """Liveness injection for a job fold — `vastlib.jobs.view._live_iids_set`.

    Its one call site here is inside the `try/except` around `jobmeta.read_job`,
    so a failure degrades to "cache not refreshed" — the same outcome a B2 read
    failure produces."""
    return jobs_view._live_iids_set()


def _box_lifecycle_soft(iid: object) -> Any:  # noqa: ANN401 — jobmeta.read_box view
    """The per-box lifecycle fold — `vastlib.jobs.view._box_lifecycle_soft`.

    Folds `jobs/nodes/<iid>/events/` to at least `{parked, drained_pending}` and
    never raises."""
    return jobs_view._box_lifecycle_soft(iid)


def cmd_job_retarget(a: argparse.Namespace) -> Any:  # noqa: ANN401 — CLI cmd, returns None
    """Move a job's ticket — `vastlib.jobs.control.cmd_job_retarget`.

    Writes the new ticket then DELETEs the old one, and `sys.exit`s (SystemExit)
    on a terminal/no-ticket/poisoned-target refusal — both cutover paths below
    catch that. The poisoned-target case (the understudy already went terminal on
    this JOB_ID, so its jobd would skip the ticket forever) is a cutover the
    supervisor must NOT record as moved; skipping and reporting is correct."""
    return jobs_control.cmd_job_retarget(a)


# --------------------------------------------------------------------------- #
# Constants. All three are read across lanes, so they sit together rather than
# beside their single reader.
# --------------------------------------------------------------------------- #

# The phases where the two-writer FENCE is open (primary parked, understudy
# taking over): the run lane suppresses poll()'s primary churn (bid/evict/
# relaunch) there so the untouched ladder cannot fight the deliberate
# retirement of the primary. Read by `_handoff_tick`, `_job_handoff_tick` and
# by `run_lane.supervise_tick`.
# moved-from: herdd._HANDOFF_FENCE_OPEN
_HANDOFF_FENCE_OPEN = ("CUTOVER", "DRAINING")

# The jobs-lane understudy proof-of-life event set: a lifecycle event of one of
# these kinds, carrying the understudy's instance_id, is what gates the primary
# destroy. Sole reader is `_handoff_job_signals`.
# moved-from: herdd._HANDOFF_JOB_PRODUCING
_HANDOFF_JOB_PRODUCING = ("checkpoint", "started", "resumed", "claimed", "done")

#: What each `precondition:`/`fence_hold:` reason means in one operator-facing
#: line. The reason strings come from bidpolicy's pure gates; keeping the prose
#: HERE keeps the pure core free of presentation and keeps `fleet log` readable.
# moved-from: herdd._HANDOFF_REFUSAL_NOTES
_HANDOFF_REFUSAL_NOTES = {
    "driver_cannot_complete":
        "the driver cannot carry a migration to completion, so it is not allowed "
        "to start one (defect #61: a fleetd jobs watch ends at `inst is None` "
        "before `complete` runs, leaving the understudy unwatched and uncapped)",
    "unresumable_running_job":
        "a RUNNING job has NO checkpoint to resume from — migrating would "
        "discard the attempt, not move it (defect #62)",
    "no_resumable_checkpoint":
        "fence HELD: parking the primary now would interrupt a RUNNING job with "
        "no checkpoint to resume from",
    "checkpoint_stale":
        "fence HELD: a RUNNING job's last checkpoint is older than "
        f"{bidpolicy.HANDOFF_CKPT_FRESH_MULT}x its own checkpoint_s, so its "
        "resumability is unproven",
}


# --------------------------------------------------------------------------- #
# The mutable per-run / per-box sub-state carried across ticks.
#
# NOTE FOR THE INTEGRATOR: `sup-state.json` also claims these two factories for
# `supervise/state.py`. They are defined here because (a) this wave's brief
# forbids driver modules from importing `supervise.state`, and (b) the two
# `*_reset` functions below depend on the factory returning a PLAIN DICT —
# `hf.clear()` + `hf.update(fresh)` is the in-place reset every holder of the
# reference (including fleetd's `rt["hf"]`) relies on. If `state.py` becomes
# the home, this module must consume it as `state._init_handoff_state()`, and
# a slotted dataclass there needs an explicit `reset_in_place()` first.
# --------------------------------------------------------------------------- #

# moved-from: herdd._init_handoff_state
def _init_handoff_state() -> dict[str, Any]:
    """The mutable per-run handoff sub-state carried across ticks (distinct from
    the pure mk_handoff_state snapshot built each tick). IDLE until the dwell
    arms it."""
    return {
        "phase": "IDLE", "over_ceiling_streak": 0,
        # the dwell is HANDOFF_DWELL_S of wall clock, so the run's START is the
        # state that matters; the counter beside it is the legacy fallback
        "over_ceiling_since": None,
        "primary_iid": None, "understudy_iid": None, "understudy_dph": None,
        "understudy_on_demand": None, "understudy_status": None,
        "understudy_live_since": None, "understudy_producing": False,
        "candidate_min_bid": None, "candidate_on_demand": None,
        "chosen_offer": None, "final_flush_seen": False,
        "epoch": None,                                # T4b write-generation, set at ARM
        "fence_ts": None,                             # wall-clock the fence opened (CUTOVER);
                                                      # feeds the flush-timeout and the
                                                      # DRAINING-stall clocks
        "stall_alarmed": False,                       # DRAINING-stall alarm once-flag
        "cutover_ts": None,                           # compact-UTC promotion moment
                                                      # (producing-signal anchor on the
                                                      # no-SIGTERM flush_timeout path)
        "ckpt_pulled_epoch": None, "handoff_started_ts": None,
        "handoff_spend_usd": 0.0, "handoffs_done": 0, "cooldown_until": 0.0,
        "primary_gone": False,
    }


# moved-from: herdd._init_job_handoff_state
def _init_job_handoff_state() -> dict[str, Any]:
    """The mutable per-box jobs-handoff sub-state carried across ticks. Extends the
    run-lane sub-state (same pure handoff_poll consumes it) with the jobs-only
    ticket bookkeeping: the JOB_IDs to move (`pending_jobs`), which of them were
    RUNNING at the fence (`running_jobs` — the final_flush wait set), and any old
    ticket whose delete failed at cutover (`retarget_incomplete`, §5)."""
    hf = _init_handoff_state()
    hf["pending_jobs"] = []
    hf["running_jobs"] = []
    hf["retarget_incomplete"] = None
    return hf


# --------------------------------------------------------------------------- #
# PURE: prices and pins. No I/O, no mutation.
# --------------------------------------------------------------------------- #

# moved-from: herdd._handoff_primary_dph
def _handoff_primary_dph(st: MutableMapping[str, Any]) -> float | None:
    """The primary's current PAID rate for the §2.3 amortization math: the live
    standing bid (`last_bid`, rolled forward by every _do_bid_move) if known,
    else the launch/relaunch cost basis (`dph_total`). None disables handoff."""
    return models._num_dph(st.get("last_bid")) or models._num_dph(st.get("dph_total"))


# moved-from: herdd._prefence_bid
def _prefence_bid(last_bid: object, dph: object) -> float | None:
    """PURE. What `fence_primary` records as the bid to restore, or None.

    The recorded value must never BE the fence pin (2026-08-08, task #62). A
    supervisor that dies mid-fence and restarts reconciles into the migration
    with `prefence_bid` lost and the primary already parked at HANDOFF_PARK_BID —
    so the box's observed `dph_total` IS $0.001, and a naive re-record would
    memorise the pin as the thing to restore. The unwind would then dutifully
    put the box back at a bid it can never win a market with, which is the exact
    wedge the unwind exists to prevent, laundered through a restart.

    None means "unknown", and the unwind's policy-target fallback owns it."""
    v = models._num_dph(last_bid) or models._num_dph(dph)
    return v if (v is not None and v > bidpolicy.HANDOFF_PARK_BID) else None


# moved-from: herdd._handoff_park_bid
def _handoff_park_bid(st: MutableMapping[str, Any]) -> float:
    """The bid we pin a fenced primary to so vast cannot auto-resume it (§4). Always
    the API-minimum HANDOFF_PARK_BID — by construction below any live market floor,
    robust to a floor DROP in the fence->drain window (a floor-relative pin is not).
    A helper so the value is asserted in one place."""
    # `st` is accepted and ignored ON PURPOSE: both lanes call this, the run
    # lane with `st` and the jobs lane with `jctx`, and the pin is deliberately
    # NOT context-relative. Dropping the parameter would touch call sites in
    # two modules this port does not own.
    return float(bidpolicy.HANDOFF_PARK_BID)


# --------------------------------------------------------------------------- #
# RUN LANE — B2 markers and box-side proofs.
#
# The two write helpers are the PRODUCER side of a wire contract with Zone S
# shell (`onstart/train.sh`, `onstart/jobd.sh`): the marker PATHS and the
# `<epoch>.json` / `promoted` names are read by `_handoff_epoch_stale` and the
# understudy dead-man watchdog on the box. Changing a path here silently
# disables a guard running on a rented machine.
# --------------------------------------------------------------------------- #

# moved-from: herdd._handoff_b2_write
def _handoff_b2_write(run_id: object, rel: str, body: str,
                      dry_run: bool = False) -> bool:
    """Write a handoff coordination marker to runs/<RUN_ID>/handoff/<rel> on B2 —
    the PRODUCER side of T6's box-side guards (onstart/train.sh consumes these):

      * `<epoch>.json` (at ARM) — the monotonic write-generation marker T6's
        `_handoff_epoch_stale` maxes over (`rclone lsf runs/<ID>/handoff/` ->
        strip `.json` -> greatest int). A box refuses to push when that max
        exceeds its own HANDOFF_EPOCH.
      * `promoted` (at CUTOVER) — T6's understudy dead-man watchdog stay-up signal
        (`rclone lsf runs/<ID>/handoff/promoted`): present => this box is canonical.

    Best-effort like every other emit (hard=False): no B2_BUCKET or --dry-run =>
    no-op (mirrors _reset_run_markers; keeps the dry-run/no-bucket driver tests
    marker-free). Returns True on a written object, else False."""
    bucket = os.environ.get("B2_BUCKET")
    if not bucket or dry_run:
        return False
    return b2._b2_rcat(f"b2:{bucket}/runs/{run_id}/handoff/{rel}", body, hard=False)


# moved-from: herdd._handoff_synced_epoch_soft
def _handoff_synced_epoch_soft(run_id: object) -> int | None:
    """Greatest epoch with a box-side `.synced` boot proof under
    runs/<RUN_ID>/handoff/ — written by train.sh AFTER its checkpoint resume pull
    completes on the understudy (sentinel `handoff-synced-marker`). The driver's
    SYNCED gate keys on THIS, never on API liveness: `loading` counts as a
    LIVE_STATE, so the old liveness proxy stamped SYNCED 48s after launch against
    a box that had not booted, with zero checkpoints staged, and fenced the
    primary into nothing (live canary handoff-canary-2, 2026-07-15). None when no
    marker / no bucket / read failure — fail-closed: no proof, no SYNCED, no
    fence. Monkeypatched in tests."""
    bucket = os.environ.get("B2_BUCKET")
    if not bucket:
        return None
    rc, out, _ = b2._rclone_soft(["lsf", f"b2:{bucket}/runs/{run_id}/handoff/"])
    if rc != 0:
        return None
    best = None
    for name in (out or "").split():
        if name.endswith(".synced"):
            try:
                e = int(name[:-len(".synced")])
            except ValueError:
                continue
            if best is None or e > best:
                best = e
    return best


# moved-from: herdd._handoff_run_signals
def _handoff_run_signals(run_id: object,
                         cutover_ts: str | None = None) -> dict[str, bool]:
    """Box-side fence signals read from the run event log: the primary's
    `final_flush` (emitted by the box-side `preempt_trap.sh` after its bounded
    final flush) and whether the understudy is PRODUCING. Default reads the local
    runmeta cache; the harness monkeypatches it. Never raises.

    PRODUCING = a `checkpoint` event after the flush, OR — because vast delivers
    NO SIGTERM on a fence-park (proven live twice, 2026-07-15; the `flush_timeout`
    cutover is the NORM, not the fallback) — a `checkpoint` event whose ts is
    later than `cutover_ts` (the compact-UTC moment `resume_understudy` promoted
    the understudy to sole writer). Without the second clause every real handoff
    wedged in DRAINING: no final_flush event ever lands, so flush-gated producing
    stayed False forever. A post-cutover checkpoint can only be the understudy's:
    the primary was parked at the fence >=HANDOFF_FENCE_TIMEOUT_S earlier (timeout
    path) or already flushed (post_flush path)."""
    try:
        evs = launch_spec._raw_events_soft(run_id)    # (ts, nonce)-sorted
    except Exception:
        return {"final_flush_seen": False, "understudy_producing": False}
    flush = any(e.get("event") == "final_flush" for e in evs)
    producing = False
    if flush:
        seen = False
        for e in evs:
            if e.get("event") == "final_flush":
                seen = True
            elif seen and e.get("event") == "checkpoint":
                producing = True
                break
    if not producing and cutover_ts:
        producing = any(e.get("event") == "checkpoint"
                        and (e.get("ts") or "") > cutover_ts for e in evs)
    return {"final_flush_seen": flush, "understudy_producing": producing}


# moved-from: herdd._handoff_observe_understudy
def _handoff_observe_understudy(st: MutableMapping[str, Any],
                                hf: MutableMapping[str, Any]) -> None:
    """Locate the understudy in the instances _observe already fetched and refresh
    its liveness/cost into hf. Matched by the run:<ID>:handoff label OR (once the
    cutover relabels it to run:<ID>) by its adopted id. SYNCED requires the
    box-side `.synced` boot proof (_handoff_synced_epoch_soft) for the armed
    epoch — API liveness alone stamped SYNCED against a still-booting box with
    zero checkpoints staged (live canary handoff-canary-2, 2026-07-15)."""
    run_id = st["run_id"]
    twin_rid = f"{run_id}{labels.HANDOFF_LABEL_SUFFIX}"   # rid = '<id>:handoff'
    uiid = hf.get("understudy_iid")
    inst = None
    for i in st.get("_instances", []) or []:
        if models._instance_run_label(i) == twin_rid or (uiid is not None
                                                         and i.get("id") == uiid):
            inst = i
            break
    if inst is None:                                  # transient absence != reap
        return
    hf["understudy_iid"] = inst.get("id")             # adopt real id over any placeholder
    hf["understudy_status"] = (inst.get("actual_status") or "").lower() or None
    dph = models._num_dph(inst.get("dph_total"))
    if dph is not None:
        hf["understudy_dph"] = dph
    live = hf["understudy_status"] in bidpolicy.LIVE_STATES
    if live and hf.get("understudy_live_since") is None:
        hf["understudy_live_since"] = st.get("now")
    if live and hf.get("ckpt_pulled_epoch") is None \
            and hf.get("phase") in ("LAUNCHING", "WARMING"):
        epoch = hf.get("epoch") or (hf.get("handoffs_done", 0) + 1)
        se = _handoff_synced_epoch_soft(run_id)
        if se is not None and se >= epoch:
            hf["ckpt_pulled_epoch"] = epoch


# moved-from: herdd._handoff_accrue
def _handoff_accrue(st: MutableMapping[str, Any],
                    hf: MutableMapping[str, Any]) -> None:
    """BOTH boxes' cost must count against --budget (HANDOFF_DESIGN §3): while the
    understudy is a SEPARATE live box (not yet st's tracked primary), add its burn
    to spend_usd (so _spend_time_exceeded sees the true double-bill) AND to
    handoff_spend_usd (the incremental-window tracker for the abort rule + cost
    event). Once the understudy becomes st's tracked box, _accrue_cost bills it —
    the id guard prevents a double-subtraction."""
    # LANE DIVERGENCE (pinned): the id guard is RAW `==` here and `str()==` in
    # `_handoff_job_accrue`. Do not unify — see the module docstring.
    if hf.get("phase") in (None, "IDLE", "DONE"):
        return
    if st.get("instance_id") == hf.get("understudy_iid"):
        return
    if hf.get("understudy_status") not in bidpolicy.LIVE_STATES:
        return
    dph = hf.get("understudy_dph")
    dt = st.get("dt", 0.0) or 0.0
    if dph and dt > 0:
        amt = (dph / 3600.0) * dt
        st["spend_usd"] = st.get("spend_usd", 0.0) + amt
        hf["handoff_spend_usd"] = hf.get("handoff_spend_usd", 0.0) + amt


# moved-from: herdd._handoff_build_state
def _handoff_build_state(st: MutableMapping[str, Any], a: argparse.Namespace,
                         hf: MutableMapping[str, Any],
                         act: Any) -> dict[str, Any]:  # noqa: ANN401 — bidpolicy.Action
    """Snapshot the observed world into the PURE mk_handoff_state (T2). primary_dph
    is the paid-bid basis; remaining_wall_h amortizes the migration; primary_evicted
    comes from poll()'s own eviction verdict this tick (fast-cutover / abort keys
    off it)."""
    # FROZEN KEY CONTRACT: `mk_handoff_state` is Zone S and its 35 keys are read
    # BY NAME by handoff_poll / _handoff_fence_hold / _handoff_candidate_ok. The
    # run lane passes 29 and takes the defaults for the six work-awareness keys
    # (`driver_can_complete` defaults TRUE here BY DESIGN — the jobs lane is the
    # one that fails closed). Keyword form, dict result: both are contract.
    rwh = st.get("remaining_wall_h", 0.0)
    primary_evicted = bool(st.get("evicted_pending")) or \
        act.kind in ("emit_evicted", "relaunch")
    hs: dict[str, Any] = bidpolicy.mk_handoff_state(  # type: ignore[no-untyped-call]
        phase=hf["phase"], over_ceiling_streak=hf["over_ceiling_streak"],
        primary_iid=hf.get("primary_iid") or st.get("instance_id"),
        primary_bid=st.get("last_bid"), primary_on_demand=st.get("on_demand"),
        primary_dph=_handoff_primary_dph(st), primary_evicted=primary_evicted,
        primary_gone=hf.get("primary_gone", False),
        understudy_iid=hf.get("understudy_iid"), understudy_dph=hf.get("understudy_dph"),
        understudy_on_demand=hf.get("understudy_on_demand"),
        understudy_status=hf.get("understudy_status"),
        understudy_live_since=hf.get("understudy_live_since"),
        understudy_producing=hf.get("understudy_producing", False),
        understudy_gone=hf.get("understudy_gone", False),
        drain_ts=hf.get("drain_ts"),
        candidate_min_bid=hf.get("candidate_min_bid"),
        candidate_on_demand=hf.get("candidate_on_demand"),
        remaining_wall_h=rwh, final_flush_seen=hf.get("final_flush_seen", False),
        fence_ts=hf.get("fence_ts"),
        ckpt_pulled_epoch=hf.get("ckpt_pulled_epoch"),
        handoff_started_ts=hf.get("handoff_started_ts"),
        handoff_spend_usd=hf.get("handoff_spend_usd", 0.0),
        handoffs_done=hf.get("handoffs_done", 0),
        cooldown_until=hf.get("cooldown_until", 0.0),
        budget_usd=st.get("budget_usd"), spend_usd=st.get("spend_usd", 0.0),
        now=st.get("now", 0.0))
    # Assigned, not passed: `HandoffSnapshot` pins the factory's 35 keys exactly.
    # Absent => Zone S falls back to the HANDOFF_DWELL_POLLS count.
    hs["over_ceiling_since"] = hf.get("over_ceiling_since")
    return hs


# moved-from: herdd._handoff_reset
def _handoff_reset(hf: MutableMapping[str, Any], *, handoffs_done: int,
                   cooldown_until: float) -> None:
    """Return the sub-state to IDLE for the next opportunity, PRESERVING the
    per-run counters (handoffs_done, cooldown_until)."""
    # IN-PLACE by contract: callers (and fleetd's `rt["hf"]`) hold this exact
    # object across ticks. clear()+update(), never a rebind.
    fresh = _init_handoff_state()
    fresh["handoffs_done"] = handoffs_done
    fresh["cooldown_until"] = cooldown_until
    hf.clear()
    hf.update(fresh)


# --------------------------------------------------------------------------- #
# RUN LANE — the effectful drivers. Terminal transitions first, then the single
# action dispatcher, then reconcile / unfence / alarm / tick / exit.
# --------------------------------------------------------------------------- #

# moved-from: herdd._handoff_complete
def _handoff_complete(st: MutableMapping[str, Any], a: argparse.Namespace,
                      hf: MutableMapping[str, Any]) -> None:
    """DONE: the understudy is confirmed producing and the primary is gone. Promote
    the understudy to st's tracked box so the untouched poll() ladder supervises it
    from here, bump handoffs_done, open the cooldown."""
    run_id = st["run_id"]
    u, udph = hf.get("understudy_iid"), hf.get("understudy_dph")
    if u is not None:
        st["instance_id"] = st["husk_id"] = u
        # `understudy_dph` is the observed `dph_total` (bid + storage) — the
        # right number for the cost basis and the WRONG one for `last_bid`,
        # which the self-floor guard compares by exact equality against the
        # chunk's `min_bid`. Take the promoted box's `dph_base` when the
        # instance body is in hand (`_observe` caches the listing).
        _uinst = next((i for i in (st.get("_instances") or [])
                       if i.get("id") == u), None)
        _ubid = (models._instance_standing_bid(_uinst)
                 if (_uinst or {}).get("is_bid") else None)
        if udph is not None:
            st["dph_total"] = udph
        # dph_base or NOTHING (review 2026-08-10, #7): the old `or udph`
        # fallback wrote bid+storage into last_bid — permanently one storage
        # sliver above every number vast echoes back, so the standing arm
        # could never match the promoted box. A None here is fail-closed:
        # bid moves stay disabled until _observe seeds/reconciles from the
        # next body that carries dph_base.
        st["last_bid"] = _ubid
        # The shared box-swap seam — but NOT the sticky on-demand clamp: this
        # seam has never cleared `on_demand_last`, so the retired primary's
        # machine keeps clamping the promoted understudy's rails until the next
        # probe re-seeds it. That is divergence D2 in AUTOBID_DESIGN.md §"One
        # core, two lanes" — an unfixed parity gap against the other four seams,
        # recorded and pinned by a test rather than silently repaired inside a
        # behavior-preserving refactor.
        ladder_core.box_swap_reset(  # type: ignore[no-untyped-call]
            st, reset_sticky_on_demand=False)
    st["evicted_pending"] = False
    st["not_live_streak"] = 0
    done = hf.get("handoffs_done", 0) + 1
    journal._sup_emit(run_id, "handoff_complete", understudy=u, dph=udph,
                      handoffs_done=done,
                      handoff_spend_usd=round(hf.get("handoff_spend_usd", 0.0), 4),
                      spend_usd=round(st.get("spend_usd", 0.0), 4))
    print(f">> handoff complete: migrated {run_id} to understudy {u} "
          f"(window cost {fmt.dollars(hf.get('handoff_spend_usd', 0.0))})")
    _handoff_reset(hf, handoffs_done=done,
                   cooldown_until=st.get("now", 0.0) + bidpolicy.HANDOFF_COOLDOWN_S)


# moved-from: herdd._handoff_abort
def _handoff_abort(st: MutableMapping[str, Any], a: argparse.Namespace,
                   hf: MutableMapping[str, Any], reason: str) -> None:
    """Pre-CUTOVER rollback (deadline / primary-evicted / unlaunchable): reap any
    understudy remnant, stay on the primary, open the cooldown. handoffs_done is
    NOT bumped (an abort is not a completed handoff). The understudy's nonce B2 key
    self-expires (its name isn't retained here — no cross-revoke of the primary's
    key, the T3 invariant)."""
    run_id = st["run_id"]
    u = hf.get("understudy_iid")
    if u is not None:
        lifecycle._destroy_soft(u, dry_run=a.dry_run)
        if not a.dry_run:
            _confirm_gone(u)
    journal._sup_emit(run_id, "handoff_abort", reason=reason, instance_id=u)
    print(f">> handoff aborted ({reason}); reaped understudy {u}, staying on primary")
    _handoff_reset(hf, handoffs_done=hf.get("handoffs_done", 0),
                   cooldown_until=st.get("now", 0.0) + bidpolicy.HANDOFF_COOLDOWN_S)


# moved-from: herdd._do_handoff_move
def _do_handoff_move(st: MutableMapping[str, Any], a: argparse.Namespace,
                     hf: MutableMapping[str, Any],
                     act: Any) -> None:  # noqa: ANN401 — bidpolicy.HandoffAction
    """Execute ONE HandoffAction (the I/O). Money-moving steps (launch/park/
    destroy) sit AFTER poll()'s spend/budget guards in the driver loop, mirroring
    _do_bid_move. Emits the HANDOFF_DESIGN §6 events."""
    # `replacement` imports THIS module, so its two builders are imported at
    # call time — module-attribute form, so a monkeypatch on either still
    # steers. See the module docstring.
    from vastlib.supervise import replacement

    run_id = st["run_id"]
    k = act.kind

    if k == "arm":
        hf["phase"] = "ARMED"
        hf["handoff_started_ts"] = st.get("now", 0.0)
        hf["primary_iid"] = st.get("instance_id")     # pin the box we will retire
        # T4b: the write-generation epoch for THIS attempt. Monotonic across the
        # run (handoffs_done increments per completed handoff), and always one above
        # the current primary's epoch — the original primary has NO HANDOFF_EPOCH
        # (== 0), a prior understudy-turned-primary was launched at epoch
        # handoffs_done. Stamp runs/<ID>/handoff/<epoch>.json now (ARM) so any stale
        # writer (a resumed husk) that reads a strictly-greater epoch refuses to push
        # (T6 _handoff_epoch_stale). Best-effort; the understudy carries the same
        # epoch in its launch env (launch_understudy below).
        epoch = hf.get("handoffs_done", 0) + 1
        hf["epoch"] = epoch
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _handoff_b2_write(run_id, f"{epoch}.json",
                          f'{{"epoch":{epoch},"run_id":"{run_id}","armed_at":"{ts}"}}\n',
                          dry_run=a.dry_run)
        journal._sup_emit(run_id, "handoff_armed",
                          over_ceiling_streak=hf["over_ceiling_streak"],
                          primary_bid=st.get("last_bid"), on_demand=st.get("on_demand"),
                          candidate_min_bid=hf.get("candidate_min_bid"),
                          candidate_on_demand=hf.get("candidate_on_demand"),
                          epoch=epoch, handoffs_done=hf.get("handoffs_done", 0))
        print(f">> handoff ARMED for {run_id}: primary bid ${st.get('last_bid')} "
              f"over ceiling; migrating to a cheaper box (epoch {epoch})")
        return

    if k == "launch_understudy":
        offer = hf.get("chosen_offer") or replacement._handoff_pick_offer(st, a)
        if offer is None:                             # market moved -> no candidate
            return _handoff_abort(st, a, hf, "no_offer")
        # epoch stamped at ARM; fall back to the monotonic default if this launch
        # is reached without a prior arm (defensive — arm always precedes launch).
        epoch = hf.get("epoch") or (hf.get("handoffs_done", 0) + 1)
        body, bid, missing = replacement._handoff_understudy_body(st, a, offer,
                                                                  epoch=epoch)
        if body is None or missing:
            return _handoff_abort(st, a, hf, "understudy_unlaunchable")
        # F4: the _launch_preflight handoff-twin allowance (:672) is dead code unless
        # this path actually calls it — a resumed/parked run:<ID>:handoff twin (a
        # crash-orphan we did not reconcile) must block a duplicate understudy. Reuse
        # this tick's instance snapshot (no extra GET); a live/parked twin -> abort.
        # Respect dry-run (preflight does real I/O and there is no real launch here).
        if not a.dry_run:
            try:
                launchmod._launch_preflight(
                    f"run:{run_id}{labels.HANDOFF_LABEL_SUFFIX}", False,
                    instances=st.get("_instances"))
            except SystemExit as e:                   # a live/parked twin already exists
                st["last_error"] = str(e)
                return _handoff_abort(st, a, hf, "understudy_unlaunchable")
        if a.dry_run:
            print(f"[dry-run] would PUT understudy ask offer={offer['id']} "
                  f"bid={bid} label=run:{run_id}{labels.HANDOFF_LABEL_SUFFIX}")
            cid = f"dry-h-{int(st.get('now', 0))}"
        else:
            okp, cid, perr = launchmod.launch_instance(offer["id"], body)
            if not okp:                               # stay ARMED, retry (deadline bounds)
                st["last_error"] = perr
                return
            ssh.attach_ssh_key_soft(cid)  # mirror _do_launch's post-launch attach
        hf["understudy_iid"] = cid
        hf["understudy_dph"] = bid
        hf["understudy_on_demand"] = models._num_dph(offer.get("dph_total"))
        hf["phase"] = "LAUNCHING"
        journal._sup_emit(run_id, "handoff_launch", offer_id=offer.get("id"), dph=bid,
                          instance_id=cid,
                          label=f"run:{run_id}{labels.HANDOFF_LABEL_SUFFIX}")
        return

    if k == "warm_wait":
        return                                        # understudy still booting

    if k == "mark_synced":
        hf["phase"] = "SYNCED"
        journal._sup_emit(run_id, "handoff_synced",
                          instance_id=hf.get("understudy_iid"),
                          ckpt_epoch=hf.get("ckpt_pulled_epoch"))
        return

    if k == "fence_primary":
        # Open the two-writer fence: park the primary; its train.sh trap does the
        # final flush. The understudy becomes a writer only AFTER final_flush_seen.
        iid = hf.get("primary_iid") or st.get("instance_id")
        pin = _handoff_park_bid(st)
        # Remember what the pin is about to overwrite: the post-cutover abort has
        # to give it back, and resuming at the park pin would leave the box
        # permanently unable to win its market.
        hf["prefence_bid"] = _prefence_bid(st.get("last_bid"), st.get("dph"))
        if a.dry_run:
            print(f"[dry-run] would park primary {iid} (handoff fence) and pin its "
                  f"bid -> ${pin} (below floor, no auto-resume)")
        elif iid is not None:
            ok, perr = lifecycle._put_state_soft(iid, "stopped")
            if ok:
                lifecycle._wait_states_soft(iid, {"stopped", "exited"}, 120)
            else:
                st["last_error"] = f"handoff fence park failed: {perr}"
            # T4b two-writer belt (§4): a PARKED bid box AUTO-RESUMES when the floor
            # drops (the box-44566398 stuck-bid leak) and would then race the
            # understudy's checkpoint writes — and the epoch guard can't stop it (the
            # primary's launch env has no HANDOFF_EPOCH, so T6's fail-safe lets it
            # push). Pin its bid below any floor so vast never auto-resumes it before
            # DRAINING destroys it. Best-effort; a failed pin just leaves the epoch
            # guard + the imminent destroy as the remaining defenses.
            lifecycle._put_bid_soft(iid, pin)
        hf["phase"] = "CUTOVER"
        hf["fence_ts"] = st.get("now", 0.0)           # opens the flush-timeout / stall clocks
        journal._sup_emit(run_id, "handoff_fence", primary=iid,
                          understudy=hf.get("understudy_iid"), pinned_bid=pin)
        return

    if k == "resume_understudy":
        # post-flush (or fast-cutover on a dead primary): the understudy becomes
        # the sole writer; relabel run:<ID>:handoff -> run:<ID> so it is canonical.
        u = hf.get("understudy_iid")
        # T4b: write the promotion marker BEFORE anything tears the primary down
        # (the drain step is a LATER tick, and the fast-cutover path skips drain
        # entirely — this is the one place both paths share). Ordering is
        # load-bearing: the understudy's dead-man watchdog (T6) parks the box if no
        # runs/<ID>/handoff/promoted is present by TTL, so a supervisor that dies
        # mid-cutover (after we commit to the understudy) must not self-park the
        # true survivor. Written here == the moment the understudy becomes canonical.
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _handoff_b2_write(run_id, "promoted",
                          f'{{"run_id":"{run_id}","understudy":"{u}",'
                          f'"epoch":{hf.get("epoch") or 0},"promoted_at":"{ts}",'
                          f'"reason":"{act.reason}"}}\n',
                          dry_run=a.dry_run)
        if a.dry_run:
            print(f"[dry-run] would relabel understudy {u} -> run:{run_id} "
                  f"and resume as sole writer ({act.reason})")
        elif u is not None:
            lifecycle._put_label_soft(u, f"run:{run_id}")
        hf["phase"] = "DRAINING"
        hf["drain_ts"] = st.get("now", 0.0)   # starts the post-cutover clock
        # compact-UTC stamp of the promotion moment, same lexicographic format
        # as runmeta event ts — the producing signal counts `checkpoint` events
        # AFTER this (the flush_timeout path has no final_flush to key on).
        hf["cutover_ts"] = datetime.datetime.now(datetime.timezone.utc)\
            .strftime("%Y%m%dT%H%M%S")
        if act.reason == "fast_cutover":              # primary already evicted/gone
            hf["primary_gone"] = True
        journal._sup_emit(run_id, "handoff_cutover", understudy=u, reason=act.reason)
        return

    if k == "drain_primary":
        iid = hf.get("primary_iid")
        okd, derr = lifecycle._destroy_soft(iid, dry_run=a.dry_run)
        if okd and (a.dry_run or _confirm_gone(iid)):
            hf["primary_gone"] = True
            lifecycle._revoke_box_keys([f"run-{run_id}"])   # primary's plain run key
        else:
            st["last_error"] = f"handoff drain: primary {iid} not gone ({derr})"
        return

    if k == "complete":
        return _handoff_complete(st, a, hf)

    if k == "abort_reap":
        return _handoff_abort(st, a, hf, act.reason)

    if k == "abort_unfence":
        # POST-cutover rollback: the understudy died (or never produced) after the
        # fence closed. Reap it, then GIVE THE PRIMARY BACK — un-pin the bid the
        # fence drove to HANDOFF_PARK_BID and resume the box. Doing only the reap
        # would leave the primary parked at an unwinnable bid, which is the same
        # wedge one step later.
        p = hf.get("primary_iid") or st.get("instance_id")
        _handoff_unfence_primary(p, hf, dry_run=a.dry_run,
                                 emit=lambda **f: journal._sup_emit(st["run_id"],
                                                                    "handoff_unfence",
                                                                    **f))
        return _handoff_abort(st, a, hf, act.reason)


# moved-from: herdd._handoff_reconcile
def _handoff_reconcile(st: MutableMapping[str, Any], a: argparse.Namespace,
                       hf: MutableMapping[str, Any]) -> None:
    """Reconcile-on-(re)start (HANDOFF_DESIGN §5 crash row / §6): adopt a live
    run:<ID>:handoff twin left by a crashed supervisor rather than orphaning it,
    and resume the machine mid-flight. Bias: keep the confirmed writer. The full
    cutover-evidence promotion + epoch matrix is T5/T6; v1 resumes a staged live
    twin at SYNCED (proceed to fence) or LAUNCHING (keep warming)."""
    run_id = st["run_id"]
    ok, data, _ = api.request_soft("GET", "v1/instances/")
    if not ok:
        return
    instances = data.get("instances", data) if isinstance(data, dict) else data
    twin_rid = f"{run_id}{labels.HANDOFF_LABEL_SUFFIX}"
    twin = next((i for i in (instances or [])
                 if models._instance_run_label(i) == twin_rid), None)
    if twin is None:
        return
    live = (twin.get("actual_status") or "").lower() in bidpolicy.LIVE_STATES
    hf["understudy_iid"] = twin.get("id")
    hf["understudy_status"] = (twin.get("actual_status") or "").lower() or None
    hf["understudy_dph"] = models._num_dph(twin.get("dph_total"))
    hf["understudy_on_demand"] = models._num_dph(twin.get("dph_total"))
    hf["primary_iid"] = st.get("instance_id")
    hf["handoff_started_ts"] = st.get("now", time.time())
    # SYNCED needs the box-side .synced boot proof, same as the observe path —
    # an adopted twin that is merely `loading` resumes at LAUNCHING and earns
    # SYNCED through the marker like any fresh understudy.
    se = _handoff_synced_epoch_soft(run_id) if live else None
    hf["phase"] = "SYNCED" if se is not None else "LAUNCHING"
    if live:
        hf["understudy_live_since"] = st.get("now", time.time())
    if se is not None:
        hf["epoch"] = se
        hf["ckpt_pulled_epoch"] = se
    journal._sup_emit(run_id, "handoff_reconciled", understudy=twin.get("id"),
                      phase=hf["phase"], live=live)
    print(f">> adopted live handoff understudy {twin.get('id')} "
          f"(resuming at {hf['phase']})")


# moved-from: herdd._handoff_unfence_primary
def _handoff_unfence_primary(iid: object, hf: MutableMapping[str, Any], *,
                             dry_run: bool = False,
                             emit: Callable[..., Any] | None = None,
                             policy_target: float | None = None) -> bool:
    """Undo `fence_primary` on the PRIMARY: restore a WINNABLE bid and resume it.

    Called from the post-cutover abort and from the fence-unwind — i.e. only when
    the understudy is dead, has provably never produced, or the cutover could not
    commit — so there is no second writer to race. Best-effort and total: a
    failure here is reported, never raised, because the abort that follows must
    happen either way (an un-reaped understudy is the more expensive half).

    The bid restored is, in order: the bid recorded BEFORE the fence pinned it to
    HANDOFF_PARK_BID (`prefence_bid`), else `policy_target` — what the bid policy
    would put for this box right now. The fallback is not decoration. Resuming at
    the $0.001 pin leaves the box permanently unable to win its market, which is
    parked-at-an-unwinnable-bid: the same wedge one step later, and (since the
    2026-08-08 autobid work) the exact first-eviction-target configuration that
    work exists to prevent. When NEITHER is known we do not resume at all — a box
    left parked is recoverable by hand and by the reaper; a box resumed at $0.001
    is a live rental that cannot defend itself — and the refusal is printed with
    the command to fix it.
    """
    if iid is None:
        return False
    restore = hf.get("prefence_bid")
    if restore is not None and restore <= bidpolicy.HANDOFF_PARK_BID:
        restore = None            # belt: never restore the pin AS the pre-fence bid
    restore = restore or policy_target
    if not restore:
        print(f"!! unfence: {iid} has NO recoverable bid (no pre-fence bid "
              f"recorded and no policy target readable) — leaving it PARKED "
              f"rather than resuming it at the ${bidpolicy.HANDOFF_PARK_BID} fence "
              f"pin, which it could never win a market with. Set a bid and resume "
              f"by hand: herdd bid {iid} --price <X> && herdd start {iid}")
        if emit is not None:
            emit(primary=iid, restored_bid=None, bid_ok=False, resume_ok=False,
                 note="no recoverable bid; left parked, NOT resumed at the pin")
        return False
    if dry_run:
        print(f"[dry-run] would unfence primary {iid}: restore bid "
              f"${restore if restore is not None else '(unknown)'} and resume")
        return True
    ok_bid = True
    if restore:
        ok_bid, berr = lifecycle._put_bid_soft(iid, restore)
        if not ok_bid:
            print(f"!! unfence: could not restore {iid}'s bid to ${restore} "
                  f"({berr}) — set it by hand: herdd bid {iid} --price {restore}")
    ok_run, rerr = lifecycle._put_state_soft(iid, "running")
    if not ok_run:
        print(f"!! unfence: could not resume primary {iid} ({rerr}) — it is still "
              f"parked; `herdd start {iid}` by hand")
    if emit is not None:
        emit(primary=iid, restored_bid=restore, bid_ok=bool(ok_bid),
             resume_ok=bool(ok_run))
    print(f">> unfenced primary {iid} (bid -> "
          f"${restore if restore is not None else 'unchanged'}, resumed)")
    return bool(ok_run)


# moved-from: herdd._handoff_stall_alarm
def _handoff_stall_alarm(hf: MutableMapping[str, Any], now: float,
                         emit: Callable[..., Any]) -> None:
    """F2: bounded observability for a stuck fence-open phase — NOT a forced
    transition. If the migration is still in CUTOVER or DRAINING HANDOFF_DEADLINE_S
    after the fence opened, alarm ONCE (emit `handoff_stall` + one print) and latch
    the once-flag. CUTOVER included (2026-07-18 review S4): CUTOVER normally exits
    at HANDOFF_FENCE_TIMEOUT_S (< the deadline), so a CUTOVER past the deadline is
    already wedged (a retarget_incomplete latch, or an understudy that died inside
    the fence window — which nothing else detects until the DRAINING stall). We
    deliberately do NOT force-destroy the primary on a timer: the primary destroy
    stays gated on understudy proof-of-life (the byte-safety invariant), so this is
    a loud alert on a wedged migration, nothing more. Shared by both lanes; `emit`
    writes the lane's telemetry event, `hf` carries the phase, fence_ts and the
    `stall_alarmed` latch (reset by _handoff_reset)."""
    fence_ts = hf.get("fence_ts")
    if hf.get("phase") in ("CUTOVER", "DRAINING") and fence_ts is not None \
            and now - fence_ts >= bidpolicy.HANDOFF_DEADLINE_S \
            and not hf.get("stall_alarmed"):
        waited = now - fence_ts
        u = hf.get("understudy_iid")
        emit(understudy=u, waited_s=round(waited, 1), phase=hf.get("phase"))
        print(f"!! handoff STALL: understudy {u} not confirmed producing {waited:.0f}s "
              f"after the fence opened ({hf.get('phase')}) — primary held parked "
              f"pending understudy proof-of-life (no forced destroy)")
        hf["stall_alarmed"] = True


# moved-from: herdd._handoff_tick
def _handoff_tick(st: MutableMapping[str, Any], a: argparse.Namespace,
                  hf: MutableMapping[str, Any],
                  act: Any) -> None:  # noqa: ANN401 — bidpolicy.Action from poll()
    """One handoff step: advance the dwell counter, observe+accrue the understudy,
    read the fence signals, then run the PURE handoff_poll and execute its move.
    Called AFTER poll() every tick when --handoff is set."""
    from vastlib.supervise import replacement

    run_id = st["run_id"]
    # remaining wall budget (hours) — the amortization horizon read by BOTH the
    # candidate filter (_handoff_pick_offer / _handoff_understudy_body) and the
    # pure state builder; stash it on st so the T3 helpers see it. NO wall budget
    # means no horizon, so the filter refuses (defect #63): the run lane has no
    # ticket queue to measure a remainder off, and the flat 24.0 that stood here
    # was the same fabrication that migrated a healthy jobs box on 2026-08-08.
    # RUN_POLICY_DEFAULTS seeds 48h, so under fleetd this branch is the rare one
    # — a bare `supervise --handoff` with no --wall-budget now gets get-and-hold
    # instead of a migration priced against an invented day.
    wall = st.get("wall_budget_s")
    st["remaining_wall_h"] = (max(0.0, (wall - st.get("wall_clock_s", 0.0)) / 3600.0)
                              if wall else None)
    # dwell hysteresis (HANDOFF_DESIGN §1): ARM only after the bid has been over
    # the preferred ceiling CONTINUOUSLY for HANDOFF_DWELL_S. The start
    # timestamp is what makes that a duration; the count is kept because it is
    # the persisted key and Zone S falls back to it when `since` is absent.
    if st.get("_over_pref"):
        hf["over_ceiling_streak"] = hf.get("over_ceiling_streak", 0) + 1
        if hf.get("over_ceiling_since") is None:
            hf["over_ceiling_since"] = st.get("now")
    else:
        hf["over_ceiling_streak"] = 0
        hf["over_ceiling_since"] = None

    _handoff_observe_understudy(st, hf)
    # pin the primary iid if an adopted (reconciled) handoff never saw ARM
    if hf.get("primary_iid") is None and hf["phase"] != "IDLE" \
            and st.get("instance_id") not in (None, hf.get("understudy_iid")):
        hf["primary_iid"] = st.get("instance_id")
    _handoff_accrue(st, hf)

    # candidate market read: once dwell is satisfied and still IDLE, price the
    # cheapest qualifying offer so the pure ARM gate (headroom + candidate) can see it.
    if hf["phase"] == "IDLE" \
            and bidpolicy._handoff_dwell_satisfied(hf, st.get("now")):  # type: ignore[no-untyped-call]
        offer = replacement._handoff_pick_offer(st, a)
        hf["chosen_offer"] = offer
        hf["candidate_min_bid"] = models._num_dph(offer.get("min_bid")) if offer else None
        # market on-demand, never the bid row's dph_total (doc 50 R1 —
        # see _offer_ondemand_ref); None makes the pure ARM gate refuse.
        hf["candidate_on_demand"] = pricing._offer_ondemand_ref(offer) if offer else None

    # box-side fence signals only matter once the fence is open.
    if hf["phase"] in _HANDOFF_FENCE_OPEN:
        sig = _handoff_run_signals(run_id, cutover_ts=hf.get("cutover_ts"))
        if sig.get("final_flush_seen"):
            hf["final_flush_seen"] = True
        if sig.get("understudy_producing"):
            hf["understudy_producing"] = True

    _handoff_stall_alarm(hf, st.get("now", 0.0),
                         lambda **f: journal._sup_emit(run_id, "handoff_stall", **f))

    hs = _handoff_build_state(st, a, hf, act)
    hact = bidpolicy.handoff_poll(hs)                  # PURE
    if hact.kind != "noop":
        _do_handoff_move(st, a, hf, hact)


# moved-from: herdd._handoff_reap_on_exit
def _handoff_reap_on_exit(st: MutableMapping[str, Any], a: argparse.Namespace,
                          hf: MutableMapping[str, Any]) -> None:
    """Stop path (budget/wall/fatal): a mid-flight PRE-cutover understudy is reaped
    so a stop never leaks a second box (HANDOFF_DESIGN §3). A post-cutover
    understudy is the run's canonical box — leave it (the primary stop path parks
    st's tracked box, which is now the understudy)."""
    if not hf:
        return
    if hf.get("phase") in bidpolicy._HANDOFF_PRE_CUTOVER \
            and hf.get("understudy_iid") is not None:
        lifecycle._destroy_soft(hf["understudy_iid"], dry_run=a.dry_run)
        journal._sup_emit(st["run_id"], "handoff_abort", reason="supervisor_stop",
                          instance_id=hf["understudy_iid"])


# --------------------------------------------------------------------------- #
# JOBS LANE — the mirror. Same ladder, same pure core, different I/O: per-JOB
# B2 markers instead of per-run, a ticket RETARGET instead of a box relabel, an
# in-memory decision journal fleetd drains into `fleet log`, and six extra
# work-awareness inputs to the pure gate. Read each function against its run-
# lane twin above; the divergences are pinned, not accidental.
# --------------------------------------------------------------------------- #

# moved-from: herdd._handoff_job_b2_write
def _handoff_job_b2_write(job_id: object, rel: str, body: str,
                          dry_run: bool = False) -> bool:
    """PRODUCER side of onstart/jobd.sh's box-side guards, per JOB (mirror of the
    run lane's _handoff_b2_write under the JOB prefix):

      * `<epoch>.json` (at ARM) — the monotonic write-generation marker jobd.sh's
        `_handoff_epoch_stale` maxes over (`rclone lsf jobs/<JID>/handoff/` :74 ->
        strip `.json` -> greatest int; a box refuses to push once that exceeds its
        own HANDOFF_EPOCH).
      * `promoted` (at CUTOVER) — the understudy-canonical signal, symmetric with
        the run lane; jobd's idle self-park (JOBD_NO_JOB_PARK_S) is the jobs-lane
        dead-man, so this is telemetry/parity here rather than a stay-up gate.

    Best-effort (hard=False): no B2_BUCKET or --dry-run => no-op."""
    bucket = os.environ.get("B2_BUCKET")
    if not bucket or dry_run:
        return False
    return b2._b2_rcat(f"b2:{bucket}/jobs/{job_id}/handoff/{rel}", body, hard=False)


# The jobs-lane mirror of `launch_spec._raw_events_soft`. It travels with this
# cluster rather than with `jobs/` because `_handoff_job_signals` is its ONLY
# caller in the tree (grep at a1f2c8a5: one call, one def) and no port manifest
# claims it. If `jobs/` grows a raw-event reader, this is the copy to delete.
# moved-from: herdd._raw_job_events_soft
def _raw_job_events_soft(job_id: str) -> list[dict[str, Any]]:
    """Raw job event dicts from the local jobmeta cache (populated by
    jobmeta.read_job). Mirror of _raw_events_soft for the run lane: `final_flush`
    is NOT folded into the job view, so the fence reads the raw stream. [] on any
    failure. Sorted (ts, nonce)."""
    cache = os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "vast-jobmeta", job_id, "events")
    out: list[dict[str, Any]] = []
    try:
        for p in glob.glob(os.path.join(cache, "*.json")):
            try:
                with open(p) as fh:
                    out.append(json.load(fh))
            except Exception:
                continue
    except Exception:
        return []
    out.sort(key=lambda e: (e.get("ts", ""), e.get("nonce", "")))
    return out


# moved-from: herdd._handoff_job_signals
def _handoff_job_signals(running_jobs: Sequence[str] | None,
                         all_jobs: Iterable[str] | None,
                         understudy_iid: object) -> dict[str, bool]:
    """Box-side fence signals for the jobs lane (analogue of _handoff_run_signals),
    read from each job's raw event log. Monkeypatched in tests. Never raises.

      * final_flush_seen — EVERY job that was RUNNING on the primary at the fence
        has emitted jobd's `final_flush` (onstart/jobd.sh:1201 preempt trap, T1).
        An empty running set (nothing was writing) is trivially fenced.
      * understudy_producing — after the retarget, the understudy's jobd has
        claimed/checkpointed a moved ticket (a lifecycle event whose instance_id ==
        the understudy). Proof-of-life gating the primary destroy (design §2.2/6)."""
    u = str(understudy_iid) if understudy_iid is not None else None
    flushed = {}
    producing = False
    for jid in all_jobs or ():
        try:
            jobmeta.read_job(jid, live_iids=_live_iids_set())     # refresh the cache
        except Exception:
            pass
        evs = _raw_job_events_soft(jid)
        flushed[jid] = any(e.get("event") == "final_flush" for e in evs)
        if u is not None and not producing:
            producing = any(str(e.get("instance_id")) == u
                            and e.get("event") in _HANDOFF_JOB_PRODUCING for e in evs)
    ff = all(flushed.get(j) for j in running_jobs) if running_jobs else True
    return {"final_flush_seen": ff, "understudy_producing": producing}


# Travels with this cluster for the same reason as `_raw_job_events_soft`: its
# only caller in the tree is `_handoff_job_understudy_synced` below, and no port
# manifest claims it. Its natural long-term home is `boxes/health.py`, beside
# `_jobd_status_line_soft` (which it reads) and `_jobd_status_pyhalf_soft`.
# moved-from: herdd._jobd_status_soft
def _jobd_status_soft(iid: object) -> str | None:
    """First token of jobs/nodes/<iid>/JOBD_STATUS (jobd's coarse per-box
    heartbeat, stamped by its MAIN LOOP: 'RUNNING n <pids>' / 'IDLE' /
    'PARKING …'). None on no bucket / absent marker / read failure. This is the
    AFFIRMATIVE jobd boot proof for the jobs-lane SYNCED gate: absence-of-park
    from the lifecycle fold is not evidence a box ever booted (a still-`loading`
    box has an empty fold and read as healthy — the same hole the run lane's
    liveness proxy had, live canary 2026-07-15).

    THE TOKEN IS NOT A LIVENESS PROOF ON ITS OWN. It is written by jobd.sh —
    the BASH half — so a box whose python half is dead stamps a perfectly
    healthy `IDLE` and keeps stamping it every 120 s. Two callers learned that
    the expensive way (FAILCLOSED_DESIGN §11.2, §11.5); both now pair this with
    `_jobd_status_pyhalf_soft`, which reads the confession on the same line. A
    third caller should assume it must do the same."""
    line = health._jobd_status_line_soft(iid)
    if line is None:
        return None
    return line.strip().split()[0].upper()


# moved-from: herdd._handoff_job_understudy_synced
def _handoff_job_understudy_synced(jctx: MutableMapping[str, Any],
                                   hf: MutableMapping[str, Any]) -> bool:
    """The jobs-lane SYNCED signal (monkeypatched in tests): the understudy is
    LIVE, its jobd MAIN LOOP has stamped JOBD_STATUS (RUNNING/IDLE — affirmative
    boot proof; a fresh iid prefix has no marker until jobd really runs), and it
    is HEALTHY (not self-parked/drained). A box that booted but whose jobd
    errored on the asset/checkpoint pull self-parks (JOBD_NO_JOB_PARK_S backstop)
    or goes not-live, so this stays False and the deadline abort (§5 'bundle/
    asset pull fails' row) reaps it.

    AND its python half must not be CONFESSING (2026-08-14, the same defect
    class as the boot SLA's milestone — FAILCLOSED_DESIGN §11.2). A box with a
    dead python half still stamps `IDLE`, because the bash half writes the
    marker; without this check the migration would declare SYNCED, retarget the
    queue onto a box that can neither claim a ticket nor emit an event, and
    then destroy the healthy primary. That is strictly worse than the incident
    it descends from: it does not merely fail to notice a dead box, it MOVES
    LIVE WORK ONTO ONE.

    Tri-state, and only `True` blocks: an understudy on a bundle older than the
    field reports None and syncs exactly as it did before. A confession costs
    one understudy (never SYNCED -> the deadline abort reaps it) and the
    primary keeps the job."""
    if hf.get("understudy_status") not in bidpolicy.LIVE_STATES:
        return False
    u = hf.get("understudy_iid")
    if u is None:
        return False
    if _jobd_status_soft(u) not in ("RUNNING", "IDLE"):
        return False
    # TRI-STATE, and ONLY `True` blocks. `is True` is load-bearing: None means
    # "the bundle is older than the field", not "confessing". No `or False`.
    if health._jobd_status_pyhalf_soft(u) is True:
        return False
    bx = _box_lifecycle_soft(u)
    return not (bx.get("parked") or bx.get("drained_pending"))


# moved-from: herdd._handoff_observe_job_understudy
def _handoff_observe_job_understudy(jctx: MutableMapping[str, Any],
                                    hf: MutableMapping[str, Any]) -> None:
    """Locate the understudy in the tick's instance snapshot (by the
    job:<primary>:handoff label OR its adopted id) and refresh liveness/cost into
    hf. When it becomes SYNCED during WARMING (jobd healthy + prewarm done), stamp
    ckpt_pulled_epoch so the pure handoff_poll advances SYNCED."""
    label = labels._job_handoff_label(jctx.get("iid"))
    uiid = hf.get("understudy_iid")
    inst = None
    for i in jctx.get("instances", []) or []:
        if i.get("label") == label or (uiid is not None and str(i.get("id")) == str(uiid)):
            inst = i
            break
    if inst is None:
        # The understudy is NOT in this tick's snapshot. Leaving `understudy_status`
        # STALE here is what let a dead understudy keep reading as `running` while
        # DRAINING waited on it forever. An EMPTY snapshot is an API blip, not
        # evidence — only a non-empty listing that omits our id says it is gone.
        if uiid is not None and (jctx.get("instances") or []):
            hf["understudy_gone"] = True
            hf["understudy_status"] = None
        return
    hf["understudy_gone"] = False
    hf["understudy_iid"] = inst.get("id")
    hf["understudy_status"] = (inst.get("actual_status") or "").lower() or None
    dph = models._num_dph(inst.get("dph_total"))
    if dph is not None:
        hf["understudy_dph"] = dph
    if hf.get("understudy_status") in bidpolicy.LIVE_STATES \
            and hf.get("understudy_live_since") is None:
        hf["understudy_live_since"] = jctx.get("now")
    if hf.get("ckpt_pulled_epoch") is None \
            and hf.get("phase") in ("LAUNCHING", "WARMING") \
            and _handoff_job_understudy_synced(jctx, hf):
        hf["ckpt_pulled_epoch"] = hf.get("handoffs_done", 0) + 1


# moved-from: herdd._handoff_job_accrue
def _handoff_job_accrue(jctx: MutableMapping[str, Any],
                        hf: MutableMapping[str, Any]) -> None:
    """BOTH boxes' cost counts against --budget while the understudy is a SEPARATE
    live box (design §3): add its burn to jctx['spend_usd'] AND hf['handoff_spend_usd']
    (the abort-rule / cost-event window tracker). Once it becomes jctx's tracked box
    the primary loop bills it — the id guard prevents a double-count."""
    # LANE DIVERGENCE (pinned): `str()==` here, RAW `==` in `_handoff_accrue`.
    # The jobs lane spells every box id as a string. Do NOT unify.
    if hf.get("phase") in (None, "IDLE", "DONE"):
        return
    if str(jctx.get("iid")) == str(hf.get("understudy_iid")):
        return
    if hf.get("understudy_status") not in bidpolicy.LIVE_STATES:
        return
    dph = hf.get("understudy_dph")
    dt = jctx.get("dt", 0.0) or 0.0
    if dph and dt > 0:
        amt = (dph / 3600.0) * dt
        jctx["spend_usd"] = jctx.get("spend_usd", 0.0) + amt
        hf["handoff_spend_usd"] = hf.get("handoff_spend_usd", 0.0) + amt


# moved-from: herdd._handoff_job_build_state
def _handoff_job_build_state(jctx: MutableMapping[str, Any],
                             hf: MutableMapping[str, Any]) -> dict[str, Any]:
    """Snapshot the observed world into the PURE mk_handoff_state (mirror of
    _handoff_build_state). While a retarget delete is stuck (§5), force
    understudy_producing False so DRAINING never destroys the primary — the husk
    stays parked (epoch-fenced) rather than risking a double-claim."""
    # FROZEN KEY CONTRACT: all 35 keys, by keyword, as a dict — see
    # `_handoff_build_state`. This lane passes the six work-awareness keys the
    # run lane omits; `driver_can_complete` fails CLOSED here (defect #61).
    producing = hf.get("understudy_producing", False) and not hf.get("retarget_incomplete")
    hs: dict[str, Any] = bidpolicy.mk_handoff_state(  # type: ignore[no-untyped-call]
        phase=hf["phase"], over_ceiling_streak=hf["over_ceiling_streak"],
        primary_iid=hf.get("primary_iid") or jctx.get("iid"),
        primary_bid=jctx.get("last_bid"), primary_on_demand=jctx.get("on_demand"),
        primary_dph=models._num_dph(jctx.get("last_bid"))
        or models._num_dph(jctx.get("dph")),
        primary_evicted=bool(jctx.get("primary_evicted")),
        primary_gone=hf.get("primary_gone", False),
        understudy_iid=hf.get("understudy_iid"), understudy_dph=hf.get("understudy_dph"),
        understudy_on_demand=hf.get("understudy_on_demand"),
        understudy_status=hf.get("understudy_status"),
        understudy_live_since=hf.get("understudy_live_since"),
        understudy_producing=producing,
        understudy_gone=hf.get("understudy_gone", False),
        drain_ts=hf.get("drain_ts"),
        candidate_min_bid=hf.get("candidate_min_bid"),
        candidate_on_demand=hf.get("candidate_on_demand"),
        remaining_wall_h=jctx.get("remaining_wall_h", 0.0),
        final_flush_seen=hf.get("final_flush_seen", False),
        fence_ts=hf.get("fence_ts"),
        ckpt_pulled_epoch=hf.get("ckpt_pulled_epoch"),
        handoff_started_ts=hf.get("handoff_started_ts"),
        handoff_spend_usd=hf.get("handoff_spend_usd", 0.0),
        handoffs_done=hf.get("handoffs_done", 0),
        cooldown_until=hf.get("cooldown_until", 0.0),
        budget_usd=jctx.get("budget_usd"), spend_usd=jctx.get("spend_usd", 0.0),
        now=jctx.get("now", 0.0),
        # --- work awareness (2026-08-08 22:17Z incident; tasks #61/#62/#67).
        # `driver_can_complete` FAILS CLOSED: a driver that has not declared
        # itself able to carry a migration to `complete` cannot arm one. fleetd
        # declares it (and only since the watch survives the primary's destroy —
        # defect #61); the legacy inline `job supervise` loop declares it because
        # it simply keeps ticking. Anything else — a new embedder, a test harness
        # driving the ladder by hand — gets the safe answer by default.
        driver_can_complete=bool(jctx.get("handoff_can_complete", False)),
        work_at_risk_h=jctx.get("work_at_risk_h", 0.0),
        running_unresumable=jctx.get("running_unresumable", 0),
        min_running_eta_s=jctx.get("min_running_eta_s"),
        ckpt_stale=bool(jctx.get("ckpt_stale", False)),
        unsafe_override=bool(jctx.get("handoff_unsafe_override", False)))
    # Assigned, not passed: `HandoffSnapshot` pins the factory's 35 keys exactly.
    # Absent => Zone S falls back to the HANDOFF_DWELL_POLLS count.
    hs["over_ceiling_since"] = hf.get("over_ceiling_since")
    return hs


# moved-from: herdd._job_handoff_reset
def _job_handoff_reset(hf: MutableMapping[str, Any], *, handoffs_done: int,
                       cooldown_until: float) -> None:
    # Undocumented in the flat file (no docstring there either). Same in-place
    # clear/update as `_handoff_reset`, seeded from the jobs factory so the three
    # jobs-only keys come back.
    fresh = _init_job_handoff_state()
    fresh["handoffs_done"] = handoffs_done
    fresh["cooldown_until"] = cooldown_until
    hf.clear()
    hf.update(fresh)


# moved-from: herdd._do_job_handoff_move
def _do_job_handoff_move(jctx: MutableMapping[str, Any],
                         hf: MutableMapping[str, Any],
                         act: Any) -> None:  # noqa: ANN401 — bidpolicy.HandoffAction
    """Execute ONE HandoffAction for the jobs lane (the I/O). Mirrors the run
    lane's _do_handoff_move; the kinds are the SAME (the pure handoff_poll is
    shared), only the I/O diverges (retarget instead of relabel; per-job markers)."""
    from vastlib.supervise import replacement

    k = act.kind
    dry = jctx.get("dry_run", False)

    if k == "arm":
        hf["phase"] = "ARMED"
        hf["handoff_started_ts"] = jctx.get("now", 0.0)
        hf["primary_iid"] = jctx.get("iid")
        epoch = hf.get("handoffs_done", 0) + 1
        hf["epoch"] = epoch
        hf["pending_jobs"] = list(jctx.get("pending_jobs", []))
        _p = models._job_primary_inst(jctx)               # P5: sizing snapshot —
        if _p is not None:                                # survives a later
            hf["primary_shape"] = {k2: _p.get(k2)         # instance-API miss
                                   for k2 in models._JOB_PRIMARY_SHAPE_KEYS}
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for jid in hf["pending_jobs"]:                    # per-JOB epoch marker (jobd.sh:74)
            _handoff_job_b2_write(jid, f"{epoch}.json",
                                  f'{{"epoch":{epoch},"job_id":"{jid}","armed_at":"{ts}"}}\n',
                                  dry_run=dry)
        journal._job_handoff_emit(jctx, "handoff_armed", epoch=epoch,
                                  over_ceiling_streak=hf["over_ceiling_streak"],
                                  primary_bid=jctx.get("last_bid"),
                                  on_demand=jctx.get("on_demand"),
                                  candidate_min_bid=hf.get("candidate_min_bid"),
                                  candidate_on_demand=hf.get("candidate_on_demand"),
                                  n_jobs=len(hf["pending_jobs"]))
        note = (f">> job handoff ARMED for box {jctx.get('iid')}: bid "
                f"${jctx.get('last_bid')} over ceiling; migrating "
                f"{len(hf['pending_jobs'])} job(s) to a cheaper box (epoch {epoch})")
        journal._job_handoff_journal(jctx, "armed", epoch=epoch,
                                     n_jobs=len(hf["pending_jobs"]),
                                     primary_bid=jctx.get("last_bid"),
                                     on_demand=jctx.get("on_demand"),
                                     candidate_min_bid=hf.get("candidate_min_bid"),
                                     remaining_wall_h=jctx.get("remaining_wall_h"),
                                     note=note)
        print(note)
        return

    if k == "launch_understudy":
        epoch = hf.get("epoch") or (hf.get("handoffs_done", 0) + 1)
        cid, dph, reason = replacement._launch_job_understudy(jctx, hf, epoch)
        if cid is None:                                   # no_offer vs unlaunchable
            return _do_job_handoff_move(
                jctx, hf, bidpolicy.HandoffAction("abort_reap",
                                                  reason or "understudy_unlaunchable"))
        hf["understudy_iid"] = cid
        hf["understudy_dph"] = dph
        hf["phase"] = "LAUNCHING"
        journal._job_handoff_emit(jctx, "handoff_launch", instance_id=cid, dph=dph,
                                  label=labels._job_handoff_label(jctx.get("iid")))
        print(f">> job handoff LAUNCH: understudy {cid} (bid ${dph}) warming")
        return

    if k == "warm_wait":
        return

    if k == "mark_synced":
        hf["phase"] = "SYNCED"
        journal._job_handoff_emit(jctx, "handoff_synced",
                                  instance_id=hf.get("understudy_iid"),
                                  ckpt_epoch=hf.get("ckpt_pulled_epoch"))
        return

    if k == "fence_primary":
        # Park the primary (cmd_job_retarget refuses a RUNNING job) and PIN its bid
        # below any market floor so vast can't auto-resume it and race the
        # understudy (§4 two-writer belt; the per-job epoch marker is the other).
        # Snapshot which pending jobs were RUNNING — the final_flush wait set.
        # Re-snapshot pending_jobs from the LIVE queue first (S1, 2026-07-18
        # review): a reconcile-adopted twin arrives with hf["pending_jobs"]==[]
        # (reconcile runs before the tick populates jctx), and tickets submitted
        # since ARM would otherwise be stranded on the retired primary. jctx's
        # copy is fresh every tick (the queue read is not gated by the fence).
        iid = hf.get("primary_iid") or jctx.get("iid")
        hf["pending_jobs"] = list(jctx.get("pending_jobs", []))
        hf["running_jobs"] = list(jctx.get("running_jobs", []))
        pin = _handoff_park_bid(jctx)
        # Remember what we are about to overwrite. A post-cutover abort has to
        # give this back, and resuming at the $0.001 park pin would leave the box
        # permanently unable to win its market.
        hf["prefence_bid"] = _prefence_bid(jctx.get("last_bid"), jctx.get("dph"))
        if dry:
            print(f"[dry-run] would park primary {iid} (job handoff fence) + pin bid "
                  f"-> ${pin}; await final_flush on {len(hf['running_jobs'])} running job(s)")
        elif iid is not None:
            ok, perr = lifecycle._put_state_soft(iid, "stopped")
            if ok:
                lifecycle._wait_states_soft(iid, {"stopped", "exited"}, 120)
            else:
                jctx["last_error"] = f"job handoff fence park failed: {perr}"
            lifecycle._put_bid_soft(iid, pin)
        hf["phase"] = "CUTOVER"
        hf["fence_ts"] = jctx.get("now", 0.0)         # opens the flush-timeout / stall clocks
        journal._job_handoff_emit(jctx, "handoff_fence", primary=iid,
                                  understudy=hf.get("understudy_iid"), pinned_bid=pin,
                                  running_jobs=len(hf["running_jobs"]))
        journal._job_handoff_journal(
            jctx, "fenced", primary=iid,
            understudy=hf.get("understudy_iid"), pinned_bid=pin,
            running_jobs=len(hf["running_jobs"]),
            note=f"two-writer fence OPEN: primary {iid} parked and "
                 f"bid-pinned to ${pin}; awaiting final_flush on "
                 f"{len(hf['running_jobs'])} running job(s) before "
                 f"understudy {hf.get('understudy_iid')} write-enables")
        return

    if k == "resume_understudy":
        # Post-flush cutover: RETARGET each pending ticket primary -> understudy
        # (same JOB_ID, log continues). cmd_job_retarget writes the new ticket then
        # DELETEs the old; we post-check the primary queue for any ticket whose
        # delete FAILED (§5 row): a stuck old ticket means the cutover is INCOMPLETE
        # — do NOT advance to drain/destroy (double-claim risk); keep the husk
        # parked (bid-pinned + epoch-fenced) and alert.
        u = hf.get("understudy_iid")
        primary = hf.get("primary_iid") or jctx.get("iid")
        fast = act.reason == "fast_cutover"
        if not hf.get("pending_jobs"):
            # S1 belt: the fast_cutover path (SYNCED -> here on primary eviction)
            # SKIPS fence_primary, so a reconcile-adopted twin can still arrive
            # with an empty snapshot — retargeting zero tickets would "complete"
            # a cutover that moved nothing. Fall back to the tick's live queue.
            hf["pending_jobs"] = list(jctx.get("pending_jobs", []))
        moved = []
        for jid in hf.get("pending_jobs", []):
            if dry:
                print(f"[dry-run] would retarget {jid}: {primary} -> {u}")
                moved.append(jid)
                continue
            try:
                cmd_job_retarget(argparse.Namespace(
                    job_id=jid, from_box=primary, box=u, dry_run=False))
                moved.append(jid)
            except SystemExit as e:                       # terminal/no-ticket/etc: skip
                print(f"!! job handoff: retarget {jid} skipped ({e})")
        # delete-failure detection: any moved ticket still in the OLD queue means
        # cmd_job_retarget's old-ticket delete failed (it warns + continues).
        stuck = []
        if not dry and not fast:
            try:
                remaining = set(jobmeta.list_queue(primary))
            except Exception:
                remaining = set()
            stuck = [j for j in moved if j in remaining]
        if stuck:
            hf["retarget_incomplete"] = stuck
            journal._job_handoff_emit(jctx, "handoff_cutover", understudy=u,
                                      reason=act.reason, retarget_incomplete=stuck)
            note = (f"!! job handoff cutover INCOMPLETE: old-ticket delete failed for "
                    f"{stuck} — NOT destroying primary {primary} (double-claim risk); "
                    f"husk stays parked + epoch-fenced. Delete "
                    f"jobs/queue/{primary}/<JID>.json by hand, then it drains.")
            journal._job_handoff_journal(jctx, "cutover", understudy=u,
                                         reason=act.reason,
                                         retarget_incomplete=stuck, note=note)
            print(note)
            return                                        # stay at CUTOVER; do not advance
        for jid in hf.get("pending_jobs", []):            # per-JOB promoted marker
            ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            _handoff_job_b2_write(jid, "promoted",
                                  f'{{"job_id":"{jid}","understudy":"{u}",'
                                  f'"epoch":{hf.get("epoch") or 0},"promoted_at":"{ts}",'
                                  f'"reason":"{act.reason}"}}\n', dry_run=dry)
        # BOX relabel (the run lane's _put_label_soft analogue). The cutover above
        # is a `job retarget` of the TICKETS, not a box relabel — but the understudy
        # is now the sole canonical writer, so its INSTANCE label must drop the
        # dead-primary handoff-twin marker. Live proof (canary-job 2026-07-15): box
        # 45006331 kept label `job:45005266:handoff` after primary 45005266 was
        # destroyed, confusing `ls`, reconcile-on-restart's :handoff twin scan
        # (_job_handoff_reconcile), and a FUTURE epoch-2 handoff from the promoted
        # box (which keys the new twin on jctx.iid == the understudy). Relabel
        # job:<oldprimary>:handoff -> job:<understudy> so it reads canonical and a
        # next handoff launches a correctly-keyed job:<understudy>:handoff twin.
        # Best-effort, mirroring the run lane (a failed relabel never wedges an
        # already-committed cutover — the retarget is what moved the work);
        # skipped on --dry-run.
        if dry:
            if u is not None:
                print(f"[dry-run] would relabel understudy {u} -> job:{u} (canonical)")
        elif u is not None:
            lok, lerr = lifecycle._put_label_soft(u, f"job:{u}")
            if not lok:
                print(f"!! job handoff: understudy {u} relabel job:{primary}:handoff "
                      f"-> job:{u} failed ({lerr}) — box stays tagged as a handoff "
                      f"twin of dead primary {primary} (cosmetic; cutover committed)")
        hf["phase"] = "DRAINING"
        if fast:
            hf["primary_gone"] = True                     # primary already evicted/gone
            # Belt (2026-07-18 review S2): an OUTBID husk is stopped-but-PRESENT
            # with its standing bid intact — the fast path skipped the fence's
            # bid pin, so a receding floor would auto-resume a box whose tickets
            # just moved (billing leak until jobd's no-job self-park). Pin it.
            # Best-effort: a truly-gone box 404s harmlessly.
            if not dry and primary is not None:
                lifecycle._put_bid_soft(primary, _handoff_park_bid(jctx))
        journal._job_handoff_emit(jctx, "handoff_cutover", understudy=u,
                                  reason=act.reason)
        note = (f">> job handoff CUTOVER: retargeted {len(moved)} job(s) to {u} "
                f"({act.reason})")
        journal._job_handoff_journal(jctx, "cutover", understudy=u, reason=act.reason,
                                     from_box=str(primary), to_box=str(u),
                                     n_jobs=len(moved), note=note)
        print(note)
        return

    if k == "drain_primary":
        if hf.get("retarget_incomplete"):                 # belt: never destroy a husk
            return                                        #   whose old ticket lingers
        iid = hf.get("primary_iid")
        okd, derr = lifecycle._destroy_soft(iid, dry_run=dry)
        if okd and (dry or _confirm_gone(iid)):
            hf["primary_gone"] = True
        else:
            jctx["last_error"] = f"job handoff drain: primary {iid} not gone ({derr})"
        return

    if k == "complete":
        done = hf.get("handoffs_done", 0) + 1
        u, udph = hf.get("understudy_iid"), hf.get("understudy_dph")
        journal._job_handoff_emit(jctx, "handoff_complete", understudy=u, dph=udph,
                                  handoffs_done=done,
                                  handoff_spend_usd=round(
                                      hf.get("handoff_spend_usd", 0.0), 4),
                                  spend_usd=round(jctx.get("spend_usd", 0.0), 4))
        note = (f">> job handoff complete: migrated box {jctx.get('iid')} -> "
                f"understudy {u} (window cost "
                f"{fmt.dollars(hf.get('handoff_spend_usd', 0.0))})")
        # Journalled like every other phase (2026-08-08). Before defect #61 was
        # fixed this transition was unreachable under fleetd, so there was
        # nothing to write; now that the watch survives the primary's destroy it
        # is the line that says the understudy inherited the budget cap.
        journal._job_handoff_journal(jctx, "complete", understudy=str(u), dph=udph,
                                     handoffs_done=done,
                                     window_cost_usd=round(
                                         hf.get("handoff_spend_usd", 0.0), 4),
                                     note=note)
        print(note)
        # promote: the loop now supervises the understudy under the normal ladder.
        jctx["_handoff_completed_iid"] = u
        jctx["_handoff_completed_dph"] = udph
        _job_handoff_reset(hf, handoffs_done=done,
                           cooldown_until=jctx.get("now", 0.0)
                           + bidpolicy.HANDOFF_COOLDOWN_S)
        return

    if k in ("abort_reap", "abort_unfence"):
        u = hf.get("understudy_iid")
        if k == "abort_unfence":
            # POST-cutover rollback. Order matters: move the tickets back BEFORE
            # resuming the primary, so a ticket never points at a box that is
            # about to run it while another copy is still queued elsewhere — the
            # same launch-then-move-then-dispose order the eviction ladder uses.
            _job_handoff_retarget_back(jctx, hf, dry=dry)
            _handoff_unfence_primary(
                hf.get("primary_iid") or jctx.get("iid"), hf, dry_run=dry,
                emit=lambda **f: journal._job_handoff_emit(jctx, "handoff_unfence",
                                                           **f),
                # belt for the $0.001 pin: if the pre-fence bid was never
                # recorded (a reconciled twin, a daemon restart mid-fence), the
                # box still gets a bid the POLICY would put rather than being
                # resumed at — or left at — the fence pin.
                policy_target=bidpolicy._bid_target(  # type: ignore[no-untyped-call]
                    jctx.get("market_min_bid"), jctx.get("max_bid"),
                    jctx.get("on_demand")))
        if u is not None:
            lifecycle._destroy_soft(u, dry_run=dry)
            if not dry:
                _confirm_gone(u)
        journal._job_handoff_emit(jctx, "handoff_abort", reason=act.reason,
                                  instance_id=u)
        note = (f">> job handoff aborted ({act.reason}); reaped understudy {u}, "
                + ("unfenced and staying on primary" if k == "abort_unfence"
                   else "staying on primary"))
        journal._job_handoff_journal(jctx, "abort", reason=act.reason, instance_id=u,
                                     abort_kind=k, note=note)
        print(note)
        _job_handoff_reset(hf, handoffs_done=hf.get("handoffs_done", 0),
                           cooldown_until=jctx.get("now", 0.0)
                           + bidpolicy.HANDOFF_COOLDOWN_S)
        return


# moved-from: herdd._job_handoff_retarget_back
def _job_handoff_retarget_back(jctx: MutableMapping[str, Any],
                               hf: MutableMapping[str, Any], *,
                               dry: bool = False) -> list[str]:
    """Move any ticket the cutover put on the understudy BACK to the primary.

    Only reached from the post-cutover abort, i.e. with the understudy dead or
    provably never producing, so there is no double-claim risk: the box those
    tickets name cannot run them.

    A cutover that moved ZERO tickets (the two-writer fence correctly refusing to
    retarget a job still RUNNING on the primary — the live 2026-08-05 shape) has
    nothing to do here, and that is the common case. Returns the ids moved back.
    """
    u = hf.get("understudy_iid")
    primary = hf.get("primary_iid") or jctx.get("iid")
    if u is None or primary is None or str(u) == str(primary):
        return []
    try:
        stranded = list(jobmeta.list_queue(u))
    except Exception as e:                                # noqa: BLE001
        print(f"!! unfence: could not read the understudy queue on {u} "
              f"({type(e).__name__}: {e}) — any ticket there is stranded; "
              f"`herdd job orphans` will find it")
        return []
    moved = []
    for jid in stranded:
        if dry:
            print(f"[dry-run] would retarget {jid} back: {u} -> {primary}")
            moved.append(jid)
            continue
        try:
            cmd_job_retarget(argparse.Namespace(
                job_id=jid, from_box=u, box=primary, dry_run=False))
            moved.append(jid)
        except SystemExit as e:
            print(f"!! unfence: retarget-back of {jid} skipped ({e})")
    if moved:
        journal._job_handoff_emit(jctx, "handoff_retarget_back", understudy=u,
                                  primary=primary, jobs=len(moved))
        print(f">> unfence: moved {len(moved)} ticket(s) back {u} -> {primary}")
    return moved


# moved-from: herdd._job_handoff_reconcile
def _job_handoff_reconcile(jctx: MutableMapping[str, Any],
                           hf: MutableMapping[str, Any]) -> None:
    """Reconcile-on-(re)start (HANDOFF_DESIGN §5 crash row): adopt a live
    job:<primary>:handoff twin left by a crashed supervisor rather than orphaning
    it, and resume the machine mid-flight. Bias: keep the confirmed writer. v1
    resumes a live twin at SYNCED (proceed to fence) or LAUNCHING (keep warming)."""
    label = labels._job_handoff_label(jctx.get("iid"))
    twin = None
    for i in jctx.get("instances", []) or []:
        if i.get("label") == label:
            twin = i
            break
    if twin is None:
        return
    live = (twin.get("actual_status") or "").lower() in bidpolicy.LIVE_STATES
    hf["understudy_iid"] = twin.get("id")
    hf["understudy_status"] = (twin.get("actual_status") or "").lower() or None
    hf["understudy_dph"] = models._num_dph(twin.get("dph_total"))
    hf["primary_iid"] = jctx.get("iid")
    hf["handoff_started_ts"] = jctx.get("now", time.time())
    hf["pending_jobs"] = list(jctx.get("pending_jobs", []))
    # SYNCED needs the affirmative jobd boot proof, same as the observe path —
    # a merely-live (possibly still-`loading`) twin resumes at LAUNCHING.
    synced = live and _handoff_job_understudy_synced(jctx, hf)
    hf["phase"] = "SYNCED" if synced else "LAUNCHING"
    if live:
        hf["understudy_live_since"] = jctx.get("now", time.time())
    if synced:
        hf["ckpt_pulled_epoch"] = hf.get("handoffs_done", 0) + 1
    journal._job_handoff_emit(jctx, "handoff_reconciled", understudy=twin.get("id"),
                              phase=hf["phase"], live=live)
    print(f">> adopted live job handoff understudy {twin.get('id')} "
          f"(resuming at {hf['phase']})")


# moved-from: herdd._job_handoff_defer
def _job_handoff_defer(jctx: MutableMapping[str, Any],
                       hf: MutableMapping[str, Any]) -> None:
    """The REFUSAL half of the §2.3 gate, said out loud once per distinct cause.

    Reached when the dwell is satisfied — the box genuinely is over its preferred
    ceiling and a genuinely cheaper offer is on the market — but the migration
    cannot pay for itself over the horizon that is actually left. Declining is
    the right answer; being silent about it was not. Until defect #63 the ONLY
    visible handoff decision was the one that fired, so `fleet status` showed an
    expensive box holding station with no reasoning attached, and the 2026-08-08
    post-mortem had to reconstruct the arithmetic from the market by hand.

    Journalled once per distinct CAUSE, not once per poll: the horizon shrinks a
    little every tick, so keying on the raw number would bury `fleet log` under a
    line a minute. The signature is what an operator would act on — whether the
    horizon is bounded at all, and which offer/primary pair is being priced.
    Same house style as `_job_eviction_replace`'s refusals: the numbers go IN the
    message, because a money decision nobody can reconstruct is not a bounded one."""
    target = bidpolicy._handoff_candidate_target(hf)  # type: ignore[no-untyped-call]
    primary = models._num_dph(jctx.get("last_bid")) or models._num_dph(jctx.get("dph"))
    if target is None or primary is None:
        return                                  # no priced candidate: nothing to explain
    rwh = jctx.get("remaining_wall_h")
    # `rwh is None` is UNKNOWN horizon and stays None-shaped in the signature and
    # in the journal field — never coerced to 0.0 (defect #67).
    sig = (rwh is None, round(target, 3), round(primary, 3))
    if hf.get("defer_sig") == sig:
        return
    hf["defer_sig"] = sig
    overhead = (primary + target) * bidpolicy.HANDOFF_WINDOW_H
    horizon = "an UNMEASURABLE" if rwh is None else f"~{int(rwh * 3600)}s of"
    note = (f"!! HANDOFF DEFERRED on {jctx.get('iid')}: "
            f"{len(jctx.get('running_jobs') or [])} running job(s), "
            f"{horizon} horizon left — migrating would cost "
            f"~{fmt.dollars(overhead)} to capture "
            f"{fmt.dollars(primary - target)}/hr; re-testing each poll")
    journal._job_handoff_journal(jctx, "deferred", primary_dph=primary,
                                 candidate_target=target,
                                 horizon_s=None if rwh is None else int(rwh * 3600),
                                 overhead_usd=round(overhead, 4),
                                 delta_dph=round(primary - target, 4),
                                 running_jobs=len(jctx.get("running_jobs") or []),
                                 note=note)
    print(note)


# moved-from: herdd._job_handoff_refusal_note
def _job_handoff_refusal_note(reason: str) -> str:
    """Operator-facing prose for a pure-core refusal reason. ETA holds carry
    their own number (`eta_<n>s`), so they are formatted rather than looked up."""
    if reason.startswith("eta_"):
        return (f"fence HELD: the closest RUNNING job is an estimated "
                f"{reason[4:]} from finishing (under "
                f"{bidpolicy.HANDOFF_FENCE_HOLD_ETA_S}s) — interrupting it now would "
                f"throw away work that is nearly done")
    return _HANDOFF_REFUSAL_NOTES.get(reason, reason)


# moved-from: herdd._job_handoff_refuse
def _job_handoff_refuse(jctx: MutableMapping[str, Any],
                        hf: MutableMapping[str, Any], reason: str) -> None:
    """The WORK-side refusals, said out loud once per distinct cause.

    Sibling of `_job_handoff_defer`, which explains the ECONOMIC refusal. These
    are the ones that would otherwise be silent in the worst way: a box sitting
    over its ceiling with a migration that is deliberately not happening, which
    from `fleet status` is indistinguishable from a migration nobody noticed.
    Deduped on the reason, not the tick, for the same reason the deferral is —
    the condition re-tests every poll and would otherwise write a line a minute."""
    if hf.get("refuse_sig") == reason:
        return
    hf["refuse_sig"] = reason
    note = (f"!! HANDOFF REFUSED on {jctx.get('iid')}: "
            f"{_job_handoff_refusal_note(reason)}; re-testing each poll")
    journal._job_handoff_journal(jctx, "refused", reason=reason,
                                 running_jobs=len(jctx.get("running_jobs") or []),
                                 work_at_risk_h=round(jctx.get("work_at_risk_h", 0.0), 4),
                                 eta_s=jctx.get("min_running_eta_s"),
                                 note=note)
    print(note)


# moved-from: herdd._job_handoff_progress_warn
def _job_handoff_progress_warn(jctx: MutableMapping[str, Any],
                               hf: MutableMapping[str, Any]) -> None:
    """HANDOFF_WARN_PCT advisory (task #67). Two lines, neither of them a gate:

      * a RUNNING job past HANDOFF_WARN_PCT is nearly done, and ANY interruption
        (handoff, eviction, a hand park) now costs most of a cell;
      * a RUNNING job past HANDOFF_WARN_PCT with `n_checkpoints: 0` is exposed to
        EVICTION as much as to us — spot delivers no signal, so nothing in this
        module can protect it. That is the honest scope boundary, and the alarm
        is the only part of this work that transfers to it.

    Once per job per condition; the latch clears when the condition does."""
    from vastlib.jobs import risk

    warned = hf.setdefault("pct_warned", {})
    for v in jctx.get("pending_views") or []:
        if not isinstance(v, dict) or v.get("display_status") != "running":
            continue
        pct = risk._job_pct(v)
        jid = v.get("job_id") or "?"
        if pct is None or pct < bidpolicy.HANDOFF_WARN_PCT:
            warned.pop(jid, None)
            continue
        key = bool(v.get("n_checkpoints"))
        if warned.get(jid) == key:
            continue
        warned[jid] = key
        note = (f"!! {jid} is {pct}% done on {jctx.get('iid')}"
                + ("" if key else " with NO checkpoint — an eviction or any "
                                  "interruption right now discards the attempt")
                + "; no migration will be started over it")
        journal._job_handoff_journal(jctx, "work_warning", job_id=jid, pct=pct,
                                     n_checkpoints=v.get("n_checkpoints") or 0,
                                     note=note)
        print(note)


# moved-from: herdd._job_handoff_tick
def _job_handoff_tick(jctx: MutableMapping[str, Any],
                      hf: MutableMapping[str, Any]) -> None:
    """One jobs-lane handoff step (mirror of _handoff_tick): advance the dwell
    counter, observe+accrue the understudy, read the fence signals, then run the
    PURE handoff_poll and execute its move. Called AFTER the primary's own poll
    machinery every tick when --handoff is set."""
    from vastlib.supervise import replacement

    # dwell hysteresis (§1): ARM only after HANDOFF_DWELL_S continuously over the
    # preferred ceiling. Count AND start timestamp, exactly as the run-lane twin.
    if jctx.get("_over_pref"):
        hf["over_ceiling_streak"] = hf.get("over_ceiling_streak", 0) + 1
        if hf.get("over_ceiling_since") is None:
            hf["over_ceiling_since"] = jctx.get("now")
    else:
        hf["over_ceiling_streak"] = 0
        hf["over_ceiling_since"] = None

    _handoff_observe_job_understudy(jctx, hf)
    # LANE DIVERGENCE (pinned): str()-shaped id guard, including the literal
    # "None". The run lane's twin compares raw. Do NOT unify.
    if hf.get("primary_iid") is None and hf["phase"] != "IDLE" \
            and str(jctx.get("iid")) not in (None, "None", str(hf.get("understudy_iid"))):
        hf["primary_iid"] = jctx.get("iid")
    _handoff_job_accrue(jctx, hf)

    # candidate market read once dwell is satisfied and still IDLE (feeds the pure
    # ARM headroom + candidate gate).
    if hf["phase"] == "IDLE" \
            and bidpolicy._handoff_dwell_satisfied(hf, jctx.get("now")):  # type: ignore[no-untyped-call]
        offer = replacement._job_understudy_offer(jctx, hf)
        hf["chosen_offer"] = offer
        hf["candidate_min_bid"] = models._num_dph(offer.get("min_bid")) if offer else None
        # market on-demand, never the bid row's dph_total (doc 50 R1 —
        # see _offer_ondemand_ref); None makes the pure ARM gate refuse.
        hf["candidate_on_demand"] = pricing._offer_ondemand_ref(offer) if offer else None

    # box-side fence signals only matter once the fence is open.
    if hf["phase"] in _HANDOFF_FENCE_OPEN:
        sig = _handoff_job_signals(hf.get("running_jobs", []),
                                   hf.get("pending_jobs", []),
                                   hf.get("understudy_iid"))
        if sig.get("final_flush_seen"):
            hf["final_flush_seen"] = True
        if sig.get("understudy_producing"):
            hf["understudy_producing"] = True

    _handoff_stall_alarm(hf, jctx.get("now", 0.0),
                         lambda **f: journal._job_handoff_emit(jctx, "handoff_stall",
                                                               **f))

    hs = _handoff_job_build_state(jctx, hf)
    hact = bidpolicy.handoff_poll(hs)                      # PURE (shared core)
    if hact.kind != "noop":
        _do_job_handoff_move(jctx, hf, hact)
        hf.pop("refuse_sig", None)            # a move retracts a standing refusal
    elif hact.reason == "candidate_reject":
        _job_handoff_defer(jctx, hf)
    elif hact.reason.startswith(("precondition:", "fence_hold:")):
        _job_handoff_refuse(jctx, hf, hact.reason.split(":", 1)[1])


# moved-from: herdd._job_handoff_reap_on_exit
def _job_handoff_reap_on_exit(jctx: MutableMapping[str, Any],
                              hf: MutableMapping[str, Any]) -> None:
    """Stop path (budget/drain/park/crash): a mid-flight PRE-cutover understudy is
    reaped so a stop never leaks a second box (§3). A post-cutover understudy is
    the canonical box (already promoted into jctx) — leave it.

    An OPEN fence is unwound here too (2026-08-08, task #62). CUTOVER means the
    primary is parked with its bid pinned to HANDOFF_PARK_BID and its tickets
    have NOT moved yet, so a supervisor that exits at that moment leaves a box
    that is off, unwinnable, and still owns the work. The pin must not outlive
    the fence window on ANY path, and 'the supervisor stopped' is a path.
    DRAINING is deliberately not unwound: there the tickets are already on the
    understudy, so resuming the primary would put a second claimant back on the
    board — the husk stays parked and the reaper owns it."""
    if not hf:
        return
    if hf.get("phase") == "CUTOVER" and hf.get("primary_iid") is not None:
        _job_handoff_retarget_back(jctx, hf, dry=jctx.get("dry_run", False))
        _handoff_unfence_primary(
            hf["primary_iid"], hf, dry_run=jctx.get("dry_run", False),
            emit=lambda **f: journal._job_handoff_emit(jctx, "handoff_unfence", **f),
            policy_target=bidpolicy._bid_target(  # type: ignore[no-untyped-call]
                jctx.get("market_min_bid"), jctx.get("max_bid"),
                jctx.get("on_demand")))
        journal._job_handoff_emit(jctx, "handoff_abort",
                                  reason="supervisor_stop_fenced",
                                  instance_id=hf.get("understudy_iid"))
    if hf.get("phase") in bidpolicy._HANDOFF_PRE_CUTOVER + ("CUTOVER",) \
            and hf.get("understudy_iid") is not None:
        lifecycle._destroy_soft(hf["understudy_iid"],
                                dry_run=jctx.get("dry_run", False))
        journal._job_handoff_emit(jctx, "handoff_abort", reason="supervisor_stop",
                                  instance_id=hf["understudy_iid"])
