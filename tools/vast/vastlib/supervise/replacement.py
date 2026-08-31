"""Replacement, understudy, re-bid and boot-SLA — the effectful drivers that
turn "we lost the box" into "we are running on a different box".

Why this module exists
----------------------
Clusters C24 (eviction-replace) + C27 (job-supervise drivers) of `herdd.py`:
everything that RENTS, CONDEMNS, RE-BIDS or RE-HOSTS. Both supervise lanes
reach it, in both directions, which is why the knot is ported as one unit
(plan §8 step 4) and why every cross-module call here — and every call INTO
here — is written in module-attribute form. A `from … import _launch_job_replacement`
anywhere in this ring is a call bound at import time, invisible to the 659
`monkeypatch.setattr` sites the suite steers these drivers with, and a test
that goes vacuously green while a real launch happens is the failure mode
plan §7.3 exists to kill.

Four properties of this file are money, not style, and a "cleanup" that
loses any of them re-buys a specific box:

* **The doc-50 dollar guards, four of them.** Never read an on-demand
  reference off a BID offer's `dph_total` (that field is the current
  interruptible price, ~min_bid + 0.5%, so substituting it manufactures the
  razor-thin bid the cushion rail exists to prevent — `_replacement_spot_walk`,
  `_handoff_understudy_body`, `_launch_job_understudy`, `_launch_job_replacement`'s
  clamp). The un-ceilinged on-demand re-probe in `_job_eviction_replace` exists
  because "no on-demand offer under the ceiling" and "no on-demand market at
  all" are DIFFERENT failures that the ceilinged probe collapses into one None.
  `max_dph` binds on the offer actually launched, not only inside the pure
  decision (R3). The returned `dph` is the REALIZED rate, never None while a
  meter runs (R4). Bought on 2026-08-05 for $3.4741/hr.
* **The `_sel` dict is a wire contract.** It is spread into BOTH the B2
  `eviction_replacement_decision` event and the `fleet log` ladder journal, and
  `fleet_report.py` schemas all ten of its keys — `disk_floor_gb` and
  `disk_blocked` included. Renaming one, or typing it into a model that drops
  unknown keys, silently breaks `fleet log` rendering. Same for the event names
  emitted here and the ladder-journal names.
* **The refusal-dedup latches are behaviour.** `_job_rebid_ladder` latches on
  `jc["rebid_refused"] == dec.reason`; `_job_eviction_replace` on
  `jc["replacement_refused"] == _reason` (INCLUDING the disk-note suffix). The
  decision is still re-MADE every tick; only the announcement dedups. Box
  47398836 wrote 79 identical refusals in 66 minutes without them.
* **The box-swap re-anchor block is deliberately duplicated** in
  `_job_pull_condemn` and `_job_eviction_replace`. Only the eviction path clears
  `not_live`/`was_live`, and `launch_dph_anchor` is reset in neither (three 2x
  replacements would license an 8x box). The 2026-08-10 review that created the
  two copies rejected sharing them.

Also load-bearing, and quieter: `is not True` in `_job_boot_sla_tick`'s pyhalf
check is back-compat (a strict `== ok` holds every older-bundle box to a
milestone it cannot signal); `pyhalf=broken` HOLDS rather than condemns; the
two boot clocks are anchored differently ON PURPOSE (`boot_running_since` for
env-setup, the instance's own `start_date` for the pull and the run lane) —
box 47166718 was condemned 82 seconds into `running` when they shared one
budget; `_job_replacement_verified` parses its bool BY HAND because
`_rebid_knob`'s `type(default)(v)` coercion makes `bool("0")` True; and the
bare `except Exception` in the pickers, the knobs, the per-ticket retarget loop
and both launchers is the policy — never kill the supervise loop — not a smell.

What is deliberately NOT here
-----------------------------
* **No lane unification.** Run-lane `_handoff_understudy_body`/`_handoff_pick_offer`
  and jobs-lane `_launch_job_understudy`/`_job_understudy_offer` are parallel by
  design (plan §5 NOTE, v1 §7, FLEET_REVIEW item 1): the run lane prices through
  `_relaunch_body`, the jobs lane builds an `argparse.Namespace` for `_do_launch`;
  the run lane's probe budget RETURNS None where `_replacement_spot_walk` BREAKS
  and keeps the priced prefix; only the jobs lane rolls the image forward and
  resizes the disk. Do not extract a shared helper.
* **No decision arithmetic.** `bidpolicy` owns `rebid_ladder`,
  `replacement_decision`, `bid_decision` and the ceilings; `ladder_core` owns
  `box_swap_reset`. This file is I/O, journaling and the order of operations.
* **No offer search.** `market.offers` owns `pick_offers` and the
  minimum-requirements candidate class; the disk floor lives in `pick_offers`,
  and `_job_replacement_offers` passes `disk_gb=` straight through rather than
  re-filtering.
* **No lane tick.** `job_supervise_tick` / `supervise_tick` are `job_lane.py` /
  `run_lane.py`; the retain-or-destroy sweep is `retention.py`. They call in
  here; nothing here drives a loop.
* **No typed state.** `st` / `jc` / `hf` are annotated `MutableMapping[str, Any]`
  on purpose this wave. `state.py` documents the key inventory and is wired into
  these annotations post-wave; tri-state float keys stay None-for-UNKNOWN (never
  `or 0.0`, defect #67), and `evicted_machines` stays a SET while
  `evicted_machine_ts` is str-keyed — the asymmetry is persisted in state.json
  and a pre-2026-08-16 file must keep degrading to permanent exclusion.

`vastlib.market.offers` is bound as `market_offers` because three ported bodies
(`_replacement_fit`, `_replacement_spot_walk`, `_replacement_ondemand_walk`) take
a parameter named `offers` and `_relaunch` assigns a local of that name — the
verbatim body wins over the import alias. `vastlib.launch.spec` is bound TWICE
for the same reason: as `spec` (what most bodies here call it) and as
`launch_spec`, because `_relaunch_body` opens with `spec = st.get("launch_spec",
{})` and would otherwise shadow the module out from under its own
`_require_image` / `_resolve_secret` / `image_login_arg` calls. Deleting either
alias is a NameError, not a tidy-up.

Provenance: moved from `tools/vast/herdd.py` at rev a1f2c8a5 (plan §8 step 4),
manifest `.port_manifests/sup-replacement.json`. Behavior-preserving: bodies
verbatim, annotations and the empty-container declarations mypy strict requires
added, cross-module names repointed to their vastlib homes.

RE-PORTED 2026-08-16 (plan §8 no-freeze drift duty) over peer commits
830579df / 8b984898 / d5b0b773 — the notify-S2b slice. Three deltas landed
here: `_job_defense_inputs` / `_job_defense_cap` were EXTRACTED out of
`_job_rebid_ladder` so the notification-priced rescue quote's ceiling and the
re-bid rung's are one derivation of the same six numbers rather than two;
`_job_eviction_replace` gained `exclusion_class`, because a notification
refines the class we ACT on and must not shorten how long we remember that a
machine took our box; and both box-swap sites (`_job_pull_condemn`,
`_job_eviction_replace`) now retire the notify latch through
`job_lane._job_notify_box_swap_reset`. `_rebid_knob` and
`_job_excluded_machines` were listed as stale and measured byte-identical —
the new code merely landed next to them.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import statistics
import subprocess
import sys
import time
from collections import namedtuple
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
    TypeVar,
)

import disksize
import imageref
import ladder_core

from vastlib.boxes import health, lifecycle, ssh
from vastlib.core import acctfault, api, config, fmt, labels, models
from vastlib.jobs import risk
from vastlib.launch import launch, spec
from vastlib.launch import spec as launch_spec
from vastlib.market import hostrep, pricing
from vastlib.market import offers as market_offers
from vastlib.storage import b2
from vastlib.supervise import handoff, job_lane, journal, retention, run_lane, serve_ident

import bidpolicy
import jobmeta
import runmeta

if TYPE_CHECKING:
    # Types only. The RUNTIME import is function-local and soft (`_train_rate_soft`)
    # for the same contract `gpu_rates` has: the module is optional here.
    from train_rates import Family as _Family
    from train_rates import RateEstimate as _RateEstimate

#: Knob value type — `_job_replacement_knob` / `_rebid_knob` coerce every source
#: with `type(default)(v)`, so the resolved value has the DEFAULT's type. That
#: coercion is also why `_job_replacement_verified` parses its boolean by hand
#: instead of routing through them (`bool("0")` is True).
_KnobT = TypeVar("_KnobT")


# --------------------------------------------------------------------------- #
# THE LAST SEAM IN THIS BLOCK IS CLOSED (step 6d, at the thinning).
# `_reset_run_markers` raised here from step 4 until the launcher cutover, on
# the theory that its home was a module plan §5 had not built yet. It is not:
# see the home ruling on the def below. The body is now MOVED (not
# reimplemented) from `herdd.py`, so the loud failure this stub existed to
# produce — step 6, when `herdd.py` starts calling this module — never had to
# fire.
#
# THE OTHER THREE SEAMS THIS BLOCK CARRIED AT STEP 4 ARE CLOSED (step 6d).
# `_boot_deadline_backoff` and `_relaunch_body` are ported below — bodies MOVED
# from `herdd.py`, not reimplemented — and `_confirm_gone` forwards to the one
# copy that already landed in `boxes/lifecycle.py` (its marker is there).
# `handoff.py`'s twin `_confirm_gone` stub closes the same way. Each module
# still OWNS the attribute its own callers patch — the precedent
# `launch/launch.py::_launch_preflight` set, and the reason `lifecycle.py`'s own
# header says a patch of `lifecycle._confirm_gone` is not seen through a
# supervise-side binding — so the 7 `monkeypatch.setattr` sites keep steering
# while there is still exactly ONE body.
# --------------------------------------------------------------------------- #

# HOME RULING (step 6d, at the thinning). The stub above pointed at
# `storage.b2`; that module's own header refuses the job — "No bucket, key or
# path policy. This module moves bytes at whatever path it is given" — and this
# function is nothing BUT path policy (which markers a relaunch stamps and which
# it clears). Its twin `_handoff_b2_write` landed in `supervise/handoff.py` for
# exactly that reason, and `_relaunch` below is its only caller. So the body
# moves here, verbatim, and this module keeps owning the attribute the suite
# steers. The bytes still leave through `storage.b2` — one transport, one body.
# moved-from: herdd._reset_run_markers
def _reset_run_markers(run_id: object, dry_run: bool = False) -> None:
    """Mirror cmd_train's launch-time marker hygiene on a relaunch (G6): stamp
    checkpoints/<id>/STATUS=RELAUNCHED and drop stale STOP/EXTEND so a prior
    run's debug-hold markers can't tear the fresh box down. Best-effort; no
    B2_BUCKET or --dry-run => skip (the supervise driver tests run with
    B2_BUCKET unset)."""
    bucket = os.environ.get("B2_BUCKET")
    if not bucket or dry_run:
        return
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    b2._b2_rcat(f"b2:{bucket}/checkpoints/{run_id}/STATUS", f"RELAUNCHED {ts}\n",
                hard=False)
    b2._rclone_soft(["deletefile", f"b2:{bucket}/checkpoints/{run_id}/STOP"])
    b2._rclone_soft(["deletefile", f"b2:{bucket}/checkpoints/{run_id}/EXTEND"])


# --------------------------------------------------------------------------- #
# THE RUN LANE'S EFFECTFUL DRIVERS (plan §8 step 6d, integrator ruling).
#
# `run_lane.py` shipped five of these as raising SEAM stubs at step 4
# (`_observe`, `_accrue_cost`, `_emit_cost`, `_do_bid_move`,
# `_supervise_boot_health`), and `sup-replacement.json` deferred `_relaunch_body`
# to `run_lane.py` while the step-4 run-lane port did not carry it. The
# integrator ruling puts all six HERE, with the rest of the effectful drivers,
# and OVERRULES the `-> supervise.run_lane` pointer the `_relaunch_body` stub
# carried:
#
#   * `_relaunch_body` lands beside its THREE callers — `_relaunch`,
#     `_handoff_understudy_body` and the missing-secret preflight — all of which
#     are in this file. Homing it in `run_lane.py` would have made one function
#     cost a sibling edge in both directions.
#   * `_supervise_boot_health` is the twin of `_supervise_boot_sla` below: same
#     verdict vocabulary (None | "condemned" | "stop_fatal" | "stop_budget"),
#     same exclusion-set write, same destroy -> `_confirm_gone` -> `_relaunch`
#     tail. Splitting the pair across two modules is how the two watchdogs
#     drift apart, and one of them is the default-ON backstop for the other.
#   * `_observe` / `_accrue_cost` / `_emit_cost` / `_do_bid_move` are the run
#     lane's I/O, and `run_lane.py`'s own header already declares that "the
#     effectful drivers ... are `replacement.py`".
#
# `run_lane.py` keeps a one-line forwarder per name so BOTH patch surfaces
# steer. `monkeypatch.setattr(replacement, "_observe", ...)` reaches every lane
# caller (the forwarder resolves the attribute at CALL time, never at import),
# and a test that patches `run_lane._observe` still shadows the lane's own call
# sites. That is the both-directions module-attribute rule this module's header
# states, applied to a lane boundary instead of a ring boundary.
#
# ONE import edge is new and deliberate: this module now imports `run_lane`, for
# `_observe`'s single `_self_floor_guard` call. That wrapper is the run lane's
# THREE lane-specific facts over the shared `ladder_core.self_floor_guard` and
# holds lane-scoped hooks (`_RUN_FLOOR_HOOKS`), so it may not be shared with the
# jobs lane's twin and stays where it is. `replacement` <-> `job_lane` is already
# a mutual sibling import for exactly this reason; the §5 DAG joins the supervise
# ring's siblings with `:` (NON-independent), so the edge is inside the
# contract, and every use of it is at call time.
# --------------------------------------------------------------------------- #

# moved-from: herdd._boot_deadline_backoff
def _boot_deadline_backoff(base_s: float, kills: object) -> float:
    """Escalating boot-SLA tolerance (owner directive 2026-08-03, requirement
    3): the first BOOT_SLA_MAX_KILLS consecutive boot-SLA kills on one watch
    use the base deadline; each further kill WIDENS it by
    BOOT_SLA_BACKOFF_MULT (600s, 600s, 1200s, ... at defaults), so an owning
    watch cannot flap-loop through a market where every available host boots
    slowly. BOOT_MAX_HOST_RETRIES remains the hard disarm above this."""
    # Two annotation-only concessions, both because `kills` is `object` (its
    # callers pass a raw `jc`/`st` value): `max(0, int(...))` is an untypeable
    # overload against `object`, and `float ** int` is `Any` in typeshed's
    # general `__pow__`. Neither narrows anything at runtime — the coercion the
    # flat body does is exactly the coercion here.
    k = max(0, int(kills or 0))  # type: ignore[call-overload]
    lim = config._boot_knob("BOOT_SLA_MAX_KILLS", cast=int)
    if k < lim:
        return float(base_s)
    return float(base_s) * (float(config._boot_knob("BOOT_SLA_BACKOFF_MULT"))  # type: ignore[no-any-return]
                            ** (k - lim + 1))


# FORWARDER, not a copy. The one body is `boxes/lifecycle.py::_confirm_gone`
# (ported step 3, `moved-from:` marker there); this module owns the ATTRIBUTE
# because the flat suite steers the name at the supervise seam and a
# `lifecycle._confirm_gone(...)` call written inline here would not see that
# patch. Both patch surfaces work: `replacement._confirm_gone` shadows this
# def, `lifecycle._confirm_gone` is resolved at call time inside it.
def _confirm_gone(iid: object, tries: int = 6) -> bool:
    """True once vast no longer reports the instance present (destroy confirmed).
    Treats {"instances": null}/None (HTTP 200 for a gone box) and HTTP 404 as
    gone. Enforces destroy-husk-before-relaunch (never launch a twin over a live
    husk)."""
    return lifecycle._confirm_gone(iid, tries)


# moved-from: herdd._observe
def _observe(st: MutableMapping[str, Any],
             a: argparse.Namespace) -> MutableMapping[str, Any]:
    """One I/O observation (the only place the loop reads the world). Mutates &
    returns st; sets st['obs_status'] in {ok,transient,fatal}. INVARIANT: a
    transient API failure yields obs_status='transient' + an 'unknown' view and
    does NOT advance the not-live streak — an outage can never look like an
    eviction. {"instances": null}/None is treated as gone, never dereferenced."""
    run_id = st["run_id"]
    now = time.time()
    st["now"] = now
    st["dt"] = now - st.get("_last_obs_t", now)
    st["_last_obs_t"] = now
    st["wall_clock_s"] = now - st["_t0"]

    ok, data, err = api.request_soft("GET", "v1/instances/")
    if not ok:
        st["last_error"] = err
        if api._classify_http(err) == "fatal":
            st["obs_status"] = "fatal"
            return st
        st["obs_status"] = "transient"
        st["actual_status"] = "unknown"
        st["view"] = {"status": "unknown", "display_status": "unknown",
                      "_cache_stale": True}
        return st                                     # streak NOT advanced

    instances = data.get("instances", data) if isinstance(data, dict) else data
    instances = instances or []
    st["_instances"] = instances                      # reused by the handoff observer
    live = lifecycle.live_run_instances(run_id, instances=instances)
    inst = live[0] if live else next(
        (i for i in instances if models._instance_run_label(i) == run_id), None)

    if inst is None:                                  # gone (never deref)
        st["present"] = False
        st["actual_status"] = None
        st["intended_status"] = None
        st["stopping_reason"] = None
        # keep last-known husk_id so a host-death relaunch can still destroy it
    else:
        st["present"] = True
        st["husk_id"] = inst.get("id")
        st["instance_id"] = inst.get("id")
        st["machine_id"] = inst.get("machine_id") or st.get("machine_id")
        st["actual_status"] = (inst.get("actual_status") or "").lower() or None
        st["intended_status"] = (inst.get("intended_status") or "").lower() or None
        st["stopping_reason"] = inst.get("status_msg") or inst.get("stopping_reason")
        dph = models._num_dph(inst.get("dph_total"))
        if dph is not None:
            st["dph_total"] = dph
        st["num_gpus"] = inst.get("num_gpus") or st.get("num_gpus")
        st["is_bid"] = bool(inst.get("is_bid"))
        # The STANDING BID is `dph_base`, NOT the billed `dph_total` (= bid +
        # storage) — see `_instance_standing_bid`. Seeding `last_bid` from the
        # total puts it one storage sliver ABOVE the number vast reports back as
        # the chunk's `min_bid`, and `market_floor_is_self` is an exact-equality
        # test by design, so the self-floor guard below could not recognise our
        # own bid. `dph` stays the fallback for a body without `dph_base`.
        #
        # Seed + echo-record + RECONCILE (review 2026-08-10, F2/M3) are the
        # shared `ladder_core.reconcile_standing_bid` — the jobs lane's tick
        # drives the identical call. This lane's two facts: the rate-limit clock
        # is named `last_bid_put_ts` here and `last_bid_put` there (both
        # persisted names, D4), and this lane has never carried the jobs lane's
        # "box reports no dph_base" warning (D3, an unfixed parity gap recorded
        # in AUTOBID_DESIGN.md §"One core, two lanes" rather than repaired
        # inside a behavior-preserving refactor).
        _true_bid = models._instance_standing_bid(inst)
        ladder_core.reconcile_standing_bid(  # type: ignore[no-untyped-call]
            st, is_bid=st["is_bid"], true_bid=_true_bid, dph=dph,
            machine_id=st.get("machine_id"),
            now=st.get("now") or time.time(),
            put_ts_key="last_bid_put_ts",
            on_reconcile=lambda old, new: print(
                f".. standing bid reconciled from the box: "
                f"${old} (lane belief) -> ${new} "
                f"(observed dph_base)"))

    live_now = st["present"] and st["actual_status"] in bidpolicy.LIVE_STATES
    st["not_live_streak"] = 0 if live_now else st.get("not_live_streak", 0) + 1
    # one soft offers read per tick -> market pressure (SPOT_DESIGN §3.2). Failure
    # degrades to None (bid actions disabled) and NEVER touches the streak above.
    # The RAW read is kept beside the guarded one: a suppressed self-echo must
    # stay distinguishable from a FAILED read, because poll() rule 2a's
    # underbid-park carve-out (_underbid_parked) is a diagnostic, not a price
    # input — collapsing both to None made vast's own underbid park read as
    # operator intent and terminally abandoned the box (review 2026-08-10, #1).
    _mr = (pricing._market_min_bid_read(st.get("machine_id"), st.get("num_gpus"))
           if st.get("machine_id") else models.MarketRead(False, False, None))
    _raw_floor = _mr.min_bid if _mr.ok else None
    st["market_min_bid_raw"] = _raw_floor
    # Tenant gate, not liveness gate (review 2026-08-10, #3): a healthy box
    # can transiently flap running->exited->running (measured, probe v2) with
    # intended_status still `running` — the chunk is still OURS, but a
    # live-only gate dropped the guard and the raw echo reached _bid_action's
    # rescue, which PUT a 20% raise priced off our own bid on every flap. A
    # genuine outbid/underbid park reports intended `stopped`, so the rescue
    # ladder keeps the raw read exactly where it needs it. (The run lane has
    # no resume-in-place rung consuming the floor while not-live; the jobs
    # lane does, so it keeps a live gate and refuses only the rescue RAISE.)
    _still_tenant = live_now or (st["present"] and st.get("is_bid")
                                 and st.get("intended_status") == "running")
    st["market_min_bid"] = run_lane._self_floor_guard(
        st, _raw_floor, live=_still_tenant,
        floors=(_mr.floors if _mr.ok else None), scaled=_mr.scaled)
    # on-demand price for the ceiling anchor (AUTOBID_DESIGN); None disables the
    # on-demand clamp + falls the default ceiling back to the median-floor path.
    st["on_demand"] = pricing._sticky_on_demand(
        st, pricing._market_ondemand_soft(st.get("machine_id"), st.get("num_gpus")))

    live_iids = {i.get("id") for i in live}
    st["view"] = spec._read_run_soft(run_id, live_iids=live_iids)
    st["stopping_actor"] = spec._last_stopping_actor(run_id)
    # STATUS marker only in the exact window final_status can infer a terminal (I4)
    if (not live_now and not st["present"]
            and st["view"].get("status") not in ("done", "failed")):
        st["status_marker"] = spec._status_marker_soft(run_id)
    else:
        st["status_marker"] = None
    st["obs_status"] = "ok"
    return st


# moved-from: herdd._accrue_cost
def _accrue_cost(st: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """spend_usd += dph_total/3600 * dt, ONLY while live (design §5). dph_total is
    the instance-body value refreshed each observe."""
    live = st.get("present") and st.get("actual_status") in bidpolicy.LIVE_STATES
    dph = st.get("dph_total")
    dt = st.get("dt", 0.0) or 0.0
    if live and dph and dt > 0:
        st["spend_usd"] = st.get("spend_usd", 0.0) + (dph / 3600.0) * dt
    return st


# moved-from: herdd._emit_cost
def _emit_cost(st: MutableMapping[str, Any], run_id: object) -> None:
    """Cumulative-to-date cost snapshot (fold takes max, so cumulative is safe)."""
    journal._sup_emit(run_id, "cost",  # type: ignore[arg-type]  # run_id is str at every call site
                      cost_usd=round(st.get("spend_usd", 0.0), 4))
    st["_last_cost_emit_t"] = time.time()


# moved-from: herdd._do_bid_move
def _do_bid_move(st: MutableMapping[str, Any], a: argparse.Namespace,
                 act: Any) -> None:  # noqa: ANN401 — bidpolicy.Action
    """Execute a poll() raise_bid/rescue_bid (SPOT_DESIGN §3.2): PUT the new bid,
    roll the cost basis (dph_total) onto it so accrual follows the raise, emit
    `bid_raised`. A rescue additionally arms the resume_wait deadline. A 429 or
    transient PUT failure is a SKIP (retry next poll), never fatal — but the
    rate-limit clock (last_bid_put_ts) is advanced on ANY real PUT, success or
    not, so a 429 storm can't re-issue every --interval (<60s) and deepen the
    throttling; the next attempt waits out the full BID_RATE_LIMIT_S window."""
    phase = {"raise_bid": "defend", "rescue_bid": "rescue",
             "lower_bid": "decay"}.get(act.kind, act.kind)
    target = bidpolicy._bid_target(  # type: ignore[no-untyped-call]
        st.get("market_min_bid"), st.get("max_bid"), st.get("on_demand"))
    iid = st.get("instance_id")
    if target is None or iid is None:
        return
    old = st.get("last_bid")
    floor = st.get("market_min_bid")
    if a.dry_run:
        print(f"[dry-run] would PUT bid {iid}: bid ${old} -> ${target}, "
              f"floor ${floor} ({phase})")
        ok, err = True, None
    else:
        st["last_bid_put_ts"] = time.time()           # real PUT issued: start the
        ok, err = lifecycle._put_bid_soft(iid, target)  # 60s clock even on failure
    if not ok:
        st["last_error"] = err                        # 429/transient -> skip, retry
        return                                         # after the window (ts advanced)
    st["last_bid"] = target
    st["dph_total"] = target                          # accrual now follows the move
    # one log line per REAL bid change, up OR down, naming both numbers and the
    # floor (never our own bid mislabeled as 'market' — the 2026-07-12 confusion).
    print(f">> bid ${old} -> ${target}, floor ${floor} ({phase})")
    # `bid_lowered` is a distinct, fold-tolerated event so a decrease is never
    # mislabeled as a raise in the run's event log.
    journal._sup_emit(st["run_id"],
                      "bid_lowered" if act.kind == "lower_bid" else "bid_raised",
                      old=old, new=target, market_min_bid=floor, phase=phase)
    if act.kind == "rescue_bid":
        st["rescue_attempted"] = True
        st["rescue_deadline"] = time.time() + a.rescue_wait


# moved-from: herdd._supervise_boot_health
def _supervise_boot_health(st: MutableMapping[str, Any], a: argparse.Namespace, *,
                           get_instance: Callable[[Any], Mapping[str, Any] | None]
                           | None = None,
                           now: Callable[[], float] | None = None) -> str | None:
    """Opt-in (`--boot-health`) boot-throughput watchdog tick for the supervise
    loop (BOOT_HEALTHCHECK phase P0). Runs ONLY while the box is pre-`running`
    (actual_status in loading/created); once it reaches running the sampler is
    retired. On a sustained-slow image pull (< BOOT_MIN_MBPS over a FULL
    BOOT_MBPS_WINDOW_S of downloading-phase samples) it CONDEMNS: emits a
    `boot_killed_slow` runmeta event, records the machine in the run's exclusion
    set, destroys the box (never park — nothing warm), and relaunches through
    the SAME eviction machinery (`_relaunch`, counting against --max-relaunch,
    with the exclusion set applied via build_search_query). Composes with — never
    replaces — the box's own fixed self-park deadline.

    Returns:
      None         — not applicable / box healthy so far (caller proceeds)
      "condemned"  — slow box torn down + relaunch attempted (caller sleeps+continues)
      "stop_fatal" / "stop_budget" — a relaunch guard tripped (caller breaks)."""
    if not getattr(a, "boot_health", False):
        return None
    get_instance = get_instance or health._get_instance_soft
    now = now or time.time
    # Retire the sampler the moment the box is no longer pre-running: a healthy
    # box that reached running (or a gone box the eviction ladder owns) starts
    # fresh next time.
    if not (st.get("present") and st.get("actual_status") in health._BOOT_LOADING_STATES):
        st["boot_sampler"] = None
        st["boot_sampler_iid"] = None
        return None
    iid = st.get("instance_id")
    if iid is None:
        return None
    if st.get("boot_sampler_iid") != iid or st.get("boot_sampler") is None:
        st["boot_sampler"] = health.BootThroughputSampler(
            min_mbps=config._boot_knob("BOOT_MIN_MBPS"),
            window_s=config._boot_knob("BOOT_MBPS_WINDOW_S", cast=int),
            deadline_s=10 ** 9, start_t=now())        # deadline owned by the box's self-park
        st["boot_sampler_iid"] = iid
    inst = get_instance(iid)
    if inst is None:
        return None                                   # failed poll: no sample
    verdict = st["boot_sampler"].feed(inst, now())
    if verdict != "slow":
        return None

    samp = st["boot_sampler"]
    window_s = int(config._boot_knob("BOOT_MBPS_WINDOW_S", cast=int))
    machine = inst.get("machine_id") or st.get("machine_id")
    journal._sup_emit(st["run_id"], "boot_killed_slow", instance_id=iid,
                      machine_id=machine, mbps=round(samp.last_mbps or 0.0, 3),
                      window_s=window_s, phase=samp.phase)
    if machine is not None and machine not in st["excluded_machines"]:
        st["excluded_machines"].append(machine)
    a.exclude_machines = list(st["excluded_machines"])     # -> build_search_query
    # Mirror poll()'s eviction guardrail: a run that keeps landing on slow hosts
    # terminates as max_relaunch instead of looping teardown+launch forever.
    if bidpolicy._guardrail_exceeded(st) == "max_relaunch":  # type: ignore[no-untyped-call]
        st["last_error"] = "max_relaunch (boot throughput kills)"
        return "stop_fatal"
    # Destroy + CONFIRM gone so _relaunch's adopt-live-twin step can't re-adopt
    # this still-live slow box as its own replacement.
    okd, derr = lifecycle._destroy_soft(iid, dry_run=getattr(a, "dry_run", False))
    if not okd:
        st["last_error"] = f"boot-kill destroy failed: {derr}"
        return "condemned"                            # retry the destroy next tick
    if not getattr(a, "dry_run", False) and not _confirm_gone(iid):
        st["last_error"] = f"boot-kill: {iid} not confirmed gone"
        return "condemned"
    st["husk_id"] = None
    st["instance_id"] = None
    st["boot_sampler"] = None
    st["boot_sampler_iid"] = None
    verdict2 = _relaunch(st, a)
    if verdict2 in ("stop_budget", "stop_fatal"):
        st["last_error"] = st.get("last_error") or verdict2
        return verdict2
    return "condemned"


# moved-from: herdd._relaunch_body
def _relaunch_body(st: MutableMapping[str, Any], a: argparse.Namespace,
                   bid: float | None, label: str | None = None,
                   key_name: str | None = None,
                   ) -> tuple[dict[str, Any], list[str]]:
    """Ask body from the captured launch spec + run: label + resume bid. Secret
    env values (spec.secret_env_keys) are re-injected here from the local env/.env
    by NAME — never read back from B2 — and image_login is re-derived from the
    local secret the image's registry needs (the spec stores only a redacted
    marker). Returns (body,
    missing): `missing` lists secret NAMES with no local value, so the caller
    refuses to launch a box with absent creds rather than 401 on the pull.
    RUNSET is forced into the box env (fixes G1: it never reached the relaunch).

    `label`/`key_name` override the launch label and the minted B2 key name; both
    default to the plain-relaunch values (`run:<run_id>` / `run-<run_id>`) so the
    eviction path is byte-identical. The handoff understudy builder passes the
    :handoff label and a nonce-suffixed key name (T3).

    NOTE the local `spec` below SHADOWS this module's `vastlib.launch.spec`
    import for the length of this body — the verbatim body wins — which is why
    `_require_image` / `_resolve_secret` / `image_login_arg` are reached through
    the second alias `launch_spec` (module header, alias note)."""
    # Secret NAMES whose absence must not refuse a relaunch. Gate (0.5) turns
    # `missing` into stop_fatal — right for a credential (a box launched without
    # one 401s on its first pull), wrong for a PERFORMANCE var: B2_CDN_PREFIX is
    # secret-classified because it is a URL bearer, but a box without it simply
    # takes the b2x -> rclone ladder, which is what every box did before the CDN
    # existed. Trading a recoverable evicted run for a slower pull is a bad deal.
    _soft = ("B2_CDN_PREFIX",)
    spec = st.get("launch_spec", {})
    # Normally the recorded spec carries it. If neither the spec nor --image has
    # one, REFUSE: relaunching a run on a stock image would silently move its
    # second half onto a different env than its first — the same faithful-replay
    # concern the digest pinning below exists for, in its most extreme form.
    image = launch_spec._require_image(spec.get("image") or getattr(a, "image", None),
                                       "relaunch")
    # ROLLING RELEASE (owner ruling 2026-08-04): a relaunch takes whatever the
    # tag means NOW. This used to replay the recorded digest when the tag had
    # moved (velvet P4a "faithful replay"); resurrecting an old digest is what
    # the ruling forbids, and it did it invisibly on the eviction path. The
    # recorded digest is still read — it is what lets us TELL the operator the
    # run's second half is on different bytes than its first. Recording, not
    # pinning; see imageref.pin_relaunch_image.
    if spec.get("image_digest") and image:
        image, _pin_state, _pin_why = imageref.pin_relaunch_image(
            image=image, spec_digest=spec.get("image_digest"),
            current_digest=imageref.image_tag_digest(image))
        if _pin_state in (imageref.PIN_ROLLED, imageref.PIN_UNVERIFIED):
            print(f"    image: {_pin_why}")
    body = {
        "image": image,
        "disk": spec.get("disk") or getattr(a, "disk", None),
        "runtype": spec.get("runtype") or getattr(a, "runtype", "ssh_direct"),
        "label": label or f"run:{st['run_id']}",
        "price": bid,
    }
    env = dict(spec.get("env") or {})
    missing = []
    for name in spec.get("secret_env_keys") or []:
        # mint a fresh run key only on the REAL relaunch (bid set, not log-only),
        # not the missing-secrets preflight that discards the body
        val = launch_spec._resolve_secret(name, run_id=st.get("run_id"),
                                          key_name=key_name,
                                          mint=bid is not None
                                          and not getattr(a, "dry_run", False))
        if val:
            env[name] = val
        elif name not in _soft:
            missing.append(name)
    if spec.get("runset") and "RUNSET" not in env:
        env["RUNSET"] = spec["runset"]
    if env:
        body["env"] = env
    if spec.get("onstart"):
        body["onstart"] = spec["onstart"]
    # SSH parity with _do_launch (2026-07-31). The spec records the PRE-inject
    # wire (cmd_train snapshots `wire`, and the injection happens later inside
    # _do_launch), and this builder PUTs its body straight to /asks/ — so every
    # supervisor-relaunched box and every handoff understudy was born WITHOUT an
    # sshd-acceptable authorized_keys. That is exactly how t211-vet-1's second
    # box (46449950) became un-debuggable: `herdd train` did pass ssh=True, but
    # the box that outlived the eviction never went through that code. Re-derive
    # from the LOCAL key here rather than storing it in the spec, so a rotated
    # workstation key lands on the replacement box.
    _wired = ssh.with_ssh_inject(body.get("onstart"))
    if _wired:
        body["onstart"] = _wired
    # image_login holds a token — never stored; re-derive from image + the
    # local signing secret. A spec that launched WITH a private pull but no
    # local secret now is a missing secret (refuse, don't 401 the pull on the
    # fresh box). One registry left since the 2026-08-22 GitLab cut, so the
    # name is unconditional: a spec naming any OTHER private host has no
    # credential path at all and must not silently relaunch unauthenticated.
    if spec.get("image_login"):
        login = launch_spec.image_login_arg(image, None)
        if login:
            body["image_login"] = login
        else:
            missing.append(imageref.R2_SECRET_ENV)
    return body, missing


# How many replacement candidates the eviction rung walks. 16 because the walk
# is FREE — one bundles request either way, and the API sorts ascending, so a
# deeper list only adds rows we then filter locally. The number is a bound on
# local work, not on spend.
# moved-from: herdd.REPLACEMENT_CANDIDATES
REPLACEMENT_CANDIDATES = 16


# moved-from: herdd._job_understudy_offer
def _job_understudy_offer(jctx: MutableMapping[str, Any],
                          hf: Mapping[str, Any] | None = None,
                          exclude_machines: Iterable[object] | None = None,
                          ) -> dict[str, Any] | None:
    """Pick the cheapest qualifying bid offer for a jobs-lane understudy, sized to
    the primary box's GPU shape (design §2.3: 'any GPU/geo that FITS the job').
    Argparse-free via pick_cheapest_offer. None on no match / no market read.
    `exclude_machines` (pull watchdog, 2026-08-02): never re-pick a machine that
    just failed the pull — the whole point of the reschedule is a DIFFERENT host."""
    primary = models._job_primary_shape(jctx, hf)
    gpu = ((primary or {}).get("gpu_name") or "").strip()
    ngpu = (primary or {}).get("num_gpus") or 1
    try:
        return market_offers.pick_cheapest_offer(gpu=(gpu,) if gpu else (),
                                                 num_gpus=ngpu, rental="bid", verified=True,
                                                 cc_allow=_replacement_cc_allow(jctx),
                                                 exclude_machines=exclude_machines or None)
    except Exception:
        return None


# moved-from: herdd._job_replacement_offer
def _job_replacement_offer(jctx: MutableMapping[str, Any],
                           exclude_machines: Iterable[object] | None = None,
                           rental: str = "bid",
                           max_dph: float | None = None,
                           cuda: float | None = None,
                           disk_gb: float | None = None,
                           ) -> dict[str, Any] | None:
    """Pick the cheapest qualifying offer for a REPLACEMENT jobs box — the
    forced-rehost lane (pull/SLA condemn, and the eviction replacement added
    2026-08-05), as opposed to `_job_understudy_offer`'s economic migration.

    Three differences from the understudy picker, all owner-directed 2026-08-05:

      * `rental` — the replacement may be rented ON-DEMAND when spot is
        structurally unsafe (`bidpolicy.replacement_decision`), so the market is
        a parameter, not a constant;
      * `cuda` — a forced rehost re-launches the PRIMARY'S OWN IMAGE, so the
        host driver floor must be enforced at pick time or the image lands on a
        host whose driver is older than its CUDA runtime and boots into
        Error-804;
      * `max_dph` — the derived price ceiling is pushed INTO the query, so an
        unaffordable market returns no offer instead of an offer we then refuse.

    Geography is deliberately unconstrained (same directive): the offer must be
    fast, not near. `pick_cheapest_offer` applies the
    `inet_down >= LAUNCH_INET_DOWN_MBPS` (1000 Mb/s) floor by default and falls
    back to an unfloored pass only if that empties the market — a slow host
    under the boot SLA still beats no host, and the SLA/pull watchdog rehosts it.
    None on no match / any API error.

    Since 2026-08-16 this is the `limit=1` face of `_job_replacement_offers`:
    the hardware filter is the MINIMUM-REQUIREMENTS candidate class, not the
    primary's exact vast gpu_name — and that class carries the CONTAINER-DISK
    floor (`disk_gb`, defaulted from `_replacement_disk_need`), so a p_alt read
    prices a box this workload could actually run on. Callers that want to
    CHOOSE (the eviction rung) walk the list; callers that only want a price or
    a fallback (the p_alt poll, `_launch_job_replacement`'s internal re-pick)
    keep this."""
    offers = _job_replacement_offers(jctx, exclude_machines, rental=rental,
                                     max_dph=max_dph, cuda=cuda, limit=1,
                                     disk_gb=disk_gb)
    return offers[0] if offers else None


# moved-from: herdd._replacement_disk_need
def _replacement_disk_need(jctx: MutableMapping[str, Any],
                           primary: Mapping[str, Any] | None = None,
                           ) -> tuple[float, str, bool]:
    """`(gb, why, known)` — the CONTAINER DISK a replacement for this watch
    must have, the number the launch will pass as `--disk`.

    One function so the SEARCH and the LAUNCH cannot disagree. `disksize.
    replacement_disk_gb` already takes the max of the three things that know
    anything about the size — the watch's immutable `launch_disk_gb` anchor
    (what the WORKLOAD was launched at), the replaced box's own allocation, and
    its measured usage plus headroom — so this dominates "at least what the old
    box had" without needing a separate rule. `known=False` means the
    inheritance chain broke and the figure is `REPLACEMENT_FALLBACK_GB`; the
    launcher says so out loud, and the search still carries it, because the
    launch is going to ask for exactly that many GB either way.

    Authoritative BECAUSE it is what we will ask vast for: a replacement search
    that filters on anything else can hand back an offer the very next call
    cannot use."""
    primary = (primary if primary is not None
               else (models._job_primary_shape(jctx, None) or {}))
    alloc, used = models._disk_gb(primary)
    gb, why = disksize.replacement_disk_gb(  # type: ignore[no-untyped-call]
        launch_gb=jctx.get("launch_disk_gb"), allocated_gb=alloc, used_gb=used)
    if gb is None:
        return disksize.REPLACEMENT_FALLBACK_GB, why, False
    return gb, why, True


def _replacement_cc_allow(jctx: MutableMapping[str, Any],
                          primary: Mapping[str, Any] | None = None,
                          ) -> tuple[int, ...]:
    """The sm levels a replacement for this watch MAY land on, `()` when the
    workload never declared any.

    Same inheritance discipline as `_replacement_disk_need`: the watch's own
    anchor first (written once from the primary's `LAUNCH_CC_ALLOW` stamp and
    durable across a fleetd restart), then the primary's env directly, for a
    watch adopted before the anchor was ever seeded. Never re-derived from the
    card the box happens to be holding — that is what made the constraint
    disappear at the first hop.

    `()` is the whole pre-2026-08-18 behaviour, which is what a box launched
    without `--cc-allow` still gets."""
    allow = market_offers.parse_cc_allow(jctx.get("launch_cc_allow"))
    if allow:
        return allow
    primary = (primary if primary is not None
               else (models._job_primary_inst(jctx) or {}))
    return market_offers.parse_cc_allow(
        models._instance_env(dict(primary or {})).get(
            market_offers.LAUNCH_CC_ALLOW_ENV))


# moved-from: herdd._job_replacement_offers
def _job_replacement_offers(jctx: MutableMapping[str, Any],
                            exclude_machines: Iterable[object] | None = None,
                            rental: str = "bid",
                            max_dph: float | None = None,
                            cuda: float | None = None,
                            limit: int = REPLACEMENT_CANDIDATES,
                            disk_gb: float | None = None,
                            ) -> list[dict[str, Any]]:
    """The replacement CANDIDATE SET, cheapest-first: every offer that meets the
    primary's minimum requirements (`_replacement_candidate_class` + num_gpus +
    the container-disk floor + the caller's cuda/ceiling, plus `pick_offers`'s
    inet floor), minus the excluded machines. [] on no match / any API error.

    This replaced a `gpu=(exact primary SKU,)` + limit-1 query on 2026-08-16.
    Both halves of that were the same bug: the rung's per-offer safety rail
    (`bidpolicy.bid_decision`) is a statement ABOUT A MACHINE, so evaluating it
    against a single pinned-SKU offer turns a selector into a veto. See
    `docs/plans/vast-fleet-sessions/2026-08-16-replacement-probe-tpd/SESSION.md`.

    The class also carries the workload's sm ALLOWLIST when its launch declared
    one (`_replacement_cc_allow`) — the axis this rung shipped without, and the
    reason an evicted A100 came back as an sm_120 RTX PRO 6000 on 2026-08-18
    with no kernel image for the job's attention path.

    `disk_gb` is the container-disk floor and defaults to
    `_replacement_disk_need` — the SAME number `_launch_job_replacement` hands
    vast as `--disk`. It was the axis the minimum-requirements class shipped
    without: VRAM, GPU count, cuda and inet were all carried and disk was not,
    so the rung ranked a 23 GB machine cheapest and rented it for a 50 GB
    workload (2026-08-16, boxes 47845159 / 47845212). Pass `disk_gb=0` to
    search WITHOUT the floor — the one legitimate use is asking whether the
    floor is what emptied the market (`_replacement_disk_shortfall`), never
    picking a box to rent."""
    primary = models._job_primary_shape(jctx, None)
    ngpu = (primary or {}).get("num_gpus") or 1
    names, ram_gb = market_offers._replacement_candidate_class(primary)
    need = (_replacement_disk_need(jctx, primary or {})[0] if disk_gb is None
            else disk_gb)
    try:
        return market_offers.pick_offers(
            gpu=names, gpu_ram_gb=ram_gb, disk_gb=need, num_gpus=ngpu,
            rental=rental,
            verified=_job_replacement_verified(jctx), max_dph=max_dph,
            cuda=cuda, cc_allow=_replacement_cc_allow(jctx, primary or {}),
            exclude_machines=exclude_machines or None, limit=limit)
    except Exception:
        return []


# moved-from: herdd._job_replacement_verified
def _job_replacement_verified(jctx: MutableMapping[str, Any]) -> bool:
    """Whether a replacement rental is restricted to VERIFIED hosts. Default
    True (unchanged), knob-able for a market where the verified book is empty:
    namespace `--replacement-verified` > env `JOB_REPLACEMENT_VERIFIED` >
    herdd.yaml `JOB_REPLACEMENT_VERIFIED` > True, the `_rebid_knob`
    precedence. Parsed as a BOOLEAN rather than through `_rebid_knob` itself,
    which coerces with `type(default)(v)` — `bool("0")` is True, so a bool knob
    routed through it would silently ignore every disable."""
    v = getattr(jctx.get("a"), "replacement_verified", None)
    if v is None:
        v = os.environ.get("JOB_REPLACEMENT_VERIFIED")
    if v in (None, ""):
        try:
            v = config.load_herdd_config().get("JOB_REPLACEMENT_VERIFIED")
        except Exception:
            v = None
    if v in (None, ""):
        return True
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() not in ("0", "false", "no", "off")


# moved-from: herdd.P_ALT_POLL_S
# moved-from: herdd.P_ALT_MAX_AGE_S
P_ALT_POLL_S = 600      # replacement-market read cadence (1-3 POST v0/bundles/
                        # per refresh — a tenth of the per-tick floor read's rate)
P_ALT_MAX_AGE_S = 1800  # a cached p_alt older than this is not a market read,
                        # it is a memory; the defense falls back to the
                        # pre-2026-08-09 ladder rather than price against it


# moved-from: herdd._job_palt_poll
def _job_palt_poll(jc: MutableMapping[str, Any], now: float,
                   own_machine: object = None) -> None:
    """Pre-eviction REPLACEMENT-market read (AUTOBID_DESIGN "Next iteration"
    §2): what would the best qualifying comparable box cost right now?

    This is the same `pick_cheapest_offer` query the replacement rung runs at
    eviction time, moved BEFORE the eviction so the one-shot job-aware defense
    (`bidpolicy.defense_ceiling`) has a price to reason against while the box
    is still ours. Two deliberate differences from the eviction-time read:

      * UN-ceilinged (`max_dph=None`) — the defense wants the true market
        price of moving, not a ceiling-filtered one;
      * ALWAYS excludes our own machine (plus the watch's evicted / pull-bad
        sets) — a p_alt read off the machine we are defending would be the
        self-referential floor (#73) wearing a different hat.

    Cached on `jc["p_alt"] / ["p_alt_ts"] / ["p_alt_machine"]`, refreshed at
    most every P_ALT_POLL_S; failures leave the previous read in place (the
    freshness gate in `_job_palt_fresh` retires it on its own). Mutates jc,
    returns nothing."""
    ts = jc.get("p_alt_ts")
    if ts is not None and now - float(ts) < _rebid_knob(
            jc, "palt_poll_s", P_ALT_POLL_S):
        return
    excl = _job_excluded_machines(jc, now)      # TTL'd by class (2026-08-16)
    if own_machine:
        excl.add(own_machine)
    offer = _job_replacement_offer(jc, sorted(excl), rental="bid",
                                   cuda=_replacement_cuda_floor())
    pa = models._num_dph((offer or {}).get("min_bid"))
    if pa is not None and pa > 0:
        jc["p_alt"] = pa
        jc["p_alt_ts"] = now
        jc["p_alt_machine"] = (offer or {}).get("machine_id")
    elif offer is None and jc.get("p_alt_ts") is None:
        # First read failed outright: stamp the attempt so a dead offers API
        # is retried on the poll cadence, not every tick.
        jc["p_alt_ts"] = now


# moved-from: herdd._job_palt_fresh
def _job_palt_fresh(jc: MutableMapping[str, Any], now: float) -> float | None:
    """The cached replacement-market price, ONLY while fresh enough to defend
    against (<= P_ALT_MAX_AGE_S old). None otherwise — a stale p_alt reverts
    the re-bid ladder to its pre-defense shape rather than pricing a live
    money move off a dead market read."""
    pa, ts = models._num_dph(jc.get("p_alt")), jc.get("p_alt_ts")
    if pa is None or pa <= 0 or ts is None:
        return None
    if now - float(ts) > P_ALT_MAX_AGE_S:
        return None
    return pa


# Launch-env keys an automatically re-rented box MUST inherit from the box it
# replaces. Deliberately a tiny allowlist, not a copy of the primary's env: most
# of what lives there is per-instance (nonces, digests, handoff epochs) and
# re-using it on a different box is wrong or unsafe.
#
# EVAL_ENV_VER earns its place because the BOX launch env is the only thing that
# can steer it. jobd's `check_venv eval` -> `onstart/fetch_eval_env.sh` resolves
# it and otherwise falls back to `rclone cat eval-env/LATEST`, and that fetch
# happens BEFORE the job's own `.job.env` is sourced — so a bundle-level pin can
# be compared on-box but cannot decide what was unpacked. `launch_jobs_box.sh`
# injects it for the box IT rents and calls that "fail-closed at both ends";
# every automatic re-rent below that lane (eviction replacement, handoff
# understudy) used to hand `_do_launch` an empty env and quietly reopen the hole.
#
# MEASURED 2026-08-16. Box 47887414 was rented with the pin, outbid before it
# ran, and fleetd's replacement 47889345 came up without it — so it provisioned
# eval-env/LATEST (20260807-0503-84d35a08) while the job pinned
# 20260816-1813-3c0a5f5b. Both queued E3 legs died rc 6 on env_identity's
# content gate, ~$0.50 of churn for zero rows. The gate did its job; nothing
# mis-graded. This makes the gate stop being the only thing standing there.
# moved-from: herdd.INHERITED_LAUNCH_ENV_KEYS
INHERITED_LAUNCH_ENV_KEYS: tuple[str, ...] = ("EVAL_ENV_VER",)


def launch_env_pin_from(inst: Mapping[str, Any] | None) -> dict[str, Any]:
    """The allowlisted launch env to anchor on the WATCH (`launch_env_pin`).

    ALLOWLISTED, never a copy: `extra_env` carries the box's B2 keys and HF
    token, and this dict is persisted to state.json. Needed because an evicted
    primary is absent from the tick snapshot exactly when a replacement is due,
    so the inheritance has nothing to read — measured twice, 2026-08-16 and
    again 2026-08-17."""
    return {k: v for k, v in (models._instance_env(dict(inst or {})) or {}).items()
            if k in INHERITED_LAUNCH_ENV_KEYS and v}


# moved-from: herdd._inherited_launch_env
def _inherited_launch_env(primary: Mapping[str, Any] | None,
                          extra: Sequence[str] = (), *,
                          pin: Mapping[str, Any] | None = None) -> list[str]:
    """`--env K=V` strings a re-rented box inherits from `primary`, plus `extra`.

    Same doctrine as the image pin and the disk sizing in these lanes: inherit
    the primary's shape, never re-derive it. Absent on both the primary and the
    watch anchor means absent here — this never invents a value, because a WRONG
    pin is worse than the unpinned fallback (unpinned dies on the content gate;
    wrong-but-plausible is what the gate exists to catch).

    `pin` is the watch's `launch_env_pin` anchor. It is not a second guess: an
    EVICTED primary is gone from the tick snapshot, so `primary` is {} exactly
    when a replacement is needed, and the anchor is that same box's own launch
    value recorded while it was still visible."""
    env = list(extra)
    have = {s.split("=", 1)[0] for s in env}
    pe = models._instance_env(dict(primary or {})) or {}
    for k in INHERITED_LAUNCH_ENV_KEYS:
        v = pe.get(k) or (pin or {}).get(k)
        if v and k not in have:
            env.append(f"{k}={v}")
    return env


# moved-from: herdd._launch_job_understudy
def _launch_job_understudy(jctx: MutableMapping[str, Any],
                           hf: MutableMapping[str, Any],
                           epoch: int,
                           ) -> tuple[Any, Any, str | None]:  # noqa: ANN401 — vast ids
    """Pre-warm the understudy (design §2.2 step 1): a `launch --jobs` bid box
    labelled job:<primary>:handoff, carrying HANDOFF_EPOCH/HANDOFF_TTL_S so jobd's
    epoch guard + dead-man honor the migration. Its B2 key is minted nonce-named by
    the --jobs launch path (_ship_b2_env) — no cross-revoke of the primary's key.
    jobd boots idle, prewarms jobs/<JOB_ID>/results/, waits for the retargeted
    ticket. Integration seam (a real launch) — the driver tests monkeypatch it.

    Returns `(iid, dph, reason)`: `reason` is None on success, else the ABORT
    reason the caller maps 1:1 (mirroring the run lane's :4729/:4735):
      * 'no_offer' — no qualifying offer, OR the picked offer no longer clears the
        §2.3 candidate filter against its live numbers (the belt-and-suspenders
        re-check: the market can move between the ARM decision and this launch);
      * 'understudy_unlaunchable' — an offer was picked but the launch call failed."""
    offer = hf.get("chosen_offer") or _job_understudy_offer(jctx, hf)
    if offer is None:
        return None, None, "no_offer"
    # belt-and-suspenders (mirror the run lane's _handoff_understudy_body): re-check
    # the §2.3 candidate filter against the ACTUALLY-picked offer's live numbers
    # before spending. A now-unviable offer is no qualifying candidate -> no_offer.
    # The candidate's on-demand rate is a live market read, never the bid row's
    # own dph_total (doc 50 R1 — that field is the interruptible price, and
    # clamping to it is what bid understudy 46934673 a razor-thin $0.401 on
    # 2026-08-06; see _offer_ondemand_ref).
    offer_od = pricing._offer_ondemand_ref(offer)
    if not bidpolicy._handoff_candidate_ok(  # type: ignore[no-untyped-call]
            models._num_dph(jctx.get("last_bid")) or models._num_dph(jctx.get("dph")),
            models._num_dph(offer.get("min_bid")), offer_od,
            jctx.get("remaining_wall_h", 0.0), jctx.get("on_demand")):
        jctx["last_error"] = "understudy offer no longer clears the §2.3 filter"
        return None, None, "no_offer"
    # Price the understudy HERE from the offer dict already in hand (D8, live
    # jobs-lane canary 2026-07-15): leaving price=None makes _do_launch re-price
    # via _offer_pricing_soft, whose v0/bundles id-filter is structurally dead
    # (returns 0 rows for a live offer id — memory vast-bundles-id-filter-dead),
    # so the auto-price failed and every jobs-lane understudy came back
    # `understudy_unlaunchable`. The run lane never hit this: _handoff_understudy_body
    # prices from the same offer dict. None => unwinnable floor (D7) => no_offer.
    price = pricing._auto_bid_price(models._num_dph(offer.get("min_bid")), offer_od)
    if price is None:
        jctx["last_error"] = "understudy offer has no winnable bid price"
        return None, None, "no_offer"
    primary = models._job_primary_shape(jctx, hf)                # P5: ARM snapshot fallback
    # ROLLING RELEASE (owner ruling 2026-08-04): the understudy adopts the
    # NEWEST image, not the primary's. velvet P3/P4 pinned the primary's launch
    # digest here so a handoff reproduced the migrating job's environment; that
    # is the digest replay the ruling retires. What survives is the NOTICE — the
    # primary's stamp is still read, and when the tag has moved the operator is
    # told the job just changed envs mid-flight (imageref.pin_relaunch_image).
    u_image = (primary or {}).get("image_uuid") or None
    _p_stamp = models._instance_env(primary or {}).get(imageref.IMAGE_DIGEST_ENV)
    if u_image and _p_stamp:
        u_image, _u_state, _u_why = imageref.pin_relaunch_image(
            image=u_image, spec_digest=_p_stamp,
            current_digest=imageref.image_tag_digest(u_image))
        if _u_state in (imageref.PIN_ROLLED, imageref.PIN_UNVERIFIED):
            print(f"    understudy image: {_u_why}", file=sys.stderr)
    # ...and size it from what the primary is USING, not from what it was
    # allocated. Copying `disk_space` was the only "derived" size in the tree and
    # it derived the wrong thing: a primary at 160G holding 17G minted another
    # 160G box. This can only shrink, never grow, and only on a real
    # measurement — a booting box reports disk_usage -1 and keeps the copy.
    _alloc, _used = models._disk_gb(primary or {})
    u_disk, _d_why = disksize.understudy_disk_gb(  # type: ignore[no-untyped-call]
        allocated_gb=_alloc, used_gb=_used)
    if u_disk is None:
        u_disk = 120
    elif u_disk < (_alloc or 0):
        print(f"    understudy disk: {_d_why}", file=sys.stderr)
    _cc = _replacement_cc_allow(jctx, primary or {})
    ns = argparse.Namespace(
        offer=offer.get("id"), type="bid", price=price,
        num_gpus=(primary or {}).get("num_gpus") or 1,
        # the arch allowlist travels with the box, same as the disk request:
        # a migration that lands on silicon the workload cannot run on is the
        # same loss as an eviction that does.
        cc_allow=(",".join(str(s) for s in _cc) if _cc else None),
        env=_inherited_launch_env(                                         # HANDOFF_TTL_S
            primary, (f"HANDOFF_EPOCH={epoch}",
                      f"HANDOFF_TTL_S={bidpolicy.HANDOFF_TTL_S}"),
            # the ARM snapshot this lane falls back to carries _JOB_PRIMARY_SHAPE_KEYS
            # only — no extra_env — so it needs the anchor for the same reason.
            pin=jctx.get("launch_env_pin")),
        # is read by no box-side reader on this lane (jobd.sh has no dead-man of its
        # own); the REAL jobs-lane dead-man is jobd's no-job park, wired below via
        # no_job_deadline=HANDOFF_TTL_S — see the comment there.
        port=None, jupyter=False, onstart=None,
        # ssh=True: an understudy is a REPLACEMENT production box, and the jobs
        # lane's `job cancel --hard` / `job attach` both need a shell on it. It
        # opted out until 2026-07-31, which made every migrated jobs box
        # un-debuggable in exactly the way box 46449950 was.
        no_hf_token=False, hf_token=None, ssh=True, ssh_key_file=None,
        # F6: enforce the same TTL as the run-lane watchdog THROUGH jobd's existing
        # no-job park (JOBD_NO_JOB_PARK_S). The understudy boots idle waiting for the
        # retargeted ticket; if the supervisor dies mid-handoff it never arrives, so
        # jobd self-parks at HANDOFF_TTL_S rather than the box default (3600s).
        jobs=True, no_idle_park=False, idle_park_grace=None,
        no_job_deadline=bidpolicy.HANDOFF_TTL_S,
        disk=int(u_disk),
        runtype="ssh_direct",
        label=labels._job_handoff_label(jctx.get("iid")),
        force=True, wait=None, template_id=None,
        login=None, no_registry_login=False,
        image=u_image,
        dry_run=jctx.get("dry_run", False))
    try:
        cid, _oid, dph = launch._do_launch(ns)
    except SystemExit as e:
        jctx["last_error"] = f"understudy launch failed: {e}"
        return None, None, "understudy_unlaunchable"
    except Exception as e:                                # never kill the loop
        jctx["last_error"] = f"understudy launch error: {e}"
        return None, None, "understudy_unlaunchable"
    if cid is None:                                       # dry-run: synthesize an iid
        cid = f"dry-jh-{int(jctx.get('now', 0))}"
    _note_arch_change(jctx, primary, offer, cid, lane="understudy")
    return cid, dph, None


# --------------------------------------------------------------------------- #
# jobs-lane boot-pull watchdog (owner directive 2026-08-02)
#
# "fleetd should trigger based on a timeout, worst case. if the box doesn't
#  pull in 10 minutes, terminate and reschedule on another box. a job in the
#  queue should continue seamlessly. [...] if the host is only pulling docker
#  at some very slow rate (<3MB/s or something) [kill earlier]."
#
# The problem is HOST QUALITY, not box death: a host that cannot pull the
# image fast is a bad host — cut it loose early and go somewhere else. The
# pull phase is GPU-UNBILLED (invoice-verified, see _BOOT_LOADING_STATES), so
# terminating a slow puller costs only the wasted pull; aggression is nearly
# free, which is why this lane is fast and blunt where the graded reap ladder
# (billed env-setup / running-but-dead shapes) stays evidence-graded. This is
# where the boot-phase split earns its keep.
#
# Control-plane only (owner ruling, same date): every signal here is the vast
# API sampled longitudinally by the supervising daemon — status_msg layer
# bytes folded through BootThroughputSampler (AGGREGATE across layers, never
# per-flow; extract-only windows can't condemn via _BOOT_DL_MIN_FRAC). The
# condemned box is never consulted.
#
# Live evidence for the seamless-reschedule requirement: box 46590907 died
# 2026-08-02 holding two phase1-cot tickets, both still `submitted`, orphaned
# against an instance that no longer exists. The reschedule below moves every
# pending ticket (same JOB_ID, log continues) BEFORE the condemned box is
# destroyed, so that outcome cannot recur on this lane.
#
# Relaunch-loop guards: BOOT_MAX_HOST_RETRIES caps the reschedules per watch
# (then the watchdog disarms and alarms instead of re-renting); every failed
# machine_id is excluded from the replacement offer; the ladder's --budget
# rail keeps billing the replacement; every terminate+relaunch is journaled
# with the measured rate and reason.
#: The refusal reason for a launch the ACCOUNT could not make, as opposed to one
#: this market could not satisfy. Kept distinct from `unlaunchable` because the
#: two have disjoint remedies and the alarm text branches on it.
REASON_ACCOUNT_BLOCKED = acctfault.REASON


def _launch_failure_reason(err: object) -> str:
    """`account_blocked` when the API refused US, `unlaunchable` otherwise.

    Measured 2026-08-25: 76 consecutive `insufficient_credit` refusals all read
    `unlaunchable`, and every alarm derived from them prescribed a market move.
    """
    return (REASON_ACCOUNT_BLOCKED if acctfault.classify(err)
            else "unlaunchable")


# --------------------------------------------------------------------------- #
# moved-from: herdd._launch_job_replacement
def _launch_job_replacement(jctx: MutableMapping[str, Any],
                            exclude_machines: Iterable[object] | None, *,
                            offer: Mapping[str, Any] | None = None,
                            rental: str = "bid",
                            price: float | None = None,
                            max_dph: float | None = None,
                            ) -> tuple[Any, Any, str | None]:  # noqa: ANN401 — vast ids
    """Launch a REPLACEMENT jobs box for a condemned or EVICTED primary: same
    shape (GPU count/name via the offer pick, image pinned to the primary's
    launch digest, disk sized from usage — the hard-won bits mirror
    _launch_job_understudy, which stays canonical for the handoff lane), but:
      * the offer pick EXCLUDES the failed machines (the entire point),
      * no HANDOFF_* env and no §2.3 price filter — this is a forced
        replacement of a condemned box, not an economic migration,
      * the primary's own label is carried, so fleetd adoption/dup-guards see
        the replacement exactly as they saw the original.

    `offer` / `rental` / `price` (added 2026-08-05 for the eviction-replacement
    ladder): the caller may hand in an offer it already priced through
    `bidpolicy.replacement_decision`, and may ask for the ON-DEMAND market —
    the rung that exists because an on-demand claim outranks every bid, so a
    spot replacement after an on-demand displacement just buys the same loss
    again. Omit all three and the behavior is the pre-existing pull-condemn one:
    pick the cheapest bid offer and auto-price it.

    `max_dph` (doc 50 R3, 2026-08-05) is the SPEND RAIL, and it binds HERE, on
    whatever offer this function actually launches — handed in or re-picked. It
    did not: the internal re-pick below ran with no ceiling and no CUDA floor,
    and on 2026-08-05 it bought a $3.4741/hr on-demand box against the $2.164
    ceiling the decision record claimed to respect. A rail that binds only in
    the pure decision function is not a rail. None = no ceiling known (no launch
    price anchor), in which case the decision ladder has already refused and
    there is nothing here to enforce.

    Returns (iid, dph, reason) with reason None on success ('no_offer' /
    'over_ceiling' / 'account_blocked' / 'unlaunchable' otherwise).
    `account_blocked` is `unlaunchable` narrowed to the case no host can fix:
    the API refused US, so the ladder is not choosing badly and the alarm must
    not send the operator to the market. `dph` is always the REALIZED
    rate: the bid price on the spot rung, the offer's `dph_total` on the
    on-demand one. `_do_launch` reads a price off SEARCHED offers only, so a
    PINNED on-demand launch returned None and the journal recorded
    `ondemand @ $None/hr` while a real meter ran — an unauditable spend
    record (doc 50 R4)."""
    is_od = rental in ("ondemand", "on-demand")
    if offer is None:
        # Same filters the DECISION probed with: the ceiling and the host CUDA
        # floor. Omitting `cuda` here was a second latent bug on this line — a
        # forced rehost re-launches the primary's OWN image, and a host whose
        # driver is below its CUDA runtime boots into Error-804, the exact
        # failure `_replacement_cuda_floor` exists to prevent.
        offer = _job_replacement_offer(jctx, exclude_machines,
                                       rental="ondemand" if is_od else "bid",
                                       max_dph=max_dph,
                                       cuda=_replacement_cuda_floor())
    if offer is None:
        # NAME THE BOUND. Since the search carries the container-disk floor, a
        # market that looks empty may simply be a market with no box big enough
        # — on host 67231's A100 PCIe book (18/23/33/47 GB) that is the usual
        # reason — and "no qualifying offer" alone sends the operator hunting
        # for a price problem that isn't there.
        _need, _, _ = _replacement_disk_need(jctx)
        # ...AND ANSWER IT: naming the floor does not say whether the floor is
        # the bound. The probe re-searches UNFLOORED and returns None when disk
        # was not the reason, so a price-emptied market reads as it did before.
        # Wired here 2026-08-17 — the pull-condemn lane was the one refusal path
        # that had the floor but not the verdict.
        _bound = _replacement_disk_shortfall(
            jctx, exclude_machines, max_dph, _replacement_cuda_floor(), _need)
        jctx["last_error"] = (
            f"no qualifying replacement offer (after exclusions; the search "
            f"requires >= {_need:g}G of container disk)"
            + (f" — {_bound}" if _bound else ""))
        return None, None, "no_offer"
    offer_dph = models._num_dph(offer.get("dph_total"))
    if is_od:
        # On-demand pays the list price; a bid price would be meaningless (and
        # _do_launch ignores it for type=ondemand). Nothing to clamp.
        price = None
    elif price is None:
        # The on-demand clamp reference must come from the ON-DEMAND market
        # (doc 50 R1). This used to pass the BID offer's own `dph_total`, which
        # is the current interruptible price (~min_bid + 0.5%) — the clamp then
        # lands the bid a tenth of a cent over the floor, i.e. it MANUFACTURES
        # the razor-thin bid the whole cushion rail exists to prevent (that is
        # how understudy 46909754 was bid $1.071 over a $1.0667 floor and lost
        # the box 45 minutes later). `_market_ondemand_soft` reads `dph_base`
        # off the machine's own offers and is soft: None disables the clamp,
        # which is the correct behavior for an unknown price — no clamp beats a
        # wrong one, and the ceiling below still bounds the bid.
        _od = pricing._market_ondemand_soft(
            offer.get("machine_id"),
            (models._job_primary_shape(jctx, None) or {}).get("num_gpus"))
        price = pricing._auto_bid_price(models._num_dph(offer.get("min_bid")), _od)
    if not is_od and price is None:
        jctx["last_error"] = "replacement offer has no winnable bid price"
        return None, None, "no_offer"
    # REFUSE, never rent, over the ceiling. On-demand is checked against the
    # offer's own list price (what we will actually be billed); spot against the
    # bid we are about to place — the offer search filters on `min_bid`, and
    # 1.20x a just-under-ceiling floor lands over the line.
    _pay = offer_dph if is_od else price
    if max_dph is not None and _pay is not None and _pay > max_dph + 1e-9:
        jctx["last_error"] = (
            f"{'on-demand' if is_od else 'spot'} replacement offer "
            f"{offer.get('id')} would bill ${_pay:.4f}/hr, over the "
            f"${max_dph:.3f} replacement ceiling")
        print(f"!! replacement REFUSED (not rented): {jctx['last_error']}")
        # A REFUSAL IS A MARKET READ, and on this lane it is the only one: the
        # ceiling is pushed into the offer search, so an unaffordable market
        # reports as an EMPTY one and the price never reaches the ceiling. Record
        # what a qualifying box was actually going to cost, so the next tick's
        # `_job_replacement_ceiling` can re-derive against it instead of refusing
        # the same 3.4% gap forever (2026-08-24).
        jctx["replacement_market_floor"] = _pay
        jctx["replacement_market_floor_ts"] = jctx.get("now") or time.time()
        return None, None, "over_ceiling"
    primary = models._job_primary_shape(jctx, None) or {}
    u_image = primary.get("image_uuid") or None
    _p_stamp = models._instance_env(primary).get(imageref.IMAGE_DIGEST_ENV)
    if u_image and _p_stamp:
        u_image, _u_state, _u_why = imageref.pin_relaunch_image(
            image=u_image, spec_digest=_p_stamp,
            current_digest=imageref.image_tag_digest(u_image))
        if _u_state in (imageref.PIN_ROLLED, imageref.PIN_UNVERIFIED):
            print(f"    replacement image: {_u_why}", file=sys.stderr)
    # INHERIT THE SIZING, NEVER RE-INVENT IT (task #69, 2026-08-08). R7
    # (2026-08-05) floored this at the PRIMARY's allocation, which is right as
    # far as it goes and covers the box-46914272 shape (evicted 2 min into boot
    # holding 5.7 of 50 GB; `used x 1.4 + 12` sized its replacement at 20 GB).
    # What it cannot cover is a primary that is not there to read: on
    # EVICTION_HOST_FAILURE the box has left the listing, so `_disk_gb` returns
    # (None, None), `understudy_disk_gb` declares unknown, and the sizing
    # reverted to the hardcoded default below — SILENTLY. And because each hop
    # floored at the LAST box rather than at the launch, one under-sized hop
    # propagated for the rest of the chain: driftr3 went 110 -> 110 -> 60 GB and
    # died on its own disk guard with rc 5
    # (DRIFT_ROSTER_R3_H200_COHORT_2026-08-06.md §8).
    #
    # `launch_disk_gb` is the watch's immutable disk anchor (written once from
    # the first observation of the ORIGINAL box, exactly like
    # `launch_dph_anchor`, and durable across a fleetd restart), so the sizing
    # is now a property of the WORKLOAD and not of whichever box last held it.
    #
    # `_replacement_disk_need` is that max(), factored out on 2026-08-16 so the
    # OFFER SEARCH filters on the same number this launch asks for: sizing the
    # rental correctly is worth nothing if the search already handed us a
    # machine with 23 GB to give.
    u_disk, _d_why, _d_known = _replacement_disk_need(jctx, primary)
    if not _d_known:
        # FAIL SAFE, NEVER SILENT. Reaching here means the inheritance chain
        # broke (a watch adopted after its box vanished, or one that never saw
        # a priced tick), and the replacement may be smaller than the job needs.
        print(f"!! replacement disk: {_d_why} — falling back to the "
              f"{u_disk:g}G default. The replacement may be SMALLER than the "
              f"job was launched at; check `herdd ls` disk_gb against the "
              f"launch's --disk before trusting the rehost.", file=sys.stderr)
    else:
        print(f"    replacement disk: {u_disk:g}G — {_d_why}", file=sys.stderr)
    # Hand the size back to the caller's journal. §8's write-up could only INFER
    # which lane sized the 60 GB box, from its label and the observed disk,
    # because no decision event carried the number: "this is inference from the
    # labels and the observed sizes, not something read off a decision event".
    jctx["last_replacement_disk_gb"] = u_disk
    # CARRY THE ARCH STAMP FORWARD, or it dies at the first hop. `_do_launch`
    # stamps `--cc-allow` into the new box's env, which is the only channel the
    # NEXT replacement can read it from — the same reasoning that makes the disk
    # sizing an anchor rather than a re-derivation.
    _cc = _replacement_cc_allow(jctx, primary)
    prim_inst = models._job_primary_inst(jctx) or {}
    ns = argparse.Namespace(
        cc_allow=(",".join(str(s) for s in _cc) if _cc else None),
        offer=offer.get("id"), type="ondemand" if is_od else "bid", price=price,
        num_gpus=primary.get("num_gpus") or 1,
        # pin=: this lane reads `primary` via _job_primary_shape(jctx, None), so an
        # EVICTED box yields {} and the env inheritance had nothing to inherit.
        env=_inherited_launch_env(primary, pin=jctx.get("launch_env_pin")),
        port=None, jupyter=False, onstart=None,
        no_hf_token=False, hf_token=None, ssh=True, ssh_key_file=None,
        jobs=True, no_idle_park=False, idle_park_grace=None,
        no_job_deadline=None,
        disk=int(u_disk),
        runtype="ssh_direct",
        label=prim_inst.get("label") or None,
        force=True, wait=None, template_id=None,
        login=None, no_registry_login=False,
        image=u_image,
        dry_run=jctx.get("dry_run", False))
    try:
        cid, _oid, dph = launch._do_launch(ns)
    except SystemExit as e:
        jctx["last_error"] = f"replacement launch failed: {e}"
        return None, None, _launch_failure_reason(e)
    except Exception as e:                                # never kill the loop
        jctx["last_error"] = f"replacement launch error: {e}"
        return None, None, _launch_failure_reason(e)
    if cid is None:                                       # dry-run: synthesize
        cid = f"dry-jr-{int(jctx.get('now', 0))}"
    # R4: never hand the caller a price of None while a meter runs. _do_launch
    # only reads `dph_total` off a SEARCHED offer, and this lane always PINS
    # one — so on the on-demand rung it came back None and the journal read
    # `ondemand @ $None/hr` for a box billing $3.4741/hr. The offer row we
    # launched from is that price.
    dph = models._num_dph(dph)
    if dph is None:
        dph = price if not is_od else offer_dph
    _note_arch_change(jctx, primary, offer, cid, lane="replacement")
    return cid, dph, None


def _note_arch_change(jctx: MutableMapping[str, Any],
                      primary: Mapping[str, Any] | None,
                      offer: Mapping[str, Any] | None,
                      new_iid: object, *, lane: str) -> bool:
    """Say it out loud when an automatic swap crossed an ARCHITECTURE boundary.
    Returns whether it did. Never blocks — refusing is `--cc-allow`'s job, and a
    swap that has already happened cannot be un-rented by an alarm.

    Loud because the two incidents this family comes from were both invisible at
    the moment they mattered: fleetd replaced an evicted card with a different
    generation, and the only thing that noticed was a workload dying on it
    (2026-08-17) or a bundle gate after the fact (2026-08-18). It is also a
    MEASUREMENT boundary — throughput, VRAM headroom and kernel availability all
    move with the silicon, so a run whose second half is on a different arch is
    not one series."""
    if not market_offers.arch_changed(primary, offer):
        return False
    old, new = market_offers.arch_label(primary), market_offers.arch_label(offer)
    print(f"!! {lane} ARCHITECTURE CHANGE: {old} -> {new} (box {new_iid}). "
          f"Measurements across this boundary are NOT comparable, and kernels "
          f"built for the old arch may have no image for the new one. Pin the "
          f"class with `--cc-allow` on the launch if this workload cannot move.",
          file=sys.stderr)
    journal._job_ladder_journal(jctx, "arch_change", iid=str(new_iid), lane=lane,
                                old_arch=old, new_arch=new,
                                note="measurements across this boundary are "
                                     "not comparable")
    journal._job_handoff_emit(jctx, "arch_change", lane=lane, box=str(new_iid),
                              old_arch=old, new_arch=new)
    return True


# moved-from: herdd._retarget_pending_tickets
def _retarget_pending_tickets(old_iid: object, new_iid: object,
                              reason: str = "pull_condemned",
                              ) -> tuple[list[Any], list[Any]]:  # noqa: ANN401 — job ids
    """Move the still-RUNNABLE queued tickets from `old_iid`'s queue to
    `new_iid`'s — the load-bearing 'a job in the queue should continue
    seamlessly' step, done BEFORE the condemned box is destroyed so no
    46590907-style orphan can exist even transiently. Same JOB_ID, event log
    continues (`retargeted` event). Returns (moved, failed) job-id lists;
    failures are reported, never raised (best-effort per ticket, and a failed
    move leaves the old ticket in place for a manual `job retarget`).

    "every ticket" until 2026-08-26, and that was the bug. `list_queue` returns
    every ticket EVER submitted to the box — jobd deletes none — so this moved
    finished jobs and forgotten week-old ones onto the successor alongside the
    live work. Terminal ones were then skipped box-side by the
    `results.DONE.json` probe, a guard this function cannot see and that a job
    which failed before its entrypoint ran never wrote; stale ones had no guard
    at all. `jobmeta.bulk_move_verdict` now filters both here, at the source.
    Skips are printed, never silent, and the skipped ticket is left in the old
    queue so it surfaces via `job orphans` rather than vanishing.

    Why the bulk move may ignore `cmd_job_retarget`'s running-job refusal — the
    refusal exists so a ticket is never claimed by TWO live jobds, and both
    callers here have positive evidence that cannot happen:

      * `reason="pull_condemned"` — a box condemned in `loading` never booted
        jobd at all, so nothing can be running;
      * `reason="evicted"` (2026-08-05) — the box is STOPPED or has left the
        instance listing entirely; a ticket whose `display_status` still reads
        `running` is recording the state jobd last synced before the machine was
        taken, not a live process. The caller destroys the old box after this
        returns, and jobd's own resume path picks the work up from the last
        synced checkpoint on the new one.

    A caller without that evidence must use `cmd_job_retarget`, which refuses."""
    moved: list[Any]
    failed: list[Any]
    moved, failed = [], []
    skipped: list[tuple[str, str]] = []
    try:
        jids = jobmeta.list_queue(str(old_iid))
    except Exception as e:
        print(f"!! retarget ({reason}): queue listing failed ({e}) — retarget "
              f"pending tickets by hand: herdd job retarget <JOB_ID> "
              f"--from {old_iid} --box {new_iid}")
        return moved, ["<queue-unreadable>"]
    statuses = _bulk_move_statuses(jids)
    for jid in jids:
        try:
            ticket = jobmeta.read_ticket(str(old_iid), jid)  # type: ignore[no-untyped-call]
            if ticket is None:
                continue
            move, why = jobmeta.bulk_move_verdict(ticket, statuses.get(jid))  # type: ignore[no-untyped-call]
            if not move:
                skipped.append((jid, why))
                print(f"   retarget ({reason}): SKIP {jid} — {why}")
                continue
            ticket = dict(ticket)
            ticket["box"] = str(new_iid)
            ticket["retargeted_from"] = str(old_iid)
            ok, _key, err = jobmeta.write_ticket(ticket)  # type: ignore[no-untyped-call]
            if not ok:
                raise RuntimeError(f"ticket write failed: {err}")
            ok, err = jobmeta.delete_ticket(str(old_iid), jid)  # type: ignore[no-untyped-call]
            if not ok:
                print(f"!! retarget ({reason}): old ticket delete failed for {jid} "
                      f"({err}) — box {old_iid} is being destroyed, so no "
                      f"double-run; delete jobs/queue/{old_iid}/{jid}.json to "
                      f"tidy")
            jobmeta.emit_event(jid, "retargeted", box=str(new_iid),
                               from_box=str(old_iid), reason=reason)
            moved.append(jid)
        except Exception as e:
            print(f"!! retarget ({reason}): could not move {jid}: {e} — move it "
                  f"by hand: herdd job retarget {jid} --from {old_iid} "
                  f"--box {new_iid}")
            failed.append(jid)
    if skipped:
        # NEVER silent: a bulk move that quietly dropped tickets would read as
        # a clean recovery. A skipped ticket stays in the old box's queue, so
        # once that box is destroyed it surfaces in `herdd job orphans` —
        # that is the reviewable end state, and the operator decides between
        # `job retarget --stale-ok` and `job dlq add`.
        print(f"   retarget ({reason}): {len(skipped)} ticket(s) NOT moved — "
              f"they stay in box {old_iid}'s queue and will show up as orphans "
              f"once it is gone (herdd job orphans)")
    return moved, failed


def _bulk_move_statuses(jids: Sequence[str]) -> dict[str, str]:
    """Folded status per job for the bulk-move filter, best effort.

    Cost is 1-2 rclone subprocesses for the whole set (`scan.fold_many`), not
    one per job. A fold that fails as a whole returns {} — every ticket then
    reads `unknown` and MOVES, which is the fail-open direction: an unreadable
    event log must never silently abandon live work on an evicted box."""
    try:
        from vastlib.jobs import scan
        views = scan.fold_many(list(jids))
    except Exception as e:      # noqa: BLE001 — best effort by contract
        print(f"   retarget: status fold unavailable ({e}) — moving every "
              f"ticket (fail-open)")
        return {}
    return {jid: str(v.get("status") or "unknown") for jid, v in views.items()}


# moved-from: herdd._job_pull_watchdog_tick
def _job_pull_watchdog_tick(jc: MutableMapping[str, Any],
                            inst: Mapping[str, Any], now: float) -> str | None:
    """Feed this tick's instance record to the per-watch pull sampler while the
    box is pre-`running`. Returns 'slow' | 'deadline' when the host stands
    condemned, else None. The sampler clock for the fixed deadline is the
    box's own start_date (billing/schedule truth, survives supervisor
    restarts); the throughput clock starts at first Downloading evidence and
    measures the AGGREGATE byte rate across layers over a FULL window with the
    >=50%-downloading vote (extract-heavy windows are progress, not
    starvation)."""
    if jc.get("pull_watchdog_disabled"):
        return None
    s = jc.get("pull_sampler")
    if s is None or jc.get("pull_sampler_iid") != jc.get("iid"):
        try:
            start = float(inst.get("start_date"))  # type: ignore[arg-type]  # guarded
        except (TypeError, ValueError):
            start = now
        s = health.BootThroughputSampler(
            min_mbps=config._boot_knob("BOOT_MIN_MBPS"),
            window_s=config._boot_knob("BOOT_MBPS_WINDOW_S", cast=int),
            deadline_s=config._boot_knob("BOOT_PULL_TIMEOUT_S"),
            start_t=start)
        jc["pull_sampler"], jc["pull_sampler_iid"] = s, jc.get("iid")
    v = s.feed(inst, now)
    return v if v in ("slow", "deadline") else None


# moved-from: herdd._job_boot_sla_tick
def _job_boot_sla_tick(jc: MutableMapping[str, Any],
                       inst: Mapping[str, Any], now: float) -> str | None:
    """Env-setup half of the jobs-lane boot SLA (owner directive 2026-08-03:
    "longer than 10 minutes to come online is unacceptable"). The pull watchdog
    above covers `loading`; this covers the RUNNING box whose jobd never came
    up. Milestone = jobd's first JOBD_STATUS stamp THAT IS NOT CONFESSING (the
    affirmative boot proof, read control-plane-side from B2 — never an on-box
    probe). Armed ONLY for a box THIS watch observed pre-running: a
    resumed/adopted box carries a STALE marker from its previous session that
    would fake the milestone, and an
    adopted mid-life box is not a fresh boot to hold to the SLA. Clock =
    start_date (billing truth, survives supervisor restarts); the deadline
    widens per _boot_deadline_backoff after repeated kills. Returns 'sla' when
    breached (the caller routes it into the SAME condemn/reschedule path as a
    pull verdict), else None. GPU bills in this phase — one more reason not to
    wait out a dead env-setup."""
    if jc.get("pull_watchdog_disabled"):
        return None
    sla = config._boot_knob("BOOT_SLA_S")
    if sla <= 0:
        return None
    iid = jc.get("iid")
    if jc.get("boot_online_iid") == iid:
        return None                      # milestone already reached
    if jc.get("boot_loading_iid") != iid:
        return None                      # never saw this box boot: not armed
    # THE MILESTONE IS NOT MERELY "A STAMP EXISTS" (amended 2026-08-14,
    # FAILCLOSED_DESIGN §1). Box 47737955 stamped JOBD_STATUS at T+9m26s, inside
    # this deadline, and was functionally dead the whole time: `jobd.py` could
    # not import its own modules, so the daemon claimed nothing and emitted
    # nothing while it billed $1.742 over 52 minutes. The bash half wrote the
    # marker, the python half was gone, and THE SLA MEASURED THE HALF THAT
    # WORKED. So a stamp counts as coming online only if the box is not, on that
    # same line, telling us it cannot run our code.
    line = health._jobd_status_line_soft(iid)
    if line is not None:
        pyhalf = health.jobd_status_pyhalf(line)
        if pyhalf is not True:
            # ok, or a bundle older than the field, or an unrecognised value.
            # BACK-COMPAT IS THE WHOLE POINT OF `is not True`: absence is not a
            # confession, and a strict `== ok` here would hold every box on an
            # older bundle to a milestone it can never signal and then destroy
            # and relaunch it at the deadline — fleet-wide, on the day this
            # ships.
            jc["boot_online_iid"] = iid  # jobd stamped, not confessing: online
            # The one place in this codebase that knows a host DID boot our
            # image and run our code. Recorded durably (2026-08-20) so the
            # reputation score has positive evidence too: without it the store
            # would only ever accumulate, and a machine we use daily and that
            # failed once last month would look identical to one we tried once
            # and that failed. Gated on the same milestone as the SLA — a stamp
            # from a confessing box below is explicitly NOT a success.
            hostrep.note_ok(inst.get("machine_id"), iid=iid, now=now)
            return None
        # Confessed broken. The milestone is NOT met — recording this boot as a
        # success is exactly the lie the incident turned on — but this must not
        # CONDEMN either, and that is not timidity. The SLA's remedy is destroy
        # + exclude the machine + relaunch elsewhere, which is right for a host
        # fault and actively harmful here: `pyhalf=broken` is a shipped-bundle
        # fault, host-independent by construction, so every replacement
        # reproduces it and the ladder burns BOOT_MAX_HOST_RETRIES boxes (and
        # blames that many innocent machines) proving it.
        #
        # The remedy belongs to the two mechanisms that already own this shape,
        # both of which PARK: jobd self-parks at JOBD_PY_BROKEN_PARK_S (300 s),
        # and fleetd's `_pyhalf_tick` parks at FLEETD_PYHALF_CONFIRM_S (600 s).
        # What the SLA owes is to stay armed, say so once, and get out of the
        # way — so that if the box does recover to `pyhalf=ok` the milestone
        # latches then, on evidence, rather than having been granted at the
        # first byte written.
        if not jc.get("boot_sla_pyhalf_said"):
            jc["boot_sla_pyhalf_said"] = True
            print(f"!! BOOT-SLA HELD {iid}: jobd stamped JOBD_STATUS but the "
                  f"line says pyhalf=broken — the box cannot run jobd.py, so "
                  f"the come-online milestone is NOT met. NOT condemning: this "
                  f"is a bundle fault and a relaunch reproduces it. The box "
                  f"self-parks at 300s and fleetd parks it at 600s; fix the "
                  f"bundle (`herdd shipcheck`) and re-ship")
            journal._job_ladder_journal(
                jc, "boot_sla_held_pyhalf_broken", iid=str(iid),
                note="jobd stamped but confessed pyhalf=broken: come-online "
                     "milestone withheld, condemn deliberately suppressed "
                     "(bundle fault, not a host fault — the park paths own it)")
        return None
    # CLOCK FROM THE RUNNING TRANSITION, NOT FROM BOX CREATION. Both this
    # deadline and the pull watchdog's are BOOT_SLA_S/BOOT_PULL_TIMEOUT_S = 600s,
    # and anchoring both at start_date makes them share ONE budget instead of
    # granting one each: a box whose pull legally takes 9 minutes then has 60
    # seconds to bootstrap jobd before this fires. That is not hypothetical —
    # box 47166718 (2026-08-08) pulled the 7 GB t212 image in 8m59s, well inside
    # the pull timeout, and was condemned 82 seconds later for "running 10m
    # without jobd ever stamping JOBD_STATUS". It had been running 82 seconds.
    #
    # The misattribution is the worse half. This verdict is emitted with
    # phase="env-setup" and suspect="our-boot-code-or-transfer-path", on the
    # documented reasoning that a host-independent recurrence implicates our
    # boot path — but a squeeze driven by PULL DURATION recurs on every slow
    # host by construction, so the telemetry points the next investigator at
    # our b2x/rclone ladder for what is a host-speed problem. Anchoring here
    # keeps the two phases separately attributable, which is the whole point of
    # having two verdicts.
    #
    # Falls back to start_date when the transition was not observed (a
    # supervisor that attached to an already-running box), which is the old
    # behaviour and still the conservative one: it can only fire EARLIER.
    anchor = (jc.get("boot_running_since")
              if jc.get("boot_running_iid") == iid else None)
    if anchor is None:
        anchor = inst.get("start_date")
    try:
        age = now - float(anchor)  # type: ignore[arg-type]  # guarded
    except (TypeError, ValueError):
        return None                      # no clock, no verdict
    if age > _boot_deadline_backoff(sla, jc.get("pull_relaunches", 0)):
        return "sla"
    return None


# The serve lane's SELF-PARK signal, ported 2026-08-16 (plan §8 step 6
# leftovers). It lands HERE, with the rest of the serve cluster
# (`_serve_status_line_soft`, `_serve_relaunch_*`, `_serve_sla_emit`), and not
# in `supervise/job_lane.py` where its only caller lives: the caller is the jobs
# tick running in `serve_mode`, and `cli/job/supervise.py` already ruled that
# cli-surface.json's transitive attribution to that COMMAND is not a home ("its
# real home is the serve lane, not this module"). Both readers parse the same
# single overwritten `serve/<SERVE_ID>/SERVE_STATUS` object, so a change to that
# marker's grammar has to be made once, in one place, or the two disagree about
# what a line means.
#
# `job_lane` reaches it in module-attribute form (`replacement._serve_self_park_soft`)
# rather than by an assignment rebind, deliberately: the two modules import each
# other, so a module-level `_serve_self_park_soft = replacement._serve_self_park_soft`
# in `job_lane` is an AttributeError whenever `replacement` is the module imported
# first (it reaches its own `import job_lane` before this def exists). The
# attribute call also keeps `monkeypatch.setattr(replacement, ...)` steering the
# tick, which is the idiom that module's header states.
# moved-from: herdd.SERVE_SELF_PARK_FRESH_S
SERVE_SELF_PARK_FRESH_S = 3600  # marker older than this can't explain THIS stop


# moved-from: herdd._serve_self_park_soft
def _serve_self_park_soft(serve_id: object, *,
                          max_age_s: float = SERVE_SELF_PARK_FRESH_S) -> bool:
    """Did this serve box park ITSELF? True only for a FRESH self-park line in
    serve/<SERVE_ID>/SERVE_STATUS (the MAX_HOURS watchdog writes `SELF_PARKED
    <ts> max_hours` before its API stop; pre-2026-08-02 wires wrote `FAILED <ts>
    max_hours` — both accepted). The marker is a single overwritten object that
    OUTLIVES the park that wrote it, so freshness is load-bearing: an hour-old
    self-park must never explain away a LATER genuine eviction (the stale-marker
    inverse of the 2026-07-11 outbid-misread-as-park regression). Returns False
    on any read/parse failure — fail toward RESCUE, the same doctrine as
    classify_job_box_stop (abandoning an outbid box is the unsafe direction)."""
    bucket = os.environ.get("B2_BUCKET")
    if not serve_id or not bucket:
        return False
    rc, out, _err = b2._rclone_soft(
        ["cat", f"b2:{bucket}/serve/{serve_id}/SERVE_STATUS"])
    if rc != 0 or not out:
        return False
    toks = (out.strip().splitlines() or [""])[0].split()
    if len(toks) < 2:
        return False
    verdict, ts = toks[0], toks[1]
    self_park = verdict == "SELF_PARKED" or (
        verdict == "FAILED" and "max_hours" in toks[2:])
    if not self_park:
        return False
    try:
        age = time.time() - datetime.datetime.strptime(
            ts, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc).timestamp()
    except ValueError:
        return False
    return -300 <= age <= max_age_s          # -300: tolerate small clock skew


# moved-from: herdd._serve_status_line_soft
def _serve_status_line_soft(
        serve_id: object) -> tuple[str | None, float | None, str | None]:
    """(token, epoch_ts|None, detail) of the first SERVE_STATUS line, or
    (None, None, None) on any failure. The marker line is the serve lane's
    whole phase-stamped boot timeline, one milestone at a time:

      `LAUNCHED <ts>`            workstation, at launch (launch_serve.sh)
      `PULLING <ts> boot`        BOX: onstart running, creds proven, pre-pull
      `PULLING <ts> base` / `adapter:<n>` / `chat_template`
                                 BOX: a B2 transfer in flight (serve_vllm.sh)
      `READY <ts> <ids>` etc.    BOX: online (FAILED/SELF_PARKED = terminal)

    so the token+detail pair discriminates WHICH boot phase a stall is in —
    the remedies differ (host-side image pull vs OUR transfer path vs OUR
    provisioning code; owner directive 2026-08-03)."""
    bucket = os.environ.get("B2_BUCKET")
    if not serve_id or not bucket:
        return None, None, None
    rc, out, _err = b2._rclone_soft(
        ["cat", f"b2:{bucket}/serve/{serve_id}/SERVE_STATUS"])
    if rc != 0 or not out:
        return None, None, None
    toks = (out.strip().splitlines() or [""])[0].split()
    if not toks:
        return None, None, None
    ts = None
    if len(toks) > 1:
        try:
            ts = datetime.datetime.strptime(
                toks[1], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=datetime.timezone.utc).timestamp()
        except ValueError:
            ts = None
    return toks[0], ts, " ".join(toks[2:])


# --------------------------------------------------------------------------- #
# WHAT is this box serving? (P3, 2026-08-24)
#
# P2 taught the box to verify its own weights and stamp the grade-A sha12 into
# the READY marker. Nothing downstream READ it except a human running
# `serve_ready.sh --expect-ident`, which means an unattended serve box could
# drift onto the wrong weights and keep answering /v1/models with the right
# name for as long as it stayed rented. This is the daemon's half: the watch
# carries the artifact it was registered for, and every tick compares intent
# against what the box says about itself.
#
# ONE STATE CONDEMNS, and it is the narrow one. A READY line whose `ident=`
# DISAGREES with the watch's pin is a box that proved it is coherently serving
# something — just not the thing every eval scored against it will be labelled
# with. That is the poisoned-training-row shape, so the box stops being a serve
# target: parked, and refused to the rescue/relaunch ladder from here on. An
# ABSENT ident (`unarmed`) and the on-box gate's own refusals (`gate_failed`)
# are LOUD BUT PASSIVE — the first is a launch that never armed the gate, the
# second is a box that already refused to serve, and neither is improved by
# fleetd parking it out from under an operator who is mid-diagnosis.
#
# NEVER a destroy. `_serve_boot_sla_condemn` may destroy because it can re-fire
# the launch spec on a different host, and a stuck boot is host-shaped. A wrong
# ARTIFACT is not host-shaped: the same relaunch on a new box pulls the same
# wrong weights, so a destroy here would spend money to reproduce the defect.
# --------------------------------------------------------------------------- #
SERVE_IDENTITY_VERDICT = "identity_mismatch"


def _serve_identity_tick(jc: MutableMapping[str, Any],
                         inst: Mapping[str, Any] | None,
                         now: float) -> str | None:
    """Compare the watch's expected identity against the box's READY marker.

    Returns `SERVE_IDENTITY_VERDICT` when the box must stop being treated as a
    healthy serve target, else None. Records the verdict on
    `jc["serve_identity"]` either way — fleetd persists that onto the watch and
    DERIVES its alarms from it, so this function never alarms directly.

    A watch with no `expect_ident` returns before any B2 read and leaves NO key
    behind: a legacy serve watch must be byte-identical to what it was.
    """
    expect = jc.get("expect_ident")
    if not expect:
        # No pin: leave NOTHING behind, including a latch from a previous
        # registration. Re-`watch`ing without `--artifact` is the operator's
        # documented way to say "stop checking this endpoint", and it has to
        # actually release the box — a latch that outlived the pin it was made
        # against would be an alarm with no way to retract it.
        jc.pop("serve_identity", None)
        jc.pop("serve_identity_condemned", None)
        return None
    if jc.get("serve_identity_condemned"):
        # Latched: re-reading the marker cannot un-condemn a box, and the whole
        # point is that no later rung gets to resurrect it. Costs no I/O.
        return SERVE_IDENTITY_VERDICT
    if inst is None:
        return None                      # no box this tick, no verdict
    serve_id = models._instance_serve_label(inst)
    tok, _ts, detail = _serve_status_line_soft(serve_id)
    rec = serve_ident.classify(expect, tok, detail)
    rec["artifact"] = jc.get("model_artifact")
    rec["serve_id"] = serve_id
    rec["iid"] = str(jc.get("iid") or "")
    prev = jc.get("serve_identity") or {}
    # `since` clocks the STATE, not the tick — an alarm that resets its age
    # every 45 s cannot tell a two-minute blip from a six-hour one.
    rec["since"] = (prev.get("since")
                    if prev.get("state") == rec["state"] else now) or now
    jc["serve_identity"] = rec
    if rec["state"] != "mismatch":
        return None

    jc["serve_identity_condemned"] = True
    iid = jc.get("iid")
    parked = False
    if jc.get("dry_run"):
        print(f"[dry-run] would PARK {iid}: serving identity {rec['observed']}, "
              f"watch expects {rec['expected']}")
    else:
        parked = lifecycle._stop_instance_soft(iid)
    rec["parked"] = parked
    journal._job_ladder_journal(
        jc, "serve_identity_mismatch", iid=str(iid),
        artifact=rec.get("artifact"), serve_id=serve_id,
        expected_ident=rec["expected"], observed_ident=rec["observed"],
        parked=parked,
        note="the box VERIFIED an identity on itself and it is not the one "
             "this watch was registered for — parked (never destroyed) and "
             "withdrawn from the rescue/relaunch ladder; every eval scored "
             "against it carries the wrong label")
    art = f" (artifact {rec['artifact']})" if rec.get("artifact") else ""
    fate = "PARKED" if parked else "PARK FAILED — park it by hand"
    print(f"!! SERVE IDENTITY MISMATCH on {iid}: the box verified "
          f"'{rec['observed']}', this watch expects '{rec['expected']}'{art}. "
          f"{fate}; the ladder will not rescue or relaunch it.")
    return SERVE_IDENTITY_VERDICT


# moved-from: herdd._serve_relaunch_dir
def _serve_relaunch_dir() -> str:
    """State dir for launch_serve.sh boot-SLA relaunch specs (<IID>.json).
    Workstation-local runtime state (never committed); keyed by instance so the
    owning watch finds the spec for exactly the box it is holding to the SLA."""
    return os.path.join(os.environ.get("XDG_STATE_HOME")
                        or os.path.expanduser("~/.local/state"),
                        "herdd", "serve-relaunch")


# moved-from: herdd._serve_relaunch_spec_load
def _serve_relaunch_spec_load(iid: object) -> dict[str, Any] | None:
    """The relaunch spec launch_serve.sh saved for this instance, or None —
    a pre-SLA launch, a pinned (--offer/--machine/--host) launch, an --on-box
    attach, or a corrupt file. None means the SLA cannot re-fire the serve and
    therefore must not destroy it (alarm-only)."""
    try:
        with open(os.path.join(_serve_relaunch_dir(), f"{iid}.json")) as fh:
            spec = json.load(fh)
        return spec if isinstance(spec, dict) and spec.get("argv") is not None \
            else None
    except Exception:
        return None


# moved-from: herdd._serve_sla_emit
def _serve_sla_emit(serve_id: object, event: str,
                    **fields: Any) -> None:  # noqa: ANN401 — open event payload
    """Best-effort runmeta event on the SERVE_ID (launch_serve.sh already emits
    `launched` there), so every SLA kill/relaunch lands in runs/<SERVE_ID>/ and
    the history is auditable. Never raises."""
    if not serve_id:
        return
    try:
        runmeta.emit_event(str(serve_id), event, actor="boot-sla", **fields)
    except Exception:
        pass


# moved-from: herdd._serve_boot_sla_tick
def _serve_boot_sla_tick(jc: MutableMapping[str, Any],
                         inst: Mapping[str, Any], now: float) -> str | None:
    """Boot SLA for a `serve` watch (owner directive 2026-08-03 — motivating
    incident: serve box 46682177, created 07:33Z, onstart began 08:12Z — 39
    minutes of image pull on an 805 Mb/s host — PULLING base 08:37Z, READY
    08:44Z, while the session sat waiting). The marker line (see
    _serve_status_line_soft) is a phase-stamped timeline, and the SLA
    DISCRIMINATES the phases because the remedies differ (owner amendment,
    same day):

      * `LAUNCHED` past the deadline — the box never started onstart: the
        image/docker pull or vast container standup, HOST-side (only
        observable from outside the container). Remedy: destroy + re-fire the
        saved launch_serve spec on a DIFFERENT host -> returns 'sla'.
      * `PULLING <ts> boot` past the deadline — onstart IS running but our
        pre-pull provisioning is stalled: OUR code. Host rotation fixes
        nothing, so: loud one-shot alarm naming the phase, no kill.
      * `PULLING <ts> base|adapter:*|chat_template` past the deadline — a B2
        transfer is stalled: suspect OUR transfer path first (serve_vllm.sh
        still pulls via rclone, whose stream clamp caps at 4-9 flows on
        per-flow-shaped hosts; b2x with ~68 flows is the sanctioned layer —
        memory b2x-is-the-transfer-layer). Same loud alarm, no kill.
      * anything else (READY/FAILED/SELF_PARKED/...) — online or terminal:
        the SLA's scope is over (other machinery owns failures).

    Every marker transition this watch observes is timestamped into
    jc["boot_marker_seen"] and shipped with the SLA events, so a breach names
    the phase that breached and its elapsed — the telemetry that would have
    answered "where did 46682177's 71 minutes go" without renting a probe box.

    Armed only while the marker still reads LAUNCHED/PULLING pre-online: a
    resumed/adopted box carries READY/SELF_PARKED and is never held to a
    fresh-boot SLA. Enforcement requires the launch_serve.sh relaunch spec
    (no spec -> disabled for this watch, one advisory note; guard/ls
    advisories still cover the box). Returns 'sla' | None."""
    if jc.get("boot_sla_disabled"):
        return None
    sla = config._boot_knob("BOOT_SLA_S")
    if sla <= 0:
        return None
    iid = jc.get("iid")
    if jc.get("boot_online_iid") == iid:
        return None
    spec = _serve_relaunch_spec_load(iid)
    if spec is None:
        jc["boot_sla_disabled"] = True
        if (inst.get("actual_status") or "").lower() in health._BOOT_LOADING_STATES:
            print(f">> serve boot-SLA: no relaunch spec for {iid} (pre-SLA "
                  f"launch, pinned launch, or --on-box attach) — SLA "
                  f"enforcement OFF for this watch (guard/ls still alarm)")
        return None
    serve_id = spec.get("serve_id") or models._instance_serve_label(inst)
    tok, mts, detail = _serve_status_line_soft(serve_id)
    if tok is None:
        return None                      # marker unreadable this tick: no verdict
    seen = jc.setdefault("boot_marker_seen", {})
    key = f"{tok} {detail}".strip() if tok == "PULLING" else tok
    seen.setdefault(key, now)
    ddl = _boot_deadline_backoff(sla, int(spec.get("sla_kills") or 0))
    if tok == "LAUNCHED":
        astat = (inst.get("actual_status") or "").lower()
        jc["boot_sla_phase"] = ("image-pull" if astat in health._BOOT_LOADING_STATES
                                else "pre-onstart (image pull/extract or "
                                     "container standup)")
        try:
            age = now - float(inst.get("start_date"))  # type: ignore[arg-type]  # guarded
        except (TypeError, ValueError):
            return None
        if age > ddl:
            return "sla"
        return None
    if tok == "PULLING":
        # onstart is running: the box came up — a stall past here is OURS, and
        # rotating hosts cannot fix it. Alarm loudly (once per sub-phase), keep
        # the box.
        if (detail or "").strip() == "boot":
            phase, suspect = "onstart-provisioning", (
                "our onstart code (serve_vllm.sh pre-pull work — the 46682177 "
                "timeline lost 25 min here; includes installing rclone, which "
                "the t211 image does NOT bake)")
        else:
            phase, suspect = f"b2-pull ({detail or '?'})", (
                "our side, not the host: measured 2026-08-03, tuned rclone "
                "and b2x BOTH saturate past the host's own rating (157 vs "
                "169 MB/s fast host; 66 vs 87 MB/s on a 321 Mb/s-rated one), "
                "so a genuine stall here is anomalous — suspect a B2/regional "
                "incident, auth/key expiry, or a regression in our pull "
                "flags, and check b2x vs rclone parity before blaming the "
                "host")
        elapsed = now - (mts if mts is not None else seen[key])
        if elapsed > ddl and jc.get("boot_sla_phase_alarmed") != key:
            jc["boot_sla_phase_alarmed"] = key
            _serve_sla_emit(serve_id, "boot_sla_phase_stall", instance_id=iid,
                            machine_id=inst.get("machine_id"), phase=phase,
                            elapsed_s=int(elapsed), deadline_s=int(ddl),
                            suspect=suspect,
                            status_msg_available=bool(inst.get("status_msg")),
                            milestones=_boot_milestones(seen, inst, now))
            print(f"!! serve boot-SLA: {serve_id} stalled in {phase} for "
                  f"{fmt._age_str(int(elapsed))} (> {int(ddl)}s) — NOT killing "
                  f"the box (this phase is {suspect}); investigate the "
                  f"transfer/provisioning path")
        return None
    jc["boot_online_iid"] = iid          # READY/FAILED/...: SLA scope is over
    return None


# moved-from: herdd._boot_milestones
def _boot_milestones(seen: Mapping[str, float], inst: Mapping[str, Any],
                     now: float) -> dict[str, int]:
    """{marker-key: seconds-after-box-start} for the SLA events — the compact
    phase-stamped boot timeline (created -> LAUNCHED -> PULLING boot ->
    PULLING base -> ...) that lets a breach event answer WHERE the time went."""
    try:
        start = float(inst.get("start_date"))  # type: ignore[arg-type]  # guarded
    except (TypeError, ValueError):
        start = None
    out = {}
    for k, t in sorted(seen.items(), key=lambda kv: kv[1]):
        out[k] = int(t - start) if start is not None else int(now - t)
    return out


# moved-from: herdd._serve_boot_sla_condemn
def _serve_boot_sla_condemn(jc: MutableMapping[str, Any],
                            inst: Mapping[str, Any]) -> str | None:
    """Destroy an SLA-breaching serve box and re-fire its saved launch_serve.sh
    spec on a DIFFERENT host (failed machine excluded, kill count carried
    forward in the next spec). Same license as the jobs pull watchdog and the
    same reconciliation with the 2026-08-03 guard ruling: the PASSIVE sweep
    never destroys a loading box because it cannot re-attach the workload — the
    OWNING lifecycle can, because re-firing the spec IS the re-attach (fresh
    keys, fresh LAUNCHED marker, a new fleetd serve watch registered by the
    script itself). Order: loop/budget guards, destroy (frees the serve:<ID>
    label the replacement re-uses — launch_serve's dup preflight would refuse a
    live twin), then re-exec. A failed relaunch AFTER the destroy is
    'unrecoverable' (loud: the serve is down, the spec path is printed).
    Returns 'sla_relaunched' on success (terminal for THIS watch — its
    successor was registered by the relaunch), None to keep supervising."""
    a = jc["a"]
    old = str(jc.get("iid"))
    machine = inst.get("machine_id")
    spec = _serve_relaunch_spec_load(old) or {}
    kills = int(spec.get("sla_kills") or 0)
    serve_id = spec.get("serve_id") or models._instance_serve_label(inst)
    try:
        age = int(time.time() - float(inst.get("start_date")))  # type: ignore[arg-type]  # guarded
    except (TypeError, ValueError):
        age = -1
    ddl = int(_boot_deadline_backoff(config._boot_knob("BOOT_SLA_S"), kills))
    phase = jc.get("boot_sla_phase") or "image-pull"
    _serve_sla_emit(serve_id, "boot_sla_condemned", instance_id=old,
                    machine_id=machine, boot_age_s=age, sla_kills=kills,
                    deadline_s=ddl, phase=phase, suspect="host",
                    inet_down=inst.get("inet_down"),
                    # some hosts stream NO status_msg for the whole pull
                    # (m=127653, instance 46726441, 2026-08-03) — record its
                    # availability so fleet data shows how common that is and
                    # why the SLA had to be wall-clock, not progress-gated.
                    status_msg_available=bool(inst.get("status_msg")),
                    milestones=_boot_milestones(
                        jc.get("boot_marker_seen") or {}, inst, time.time()))
    print(f"!! BOOT-SLA-CONDEMNED serve box {old} (serve {serve_id}): no "
          f"onstart progress {fmt._age_str(max(age, 0))} after start (> {ddl}s "
          f"SLA; phase {phase}; machine {machine}) — host-side, so: destroy + "
          f"re-fire launch_serve on a different host")
    max_r = config._boot_knob("BOOT_MAX_HOST_RETRIES", cast=int)
    if kills >= max_r:
        jc["boot_sla_disabled"] = True
        _serve_sla_emit(serve_id, "boot_sla_exhausted", sla_kills=kills)
        print(f"!! serve boot-SLA: {kills} kills already burned "
              f"(BOOT_MAX_HOST_RETRIES={max_r}) — NOT re-renting (loop guard; "
              f"the image or the market may be the problem). Box {old} kept; "
              f"investigate, then relaunch by hand.")
        return None
    if a.budget is not None and jc.get("spend_usd", 0.0) >= a.budget:
        print(f"!! serve boot-SLA: budget already consumed "
              f"(${jc.get('spend_usd', 0.0):.2f} >= ${a.budget}) — a relaunch "
              f"must not escape --budget; alarming only")
        return None
    if jc.get("dry_run"):
        print(f"[dry-run] would destroy {old} and re-fire launch_serve.sh for "
              f"{serve_id} excluding machine {machine}")
        return None
    hostrep.note_strike(machine, "boot_sla", iid=old,
                        note=f"serve boot SLA, kill #{kills + 1}")
    # The relaunch spec carries this watch's own exclusions forward; the durable
    # block list is unioned in so a serve relaunch cannot land on a host another
    # lane condemned yesterday. `launch_serve.sh` re-enters `pick_offers`, which
    # would filter it again — belt and braces, and this arm is the one whose
    # exclusions the operator reads in the log line below.
    excl = hostrep.with_blocked(
        {int(m) for m in (spec.get("exclude_machines") or [])}
        | ({int(machine)} if machine is not None else set())) or []
    failed = lifecycle._destroy_and_revoke([old], jc.get("instances") or [],
                                           "boot_sla_destroy")
    if failed:
        print(f"!! serve boot-SLA: destroy of {old} FAILED — retrying next "
              f"tick (never relaunch over a live serve:<ID> twin)")
        return None
    script = os.path.join(config._HERE, os.path.basename(spec.get("script")
                                                  or "launch_serve.sh"))
    cmd = ["bash", script, *[str(x) for x in (spec.get("argv") or [])],
           "--serve-id", str(serve_id), "--sla-kills", str(kills + 1)]
    for m in excl:
        cmd += ["--exclude-machine", str(m)]
    print(f">> serve boot-SLA: re-firing {os.path.basename(script)} for "
          f"{serve_id} (kill #{kills + 1}, excluding machines {excl})")
    try:
        r = subprocess.run(cmd, timeout=1800)
        ok = r.returncode == 0
    except Exception as e:
        print(f"!! serve boot-SLA: relaunch exec failed ({e})")
        ok = False
    if not ok:
        _serve_sla_emit(serve_id, "boot_sla_relaunch_failed",
                        sla_kills=kills + 1)
        print(f"!! serve boot-SLA: relaunch FAILED after destroying {old} — "
              f"serve {serve_id} is DOWN; re-run launch_serve.sh by hand "
              f"(spec: {os.path.join(_serve_relaunch_dir(), old + '.json')})")
        return "unrecoverable"
    _serve_sla_emit(serve_id, "boot_sla_relaunched", from_instance=old,
                    sla_kills=kills + 1, excluded_machines=excl)
    print(">> serve boot-SLA: replacement launched; launch_serve registered "
          "its fleetd watch + wrote the next relaunch spec — this watch is "
          "done")
    return "sla_relaunched"


# moved-from: herdd._job_pull_condemn
def _job_pull_condemn(jc: MutableMapping[str, Any], inst: Mapping[str, Any],
                      verdict: str) -> None:
    """Terminate-and-reschedule a pull-condemned jobs box (the FLEETD_DESIGN §8
    amendment's authorized destroy). Order matters: launch the replacement,
    MOVE the queue, then destroy — the condemned box is the last thing to go,
    so a failure at any step leaves a recoverable state, never an orphaned
    queue. Mutates jc to track the replacement and returns None (keep
    supervising). On exhausted retries or an unlaunchable replacement it
    alarms and leaves the (GPU-unbilled) box alone rather than orphaning its
    tickets."""
    a = jc["a"]
    old = str(jc.get("iid"))
    s = jc.get("pull_sampler")
    mbps = round(s.last_mbps, 2) if s is not None and s.last_mbps is not None \
        else None
    machine = inst.get("machine_id")
    age = int(time.time() - float(inst.get("start_date") or time.time()))
    if verdict == "slow":
        why = (f"host pulled at {mbps} MB/s aggregate (< "
               f"{config._boot_knob('BOOT_MIN_MBPS'):g} MB/s floor)")
    elif verdict == "sla":
        # Report the ENV-SETUP age, not the box age. The old wording read
        # "box running 10m" for box 47166718, which had been running 82s after
        # a 9-minute pull — the number named the wrong phase and made a
        # host-speed problem look like a hung bootstrap.
        _since = (jc.get("boot_running_since")
                  if jc.get("boot_running_iid") == jc.get("iid") else None)
        _env_age = int(time.time() - _since) if _since else None
        _phr = (f"box running {fmt._age_str(_env_age)}" if _env_age is not None
                else f"box up {fmt._age_str(age)} (running-transition not observed)")
        why = (f"{_phr} without jobd ever stamping JOBD_STATUS (boot SLA "
               f"{int(config._boot_knob('BOOT_SLA_S'))}s base"
               + (f", after a {fmt._age_str(age - _env_age)} pull"
                  if _env_age is not None else "") + ")")
    else:
        why = (f"pull not finished in {fmt._age_str(age)} "
               f"(> {int(config._boot_knob('BOOT_PULL_TIMEOUT_S'))}s timeout)")
    bad = jc.setdefault("pull_bad_machines", set())
    if machine is not None:
        bad.add(machine)
        # ...and DURABLY, outliving this watch (2026-08-20). `pull_bad_machines`
        # dies with `jc`, which is how machine 72425 was condemned on 08-17 and
        # rented again from a clean slate on 08-20. `sla` is charged as a
        # distinct kind because its own `suspect` field above says the fault may
        # be ours — the score treats it the same today, but a future audit that
        # finds our boot path at fault can `hostrep.forget` exactly those.
        hostrep.note_strike(machine, {"slow": "pull_slow", "sla": "boot_sla"}
                            .get(verdict, "pull_timeout"), iid=old, note=why)
    journal._job_handoff_emit(jc, "pull_condemned", verdict=verdict, mbps=mbps,
                              machine_id=machine, boot_age_s=age,
                              total_bytes=(s.total_bytes if s else None),
                              # phase discrimination (owner amendment 2026-08-03): the
                              # remedies differ, so the event names the cause. slow/
                              # deadline = the docker image pull, host-side; sla = the
                              # billed env-setup phase (onstart/jobd bootstrap incl.
                              # the B2 jobd-bundle pull) — if it recurs across hosts,
                              # suspect OUR boot/transfer path (b2x vs rclone flows),
                              # not the hosts.
                              phase=("env-setup" if verdict == "sla" else "image-pull"),
                              suspect=("our-boot-code-or-transfer-path"
                                       if verdict == "sla" else "host"),
                              inet_down=inst.get("inet_down"),
                              # null on some hosts for the ENTIRE pull (m=127653,
                              # 2026-08-03) — when False, the throughput floor was
                              # blind and only the wall-clock deadline could fire.
                              status_msg_available=bool(inst.get("status_msg")))
    _label = "BOOT-SLA-CONDEMNED" if verdict == "sla" else "PULL-CONDEMNED"
    _cost = ("env-setup bills FULL GPU rate, so waiting out a dead boot is "
             "the expensive option" if verdict == "sla" else
             "pull is GPU-unbilled, so this costs only the wasted pull")
    print(f"!! {_label} {old}: {why} (machine {machine}, "
          f"phase {getattr(s, 'phase', '?')}) — bad host; terminate + "
          f"reschedule ({_cost})")
    # Task #78: condemning a box is a money-moving decision (it commits us to
    # renting another one), and on 2026-08-08 23:27:56Z the entire condemn ->
    # launch -> retarget -> destroy chain reached `fleet log` as NOTHING.
    journal._job_ladder_journal(jc, "jobs_box_condemned", iid=str(old),
                                verdict=verdict, machine_id=machine, mbps=mbps,
                                boot_age_s=age, phase=getattr(s, "phase", None),
                                dph=models._num_dph((inst or {}).get("dph_total")),
                                spend_usd=round(jc.get("spend_usd", 0.0), 4),
                                budget_usd=getattr(a, "budget", None),
                                note=f"{_label}: {why} — {_cost}")
    retries = jc.get("pull_relaunches", 0)
    max_r = config._boot_knob("BOOT_MAX_HOST_RETRIES", cast=int)
    if retries >= max_r:
        jc["pull_watchdog_disabled"] = True
        journal._job_handoff_emit(jc, "pull_relaunch_exhausted", relaunches=retries)
        print(f"!! pull watchdog: {retries} reschedules already burned "
              f"(BOOT_MAX_HOST_RETRIES={max_r}) — NOT re-renting (loop "
              f"guard; the image itself may be the problem). Box {old} kept "
              f"(GPU-unbilled) with its queue; investigate, then `job "
              f"retarget` by hand.")
        return None
    if a.budget is not None and jc.get("spend_usd", 0.0) >= a.budget:
        print(f"!! pull watchdog: budget already consumed "
              f"(${jc.get('spend_usd', 0.0):.2f} >= ${a.budget}) — a "
              f"reschedule must not escape --budget; alarming only")
        return None
    if jc.get("dry_run"):
        print(f"[dry-run] would destroy {old} and reschedule its queue on a "
              f"fresh box (excluding machines {sorted(bad)})")
        return None
    # Same price rail as the eviction lane (doc 50 R3): a forced rehost is an
    # autonomous rental too, and "the box wouldn't pull" is not a licence to buy
    # a 3x one. A refusal here is safe by construction — the condemned box is
    # KEPT with its queue and the next tick retries.
    # INHERIT THE CONDEMNED BOX'S RENTAL TYPE. This defaulted to `rental="bid"`
    # (the _launch_job_replacement signature default) and so silently rehosted
    # an ON-DEMAND box onto spot. On-demand is not a price preference the
    # rehost may re-optimise: it is chosen when an interruption would destroy
    # the work rather than merely delay it — a paired A/B or a serial
    # measurement ladder, where losing one arm mid-run confounds the
    # comparison instead of restarting it. Measured 2026-08-08: on-demand box
    # 47165024 (a 9-cell DDP ladder, launched --type ondemand for exactly that
    # reason) was pull-condemned on a slow Hong Kong host, silently rehosted to
    # a spot box, and outbid ~2 min later, with the operator's choice nowhere
    # in the journal. The EVICTION lane never had this hole — it threads
    # `rental=dec.rental` from bidpolicy.replacement_decision, which has an
    # explicit on-demand rung. Only this lane dropped it on the floor.
    # Only an EXPLICIT is_bid=False upgrades the rehost to on-demand. An absent
    # field means we cannot tell, and guessing "ondemand" there would silently
    # DOUBLE the bill for a spot box on an API shape change — so unknown keeps
    # the pre-existing bid behaviour, and only a known on-demand box is
    # protected.
    _rental = "ondemand" if inst.get("is_bid") is False else "bid"
    _ceiling = _job_replacement_ceiling(jc)
    cid, dph, reason = _launch_job_replacement(jc, sorted(bad),
                                               rental=_rental,
                                               max_dph=_ceiling)
    if reason is not None:
        _job_note_replacement_refusal(jc, reason, _ceiling)
        print(f"!! pull watchdog: replacement launch failed ({reason}: "
              f"{jc.get('last_error')}) — condemned box KEPT so its queue is "
              f"not orphaned; will retry next tick")
        journal._job_ladder_journal(jc, "jobs_box_launch_failed", iid=str(old),
                                    lane="pull_reschedule", reason=reason,
                                    detail=jc.get("last_error"), ceiling=_ceiling,
                                    refusals=jc.get("replacement_refusals"),
                                    note="condemned box KEPT with its queue; retrying "
                                         "next tick")
        jc.pop("pull_sampler", None)         # re-arm: fresh window on retry
        return None
    _job_clear_replacement_refusals(jc)
    journal._job_ladder_journal(jc, "jobs_box_launched", iid=str(cid),
                                lane="pull_reschedule", from_box=str(old),
                                rental=_rental, dph=models._num_dph(dph), ceiling=_ceiling,
                                disk_gb=jc.get("last_replacement_disk_gb"),
                                launch_disk_gb=jc.get("launch_disk_gb"),
                                excluded_machines=sorted(bad),
                                budget_usd=getattr(a, "budget", None),
                                spend_usd=round(jc.get("spend_usd", 0.0), 4),
                                note="autonomous rental: the condemned host could not "
                                     "finish its pull, so the queue is being rehosted")
    moved, failed = _retarget_pending_tickets(old, cid)
    journal._job_ladder_journal(jc, "jobs_queue_retargeted", iid=str(cid),
                                lane="pull_reschedule", from_box=str(old),
                                to_box=str(cid), moved_jobs=len(moved),
                                failed_moves=len(failed),
                                job_ids=[str(m) for m in moved],
                                note="tickets moved BEFORE the old box dies — a ticket "
                                     "must never point at a box that is already gone")
    journal._job_handoff_emit(jc, "pull_rescheduled", from_box=old, to_box=str(cid),
                              verdict=verdict, mbps=mbps, machine_id=machine,
                              moved_jobs=len(moved), failed_moves=len(failed),
                              dph=dph,
                              # same sizing record as the eviction lane (task #69) —
                              # this lane rehosts a box condemned mid-PULL, whose usage
                              # snapshot is the least trustworthy of all
                              disk_gb=jc.get("last_replacement_disk_gb"),
                              launch_disk_gb=jc.get("launch_disk_gb"))
    print(f">> pull-reschedule: {old} (machine {machine}, {mbps} MB/s) -> "
          f"{cid}; {len(moved)} ticket(s) moved"
          + (f", {len(failed)} FAILED (see above)" if failed else "")
          + "; destroying the condemned box")
    failed_destroy = lifecycle._destroy_and_revoke([old], jc.get("instances") or [],
                                                   "pull_condemned_destroy")
    if failed_destroy:
        print(f"!! pull watchdog: destroy of {old} FAILED — it still bills "
              f"storage; destroy it by hand (queue already moved to {cid})")
    journal._job_ladder_journal(jc, "jobs_box_destroyed", iid=str(old),
                                lane="pull_reschedule", to_box=str(cid),
                                actor="fleetd:pull-watchdog", ok=not failed_destroy,
                                machine_id=machine, verdict=verdict,
                                note="condemned box destroyed by the LADDER, not by an "
                                     "operator — the `operator_intent_destroy` line "
                                     "next to this one carries the workstation user "
                                     "because `fleet_operator_intent` names the host, "
                                     "and its `reason` is the real actor"
                                     if not failed_destroy else
                                     "destroy FAILED — the box still bills allocated "
                                     "storage; `herdd reap` or destroy by hand")
    # Track the replacement: fresh box, fresh market anchors, fresh sampler —
    # the SAME reset block as the eviction-replacement lane (review 2026-08-10,
    # H2: this lane wrote the ON-DEMAND list price into last_bid, arming the
    # defend/decay ladder and the tick-1 preferred-ceiling alarm against a box
    # that cannot be outbid, and carried the OLD machine's sticky on-demand
    # price into every rail of the new one).
    jc["iid"] = str(cid)
    jc["pull_relaunches"] = retries + 1
    jc.pop("pull_sampler", None)
    jc.pop("pull_sampler_iid", None)
    jc["last_bid"] = dph if _rental == "bid" else None
    jc["first_seen_dph"] = dph if _rental == "bid" else None
    jc["floor_samples"] = []
    jc["decay_streak"] = 0
    jc["decay_streak_since"] = None      # a new box starts a fresh decay dwell
    jc["rescue_deadline"] = None
    jc["rebid_rungs"] = 0
    jc["rebid_refused"] = None
    # the shared box-swap seam: self-floor episode + echo window + the sticky
    # on-demand clamp (per MACHINE, and a rehost can land on the SAME machine,
    # where the old echoes would suppress a genuine competitor floor at a price
    # we recently held — review 2026-08-10, #4)
    ladder_core.box_swap_reset(jc)  # type: ignore[no-untyped-call]
    jc.pop("evicted_announced", None)  # the old box's eviction is history
    job_lane._job_evicted_latch_reset(jc)
    job_lane._job_notify_box_swap_reset(jc)  # ...and so is the row that
                                             # labelled it
    jc["pref_alarmed"] = False
    jc["ceiling_escalated"] = False
    # the under-delivered-disk warning is per BOX: the replacement can land on a
    # host that short-changes it too, and a latch from the dead box would eat it
    jc.pop("disk_shortfall_said", None)
    return None


# --------------------------------------------------------------------------- #
# Automatic EVICTION REPLACEMENT (owner directive 2026-08-05)
#
# The gap this closes, in the owner's words: fleetd "declares an evicted spot box
# `unrecoverable` and stops — the fix is a new box, which it won't rent itself,
# so a human must hand-rescue. That just cost two hand-rescues in one training
# run." (docs/plans/witness/g2_push/V7_TRAIN_RUN_2026-08-05.md; readout
# FLEETD_AUTOREPLACE_2026-08-05.md.)
#
# The bid ladder above is complete for what it covers — it defends, it rescues,
# and on 2026-08-05 it correctly declined to bid at all, twice, because
# `_bid_target` could see that no legal bid could win. What it lacks is the rung
# BELOW a bid: rent a different box. `_job_pull_condemn` already built every
# impure piece of that (offer -> launch -> retarget -> destroy -> re-anchor the
# supervisor); this reuses them verbatim and puts a spend-bounded decision in
# front, so eviction recovery becomes automatic without becoming unbounded.
#
# `bidpolicy.replacement_decision` owns the arithmetic (budget remainder, count
# cap, ceiling derived from the original launch price, spot-vs-on-demand rung);
# everything here is I/O and journaling.
# --------------------------------------------------------------------------- #
# moved-from: herdd._job_replacement_knob
def _job_replacement_knob(jc: MutableMapping[str, Any], name: str,
                          default: _KnobT) -> _KnobT:
    """Per-watch replacement knob: the policy namespace first (`fleet watch
    --max-replacements` / the inline CLI flags), then the env override, then the
    bidpolicy default. Env exists so a running daemon can be tightened without a
    re-registration; a `None` on the namespace means "not set", never 0."""
    v = getattr(jc.get("a"), name, None)
    if v is None:
        v = os.environ.get(f"JOB_{name.upper()}")
    if v in (None, ""):
        return default
    try:
        return type(default)(v)  # type: ignore[call-arg]  # type[_KnobT] ctor
    except (TypeError, ValueError):
        return default


# moved-from: herdd._rebid_knob
def _rebid_knob(jc: MutableMapping[str, Any], name: str,
                default: _KnobT) -> _KnobT:
    """Per-watch re-bid-ladder knob, resolved namespace > env > herdd.yaml >
    bidpolicy default. Sibling of `_job_replacement_knob`, with the yaml rung
    added because these are policy numbers an operator tunes per FLEET, not per
    watch (the audit's directive was "make the ladder parameters config, not
    constants"). Names are `rebid_step` / `rebid_max_rungs` / `rebid_wait_s` /
    `rebid_ceiling_mult`; env is `JOB_REBID_*`, yaml is the same upper-cased key.
    A malformed value is skipped, never fatal — a bad knob must not disarm the
    ladder."""
    v = getattr(jc.get("a"), name, None)
    if v is None:
        v = os.environ.get(f"JOB_{name.upper()}")
    if v in (None, ""):
        try:
            v = config.load_herdd_config().get(f"JOB_{name.upper()}")
        except Exception:
            v = None
    if v in (None, ""):
        return default
    try:
        return type(default)(v)  # type: ignore[call-arg]  # type[_KnobT] ctor
    except (TypeError, ValueError):
        return default


# moved-from: herdd._job_defense_inputs
def _job_defense_inputs(jc: MutableMapping[str, Any],
                        now: float) -> dict[str, Any]:
    """The six job-aware defense inputs `bidpolicy.defense_ceiling` reads, off
    this tick's `jc` (AUTOBID_DESIGN "Next iteration" §3).

    p_alt: the pre-eviction replacement-market read, only while fresh. R: the
    progress-ETA work horizon — NOT the timeout ceiling (the 2026-08-08
    5x-inflation lesson lives in `_jobs_work_horizon_h`'s docstring); None
    (unknown) falls back to the policy prior inside `defense_ceiling`. L: the
    widest checkpoint interval among pending tickets, the same bound
    `spot_breakeven` uses. All None-safe: with no fresh p_alt every consumer is
    byte-identical to its pre-defense self.

    EXTRACTED (S2b review round 2). It had exactly one caller — the re-bid rung
    — until the notification-priced rescue quote had to be bounded by the same
    ceiling, and a ceiling assembled from two different derivations of the same
    six numbers is not the same ceiling. Both callers read this, and both run
    on the same tick, off the same `pending_views` and the same p_alt poll,
    which is what makes the equality §6.7-6 claims literally true rather than
    approximately true."""
    views = jc.get("pending_views") or []
    remaining_h: float | None
    try:
        remaining_h = risk._jobs_work_horizon_h(views, now)
    except Exception:
        remaining_h = None
    ckpt_h = 0.0
    for v in views:
        cs = models._num_dph((v or {}).get("checkpoint_s"))
        if cs and cs > 0:
            ckpt_h = max(ckpt_h, cs / 3600.0)
    return {"p_alt": _job_palt_fresh(jc, now), "remaining_h": remaining_h,
            "ckpt_interval_h": ckpt_h, "defend": risk._jobs_defend_hint(views),
            "prior_runtime_h": risk._jobs_prior_runtime_h(views, now),
            "setup_h": _rebid_knob(jc, "spot_setup_h", bidpolicy.SPOT_SETUP_H)}


# moved-from: herdd._job_defense_cap
def _job_defense_cap(jc: MutableMapping[str, Any],
                     now: float) -> float | None:
    """The job-aware defense ceiling this tick, or None when no fresh `p_alt`
    makes one derivable. The scalar the rescue quote's ceiling needs; the re-bid
    rung derives the same number inside `bidpolicy.rebid_ladder` from the same
    inputs."""
    try:
        cap: float | None = bidpolicy.defense_ceiling(  # type: ignore[no-untyped-call]
            **_job_defense_inputs(jc, now))[0]
        return cap
    except Exception:                     # a ceiling read never kills a tick
        return None


# moved-from: herdd._job_rebid_ladder
def _job_rebid_ladder(jc: MutableMapping[str, Any], a: argparse.Namespace,
                      iid: object, market: float | None,
                      on_demand: float | None, eviction_class: str | None,
                      now: float) -> bool:
    """The rung BELOW a stalled single rescue and ABOVE renting a replacement
    (autobid audit 2026-08-08). Returns True when a rung was placed and the
    supervisor should keep watching this box; False when the ladder refuses (the
    caller then falls through to `_job_eviction_replace` exactly as before).

    Why this ordering: a re-bid keeps the box's disk — rehydrated env, base
    model, dataset, newest checkpoint — where a replacement pays a MEASURED
    11m35s of setup on a cold one and loses whatever never reached B2. The
    ladder's total wall budget (REBID_MAX_RUNGS x REBID_WAIT_S = 15 min) is
    deliberately one replacement's setup cost, so preferring it can never cost
    more time than the thing it is avoiding.

    Every outcome is journaled to the box's own event log — `rebid_ladder`
    (money moved) or `rebid_refused` (bound hit) — the same channel
    `eviction_replacement_decision` uses, and a refusal is additionally hung on
    `jc['replacement_refused']` so fleetd's `rescue_stalled` alarm names the
    bound instead of prescribing a raise the ladder has already proved
    impossible."""
    # Job-aware ONE-SHOT defense inputs — `_job_defense_inputs`, shared with the
    # notification-priced rescue quote's ceiling so the two cannot derive the
    # same six numbers two ways (review round 2).
    _di = _job_defense_inputs(jc, now)
    _p_alt, _remaining_h = _di["p_alt"], _di["remaining_h"]
    _ckpt_h, _defend = _di["ckpt_interval_h"], _di["defend"]
    _prior_h = _di["prior_runtime_h"]
    dec = bidpolicy.rebid_ladder(  # type: ignore[no-untyped-call]
        last_bid=models._num_dph(jc.get("last_bid")),
        market_min_bid=market, on_demand=on_demand,
        max_bid=jc.get("max_bid"),
        rungs_used=int(jc.get("rebid_rungs", 0) or 0),
        launch_dph_anchor=models._num_dph(jc.get("launch_dph_anchor")),
        eviction_class=eviction_class,
        budget_usd=getattr(a, "budget", None),
        spend_usd=jc.get("spend_usd", 0.0),
        step=_rebid_knob(jc, "rebid_step", bidpolicy.REBID_STEP),
        max_rungs=_rebid_knob(jc, "rebid_max_rungs", bidpolicy.REBID_MAX_RUNGS),
        ceiling_mult=_rebid_knob(jc, "rebid_ceiling_mult",
                                 bidpolicy.REBID_CEILING_MULT),
        p_alt=_p_alt, remaining_h=_remaining_h, ckpt_interval_h=_ckpt_h,
        defend=_defend, prior_runtime_h=_prior_h, setup_h=_di["setup_h"])
    if dec.action != "rebid":
        # Journal on CHANGE, not per tick: a stuck eviction re-evaluates this
        # every ~50 s, and on 2026-08-10 box 47398836 wrote 79 byte-identical
        # refusals in 66 min. The latch is the reason STRING — a refusal whose
        # numbers move (different bound, different headroom) is new information
        # and journals again; the same sentence twice is not. The decision is
        # still re-MADE every tick (that is how rung zero eventually won that
        # box back); only the announcement dedups.
        _said = jc.get("rebid_refused") == dec.reason
        jc["rebid_refused"] = dec.reason
        if not _said:
            journal._job_handoff_emit(jc, "rebid_refused", reason=dec.reason,
                                      eviction_class=eviction_class,
                                      rungs_used=int(jc.get("rebid_rungs", 0) or 0),
                                      ceiling=dec.ceiling, last_bid=jc.get("last_bid"),
                                      market_min_bid=market, on_demand=on_demand)
            journal._job_ladder_journal(jc, "jobs_rebid_refused", iid=str(iid),
                                        reason=dec.reason, eviction_class=eviction_class,
                                        rungs_used=int(jc.get("rebid_rungs", 0) or 0),
                                        ceiling=dec.ceiling, last_bid=jc.get("last_bid"),
                                        market_min_bid=market, on_demand=on_demand,
                                        p_alt=_p_alt, remaining_h=_remaining_h,
                                        defend=_defend, prior_runtime_h=_prior_h,
                                        note="the warm box cannot be bought back inside "
                                             "the ceiling — falling through to the "
                                             "replacement rung")
            print(f"!! re-bid ladder REFUSED: {dec.reason}")
        return False
    if getattr(a, "dry_run", False) or jc.get("dry_run"):
        print(f"[dry-run] would re-bid {iid} to ${dec.price} ({dec.reason})")
        return False
    ok, err = lifecycle._put_bid_soft(iid, dec.price)
    if not ok:                                    # 429/transient -> retry next tick
        jc["last_error"] = err
        print(f"!! re-bid PUT failed ({err}) — retrying on the next tick")
        return True                               # keep the box; do NOT replace yet
    old = jc.get("last_bid")
    jc["last_bid"], jc["last_bid_put"] = dec.price, now
    jc["rebid_rungs"] = int(jc.get("rebid_rungs", 0) or 0) + 1
    jc["rebid_refused"] = None
    jc["rescue_deadline"] = now + _rebid_knob(jc, "rebid_wait_s",
                                              bidpolicy.REBID_WAIT_S)
    print(f">> RE-BID {iid}: ${old} -> ${dec.price} ({dec.reason})")
    journal._job_handoff_emit(jc, "rebid_ladder", old=old, new=dec.price,
                              reason=dec.reason, ceiling=dec.ceiling,
                              rungs_used=jc["rebid_rungs"], rungs_left=dec.rungs_left,
                              market_min_bid=market, on_demand=on_demand,
                              eviction_class=eviction_class)
    # Raising a standing bid IS moving money — three rungs walked 47214941 from
    # $2.134 to $3.216 on 2026-08-08 and `fleet log` recorded none of it.
    journal._job_ladder_journal(jc, "jobs_rebid_rung", iid=str(iid), old_bid=old,
                                new_bid=dec.price, reason=dec.reason,
                                ceiling=dec.ceiling, rungs_used=jc["rebid_rungs"],
                                rungs_left=dec.rungs_left, market_min_bid=market,
                                on_demand=on_demand, eviction_class=eviction_class,
                                p_alt=_p_alt, remaining_h=_remaining_h,
                                defend=_defend, prior_runtime_h=_prior_h,
                                note="keeping the WARM box rather than paying a "
                                     "replacement's setup; the rung waits "
                                     "rebid_wait_s before the next one")
    return True


#: How long an observed qualifying-offer price is market EVIDENCE rather than a
#: memory. Matched to `P_ALT_MAX_AGE_S` on purpose — the two are the same kind
#: of reading and a ceiling that outlived one but not the other would be priced
#: against a market half of it can no longer see.
MARKET_FLOOR_MAX_AGE_S = P_ALT_MAX_AGE_S


def _job_market_floor_fresh(jc: MutableMapping[str, Any],
                            now: float) -> float | None:
    """The price a qualifying replacement offer was last observed to bill, while
    still fresh. Written by `_launch_job_replacement`'s `over_ceiling` refusal —
    a refusal IS a market read, and it is the only one on the pull-reschedule
    lane, whose ceiling-filtered search reports an unaffordable market as an
    empty one (29 of the 36 refusals on 2026-08-24 read `no_offer`)."""
    p, ts = models._num_dph(jc.get("replacement_market_floor")), \
        jc.get("replacement_market_floor_ts")
    if p is None or p <= 0 or ts is None:
        return None
    return None if now - float(ts) > MARKET_FLOOR_MAX_AGE_S else p


# moved-from: herdd._job_replacement_ceiling
def _job_replacement_ceiling(jc: MutableMapping[str, Any]) -> float | None:
    """Price ceiling for ANY autonomous replacement rental on this watch. Base:
    `replace_ceiling_mult` x the ORIGINAL launch dph (never the current box's —
    three replacements at 2x each would license an 8x box). None when no launch
    anchor was ever observed: the eviction ladder refuses outright in that case,
    and the pull-condemn lane keeps its pre-2026-08-05 behavior (it has its own
    budget + retry rails) rather than inventing a ceiling out of nothing.

    Shared by the decision probe and the launch path on purpose (doc 50 R3): the
    two used to derive it independently, and the launcher's copy was `None`.

    Since 2026-08-24 the base is RE-PRICED against live market evidence when —
    and only when — the market has provably outrun it (`bidpolicy.
    replacement_ceiling`; incident REPLACEMENT_CEILING_WEDGE_2026-08-24.md). The
    anchor itself is still never rewritten and every escalation is computed from
    it, so the 8x-compounding argument the immutability was written for is
    untouched. A re-derivation that MOVES the ceiling is journaled once per
    change — an autonomous spend bound that widens silently is not a bound."""
    anchor = models._num_dph(jc.get("launch_dph_anchor"))
    mult = _job_replacement_knob(jc, "replace_ceiling_mult",
                                 bidpolicy.REPLACE_CEILING_MULT)
    now = jc.get("now") or time.time()
    di = _job_defense_inputs(jc, now)
    a = jc.get("a")
    budget = getattr(a, "budget", None) if a is not None else None
    dec = bidpolicy.replacement_ceiling(     # type: ignore[no-untyped-call]
        launch_dph_anchor=anchor, ceiling_mult=mult,
        market_floor=_job_market_floor_fresh(jc, now),
        p_alt=di["p_alt"],
        # The sticky clamp is the price of the machine we HELD, and it is popped
        # on a box swap — so it is often absent here. That is the safe direction:
        # a missing bound drops out of the `min`, and the escalation cap and the
        # budget bound still hold. When it IS present it can only tighten.
        on_demand=jc.get("on_demand_last"),
        budget_left=(round(float(budget) - float(jc.get("spend_usd", 0.0) or 0.0), 4)
                     if budget is not None else None),
        horizon_h=di["remaining_h"],
        escalation_cap_mult=_job_replacement_knob(
            jc, "replace_escalation_cap_mult",
            bidpolicy.REPLACE_ESCALATION_CAP_MULT))
    if dec.price is not None and dec.price != jc.get("replacement_ceiling_last"):
        jc["replacement_ceiling_last"] = dec.price
        if dec.escalated or dec.market_ref is not None:
            journal._job_ladder_journal(
                jc, "jobs_replacement_ceiling_repriced", iid=str(jc.get("iid")),
                ceiling=dec.price, base_ceiling=dec.base,
                launch_dph_anchor=anchor, market_ref=dec.market_ref,
                market_source=dec.source, escalated=bool(dec.escalated),
                bound=dec.bound, budget_usd=budget,
                spend_usd=round(jc.get("spend_usd", 0.0) or 0.0, 4),
                note="the replacement ceiling was RE-DERIVED against live market "
                     "evidence: the launch anchor is unchanged and every "
                     "escalation is computed from it, so N swaps cannot compound"
                     if dec.escalated else
                     "the market moved above the base ceiling and the "
                     "escalation was REFUSED by a tighter bound — replacements "
                     "stay bounded at the base")
    # `bidpolicy` is untyped (Zone S), so the namedtuple field reads as Any.
    return models._num_dph(dec.price)


def _job_note_replacement_refusal(jc: MutableMapping[str, Any], reason: str,
                                  ceiling: float | None) -> None:
    """Count CONSECUTIVE failed LAUNCH ATTEMPTS on an autonomous-replacement
    lane, and remember the market gap standing when the streak began.

    A single refusal is the ladder working: it names its bound, keeps the watch,
    and retries. A refusal that repeats every tick with the market unchanged is
    a WEDGE, and it had no representation anywhere — `BOOT_MAX_HOST_RETRIES`
    counts SUCCESSFUL relaunches (`pull_relaunches`, incremented only past the
    launch), so a lane that can never launch never trips it. This counter is
    what `fleetd`'s derived `replacement_wedged` alarm reads; it is durable
    (REPLACEMENT_STATE_KEYS) because a restart that reset it would re-arm the
    silence the 2026-08-24 wedge ran in for 33 minutes.

    SCOPE, and it is the whole value of the alarm: only a LAUNCH that was
    attempted and failed counts. A `replacement_decision` that declines to rent
    is not a wedge — it escalates to `unrecoverable` and `rescue_stalled` says
    which bound stopped the spend — and counting it would put a second alarm on
    every budget-exhausted watch, i.e. fire on the normal workflow."""
    n = int(jc.get("replacement_refusals", 0) or 0) + 1
    jc["replacement_refusals"] = n
    if n == 1:
        jc["replacement_refusals_since"] = jc.get("now") or time.time()
    jc["replacement_refusal_reason"] = reason
    jc["replacement_refusal_ceiling"] = ceiling


def _job_clear_replacement_refusals(jc: MutableMapping[str, Any]) -> None:
    """Any autonomous rental that actually happened ends the streak — the lane
    is not wedged, whatever it refused on the way here."""
    for k in ("replacement_refusals", "replacement_refusals_since",
              "replacement_refusal_reason", "replacement_refusal_ceiling"):
        jc.pop(k, None)


# moved-from: herdd._job_replacement_fast_deaths
def _job_replacement_fast_deaths(jc: MutableMapping[str, Any],
                                 now: float) -> int:
    """How many replacements this watch rented died inside SPOT_FASTDEATH_S.
    A hostile market kills replacements fast; an unlucky one does not. Only
    `bid` rentals count — an on-demand box that dies early is a host failure,
    not evidence about the spot market, and must not push the ladder further
    toward on-demand on reasoning it did not earn."""
    n = 0
    for r in jc.get("replacement_history") or []:
        if r.get("rental") != "bid":
            continue
        died = r.get("died_ts")
        if died is None:
            continue
        if died - r.get("ts", died) < bidpolicy.SPOT_FASTDEATH_S:
            n += 1
    return n


# moved-from: herdd._job_observed_lifetime_h
def _job_observed_lifetime_h(jc: MutableMapping[str, Any]) -> float | None:
    """This lane's OBSERVED spot lifetime in hours — the `expected_lifetime_h`
    input to `bidpolicy.spot_breakeven`, wired 2026-08-08.

    Built from `replacement_history` alone, and only from `bid` rentals that
    have already died: an on-demand box's lifetime says nothing about the spot
    market, and a box still running has no lifetime yet (counting it would read
    as a very short one and bias the ladder toward the expensive rung — the
    failure direction that costs money).

    The MEDIAN, not the mean: the distribution we actually observe is a couple
    of ~2-minute deaths mixed with multi-hour survivals (46934302 11 min,
    46935445 <1 min, 46936034 2h29m on one JOB_ID), and one outlier either way
    should not decide a rental. None until at least one bid replacement has
    died — an unknown lifetime never fires the trigger."""
    lives = []
    for r in jc.get("replacement_history") or []:
        if r.get("rental") != "bid":
            continue
        died, born = r.get("died_ts"), r.get("ts")
        if died is None or born is None or died <= born:
            continue
        lives.append((died - born) / 3600.0)
    return round(statistics.median(lives), 4) if lives else None


# moved-from: herdd._job_note_replacement_death
def _job_note_replacement_death(jc: MutableMapping[str, Any],
                                now: float) -> None:
    """Stamp `died_ts` on the most recent replacement that has not got one yet.
    Called from the eviction path, so 'this replacement is gone' is recorded at
    the moment we notice, which is what the fast-death window measures."""
    for r in reversed(jc.get("replacement_history") or []):
        if r.get("died_ts") is None:
            r["died_ts"] = now
            return


# --------------------------------------------------------------------------- #
# replacement SELECTION — minimum requirements, then tokens per dollar
#
# Owner ruling 2026-08-16 (incident doc:
# docs/plans/vast-fleet-sessions/2026-08-16-replacement-probe-tpd/SESSION.md):
# "replacement selection = minimum requirements + best tokens-per-dollar."
# Filter on HARD requirements — per-card VRAM, num_gpus, the cuda floor, the
# inet floor, the price ceiling — then rank the survivors by measured training
# tok/s per effective $/hr. Upgrading to a bigger/faster card is CORRECT when it
# is the better deal: finishing sooner is less eviction exposure.
#
# What this replaced: a single `pick_cheapest_offer` on the primary's exact vast
# gpu_name. On 2026-08-16 that returned one H100 NVL whose bid floor sat at
# 91.5% of its own on-demand price; `bid_decision`'s cushion rail correctly
# refused to hold THAT machine on spot, the rung read the refusal as "spot is
# unsafe", and the ladder rented on-demand at $1.603/hr with a $0.40 H200 NVL
# spot offer unqueried on the same market. The rail was right; the candidate set
# was a sample of one. The 0.75 ceiling constant is deliberately untouched
# (AUTOBID_RECALIBRATION_2026-08-09 item A) — the lever is the candidate class.
# --------------------------------------------------------------------------- #
# moved-from: herdd.ReplacementCandidate
# `rate_source` defaults so every pre-existing 6-arg construction still builds:
# it is a PROVENANCE label for the journal (which numbers picked the box), never
# an input to the ranking, and it is uniform across a candidate set by
# construction — see `_rank_rates`.
ReplacementCandidate = namedtuple(
    "ReplacementCandidate", "offer price ondemand rate tpd reason rate_source",
    defaults=(None,))


# How many of the (cheapest-first) candidates get a REAL per-machine on-demand
# probe. Each probe is one bundles POST and the eviction decision re-runs every
# supervise tick while an eviction is stuck, so this is a rate bound, not a
# quality one — offers are deduped by machine_id first, and a candidate past the
# budget is DROPPED rather than priced off a number we did not read.
# moved-from: herdd.REPLACEMENT_OD_PROBES
REPLACEMENT_OD_PROBES = 8


# moved-from: herdd._gpu_rate_soft
def _gpu_rate_soft(gpu_name: str | None,
                   num_gpus: int | None = 1) -> float | None:
    """Expected training throughput (tok/s) for a GPU class, or None.

    Interface — deliberately narrow, imported lazily so this lane works with or
    without the module `gpu_rates` (a sibling of this file, being built in
    parallel with this change, hence the lazy import):

        rate_for(gpu_name, num_gpus=1, shape=None) -> float | None

    None (module absent, class unmeasured, or any error) is a FIRST-CLASS
    answer, never a zero: the caller degrades to cheapest-effective-price
    ranking among minimum-requirements survivors, which is by itself the fix
    for the 2026-08-16 incident. A rate we have not measured must not be
    guessed — an invented tok/s silently re-ranks real money."""
    if not gpu_name:
        return None
    try:
        import gpu_rates
    except Exception:
        return None
    try:
        r = gpu_rates.rate_for(gpu_name, num_gpus=num_gpus or 1)
    except Exception:
        return None
    try:
        r = float(r)  # type: ignore[arg-type]  # guarded
    except (TypeError, ValueError):
        return None
    return r if r > 0 else None


def _train_rate_soft(family: _Family | None, gpu_name: str | None,
                     num_gpus: int | None = 1,
                     gpu_ram_gb: float | None = None) -> _RateEstimate | None:
    """`train_rates.RateEstimate` for this (job family, card, count, VRAM), or
    None — the JOB-SHAPED rate, as opposed to `_gpu_rate_soft`'s card-class one.

    Lazy and total for `_gpu_rate_soft`'s reason, which binds harder here:
    `train_rates` is a sibling of this package rather than a dependency of it, so
    an ImportError must degrade the ranking and never break a replacement."""
    if family is None or not gpu_name:
        return None
    try:
        import train_rates
    except Exception:
        return None
    try:
        est = train_rates.rate_for_offer(family, gpu_name, num_gpus or 1,
                                         gpu_ram_gb)
        return est if est is not None and float(est.tok_s) > 0 else None
    except Exception:
        return None


def _train_family_soft(env: Mapping[str, Any] | None, assets: object = None,
                       world_size: int | None = 1) -> _Family | None:
    """`train_rates.Family` for a jobs-v2 env block, or None. `family_from_env`
    already never raises on an unmappable env (an eval, a generation sweep, a
    probe); this adds the import guard."""
    try:
        import train_rates
    except Exception:
        return None
    try:
        return train_rates.family_from_env(dict(env or {}),
                                           world_size=int(world_size or 1),
                                           assets=assets)
    except Exception:
        return None


def _supervised_job_config(jctx: MutableMapping[str, Any]) -> dict[str, Any] | None:
    """The jobs-v2 CONFIG of the work this box is running, or None.

    The env that fixes a training job's shape (BASE_SLUG / MAX_SEQ / BATCH /
    GRAD_ACCUM / …) is a BUNDLE env that jobd exports box-side. It is never in
    the instance record's `extra_env`, so the primary's shape cannot answer it —
    but the queue TICKET carries the canonical config (`jobmeta.make_ticket`),
    and this lane already reads tickets off that queue
    (`_retarget_pending_tickets`). No new plumbing, one small object.

    Cached on the watch: a stuck eviction re-runs this ~every 50 s and the read
    is a B2 `cat`. Keyed on the job id so a moved queue re-reads. The MISS is
    cached too — most misses are permanent (no ticket, an eval bundle), and the
    cost of caching a transient one is a ranking that stays at today's accuracy
    for the rest of the watch, never a refusal.

    Serve boxes have no queue and no training shape — None, and the ladder keeps
    its card-class rate exactly as today."""
    if jctx.get("serve_mode"):
        return None
    views = [v for v in (jctx.get("pending_views") or []) if v.get("job_id")]
    if not views:
        return None
    # The RUNNING ticket is the one whose throughput a replacement buys back;
    # otherwise the head of the queue, which is what the new box starts on.
    view = next((v for v in views
                 if str(v.get("status") or "") == "running"), views[0])
    jid = str(view["job_id"])
    cached = jctx.get("_job_cfg_cache")
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == jid:
        return cached[1] if isinstance(cached[1], dict) else None
    cfg = None
    try:
        ticket = jobmeta.read_ticket(str(jctx.get("iid")), jid)  # type: ignore[no-untyped-call]
        if isinstance(ticket, dict) and isinstance(ticket.get("config"), dict):
            cfg = ticket["config"]
    except Exception:
        cfg = None
    jctx["_job_cfg_cache"] = (jid, cfg)
    return cfg


def _job_train_family(jctx: MutableMapping[str, Any],
                      world_size: int | None = 1) -> _Family | None:
    """The supervised job's `train_rates.Family`, or None when this watch is not
    running an identifiable training shape. `world_size` is the card count being
    ranked — `eff_batch` includes it, so a family built at the wrong count names
    a job nobody ran."""
    cfg = _supervised_job_config(jctx)
    if not cfg:
        return None
    return _train_family_soft(cfg.get("env"), cfg.get("assets"), world_size)


def _rank_rates(offers: Sequence[Mapping[str, Any]] | None,
                num_gpus: int | None = 1,
                family: _Family | None = None,
                ) -> tuple[list[float | None], str | None]:
    """`(per-offer tok/s, rate_source)` for a candidate SET.

    Set-level and not per-offer, which is the same rule `_replacement_rank`
    keeps one level up: a family-specific rate for one card ranked against
    `gpu_rates`' DEFAULT_SHAPE for another compares two different training jobs,
    and the mismatch always favours whichever card happens to have an anchor at
    the job's own shape. So the job-shaped source is taken only when it answers
    for EVERY candidate; otherwise the whole set falls back to today's path,
    which is byte-identical to pre-2026-08-28 whenever no family is derivable.

    `rate_source` is journalled, never ranked on. A `train_rates` tier is the
    WORST tier in the set for the same comparability reason — one provisional
    (stale, therefore a floor) anchor makes the whole comparison a floor."""
    rows = list(offers or [])
    if family is not None and rows:
        ok = [e for e in (_train_rate_soft(family, o.get("gpu_name"), num_gpus,
                                           models._gpu_ram_gb(o.get("gpu_ram")))
                          for o in rows) if e is not None]
        if len(ok) == len(rows):
            tier = ("measured" if all(e.tier == "measured" for e in ok)
                    else "provisional")
            return [float(e.tok_s) for e in ok], f"train_rates:{tier}"
    rates = [_gpu_rate_soft(o.get("gpu_name"), num_gpus) for o in rows]
    return rates, ("gpu_rates" if any(r is not None for r in rates) else None)


# moved-from: herdd._replacement_rank
def _replacement_rank(
        cands: Sequence[ReplacementCandidate]) -> list[ReplacementCandidate]:
    """Candidates best-first: tokens-per-dollar DESC when every one of them has
    a measured rate, cheapest effective price ASC otherwise.

    All-or-nothing on purpose. Mixing a measured tok/s against an assumed one
    ranks money on an assumption, and the assumption always favours whichever
    class we happened to have measured; falling back for the whole set keeps the
    comparison honest and reproduces the pre-2026-08-16 pick exactly.

    HOST REPUTATION (2026-08-20) enters as a multiplier on the SORT KEY and
    nowhere else. `pick_offers` already handed this list back reputation-ordered,
    but both branches below re-sort and would throw that away — and the tok/s
    branch has to, since a rate table can reorder anything. Applied to the key
    rather than to `c.price` on purpose: `price` is the effective price the
    structural bid rail reasons about (`bidpolicy.replacement_decision`), and
    inflating it would make an advisory score change what we are willing to PAY.
    A penalized host still wins when it is the only affordable one."""
    def _p(c: ReplacementCandidate) -> float:
        try:
            return hostrep.penalty((c.offer or {}).get("machine_id"), verd=_verd)
        except Exception:
            return 1.0
    try:
        _verd = hostrep.verdicts()
    except Exception:
        _verd = {}
    if cands and all(c.rate is not None for c in cands):
        return sorted(cands, key=lambda c: (-((c.tpd or 0.0) / _p(c)),
                                            c.price * _p(c)))
    return sorted(cands, key=lambda c: c.price * _p(c))


# moved-from: herdd._replacement_fit
def _replacement_fit(offers: Sequence[Mapping[str, Any]] | None,
                     num_gpus: int | None) -> list[Mapping[str, Any]]:
    """Candidates whose GPU COUNT matches the primary's. The bundles query asks
    `num_gpus >= n` (unchanged), so a bigger chunk can come back and would bill
    for cards the job cannot use. When the exact-size set is empty the full list
    stands: a differently-sized box beats no box, and the ceiling still binds
    the price."""
    if not offers or not num_gpus:
        return list(offers or [])
    exact = [o for o in offers if o.get("num_gpus") == num_gpus]
    return exact or list(offers)


# moved-from: herdd._replacement_spot_walk
def _replacement_spot_walk(offers: Sequence[Mapping[str, Any]] | None,
                           ceiling: float | None, num_gpus: int | None,
                           max_probes: int = REPLACEMENT_OD_PROBES,
                           family: _Family | None = None,
                           ) -> list[ReplacementCandidate]:
    """Evaluate the SPOT candidate set, cheapest-first: one `ReplacementCandidate`
    per offer we could price, with `price=None` on a candidate the structural
    safety rail vetoed (so the caller can still journal WHY the cheapest one was
    refused). Offers must arrive cheapest-first; the return keeps that order.

    DOC-50 GUARD, and the reason this walk exists as its own function: the
    on-demand reference for a candidate is read from the ON-DEMAND MARKET via
    `_market_ondemand_soft(machine_id, num_gpus)` — NEVER from the bid offer's
    own `dph_total`, which on a bid-type vast offer is the current interruptible
    price (~min_bid + 0.5%). Substituting it makes every candidate look like a
    machine whose on-demand rate sits a tenth of a cent over its spot floor,
    which is `thin` by construction and cost $3.4741/hr on 2026-08-05
    (bidpolicy.replacement_decision docstring; doc 50).

    Probes are deduped per machine and bounded by `max_probes`.

    `family` (2026-08-28) is the supervised job's `train_rates.Family` when one
    is derivable; rates then come from anchors at THIS job's shape instead of
    `gpu_rates`' DEFAULT_SHAPE. Set-level, so the comparison stays honest —
    `_rank_rates`. None reproduces the previous behaviour exactly."""
    priced: list[tuple[Mapping[str, Any], Any, float | None]]
    seen: dict[Any, float | None]
    priced, seen, probes = [], {}, 0
    for o in offers or []:
        floor = models._num_dph(o.get("min_bid"))
        if floor is None or floor <= 0:
            continue
        mid = o.get("machine_id")
        if mid in seen:
            od = seen[mid]
        else:
            if probes >= max_probes:
                break            # bounded: drop the tail, never guess its price
            probes += 1
            od = pricing._market_ondemand_soft(mid, num_gpus)
            seen[mid] = od
        dec = bidpolicy.bid_decision(floor, ceiling, od)  # type: ignore[no-untyped-call]
        priced.append((o, dec, od))
    rates, src = _rank_rates([o for o, _d, _od in priced], num_gpus, family)
    out: list[ReplacementCandidate] = []
    for (o, dec, od), rate in zip(priced, rates):
        out.append(ReplacementCandidate(
            o, dec.price, od, rate,
            (rate / dec.price) if (rate and dec.price) else None, dec.reason,
            src))
    return out


# moved-from: herdd._offer_disk_gb
def _offer_disk_gb(offer: Mapping[str, Any] | None) -> float | None:
    """An offer's advertised container disk in GB, or None. Vast reports
    `disk_space` in GB on a bundle (unlike `gpu_ram`, which is MiB) — the two
    units live one field apart, which is why this is a named reader."""
    try:
        v = float((offer or {}).get("disk_space"))  # type: ignore[arg-type]  # guarded
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


# moved-from: herdd._replacement_disk_shortfall
def _replacement_disk_shortfall(jctx: MutableMapping[str, Any],
                                exclude_machines: Iterable[object] | None,
                                ceiling: float | None, cuda: float | None,
                                need: float | None) -> str | None:
    """A one-line explanation when the DISK FLOOR is what emptied the
    replacement market, or None when it is not.

    Costs one extra bundles read and runs ONLY on the path that is already
    about to refuse — never on a path that can rent. The question it answers is
    the one the operator otherwise cannot: `no_market_read` reads identically
    whether the market is empty, unaffordable, or full of machines too small to
    hold the job. On 2026-08-16 it was the third one every time, and nothing in
    the record said so.

    Deliberately probes the SPOT book unfloored: if a box exists at all it
    shows up there first, and the caller only needs to know that price was not
    the binding constraint."""
    if not need:
        return None
    try:
        loose = _job_replacement_offers(jctx, exclude_machines, rental="bid",
                                        max_dph=ceiling, cuda=cuda, limit=4,
                                        disk_gb=0)
    except Exception:
        return None
    sizes: list[tuple[Any, Any]]
    sizes = [(_offer_disk_gb(o), o) for o in loose or []]
    sizes = [(g, o) for g, o in sizes if g is not None]
    if not sizes:
        return None                      # the market was empty for other reasons
    best_gb, best = max(sizes, key=lambda t: t[0])
    if best_gb >= need:
        return None                      # disk was not the bound; price/rail was
    return (f"DISK FLOOR: no offer in class carries the {need:g}G of container "
            f"disk this workload needs — the biggest of {len(sizes)} affordable "
            f"candidate(s) has {best_gb:g}G (machine {best.get('machine_id')}, "
            f"{best.get('gpu_name')}). Renting one anyway is how boxes 47845159 "
            f"(23G) and 47845212 (47G) were minted on 2026-08-16; raise nothing "
            f"— a bigger bid buys no more disk. Rehost by hand on a host with "
            f"the space, or shrink the job's --disk")


# moved-from: herdd._replacement_ondemand_walk
def _replacement_ondemand_walk(offers: Sequence[Mapping[str, Any]] | None,
                               num_gpus: int | None,
                               family: _Family | None = None,
                               ) -> list[ReplacementCandidate]:
    """The ON-DEMAND rung's candidate set. `dph_total` on an ONDEMAND-type offer
    IS the on-demand price — the doc-50 ban is on reading it off a BID-type
    offer — so no per-machine probe is needed here. `family` as in
    `_replacement_spot_walk`."""
    priced = [(o, dph) for o, dph in
              ((o, models._num_dph(o.get("dph_total"))) for o in offers or [])
              if dph is not None and dph > 0]
    rates, src = _rank_rates([o for o, _ in priced], num_gpus, family)
    return [ReplacementCandidate(o, dph, dph, rate,
                                 (rate / dph) if rate else None, None, src)
            for (o, dph), rate in zip(priced, rates)]


# How long an eviction excludes the machine we lost, BY CLASS. 30 minutes for
# the two classes that describe a market state rather than a broken host:
#
#   * `outbid` — the floor that displaced us is a price, and prices come back
#     down. On 2026-08-16 the lost machine's floor never moved at all ($0.80 all
#     night); only its listing blinked. A permanent exclusion there removes the
#     one machine we hold the most information about from every later probe.
#   * `host_stop` — the host stopped the chunk; ONE of those is a minute-scale
#     event, so the first one ages out here. It is host evidence all the same:
#     it earns a durable strike, and a repeat escalates this TTL (below).
#
# `host_failure` and `ondemand_displaced` stay excluded for the life of the
# watch: a broken host stays broken, and an on-demand renter sitting on the
# GPUs holds them until they leave. 30 min is one replacement's setup cost
# (~12 min) with margin — long enough that a re-probe cannot immediately
# re-rent into the same live contest, short enough to matter within a watch.
# moved-from: herdd.EVICTED_EXCLUSION_TTL_S
EVICTED_EXCLUSION_TTL_S = 1800.0


# A REPEAT `host_stop` from one machine inside a watch is no longer a
# minute-scale blip, so its exclusion stops expiring on the same clock: the TTL
# multiplies per extra stop, bounded so even a serial stopper returns to the
# market the same day. The first stop keeps the 30 min above — one stop is
# enough to move off the host, not enough to write it off.
HOST_STOP_TTL_ESCALATION = 4.0
HOST_STOP_TTL_MAX_S = 21600.0            # 6 h


# moved-from: herdd.EVICTED_TTL_CLASSES
EVICTED_TTL_CLASSES = (bidpolicy.EVICTION_OUTBID, bidpolicy.EVICTION_HOST_STOP)


# The classes that are NOT evidence about the host, so may not become a durable
# strike. `outbid` is a price we lost, and charging it would blacklist whichever
# machines sit in a contested band — i.e. punish a host for being popular.
# `no_credit` NAMES our own account as the cause, so a strike written under it
# is a contradiction in one line.
#
# `host_stop` is NOT in here even though it shares the TTL set: its own
# classification is "box present, chunk still listed and rentable, our bid still
# clears the floor" — nobody took the box, the host stopped it. That is a fact
# about the host, and it was the one eviction class fleetd could never remember
# across watches.
STRIKE_FREE_EVICTION_CLASSES = (bidpolicy.EVICTION_OUTBID,
                                bidpolicy.EVICTION_NO_CREDIT)


# Which durable strike kind an eviction class earns. Anything not named here
# that is also not strike-free reads as a generic host failure.
EVICTION_STRIKE_KINDS: dict[str | None, str] = {
    bidpolicy.EVICTION_HOST_STOP: "host_stop"}


# moved-from: herdd._job_note_evicted_machine
def _job_note_evicted_machine(jc: MutableMapping[str, Any], machine: object,
                              eviction_class: str | None, now: float) -> None:
    """Record the machine we were just evicted from, with the timestamp and the
    class the TTL is keyed on. Writes BOTH the legacy `evicted_machines` set
    (every existing reader, and the journal field) and the
    `evicted_machine_ts` sidecar; the sidecar is what makes the exclusion
    expire, and its absence means "permanent", so a watch restored from a
    pre-2026-08-16 state.json degrades to the old always-permanent behaviour
    instead of silently un-excluding a broken host.

    `host_stops` on the sidecar counts this machine's host stops within the
    watch, which is what escalates the exclusion TTL (`_evicted_ttl_s`). It is
    carried across a later eviction of another class so a host cannot launder
    its count through one outbid; absent means "one or none", so a restored
    pre-existing record reads as the old flat TTL."""
    if machine is None:
        return
    jc.setdefault("evicted_machines", set()).add(machine)
    meta = jc.setdefault("evicted_machine_ts", {})
    prev = meta.get(str(machine))
    stops = _int0(prev.get("host_stops") if isinstance(prev, Mapping) else None)
    if eviction_class == bidpolicy.EVICTION_HOST_STOP:
        stops += 1
    meta[str(machine)] = {"ts": float(now), "class": eviction_class,
                          **({"host_stops": stops} if stops else {})}
    # A DURABLE strike only for the classes that are evidence about the HOST —
    # see STRIKE_FREE_EVICTION_CLASSES for which those are and why.
    # `core.acctfault`'s latch also refuses this write while a refusal is fresh,
    # but the latch is 15 minutes and the credit signal that produces
    # `no_credit` is stored — a re-classification after the latch expires would
    # land the strike anyway.
    if eviction_class not in STRIKE_FREE_EVICTION_CLASSES:
        hostrep.note_strike(machine,
                            EVICTION_STRIKE_KINDS.get(eviction_class,
                                                      "host_failure"),
                            now=now, note=f"eviction class {eviction_class}")


def _int0(v: object) -> int:
    """A counter read off restored JSON: anything unreadable is zero, because a
    malformed sidecar must not take a probe down."""
    if not isinstance(v, (int, float, str)):
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _evicted_ttl_s(rec: Mapping[str, Any], base: float) -> float:
    """How long THIS eviction record excludes its machine. Flat `base` except
    for a repeated `host_stop`, where it grows by HOST_STOP_TTL_ESCALATION per
    extra stop up to HOST_STOP_TTL_MAX_S — a host that stopped us twice will
    stop the replacement too, and price alone keeps sending the fleet back."""
    stops = _int0(rec.get("host_stops"))
    if rec.get("class") != bidpolicy.EVICTION_HOST_STOP or stops <= 1:
        return base
    # `max(base, ...)`: the cap may never shorten an operator's own TTL.
    return max(base, min(base * (HOST_STOP_TTL_ESCALATION ** (stops - 1)),
                         HOST_STOP_TTL_MAX_S))


# moved-from: herdd._job_excluded_machines
def _job_excluded_machines(jc: MutableMapping[str, Any],
                           now: float | None = None) -> set[Any]:  # noqa: ANN401 — vast machine ids
    """The machine ids a replacement probe must not consider: the pull-bad set
    (always permanent — that host failed to pull our image) plus the evicted set
    minus whatever has aged past its class TTL (see EVICTED_EXCLUSION_TTL_S,
    and `_evicted_ttl_s` for the repeat-`host_stop` escalation).

    Read-time filtering, not pruning: `evicted_machines` stays the historical
    record the journal prints, and an expired entry that gets re-evicted simply
    re-stamps its timestamp.

    Since 2026-08-20 the DURABLE block list is unioned in, so a probe on a fresh
    watch still refuses a host an earlier session condemned. Included here — and
    not left to `pick_offers`, which filters too — because this set is also what
    the ladder JOURNALS as `excluded_machines`, and an exclusion the operator
    cannot see in the journal is one they will re-litigate by hand."""
    now = time.time() if now is None else now
    ttl = _rebid_knob(jc, "evicted_ttl_s", EVICTED_EXCLUSION_TTL_S)
    meta = jc.get("evicted_machine_ts") or {}
    out = set(jc.get("pull_bad_machines") or set())
    try:
        out |= hostrep.blocked_machines(now)
    except Exception:
        pass                          # advisory layer: never break a probe
    for m in (jc.get("evicted_machines") or set()):
        rec = meta.get(str(m))
        if isinstance(rec, dict) and rec.get("class") in EVICTED_TTL_CLASSES:
            ts = rec.get("ts")
            try:
                aged = (ts is not None
                        and (now - float(ts)) >= _evicted_ttl_s(rec, float(ttl)))
            except (TypeError, ValueError):
                aged = False
            if aged:
                continue
        out.add(m)
    return out


# moved-from: herdd._job_eviction_replace
def _job_eviction_replace(jc: MutableMapping[str, Any],
                          hf: MutableMapping[str, Any] | None,
                          eviction_class: str, why: object,
                          exclusion_class: str | None = None) -> bool:
    """Rent a REPLACEMENT box for an evicted jobs/serve primary and move the
    queue onto it. Returns True when the watch now supervises a new box (caller
    keeps supervising), False when the ladder refused — in which case the caller
    falls through to its existing `unrecoverable` verdict, unchanged.

    Order is the pull-condemn order and for the same reason: launch, MOVE THE
    QUEUE, then destroy the old box. A ticket must never exist pointing at a box
    that is already gone (the 46590907 orphan shape). The old box may already
    have left the listing (host failure) — then there is nothing to destroy and
    the retarget is the whole job.

    Refusals are LOUD and recorded on `jc["replacement_refused"]` so fleetd can
    put the arithmetic in its alarm instead of the generic "raise the bid" text
    that sent the v7 operator to do this by hand.

    `exclusion_class` (S2b review round 1, F2/M2) is the class the MACHINE
    EXCLUSION is keyed on, and it is deliberately allowed to differ from
    `eviction_class`. A notification row refines the class we ACT on; it must
    not shorten how long we remember that this machine took our box. The
    exclusion TTL is a memory of a HOST, and a control-plane message about a
    price is not evidence about the host — but `EVICTED_TTL_CLASSES` maps
    `outbid`/`host_stop` to a 30-minute TTL and everything else to permanent, so
    a row that refined `unknown -> outbid` un-excluded the machine at t+30m and
    let the very next replacement probe re-rent the machine we were just
    displaced from. Two of those deaths inside SPOT_FASTDEATH_S flip
    `prefer_od` and the ladder buys an on-demand box (measured 8.3x at
    anchor $2.00). So exclusion reads the BARE classification and is
    byte-identical to pre-S2b in every state. None = "same as
    `eviction_class`", which is every pre-S2b caller and every test."""
    a = jc["a"]
    now = jc.get("now") or time.time()
    old = str(jc["iid"])
    inst = job_lane._job_sup_inst(jc, old) or {}
    machine = inst.get("machine_id")

    _job_note_replacement_death(jc, now)
    used = int(jc.get("replacements", 0))
    max_repl = _job_replacement_knob(jc, "max_replacements",
                                     bidpolicy.MAX_REPLACEMENTS)
    ceil_mult = _job_replacement_knob(jc, "replace_ceiling_mult",
                                      bidpolicy.REPLACE_CEILING_MULT)
    anchor = models._num_dph(jc.get("launch_dph_anchor"))
    fast = _job_replacement_fast_deaths(jc, now)

    # Probe BOTH markets before deciding: the rung choice needs the candidate's
    # numbers, and a ceiling we cannot price against is a refusal, not a guess.
    # The ceiling is pushed into the query (max_dph) so an unaffordable market
    # simply returns nothing.
    ceiling = _job_replacement_ceiling(jc)
    if machine is not None:
        # Exclude the lost machine BEFORE the search, not after the launch: on an
        # outbid it is a contested machine, on a host failure a broken one, and
        # on an on-demand claim the renter is still sitting on it. Re-renting the
        # machine we were just evicted from is the one host choice we have
        # positive evidence against.
        #
        # TTL'd BY CLASS since 2026-08-16 (EVICTED_EXCLUSION_TTL_S): on an
        # `outbid`/`host_stop` that evidence is a market state and it expires;
        # on a host failure / on-demand claim it does not.
        #
        # Keyed on the BARE class, never the notify-refined one — see
        # `exclusion_class` above. A notification is evidence about a PRICE; the
        # exclusion is a memory of a HOST.
        _job_note_evicted_machine(
            jc, machine,
            eviction_class if exclusion_class is None else exclusion_class, now)
    excl = sorted(_job_excluded_machines(jc, now))
    cuda = _replacement_cuda_floor()
    ngpu = (models._job_primary_shape(jc, None) or {}).get("num_gpus") or 1
    # WHAT JOB is being replaced, as a rate key (2026-08-28). Without it the
    # ladder ranked every candidate at `gpu_rates`' DEFAULT_SHAPE — the slowest
    # measured shape for the card, and not necessarily this job's — so a rate
    # that decided real money could belong to a different training run. Built at
    # the primary's card count because `eff_batch` includes world size. None
    # (serve box, eval bundle, unreadable ticket) keeps the card-class path.
    fam = _job_train_family(jc, ngpu)

    # SPOT rung: a CANDIDATE SET, not a sample of one. Fetch the minimum-
    # requirements class cheapest-first, run the per-offer structural rail
    # (`bid_decision`, against each candidate's OWN on-demand price) over it,
    # and rank the survivors by tokens-per-dollar — cheapest effective price
    # when no rate table is installed, which is exactly the old pick.
    #
    # With NO survivor the cheapest candidate is still handed to the decision,
    # so the refusal the operator reads is the real rail arithmetic
    # ("escalate_over_ceiling: surviving this floor needs $X...") rather than a
    # bare "no market read", and the ladder falls through to the on-demand rung
    # exactly as it did before.
    spot_cands = _replacement_spot_walk(
        _replacement_fit(_job_replacement_offers(jc, excl, rental="bid",
                                                 max_dph=ceiling, cuda=cuda),
                         ngpu),
        ceiling, ngpu, family=fam)
    _spot_ranked = _replacement_rank([c for c in spot_cands
                                      if c.price is not None])
    spot_pick = _spot_ranked[0] if _spot_ranked else (
        spot_cands[0] if spot_cands else None)
    spot_offer = spot_pick.offer if spot_pick else None
    # The candidate's OWN machine on-demand rate — the reference the survival
    # rail is about (bidpolicy.replacement_decision `spot_ondemand`), never the
    # bid offer's `dph_total` (doc 50).
    spot_ondemand = spot_pick.ondemand if spot_pick else None
    od_cands = _replacement_rank(_replacement_ondemand_walk(
        _replacement_fit(_job_replacement_offers(jc, excl, rental="ondemand",
                                                 max_dph=ceiling, cuda=cuda),
                         ngpu), ngpu, family=fam))
    od_offer = od_cands[0].offer if od_cands else None
    # The on-demand REFERENCE price must come from the ON-DEMAND market and must
    # be UN-CEILINGED (doc 50 R1/R2). Two distinct failures were being collapsed
    # into one silent guess on 2026-08-05:
    #
    #   * "no on-demand offer UNDER the ceiling" — a real market read that the
    #     ceiling rail exists to refuse loudly ("cheapest on-demand $3.47 > the
    #     $2.164 ceiling"). The ceilinged probe returns None for it, which reads
    #     identically to...
    #   * "no on-demand market at all" — genuinely no price, which must clamp
    #     nothing and license nothing.
    #
    # The old code answered BOTH by substituting the SPOT offer's `dph_total`,
    # which on a bid-type offer is the interruptible price (~min_bid + 0.5%) —
    # so the bid got clamped a tenth of a cent over its own floor, the cushion
    # read 1.001x, `thin` fired, and the ladder bought an on-demand box it had
    # no quote for at $3.4741/hr. Never substitute a bid price for an on-demand
    # one: re-probe the on-demand market WITHOUT the ceiling instead, and let
    # `replacement_decision` refuse it on price.
    od_ref_offer = od_offer or _job_replacement_offer(
        jc, excl, rental="ondemand", cuda=cuda)
    od_reference = models._num_dph((od_ref_offer or {}).get("dph_total"))
    dec = bidpolicy.replacement_decision(  # type: ignore[no-untyped-call]
        eviction_class=eviction_class,
        replacements_used=used,
        budget_usd=a.budget,
        spend_usd=jc.get("spend_usd", 0.0),
        launch_dph_anchor=anchor,
        offer_min_bid=models._num_dph((spot_offer or {}).get("min_bid")),
        offer_ondemand=od_reference,
        # The SURVIVAL rail is about the machine we would bid on, the on-demand
        # rung is about the cheapest box in class; they are different numbers
        # once the candidate class is wider than one SKU (2026-08-16).
        spot_ondemand=spot_ondemand,
        fast_deaths=fast,
        max_replacements=max_repl,
        ceiling_mult=ceil_mult,
        # ONE derivation, not two. The search above already ran `max_dph=ceiling`
        # off `_job_replacement_ceiling`; letting the decision re-derive its own
        # meant they agreed only because the formula was copied — and since
        # 2026-08-24 the derivation reads live market evidence, so a copy would
        # be a probe and a decision priced against different markets.
        ceiling=ceiling,
        ceiling_basis=(f"re-derived replacement ceiling on a ${anchor:.4f} "
                       f"launch anchor"
                       if (ceiling is not None and anchor
                           and abs(ceiling - round(ceil_mult * anchor, 3)) > 1e-9)
                       else None),
        # the livelock trigger (spot_breakeven, wired 2026-08-08): OBSERVED
        # lifetime from this lane's own history, setup cost from config.
        # `observed_lifetime_h` is None until a bid replacement of THIS lane has
        # died, which is why the trigger could never fire on a first eviction —
        # so `prior_lifetime_h` (recalibration 2026-08-09, item B) stands in, and
        # `dec.lifetime_basis` records which of the two the rung ran on.
        observed_lifetime_h=_job_observed_lifetime_h(jc),
        setup_h=_rebid_knob(jc, "spot_setup_h", bidpolicy.SPOT_SETUP_H),
        prior_lifetime_h=_rebid_knob(jc, "spot_prior_lifetime_h",
                                     bidpolicy.SPOT_PRIOR_LIFETIME_H))

    # WHICH candidate set the decision was made over, and on what it was ranked
    # (2026-08-16). "$1.603 on-demand" in a journal is unreadable without the
    # answer to "against how many spot candidates, and why did none of them
    # win?" — that question is what turned tonight's incident into an hour of
    # log archaeology. `ranked_by: price` is the shipped default until
    # `gpu_rates` carries the class; `tokens_per_dollar` means it did.
    # The container-disk floor the candidate class was searched under, and —
    # when both rungs came back empty — whether that floor is what emptied it.
    # A hard fit requirement that appears in NO decision field is exactly how
    # this went unnoticed: every 2026-08-16 record says `spot_candidates: 4`
    # and nothing about the 23 GB the winner had.
    _disk_need = _replacement_disk_need(jc)[0]
    _disk_note = (_replacement_disk_shortfall(jc, excl, ceiling, cuda,
                                              _disk_need)
                  if not spot_cands and not od_cands else None)
    _ranked_by = ("tokens_per_dollar"
                  if (_spot_ranked and all(c.rate is not None
                                           for c in _spot_ranked))
                  else "price")
    _sel = dict(
        disk_floor_gb=_disk_need,
        disk_blocked=bool(_disk_note),
        spot_candidates=len(spot_cands),
        spot_survivors=len(_spot_ranked),
        ranked_by=_ranked_by,
        # WHICH numbers picked the box (2026-08-28). `ranked_by` says a rate
        # ranked it; this says whose — `train_rates:measured|provisional` for
        # anchors at THIS job's shape, `gpu_rates` for the card-class table.
        # Scoped to the spot rung exactly as `ranked_by` is, and None when
        # price ranked, because then no rate picked anything.
        rate_source=(_spot_ranked[0].rate_source
                     if _ranked_by == "tokens_per_dollar" else None),
        spot_gpu=(spot_offer or {}).get("gpu_name"),
        spot_machine=(spot_offer or {}).get("machine_id"),
        spot_ondemand=spot_ondemand,
        ondemand_candidates=len(od_cands),
        ondemand_gpu=(od_offer or {}).get("gpu_name"))

    # EVERY autonomous rental decision is logged with its price math, refusals
    # included — an unauditable spend decision is not a bounded one. But a
    # decision is logged when it is MADE, not every tick it stays true: a
    # stuck eviction re-runs this ~every 50 s, and on 2026-08-10 box 47398836
    # journaled "replacement cap reached (3/3)" 79 times in 66 min. A rent
    # always logs (money moves each time); a REFUSAL logs when its reason
    # string differs from the one already standing on `replacement_refused` —
    # changed numbers are a changed reason and log again. The decision itself
    # is still re-made every tick.
    #
    # A refusal the disk floor caused says so IN THE REASON, next to the price
    # arithmetic and in the same style — `_report_stalled`'s contract is that
    # the alarm names which bound stopped the spend, and "raise the bid" is
    # actively wrong advice when the bound is gigabytes.
    _reason = dec.reason + (f"; {_disk_note}" if _disk_note else "")
    _repeat_refusal = (dec.action != "rent"
                       and jc.get("replacement_refused") == _reason)
    if not _repeat_refusal:
        journal._job_handoff_emit(jc, "eviction_replacement_decision",
                                  eviction_class=eviction_class, why=why,
                                  action=dec.action, rental=dec.rental, price=dec.price,
                                  reason=_reason, ceiling=dec.ceiling,
                                  # PRIOR vs OBSERVED: an escalation made on an assumed
                                  # lifetime must never read like one made on evidence.
                                  lifetime_basis=dec.lifetime_basis,
                                  budget_left=dec.budget_left, budget_usd=a.budget,
                                  spend_usd=round(jc.get("spend_usd", 0.0), 4),
                                  replacements_used=used, max_replacements=max_repl,
                                  fast_deaths=fast, launch_dph_anchor=anchor,
                                  machine_id=machine,
                                  # The two market reads the decision was made on, kept
                                  # SEPARATE and named: `offer_ondemand` is the cheapest
                                  # on-demand price in class (un-ceilinged, None = no
                                  # on-demand market read), `ondemand_under_ceiling` says
                                  # whether any on-demand offer was actually rentable
                                  # within the ceiling. Tonight's record showed neither
                                  # and a spot price labelled "on-demand".
                                  offer_min_bid=models._num_dph(
                                      (spot_offer or {}).get("min_bid")),
                                  offer_ondemand=od_reference,
                                  ondemand_under_ceiling=od_offer is not None,
                                  **_sel)
        # ...and to `fleet log`, which is where an operator actually looks
        # (task #78). B2 box events are durable and invisible: on 2026-08-08
        # the whole decision record existed in `jobs/nodes/<IID>/events/`
        # while the daemon's own journal showed a `tick` and nothing else.
        journal._job_ladder_journal(jc, "eviction_replacement_decision", iid=str(old),
                                    eviction_class=eviction_class, why=why,
                                    action=dec.action, rental=dec.rental,
                                    price=dec.price,
                                    reason=_reason, ceiling=dec.ceiling,
                                    lifetime_basis=dec.lifetime_basis,
                                    budget_left=dec.budget_left, budget_usd=a.budget,
                                    spend_usd=round(jc.get("spend_usd", 0.0), 4),
                                    replacements_used=used, max_replacements=max_repl,
                                    launch_dph_anchor=anchor, machine_id=machine,
                                    offer_min_bid=models._num_dph(
                                        (spot_offer or {}).get("min_bid")),
                                    offer_ondemand=od_reference,
                                    ondemand_under_ceiling=od_offer is not None,
                                    **_sel)
        print(f"!! EVICTED {old}: {why} (class {eviction_class}) — "
              f"replacement: {dec.action.upper()} ({_reason})")
    if dec.action != "rent":
        # DELIBERATELY NOT counted toward the wedge. A decision that declines to
        # rent already escalates: it sets `replacement_refused`, the tick returns
        # `unrecoverable`, and `rescue_stalled` alarms NAMING THE BOUND. The
        # wedge counter exists for the shape with no escalation at all — a lane
        # that keeps ATTEMPTING a launch and keeps being refused — so counting
        # this here would double-alarm every budget-exhausted or cap-reached
        # watch on a state that is already loud.
        jc["replacement_refused"] = _reason
        return False
    if jc.get("dry_run"):
        print(f"[dry-run] would rent a {dec.rental} replacement at "
              f"${dec.price}/hr and move {old}'s queue onto it")
        jc["replacement_refused"] = None
        return False

    # `od_ref_offer` is the fallback ONLY because the market can move between
    # the two probes; it is not a licence to exceed the ceiling — the launch
    # path re-checks `max_dph` against whatever offer it ends up with.
    offer = ((od_offer or od_ref_offer) if dec.rental == "ondemand"
             else spot_offer)
    cid, dph, reason = _launch_job_replacement(jc, excl, offer=offer,
                                               rental=dec.rental,
                                               price=dec.price,
                                               max_dph=ceiling)
    if reason is not None:
        jc["replacement_refused"] = f"{reason}: {jc.get('last_error')}"
        _job_note_replacement_refusal(jc, reason, ceiling)
        journal._job_handoff_emit(jc, "eviction_replacement_failed", reason=reason,
                                  detail=jc.get("last_error"), rental=dec.rental)
        journal._job_ladder_journal(jc, "jobs_box_launch_failed", iid=str(old),
                                    lane="eviction_replacement", reason=reason,
                                    detail=jc.get("last_error"), rental=dec.rental,
                                    price=dec.price, ceiling=ceiling,
                                    refusals=jc.get("replacement_refusals"),
                                    note="the old box and its queue are left exactly as "
                                         "they are; retrying on the next eviction tick")
        print(f"!! eviction replacement launch FAILED ({reason}: "
              f"{jc.get('last_error')}) — the old box and its queue are left "
              f"exactly as they are; retrying on the next eviction tick")
        return False

    # REALIZED price vs decided price. `$None/hr` in the journal while a
    # $3.4741/hr meter runs is not a spend record (doc 50 R4), so
    # `_launch_job_replacement` now always returns a number — and when it
    # differs materially from what the decision priced, that is a market move
    # between decide and launch and it gets its own alarm rather than a quiet
    # field nobody diffs.
    dph = models._num_dph(dph)
    if dph is not None and dec.price is not None and dph > dec.price + 1e-3:
        print(f"!! eviction replacement OVERPRICED: decided ${dec.price:.4f}/hr, "
              f"box {cid} bills ${dph:.4f}/hr (ceiling ${dec.ceiling}) — the "
              f"market moved between the decision and the launch; the box is "
              f"rented and the queue is moving, budget ${a.budget} still caps it")
        journal._job_handoff_emit(jc, "eviction_replacement_overpriced", box=str(cid),
                                  decided_price=dec.price, realized_dph=dph,
                                  ceiling=dec.ceiling, rental=dec.rental)
        journal._job_ladder_journal(jc, "eviction_replacement_overpriced", iid=str(cid),
                                    decided_price=dec.price, realized_dph=dph,
                                    ceiling=dec.ceiling, rental=dec.rental,
                                    note="the market moved between the decision and the "
                                         "launch; the box is rented and the queue is "
                                         "moving, --budget still caps it")
    journal._job_ladder_journal(jc, "jobs_box_launched", iid=str(cid),
                                lane="eviction_replacement", from_box=str(old),
                                rental=dec.rental, dph=dph, decided_price=dec.price,
                                ceiling=dec.ceiling, eviction_class=eviction_class,
                                disk_gb=jc.get("last_replacement_disk_gb"),
                                launch_disk_gb=jc.get("launch_disk_gb"),
                                excluded_machines=excl,
                                budget_usd=a.budget,
                                spend_usd=round(jc.get("spend_usd", 0.0), 4),
                                replacements_used=used + 1,
                                max_replacements=max_repl,
                                note="autonomous rental on the eviction rung")
    moved, failed = _retarget_pending_tickets(old, cid, reason="evicted")
    journal._job_ladder_journal(jc, "jobs_queue_retargeted", iid=str(cid),
                                lane="eviction_replacement", from_box=str(old),
                                to_box=str(cid), moved_jobs=len(moved),
                                failed_moves=len(failed),
                                job_ids=[str(m) for m in moved],
                                note="tickets moved BEFORE the old box is retained or "
                                     "destroyed")
    # The lost box is EVIDENCE, not garbage (owner directive 2026-08-05) — it is
    # retained for a bounded window rather than destroyed. A failure here must
    # never fail the replacement: the queue has already moved, which is the
    # important half.
    try:
        retention._job_retain_or_destroy(jc, old, inst, eviction_class, now, new_iid=cid)
    except Exception as e:
        print(f"!! eviction retention errored ({type(e).__name__}: {e}) — the "
              f"lost box {old} is still listed and still bills storage; "
              f"`herdd reap` will reclaim it (queue already moved to {cid})")
    jc["replacements"] = used + 1
    jc.setdefault("replacement_history", []).append(
        {"iid": str(cid), "from_iid": old, "ts": now, "rental": dec.rental,
         "price": dec.price, "dph": dph, "class": eviction_class,
         "died_ts": None})
    jc["replacement_refused"] = None
    _job_clear_replacement_refusals(jc)
    journal._job_handoff_emit(jc, "eviction_replaced", from_box=old, to_box=str(cid),
                              eviction_class=eviction_class, rental=dec.rental,
                              dph=dph, decided_price=dec.price, ceiling=dec.ceiling,
                              # the SIZING, journaled (task #69): `disk_gb` is what was
                              # rented, `launch_disk_gb` the anchor it inherited from.
                              # A rehost whose size nobody can reconstruct is how a
                              # 110 GB job spent three hours on a 60 GB box.
                              disk_gb=jc.get("last_replacement_disk_gb"),
                              launch_disk_gb=jc.get("launch_disk_gb"),
                              moved_jobs=len(moved),
                              failed_moves=len(failed),
                              replacements_used=jc["replacements"],
                              max_replacements=max_repl)
    print(f">> eviction-replaced: {old} -> {cid} ({dec.rental} @ "
          f"${dph}/hr); {len(moved)} ticket(s) moved"
          + (f", {len(failed)} FAILED (see above)" if failed else "")
          + f"; replacement {jc['replacements']}/{max_repl}")

    # Re-anchor the supervisor on the replacement — the same block the
    # pull-condemn path uses. `launch_dph_anchor` is deliberately NOT reset: the
    # price ceiling must stay derived from the ORIGINAL launch, or three
    # replacements could ratchet it 2x each and buy an 8x box.
    jc["iid"] = str(cid)
    # BID ladder anchors only — an ON-DEMAND box has no standing bid, and
    # writing the list price into `last_bid`/`first_seen_dph` would arm the
    # defend/decay moves (and `classify_eviction`'s "our bid was at on-demand"
    # test) against a box that cannot be outbid. R4 made `dph` a real number for
    # both rungs; the ladder state stays rung-aware, exactly as it was when the
    # on-demand rung simply had no price to write.
    jc["last_bid"] = dph if dec.rental == "bid" else None
    jc["first_seen_dph"] = dph if dec.rental == "bid" else None
    jc["floor_samples"] = []
    jc["decay_streak"] = 0
    jc["decay_streak_since"] = None      # a new box starts a fresh decay dwell
    jc["rescue_deadline"] = None
    jc["not_live"] = 0
    jc["was_live"] = None
    # a fresh box gets a fresh re-bid ladder (per eviction cycle, not per watch)
    jc["rebid_rungs"] = 0
    jc["rebid_refused"] = None
    jc.pop("pull_sampler", None)
    jc.pop("pull_sampler_iid", None)
    # The shared box-swap seam. The sticky on-demand price is per MACHINE, so
    # carrying it over would clamp the new box against the old one's price; the
    # echo window has to go for the mirror-image reason, because a replacement
    # CAN land on the same machine (review 2026-08-10, #4); and the self-floor
    # episode is per BOX (L7/L8) — a suppression latched on the dead box would
    # swallow the replacement's own first alarm.
    ladder_core.box_swap_reset(jc)  # type: ignore[no-untyped-call]
    # ...and so are the lane's own latches: a stale `evicted_announced` made the
    # replacement's first live tick journal "eviction survived; no replacement
    # was rented" seconds after fleetd journaled jobs_replaced for the same
    # watch, and a latched preferred-ceiling / hard-ceiling alarm would swallow
    # the replacement's own.
    jc.pop("evicted_announced", None)
    job_lane._job_evicted_latch_reset(jc)
    job_lane._job_notify_box_swap_reset(jc)
    jc["pref_alarmed"] = False
    jc["ceiling_escalated"] = False
    # the under-delivered-disk warning is per BOX: the replacement can land on a
    # host that short-changes it too, and a latch from the dead box would eat it
    jc.pop("disk_shortfall_said", None)
    return True


# moved-from: herdd._replacement_cuda_floor
def _replacement_cuda_floor() -> float | None:
    """Host `cuda_max_good` floor for an automatic replacement rental. The same
    `config.LAUNCH_CUDA_MAX_GOOD` the `launch`/`train` CLI defaults to (the
    image's own CUDA runtime — memory `vast-cuda-driver-floor`), because a
    replacement re-launches the primary's own image and an under-driver host
    boots into Error-804 instead of resuming the job. `REPLACEMENT_CUDA_FLOOR=0`
    disables it; set it higher for a fleet on a newer-CUDA image."""
    v = os.environ.get("REPLACEMENT_CUDA_FLOOR")
    if v in (None, ""):
        return config.LAUNCH_CUDA_MAX_GOOD
    try:
        return float(v) or None
    except ValueError:
        return config.LAUNCH_CUDA_MAX_GOOD


# moved-from: herdd._supervise_boot_sla
def _supervise_boot_sla(st: MutableMapping[str, Any], a: argparse.Namespace, *,
                        get_instance: Callable[[Any], Mapping[str, Any] | None]
                        | None = None,
                        now: Callable[[], float] | None = None) -> str | None:
    """Default-ON come-online boot SLA for the run lane (owner directive
    2026-08-03: "longer than 10 minutes to come online is unacceptable"). A
    supervised box must reach `running` — the loading->running flip, the best
    cheaply-observable run-lane milestone per BOOT_OBSERVABILITY.md (the billed
    env-setup phase after it is covered by the box's own self-park deadline and
    the guard env-setup advisory) — within the backoff-widened BOOT_SLA_S of
    its start_date. Breach: emit `boot_sla_condemned`, record the machine in
    the run's exclusion set, destroy, and relaunch through the SAME eviction
    machinery as the throughput watchdog above (counts against
    --max-relaunch). Armed only for a box THIS supervisor observed pre-running
    and clocked on the instance's own start_date (no start_date -> no SLA).
    Opt out with --no-boot-sla or BOOT_SLA_S<=0.

    Guard-vs-owner boundary (the 2026-08-03 reconciliation): the PASSIVE
    sweeps (`guard`, reap, fleetd stray alarms) still never destroy a loading
    box — loading is GPU-unbilled, sometimes recovers (46682177 cleared at
    40 m), and an unattended sweep cannot know who is deliberately waiting.
    THIS path may destroy one: it is the lifecycle that OWNS the box for a
    run, the replacement is launched in the same tick, and the run's durable
    state lives on B2 — so the outcome is a reschedule onto a better host,
    never a loss. Same contract as _supervise_boot_health:
    None | "condemned" | "stop_fatal"/"stop_budget"."""
    if not getattr(a, "boot_sla", True):
        return None
    sla = config._boot_knob("BOOT_SLA_S")
    if sla <= 0:
        return None
    get_instance = get_instance or health._get_instance_soft
    now = now or time.time
    iid = st.get("instance_id")
    pre = (st.get("present") and iid is not None
           and st.get("actual_status") in health._BOOT_LOADING_STATES)
    if not pre:
        if (st.get("actual_status") == "running"
                and st.get("boot_sla_armed_iid") == iid):
            st["boot_sla_armed_iid"] = None    # milestone reached: SLA met
            st["boot_sla_kills"] = 0           # consecutive-kill counter resets
        return None
    st["boot_sla_armed_iid"] = iid
    inst = get_instance(iid)
    if inst is None:
        return None                            # failed poll: no verdict
    try:
        age = now() - float(inst.get("start_date"))  # type: ignore[arg-type]  # guarded
    except (TypeError, ValueError):
        return None                            # no clock, no verdict
    kills = int(st.get("boot_sla_kills") or 0)
    ddl = _boot_deadline_backoff(sla, kills)
    if age <= ddl:
        return None
    machine = inst.get("machine_id") or st.get("machine_id")
    journal._sup_emit(st["run_id"], "boot_sla_condemned", instance_id=iid,
                      machine_id=machine, boot_age_s=int(age), sla_kills=kills,
                      deadline_s=int(ddl),
                      # pre-`running` = the docker image pull / container standup:
                      # host-side by construction (nothing of ours has run yet). The
                      # post-running phases (B2 pulls, env ladder) are phase-stamped by
                      # the box itself in boot_phases.tsv / onstart.log
                      # (BOOT_OBSERVABILITY.md) and are NOT this SLA's verdict.
                      phase="image-pull", suspect="host",
                      inet_down=inst.get("inet_down"),
                      status_msg_available=bool(inst.get("status_msg")))
    print(f"!! BOOT-SLA-CONDEMNED {iid}: not running {int(age)}s after start "
          f"(> {int(ddl)}s boot SLA; machine {machine}) — destroy + relaunch "
          f"on a different host (loading is GPU-unbilled; the kill costs only "
          f"the wasted pull)")
    if machine is not None and machine not in st["excluded_machines"]:
        st["excluded_machines"].append(machine)
    a.exclude_machines = list(st["excluded_machines"])   # -> build_search_query
    if bidpolicy._guardrail_exceeded(st) == "max_relaunch":  # type: ignore[no-untyped-call]
        st["last_error"] = "max_relaunch (boot SLA kills)"
        return "stop_fatal"
    okd, derr = lifecycle._destroy_soft(iid, dry_run=getattr(a, "dry_run", False))
    if not okd:
        st["last_error"] = f"boot-SLA destroy failed: {derr}"
        return "condemned"                     # retry the destroy next tick
    if not getattr(a, "dry_run", False) and not _confirm_gone(iid):
        st["last_error"] = f"boot-SLA: {iid} not confirmed gone"
        return "condemned"
    st["husk_id"] = None
    st["instance_id"] = None
    st["boot_sampler"] = None
    st["boot_sampler_iid"] = None
    st["boot_sla_armed_iid"] = None
    st["boot_sla_kills"] = kills + 1
    verdict2 = _relaunch(st, a)
    if verdict2 in ("stop_budget", "stop_fatal"):
        st["last_error"] = st.get("last_error") or verdict2
        return verdict2
    return "condemned"


# moved-from: herdd._handoff_understudy_body
def _handoff_understudy_body(
        st: MutableMapping[str, Any], a: argparse.Namespace,
        offer: Mapping[str, Any], epoch: int = 1,
) -> tuple[dict[str, Any] | None, float | None, Any]:  # noqa: ANN401 — see docstring
    """Assemble the UNDERSTUDY launch body from ONE chosen offer, reusing the
    Phase-1 relaunch body builder (`_relaunch_body`) and auto-pricing
    (`_auto_bid_price`). Returns `(body, bid, missing)`:

      * on success: `body` (dict, label run:<RUN>:handoff), `bid` (auto-priced
        $/hr), `missing` (secret NAMES with no local value — caller refuses the
        launch if non-empty, exactly like the eviction path);
      * on rejection: `(None, None, reason)` where reason is 'candidate_reject'
        (offer fails the §2.3 `_handoff_candidate_ok` filter — reuse the T2 pure
        test, never re-derive the inequality here) or 'no_price' (offer has no
        min_bid to bid off).

    The two handoff-only deviations from a plain relaunch body:
      * label = run:<RUN>:handoff — the twin marker _launch_preflight lets coexist
        with the live primary; cutover later relabels it to run:<RUN> (§2.1);
      * B2 key name = run-<RUN>-h<nonce>. `b2_mint_key.mint()` is revoke-then-mint
        BY NAME, and the primary holds `run-<RUN>`; a colliding name would revoke
        the primary's live key mid-run (the box-44566398 incident). The nonce
        (same time+random shape as the --jobs launch key) makes the understudy's
        scoped key independent of the primary's.

    The candidate filter is re-checked HERE against the actually chosen offer
    (belt-and-suspenders: the market can move between the ARM decision and this
    launch)."""
    run_id = st["run_id"]
    mb = models._num_dph(offer.get("min_bid"))
    # The candidate's on-demand rate comes from the LIVE MARKET, never the bid
    # offer's own dph_total (the doc 50 R1 defect — that field is the
    # interruptible price, and clamping to it bid understudy 46909754 $1.071
    # over a $1.0667 floor; see _offer_ondemand_ref). None -> the filter
    # refuses below, which keeps the driver get-and-hold rather than migrating
    # onto an unpriceable box.
    od = pricing._offer_ondemand_ref(offer)
    # §2.3 gate on the concrete offer (own-preferred-ceiling viability AND the
    # conservative 2x-window amortization) — reuse the T2 pure helper as-is.
    if not bidpolicy._handoff_candidate_ok(  # type: ignore[no-untyped-call]
            handoff._handoff_primary_dph(st), mb, od,
            st.get("remaining_wall_h", 0.0),
            st.get("on_demand")):
        return None, None, "candidate_reject"
    bid = pricing._auto_bid_price(mb, od)                    # 1.2x floor, clamped < on-demand
    if bid is None:
        return None, None, "no_price"
    # nonce-suffixed key name — see docstring (never plain run-<RUN>).
    nonce = f"{int(time.time())}-{random.randint(0, 0xffff):04x}"
    key_name = f"run-{run_id}-h{nonce}"
    label = f"run:{run_id}{labels.HANDOFF_LABEL_SUFFIX}"
    body, missing = _relaunch_body(st, a, bid, label=label, key_name=key_name)
    # T4b: the two box-side contracts T6 (onstart/train.sh) consumes. The
    # understudy MUST get a strictly HIGHER HANDOFF_EPOCH than the primary (the
    # driver arms `epoch = handoffs_done + 1`; the original primary carries no
    # HANDOFF_EPOCH at all == epoch 0), so its epoch marker maxes the run's
    # runs/<ID>/handoff/ and its own push is never self-refused. HANDOFF_TTL_S is
    # the dead-man deadline (= HANDOFF_DEADLINE_S + margin) — self-park if no
    # `promoted` marker lands, i.e. the supervisor died mid-handoff.
    if body is not None:
        env = dict(body.get("env") or {})
        env["HANDOFF_EPOCH"] = str(epoch)
        env["HANDOFF_TTL_S"] = str(bidpolicy.HANDOFF_TTL_S)
        body["env"] = env
    return body, bid, missing


# moved-from: herdd._has_relaunched_after_last_evicted
def _has_relaunched_after_last_evicted(run_id: object) -> bool:
    """Idempotence probe for adopt-backfill: is there already a `relaunched`
    event after the most recent `evicted`?"""
    evs = spec._raw_events_soft(run_id)
    last_ev = max((i for i, e in enumerate(evs) if e.get("event") == "evicted"),
                  default=-1)
    tail = evs[last_ev + 1:] if last_ev >= 0 else evs
    return any(e.get("event") == "relaunched" for e in tail)


# moved-from: herdd._handoff_pick_offer
def _handoff_pick_offer(st: MutableMapping[str, Any],
                        a: argparse.Namespace) -> dict[str, Any] | None:
    """Reuse the Phase-1 spot search (`_search_offers_soft` + the run's captured
    search filters on `a`) to find the cheapest offer that clears the §2.3
    candidate filter for THIS primary. Offers arrive min_bid-ascending
    (build_search_query), so the first qualifier is the cheapest. Returns the
    offer dict or None (no qualifying offer / no market read — the driver then
    stays get-and-hold, never handing off blind)."""
    primary_dph = handoff._handoff_primary_dph(st)
    rwh = st.get("remaining_wall_h", 0.0)
    # Each candidate's on-demand rate needs a per-machine market read
    # (_offer_ondemand_ref — never the bid row's own dph_total, doc 50 R1).
    # Memoize per machine and bound the probes: offers arrive min_bid-ascending
    # so the qualifier is normally in the first few rows, and a market where
    # none of the 8 cheapest machines yields a qualifying read is one we refuse
    # to hand off into blind (the driver then stays get-and-hold).
    od_by_machine, probes = {}, 0
    for o in market_offers._search_offers_soft(a):
        mid = o.get("machine_id")
        if mid not in od_by_machine:
            if probes >= pricing.HANDOFF_ODPROBE_MAX:
                return None
            probes += 1
            od_by_machine[mid] = pricing._offer_ondemand_ref(o)
        if bidpolicy._handoff_candidate_ok(  # type: ignore[no-untyped-call]
                primary_dph, models._num_dph(o.get("min_bid")),
                od_by_machine[mid], rwh, st.get("on_demand")):
            return o
    return None


# moved-from: herdd._relaunch
def _relaunch(st: MutableMapping[str, Any], a: argparse.Namespace) -> str:
    """Eviction recovery. Returns 'relaunched' | 'noop' | 'stop_budget' |
    'stop_fatal'.

    INVARIANTS enforced here:
      * reconcile-adopt an existing live twin (idempotent across a supervisor
        restart) BEFORE launching anything — never a second box;
      * destroy the husk and CONFIRM it gone BEFORE the new PUT (no double-writer,
        stop storage bill);
      * max-bid guard (design §4 last row) refuses when no offer's bid fits;
      * emit `relaunched` STRICTLY after new_contract returns (no phantom runs);
        any transient/partial path emits nothing and returns 'noop' to retry."""
    run_id = st["run_id"]

    # (0) reconcile-adopt a live non-husk twin (crashed-mid-relaunch idempotence)
    ok, data, _ = api.request_soft("GET", "v1/instances/")
    if ok:
        instances = data.get("instances", data) if isinstance(data, dict) else data
        twins = [i for i in lifecycle.live_run_instances(run_id, instances=instances or [])
                 if i.get("id") != st.get("husk_id")]
        if twins:
            adopt = twins[0]
            st["instance_id"] = st["husk_id"] = adopt.get("id")
            st["dph_total"] = models._num_dph(adopt.get("dph_total")) or st.get("dph_total")
            if not _has_relaunched_after_last_evicted(run_id):
                st["relaunch_count"] = st.get("relaunch_count", 0) + 1
                journal._sup_emit(run_id, "relaunched", instance_id=adopt.get("id"),
                                  offer_id=adopt.get("offer_id"),
                                  bid_price=models._num_dph(adopt.get("dph_total")),
                                  relaunch_count=st["relaunch_count"],
                                  spent_usd=round(st.get("spend_usd", 0.0), 4),
                                  backoff_s=0, adopted=True)
            st["evicted_pending"] = False
            st["backoff_deadline"] = 0
            return "relaunched"

    # (0.5) refuse BEFORE destroying the husk if the spec names secret env keys we
    # cannot re-inject from the local env/.env — never trade a recoverable stopped
    # box for a fresh box launched with missing creds (SPOT_DESIGN §3.1).
    _, missing = _relaunch_body(st, a, None)
    if missing:
        miss = ",".join(sorted(set(missing)))
        st["last_error"] = f"missing_secret_env:{miss}"
        journal._sup_emit(run_id, "relaunch_refused", reason="missing_secret_env",
                          missing=miss)
        return "stop_fatal"

    # (1) destroy the husk BEFORE any new PUT, and confirm it is gone
    if st.get("husk_id") is not None:
        okd, derr = lifecycle._destroy_soft(st["husk_id"], dry_run=a.dry_run)
        if not okd:
            st["last_error"] = f"husk-destroy failed: {derr}"
            return "noop"                             # retry; husk may still die
        if not a.dry_run and not _confirm_gone(st["husk_id"]):
            st["last_error"] = f"husk {st['husk_id']} not confirmed gone"
            return "noop"                             # never a twin over a live husk
        st["husk_id"] = None

    # (2) cheapest offer whose auto-priced bid fits under --max-bid.
    #
    # THE ON-DEMAND REFERENCE IS A MARKET READ, NEVER THE OFFER ROW (doc 50 R1;
    # this was the LAST unfixed instance of that defect, found by the 2026-08-08
    # autobid audit). `_search_offers_soft` returns BID-view rows, where
    # `dph_total` is the CURRENT INTERRUPTIBLE price (min_bid + a storage sliver,
    # API-verified: min_bid 0.2667 / dph_total 0.2711 on a machine whose
    # on-demand view lists 0.5111). Clamping to it landed every relaunch bid a
    # rounding unit over its own floor — the razor-thin shape that killed
    # 46848347, 46909754, 46880245 and the v11 resume box. The launch, handoff
    # and eviction-replacement paths were fixed 2026-08-06/08-05; this one still
    # carried it, along with the `+1e-4` nudge that P2 (2026-07-18) removed
    # everywhere else and which made the first supervised poll decay a fresh
    # relaunch bid.
    #
    # The market probe is one soft POST per MACHINE, so it runs only for the
    # chosen offer: rank the candidates unclamped (which is also the correct
    # ranking — the clamp can only lower a bid, so an unclamped price is an upper
    # bound on every candidate's real one), then price the winner properly.
    offers = market_offers._search_offers_soft(a)
    max_bid = st.get("max_bid")
    priced = []
    for o in offers:
        mb = models._num_dph(o.get("min_bid"))
        if mb is None:
            continue
        bid = pricing._auto_bid_price(mb, None)              # rank unclamped; see above
        if bid is None:
            continue
        # The affordability PRE-filter is on the FLOOR, not on the unclamped rank
        # price: the on-demand clamp can only LOWER a bid, so filtering the rank
        # price against --max-bid would drop offers that are affordable once
        # priced properly. A floor over --max-bid is unaffordable by definition
        # (no winning bid sits under its own floor), so that filter is exact.
        if max_bid is not None and mb > max_bid:
            continue
        priced.append((bid, o))
    if not priced:
        st["last_error"] = f"no_offer_under_max_bid (max_bid={max_bid})"
        return "stop_budget"
    priced.sort(key=lambda t: t[0])
    # Price the ranked head properly — one soft on-demand probe per MACHINE, so
    # walk at most RELAUNCH_ODPROBE_MAX of them. `_bid_target` (not
    # `_auto_bid_price`) so --max-bid binds through the same D7 branch: a ceiling
    # that cannot afford the offer yields None and we try the next candidate.
    bid = offer = None
    for _rank_bid, o in priced[:pricing.RELAUNCH_ODPROBE_MAX]:
        _od = pricing._offer_ondemand_ref(o, getattr(a, "num_gpus", None))
        _bid = bidpolicy._bid_target(  # type: ignore[no-untyped-call]
            models._num_dph(o.get("min_bid")), max_bid, _od)
        if _bid is not None:
            bid, offer = _bid, o
            break
    if offer is None:
        st["last_error"] = f"no_offer_under_max_bid (max_bid={max_bid})"
        return "stop_budget"

    # (3) PUT — emit `relaunched` ONLY after new_contract
    body, _ = _relaunch_body(st, a, bid)             # secrets checked at (0.5)
    if a.dry_run:
        print(f"[dry-run] would PUT ask offer={offer['id']} bid={bid} "
              f"body-keys={sorted(body)}")
        cid = f"dry-{int(time.time())}"
    else:
        okp, cid, perr = lifecycle.launch_instance(offer["id"], body)
        if not okp:
            st["last_error"] = perr
            return "noop" if api._classify_http(perr) == "transient" else "stop_fatal"
        ssh.attach_ssh_key_soft(cid)          # mirror _do_launch's post-launch attach

    # (4) reset run markers on the fresh box, mirroring cmd_train's launch-time
    # hygiene (G6): a stale STOP/EXTEND or a prior DONE/FAILED STATUS would let
    # babysit tear the new box down before it boots.
    _reset_run_markers(run_id, dry_run=a.dry_run)

    st["relaunch_count"] = st.get("relaunch_count", 0) + 1
    st["instance_id"] = st["husk_id"] = cid
    # `bid` is the price we PUT, i.e. the STANDING bid (`dph_base` semantics) —
    # correct for `last_bid`, which the exact-equality self-floor guard compares
    # against the chunk's `min_bid`. `dph_total` is only the cost basis until the
    # first real observation refreshes it (there it is bid + storage, which is
    # why the two must not be conflated — see `_instance_standing_bid`).
    st["last_bid"] = bid
    st["dph_total"] = bid
    # The shared box-swap seam: episode latch + echo window + the per-MACHINE
    # sticky on-demand clamp. One call, four call sites across two lanes
    # (relaunch here; pull-reschedule, eviction replacement and handoff
    # promotion on the jobs lane) — the echo window in particular MUST clear,
    # because the relaunch offer pick can land on the SAME machine we just left,
    # where the old entries would suppress a genuine competitor floor at a price
    # we recently held (review 2026-08-10, #4).
    ladder_core.box_swap_reset(st)  # type: ignore[no-untyped-call]
    backoff_s = min(1800, 120 * (2 ** (st["relaunch_count"] - 1)))
    st["backoff_deadline"] = time.time() + backoff_s
    st["evicted_pending"] = False
    journal._sup_emit(run_id, "relaunched", instance_id=cid, offer_id=offer["id"],
                      bid_price=bid, relaunch_count=st["relaunch_count"],
                      spent_usd=round(st.get("spend_usd", 0.0), 4), backoff_s=backoff_s)
    return "relaunched"
