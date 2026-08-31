"""The RUN lane's supervision loop — one tick of policy, two drivers.

Why this module exists
----------------------
`supervise_init` / `supervise_tick` / `supervise_finalize` are the whole of the
run lane's money policy, and they have exactly **two** callers: the CLI's
inline `while` loop (`cmd_supervise`) and `fleetd`'s `run` profile
(`FLEETD_DESIGN.md` §3, `fleetd.py`'s `Hooks.run_init/run_tick/run_finalize`).
The three were extracted out of the inline loop precisely so there would be one
copy of the policy rather than a daemon fork of it, and this module is where
that copy now lives. Everything the trio needs that is *not* run-lane policy —
the journal, the handoff ladder, the replacement drivers, the market, the
lifecycle PUTs — it calls through a module attribute, so the seam stays
patchable and no lane owns another lane's code.

Alongside them sit the run lane's two self-floor pieces (`_RunLaneFloorHooks`
+ `_self_floor_guard`) and the launch-capture chain
(`supervise_init -> _init_state -> _capture_launch_spec -> _read_onstart`),
which is single-caller from top to bottom and would be a cross-module
monkeypatch in any other arrangement.

Three contracts that are easy to "clean up" and must not be
----------------------------------------------------------
* **`destroy_on_park_failure=True` is the DEFAULT and `fleetd` passes False.**
  The inline loop keeps the frozen `SUPERVISE_DESIGN` §5 behavior (a hard cap
  parks the box and destroys it if the park does not take, so the bill always
  stops); the daemon parks and *alarms*, never destroys (`FLEETD_DESIGN` §3/§8).
  Flipping the default silently hands the daemon a destroy it must never have.
* **`_init_state` MUTATES its argparse namespace.** It backfills
  `a.gpu / a.gpu_ram / a.cuda / a.num_gpus` from the captured launch spec so
  the eviction relaunch and the handoff candidate pick search the SAME market
  the box came from. Without it a `train --gpu-ram 24 --cuda 12.8 --supervise`
  child searched the whole market and the understudy landed on an 8 GB GTX 1080
  that could not run the cu128 image (handoff-canary-2, 2026-07-15). A "cleaner"
  version that returns a new namespace breaks `_relaunch` and
  `_handoff_pick_offer`.
* **The fence in `supervise_tick` is pre-OR-post tick.** `act` was computed
  against the pre-tick world, so on the tick where `complete` resets the phase
  to IDLE the pre-tick `DRAINING` must still suppress the stale act — otherwise
  the destroyed primary's `emit_evicted` fires against the freshly promoted
  understudy (same live canary).

What is deliberately NOT here
-----------------------------
* **The jobs lane.** `_JobLaneFloorHooks` and `job_supervise_tick` are its
  twins in `job_lane.py` and they are DELIBERATELY divergent: D1 (this lane's
  tenancy gate is lenient and tolerates a running->exited->running flap; the
  jobs lane keeps a strict gate because of its resume-in-place rung) and D5
  (this lane ASSIGNS `self_floor_at = None`, the jobs lane pops the key). Plan
  §5's NOTE and `FLEET_REVIEW_2026-08-14.md` item 1 pin all six divergences as
  intentional. Never unify the two, never share a base class beyond
  `ladder_core.LaneHooks`.
* **The bid state machine.** `poll`, `mk_poll_state`, the ceiling refresh, the
  decay streak and the handoff trigger are `bidpolicy` (Zone S); the self-floor
  transitions are `ladder_core.self_floor_guard`. Both are imported bare-name
  and called by attribute, never re-homed into the package.
* **The driver loop and its `time.sleep(a.interval)`.** `cmd_supervise` (CLI,
  plan step 6) and the fleetd tick own it; the only thing that left the tick
  body when it was extracted was that sleep.
* **The effectful drivers.** `_relaunch` and the boot SLA are
  `replacement.py`; the handoff state, tick, reconcile and reap are
  `handoff.py`. This module calls those by module attribute. The five the step-4
  wave left unported — `_observe`, `_accrue_cost`, `_emit_cost`, `_do_bid_move`,
  `_supervise_boot_health` — were raising SEAM stubs here and are now
  **forwarders**: the integrator ruling (step 6d) moved all five bodies into
  `replacement.py` with the rest of the effectful drivers, and `_relaunch_body`
  landed there too rather than here, beside its three callers. A forwarder, not
  a `_observe = replacement._observe` rebind, because the rebind binds at import
  time and would make `monkeypatch.setattr(replacement, "_observe", ...)`
  invisible to this lane — and because the two modules import each other, so the
  rebind is an AttributeError on one of the two import orders.
* **The `st` schema.** `state.py` documents and types the ~30 keys `_init_state`
  seeds on top of `bidpolicy.mk_poll_state`; the dicts here are annotated
  structurally (`MutableMapping[str, Any]`) until that wiring lands. The one
  key fleetd persists across a daemon restart is `bid_history`
  (`RUN_STATE_KEYS`) — everything else is rebuilt by `_init_state` on re-adopt.

Provenance: moved verbatim from `tools/vast/herdd.py` (plan §8 step 4,
2026-08-16), behavior-preserving — bodies copied, annotations added, cross-module
calls rewritten to module-attribute form. The launch-capture trio came here by
integrator ruling over three competing manifests: the chain is single-caller and
entirely run-lane, and keeping it whole keeps `_capture_launch_spec`'s patch
sites intra-module.
"""

from __future__ import annotations

import argparse
import base64
import os
import time
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import ladder_core

from vastlib.boxes import lifecycle
from vastlib.core import fmt, models
from vastlib.launch import spec as launch_spec
from vastlib.storage import b2
from vastlib.supervise import handoff, journal, replacement

import bidpolicy
import runmeta

# --------------------------------------------------------------------------- #
# THE FIVE DRIVER FORWARDERS (step 6d). These were raising SEAM stubs at step 4
# — no port manifest claimed them and no module in that wave carried them. The
# integrator ruling landed all five bodies in `supervise/replacement.py`, with
# the rest of the effectful drivers (that module's header states the ruling and
# its three reasons); what stays here is one forwarder per name.
#
# WHY A FORWARDER AND NOT A REBIND. `_observe = replacement._observe` at module
# level binds the function OBJECT at import time, so
# `monkeypatch.setattr(replacement, "_observe", ...)` — the patch surface the
# ruling requires to steer every lane caller — would not be seen through it. It
# is also an AttributeError whenever `replacement` is the module imported first
# (the two import each other). The forwarder resolves `replacement.<name>` at
# CALL time, so BOTH patch surfaces steer: a test that patches the replacement
# attribute reaches this lane, and a test that patches `run_lane._observe`
# shadows this def for the lane's own call sites below.
#
# They are still called BARE from `supervise_tick` (module global, resolved at
# call time) for the same late-binding reason the journal emitters are.
# --------------------------------------------------------------------------- #

def _observe(st: MutableMapping[str, Any],
             a: argparse.Namespace) -> MutableMapping[str, Any]:
    """The tick's ONE read of the world: the instances listing, the market floor
    (through `_self_floor_guard`), tenancy, and every `obs_status` the tick
    branches on. Returns the same `st` it was handed, mutated.
    Body: `replacement._observe` (moved from `herdd._observe`)."""
    return replacement._observe(st, a)


def _accrue_cost(st: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Bills the elapsed slice at the live dph; a not-live box accrues nothing.
    Body: `replacement._accrue_cost` (moved from `herdd._accrue_cost`).

    `workflows.ctl._workflow_budget_tick` reuses THIS name (`run_lane._accrue_cost`)
    as its cost seam — the forwarder is what keeps that call reaching the one
    body rather than a second copy of the accrual arithmetic."""
    return replacement._accrue_cost(st)


def _emit_cost(st: MutableMapping[str, Any], run_id: object) -> None:
    """The cumulative-cost journal row, emitted on a ~15 min period and at every
    transition. Body: `replacement._emit_cost` (moved from `herdd._emit_cost`)."""
    replacement._emit_cost(st, run_id)


def _do_bid_move(st: MutableMapping[str, Any], a: argparse.Namespace,
                 act: Any) -> None:  # noqa: ANN401 — bidpolicy.Action
    """Executes one rung of the ladder (the bid PUT) for a raise/rescue/lower
    Action. Body: `replacement._do_bid_move` (moved from `herdd._do_bid_move`)."""
    replacement._do_bid_move(st, a, act)


def _supervise_boot_health(st: MutableMapping[str, Any], a: argparse.Namespace, *,
                           get_instance: Callable[[Any], Mapping[str, Any] | None]
                           | None = None,
                           now: Callable[[], float] | None = None) -> str | None:
    """Opt-in (`--boot-health`) boot-throughput watchdog: condemn a box crawling
    through its image pull BEFORE the box's own fixed self-park deadline.
    Returns None / 'condemned' / 'stop_fatal' / 'stop_budget' — the same verdict
    vocabulary as `replacement._supervise_boot_sla`, which is the default-ON
    blunt backstop beside it. Body: `replacement._supervise_boot_health` (moved
    from `herdd._supervise_boot_health`); `sup-replacement.json` deferred it
    to this module and the integrator ruling put it back with its twin.

    Both keyword seams pass straight through as-is, including None: the body
    resolves its own defaults with `get_instance or health._get_instance_soft`
    and `now or time.time`, so "not passed" and "passed None" are the same call
    — one statement of the defaults, in the body. (The step-4 stub annotated
    `now` as `float | None`; it has always been a CLOCK, `now()`, and the
    annotation is corrected here to match the body it forwards to.)"""
    return replacement._supervise_boot_health(st, a, get_instance=get_instance,
                                              now=now)


# --------------------------------------------------------------------------- #
# Launch capture — supervise_init -> _init_state -> _capture_launch_spec ->
# _read_onstart. Single-caller top to bottom (grep at a1f2c8a5).
# --------------------------------------------------------------------------- #

# moved-from: herdd._read_onstart
def _read_onstart(x: str | None) -> str | None:
    if not x:
        return None
    return open(x).read() if os.path.isfile(x) else x


# moved-from: herdd._capture_launch_spec
def _capture_launch_spec(run_id: object,
                         a: argparse.Namespace) -> tuple[dict[str, Any], float | None]:
    """Snapshot the ORIGINAL launch parameters so a relaunch reproduces the box.
    Prefers the declarative runs/<RUN_ID>/spec.json (cmd_train writes it before the
    launch PUT); a transient read failure or a pre-spec run degrades to the old
    event-scrape at lower fidelity (fold_events keeps only instance_id/gpu/offer_id/
    dph/config, so image/disk/onstart/env/runtype/image_login came from the RAW
    `launched`/`supervised` events). Secret VALUES are NEVER in the spec — only
    secret_env_keys (names); _relaunch_body re-injects them from the local env.
    Returns (internal_spec, orig_bid)."""
    orig_bid = models._num_dph(getattr(a, "price", None))
    # original bid from the launched/supervised event dph (both paths use this)
    if orig_bid is None:
        for e in launch_spec._raw_events_soft(run_id):
            if e.get("event") in ("launched", "supervised"):
                b = models._num_dph(e.get("bid_price") or e.get("dph") or e.get("price"))
                if b is not None:
                    orig_bid = b
                    break

    sj = launch_spec._read_spec_soft(run_id)
    spec: dict[str, Any] = {}
    if sj.get("v") == 1:                              # spec-first (the good path)
        for k in ("image", "disk", "runtype", "runset", "image_login",
                  "gpu", "gpu_ram", "num_gpus", "cuda"):
            if sj.get(k) is not None:
                spec[k] = sj[k]
        spec["env"] = dict(sj.get("env") or {})
        spec["secret_env_keys"] = list(sj.get("secret_env_keys") or [])
        b64 = sj.get("onstart_b64")
        if b64:
            try:
                spec["onstart"] = base64.b64decode(b64).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                pass
        if orig_bid is None:
            orig_bid = models._num_dph((sj.get("bid") or {}).get("orig"))
    else:                                             # legacy event-scrape fallback
        for e in launch_spec._raw_events_soft(run_id):
            if e.get("event") in ("launched", "supervised"):
                for k in ("image", "disk", "onstart", "env", "runtype",
                          "image_login", "runset"):
                    if e.get(k) is not None and k not in spec:
                        spec[k] = e[k]

    # CLI flags fill any gap the spec/events left, and --env overrides (last wins)
    for k, v in (("image", getattr(a, "image", None)),
                 ("disk", getattr(a, "disk", None)),
                 ("onstart", _read_onstart(getattr(a, "onstart", None))),
                 ("runtype", getattr(a, "runtype", None))):
        if v is not None:
            spec.setdefault(k, v)
    env = {}
    for kv in getattr(a, "env", None) or []:
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v
    if env:
        merged = dict(spec.get("env") or {})
        merged.update(env)
        spec["env"] = merged
    return spec, orig_bid


# moved-from: herdd._init_state
def _init_state(a: argparse.Namespace) -> dict[str, Any]:
    """Full driver state: pure-poll subset + accumulators + captured launch spec.
    max_bid is the PRE-first-observe seed only (BID_FALLBACK_DPH_MULT x the run's
    original bid); once the loop reads the live on-demand price it recomputes the
    default ceiling per tick via _default_max_bid (AUTOBID_DESIGN). An explicit
    --max-bid is fixed for the run.

    MUTATES `a` — see the module docstring; that is load-bearing, not a leak."""
    run_id = runmeta.validate_run_id(a.run_id)
    now = time.time()
    spec, orig_bid = _capture_launch_spec(run_id, a)
    # Backfill the OFFER-SEARCH filters from the captured spec when the operator
    # didn't pass them on `supervise` itself: build_search_query(a) drives BOTH
    # the eviction relaunch (_relaunch) and the handoff candidate pick
    # (_handoff_pick_offer). Without this, a `train --gpu-ram 24 --cuda 12.8
    # --supervise` child searched the WHOLE market — the live canary's
    # understudy landed on an 8GB GTX 1080 that could not run the cu128 image
    # (handoff-canary-2, 2026-07-15).
    if not getattr(a, "gpu", None) and spec.get("gpu"):
        a.gpu = list(spec["gpu"])
    if not getattr(a, "gpu_ram", None) and spec.get("gpu_ram"):
        a.gpu_ram = float(spec["gpu_ram"])
    if not getattr(a, "cuda", None) and spec.get("cuda"):
        a.cuda = float(spec["cuda"])
    if getattr(a, "num_gpus", 1) == 1 and (spec.get("num_gpus") or 1) > 1:
        a.num_gpus = int(spec["num_gpus"])
    max_bid = a.max_bid if a.max_bid is not None else (
        round(bidpolicy.BID_FALLBACK_DPH_MULT * orig_bid, 3) if orig_bid else None)
    st: dict[str, Any] = bidpolicy.mk_poll_state(   # type: ignore[no-untyped-call]
        max_relaunch=a.max_relaunch, budget_usd=a.budget,
        wall_budget_s=a.wall_budget, max_bid=max_bid,
        last_bid=orig_bid, defend_at=a.defend_at)
    st.update({
        "run_id": run_id, "_t0": now, "_last_obs_t": now, "_last_cost_emit_t": now,
        "dph_total": None, "dt": 0.0, "launch_spec": spec,
        "husk_id": None, "instance_id": None, "evicted_pending": False,
        "backoff_deadline": 0, "obs_status": "ok", "last_error": None,
        "machine_id": None, "market_min_bid": None, "last_bid_put_ts": 0.0,
        "is_bid": False,                  # tenant gate for the self-floor guard
        "self_floor_at": None,            # last floor suppressed as self-referential
        "rescue_deadline": 0, "rescue_attempted": False, "now": now,
        "on_demand": None, "num_gpus": None, "floor_samples": [],
        "first_seen_dph": orig_bid, "strict_ceiling": bool(getattr(a, "strict_ceiling", False)),
        "explicit_max_bid": a.max_bid is not None,
        # boot-throughput watchdog (opt-in via --boot-health): per-box sampler +
        # the running exclusion set of machines condemned for a slow image pull.
        "boot_sampler": None, "boot_sampler_iid": None, "excluded_machines": [],
    })
    return st


# --------------------------------------------------------------------------- #
# The run lane's self-floor surface (task #73). LANE MIRRORING IS PINNED: the
# jobs lane's `_JobLaneFloorHooks` / guard are deliberately different — plan §5
# NOTE, FLEET_REVIEW_2026-08-14 item 1. Do not unify.
# --------------------------------------------------------------------------- #

# moved-from: herdd._RunLaneFloorHooks
class _RunLaneFloorHooks(ladder_core.LaneHooks):
    """The RUN lane's observation surface for the shared self-floor guard: the
    operator prints, and `_sup_emit` into the RUN's own event log (the jobs
    lane's twin journals into the BOX's log instead — see `_JobLaneFloorHooks`).

    The emitters are looked up as module globals inside the method bodies, on
    purpose: the suite monkeypatches the emitter, and a bound reference captured
    at class-definition time would not see it. In the package that means
    `journal._sup_emit(...)` — an attribute read on the journal module at CALL
    time — and never `from .journal import _sup_emit`, which would make every
    patch of it vacuously green."""

    def scaled_read(self, st: MutableMapping[str, Any], market: object) -> None:
        print(f".. offers list no {st.get('num_gpus')}-GPU chunk while we "
              f"are the live tenant (rescaled floor ${market}) — listing "
              f"mid-flap; treating as a failed read, no bid moves")

    def self_floor(self, st: MutableMapping[str, Any], *, market_min_bid: object,
                   match: Any, surviving_floor: object,  # noqa: ANN401 — ladder_core row
                   visible: bool) -> None:
        _which = ("our own standing bid" if match.kind == "standing" else
                  f"a bid we held {match.age_s:.0f}s ago (${match.price})")
        _surv_note = (f"; sibling floor ${surviving_floor} stays the market"
                      if visible else "")
        print(f".. market floor ${market_min_bid} == {_which} — that is the "
              f"price to displace OURSELVES, not a competing bidder"
              f"{_surv_note}")
        journal._sup_emit(st["run_id"], "bid_self_floor", market_min_bid=market_min_bid,
                          standing_bid=models._num_dph(st.get("last_bid")),
                          machine_id=st.get("machine_id"),
                          instance_id=st.get("instance_id"),
                          matched=match.kind, matched_bid=match.price,
                          matched_age_s=(None if match.age_s is None
                                         else round(match.age_s, 1)),
                          surviving_floor=surviving_floor)

    def floor_blind(self, st: MutableMapping[str, Any], *, since_s: float) -> None:
        _mins = since_s / 60.0
        print(f"!! floor-blind {_mins:.0f} min: every offers read has "
              f"matched our own bid series — no market signal on this "
              f"machine; bid ${st.get('last_bid')} is held, not decaying")
        journal._sup_emit(st["run_id"], "bid_floor_blind", since_s=round(since_s, 1),
                          standing_bid=models._num_dph(st.get("last_bid")),
                          machine_id=st.get("machine_id"),
                          instance_id=st.get("instance_id"))

    def episode_end(self, st: MutableMapping[str, Any], *, market: object) -> None:
        print(f".. market floor ${market} is a real competing read again "
              f"(no longer any bid of ours within the echo window)")


# MODULE-LEVEL SIDE EFFECT, harmless and load-bearing: the guard passes THIS
# instance as `hooks=`, and the jobs lane has the mirrored `_JOB_FLOOR_HOOKS`.
# No I/O in __init__; keep one instance and keep the name.
# moved-from: herdd._RUN_FLOOR_HOOKS
_RUN_FLOOR_HOOKS = _RunLaneFloorHooks()


# `floors` widened from `list[float] | None` to `Sequence[float] | None` at step
# 6d — annotation only, no behavior. Its ONE caller is `replacement._observe`,
# which passes `MarketRead.floors`, and `core/models.py` declares that field
# `Sequence[float]`. The step-4 annotation was authored while that caller was
# still a raising stub, so nothing had ever type-checked the pair.
# moved-from: herdd._self_floor_guard
def _self_floor_guard(st: MutableMapping[str, Any], market: float | None, *,
                      live: object,
                      floors: Sequence[float] | None = None,
                      scaled: bool = False) -> float | None:
    """THE SELF-REFERENTIAL FLOOR (task #73), run lane. Returns the floor to use
    this tick — `market`, or None when that "market floor" is our own standing
    bid read back.

    The state machine itself is `ladder_core.self_floor_guard`, shared verbatim
    with the jobs lane's tick (2026-08-14, FLEET_REVIEW item 1); this wrapper is
    the run lane's THREE lane-specific facts and nothing else:

      * the tenancy gate — `live` here is `_observe`'s `_still_tenant`, which
        tolerates a running->exited->running FLAP with `intended_status` still
        `running` (review 2026-08-10, #3). The jobs lane deliberately keeps a
        strict live gate instead, because it has a resume-in-place rung that
        consumes the floor while not-live; it refuses only the rescue RAISE.
        Divergence D1 in AUTOBID_DESIGN.md §"One core, two lanes" — INTENTIONAL,
        and preserved by passing the gate in rather than computing it inside;
      * the observation surface (`_RUN_FLOOR_HOOKS`: `_sup_emit` into the run's
        event log, not `_job_ladder_journal` into the box's);
      * the dedup-latch clear shape — this lane ASSIGNS `self_floor_at = None`
        where the jobs lane pops the key (both read back None; D5).

    Why the guard exists, unchanged: on a chunk we are the live tenant of,
    vast's `min_bid` is the price to displace the current tenant — us — so the
    offers read can hand back our own last PUT labelled "the market". Multiply
    it and the defend ladder chases itself: 2.697 -> 2.818 -> 3.100 -> 3.410 in
    five minutes on 47214941, on a machine whose true floor was ~$1.33
    (FLEETD_INCIDENT_2026-08-08). Suppressive only — it can lower no rail and
    raise no ceiling — and the suppressed read is treated as a FAILED read, so
    it neither moves the bid nor reaches `floor_samples` (whose median is the
    fallback `max_bid`; the run lane folds the floor in
    `_refresh_default_ceiling`, downstream of here). Row-level suppression (F3)
    and the rescaled-while-tenant refusal (F8) live in the core."""
    return ladder_core.self_floor_guard(   # type: ignore[no-any-return,no-untyped-call]
        st, market, tenant=bool(live and st.get("is_bid")),
        floors=floors, scaled=scaled,
        machine_id=st.get("machine_id"),
        now=st.get("now") or time.time(),
        hooks=_RUN_FLOOR_HOOKS)


# --------------------------------------------------------------------------- #
# The tick, the exit path, the bootstrap — the three fleetd binds by name.
# --------------------------------------------------------------------------- #

# moved-from: herdd.supervise_tick
def supervise_tick(st: MutableMapping[str, Any], a: argparse.Namespace,
                   hf: MutableMapping[str, Any],
                   handoff_on: bool) -> Any:  # noqa: ANN401 — bidpolicy.Action | None
    """ONE supervise tick — the extracted body of `cmd_supervise`'s while-loop.
    The legacy inline loop and the fleetd `run` profile (FLEETD_DESIGN §3) call
    THIS, so there is exactly one copy of the policy. SUPERVISE_DESIGN semantics
    are untouched (observe -> accrue -> heartbeat -> PURE poll() -> execute); the
    only thing that left the body is the `time.sleep(a.interval)` every
    continue-path shared, which the caller now owns.

    Returns None to keep supervising, or the terminal Action to exit on."""
    run_id = st["run_id"]
    st = _observe(st, a)                      # I/O

    if st["obs_status"] == "fatal":           # API outage != eviction; but a
        return bidpolicy.Action("stop_fatal",  # fatal (401/404/no-key) stops.
                                f"observe_fatal:{st.get('last_error')}")

    if st["obs_status"] == "transient":       # outage: heartbeat unknown, hold
        st = _accrue_cost(st)                 # (not live -> no accrual)
        journal._sup_emit(run_id, "heartbeat", actual_status="unknown",
                          relaunch_count=st["relaunch_count"],
                          spent_usd=round(st["spend_usd"], 4),
                          last_error=st.get("last_error"))
        if (st["wall_budget_s"] is not None
                and st["wall_clock_s"] >= st["wall_budget_s"]):
            return bidpolicy.Action("stop_budget", "wall_budget")  # HARD stop honored
        return None

    st = _accrue_cost(st)

    # Boot-throughput watchdog (opt-in --boot-health): condemn a box
    # crawling through its image pull EARLY, before the box's own fixed
    # self-park deadline. Runs only while pre-`running`; a no-op once the
    # box boots. Composes with, never replaces, that deadline.
    bh = _supervise_boot_health(st, a)
    if bh in ("stop_fatal", "stop_budget"):
        return bidpolicy.Action(bh, st.get("last_error") or bh)
    if bh == "condemned":
        return None

    # Come-online boot SLA (default ON — owner directive 2026-08-03): the
    # deadline the OWNING lifecycle enforces on its own box. Composes with the
    # opt-in throughput watchdog above (which condemns a provably-starved pull
    # EARLIER); this is the blunt "not running in 10 minutes" backstop.
    bs = replacement._supervise_boot_sla(st, a)
    if bs in ("stop_fatal", "stop_budget"):
        return bidpolicy.Action(bs, st.get("last_error") or bs)
    if bs == "condemned":
        return None

    bidpolicy._refresh_default_ceiling(st)    # type: ignore[no-untyped-call]  # per-tick ceiling
    over_pref, pref = bidpolicy._preferred_ceiling_alarm(st)  # type: ignore[no-untyped-call]  # ALARM
    (trigger_on, _p2, policy_target,
     trigger_why) = bidpolicy._handoff_trigger(st)  # type: ignore[no-untyped-call]  # TRIGGER
    if over_pref and not st.get("_pref_alarmed"):    # once per breach
        print(f">> bid ${st.get('last_bid')} over preferred ceiling ${pref} "
              f"({bidpolicy.BID_CEILING_ONDEMAND_FRAC:g}x on-demand ${st.get('on_demand')}) "
              f"— get-and-hold; handoff (opt-in) "
              + ("would migrate to a cheaper box" if trigger_on else
                 f"will NOT arm ({trigger_why}"
                 + (f", policy target ${policy_target}" if policy_target else "")
                 + ")"))
        journal._sup_emit(run_id, "bid_over_preferred_ceiling",
                          bid=st.get("last_bid"), preferred=pref,
                          on_demand=st.get("on_demand"))
        st["_pref_alarmed"] = True
    elif not over_pref:
        st["_pref_alarmed"] = False
    # The dwell counter reads the TRIGGER, not the alarm — the 2026-08-08
    # trigger-domain fix applies to BOTH lanes. The survival cushion that walked
    # the incident's bid to $3.41 is in `_bid_target`, which both lanes share, so
    # a run-lane box on a tight machine would arm on its own policy bid for
    # exactly the same reason. See `_handoff_trigger` and HANDOFF_DESIGN §11.
    st["_over_pref"] = trigger_on              # handoff dwell counter input
    if time.time() - st.get("_last_cost_emit_t", 0) >= 900:   # ~15 min
        _emit_cost(st, run_id)
    journal._sup_emit(run_id, "heartbeat", actual_status=st.get("actual_status"),
                      relaunch_count=st["relaunch_count"],
                      spent_usd=round(st["spend_usd"], 4),
                      last_error=st.get("last_error"))
    st["backoff_ready"] = time.time() >= st.get("backoff_deadline", 0)

    # rescue-wait resolution (SPOT_DESIGN §3.2): a rescue_bid raised the
    # stopped box's bid; give vast up to --rescue-wait to auto-resume it
    # before falling back to the unchanged destroy+relaunch path.
    live_now = st.get("present") and st.get("actual_status") in bidpolicy.LIVE_STATES
    if st.get("rescue_deadline"):
        if live_now:                          # the SAME box came back -> pause over
            journal._sup_emit(run_id, "rescued", instance_id=st.get("instance_id"),
                              relaunch_count=st["relaunch_count"])
            st["rescue_deadline"] = 0
            st["rescue_attempted"] = False
            st["not_live_streak"] = 0
            st["evicted_pending"] = False
        elif (time.time() < st["rescue_deadline"]
              and not (st["wall_budget_s"] is not None
                       and st["wall_clock_s"] >= st["wall_budget_s"])):
            return None                       # still scheduling: keep waiting
                                              # (HARD wall cap still ends it)
        else:
            st["rescue_deadline"] = 0         # timed out / capped; attempted stays
    elif live_now:                            # healthy box -> no pending rescue
        st["rescue_attempted"] = False

    # advance BEFORE poll reads it — the count AND the run's start timestamp,
    # which is what keeps the decay dwell a DURATION (BID_DECAY_S) instead of a
    # poll count a shorter tick silently re-tunes.
    st["decay_streak"], st["decay_streak_since"] = \
        bidpolicy.next_decay_state(st)  # type: ignore[no-untyped-call]
    # poll() must know a fence is open BEFORE it reads intended_status:
    # the fence's own park (and the drain's own destroy) otherwise trip
    # its operator-intent rows 2a/2b and the supervisor exits mid-cutover
    # (live canary handoff-canary-2, 2026-07-15).
    pre_phase = hf["phase"] if handoff_on else None
    st["handoff_fenced"] = bool(handoff_on
                                and pre_phase in handoff._HANDOFF_FENCE_OPEN)
    act = bidpolicy.poll(st)                  # PURE (the primary's ladder)

    # handoff is a SEPARATE decision on the SECOND instance, run AFTER
    # poll() (HANDOFF_DESIGN §6). While its two-writer fence is open the
    # driver suppresses poll()'s primary churn so the untouched ladder
    # can't fight the deliberate retirement of the primary. Terminal /
    # budget / wall stops (the returned Action below) still win — poll()'s
    # money-safety precedence is unchanged. `fenced` is pre-OR-post tick:
    # `act` was computed against the pre-tick world, so on the tick where
    # `complete` resets the phase to IDLE the pre-tick DRAINING must still
    # suppress the stale act (else the destroyed primary's emit_evicted
    # fires against the freshly promoted understudy state).
    if handoff_on:
        handoff._handoff_tick(st, a, hf, act)
    fenced = handoff_on and (pre_phase in handoff._HANDOFF_FENCE_OPEN
                             or hf["phase"] in handoff._HANDOFF_FENCE_OPEN)

    if act.kind == "noop":
        return None
    if act.kind in ("raise_bid", "rescue_bid", "lower_bid"):
        if not fenced:
            _do_bid_move(st, a, act)
            if act.kind == "lower_bid":
                st["decay_streak"] = 0        # PUT issued -> restart the run
                st["decay_streak_since"] = None
        return None
    if act.kind == "emit_evicted":
        if not fenced:                        # handoff drives the primary now
            journal._sup_emit(run_id, "evicted", instance_id=st.get("husk_id"),
                              reason=act.reason)
            st["evicted_pending"] = True
            # eviction ends the self-floor episode — the run-lane twin of the
            # reset in _job_announce_eviction (a frozen `self_floor_since`
            # across a stopped gap fakes a "continuous" floor-blind alarm)
            ladder_core.self_floor_reset(st)   # type: ignore[no-untyped-call]
            st["backoff_deadline"] = time.time() + min(
                1800, 120 * (2 ** st.get("relaunch_count", 0)))
            _emit_cost(st, run_id)            # transition -> cost snapshot
        return None
    if act.kind == "relaunch":
        if fenced:                            # primary is being retired
            return None
        verdict = replacement._relaunch(st, a)          # §S3
        if verdict in ("stop_budget", "stop_fatal"):
            return bidpolicy.Action(verdict, st.get("last_error") or verdict)
        return None                           # 'relaunched'/'noop' -> loop on
    return act                                # stop_terminal/stop_budget/stop_fatal


# moved-from: herdd.supervise_finalize
def supervise_finalize(st: MutableMapping[str, Any], a: argparse.Namespace,
                       act: Any,  # noqa: ANN401 — bidpolicy.Action
                       hf: MutableMapping[str, Any], handoff_on: bool, *,
                       destroy_on_park_failure: bool = True) -> None:
    """Exit path shared by the inline loop and fleetd: reap a mid-flight twin,
    emit the final cumulative cost, PARK-then-destroy-fallback on a HARD cap,
    and emit `supervisor_exiting`.

    `destroy_on_park_failure=False` drops the destroy fallback: fleetd never
    destroys on its own (FLEETD_DESIGN §3/§8 — a budget breach is a resumable
    PARK plus an alarm), so the daemon keeps alarming until a human parks or
    destroys the box explicitly. The inline loop keeps the frozen
    SUPERVISE_DESIGN §5 behavior (destroy if the park does not take, so the cap
    still guarantees the GPU bill stops)."""
    run_id = st["run_id"]
    if handoff_on:
        handoff._handoff_reap_on_exit(st, a, hf)     # never leak a pre-cutover twin
    _emit_cost(st, run_id)                            # final cumulative cost
    if act.kind == "stop_budget" and st.get("present") and st.get("instance_id"):
        # A HARD cap tripped while the box is still live — PARK it to stop the
        # GPU bill (the dominant cost) while keeping disk/checkpoints warm for
        # diagnosis (2026-07-10 suspend-by-default ruling; storage still bills
        # until destroyed). If the stop doesn't take, fall back to DESTROY so
        # the cap still guarantees the bill stops. Emit intent first so a
        # reader sees why it died.
        iid = st["instance_id"]
        journal._sup_emit(run_id, "stopping", reason=f"supervisor_{act.reason}")
        ok, perr = lifecycle._put_state_soft(iid, "stopped")
        if ok:
            ok, _state = lifecycle._wait_states_soft(iid, {"stopped", "exited"}, 120)
        if ok:
            print(f">> {act.reason} cap hit — parked instance {iid} (GPU billing "
                  f"stopped; disk bills until: herdd destroy {iid} -y)")
        elif not destroy_on_park_failure:
            print(f"!! {act.reason} cap hit — PARK FAILED for {iid} ({perr}). "
                  f"fleetd never destroys on its own: park it by hand "
                  f"(herdd stop {iid}) — the box is STILL BILLING")
        else:
            okd, derr = lifecycle._destroy_soft(iid)
            print(f">> {act.reason} cap hit — park failed ({perr}); destroyed {iid}"
                  + ("" if okd else f" (FAILED: {derr} — DESTROY MANUALLY: "
                                    f"herdd destroy {iid} -y)"))
    journal._sup_emit(run_id, "supervisor_exiting", reason=act.reason)
    print(f"supervisor for {run_id} exiting: {act.reason} "
          f"(relaunches={st['relaunch_count']}, spent={fmt.dollars(st['spend_usd'])})")


# moved-from: herdd.supervise_init
def supervise_init(a: argparse.Namespace) -> tuple[dict[str, Any],
                                                   MutableMapping[str, Any], bool]:
    """State + handoff bootstrap shared by the inline loop and the fleetd `run`
    profile: capture the launch spec, emit the start/policy events, reconcile a
    handoff twin left behind by a crashed supervisor. Returns (st, hf,
    handoff_on)."""
    run_id = runmeta.validate_run_id(a.run_id)
    b2._ensure_b2_remote()
    st = _init_state(a)
    journal._sup_emit(run_id, "supervisor_started")
    journal._sup_emit(run_id, "supervised",
                      max_relaunch=st["max_relaunch"], max_bid=st["max_bid"],
                      defend_at=st["defend_at"], rescue_wait_s=a.rescue_wait,
                      budget_usd=st["budget_usd"], wall_budget=st["wall_budget_s"],
                      backoff="120s*2^n cap 30m",
                      image=st["launch_spec"].get("image"),
                      disk=st["launch_spec"].get("disk"),
                      runtype=st["launch_spec"].get("runtype"),
                      runset=st["launch_spec"].get("runset"))
    # handoff is the DEFAULT over the preferred ceiling (2026-07-15 flip);
    # --strict-ceiling forces terminate-above-the-line, so it wins over the
    # default-on; --no-handoff sets a.handoff False directly.
    handoff_on = getattr(a, "handoff", True) and not getattr(a, "strict_ceiling", False)
    hf = handoff._init_handoff_state()
    if handoff_on:
        st["now"] = time.time()
        handoff._handoff_reconcile(st, a, hf)         # adopt a crashed-mid-flight twin
    return st, hf, handoff_on
