"""The supervise knot's SHARED MEMORY, described — `st`, `jc`, and the two `hf`s.

Why this module exists
----------------------
Four clusters (~5,400 body lines, 23/15/13/13 cross-edges) communicate almost
entirely by mutating three plain dicts in place. Nothing in the tree said what
was in them. This file is the first complete inventory: **170 distinct keys**
across four dict identities, each declared with the writer that creates it, so
that a reader of `replacement.py` or `job_lane.py` can find out what
`jc["p_alt_ts"]` is without grepping 20k lines.

Every count in this docstring was re-measured from source; a few differ from the
step-4 mapping manifest (which reported 163 keys, 50 late-bound and 17 pops).
The measurement wins, and `test_vastlib_supervise_state.py` re-derives it on
every run.

RE-MEASURED 2026-08-16 (notify-S2b drift re-port). The union was 163 at rev
`86840142` — the "164" this paragraph used to claim was off by one against its
own declarations, which is exactly the rot a documentation-only module is prone
to and the reason the tripwire test derives from SOURCE and not from here. The
S2b slice added **7 distinct keys**: `notify_min_bid` / `rebid_ceiling_mult` /
`defense_cap` onto `RunState` via the Zone-S `mk_poll_state` factory (plus
`launch_dph_anchor`, which `JobContext` already declared), and `notify_rows` /
`notify_matched` / `notify_consumed_ids` / `notify_quote_said` onto
`JobContext`. Four durable keys joined :data:`REPLACEMENT_STATE_KEYS` in the
same slice (14 -> 18).

It is *documentation that mypy enforces*. Once the lane modules annotate their
parameters with these types (post-wave, see "What is deliberately NOT here"), a
typo'd key is a red gate instead of a silent `None`.

TypedDict, not dataclass — and why the plan's wording was overturned
--------------------------------------------------------------------
Plan §5 says "typed dataclasses for the st/jc/handoff dicts (mutable, slots) …
dict-compat via `dataclasses.asdict` where journals persist them". The step-4
mapping pass measured that premise false on five independent counts, and the
integrator ruled TypedDict (2026-08-16). The five, kept here because each one is
a live invariant somebody will otherwise re-propose:

1. **Zone-S boundary.** 25 of `st`'s keys and all 35 of the per-tick handoff
   snapshot's keys are built by `bidpolicy.mk_poll_state` /
   `bidpolicy.mk_handoff_state` — and `bidpolicy.py` is a SHIPPED flat leaf
   (plan §3: stdlib-only, never imports `vastlib`). A dataclass in Zone P could
   only wrap or duplicate those factories, i.e. add a second representation of a
   money-path state with round-trip-loss risk. That is exactly why
   `core/models.py` deferred pydantic.
2. **The key sets are OPEN.** 74 keys are created *after* construction (12 on
   `st`, 55 on `jc`, 7 on `hf`); 7 of those are written by `ladder_core.py`,
   which receives `st` or `jc` under the parameter name `ctx`.
   `@dataclass(slots=True)` forbids attribute creation outside `__slots__`, so
   every one would have to be pre-declared with a default — which CHANGES
   SEMANTICS, because the tick bodies branch on key-ABSENCE
   (`jc.get("boot_marker_seen")`, `st.get("boot_sla_armed_iid")`).
3. **Key DELETION is first-class.** 28 `.pop()` sites (16 in `herdd.py`, 2 in
   `fleetd.py`, 10 in `ladder_core.py`) plus two full `hf.clear()` resets make
   absence a load-bearing state: re-arm a fresh pull sampler, retract a standing
   refusal, drain an empty journal queue. A dataclass has no key-absent state;
   every pop would become a sentinel assignment and every reader would need a
   matching edit.
4. **The dict PROTOCOL is used, not attribute access.** `jc.update(**18
   kwargs)` in one statement; `jctx["_handoff_completed_iid"] = u` paired with
   `jc.pop("_handoff_completed_iid", None)` as an out-parameter channel between
   two functions; three-arg `.get(k, default)` where the default DIFFERS per
   call site for the same key (`st.get("remaining_wall_h", 0.0)` vs
   `jctx.get("remaining_wall_h")` bare).
5. **Tri-state floats.** See §"Tri-state" below — dataclass field defaults
   invite `0.0` and would re-open defect #67.

`total=False` is required, not stylistic: it is the only form that models 74
late-bound keys and 28 pop sites honestly. A `total=True` TypedDict would claim
every key is present at construction, which is false for all four shapes.

FOUR shapes, deliberately not merged
------------------------------------
* :data:`RunState` — the run lane's `st`.
* :data:`JobContext` — the jobs lane's `jc`, which every `_job_handoff_*`
  function names `jctx`. **One type, two parameter names**; there is no second
  dict.
* :data:`HandoffCarry` / :data:`JobHandoffCarry` — the MUTABLE per-run handoff
  carry (`_init_handoff_state`, 23 keys / `_init_job_handoff_state`, +3).
* :data:`HandoffSnapshot` — the PURE per-tick snapshot from
  `bidpolicy.mk_handoff_state` (35 keys), rebuilt from scratch every tick and
  never mutated.

The last two are both called `hf` at their call sites and they are NOT
interchangeable (mapping hazard H3). Measured set difference: the snapshot has
16 keys the carry lacks (`budget_usd`, `spend_usd`, `now`, `primary_bid`,
`primary_dph`, `primary_on_demand`, `primary_evicted`, `drain_ts`,
`understudy_gone`, `remaining_wall_h`, `driver_can_complete`, `work_at_risk_h`,
`running_unresumable`, `min_running_eta_s`, `ckpt_stale`, `unsafe_override`);
the carry has 4 the snapshot lacks (`chosen_offer`, `cutover_ts`, `epoch`,
`stall_alarmed`). A single `HandoffState` over both would be wrong in both
directions.

Lane mirroring is PINNED
------------------------
`JobHandoffCarry` inherits `HandoffCarry` and adds three keys; `JobContext`
mirrors much of `RunState` under different names (`not_live` vs
`not_live_streak`, `last_bid_put` vs `last_bid_put_ts`, `t0` vs `_t0`,
`pref_alarmed` vs `_pref_alarmed`). That divergence is deliberate (plan §5
NOTE, FLEET_REVIEW item 1: six pinned divergences). **Do not unify the names to
make the types tidier** — the runtime dicts would still differ, and the type
would then lie.

Tri-state floats — None means UNKNOWN, never 0.0 (defect #67)
-------------------------------------------------------------
:data:`TRISTATE_FLOAT_KEYS` lists them. `min_running_eta_s`,
`remaining_wall_h`, `timeout_ceiling_h` and `work_horizon` readings are
`float | None`, and `None` means "no progress signal exists" — which REFUSES a
migration rather than assuming one. `mk_handoff_state`'s own docstring pins it:
"Tri-state on purpose: None means unknown, never 0 and never infinite."

The asymmetry the port must preserve, measured at `_job_handoff_build_state`:
`remaining_wall_h` and `work_at_risk_h` are read as `jctx.get(k, 0.0)` (a 0.0
default) while `min_running_eta_s` is read BARE (`jctx.get(k)`, None survives).
Both readings are correct for their field and neither may be "harmonised". The
2026-08-08 22:17Z incident priced a migration against `remaining_wall_h: 9.904`
— 36000 s of hang ceiling minus 345 s elapsed — when the real work left was
1-2 h, inflating the projected saving ~5x. Collapsing an UNKNOWN to 0.0 is how
that class of bug gets back in. **No `or 0.0`, no 0.0 field defaults, ever.**

FOUND-NOT-FIXED: the `st["dph"]` phantom (mapping hazard H4)
------------------------------------------------------------
`herdd.py:8956` reads::

    hf["prefence_bid"] = _prefence_bid(st.get("last_bid"), st.get("dph"))

`dph` is **never written on the run-lane `st`** — the run lane's key is
`dph_total` (the jobs lane's `jc` is the one that carries `dph`, written by its
per-tick `jc.update`). The read therefore always yields `None`, and
`_prefence_bid`'s second argument has been dead since it was written.

This is recorded, **not repaired**. Repairing it would change what price the
pre-fence bid is computed against — a money-path behavior change, which plan
§7.4 forbids inside a port ("any test that cannot pass without an expectation
change is a found behavior drift — stop, diagnose"). It is a separate,
owner-visible change.

`dph` is deliberately **absent from** :data:`RunState`. When `run_lane.py`
annotates that parameter (post-wave), mypy will flag the read as
`typeddict-item`. That error is the type system finding the defect, not a port
regression: fix the *defect* under its own change, do not add `dph` to
`RunState` to silence it.

Name-shadowing trap (mapping hazard H5)
---------------------------------------
`herdd.py:18426-18437` binds a LOCAL `st = json.load(fleet_state_path())` —
that is **fleetd's `state.json`**, whose `watches` / `ceiling_by_box` keys
belong to `fleet/state.py`'s schema, not to this module's `RunState`. The same
name is used for two unrelated shapes; it is the only such shadow in the file.
`fleetd.py`'s own `st[...]` sites (`version`, `watches`, `strays`, `destroys`,
`intents`, `spend_by_box`, `meta`, `alarms`, `ceilings`, `ceiling_by_box`,
`notify`) are that other schema too. None of them are declared here.

What is deliberately NOT here
-----------------------------
* **No constructors.** `_init_state` / `supervise_init` land in `run_lane.py`;
  `job_supervise_init` / `_init_job_handoff_state` in `job_lane.py`;
  `_init_handoff_state` with the run lane. They are effectful (they emit
  journal events, mutate `a`, reconcile a crashed twin) and belong with the
  lane that owns their I/O. This module describes their output.
* **No re-export of the Zone-S factories.** `bidpolicy.mk_poll_state` /
  `mk_handoff_state` stay exactly where they are and this module does not
  import them; the `herdd.py` module-level aliases that `test_supervise.py`
  reaches through are a step-6/7 concern, handled by the rename table. This
  file has **no runtime imports at all** beyond `typing`, which is what keeps
  it free to be imported from anywhere in the DAG.
* **No `dataclasses.asdict` seam.** Plan §5's line describes a seam that does
  not exist: nothing serializes `st`/`jc`/`hf` whole. Journals lift scalars out
  by hand (`**fields`); `state.json` takes two hand-written key PROJECTIONS
  (:data:`REPLACEMENT_STATE_KEYS`, :data:`RUN_STATE_KEYS`) with explicit
  coercion. `jc` is not even JSON-serialisable — it holds a live
  `argparse.Namespace` under `jc["a"]` and a python `set` under
  `jc["evicted_machines"]`, which is precisely WHY the projections exist.
* **No validation, no runtime behavior.** A TypedDict is erased at runtime;
  these dicts stay plain dicts and every subscript / `.get` / `.pop` /
  `.update` site in the ported bodies is byte-identical to what it was.
* **No annotation of the lane modules, this wave.** `handoff.py`,
  `replacement.py`, `run_lane.py`, `job_lane.py` and `retention.py` are written
  concurrently with this file and annotate `st` / `jc` / `hf` as
  `MutableMapping[str, Any]`. Wiring these types into their signatures is a
  post-wave pass, deliberately separated so a typing disagreement cannot block
  a behavior-preserving move.

Provenance
----------
New code, 2026-08-16, plan §8 step 4. It moves nothing, so it carries no
`moved-from:` marker (README §2 rule 7). The key inventory was measured against
`herdd.py` / `fleetd.py` / `bidpolicy.py` / `ladder_core.py` at rev
`86840142` (manifest `.port_manifests/sup-state.json`, built at `a1f2c8a5`).
The `file:line` citations on each key are rev-pinned provenance and WILL drift;
`test_vastlib_supervise_state.py` re-derives the key sets from source on every
run and is the authority when the two disagree.
"""

from __future__ import annotations

from typing import Any, TypedDict

from vastlib.fleet import state as fleet_state

# --------------------------------------------------------------------------- #
# the handoff sub-state — two shapes, one name at the call sites (hazard H3)
# --------------------------------------------------------------------------- #


class HandoffCarry(TypedDict, total=False):
    """The MUTABLE per-run handoff sub-state carried across ticks.

    Built by `herdd._init_handoff_state()` (23 keys) and thereafter mutated in
    place by the handoff driver. Distinct from :class:`HandoffSnapshot`, which is
    rebuilt from scratch every tick from this carry plus the lane context.

    `_handoff_reset` / `_job_handoff_reset` return it to IDLE by
    `hf.clear(); hf.update(_init_*_handoff_state())`, preserving only
    `handoffs_done` and `cooldown_until`. That `clear()` also drops every
    late-bound key below — absence after a reset is the intended state, and is a
    second reason this cannot be a dataclass.
    """

    # --- declared by _init_handoff_state (herdd.py:8641-8663) --------------
    #: IDLE|ARMED|LAUNCHING|WARMING|SYNCED|CUTOVER|DRAINING|DONE|ABORT
    phase: str
    over_ceiling_streak: int          # the dwell counter that arms a handoff
    over_ceiling_since: float | None  # handoff.py:1000 (run) / :1844 (jobs) —
                                      # when the current over-ceiling run began.
                                      # The dwell is HANDOFF_DWELL_S of wall
                                      # clock; the counter above is the fallback
                                      # for a carry written before this key
    primary_iid: str | None
    understudy_iid: str | None
    understudy_dph: float | None
    understudy_on_demand: float | None
    understudy_status: str | None
    understudy_live_since: float | None
    understudy_producing: bool
    candidate_min_bid: float | None
    candidate_on_demand: float | None
    chosen_offer: dict[str, Any] | None   # NOT in HandoffSnapshot
    final_flush_seen: bool
    epoch: str | None                 # T4b write-generation, set at ARM; NOT in
                                      # HandoffSnapshot
    fence_ts: float | None            # wall-clock the fence opened (CUTOVER)
    stall_alarmed: bool               # DRAINING-stall once-flag; NOT in snapshot
    cutover_ts: str | None            # compact-UTC promotion moment; NOT in
                                      # snapshot
    ckpt_pulled_epoch: str | None
    handoff_started_ts: float | None
    handoff_spend_usd: float
    handoffs_done: int
    cooldown_until: float
    primary_gone: bool

    # --- LATE-BOUND: created mid-tick, never declared by any constructor -----
    # (7 keys; each is absent until the write below fires, and absent again
    # after a _handoff_reset. Readers use .get() and branch on absence.)
    drain_ts: float | None            # herdd.py:9003 — post-cutover DRAINING
                                      # clock start
    prefence_bid: float | None        # herdd.py:8956 / :16542 — see the H4
                                      # FOUND-NOT-FIXED note in the module
                                      # docstring: the run-lane write reads a
                                      # phantom st["dph"] and always gets None
    primary_shape: dict[str, Any] | None   # herdd.py:16477 (jobs lane only)
    understudy_gone: bool             # herdd.py:16302/:16305
    defer_sig: Any                    # herdd.py:16834 — deferral signature,
                                      # once-per-condition latch
    refuse_sig: Any                   # herdd.py:16895, POPPED at :16984 —
                                      # ABSENCE retracts a standing refusal
    pct_warned: dict[str, Any]        # herdd.py:16917 (setdefault) — per-job
                                      # percentage-warning latches


class JobHandoffCarry(HandoffCarry, total=False):
    """The jobs lane's handoff carry: :class:`HandoffCarry` plus three keys.

    `herdd._init_job_handoff_state()` is literally `_init_handoff_state()`
    followed by three assignments. LANE MIRRORING IS PINNED (plan §5 NOTE): this
    inherits rather than merges, and the run lane must never grow these three.
    """

    pending_jobs: list[str]           # herdd.py:12861 — the JOB_IDs to move
    running_jobs: list[str]           # herdd.py:12862 — RUNNING at the fence
                                      # (the final_flush wait set)
    retarget_incomplete: str | None   # herdd.py:12863 — an old ticket whose
                                      # delete failed at cutover (§5)


class HandoffSnapshot(TypedDict, total=False):
    """The PURE per-tick handoff snapshot — `bidpolicy.mk_handoff_state`'s output.

    Zone S owns this shape (`bidpolicy.py:2245-2320`, 35 keys, all
    kwargs-with-defaults). It is declared here because both lanes' build
    functions (`_handoff_build_state`, `_job_handoff_build_state`) return it and
    `handoff_poll` consumes it — but the constructor is NOT ported and NOT
    wrapped (plan §3: shipped leaves stay stdlib and flat).

    `test_vastlib_supervise_state.py` asserts this TypedDict's keys are a
    superset of what the live factory emits, so a Zone-S drift goes red HERE
    rather than silently widening the dict.
    """

    phase: str
    over_ceiling_streak: int
    primary_iid: str | None
    primary_bid: float | None         # not in HandoffCarry
    primary_on_demand: float | None   # not in HandoffCarry
    primary_dph: float | None         # not in HandoffCarry
    primary_evicted: bool             # not in HandoffCarry
    primary_gone: bool
    understudy_iid: str | None
    understudy_dph: float | None
    understudy_on_demand: float | None
    understudy_status: str | None
    understudy_live_since: float | None
    understudy_producing: bool
    understudy_gone: bool             # not in HandoffCarry
    drain_ts: float | None
    candidate_min_bid: float | None
    candidate_on_demand: float | None
    remaining_wall_h: float | None    # TRI-STATE; read as .get(k, 0.0) from the
                                      # lane ctx, see the module docstring
    final_flush_seen: bool
    fence_ts: float | None
    ckpt_pulled_epoch: str | None
    handoff_started_ts: float | None
    handoff_spend_usd: float
    handoffs_done: int
    cooldown_until: float
    budget_usd: float | None          # not in HandoffCarry
    spend_usd: float                  # not in HandoffCarry
    now: float                        # not in HandoffCarry
    # --- work awareness (2026-08-08, tasks #61/#62/#67) ----------------------
    driver_can_complete: bool         # FAILS CLOSED: a driver that has not
                                      # declared it can carry a migration to
                                      # `complete` cannot arm one (defect #61)
    work_at_risk_h: float             # hours a migration would DISCARD
    running_unresumable: int          # RUNNING jobs with no checkpoint (#62)
    min_running_eta_s: float | None   # TRI-STATE: None = unknown, never 0 and
                                      # never infinite. Read BARE (no default).
    ckpt_stale: bool
    unsafe_override: bool             # HANDOFF_DESIGN §11 — skips the ARM
                                      # preconditions ONLY, never the fence rails


# --------------------------------------------------------------------------- #
# the eviction-retention record — a jc["retained_boxes"] element
# --------------------------------------------------------------------------- #

class QuiesceRecord(TypedDict, total=False):
    """What `_job_quiesce_box` did to a box it parked (`herdd.py:15197-15215`).

    Stored on :data:`RetainedBox` under `quiesce` so `fleet log` can say what was
    done and the salvage runbook knows a resume needs the bid raised back.

    `stopped` / `bid_pinned` are TRI-STATE: `None` under `--dry-run` means "no
    stop and no bid pin was issued", which is not the same as `False` ("tried and
    failed"). The dry-run branch sets both to None explicitly and appends the
    reason to `errors`.
    """

    stopped: bool | None
    bid_pinned: float | None
    prior_bid: float | None
    errors: list[str]
    ts: float | None
    why: str


#: One retained (or destroyed, or already-gone) box, as
#: `_job_retention_record` builds it (`herdd.py:15267-15279`) and the
#: retention sweep advances it (`:15506-15560`). Appended to
#: `jc["retained_boxes"]` by three `jc.setdefault("retained_boxes", []).append`
#: sites (:15290, :15314, :15348).
#:
#: WIRE FORMAT — `retained_boxes` is a member of :data:`REPLACEMENT_STATE_KEYS`,
#: so this whole record round-trips through fleetd's `state.json` under
#: `w["replacement"]["retained_boxes"]` and is read back by `fleet status`
#: (`fleetd.py:711/:754/:845`). Renaming a field here silently drops a retention
#: DEADLINE across a daemon restart — i.e. a box nobody follows to a terminal
#: outcome.
#:
#: Functional TypedDict syntax is forced: `class` is a Python keyword and the
#: record really does carry a key spelled `class` (the eviction class).
RetainedBox = TypedDict(
    "RetainedBox",
    {
        "iid": str,
        # retained | destroyed | destroy_failed | already_gone | expired |
        # reaped | retention_lost
        "status": str | None,
        "class": str,                     # eviction class — keyword-named key
        "retained_ts": float,
        "deadline_ts": float | None,
        "retention_h": float,
        "cost_usd": float | None,
        "cost_hi_usd": float | None,
        "storage_day_usd": float | None,
        "replacement_iid": str | None,
        "label": str | None,
        "keep_labeled": bool,
        # --- added after construction ---------------------------------------
        "quiesce": QuiesceRecord,          # herdd.py:15340 (_job_quiesce_box)
        "salvage": dict[str, Any],        # herdd.py:15280/:15347 — salvage.py
                                          # owns this shape; not modelled here
        "ended_ts": float,                # herdd.py:15519 (sweep, terminal)
        "live_dph": float | None,         # _job_retention_liveness: a retained
        "live_multiple": float | None,    # box found LIVE is billing GPU rate
        "live_since_ts": float | None,    # right now, window or no window
        "requiesces": int,
        "resurrections": int,
    },
    total=False,
)


# --------------------------------------------------------------------------- #
# the run lane's `st`
# --------------------------------------------------------------------------- #


class RunState(TypedDict, total=False):
    """The run lane's driver state — `herdd._init_state(a)`'s output, mutated.

    54 keys at construction (29 from the Zone-S `bidpolicy.mk_poll_state`
    subset + 30 from `_init_state`'s `st.update`, overlapping by 5), 8 more
    created mid-tick, and 4 more written only by `ladder_core.py`, which
    receives this dict as `ctx` — 66 in all. (`ladder_core` writes 7 keys;
    `last_bid`, `first_seen_dph` and `self_floor_at` are already declared by the
    constructors.)

    `st["dph"]` is NOT declared — see the FOUND-NOT-FIXED note in the module
    docstring (hazard H4). `st["watches"]` / `st["ceiling_by_box"]` are NOT
    declared either: those belong to fleetd's `state.json`, which shadows the
    name `st` at `herdd.py:18426` (hazard H5).
    """

    # --- the pure-poll subset: bidpolicy.mk_poll_state (bidpolicy.py:426-450) -
    # ZONE S owns these 29 (25 before notify S2b). The portable lane hand-builds
    # them, so their names are a contract with a shipped leaf.
    view: dict[str, Any]              # this tick's instance body
    present: bool
    actual_status: str | None
    intended_status: str | None
    status_marker: str | None
    stopping_actor: str | None
    stopping_reason: str | None
    not_live_streak: int
    backoff_ready: bool
    relaunch_count: int
    spend_usd: float
    wall_clock_s: float
    max_relaunch: int
    budget_usd: float | None
    wall_budget_s: float
    max_bid: float | None
    last_bid: float | None            # also written by ladder_core (WIRE FORMAT)
    market_min_bid: float | None
    last_bid_put_ts: float
    rescue_attempted: bool
    now: float
    defend_at: float | None
    decay_streak: int
    decay_streak_since: float | None  # run_lane.py:536 / :569 — the decay dwell
                                      # is a DURATION (bidpolicy.BID_DECAY_S);
                                      # this is when the current candidate run
                                      # started. Absent => the legacy poll count
    on_demand: float | None
    handoff_fenced: bool
    # The four the notify-S2b slice added to the Zone-S factory (2026-08-16,
    # bidpolicy.py `mk_poll_state`). They are the BOUNDS + the price of the
    # notification-priced rescue quote, and `bidpolicy.notify_rescue_bound` is
    # the only consumer: with `notify_min_bid` None the other three are inert,
    # which is what keeps the pre-S2b rescue byte-identical. Declared here for
    # the same reason as the other 25 — Zone S owns the names, so a key added
    # there and not here is a silently incomplete inventory.
    notify_min_bid: float | None      # the displacing price off a MATCHED outbid
                                      # row; None = no row (every box with the
                                      # driver gate off). TRI-STATE.
    launch_dph_anchor: float | None    # the launch price the rescue ceiling is a
                                      # multiple of; None = no derivable ceiling,
                                      # which REFUSES the quote outright
    rebid_ceiling_mult: float          # the per-watch knob (`_rebid_knob`), so
                                      # the rescue ceiling IS the re-bid rung's
    defense_cap: float | None          # the job-aware defense ceiling this tick
                                      # (`_job_defense_cap`); None = no fresh
                                      # p_alt, so no derivable defense. TRI-STATE.

    # --- added by _init_state's st.update (herdd.py:7719-7733) -------------
    run_id: str
    _t0: float                        # loop start (run lane spells it `_t0`;
                                      # the jobs lane spells the same thing `t0`
                                      # — pinned mirror divergence, do not unify)
    _last_obs_t: float
    _last_cost_emit_t: float
    dph_total: float | None           # the run lane's price key. The jobs lane's
                                      # is `dph`; see hazard H4.
    dt: float
    launch_spec: dict[str, Any]       # captured at init, drives _relaunch AND
                                      # the handoff candidate pick
    husk_id: str | None
    instance_id: str | None
    evicted_pending: bool
    backoff_deadline: float
    obs_status: str
    last_error: str | None
    machine_id: Any                   # int on the wire, str after a JSON
                                      # round-trip — compared with str() at every
                                      # site that matters
    is_bid: bool                      # tenant gate for the self-floor guard
    self_floor_at: float | None       # WIRE FORMAT (ladder_core also writes it)
    rescue_deadline: float
    num_gpus: int | None
    floor_samples: list[float]        # bidpolicy._observe_floor appends; its
                                      # median seeds the default ceiling
    first_seen_dph: float | None      # WIRE FORMAT (ladder_core also writes it)
    strict_ceiling: bool
    explicit_max_bid: bool
    boot_sampler: Any                 # boxes.health.BootThroughputSampler | None
                                      # (kept Any so this module imports nothing)
    boot_sampler_iid: str | None
    excluded_machines: list[Any]      # machines condemned for a slow image pull;
                                      # copied onto a.exclude_machines each tick

    # --- LATE-BOUND: created mid-tick, never declared by _init_state ---------
    _instances: list[dict[str, Any]]  # herdd.py:7857 — this tick's single fetch
    market_min_bid_raw: float | None  # herdd.py:7919 — pre-self-floor read
    boot_sla_armed_iid: str | None    # herdd.py:8167/:8170/:8215
    boot_sla_kills: int               # herdd.py:8168/:8216
    on_demand_last: float | None      # herdd.py:1202; POPPED by
                                      # ladder_core.py:215 on a box swap
    remaining_wall_h: float | None    # herdd.py:9181 — TRI-STATE, but read as
                                      # .get(k, 0.0) at :8278/:8312/:8760
    _pref_alarmed: bool               # herdd.py:9301/:9303
    _over_pref: bool                  # herdd.py:9309 — handoff dwell input

    # --- written by ladder_core.py, which names this dict `ctx` --------------
    # WIRE FORMAT. ladder_core.py:50-56 pins it verbatim: "key names here are a
    # wire format … Renaming one silently drops durable state across a daemon
    # restart." `bid_history` is additionally the whole of RUN_STATE_KEYS.
    bid_history: list[list[Any]]      # ladder_core.py:129-160 — [ts_first, price,
                                      # machine_id, ts_last]; legacy entries are
                                      # 3-field. WIRE FORMAT + RUN_STATE_KEYS.
    self_floor_since: float | None    # ladder_core.py:408-413; POPPED at :180
    self_floor_sustained_said: bool   # ladder_core.py:414; POPPED at :181
    _scaled_floor_said: bool          # ladder_core.py:368-369; POPPED at :372


# --------------------------------------------------------------------------- #
# the jobs lane's `jc` (a.k.a. `jctx`)
# --------------------------------------------------------------------------- #


class JobContext(TypedDict, total=False):
    """The jobs lane's per-tick context — `herdd.job_supervise_init(a)`'s `jc`.

    **`jctx` is not a second type.** Every `_job_handoff_*` def names its
    parameter `jctx`; `job_supervise_tick` passes `jc`. Same object, same shape.

    32 keys at construction, 59 more created later — 91 in all. 13 of the 59
    come from the single 18-kwarg `jc.update(...)` at `herdd.py:17803-17844`
    (the other 5 kwargs re-write declared keys), which runs on every
    handoff-enabled tick; 5 come from `ladder_core.py`.

    NOT JSON-SERIALISABLE by design: `jc["a"]` is a live `argparse.Namespace`
    and `jc["evicted_machines"]` is a python `set`. That is why the durable
    projection (:data:`REPLACEMENT_STATE_KEYS`) exists and why it coerces.
    """

    # --- declared by job_supervise_init (herdd.py:17038-17084) -------------
    a: Any                            # LIVE argparse.Namespace — not serialisable
    iid: str                          # REASSIGNED to the understudy after a
                                      # completed handoff (:17850)
    dry_run: bool
    budget_usd: float | None
    spend_usd: float
    instances: list[dict[str, Any]]
    pending_jobs: list[str]
    running_jobs: list[str]
    # loop-carried accumulators (locals in the pre-fleetd inline loop)
    last_bid: float | None            # also written by ladder_core (WIRE FORMAT)
    max_bid: float | None
    first_seen_dph: float | None      # also written by ladder_core (WIRE FORMAT)
    floor_samples: list[float]
    decay_streak: int
    decay_streak_since: float | None  # job_lane.py:1634 — the run's start, so
                                      # the decay dwell is BID_DECAY_S of wall
                                      # clock and not N ticks of an interval
                                      # the unit template can change
    not_live: int                     # run lane spells it `not_live_streak`
    was_live: bool | None
    rescue_deadline: float | None
    last_bid_put: float               # run lane spells it `last_bid_put_ts`
    t_prev: float
    t0: float                         # run lane spells it `_t0`
    pref_alarmed: bool                # run lane spells it `_pref_alarmed`
    reconciled: bool
    # automatic eviction replacement (owner directive 2026-08-05); fleetd seeds
    # these from the durable watch record on every daemon start.
    replacements: int                 # REPLACEMENT_STATE_KEYS (spend bound)
    replacement_history: list[dict[str, Any]]   # REPLACEMENT_STATE_KEYS
    replacement_refused: str | None
    launch_dph_anchor: float | None   # REPLACEMENT_STATE_KEYS (price anchor)
    launch_disk_gb: float | None      # REPLACEMENT_STATE_KEYS (task #69)
    launch_cc_allow: list[int]        # REPLACEMENT_STATE_KEYS — sm levels the
                                      # launch declared (`--cc-allow` ->
                                      # LAUNCH_CC_ALLOW); [] = unconstrained
    launch_env_pin: dict[str, Any]    # REPLACEMENT_STATE_KEYS — allowlisted
                                      # launch env (EVAL_ENV_VER) the rehost
                                      # lanes re-apply; never the whole env
    evicted_machines: set[Any]        # REPLACEMENT_STATE_KEYS — a python SET;
                                      # persisted as a sorted LIST, restored as
                                      # a set. The only coerced key.
    evicted_machine_ts: dict[str, Any]  # REPLACEMENT_STATE_KEYS — machine id
                                      # STRINGIFIED (state.json is JSON) ->
                                      # {"ts", "class"}; makes an exclusion EXPIRE
    handoff_on: bool                  # default-on over the ceiling;
                                      # --strict-ceiling and serve_mode force off
    handoff_can_complete: bool        # FAILS CLOSED (defect #61)
    handoff_unsafe_override: bool     # HANDOFF_DESIGN §11
    serve_mode: bool
    # --- the serve lane's identity pin (job_lane.py:326-327) ----------------
    # WHAT this box is supposed to be serving, off the WATCH POLICY — not a
    # launch anchor like the four above. Both None on every watch registered
    # without `fleet watch --artifact`, which is what makes the whole check a
    # no-op for a pre-P3 serve watch rather than a code path it must survive.
    model_artifact: str | None        # registry slug; telemetry + the pin's
                                      # derivation source
    expect_ident: str | None          # grade-A fingerprint_sha256[:12] — the
                                      # value actually compared against the
                                      # READY marker's `ident=` field
    # The VERDICT, written by replacement._serve_identity_tick (:1700) and
    # popped by it when the pin is dropped — absence is load-bearing (see
    # POPPED_KEYS). fleetd mirrors both onto the watch record every tick
    # (fleet/state.py SERVE_IDENTITY_KEYS) and DERIVES its alarms from them.
    serve_identity: dict[str, Any]    # {state, expected, observed, reason,
                                      # artifact, serve_id, iid, since, parked}
    serve_identity_condemned: bool    # the withdrawal LATCH: no later rung of
                                      # the ladder may rescue this box, and a
                                      # daemon restart must not forget it

    # --- written every tick by jc.update(**18) (herdd.py:17803-17844) ------
    # 5 of the 18 (iid, last_bid, budget_usd, pending_jobs, running_jobs) are
    # declared above; the 13 below exist only from the first such tick onward.
    now: float
    dt: float
    on_demand: float | None
    dph: float | None                 # the jobs lane's price key (the run lane's
                                      # is `dph_total`) — see hazard H4
    market_min_bid: float | None
    remaining_wall_h: float | None    # TRI-STATE (defect #67): the progress ETA
                                      # capped by the timeout ceiling and the
                                      # --wall-budget remainder. UNKNOWN refuses.
                                      # Read as .get(k, 0.0) at :16360.
    timeout_ceiling_h: float | None   # TRI-STATE — the hang-detector bound, kept
                                      # so a deferral can say WHICH bound it hit
    work_at_risk_h: float             # hours a migration would DISCARD
    running_unresumable: int
    min_running_eta_s: float | None   # TRI-STATE; read BARE at :16381
    ckpt_stale: bool
    _over_pref: bool
    primary_evicted: bool

    # --- LATE-BOUND: everything else, by writer ------------------------------
    # boot SLA / boot observability
    boot_loading_iid: str | None      # herdd.py:17571
    boot_marker_seen: dict[str, Any]  # herdd.py:13872 (setdefault)
    boot_online_iid: str | None       # herdd.py:13682/:13920
    boot_running_iid: str | None      # herdd.py:17588
    boot_running_since: float | None  # herdd.py:17589
    boot_sla_disabled: bool           # herdd.py:13862/:13981
    boot_sla_phase: str | None        # herdd.py:13878
    boot_sla_phase_alarmed: bool      # herdd.py:13908
    boot_sla_pyhalf_said: bool        # herdd.py:13701
    # pull watchdog — ABSENCE re-arms a fresh sampling window
    pull_sampler: Any                 # herdd.py:13634; POPPED :14164/:14222/
                                      # :16192/:17582
    pull_sampler_iid: str | None      # herdd.py:13634; POPPED :14223/:16193/
                                      # :17583
    pull_bad_machines: Any            # herdd.py:14070 (setdefault)
    pull_relaunches: int              # herdd.py:14221
    pull_watchdog_disabled: bool      # herdd.py:14111
    # eviction / rebid ladder
    evicted_announced: str | None     # herdd.py:17252; POPPED :14236/:16206/
                                      # :17880/:17906 — the once-per-cycle
                                      # announce latch. REPLACEMENT_STATE_KEYS.
    evicted_class: str | None         # the class the cycle was ANNOUNCED with —
                                      # what the host-stop escalation reads, so
                                      # ladder and journal cannot disagree.
                                      # REPLACEMENT_STATE_KEYS.
    evicted_since: float | None       # when that announcement landed: the
                                      # escalation's clock. REPLACEMENT_STATE_KEYS.
    rebid_rungs: int                  # herdd.py:14398 — REPLACEMENT_STATE_KEYS
                                      # (a bound on WALL TIME as well as money)
    rebid_refused: str | None         # herdd.py:14369/:14399
    resume_tries: int                 # herdd.py:17161 — REPLACEMENT_STATE_KEYS
    resume_refused: str | None        # herdd.py:17154/:17157
    ceiling_escalated: bool           # herdd.py:14238/:16208/:17786/:17793
    last_replacement_disk_gb: float | None   # herdd.py:13516
    retained_boxes: list[RetainedBox]  # herdd.py:15290/:15314/:15348
                                      # (setdefault+append) — REPLACEMENT_STATE_KEYS
    # defense-controller observations (AUTOBID_DESIGN, 2026-08-09)
    entry_floor: float | None         # herdd.py:17469 — the PRE-RENT market
                                      # floor, unrecoverable after launch (#73).
                                      # REPLACEMENT_STATE_KEYS.
    p_alt: float | None               # herdd.py:13214 — REPLACEMENT_STATE_KEYS
    p_alt_machine: Any                # herdd.py:13216
    p_alt_ts: float | None            # herdd.py:13215/:13220 — travels WITH the
                                      # price so _job_palt_fresh survives a
                                      # restart. REPLACEMENT_STATE_KEYS.
    # replacement-ceiling re-pricing (REPLACEMENT_CEILING_WEDGE_2026-08-24).
    # The first pair is a MARKET READ taken from a refusal — on the pull lane it
    # is the only one available, because the ceiling is pushed into the offer
    # search and an unaffordable market comes back empty. The rest is the
    # consecutive-refusal streak the derived `replacement_wedged` alarm reads.
    replacement_market_floor: float | None      # supervise/replacement.py:1238
    replacement_market_floor_ts: float | None   # supervise/replacement.py:1239
    replacement_refusals: int                   # supervise/replacement.py:2591
    replacement_refusals_since: float | None    # supervise/replacement.py:2593
    replacement_refusal_reason: str | None      # supervise/replacement.py:2594
    replacement_refusal_ceiling: float | None   # supervise/replacement.py:2595
    replacement_ceiling_last: float | None      # supervise/replacement.py:2557 —
                                      # the once-per-CHANGE journal latch. NOT
                                      # durable on purpose: it only suppresses a
                                      # duplicate line, and a restart re-emitting
                                      # the current ceiling once is the harmless
                                      # direction for a spend bound.
    # The supervised job's TICKET CONFIG, `(job_id, config|None)`, memoized so a
    # stuck eviction re-running every ~50 s costs one B2 `cat` and not one per
    # tick. Deliberately NOT durable: it is a cache of a remote object, and a
    # restart re-reading it once is free.
    _job_cfg_cache: Any               # supervise/replacement.py:2888
    dph_base_missing_said: bool       # herdd.py:17420
    disk_shortfall_said: bool         # once-per-box: vast allocated less
                                      # container disk than the launch asked for
    # vast's own outbid notification, as EVIDENCE (notify S2b, NOTIFY_DESIGN
    # §6.3, 2026-08-16). FOUR keys with three different lifetimes, and the
    # differences are the design, not an accident:
    #   * `notify_rows` is the DRIVER's inbox slice. fleetd writes it every tick
    #     when the deploy gate is armed and POPS it when the gate is off, which
    #     is why the inline `job supervise` CLI never sets it and the ladder is
    #     exactly its pre-S2b self there.
    #   * `notify_matched` / `notify_quote_said` are PER-EVICTION-CYCLE latches,
    #     cleared on return-to-live and on every box swap.
    #   * `notify_consumed_ids` is DEDUP MEMORY and deliberately survives both a
    #     return-to-live and a deploy-gate flap; only a box swap retires it.
    # `notify_matched` and `notify_consumed_ids` are REPLACEMENT_STATE_KEYS —
    # durable, so a hand-edited state.json can hand back any JSON shape, which
    # is why both readers are isinstance-guarded.
    notify_rows: list[dict[str, Any]]  # fleetd.py:2570; POPPED fleetd.py:2566
                                      # when the deploy gate is off
    notify_matched: dict[str, Any]    # herdd.py:17344; POPPED :17374 and
                                      # fleetd.py:2567 — REPLACEMENT_STATE_KEYS
    notify_consumed_ids: list[str]    # herdd.py:17339; POPPED :17385 (box swap
                                      # ONLY) — REPLACEMENT_STATE_KEYS
    notify_quote_said: bool           # herdd.py:17527; POPPED :17375 and
                                      # fleetd.py:2568
    # queue view + market read
    pending_views: list[dict[str, Any]]   # herdd.py:17633 — the per-tick job
                                      # views every risk metric reads
    _market_read: Any                 # herdd.py:17221
    last_error: str | None            # herdd.py:13265 and 11 more
    stop_intent: bool                 # NEVER written in herdd — the DRIVER
                                      # seeds it (:17521 reads it) to say a
                                      # journaled operator/fleetd stop exists
    rescue_put_failures: int          # herdd.py:18007; POPPED :18003/:18011
    # OUT-PARAMETER channels: absent unless the event fired this tick
    _handoff_completed_iid: str       # written herdd.py:16694, POPPED :17847
    _handoff_completed_dph: float     # written herdd.py:16695, POPPED :17860
    # QUEUE channels drained by fleetd with .pop() — absence is "empty"
    ladder_journal: list[Any]         # herdd.py:16448 (setdefault+append);
                                      # POPPED fleetd.py:3080
    handoff_journal: list[Any]        # herdd.py:16417 (setdefault+append);
                                      # POPPED fleetd.py:3116

    # --- written by ladder_core.py, which names this dict `ctx` --------------
    # WIRE FORMAT — see RunState's note. `bid_history` is also a member of
    # REPLACEMENT_STATE_KEYS on this lane.
    bid_history: list[list[Any]]      # ladder_core.py:129-160
    self_floor_at: float | None       # ladder_core.py:399-400; POPPED at :179
    self_floor_since: float | None    # ladder_core.py:408-413; POPPED at :180
    self_floor_sustained_said: bool   # ladder_core.py:414; POPPED at :181
    _scaled_floor_said: bool          # ladder_core.py:368-369; POPPED at :372
    on_demand_last: float | None      # POPPED by ladder_core.py:215 on a box swap


# --------------------------------------------------------------------------- #
# the two durable PROJECTIONS — the only parts of jc/st that outlive the process
# --------------------------------------------------------------------------- #

#: RE-EXPORT of `vastlib.fleet.state.REPLACEMENT_STATE_KEYS`, which became THE
#: DEFINITION when `fleet/state.py` landed (plan §8 step 5; integrator ruling
#: 2026-08-16). This module deliberately holds NO literal of its own any more:
#: the mirror test was written because a copy that can drift is worse than no
#: copy, and a third copy would have been the drift it was guarding against.
#: `fleetd.py` keeps its live copy until step 6 (add-only), so
#: `test_vastlib_supervise_state.py` still AST-parses `fleetd.py` AND asserts
#: object identity with the definition.
#:
#: WIRE FORMAT. These `jc` keys are persisted under `w["replacement"]` in
#: fleetd's `state.json` by `_replacement_state_persist` (fleetd.py:484-501) and
#: restored by `_replacement_state_restore` (:430-449). Every one is a SPEND or
#: WALL-TIME bound: a restart that forgot `replacements` hands the ladder a
#: fresh budget of autonomous rentals, one that forgot `launch_dph_anchor`
#: re-derives the ceiling from the REPLACEMENT's price (ratcheting 2x per
#: restart), one that forgot `rebid_rungs` turns a bounded 15-minute stall into
#: an unbounded loop of them.
#:
#: TWO COERCIONS, and they are asymmetric:
#:   * `evicted_machines` — a python `set` in memory, persisted as
#:     `sorted(v)` (a list, because state.json is JSON), restored with
#:     `set(v)`. It is the ONLY key the restore special-cases.
#:   * `evicted_machine_ts` — keys are machine ids STRINGIFIED. There is no
#:     restore-side coercion: an int-keyed in-memory dict and a string-keyed
#:     restored one behave identically because every lookup goes through
#:     `str(m)`. A watch persisted before this key existed restores the set with
#:     no sidecar and every entry reads permanent — degraded, never wrong.
#: The 18 keys, in order, are enumerated with their per-key spend rationale at
#: the definition site (`vastlib/fleet/state.py`). Two of them are worth
#: repeating here because the TypedDicts above are annotated against them:
#: notify S2b (2026-08-16) added `notify_matched` (the per-cycle latch) and
#: `notify_consumed_ids` (the dedup memory), both durable because a daemon
#: restart mid-eviction must not re-price a spent row; `rescue_deadline` /
#: `rescue_put_failures` joined in the same slice, because a restart used to
#: re-arm a spent ONE-SHOT rescue.
REPLACEMENT_STATE_KEYS: tuple[str, ...] = fleet_state.REPLACEMENT_STATE_KEYS

#: RE-EXPORT of `vastlib.fleet.state.RUN_STATE_KEYS`, same reasoning and same
#: drift test as above.
#:
#: WIRE FORMAT, and a standing oddity worth stating plainly: the run lane's
#: ENTIRE durable state is one key, and that key is written by `ladder_core.py`
#: — a module that never declares it and does not import either lane. `st`'s own
#: constructor contributes nothing to what survives a daemon restart.
#:
#: `last_bid` is deliberately NOT persisted (review 2026-08-10, F2): a restart
#: re-derives it from the box's own `dph_base` via the reconcile path, which
#: cannot go stale, while a persisted belief could.
RUN_STATE_KEYS: tuple[str, ...] = fleet_state.RUN_STATE_KEYS


# --------------------------------------------------------------------------- #
# absence-as-state, and the keys that must never become 0.0
# --------------------------------------------------------------------------- #

#: Keys deleted at runtime, by dict identity. For every one of them ABSENCE is a
#: distinct, load-bearing state — re-arm a fresh sampler window, retract a
#: standing refusal, drain an empty journal queue, forget a completed migration.
#: 28 `.pop()` sites at rev 86840142 — 16 in herdd.py (:14164, :14222, :14223,
#: :14236, :16192, :16193, :16206, :16984, :17582, :17583, :17847, :17860,
#: :17880, :17906, :18003, :18011), 2 in fleetd.py (:3080, :3116) and 10 in
#: ladder_core.py (:179-181, :215, :372, :408-409, :432, :435-436) on whichever
#: dict it was handed as `ctx` — plus the two whole-dict `hf.clear()` resets at
#: herdd.py:8795 / :16457.
#:
#: This is the single hardest constraint on ever converting these to
#: dataclasses: a dataclass has no key-absent state.
POPPED_KEYS: dict[str, tuple[str, ...]] = {
    "RunState": (
        "on_demand_last",
        "self_floor_at",
        "self_floor_since",
        "self_floor_sustained_said",
        "_scaled_floor_said",
    ),
    "JobContext": (
        "pull_sampler",
        "pull_sampler_iid",
        "evicted_announced",
        # popped with it at all four seams — a stale clock would date the NEXT
        # eviction from the PREVIOUS one
        "evicted_class",
        "evicted_since",
        "rescue_put_failures",
        # per-BOX, so both box-swap paths retire it: the replacement can be
        # short-changed too, and a latch from the dead box would eat that warning
        "disk_shortfall_said",
        # notify S2b. Three different absences: `notify_rows` absent = the
        # deploy gate is OFF (fleetd.py:2566), `notify_matched` /
        # `notify_quote_said` absent = this eviction cycle has not matched a row
        # (or the cycle ended), `notify_consumed_ids` absent = a BOX SWAP
        # retired the memory, which is the only thing that may.
        "notify_rows",
        "notify_matched",
        "notify_quote_said",
        "notify_consumed_ids",
        "_handoff_completed_iid",
        "_handoff_completed_dph",
        "ladder_journal",
        "handoff_journal",
        "on_demand_last",
        "self_floor_at",
        "self_floor_since",
        "self_floor_sustained_said",
        "_scaled_floor_said",
        # the serve identity verdict + its withdrawal latch. Absence means "this
        # watch is not checking", and it has to be reachable from BOTH sides:
        # an operator re-`watch`ing without --artifact says stop checking, and a
        # verdict (or a latch) that outlived the pin it was made against would
        # be an alarm with nothing able to retract it — derived alarms have no
        # clear path by construction.
        "serve_identity",
        "serve_identity_condemned",
    ),
    "HandoffCarry": ("refuse_sig",),
}

#: Tri-state float keys: `None` means UNKNOWN, and UNKNOWN REFUSES. Never
#: default one of these to `0.0`, never write `or 0.0` at a read site, and never
#: give one a dataclass field default — that is defect #67 (the 2026-08-08
#: 22:17Z incident, ~5x-inflated projected saving) re-opened.
#:
#: The per-site read defaults are NOT uniform and the asymmetry is deliberate;
#: `_job_handoff_build_state` reads `remaining_wall_h` with a 0.0 default and
#: `min_running_eta_s` bare, in adjacent lines.
TRISTATE_FLOAT_KEYS: dict[str, tuple[str, ...]] = {
    "RunState": ("remaining_wall_h",),
    "JobContext": ("remaining_wall_h", "timeout_ceiling_h", "min_running_eta_s"),
    "HandoffSnapshot": ("remaining_wall_h", "min_running_eta_s"),
}

#: The keys `ladder_core.py` writes into whichever dict it received as `ctx`.
#: WIRE FORMAT, pinned by ladder_core.py:50-56 verbatim: "key names here are a
#: wire format … Renaming one silently drops durable state across a daemon
#: restart." `ladder_core` is a stdlib-only leaf that sibling-imports
#: `bidpolicy` (Zone S) and must NOT import `vastlib`; its identity is pinned by
#: `test_ladder_core.py` and it is NOT ported in this step.
LADDER_CORE_CTX_KEYS: tuple[str, ...] = (
    "bid_history",
    "self_floor_at",
    "self_floor_since",
    "self_floor_sustained_said",
    "_scaled_floor_said",
    "last_bid",
    "first_seen_dph",
)
