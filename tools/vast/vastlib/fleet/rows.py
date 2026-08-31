"""vastlib.fleet.rows — what `fleet status` is allowed to say, and what it must not.

Why this exists
---------------
`fleetd.py` computes its operator-facing tables inline, next to the tick loop
that mutates the fleet. Every function in this module is a PURE fold over the
already-persisted `state.json` document (plus, for `reconcile_rows`, one
instance listing the caller already fetched): no API call, no socket, no clock
read, no env read except the one config lookup `ceiling_rows` needs for its
fail-closed default. That is the whole point of separating them — the numbers an
operator uses to decide whether a box is still costing them money can be pinned
by a unit test instead of by a live fleet.

The conventions this module is not allowed to "clean up"
--------------------------------------------------------
* **Explicit `None` is the render contract, not a gap to fill.** `stray_rows`
  emits `spend_usd=None` / `budget_usd=None` and `reconcile_rows` emits
  `divergence_pct=None` whenever the upper bound is falsy (INCLUDING `ub == 0`).
  These say "fleetd has no figure for this box", and turning one into `0.0`
  re-prints exactly the reassuring number the retention `live_*` fields were
  added to kill (box 47833510, 2026-08-16). `profile="-"` and
  `state="UNWATCHED"` are literals for the same reason: `fleet status`
  concatenates stray rows with watch rows and the columns have to line up.
* **`_num` is NOT `core.models._num_dph`.** This file calls both. `_num`
  (ceiling arithmetic) rejects NaN and ±inf and answers `None`; a NaN cap
  compares false against every bound, so it would read as a ceiling no spend can
  breach — the "unlimited" the ceiling ledger refuses to express.
  `models._num_dph` (`reconcile_rows`, price coercion) is a different function
  with different rejections. They are deliberately not merged, and the two
  call sites are not interchangeable.
* **The alarm KEY is schema.** `retention_alarms` emits `retention:<iid>:live`,
  and that string is the identity fleetd raises/resolves and dedups against
  across a daemon restart (it lands in `state["alarms"]` and in the journal's
  `alarm_raised` / `alarm_resolved` rows). Reformatting it silently double-raises
  every retention alarm once.
* **`RETENTION_NOTES` and `_RETENTION_FATE` are text an operator acts on.**
  `_RETENTION_FATE`'s clause is spliced verbatim into the `jobs_replaced`
  journal note — i.e. it is journal-schema-adjacent, not prose — and
  `RETENTION_NOTES` substrings are asserted by the retention tests. Nothing here
  gets reflowed.
* **`workload_evidence`'s ORDER is the contract.** booting -> boot age -> jobd
  heartbeat -> zombie short-circuit -> label tokens -> jobs-lane box. The zombie
  test sits ABOVE the label test so a MEASURED dead workload is never rescued by
  a mere label; moving it is a safety-net regression that no assertion on the
  return value would catch.
* **`watch_box_iid`'s `"None"` guard is a state.json fact.** A `str(None)`
  round-trip through the state document leaves the four-character string
  `"None"` where an id should be, and reading it as an id is how two spend-control
  incidents happened (2026-08-05, 46866652 -> 46867184 -> 46867793). The guard
  belongs next to the schema, which is why this symbol is here and not in the
  daemon.

What is deliberately NOT here
-----------------------------
* **`iso`.** It lives in `fleet.state` (it is the journal's `ts_iso` producer)
  and is called module-attribute-style, `fleet_state.iso(...)`. It is NOT
  `supervise.journal._iso_z`: that one answers `None` for a falsy timestamp
  while this one raises, and every caller here guards first. Merging them turns
  a would-be crash into a silent `None` in a journal field.
* **`_Policy` / `_redirect_policy` / `make_policy`.** Textually adjacent and
  nearly pure, but they are WATCH-REGISTRATION policy: they read
  `vastconf.jobs_handoff_enabled()` and the two `*_POLICY_DEFAULTS` namespaces,
  and `make_policy` re-applies the SAFE-OFF handoff switch on every rebuild.
  A decision the daemon makes, not a row. They travel with `fleet/daemon.py`.
* **The guard lattice itself.** `workload_evidence` READS it —
  `health.GUARD_BOOTING` and `health.verdict_is_zombie` — and never re-derives
  membership. fleetd's own re-derived presentation of the same verdicts is
  daemon-side (`_health_alarm_msg`); one lattice, in `boxes.health`.
* **Alarm latching / raising.** These functions return alarm TUPLES; whether an
  alarm latches, journals, or is already lit is `fleet/daemon.py`'s question.
  A retention alarm is self-retracting by construction (it is derived from
  `live_since_ts` every read), so it is never stored as a decision.

Provenance: verbatim-with-types move of 19 symbols from `tools/vast/fleetd.py`
(plan §8 step 5, 2026-08-16), each carrying its `# moved-from:` marker.
ADD-ONLY: `fleetd.py` keeps its live copies until step 6, so nothing in the repo
calls this module yet except `tools/vast/test_vastlib_fleet_rows.py`, which pins
the two implementations against each other.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Final, TypedDict

from vastlib.boxes import health
from vastlib.core import config, models
from vastlib.fleet import state as fleet_state

#: The persisted `state.json` document (or any mapping shaped like it). Typed
#: structurally rather than as a TypedDict because every function here reads it
#: DEFENSIVELY — `state.get(k) or {}`, `isinstance(w, dict)` — precisely so a
#: state file written by an older daemon (or quarantined and rebuilt empty)
#: renders rather than raises. The authoritative key inventory is in
#: `fleet/state.py`.
StateDoc = Mapping[str, Any]


# --------------------------------------------------------------------------- #
# the ceiling ledger's read side — a spend cap that OUTLIVES the watch that
# armed it. The ledger itself (`state["ceilings"]`, `state["ceiling_by_box"]`)
# is written by the daemon; everything below only reads it, and reads it
# FAIL-CLOSED: an unreadable ceiling resolves to the conservative provisional
# default, never to "unlimited". "Unlimited" is not expressible by any
# automatic path — the only code that can produce a `budget_usd` of None is an
# operator typing `--profile bare` with no `--budget`.
# --------------------------------------------------------------------------- #
# moved-from: fleetd._HANDOFF_LABEL_RE
_HANDOFF_LABEL_RE: re.Pattern[str] | None = None   # compiled lazily; see below


# moved-from: fleetd.handoff_predecessor
def handoff_predecessor(label: str | None) -> str | None:
    """The PRIMARY box id encoded in a handoff understudy's label, or None.

    The jobs ladder labels the box it rents to take over a migration
    `job:<primary_iid>:handoff` (observed verbatim on 47215526, 2026-08-08).
    That label is the only control-plane evidence that links the successor to
    the watch whose ceiling it should draw on when the ladder's own
    `understudy_iid` never reached us (a daemon restart mid-migration, or a
    handoff armed by an inline `job supervise`). Control-plane only — no on-box
    probe (owner ruling 2026-08-02)."""
    global _HANDOFF_LABEL_RE
    if _HANDOFF_LABEL_RE is None:
        import re
        _HANDOFF_LABEL_RE = re.compile(r"^job:(\d+):handoff$")
    m = _HANDOFF_LABEL_RE.match((label or "").strip())
    return m.group(1) if m else None


# moved-from: fleetd._num
def _num(v: Any) -> float | None:  # noqa: ANN401 — an arbitrary state.json leaf
    """A finite float, or None. `v != v` is the NaN test — a NaN cap compares
    false against every bound, so it would read as a ceiling no spend can
    breach, which is exactly the "unlimited" this design refuses to express.

    NOT `core.models._num_dph`, which this module also calls: that one is the
    price coercion and rejects a different set of inputs. See the module
    docstring."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (f != f or f in (float("inf"), float("-inf"))) else f


# moved-from: fleetd.normalize_ceiling
def normalize_ceiling(rec: object, default_cap: float) -> tuple[float, float, str | None]:
    """PURE. `(cap_usd, spend_usd, degraded_reason)` for a ledger record.

    FAIL-CLOSED: the returned cap is ALWAYS a positive float. A record that is
    not a dict, or whose `cap_usd` is missing / None / non-numeric / NaN /
    <= 0, yields `default_cap` and a reason string the caller journals. An
    unreadable spend yields 0.0 — we genuinely do not know the box spent
    anything, and the load-bearing half of "fail closed" is the CAP.

    The reason strings are journaled VERBATIM, so their text is an observable
    contract, not a diagnostic nicety.
    """
    if not isinstance(rec, dict):
        return default_cap, 0.0, "ceiling record is not an object"
    cap = _num(rec.get("cap_usd"))
    spend = _num(rec.get("spend_usd"))
    reason = None
    if cap is None:
        reason = f"cap_usd={rec.get('cap_usd')!r} is unreadable"
        cap = default_cap
    elif cap <= 0:
        reason = f"cap_usd={cap} is not positive"
        cap = default_cap
    if spend is None or spend < 0:
        spend = 0.0
    return cap, spend, reason


class CeilingRow(TypedDict):
    """One durable ceiling, as `fleet status` / `fleet ceilings` renders it. The
    key set is a presentation contract for the fleet CLI."""

    ceiling_id: str
    cap_usd: float
    spend_usd: float
    remaining_usd: float
    source: Any
    degraded: str | None
    origin_target: Any
    requester: Any
    epochs: Any
    members: Any
    last_verdict: Any
    live_boxes: list[Any]


# moved-from: fleetd.ceiling_rows
def ceiling_rows(state: StateDoc, now: float | None = None) -> list[CeilingRow]:
    """Every durable ceiling, for `fleet status` / `fleet ceilings`. A ceiling
    with no live watch is the interesting case: it is the headroom a future
    re-arm or auto-adoption will inherit, and before this existed that number
    was silently 0.

    `now` is accepted and unused — it is part of the row-builder call shape the
    daemon uses uniformly, and dropping it would be the one signature change in
    this file."""
    live: dict[str, list[Any]] = {}
    for target, w in (state.get("watches") or {}).items():
        cid = w.get("ceiling_id")
        if cid:
            live.setdefault(str(cid), []).append(w.get("iid") or target)
    rows: list[CeilingRow] = []
    default_cap = config.fleetd_adopt_default_budget_usd()
    for cid, rec in sorted((state.get("ceilings") or {}).items()):
        cap, spend, degraded = normalize_ceiling(rec, default_cap)
        rows.append({"ceiling_id": cid, "cap_usd": round(cap, 4),
                     "spend_usd": round(spend, 4),
                     "remaining_usd": round(cap - spend, 4),
                     "source": rec.get("source") if isinstance(rec, dict) else None,
                     "degraded": degraded,
                     "origin_target": (rec.get("origin_target")
                                       if isinstance(rec, dict) else None),
                     "requester": (rec.get("requester")
                                   if isinstance(rec, dict) else None),
                     "epochs": (rec.get("epochs") if isinstance(rec, dict) else None),
                     "members": (rec.get("members") if isinstance(rec, dict) else None),
                     "last_verdict": (rec.get("last_verdict")
                                      if isinstance(rec, dict) else None),
                     "live_boxes": live.get(cid) or []})
    return rows


# --------------------------------------------------------------------------- #
# eviction retention — the cost disclosure a retained box owes its operator
# --------------------------------------------------------------------------- #
# moved-from: fleetd.RETENTION_NOTES
RETENTION_NOTES: Final[dict[str, str]] = {
    "retained": "EVICTED box held for salvage of state that never reached B2; "
                "it carries a self-expiring keep label and bills ALLOCATED disk "
                "until the reaper takes it (FLEETD_DESIGN 'Salvaging a "
                "retained box')",
    "expired": "retention window closed — the keep label no longer holds the "
               "box; `herdd reap` reclaims it on its next 15-minute pass",
    "reaped": "retention ended as designed: the box was reclaimed at/after its "
              "deadline",
    "retention_lost": "box VANISHED before its deadline — a spot host reclaimed "
                      "the stopped bid instance and its disk; retention on an "
                      "evicted spot box is BEST-EFFORT (box 44612403)",
    "already_gone": "the lost box was never retainable: already out of the "
                    "listing when the replacement landed",
    "destroyed": "backstop destroy: the window had been closed for over the "
                 "grace period and `herdd reap` had not reclaimed it",
    "destroy_failed": "DESTROY FAILED — the box still bills storage; destroy it "
                      "by hand",
}


# moved-from: fleetd._retention_status_map
def _retention_status_map(jc: Mapping[str, Any]) -> dict[str, Any]:
    """`{iid: record}` for the eviction-retention records on a jobs context.

    Reads a JOBS-CONTEXT dict (`supervise.state.JobContext`), not `state.json`
    — the one function in this file whose input is the supervise-side shape."""
    return {str(r.get("iid")): r for r in (jc.get("retained_boxes") or [])
            if r.get("iid") is not None}


# What became of the box a replacement replaced, as (status, clause). The clause
# goes verbatim into `jobs_replaced`'s note, which is the line an operator reads
# when they want to know whether the old box is still costing them money.
# moved-from: fleetd._RETENTION_FATE
_RETENTION_FATE: Final[dict[str, str]] = {
    "retained": "the old box RETAINED (stopped, bid pinned) for salvage — it "
                "still bills allocated disk until its keep label expires",
    "expired": "the old box's retention window has CLOSED — `herdd reap` "
               "reclaims it on its next pass",
    "destroyed": "the old box destroyed",
    "destroy_failed": "the old box's DESTROY FAILED — it still bills; destroy "
                      "it by hand",
    "already_gone": "the old box was already out of the listing — nothing to "
                    "destroy",
    "reaped": "the old box has since been reclaimed",
    "retention_lost": "the old box vanished before its deadline (spot reclaim)",
}


# moved-from: fleetd._retention_fate
def _retention_fate(ret_map: Mapping[str, Any] | None,
                    iid: object) -> tuple[str | None, str]:
    """PURE. `(status, clause)` for the box a replacement just replaced. Unknown
    (no retention record — an SLA relaunch, a pull-reschedule, a shape this
    function has not met) answers `(None, "the old box handed off")` rather than
    naming an outcome it cannot see: an honest vague line beats a confident
    wrong one, which is the whole point of this helper.

    The default clause is load-bearing, not a stub."""
    rec = (ret_map or {}).get(str(iid)) or {}
    # `status: Any` (rather than the inferred `Any | None`) is a mypy-strict
    # annotation only: the lookup below must stay a plain `.get` with the
    # default clause, because an UNKNOWN status is exactly the case the default
    # exists for.
    status: Any = rec.get("status")
    return status, _RETENTION_FATE.get(status, "the old box handed off")


class RetentionRow(TypedDict):
    """One retained (or awaiting-reclaim) box, as `fleet status` renders it.

    `est_cost_usd` is ALLOCATED-DISK billing. The `live_*` triple is what says
    whether that is still the truth: a retained box that is LIVE bills the GPU
    rate, and the row has to show it or the table repeats the same reassuring
    number the incident was hiding behind."""

    iid: Any
    target: str
    status: Any
    eviction_class: Any
    deadline: str | None
    left_s: float | None
    est_cost_usd: Any
    est_cost_hi_usd: Any
    keep_labeled: Any
    live_since: str | None
    live_dph: Any
    live_cost_usd: Any
    replacement_iid: Any


# moved-from: fleetd.retention_rows
def retention_rows(state: StateDoc, now: float) -> list[RetentionRow]:
    """Every RETAINED (or awaiting-reclaim) box across all watches, for
    `fleet status`. A retained box is one a human did not choose to keep and may
    not know is still billing — surfacing it is half of the cost disclosure the
    retention window owes (the journal is the other half)."""
    rows: list[RetentionRow] = []
    for target, w in sorted((state.get("watches") or {}).items()):
        for rec in ((w.get("replacement") or {}).get("retained_boxes") or []):
            if rec.get("status") not in ("retained", "expired"):
                continue
            dl = rec.get("deadline_ts")
            rows.append({"iid": rec.get("iid"), "target": target,
                         "status": rec.get("status"),
                         "eviction_class": rec.get("class"),
                         "deadline": fleet_state.iso(dl) if dl else None,
                         "left_s": round(dl - now, 1) if dl else None,
                         "est_cost_usd": rec.get("cost_usd"),
                         "est_cost_hi_usd": rec.get("cost_hi_usd"),
                         "keep_labeled": rec.get("keep_labeled"),
                         # The disclosed cost above is ALLOCATED DISK. These
                         # three say whether that is still the truth: a retained
                         # box that is LIVE is billing the GPU rate, and the row
                         # has to show it or `fleet status` repeats the same
                         # reassuring number the incident was hiding behind.
                         "live_since": (fleet_state.iso(rec["live_since_ts"])
                                        if rec.get("live_since_ts") else None),
                         "live_dph": rec.get("live_dph"),
                         "live_cost_usd": rec.get("live_cost_usd"),
                         "replacement_iid": rec.get("replacement_iid")})
    return rows


# moved-from: fleetd.retention_alarms
def retention_alarms(state: StateDoc, now: float) -> list[tuple[str, str]]:
    """PURE. A standing alarm per RETAINED box that is currently LIVE.

    Derived from `live_since_ts`, which `_job_retention_liveness` sets and
    clears against the instance listing every tick — so the alarm appears the
    tick a resurrection is seen and retracts itself the tick the box is
    stopped, with nobody acking anything.

    The gap this closes (2026-08-16, box 47833510): a retained box came back
    RUNNING and NOTHING alarmed. It is exempt from `herdd reap` by its own
    `keep:` label; the retention sweep skipped it as "a human mid-salvage"; and
    the stray sweep dropped its record every cycle because auto-adopt is refused
    by the watch filed under the same id. Three safety nets, all of which had a
    reason not to look.

    The `retention:<iid>:live` KEY is schema — it is the identity the daemon
    raises, resolves and dedups against across a restart."""
    out: list[tuple[str, str]] = []
    for target, w in sorted((state.get("watches") or {}).items()):
        if not isinstance(w, dict):
            continue
        for rec in ((w.get("replacement") or {}).get("retained_boxes") or []):
            if not isinstance(rec, dict) or not rec.get("live_since_ts"):
                continue
            if rec.get("status") not in ("retained", "expired"):
                continue
            iid = rec.get("iid")
            dph = rec.get("live_dph")
            usd = rec.get("live_cost_usd")
            mult = rec.get("live_multiple")
            rate = (f"${dph:.4f}/hr" if isinstance(dph, (int, float))
                    else "an unknown rate")
            spent = (f"${usd:.2f} so far" if isinstance(usd, (int, float))
                     else "cost unknown")
            over = (f", {mult:g}x the disclosed storage-only rate"
                    if mult else "")
            out.append((f"retention:{iid}:live",
                        f"{iid}: RETAINED box is RUNNING again — {rate}, "
                        f"{spent}{over}, no queue and no watch "
                        f"({int(now - rec['live_since_ts'])}s). The ladder has "
                        f"re-parked it {int(rec.get('requiesces') or 0)}x; if it "
                        f"keeps coming back, salvage what you need and "
                        f"`fleet destroy {iid}`"))
    return out


# --------------------------------------------------------------------------- #
# recovery-in-flight (the `fleet restart` guard) — recalibration 2026-08-09,
# item C.
#
# 2026-08-08 23:24:37Z: an unrelated fleetd redeploy landed two minutes after a
# human destroyed box 47214941, in the middle of a recovery chain the ladder was
# still driving. The restarted daemon reconciled the stale watch and ran its OWN
# chain — condemn 47219058, launch 47219872, move a ticket, destroy — duplicating
# work that was already in flight. Cost: ~$0.9 of duplicated recovery and two
# actors on one job.
#
# `systemctl restart` is not the problem; restarting BLIND is. The state file
# already knows: the ladder's per-eviction-cycle counters are durable
# (REPLACEMENT_STATE_KEYS) precisely so a restart cannot forget them, which means
# the same file can be read one moment EARLIER to say "something is mid-recovery,
# are you sure". This is a pure fold over state.json — no API, no daemon call, so
# it works when the daemon is wedged, which is exactly when a restart is typed.
# --------------------------------------------------------------------------- #
# moved-from: fleetd.UNWATCHED_STALE_S
UNWATCHED_STALE_S: Final = 900.0  # a stray record whose last LIVE sighting is older
                                  # than this is history, not a fleet row (item D)


class RecoveryRow(TypedDict):
    """One thing a `fleet restart` would interrupt. `kind` is one of
    `rebid_ladder` / `resume_in_place` / `replacement` / `unrecoverable` /
    `destroy_queued`; `detail` is the operator-facing sentence."""

    target: str
    iid: Any
    kind: str
    detail: str


# moved-from: fleetd.recoveries_in_flight
def recoveries_in_flight(state: StateDoc) -> list[RecoveryRow]:
    """PURE. What a `fleet restart` would interrupt, as a list of
    `{target, iid, kind, detail}` — empty when nothing is mid-recovery.

    The kinds, and the concrete durable field each is read from:

      * `rebid_ladder`   — `replacement.rebid_rungs > 0`. Per EVICTION CYCLE and
        cleared on any return to live, so a non-zero value means a rung has been
        placed and its REBID_WAIT_S window has not resolved.
      * `resume_in_place`— `replacement.resume_tries > 0`. Same lifecycle: rung
        zero has issued a `start` and the box has not come back yet.
      * `replacement`    — a `retained_boxes` record still `retained`: the ladder
        rented a replacement and is holding the evicted box's disk for salvage.
        The window is bounded and journaled, but a restart mid-window is how the
        duplicate chain happened.
      * `unrecoverable`  — `unrecoverable_since` set: the ladder gave up and the
        watch is being KEPT so an auto-resume re-arms it. A restart here
        re-initialises the ladder and re-runs its decisions from scratch.
      * `destroy_queued` — an entry in `state["destroys"]`: a destroy the daemon
        re-checks every tick and executes at most once.

    Handoff phase is deliberately NOT here: it lives in runtime state only and
    never reaches state.json, so this fold cannot see it. Handoff is SAFE-OFF
    fleet-wide; if it is ever re-enabled its phase has to be persisted before
    this guard can claim to cover it. Said out loud rather than implied, because
    a guard that silently does not cover a case is worse than no guard."""
    out: list[RecoveryRow] = []
    for target, w in sorted((state.get("watches") or {}).items()):
        if not isinstance(w, dict):
            continue
        iid = w.get("iid") or target
        repl = w.get("replacement") or {}
        if not isinstance(repl, dict):
            repl = {}
        rungs = repl.get("rebid_rungs") or 0
        if rungs:
            out.append({"target": target, "iid": iid, "kind": "rebid_ladder",
                        "detail": f"{int(rungs)} re-bid rung(s) placed this "
                                  f"eviction cycle, waiting on the box"})
        tries = repl.get("resume_tries") or 0
        if tries:
            out.append({"target": target, "iid": iid, "kind": "resume_in_place",
                        "detail": f"{int(tries)} in-place start(s) issued, "
                                  f"waiting for the box to come back"})
        for rec in (repl.get("retained_boxes") or []):
            if not isinstance(rec, dict) or rec.get("status") != "retained":
                continue
            out.append({"target": target, "iid": rec.get("iid"),
                        "kind": "replacement",
                        "detail": f"evicted box held for salvage, replaced by "
                                  f"{rec.get('replacement_iid')}"})
        if w.get("unrecoverable_since"):
            out.append({"target": target, "iid": iid, "kind": "unrecoverable",
                        "detail": "the ladder gave up and the watch is being "
                                  "KEPT so an auto-resume re-arms it; a restart "
                                  "re-runs its decisions from scratch"})
    for iid, req in sorted((state.get("destroys") or {}).items()):
        out.append({"target": iid, "iid": iid, "kind": "destroy_queued",
                    "detail": f"destroy queued "
                              f"(when={(req or {}).get('when')})"})
    return out


class StrayRow(TypedDict):
    """One UNWATCHED box, shaped so `fleet status` can concatenate it with the
    watch rows. `profile`, `state`, `spend_usd` and `budget_usd` are FIXED
    literals — see the module docstring on why the two Nones stay None."""

    target: str
    iid: str
    profile: str
    state: str
    spend_usd: None
    budget_usd: None
    last_seen_s: float
    paused: bool
    last_action: str | None


# moved-from: fleetd.stray_rows
def stray_rows(state: StateDoc, now: float,
               stale_s: float | None = None) -> list[StrayRow]:
    """PURE. The UNWATCHED rows `fleet status` should render — which is NOT every
    record in `state["strays"]` (recalibration 2026-08-09, item D).

    `_tick_strays` only iterates instances the API currently lists, so it can
    create a stray record but never prunes one for a box that has LEFT the
    listing. A destroyed box therefore keeps its record forever and renders as an
    `UNWATCHED  -  $0.000` row indefinitely: a fleet table padded with boxes that
    stopped existing hours ago, which is how a real stray gets lost in the noise.

    A record earns a row only if it was seen LIVE within `stale_s`. `live_ts` is
    stamped by `_tick_strays` on every reconcile that saw the box live and is
    already the alarm-derivation gate for the same reason (a record outliving its
    box must not alarm), so this reuses the existing evidence rather than adding a
    second clock. A record with NO `live_ts` predates that field and is treated as
    stale — the conservative direction, since it cannot have been seen live by a
    daemon that stamps it.

    Boxes with a queued destroy never appear at all: the operator has already
    decided that box's fate, and offering it as an unwatched stray invites a
    second decision on it."""
    stale_s = UNWATCHED_STALE_S if stale_s is None else stale_s
    doomed = {str(k) for k in (state.get("destroys") or {})}
    rows: list[StrayRow] = []
    for iid, s in sorted((state.get("strays") or {}).items()):
        if not isinstance(s, dict) or str(iid) in doomed:
            continue
        live_ts = s.get("live_ts")
        if not isinstance(live_ts, (int, float)) or (now - live_ts) > stale_s:
            continue
        rows.append({"target": iid, "iid": iid, "profile": "-",
                     "state": "UNWATCHED", "spend_usd": None,
                     "budget_usd": None,
                     "last_seen_s": round(now - live_ts, 1),
                     "paused": bool(s.get("paused_until")),
                     "last_action": "parked" if s.get("parked_ts") else None})
    return rows


# --------------------------------------------------------------------------- #
# spend reconciliation (`fleet spend --reconcile`) — recalibration 2026-08-09,
# item E.
#
# The 2026-08-08 night's watch accounting saw ~$4.09 of a ~$5.66 invoice. Two
# named causes, both structural rather than arithmetic:
#
#   * fleetd accrues from WATCH ADOPTION, and the box bills from `start_date`.
#     Every boot/loading window before a `fleet watch` lands is invisible —
#     47214941 was watched at 22:09:09 having been launched earlier, and the
#     understudy was never watched at all.
#   * a box nobody watched bills exactly the same as one somebody did.
#
# WHAT THE API DOES NOT GIVE US, stated so nobody re-derives it: there is no
# per-instance invoice read. The instance body carries `start_date`, `dph_total`,
# `dph_base` and `storage_total_cost`; the invoice lives on the account, not the
# instance, and cannot be attributed back to a box from the API. So this is NOT a
# reconciliation against the bill — it is a reconciliation against an INDEPENDENT
# ESTIMATE built from the box's own billing anchor, which is the part that was
# actually missing. See AUTOBID_DESIGN "Spend reconciliation".
#
# The estimate is an UPPER BOUND on GPU-rate billing and is labelled as one:
# vast does not bill GPU during `loading` (invoice-verified 2026-07-20 and
# 2026-08-02) and the API exposes no loading->running timestamp, so a box that
# spent 20 minutes pulling an image is over-estimated by exactly that. A box that
# was parked mid-life is over-estimated too. A divergence therefore reads as
# "this much of the box's billed life fleetd never watched", never as "we were
# undercharged by $X" — an estimate quoted as an invoice is how the $0.155-vs-
# $0.041 loading overstatement got into a billing measurement in the first place.
# --------------------------------------------------------------------------- #
class ReconcileRow(TypedDict):
    """One box's accrued-vs-estimated spend. `divergence_pct` is None whenever
    the upper bound is falsy — INCLUDING `ub == 0` — because a percentage of
    nothing is not 0%, it is unknown."""

    iid: str
    target: str | None
    watched: bool
    present: bool
    accrued_usd: float
    dph_total: float | None
    age_s: float | None
    unwatched_head_s: float | None
    upper_bound_usd: float | None
    divergence_usd: float | None
    divergence_pct: float | None


# moved-from: fleetd.reconcile_rows
def reconcile_rows(state: StateDoc,
                   instances: Iterable[Mapping[str, Any]] | None,
                   now: float) -> list[ReconcileRow]:
    """PURE. Per-box `{iid, accrued_usd, upper_bound_usd, divergence_usd, ...}`
    comparing fleetd's accrued spend against the independent start_date estimate.

    Covers every box in the instance listing, watched or not: an UNWATCHED box
    accrues nothing at all, and its whole bill is the divergence — which is the
    single largest line the 2026-08-08 accounting missed."""
    by_iid = {str(i.get("id")): i for i in (instances or []) if i.get("id")}
    accrued = {str(k): v for k, v in (state.get("spend_by_box") or {}).items()}
    watch_by_iid: dict[str, tuple[str | None, Mapping[str, Any]]] = {}
    for target, w in (state.get("watches") or {}).items():
        if isinstance(w, dict):
            watch_by_iid[str(w.get("iid") or target)] = (target, w)
    rows: list[ReconcileRow] = []
    for iid in sorted(set(by_iid) | set(accrued)):
        inst: Mapping[str, Any] = by_iid.get(iid) or {}
        target, w = watch_by_iid.get(iid, (None, {}))
        got = round(float(accrued.get(iid) or 0.0), 4)
        dph = models._num_dph(inst.get("dph_total"))
        # MECHANICAL: the flat copy calls `inst.get("start_date")` twice, once
        # as the guard and once as the argument, which strict mypy reads as
        # `Any | None` at the `float()`. Bound to one local instead — a plain
        # mapping lookup has no side effect, so the two are identical, and the
        # `try` still catches the garbage `start_date` the guard lets through.
        raw_start: Any = inst.get("start_date")
        try:
            start = float(raw_start) if raw_start else None
        except (TypeError, ValueError):
            start = None
        age_s = (now - start) if start else None
        ub = (round(dph * age_s / 3600.0, 4)
              if (dph and age_s and age_s > 0) else None)
        created = w.get("created_ts") if isinstance(w, dict) else None
        rows.append({
            "iid": iid, "target": target,
            "watched": bool(w),
            "present": iid in by_iid,
            "accrued_usd": got,
            "dph_total": dph,
            "age_s": round(age_s, 1) if age_s is not None else None,
            # the window the box billed before any watch existed — the named
            # cause of the 2026-08-08 shortfall, and the number that says whether
            # a divergence is "we adopted late" or "our arithmetic drifted"
            "unwatched_head_s": (round(created - start, 1)
                                 if (created and start and created > start)
                                 else None),
            "upper_bound_usd": ub,
            "divergence_usd": (round(ub - got, 4) if ub is not None else None),
            "divergence_pct": (round(100.0 * (ub - got) / ub, 1)
                               if (ub and ub > 0) else None),
        })
    return rows


# --------------------------------------------------------------------------- #
# the three pure predicates the safety net and the ladder rebuild ask
# --------------------------------------------------------------------------- #
# moved-from: fleetd.watch_box_iid
def watch_box_iid(w: Mapping[str, Any] | None) -> str | None:
    """The box a NON-RUN watch's ladder is supervising RIGHT NOW, or None.

    `w["iid"]` diverges from the watch KEY whenever the jobs/serve ladder moved
    the watch onto a different box — an eviction replacement it rented itself,
    or a handoff understudy. The key is only where the watch STARTED; this is
    the authoritative "which box is this". Two spend-control incidents came from
    treating the key as authoritative (2026-08-05, boxes 46866652->46867184 and
    46867184->46867793: a daemon restart and an operator re-`watch` each rebuilt
    the ladder against the ORIGINAL, already-destroyed id, so the watch died
    `instance_gone` a minute later and the safety net re-adopted the live
    replacement as `bare` with NO cap).

    None for a run watch (its iid is re-resolved from the run label every tick,
    and `run:<ID>` is not an instance id) and for the unset/garbage values a
    `str(None)` round-trip through state.json can leave behind."""
    w = w or {}
    if w.get("profile") == "run":
        return None
    iid = w.get("iid")
    return None if iid in (None, "", "None") else str(iid)


# moved-from: fleetd.EXEMPT_LABEL_TOKENS
EXEMPT_LABEL_TOKENS: Final = ("nofleet",)  # boxes that opt out of the safety net entirely


# moved-from: fleetd.label_exempt
def label_exempt(label: str | None) -> bool:
    """A `nofleet` token anywhere in the `:`-separated label opts a box out of
    the safety net (the workflowctl escape hatch — B1c).

    TOKEN match, never substring: the same grammar rule `core.labels` applies to
    `keep:`, and for the same reason — a box labelled `nofleetd-probe` is not
    exempt."""
    return any(t.strip().lower() in EXEMPT_LABEL_TOKENS
               for t in (label or "").split(":"))


# moved-from: fleetd.BOOT_EVIDENCE_S
BOOT_EVIDENCE_S: Final = 1800.0  # a freshly booted box is busy by construction
# moved-from: fleetd.JOBD_FRESH_S
JOBD_FRESH_S: Final = 900.0      # jobd heartbeat younger than this = alive workload

# moved-from: fleetd.workload_evidence
def workload_evidence(inst: Mapping[str, Any] | None,
                      health_row: Mapping[str, Any] | None = None) -> str | None:
    """PURE (review B1). Why an UNWATCHED box looks like live work — a string,
    or None when nothing says so. Evidence => auto-adopt + alarm; no evidence
    (past the grace window) => the safety-net park. Order matters: a measured
    zombie verdict must not be rescued by a mere label, and a MEASUREMENT
    outranks a label (CPU work sits above the token loop, below the zombie
    check)."""
    inst = inst or {}
    row = health_row or {}
    ev = row.get("evidence") or {}
    verdict = row.get("verdict")
    label = inst.get("label") or ""
    if verdict == health.GUARD_BOOTING:
        return "booting"
    boot_age = ev.get("boot_age_s")
    if boot_age is not None and boot_age < BOOT_EVIDENCE_S:
        return f"booted {int(boot_age)}s ago"
    hb = ev.get("jobd_hb_age_s")
    if hb is not None and hb < JOBD_FRESH_S:
        return f"jobd heartbeat {int(hb)}s old"
    if health.verdict_is_zombie(verdict):
        return None                       # measured dead workload: no evidence
    # A dedicated CPU box (compile/search work, no model endpoint) is invisible
    # to every signal above: no jobd, no GPU, and a label the token loop below
    # does not know. Live 2026-08-21, box 48259065 four hours into an rb3 A/B
    # returned None here and was one unwatched grace window from being parked
    # mid-run. This is NOT the co-tenant compile farm ruled dead the same day
    # (`0a9f1926`) — that was a sidecar stealing cores from a GPU box.
    cpu = ev.get("cpu_util")
    if isinstance(cpu, (int, float)) and cpu > health.CPU_BUSY_UTIL:
        cores = ev.get("cpu_cores_effective")
        of = f"/{cores:g}" if isinstance(cores, (int, float)) else ""
        return f"cpu {cpu:.2f}{of} busy"
    toks = [t.strip().lower() for t in label.split(":") if t.strip()]
    for t in ("run", "serve", "jobs", "wf", "workflow", "stage", "eval", "keep"):
        if t in toks:
            return f"label {label!r}"
    if ev.get("is_jobs_box"):
        return "jobs-lane box"
    return None
