"""Retain-or-destroy: what happens to the box we were just evicted FROM.

Why this module exists
----------------------
The eviction ladder's job is the REPLACEMENT — rent a new box, retarget the
queue, re-anchor the supervisor. This module owns the other half, which is
about the box that was lost, and which exists because of an owner directive
(2026-08-05):

    "don't have fleetd destroy the box immediately please. we should eat a few
     hours of parked host time just in case we have bugs that lose data."

So the default is RETAIN for `--replacement-retention-hours` (3h). Its disk can
hold state that never reached B2 — checkpoint sync is periodic, so up to one
interval of training progress plus any non-checkpoint artifact exists only
there. Around that one decision sit four mechanisms, each of which was added
after the previous set failed in production:

* **A self-expiring `keep:` label** is the whole expiry mechanism. `herdd
  reap`'s 15-minute systemd timer destroys stopped boxes idle past 2h unless
  the label carries a `keep` token, and `_reap_kept` reads a `until-<TS>`
  deadline inside that token. The window therefore survives a fleetd restart, a
  watch that ended, and a sleeping workstation, with no daemon in the loop.
* **A backstop destroy** (`_job_retention_sweep`), for the case where the reap
  timer is not installed at all. Past `deadline + RETENTION_BACKSTOP_GRACE_H`
  the ladder that created the retention finishes it.
* **Quiesce** (`_job_quiesce_box`) — stop + pin the bid below any floor. The
  keep label stops the REAPER; it does nothing about VAST putting the box back
  on a GPU. Box 47833510 (2026-08-16) came back on its own an hour after
  eviction, via a start the rescue rung had queued and a standing bid nothing
  unwound, and billed GPU rate unwatched inside a window whose disclosed price
  was storage-only.
* **Salvage** (`boxes/salvage.py`, armed here at the moment of eviction). The
  race is HOST RECLAMATION, not the window: box 46859541 was gone ~30 min after
  its eviction, far inside a 3h retention, and took an unsynced checkpoint with
  it. A window protects against the operator being slow, and the operator being
  slow is not what loses the data.

The five tick drivers at the top of this file are the salvage half of that: they
mutate the job context, journal through `supervise.journal`, read the
replacement knobs and print the operator lines. The state machine they drive is
`boxes/salvage.py`, which is pure and answers "advance this record one step" —
never "when should a record exist". That split is why `_job_salvage_advance`
reads as an injection site: it is the one place where the eight vast-side I/O
closures meet the transport-free orchestrator.

What is deliberately NOT here
-----------------------------
* **The run lane.** Every symbol here is the JOB lane (`jc` dicts, `_job_*`
  names). The run lane has its own eviction/retention wording and the two stay
  MIRRORED — the six divergences are pinned deliberately (plan §5 NOTE, v1 §7 /
  `FLEET_REVIEW_2026-08-14.md` item 1). Do not unify a hook, a tick, a journal
  or an accruer across lanes while porting.
* **`_job_eviction_replace`** — the caller of `_job_retain_or_destroy` — and
  `_job_replacement_knob`, which this module READS. Both are `replacement.py`.
  The knob is imported rather than copied because its precedence (namespace >
  `JOB_<NAME>` env > bidpolicy default) is one behavior, and a second copy is a
  second thing to drift.
* **The retention POLICY arithmetic.** `bidpolicy.retention_plan` /
  `retention_live_cost` own the deadline, the cost band and the live multiple;
  everything here is I/O, journaling and the operator's text.
* **The keep-label READ side.** `boxes/reap.py` owns `_keep_retention_info`;
  this module owns only the WRITE, via `core.labels.retention_keep_label`.
  Neither imports the other — both go through `core.labels`.
* **No dry-run gate to hoist.** The threading is per-function and inconsistent
  BY DESIGN: `_job_quiesce_box` returns `None`/`None` plus an errors note,
  `_job_retain_or_destroy` skips only the label PUT, `_job_retention_sweep`
  skips only the backstop destroy, `_job_salvage_advance` passes `dry_run` down
  into `salvage.advance`. Each site is verbatim; a single hoisted gate would
  change three of the four.

Contracts that look like implementation detail
----------------------------------------------
* **The `retained_boxes` record keys are frozen.** fleetd reads them directly
  (`_retention_status_map`, `retention_rows`, `retention_alarms`, the ceiling
  rows) and `_replacement_state_persist` writes them into `state.json` — a plan
  §4 load-compat contract. `status`, `class`, `deadline_ts`, `retention_h`,
  `cost_usd`, `cost_hi_usd`, `keep_labeled`, `replacement_iid`,
  `storage_day_usd`, `live_since_ts`, `live_dph`, `live_cost_usd`,
  `live_multiple`, `resurrections`, `requiesces`, `quiesce`, `salvage`,
  `ended_ts`: renaming one silently blanks a fleetd row.
* **The seven status spellings** — `retained`, `expired`, `reaped`,
  `retention_lost`, `destroyed`, `destroy_failed`, `already_gone` — are the
  keys of fleetd's `RETENTION_NOTES`. Add, never repurpose.
* **The printed operator text is asserted on.** `QUIESCE FAILED`, the `2h idle
  mark` warning, the `$0.27-$0.58` money formatting, `RETENTION_REQUIESCE_MAX`,
  and the literal path `tools/vast/RETENTION_SALVAGE.md`. Reflowing an f-string
  here is an expectation change (plan §7.4), i.e. a found drift.
* **`_job_retention_sweep` calls `_job_salvage_sweep` FIRST and
  unconditionally**, before the empty-record early return: a copy already in
  flight keeps making progress after the retention record goes terminal, and its
  result is the whole point of holding the box.
* **`_job_salvage_sweep` swallows every per-record exception** by design — a
  salvage failure must never take down the supervision loop keeping the
  REPLACEMENT alive. The printed retry line is the only signal it happened.
* **There is no `--retention-backstop-hours` CLI flag**, on purpose:
  `retention_backstop_hours` resolves from a policy-namespace attribute or
  `$JOB_RETENTION_BACKSTOP_HOURS` only. Do not "fix" that here.

Provenance: moved verbatim from `tools/vast/herdd.py` (plan §8 step 4,
2026-08-16). Behavior-preserving: bodies copied, annotations added, and every
cross-module call respelled to module-attribute form (`journal._iso_z`,
`lifecycle._put_state_soft`, `models._num_dph`, `labels.retention_keep_label`,
`salvage.advance`, `replacement._job_replacement_knob`) so the suite's
`monkeypatch.setattr` idiom keeps steering after the port. Every symbol carries
its `# moved-from:` marker (grammar: `vastlib/README.md` §2).
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping

from vastlib.boxes import lifecycle, remote, salvage
from vastlib.core import labels, models
from vastlib.supervise import journal, replacement

import bidpolicy


# --------------------------------------------------------------------------- #
# The salvage tick drivers. `boxes/salvage.py` is the state machine; these five
# decide WHEN a record exists and advance it from the retention sweep.
# --------------------------------------------------------------------------- #
# moved-from: herdd._job_salvage_start
def _job_salvage_start(jc: MutableMapping[str, Any], old: int | str,
                       new_iid: int | str | None,
                       now: float) -> salvage.Record | None:
    """Arm salvage for a just-evicted box. Returns the record, or None when
    disarmed. Does NOT do the copy — `advance` does, from the retention sweep, so
    the eviction path never blocks the daemon."""
    if not salvage._salvage_enabled(jc):
        return None
    rec = salvage.new_record(
        old, now=now, dest_candidates=salvage._salvage_dest_candidates(jc, new_iid, old),
        keep_n=replacement._job_replacement_knob(jc, "salvage_keep_n", salvage.SALVAGE_KEEP_N),
        max_gb=replacement._job_replacement_knob(jc, "salvage_max_gb", salvage.SALVAGE_MAX_GB))
    journal._job_handoff_emit(jc, "eviction_salvage_armed", box=str(old),
                              dest_candidates=rec["dest_candidates"],
                              deadline=journal._iso_z(rec["deadline_ts"]),
                              note="instance->instance disk salvage armed at the moment "
                                   "of eviction; the race is HOST RECLAMATION (box "
                                   "46859541 was gone ~30 min after its eviction), not "
                                   "the retention window")
    return rec


# moved-from: herdd._job_salvage_advance
def _job_salvage_advance(jc: MutableMapping[str, Any], rec: salvage.Record,
                         now: float) -> salvage.Record:
    """One bounded salvage step, driven by the retention sweep's tick.

    Journals a terminal outcome exactly once and prints the LOUD ones loudly:
    a partial or unverifiable copy is NOT a salvage, and the operator has to know
    that before anything resumes from those bytes."""
    statuses = salvage._salvage_statuses(jc)
    salvage.advance(
        rec, now=now, execute=remote._vast_execute_soft,
        copy_direct=remote._vast_copy_direct_soft, statuses=statuses,
        free_gb=salvage._salvage_free_gb(jc),
        dest_execute=salvage._mk_salvage_dest_exec(statuses),
        prepare_dest=salvage._mk_salvage_prepare_dest(statuses),
        copy_status=salvage._salvage_copy_status,
        b2_bytes=salvage._salvage_b2_bytes, push_to_b2=salvage._salvage_push_to_b2,
        dry_run=bool(jc.get("dry_run")))
    if rec.get("phase") != "done" or rec.get("_journaled"):
        return rec
    rec["_journaled"] = True
    outcome = rec.get("outcome")
    journal._job_handoff_emit(jc, "eviction_salvage_ended", box=rec.get("dead_iid"),
                              outcome=outcome, dest=rec.get("dest_iid"),
                              bytes=rec.get("bytes"), detail=rec.get("detail"),
                              items=[{"job_id": it.get("job_id"), "name": it.get("name"),
                                      "bytes": it.get("bytes"),
                                      "verify": it.get("verify"), "b2": it.get("b2")}
                                     for it in rec.get("items") or []])
    mark = "!!" if outcome in salvage.LOUD_OUTCOMES else ">>"
    print(f"{mark} salvage {outcome} for lost box {rec.get('dead_iid')}: "
          f"{rec.get('detail')}")
    return rec


# moved-from: herdd.SALVAGE_DEFER_GRACE_S
SALVAGE_DEFER_GRACE_S = 900.0     # slack past a salvage record's own deadline
                                  # before the retention backstop stops waiting
                                  # for it. One tick can be minutes; this is
                                  # room for the LAST verification attempt, not
                                  # an extension of the transfer window.


# moved-from: herdd._salvage_defer_until
def _salvage_defer_until(sal: Mapping[str, Any]) -> float:
    """Wall-clock bound on how long a non-terminal salvage record may hold off
    the retention backstop. Reads the record's own deadline, so a record that
    was never advanced (or that cannot be advanced) still expires."""
    try:
        dl = float(sal.get("deadline_ts") or 0.0)
    except (TypeError, ValueError):
        dl = 0.0
    if dl <= 0:
        try:
            dl = float(sal.get("started_ts") or 0.0) + salvage.SALVAGE_DEADLINE_S
        except (TypeError, ValueError):
            dl = 0.0
    return dl + SALVAGE_DEFER_GRACE_S


# moved-from: herdd._job_salvage_sweep
def _job_salvage_sweep(jc: MutableMapping[str, Any], now: float) -> None:
    """Advance every armed salvage record on this watch, once per tick. Errors are
    contained: a salvage failure must never take down the supervision loop that
    is keeping the REPLACEMENT alive."""
    for r in (jc.get("retained_boxes") or []):
        rec = r.get("salvage")
        if not isinstance(rec, dict) or rec.get("phase") == "done":
            continue
        try:
            _job_salvage_advance(jc, rec, now)
        except Exception as e:                        # noqa: BLE001
            print(f"!! salvage step errored for {rec.get('dead_iid')} "
                  f"({type(e).__name__}: {e}) — retrying next tick")


# --------------------------------------------------------------------------- #
# Retention proper: hold the lost box, keep it asleep, follow it to an outcome.
# --------------------------------------------------------------------------- #
# moved-from: herdd.RETENTION_REQUIESCE_MAX
RETENTION_REQUIESCE_MAX = 3      # how many times the sweep will re-park one
                                 # retained box before it only alarms. A host
                                 # that keeps re-placing the instance is not
                                 # something we can win by PUT-ing at it every
                                 # 45s; past this the operator owns it.


# moved-from: herdd._job_quiesce_box
def _job_quiesce_box(jc: MutableMapping[str, Any], iid: int | str,
                     inst: Mapping[str, Any] | None, *,
                     why: str) -> dict[str, Any]:
    """Put a box we are HOLDING (not using) to sleep, so it cannot bill GPU rate.

    Two moves, both best-effort, both needed — this is the fix for the 2026-08-16
    resurrection of box 47833510 (session doc:
    docs/plans/vast-fleet-sessions/2026-08-16-replacement-probe-tpd/SESSION.md):

      1. `stop`. The evicted box is already `exited`, so this looks like a no-op
         — it is not. The rescue rung's in-place `start` attempts had been
         answered by vast with *"Required resources are currently unavailable,
         state change queued"*, which leaves the instance with a PENDING desired
         state of running. Nothing in the replacement path cancelled it, so when
         machine 34985 had capacity again ~65 minutes later vast dutifully
         executed the queued start. A `stop` is what withdraws it.

      2. Pin the bid to `RETENTION_PARK_BID` ($0.001), on BID instances only.
         Independently of any queued start, vast auto-resumes a stopped bid
         instance whose standing bid clears the machine's floor, and the rescue
         ladder had just raised this one $0.96 -> $1.20 against a floor of $0.80
         with nothing to unwind it. A floor-relative pin would not do: the floor
         DROPS, which is precisely when a resume fires. Same reasoning and same
         value as the handoff fence's `HANDOFF_PARK_BID` (box-44566398 stuck-bid
         leak) — that lane learned this in July and the retention lane never got
         the lesson.

    `_put_bid_soft` documents that a bid change is accepted on a stopped
    instance, and the handoff fence has been doing exactly this pair since the
    T4b belt; neither is new API surface.

    Returns a record `{"stopped", "bid_pinned", "prior_bid", "errors", "ts"}` —
    stored on the retention record so `fleet log` can say what was done, and so
    the salvage runbook knows a resume needs the bid raised back."""
    rec: dict[str, Any] = {"stopped": False, "bid_pinned": None, "prior_bid": None,
                           "errors": [], "ts": jc.get("now"), "why": why}
    inst = inst or {}
    is_bid = bool(inst.get("is_bid"))
    prior = models._num_dph(inst.get("dph_total"))
    rec["prior_bid"] = prior if (is_bid and prior
                                 and prior > bidpolicy.RETENTION_PARK_BID) else None
    if jc.get("dry_run"):
        rec["stopped"] = rec["bid_pinned"] = None
        rec["errors"].append("dry_run: no stop, no bid pin issued")
        return rec
    ok, err = lifecycle._put_state_soft(iid, "stopped")
    rec["stopped"] = bool(ok)
    if not ok:
        rec["errors"].append(f"stop: {err}")
    if is_bid:
        okb, errb = lifecycle._put_bid_soft(iid, bidpolicy.RETENTION_PARK_BID)
        rec["bid_pinned"] = bidpolicy.RETENTION_PARK_BID if okb else None
        if not okb:
            rec["errors"].append(f"bid_pin: {errb}")
    return rec


# moved-from: herdd._quiesce_summary
def _quiesce_summary(q: object) -> str:
    """One human clause for a `_job_quiesce_box` record."""
    if not isinstance(q, dict):
        return "not quiesced"
    if q.get("errors") and not (q.get("stopped") or q.get("bid_pinned")):
        return "QUIESCE FAILED (" + "; ".join(q["errors"]) + ")"
    bits = []
    if q.get("stopped"):
        bits.append("stopped (queued start withdrawn)")
    if q.get("bid_pinned") is not None:
        bits.append(f"bid pinned ${q['bid_pinned']:g} (below any floor)")
    if q.get("errors"):
        bits.append("partial: " + "; ".join(q["errors"]))
    return ", ".join(bits) or "nothing to quiesce"


# moved-from: herdd._job_retain_or_destroy
def _job_retain_or_destroy(jc: MutableMapping[str, Any], old: int | str,
                           inst: Mapping[str, Any] | None, eviction_class: str,
                           now: float,
                           new_iid: int | str | None = None) -> dict[str, Any]:
    """Dispose of the box we were just evicted from, once the replacement is
    renting and the queue has moved. Owner directive 2026-08-05:

      "don't have fleetd destroy the box immediately please. we should eat a few
       hours of parked host time just in case we have bugs that lose data."

    Default is RETAIN for `--replacement-retention-hours` (3h), because the lost
    box's disk can hold state that never reached B2 — checkpoint sync is
    periodic, so up to one interval of training progress plus any non-checkpoint
    artifact exists only there. `0` restores the immediate destroy.

    Retention is a `keep:` label with an embedded deadline, and that label is the
    WHOLE expiry mechanism: `herdd reap` (the 15-minute systemd timer) destroys
    stopped boxes idle past 2h unless the label carries a `keep` token, and
    `_reap_kept` now reads a `until-<TS>` deadline inside that token. So the
    window is defended and self-terminating with no daemon in the loop — it
    survives a fleetd restart, a watch that ended, and a sleeping workstation.
    `_job_retention_sweep` adds a backstop destroy for the case where the reap
    timer is not installed at all.

    Three outcomes, journaled distinctly, because retention of an evicted SPOT
    box is BEST-EFFORT by construction (a host can reclaim a stopped bid
    instance and its disk within minutes — box 44612403, `herdd start` -> 404
    `no_such_instance`): `retained`, `already_gone`, and later `retention_lost`
    (see the sweep). Returns the record it appended."""
    hours = replacement._job_replacement_knob(jc, "replacement_retention_hours",
                                              bidpolicy.REPLACEMENT_RETENTION_H)
    plan = bidpolicy.retention_plan(  # type: ignore[no-untyped-call]
        retention_h=hours, present=bool(inst), now=now,
        storage_day_usd=models._storage_day(inst or {}))
    rec: dict[str, Any] = {"iid": str(old), "status": None, "class": eviction_class,
                           "retained_ts": now, "deadline_ts": plan.deadline_ts,
                           "retention_h": hours, "cost_usd": plan.cost_usd,
                           "cost_hi_usd": plan.cost_hi_usd,
                           "storage_day_usd": models._storage_day(inst or {}),
                           "replacement_iid": str(new_iid) if new_iid is not None else None,
                           "label": (inst or {}).get("label"), "keep_labeled": False}

    if plan.action == "already_gone":
        rec["status"] = "already_gone"
        # Record the salvage outcome even here: `dead_box_gone` is the measured
        # failure rate of the whole idea, and folding it into "we didn't try"
        # would hide exactly the number that decides whether salvage is worth it.
        rec["salvage"] = {"phase": "done", "outcome": salvage.OUTCOME_DEAD_GONE,
                          "dead_iid": str(old), "items": [], "bytes": 0,
                          "_journaled": True,
                          "detail": "the box had already left the listing when "
                                    "the replacement landed — its disk was gone "
                                    "before salvage could look at it"}
        journal._job_handoff_emit(jc, "eviction_box_already_gone", box=str(old),
                                  eviction_class=eviction_class, reason=plan.reason)
        print(f".. lost box {old} is already out of the listing — nothing to "
              f"retain or destroy (spot hosts reclaim stopped bid instances)")
        jc.setdefault("retained_boxes", []).append(rec)
        return rec

    if plan.action == "destroy":
        rec["status"] = "destroyed"
        failed_destroy = lifecycle._destroy_and_revoke([old], jc.get("instances") or [],
                                                       "eviction_replaced_destroy")
        if failed_destroy:
            rec["status"] = "destroy_failed"
            print(f"!! eviction replacement: destroy of {old} FAILED — it still "
                  f"bills storage; destroy it by hand (queue already moved to "
                  f"{new_iid})")
        # Task #78. `jobs_box_retention` only journals a RETAINED box; the
        # `retention_h: 0` configuration destroys immediately and said nothing.
        journal._job_ladder_journal(jc, "jobs_box_destroyed", iid=str(old),
                                    lane="eviction_replacement", to_box=str(new_iid),
                                    actor="fleetd:eviction-ladder",
                                    ok=not failed_destroy,
                                    eviction_class=eviction_class, reason=plan.reason,
                                    note="destroyed by the LADDER (retention window is "
                                         "0h) — the neighbouring `operator_intent_"
                                         "destroy` carries the workstation user because "
                                         "`fleet_operator_intent` names the host; its "
                                         "`reason` field is the real actor")
        jc.setdefault("retained_boxes", []).append(rec)
        return rec

    # RETAIN. Stamp the self-expiring keep label FIRST — an unlabelled retained
    # box is a lie: `herdd reap` would destroy it at the 2h idle mark, i.e.
    # BEFORE a 3h window closes, and the salvage promise would silently evaporate.
    label = (inst or {}).get("label") or ""
    new_label = labels.retention_keep_label(label, f"evicted-{eviction_class}",
                                            plan.deadline_ts)
    ok = False
    if not jc.get("dry_run"):
        ok, err = lifecycle._put_label_soft(old, new_label)
        if not ok:
            print(f"!! could not stamp the retention keep label on {old} "
                  f"({err}) — the box is KEPT anyway, but `herdd reap` may "
                  f"destroy it at the 2h idle mark; re-stamp by hand: "
                  f"herdd label {old} '{new_label}'")
    rec["keep_labeled"] = bool(ok)
    rec["label"] = new_label if ok else label
    rec["status"] = "retained"
    # QUIESCE. The keep label stops the REAPER from taking the box; it does
    # nothing about VAST putting it back on a GPU. Withdraw the queued start and
    # pin the bid below any floor, so the storage-only price we are about to
    # disclose is the price this box will actually charge. Salvage reads an
    # `exited` instance's filesystem and enters no GPU contract, so stopping the
    # box costs the salvage path nothing.
    rec["quiesce"] = _job_quiesce_box(jc, old, inst, why="retention")
    # ARM SALVAGE NOW, not when an operator gets round to the runbook. The
    # retention window protects against the wrong failure: box 46859541 was
    # reclaimed by its host ~30 min after eviction, well inside a 3h window, and
    # the checkpoint went with it. The first tick of the retention sweep does the
    # survey + copy; this call only stamps the record so the eviction path
    # itself stays non-blocking.
    rec["salvage"] = _job_salvage_start(jc, old, new_iid, now)
    jc.setdefault("retained_boxes", []).append(rec)
    journal._job_handoff_emit(jc, "eviction_box_retained", box=str(old),
                              eviction_class=eviction_class,
                              deadline=journal._iso_z(plan.deadline_ts),
                              retention_h=hours, est_cost_usd=plan.cost_usd,
                              est_cost_hi_usd=plan.cost_hi_usd,
                              storage_day_usd=rec["storage_day_usd"],
                              keep_labeled=rec["keep_labeled"], label=rec["label"],
                              replacement_iid=rec["replacement_iid"],
                              quiesce=_quiesce_summary(rec["quiesce"]),
                              reason=plan.reason)
    journal._job_ladder_journal(jc, "jobs_box_quiesced", iid=str(old),
                                lane="eviction_replacement", to_box=str(new_iid),
                                actor="fleetd:eviction-ladder",
                                eviction_class=eviction_class,
                                stopped=rec["quiesce"].get("stopped"),
                                bid_pinned=rec["quiesce"].get("bid_pinned"),
                                prior_bid=rec["quiesce"].get("prior_bid"),
                                errors=rec["quiesce"].get("errors") or None,
                                note="RETAINED box put to sleep — " +
                                     _quiesce_summary(rec["quiesce"]) +
                                     ". Without this a retained SPOT box comes back on "
                                     "its own (queued start / standing bid clearing the "
                                     "floor) and bills GPU rate unwatched, which is "
                                     "incident 47833510, 2026-08-16")
    cost = (f"~${plan.cost_usd:.2f}" if plan.cost_usd == plan.cost_hi_usd
            else f"~${plan.cost_usd:.2f}-${plan.cost_hi_usd:.2f}")
    print(f">> RETAINED lost box {old} until {journal._iso_z(plan.deadline_ts)} "
          f"({hours:g}h, {cost} storage) for salvage of state that never "
          f"reached B2 — it self-expires and `herdd reap` reclaims it. "
          f"Quiesced: {_quiesce_summary(rec['quiesce'])}. "
          f"Salvage runbook: tools/vast/RETENTION_SALVAGE.md")
    if rec["quiesce"].get("prior_bid"):
        print(f".. {old}'s standing bid was ${rec['quiesce']['prior_bid']:.4f} "
              f"before the pin — a manual resume for salvage has to raise it "
              f"back (`herdd bid {old} --price <P>`)")
    return rec


# moved-from: herdd._job_retention_liveness
def _job_retention_liveness(jc: MutableMapping[str, Any], r: MutableMapping[str, Any],
                            inst: Mapping[str, Any] | None, now: float) -> None:
    """One retained box, one tick: is it LIVE, and if so what does that cost?

    Mutates the retention record in place:

      * `live_since_ts` / `live_dph` / `live_cost_usd` / `live_multiple` — set
        while the box is live, cleared the moment it is not. `fleet status`
        derives its standing alarm from `live_since_ts`, so the alarm retracts
        itself without anyone acking it.
      * `resurrections` / `requiesces` — how many times this has happened and
        how many times we answered. Counted, not just flagged, because "it came
        back once" and "the host keeps re-placing it" want different operators.

    Acts by RE-QUIESCING, never by destroying (see `_job_retention_sweep`'s
    docstring). Only when the retain path quiesced it in the first place: if the
    box was never put to sleep by us, someone else owns its state and the honest
    move is the alarm alone."""
    iid = str(r.get("iid"))
    astat = ((inst or {}).get("actual_status") or "").lower()
    if astat not in bidpolicy.LIVE_STATES:
        for k in ("live_since_ts", "live_dph", "live_cost_usd", "live_multiple"):
            r.pop(k, None)
        return
    dph = models._num_dph((inst or {}).get("dph_total"))
    first = r.get("live_since_ts") is None
    if first:
        r["live_since_ts"] = now
        r["resurrections"] = int(r.get("resurrections") or 0) + 1
    r["live_dph"] = dph
    usd, mult = bidpolicy.retention_live_cost(  # type: ignore[no-untyped-call]
        dph, now - (r.get("live_since_ts") or now),
        storage_day_usd=r.get("storage_day_usd"))
    r["live_cost_usd"], r["live_multiple"] = usd, mult
    # `: Any` is a typing-forced annotation, not a body change: `isinstance` on a
    # re-evaluated `r.get(...)` narrows nothing, so mypy sees `Any | dict | None`
    # and rejects the four `.get` calls below. The expression is verbatim.
    last_q: Any = r.get("quiesce") if isinstance(r.get("quiesce"), dict) else {}
    # Re-park on the FIRST live tick of a resurrection, and again only if the
    # last attempt errored — a successful `stop` takes a few ticks to show as
    # `exited`, and PUT-ing at the box once per 45s in the meantime is how a
    # rate limit gets earned. `RETENTION_REQUIESCE_MAX` bounds both paths.
    ours = bool(last_q.get("stopped") or last_q.get("bid_pinned") is not None
                or last_q.get("errors"))
    tries = int(r.get("requiesces") or 0)
    retry = (ours and bool(last_q.get("errors"))
             and last_q.get("why") == "retention_resurrection"
             and tries < RETENTION_REQUIESCE_MAX)
    if not (first or retry):
        return                       # alarm is standing; nothing new to do
    money = (f"${usd:.4f} so far at ${dph:.4f}/hr" if usd is not None and dph
             else "an unreadable rate")
    over = f", {mult:g}x the disclosed storage-only rate" if mult else ""
    if ours and tries < RETENTION_REQUIESCE_MAX:
        r["requiesces"] = tries + 1
        q = _job_quiesce_box(jc, iid, inst, why="retention_resurrection")
        r["quiesce"] = q
        acted = _quiesce_summary(q)
    elif ours:
        acted = (f"NOT re-parked: already re-parked {tries} times "
                 f"(RETENTION_REQUIESCE_MAX) — the host keeps re-placing it, "
                 f"destroy or salvage it by hand")
    else:
        acted = ("NOT re-parked: this box was never quiesced by the ladder, so "
                 "someone else owns its state — alarming only")
    journal._job_ladder_journal(jc, "jobs_retained_box_resurrected", iid=iid,
                                lane="eviction_replacement",
                                actor="fleetd:retention-sweep",
                                eviction_class=r.get("class"),
                                replacement_iid=r.get("replacement_iid"),
                                dph=dph, live_cost_usd=usd, live_multiple=mult,
                                resurrections=r.get("resurrections"),
                                requiesces=r.get("requiesces"),
                                deadline=journal._iso_z(r.get("deadline_ts")),
                                note=f"a RETAINED box is RUNNING again with no queue "
                                     f"and no watch — {money}{over}. Retention prices "
                                     f"ALLOCATED DISK, so this is money nobody decided "
                                     f"to spend. {acted}")
    print(f"!! retained box {iid} is RUNNING again ({money}{over}) — {acted}")


# moved-from: herdd._job_retention_sweep
def _job_retention_sweep(jc: MutableMapping[str, Any], now: float) -> None:
    """Follow every retained box to a terminal outcome, once per tick.

    Outcome classes, kept DISTINCT because they mean different things about how
    much we can trust retention:

      * `retained`       — still listed, window open (salvage is possible now)
      * `expired`        — window closed; the label no longer keeps it, so the
                           reaper's next pass reclaims it
      * `reaped`         — gone at/after the deadline: the window worked and
                           ended as designed
      * `retention_lost` — gone BEFORE the deadline. On a spot box that is the
                           host reclaiming a stopped bid instance and its disk
                           (44612403); it is the measured failure rate of the
                           retention promise, so it is never folded into
                           `reaped`
      * `destroyed`      — the backstop below killed it

    The backstop exists because the primary expiry path is `herdd reap`'s
    systemd timer, which a workstation may simply not have installed. Past
    `deadline + RETENTION_BACKSTOP_GRACE_H` the ladder that created the
    retention finishes it. It NEVER destroys a box that is currently live: a
    running retained box is a human mid-salvage, and killing that is exactly the
    data loss retention exists to prevent.

    RESURRECTION (2026-08-16, box 47833510). "Currently live" used to be read as
    "a human resumed it" on no evidence, and the sweep's only response was to
    skip the record — inside the window it did not even look. It was not a
    human: vast had executed a start the rescue rung queued an hour earlier, on
    a box whose standing bid nothing had unwound. Nothing else could have caught
    it either — the `keep:` label exempts the box from `herdd reap`, and the
    stray sweep drops the record every cycle because auto-adopt is refused by
    the watch that is filed under this very id. So a live retained box is now
    checked EVERY tick, at any point in the window, RE-QUIESCED (up to
    `RETENTION_REQUIESCE_MAX` times) and flagged `live_since_ts` so `fleet
    status` carries a standing alarm with the money on it. Re-parking is not
    destroying: the disk is untouched, and the box has no queue by construction
    (its tickets were retargeted before it was retained)."""
    # Salvage first, and UNCONDITIONALLY — a host-to-host copy that is already in
    # flight keeps making progress even after the retention record has gone
    # terminal, and its result is the whole point of holding the box.
    _job_salvage_sweep(jc, now)
    recs = [r for r in (jc.get("retained_boxes") or [])
            if r.get("status") in ("retained", "expired")]
    if not recs:
        return
    by_id = {str(i.get("id")): i for i in (jc.get("instances") or [])}
    grace_h = replacement._job_replacement_knob(jc, "retention_backstop_hours",
                                                bidpolicy.RETENTION_BACKSTOP_GRACE_H)
    for r in recs:
        iid = str(r.get("iid"))
        dl = r.get("deadline_ts") or 0.0
        inst = by_id.get(iid)
        if inst is None:
            past = now >= dl
            r["status"] = "reaped" if past else "retention_lost"
            r["ended_ts"] = now
            journal._job_handoff_emit(jc, "eviction_retention_ended", box=iid,
                                      outcome=r["status"], eviction_class=r.get("class"),
                                      deadline=journal._iso_z(dl),
                                      held_s=round(now - (r.get("retained_ts") or now), 1),
                                      note=("window closed and the box was reclaimed as "
                                            "designed" if past else
                                            "box VANISHED before its deadline — a spot "
                                            "host reclaimed the stopped bid instance and "
                                            "its disk; salvage was not possible"))
            print(f".. retention {r['status']}: {iid} "
                  + ("(window had closed)" if past else
                     "(GONE before the deadline — spot reclaim; anything only "
                     "on that disk is lost)"))
            continue
        # BEFORE the window check: a retained box that is LIVE is billing GPU
        # rate right now, and the window has nothing to do with it.
        _job_retention_liveness(jc, r, inst, now)
        if now < dl:
            continue
        if r["status"] == "retained":
            r["status"] = "expired"
            journal._job_handoff_emit(jc, "eviction_retention_expired", box=iid,
                                      eviction_class=r.get("class"),
                                      deadline=journal._iso_z(dl),
                                      note="keep label self-expired; `herdd reap` "
                                           "reclaims the disk on its next pass")
            print(f".. retention window CLOSED for {iid} — its keep label has "
                  f"expired; `herdd reap` reclaims it (backstop in {grace_h:g}h)")
        astat = (inst.get("actual_status") or "").lower()
        if astat in bidpolicy.LIVE_STATES:
            # Still hands off the DESTROY — a live box may be a human
            # mid-salvage and killing it is the data loss retention exists to
            # prevent. `_job_retention_liveness` above has already alarmed and
            # tried to re-park it, so "live" is no longer silent.
            continue
        sal = r.get("salvage")
        if isinstance(sal, dict) and sal.get("phase") != "done" \
                and now < _salvage_defer_until(sal):
            # AUTOMATED salvage is still in flight. Destroying the source now
            # aborts a host-to-host copy that is mid-transfer — the same data
            # loss the backstop exists downstream of.
            #
            # The deferral is bounded by the salvage record's OWN deadline plus a
            # grace, NOT by "phase != done". A record whose `advance` keeps
            # throwing (a persistent API shape change, a bug) never reaches a
            # terminal phase, and an unbounded deferral on that would hold a
            # billing box open forever — the exact waste the backstop exists to
            # stop. Past the bound the destroy proceeds and the loss is recorded.
            print(f".. retention backstop for {iid} DEFERRED — salvage is still "
                  f"in flight (phase {sal.get('phase')}); bounded by its own "
                  f"deadline {journal._iso_z(_salvage_defer_until(sal))}")
            continue
        if now < dl + float(grace_h) * 3600.0 or jc.get("dry_run"):
            continue
        failed = lifecycle._destroy_and_revoke([iid], jc.get("instances") or [],
                                               "eviction_retention_expired_destroy")
        r["status"] = "destroy_failed" if failed else "destroyed"
        r["ended_ts"] = now
        journal._job_handoff_emit(jc, "eviction_retention_destroyed", box=iid,
                                  outcome=r["status"], eviction_class=r.get("class"),
                                  deadline=journal._iso_z(dl), grace_h=grace_h,
                                  note="backstop: the retention window closed over "
                                       f"{grace_h:g}h ago and `herdd reap` had not "
                                       "reclaimed the box (timer not installed?)")
