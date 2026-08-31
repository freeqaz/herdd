"""Durability predicate for parked boxes — is this box's work safely on B2?

Design of record: docs/plans/parked-box-lifecycle.md (P1/P2; owner-ratified
2026-07-30). The policy lives in PURE, I/O-free classifiers here so `herdd
reap` and `fleetd` import one copy, matching the house pattern
(`classify_job_box_stop`, `classify_box_health`, `credbroker.verify_instance`).
Impure evidence-gathering stays at the edges, behind injectable seams.

THE ASYMMETRY THAT SHAPES EVERYTHING. You can prove work IS on B2 (an upload
returned rc=0; a listing is non-empty). You cannot prove it is NOT somewhere. So
the verdict is three-valued and UNKNOWN means HOLD. This deliberately inverts
fleetd's usual convention — elsewhere unreadable evidence fails STRICT, because
the strict action there is a park; here the strict action is an irreversible
destroy, so unreadable evidence must fail LENIENT.

WHY EVENT ORDERING IS NOT PROOF (the defect this module exists to avoid). The
run lane's terminal event looks like it implies a completed flush: it is emitted
after one. It does not. `onstart/train.sh`'s teardown runs with errexit OFF (:589,
`set -uo pipefail` :42 — deliberately, so teardown completes even when a push
fails), the flush retries 3× and falls through (:606-609), and the terminal event
is minted from the TRAINING exit code (:653-658), not the flush result. A B2 flake
that eats the bulk checkpoint copy but lands the small event object therefore
mints `done` on the box holding the only complete copy. So DURABLE requires
POSITIVE LANDING EVIDENCE — an actual listing — and it is branch-complete,
because that flush loop runs for every RC:

    terminal `done`                      -> non-empty artifacts/<RID>/
    terminal `failed` (not max_hours)    -> non-empty payload-filtered
                                            checkpoints/<RID>/  (a failed run
                                            never pushes artifacts: :611 gates
                                            that push on RC == 0)

Neither branch can be witnessed by `checkpoint_sync_failed`: the sync watcher is
killed at train.sh:595, BEFORE both teardown pushes, so its silence is not
evidence of success.
"""
from __future__ import annotations

DURABLE = "DURABLE"      # work provably on B2; safe to destroy
UNSYNCED = "UNSYNCED"     # positive evidence work is NOT on B2; never destroy on a soft fuse
UNKNOWN = "UNKNOWN"      # cannot tell -> HOLD

# Payload recognition is an ALLOWLIST (§3.2a), not a denylist. The 2026-07-30
# review replay caught the denylist minting DURABLE off `chainmine.log` — the
# chain-mining runset rcats its log into checkpoints/<RID>/ on EVERY exit
# (runsets/chain-mining/train.sh push_log), and other lanes write EVAL_STATUS,
# ARM_DONE, README.md, chat_template.jinja, debug STOP/EXTEND there too. New
# non-payload names appear whenever a runset grows a marker; new PAYLOAD shapes
# are rare and enumerable. So enumerate the payload.
CKPT_PAYLOAD_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
CKPT_PAYLOAD_NAMES = ("trainer_state.json", "adapter_config.json")

_ORDER_MARKERS = ("done", "failed", "final_flush")

# Any of these AFTER the terminal marker means a later session ran on this
# run-id (supervise relaunch, `herdd start`, handoff twin, interactive
# resume-guard reuse) — whatever it produced is invisible to this stream, so
# the terminal marker no longer describes the box.
POST_TERMINAL_ACTIVITY = ("launched", "relaunched", "resumed", "running",
                          "supervisor_started", "runset_cmd_start")

# A failed-branch payload is only landing evidence if it is not obviously OLDER
# than the terminal: the final flush and the terminal event are pushed
# independently (train.sh:606-609 vs :653-658), so a B2 flake can strand
# hours-old mid-run checkpoints as the "payload" while the newest state died
# with the box. Window = one checkpoint interval (300 s) + slack for a slow
# multi-GB flush that STARTED before the terminal was minted.
FRESH_WINDOW_S = 900.0


def ckpt_payload_names(names):
    """PURE. The payload objects in a `checkpoints/<RID>/` listing.

    `checkpoints/` is non-empty from logs alone, so allowlist payload shapes
    before treating a listing as landing evidence: anything under a
    `checkpoint-*` path component (the trainer's save dirs), or a file whose
    name is a known weight/state shape.
    """
    out = []
    for n in names or ():
        s = str(n)
        parts = [p for p in s.split("/") if p]
        if not parts:
            continue
        base = parts[-1]
        if any(p.startswith("checkpoint-") for p in parts):
            out.append(n)
        elif base in CKPT_PAYLOAD_NAMES or base.endswith(CKPT_PAYLOAD_SUFFIXES):
            out.append(n)
    return out


def payload_is_fresh(*, newest_payload_ts, terminal_ts, window_s=FRESH_WINDOW_S):
    """PURE tri-state freshness of failed-branch payload vs the terminal event.

    Epoch seconds; None in ⇒ None out (unknown freshness must HOLD, never
    accelerate a destroy — invariant I3). The §3.2a objection to listing
    recency ("contaminated by onstart.log rewrites") does not apply here:
    the contaminating objects are exactly the ones `ckpt_payload_names`
    filters out, so the newest PAYLOAD mtime is meaningful.
    """
    if newest_payload_ts is None or terminal_ts is None:
        return None
    return float(newest_payload_ts) >= float(terminal_ts) - float(window_s)


def classify_run_durability(*, events, fail_reason=None,
                            artifacts_present=None, ckpt_payload_present=None,
                            ckpt_payload_fresh=None):
    """PURE. `(verdict, reasons)` for a run/train box.

    `events`      chronological event NAMES from the RAW stream (not the fold —
                  `checkpoint_sync_failed` is not a whitelisted fold key and must
                  be read raw, and its ORDER relative to the terminal marker
                  decides the verdict).
    `fail_reason` the fold's `fail_reason` for a terminal `failed`.
    `artifacts_present` / `ckpt_payload_present` / `ckpt_payload_fresh`
                  tri-state evidence: True / False / None=unreadable/unknown.
                  None must never read as False — an unreadable listing is not
                  evidence of absence — and never as True either (I3: unknown
                  evidence never accelerates a destroy). `ckpt_payload_fresh`
                  is `payload_is_fresh(...)`: whether the newest PAYLOAD object
                  is at least as new as the terminal minus FRESH_WINDOW_S.

    SCOPE: the verdict describes the RUN STREAM, and is applicable only to the
    box that emitted the terminal marker. A handoff twin and a supervise
    relaunch share one `runs/<RID>/events/` stream, so a consumer must never
    apply a run verdict to a box other than the terminal emitter — the
    POST_TERMINAL_ACTIVITY guard below catches the recorded cases, but the
    per-box applicability rule is the consumer's to enforce.
    """
    ev = [str(e) for e in (events or [])]
    if not ev:
        return UNKNOWN, ["no readable events"]

    # The MAX_HOURS death path proves NOTHING was flushed: the watchdog writes the
    # terminal STATUS + failed event inline (train.sh:110-136) and touches no
    # checkpoint, then writes /workspace/.run_terminal (:132), which is the
    # preempt trap's first-line short-circuit (preempt_trap.sh:33) — so no final
    # flush runs at all. Compute-only loss, but never DURABLE.
    if "failed" in ev and str(fail_reason or "").lower() == "max_hours":
        return UNSYNCED, ["terminal failed fail_reason=max_hours: the watchdog "
                          "death path performs no final flush (train.sh:132 -> "
                          "preempt_trap.sh:33)"]

    idx = max((i for i, e in enumerate(ev) if e in _ORDER_MARKERS), default=-1)
    if idx < 0:
        if "checkpoint_sync_failed" in ev:
            return UNSYNCED, ["checkpoint_sync_failed with no terminal marker: "
                              "a mid-run sync is known to have failed"]
        return UNKNOWN, ["no terminal event and no final_flush marker"]

    marker = ev[idx]
    if "checkpoint_sync_failed" in ev[idx + 1:]:
        return UNSYNCED, [f"checkpoint_sync_failed AFTER the {marker} marker"]

    # Post-terminal activity makes the terminal marker STALE, not wrong: a later
    # session (supervise relaunch via _reset_run_markers, `herdd start` on the
    # parked box, the resume-guard's interactive idle, a handoff twin) ran on
    # this run-id and wrote no terminal of its own. base-bakeoff-04 and
    # tuner-v0 are the recorded cases: days of resumed sessions after a
    # terminal `failed`, invisible to a marker-keyed read.
    stale_by = [e for e in ev[idx + 1:] if e in POST_TERMINAL_ACTIVITY]
    if stale_by:
        return UNKNOWN, [f"{marker} marker is STALE: post-terminal activity "
                         f"({', '.join(sorted(set(stale_by)))}) — a later session "
                         "ran on this run-id and its work is invisible to the "
                         "event stream, so HOLD"]

    # `final_flush` proves the flush RAN, not that it completed — the emitter is
    # `timeout 45 ... || true` (preempt_trap.sh:50). Treat it as an ordering
    # marker only; it still needs landing evidence below.
    if marker == "done":
        if artifacts_present is None:
            return UNKNOWN, ["terminal done, but artifacts/<RID>/ was unreadable"]
        if not artifacts_present:
            # Two readings and we cannot separate them: the artifact push fell
            # through its 3 retries, OR the run legitimately had nothing to push
            # (empty CKPT_DIR). Both HOLD; do not claim UNSYNCED we cannot prove.
            return UNKNOWN, ["terminal done but artifacts/<RID>/ is EMPTY: either "
                             "the push fell through its retries (train.sh:606-609, "
                             "errexit off :589) or the run produced no payload — "
                             "indistinguishable, so HOLD"]
        return DURABLE, ["terminal done + non-empty artifacts/<RID>/, no later "
                         "checkpoint_sync_failed"]

    # `failed` (not max_hours) and `final_flush` both land only into
    # checkpoints/<RID>/ — artifacts are gated on RC == 0 (train.sh:611).
    if ckpt_payload_present is None:
        return UNKNOWN, [f"{marker} marker, but checkpoints/<RID>/ was unreadable"]
    if not ckpt_payload_present:
        return UNKNOWN, [f"{marker} marker but checkpoints/<RID>/ has no payload "
                         "objects (log-only): the flush fell through its retries, "
                         "and the watcher dies at train.sh:595 so no "
                         "checkpoint_sync_failed can witness it"]
    # Presence is not enough on this branch: the final flush and the terminal
    # event are pushed independently, and there is NO second copy (artifacts are
    # RC==0-gated), so payload provably OLDER than the terminal window means only
    # mid-run checkpoints landed and the newest state died with the box.
    if ckpt_payload_fresh is False:
        return UNKNOWN, [f"{marker} marker with payload, but the newest payload "
                         "object predates the terminal by more than "
                         f"{int(FRESH_WINDOW_S)}s — the final flush likely fell "
                         "through; only mid-run checkpoints landed, so HOLD"]
    if ckpt_payload_fresh is None:
        return UNKNOWN, [f"{marker} marker with payload, but payload freshness is "
                         "unverifiable (no mtimes) — unknown evidence never "
                         "accelerates a destroy (I3), so HOLD"]
    return DURABLE, [f"{marker} marker + fresh payload objects in "
                     "checkpoints/<RID>/, no later checkpoint_sync_failed"]


def classify_box_run_durability(*, box_iid, terminal_emitter_iid, **kw):
    """PURE. `classify_run_durability` + the per-box applicability rule
    (§11a-R2 precondition d): a run verdict belongs to the box that emitted the
    terminal marker. Handoff twins and same-RID relaunches share ONE
    `runs/<RID>/events/` stream, so DURABLE read off another box's terminal
    must never license destroying THIS box. Unknown emitter ⇒ the DURABLE
    claim cannot be attributed ⇒ HOLD (I3). Non-DURABLE verdicts pass through:
    holding is always applicable."""
    verdict, reasons = classify_run_durability(**kw)
    if verdict != DURABLE:
        return verdict, reasons
    if terminal_emitter_iid is None or box_iid is None:
        return UNKNOWN, reasons + [
            "terminal emitter unknown — a DURABLE verdict cannot be attributed "
            "to this box, so HOLD"]
    if str(terminal_emitter_iid) != str(box_iid):
        return UNKNOWN, reasons + [
            f"terminal was emitted by box {terminal_emitter_iid}, not {box_iid} "
            "(handoff twin / relaunch share one stream) — HOLD"]
    return verdict, reasons


def terminal_emitter(raw_events):
    """PURE. iid of the box that emitted the LAST ordering marker, from raw
    event BODIES (not names): actor `box_<iid>` preferred, else the marker
    event's own `instance_id`. Attribution follows the governing (last) marker
    — an unattributable last marker returns None even if an earlier one was
    attributed, because the earlier marker is not the one being trusted."""
    emitter = None
    for e in raw_events or []:
        if str((e or {}).get("event") or "") not in _ORDER_MARKERS:
            continue
        actor = str(e.get("actor") or "")
        if actor.startswith("box_"):
            emitter = actor[4:] or None
        elif e.get("instance_id") is not None:
            emitter = str(e.get("instance_id"))
        else:
            emitter = None
    return emitter


ZOMBIE_DESTROY = "destroy"
ZOMBIE_PARK = "park"
ZOMBIE_ALARM = "alarm"


def zombie_action(*, verdict, is_jobs_box, jobd_ever_stamped, jobd_hb_read,
                  label_kept, confirmed):
    """PURE. What may the AUTOMATIC sweep (the reap timer's live lane) do to a
    box in a zombie verdict? -> (ZOMBIE_DESTROY | ZOMBIE_PARK | ZOMBIE_ALARM,
    why).

    Why this exists: zombie 46256890 (2026-07-30, `loading` 3 h) produced the
    first live-lane rule — destroy the provably-workless loading stall
    (`stall_sweepable`). Zombie 46633685 (2026-08-02) then sat 31 min dead on
    an on-demand serve box: the reaper SAW it, classified it, and declined —
    "not a jobs box — no never-ran proof, alarm only" — while the expensive
    running-but-dead shape (ZOMBIE_NO_JOBD) had no automatic action at all.
    This generalizes the lane with GRADED actions, so the strength of the
    action always matches the strength of the evidence:

      DESTROY — only with the workless proof AND a BILLED phase: a RUNNING
        jobs-lane box whose `jobd_ever_stamped` is False. The JOBD_STATUS
        marker persists on B2 across park/resume, so a READABLE listing
        without it proves no session ever ran against this disk. The phase
        qualifier is the 2026-08-03 amendment below.
      PARK — death is measured but worklessness is not provable, OR the box is
        still in the GPU-UNBILLED `loading` phase: any loading stall (jobs or
        not), a resumed jobs box stalling with disk history, or a running jobs
        box whose jobd heartbeat was affirmatively READ and is stale (jobd
        existed and died; jobs are interruption-tolerant and resume on the next
        jobd boot). Parking ends the GPU-rate bleed immediately, keeps the
        disk, and hands the box to the existing 2 h idle reaper — with the same
        `keep`-label escape a human park gets. The gentle action is
        deliberately still an action: alarm-only is how 46633685 billed a full
        GPU for half an hour.

    AMENDMENT 2026-08-03 (boxes 46682313 + 46682177) — **the `loading` phase is
    not DESTROY-eligible.** Two co-resident boxes on the same image hit the same
    slow pull that morning: serve box 46682177 was flagged ZOMBIE_LOADING_STALL
    at 27m and *came up healthy at 40m* (journal: `health_alarm_cleared
    verdict=OK`); jobs box 46682313, identical shape, was destroyed at 38m —
    90 s before its peer finished pulling. The verdict was a false positive in a
    PROVEN case, and the destroy branch above fired on it only because the box
    happened to be jobs-lane. Two facts make that trade always wrong:

      * `loading` is GPU-UNBILLED (invoice-verified; storage only, ~$0.01/hr —
        46682313 accrued $0.173 total, $0.00 GPU). Destroying there spends an
        IRREVERSIBLE action to save a negligible amount.
      * "jobd never stamped" is not evidence of death during `loading` — jobd
        cannot stamp before the container exists. It is a tautology of the
        phase, so it can never distinguish a slow pull from a dead one.

    So a loading stall now PARKs whatever lane it is in. Terminal removal still
    happens — a parked box lands in the reaper's 2 h idle fuse — but every step
    to it is recoverable (`herdd start`, or a `keep` label). The jobs/non-jobs
    asymmetry in this phase is gone: jobs boxes got the gentler treatment
    non-jobs boxes already had; nothing anywhere got more aggressive.
      ALARM — everything weaker: unconfirmed sightings (one snapshot is not
        no-progress evidence), `keep` labels, ZOMBIE_TICKET_UNCLAIMED (jobd is
        ALIVE — never auto-touch a functioning box over a claiming bug),
        ZOMBIE_PYHALF (see below), and any unreadable evidence (I3: unreadable
        never accelerates an automatic action — a local B2 outage must degrade
        to alarms, not fleet-wide parks).

    ZOMBIE_PYHALF is ALARM here even though the evidence is the STRONGEST this
    function ever sees — the box's own offline capability check, self-reported.
    Strength of evidence is not the only input; the remedy has to match the
    fault. That verdict says the shipped BUNDLE cannot import its own modules
    (FAILCLOSED_DESIGN §4), which is host-independent by construction, so the
    reschedule-onto-a-different-host remedy the other verdicts carry would
    reproduce the fault on every replacement and burn BOOT_MAX_HOST_RETRIES
    boxes proving it. Enforcement for this shape already exists, is gentler,
    and is faster than any sweep: the box self-parks at JOBD_PY_BROKEN_PARK_S
    (300 s) and fleetd's `_pyhalf_tick` parks it at FLEETD_PYHALF_CONFIRM_S
    (600 s), against this lane's ~40-55 min effective floor.

    `confirmed` is the caller's no-progress attestation (herdd's zombie
    ledger: same verdict persisted >= REAP_ZOMBIE_CONFIRM_S across passes with
    pull bytes, jobd heartbeat, box download traffic AND disk usage all flat —
    the last two are the env-setup liveness signals of the 2026-08-02 boot-
    phase split: a running jobs box mid-provision has flat pull bytes and no
    heartbeat BY DEFINITION, so without them a healthy long install would
    confirm as dead). `jobd_hb_read` is True only when a JOBD_STATUS heartbeat
    stamp was successfully read this pass."""
    v = str(verdict or "")
    if v not in ("ZOMBIE_LOADING_STALL", "ZOMBIE_NO_JOBD",
                 "ZOMBIE_TICKET_UNCLAIMED", "ZOMBIE_PYHALF"):
        return ZOMBIE_ALARM, "not an auto-actionable zombie verdict"
    # Named explicitly rather than left to the fall-through above, so the
    # refusal reads as a DECISION with a reason instead of an omission somebody
    # later "fixes" by adding the verdict to the tuple.
    if v == "ZOMBIE_PYHALF":
        return ZOMBIE_ALARM, ("the box confesses a BUNDLE fault (pyhalf=broken) "
                              "— host-independent, so a reschedule reproduces "
                              "it; the box self-parks at 300s and fleetd parks "
                              "it at 600s")
    if label_kept:
        return ZOMBIE_ALARM, "kept (label opt-out)"
    if v == "ZOMBIE_TICKET_UNCLAIMED":
        return ZOMBIE_ALARM, ("jobd is alive (ticket-claiming bug) — never "
                              "auto-touch a functioning box")
    if not confirmed:
        return ZOMBIE_ALARM, ("unconfirmed — needs the verdict to persist "
                              "with no pull/heartbeat progress")
    # ORDER MATTERS (2026-08-03): the loading branch is checked BEFORE the
    # workless-proof destroy, so a GPU-unbilled box can never be destroyed on
    # a timer — see the AMENDMENT in the docstring. "jobd never stamped" is
    # vacuous while the container does not yet exist.
    if v == "ZOMBIE_LOADING_STALL":
        return ZOMBIE_PARK, ("boot stalled in the GPU-UNBILLED loading phase — "
                             "park, never destroy (recoverable; the idle "
                             "reaper finishes in 2h unless kept/resumed)")
    if bool(is_jobs_box) and jobd_ever_stamped is False:
        return ZOMBIE_DESTROY, ("provably workless AND billing: running, full "
                                "GPU rate, jobd never stamped JOBD_STATUS "
                                "(readable absence)")
    # ZOMBIE_NO_JOBD without the never-ran proof:
    if jobd_hb_read:
        return ZOMBIE_PARK, ("jobd existed and died (heartbeat read, stale) — "
                             "park ends the GPU burn, disk kept 2h")
    return ZOMBIE_ALARM, ("jobd evidence unreadable — never act on "
                          "unreadable (I3)")


def stall_sweepable(*, verdict, is_jobs_box, jobd_ever_stamped, label_kept):
    """PURE. May the reaper auto-DESTROY this loading-stalled box?

    **RETIRED 2026-08-03: this is now ALWAYS False.** It is kept — still
    derived from `zombie_action`, never hard-coded — as the executable
    statement of the amendment above: no loading-phase box is destroy-eligible,
    because the phase is GPU-unbilled and destroy is irreversible. If a future
    edit to `zombie_action` ever makes this return True again, that edit has
    re-armed the branch that destroyed 46682313 90 s before its co-resident
    peer finished the same pull, and the test asserting False will say so.

    (Original 2026-07-30 rule, for history: True when a jobs-lane box's jobd
    had NEVER written jobs/nodes/<iid>/JOBD_STATUS — readable absence on B2 —
    with no `keep` token. The flaw was that jobd CANNOT stamp during `loading`,
    so the "proof" was a tautology of the phase.)"""
    act, _why = zombie_action(verdict=verdict, is_jobs_box=is_jobs_box,
                              jobd_ever_stamped=jobd_ever_stamped,
                              jobd_hb_read=False, label_kept=label_kept,
                              confirmed=True)
    return (act == ZOMBIE_DESTROY
            and str(verdict or "") == "ZOMBIE_LOADING_STALL")


def classify_jobs_durability(*, tickets):
    """PURE. `(verdict, reasons)` for a jobs-lane box.

    Each ticket: {id, status, results (list|None), declared_globs (int|None),
    events (chronological names)}. `results=None` means unreadable.

    The jobs lane is genuinely fail-safe by construction — `.uploaded` is written
    only when `rclone copy` returns 0 (onstart/jobd.sh:1283), the manifest is
    built from it, and the fold sets results only from an event carrying that
    manifest — so a failed upload yields an empty manifest, which is falsy, which
    is not durable. Two corrections keep it honest:

      * jobd writes the DONE marker EVEN WHEN PUBLISH-VERIFY FAILS, emitting
        `publish_verify_failed`. A trailing one of those is not durable.
      * a job whose `results:` globs matched nothing folds to `[]`, which is
        falsy forever — a benign permanent HOLD. Distinguish it: ZERO DECLARED
        globs is durable (nothing was ever meant to land); declared-some but
        folded-empty is a real HOLD.
    """
    ts = list(tickets or [])
    if not ts:
        return UNKNOWN, ["no readable job tickets"]

    reasons, verdict = [], DURABLE
    for t in ts:
        jid = t.get("id") or "<job>"
        status = str(t.get("status") or "").lower()
        if status not in ("done", "failed", "cancelled"):
            return UNKNOWN, [f"job {jid} is {status or 'unknown'} — not terminal"]

        ev = [str(e) for e in (t.get("events") or [])]
        if ev and "publish_verify_failed" in ev:
            last_ok = max((i for i, e in enumerate(ev)
                           if e in ("results_uploaded", "publish_verified")), default=-1)
            if max(i for i, e in enumerate(ev) if e == "publish_verify_failed") > last_ok:
                return UNSYNCED, [f"job {jid}: publish_verify_failed after the last "
                                  "successful publish — jobd writes DONE anyway "
                                  "(onstart/jobd.sh finalize), so DONE is not proof"]

        res, declared = t.get("results"), t.get("declared_globs")
        if res is None:
            return UNKNOWN, [f"job {jid}: results manifest unreadable"]
        if not res:
            if declared == 0:
                reasons.append(f"job {jid}: zero declared results globs — nothing "
                               "was ever meant to land (durable, not a HOLD)")
                continue
            return UNKNOWN, [f"job {jid}: declared {declared} results glob(s) but the "
                             "folded manifest is EMPTY — either the publish failed "
                             "or the globs matched nothing; both HOLD"]
        reasons.append(f"job {jid}: {len(res)} result file(s) on B2")
    return verdict, reasons


# --- orphaned queue tickets (the OTHER half of a box death) ------------------
# `zombie_action` above answers "what do we do to a dead BOX?". This answers the
# question nobody was asking: "what happens to the TICKETS it was holding?"
#
# A jobs-v2 ticket (jobs/queue/<box>/<jid>.json) is a pointer into ONE box's
# queue. Every ORDERLY box death moves it first — the boot-pull watchdog
# retargets every pending ticket to the replacement box BEFORE it destroys the
# condemned one (herdd `_job_pull_condemn`, 2026-08-02), `job cancel` deletes
# it, `job retarget` moves it. A DISORDERLY death — a manual `destroy`, the idle
# reaper, an eviction that nothing rescued — moves nothing, and the ticket is
# left pointing at an instance id that will never exist again.
#
# Nothing then reports it. The fold is honest about what it knows (`submitted`,
# because a `submitted` event is genuinely the newest one) and liveness is
# injected from the vast API only for a box that CLAIMED the job, so an unclaimed
# ticket has no instance_id to test and displays as plain `queued` forever. That
# is how box 46590907 left two phase1-cot tickets reading `submitted` on
# 2026-08-02 while the arms had already been resubmitted elsewhere: a returning
# agent reads "pending", not "dead".
#
# THE DISTINCTION THAT MAKES THIS SAFE: presence, not liveness. A PARKED box is
# absent from the live set and its tickets are perfectly healthy — jobd claims
# them on the next `herdd start`. Only a box that is absent from the ACCOUNT
# can never come back. So the input is `box_present` (is this instance id in
# `v1/instances/` at all, any actual_status), and it is THREE-valued: `None`
# means the listing could not be read, which must never read as "destroyed" —
# a soft API failure returns an empty instance list, and treating that as
# absence would classify the entire fleet's queue as orphaned at once.
TICKET_OK = "OK"                              # target box exists (live, loading or parked)
TICKET_UNKNOWN = "UNKNOWN"                    # instance listing unreadable -> no verdict
TICKET_ORPHAN_UNCLAIMED = "ORPHAN_UNCLAIMED"  # box gone, never claimed: pending FOREVER
TICKET_ORPHAN_INTERRUPTED = "ORPHAN_INTERRUPTED"   # box gone mid-run: may be resumable
TICKET_ORPHAN_TERMINAL = "ORPHAN_TERMINAL"    # box gone, job already ended: stale pointer

TICKET_ORPHANS = frozenset({TICKET_ORPHAN_UNCLAIMED, TICKET_ORPHAN_INTERRUPTED,
                            TICKET_ORPHAN_TERMINAL})
# The two an operator must ACT on. A terminal orphan is litter: the job reached
# an outcome, its results/events are on B2, and the leftover pointer is what
# makes `job ls` still list the box (which is a feature — it is the only per-box
# history view a returning agent gets). Reported, never swept.
TICKET_ORPHANS_STUCK = frozenset({TICKET_ORPHAN_UNCLAIMED, TICKET_ORPHAN_INTERRUPTED})


def ticket_orphan_verdict(*, box_present, job_status):
    """PURE. `(verdict, why)` for one queue ticket.

    `box_present` — True / False / None(unreadable), per the three-valued rule
    above. `job_status` — the FOLDED status (`jobmeta.fold_events`'s `status`,
    not `display_status`): submitted | claimed | started | done | failed |
    cancelled | unknown.

    Grading the orphan by how far the job got is what keeps the remedy correct:
    an UNCLAIMED orphan never ran, so cancelling it loses nothing; an
    INTERRUPTED one may have checkpoints under jobs/<JOB_ID>/checkpoints/ and
    wants `job retarget` (or `job requeue`) onto a live box FIRST — cancelling
    it is the destructive answer to a recoverable state."""
    if box_present is None:
        return TICKET_UNKNOWN, "instance listing unreadable — presence unproven"
    if box_present:
        return TICKET_OK, "target box exists"
    st = str(job_status or "unknown").lower()
    if st in ("done", "failed", "cancelled"):
        return (TICKET_ORPHAN_TERMINAL,
                f"target box is gone; job already {st} — stale queue pointer, "
                f"no work is stuck")
    if st in ("claimed", "started"):
        return (TICKET_ORPHAN_INTERRUPTED,
                "target box is gone mid-run; the job can never resume there — "
                "retarget/requeue it onto a live box, or cancel it")
    return (TICKET_ORPHAN_UNCLAIMED,
            f"target box is gone and the ticket was never claimed (status "
            f"{st}) — it stays `{st}` forever; nothing will ever run it")


def replay(*, runs, artifacts_rids, ckpt_objects, runs_dirs=None):
    """PURE. Score the predicate against recorded history (§11a).

    Replaces the review's 48 h dry-run soak, on the owner's 2026-07-30 ruling.
    A soak samples only the boxes that happen to park inside its window; replay
    sees the recorded history, including the shapes that already went wrong.

    WHAT REPLAY CAN AND CANNOT SHOW (2026-07-30 review, honest version). The
    predicate's booleans are derived from the same listings replay scores
    against, so "shipped predicate: 0 false-DURABLE" is a WIRING check, not an
    empirical result — `DURABLE ∧ no-evidence` is unsatisfiable by construction.
    The empirical content is elsewhere: (a) the deltas against the earlier,
    listing-blind predicates (which had real false-DURABLEs: chainmine-rb3-s2/s3
    off a log file, base-bakeoff-04/tuner-v0 off stale terminals); (b) the
    event-vs-listing disagreements (`done_but_empty_artifacts`); (c) the named
    rows, which a human can spot-check against B2. Replay can FALSIFY
    DURABLE-minting logic; it can never certify it complete — nothing recorded
    can prove the destroyed disk held nothing else.

    Population is the UNION of every rid-shaped prefix (`runs/` dirs, event
    streams, `artifacts/`, `checkpoints/`) — checkpoint-only legacy runs and
    event-less dirs classify UNKNOWN rather than silently vanishing.

    `runs`           {rid: {"events": [names...], "fail_reason": str|None,
                            "terminal_ts": epoch|None}}
    `artifacts_rids` set of rids with a non-empty artifacts/<rid>/ prefix
    `ckpt_objects`   {rid: [{"name": str, "mtime": epoch|None}, ...]}
                     (plain-string entries are accepted as name-only)
    `runs_dirs`      set of rids that have a runs/<rid>/ dir at all

    Jobs lane: NOT replayed here — validated by unit tests only. Implement a
    jobs replay before arming any jobs-box destroy.
    """
    runs = runs or {}
    ckpt_objects = ckpt_objects or {}
    art_set = set(artifacts_rids or ())
    population = sorted(set(runs) | art_set | set(ckpt_objects)
                        | set(runs_dirs or ()))
    rows, counts = [], {DURABLE: 0, UNSYNCED: 0, UNKNOWN: 0}
    false_durable, held_but_complete, done_empty_artifacts = [], [], []
    stale_payload_failed = []
    for rid in population:
        r = runs.get(rid) or {}
        ev = r.get("events") or []
        art = rid in art_set
        objs = []
        for o in ckpt_objects.get(rid) or []:
            if isinstance(o, str):
                objs.append({"name": o, "mtime": None})
            else:
                objs.append(o)
        pay_names = set(ckpt_payload_names([o["name"] for o in objs]))
        pay_mtimes = [o["mtime"] for o in objs
                      if o["name"] in pay_names and o.get("mtime") is not None]
        newest = max(pay_mtimes) if pay_mtimes else None
        fresh = payload_is_fresh(newest_payload_ts=newest,
                                 terminal_ts=r.get("terminal_ts"))
        verdict, reasons = classify_run_durability(
            events=ev, fail_reason=r.get("fail_reason"),
            artifacts_present=art, ckpt_payload_present=bool(pay_names),
            ckpt_payload_fresh=fresh)
        counts[verdict] = counts.get(verdict, 0) + 1
        rows.append({"rid": rid, "verdict": verdict, "artifacts": art,
                     "ckpt_payload": len(pay_names), "fresh": fresh,
                     "events": ev[-4:], "reason": reasons[0] if reasons else ""})
        if verdict == DURABLE and not (art or pay_names):
            false_durable.append(rid)          # wiring check — see docstring
        if "done" in ev and not art:
            done_empty_artifacts.append(rid)
        if "failed" in ev and pay_names and fresh is False:
            stale_payload_failed.append(rid)
        if verdict != DURABLE and "done" in ev and (art or pay_names):
            held_but_complete.append(rid)
    return {
        "n": len(rows), "counts": counts, "rows": rows,
        "false_durable": false_durable,
        "done_but_empty_artifacts": done_empty_artifacts,
        "stale_payload_failed": stale_payload_failed,
        "held_but_complete": held_but_complete,
        "arming_blocked": bool(false_durable),
    }


def replay_jobs(*, jobs):
    """PURE. Score `classify_jobs_durability` per historical job — §11a-R2
    precondition (c). Each job is classified as a box-of-one; the per-box
    grouping is an AND over tickets, so per-job replay exercises every branch
    the grouping folds over. Same honesty bound as `replay()`: verdicts are
    computed from the same folds the booleans came from, so the value is the
    named rows and the disagreement classes, not a self-consistent zero.

    `jobs` {jid: {"status": str, "results": list|None,
                  "declared_globs": int|None, "events": [names...]}}
    """
    rows, counts = [], {DURABLE: 0, UNSYNCED: 0, UNKNOWN: 0}
    verify_failed_final, done_empty_manifest = [], []
    for jid, t in sorted((jobs or {}).items()):
        ticket = {"id": jid, "status": t.get("status"),
                  "results": t.get("results"),
                  "declared_globs": t.get("declared_globs"),
                  "events": t.get("events") or []}
        verdict, reasons = classify_jobs_durability(tickets=[ticket])
        counts[verdict] = counts.get(verdict, 0) + 1
        rows.append({"jid": jid, "verdict": verdict,
                     "status": t.get("status"),
                     "n_results": (None if t.get("results") is None
                                   else len(t.get("results"))),
                     "reason": reasons[0] if reasons else ""})
        if verdict == UNSYNCED:
            verify_failed_final.append(jid)
        if (str(t.get("status") or "").lower() == "done"
                and not t.get("results")):
            done_empty_manifest.append(jid)
    return {"n": len(rows), "counts": counts, "rows": rows,
            "verify_failed_final": verify_failed_final,
            "done_empty_manifest": done_empty_manifest,
            "arming_blocked": False}


def keep_lease_state(*, label, deadline_ts=None, now_ts=None, reap_kept=None):
    """PURE. `(state, reason)` for the keep LEASE — §5.1 option (b).

    The label carries PRESENCE only; the deadline lives in workstation-local
    state. That split is deliberate: a vast label is one string already carrying
    identity (`run:<RID>`), the handoff-twin distinction (`:handoff`) and the reap
    opt-out (`keep`), and adding a fourth meaning broke a dup guard once already
    (see herdd._label_value). Consequence, stated plainly: a lease-unaware
    reader honors the token FOREVER — expiry is advisory to anything that has not
    read the local lease.

    states: `none` | `held` | `expired`
    """
    kept = reap_kept(label) if reap_kept else ("keep" in
                                              (t.strip().lower()
                                               for t in str(label or "").split(":")))
    if not kept:
        return "none", "no keep token in the label"
    if deadline_ts is None:
        return "held", ("keep token present, no local lease deadline — honored "
                        "indefinitely by any lease-unaware reader")
    if now_ts is not None and now_ts >= deadline_ts:
        return "expired", f"keep lease expired at {deadline_ts}"
    return "held", f"keep lease held until {deadline_ts}"


# --- impure edge: gather the evidence replay needs, in 3 rclone calls ---------

def _ev_epoch(ts):
    """Event-stream ts (20260710T083707393Z) -> epoch, or None."""
    try:
        import calendar
        import time as _t
        return float(calendar.timegm(_t.strptime(str(ts)[:15], "%Y%m%dT%H%M%S")))
    except (ValueError, TypeError):
        return None


def _mod_epoch(s):
    """rclone lsjson ModTime (2026-07-10T08:37:07.000Z, UTC) -> epoch, or None."""
    try:
        import calendar
        import time as _t
        return float(calendar.timegm(_t.strptime(str(s)[:19], "%Y-%m-%dT%H:%M:%S")))
    except (ValueError, TypeError):
        return None


def _env_bucket_runner(bucket, runner):
    """Shared impure-edge setup: repo .env, bucket, injectable rclone runner."""
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import vastconf
    import herdd
    vastconf.load_env()
    bucket = bucket or _os.environ.get("B2_BUCKET")
    if not bucket:
        raise RuntimeError("B2_BUCKET not set (env or .env)")
    return bucket, (runner or (lambda args: herdd._rclone_soft(args)[:2]))


def _replay_cache_dir(cache_dir=None):
    import os as _os
    cache = cache_dir or _os.path.join(
        _os.environ.get("XDG_CACHE_HOME", _os.path.expanduser("~/.cache")),
        "vast-parked-replay")
    _os.makedirs(cache, exist_ok=True)
    return cache


def gather_run_evidence(rid, *, bucket=None, cache_dir=None, runner=None):
    """IMPURE. Per-run evidence for ONE reap candidate — the kwargs of
    `classify_box_run_durability` minus `box_iid`. Scoped to the RID (4 rclone
    calls) so a 15-min reap pass stays cheap; the bulk `gather_replay_evidence`
    is for whole-history replay. Unreadable prefixes yield tri-state None,
    which the classifier HOLDs on (I3) — never an exception for a listing.
    """
    import glob
    import json as _json
    import os as _os
    bucket, run = _env_bucket_runner(bucket, runner)
    base = f"b2:{bucket}"
    cache = _replay_cache_dir(cache_dir)

    events, fail_reason, term_ts, emitter = [], None, None, None
    rc, _ = run(["copy", f"{base}/runs/{rid}/events/",
                 _os.path.join(cache, rid, "events"),
                 "--transfers", "16", "--checkers", "16"])
    raw = []
    if rc == 0:
        for fn in sorted(glob.glob(_os.path.join(cache, rid, "events", "*.json"))):
            try:
                with open(fn) as fh:
                    raw.append(_json.load(fh))
            except (OSError, ValueError):
                continue
        raw.sort(key=lambda e: str(e.get("ts") or ""))
        events = [str(e.get("event") or "") for e in raw]
        emitter = terminal_emitter(raw)
        for e in raw:
            name = str(e.get("event") or "")
            if name in _ORDER_MARKERS:
                term_ts = _ev_epoch(e.get("ts")) or term_ts
            if name == "failed" and e.get("reason"):
                fail_reason = str(e.get("reason"))

    rc, out = run(["lsf", f"{base}/artifacts/{rid}/"])
    artifacts_present = (bool((out or "").strip()) if rc == 0 else None)

    rc, out = run(["lsjson", "-R", "--files-only", f"{base}/checkpoints/{rid}/"])
    ckpt_present, ckpt_fresh = None, None
    if rc == 0:
        try:
            entries = _json.loads(out or "[]")
        except ValueError:
            entries = None
        if entries is not None:
            names = [str(e.get("Path") or "") for e in entries]
            pay = set(ckpt_payload_names(names))
            ckpt_present = bool(pay)
            mts = [_mod_epoch(e.get("ModTime")) for e in entries
                   if str(e.get("Path") or "") in pay]
            mts = [m for m in mts if m is not None]
            ckpt_fresh = payload_is_fresh(
                newest_payload_ts=max(mts) if mts else None,
                terminal_ts=term_ts)
    return {"events": events, "fail_reason": fail_reason,
            "artifacts_present": artifacts_present,
            "ckpt_payload_present": ckpt_present,
            "ckpt_payload_fresh": ckpt_fresh,
            "terminal_emitter_iid": emitter}


def job_ticket(jid, *, bucket=None, runner=None):
    """IMPURE. One job's `replay_jobs`/`classify_jobs_durability` ticket, via
    jobmeta's own fold + its incremental event cache (one rclone copy per job,
    then local reads — never one `cat` per event)."""
    import json as _json
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import jobmeta
    view = jobmeta.read_job(jid)   # jobmeta resolves bucket/runner itself
    cache = _os.path.join(
        _os.environ.get("XDG_CACHE_HOME", _os.path.expanduser("~/.cache")),
        "vast-jobmeta", jid, "events")
    evs = []
    try:
        for name in sorted(_os.listdir(cache)):
            if not name.endswith(".json"):
                continue
            try:
                with open(_os.path.join(cache, name)) as fh:
                    evs.append(_json.load(fh))
            except (OSError, ValueError):
                continue
    except OSError:
        pass
    evs.sort(key=lambda e: (str(e.get("ts") or ""), str(e.get("nonce") or "")))
    return {"status": view.get("status"), "results": view.get("results"),
            "declared_globs": view.get("declared_globs"),
            "events": [str(e.get("event") or "") for e in evs]}


def gather_jobs_replay_evidence(*, bucket=None, runner=None):
    """IMPURE. Every recorded job's ticket, for `replay_jobs` (§11a-R2 (c)).
    Population = `jobs/` dir listing minus the non-job `nodes/` prefix; the
    listing RAISES on failure (same rule as the run gather — a silently-empty
    population corrupts the report an arming decision reads)."""
    bucket, run = _env_bucket_runner(bucket, runner)
    rc, out = run(["lsf", "--dirs-only", f"b2:{bucket}/jobs/"])
    if rc != 0:
        raise RuntimeError(f"jobs/ listing failed rc={rc}")
    jids = sorted({l.strip().rstrip("/") for l in (out or "").splitlines()
                   if l.strip() and l.strip().rstrip("/") != "nodes"})
    return {"jobs": {jid: job_ticket(jid) for jid in jids}}


def gather_replay_evidence(*, bucket=None, cache_dir=None, runner=None):
    """IMPURE. Evidence for `replay()` over the WHOLE recorded history.

    Deliberately 3 calls, not 3-per-run: one bulk `copy` of every event body
    (incremental — the keys are immutable, so a warm run re-GETs nothing), plus
    one prefix listing each for artifacts/ and checkpoints/. Costs no GPU, no
    rental, and seconds. That is what makes replay a cheaper gate than a soak.
    """
    import glob
    import json as _json
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import runmeta
    import vastconf
    import herdd

    vastconf.load_env()          # repo-root .env, same resolution as the CLI
    bucket = bucket or _os.environ.get("B2_BUCKET")
    if not bucket:
        raise RuntimeError("B2_BUCKET not set (env or .env)")
    run = runner or (lambda args: herdd._rclone_soft(args)[:2])
    base = f"b2:{bucket}"
    cache = cache_dir or _os.path.join(
        _os.environ.get("XDG_CACHE_HOME", _os.path.expanduser("~/.cache")),
        "vast-parked-replay")
    _os.makedirs(cache, exist_ok=True)

    rc, _ = run(["copy", f"{base}/runs/", cache, "--include", "*/events/*.json",
                 "--fast-list", "--transfers", "16", "--checkers", "16"])
    if rc not in (0,):
        raise RuntimeError(f"bulk event copy failed rc={rc}")

    runs = {}
    for d in sorted(glob.glob(_os.path.join(cache, "*", "events"))):
        rid = _os.path.basename(_os.path.dirname(d))
        raw = []
        for fn in sorted(glob.glob(_os.path.join(d, "*.json"))):
            try:
                with open(fn) as fh:
                    raw.append(_json.load(fh))
            except (OSError, ValueError):
                continue
        if not raw:
            continue
        raw.sort(key=lambda e: str(e.get("ts") or ""))
        try:
            view = runmeta.fold_events(raw)
        except Exception:
            view = {}
        names = [str(e.get("event") or "") for e in raw]
        term_ts = None
        for e in raw:                       # ts of the LAST ordering marker
            if str(e.get("event") or "") in _ORDER_MARKERS:
                term_ts = _ev_epoch(e.get("ts")) or term_ts
        runs[rid] = {"events": names,
                     "fail_reason": (view or {}).get("fail_reason"),
                     "terminal_ts": term_ts}

    # Listings RAISE on failure: a silently-empty listing cannot mint DURABLE
    # (absence still reads UNKNOWN), but it would corrupt the replay report an
    # arming decision reads — population and delta counts would be wrong.
    rc, out = run(["lsf", "--dirs-only", f"{base}/runs/"])
    if rc != 0:
        raise RuntimeError(f"runs/ listing failed rc={rc}")
    runs_dirs = {l.strip().rstrip("/") for l in (out or "").splitlines() if l.strip()}

    rc, out = run(["lsf", "--dirs-only", f"{base}/artifacts/"])
    if rc != 0:
        raise RuntimeError(f"artifacts/ listing failed rc={rc}")
    art = {l.strip().rstrip("/") for l in (out or "").splitlines() if l.strip()}

    rc, out = run(["lsjson", "-R", "--files-only", f"{base}/checkpoints/",
                   "--fast-list"])
    if rc != 0:
        raise RuntimeError(f"checkpoints/ listing failed rc={rc}")
    try:
        entries = _json.loads(out or "[]")
    except ValueError as e:
        raise RuntimeError(f"checkpoints/ lsjson unparseable: {e}")

    ckpt = {}
    for ent in entries:
        p = str(ent.get("Path") or "")
        if "/" not in p:
            continue
        rid, rest = p.split("/", 1)
        ckpt.setdefault(rid, []).append(
            {"name": rest, "mtime": _mod_epoch(ent.get("ModTime"))})
    return {"runs": runs, "artifacts_rids": art, "ckpt_objects": ckpt,
            "runs_dirs": runs_dirs}


def _main(argv=None):
    import argparse
    import json as _json
    ap = argparse.ArgumentParser(
        prog="parked_lifecycle",
        description="durability predicate for parked boxes; `replay` scores it "
                    "against recorded history (docs/plans/parked-box-lifecycle.md §11a)")
    ap.add_argument("cmd", choices=["replay", "replay-jobs"])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--show", type=int, default=12, help="disagreeing rows to print")
    a = ap.parse_args(argv)

    if a.cmd == "replay-jobs":
        rep = replay_jobs(**gather_jobs_replay_evidence())
        if a.json:
            print(_json.dumps(rep, indent=2, default=str))
            return 0
        c = rep["counts"]
        print(f"== jobs replay over {rep['n']} recorded jobs ==")
        print(f"  DURABLE {c.get(DURABLE,0)}  UNSYNCED {c.get(UNSYNCED,0)}  "
              f"UNKNOWN/HOLD {c.get(UNKNOWN,0)}")
        vf = rep["verify_failed_final"]
        print(f"\n  trailing publish_verify_failed (DONE written anyway): {len(vf)}")
        for j in vf[:a.show]:
            print(f"    !! {j}")
        de = rep["done_empty_manifest"]
        print(f"  done with EMPTY results manifest (declared unknowable pre-"
              f"n_results_globs, so held): {len(de)}")
        for j in de[:a.show]:
            print(f"     - {j}")
        return 0

    ev = gather_replay_evidence()
    rep = replay(**ev)
    if a.json:
        print(_json.dumps(rep, indent=2, default=str))
        return 1 if rep["arming_blocked"] else 0

    c = rep["counts"]
    print(f"== replay over {rep['n']} recorded runs ==")
    print(f"  DURABLE {c.get(DURABLE,0)}  UNSYNCED {c.get(UNSYNCED,0)}  "
          f"UNKNOWN/HOLD {c.get(UNKNOWN,0)}")
    fd = rep["false_durable"]
    print(f"\n  FALSE DURABLE (wiring check — tautologically 0 for the shipped "
          f"predicate, see replay() docstring): {len(fd)}")
    for r in fd[:a.show]:
        print(f"    !! {r}")
    sp = rep["stale_payload_failed"]
    print(f"  failed-branch payload STALER than the terminal window: {len(sp)}")
    for r in sp[:a.show]:
        print(f"     - {r}")
    de = rep["done_but_empty_artifacts"]
    print(f"  terminal `done` with EMPTY artifacts/ (the shape the review found): {len(de)}")
    for r in de[:a.show]:
        print(f"     - {r}")
    hb = rep["held_but_complete"]
    print(f"  HELD despite looking complete (cost / alarm-fatigue class): {len(hb)}")
    for r in hb[:a.show]:
        print(f"     - {r}")
    print(f"\n  arming {'BLOCKED' if rep['arming_blocked'] else 'not blocked by replay'}")
    return 1 if rep["arming_blocked"] else 0


if __name__ == "__main__":
    import sys as _s
    _s.exit(_main())
