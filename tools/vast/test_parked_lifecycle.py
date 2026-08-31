"""The durability predicate, including every adversarial case §11 names.

Design: docs/plans/parked-box-lifecycle.md. The class of bug these exist to
prevent is a FALSE DURABLE — a verdict that says "safe to destroy" about a box
holding the only copy of some work. Every test below that asserts a non-DURABLE
verdict is guarding that, so a change which flips one to DURABLE is a data-loss
regression, not a test that needs updating.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parked_lifecycle as pl  # noqa: E402

D, U, UNK = pl.DURABLE, pl.UNSYNCED, pl.UNKNOWN


def v(**kw):
    return pl.classify_run_durability(**kw)[0]


# --- run lane: the defect the whole module exists for ------------------------

def test_done_with_artifacts_is_durable():
    assert v(events=["launched", "checkpoint", "done"], artifacts_present=True) == D


def test_done_with_EMPTY_artifacts_holds():
    """THE data-loss shape. train.sh's flush retries 3x then falls through with
    errexit off, and `done` is minted from the TRAINING rc — so the terminal event
    can exist while the payload never landed."""
    assert v(events=["launched", "checkpoint", "done"], artifacts_present=False) == UNK


def test_done_with_UNREADABLE_artifacts_holds_not_destroys():
    """None must never collapse to False: an unreadable listing is not evidence
    of absence."""
    assert v(events=["done"], artifacts_present=None) == UNK


def test_failed_branch_also_requires_landing_evidence():
    """The fix was originally applied to `done` only, leaving this alive on the
    branch where the run had ALREADY gone wrong. The flush loop runs for every RC.
    """
    assert v(events=["launched", "failed"], fail_reason="rc=1",
             ckpt_payload_present=False) == UNK
    assert v(events=["launched", "failed"], fail_reason="rc=1",
             ckpt_payload_present=True, ckpt_payload_fresh=True) == D


def test_failed_branch_payload_must_be_fresh():
    """F4 (2026-07-30 review): the final flush and the terminal event are pushed
    independently and there is no second copy on this branch, so hours-old
    mid-run checkpoints can sit as the 'payload' while the newest state died
    with the box. Stale ⇒ HOLD; unknown freshness ⇒ HOLD (I3), never DURABLE."""
    assert v(events=["failed"], fail_reason="rc=1",
             ckpt_payload_present=True, ckpt_payload_fresh=False) == UNK
    assert v(events=["failed"], fail_reason="rc=1",
             ckpt_payload_present=True, ckpt_payload_fresh=None) == UNK


def test_failed_never_consults_artifacts():
    """A failed run never pushes artifacts (train.sh:611 gates it on RC == 0), so
    an artifacts-based check would hold every failed run forever."""
    assert v(events=["failed"], fail_reason="rc=2", artifacts_present=False,
             ckpt_payload_present=True, ckpt_payload_fresh=True) == D


def test_max_hours_is_unsynced_never_durable():
    assert v(events=["launched", "failed"], fail_reason="max_hours",
             ckpt_payload_present=True) == U


def test_sync_failure_after_the_marker_wins():
    assert v(events=["done", "checkpoint_sync_failed"], artifacts_present=True) == U


def test_sync_failure_BEFORE_the_marker_is_superseded():
    """A mid-run sync failure that a later successful terminal flush displaced
    must not hold the box forever."""
    assert v(events=["checkpoint_sync_failed", "checkpoint", "done"],
             artifacts_present=True) == D


def test_sync_failure_with_no_terminal_marker_is_unsynced():
    assert v(events=["launched", "checkpoint_sync_failed"]) == U


def test_final_flush_marker_needs_checkpoint_payload():
    assert v(events=["preempted", "final_flush"], ckpt_payload_present=True,
             ckpt_payload_fresh=True) == D
    assert v(events=["preempted", "final_flush"], ckpt_payload_present=False) == UNK


def test_post_terminal_activity_makes_the_marker_stale():
    """F3 (2026-07-30 review): base-bakeoff-04 and tuner-v0 had DAYS of
    resumed sessions after their terminal `failed` (supervise relaunch,
    `herdd start`, the resume-guard's interactive idle) — sessions that write
    no terminal of their own, so whatever they produced is invisible. A
    marker-keyed read minted DURABLE for both. Any activity event after the
    terminal ⇒ the verdict must HOLD."""
    base_bakeoff_shape = ["launched", "supervisor_started", "failed",
                          "resumed", "launched", "resumed"]
    assert v(events=base_bakeoff_shape, fail_reason="rc=143",
             ckpt_payload_present=True, ckpt_payload_fresh=True) == UNK
    assert v(events=["done", "resumed"], artifacts_present=True) == UNK
    # non-activity trailers (CLI stop, heartbeats) do NOT stale the marker
    assert v(events=["done", "stopping", "heartbeat"], artifacts_present=True) == D


def test_no_events_and_no_marker_hold():
    assert v(events=[]) == UNK
    assert v(events=["launched", "checkpoint"]) == UNK


def test_zero_checkpoint_events_is_not_a_chronic_hold():
    """SAVE_STEPS=0 emits no `checkpoint` events at all. The predicate must not
    require them — that was the rejected freshness heuristic, which would have
    held such a box forever and trained everyone to ignore the alarm."""
    assert v(events=["launched", "done"], artifacts_present=True) == D


# --- the checkpoints/ listing is log-polluted --------------------------------
# ALLOWLIST semantics (2026-07-30 review F2): the original denylist named 4
# files, but chain-mining rcats `chainmine.log` into checkpoints/<RID>/ on
# every exit and other lanes write EVAL_STATUS / ARM_DONE / README.md there —
# so the denylist minted two real false-DURABLEs (chainmine-rb3-s2/s3) off a
# log file. Non-payload names grow with every runset; payload shapes don't.

def test_log_only_checkpoint_listing_is_not_payload():
    assert pl.ckpt_payload_names(
        ["STATUS", "onstart.log", "boot_phases.tsv", "HANDOFF_EPOCH"]) == []


def test_chainmine_log_is_not_payload():
    """THE recorded false-DURABLE: chainmine-rb3-s2/s3 hold only STATUS +
    chainmine.log on B2, ended terminal `failed rc=1`, and the denylist read
    the log as payload. A log file must never be landing evidence."""
    assert pl.ckpt_payload_names(["STATUS", "chainmine.log"]) == []
    assert pl.ckpt_payload_names(["EVAL_STATUS", "ARM_DONE", "README.md",
                                  "chat_template.jinja", "STOP", "EXTEND"]) == []


def test_payload_objects_survive_the_filter():
    names = ["STATUS", "checkpoint-500/adapter_model.safetensors", "onstart.log"]
    assert pl.ckpt_payload_names(names) == ["checkpoint-500/adapter_model.safetensors"]


def test_top_level_weight_shapes_are_payload():
    """tuner-v0's adapter lives at the prefix root, not under checkpoint-*.
    Weight/state shapes count wherever they sit."""
    names = ["adapter_model.safetensors", "adapter_config.json",
             "trainer_state.json", "model.bin", "README.md", "chainmine.log"]
    assert pl.ckpt_payload_names(names) == \
        ["adapter_model.safetensors", "adapter_config.json",
         "trainer_state.json", "model.bin"]


def test_checkpoint_dir_component_admits_its_contents():
    """Anything under a checkpoint-* path component is trainer save-dir output,
    whatever its basename."""
    assert pl.ckpt_payload_names(["checkpoint-1/STATUS_REPORT.json"]) == \
        ["checkpoint-1/STATUS_REPORT.json"]
    assert pl.ckpt_payload_names(["arms/a0/checkpoint-40/rng_state.pth"]) == \
        ["arms/a0/checkpoint-40/rng_state.pth"]


def test_payload_freshness_tristate():
    f = pl.payload_is_fresh
    assert f(newest_payload_ts=1000.0, terminal_ts=1000.0) is True
    assert f(newest_payload_ts=1000.0 - pl.FRESH_WINDOW_S, terminal_ts=1000.0) is True
    assert f(newest_payload_ts=1000.0 - pl.FRESH_WINDOW_S - 1, terminal_ts=1000.0) is False
    assert f(newest_payload_ts=None, terminal_ts=1000.0) is None
    assert f(newest_payload_ts=1000.0, terminal_ts=None) is None


# --- jobs lane ---------------------------------------------------------------

def jv(**kw):
    return pl.classify_jobs_durability(**kw)[0]


def test_job_done_with_results_is_durable():
    assert jv(tickets=[{"id": "j1", "status": "done", "results": ["a", "b"],
                        "declared_globs": 1, "events": ["results_uploaded", "done"]}]) == D


def test_publish_verify_failed_after_publish_is_not_durable():
    """jobd writes the DONE marker even when verify fails."""
    assert jv(tickets=[{"id": "j1", "status": "done", "results": ["a"],
                        "declared_globs": 1,
                        "events": ["results_uploaded", "publish_verify_failed", "done"]}]) == U


def test_publish_verify_failed_then_a_later_success_is_durable():
    assert jv(tickets=[{"id": "j1", "status": "done", "results": ["a"],
                        "declared_globs": 1,
                        "events": ["publish_verify_failed", "results_uploaded", "done"]}]) == D


def test_zero_declared_globs_is_durable_not_a_permanent_hold():
    """A job that never declared results folds to [] forever. Holding it would be
    a permanent false alarm on a legitimately empty job."""
    assert jv(tickets=[{"id": "j1", "status": "done", "results": [],
                        "declared_globs": 0, "events": ["done"]}]) == D


def test_declared_but_empty_manifest_holds():
    assert jv(tickets=[{"id": "j1", "status": "done", "results": [],
                        "declared_globs": 3, "events": ["done"]}]) == UNK


def test_unreadable_manifest_holds():
    assert jv(tickets=[{"id": "j1", "status": "done", "results": None,
                        "declared_globs": 1, "events": ["done"]}]) == UNK


def test_a_non_terminal_ticket_holds_the_whole_box():
    assert jv(tickets=[{"id": "j1", "status": "done", "results": ["a"],
                        "declared_globs": 1, "events": ["done"]},
                       {"id": "j2", "status": "running", "results": None,
                        "declared_globs": 1, "events": []}]) == UNK


def test_no_tickets_holds():
    assert jv(tickets=[]) == UNK


# --- keep lease (§5.1 option b) ----------------------------------------------

def test_keep_lease_states():
    assert pl.keep_lease_state(label="run:r1")[0] == "none"
    assert pl.keep_lease_state(label="run:r1:keep")[0] == "held"
    assert pl.keep_lease_state(label="run:r1:keep", deadline_ts=100,
                              now_ts=50)[0] == "held"
    assert pl.keep_lease_state(label="run:r1:keep", deadline_ts=100,
                              now_ts=100)[0] == "expired"


def test_keep_token_with_no_lease_is_honored_indefinitely():
    """The honest cost of option (b): a lease-unaware reader sees only presence.
    This is the 46193810 failure mode (12h at $2.13/day on a blanket keep) and the
    doc must not pretend otherwise."""
    state, why = pl.keep_lease_state(label="keep:fleetd-park")
    assert state == "held" and "indefinitely" in why


def test_keep_lease_delegates_to_the_real_reap_predicate():
    """Wired to herdd._reap_kept so the two cannot drift on what a token is."""
    import herdd
    for lab in ["keep", "run:r1:keep", "keep:why", "KEEP"]:
        assert pl.keep_lease_state(label=lab, reap_kept=herdd._reap_kept)[0] == "held"
    for lab in ["run:r1", "", "keeper"]:
        assert pl.keep_lease_state(label=lab, reap_kept=herdd._reap_kept)[0] == "none"


# --- the invariant, stated as a test ----------------------------------------

def test_I3_unreadable_evidence_never_accelerates_a_destroy():
    """No combination of unreadable evidence may yield DURABLE."""
    for kw in [dict(events=["done"], artifacts_present=None),
               dict(events=["failed"], fail_reason="rc=1", ckpt_payload_present=None),
               dict(events=["failed"], fail_reason="rc=1",
                    ckpt_payload_present=True, ckpt_payload_fresh=None),
               dict(events=["final_flush"], ckpt_payload_present=None),
               dict(events=[])]:
        assert pl.classify_run_durability(**kw)[0] != D, kw
    for t in [None, [], [{"id": "j", "status": "done", "results": None,
                          "declared_globs": 1, "events": []}]]:
        assert pl.classify_jobs_durability(tickets=t)[0] != D, t


def test_every_verdict_carries_a_reason():
    """The journal is the only record a destroyed box leaves, so a verdict with
    no reason is unauditable."""
    verdict, reasons = pl.classify_run_durability(events=["done"], artifacts_present=True)
    assert reasons and all(isinstance(r, str) and r for r in reasons)
    verdict, reasons = pl.classify_jobs_durability(tickets=[])
    assert reasons and all(isinstance(r, str) and r for r in reasons)


# --- per-box applicability (§11a-R2 precondition d) --------------------------

def test_durable_applies_only_to_the_terminal_emitter():
    """Handoff twins and same-RID relaunches share one runs/<RID>/events/
    stream: DURABLE read off ANOTHER box's terminal must not license destroying
    this box. Unknown emitter cannot be attributed => HOLD (I3)."""
    kw = dict(events=["launched", "done"], artifacts_present=True)
    assert pl.classify_box_run_durability(
        box_iid="111", terminal_emitter_iid="111", **kw)[0] == D
    assert pl.classify_box_run_durability(
        box_iid="222", terminal_emitter_iid="111", **kw)[0] == UNK
    assert pl.classify_box_run_durability(
        box_iid="111", terminal_emitter_iid=None, **kw)[0] == UNK
    # int/str iid representations must not defeat the match
    assert pl.classify_box_run_durability(
        box_iid=111, terminal_emitter_iid="111", **kw)[0] == D


def test_non_durable_verdicts_pass_through_regardless_of_emitter():
    """Holding is always applicable — the emitter rule only guards DURABLE."""
    v, _ = pl.classify_box_run_durability(
        box_iid="222", terminal_emitter_iid="111",
        events=["failed"], fail_reason="max_hours", ckpt_payload_present=True)
    assert v == U


def test_terminal_emitter_follows_the_last_marker():
    evs = [{"event": "launched", "instance_id": "100", "actor": "cli:host"},
           {"event": "failed", "actor": "box_100"},
           {"event": "done", "actor": "box_200"}]
    assert pl.terminal_emitter(evs) == "200"
    # an unattributable LAST marker resets attribution — the earlier marker is
    # not the one being trusted
    evs.append({"event": "failed", "actor": "supervisor"})
    assert pl.terminal_emitter(evs) is None
    assert pl.terminal_emitter([]) is None


# --- the stall sweep rule (zombie 46256890) ----------------------------------

def test_stall_sweep_is_retired_no_loading_box_is_destroy_eligible():
    """RETIRED 2026-08-03 (boxes 46682313/46682177): the loading phase is not
    destroy-eligible at all, so this predicate is now False for EVERY input —
    including the one shape it used to license.

    The old rule read "jobs box + readable absence of JOBD_STATUS = provably
    workless". During `loading` that proof is vacuous: jobd cannot stamp before
    the container exists, so the predicate was really just "is this a jobs
    box?", and it destroyed 46682313 at 38m of a slow-but-live pull — 90 s
    before its co-resident twin on the same image cleared the identical verdict
    to OK. `loading` is GPU-unbilled, so the destroy saved ~$0.01/hr and spent
    an irreversible action to do it.

    Kept (still DERIVED from zombie_action, never hard-coded) so that any
    future edit re-arming that branch fails here rather than in production."""
    ok = dict(verdict="ZOMBIE_LOADING_STALL", is_jobs_box=True,
              jobd_ever_stamped=False, label_kept=False)
    assert not pl.stall_sweepable(**ok)                                  # was True
    assert not pl.stall_sweepable(**{**ok, "jobd_ever_stamped": True})   # resumed
    assert not pl.stall_sweepable(**{**ok, "jobd_ever_stamped": None})   # I3
    assert not pl.stall_sweepable(**{**ok, "is_jobs_box": False})
    assert not pl.stall_sweepable(**{**ok, "label_kept": True})
    assert not pl.stall_sweepable(**{**ok, "verdict": "ZOMBIE_NO_JOBD"})
    assert not pl.stall_sweepable(**{**ok, "verdict": None})


# --- the graded zombie policy (zombie 46633685) ------------------------------

def _za(**kw):
    base = dict(verdict="ZOMBIE_LOADING_STALL", is_jobs_box=False,
                jobd_ever_stamped=None, jobd_hb_read=False,
                label_kept=False, confirmed=True)
    base.update(kw)
    return pl.zombie_action(**base)[0]


def test_zombie_destroy_needs_the_workless_proof_AND_a_billed_phase():
    """DESTROY needs BOTH halves: the workless proof (jobs box + readable
    ABSENCE of JOBD_STATUS) AND a phase that actually bills. The running shape
    (46633685-class GPU burn) still destroys. The loading shape does NOT, as of
    2026-08-03 — see test_loading_stall_is_never_destroy_eligible below."""
    assert _za(verdict="ZOMBIE_NO_JOBD", is_jobs_box=True,
               jobd_ever_stamped=False) == pl.ZOMBIE_DESTROY
    assert _za(verdict="ZOMBIE_LOADING_STALL", is_jobs_box=True,
               jobd_ever_stamped=False) == pl.ZOMBIE_PARK


def test_loading_stall_is_never_destroy_eligible():
    """THE 46682313 REGRESSION TEST. A box stalled in `loading` is GPU-unbilled
    (invoice-verified: storage only, ~$0.01/hr — the destroyed box accrued
    $0.173 total, $0.00 GPU), and the verdict itself is a proven false positive
    at these ages: co-resident 46682177, same image, was flagged
    ZOMBIE_LOADING_STALL at 27m and came up healthy at 40m.

    So NO combination of lane and evidence may destroy in this phase — the
    strongest licensed action is a recoverable PARK, whatever `is_jobs_box` and
    `jobd_ever_stamped` say. The old policy destroyed exactly the first case
    below and parked the second, which is how two boxes hitting one slow pull
    got opposite fates."""
    for jobs, stamped in ((True, False), (True, True), (True, None),
                          (False, False), (False, None)):
        act, why = pl.zombie_action(
            verdict="ZOMBIE_LOADING_STALL", is_jobs_box=jobs,
            jobd_ever_stamped=stamped, jobd_hb_read=False,
            label_kept=False, confirmed=True)
        assert act != pl.ZOMBIE_DESTROY, (jobs, stamped, why)
        assert act == pl.ZOMBIE_PARK, (jobs, stamped, why)
        assert "UNBILLED" in why


def test_zombie_stall_without_proof_parks_never_destroys():
    """The 2026-08-02 incident shape: a serve-lane box dead in `loading` past
    the deadline. No never-ran proof exists for a non-jobs box, so the action
    degrades to PARK — end the bleed, keep the disk, 2h idle fuse follows."""
    assert _za(is_jobs_box=False) == pl.ZOMBIE_PARK
    # resumed jobs box (marker present): disk has history -> park, not destroy
    assert _za(is_jobs_box=True, jobd_ever_stamped=True) == pl.ZOMBIE_PARK
    # unreadable marker on a jobs box: I3 blocks destroy; stall still parks
    assert _za(is_jobs_box=True, jobd_ever_stamped=None) == pl.ZOMBIE_PARK


def test_zombie_no_jobd_grades_on_heartbeat_readability():
    """Running-but-dead: an affirmatively READ stale heartbeat proves jobd
    existed and died -> park. An UNREADABLE heartbeat could be a local B2
    outage making the whole fleet look dead -> alarm only (I3)."""
    assert _za(verdict="ZOMBIE_NO_JOBD", is_jobs_box=True,
               jobd_ever_stamped=True, jobd_hb_read=True) == pl.ZOMBIE_PARK
    assert _za(verdict="ZOMBIE_NO_JOBD", is_jobs_box=True,
               jobd_ever_stamped=True, jobd_hb_read=False) == pl.ZOMBIE_ALARM
    assert _za(verdict="ZOMBIE_NO_JOBD", is_jobs_box=True,
               jobd_ever_stamped=None, jobd_hb_read=False) == pl.ZOMBIE_ALARM


def test_zombie_confirmation_gates_every_action():
    """One snapshot is never no-progress evidence: unconfirmed sightings stay
    alarm-only regardless of proof strength."""
    assert _za(is_jobs_box=True, jobd_ever_stamped=False,
               confirmed=False) == pl.ZOMBIE_ALARM
    assert _za(is_jobs_box=False, confirmed=False) == pl.ZOMBIE_ALARM
    assert _za(verdict="ZOMBIE_NO_JOBD", is_jobs_box=True,
               jobd_ever_stamped=True, jobd_hb_read=True,
               confirmed=False) == pl.ZOMBIE_ALARM


def test_zombie_keep_and_alive_shapes_are_untouchable():
    assert _za(label_kept=True, is_jobs_box=True,
               jobd_ever_stamped=False) == pl.ZOMBIE_ALARM
    # jobd ALIVE (ticket-claiming bug): never auto-touch a functioning box
    assert _za(verdict="ZOMBIE_TICKET_UNCLAIMED", is_jobs_box=True,
               jobd_ever_stamped=True, jobd_hb_read=True) == pl.ZOMBIE_ALARM
    assert _za(verdict="OK") == pl.ZOMBIE_ALARM
    assert _za(verdict=None) == pl.ZOMBIE_ALARM


# --- jobs replay -------------------------------------------------------------

def test_replay_jobs_branches():
    rep = pl.replay_jobs(jobs={
        "j-good": {"status": "done", "results": ["a"], "declared_globs": 1,
                   "events": ["submitted", "results_uploaded", "done"]},
        "j-verify": {"status": "done", "results": ["a"], "declared_globs": 1,
                     "events": ["results_uploaded", "publish_verify_failed",
                                "done"]},
        "j-empty": {"status": "done", "results": [], "declared_globs": None,
                    "events": ["done"]},
        "j-running": {"status": "running", "results": None,
                      "declared_globs": None, "events": ["claimed"]},
    })
    by = {r["jid"]: r["verdict"] for r in rep["rows"]}
    assert by["j-good"] == D
    assert by["j-verify"] == U
    assert rep["verify_failed_final"] == ["j-verify"]
    # declared unknowable (pre-n_results_globs stream) + empty manifest => HOLD
    assert by["j-empty"] == UNK
    assert rep["done_empty_manifest"] == ["j-empty"]
    assert by["j-running"] == UNK


# --- replay: the recorded failure shapes, end to end -------------------------

def test_replay_chainmine_shape_holds_not_durable():
    """The full F2 shape as recorded on B2: terminal failed rc=1, checkpoints/
    holds only STATUS + chainmine.log. The denylist predicate said DURABLE."""
    rep = pl.replay(
        runs={"chainmine-rb3-s2": {"events": ["launched", "failed"],
                                   "fail_reason": "rc=1", "terminal_ts": 2000.0}},
        artifacts_rids=set(),
        ckpt_objects={"chainmine-rb3-s2": [
            {"name": "STATUS", "mtime": 1990.0},
            {"name": "chainmine.log", "mtime": 1999.0}]})
    row = rep["rows"][0]
    assert row["verdict"] == pl.UNKNOWN and row["ckpt_payload"] == 0
    assert not rep["arming_blocked"]


def test_replay_stale_payload_is_flagged_and_held():
    """F4: payload exists but its newest object predates the terminal by more
    than the flush window — only mid-run checkpoints landed."""
    rep = pl.replay(
        runs={"r1": {"events": ["launched", "failed"], "fail_reason": "rc=1",
                     "terminal_ts": 100000.0}},
        artifacts_rids=set(),
        ckpt_objects={"r1": [{"name": "checkpoint-40/adapter_model.safetensors",
                              "mtime": 100000.0 - pl.FRESH_WINDOW_S - 3600}]})
    assert rep["rows"][0]["verdict"] == pl.UNKNOWN
    assert rep["stale_payload_failed"] == ["r1"]


def test_replay_population_is_the_union_of_prefixes():
    """F5: 20 checkpoint-only legacy prefixes and 2 event-less runs/ dirs were
    invisible to an events-derived population. They must appear, as UNKNOWN."""
    rep = pl.replay(
        runs={"r-ev": {"events": ["launched", "done"], "fail_reason": None,
                       "terminal_ts": 10.0}},
        artifacts_rids={"r-ev", "r-art-only"},
        ckpt_objects={"r-ckpt-only": [
            {"name": "adapter_model.safetensors", "mtime": 5.0}]},
        runs_dirs={"r-ev", "r-dir-no-events"})
    by = {r["rid"]: r["verdict"] for r in rep["rows"]}
    assert set(by) == {"r-ev", "r-art-only", "r-ckpt-only", "r-dir-no-events"}
    assert by["r-ev"] == pl.DURABLE          # done + artifacts present
    assert by["r-art-only"] == pl.UNKNOWN    # no events -> HOLD
    assert by["r-ckpt-only"] == pl.UNKNOWN
    assert by["r-dir-no-events"] == pl.UNKNOWN
