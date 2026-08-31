"""Portable tests for the durable zombie-sweep layer (BOOT_HEALTHCHECK_DESIGN.md
Track C): herdd's pure `classify_box_health`, the JOBD_STATUS heartbeat
parsing, `gather_fleet_health`'s bounded B2 reads, the `_render_ls` loud scream,
and the `guard` subcommand's --fix gating + exit codes.

Toolchain-free lane (`pytest -m "not integration"`): NO vast API, NO B2/rclone,
NO network — every seam is injected (`now=`) or monkeypatched.
"""
import argparse
import datetime
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vastlib.boxes import health as vhealth  # noqa: E402
from vastlib.boxes import lifecycle as vlifecycle  # noqa: E402
from vastlib.boxes import reap as vreap  # noqa: E402
from vastlib.cli import _ls_render as vlsrender  # noqa: E402
from vastlib.cli import guard as vguard  # noqa: E402
from vastlib.core import fmt as vfmt  # noqa: E402
from vastlib.fleet import client as vfleetclient  # noqa: E402
from vastlib.jobs import view as vjobsview  # noqa: E402
from vastlib.storage import b2 as vb2  # noqa: E402


# --- ts builders (mirror the two on-wire stamp forms) ------------------------ #
def _jm_ts(epoch):
    """jobmeta/runmeta now_ts form: colon-free YYYYMMDDTHHMMSSmmmZ."""
    dt = datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%S") + "000Z"


def _jm_job_id(epoch, slug="arm"):
    """`job submit`'s JOB_ID form — the timestamp prefix is the SUBMIT stamp,
    and the only place the fold carries one for a running job."""
    dt = datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%S") + f"-{slug}-0000"


def _ftz(epoch):
    """JOBD_STATUS heartbeat form: %Y-%m-%dT%H:%M:%SZ (box-side date -u +%FT%TZ)."""
    dt = datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


NOW = 1_000_000.0
LOAD_DL = 1500        # GUARD_LOADING_DEADLINE_S default
STALE = 600           # GUARD_JOBD_STALE_S default


def _box(status="running", *, age=None, cred_jobs=False, iid=42, mid=7):
    b = {"id": iid, "actual_status": status, "machine_id": mid}
    if age is not None:
        b["start_date"] = NOW - age
    if cred_jobs:
        b["extra_env"] = [["CRED_ROLE", "jobs"]]
    return b


# --- pure ts parsers --------------------------------------------------------- #
def test_iso_ftz_to_epoch_roundtrip():
    assert vhealth._iso_ftz_to_epoch(_ftz(NOW)) == NOW
    assert vhealth._iso_ftz_to_epoch("") is None
    assert vhealth._iso_ftz_to_epoch("not-a-ts") is None
    assert vhealth._iso_ftz_to_epoch(None) is None


def test_jobd_status_hb_epoch():
    # the ts is scanned for (state field is 1 OR 2 tokens; extra trails after).
    assert vhealth._jobd_status_hb_epoch(f"IDLE {_ftz(NOW)}") == NOW              # 1-word state
    assert vhealth._jobd_status_hb_epoch(f"STAGING {_ftz(NOW)} staging=a mbps=3") == NOW
    # "RUNNING <njobs> <ts> staging=.. mbps=.. pids" — ts is the 3rd token
    running = f"RUNNING 2 {_ftz(NOW)} staging=asset mbps=12 pid1 pid2"
    assert vhealth._jobd_status_hb_epoch(running) == NOW
    assert vhealth._jobd_status_hb_epoch("IDLE") is None
    assert vhealth._jobd_status_hb_epoch("") is None
    assert vhealth._jobd_status_hb_epoch("RUNNING garbage extra") is None


def test_unclaimed_ticket_age():
    jobs = [
        {"display_status": "running", "last_event_ts": _jm_ts(NOW - 10)},
        {"display_status": "submitted", "last_event_ts": _jm_ts(NOW - 2000)},
        {"display_status": "submitted", "last_event_ts": _jm_ts(NOW - 500)},
    ]
    # oldest SUBMITTED ticket wins; running/claimed are ignored
    assert vhealth._guard_unclaimed_ticket_age(jobs, NOW) == pytest.approx(2000, abs=1)
    assert vhealth._guard_unclaimed_ticket_age([], NOW) is None


# --- classify_box_health: BOOTING vs LOADING_STALL --------------------------- #
def test_loading_within_deadline_is_booting():
    h = vhealth.classify_box_health(_box("loading", age=LOAD_DL - 1), now=NOW)
    assert h.verdict == vhealth.GUARD_BOOTING
    assert h.verdict not in vhealth._ZOMBIE_VERDICTS


def test_loading_at_deadline_boundary_still_booting():
    # boundary: age == deadline is NOT past it (strict >), so still BOOTING
    h = vhealth.classify_box_health(_box("loading", age=LOAD_DL), now=NOW)
    assert h.verdict == vhealth.GUARD_BOOTING


def test_loading_past_deadline_is_zombie_stall():
    h = vhealth.classify_box_health(_box("loading", age=LOAD_DL + 1), now=NOW)
    assert h.verdict == vhealth.GUARD_ZOMBIE_LOADING_STALL
    assert h.age_s == LOAD_DL + 1


def test_created_past_deadline_is_zombie_stall():
    h = vhealth.classify_box_health(_box("created", age=4 * 3600), now=NOW)
    assert h.verdict == vhealth.GUARD_ZOMBIE_LOADING_STALL


# --- classify_box_health: running lane --------------------------------------- #
def test_stopped_box_is_ok():
    h = vhealth.classify_box_health(_box("stopped", age=99999), now=NOW)
    assert h.verdict == vhealth.GUARD_OK


def test_running_non_jobs_box_is_ok():
    # a plain train/serve box (no CRED_ROLE=jobs, no tickets) never gets a
    # jobd expectation -> OK even if very old with no heartbeat
    h = vhealth.classify_box_health(_box("running", age=99999), now=NOW)
    assert h.verdict == vhealth.GUARD_OK
    assert h.evidence["is_jobs_box"] is False


def test_running_jobs_box_fresh_jobd_is_ok():
    h = vhealth.classify_box_health(_box("running", age=99999, cred_jobs=True),
                                    jobd_hb_epoch=NOW - 30, now=NOW)
    assert h.verdict == vhealth.GUARD_OK


def test_running_jobs_box_stale_jobd_is_zombie():
    h = vhealth.classify_box_health(_box("running", age=99999, cred_jobs=True),
                                    jobd_hb_epoch=NOW - (STALE + 60), now=NOW)
    assert h.verdict == vhealth.GUARD_ZOMBIE_NO_JOBD
    assert h.age_s == STALE + 60


def test_running_jobs_box_absent_jobd_old_box_is_zombie():
    # jobd never once stamped AND the box is well past the boot horizon
    h = vhealth.classify_box_health(_box("running", age=LOAD_DL + 100, cred_jobs=True),
                                    jobd_hb_epoch=None, now=NOW)
    assert h.verdict == vhealth.GUARD_ZOMBIE_NO_JOBD


def test_running_jobs_box_absent_jobd_young_box_is_booting_envsetup():
    # just reached running: jobd hasn't had time to stamp -> NOT flagged as a
    # zombie, but no longer the old misleading OK either: it is BOOTING in the
    # env-setup phase (billed GPU — onstart/jobd bootstrap provisioning).
    h = vhealth.classify_box_health(_box("running", age=90, cred_jobs=True),
                                    jobd_hb_epoch=None, now=NOW)
    assert h.verdict == vhealth.GUARD_BOOTING
    assert h.verdict not in vhealth._ZOMBIE_VERDICTS
    assert h.evidence["phase"] == "env-setup"
    assert "BILLED" in h.reason


def test_jobs_box_detected_via_folded_tickets():
    # no CRED_ROLE marker (ssh `job attach` box) but a ticket folded on -> jobs
    jobs = [{"display_status": "running", "last_event_ts": _jm_ts(NOW - 10)}]
    h = vhealth.classify_box_health(_box("running", age=99999), jobs=jobs,
                                    jobd_hb_epoch=None, now=NOW)
    assert h.evidence["is_jobs_box"] is True
    # absent jobd on an old box -> zombie
    assert h.verdict == vhealth.GUARD_ZOMBIE_NO_JOBD


def test_ticket_unclaimed_when_jobd_alive():
    # jobd heartbeat fresh, but a submitted ticket sits unclaimed past deadline
    jobs = [{"display_status": "submitted",
             "last_event_ts": _jm_ts(NOW - (LOAD_DL + 200))}]
    h = vhealth.classify_box_health(_box("running", age=99999, cred_jobs=True),
                                    jobs=jobs, jobd_hb_epoch=NOW - 30, now=NOW)
    assert h.verdict == vhealth.GUARD_ZOMBIE_TICKET_UNCLAIMED
    assert h.evidence["ticket_age_s"] == LOAD_DL + 200


def test_a_ticket_queued_behind_running_work_is_not_a_zombie():
    """THE LIVE FALSE POSITIVE, 2026-08-17, box 47976929.

    Two tickets on a one-card box: `v13-chain-full-9b-r64-train` claimed and
    running, `gc-hybrid-prep-9b-r64` submitted five minutes later and queued
    behind it. jobd is doing exactly what JOBS_DESIGN §2 says — a `gpus: "all"`
    arm blocks the FIFO, which is the shape we RECOMMEND for same-model arms —
    and the old rule called it a zombie at 25 min and re-raised the alarm on
    every reaper pass for 84 min.

    The queue age is still recorded: withholding the verdict must not also
    withhold the fact, or `guard --json` loses the only number that would let
    an operator notice a queue that never drains."""
    claimed = NOW - (LOAD_DL + 900)
    jobs = [{"job_id": _jm_job_id(claimed - 60),
             "display_status": "running", "started_at": _jm_ts(claimed),
             "last_event_ts": _jm_ts(NOW - 5)},
            {"display_status": "submitted",
             "last_event_ts": _jm_ts(NOW - (LOAD_DL + 200))}]
    h = vhealth.classify_box_health(_box("running", age=99999, cred_jobs=True),
                                    jobs=jobs, jobd_hb_epoch=NOW - 30, now=NOW)
    assert h.verdict == vhealth.GUARD_OK
    assert h.evidence["ticket_fifo_blocked"] is True
    assert h.evidence["ticket_age_s"] == LOAD_DL + 200


def test_a_claim_that_jumped_the_queue_still_flags():
    """The other side of the same gate — and the reason it is a comparison and
    not just "is anything running".

    Here the ticket was queued BEFORE the running job was SUBMITTED. Strict
    FIFO forbids that: the oldest ticket that does not fit blocks the younger
    ones, so running work that entered the queue after this ticket means the
    ticket was SKIPPED, not waiting. That is a real claiming bug and still
    deserves the alarm.

    Note both sides are submit times. Claim time cannot order this — under a
    batch submit every claim post-dates every submit, which is what made the
    first version of this gate fire on the happy path (box 47999495)."""
    queued = NOW - (LOAD_DL + 900)
    jobs = [{"display_status": "submitted", "last_event_ts": _jm_ts(queued)},
            {"job_id": _jm_job_id(queued + 60),
             "display_status": "running", "started_at": _jm_ts(queued + 120),
             "last_event_ts": _jm_ts(NOW - 5)}]
    h = vhealth.classify_box_health(_box("running", age=99999, cred_jobs=True),
                                    jobs=jobs, jobd_hb_epoch=NOW - 30, now=NOW)
    assert h.verdict == vhealth.GUARD_ZOMBIE_TICKET_UNCLAIMED
    assert h.evidence["ticket_fifo_blocked"] is False
    assert "FIFO skip" in h.reason


def test_an_idle_daemon_holding_a_ticket_names_that_shape():
    """No running job at all: jobd is up, executing nothing, and still has not
    claimed. Unambiguous, and the reason must SAY so — the operator's next move
    differs from the skip case above."""
    jobs = [{"display_status": "submitted",
             "last_event_ts": _jm_ts(NOW - (LOAD_DL + 200))}]
    h = vhealth.classify_box_health(_box("running", age=99999, cred_jobs=True),
                                    jobs=jobs, jobd_hb_epoch=NOW - 30, now=NOW)
    assert h.verdict == vhealth.GUARD_ZOMBIE_TICKET_UNCLAIMED
    assert "running nothing" in h.reason


def test_no_jobd_precedes_ticket_check():
    # both a dead jobd AND an old ticket -> NO_JOBD (root cause) wins
    jobs = [{"display_status": "submitted",
             "last_event_ts": _jm_ts(NOW - (LOAD_DL + 200))}]
    h = vhealth.classify_box_health(_box("running", age=99999, cred_jobs=True),
                                    jobs=jobs, jobd_hb_epoch=None, now=NOW)
    assert h.verdict == vhealth.GUARD_ZOMBIE_NO_JOBD


# --- boot-phase split (2026-08-02): loading (GPU unbilled) vs env-setup ------ #
ENV_DL = 900          # GUARD_ENVSETUP_DEADLINE_S default


def test_phase_evidence_loading():
    h = vhealth.classify_box_health(_box("loading", age=100), now=NOW)
    assert h.evidence["phase"] == "loading"
    assert "unbilled" in h.reason        # the pull phase costs schedule, not GPU $


def test_phase_evidence_loading_stall_names_unbilled():
    h = vhealth.classify_box_health(_box("loading", age=LOAD_DL + 1), now=NOW)
    assert h.evidence["phase"] == "loading"
    assert "unbilled" in h.reason.lower()
    # ...and prescribes the RECOVERABLE remedy, not a destroy (2026-08-03).
    assert "park" in h.reason.lower() and "destroy" not in h.reason.split(
        "not destroy")[0].lower()


def test_phase_evidence_up_when_jobd_stamped():
    h = vhealth.classify_box_health(_box("running", age=99999, cred_jobs=True),
                                    jobd_hb_epoch=NOW - 30, now=NOW)
    assert h.evidence["phase"] == "up"
    # stale-heartbeat boxes are also past env-setup (jobd existed):
    h2 = vhealth.classify_box_health(_box("running", age=99999, cred_jobs=True),
                                     jobd_hb_epoch=NOW - (STALE + 60), now=NOW)
    assert h2.evidence["phase"] == "up"


def test_phase_unknowable_for_non_jobs_running_box():
    # no workload contract the API can check -> phase None, never a guess
    h = vhealth.classify_box_health(_box("running", age=99999), now=NOW)
    assert h.evidence["phase"] is None


def test_envsetup_deadline_is_tighter_than_loading_deadline():
    """The 2026-08-02 phase split: env-setup bills FULL GPU (invoice-verified —
    billing starts at the loading→running flip), so a never-stamped running box
    is condemned at GUARD_ENVSETUP_DEADLINE_S (900s), not the loading deadline
    (1500s) the old single-phase code reused."""
    age = (ENV_DL + LOAD_DL) // 2                 # 1200s: between the two
    h = vhealth.classify_box_health(_box("running", age=age, cred_jobs=True),
                                    jobd_hb_epoch=None, now=NOW)
    assert h.verdict == vhealth.GUARD_ZOMBIE_NO_JOBD
    assert h.evidence["phase"] == "env-setup"
    assert "env-setup" in h.reason


def test_envsetup_deadline_boundary_still_booting():
    h = vhealth.classify_box_health(_box("running", age=ENV_DL, cred_jobs=True),
                                    jobd_hb_epoch=None, now=NOW)
    assert h.verdict == vhealth.GUARD_BOOTING


def test_env_knob_overrides_envsetup_deadline(monkeypatch):
    monkeypatch.setenv("GUARD_ENVSETUP_DEADLINE_S", "60")
    h = vhealth.classify_box_health(_box("running", age=100, cred_jobs=True),
                                    jobd_hb_epoch=None, now=NOW)
    assert h.verdict == vhealth.GUARD_ZOMBIE_NO_JOBD


def test_guard_evidence_bits_carry_phase_and_cost():
    assert "phase env-setup (BILLED)" in vreap._guard_evidence_bits(
        {"phase": "env-setup", "boot_age_s": 100})
    assert "phase loading (GPU unbilled)" in vreap._guard_evidence_bits(
        {"phase": "loading", "boot_age_s": 100})
    assert "phase" not in vreap._guard_evidence_bits({"boot_age_s": 100})


# --- knob override ----------------------------------------------------------- #
def test_env_knob_overrides_loading_deadline(monkeypatch):
    monkeypatch.setenv("GUARD_LOADING_DEADLINE_S", "60")
    # 100s loading would be BOOTING at the 1500 default, but STALL at 60
    h = vhealth.classify_box_health(_box("loading", age=100), now=NOW)
    assert h.verdict == vhealth.GUARD_ZOMBIE_LOADING_STALL


# --- gather_fleet_health: bounded B2 reads ----------------------------------- #
def test_gather_skips_b2_read_when_fold_heartbeat_fresh(monkeypatch):
    calls = {"n": 0}

    def fake_read(iid):
        calls["n"] += 1
        return None

    monkeypatch.setattr(vhealth, "_jobd_heartbeat_epoch_soft", fake_read)
    ins = [_box("running", age=99999, cred_jobs=True, iid=1)]
    jobs = {"1": [{"display_status": "running",
                   "last_heartbeat_ts": _jm_ts(NOW - 30)}]}
    out = vhealth.gather_fleet_health(ins, jobs, now=NOW)
    assert out["1"]["verdict"] == vhealth.GUARD_OK
    assert calls["n"] == 0                       # fresh fold -> no B2 read


def test_gather_reads_b2_for_idle_jobs_box(monkeypatch):
    calls = {"n": 0}

    def fake_read(iid):
        calls["n"] += 1
        return NOW - 30                          # fresh JOBD_STATUS

    monkeypatch.setattr(vhealth, "_jobd_heartbeat_epoch_soft", fake_read)
    ins = [_box("running", age=99999, cred_jobs=True, iid=2)]
    out = vhealth.gather_fleet_health(ins, {}, now=NOW)   # idle: no jobs folded
    assert out["2"]["verdict"] == vhealth.GUARD_OK
    assert calls["n"] == 1                       # idle box -> one B2 read


def test_gather_no_reads_when_no_live_jobs_boxes(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(vhealth, "_jobd_heartbeat_epoch_soft",
                        lambda iid: calls.__setitem__("n", calls["n"] + 1))
    ins = [_box("stopped", age=99999, iid=3),
           _box("running", age=99999, iid=4)]       # non-jobs running box
    out = vhealth.gather_fleet_health(ins, {}, now=NOW)
    assert calls["n"] == 0                        # latency ~0 on a healthy fleet
    assert all(out[k]["verdict"] == vhealth.GUARD_OK for k in out)


# --- _render_ls loud scream -------------------------------------------------- #
def _pal():
    return vfmt._Pal(False)                          # color-off: plain identity


def test_render_ls_screams_for_zombie():
    ins = [_box("loading", age=4 * 3600, iid=55)]
    health = vhealth.gather_fleet_health(ins, {}, now=NOW)
    data = {"instances": ins, "live_ids": [55], "health": health}
    out = "\n".join(vlsrender._render_ls(data, _pal()))
    assert "ZOMBIE" in out and "55" in out and "loading-stall" in out


def test_render_ls_silent_when_healthy():
    ins = [_box("running", age=99999, iid=66)]     # non-jobs box -> OK
    health = vhealth.gather_fleet_health(ins, {}, now=NOW)
    data = {"instances": ins, "live_ids": [66], "health": health}
    out = "\n".join(vlsrender._render_ls(data, _pal()))
    assert "ZOMBIE" not in out
    assert "booting[" not in out                   # no phase noise when healthy


def test_render_ls_names_the_boot_phase(monkeypatch):
    """A returning agent must see WHICH phase a slow box is in — the correct
    response differs (loading = unbilled, patient; env-setup = billed)."""
    monkeypatch.setattr(vhealth, "_jobd_heartbeat_epoch_soft", lambda iid: None)
    ins = [_box("loading", age=300, iid=71),
           _box("running", age=300, cred_jobs=True, iid=72)]
    health = vhealth.gather_fleet_health(ins, {}, now=NOW)
    data = {"instances": ins, "live_ids": [71, 72], "health": health}
    out = "\n".join(vlsrender._render_ls(data, _pal()))
    assert "71  booting[loading]" in out and "GPU unbilled" in out
    assert "72  booting[env-setup]" in out and "BILLED full GPU" in out


def test_render_minimal_carries_the_phase_column(monkeypatch):
    monkeypatch.setattr(vhealth, "_jobd_heartbeat_epoch_soft", lambda iid: None)
    ins = [_box("loading", age=300, iid=81),
           _box("running", age=300, cred_jobs=True, iid=82),
           _box("running", age=300, iid=83)]       # non-jobs: unknowable
    health = vhealth.gather_fleet_health(ins, {}, now=NOW)
    data = {"instances": ins, "live_ids": [81, 82, 83], "health": health}
    lines = vlsrender._render_minimal(data).splitlines()
    header = lines[0].split("\t")
    # BY NAME, not by position. This test is about what `phase` CONTAINS; it
    # read `header[-1]` only because phase happened to be last, so appending a
    # column (`cpu_util`, 2026-08-21) failed it for no reason of its own. The
    # column ORDER has its own pin — test_vastlib_cli_helpers.py::
    # test_minimal_tsv_column_order_is_frozen — which is where an append is
    # supposed to be caught, and was.
    assert "phase" in header
    got = {r.split("\t")[1]: dict(zip(header, r.split("\t")))["phase"]
           for r in lines[1:]}
    assert got == {"81": "loading", "82": "env-setup", "83": ""}


# --- the 46682313 / 46682177 false positive (2026-08-03) --------------------- #
# Both boxes pulled train-t211-latest that morning; three boxes on three hosts
# sat 27-40 min in `loading`, so "pull stall = change HOST" did not apply. Serve
# box 46682177 was flagged ZOMBIE_LOADING_STALL at 27m and cleared to OK at 40m
# (fleetd journal). Jobs box 46682313, identical shape, was destroyed at 38m —
# 90 s earlier. A wall-clock age cannot tell slow from dead; the pull output can.
_PULLING = ("cafe00: Downloading [=======>      ]  1.25 GB / 4.10 GB\n"
            "beef11: Download complete\n")
_PULL_DONE = "cafe00: Pull complete\nbeef11: Pull complete\n"


def test_slow_but_advancing_pull_is_advisory_not_a_zombie():
    """THE REGRESSION TEST. Past the 1500s deadline, still visibly pulling,
    inside the hard bound -> LOADING_SLOW: an ADVISORY verdict, so it alarms
    but licenses nothing. At 27m and 33m — the exact ages at which 46682177 was
    (falsely) condemned — it must not be a zombie."""
    for age in (27 * 60, 33 * 60, 40 * 60):
        b = _box("loading", age=age)
        b["status_msg"] = _PULLING
        h = vhealth.classify_box_health(b, now=NOW)
        assert h.verdict == vhealth.GUARD_LOADING_SLOW, age
        assert h.verdict not in vhealth._ZOMBIE_VERDICTS
        assert h.verdict in vhealth._ADVISORY_VERDICTS
        assert h.evidence["pull_active"] is True
        assert h.evidence["phase"] == "loading"


def test_inert_pull_past_the_deadline_is_still_a_zombie():
    """The lever is EVIDENCE, not patience: no pull activity past the deadline
    still classifies as a stall (it just can no longer license a destroy)."""
    b = _box("loading", age=27 * 60)
    b["status_msg"] = _PULL_DONE            # nothing downloading/extracting
    h = vhealth.classify_box_health(b, now=NOW)
    assert h.verdict == vhealth.GUARD_ZOMBIE_LOADING_STALL
    assert h.evidence["pull_active"] is False


def test_loading_hard_bound_catches_a_frozen_status_msg():
    """A progress-based bound still needs a stop, or a status_msg frozen
    mid-`Downloading` holds a dead box forever (the ~10h 45373337 shape). Past
    GUARD_LOADING_HARD_S it is a zombie even while 'pulling'."""
    b = _box("loading", age=2 * 3600)
    b["status_msg"] = _PULLING
    h = vhealth.classify_box_health(b, now=NOW)
    assert h.verdict == vhealth.GUARD_ZOMBIE_LOADING_STALL
    assert "hard bound" in h.reason


def test_no_status_msg_still_classifies_as_before():
    """Degrade safely when the API gives us nothing to read: the verdict is
    unchanged from the pre-2026-08-03 behavior (a stall), and only the ACTION
    softened."""
    h = vhealth.classify_box_health(_box("loading", age=27 * 60), now=NOW)
    assert h.verdict == vhealth.GUARD_ZOMBIE_LOADING_STALL
    assert h.evidence["pull_active"] is False


def test_slow_pull_verdict_is_identical_for_jobs_and_nonjobs():
    """The lane must not change the verdict — that asymmetry is what gave two
    co-resident boxes on one slow pull opposite fates."""
    verdicts = set()
    for jobs in (True, False):
        b = _box("loading", age=30 * 60, cred_jobs=jobs)
        b["status_msg"] = _PULLING
        verdicts.add(vhealth.classify_box_health(b, now=NOW).verdict)
    assert verdicts == {vhealth.GUARD_LOADING_SLOW}


# --- guard subcommand: gating + exit codes ----------------------------------- #
def _wire_guard(monkeypatch, ins, jobs=None, jobd_epoch=None,
                ever_stamped=None, parked=None):
    monkeypatch.setattr(vlifecycle, "_instances", lambda: ins)
    monkeypatch.setattr(vlifecycle, "_instances_soft", lambda: ins)
    monkeypatch.setattr(vjobsview, "_fold_fleet_jobs", lambda live: jobs or {})
    monkeypatch.setattr(vhealth, "_jobd_heartbeat_epoch_soft", lambda iid: jobd_epoch)
    monkeypatch.setattr(vhealth, "_jobd_ever_stamped", lambda iid: ever_stamped)
    destroyed = []
    monkeypatch.setattr(vlifecycle, "destroy_box",
                        lambda iid: (destroyed.append(iid), (True, None))[1])
    monkeypatch.setattr(vlifecycle, "stop_box",
                        lambda iid: ((parked if parked is not None
                                      else []).append(iid), (True, None))[1])
    monkeypatch.setattr(vlifecycle, "_emit_stopping_intent",
                        lambda *a, **k: None)
    # two resolution sites for ONE seam: the destroy lane reaches it through
    # `lifecycle._destroy_and_revoke`'s own globals, the park lane through
    # `cli.guard`'s `fleet_client.` module attribute.
    monkeypatch.setattr(vlifecycle, "fleet_operator_intent", lambda *a, **k: None)
    monkeypatch.setattr(vfleetclient, "fleet_operator_intent", lambda *a, **k: None)
    monkeypatch.setattr(vlifecycle, "_revoke_box_keys", lambda names: None)
    monkeypatch.setenv("NO_COLOR", "1")
    return destroyed


def _guard_args(fix=False, yes=False, json=False, force=False):
    return argparse.Namespace(fix=fix, yes=yes, json=json, force=force)


def test_guard_clean_fleet_exits_zero(monkeypatch, capsys):
    _wire_guard(monkeypatch, [_box("running", age=99999, iid=1)])
    vguard.run(argparse.Namespace(fix=False, yes=False, json=False))
    out = capsys.readouterr().out
    assert "no zombies" in out


def test_guard_detects_and_exits_nonzero(monkeypatch, capsys):
    destroyed = _wire_guard(monkeypatch, [_box("loading", age=4 * 3600, iid=9)])
    with pytest.raises(SystemExit) as e:
        vguard.run(argparse.Namespace(fix=False, yes=False, json=False))
    assert e.value.code == 2
    assert destroyed == []                        # no --fix -> nothing destroyed
    assert "ZOMBIE_LOADING_STALL" in capsys.readouterr().out


def test_guard_fix_without_yes_is_dry_run(monkeypatch, capsys):
    destroyed = _wire_guard(monkeypatch, [_box("loading", age=4 * 3600, iid=9)])
    with pytest.raises(SystemExit) as e:
        vguard.run(argparse.Namespace(fix=True, yes=False, json=False))
    assert e.value.code == 2
    assert destroyed == []                        # preview only
    assert "WOULD DESTROY" in capsys.readouterr().out


def test_guard_fix_yes_parks_a_loading_zombie_and_spares_the_healthy(
        monkeypatch, capsys):
    """`--fix -y` acts on ONLY the zombie — and, since 2026-08-03, the action
    for a GPU-unbilled loading stall is a recoverable PARK, not a destroy."""
    ins = [_box("loading", age=4 * 3600, iid=9),          # zombie
           _box("running", age=99999, iid=1)]             # healthy non-jobs box
    parked = []
    destroyed = _wire_guard(monkeypatch, ins, parked=parked)
    vguard.run(_guard_args(fix=True, yes=True))
    assert parked == [9] and destroyed == []
    assert "parked zombie 9" in capsys.readouterr().out


def test_guard_fix_yes_destroys_the_billing_workless_shape(monkeypatch, capsys):
    """The expensive shape is untouched by the amendment: a RUNNING jobs box
    burning full GPU whose jobd never stamped JOBD_STATUS still destroys."""
    ins = [_box("running", age=4 * 3600, cred_jobs=True, iid=11),
           _box("running", age=99999, iid=1)]
    parked = []
    destroyed = _wire_guard(monkeypatch, ins, jobd_epoch=None,
                            ever_stamped=False, parked=parked)
    vguard.run(_guard_args(fix=True, yes=True))
    assert destroyed == [11] and parked == []
    assert "destroyed zombie 11" in capsys.readouterr().out


def test_guard_force_restores_the_blunt_destroy(monkeypatch, capsys):
    """`--force` is the human escape hatch — and is the ONLY way to reach the
    behavior that destroyed 46682313 mid-pull."""
    parked = []
    destroyed = _wire_guard(monkeypatch, [_box("loading", age=4 * 3600, iid=9)],
                            parked=parked)
    vguard.run(_guard_args(fix=True, yes=True, force=True))
    assert destroyed == [9] and parked == []


def test_guard_never_touches_running_job_box(monkeypatch, capsys):
    # a live jobs box actively running a job (fresh fold heartbeat) is OK and
    # must never be swept (the M2-box safety invariant). cmd_guard uses the real
    # clock, so the fold heartbeat is stamped relative to time.time().
    import time as _t
    ins = [_box("running", age=99999, cred_jobs=True, iid=77)]
    jobs = {"77": [{"display_status": "running",
                    "last_heartbeat_ts": _jm_ts(_t.time() - 20)}]}
    destroyed = _wire_guard(monkeypatch, ins, jobs=jobs)
    vguard.run(argparse.Namespace(fix=True, yes=True, json=False))
    assert destroyed == []
    assert "no zombies" in capsys.readouterr().out


def test_guard_json(monkeypatch, capsys):
    _wire_guard(monkeypatch, [_box("loading", age=4 * 3600, iid=9)])
    with pytest.raises(SystemExit) as e:
        vguard.run(argparse.Namespace(fix=False, yes=False, json=True))
    assert e.value.code == 2
    import json as _json
    doc = _json.loads(capsys.readouterr().out)
    assert doc["zombies"] == [9]


# --- STALE_IMAGE: the ADVISORY verdict (velvet plan P1) ---------------------- #
# The incident: three frontier-wave jobs died on box 46240842 whose baked
# eval-env predated a module they imported. The signal existed; nothing that
# schedules work consulted it. P1 surfaces it WITHOUT letting it destroy
# anything — a stale box is healthy, just running old code.
REG = "registry.example.com"
OURS = f"{REG}/train:t215-latest"
DIG_OLD = "sha256:" + "a" * 64
DIG_NEW = "sha256:" + "b" * 64


def _stamped_box(status="running", *, age=99999, iid=5, digest=DIG_OLD,
                 image=OURS, cred_jobs=False):
    b = _box(status, age=age, iid=iid, cred_jobs=cred_jobs)
    env = [["HERDD_IMAGE_DIGEST", digest]]
    if cred_jobs:
        env.append(["CRED_ROLE", "jobs"])
    b["extra_env"] = env
    b["image"] = image
    return b


def _wire_digest(monkeypatch, current):
    """Steer the TTL resolver, which is what gather_fleet_health calls."""
    import imageref
    imageref.clear_ttl_cache()
    monkeypatch.setattr(imageref, "image_tag_digest", lambda img: current)


def test_stale_image_becomes_an_advisory_verdict_on_an_otherwise_OK_box(monkeypatch):
    _wire_digest(monkeypatch, DIG_NEW)
    h = vhealth.gather_fleet_health([_stamped_box()], {})["5"]
    assert h["verdict"] == vhealth.GUARD_STALE_IMAGE
    assert h["evidence"]["image_state"] == "stale"


def test_fresh_and_unresolved_leave_the_verdict_OK(monkeypatch):
    _wire_digest(monkeypatch, DIG_OLD)                    # matches the stamp
    assert vhealth.gather_fleet_health([_stamped_box()], {})["5"]["verdict"] == vhealth.GUARD_OK
    _wire_digest(monkeypatch, None)                       # resolution failed
    h = vhealth.gather_fleet_health([_stamped_box()], {})["5"]
    assert h["verdict"] == vhealth.GUARD_OK, "unresolved must not alarm in P1"
    assert h["evidence"]["image_state"] == "unresolved"


def test_a_zombie_verdict_is_NEVER_masked_by_staleness(monkeypatch):
    """The advisory overlays OK only. A dead box that is ALSO stale must still
    read as the zombie — that verdict is destroy-relevant and drives --fix."""
    _wire_digest(monkeypatch, DIG_NEW)
    box = _stamped_box("loading", age=4 * 3600, iid=9)
    h = vhealth.gather_fleet_health([box], {})["9"]
    assert h["verdict"] == vhealth.GUARD_ZOMBIE_LOADING_STALL
    # ...and the staleness is still legible in evidence, not lost
    assert h["evidence"]["image_state"] == "stale"


def test_guard_fix_NEVER_destroys_a_stale_image_box(monkeypatch, capsys):
    """THE safety invariant of P1. STALE_IMAGE is deliberately absent from
    _GUARD_ZOMBIE_VERDICTS: destroying a healthy box over a fixable condition
    would throw away a warm disk. Adding it to that set must fail here."""
    _wire_digest(monkeypatch, DIG_NEW)
    destroyed = _wire_guard(monkeypatch, [_stamped_box(iid=5)])
    vguard.run(argparse.Namespace(fix=True, yes=True, json=False))
    assert destroyed == []
    out = capsys.readouterr().out
    assert "no zombies" in out and "STALE_IMAGE" in out


def test_guard_exits_zero_on_advisory_only_and_reports_it(monkeypatch, capsys):
    """Advisory must not turn into a nonzero exit — schedulers and CI read it."""
    _wire_digest(monkeypatch, DIG_NEW)
    _wire_guard(monkeypatch, [_stamped_box(iid=5)])
    vguard.run(argparse.Namespace(fix=False, yes=False, json=False))
    out = capsys.readouterr().out
    assert "1 advisory" in out and "does NOT touch these" in out


def test_guard_json_separates_advisory_from_zombies(monkeypatch, capsys):
    """Advisory rows are reported but exit ZERO — the exit code stays the
    zombie signal, so a scheduler polling `guard --json` is not blocked by a
    box that merely needs a refresh."""
    _wire_digest(monkeypatch, DIG_NEW)
    _wire_guard(monkeypatch, [_stamped_box(iid=5)])
    with pytest.raises(SystemExit) as e:
        vguard.run(argparse.Namespace(fix=False, yes=False, json=True))
    assert e.value.code == 0
    import json as _json
    doc = _json.loads(capsys.readouterr().out)
    assert doc["zombies"] == [] and doc["advisory"] == [5]


def test_foreign_registry_box_costs_no_lookup_and_never_alarms(monkeypatch):
    """The vllm/vllm-openai serve lane. A two-valued gate would refuse it."""
    import imageref
    imageref.clear_ttl_cache()
    calls = []
    monkeypatch.setattr(imageref, "image_tag_digest",
                        lambda img: calls.append(img) or DIG_NEW)
    box = _stamped_box(iid=5, image="vllm/vllm-openai:latest")
    h = vhealth.gather_fleet_health([box], {})["5"]
    assert h["verdict"] == vhealth.GUARD_OK
    assert h["evidence"]["image_state"] == "not_applicable"
    assert calls == [], "a foreign-registry box must not cost a registry call"


def test_unstamped_box_costs_no_lookup(monkeypatch):
    import imageref
    imageref.clear_ttl_cache()
    calls = []
    monkeypatch.setattr(imageref, "image_tag_digest",
                        lambda img: calls.append(img) or DIG_NEW)
    box = _box("running", age=99999, iid=5)
    box["image"] = OURS                       # our registry, but no stamp
    h = vhealth.gather_fleet_health([box], {})["5"]
    assert h["verdict"] == vhealth.GUARD_OK
    assert calls == []


# --- reap: the automatic zombie lane (zombie 46633685) ----------------------- #
# The 2026-08-02 gap: 46633685 (serve lane) sat 31 min dead in `loading` on an
# on-demand box; the reaper SAW it, classified ZOMBIE_LOADING_STALL, and
# declined ("not a jobs box — no never-ran proof, alarm only") while manual
# `guard --fix` was the only destroyer of running-but-dead shapes. The lane
# below acts automatically — graded destroy/park via parked_lifecycle.
# zombie_action — but ONLY after the zombie ledger confirms no progress.
import time as _time  # noqa: E402


def _wire_reap(monkeypatch, tmp_path, ins, *, jobd_epoch=None,
               ever_stamped=None, ledger=None):
    """Wire cmd_reap hermetically: fake fleet, fake B2, tmp ledgers. Returns
    (destroyed, parked, ledger_path)."""
    monkeypatch.setattr(vlifecycle, "_instances", lambda: ins)
    monkeypatch.setattr(vlifecycle, "_instances_soft", lambda: ins)
    monkeypatch.setattr(vreap, "_fold_fleet_jobs", lambda live: {})
    monkeypatch.setattr(vhealth, "_jobd_heartbeat_epoch_soft", lambda iid: jobd_epoch)
    monkeypatch.setattr(vhealth, "_jobd_ever_stamped", lambda iid: ever_stamped)
    destroyed, parked = [], []
    monkeypatch.setattr(vlifecycle, "destroy_box",
                        lambda iid: (destroyed.append(iid), (True, None))[1])
    monkeypatch.setattr(vlifecycle, "stop_box",
                        lambda iid: (parked.append(iid), (True, None))[1])
    monkeypatch.setattr(vlifecycle, "_emit_stopping_intent", lambda *a, **k: None)
    monkeypatch.setattr(vlifecycle, "fleet_operator_intent", lambda *a, **k: None)
    monkeypatch.setattr(vlifecycle, "_revoke_box_keys", lambda names: None)
    monkeypatch.setattr(vreap, "_IDLE_LEDGER", str(tmp_path / "idle.json"))
    zl = tmp_path / "zombie.json"
    monkeypatch.setattr(vreap, "_ZOMBIE_LEDGER", str(zl))
    if ledger is not None:
        zl.write_text(__import__("json").dumps(ledger))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("HERDD_REAP_DURABILITY", "0")
    return destroyed, parked, zl


def _reap_args(yes=True):
    return argparse.Namespace(idle_hours=None, yes=yes, json=False)


def _seed(iid, verdict, *, age_s=2000, pull_bytes=0, hb=None, inet=None,
          disk=None):
    """A zombie-ledger entry whose first sighting is `age_s` in the past —
    past the 900 s REAP_ZOMBIE_CONFIRM_S default when age_s >= 900."""
    return {str(iid): {"first": _time.time() - age_s, "verdict": verdict,
                       "pull": {"layers": {"cafe00": pull_bytes},
                                "downloading": False, "extracting": False,
                                "total_bytes": pull_bytes},
                       "hb": hb, "inet": inet, "disk": disk}}


def _serve_stall_box(iid=9):
    b = _box("loading", age=4 * 3600, iid=iid)
    b["label"] = "serve:serve-probe"
    return b


def test_reap_zombie_first_sighting_alarms_and_seeds_the_ledger(
        monkeypatch, tmp_path, capsys):
    destroyed, parked, zl = _wire_reap(monkeypatch, tmp_path,
                                       [_serve_stall_box()])
    vreap.cmd_reap(_reap_args(yes=True))
    assert destroyed == [] and parked == []
    out = capsys.readouterr().out
    assert "alarm" in out and "unconfirmed" in out
    import json as _json
    led = _json.loads(zl.read_text())
    assert "9" in led and led["9"]["verdict"] == vhealth.GUARD_ZOMBIE_LOADING_STALL


def test_reap_zombie_confirmed_nonjobs_stall_is_parked_not_destroyed(
        monkeypatch, tmp_path, capsys):
    """The 46633685 shape, one confirmed pass later: PARK (never destroy — no
    workless proof exists for a serve box), landing it in the 2h idle lane."""
    destroyed, parked, _ = _wire_reap(
        monkeypatch, tmp_path, [_serve_stall_box()],
        ledger=_seed(9, vhealth.GUARD_ZOMBIE_LOADING_STALL))
    vreap.cmd_reap(_reap_args(yes=True))
    assert parked == [9] and destroyed == []
    assert "parked zombie 9" in capsys.readouterr().out


def test_reap_zombie_pull_progress_resets_the_clock(
        monkeypatch, tmp_path, capsys):
    """A slow-but-moving image pull must never be condemned: advancing
    status_msg bytes reset the confirmation window even when the ledger says
    the verdict is old."""
    b = _serve_stall_box()
    b["status_msg"] = ("cafe00: Downloading [=====>    ]  "
                       "150.0 MB / 900.0 MB")
    destroyed, parked, _ = _wire_reap(
        monkeypatch, tmp_path, [b],
        ledger=_seed(9, vhealth.GUARD_ZOMBIE_LOADING_STALL, pull_bytes=50_000_000))
    vreap.cmd_reap(_reap_args(yes=True))
    assert parked == [] and destroyed == []
    assert "pull bytes advancing" in capsys.readouterr().out


def test_reap_zombie_running_never_stamped_jobs_box_is_destroyed(
        monkeypatch, tmp_path, capsys):
    """The EXPENSIVE shape with the workless proof: running, billing full GPU,
    jobd never stamped JOBD_STATUS (readable absence) -> destroy, through
    _destroy_and_revoke so the ephemeral B2 key dies with the box."""
    b = _box("running", age=4 * 3600, cred_jobs=True, iid=11)
    destroyed, parked, _ = _wire_reap(
        monkeypatch, tmp_path, [b], jobd_epoch=None, ever_stamped=False,
        ledger=_seed(11, vhealth.GUARD_ZOMBIE_NO_JOBD))
    vreap.cmd_reap(_reap_args(yes=True))
    assert destroyed == [11] and parked == []
    assert "destroyed zombie 11" in capsys.readouterr().out


def test_reap_zombie_loading_jobs_box_is_PARKED_not_destroyed(
        monkeypatch, tmp_path, capsys):
    """THE 46682313 SHAPE, at the automatic lane. A jobs box stalled in
    `loading` with the "never stamped" marker absent used to satisfy the
    provably-workless destroy — but jobd CANNOT stamp before the container
    exists, so that proof is vacuous during `loading`, and the phase is
    GPU-unbilled anyway. Now: PARK (recoverable), the same action its
    co-resident serve box would have got."""
    b = _box("loading", age=4 * 3600, cred_jobs=True, iid=14)
    destroyed, parked, _ = _wire_reap(
        monkeypatch, tmp_path, [b], jobd_epoch=None, ever_stamped=False,
        ledger=_seed(14, vhealth.GUARD_ZOMBIE_LOADING_STALL))
    vreap.cmd_reap(_reap_args(yes=True))
    assert parked == [14] and destroyed == []
    assert "parked zombie 14" in capsys.readouterr().out


def test_reap_zombie_stale_heartbeat_with_history_is_parked(
        monkeypatch, tmp_path):
    """jobd existed and died (heartbeat READ, stale; marker present): the disk
    has history, so the automatic action degrades to park."""
    hb = _time.time() - 3600
    b = _box("running", age=4 * 3600, cred_jobs=True, iid=12)
    destroyed, parked, _ = _wire_reap(
        monkeypatch, tmp_path, [b], jobd_epoch=hb, ever_stamped=True,
        ledger=_seed(12, vhealth.GUARD_ZOMBIE_NO_JOBD, hb=hb))
    vreap.cmd_reap(_reap_args(yes=True))
    assert parked == [12] and destroyed == []


def test_reap_zombie_unreadable_evidence_stays_alarm_only(
        monkeypatch, tmp_path):
    """I3: an unreadable JOBD_STATUS (e.g. a LOCAL B2 outage that makes every
    jobs box look dead) must degrade to alarms, never to fleet-wide action."""
    b = _box("running", age=4 * 3600, cred_jobs=True, iid=13)
    destroyed, parked, _ = _wire_reap(
        monkeypatch, tmp_path, [b], jobd_epoch=None, ever_stamped=None,
        ledger=_seed(13, vhealth.GUARD_ZOMBIE_NO_JOBD))
    vreap.cmd_reap(_reap_args(yes=True))
    assert destroyed == [] and parked == []


def test_reap_zombie_keep_label_is_never_touched(monkeypatch, tmp_path):
    b = _serve_stall_box()
    b["label"] = "serve:probe keep:debugging-this-boot"
    destroyed, parked, _ = _wire_reap(
        monkeypatch, tmp_path, [b],
        ledger=_seed(9, vhealth.GUARD_ZOMBIE_LOADING_STALL))
    vreap.cmd_reap(_reap_args(yes=True))
    assert destroyed == [] and parked == []


def test_reap_zombie_preview_exits_2_and_touches_nothing(
        monkeypatch, tmp_path, capsys):
    destroyed, parked, _ = _wire_reap(
        monkeypatch, tmp_path, [_serve_stall_box()],
        ledger=_seed(9, vhealth.GUARD_ZOMBIE_LOADING_STALL))
    with pytest.raises(SystemExit) as e:
        vreap.cmd_reap(_reap_args(yes=False))
    assert e.value.code == 2
    assert destroyed == [] and parked == []
    assert "PARK 1" in capsys.readouterr().out


def test_reap_zombie_lane_disabled_by_env(monkeypatch, tmp_path, capsys):
    for knob in ("HERDD_REAP_ZOMBIE", "HERDD_REAP_STALL"):  # legacy alias
        destroyed, parked, _ = _wire_reap(
            monkeypatch, tmp_path, [_serve_stall_box()],
            ledger=_seed(9, vhealth.GUARD_ZOMBIE_LOADING_STALL))
        monkeypatch.setenv(knob, "0")
        vreap.cmd_reap(_reap_args(yes=True))
        monkeypatch.delenv(knob)
        assert destroyed == [] and parked == []
        assert "zombie" not in capsys.readouterr().out.lower()


# --- env-setup liveness signals in the confirm ledger (2026-08-02) ----------- #
# The interaction bug this closes: a RUNNING jobs box mid-provision (onstart
# pulling weights/env, jobd not yet started) has flat pull bytes and no jobd
# heartbeat BY DEFINITION, so under the pull/heartbeat-only confirm rule a
# perfectly healthy env-setup confirmed as a stall — and, carrying the
# never-ran proof, was DESTROYED mid-install. The box's own download counter
# (`inet_down_billed`) and `disk_usage` are the API-visible env-setup progress.
def _envsetup_box(iid=21, *, inet=None, disk=None, age=4 * 3600):
    b = _box("running", age=age, cred_jobs=True, iid=iid)
    if inet is not None:
        b["inet_down_billed"] = inet
    if disk is not None:
        b["disk_usage"] = disk
    return b


def test_reap_envsetup_download_progress_resets_the_clock(
        monkeypatch, tmp_path, capsys):
    """Regression for the healthy-install-looks-like-a-stall bug: an aged
    NO_JOBD verdict whose box moved >50 MB of download traffic since the last
    pass must NOT be acted on — pull bytes and heartbeat are flat by
    definition during env-setup, so this is the signal that saves it."""
    b = _envsetup_box(21, inet=1_200_000.0)          # KB, cumulative
    destroyed, parked, _ = _wire_reap(
        monkeypatch, tmp_path, [b], jobd_epoch=None, ever_stamped=False,
        ledger=_seed(21, vhealth.GUARD_ZOMBIE_NO_JOBD, inet=1_000_000.0))
    vreap.cmd_reap(_reap_args(yes=True))
    assert destroyed == [] and parked == []
    assert "download traffic advancing" in capsys.readouterr().out


def test_reap_envsetup_disk_progress_resets_the_clock(
        monkeypatch, tmp_path, capsys):
    b = _envsetup_box(22, disk=21.0)                 # GB
    destroyed, parked, _ = _wire_reap(
        monkeypatch, tmp_path, [b], jobd_epoch=None, ever_stamped=False,
        ledger=_seed(22, vhealth.GUARD_ZOMBIE_NO_JOBD, disk=17.0))
    vreap.cmd_reap(_reap_args(yes=True))
    assert destroyed == [] and parked == []
    assert "disk usage advancing" in capsys.readouterr().out


def test_reap_envsetup_flat_signals_still_confirm_and_destroy(
        monkeypatch, tmp_path):
    """The widened signal set must not immortalize a truly dead box: flat
    download counter + flat disk + flat pull + no heartbeat past the confirm
    window keeps the workless-proof destroy."""
    b = _envsetup_box(23, inet=1_000_000.0, disk=17.0)
    destroyed, parked, _ = _wire_reap(
        monkeypatch, tmp_path, [b], jobd_epoch=None, ever_stamped=False,
        ledger=_seed(23, vhealth.GUARD_ZOMBIE_NO_JOBD, inet=1_000_000.0, disk=17.0))
    vreap.cmd_reap(_reap_args(yes=True))
    assert destroyed == [23] and parked == []


def test_reap_envsetup_subthreshold_noise_does_not_reset(
        monkeypatch, tmp_path):
    """Idle-noise traffic (jobd-less heartbeat chatter, DNS, retries) below the
    50 MB / 0.5 GB epsilons must not keep a dead box alive forever."""
    b = _envsetup_box(24, inet=1_020_000.0, disk=17.2)     # +20 MB, +0.2 GB
    destroyed, parked, _ = _wire_reap(
        monkeypatch, tmp_path, [b], jobd_epoch=None, ever_stamped=False,
        ledger=_seed(24, vhealth.GUARD_ZOMBIE_NO_JOBD, inet=1_000_000.0, disk=17.0))
    vreap.cmd_reap(_reap_args(yes=True))
    assert destroyed == [24] and parked == []


def test_reap_envsetup_first_reading_never_counts_as_progress(
        monkeypatch, tmp_path):
    """A ledger with no prior inet/disk reading (pre-split entries, or a failed
    read last pass) must neither fake progress nor block confirmation — the
    comparison needs two READ values."""
    b = _envsetup_box(25, inet=9_999_999.0)
    destroyed, parked, _ = _wire_reap(
        monkeypatch, tmp_path, [b], jobd_epoch=None, ever_stamped=False,
        ledger=_seed(25, vhealth.GUARD_ZOMBIE_NO_JOBD))    # inet/disk absent
    vreap.cmd_reap(_reap_args(yes=True))
    assert destroyed == [25] and parked == []


def test_reap_ledger_persists_inet_and_disk_readings(
        monkeypatch, tmp_path):
    b = _envsetup_box(26, inet=2_000_000.0, disk=33.0)
    destroyed, parked, zl = _wire_reap(
        monkeypatch, tmp_path, [b], jobd_epoch=None, ever_stamped=None)
    vreap.cmd_reap(_reap_args(yes=True))
    import json as _json
    led = _json.loads(zl.read_text())
    assert led["26"]["inet"] == 2_000_000.0
    assert led["26"]["disk"] == 33.0


# --- the v9 false zombie (2026-08-07) ---------------------------------------- #
# Two ZOMBIE_NO_JOBD alarms that night named LIVE, healthy boxes: 47042386 at
# 04:14:31 and 47045282 at 05:02:56 ("jobd heartbeat 11m stale ... destroy +
# relaunch"), the latter while 20260806T212132-v9-gemma4-dec-train-8818 was at
# step 142 of 156 with seven attempts behind it and 389 W on the card. On an
# UNWATCHED box that verdict parks the machine (test_fleetd:
# test_a_measured_zombie_is_not_saved_by_its_label), so a false one is not just
# noise.
#
# Two independent faults compose into it, both exercised here against the REAL
# event log (tools/vast/testfixtures/jobmeta/v9-gemma4-requeue-chain.jsonl):
#
#   1. `_fold_fleet_jobs` froze the job's view on disk when it went `failed` on
#      47041615 at 03:51:33 — but `failed` is the ONE re-openable terminal
#      (`jobmeta.fold_events`' requeue un-stick). Two operator requeues later
#      the job was running on 47045282 and heartbeating every 60 s, while `ls`
#      and fleetd still read a view whose last_heartbeat_ts said 03:40:02.
#   2. With no fresh folded heartbeat, `_fleet_jobd_hb_epoch` fell back to the
#      JOBD_STATUS marker — which jobd.sh stamps only on transitions, so it sat
#      at the 04:51:46 job-spawn stamp until 05:19:16. A 27-minute gap on a
#      100%-busy box, against a 600 s deadline.
import json as _json  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobmeta as _jm  # noqa: E402

V9_JOB_ID = "20260806T212132-v9-gemma4-dec-train-8818"
V9_LIVE_BOX = "47045282"
V9_DEAD_BOX = "47041615"
V9_FAIL_TS = "20260807T035133081Z"          # the failure the cache froze
V9_JOBD_STATUS_TS = "20260807T045146000Z"   # last transition stamp before 05:19


def _v9_events():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "testfixtures", "jobmeta", "v9-gemma4-requeue-chain.jsonl")
    with open(p) as fh:
        return [_json.loads(l) for l in fh if l.strip()]


def _v9_now():
    """Two minutes after the newest event in the log — inside the 600 s
    deadline, so a correctly-folded view proves life on its own."""
    return max(vfmt._ts_to_epoch(e["ts"]) for e in _v9_events()) + 120.0


def _wire_v9_fold(monkeypatch, tmp_path, *, poison=True):
    """Reproduce the ls job fold as it stood that night: the ticket is queued on
    the LIVE box, B2 holds the full log, and the disk cache holds the view
    frozen at the 03:51:33 failure (mtime 03:53 on the real box)."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(vb2, "_ensure_b2_remote", lambda *a, **k: None)
    monkeypatch.setattr(_jm, "list_all_queued",
                        lambda *a, **k: [(V9_LIVE_BOX, V9_JOB_ID)])
    reads = {"n": 0}

    def fake_read_job(jid, live_iids=(), **kw):
        reads["n"] += 1
        return _jm.fold_events(_v9_events(), live_iids=live_iids)

    monkeypatch.setattr(_jm, "read_job", fake_read_job)
    if poison:
        d = tmp_path / "vast-jobmeta" / V9_JOB_ID
        d.mkdir(parents=True)
        frozen = _jm.fold_events([e for e in _v9_events()
                                  if e["ts"] <= V9_FAIL_TS],
                                 live_iids={V9_DEAD_BOX})
        assert frozen["status"] == "failed"        # correct when it was written
        (d / "view.json").write_text(_json.dumps(frozen))
    return reads


def test_v9_ls_fold_refuses_to_trust_a_frozen_failed_view(monkeypatch, tmp_path):
    """A `failed` fold is re-openable, so it may never be cached as final. The
    box must see the LIVE job, not the dead attempt."""
    reads = _wire_v9_fold(monkeypatch, tmp_path)
    by_box = vjobsview._fold_fleet_jobs({V9_LIVE_BOX})
    assert reads["n"] == 1, "a re-openable `failed` must be re-read from B2"
    jobs = by_box[V9_LIVE_BOX]
    assert [j["instance_id"] for j in jobs] == [V9_LIVE_BOX]
    assert jobs[0]["last_heartbeat_ts"] > V9_FAIL_TS
    assert jobs[0]["display_status"] == "running"


def test_v9_ls_fold_still_caches_a_done_view(monkeypatch, tmp_path):
    """Inertness: the cost control survives for the terminals that ARE sticky."""
    _wire_v9_fold(monkeypatch, tmp_path, poison=False)
    monkeypatch.setattr(_jm, "read_job",
                        lambda jid, live_iids=(), **kw: {
                            "status": "done", "display_status": "done",
                            "instance_id": V9_LIVE_BOX})
    vjobsview._fold_fleet_jobs({V9_LIVE_BOX})
    body = _json.loads((tmp_path / "vast-jobmeta" / V9_JOB_ID
                        / "view.json").read_text())
    assert body["status"] == "done"
    assert body[vjobsview._JOB_VIEW_CACHE_KEY] == vjobsview._JOB_VIEW_CACHE_V

    def boom(*a, **k):
        raise AssertionError("a `done` view must not be re-read from B2")

    monkeypatch.setattr(_jm, "read_job", boom)
    assert vjobsview._fold_fleet_jobs({V9_LIVE_BOX})[V9_LIVE_BOX][0]["status"] == "done"


def test_v9_unstamped_cache_entry_is_re_read(monkeypatch, tmp_path):
    """Every view.json written before this fix is still on disk, frozen at a
    `failed` that has since been requeued. An unstamped body is not trusted."""
    _wire_v9_fold(monkeypatch, tmp_path, poison=False)
    d = tmp_path / "vast-jobmeta" / V9_JOB_ID
    d.mkdir(parents=True)
    (d / "view.json").write_text(_json.dumps(
        {"status": "done", "display_status": "done", "instance_id": "9999"}))
    got = vjobsview._fold_fleet_jobs({V9_LIVE_BOX})
    assert got[V9_LIVE_BOX][0]["instance_id"] == V9_LIVE_BOX


def test_v9_live_training_box_is_not_a_zombie(monkeypatch, tmp_path):
    """End to end, on the real log and the real 27-minute JOBD_STATUS gap: the
    box that was told to `destroy + relaunch` at step 142/156 classifies OK."""
    _wire_v9_fold(monkeypatch, tmp_path)
    monkeypatch.setattr(vhealth, "_jobd_heartbeat_epoch_soft",
                        lambda iid: vfmt._ts_to_epoch(V9_JOBD_STATUS_TS))
    now = _v9_now()
    ins = [{"id": int(V9_LIVE_BOX), "actual_status": "running", "machine_id": 1,
            "start_date": now - 99999, "extra_env": [["CRED_ROLE", "jobs"]]}]
    health = vhealth.gather_fleet_health(ins, vjobsview._fold_fleet_jobs({V9_LIVE_BOX}),
                                         now=now)
    h = health[V9_LIVE_BOX]
    assert h["verdict"] == vhealth.GUARD_OK, h["reason"]
    assert h["evidence"]["jobd_hb_src"] == "jobs"


def test_v9_stale_jobd_status_alone_never_outranks_a_live_job(monkeypatch):
    """The guard rule the incident cost us: the transition-driven JOBD_STATUS
    marker can move the liveness epoch FORWARD, never backward. With a correctly
    folded job the 27-minute marker gap is irrelevant."""
    monkeypatch.setattr(vhealth, "_jobd_heartbeat_epoch_soft",
                        lambda iid: vfmt._ts_to_epoch(V9_JOBD_STATUS_TS))
    now = _v9_now()
    live = _jm.fold_events(_v9_events(), live_iids={V9_LIVE_BOX})
    ep, src, pyh = vhealth._fleet_jobd_hb_epoch(V9_LIVE_BOX, [live], now)
    assert src == "jobs" and (now - ep) <= STALE
    # A fresh fold skips the JOBD_STATUS read entirely, so pyhalf is UNKNOWN
    # here — never False, and never True. Unknown is the safe default.
    assert pyh is None


def test_job_liveness_epoch_takes_every_jobd_written_stamp():
    """A checkpoint or a bare `started` is as much proof of life as a heartbeat
    — after a preempt-resume it is the ONLY one for a while."""
    assert vhealth._job_liveness_epoch({}) is None
    assert vhealth._job_liveness_epoch(
        {"last_heartbeat_ts": _jm_ts(NOW - 900),
         "last_checkpoint_ts": _jm_ts(NOW - 30)}) == NOW - 30
    assert vhealth._job_liveness_epoch(
        {"last_heartbeat_ts": _jm_ts(NOW - 900),
         "last_event_ts": _jm_ts(NOW - 10)}) == NOW - 10


def test_weak_jobd_status_only_zombie_says_it_is_the_weak_signal(monkeypatch):
    """A box with NO folded job still alarms — a missed zombie costs money — but
    it must not claim evidence it does not have."""
    monkeypatch.setattr(vhealth, "_jobd_heartbeat_epoch_soft", lambda iid: NOW - 5000)
    ins = [_box("running", age=99999, cred_jobs=True, iid=1)]
    h = vhealth.gather_fleet_health(ins, {}, now=NOW)["1"]
    assert h["verdict"] == vhealth.GUARD_ZOMBIE_NO_JOBD
    assert h["evidence"]["jobd_hb_src"] == "jobd-status"
    assert "WEAK" in h["reason"] and "confirm" in h["reason"]


def test_quiet_job_on_a_live_box_is_still_a_zombie(monkeypatch):
    """The other half of the asymmetry: degrading toward alive must not blind
    the guard. When the job's OWN events go quiet, that is the strong reading
    and the verdict stands."""
    monkeypatch.setattr(vhealth, "_jobd_heartbeat_epoch_soft", lambda iid: None)
    ins = [_box("running", age=99999, cred_jobs=True, iid=1)]
    jobs = {"1": [{"display_status": "running",
                   "last_heartbeat_ts": _jm_ts(NOW - 5000),
                   "last_event_ts": _jm_ts(NOW - 5000)}]}
    h = vhealth.gather_fleet_health(ins, jobs, now=NOW)["1"]
    assert h["verdict"] == vhealth.GUARD_ZOMBIE_NO_JOBD
    assert h["evidence"]["jobd_hb_src"] == "jobs"
    assert "daemon dead" in h["reason"]


# --------------------------------------------------------------------------- #
# ZOMBIE_PYHALF — the box's own fail-closed confession, surfaced as a verdict
# (FAILCLOSED_DESIGN §5/§8; the herdd half the fail-closed change could not
# reach). Every test here is about ONE property: only `True` acts, because
# `None` is what every box on a bundle older than the field reports and reading
# absence as broken would flag the entire in-flight fleet the day it lands.
# --------------------------------------------------------------------------- #
def test_jobd_status_pyhalf_is_tri_state():
    ok = f"IDLE {_ftz(NOW)} pyhalf=ok"
    broken = (f"IDLE {_ftz(NOW)} pyhalf=broken "
              f"pyreason=jobd.py_selftest_rc=3:_No_module_named_bidpolicy")
    assert vhealth.jobd_status_pyhalf(ok) is False
    assert vhealth.jobd_status_pyhalf(broken) is True
    assert vhealth.jobd_status_pyhalf(f"RUNNING 2 {_ftz(NOW)} pids") is None  # old bundle
    assert vhealth.jobd_status_pyhalf("") is None
    assert vhealth.jobd_status_pyhalf(None) is None
    # a value we do not recognise is NOT broken
    assert vhealth.jobd_status_pyhalf(f"IDLE {_ftz(NOW)} pyhalf=weird") is None
    # the field rides at the END of the line, after a 2-token state and extras
    assert vhealth.jobd_status_pyhalf(
        f"RUNNING 2 {_ftz(NOW)} staging=asset mbps=12 pid1 pyhalf=broken") is True


def test_fleetd_and_herdd_parse_pyhalf_with_one_implementation():
    """The alarm (this module) and the teeth (fleetd._pyhalf_tick) must never
    disagree about what a box said. One parse, two readers."""
    import fleetd
    line = f"IDLE {_ftz(NOW)} pyhalf=broken"
    assert fleetd.pyhalf_broken(line) is vhealth.jobd_status_pyhalf(line) is True
    assert fleetd.pyhalf_broken(f"IDLE {_ftz(NOW)}") is None


def test_pyhalf_broken_is_its_own_verdict():
    """A confessed-broken box keeps its (now periodic) beacon FRESH, so rule 4
    is silent on it. Without this verdict the box classifies OK."""
    h = vhealth.classify_box_health(_box("running", age=99999, cred_jobs=True),
                                    jobd_hb_epoch=NOW - 30, jobd_hb_src="jobd-status",
                                    jobd_pyhalf=True, now=NOW)
    assert h.verdict == vhealth.GUARD_ZOMBIE_PYHALF
    assert h.evidence["pyhalf"] is True
    assert h.evidence["phase"] == "up"
    assert "pyhalf=broken" in h.reason
    # the reason must STEER AWAY from the reschedule remedy: it is a bundle
    # fault, so a different host reproduces it.
    assert "do NOT destroy" in h.reason


def test_pyhalf_ok_is_not_a_zombie():
    h = vhealth.classify_box_health(_box("running", age=99999, cred_jobs=True),
                                    jobd_hb_epoch=NOW - 30, jobd_pyhalf=False, now=NOW)
    assert h.verdict == vhealth.GUARD_OK
    assert h.evidence["pyhalf"] is False


def test_pyhalf_absent_field_never_reads_as_broken():
    """BACK-COMPAT, the load-bearing one. A box on a bundle older than the
    field reports None, and every verdict must come out exactly as it did
    before the field existed."""
    fresh = vhealth.classify_box_health(_box("running", age=99999, cred_jobs=True),
                                        jobd_hb_epoch=NOW - 30, now=NOW)
    assert fresh.verdict == vhealth.GUARD_OK
    assert fresh.evidence["pyhalf"] is None
    stale = vhealth.classify_box_health(_box("running", age=99999, cred_jobs=True),
                                        jobd_hb_epoch=NOW - 5000, now=NOW)
    assert stale.verdict == vhealth.GUARD_ZOMBIE_NO_JOBD
    never = vhealth.classify_box_health(_box("running", age=99999, cred_jobs=True),
                                        now=NOW)
    assert never.verdict == vhealth.GUARD_ZOMBIE_NO_JOBD


def test_pyhalf_outranks_the_symptom_it_causes():
    """A confessed-broken box REFUSES to claim tickets (jobd.sh poll_once
    returns early), so its queue ages out into ZOMBIE_TICKET_UNCLAIMED — a
    verdict whose text sends the reader hunting a claiming bug that is not
    there. The confession has to win."""
    jobs = [{"display_status": "submitted", "last_event_ts": _jm_ts(NOW - 9000)}]
    sym = vhealth.classify_box_health(_box("running", age=99999, cred_jobs=True),
                                      jobs=jobs, jobd_hb_epoch=NOW - 30, now=NOW)
    assert sym.verdict == vhealth.GUARD_ZOMBIE_TICKET_UNCLAIMED     # the old reading
    h = vhealth.classify_box_health(_box("running", age=99999, cred_jobs=True),
                                    jobs=jobs, jobd_hb_epoch=NOW - 30,
                                    jobd_pyhalf=True, now=NOW)
    assert h.verdict == vhealth.GUARD_ZOMBIE_PYHALF


def test_pyhalf_never_fires_on_a_non_jobs_or_dead_box():
    """Rule ordering: the confession is checked AFTER the not-live and
    not-a-jobs-box exits, so a stray field can never reclassify either."""
    assert vhealth.classify_box_health(_box("running", age=99999),
                                       jobd_pyhalf=True, now=NOW).verdict == vhealth.GUARD_OK
    assert vhealth.classify_box_health(_box("stopped", age=99999, cred_jobs=True),
                                       jobd_pyhalf=True, now=NOW).verdict == vhealth.GUARD_OK


def test_pyhalf_is_a_loud_zombie_verdict_but_licenses_nothing():
    """In the zombie set (so `ls`/`guard` scream and exit 2), and ALARM in the
    graded policy even with `confirmed=True` — the remedy for a bundle fault is
    not a reschedule onto an innocent host."""
    import parked_lifecycle as pl
    assert vhealth.GUARD_ZOMBIE_PYHALF in vhealth._ZOMBIE_VERDICTS
    assert vhealth._VERDICT_SHORT[vhealth.GUARD_ZOMBIE_PYHALF]
    action, why = pl.zombie_action(verdict=vhealth.GUARD_ZOMBIE_PYHALF,
                                   is_jobs_box=True, jobd_ever_stamped=True,
                                   jobd_hb_read=True, label_kept=False,
                                   confirmed=True)
    assert action == pl.ZOMBIE_ALARM
    assert "bundle fault" in why.lower()


def test_guard_fix_holds_a_pyhalf_box(monkeypatch):
    # (the never-ran listing is still paid for any jobs-lane row — harmless
    # here, since the graded policy ignores it for this verdict)
    monkeypatch.setattr(vhealth, "_jobd_ever_stamped", lambda iid: False)
    rows = [vhealth.classify_box_health(_box("running", age=99999, cred_jobs=True,
                                             iid=1),
                                        jobd_hb_epoch=NOW - 30, jobd_pyhalf=True,
                                        now=NOW)._asdict()]
    plan = vreap._guard_fix_plan(rows, {"1": {"id": 1, "label": ""}})
    assert [act for _h, act, _w in plan] == ["alarm"]


def test_guard_evidence_bits_name_the_confession():
    h = vhealth.classify_box_health(_box("running", age=99999, cred_jobs=True),
                                    jobd_hb_epoch=NOW - 30, jobd_pyhalf=True, now=NOW)
    assert "pyhalf BROKEN" in vreap._guard_evidence_bits(h.evidence)
    ok = vhealth.classify_box_health(_box("running", age=99999, cred_jobs=True),
                                     jobd_hb_epoch=NOW - 30, jobd_pyhalf=False,
                                     now=NOW)
    assert "pyhalf" not in vreap._guard_evidence_bits(ok.evidence)


def test_gather_fleet_health_folds_the_confession(monkeypatch):
    """End to end through the read path: an IDLE jobs box (no fresh folded job)
    pays the JOBD_STATUS read, and the pyhalf field on it reaches the verdict."""
    monkeypatch.setenv("B2_BUCKET", "bkt")
    line = f"IDLE {_ftz(NOW - 30)} pyhalf=broken pyreason=selftest_rc3"
    monkeypatch.setattr(vb2, "_rclone_soft", lambda args: (0, line + "\n", ""))
    ins = [_box("running", age=99999, cred_jobs=True, iid=1)]
    h = vhealth.gather_fleet_health(ins, {}, now=NOW)["1"]
    assert h["verdict"] == vhealth.GUARD_ZOMBIE_PYHALF
    assert h["evidence"]["pyhalf"] is True


def test_gather_fleet_health_unreadable_marker_is_not_a_confession(monkeypatch):
    """I3, restated for this field: a B2 outage makes every marker unreadable
    at once. Unknown, never broken."""
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(vb2, "_rclone_soft", lambda args: (1, "", "b2 down"))
    ins = [_box("running", age=99999, cred_jobs=True, iid=1)]
    h = vhealth.gather_fleet_health(ins, {}, now=NOW)["1"]
    assert h["evidence"]["pyhalf"] is None
    assert h["verdict"] != vhealth.GUARD_ZOMBIE_PYHALF


def test_gather_fleet_health_pyhalf_read_is_bounded_by_the_fold(monkeypatch):
    """The read budget is unchanged: a box with a FRESH folded job event pays
    no JOBD_STATUS read at all, so its pyhalf is unknown rather than ok. The
    confession still surfaces within GUARD_JOBD_STALE_S, because a broken
    python half is exactly what stops those job events."""
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(vb2, "_rclone_soft",
                        lambda args: pytest.fail("fresh fold must read nothing"))
    ins = [_box("running", age=99999, cred_jobs=True, iid=1)]
    jobs = {"1": [{"display_status": "running",
                   "last_heartbeat_ts": _jm_ts(NOW - 30),
                   "last_event_ts": _jm_ts(NOW - 30)}]}
    h = vhealth.gather_fleet_health(ins, jobs, now=NOW)["1"]
    assert h["verdict"] == vhealth.GUARD_OK
    assert h["evidence"]["pyhalf"] is None
