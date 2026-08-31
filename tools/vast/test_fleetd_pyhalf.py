"""fleetd teeth for a self-reported dead python half (FAILCLOSED_DESIGN §8).

After box 47737955 billed $1.742 doing nothing, the obvious move was to give
fleetd's ZOMBIE_NO_JOBD alarm teeth — it fired correctly at T+20min and nothing
acted on it. These tests encode why that would have been the wrong fix and what
was done instead.

ZOMBIE_NO_JOBD is derived from JOBD_STATUS staleness, and JOBD_STATUS used to be
stamped only on state transitions, so it goes stale on every HEALTHY IDLE jobs
box too. Enforcing on it parks working rented boxes — the same bug pointing the
other way. So the teeth bite on `pyhalf=broken`: not an inference from silence
but a confession the box writes after a deterministic offline capability check
fails.

Reuses test_fleetd.py's FakeHooks (the single fake-transport source of truth)
with one added reader, rather than forking it.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fleetd import FakeHooks, events, journal                # noqa: E402
from vastlib.fleet import daemon                                  # noqa: E402


class PyHalfHooks(FakeHooks):
    """FakeHooks plus the JOBD_STATUS body reader. `status_map` is iid -> the
    raw marker line, or None for "unreadable" (which is a DIFFERENT answer from
    a line that says ok, and the tests below hold it to that)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.status_map = {}

    def jobd_status_line(self, iid):
        return self.status_map.get(str(iid))


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    monkeypatch.setenv("FLEETD_STATE_DIR", str(tmp_path / "state"))
    for k in ("FLEETD_GLOBAL_BUDGET_USD", "FLEETD_UNWATCHED_GRACE_S",
              "FLEETD_UNWATCHED_GRACE_EXPENSIVE_S", "FLEETD_EXPENSIVE_DPH",
              "FLEETD_PYHALF_CONFIRM_S"):
        monkeypatch.delenv(k, raising=False)
    return daemon.Fleet(str(tmp_path / "state"), hooks=PyHalfHooks())


BROKEN = ("IDLE 2026-08-13T04:12:00Z pyhalf=broken "
          "pyreason=boot_selftest_rc=3:_ModuleNotFoundError:_No_module_named_'bidpolicy'")
OK = "IDLE 2026-08-13T04:12:00Z pyhalf=ok"
OLD_BUNDLE = "IDLE 2026-08-13T04:12:00Z"          # no pyhalf field at all


# --------------------------------------------------------------------------
# the pure parser
# --------------------------------------------------------------------------

def test_pyhalf_broken_is_tristate():
    assert daemon.pyhalf_broken(BROKEN) is True
    assert daemon.pyhalf_broken(OK) is False
    # A bundle too old to carry the field is NOT a sick box. This is what makes
    # the daemon-side change safe to deploy ahead of the boxes.
    assert daemon.pyhalf_broken(OLD_BUNDLE) is None
    assert daemon.pyhalf_broken(None) is None
    assert daemon.pyhalf_broken("") is None
    # RUNNING carries a job count before the stamp; the field still parses
    assert daemon.pyhalf_broken("RUNNING 2 2026-08-13T04:12:00Z x=1 pyhalf=broken") is True


# --------------------------------------------------------------------------
# enforcement
# --------------------------------------------------------------------------

def test_confessed_broken_half_parks_after_the_confirm_window(fleet):
    """The teeth. A jobs box that says it cannot run jobd.py is parked — after
    a confirm window, and PARKED not destroyed (fleetd never originates a
    destroy: FLEETD_DESIGN §3/§8)."""
    fleet.hooks.box(4001, dph=2.4)
    fleet.hooks.status_map["4001"] = BROKEN
    fleet.watch("4001", "jobs", budget_usd=100.0)

    fleet.tick()                                   # first sighting: arm only
    assert fleet.hooks.parked == []
    assert "pyhalf_broken_seen" in events(fleet)

    fleet.hooks.advance(daemon.PYHALF_CONFIRM_S + 1)
    fleet.tick()
    assert fleet.hooks.parked == ["4001"]
    assert fleet.hooks.destroyed == [], "fleetd must never originate a destroy"
    assert "pyhalf_parked" in events(fleet)
    assert fleet.state["watches"]["4001"]["state"] == "pyhalf_parked"
    assert fleet.state["watches"]["4001"]["dormant"] is True
    assert fleet.hooks.kept == ["4001"], "a fleetd park is a resumability promise"
    assert any("PYHALF BROKEN" in a for a in fleet.alarms)


def test_one_sighting_is_not_enough(fleet):
    """Proportionality: the confirm window is real, not decorative."""
    fleet.hooks.box(4002, dph=2.4)
    fleet.hooks.status_map["4002"] = BROKEN
    fleet.watch("4002", "jobs", budget_usd=100.0)
    fleet.tick()
    fleet.hooks.advance(daemon.PYHALF_CONFIRM_S - 60)
    fleet.tick()
    assert fleet.hooks.parked == []


# --------------------------------------------------------------------------
# false-positive guards — each of these is a way to park a HEALTHY rented box
# --------------------------------------------------------------------------

def test_a_healthy_box_is_never_parked(fleet):
    fleet.hooks.box(4003, dph=2.4)
    fleet.hooks.status_map["4003"] = OK
    fleet.watch("4003", "jobs", budget_usd=100.0)
    for _ in range(5):
        fleet.tick()
        fleet.hooks.advance(daemon.PYHALF_CONFIRM_S)
    assert fleet.hooks.parked == []


def test_an_old_bundle_without_the_field_is_never_parked(fleet):
    """Every box in the fleet on the day this lands reports OLD_BUNDLE. If that
    read as broken, the change would park the entire fleet on deploy."""
    fleet.hooks.box(4004, dph=2.4)
    fleet.hooks.status_map["4004"] = OLD_BUNDLE
    fleet.watch("4004", "jobs", budget_usd=100.0)
    for _ in range(5):
        fleet.tick()
        fleet.hooks.advance(daemon.PYHALF_CONFIRM_S)
    assert fleet.hooks.parked == []


def test_an_unreadable_marker_never_accumulates_confirm_time(fleet):
    """THE FLEET-WIDE FAILURE MODE. A B2 outage makes every marker unreadable at
    once. `None` must be "we do not know", never "broken" — and it must also
    RESET a confirmation already in progress, so an outage cannot finish a park
    that a real fault started but that has since cleared."""
    fleet.hooks.box(4005, dph=2.4)
    fleet.hooks.status_map["4005"] = BROKEN
    fleet.watch("4005", "jobs", budget_usd=100.0)
    fleet.tick()                                   # arm
    assert fleet.state["watches"]["4005"].get("_pyhalf_since") is not None

    fleet.hooks.status_map["4005"] = None          # B2 goes away
    fleet.hooks.advance(daemon.PYHALF_CONFIRM_S + 1)
    fleet.tick()
    assert fleet.hooks.parked == []
    assert fleet.state["watches"]["4005"].get("_pyhalf_since") is None


def test_recovery_clears_the_clock_and_the_alarm(fleet):
    """A transient that resolves must leave no residue — otherwise the next
    unrelated blip inherits a nearly-expired confirm window."""
    fleet.hooks.box(4006, dph=2.4)
    fleet.hooks.status_map["4006"] = BROKEN
    fleet.watch("4006", "jobs", budget_usd=100.0)
    fleet.tick()
    fleet.hooks.advance(daemon.PYHALF_CONFIRM_S - 30)
    fleet.tick()
    assert any("PYHALF BROKEN" in a for a in fleet.alarms)

    fleet.hooks.status_map["4006"] = OK
    fleet.tick()
    assert fleet.state["watches"]["4006"].get("_pyhalf_since") is None
    assert not any("PYHALF BROKEN" in a for a in fleet.alarms)

    # ...and the window starts over from scratch, not from where it left off
    fleet.hooks.status_map["4006"] = BROKEN
    fleet.tick()
    fleet.hooks.advance(daemon.PYHALF_CONFIRM_S - 30)
    fleet.tick()
    assert fleet.hooks.parked == []


def test_non_jobs_profiles_are_out_of_scope(fleet):
    """A train/serve/bare box carries its own supervise watchdog and has no jobd
    to confess with; the marker on such an iid would be a leftover."""
    fleet.hooks.box(4007, dph=2.4)
    fleet.hooks.status_map["4007"] = BROKEN
    fleet.watch("4007", "bare", budget_usd=100.0)
    fleet.tick()
    fleet.hooks.advance(daemon.PYHALF_CONFIRM_S + 1)
    fleet.tick()
    assert fleet.hooks.parked == []


def test_enforcement_can_be_disabled(fleet, monkeypatch):
    """The escape hatch, matching the unwatched-grace convention: <= 0 leaves
    the condition as an alarm and never acts."""
    monkeypatch.setenv("FLEETD_PYHALF_CONFIRM_S", "0")
    fleet.hooks.box(4008, dph=2.4)
    fleet.hooks.status_map["4008"] = BROKEN
    fleet.watch("4008", "jobs", budget_usd=100.0)
    fleet.tick()
    fleet.hooks.advance(100000)
    fleet.tick()
    assert fleet.hooks.parked == []


def test_a_hooks_double_without_the_reader_disarms_rather_than_raising(fleet):
    """fleetd's own FakeHooks does not inherit from Hooks. A missing reader must
    leave the teeth unarmed, not raise through the whole tick and take every
    other watch down with it."""
    fleet.hooks.box(4009, dph=2.4)
    fleet.watch("4009", "jobs", budget_usd=100.0)
    saved = PyHalfHooks.jobd_status_line            # grab it BEFORE the del
    del PyHalfHooks.jobd_status_line                # simulate the narrower double
    try:
        fleet.tick()
        fleet.hooks.advance(daemon.PYHALF_CONFIRM_S + 1)
        fleet.tick()
    finally:
        PyHalfHooks.jobd_status_line = saved
    assert fleet.hooks.parked == []


def test_park_failure_is_recorded_and_keeps_alarming(fleet):
    """A wedged box may refuse the park. That must be visible, not swallowed —
    it is the case where a human has to intervene."""
    fleet.hooks.box(4010, dph=2.4)
    fleet.hooks.status_map["4010"] = BROKEN
    fleet.hooks.park_ok = False
    fleet.watch("4010", "jobs", budget_usd=100.0)
    fleet.tick()
    fleet.hooks.advance(daemon.PYHALF_CONFIRM_S + 1)
    fleet.tick()
    assert "pyhalf_park_failed" in events(fleet)
    assert fleet.state["watches"]["4010"]["state"] == "pyhalf_park_failed"
    assert fleet.state["watches"]["4010"]["dormant"] is False
    assert any("PYHALF BROKEN" in a for a in fleet.alarms)


def test_the_journal_names_the_reason_the_box_gave(fleet):
    """Whoever reads the journal must be able to act without sshing anywhere:
    the park line has to carry why, and the alarm has to say where to look."""
    fleet.hooks.box(4011, dph=2.4)
    fleet.hooks.status_map["4011"] = BROKEN
    fleet.watch("4011", "jobs", budget_usd=100.0)
    fleet.tick()
    fleet.hooks.advance(daemon.PYHALF_CONFIRM_S + 1)
    fleet.tick()
    rows = [r for r in journal(fleet) if r["event"] == "pyhalf_parked"]
    assert rows and rows[0]["iid"] == "4011"
    assert rows[0]["held_s"] >= daemon.PYHALF_CONFIRM_S
    assert "NOT destroyed" in rows[0]["note"]


def test_a_wedge_that_ate_no_work_is_left_to_the_reaper(fleet):
    """Item 1, `FLEET_REVIEW_2026-08-20`. The disk of a box wedged MID-WORK is
    worth the permanent keep-token; one that wedged after everything shipped
    (queue all-terminal, results on B2) holds nothing worth $2-$4/day."""
    fleet.hooks.box(4012, dph=2.4)
    fleet.hooks.status_map["4012"] = BROKEN
    fleet.hooks.drained_map["4012"] = True
    fleet.hooks.results_map["4012"] = True
    fleet.watch("4012", "jobs", budget_usd=100.0)
    fleet.tick()
    fleet.hooks.advance(daemon.PYHALF_CONFIRM_S + 1)
    fleet.tick()
    assert fleet.hooks.parked == ["4012"] and fleet.hooks.kept == []
    assert "keep_label_skipped" in events(fleet)


def test_a_wedge_with_unfinished_tickets_keeps_its_disk(fleet):
    """The fail-open side: jobd is dead, so its tickets are non-terminal — the
    reason and the bundle that caused it must survive on the disk."""
    fleet.hooks.box(4013, dph=2.4)
    fleet.hooks.status_map["4013"] = BROKEN
    fleet.hooks.drained_map["4013"] = False
    fleet.watch("4013", "jobs", budget_usd=100.0)
    fleet.tick()
    fleet.hooks.advance(daemon.PYHALF_CONFIRM_S + 1)
    fleet.tick()
    assert fleet.hooks.kept == ["4013"]
    assert "keep_label_skipped" not in events(fleet)
