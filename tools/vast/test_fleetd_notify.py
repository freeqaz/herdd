"""fleetd's notification poll — NOTIFY_DESIGN S2a, OBSERVABILITY ONLY.

The bar these tests hold: the poll may add journal rows and it may not change
anything else. So beside the cursor/dedup/gap/health behaviour there is an
explicit test that a tick with notifications wired produces exactly the same
fleet actions as one without (`test_the_poll_changes_no_fleet_behaviour`) —
S2b is where rows become inputs, and until that review happens a regression
into policy is the failure mode worth spending a test on.

Payloads are the REAL captured inbox (`testfixtures/notify/inbox_2026-08-16.json`,
50 rows / 16 outbids); the fake hook replays slices of it. No network.
"""
from __future__ import annotations

import copy
import json
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fleetd                                                    # noqa: E402
import notify                                                    # noqa: E402
from test_fleetd import FakeHooks, journal, events                # noqa: E402

_FIX = pathlib.Path(__file__).resolve().parent / "testfixtures" / "notify"


def _inbox():
    with open(_FIX / "inbox_2026-08-16.json") as fh:
        return json.load(fh)


def _rows():
    return sorted(_inbox()["notifications"], key=lambda r: r["created_at"])


def _envelope(rows):
    return {"success": True, "notifications": list(rows), "last_seen_at": None,
            "seen_through_at": None, "unread_count": len(rows)}


class NotifyHooks(FakeHooks):
    """FakeHooks plus a scriptable notification poll."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.notify_queue = []            # list of (payload, err) per tick
        self.notify_calls = 0
        self.notify_raises = None

    def notifications(self):
        self.notify_calls += 1
        if self.notify_raises is not None:
            raise self.notify_raises
        if not self.notify_queue:
            return _envelope([]), None
        return self.notify_queue.pop(0)


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    monkeypatch.setenv("FLEETD_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("FLEETD_GLOBAL_BUDGET_USD", raising=False)
    monkeypatch.delenv("FLEETD_NOTIFY", raising=False)
    return fleetd.Fleet(str(tmp_path / "state"), hooks=NotifyHooks())


def _seen(f):
    return [r for r in journal(f) if r["event"] == notify.SEEN_EVENT]


def _notify_state(f):
    return f.state.get("notify") or {}


# --------------------------------------------------------------------------- #
# the poll itself
# --------------------------------------------------------------------------- #
def test_one_poll_per_tick_and_every_row_journaled_once(fleet):
    rows = _rows()
    fleet.hooks.notify_queue = [(_envelope(rows[:30]), None),
                                (_envelope(rows[10:]), None)]
    fleet.tick()
    fleet.hooks.advance(45)
    fleet.tick()
    assert fleet.hooks.notify_calls == 2
    ids = [r["event_id"] for r in _seen(fleet)]
    assert len(ids) == len(set(ids)) == 50, "overlapping polls must not re-journal"


def test_a_seen_row_carries_the_associated_id_verbatim(fleet):
    row = [r for r in _rows() if r["notif_type"] == "outbid"
           and r["associated_id"]["instance_id"] == 47840057][0]
    fleet.hooks.notify_queue = [(_envelope([row]), None)]
    fleet.tick()
    rec = _seen(fleet)[0]
    assert rec["notif_type"] == "outbid"
    assert rec["event_id"] == row["event_id"]
    assert rec["associated_id"] == row["associated_id"]
    assert rec["created_at"] == row["created_at"]
    # the box id is lifted to `iid` so `fleet report --box` / `fleet log --iid`
    # find the row beside the eviction it explains
    assert rec["iid"] == "47840057" and rec["machine_id"] == 56759


def test_cursor_persists_across_a_restart(fleet, tmp_path):
    rows = _rows()
    fleet.hooks.notify_queue = [(_envelope(rows), None)]
    fleet.tick()
    assert len(_seen(fleet)) == 50
    cur = _notify_state(fleet)["cursor"]
    assert cur["last_created_at"] == max(r["created_at"] for r in rows)

    reborn = fleetd.Fleet(fleet.dir, hooks=NotifyHooks())
    reborn.hooks.t = fleet.hooks.t + 600.0
    reborn.hooks.notify_queue = [(_envelope(rows), None)]
    reborn.tick()
    # the journal is the same file, so a re-consumed window would show up as
    # 100 rows rather than 50
    assert len(_seen(reborn)) == 50, (
        "a restart must not re-journal the window the previous process "
        "consumed — the cursor is the only thing that remembers")
    assert reborn.state["notify"]["cursor"] == cur


def test_a_corrupt_cursor_is_survivable(fleet):
    fleet.state["notify"] = {"cursor": "not-a-cursor"}
    fleet.hooks.notify_queue = [(_envelope(_rows()[:3]), None)]
    fleet.tick()
    assert len(_seen(fleet)) == 3
    assert isinstance(_notify_state(fleet)["cursor"]["recent_ids"], list)


def test_a_full_unknown_window_journals_a_gap(fleet):
    rows = _rows()
    fleet.hooks.notify_queue = [(_envelope(rows[:5]), None)]
    fleet.tick()
    storm = copy.deepcopy(rows)
    top = max(r["created_at"] for r in rows)
    for i, r in enumerate(storm):
        r["event_id"] = f"{i:032x}"
        r["created_at"] = top + 100 + i
    fleet.hooks.notify_queue = [(_envelope(storm), None)]
    fleet.hooks.advance(45)
    fleet.tick()
    gaps = [r for r in journal(fleet) if r["event"] == notify.GAP_EVENT]
    assert len(gaps) == 1 and gaps[0]["rows"] == 50 and gaps[0]["window"] == 50
    assert _notify_state(fleet)["gaps"] == 1


def test_the_first_poll_is_not_a_gap(fleet):
    """A fresh install sees 50 unknown rows by definition — crying gap there
    would train the operator to ignore the one alarm that means a storm."""
    fleet.hooks.notify_queue = [(_envelope(_rows()), None)]
    fleet.tick()
    assert [r for r in journal(fleet) if r["event"] == notify.GAP_EVENT] == []
    assert len(_seen(fleet)) == 50


# --------------------------------------------------------------------------- #
# degradation (D2)
# --------------------------------------------------------------------------- #
def test_poll_errors_journal_on_state_change_only(fleet):
    err = "network <urlopen error timed out> on GET v0/notifications/inbox/"
    fleet.hooks.notify_queue = [(None, err)] * 4
    for _ in range(4):
        fleet.tick()
        fleet.hooks.advance(45)
    health = [r for r in journal(fleet) if r["event"] == notify.POLL_ERROR_EVENT]
    assert len(health) == 1, "4 failing ticks are ONE fact, not four"
    assert health[0]["state"] == "failing" and "timed out" in health[0]["error"]
    assert _notify_state(fleet)["consecutive_failures"] == 4

    fleet.hooks.notify_queue = [(_envelope(_rows()[:2]), None)]
    fleet.tick()
    health = [r for r in journal(fleet) if r["event"] == notify.POLL_ERROR_EVENT]
    assert [h["state"] for h in health] == ["failing", "ok"]
    assert health[-1]["failures"] == 4
    assert _notify_state(fleet)["poll_ok"] is True
    assert len(_seen(fleet)) == 2, "recovery must resume consuming rows"


def test_a_retired_endpoint_degrades_to_current_behaviour(fleet):
    """The inbox is commented out of vast's published spec. Its 404 is an
    expected end state: one journal line, then exactly the pre-notify daemon."""
    gone = "HTTP 404 on GET v0/notifications/inbox/: {'detail': 'Not Found'}"
    fleet.hooks.box(4711, label="jobs:x")
    fleet.hooks.notify_queue = [(None, gone)] * 3
    for _ in range(3):
        fleet.tick()
        fleet.hooks.advance(45)
    health = [r for r in journal(fleet) if r["event"] == notify.POLL_ERROR_EVENT]
    assert len(health) == 1 and health[0]["gone"] is True
    assert notify.SEEN_EVENT not in events(fleet)
    assert "watch_error" not in events(fleet)


def test_a_hook_that_raises_is_a_failing_poll_not_a_dead_tick(fleet):
    fleet.hooks.notify_raises = RuntimeError("transport exploded")
    fleet.hooks.box(4712)
    fleet.tick()
    health = [r for r in journal(fleet) if r["event"] == notify.POLL_ERROR_EVENT]
    assert len(health) == 1 and "transport exploded" in health[0]["error"]
    assert fleet.state["meta"].get("last_ok_tick_ts"), "the tick still completed"


def test_an_unreadable_payload_is_a_failing_poll(fleet, monkeypatch):
    """`notify.poll` is total by construction, so a raise there is a BUG — it
    must still be announced once and never per tick."""
    def boom(*a, **kw):
        raise ValueError("cursor logic bug")
    monkeypatch.setattr(notify, "poll", boom)
    for _ in range(3):
        fleet.tick()
        fleet.hooks.advance(45)
    health = [r for r in journal(fleet) if r["event"] == notify.POLL_ERROR_EVENT]
    assert len(health) == 1 and health[0]["state"] == "failing"


def test_malformed_and_unknown_rows_are_tolerated(fleet):
    """Three shapes the server may legitimately send us tomorrow: an unknown
    `notif_type` (30 keys exist, we have observed 3), a row with no
    `associated_id` (the documented webhook payload has none), and a row with
    extra keys."""
    base = _rows()[0]
    unknown = copy.deepcopy(base)
    unknown.update(event_id="a" * 32, notif_type="upcoming_downtime",
                   associated_id={"machine_id": 999, "when": "2026-08-20"})
    bare = copy.deepcopy(base)
    bare.update(event_id="b" * 32, notif_type="low_credit")
    bare.pop("associated_id")
    extra = copy.deepcopy(base)
    extra.update(event_id="c" * 32, brand_new_field={"nested": [1, 2]})
    junk = {"event_id": "d" * 32}                       # no type, no stamp
    fleet.hooks.notify_queue = [(_envelope([unknown, bare, extra, junk]), None)]
    fleet.tick()
    got = {r["event_id"]: r for r in _seen(fleet)}
    assert len(got) == 4
    assert got["a" * 32]["associated_id"] == {"machine_id": 999,
                                              "when": "2026-08-20"}
    assert "associated_id" not in got["b" * 32] and "iid" not in got["b" * 32]
    assert got["c" * 32]["notif_type"] == base["notif_type"]
    assert "notif_type" not in got["d" * 32]


def test_a_payload_that_is_not_an_envelope_is_survivable(fleet):
    for payload in ({"success": True}, {"notifications": "nope"}, [], "garbage"):
        fleet.hooks.notify_queue = [(payload, None)]
        fleet.tick()
        fleet.hooks.advance(45)
    assert _seen(fleet) == []
    assert [r for r in journal(fleet)
            if r["event"] == notify.POLL_ERROR_EVENT] == []


def test_the_knob_turns_the_poll_off(fleet, monkeypatch):
    monkeypatch.setenv("FLEETD_NOTIFY", "0")
    fleet.hooks.notify_queue = [(_envelope(_rows()), None)]
    fleet.tick()
    assert fleet.hooks.notify_calls == 0 and _seen(fleet) == []


def test_a_hooks_object_without_the_seam_is_silent(fleet):
    """An older Hooks (or any fake that predates S2a) must not read as a failing
    poll — that would journal a transition on a daemon that simply cannot poll."""
    fleet.hooks = FakeHooks()
    fleet.tick()
    assert [r for r in journal(fleet)
            if r["event"].startswith("notify_")] == []


# --------------------------------------------------------------------------- #
# the boundary: observability only
# --------------------------------------------------------------------------- #
def test_the_poll_changes_no_fleet_behaviour(tmp_path, monkeypatch):
    """Same fleet, same script, notifications on vs off: identical actions and
    identical non-notify journal. S2b is where rows become inputs."""
    monkeypatch.delenv("FLEETD_NOTIFY", raising=False)

    def run(with_rows):
        d = tmp_path / ("on" if with_rows else "off")
        h = NotifyHooks()
        h.box(5001, label="jobs:one", dph=1.0)
        h.box(5002, label=None, dph=3.0)
        if with_rows:
            h.notify_queue = [(_envelope(_rows()), None)]
        f = fleetd.Fleet(str(d), hooks=h)
        f.watch("5001", "bare", budget_usd=5.0)
        for _ in range(4):
            f.tick()
            h.advance(300)
        return f, h

    on, hon = run(True)
    off, hoff = run(False)
    assert (hon.parked, hon.resumed, hon.destroyed, hon.kept) == \
           (hoff.parked, hoff.resumed, hoff.destroyed, hoff.kept)
    strip = lambda f: [r["event"] for r in journal(f)
                       if not r["event"].startswith("notify_")]
    assert strip(on) == strip(off)
    assert [r["event"] for r in journal(on)
            if r["event"] == notify.SEEN_EVENT], "the ON arm did poll"
    assert on.alarms == off.alarms
