"""Tests for `notify` — vast's notification channel (NOTIFY_DESIGN S1 + S2a).

Every fixture in here is a REAL captured payload, not a hand-written shape:
`testfixtures/notify/*.json` are the verbatim bodies of the three read-only
probes of record (2026-08-16, this account) — 50 inbox rows including the 16
outbids, the 30-entry type catalog, and the empty webhook list. Synthetic rows
appear only where the point IS the malformation (a row with no `associated_id`,
an unknown `notif_type`), and each of those is a real row with one field
removed or renamed, so the surrounding shape is still the server's.

No network: the HTTP seam is `herdd.request_soft`, monkeypatched.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import notify                                                    # noqa: E402
from vastlib.core import api                                     # noqa: E402
from vastlib.cli.notify import _get, inbox as cli_inbox          # noqa: E402
from vastlib.cli.notify import types as cli_types                # noqa: E402
from vastlib.cli.notify import webhooks as cli_webhooks          # noqa: E402

_FIX = pathlib.Path(__file__).resolve().parent / "testfixtures" / "notify"


def _load(name):
    with open(_FIX / name) as fh:
        return json.load(fh)


@pytest.fixture
def inbox():
    """The real 50-row inbox response (2026-08-16)."""
    return _load("inbox_2026-08-16.json")


@pytest.fixture
def types():
    return _load("types_2026-08-16.json")


@pytest.fixture
def webhooks():
    return _load("webhooks_2026-08-16.json")


def _rows(payload):
    return payload["notifications"]


def _by_type(payload, kind):
    return [r for r in _rows(payload) if r.get("notif_type") == kind]


def _envelope(rows, **kw):
    """An inbox response carrying exactly `rows` — the envelope's own fields are
    the server's, only the row list is chosen."""
    env = {"success": True, "notifications": list(rows), "last_seen_at": None,
           "seen_through_at": None, "unread_count": len(rows)}
    env.update(kw)
    return env


# --------------------------------------------------------------------------- #
# the fixtures are what the design says they are
# --------------------------------------------------------------------------- #
def test_captured_fixture_matches_the_documented_ground_truths(inbox):
    """NOTIFY_DESIGN §1.3/§1.4 in assertion form. If a re-capture ever breaks
    this, the DOC is what changed and the doc loses to the measurement."""
    rows = _rows(inbox)
    assert len(rows) == notify.WINDOW == 50
    assert all(len(r["event_id"]) == 32 for r in rows)
    assert {r["user_context"] for r in rows} == {"client"}
    outbids = _by_type(inbox, "outbid")
    assert len(outbids) == 16
    for r in outbids:                       # the structured id the webhook lacks
        assert set(r["associated_id"]) == {"instance_id", "machine_id",
                                           "your_bid", "new_min_bid"}
    # §1.4's caveat, in the data: displacement BELOW our own bid is real.
    below = [r for r in outbids
             if notify.new_min_bid(r) <= notify.your_bid(r)]
    assert [notify.instance_id(r) for r in below] == [47840057]
    assert (notify.your_bid(below[0]), notify.new_min_bid(below[0])) == (0.16, 0.15)


def test_accessors_are_total_over_junk():
    """Every accessor answers on a row that is not a row at all — the tick may
    not raise on anything the server sends (D2)."""
    for junk in (None, {}, {"associated_id": "not-a-dict"}, {"created_at": "x"},
                 {"event_id": "", "notif_type": None}):
        assert notify.event_id(junk) is None or isinstance(notify.event_id(junk), str)
        assert notify.notif_type(junk) is None
        assert notify.created_at(junk) is None
        assert notify.associated(junk) == {}
        assert notify.instance_id(junk) is None
        assert notify.machine_id(junk) is None
    assert notify.inbox_rows("nonsense") == []
    assert notify.inbox_rows({"notifications": [1, "two", {"a": 1}]}) == [{"a": 1}]


def test_full_key_reassembles_the_webhook_form(inbox):
    """The inbox ships the UNPREFIXED slug; webhooks subscribe `client:outbid`."""
    r = _by_type(inbox, "outbid")[0]
    assert notify.notif_type(r) == "outbid"
    assert notify.full_key(r) == "client:outbid"


def test_is_gone_only_fires_on_404():
    assert notify.is_gone("HTTP 404 on GET v0/notifications/inbox/: {}")
    assert not notify.is_gone("HTTP 500 on GET v0/notifications/inbox/: {}")
    assert not notify.is_gone("network <urlopen error timed out>")
    assert not notify.is_gone(None)


# --------------------------------------------------------------------------- #
# the cursor (D3)
# --------------------------------------------------------------------------- #
def test_first_poll_takes_everything_and_is_not_a_gap(inbox):
    res = notify.poll(inbox, None)
    assert res.initialized is True
    assert len(res.new) == 50 and res.rows_seen == 50
    assert res.gap is False, ("a full window on the FIRST poll is initialization, "
                              "not a hole — flagging it would cry gap on install")
    assert res.new[0]["created_at"] < res.new[-1]["created_at"]   # oldest first
    assert res.cursor["last_created_at"] == max(r["created_at"] for r in _rows(inbox))
    assert len(res.cursor["recent_ids"]) == 50


def test_second_poll_of_the_same_window_yields_nothing(inbox):
    first = notify.poll(inbox, None)
    again = notify.poll(inbox, first.cursor)
    assert again.new == [] and again.gap is False
    assert again.cursor["recent_ids"] == first.cursor["recent_ids"]


def test_overlapping_polls_emit_each_event_exactly_once(inbox):
    """The real overlap shape: poll N sees rows 0..39, poll N+1 sees 10..49."""
    rows = sorted(_rows(inbox), key=lambda r: r["created_at"])
    a = notify.poll(_envelope(rows[:40]), None)
    b = notify.poll(_envelope(rows[10:]), a.cursor)
    assert [r["event_id"] for r in b.new] == [r["event_id"] for r in rows[40:]]
    seen = [r["event_id"] for r in a.new] + [r["event_id"] for r in b.new]
    assert len(seen) == len(set(seen)) == 50


def test_a_row_reordered_within_the_slop_is_still_new(inbox):
    """Rows are stamped by the producer, so a poll can deliver one slightly
    older than our high-water mark. Inside REORDER_SLOP_S it is an event we have
    never seen; outside it is history the cursor already passed."""
    rows = sorted(_rows(inbox), key=lambda r: r["created_at"])
    late = copy.deepcopy(rows[0])
    late["event_id"] = "0" * 32
    first = notify.poll(_envelope(rows[1:]), None)
    top = first.cursor["last_created_at"]

    late["created_at"] = top - 10.0                     # inside the slop
    assert len(notify.poll(_envelope([late]), first.cursor).new) == 1
    late["created_at"] = top - notify.REORDER_SLOP_S - 60.0    # outside it
    assert notify.poll(_envelope([late]), first.cursor).new == []


def test_a_full_window_of_unknown_rows_is_a_gap(inbox):
    """D4: 50 rows, none of them known, means the storm outran our poll — the
    LABELING has a hole. Reconcile is unaffected."""
    rows = sorted(_rows(inbox), key=lambda r: r["created_at"])
    seeded = notify.poll(_envelope(rows[:5]), None)
    later = copy.deepcopy(rows)
    for i, r in enumerate(later):                       # a wholly unseen window
        r["event_id"] = f"{i:032x}"
        r["created_at"] = seeded.cursor["last_created_at"] + 100.0 + i
    res = notify.poll(_envelope(later), seeded.cursor)
    assert res.gap is True and len(res.new) == 50
    # one known row in the window is enough to prove continuity
    later[0]["event_id"] = rows[0]["event_id"]
    later[0]["created_at"] = rows[0]["created_at"]
    assert notify.poll(_envelope(later), seeded.cursor).gap is False


def test_a_short_window_is_never_a_gap(inbox):
    rows = sorted(_rows(inbox), key=lambda r: r["created_at"])
    seeded = notify.poll(_envelope(rows[:5]), None)
    assert notify.poll(_envelope(rows[5:]), seeded.cursor).gap is False


def test_recent_ids_are_bounded_and_drop_the_oldest(inbox):
    rows = sorted(_rows(inbox), key=lambda r: r["created_at"])
    cursor = None
    for batch in range(6):                              # 6 x 50 = 300 ids > 200
        window = []
        for i, r in enumerate(rows):
            c = copy.deepcopy(r)
            c["event_id"] = f"{batch:02d}{i:030x}"
            c["created_at"] = r["created_at"] + 10000.0 * batch
            window.append(c)
        cursor = notify.poll(_envelope(window), cursor).cursor
    ids = cursor["recent_ids"]
    assert len(ids) == notify.RECENT_IDS_MAX
    # 300 ids seen, the newest 200 kept: batches 0 and 1 rolled off, 2..5 stayed.
    assert {i[:2] for i in ids} == {"02", "03", "04", "05"}
    # and the bound is far wider than a window, so nothing currently in the feed
    # can ever fall out of the id set between polls
    assert notify.RECENT_IDS_MAX >= 4 * notify.WINDOW


def test_a_truncated_cursor_cannot_resurrect_history(inbox):
    """If the id set is lost or rolls, `last_created_at` still bounds the past —
    otherwise a state edit would re-journal three days of rows as 'new'.

    The floor is deliberately soft by REORDER_SLOP_S, so the newest few minutes
    can re-emit; three days of history cannot."""
    first = notify.poll(inbox, None)
    top = first.cursor["last_created_at"]
    amputated = {"last_created_at": top, "recent_ids": []}
    again = notify.poll(inbox, amputated).new
    assert all(top - r["created_at"] <= notify.REORDER_SLOP_S for r in again)
    assert len(again) < 5 and len(_rows(inbox)) == 50


def test_cursor_normalizes_any_persisted_junk():
    for junk in (None, "x", 7, {"last_created_at": "bad"}, {"recent_ids": "abc"},
                 {"recent_ids": [None, "", "a"], "last_created_at": None}):
        cur = notify.normalize_cursor(junk)
        assert set(cur) == {"last_created_at", "recent_ids"}
        assert isinstance(cur["recent_ids"], list)


def test_rows_without_an_event_id_are_dropped_not_looped(inbox):
    """An unjournalable row would otherwise re-emit on every poll, forever."""
    row = copy.deepcopy(_by_type(inbox, "outbid")[0])
    row.pop("event_id")
    res = notify.poll(_envelope([row]), None)
    assert res.new == [] and res.rows_seen == 1


# --------------------------------------------------------------------------- #
# what gets journaled (D4)
# --------------------------------------------------------------------------- #
def test_journal_fields_carry_the_associated_id_verbatim(inbox):
    row = [r for r in _by_type(inbox, "outbid")
           if notify.instance_id(r) == 47840057][0]
    f = notify.journal_fields(row)
    assert f["event_id"] == row["event_id"]
    assert f["notif_type"] == "outbid"
    assert f["created_at"] == row["created_at"]
    assert f["associated_id"] == row["associated_id"]   # VERBATIM, not reshaped
    assert f["iid"] == "47840057" and f["machine_id"] == 56759


def test_journal_fields_survive_an_unknown_type_and_a_missing_associated_id(inbox):
    """Unknown types WILL appear (the catalog has 30 keys; we have seen 3), and
    a future one may carry no structured id at all (§1.7)."""
    row = copy.deepcopy(_rows(inbox)[0])
    row["notif_type"] = "upcoming_downtime"            # real key, never seen live
    row.pop("associated_id")
    row["some_new_field"] = {"nested": [1, 2]}         # extra keys tolerated
    f = notify.journal_fields(row)
    assert f["notif_type"] == "upcoming_downtime"
    assert f["associated_id"] is None and "iid" not in f
    assert notify.poll(_envelope([row]), None).new == [row]


# --------------------------------------------------------------------------- #
# rendering (S1)
# --------------------------------------------------------------------------- #
def test_render_inbox_shows_the_displacement_for_outbids(inbox):
    now = max(r["created_at"] for r in _rows(inbox)) + 60.0
    out = notify.render_inbox(inbox, now)
    assert "inbox: 50 row(s)" in out and "unread=50" in out
    assert "outbid" in out and "47848147" in out and "138918" in out
    assert "bid $0.40/hr -> min $1.27/hr" in out
    assert "BELOW our bid" in out, "the $0.16 -> $0.15 row must not read as normal"
    body = out.splitlines()[2:]
    assert len(body) == 50


def test_render_inbox_limit_and_empty(inbox):
    now = max(r["created_at"] for r in _rows(inbox)) + 60.0
    out = notify.render_inbox(inbox, now, limit=5)
    assert "showing 5" in out and len(out.splitlines()) == 2 + 5
    newest = max(_rows(inbox), key=lambda r: r["created_at"])
    assert str(notify.instance_id(newest)) in out.splitlines()[2]
    empty = notify.render_inbox({"success": True, "notifications": []}, now)
    assert "0 row(s)" in empty and "empty" in empty


def test_render_types_lists_keys_topics_and_default_channels(types):
    out = notify.render_types(types)
    assert "notification types: 30" in out
    assert "client:outbid" in out and "instance" in out
    line = [ln for ln in out.splitlines() if "client:outbid" in ln][0]
    assert "in_app" in line
    assert "webhooks" not in line, ("no webhook channel is enabled on any type "
                                    "today — §1.1")


def test_render_webhooks_says_none_rather_than_erroring(webhooks):
    out = notify.render_webhooks(webhooks)
    assert "0 of 4 slot(s) used" in out
    assert "none" in out and "D5" in out
    # and a populated list renders without knowing vast's field spelling
    out2 = notify.render_webhooks({"webhooks": [
        {"id": 7, "name": "fleetd", "url": "https://example.invalid/hook",
         "event_types": ["client:outbid", "client:instance_stopped"]}]})
    assert "fleetd" in out2 and "client:outbid,client:instance_stopped" in out2


# --------------------------------------------------------------------------- #
# the CLI (S1) — HTTP seam mocked, nothing else
# --------------------------------------------------------------------------- #
def _mock_http(monkeypatch, by_path):
    """by_path: path -> (ok, data, err). Any other path is a test bug."""
    calls = []

    def fake(method, path, body=None, timeout=60, retries=5, _sleep=None):
        calls.append((method, path))
        assert path in by_path, f"unexpected request {method} {path}"
        return by_path[path]

    monkeypatch.setattr(api, "request_soft", fake)
    return calls


def test_cli_inbox_table_and_json(monkeypatch, capsys, inbox):
    calls = _mock_http(monkeypatch, {notify.INBOX_PATH: (True, inbox, None)})
    cli_inbox.run(argparse.Namespace(json=False, limit=3))
    out = capsys.readouterr().out
    assert calls == [("GET", "v0/notifications/inbox/")]
    assert "showing 3" in out and len(out.strip().splitlines()) == 2 + 3

    cli_inbox.run(argparse.Namespace(json=True, limit=3))
    emitted = json.loads(capsys.readouterr().out)
    assert emitted == inbox, "--json must emit the payload VERBATIM"


def test_cli_inbox_404_is_a_one_liner_with_its_own_exit_code(monkeypatch, capsys):
    """The endpoint is hidden and revocable; its disappearance is an expected
    end state, so it degrades to one line and a distinct rc — never a traceback."""
    err = "HTTP 404 on GET v0/notifications/inbox/: {'detail': 'Not Found'}"
    _mock_http(monkeypatch, {notify.INBOX_PATH: (False, None, err)})
    with pytest.raises(SystemExit) as e:
        cli_inbox.run(argparse.Namespace(json=False, limit=0))
    assert e.value.code == _get.NOTIFY_GONE_RC != 0
    cap = capsys.readouterr()
    assert cap.out == ""
    assert "hidden endpoint gone" in cap.err and "NOTIFY_DESIGN.md" in cap.err
    assert "Traceback" not in cap.err


def test_cli_inbox_other_errors_exit_generically(monkeypatch):
    _mock_http(monkeypatch, {notify.INBOX_PATH: (False, None, "HTTP 500 on GET x")})
    with pytest.raises(SystemExit) as e:
        cli_inbox.run(argparse.Namespace(json=False, limit=0))
    assert e.value.code != _get.NOTIFY_GONE_RC
    assert "HTTP 500" in str(e.value.code)


def test_cli_types_and_webhooks(monkeypatch, capsys, types, webhooks):
    _mock_http(monkeypatch, {notify.TYPES_PATH: (True, types, None),
                             notify.WEBHOOKS_PATH: (True, webhooks, None)})
    cli_types.run(argparse.Namespace(json=False))
    assert "client:outbid" in capsys.readouterr().out
    cli_webhooks.run(argparse.Namespace(json=False))
    out = capsys.readouterr().out
    assert "0 of 4 slot(s) used" in out
    cli_webhooks.run(argparse.Namespace(json=True))
    assert json.loads(capsys.readouterr().out) == {"success": True, "webhooks": []}


def test_cli_reads_only(monkeypatch, inbox, types, webhooks):
    """Nothing under `notify` may write — in particular the PUT that would mark
    the feed seen (D3: our cursor is ours, and the console UI may want theirs)."""
    seen = _mock_http(monkeypatch, {notify.INBOX_PATH: (True, inbox, None),
                                    notify.TYPES_PATH: (True, types, None),
                                    notify.WEBHOOKS_PATH: (True, webhooks, None)})
    cli_inbox.run(argparse.Namespace(json=False, limit=0))
    cli_types.run(argparse.Namespace(json=False))
    cli_webhooks.run(argparse.Namespace(json=False))
    assert {m for m, _ in seen} == {"GET"}


def test_notify_help_ends_with_a_docs_list():
    """Same contract as every other subcommand: `-h` names the runbook."""
    import subprocess
    here = pathlib.Path(__file__).resolve().parent
    for argv in (["notify"], ["notify", "inbox"], ["notify", "types"],
                 ["notify", "webhooks"]):
        r = subprocess.run([sys.executable, str(here / "herdd.py"), *argv,
                            "--help"], capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        assert "docs:" in r.stdout, argv
        tail = r.stdout[r.stdout.index("docs:"):].strip().splitlines()
        assert len(tail) >= 2 and all(ln.startswith("  ") for ln in tail[1:])
        assert "NOTIFY_DESIGN.md" in r.stdout
