"""Portable tests for the jobs-v2 dead-letter queue (jobmeta DLQ half).

Toolchain-free lane: no rclone, no B2, no network, no creds — everything runs
against `test_jobmeta.FakeB2`, the same in-memory rclone-shaped runner the
ticket tests use.

What these pin, and why each one is here rather than left to review:

* The DLQ is a MOVE. `job cancel` deletes the queue pointer, and the pointer is
  the ONLY place a job's frozen `config` (the whole resolved env) is recorded —
  the `submitted` event carries bundle_sha256/entrypoint/timeout_s and not the
  env. So a retirement that dropped the body would destroy the only evidence of
  what the job would have run. `test_dead_letter_preserves_the_frozen_config`
  is that property.
* The WRITE-BEFORE-DELETE order. If the DLQ write fails, the queue must be
  untouched: a ticket is never destroyed without its record.
* Retiring a POINTER is not ending a JOB. `dead_letter_ticket` must not emit a
  terminal `cancelled` event, or a job still wanted on another box would fold
  dead forever (`cancelled` is unconditionally sticky).
* Staleness FAILS OPEN on a bad timestamp, because the bound exists to catch
  forgotten backlog and a malformed ts would otherwise refuse exactly the
  recovery most worth doing.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobmeta as jm  # noqa: E402
from test_jobmeta import FakeB2  # noqa: E402


def _cfg(**env):
    return {"version": 1, "name": "p", "entrypoint": "run.sh", "timeout_s": 60,
            "env": dict(env), "results": [], "needs": {"gpu": False,
                                                       "venv": "none"}}


def _ticket(fake, jid="20260819T072006-a5t-cap-p0-rb3-a20a", box="44",
            submitted_ts="20260819T072006000Z", **env):
    tk = jm.make_ticket(jid, "shaXYZ", "cli:h", _cfg(**env), box)
    tk["submitted_ts"] = submitted_ts
    ok, _key, _err = jm.write_ticket(tk, runner=fake, bucket="bkt")
    assert ok
    return tk


# --------------------------------------------------------------------------- #
# age / staleness — the first readers of submitted_ts in the system's history
# --------------------------------------------------------------------------- #
def test_ticket_age_days_reads_submitted_ts():
    tk = {"submitted_ts": "20260819T000000000Z"}
    age = jm.ticket_age_days(tk, now="20260826T000000000Z")
    assert age == pytest.approx(7.0)


def test_ticket_age_days_is_none_when_unparseable():
    assert jm.ticket_age_days({}, now="20260826T000000000Z") is None
    assert jm.ticket_age_days({"submitted_ts": "not-a-ts"},
                              now="20260826T000000000Z") is None


def test_staleness_bound_is_the_revival_gate():
    old = {"submitted_ts": "20260819T000000000Z"}
    fresh = {"submitted_ts": "20260825T120000000Z"}
    now = "20260826T000000000Z"

    stale, why = jm.ticket_staleness(old, now=now)
    assert stale is True
    assert "7.0d" in why and "frozen at submit" in why

    stale, why = jm.ticket_staleness(fresh, now=now)
    assert stale is False
    assert "within" in why


def test_staleness_fails_OPEN_on_an_unparseable_timestamp():
    """A malformed ts must not become a refusal — see the module docstring."""
    stale, why = jm.ticket_staleness({"submitted_ts": "garbage"},
                                     now="20260826T000000000Z")
    assert stale is False
    assert "age unknown" in why


def test_staleness_bound_is_caller_overridable():
    tk = {"submitted_ts": "20260825T000000000Z"}     # 1 day old
    now = "20260826T000000000Z"
    assert jm.ticket_staleness(tk, now=now)[0] is False
    assert jm.ticket_staleness(tk, now=now, max_age_days=0.5)[0] is True


# --------------------------------------------------------------------------- #
# the retirement itself
# --------------------------------------------------------------------------- #
def test_dead_letter_preserves_the_frozen_config():
    fake = FakeB2()
    _ticket(fake, FLA_REQUIRED="0", MAX_SEQ="20480")

    res = jm.dead_letter_ticket(
        "44", "20260819T072006-a5t-cap-p0-rb3-a20a",
        reason="a5t-cap sweep already completed 20/20 arms elsewhere",
        actor="cli:test", verdict="ORPHAN_UNCLAIMED",
        now="20260826T000000000Z", runner=fake, bucket="bkt")
    assert res["status"] == "dead_lettered"
    assert res["ticket_deleted"] is True

    # the queue pointer is gone ...
    assert jm.list_queue("44", runner=fake, bucket="bkt") == []
    # ... and the frozen env survives, verbatim, in the DLQ
    entry = jm.read_dlq_entry("44", "20260819T072006-a5t-cap-p0-rb3-a20a",
                              runner=fake, bucket="bkt")
    assert entry["config"]["env"] == {"FLA_REQUIRED": "0", "MAX_SEQ": "20480"}
    assert entry["bundle_sha256"] == "shaXYZ"

    dl = entry[jm.DEAD_LETTER_MARK]
    assert dl["verdict"] == "ORPHAN_UNCLAIMED"
    assert dl["actor"] == "cli:test"
    assert "20/20 arms" in dl["reason"]
    assert dl["source_key"] == ("jobs/queue/44/"
                                "20260819T072006-a5t-cap-p0-rb3-a20a.json")
    # submitted 20260819T072006Z, retired 20260826T000000Z -> 6.69 d, not 7:
    # the age is real elapsed time, not a date subtraction.
    assert dl["age_days_at_retirement"] == pytest.approx(6.694, abs=0.01)


def test_dead_letter_does_NOT_delete_the_ticket_when_the_dlq_write_fails():
    """Write-before-delete: a failed retirement leaves the queue intact."""
    fake = FakeB2()
    _ticket(fake)

    real = fake.__call__

    def refuse_dlq_write(args, input=None):
        if args[0] == "rcat" and jm.DLQ_PREFIX in args[1]:
            return 1, "", "simulated B2 failure"
        return real(args, input=input)

    res = jm.dead_letter_ticket(
        "44", "20260819T072006-a5t-cap-p0-rb3-a20a", reason="x",
        runner=refuse_dlq_write, bucket="bkt")

    assert res["status"] == "dlq_failed"
    assert "simulated B2 failure" in res["err"]
    # the ticket is STILL queued — nothing was destroyed
    assert jm.list_queue("44", runner=fake, bucket="bkt") == [
        "20260819T072006-a5t-cap-p0-rb3-a20a"]


def test_dead_letter_emits_no_terminal_event():
    """Retiring a POINTER is not ending a JOB. A `cancelled` event is sticky
    unconditionally in the fold, so emitting one here would kill a job that is
    still wanted on another box."""
    fake = FakeB2()
    _ticket(fake)
    jm.dead_letter_ticket("44", "20260819T072006-a5t-cap-p0-rb3-a20a",
                          reason="x", runner=fake, bucket="bkt")
    assert jm.has_events("20260819T072006-a5t-cap-p0-rb3-a20a",
                         runner=fake, bucket="bkt") is False


def test_dead_letter_of_a_missing_ticket_is_not_an_error():
    fake = FakeB2()
    res = jm.dead_letter_ticket("44", "nope", reason="x", runner=fake,
                                bucket="bkt")
    assert res["status"] == "no_ticket"


def test_dead_letter_is_idempotent():
    fake = FakeB2()
    jid = "20260819T072006-a5t-cap-p0-rb3-a20a"
    _ticket(fake)
    assert jm.dead_letter_ticket("44", jid, reason="x", runner=fake,
                                 bucket="bkt")["status"] == "dead_lettered"
    again = jm.dead_letter_ticket("44", jid, reason="x", runner=fake,
                                  bucket="bkt")
    assert again["status"] == "already_dead_lettered"


# --------------------------------------------------------------------------- #
# listing + the deliberate way back
# --------------------------------------------------------------------------- #
def test_list_dlq_scopes_by_box_and_globally():
    fake = FakeB2()
    _ticket(fake, jid="20260819T072006-a-1", box="44")
    _ticket(fake, jid="20260819T072007-a-2", box="44")
    _ticket(fake, jid="20260819T072008-b-3", box="55")
    for box, jid in (("44", "20260819T072006-a-1"),
                     ("44", "20260819T072007-a-2"),
                     ("55", "20260819T072008-b-3")):
        jm.dead_letter_ticket(box, jid, reason="x", runner=fake, bucket="bkt")

    assert jm.list_dlq("44", runner=fake, bucket="bkt") == [
        ("44", "20260819T072006-a-1"), ("44", "20260819T072007-a-2")]
    assert jm.list_dlq(runner=fake, bucket="bkt") == [
        ("44", "20260819T072006-a-1"), ("44", "20260819T072007-a-2"),
        ("55", "20260819T072008-b-3")]


def test_list_dlq_raises_rather_than_reading_a_broken_bucket_as_empty():
    """Same rule as list_queue: an unreadable listing is never an empty DLQ."""
    def broken(args, input=None):
        return 1, "", "revoked key"

    with pytest.raises(jm.QueueUnreadable):
        jm.list_dlq(runner=broken, bucket="bkt")


def test_restore_puts_the_ticket_back_on_a_live_box():
    fake = FakeB2()
    jid = "20260819T072006-a5t-cap-p0-rb3-a20a"
    _ticket(fake, FLA_REQUIRED="0")
    jm.dead_letter_ticket("44", jid, reason="x", runner=fake, bucket="bkt")

    res = jm.restore_dlq_entry(jid, "44", "99", actor="cli:test",
                               runner=fake, bucket="bkt")
    assert res["status"] == "restored"
    assert res["box"] == "99"
    assert res["dlq_entry_deleted"] is True

    # back on the LIVE box's queue, config intact, provenance stamped
    assert jm.list_queue("99", runner=fake, bucket="bkt") == [jid]
    tk = jm.read_ticket("99", jid, runner=fake, bucket="bkt")
    assert tk["config"]["env"] == {"FLA_REQUIRED": "0"}
    assert tk["retargeted_from"] == "44"
    assert jm.DEAD_LETTER_MARK not in tk
    # and the DLQ no longer holds it
    assert jm.list_dlq(runner=fake, bucket="bkt") == []


def test_restore_of_an_absent_entry_is_a_clean_no_op():
    fake = FakeB2()
    assert jm.restore_dlq_entry("nope", "44", "99", runner=fake,
                                bucket="bkt")["status"] == "no_entry"


def test_dlq_key_mirrors_the_queue_layout():
    assert jm.dlq_key("44", "j1") == "jobs/dlq/44/j1.json"


# --------------------------------------------------------------------------- #
# the bulk-move filter — fleetd's eviction / pull-condemn replacement
# --------------------------------------------------------------------------- #
NOW = "20260826T000000000Z"
_OLD = {"submitted_ts": "20260819T072006000Z"}      # 6.7 d
_FRESH = {"submitted_ts": "20260825T200000000Z"}    # 0.2 d


@pytest.mark.parametrize("status", ["done", "failed", "cancelled"])
def test_bulk_move_refuses_terminal_jobs(status):
    """A finished job needs no successor. Until 2026-08-26 it was moved anyway
    and skipped box-side by results.DONE.json — a marker a pre-entrypoint
    failure never writes, which is how a `failed` ticket could re-run."""
    move, why = jm.bulk_move_verdict(_FRESH, status, now=NOW)
    assert move is False
    assert f"already {status}" in why


def test_bulk_move_refuses_a_stale_ticket():
    move, why = jm.bulk_move_verdict(_OLD, "submitted", now=NOW)
    assert move is False
    assert "6.7d old" in why and "not today's bundle" in why


def test_bulk_move_carries_live_work():
    for status in ("submitted", "claimed", "started"):
        move, why = jm.bulk_move_verdict(_FRESH, status, now=NOW)
        assert move is True, status
        assert status in why


def test_bulk_move_fails_OPEN_on_unknown_status():
    """An unreadable event log must never silently abandon live work."""
    move, _why = jm.bulk_move_verdict(_FRESH, "unknown", now=NOW)
    assert move is True
    move, _why = jm.bulk_move_verdict(_FRESH, None, now=NOW)
    assert move is True


def test_bulk_move_terminal_check_precedes_the_age_check():
    """Both refuse, but the REASON matters: a reader must not be told a
    finished job was dropped for being old."""
    _move, why = jm.bulk_move_verdict(_OLD, "done", now=NOW)
    assert "already done" in why
