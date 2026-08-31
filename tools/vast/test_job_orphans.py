"""Portable tests for orphaned queue tickets — a jobs-v2 ticket whose target box
was destroyed without an orderly handoff (JOBS_DESIGN "Orphans").

Two layers, mirroring the house split:
  * the PURE verdict (`parked_lifecycle.ticket_orphan_verdict`) — the truth
    table, including the three-valued presence rule that keeps a soft vast API
    failure from classifying the whole fleet's queue as orphaned;
  * the herdd surfaces — `_present_iids_set`'s None, `job ls`'s GONE/ORPHAN
    render, and `job orphans` (report exit codes, --resolve gating, the recorded
    reason, and the guarantee that a LIVE box's tickets are never written to).

Toolchain-free lane (`pytest -m "not integration"`): NO vast API, NO B2/rclone,
NO network — every seam is monkeypatched.
"""
import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobmeta  # noqa: E402
import parked_lifecycle as pl  # noqa: E402
from vastlib.boxes import lifecycle  # noqa: E402
from vastlib.core import api  # noqa: E402
from vastlib.jobs import control, runlocal, scan, view  # noqa: E402
from vastlib.storage import b2  # noqa: E402


# --- pure verdict ----------------------------------------------------------- #
@pytest.mark.parametrize("status", ["submitted", "unknown", ""])
def test_unclaimed_on_a_destroyed_box_is_the_stuck_orphan(status):
    verdict, why = pl.ticket_orphan_verdict(box_present=False, job_status=status)
    assert verdict == pl.TICKET_ORPHAN_UNCLAIMED
    assert "never claimed" in why
    assert verdict in pl.TICKET_ORPHANS_STUCK


@pytest.mark.parametrize("status", ["claimed", "started"])
def test_claimed_on_a_destroyed_box_is_interrupted_not_unclaimed(status):
    """Graded on purpose: this one may have checkpoints, so the remedy is
    retarget/requeue, and cancel must be opt-in."""
    verdict, why = pl.ticket_orphan_verdict(box_present=False, job_status=status)
    assert verdict == pl.TICKET_ORPHAN_INTERRUPTED
    assert "retarget" in why
    assert verdict in pl.TICKET_ORPHANS_STUCK


@pytest.mark.parametrize("status", ["done", "failed", "cancelled"])
def test_terminal_on_a_destroyed_box_is_a_stale_pointer_not_stuck(status):
    verdict, _ = pl.ticket_orphan_verdict(box_present=False, job_status=status)
    assert verdict == pl.TICKET_ORPHAN_TERMINAL
    assert verdict in pl.TICKET_ORPHANS       # reported…
    assert verdict not in pl.TICKET_ORPHANS_STUCK   # …never swept


@pytest.mark.parametrize("status", ["submitted", "started", "failed"])
def test_a_present_box_is_never_an_orphan_however_dead_the_job_looks(status):
    """PARKED counts as present. The whole safety of the lane rests on this: a
    stopped box is absent from the LIVE set and its tickets are healthy."""
    assert pl.ticket_orphan_verdict(
        box_present=True, job_status=status)[0] == pl.TICKET_OK


@pytest.mark.parametrize("status", ["submitted", "started", "failed"])
def test_unreadable_listing_never_mints_an_orphan(status):
    verdict, why = pl.ticket_orphan_verdict(box_present=None, job_status=status)
    assert verdict == pl.TICKET_UNKNOWN
    assert verdict not in pl.TICKET_ORPHANS
    assert "unreadable" in why


# --- presence helper -------------------------------------------------------- #
def test_present_iids_set_includes_stopped_boxes_and_stringifies(monkeypatch):
    monkeypatch.setattr(runlocal, "_JOB_LOCAL", False)
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (True, {"instances": [
                            {"id": 1, "actual_status": "running"},
                            {"id": 2, "actual_status": "stopped"}]}, None))
    assert view._present_iids_set() == {"1", "2"}


def test_present_iids_set_is_none_when_the_api_fails(monkeypatch):
    """The distinction _instances_soft() cannot make: an API error and an empty
    account both come back as []. Reading the first as "all destroyed" would
    orphan the entire queue at once."""
    monkeypatch.setattr(runlocal, "_JOB_LOCAL", False)
    monkeypatch.setattr(api, "request_soft", lambda *a, **k: (False, None, "boom"))
    assert view._present_iids_set() is None
    monkeypatch.setattr(api, "request_soft", lambda *a, **k: (True, {"instances": []}, None))
    assert view._present_iids_set() == set()          # genuinely empty account


def test_live_iids_set_is_strings(monkeypatch):
    """Regression: the vast API types an id as int, every jobs-lane spelling is a
    str, and the direct membership tests (`job ls`'s live=, `job cancel --hard`)
    therefore always said False."""
    monkeypatch.setattr(runlocal, "_JOB_LOCAL", False)
    monkeypatch.setattr(lifecycle, "_instances_soft",
                        lambda: [{"id": 46648873, "actual_status": "running"},
                                 {"id": 5, "actual_status": "stopped"}])
    assert view._live_iids_set() == {"46648873"}


# --- wiring for the CLI lanes ----------------------------------------------- #
def _wire(monkeypatch, *, queue, folds, present, live=None):
    """queue: [(box, job_id)] · folds: {job_id: status} · present: set|None.
    `live` defaults to "every present box is also running"."""
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda *a, **k: None)
    monkeypatch.setattr(view, "_present_iids_set", lambda: present)
    monkeypatch.setattr(view, "_live_iids_set",
                        lambda: set(present or ()) if live is None else set(live))
    monkeypatch.setattr(jobmeta, "list_all_queued", lambda **k: list(queue))
    monkeypatch.setattr(jobmeta, "list_queue",
                        lambda box, **k: [j for b, j in queue if b == str(box)])
    monkeypatch.setattr(jobmeta, "read_box", lambda box, **k: {})

    def _one(jid):
        st = folds[jid]
        disp = {"submitted": "queued", "claimed": "interrupted",
                "started": "interrupted"}.get(st, st)
        return {"status": st, "display_status": disp, "n_events": 1,
                "done_marker": False}

    # `job orphans` and `job ls` fold the WHOLE queue through one bulk call
    # (`vastlib.jobs.scan.fold_many`, added 2026-08-17) instead of a
    # `jobmeta.read_job` per ticket — 275 tickets cost 275 rclone subprocesses
    # and 139 s the old way. `read_job` is still patched below because the
    # single-job surfaces (`job status`, `_job_view`) go on using it.
    monkeypatch.setattr(scan, "fold_many",
                        lambda jids, **k: {j: _one(j) for j in jids})
    monkeypatch.setattr(jobmeta, "read_job", lambda jid, **k: _one(jid))

    writes = {"cancel": [], "events": [], "deleted": []}
    monkeypatch.setattr(jobmeta, "write_cancel_marker",
                        lambda jid, **k: (writes["cancel"].append((jid, k)), (True, ""))[1])
    monkeypatch.setattr(jobmeta, "emit_event",
                        lambda jid, ev, **k: writes["events"].append((jid, ev, k)))
    monkeypatch.setattr(jobmeta, "delete_ticket",
                        lambda box, jid, **k: (writes["deleted"].append((box, jid)),
                                               (True, ""))[1])
    monkeypatch.setattr(lifecycle, "_cli_actor", lambda: "cli:test")
    return writes


def _args(**kw):
    base = dict(box=None, job=None, all=False, json=False, resolve=False,
                reason=None, include_interrupted=False, dry_run=False, yes=False)
    base.update(kw)
    return argparse.Namespace(**base)


# the 2026-08-02 shape: a destroyed box (46590907) holding two never-claimed
# tickets, and a LIVE box (46648873) running one job and holding one done.
QUEUE = [("46590907", "j-full"), ("46590907", "j-format"),
         ("46648873", "j-live"), ("46648873", "j-done"),
         ("46636056", "j-failed")]
FOLDS = {"j-full": "submitted", "j-format": "submitted",
         "j-live": "started", "j-done": "done", "j-failed": "failed"}
PRESENT = {"46648873"}


# --- job ls ----------------------------------------------------------------- #
def test_job_ls_marks_the_gone_box_and_its_orphans(monkeypatch, capsys):
    _wire(monkeypatch, queue=QUEUE, folds=FOLDS, present=PRESENT)
    view.cmd_job_ls(argparse.Namespace(box=None))
    out = capsys.readouterr().out
    assert "== box 46590907: GONE" in out
    assert out.count("!! ORPHAN") == 2
    assert "2 ORPHANED ticket(s)" in out
    # the live box is untouched by the new render
    assert "== box 46648873: live=True ==" in out
    # a terminal job on a dead box is litter, not an orphan
    assert "(stale pointer)" in out


def test_job_ls_says_so_when_it_cannot_classify(monkeypatch, capsys):
    _wire(monkeypatch, queue=QUEUE, folds=FOLDS, present=None)
    view.cmd_job_ls(argparse.Namespace(box=None))
    out = capsys.readouterr().out
    assert "orphan detection skipped" in out
    assert "ORPHAN" not in out.replace("orphan detection skipped", "")
    assert "GONE" not in out


# --- job orphans: report ---------------------------------------------------- #
def test_orphans_report_exits_2_and_names_the_stuck_ones(monkeypatch, capsys):
    _wire(monkeypatch, queue=QUEUE, folds=FOLDS, present=PRESENT)
    with pytest.raises(SystemExit) as e:
        control.cmd_job_orphans(_args())
    assert e.value.code == 2
    out = capsys.readouterr().out
    assert "2 STUCK orphan(s)" in out
    assert "j-full" in out and "j-format" in out
    assert "stale pointer" in out              # j-failed, reported not swept
    assert "j-live" not in out                 # healthy: hidden without --all


def test_orphans_report_is_quiet_and_exits_0_when_clean(monkeypatch, capsys):
    _wire(monkeypatch, queue=[("46648873", "j-live")], folds={"j-live": "started"},
          present=PRESENT)
    control.cmd_job_orphans(_args())
    assert "no stuck orphans" in capsys.readouterr().out


def test_orphans_report_exits_1_on_an_unreadable_listing(monkeypatch, capsys):
    _wire(monkeypatch, queue=QUEUE, folds=FOLDS, present=None)
    with pytest.raises(SystemExit) as e:
        control.cmd_job_orphans(_args())
    assert e.value.code == 1
    assert "cannot tell a destroyed box from a parked one" in capsys.readouterr().out


# --- job orphans: resolve --------------------------------------------------- #
def test_resolve_requires_a_reason(monkeypatch, capsys):
    w = _wire(monkeypatch, queue=QUEUE, folds=FOLDS, present=PRESENT)
    with pytest.raises(SystemExit) as e:
        control.cmd_job_orphans(_args(resolve=True, yes=True))
    assert "requires --reason" in str(e.value.code)
    assert w["events"] == []


def test_resolve_requires_yes(monkeypatch, capsys):
    w = _wire(monkeypatch, queue=QUEUE, folds=FOLDS, present=PRESENT)
    with pytest.raises(SystemExit) as e:
        control.cmd_job_orphans(_args(resolve=True, reason="superseded"))
    assert "re-run with -y" in str(e.value.code)
    assert w["events"] == [] and w["deleted"] == []


def test_resolve_dry_run_writes_nothing(monkeypatch, capsys):
    w = _wire(monkeypatch, queue=QUEUE, folds=FOLDS, present=PRESENT)
    control.cmd_job_orphans(_args(resolve=True, reason="superseded", dry_run=True))
    out = capsys.readouterr().out
    assert out.count("[dry-run] would resolve") == 2
    assert w["events"] == [] and w["deleted"] == [] and w["cancel"] == []


def test_resolve_cancels_only_the_unclaimed_orphans_and_records_why(
        monkeypatch, capsys):
    w = _wire(monkeypatch, queue=QUEUE, folds=FOLDS, present=PRESENT)
    control.cmd_job_orphans(_args(resolve=True, yes=True,
                            reason="superseded by the resubmitted arms"))
    jids = sorted(j for j, _, _ in w["events"])
    assert jids == ["j-format", "j-full"]
    assert sorted(w["deleted"]) == [("46590907", "j-format"), ("46590907", "j-full")]
    for jid, ev, fields in w["events"]:
        assert ev == "cancelled"                     # frozen vocabulary, no new kind
        assert fields["orphan"] == pl.TICKET_ORPHAN_UNCLAIMED
        assert fields["orphan_box"] == "46590907"
        # the reason carries BOTH the machine-checked evidence and the note
        assert "target box 46590907 no longer exists" in fields["reason"]
        assert "superseded by the resubmitted arms" in fields["reason"]


def test_resolve_never_touches_a_live_boxs_tickets(monkeypatch, capsys):
    """The live-job guarantee: `j-live` is `started` on a box that EXISTS, so it
    is OK, never selected, and no write names it."""
    w = _wire(monkeypatch, queue=QUEUE, folds=FOLDS, present=PRESENT)
    control.cmd_job_orphans(_args(resolve=True, yes=True, reason="x",
                            include_interrupted=True))
    touched = {j for j, _, _ in w["events"]} | {j for _, j in w["deleted"]}
    assert "j-live" not in touched and "j-done" not in touched
    assert "j-failed" not in touched              # terminal orphan is not swept


def test_interrupted_orphans_are_skipped_by_default_with_the_retarget_hint(
        monkeypatch, capsys):
    q = [("46590907", "j-mid")]
    w = _wire(monkeypatch, queue=q, folds={"j-mid": "started"}, present=PRESENT)
    control.cmd_job_orphans(_args(resolve=True, yes=True, reason="x"))
    out = capsys.readouterr().out
    assert "SKIP j-mid" in out and "job retarget j-mid" in out
    assert w["events"] == []


def test_interrupted_orphans_are_cancelled_with_the_opt_in(monkeypatch, capsys):
    q = [("46590907", "j-mid")]
    w = _wire(monkeypatch, queue=q, folds={"j-mid": "started"}, present=PRESENT)
    control.cmd_job_orphans(_args(resolve=True, yes=True, reason="x",
                            include_interrupted=True))
    assert [j for j, _, _ in w["events"]] == ["j-mid"]
    assert w["events"][0][2]["orphan"] == pl.TICKET_ORPHAN_INTERRUPTED


def test_job_filter_narrows_the_resolve_to_one_ticket(monkeypatch, capsys):
    w = _wire(monkeypatch, queue=QUEUE, folds=FOLDS, present=PRESENT)
    control.cmd_job_orphans(_args(resolve=True, yes=True, reason="x", job="j-full"))
    assert [j for j, _, _ in w["events"]] == ["j-full"]


# --- the shared cancel core -------------------------------------------------- #
def test_job_cancel_writes_is_the_one_copy(monkeypatch):
    """`job cancel` and `job orphans --resolve` must not drift: both go through
    _job_cancel_writes, in marker -> event -> ticket-delete order."""
    w = _wire(monkeypatch, queue=[], folds={}, present=set())
    warn = control._job_cancel_writes("j-x", "77", reason="because", actor="cli:test",
                                orphan="ORPHAN_UNCLAIMED")
    assert warn == []
    assert w["cancel"][0][0] == "j-x"
    assert w["events"][0][1] == "cancelled"
    assert w["events"][0][2]["reason"] == "because"
    assert w["events"][0][2]["orphan"] == "ORPHAN_UNCLAIMED"
    assert w["deleted"] == [("77", "j-x")]


def test_job_cancel_writes_reports_failures_without_raising(monkeypatch):
    w = _wire(monkeypatch, queue=[], folds={}, present=set())
    monkeypatch.setattr(jobmeta, "write_cancel_marker",
                        lambda jid, **k: (False, "403"))
    monkeypatch.setattr(jobmeta, "delete_ticket",
                        lambda box, jid, **k: (False, "nope"))
    warn = control._job_cancel_writes("j-x", "77", reason="r", actor="cli:test")
    assert len(warn) == 2 and "CANCEL marker write failed" in warn[0]
    assert "ticket delete failed" in warn[1]
    assert w["events"][0][1] == "cancelled"    # the terminal event still lands
