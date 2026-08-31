"""`vastlib.jobs.control` — the ported job WRITE layer, held to its traps.

Why this file exists
--------------------
Every function under test here mutates B2, and each mutation can double-run a
training job or end one that was still working. Four properties survive the move
only if something checks them, and the existing flat tests could not check the
ported copy while it was being written (they drove `herdd`'s copy, which
stayed live through the add-only phase and was re-run UNEDITED as this port's
gate; at plan §8 step 6d that copy is gone and they reach this module):

1. **`cmd_job_orphans`' exit codes** — 1 = the instance listing was UNREADABLE
   (no verdict minted), 2 = stuck orphans found without `--resolve`, 0 = clean.
   A frozen shell contract with ZERO coverage anywhere in the tree until now,
   and the 1 is not an error: it is `view._present_iids_set`'s tri-state
   reaching the shell.
2. **`_job_cancel_writes`' ORDER** — CANCEL marker, then the terminal
   `cancelled` event, then the ticket delete. Each step is independently correct
   only in that order, and it is ONE copy shared by `job cancel` and
   `job orphans --resolve`.
3. **`_job_cancel_kill_script` never puts the JOB_ID on the remote cmdline.**
   A bare `pkill -f <jid>` over `ssh --exec` matches its own wrapper and can
   kill the session (the pkill-self-match footgun — same class as this box's
   standing wait-loop rule). The script ships base64'd as a FILE; nothing
   asserted that before.
4. **The reads are called MODULE-ATTRIBUTE-style on `jobs.view`** (plan §8b), so
   a patch on `view._live_iids_set` steers these commands. A `from .view import
   _live_iids_set` would make the fleetd/handoff patch sites vacuous.

What is deliberately NOT here
-----------------------------
* No repoint of `test_job_retarget.py` / `test_job_requeue.py` /
  `test_job_orphans.py`. They own the 19-case retarget ladder and the eight
  requeue refusals against `herdd`'s live copy; duplicating them here would
  fork the expectations. This file covers what they do NOT: the exit codes, the
  write ORDER, the kill script, and the module-attribute seam.
* No network, no ssh, no rclone. `subprocess.run`, `b2._ensure_b2_remote` and
  every `jobmeta` mutation are stubbed as module attributes.

Provenance: created 2026-08-16 alongside `vastlib/jobs/control.py`, plan §8
step 5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

import jobmeta                                         # noqa: E402  Zone S
import parked_lifecycle as pl                          # noqa: E402  absorbed sibling

from vastlib.boxes import lifecycle, ssh               # noqa: E402
from vastlib.jobs import control, scan, view                 # noqa: E402
from vastlib.storage import b2                         # noqa: E402


def _ns(**kw):
    return argparse.Namespace(**kw)


@pytest.fixture(autouse=True)
def _no_b2_no_actor(monkeypatch):
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(lifecycle, "_cli_actor", lambda: "tester@box")


# --------------------------------------------------------------------------- #
# 1. _job_cancel_writes — THE ORDER
# --------------------------------------------------------------------------- #
def _wire_writes(monkeypatch, *, marker_ok=True, delete_ok=True):
    log = []

    def _marker(jid, actor=None, reason=None):
        log.append(("marker", jid, reason))
        return (marker_ok, None if marker_ok else "rclone rcat failed")

    def _emit(jid, kind, **kw):
        log.append(("event", jid, kind, kw))

    def _del(box, jid):
        log.append(("delete", box, jid))
        return (delete_ok, None if delete_ok else "rclone deletefile failed")

    monkeypatch.setattr(jobmeta, "write_cancel_marker", _marker)
    monkeypatch.setattr(jobmeta, "emit_event", _emit)
    monkeypatch.setattr(jobmeta, "delete_ticket", _del)
    return log


def test_cancel_writes_go_marker_then_event_then_delete(monkeypatch):
    """The marker must be visible to a RUNNING jobd before the ticket vanishes —
    the ticket delete cannot stop a job that is already running."""
    log = _wire_writes(monkeypatch)
    assert control._job_cancel_writes("j-1", "41", reason="why",
                                      actor="tester@box") == []
    assert [row[0] for row in log] == ["marker", "event", "delete"]


def test_cancel_writes_skip_the_delete_when_no_box_is_known(monkeypatch):
    log = _wire_writes(monkeypatch)
    control._job_cancel_writes("j-1", None, reason="why", actor="tester@box")
    assert [row[0] for row in log] == ["marker", "event"]
    assert log[1][3]["box"] is None


def test_cancel_writes_carry_extra_onto_the_event(monkeypatch):
    """`orphan=` / `orphan_box=` ride onto the event because the fold tolerates
    unknown keys by contract — that is how the log distinguishes an operator
    kill from a swept corpse."""
    log = _wire_writes(monkeypatch)
    control._job_cancel_writes("j-1", "41", reason="why", actor="a",
                               orphan=pl.TICKET_ORPHAN_UNCLAIMED, orphan_box="41")
    kw = log[1][3]
    assert kw["orphan"] == pl.TICKET_ORPHAN_UNCLAIMED and kw["orphan_box"] == "41"


def test_cancel_writes_warn_but_never_raise_on_a_failed_write(monkeypatch):
    """The docstring's promise: returns warning lines for the caller; raises
    nothing. A failed marker write must not abort the event + the delete."""
    _wire_writes(monkeypatch, marker_ok=False, delete_ok=False)
    warns = control._job_cancel_writes("j-1", "41", reason="why", actor="a")
    assert len(warns) == 2
    assert any("CANCEL marker write failed" in w for w in warns)
    assert any("ticket delete failed" in w for w in warns)


# --------------------------------------------------------------------------- #
# 2. The remote kill script — the pkill-self-match footgun
# --------------------------------------------------------------------------- #
def test_the_kill_script_quotes_the_job_id_and_walks_the_pid_tree():
    src = control._job_cancel_kill_script("20260806T082213-v11-aff8")
    assert src.startswith("#!/usr/bin/env bash\n")
    assert "JID='20260806T082213-v11-aff8'" in src or \
           "JID=20260806T082213-v11-aff8" in src
    assert "/workspace/jobs/$JID/.running" in src
    assert "kt(){" in src                       # recursive pid-tree walk
    assert 'pkill -KILL -f "/workspace/jobs/$JID/"' in src


def test_the_kill_script_shell_quotes_a_hostile_job_id():
    src = control._job_cancel_kill_script("j; rm -rf /")
    assert "rm -rf /'" in src and "JID='j; rm -rf /'" in src


def test_the_remote_cmdline_never_contains_the_job_id(monkeypatch):
    """THE FOOTGUN. A bare `pkill -f <jid>` over ssh matches its OWN wrapper
    cmdline and can kill the session, so the script ships as a base64 blob and
    runs from a file. The JOB_ID must appear nowhere in the argv."""
    jid = "20260806T082213-v11-aff8"
    seen = {}

    class _R:
        returncode = 0
        stdout = "killed"
        stderr = ""

    def _run(argv, **kw):
        seen["argv"] = argv
        return _R()

    monkeypatch.setattr(lifecycle, "_get_instance", lambda iid: {"id": 41})
    monkeypatch.setattr(ssh, "_pick_ssh_endpoint", lambda i, **k: ("h", 22, None))
    monkeypatch.setattr(control.subprocess, "run", _run)
    control._ssh_kill_job("41", jid)
    joined = " ".join(seen["argv"])
    assert jid not in joined
    assert "base64 -d > /tmp/jobcancel_kill.sh" in joined


def test_ssh_kill_catches_the_systemexit_a_missing_instance_raises(monkeypatch,
                                                                   capsys):
    """`_get_instance` sys.exits. The B2-side cancel already made the job
    terminal, so this is a print-and-return, never a failure."""
    def _boom(iid):
        raise SystemExit("error: no such instance")

    monkeypatch.setattr(lifecycle, "_get_instance", _boom)
    control._ssh_kill_job("41", "j-1")
    assert "instance 41 not found" in capsys.readouterr().out


def test_ssh_kill_skips_a_box_with_no_endpoint(monkeypatch, capsys):
    monkeypatch.setattr(lifecycle, "_get_instance",
                        lambda iid: {"id": 41, "actual_status": "loading"})
    monkeypatch.setattr(ssh, "_pick_ssh_endpoint", lambda i, **k: (None, None, None))
    monkeypatch.setattr(control.subprocess, "run",
                        lambda *a, **k: pytest.fail("no endpoint => no ssh"))
    control._ssh_kill_job("41", "j-1")
    assert "no ssh endpoint" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 3. cmd_job_cancel
# --------------------------------------------------------------------------- #
def _cancel_args(**kw):
    base = dict(job_id="j-1", box=None, reason=None, dry_run=False, hard=False)
    base.update(kw)
    return _ns(**base)


def _pin_fold(monkeypatch, v):
    monkeypatch.setattr(jobmeta, "validate_job_id", lambda jid: None)
    monkeypatch.setattr(view, "_live_iids_set", lambda: {"41"})
    monkeypatch.setattr(jobmeta, "read_job", lambda jid, live_iids=None: dict(v))
    monkeypatch.setattr(jobmeta, "list_all_queued", lambda: [])


def test_cancel_is_an_idempotent_noop_on_an_already_terminal_job(monkeypatch,
                                                                 capsys):
    _pin_fold(monkeypatch, {"status": "done", "display_status": "done",
                            "n_events": 4, "instance_id": "41"})
    _wire_writes(monkeypatch)
    monkeypatch.setattr(jobmeta, "write_cancel_marker",
                        lambda *a, **k: pytest.fail("terminal => no writes"))
    control.cmd_job_cancel(_cancel_args())
    assert "already terminal (done)" in capsys.readouterr().out


def test_cancel_refuses_an_unknown_job(monkeypatch):
    _pin_fold(monkeypatch, {"status": "submitted", "display_status": "queued",
                            "n_events": 0, "instance_id": None,
                            "target_box": None})
    with pytest.raises(SystemExit) as ei:
        control.cmd_job_cancel(_cancel_args())
    assert "unknown job j-1" in str(ei.value.code)


def test_cancel_dry_run_enumerates_all_four_effects(monkeypatch, capsys):
    _pin_fold(monkeypatch, {"status": "started", "display_status": "running",
                            "n_events": 4, "instance_id": "41"})
    monkeypatch.setattr(jobmeta, "write_cancel_marker",
                        lambda *a, **k: pytest.fail("--dry-run writes nothing"))
    control.cmd_job_cancel(_cancel_args(dry_run=True, hard=True))
    out = capsys.readouterr().out
    for expect in ("write marker jobs/j-1/CANCEL", "emit terminal `cancelled`",
                   "delete ticket jobs/queue/41/j-1.json",
                   "--hard: ssh 41"):
        assert expect in out


def test_hard_cancel_only_ssh_es_a_box_in_the_LIVE_set(monkeypatch, capsys):
    """The str/int membership bug lived exactly here: `box` is a string and the
    API's ids are ints, so a naive check was ALWAYS False."""
    _pin_fold(monkeypatch, {"status": "started", "display_status": "running",
                            "n_events": 4, "instance_id": "41"})
    _wire_writes(monkeypatch)
    killed = []
    monkeypatch.setattr(control, "_ssh_kill_job", lambda iid, jid: killed.append(iid))
    control.cmd_job_cancel(_cancel_args(hard=True))
    assert killed == ["41"]

    killed.clear()
    monkeypatch.setattr(view, "_live_iids_set", lambda: {"99"})
    control.cmd_job_cancel(_cancel_args(hard=True))
    assert killed == []
    assert "box 41 is not live" in capsys.readouterr().out


def test_cancel_scans_the_queue_when_the_fold_names_no_box(monkeypatch):
    _pin_fold(monkeypatch, {"status": "submitted", "display_status": "queued",
                            "n_events": 2, "instance_id": None,
                            "target_box": None})
    monkeypatch.setattr(jobmeta, "list_all_queued", lambda: [("77", "other"),
                                                             ("88", "j-1")])
    log = _wire_writes(monkeypatch)
    control.cmd_job_cancel(_cancel_args())
    assert ("delete", "88", "j-1") in log


# --------------------------------------------------------------------------- #
# 3b. cmd_job_flush — cancel's inverse: one marker, no state change
# --------------------------------------------------------------------------- #
def _flush_args(**kw):
    base = dict(job_id="j-1", reason=None, dry_run=False)
    base.update(kw)
    return _ns(**base)


def _wire_flush(monkeypatch, *, ok=True, pending=False):
    log = []

    def _marker(jid, actor=None, reason=None):
        log.append(("flush", jid, reason, actor))
        return (ok, None if ok else "rclone rcat failed")

    monkeypatch.setattr(jobmeta, "write_checkpoint_now_marker", _marker)
    monkeypatch.setattr(jobmeta, "has_checkpoint_now_marker", lambda jid: pending)
    # A flush must touch NOTHING else: no event, no ticket, no cancel marker.
    for name in ("emit_event", "delete_ticket", "write_cancel_marker"):
        monkeypatch.setattr(jobmeta, name,
                            lambda *a, _n=name, **k: pytest.fail(
                                f"flush must not call {_n}"))
    return log


def test_flush_writes_the_marker_and_nothing_else(monkeypatch, capsys):
    _pin_fold(monkeypatch, {"status": "started", "display_status": "running",
                            "n_events": 4, "instance_id": "41"})
    log = _wire_flush(monkeypatch)
    control.cmd_job_flush(_flush_args(reason="pre-park"))
    assert log == [("flush", "j-1", "pre-park", "tester@box")]
    out = capsys.readouterr().out
    assert "flush requested" in out and "does NOT stop the job" in out


def test_flush_refuses_a_terminal_job(monkeypatch):
    """Nothing is running to consume the marker, so writing one is just litter."""
    _pin_fold(monkeypatch, {"status": "done", "display_status": "done",
                            "n_events": 4, "instance_id": "41"})
    _wire_flush(monkeypatch,
                ok=False)  # any write at all would fail the wiring above
    monkeypatch.setattr(jobmeta, "write_checkpoint_now_marker",
                        lambda *a, **k: pytest.fail("terminal => no writes"))
    with pytest.raises(SystemExit) as ei:
        control.cmd_job_flush(_flush_args())
    assert "terminal (done)" in str(ei.value.code)


def test_flush_refuses_an_unknown_job(monkeypatch):
    _pin_fold(monkeypatch, {"status": "submitted", "display_status": "queued",
                            "n_events": 0, "instance_id": None})
    monkeypatch.setattr(jobmeta, "write_checkpoint_now_marker",
                        lambda *a, **k: pytest.fail("unknown => no writes"))
    with pytest.raises(SystemExit) as ei:
        control.cmd_job_flush(_flush_args())
    assert "unknown job j-1" in str(ei.value.code)


def test_flush_warns_but_proceeds_on_a_not_yet_running_job(monkeypatch, capsys):
    _pin_fold(monkeypatch, {"status": "submitted", "display_status": "queued",
                            "n_events": 2, "instance_id": "41"})
    log = _wire_flush(monkeypatch)
    control.cmd_job_flush(_flush_args())
    assert len(log) == 1
    assert "not running" in capsys.readouterr().out


def test_flush_dry_run_writes_nothing(monkeypatch, capsys):
    _pin_fold(monkeypatch, {"status": "started", "display_status": "running",
                            "n_events": 4, "instance_id": "41"})
    _wire_flush(monkeypatch)
    monkeypatch.setattr(jobmeta, "write_checkpoint_now_marker",
                        lambda *a, **k: pytest.fail("--dry-run writes nothing"))
    control.cmd_job_flush(_flush_args(dry_run=True))
    assert "write marker jobs/j-1/CHECKPOINT_NOW" in capsys.readouterr().out


def test_flush_says_so_when_a_marker_is_still_pending(monkeypatch, capsys):
    """Two flushes before the box polls once are still ONE flush — say it rather
    than let the operator read the second `>>` as a second sync."""
    _pin_fold(monkeypatch, {"status": "started", "display_status": "running",
                            "n_events": 4, "instance_id": "41"})
    _wire_flush(monkeypatch, pending=True)
    control.cmd_job_flush(_flush_args())
    assert "already pending" in capsys.readouterr().out


def test_flush_exits_nonzero_when_the_marker_write_fails(monkeypatch):
    _pin_fold(monkeypatch, {"status": "started", "display_status": "running",
                            "n_events": 4, "instance_id": "41"})
    _wire_flush(monkeypatch, ok=False)
    with pytest.raises(SystemExit) as ei:
        control.cmd_job_flush(_flush_args())
    assert "no flush requested" in str(ei.value.code)


# --------------------------------------------------------------------------- #
# 4. cmd_job_orphans — THREE frozen exit codes, first coverage
# --------------------------------------------------------------------------- #
def _orphan_args(**kw):
    base = dict(box=None, job=None, json=False, all=False, resolve=False,
                reason=None, include_interrupted=False, dry_run=False, yes=False)
    base.update(kw)
    return _ns(**base)


def _wire_scan(monkeypatch, rows, present):
    monkeypatch.setattr(control, "_job_orphan_scan",
                        lambda box, job: (list(rows), present))


def _row(box="41", jid="j-1", verdict=pl.TICKET_ORPHAN_UNCLAIMED, status="submitted"):
    return {"box": box, "job_id": jid, "status": status,
            "display_status": status, "n_events": 2, "verdict": verdict,
            "why": "box absent from the account"}


def test_orphans_exits_1_when_the_listing_is_unreadable(monkeypatch, capsys):
    """The tri-state reaching the shell. NOT an error code — "no verdict was
    minted", which is a different thing from "clean"."""
    _wire_scan(monkeypatch, [_row()], None)
    with pytest.raises(SystemExit) as ei:
        control.cmd_job_orphans(_orphan_args())
    assert ei.value.code == 1
    assert "NO orphan verdict was minted" in capsys.readouterr().out


def test_orphans_exits_2_on_stuck_orphans_without_resolve(monkeypatch, capsys):
    _wire_scan(monkeypatch, [_row()], {"99"})
    with pytest.raises(SystemExit) as ei:
        control.cmd_job_orphans(_orphan_args())
    assert ei.value.code == 2
    out = capsys.readouterr().out
    assert "1 STUCK orphan(s)" in out
    # The advice must not send an operator straight to `retarget`: measured
    # 2026-08-26, four of six interrupted orphans on this bucket held no
    # checkpoints at all, and retarget re-runs the ticket's FROZEN bundle.
    assert "job retarget" in out
    assert "CHECK before moving it" in out
    assert "FROZEN" in out


def test_orphans_is_exit_0_when_clean(monkeypatch, capsys):
    """The clean table path RETURNS (exit 0) rather than sys.exit(0) — same code
    at the shell, and pinned so a "tidy-up" to an explicit exit stays visible."""
    _wire_scan(monkeypatch, [_row(verdict=pl.TICKET_OK)], {"41"})
    control.cmd_job_orphans(_orphan_args())
    assert "no stuck orphans" in capsys.readouterr().out


def test_orphans_json_path_exits_0_clean_and_2_stuck(monkeypatch, capsys):
    """`--json` skips the table block, so it reaches the explicit
    `sys.exit(2 if stuck else 0)` — the branch a scripted caller reads."""
    _wire_scan(monkeypatch, [_row(verdict=pl.TICKET_OK)], {"41"})
    with pytest.raises(SystemExit) as ei:
        control.cmd_job_orphans(_orphan_args(json=True))
    assert ei.value.code == 0

    _wire_scan(monkeypatch, [_row()], {"99"})
    with pytest.raises(SystemExit) as ei:
        control.cmd_job_orphans(_orphan_args(json=True))
    assert ei.value.code == 2


def test_orphans_returns_without_exiting_on_an_empty_queue(monkeypatch, capsys):
    _wire_scan(monkeypatch, [], {"41"})
    control.cmd_job_orphans(_orphan_args())       # no SystemExit at all
    assert "no queued tickets." in capsys.readouterr().out


def test_orphans_json_reports_listing_readable_alongside_the_rows(monkeypatch,
                                                                  capsys):
    _wire_scan(monkeypatch, [_row()], None)
    with pytest.raises(SystemExit) as ei:
        control.cmd_job_orphans(_orphan_args(json=True))
    assert ei.value.code == 1
    body = json.loads(capsys.readouterr().out)
    assert body["listing_readable"] is False
    assert set(body["rows"][0]) == {"box", "job_id", "status", "display_status",
                                    "n_events", "verdict", "why"}


def test_orphans_stale_terminal_pointers_are_left_in_place(monkeypatch, capsys):
    """A stale pointer is not stuck: results and events are on B2, and the ticket
    is what keeps the box's history visible in `job ls`."""
    _wire_scan(monkeypatch, [_row(verdict=pl.TICKET_ORPHAN_TERMINAL,
                                  status="done")], {"99"})
    control.cmd_job_orphans(_orphan_args())
    out = capsys.readouterr().out
    assert "Left in place ON PURPOSE" in out and "no stuck orphans" in out


# --- the resolve guard rails ---------------------------------------------- #
def test_resolve_requires_a_reason(monkeypatch):
    _wire_scan(monkeypatch, [_row()], {"99"})
    with pytest.raises(SystemExit) as ei:
        control.cmd_job_orphans(_orphan_args(resolve=True))
    assert "--resolve requires --reason" in str(ei.value.code)


def test_resolve_requires_y_before_any_write(monkeypatch):
    _wire_scan(monkeypatch, [_row()], {"99"})
    monkeypatch.setattr(jobmeta, "write_cancel_marker",
                        lambda *a, **k: pytest.fail("no -y => no writes"))
    with pytest.raises(SystemExit) as ei:
        control.cmd_job_orphans(_orphan_args(resolve=True, reason="superseded"))
    assert "re-run with -y" in str(ei.value.code)


def test_resolve_skips_an_interrupted_orphan_unless_asked(monkeypatch, capsys):
    """An ORPHAN_INTERRUPTED job may have checkpoints — ending it is losing work."""
    _wire_scan(monkeypatch, [_row(verdict=pl.TICKET_ORPHAN_INTERRUPTED)], {"99"})
    monkeypatch.setattr(jobmeta, "write_cancel_marker",
                        lambda *a, **k: pytest.fail("skipped rows are not written"))
    control.cmd_job_orphans(_orphan_args(resolve=True, reason="why", yes=True))
    out = capsys.readouterr().out
    assert "SKIP j-1" in out and "nothing to resolve." in out


def test_resolve_re_asserts_that_the_box_is_absent_before_writing(monkeypatch):
    """Belt-and-suspenders: a future refactor of the filters must not be able to
    widen the blast radius onto a LIVE box's queue."""
    _wire_scan(monkeypatch, [_row()], {"41"})     # the row's box IS present
    monkeypatch.setattr(jobmeta, "write_cancel_marker",
                        lambda *a, **k: pytest.fail("a present box is never written"))
    with pytest.raises(SystemExit) as ei:
        control.cmd_job_orphans(_orphan_args(resolve=True, reason="why", yes=True))
    assert "box 41 exists (internal inconsistency)" in str(ei.value.code)


def test_resolution_is_a_cancel_through_the_same_three_writes(monkeypatch, capsys):
    _wire_scan(monkeypatch, [_row()], {"99"})
    log = _wire_writes(monkeypatch)
    control.cmd_job_orphans(_orphan_args(resolve=True, reason="superseded",
                                         yes=True))
    assert [row[0] for row in log] == ["marker", "event", "delete"]
    kw = log[1][3]
    assert kw["orphan"] == pl.TICKET_ORPHAN_UNCLAIMED and kw["orphan_box"] == "41"
    assert "superseded" in kw["reason"] and "no longer exists" in kw["reason"]
    assert "resolved 1 orphan(s)" in capsys.readouterr().out


def test_resolution_preserves_the_frozen_config_in_the_dlq(monkeypatch, capsys):
    """Resolving an orphan ends with `delete_ticket`, and the ticket is the ONLY
    place a job's frozen `config` lives — the submitted event carries
    bundle_sha256/entrypoint/timeout_s but not the env. So this was the one
    operation in the system that could destroy the record of what a job would
    have run. It now copies the ticket to the DLQ first."""
    _wire_scan(monkeypatch, [_row()], {"99"})
    _wire_writes(monkeypatch)
    ticket = {"v": 1, "job_id": "j-1", "box": "41", "bundle_sha256": "sha9",
              "submitted_ts": "20260819T072006000Z",
              "config": {"env": {"FLA_REQUIRED": "0"}}}
    monkeypatch.setattr(jobmeta, "read_ticket", lambda *a, **k: ticket)
    seen = {}

    def _dlq(tk, **kw):
        seen["ticket"], seen["kw"] = tk, kw
        return True, jobmeta.dlq_key(tk["box"], tk["job_id"]), ""

    monkeypatch.setattr(jobmeta, "write_dlq_entry", _dlq)
    control.cmd_job_orphans(_orphan_args(resolve=True, reason="superseded",
                                         yes=True))
    assert seen["ticket"]["config"]["env"] == {"FLA_REQUIRED": "0"}
    assert seen["kw"]["verdict"] == pl.TICKET_ORPHAN_UNCLAIMED
    assert "retired to jobs/dlq/41/j-1.json" in capsys.readouterr().out


def test_a_failed_dlq_write_does_not_block_the_cancel(monkeypatch, capsys):
    """Ending a job that can never run again is still right — but the operator
    must be told the frozen config went with it."""
    _wire_scan(monkeypatch, [_row()], {"99"})
    log = _wire_writes(monkeypatch)
    monkeypatch.setattr(jobmeta, "read_ticket",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("B2_BUCKET not set")))
    control.cmd_job_orphans(_orphan_args(resolve=True, reason="x", yes=True))
    assert [row[0] for row in log] == ["marker", "event", "delete"]
    out = capsys.readouterr().out
    assert "DLQ write FAILED" in out and "frozen config is NOT kept" in out


def test_resolve_dry_run_writes_nothing(monkeypatch, capsys):
    _wire_scan(monkeypatch, [_row()], {"99"})
    monkeypatch.setattr(jobmeta, "write_cancel_marker",
                        lambda *a, **k: pytest.fail("--dry-run writes nothing"))
    control.cmd_job_orphans(_orphan_args(resolve=True, reason="why", dry_run=True))
    out = capsys.readouterr().out
    assert "[dry-run] would resolve j-1" in out
    assert "resolved 1 orphan(s)" not in out


# --------------------------------------------------------------------------- #
# 5. _job_orphan_scan — the tri-state is passed THROUGH, not flattened
# --------------------------------------------------------------------------- #
def test_orphan_scan_returns_presence_unchanged_when_unreadable(monkeypatch):
    monkeypatch.setattr(jobmeta, "list_all_queued", lambda: [("41", "j-1")])
    monkeypatch.setattr(view, "_present_iids_set", lambda: None)
    monkeypatch.setattr(view, "_live_iids_set", lambda: set())
    # The scan folds the queue in ONE bulk call (`jobs/scan.py`), not one
    # `jobmeta.read_job` per ticket — 275 tickets used to cost 275 rclone
    # subprocesses and 139 s. `fold_many` is the seam now; patch it, not read_job.
    monkeypatch.setattr(scan, "fold_many", lambda jids, live_iids=(): {
        j: {"status": "submitted", "display_status": "queued", "n_events": 2}
        for j in jids})
    rows, present = control._job_orphan_scan()
    assert present is None
    assert rows[0]["verdict"] == pl.TICKET_UNKNOWN


def test_orphan_scan_turns_an_unreadable_log_into_a_row_not_an_abort(monkeypatch):
    """One job's broken log becomes ONE unknown row; the other 274 still get a
    verdict. `fold_many` reports that per job as `scan_error` rather than
    raising, precisely so a single bad log cannot blank the report."""
    monkeypatch.setattr(jobmeta, "list_all_queued", lambda: [("41", "bad"),
                                                             ("41", "ok")])
    monkeypatch.setattr(view, "_present_iids_set", lambda: {"41"})
    monkeypatch.setattr(view, "_live_iids_set", lambda: {"41"})

    def _fold(jids, live_iids=()):
        out = {}
        for j in jids:
            if j == "bad":
                out[j] = {"scan_error": "cannot fold"}
            else:
                out[j] = {"status": "submitted", "display_status": "queued",
                          "n_events": 2}
        return out

    monkeypatch.setattr(scan, "fold_many", _fold)
    rows, _ = control._job_orphan_scan()
    assert [r["verdict"] for r in rows] == [pl.TICKET_UNKNOWN, pl.TICKET_OK]
    assert "unreadable: cannot fold" in rows[0]["why"]


def test_orphan_scan_exits_rather_than_calling_a_failed_listing_clean(monkeypatch):
    """A listing that failed WHOLESALE must not degrade to "no events", which
    folds every ticket to unclaimed and reports the entire fleet as orphaned.
    Loud exit, no verdict — the same posture as `_present_iids_set`'s None."""
    monkeypatch.setattr(jobmeta, "list_all_queued", lambda: [("41", "j-1")])
    monkeypatch.setattr(view, "_present_iids_set", lambda: {"99"})
    monkeypatch.setattr(view, "_live_iids_set", lambda: set())

    def _boom(jids, live_iids=()):
        raise scan.ScanError("bulk job listing failed rc=3")

    monkeypatch.setattr(scan, "fold_many", _boom)
    with pytest.raises(SystemExit) as ei:
        control._job_orphan_scan()
    assert "bulk job scan failed" in str(ei.value.code)


# --------------------------------------------------------------------------- #
# 6. retarget — the parts the flat file does not cover
# --------------------------------------------------------------------------- #
def test_drop_stale_refuses_to_delete_the_retarget_target(monkeypatch):
    """Assert rather than trust the caller's set arithmetic: the ONE box this
    must never touch is the one the ticket was just written to."""
    monkeypatch.setattr(jobmeta, "delete_ticket",
                        lambda b, j: pytest.fail("must exit before deleting"))
    with pytest.raises(SystemExit) as ei:
        control._retarget_drop_stale(_ns(dry_run=False), "j-1", "42", {"42"})
    assert "is the retarget TARGET" in str(ei.value.code)


def test_drop_stale_warns_but_does_not_fail_on_a_failed_delete(monkeypatch,
                                                               capsys):
    """The write already landed; a failed delete is a loud warning, not an exit."""
    monkeypatch.setattr(jobmeta, "delete_ticket", lambda b, j: (False, "rclone 1"))
    control._retarget_drop_stale(_ns(dry_run=False), "j-1", "42", {"41"})
    assert "may double-run j-1" in capsys.readouterr().out


def test_retarget_queued_boxes_is_sorted_and_deduped(monkeypatch):
    monkeypatch.setattr(jobmeta, "list_all_queued",
                        lambda: [("77", "j-1"), ("41", "j-1"), ("41", "j-1"),
                                 ("99", "other")])
    assert control._retarget_queued_boxes("j-1") == ["41", "77"]


def test_vram_advisory_never_raises_and_never_refuses(monkeypatch, capsys):
    """A recovery must never be blocked by the sizing gate — `retarget` is what
    fleetd drives on an eviction."""
    monkeypatch.setattr(jobmeta, "vram_gate_findings",
                        lambda cfg: (_ for _ in ()).throw(RuntimeError("boom")))
    control._vram_advisory({"needs": {}}, where="retarget")     # no raise
    assert capsys.readouterr().err == ""


def test_vram_advisory_relabels_the_finding_with_its_lane(monkeypatch, capsys):
    monkeypatch.setattr(jobmeta, "vram_gate_findings", lambda cfg: ["f"])
    monkeypatch.setattr(jobmeta, "vram_gate_report",
                        lambda f: (["!! vram: floor 80 GiB > 24 GiB"], True))
    control._vram_advisory({}, where="requeue")
    assert "!! vram (requeue): floor 80" in capsys.readouterr().err


def test_retarget_reads_liveness_through_the_view_module_attribute(monkeypatch):
    """Plan §8b. A `from .view import _live_iids_set` would make this patch — and
    every handoff/fleetd patch site — vacuous."""
    seen = []
    monkeypatch.setattr(view, "_live_iids_set", lambda: seen.append(1) or {"41"})
    monkeypatch.setattr(jobmeta, "read_job", lambda jid, live_iids=None: {
        "status": "done", "display_status": "done"})
    with pytest.raises(SystemExit):
        control.cmd_job_retarget(_ns(job_id="j-1", box="42", from_box=None,
                                     dry_run=False))
    assert seen == [1]


def test_retarget_refuses_a_terminal_job(monkeypatch):
    monkeypatch.setattr(view, "_live_iids_set", lambda: set())
    monkeypatch.setattr(jobmeta, "read_job", lambda jid, live_iids=None: {
        "status": "done", "display_status": "done"})
    with pytest.raises(SystemExit) as ei:
        control.cmd_job_retarget(_ns(job_id="j-1", box="42", from_box="41",
                                     dry_run=False))
    assert "already terminal" in str(ei.value.code)


def test_retarget_names_the_box_the_FOLD_says_not_a_stale_from(monkeypatch):
    """A wrong `--from` on a running job used to assert liveness of a box the
    fold never named."""
    monkeypatch.setattr(view, "_live_iids_set", lambda: {"41"})
    monkeypatch.setattr(jobmeta, "read_job", lambda jid, live_iids=None: {
        "status": "started", "display_status": "running",
        "target_box": "41", "instance_id": "41"})
    with pytest.raises(SystemExit) as ei:
        control.cmd_job_retarget(_ns(job_id="j-1", box="42", from_box="77",
                                     dry_run=False))
    msg = str(ei.value.code)
    assert "RUNNING on live box 41" in msg
    assert "--from 77 is stale" in msg


# --------------------------------------------------------------------------- #
# 7. requeue — the eight refusals are `test_job_requeue.py`'s; the SEAM is ours
# --------------------------------------------------------------------------- #
def test_requeue_eligibility_is_exactly_failed():
    assert control._requeue_refusal({"status": "failed", "n_events": 3}) is None


@pytest.mark.parametrize("v,needle", [
    ({"n_events": 0}, "it never invents one"),
    ({"n_events": 3, "status": "done"}, "`done` is STICKY"),
    ({"n_events": 3, "status": "cancelled"}, "never-revive"),
    ({"n_events": 3, "status": "failed", "reopened": True}, "already re-opened"),
    ({"n_events": 3, "status": "claimed"}, "TERMINAL-FAILED jobs only"),
    ({"n_events": 3, "status": "submitted"}, "still QUEUED"),
    ({"n_events": 3, "status": "interrupted"}, "not `failed`"),
])
def test_each_requeue_refusal_names_a_different_next_step(v, needle):
    why = control._requeue_refusal(v)
    assert why is not None and needle in why


def test_requeue_gate_1_reads_the_FRESH_fold(monkeypatch):
    """A cached fold that is a few minutes behind is exactly how you requeue a
    job that is still running."""
    monkeypatch.setattr(jobmeta, "validate_job_id", lambda jid: None)
    monkeypatch.setattr(view, "_live_iids_set", lambda: set())
    monkeypatch.setattr(jobmeta, "read_job",
                        lambda *a, **k: pytest.fail("requeue must use read_job_fresh"))
    monkeypatch.setattr(jobmeta, "read_job_fresh", lambda jid, live_iids=None: {
        "status": "done", "display_status": "done", "n_events": 4})
    with pytest.raises(SystemExit) as ei:
        control.cmd_job_requeue(_ns(job_id="j-1", box="42", from_box="41",
                                    bundle=str(VAST_DIR), dry_run=False, env=None))
    assert "`done` is STICKY" in str(ei.value.code)


def test_requeue_refuses_the_box_the_job_failed_on(monkeypatch):
    """That box's jobd keeps a LOCAL .terminal breadcrumb and would skip the
    ticket forever, silently."""
    monkeypatch.setattr(jobmeta, "validate_job_id", lambda jid: None)
    monkeypatch.setattr(view, "_live_iids_set", lambda: set())
    monkeypatch.setattr(jobmeta, "read_job_fresh", lambda jid, live_iids=None: {
        "status": "failed", "display_status": "failed", "n_events": 6,
        "instance_id": "41", "bundle_sha256": "a" * 64})
    with pytest.raises(SystemExit) as ei:
        control.cmd_job_requeue(_ns(job_id="j-1", box="41", from_box=None,
                                    bundle=str(VAST_DIR), dry_run=False, env=None))
    assert "local terminal cache would skip the ticket forever" in str(ei.value.code)


def test_requeue_refuses_bundle_drift(monkeypatch):
    """No `--allow-bundle-drift` by design: a changed bundle is a DIFFERENT
    experiment and must not inherit the failed job's log + checkpoints."""
    monkeypatch.setattr(jobmeta, "validate_job_id", lambda jid: None)
    monkeypatch.setattr(view, "_live_iids_set", lambda: set())
    monkeypatch.setattr(jobmeta, "read_job_fresh", lambda jid, live_iids=None: {
        "status": "failed", "display_status": "failed", "n_events": 6,
        "instance_id": "41", "bundle_sha256": "a" * 64})
    monkeypatch.setattr(jobmeta, "bundle_sha256", lambda src: "b" * 64)
    with pytest.raises(SystemExit) as ei:
        control.cmd_job_requeue(_ns(job_id="j-1", box="42", from_box="41",
                                    bundle=str(VAST_DIR), dry_run=False, env=None))
    assert "bundle DRIFT" in str(ei.value.code)


# --------------------------------------------------------------------------- #
# 8. TWIN IDENTITY — one copy since plan §8 step 6d
# --------------------------------------------------------------------------- #
# This section used to run `control._job_cancel_kill_script` and
# `control._requeue_refusal` against `herdd`'s originals on the same inputs.
# The header above it said the flat copies "are live through the add-only
# phase" and that `herdd.py` keeps its originals "until plan step 6"; step 6d
# arrived and it does not. The launcher re-exports both names from
# `vastlib.jobs.control` by identity, so those comparisons became `x == x` and
# are deleted. The artifacts they protected keep their real coverage: the kill
# script's quoting and JOB_ID-off-the-cmdline property at §3 of this file, the
# refusal strings at §7 here and in `test_job_requeue.py`, which now drives
# this module directly.
import herdd                                        # noqa: E402


def test_the_launcher_re_exports_rather_than_redefines():
    """A second body under either name would resurrect the twin-drift hazard.

    `test_job_requeue.py` asserts on the refusal substrings and the kill script
    is executed verbatim on a real box, so a divergent launcher copy would be
    an operator-visible difference between `herdd.py job cancel` and every
    other caller — silently, because nothing else compares them any more.
    """
    for name in ("_job_cancel_kill_script", "_requeue_refusal",
                 "_vram_advisory"):
        assert getattr(herdd, name) is getattr(control, name), (
            f"herdd.{name} is a second body again — the launcher must "
            f"re-export vastlib.jobs.control's object, never redefine it")
