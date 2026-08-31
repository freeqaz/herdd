"""Portable tests for `herdd job retarget` — moving a queued/interrupted job's
ticket to another box (task #75).

Runs in the toolchain-free lane (`pytest -m "not integration"`): NO vast API, NO
B2/rclone, NO network, NO creds. The fold views are built by the REAL
`jobmeta.fold_events` over synthetic event bodies, and the only stubs are the
seams that would touch the network.

THE DEFECT THIS FILE EXISTS FOR. `cmd_job_retarget` had zero real coverage and
hard-exited the moment `read_ticket(--from, JOB_ID)` came back None:

    error: no ticket at jobs/queue/47219058/<JOB_ID>.json

On 2026-08-08 that was the whole reason "no CLI path moves a claimed/interrupted
job off a dead box" (V10_SPOT_PROVISIONING §6) and the recovery was a hand-run
`jobmeta.make_ticket` snippet. The diagnosis in that write-up — "a queue ticket
jobd already consumed" — is wrong about the mechanism, and the mechanism is what
makes the fix obvious:

  * jobd NEVER deletes a queue ticket. It only `cat`/`lsf`/`copyto`s
    jobs/queue/<IID>/ (jobd.sh). Claiming a job does not consume its pointer.
  * What moves a ticket is fleetd: `_retarget_pending_tickets` rewrites `box`
    and deletes the old pointer during an eviction/pull replacement — which is
    exactly what fleetd's parallel chain was doing that night.

So the ordinary failure is a LIVE ticket under a box the operator has not heard
of, with `--from` naming the old one. Step 1 scans the queue; step 2
(`--reconstruct`, opt-in) rebuilds a ticket only when none survives anywhere.
"""
import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobmeta as jm  # noqa: E402
from vastlib.boxes import lifecycle  # noqa: E402
from vastlib.jobs import control, view as jobs_view  # noqa: E402
from vastlib.storage import b2  # noqa: E402


_JID = "20260808T220906-driftr3-v10-27b-gen-269f"
_OLD, _NEW, _THIRD = "47219058", "47226953", "47219872"
_SHA = "a" * 64


# --------------------------------------------------------------------------- #
# helpers (lifted from test_job_requeue.py / test_job_orphans.py)
# --------------------------------------------------------------------------- #
def _ev(event, ts, **fields):
    d = {"v": 1, "ts": ts, "actor": f"box:{_OLD}", "event": event, "job_id": _JID,
         "nonce": ts[-4:] + event[:2]}
    d.update(fields)
    return d


def _T(n):
    return f"20260808T2209{n:02d}000Z"


def _view(*evs, live=()):
    return jm.fold_events(list(evs), live_iids=live)


def _queued_view(box=_OLD, sha=_SHA):
    """`submitted` only — the ticket sits in a queue, never claimed."""
    return _view(_ev("submitted", _T(1), actor="cli:h", box=box,
                     bundle_sha256=sha, name="driftr3-v10-27b-gen",
                     entrypoint="run.sh", timeout_s=600))


def _interrupted_view(box=_OLD, sha=_SHA, live=()):
    """claimed+started on a box that is NOT live -> display_status
    `interrupted` (jobmeta.fold_events). This is the shape the running fence
    must pass and the incident's own job was in."""
    return _view(
        _ev("submitted", _T(1), actor="cli:h", box=box, bundle_sha256=sha,
            name="driftr3-v10-27b-gen", entrypoint="run.sh", timeout_s=600),
        _ev("claimed", _T(2), instance_id=box),
        _ev("started", _T(3), instance_id=box), live=live)


def _ticket(box=_OLD, **over):
    # `submitted_ts` is NOW, not _T(1). The event timestamps are fixed because
    # only their ORDER matters, but submitted_ts is compared against the wall
    # clock by the staleness gate (jobmeta.ticket_staleness), so a hardcoded
    # 2026-08-08 value made every happy-path case here a time bomb that started
    # refusing three days after that date. Tickets deliberately older than the
    # bound are built with an explicit submitted_ts override, below.
    t = {"v": 1, "job_id": _JID, "bundle_sha256": _SHA, "box": box,
         "submitted_ts": jm.runmeta.now_ts(), "actor": "cli:h",
         "config": {"version": 1, "name": "driftr3-v10-27b-gen",
                    "entrypoint": "run.sh", "timeout_s": 600,
                    "env": {"WAVE": "A"}}}
    t.update(over)
    return t


def _args(**over):
    ns = argparse.Namespace(job_id=_JID, box=_NEW, from_box=None,
                            reconstruct=False, dry_run=False, stale_ok=False)
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture
def offline(monkeypatch):
    """Neutralize the network seams. Anything else that would touch B2 is left
    UNPATCHED on purpose: a test that reaches it fails loudly."""
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(jobs_view, "_live_iids_set", lambda *a, **k: set())
    monkeypatch.setattr(lifecycle, "_cli_actor", lambda: "cli:test")
    return monkeypatch


class Q:
    """Recorder for the B2 mutations `retarget` is allowed to make."""

    def __init__(self, offline, view, *, tickets=None, queued=None, live=()):
        self.writes, self.deletes, self.events = [], [], []
        self.tickets = dict(tickets or {})          # box -> ticket|None
        self.queued = list(queued if queued is not None
                           else [(b, _JID) for b in self.tickets])
        offline.setattr(jm, "read_job", lambda jid, **k: view)
        offline.setattr(jm, "read_ticket",
                        lambda box, jid, **k: self.tickets.get(str(box)))
        offline.setattr(jm, "list_all_queued",
                        lambda **k: list(self.queued))
        offline.setattr(jm, "write_ticket", self._write)
        offline.setattr(jm, "delete_ticket", self._delete)
        offline.setattr(jm, "emit_event", self._emit)
        if live:
            offline.setattr(jobs_view, "_live_iids_set", lambda *a, **k: set(live))

    def _write(self, ticket, **k):
        self.writes.append(dict(ticket))
        return True, f"jobs/queue/{ticket['box']}/{ticket['job_id']}.json", ""

    def _delete(self, box, jid, **k):
        self.deletes.append((str(box), jid))
        return True, ""

    def _emit(self, jid, event, **fields):
        self.events.append((jid, event, fields))
        return {}


# --------------------------------------------------------------------------- #
# 1. the unchanged contract: happy path + the fences
# --------------------------------------------------------------------------- #
def test_a_stale_ticket_is_refused_because_retarget_re_runs_FROZEN_bytes(offline):
    """The 2026-08-24 screen-v1 incident, as a test.

    A retarget re-runs the ticket's frozen `config` + `bundle_sha256`, not the
    repo's current bundle. Four screen-v1 arms were retargeted onto a live box
    days after submit; they re-ran a superseded fla gate and re-emitted
    provenance sidecars that had already been corrected by hand. Nothing in the
    system said a word — there was no age check anywhere in ticket selection.
    """
    Q(offline, _interrupted_view(),
      tickets={_OLD: _ticket(submitted_ts="20260808T220901000Z")})
    with pytest.raises(SystemExit) as e:
        control.cmd_job_retarget(_args(from_box=_OLD))
    msg = str(e.value)
    assert "STALE" in msg
    assert "FROZEN bytes" in msg
    assert "job dlq add" in msg            # the retirement route is offered


def test_stale_ok_moves_the_ticket_anyway(offline):
    """The override exists, and it is the only way through — the refusal must
    never be silently bypassable."""
    q = Q(offline, _interrupted_view(),
          tickets={_OLD: _ticket(submitted_ts="20260808T220901000Z")})
    control.cmd_job_retarget(_args(from_box=_OLD, stale_ok=True))
    assert len(q.writes) == 1
    assert q.writes[0]["box"] == _NEW


def test_a_fresh_ticket_is_not_gated(offline):
    """The bound catches forgotten backlog, not ordinary same-day recovery."""
    q = Q(offline, _interrupted_view(), tickets={_OLD: _ticket()})
    control.cmd_job_retarget(_args(from_box=_OLD))
    assert len(q.writes) == 1


def test_happy_path_moves_the_ticket_and_deletes_the_old_pointer(offline, capsys):
    q = Q(offline, _interrupted_view(), tickets={_OLD: _ticket()})
    control.cmd_job_retarget(_args(from_box=_OLD))

    assert len(q.writes) == 1
    assert q.writes[0]["box"] == _NEW
    assert q.writes[0]["retargeted_from"] == _OLD          # jobd's checkpoint cue
    assert q.writes[0]["config"] == _ticket()["config"]    # env pins ride along
    assert q.deletes == [(_OLD, _JID)]
    assert [e for _j, e, _f in q.events] == ["retargeted"]
    assert q.events[0][2] == {"box": _NEW, "from_box": _OLD}
    assert f"{_OLD} -> {_NEW}" in capsys.readouterr().out


def test_the_move_names_the_watch_the_new_box_still_owes(offline, capsys):
    """Nothing on the retarget path registers a watch, and the fresh box it
    moves work onto carries at most a `bare` one from `launch --jobs`. The move
    is also the first moment a `jobs` watch is SAFE to arm — the ticket it just
    wrote is non-terminal — so the two facts belong on this line."""
    Q(offline, _interrupted_view(), tickets={_OLD: _ticket()})
    control.cmd_job_retarget(_args(from_box=_OLD))
    out = capsys.readouterr().out
    assert "no bid defense" in out
    assert f"fleet watch {_NEW} --profile jobs --budget <USD> --standing" in out


def test_a_dry_run_promises_no_watch_it_has_not_earned(offline, capsys):
    """A dry run writes no ticket, so the "safe to arm now" claim would be
    false — and an operator who armed on it would park the box."""
    Q(offline, _interrupted_view(), tickets={_OLD: _ticket()})
    control.cmd_job_retarget(_args(from_box=_OLD, dry_run=True))
    assert "no bid defense" not in capsys.readouterr().out


def test_an_interrupted_job_is_retargetable_and_a_running_one_is_not(offline):
    """The fence is `display_status`, and `fold_events` maps claimed/started on a
    box that is NOT live to `interrupted` — which is why the incident's job
    cleared it and only the missing-ticket exit stood in the way."""
    assert _interrupted_view()["display_status"] == "interrupted"
    assert _interrupted_view(live={_OLD})["display_status"] == "running"

    q = Q(offline, _interrupted_view(live={_OLD}), tickets={_OLD: _ticket()},
          live={_OLD})
    with pytest.raises(SystemExit) as e:
        control.cmd_job_retarget(_args(from_box=_OLD))
    assert "is RUNNING on live box" in str(e.value)
    assert f"live box {_OLD}" in str(e.value)
    assert q.writes == [] and q.deletes == []


def test_running_fence_names_the_true_box_not_a_stale_from(offline):
    """THE DEFECT (drill, 2026-08-09). `old_box` takes `a.from_box` FIRST, so a
    wrong/stale `--from` on a job actually running elsewhere used to print
    'is RUNNING on live box <the wrong id>' — asserting liveness of a box the
    fold never named at all. The refusal is correct (never let a running job
    get double-queued); the message must name the box the job is ACTUALLY on
    (the folded view's target_box) and say the --from was stale."""
    v = _interrupted_view(box=_THIRD, live={_THIRD})
    assert v["display_status"] == "running" and v["target_box"] == _THIRD

    q = Q(offline, v, tickets={_THIRD: _ticket(box=_THIRD)}, live={_THIRD})
    with pytest.raises(SystemExit) as e:
        control.cmd_job_retarget(_args(from_box=_OLD))     # WRONG/stale --from
    msg = str(e.value)
    assert f"is RUNNING on live box {_THIRD}" in msg   # the TRUE box, not _OLD
    assert f"live box {_OLD}" not in msg               # never assert liveness of it
    assert f"--from {_OLD} is stale" in msg
    assert q.writes == [] and q.deletes == []


def test_a_terminal_job_is_refused(offline):
    v = _view(_ev("submitted", _T(1), actor="cli:h", box=_OLD),
              _ev("done", _T(4), instance_id=_OLD, rc=0))
    Q(offline, v, tickets={_OLD: _ticket()})
    with pytest.raises(SystemExit) as e:
        control.cmd_job_retarget(_args(from_box=_OLD))
    assert "already terminal" in str(e.value)


def test_same_source_and_target_is_refused(offline):
    Q(offline, _interrupted_view(), tickets={_OLD: _ticket()})
    with pytest.raises(SystemExit) as e:
        control.cmd_job_retarget(_args(from_box=_OLD, box=_OLD))
    assert "same" in str(e.value)


def test_a_box_that_already_went_terminal_on_the_job_is_refused_as_a_target(offline):
    """Retarget refuses a target box whose jobd already emitted a terminal event
    for this JOB_ID — that box holds a `.terminal` breadcrumb and would skip the
    ticket forever, so the move is a silent no-op."""
    # failed on _NEW, then requeued (`resumed` newer) -> non-terminal, so the
    # move is otherwise allowed; _NEW is the poisoned box.
    v = _view(_ev("submitted", _T(1), actor="cli:h", box=_OLD, bundle_sha256=_SHA),
              _ev("failed", _T(4), actor=f"box:{_NEW}", instance_id=_NEW, rc=1,
                  reason="ticket parse failed (rc=1)"),
              _ev("resumed", _T(5), actor="cli:h", box=_OLD, kind="requeue"))
    q = Q(offline, v, tickets={_OLD: _ticket()})
    with pytest.raises(SystemExit) as e:
        control.cmd_job_retarget(_args(from_box=_OLD))
    assert "skip the ticket forever" in str(e.value)
    assert q.writes == [] and q.deletes == [] and q.events == []


def test_a_box_that_never_touched_the_job_is_still_a_legal_target(offline):
    """The poison gate is evidence-based: a terminal event from a DIFFERENT box
    does not refuse an untouched target."""
    v = _view(_ev("submitted", _T(1), actor="cli:h", box=_OLD, bundle_sha256=_SHA),
              _ev("failed", _T(4), actor=f"box:{_THIRD}", instance_id=_THIRD, rc=1),
              _ev("resumed", _T(5), actor="cli:h", box=_OLD, kind="requeue"))
    q = Q(offline, v, tickets={_OLD: _ticket()})
    control.cmd_job_retarget(_args(from_box=_OLD))
    assert [w["box"] for w in q.writes] == [_NEW]
    assert q.deletes == [(_OLD, _JID)]


def test_a_claim_time_failure_with_no_instance_id_still_names_its_box(offline):
    """The poison gate must read the box out of the `box:<iid>` actor stamp: a
    claim-time `failed` never reaches the fold's `view["instance_id"]`, and this
    fixture drops the per-event stamp too (jobd itself always sets it), so the
    actor is the only attribution left."""
    v = _view(_ev("submitted", _T(1), actor="cli:h", box=_OLD, bundle_sha256=_SHA),
              _ev("failed", _T(4), actor=f"box:{_NEW}", rc=1,
                  reason="ticket parse failed (rc=1)"),
              _ev("resumed", _T(5), actor="cli:h", box=_OLD, kind="requeue"))
    Q(offline, v, tickets={_OLD: _ticket()})
    with pytest.raises(SystemExit) as e:
        control.cmd_job_retarget(_args(from_box=_OLD))
    assert _NEW in str(e.value) and "terminal" in str(e.value)


def test_dry_run_makes_no_b2_mutation(offline, capsys):
    q = Q(offline, _interrupted_view(), tickets={_OLD: _ticket()})
    control.cmd_job_retarget(_args(from_box=_OLD, dry_run=True))
    assert q.writes == [] and q.deletes == [] and q.events == []
    out = capsys.readouterr().out
    assert "[dry-run]" in out and f"jobs/queue/{_NEW}/{_JID}.json" in out


# --------------------------------------------------------------------------- #
# 2. THE INCIDENT REGRESSION — the ticket already moved
# --------------------------------------------------------------------------- #
def test_absent_ticket_is_found_at_the_box_fleetd_moved_it_to(offline, capsys):
    """2026-08-08, the dead end. fleetd's `_retarget_pending_tickets` had already
    moved the ticket onto its replacement box while the operator was hand-
    recovering from the other side; `--from <the old box>` then hit

        error: no ticket at jobs/queue/47219058/<JOB_ID>.json

    and there was no CLI path forward. jobd deletes no tickets, so a missing one
    at `--from` means MOVED, not consumed — scan the queue and proceed."""
    q = Q(offline, _interrupted_view(),
          tickets={_OLD: None, _THIRD: _ticket(box=_THIRD)},
          queued=[(_THIRD, _JID)])
    control.cmd_job_retarget(_args(from_box=_OLD))

    assert len(q.writes) == 1
    assert q.writes[0]["box"] == _NEW
    assert q.writes[0]["retargeted_from"] == _THIRD    # the REAL source
    assert q.deletes == [(_THIRD, _JID)]
    assert q.events[0][2] == {"box": _NEW, "from_box": _THIRD}
    out = capsys.readouterr().out
    assert f"the ticket is at {_THIRD}" in out
    assert "jobd never deletes them" in out


def test_already_at_the_target_is_an_idempotent_success(offline, capsys):
    """fleetd got there first and moved it exactly where we were asked to move
    it. Reporting "no ticket at jobs/queue/<old>/" and exiting 1 is a lie about
    the outcome — the world is already in the requested state."""
    q = Q(offline, _interrupted_view(), tickets={_OLD: None, _NEW: _ticket(box=_NEW)},
          queued=[(_NEW, _JID)])
    control.cmd_job_retarget(_args(from_box=_OLD))

    assert q.writes == [] and q.deletes == [] and q.events == []
    assert f"ALREADY queued at {_NEW}" in capsys.readouterr().out


def test_a_stale_leftover_pointer_is_swept_on_the_idempotent_path(offline):
    """Already at the target AND a leftover elsewhere: the leftover is the whole
    reason the delete exists (the other box would double-run on resume), and the
    target's own ticket must never be the one deleted."""
    q = Q(offline, _interrupted_view(),
          tickets={_OLD: None, _NEW: _ticket(box=_NEW), _THIRD: _ticket(box=_THIRD)},
          queued=[(_NEW, _JID), (_THIRD, _JID)])
    control.cmd_job_retarget(_args(from_box=_OLD))
    assert q.deletes == [(_THIRD, _JID)]
    assert q.writes == []


def test_two_pointers_and_neither_is_the_target_is_refused(offline):
    """A double-run in waiting; picking one for the operator would hide it."""
    Q(offline, _interrupted_view(), tickets={_OLD: None},
      queued=[(_THIRD, _JID), ("999", _JID)])
    with pytest.raises(SystemExit) as e:
        control.cmd_job_retarget(_args(from_box=_OLD))
    assert "MORE THAN ONE box" in str(e.value)


def test_no_ticket_anywhere_names_reconstruct_instead_of_dead_ending(offline):
    q = Q(offline, _interrupted_view(), tickets={_OLD: None}, queued=[])
    with pytest.raises(SystemExit) as e:
        control.cmd_job_retarget(_args(from_box=_OLD))
    msg = str(e.value)
    assert "--reconstruct" in msg
    assert "jobd does not delete tickets" in msg
    assert q.writes == []


# --------------------------------------------------------------------------- #
# 3. --reconstruct (opt-in): mint a ticket from the submitted bundle
# --------------------------------------------------------------------------- #
def _bundle_seam(offline, tmp_path, *, exists=True, cfg=None):
    """Stub the bundle round-trip. `load_job_config`/`validate_job_config` are
    the real contract; here they hand back the canonical config a ticket needs."""
    cfg = cfg or {"version": 1, "name": "driftr3-v10-27b-gen",
                  "entrypoint": "run.sh", "timeout_s": 600}
    # `control`'s reconstruct staging path reads the CONSTANT `view._REPO_ROOT`
    # (control.py:172), not a `_repo_root()` call — the port hoisted the depth
    # computation to a module constant recomputed for the package layout
    # (jobs/view.py's banner rules on it). Same seam, different shape: patch the
    # value, not a callable.
    offline.setattr(jobs_view, "_REPO_ROOT", str(tmp_path))
    offline.setattr(jm, "bundle_exists", lambda sha, **k: exists)
    offline.setattr(jm, "download_bundle",
                    lambda sha, path, **k: (open(path, "wb").close(), (True, ""))[1])
    offline.setattr(jm, "extract_bundle",
                    lambda blob, dest, expect_sha=None, **k: expect_sha)
    offline.setattr(jm, "load_job_config", lambda d: dict(cfg))

    def _validate(raw, d, *, materialized=False):
        # ASSERTED, not merely tolerated: the tree here came out of a tar, so
        # its `includes:` are already files. Validating it as an authoring tree
        # is what broke reconstruct for every migrated bundle — the real
        # `resolve_includes` refuses "declared AND present". A double that
        # accepted either way would let that regress silently.
        assert materialized is True, (
            "reconstruct must validate the EXTRACTED tree as materialized")
        return (dict(cfg), [])

    offline.setattr(jm, "validate_job_config", _validate)
    return cfg


def test_reconstruct_mints_a_ticket_from_the_submitted_bundle(offline, tmp_path,
                                                              capsys):
    """The 2026-08-08 hand-recovery, promoted into the CLI: same JOB_ID,
    `bundle_sha256` off the folded `submitted` event, config out of the bundle,
    and `retargeted_from` stamped so the new box's jobd pulls the checkpoints
    back instead of restarting from zero."""
    q = Q(offline, _interrupted_view(), tickets={_OLD: None}, queued=[])
    cfg = _bundle_seam(offline, tmp_path)
    control.cmd_job_retarget(_args(from_box=_OLD, reconstruct=True))

    assert len(q.writes) == 1
    t = q.writes[0]
    assert t["job_id"] == _JID and t["box"] == _NEW
    assert t["bundle_sha256"] == _SHA               # from the `submitted` event
    assert t["config"] == cfg
    assert t["retargeted_from"] == _OLD             # checkpoint continuity
    assert q.deletes == []                          # nothing stale to sweep
    assert [e for _j, e, _f in q.events] == ["retargeted"]
    out = capsys.readouterr().out
    assert "RECONSTRUCTED from bundle" in out
    assert "`--env` pins are NOT in the bundle" in out   # the loud warning


def test_reconstruct_is_opt_in_only(offline, tmp_path):
    """Without the flag the same world is an error, not a silent mint: a
    reconstruction that drops submit-time --env pins is a DIFFERENT run."""
    q = Q(offline, _interrupted_view(), tickets={_OLD: None}, queued=[])
    _bundle_seam(offline, tmp_path)
    with pytest.raises(SystemExit):
        control.cmd_job_retarget(_args(from_box=_OLD))
    assert q.writes == []


def test_reconstruct_refuses_while_a_ticket_still_exists_on_a_live_box(offline,
                                                                      tmp_path):
    """THE double-run hazard. The queue listing names a box but the per-box read
    came back None (a failed/raced `cat`) — minting a second pointer would queue
    the job twice, and on a LIVE box both would run."""
    q = Q(offline, _interrupted_view(), tickets={_OLD: None, _THIRD: None},
          queued=[(_THIRD, _JID)], live={_THIRD})
    _bundle_seam(offline, tmp_path)
    with pytest.raises(SystemExit) as e:
        control.cmd_job_retarget(_args(from_box=_OLD, reconstruct=True))
    msg = str(e.value)
    assert "refusing to --reconstruct" in msg
    assert f"{_THIRD} is LIVE" in msg
    assert f"--from {_THIRD}" in msg                 # names the correct next step
    assert q.writes == []


def test_reconstruct_refuses_without_a_bundle_sha(offline, tmp_path):
    v = _view(_ev("submitted", _T(1), actor="cli:h", box=_OLD),   # no sha
              _ev("claimed", _T(2), instance_id=_OLD))
    q = Q(offline, v, tickets={_OLD: None}, queued=[])
    _bundle_seam(offline, tmp_path)
    with pytest.raises(SystemExit) as e:
        control.cmd_job_retarget(_args(from_box=_OLD, reconstruct=True))
    assert "no `submitted` event carrying a bundle_sha256" in str(e.value)
    assert q.writes == []


def test_reconstruct_refuses_when_the_bundle_object_is_gone(offline, tmp_path):
    q = Q(offline, _interrupted_view(), tickets={_OLD: None}, queued=[])
    _bundle_seam(offline, tmp_path, exists=False)
    with pytest.raises(SystemExit) as e:
        control.cmd_job_retarget(_args(from_box=_OLD, reconstruct=True))
    assert "is not on B2" in str(e.value)
    assert "job submit" in str(e.value)
    assert q.writes == []


def test_reconstruct_verifies_the_downloaded_bundle_against_the_recorded_sha(
        offline, tmp_path):
    """`extract_bundle(expect_sha=...)` is the integrity gate, and it must be
    handed the sha the `submitted` event recorded — not the object name."""
    seen = {}
    Q(offline, _interrupted_view(), tickets={_OLD: None}, queued=[])
    _bundle_seam(offline, tmp_path)
    offline.setattr(jm, "extract_bundle",
                    lambda blob, dest, expect_sha=None, **k: seen.setdefault(
                        "expect", expect_sha))
    control.cmd_job_retarget(_args(from_box=_OLD, reconstruct=True))
    assert seen["expect"] == _SHA


# --------------------------------------------------------------------------- #
# 4. the stale-pointer sweep never touches the target's own queue
# --------------------------------------------------------------------------- #
def test_the_sweep_refuses_to_delete_the_target_ticket(offline):
    """Belt-and-suspenders, mirroring `cmd_job_orphans`' pre-write box check: a
    future refactor of the scan must not be able to delete the pointer the move
    just wrote."""
    with pytest.raises(SystemExit) as e:
        control._retarget_drop_stale(_args(), _JID, _NEW, {_NEW})
    assert "is the retarget TARGET" in str(e.value)


def test_a_failed_delete_warns_and_does_not_abort(offline, capsys):
    q = Q(offline, _interrupted_view(), tickets={_OLD: _ticket()})
    offline.setattr(jm, "delete_ticket",
                    lambda box, jid, **k: (False, "rclone: 403"))
    control.cmd_job_retarget(_args(from_box=_OLD))
    assert len(q.writes) == 1                       # the move still landed
    out = capsys.readouterr().out
    assert "old ticket delete failed" in out and "double-run" in out


# --------------------------------------------------------------------------- #
# 5. the programmatic core keeps its resumable-no-op contract (workflowctl)
# --------------------------------------------------------------------------- #
def test_retarget_ticket_still_no_ops_on_a_missing_source():
    """`workflowctl`'s box-loss recovery depends on this: a crash between
    box-launch and ticket-move must be a resumable no-op, not fatal. The CLI's
    new scan/reconstruct behavior is deliberately NOT pushed down here."""
    calls = []

    def runner(argv, **k):
        calls.append(argv[0])
        return 1, "", "not found"

    res = jm.retarget_ticket(_JID, _OLD, _NEW, runner=runner, bucket="test-bkt")
    assert res == {"status": "no_ticket", "job_id": _JID}
    assert "rcat" not in calls and "deletefile" not in calls


# --------------------------------------------------------------------------- #
# 6. the fleetd WAKE — a moved ticket re-arms the destination's standing watch
# --------------------------------------------------------------------------- #
# A STANDING jobs watch re-arms on a TICKET, and until 2026-08-27 the only thing
# that could see one was the daemon's own queue poll — which reads nothing at all
# while the box is parked and `unknown` when the B2 listing will not answer.
# Measured on the live daemon that night: `jobs_watch_standing_resumed` had fired
# 0 times against 84 drains. Tickets were retargeted onto a drained standing box,
# it was evicted minutes later, and the dormant watch journaled nothing — no bid
# rescue, no replacement ladder, the work stranded on an exited box.
def _wakes(offline):
    seen = []
    offline.setattr(control.fleet_client, "fleet_ticket_placed",
                    lambda box, jid=None, **kw: seen.append((str(box), jid, kw)))
    return seen


def test_a_retarget_tells_fleetd_the_destination_now_holds_a_ticket(offline):
    wakes = _wakes(offline)
    Q(offline, _interrupted_view(), tickets={_OLD: _ticket()})
    control.cmd_job_retarget(_args(from_box=_OLD))
    assert wakes == [(_NEW, _JID, {"source": "job retarget"})]


def test_a_dry_run_wakes_nothing(offline):
    """Same rule as the watch hint: no ticket was written, so there is nothing
    to wake a ladder over — and waking one would re-enter it against a queue
    that has not changed."""
    wakes = _wakes(offline)
    Q(offline, _interrupted_view(), tickets={_OLD: _ticket()})
    control.cmd_job_retarget(_args(from_box=_OLD, dry_run=True))
    assert wakes == []


def test_an_idempotent_retarget_wakes_nothing_either(offline):
    """The ticket was already where we were asked to put it (usually fleetd's
    own replacement move), so this command wrote nothing."""
    wakes = _wakes(offline)
    Q(offline, _interrupted_view(), tickets={}, queued=[(_NEW, _JID)])
    control.cmd_job_retarget(_args(from_box=_OLD))
    assert wakes == []


def test_the_watch_hint_is_silent_where_a_spend_capable_watch_exists(offline,
                                                                     capsys):
    """"Arm the ladder AFTER the tickets exist" is wrong advice onto a box whose
    standing watch this very ticket just woke — and re-arming is not free:
    `fleet watch` states the whole watch, which is how a cap gets granted a
    second time."""
    _wakes(offline)
    offline.setattr(control.fleet_client, "fleet_watch_supervision",
                    lambda box: ("policy", {"profile": "jobs", "standing": True}))
    Q(offline, _interrupted_view(), tickets={_OLD: _ticket()})
    control.cmd_job_retarget(_args(from_box=_OLD))
    assert "no bid defense" not in capsys.readouterr().out
