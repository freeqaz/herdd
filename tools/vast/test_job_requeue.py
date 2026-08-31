"""Portable tests for `herdd job requeue` — the one-command re-open of a
TERMINAL-FAILED job (task #108).

Runs in the toolchain-free lane (`pytest -m "not integration"`): NO vast API, NO
B2/rclone, NO network, NO creds. The fold views are built by the REAL
`jobmeta.fold_events` over synthetic event bodies (so a change to the un-stick
rule shows up here), and the only stubs are the two seams that would touch the
network — `_ensure_b2_remote` / `_live_iids_set` — plus the jobmeta calls a test
deliberately stops short of.

What it pins, and why each one is a fail-closed gate rather than a warning:

  * STATUS. requeue re-opens `failed` ONLY. `done`/`cancelled` are sticky by
    design, a running/queued job would be DOUBLE-RUN, and an already-requeued
    job is live work.
  * BUNDLE IDENTITY. Same JOB_ID means the same event log and the same
    checkpoints/ prefix; a drifted bundle would silently mix two experiments
    under one id. There is deliberately no --allow-bundle-drift.
  * TARGET BOX. The box that failed the job carries a local terminal breadcrumb
    its jobd checks before any B2 read, so a ticket requeued back onto it would
    be skipped forever, silently.

The jobd half (honouring the requeue mark over the prior attempt's
results.DONE.json, and the loop bound) lives in test_jobd.py; the ticket/event
core lives in test_jobmeta.py.
"""
import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobmeta as jm  # noqa: E402
import herdd as vc  # noqa: E402  (path anchor for the real-argparse --help drive)
from vastlib.jobs import control, view  # noqa: E402
from vastlib.storage import b2  # noqa: E402


_JID = "20260730T101112-data-rb3-wide-4cd4"
_OLD, _NEW = "44", "55"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ev(event, ts, **fields):
    d = {"v": 1, "ts": ts, "actor": "box:44", "event": event, "job_id": _JID,
         "nonce": ts[-4:] + event[:2]}
    d.update(fields)
    return d


def _T(n):
    return f"20260730T1011{n:02d}000Z"


def _view(*evs, live=()):
    return jm.fold_events(list(evs), live_iids=live)


def _failed_view(sha="deadbeef", live=()):
    return _view(
        _ev("submitted", _T(1), actor="cli:h", box=_OLD, bundle_sha256=sha,
            name="data-rb3-wide", entrypoint="run.sh", timeout_s=600),
        _ev("claimed", _T(2), instance_id=_OLD),
        _ev("started", _T(3), instance_id=_OLD),
        _ev("failed", _T(4), instance_id=_OLD, rc=16,
            reason="rc=16"), live=live)


def _bundle(tmp_path, body="echo hi\n", name="bundle"):
    d = tmp_path / name
    d.mkdir()
    (d / "run.sh").write_text(body)
    (d / "job-config.yaml").write_text(
        "version: 1\nname: data-rb3-wide\nentrypoint: run.sh\ntimeout_s: 600\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n")
    return str(d)


def _args(bundle, **over):
    ns = argparse.Namespace(job_id=_JID, box=_NEW, bundle=bundle, from_box=None,
                            env=None, dry_run=False)
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture
def offline(monkeypatch):
    """Neutralize the two network seams. Anything else that would touch B2 is
    left UNPATCHED on purpose: a test that reaches it fails loudly."""
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(view, "_live_iids_set", lambda *a, **k: set())
    return monkeypatch


def _run(offline, view, bundle, **over):
    # `jm` IS the module `vastlib.jobs.control` imports as `jobmeta` (one shared
    # object), so patching it here steers the subject exactly as `vc.jobmeta` did.
    offline.setattr(jm, "read_job_fresh", lambda *a, **k: view)
    return control.cmd_job_requeue(_args(bundle, **over))


# --------------------------------------------------------------------------- #
# _requeue_refusal — the pure status policy
# --------------------------------------------------------------------------- #
def test_refusal_none_for_a_terminal_failed_job():
    assert control._requeue_refusal(_failed_view()) is None


def test_refusal_done_is_sticky():
    v = _view(_ev("submitted", _T(1), actor="cli:h", box=_OLD),
              _ev("started", _T(2), instance_id=_OLD),
              _ev("done", _T(3), instance_id=_OLD, rc=0))
    why = control._requeue_refusal(v)
    assert why and "done" in why and "STICKY" in why


def test_refusal_cancelled_is_sticky():
    v = _view(_ev("submitted", _T(1), actor="cli:h", box=_OLD),
              _ev("cancelled", _T(3), reason="operator"))
    why = control._requeue_refusal(v)
    assert why and "cancelled" in why and "never-revive" in why


def test_refusal_running_job():
    v = _view(_ev("submitted", _T(1), actor="cli:h", box=_OLD),
              _ev("claimed", _T(2), instance_id=_OLD),
              _ev("started", _T(3), instance_id=_OLD), live={_OLD})
    why = control._requeue_refusal(v)
    assert why and "running" in why and "retarget" in why


def test_refusal_queued_job():
    v = _view(_ev("submitted", _T(1), actor="cli:h", box=_OLD))
    why = control._requeue_refusal(v)
    assert why and "QUEUED" in why


def test_refusal_already_reopened():
    v = _view(*[_ev(e, _T(i + 1), instance_id=_OLD, actor="cli:h")
                for i, e in enumerate(("submitted", "claimed", "started"))],
              _ev("failed", _T(4), instance_id=_OLD, rc=16),
              _ev("resumed", _T(5), kind="requeue", instance_id=_NEW))
    assert v["reopened"] is True
    why = control._requeue_refusal(v)
    assert why and "already re-opened" in why


def test_refusal_unknown_job():
    why = control._requeue_refusal(_view())
    assert why and "no events" in why


# --------------------------------------------------------------------------- #
# cmd_job_requeue — the gates
# --------------------------------------------------------------------------- #
def test_cli_refuses_done(tmp_path, offline, capsys):
    v = _view(_ev("submitted", _T(1), actor="cli:h", box=_OLD,
                  bundle_sha256="x"),
              _ev("done", _T(3), instance_id=_OLD, rc=0))
    with pytest.raises(SystemExit) as e:
        _run(offline, v, _bundle(tmp_path))
    assert "refusing to requeue" in str(e.value) and "`done`" in str(e.value)


def test_cli_refuses_cancelled(tmp_path, offline):
    v = _view(_ev("submitted", _T(1), actor="cli:h", box=_OLD),
              _ev("cancelled", _T(3), reason="operator"))
    with pytest.raises(SystemExit) as e:
        _run(offline, v, _bundle(tmp_path))
    assert "cancelled" in str(e.value)


def test_cli_refuses_a_drifted_bundle_fail_closed(tmp_path, offline):
    """The sha the CLI recomputes is the SAME content address `job submit`
    records; a mismatch is refused with both hashes named. No override exists."""
    d = _bundle(tmp_path)
    v = _failed_view(sha="0" * 64)              # not this bundle's hash
    with pytest.raises(SystemExit) as e:
        _run(offline, v, d)
    msg = str(e.value)
    assert "bundle DRIFT" in msg
    assert jm.bundle_sha256(d) in msg and "0" * 64 in msg
    assert "job submit" in msg                  # names the correct next step


def test_cli_refuses_the_box_it_failed_on(tmp_path, offline):
    d = _bundle(tmp_path)
    v = _failed_view(sha=jm.bundle_sha256(d))
    with pytest.raises(SystemExit) as e:
        _run(offline, v, d, box=_OLD)
    assert "that is the box it failed on" in str(e.value)
    assert "local terminal cache" in str(e.value)


def test_cli_refuses_a_missing_bundle_dir(tmp_path, offline):
    with pytest.raises(SystemExit) as e:
        _run(offline, _failed_view(), str(tmp_path / "nope"))
    assert "not a directory" in str(e.value)


def test_cli_refuses_when_the_submit_event_has_no_sha(tmp_path, offline):
    v = _view(_ev("submitted", _T(1), actor="cli:h", box=_OLD),   # no sha
              _ev("failed", _T(4), instance_id=_OLD, rc=16))
    with pytest.raises(SystemExit) as e:
        _run(offline, v, _bundle(tmp_path))
    assert "no `submitted` event carrying a bundle_sha256" in str(e.value)


# --------------------------------------------------------------------------- #
# cmd_job_requeue — the happy path (composition over reimplementation)
# --------------------------------------------------------------------------- #
def test_cli_dry_run_makes_no_b2_mutation(tmp_path, offline, capsys):
    d = _bundle(tmp_path)
    offline.setattr(jm, "read_ticket", lambda *a, **k: None)
    def _boom(*a, **k):
        raise AssertionError("--dry-run must not write anything")
    offline.setattr(jm, "requeue_ticket", _boom)
    offline.setattr(jm, "bundle_exists", _boom)
    _run(offline, _failed_view(sha=jm.bundle_sha256(d)), d, dry_run=True)
    out = capsys.readouterr().out
    assert "[dry-run]" in out and "retargeted_from=44" in out
    assert jm.REQUEUE_TICKET_MARK in out


def test_cli_composes_requeue_ticket_reusing_the_surviving_ticket(tmp_path, offline,
                                                                  capsys):
    """When the original queue ticket survived, its config is reused VERBATIM —
    that is where the first submit's `--env` pins live (they are ticket-side; the
    bundle sha is invariant under them), so rebuilding from the bundle alone
    would silently drop them."""
    d = _bundle(tmp_path)
    sha = jm.bundle_sha256(d)
    cfg = {"version": 1, "name": "data-rb3-wide", "entrypoint": "run.sh",
           "timeout_s": 600, "env": {"WAVE": "A", "SHARD": "3"}}
    offline.setattr(jm, "read_ticket",
                    lambda box, jid, **k: {"box": box, "job_id": jid,
                                           "bundle_sha256": sha, "config": cfg})
    offline.setattr(jm, "bundle_exists", lambda *a, **k: True)
    seen = {}
    def _fake(jid, box, config, bundle_sha, **kw):
        seen.update(job_id=jid, box=box, config=config, sha=bundle_sha, **kw)
        return {"status": "requeued", "job_id": jid, "key": f"jobs/queue/{box}/{jid}.json",
                "old_ticket_deleted": True, "requeued_ts": "20260731T000000Z"}
    offline.setattr(jm, "requeue_ticket", _fake)

    _run(offline, _failed_view(sha=sha), d)

    assert seen["job_id"] == _JID and seen["box"] == _NEW
    assert seen["old_box"] == _OLD                 # from the fold, not guessed
    assert seen["sha"] == sha
    assert seen["config"] == cfg                   # env pins preserved
    assert seen["attempt"] == 2                    # prior attempts + 1
    out = capsys.readouterr().out
    assert "reused from the surviving queue ticket" in out
    assert "no copy needed" in out                 # checkpoints ride the JOB_ID
    # Retarget's sibling owes the same line: the ticket now exists on the new
    # box, which is both the earliest and the last convenient moment to arm the
    # ladder, and nothing else on this path will say so.
    assert f"fleet watch {_NEW} --profile jobs --budget <USD> --standing" in out


def test_cli_rebuilds_the_config_when_no_ticket_survived(tmp_path, offline, capsys):
    d = _bundle(tmp_path)
    sha = jm.bundle_sha256(d)
    offline.setattr(jm, "read_ticket", lambda *a, **k: None)
    offline.setattr(jm, "bundle_exists", lambda *a, **k: True)
    seen = {}
    offline.setattr(jm, "requeue_ticket",
                    lambda jid, box, config, s, **kw: seen.update(config=config, **kw)
                    or {"status": "requeued", "job_id": jid, "key": "k",
                        "old_ticket_deleted": None, "requeued_ts": "t"})
    _run(offline, _failed_view(sha=sha), d, env=["WAVE=A"])
    assert seen["config"]["env"]["WAVE"] == "A"    # re-applied pin rides the ticket
    out = capsys.readouterr().out
    assert "REBUILT from" in out
    assert "`--env` pins are NOT in the bundle" in out   # the loud caveat


def test_cli_rejects_env_when_the_original_ticket_survived(tmp_path, offline):
    d = _bundle(tmp_path)
    sha = jm.bundle_sha256(d)
    offline.setattr(jm, "read_ticket",
                    lambda box, jid, **k: {"box": box, "config": {"name": "x"}})
    with pytest.raises(SystemExit) as e:
        _run(offline, _failed_view(sha=sha), d, env=["WAVE=A"])
    assert "--env is only for the REBUILT-from-bundle path" in str(e.value)


def test_cli_reuploads_a_missing_bundle_object(tmp_path, offline):
    """Cheap insurance: the bundle object should already be on B2 from the first
    submit, but if it expired the box would die at "bundle download failed"."""
    d = _bundle(tmp_path)
    sha = jm.bundle_sha256(d)
    offline.setattr(jm, "read_ticket", lambda *a, **k: None)
    offline.setattr(jm, "bundle_exists", lambda *a, **k: False)
    # `control`'s reconstruct staging path reads the CONSTANT `view._REPO_ROOT`
    # (same seam test_job_retarget.py patches) — the old `vc._repo_root` patch
    # became a DEAD patch after migration and the test wrote a REAL tarball
    # into the repo's out/jobs/_bundles (caught by the 6e verifier). Patch the
    # value, not a callable.
    offline.setattr(view, "_REPO_ROOT", str(tmp_path / "repo"))
    uploaded = {}
    offline.setattr(jm, "upload_bundle",
                    lambda path, s, **k: (uploaded.update(path=path, sha=s), (True, ""))[1])
    offline.setattr(jm, "requeue_ticket",
                    lambda *a, **k: {"status": "requeued", "job_id": _JID, "key": "k",
                                     "old_ticket_deleted": None, "requeued_ts": "t"})
    _run(offline, _failed_view(sha=sha), d)
    assert uploaded["sha"] == sha
    assert uploaded["path"].endswith(f"{sha}.tar.zst")


def test_cli_warns_when_the_stale_ticket_delete_failed(tmp_path, offline, capsys):
    d = _bundle(tmp_path)
    sha = jm.bundle_sha256(d)
    offline.setattr(jm, "read_ticket", lambda *a, **k: None)
    offline.setattr(jm, "bundle_exists", lambda *a, **k: True)
    offline.setattr(jm, "requeue_ticket",
                    lambda *a, **k: {"status": "requeued", "job_id": _JID, "key": "k",
                                     "old_ticket_deleted": False,
                                     "delete_err": "403", "requeued_ts": "t"})
    _run(offline, _failed_view(sha=sha), d)
    out = capsys.readouterr().out
    assert "double-run" in out and "delete jobs/queue/44/" in out


def test_cli_is_registered_and_documents_the_gates():
    """The subcommand exists in the real argparse tree (an agent reads --help)."""
    import subprocess
    r = subprocess.run([sys.executable, vc.__file__, "job", "requeue", "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    for token in ("--box", "--bundle", "--from", "--dry-run", "hash IDENTICALLY"):
        assert token in r.stdout, f"{token} missing from `job requeue --help`"
    # no drift override in the FLAG SET (a drifted bundle is a different
    # experiment — the docstring names the decision, the parser must not offer it)
    assert "allow-bundle-drift" not in r.stdout.split("options:", 1)[-1]


# --------------------------------------------------------------------------- #
# the fleetd WAKE — a re-opened job re-arms the destination's standing watch
# --------------------------------------------------------------------------- #
# Retarget's sibling seam, and the same 2026-08-27 defect: a STANDING jobs watch
# re-arms on a TICKET, and the only thing that could see one was the daemon's own
# queue poll — silent on a parked box, `unknown` on a B2 blip. It had fired 0
# times against 84 drains. `requeue` places a live ticket exactly as `retarget`
# does, so it owes the same announcement.
def test_a_requeue_tells_fleetd_the_destination_now_holds_a_ticket(tmp_path,
                                                                   offline):
    d = _bundle(tmp_path)
    seen = []
    offline.setattr(control.fleet_client, "fleet_ticket_placed",
                    lambda box, jid=None, **kw: seen.append((str(box), jid, kw)))
    offline.setattr(jm, "read_ticket", lambda *a, **k: None)
    offline.setattr(jm, "bundle_exists", lambda *a, **k: True)
    offline.setattr(jm, "requeue_ticket",
                    lambda *a, **k: {"status": "requeued", "job_id": _JID,
                                     "key": "k", "old_ticket_deleted": True,
                                     "requeued_ts": "t"})
    _run(offline, _failed_view(sha=jm.bundle_sha256(d)), d)
    assert seen == [(_NEW, _JID, {"source": "job requeue"})]


def test_a_requeue_dry_run_wakes_nothing(tmp_path, offline):
    d = _bundle(tmp_path)
    seen = []
    offline.setattr(control.fleet_client, "fleet_ticket_placed",
                    lambda box, jid=None, **kw: seen.append((str(box), jid, kw)))
    offline.setattr(jm, "read_ticket", lambda *a, **k: None)
    _run(offline, _failed_view(sha=jm.bundle_sha256(d)), d, dry_run=True)
    assert seen == []
