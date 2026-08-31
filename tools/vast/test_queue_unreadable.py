"""A FAILED queue listing must never be readable as "this box has no work".

`jobmeta.list_queue` returned `[]` when its `rclone lsf` exited non-zero, so a
broken rclone config, a revoked key, a network partition and a B2 outage all
folded to the same answer as a genuinely empty queue. On 2026-08-22 that
disarmed the eviction-replacement ladder: fleetd's own event named the pending
job while the tick that decides whether to rescue it saw no work at all, and a
spot box was lost with a live ticket. Write-up: <upstream-bench>
`archive/runs/2026-08-22-wave3-p0c-9b-thinking-pilot/analysis/FLEETD_BLIND_QUEUE.md`.

The listing is now a tri-state at every consumer: tickets, none, or unreadable.
"""
import argparse

import pytest

import jobmeta
from vastlib.fleet import hooks
from vastlib.jobs import control, scan, view
from vastlib.storage import b2

BUCKET = "bkt"
DNS_ERR = "dial tcp: lookup example.invalid: no such host"

# Captured at import: every helper below monkeypatches these NAMES on the module
# the consumer reads, so calling through the name would recurse into the stub.
_REAL_LIST_QUEUE = jobmeta.list_queue
_REAL_LIST_ALL_QUEUED = jobmeta.list_all_queued


def _fail(rc=1, err=DNS_ERR):
    """A jobmeta runner whose listing FAILS — the live 2026-08-22 shape."""
    def _runner(args, **kw):
        return rc, "", err
    return _runner


def _ok(lines=()):
    def _runner(args, **kw):
        return 0, "".join(f"{ln}\n" for ln in lines), ""
    return _runner


# --------------------------------------------------------------------------- #
# 1. the primitive
# --------------------------------------------------------------------------- #

def test_list_queue_raises_instead_of_reporting_an_empty_queue():
    with pytest.raises(jobmeta.QueueUnreadable) as e:
        jobmeta.list_queue("48392137", runner=_fail(), bucket=BUCKET)
    assert "48392137" in str(e.value)
    assert DNS_ERR in str(e.value), "the transport error must reach the operator"


def test_list_all_queued_raises_instead_of_reporting_an_empty_fleet():
    with pytest.raises(jobmeta.QueueUnreadable) as e:
        jobmeta.list_all_queued(runner=_fail(), bucket=BUCKET)
    assert DNS_ERR in str(e.value)


def test_a_genuinely_empty_queue_is_still_an_empty_list():
    """The control. Without it "raise on empty" would pass every test above."""
    assert jobmeta.list_queue("41", runner=_ok(), bucket=BUCKET) == []
    assert jobmeta.list_all_queued(runner=_ok(), bucket=BUCKET) == []


def test_a_populated_queue_is_unchanged():
    assert jobmeta.list_queue("41", runner=_ok(["j-b.json", "j-a.json"]),
                              bucket=BUCKET) == ["j-a", "j-b"]
    assert jobmeta.list_all_queued(runner=_ok(["41/j-a.json"]),
                                   bucket=BUCKET) == [("41", "j-a")]


def test_QueueUnreadable_is_not_a_JobmetaError():
    """A dozen CLI paths `except jobmeta.JobmetaError` and exit tidily; making
    this one of those would re-hide the transport failure it exists to show."""
    assert issubclass(jobmeta.QueueUnreadable, RuntimeError)
    assert not issubclass(jobmeta.QueueUnreadable, jobmeta.JobmetaError)


# --------------------------------------------------------------------------- #
# 2. fleetd — the consumer that decides whether to keep defending a box
# --------------------------------------------------------------------------- #

def _hooked(monkeypatch, runner):
    """Drive the REAL `list_queue` (with a stubbed runner) through the REAL
    fleetd hook, so the tri-state is tested across the seam, not either side."""
    monkeypatch.setattr(hooks.jobmeta, "list_queue",
                        lambda iid: _REAL_LIST_QUEUE(iid, runner=runner,
                                                     bucket=BUCKET))


def _hooks_obj():
    return hooks.Hooks(dry_run=True)


def test_fleetd_drained_says_UNKNOWN_not_TRUE_on_a_failed_listing(monkeypatch):
    """The money assertion. `drained() is True` is what lets fleetd stop
    defending a box; unfixed, a failed listing produced exactly that."""
    _hooked(monkeypatch, _fail())
    assert _hooks_obj().drained(48392137) is None


def test_fleetd_drained_is_still_TRUE_on_a_genuinely_empty_queue(monkeypatch):
    """The control: drain IS provable from a READ absence."""
    _hooked(monkeypatch, _ok())
    assert _hooks_obj().drained(48392137) is True


def test_fleetd_results_present_stays_unknown_either_way(monkeypatch):
    for runner in (_fail(), _ok()):
        _hooked(monkeypatch, runner)
        assert _hooks_obj().results_present(48392137) is None


# --------------------------------------------------------------------------- #
# 3. the operator-facing listings — a refusal, never a clean bill of health
# --------------------------------------------------------------------------- #

def _cli_env(monkeypatch, runner):
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(view, "_live_iids_set", lambda: set())
    monkeypatch.setattr(view, "_present_iids_set", lambda: set())
    monkeypatch.setattr(scan, "fold_many", lambda jids, live_iids=(): {})
    monkeypatch.setattr(jobmeta, "list_all_queued",
                        lambda **k: _REAL_LIST_ALL_QUEUED(runner=runner,
                                                          bucket=BUCKET))
    monkeypatch.setattr(jobmeta, "list_queue",
                        lambda box, **k: _REAL_LIST_QUEUE(box, runner=runner,
                                                          bucket=BUCKET))


@pytest.mark.parametrize("box", [None, "48392137"])
def test_job_ls_refuses_rather_than_print_no_queued_jobs(monkeypatch, capsys, box):
    """`no queued jobs.` off a failed listing is the operator-facing half of the
    same defect: it reads as a clean, empty fleet."""
    _cli_env(monkeypatch, _fail())
    with pytest.raises(SystemExit) as e:
        view.cmd_job_ls(argparse.Namespace(box=box))
    assert DNS_ERR in str(e.value)
    assert "no queued jobs." not in capsys.readouterr().out


def test_job_ls_still_says_no_queued_jobs_when_the_queue_really_is_empty(
        monkeypatch, capsys):
    _cli_env(monkeypatch, _ok())
    view.cmd_job_ls(argparse.Namespace(box=None))
    assert "no queued jobs." in capsys.readouterr().out


def test_retarget_refuses_to_conclude_a_ticket_is_nowhere(monkeypatch):
    """`_retarget_queued_boxes` feeds a path that DELETES queue pointers and can
    rebuild a ticket. "Not queued anywhere" is not a conclusion an unreadable
    listing can support."""
    monkeypatch.setattr(control.jobmeta, "list_all_queued",
                        lambda **k: _REAL_LIST_ALL_QUEUED(runner=_fail(),
                                                          bucket=BUCKET))
    with pytest.raises(SystemExit) as e:
        control._retarget_queued_boxes("20260822T115852-p0c-9b-think-pilot-3733")
    assert "Refusing to retarget on an unreadable queue" in str(e.value)


def test_retarget_still_answers_empty_when_the_queue_is_readable(monkeypatch):
    monkeypatch.setattr(control.jobmeta, "list_all_queued",
                        lambda **k: _REAL_LIST_ALL_QUEUED(runner=_ok(),
                                                          bucket=BUCKET))
    assert control._retarget_queued_boxes("j-a") == []


# --------------------------------------------------------------------------- #
# 4. the pin: nothing may reintroduce the fail-open shape
# --------------------------------------------------------------------------- #

def test_no_queue_lister_returns_an_empty_list_on_a_nonzero_rc():
    import inspect
    for fn in (_REAL_LIST_QUEUE, _REAL_LIST_ALL_QUEUED):
        src = inspect.getsource(fn)
        assert "raise QueueUnreadable" in src, fn.__name__
        assert "return []" not in src, (
            f"{fn.__name__} fails open again: a failed listing must not be "
            f"spelled the same as an empty queue")
