"""`vastlib.boxes.remote` — the two transports, and the trap they share.

Why this file exists
--------------------
`ok=True` here means **the transport worked**, never "the remote command
succeeded". vast's `execute` endpoint answers HTTP 200 with the command's
stderr in the body and never surfaces an exit code, so `_ssh_exec_soft` mirrors
that on purpose: a failing remote `ls` comes back `ok=True` with its error text
concatenated into `data`. `salvage.survey_dest_files` then splits
absent/unreadable/listing off that TEXT — which is why "improving" `ok` to mean
command success, or normalising the failure payload from `""` to `None`, would
collapse a three-outcome verification into two and make `unverifiable` nearly
unreachable. Every assertion below that looks pedantic about `data == ""` is
pinning exactly that.

The second thing under test is the nonce guard. `result_url` is a FIXED
PER-INSTANCE log path, so a naive read can return another caller's output or
this caller's own previous output — and salvage treats that listing as its
ORACLE. The tests cover both arms: the correlated path (refuse any body without
this call's nonce) and the DEGRADED fallback (vast refused the bracketed shape
-> bare command + read-twice-and-compare, announced on stderr).

`_EXEC_NONCE_TAG` stays the literal `herdd_exec`. It is written into a live
box's log that a half-migrated tree may still be reading with the old code, and
the suite extracts it with `__herdd_exec_([0-9a-f]+)_BEGIN__` — the regex
below is the same one, deliberately.

What is deliberately NOT here
-----------------------------
* No live API and no network. `_vast_execute_soft` and `_vast_copy_direct_soft`
  are PUTs: every test stubs `vastlib.core.api.request_soft` as a MODULE
  ATTRIBUTE, and one test deliberately does not, to show that what answers an
  unstubbed PUT is the conftest guard rather than vast.
* No re-testing of `salvage`'s parsing (`test_vastlib_boxes_salvage.py` owns
  it) and no repoint of `test_salvage.py`, whose 31 deferred-seam patches still
  drive `herdd`'s live copies.
* No assertion on the nonce VALUE (it is `secrets.token_hex`), only on its
  shape and on the fact that both markers carry the same one.

Provenance: created 2026-08-16 alongside `vastlib/boxes/remote.py`, plan §8
step 3.
"""

from __future__ import annotations

import io
import re
import subprocess
import sys
from contextlib import redirect_stderr
from pathlib import Path

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

from vastlib.boxes import lifecycle, remote, ssh       # noqa: E402
from vastlib.core import api                           # noqa: E402

_NONCE_RE = re.compile(r"__herdd_exec_([0-9a-f]+)_BEGIN__")


class _Resp:
    """Minimal urlopen context manager."""

    def __init__(self, body, status=200):
        self._body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body.encode()


def _urlopen_returning(*bodies):
    """A urlopen stub that yields each body in turn, repeating the last."""
    seq = list(bodies)

    def _open(url, timeout=None):
        return _Resp(seq.pop(0) if len(seq) > 1 else seq[0])

    return _open


def _put_ok(result_url="https://example.invalid/log", **extra):
    payload = {"success": True, "result_url": result_url}
    payload.update(extra)

    def _stub(method, path, body=None, *a, **k):
        return True, payload, None

    return _stub


# --------------------------------------------------------------------------- #
# the nonce protocol
# --------------------------------------------------------------------------- #
def test_nonce_markers_share_one_hex_nonce_and_keep_the_frozen_tag():
    begin, end = remote._exec_nonce_markers()
    m = _NONCE_RE.match(begin)
    assert m, begin
    assert end == f"__herdd_exec_{m.group(1)}_END__"
    assert remote._EXEC_NONCE_TAG == "herdd_exec"
    assert len(m.group(1)) >= 24                       # cannot collide with a name


def test_exec_wrap_uses_semicolons_so_markers_survive_a_failing_command():
    assert remote._exec_wrap("ls /nope", "B", "E") == "echo B; ls /nope; echo E"
    assert "&&" not in remote._exec_wrap("ls", "B", "E")


def test_extract_takes_the_last_begin_block_and_matches_by_containment():
    text = "\n".join(["/w/B", "stale", "/w/E", "/w/B", "fresh", "/w/E"])
    assert remote._exec_extract_nonce_block(text, "B", "E") == "fresh"


def test_extract_returns_none_for_an_incomplete_or_foreign_body():
    assert remote._exec_extract_nonce_block("B\npartial", "B", "E") is None
    assert remote._exec_extract_nonce_block("someone else's output", "B", "E") is None


@pytest.mark.parametrize("err,expected", [
    ("400 invalid_args: unsupported", True),
    ("HTTP 400 on PUT: not allowed", True),
    ("HTTP 404 on PUT: instance not found", False),      # the box is GONE
    ("network timeout", False),
])
def test_refusal_classification_never_falls_back_on_a_404(err, expected):
    assert remote._exec_refusal_is_validation(err) is expected


# --------------------------------------------------------------------------- #
# _vast_execute_soft
# --------------------------------------------------------------------------- #
def test_execute_returns_the_nonce_block_and_strips_writeable_path(monkeypatch):
    monkeypatch.setattr(api, "request_soft",
                        _put_ok(writeable_path="/workspace/"))
    sent = {}

    def _capture(method, path, body=None, *a, **k):
        sent["path"] = path
        sent["command"] = body["command"]
        return True, {"success": True, "result_url": "u",
                      "writeable_path": "/workspace/"}, None

    monkeypatch.setattr(api, "request_soft", _capture)
    monkeypatch.setattr(remote.urllib.request, "urlopen", lambda *a, **k: None)

    def _later(url, timeout=None):
        n = _NONCE_RE.search(sent["command"]).group(1)
        return _Resp(f"__herdd_exec_{n}_BEGIN__\n"
                     f"/workspace/out/checkpoint-50\n"
                     f"__herdd_exec_{n}_END__\n")

    monkeypatch.setattr(remote.urllib.request, "urlopen", _later)
    ok, data, err = remote._vast_execute_soft(41, "ls -lR /w", _sleep=lambda s: None)
    assert (ok, err) == (True, None)
    assert data == "out/checkpoint-50"                 # prefix stripped, anchored
    assert sent["path"] == "/v0/instances/command/41/"


def test_execute_keeps_polling_past_a_body_that_is_not_ours(monkeypatch):
    monkeypatch.setattr(api, "request_soft", _put_ok())
    monkeypatch.setattr(remote.urllib.request, "urlopen",
                        _urlopen_returning("someone else's listing\n"))
    ok, data, err = remote._vast_execute_soft(
        41, "ls", tries=2, _sleep=lambda s: None, _now=lambda: 0.0)
    assert ok is False
    assert data == ""                                  # NOT None (shape-A trap)
    assert "does not carry this call's nonce" in err


def test_execute_degrades_to_read_twice_when_vast_refuses_the_wrapped_shape(
        monkeypatch):
    calls = []

    def _stub(method, path, body=None, *a, **k):
        calls.append(body["command"])
        if len(calls) == 1:
            return False, None, "HTTP 400 invalid_args: unsupported"
        return True, {"success": True, "result_url": "u"}, None

    monkeypatch.setattr(api, "request_soft", _stub)
    monkeypatch.setattr(remote.urllib.request, "urlopen",
                        _urlopen_returning("body-A\n", "body-A\n"))
    err_io = io.StringIO()
    with redirect_stderr(err_io):
        ok, data, err = remote._vast_execute_soft(
            41, "ls -lR /w", _sleep=lambda s: None, _now=lambda: 0.0)
    assert (ok, err) == (True, None)
    # The WHOLE body, newline and all: with no `writeable_path` there is
    # nothing to strip, and degraded mode returns what it read rather than a
    # nonce-delimited block.
    assert data == "body-A\n"
    assert calls[1] == "ls -lR /w"                     # the BARE command
    assert "WEAKER guard" in err_io.getvalue()         # said out loud, on stderr


def test_execute_reports_a_success_without_a_result_url(monkeypatch):
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (True, {"success": True}, None))
    ok, data, err = remote._vast_execute_soft(41, "ls", _sleep=lambda s: None)
    assert (ok, data) == (False, "")
    assert "no result_url" in err


def test_execute_surfaces_the_api_refusal_text(monkeypatch):
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (True, {"success": False,
                                                "msg": "only avail on stopped"},
                                         None))
    ok, data, err = remote._vast_execute_soft(41, "ls", _sleep=lambda s: None)
    assert (ok, data, err) == (False, "", "only avail on stopped")


def test_unstubbed_put_is_refused_by_the_conftest_guard():
    """The PUT never reaches vast: the autouse guard answers it (H4)."""
    ok, data, err = remote._vast_execute_soft(41, "ls", tries=1,
                                              _sleep=lambda s: None)
    assert (ok, data) == (False, "")
    assert "test isolation" in err


# --------------------------------------------------------------------------- #
# _strip_writeable_path / _int_or / copy_direct
# --------------------------------------------------------------------------- #
def test_strip_is_anchored_to_line_starts():
    text = "/w/a\nb/w/c\n"
    assert remote._strip_writeable_path(text, "/w/") == "a\nb/w/c"
    assert remote._strip_writeable_path(text, "") == text


def test_int_or_passes_a_non_numeric_id_through():
    assert remote._int_or("41") == 41
    assert remote._int_or("C.41") == "C.41"
    assert remote._int_or(None) is None


def test_copy_direct_ok_means_only_that_vast_accepted_it(monkeypatch):
    sent = {}

    def _stub(method, path, body=None, *a, **k):
        sent.update(method=method, path=path, body=body)
        return True, {"success": True, "msg": "started"}, None

    monkeypatch.setattr(api, "request_soft", _stub)
    ok, msg, err = remote._vast_copy_direct_soft(41, "/src", "42", "/dst")
    assert (ok, msg, err) == (True, "started", None)
    assert sent["method"] == "PUT"
    assert sent["body"]["src_id"] == 41 and sent["body"]["dst_id"] == 42


def test_copy_direct_failure_carries_an_empty_data_slot(monkeypatch):
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (False, None, "HTTP 404 …"))
    ok, msg, err = remote._vast_copy_direct_soft(41, "/src", 42, "/dst")
    assert (ok, msg) == (False, "")
    assert err == "HTTP 404 …"


# --------------------------------------------------------------------------- #
# _ssh_exec_soft — the transport-only-ok contract, verbatim
# --------------------------------------------------------------------------- #
def _wire_ssh(monkeypatch, proc):
    monkeypatch.setattr(lifecycle, "_get_instance",
                        lambda iid: {"id": iid, "actual_status": "running"})
    monkeypatch.setattr(ssh, "_pick_ssh_endpoint",
                        lambda i, **k: ("1.2.3.4", 40001, "direct"))
    monkeypatch.setattr(remote.subprocess, "run", lambda *a, **k: proc)


class _Run:
    def __init__(self, rc, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = rc, stdout, stderr


def test_a_failing_remote_command_is_still_transport_ok(monkeypatch):
    _wire_ssh(monkeypatch, _Run(2, "", "ls: cannot access '/x': No such file"))
    ok, text, err = remote._ssh_exec_soft(41, "ls -lR /x")
    assert ok is True and err is None                  # the TRANSPORT worked
    assert "No such file" in text                      # stderr lands in `data`


def test_stdout_and_stderr_are_concatenated_in_that_order(monkeypatch):
    _wire_ssh(monkeypatch, _Run(0, "OUT", "ERR"))
    assert remote._ssh_exec_soft(41, "x").data == "OUTERR"


def test_rc_255_is_the_only_shape_that_flips_ok(monkeypatch):
    _wire_ssh(monkeypatch, _Run(255, "", "Permission denied (publickey)."))
    ok, text, err = remote._ssh_exec_soft(41, "x")
    assert (ok, text) == (False, "")
    assert "publickey" in err


def test_a_missing_instance_becomes_a_soft_failure(monkeypatch):
    def _boom(iid):
        raise SystemExit("error: HTTP 404")

    monkeypatch.setattr(lifecycle, "_get_instance", _boom)
    ok, text, err = remote._ssh_exec_soft(41, "x")
    assert (ok, text) == (False, "")
    assert err == "instance 41 is not listed"


def test_no_endpoint_names_the_status(monkeypatch):
    monkeypatch.setattr(lifecycle, "_get_instance",
                        lambda iid: {"actual_status": "loading"})
    monkeypatch.setattr(ssh, "_pick_ssh_endpoint", lambda i, **k: (None, None, None))
    ok, text, err = remote._ssh_exec_soft(41, "x")
    assert (ok, text) == (False, "")
    assert "loading" in err


def test_a_timeout_is_a_soft_failure(monkeypatch):
    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=180)

    monkeypatch.setattr(lifecycle, "_get_instance", lambda iid: {"id": iid})
    monkeypatch.setattr(ssh, "_pick_ssh_endpoint", lambda i, **k: ("h", 22, "direct"))
    monkeypatch.setattr(remote.subprocess, "run", _boom)
    ok, text, err = remote._ssh_exec_soft(41, "x", timeout=180)
    assert (ok, text) == (False, "")
    assert "timed out after 180s" in err


def test_pick_ssh_endpoint_is_reached_through_the_ssh_MODULE(monkeypatch):
    """Module-attribute pin: four existing patch sites ride this edge."""
    seen = []
    monkeypatch.setattr(lifecycle, "_get_instance", lambda iid: {"id": iid})
    monkeypatch.setattr(ssh, "_pick_ssh_endpoint",
                        lambda i, **k: seen.append(i) or ("h", 22, "direct"))
    monkeypatch.setattr(remote.subprocess, "run", lambda *a, **k: _Run(0, "ok"))
    assert remote._ssh_exec_soft(41, "x").ok is True
    assert seen == [{"id": 41}]
