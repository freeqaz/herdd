"""`vastlib.fleet.client` — the ported fleetd client, held to its traps.

Why this file exists
--------------------
The client is a wire protocol, two live-fleet guards and an error taxonomy
expressed as string prefixes. None of those fail loudly when they break, and
none of them are covered for the PORTED copy by the existing suite —
`test_fleetd.py` drove `herdd.fleet_request` and stayed untouched through the
add-only phase (plan §8a); since step 6d that name reaches this module. So:

1. **The collapsed protocol constants.** `herdd.FLEET_PROTO_VERSION` and
   `fleetd.VERSION` were independent literals, as were `FLEET_UNIT_NAME` and
   `UNIT_NAME`. `Server.handle` REFUSES a request whose `v` does not match, so a
   divergence is a total client outage against a daemon that keeps running and
   keeps looking healthy. Each collapsed constant is asserted equal to BOTH
   originals here, for as long as the flat files exist.
2. **The socket-module trap.** `herdd.fleet_request` connects through a
   function-local `import socket as _socket` while its `except` clause names
   `socket.timeout` off the module-level import. Half-ported, that clause raises
   `NameError` INSIDE the `try` — a transport blip becomes an exception on the
   one path whose entire purpose is to degrade into a fallback.
   `test_recv_timeout_degrades_to_a_timeout_error` fires a real
   `socket.timeout` through the ported code and asserts the taxonomy answer.
3. **Both live-fleet guards.** `fleet_sock_path` must read `FLEETD_SOCK` on
   EVERY call (conftest's autouse fixture is the suite's only protection from
   the real daemon, and plan §4 requires a meta-test that the guard target
   EXISTS — the hasattr-vacuous failure mode), and `_fleet_delegation_disabled`
   must keep reading `PYTEST_CURRENT_TEST` per call.
4. **The `__file__` depth** behind `_git_rev_short`, which fails SILENTLY: the
   naive expression still names a directory inside this repo, so `git rev-parse`
   keeps working and keeps returning a plausible answer.
5. **The dashboard's read-only op literals.** `fleetd.sock` accepts `destroy`
   and `pause` on the same transport as `status`; `test_dash_cache.py:409` pins
   `('status', 'spend')` for `herdd`'s copy and that assertion is re-anchored
   here for the ported one.

What is deliberately NOT here
-----------------------------
* No repoint of any existing test. `test_fleetd.py`, `test_dash_cache.py` and
  `test_job_submit_preflight.py` still drive `herdd`/`fleetd` and still pass;
  they migrate with their callers at plan steps 6-7.
* No daemon behavior. `Server.handle`'s op semantics belong to
  `fleet/daemon.py`; this file speaks to a socket that answers with canned
  lines, which is exactly what the client contract is.

Provenance: created 2026-08-16 alongside `vastlib/fleet/client.py`, plan §8
step 5.
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import sys
import threading
from pathlib import Path

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

import fleetd                                          # noqa: E402  twin, still live
import herdd                                         # noqa: E402  twin, still live

from vastlib.core import result                        # noqa: E402
from vastlib.fleet import client                       # noqa: E402
from vastlib.storage import dashcache                  # noqa: E402


# --------------------------------------------------------------------------- #
# 1 — the collapsed protocol constants (divergence == total client outage)
# --------------------------------------------------------------------------- #
def test_both_launchers_re_export_the_collapsed_constants() -> None:
    """Was three tests comparing `client.<C>` to `herdd.<C>` and `fleetd.<C>`
    — two independent literals each, and a divergence between them was a total
    client outage (the client would speak a protocol version the daemon
    rejects). Plan §8 step 6d collapsed the last of that: both launchers now
    re-export these from `fleet.client` by identity, and `fleetd.VERSION` is
    bound to `FLEET_PROTO_VERSION` by explicit ruling (its docstring) rather
    than to `state.VERSION`, which is a different number that happens to be
    equal. So the assertion is the binding — a literal reappearing in either
    launcher re-opens exactly the outage this section existed for."""
    assert herdd.FLEET_PROTO_VERSION is client.FLEET_PROTO_VERSION
    assert fleetd.VERSION is client.FLEET_PROTO_VERSION
    assert herdd.FLEET_UNIT_NAME is client.FLEET_UNIT_NAME
    assert fleetd.UNIT_NAME is client.FLEET_UNIT_NAME
    assert herdd.FLEET_SOCK_TIMEOUT_S is client.FLEET_SOCK_TIMEOUT_S


def test_sock_timeout_is_the_default_arg_every_request_inherits() -> None:
    """A default-arg binding: every unspecified `fleet_request` inherits it."""
    import inspect
    assert (inspect.signature(client.fleet_request).parameters["_timeout"].default
            == client.FLEET_SOCK_TIMEOUT_S)


def test_on_disk_filenames_match_the_daemons_constants() -> None:
    """The third and fourth copies of these two literals are now zero."""
    assert client.STATE_NAME == fleetd.STATE_NAME
    assert client.JOURNAL_NAME == fleetd.JOURNAL_NAME


# --------------------------------------------------------------------------- #
# 2 — paths: env read on EVERY call, and the same answers as the flat copy
# --------------------------------------------------------------------------- #
def test_paths_follow_the_env_on_every_call(monkeypatch, tmp_path) -> None:
    """Was `test_paths_match_herdd_for_the_same_env`, four resolvers compared
    across the two namespaces; one body each since step 6d. The property those
    comparisons rode on — the env var is read per call, not snapshotted — is
    what is asserted here and in `test_sock_path_is_never_cached` below."""
    monkeypatch.setenv("FLEETD_STATE_DIR", str(tmp_path / "st"))
    monkeypatch.delenv("FLEETD_SOCK", raising=False)
    assert client.fleet_state_dir() == str(tmp_path / "st")
    for fn in (client.fleet_sock_path, client.fleet_state_path,
               client.fleet_journal_path):
        assert fn().startswith(str(tmp_path / "st"))


def test_default_state_dir_is_the_documented_one(monkeypatch) -> None:
    monkeypatch.delenv("FLEETD_STATE_DIR", raising=False)
    monkeypatch.delenv("FLEETD_SOCK", raising=False)
    assert client.fleet_state_dir() == os.path.expanduser(
        "~/.local/state/vast-fleetd")
    assert client.fleet_sock_path().endswith("/vast-fleetd/fleetd.sock")


def test_sock_path_is_never_cached(monkeypatch, tmp_path) -> None:
    """The live-fleet guard: conftest redirects FLEETD_SOCK per test, so a
    snapshot taken at import time silently re-arms the leak it prevents."""
    monkeypatch.setenv("FLEETD_SOCK", str(tmp_path / "a.sock"))
    first = client.fleet_sock_path()
    monkeypatch.setenv("FLEETD_SOCK", str(tmp_path / "b.sock"))
    assert client.fleet_sock_path() != first
    assert client.fleet_sock_path() == str(tmp_path / "b.sock")


def test_conftest_socket_guard_target_exists_and_bites() -> None:
    """Plan §4's meta-test: the fixture's target must EXIST in the new home.

    `_isolate_fleetd_socket` is autouse, so with no override this call has to
    come back `nodaemon:` — if the ported client read the path from anywhere
    else, this test would reach the workstation's real daemon and pass anyway,
    which is why the assertion is on the error CLASS and not on truthiness.
    """
    assert callable(client.fleet_sock_path)
    assert "FLEETD_SOCK" in os.environ
    ok, data, err = client.fleet_request("ping", _timeout=1, _retries=0)
    assert (ok, data) == (False, None)
    assert str(err).startswith("nodaemon:")


# --------------------------------------------------------------------------- #
# 3 — the __file__ depth behind _git_rev_short (silent by design)
# --------------------------------------------------------------------------- #
def test_tools_dir_matches_herdd_computation() -> None:
    herdd_py = os.path.abspath(str(VAST_DIR / "herdd.py"))
    assert client._TOOLS_DIR == os.path.dirname(os.path.dirname(herdd_py))
    assert os.path.isdir(os.path.join(client._TOOLS_DIR, "vast"))


def test_naive_file_arithmetic_here_would_be_wrong() -> None:
    """Two dirnames from this module lands in `tools/vast/vastlib` — still a
    real directory inside the repo, so `git -C` would keep succeeding."""
    naive = os.path.dirname(os.path.dirname(os.path.abspath(client.__file__)))
    assert naive.endswith(os.path.join("vast", "vastlib"))
    assert naive != client._TOOLS_DIR


def test_git_rev_short_reads_this_checkout(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _P:
        stdout = "abc1234\n"

    def _run(argv, **kw):
        seen["argv"] = argv
        return _P()

    monkeypatch.setattr(client.subprocess, "run", _run)
    assert client._git_rev_short() == "abc1234"
    assert seen["argv"][:3] == ["git", "-C", client._TOOLS_DIR]


def test_git_rev_short_swallows_everything(monkeypatch) -> None:
    def _boom(*a, **k):
        raise OSError("no git")

    monkeypatch.setattr(client.subprocess, "run", _boom)
    assert client._git_rev_short() is None


# --------------------------------------------------------------------------- #
# 4 — the transport: one real socket, and the whole error taxonomy
# --------------------------------------------------------------------------- #
class _FakeDaemon:
    """A real AF_UNIX server that answers each connection from `replies`.

    Real sockets on purpose: the client's contract is bytes on a wire (one JSON
    line out, read to newline, parse), and a mocked `socket.socket` would let a
    connect/except mismatch pass unnoticed — trap 2 in the module docstring.
    """

    def __init__(self, path: str, replies: list[bytes | None]) -> None:
        self.path = path
        self.replies = list(replies)
        self.requests: list[str] = []
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(path)
        self.srv.listen(8)
        self.srv.settimeout(5.0)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while self.replies:
            reply = self.replies.pop(0)
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            with conn:
                conn.settimeout(5.0)
                try:
                    buf = b""
                    while b"\n" not in buf:
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        buf += chunk
                    self.requests.append(buf.decode().strip())
                    if reply is not None:
                        conn.sendall(reply)
                except OSError:
                    return

    def close(self) -> None:
        try:
            self.srv.close()
        except OSError:
            pass


@pytest.fixture
def daemon(tmp_path, monkeypatch):                                 # noqa: ANN001
    made: list[_FakeDaemon] = []

    def _make(*replies: bytes | None) -> _FakeDaemon:
        d = _FakeDaemon(str(tmp_path / "fleetd.sock"), list(replies))
        monkeypatch.setenv("FLEETD_SOCK", d.path)
        made.append(d)
        return d

    yield _make
    for d in made:
        d.close()


def test_round_trip_request_envelope_and_payload(daemon) -> None:  # noqa: ANN001
    d = daemon(b'{"ok": true, "data": {"watches": 2}}\n')
    ok, data, err = client.fleet_request("status", target="46193810")
    assert (ok, data, err) == (True, {"watches": 2}, None)
    sent = json.loads(d.requests[0])
    assert sent == {"v": client.FLEET_PROTO_VERSION, "op": "status",
                    "args": {"target": "46193810"}}
    # sort_keys=True on the wire: the daemon's line-oriented log is diffable.
    assert d.requests[0] == json.dumps(sent, sort_keys=True)


def test_returns_the_shape_A_result_triple(daemon) -> None:        # noqa: ANN001
    daemon(b'{"ok": true, "data": 7}\n')
    r = client.fleet_request("ping")
    assert isinstance(r, result.Soft)
    assert tuple(r) == (True, 7, None)                    # tuple-unpack compatible


def test_refused_is_the_only_real_decision(daemon) -> None:        # noqa: ANN001
    daemon(b'{"ok": false, "error": "no such watch", "data": {"x": 1}}\n')
    ok, data, err = client.fleet_request("unwatch", target="9")
    assert ok is False
    assert data == {"x": 1}, "a refusal still carries its payload"
    assert err == "refused:no such watch"
    assert err[8:] == "no such watch", "the four callers slice exactly 8 chars"


def test_refused_without_a_message(daemon) -> None:                # noqa: ANN001
    daemon(b'{"ok": false}\n')
    _ok, _data, err = client.fleet_request("watch")
    assert err == "refused:refused"


def test_malformed_response(daemon) -> None:                       # noqa: ANN001
    daemon(b"not json at all\n")
    ok, _data, err = client.fleet_request("status")
    assert ok is False
    assert str(err).startswith("malformed response: ")


def test_missing_socket_is_nodaemon(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FLEETD_SOCK", str(tmp_path / "nope.sock"))
    ok, data, err = client.fleet_request("ping", _retries=0)
    assert (ok, data, err) == (False, None, "nodaemon:FileNotFoundError")


def test_a_socket_nobody_listens_on_is_nodaemon(monkeypatch, tmp_path) -> None:
    """A stale socket file left by a crashed daemon: refused, not absent."""
    path = str(tmp_path / "stale.sock")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(path)
    s.close()                                   # file remains, nothing listening
    monkeypatch.setenv("FLEETD_SOCK", path)
    ok, _data, err = client.fleet_request("ping", _retries=0)
    assert ok is False
    assert str(err).startswith("nodaemon:")


def test_recv_timeout_degrades_to_a_timeout_error(monkeypatch, tmp_path) -> None:
    """THE socket-alias trap. A daemon that accepts and never answers must
    produce `timeout:`, not a NameError from an except clause naming a module
    the connect did not use."""
    path = str(tmp_path / "silent.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(4)
    srv.settimeout(5.0)
    held: list[socket.socket] = []

    def _accept_and_hold() -> None:
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            held.append(conn)                   # accepted, never answered

    t = threading.Thread(target=_accept_and_hold, daemon=True)
    t.start()
    monkeypatch.setenv("FLEETD_SOCK", path)
    try:
        ok, data, err = client.fleet_request("status", _timeout=0.1, _retries=0)
    finally:
        srv.close()
        for c in held:
            c.close()
    assert (ok, data) == (False, None)
    assert str(err).startswith("timeout:"), err


def test_empty_reply_retries_once_with_a_backed_off_sleep(daemon,      # noqa: ANN001
                                                          monkeypatch) -> None:
    d = daemon(None, b'{"ok": true, "data": "second"}\n')
    slept: list[float] = []
    monkeypatch.setattr(client.time, "sleep", slept.append)
    ok, data, err = client.fleet_request("status", _retries=1)
    assert (ok, data, err) == (True, "second", None)
    assert slept == [0.5], "0.5 * (attempt + 1) on the first retry"
    assert len(d.requests) == 2


def test_exhausted_retries_report_the_transport_error(daemon,         # noqa: ANN001
                                                      monkeypatch) -> None:
    daemon(None)
    monkeypatch.setattr(client.time, "sleep", lambda _s: None)
    ok, _data, err = client.fleet_request("status", _retries=0)
    assert ok is False
    assert err == "timeout:empty response"


def test_fleet_daemon_up_is_ping_truthiness(monkeypatch) -> None:
    calls: list[tuple] = []

    def _req(op, **kw):
        calls.append((op, kw))
        return result.Soft(True, {}, None)

    monkeypatch.setattr(client, "fleet_request", _req)
    assert client.fleet_daemon_up() is True
    assert calls == [("ping", {"_timeout": 5})]
    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: result.Soft(False, None, "nodaemon:x"))
    assert client.fleet_daemon_up() is False


# --------------------------------------------------------------------------- #
# 5 — the second live-fleet guard, and the policy envelope
# --------------------------------------------------------------------------- #
def _ns(**kw):                                                     # noqa: ANN202
    import argparse
    base = dict(no_fleet=False, dry_run=False, follow=False, id=41, budget=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_delegation_is_disabled_inside_a_pytest_run() -> None:
    """Guard #2, independent of conftest's socket redirect: a test run must
    never register a watch with the live daemon."""
    assert os.environ.get("PYTEST_CURRENT_TEST")
    assert client._fleet_delegation_disabled(_ns()) is True


def test_delegation_opt_outs_are_read_per_call(monkeypatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("FLEETD_DISABLE", raising=False)
    assert client._fleet_delegation_disabled(_ns()) is False
    assert client._fleet_delegation_disabled(_ns(no_fleet=True)) is True
    assert client._fleet_delegation_disabled(_ns(dry_run=True)) is True
    monkeypatch.setenv("FLEETD_DISABLE", "1")
    assert client._fleet_delegation_disabled(_ns()) is True
    monkeypatch.setenv("FLEETD_DISABLE", "0")               # only "1" disables
    assert client._fleet_delegation_disabled(_ns()) is False
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "x")
    assert client._fleet_delegation_disabled(_ns()) is True


# `test_delegation_opt_outs_match_the_flat_copy` drove three namespaces through
# both copies. One copy since step 6d; the opt-outs are asserted by value in
# the test above, including the `FLEETD_DISABLE` env branch a parity sweep of
# three namespaces could not reach.


def test_fleet_policy_is_the_wire_shape(monkeypatch) -> None:
    import argparse
    a = argparse.Namespace(id=41, budget=3.5, follow=False, tags=["a"],
                           opts={"k": 1}, missing=None,
                           func=lambda x: x, jobfunc=1, cmd="c", jobcmd="j",
                           wfcmd="w", obj=object())
    pol = client._fleet_policy(a)
    assert set(pol) == {"id", "budget", "follow", "tags", "opts", "missing"}
    # the key names ARE the argparse dests: the daemon rebuilds Namespace(**pol)
    assert argparse.Namespace(**pol).budget == 3.5


def test_delegate_registers_and_returns_true(monkeypatch, capsys) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    seen: dict = {}

    def _req(op, **kw):
        seen.update({"op": op, **kw})
        return result.Soft(True, {"note": "adopted a ceiling"}, None)

    monkeypatch.setattr(client, "fleet_request", _req)
    assert client._fleet_delegate(_ns(budget=4.0), "run:R1", "run", 4.0) is True
    assert (seen["op"], seen["target"], seen["profile"]) == ("watch", "run:R1",
                                                             "run")
    assert seen["budget_usd"] == 4.0 and "requester" in seen
    out = capsys.readouterr().out
    assert "registered watch run:R1" in out and "$4.00" in out
    assert "adopted a ceiling" in out


def test_delegate_falls_back_inline_on_any_transport_trouble(monkeypatch,
                                                             capsys) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    for err in ("nodaemon:FileNotFoundError", "timeout:x", "socket:y"):
        monkeypatch.setattr(client, "fleet_request",
                            lambda *a, _e=err, **k: result.Soft(False, None, _e))
        assert client._fleet_delegate(_ns(), "41", "jobs", None) is False
        assert "legacy inline supervisor" in capsys.readouterr().out


def test_delegate_exits_only_on_a_refusal(monkeypatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: result.Soft(False, None,
                                                    "refused:budget missing"))
    with pytest.raises(SystemExit) as e:
        client._fleet_delegate(_ns(), "41", "jobs", None)
    assert "budget missing" in str(e.value)
    assert "--no-fleet" in str(e.value)


def test_delegate_follows_the_daemons_redirect(monkeypatch, capsys) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: result.Soft(
                            True, {"redirected_from": "41",
                                   "target": "run:R9"}, None))
    assert client._fleet_delegate(_ns(), "41", "jobs", None) is True
    out = capsys.readouterr().out
    assert "CURRENT box" in out
    assert "registered watch run:R9" in out, "the confirmation names the "\
                                             "daemon's key, not ours"


def test_delegate_follow_blocks_and_exits_with_fleet_follows_code(
        monkeypatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: result.Soft(True, {}, None))
    monkeypatch.setattr(client, "fleet_follow", lambda *a, **k: 3)
    with pytest.raises(SystemExit) as e:
        client._fleet_delegate(_ns(follow=True), "41", "jobs", None)
    assert e.value.code == 3


def test_the_two_delegate_wrappers_target_the_right_watch(monkeypatch) -> None:
    seen: list[tuple] = []
    monkeypatch.setattr(client, "_fleet_delegate",
                        lambda a, t, p, b: seen.append((t, p, b)) or True)
    client.fleet_delegate_supervise(_ns(budget=2.0), "R1")
    client.fleet_delegate_job_supervise(_ns(id=41, budget=1.0))
    assert seen == [("run:R1", "run", 2.0), ("41", "jobs", 1.0)]


# --------------------------------------------------------------------------- #
# 6 — the best-effort announcements (boxes/ and launch/ call these)
# --------------------------------------------------------------------------- #
def test_operator_intent_is_short_and_never_fatal(monkeypatch) -> None:
    seen: dict = {}

    def _req(op, **kw):
        seen.update({"op": op, **kw})
        return result.Soft(True, {"note": "watch is dormant"}, None)

    monkeypatch.setattr(client, "fleet_request", _req)
    assert client.fleet_operator_intent(41, "stop") == {"note": "watch is dormant"}
    # B2: a pre-mutation notice must not make the mutation wait on a slow tick.
    assert (seen["_timeout"], seen["_retries"]) == (5, 0)
    assert (seen["op"], seen["target"], seen["kind"]) == ("operator_intent",
                                                          "41", "stop")
    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: result.Soft(False, {"x": 1}, "nodaemon:x"))
    assert client.fleet_operator_intent(41, "stop") is None


def test_note_operator_stop_prints_only_a_note(monkeypatch, capsys) -> None:
    monkeypatch.setattr(client, "fleet_operator_intent",
                        lambda iid, kind: {"note": "dormant until you resume"})
    client.fleet_note_operator_stop(41)
    assert ">> dormant until you resume" in capsys.readouterr().out
    monkeypatch.setattr(client, "fleet_operator_intent", lambda iid, kind: None)
    client.fleet_note_operator_stop(41)
    assert capsys.readouterr().out == ""


def test_watch_best_effort_is_quiet_without_a_daemon(monkeypatch, capsys) -> None:
    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: result.Soft(False, None, "nodaemon:x"))
    assert client.fleet_watch_best_effort(41) is False
    out = capsys.readouterr()
    assert out.out == "" and out.err == "", "no daemon, no problem — B1b"


def test_watch_best_effort_warns_on_any_other_error(monkeypatch, capsys) -> None:
    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: result.Soft(False, None, "refused:nope"))
    assert client.fleet_watch_best_effort(41, "jobs") is False
    out = capsys.readouterr()
    assert out.out == ""
    assert "registration skipped" in out.err, "the warning is on stderr"


def test_watch_best_effort_registration(monkeypatch, capsys) -> None:
    seen: dict = {}

    def _req(op, **kw):
        seen.update({"op": op, **kw})
        return result.Soft(True, {}, None)

    monkeypatch.setattr(client, "fleet_request", _req)
    assert client.fleet_watch_best_effort(41, "bare",
                                          policy={"attached": True}) is True
    assert (seen["_timeout"], seen["_retries"]) == (5, 0)
    assert seen["policy"] == {"attached": True}
    assert seen["target"] == "41", "always a string on the wire"
    assert "registered 41 with fleetd" in capsys.readouterr().out


def test_registration_alone_never_prints_the_ladder_hint(monkeypatch, capsys) -> None:
    """Three of the four callers must NOT be told to arm a `jobs` ladder —
    workflowctl registers `bare` so fleetd never fights its in-process
    supervisor, and a train box upgrades to `--profile run`. So the advice is
    the caller's to ask for, and this funnel stays silent about it."""
    monkeypatch.setattr(client, "fleet_request",
                        lambda op, **kw: result.Soft(True, {}, None))
    client.fleet_watch_best_effort(41, "bare")
    assert "bid defense" not in capsys.readouterr().out


def test_the_bare_hint_names_the_gap_and_the_command(capsys) -> None:
    """`bare` reads like supervision and arms none of the money moves. The hint
    is printed where the operator still has the id in hand, so it has to name
    both what is missing and the exact command that closes it."""
    assert client.print_bare_watch_hint(41, "jobs") is True
    out = capsys.readouterr().out
    assert "no bid defense" in out and "no eviction replacement" in out
    assert "fleet watch 41 --profile jobs --budget <USD> --standing" in out
    assert len(out.rstrip("\n").splitlines()) <= 3, "it fires on every jobs launch"


def test_the_bare_hint_tracks_the_profile_it_is_given(capsys) -> None:
    """`--standing` and the ticket-ordering caveat are jobs-profile facts. A
    `run` box upgrade carries neither, and printing them there would be wrong
    advice dressed as a checklist."""
    client.print_bare_watch_hint("run:demo", "run")
    out = capsys.readouterr().out
    assert "--profile run --budget <USD>" in out and "--standing" not in out
    assert "tickets" not in out


def test_the_ticket_hint_is_silent_under_a_spend_capable_watch(monkeypatch,
                                                               capsys) -> None:
    """`job retarget`/`job requeue` print the bare hint at the seam where the
    ticket lands. Onto a box whose standing watch that ticket just woke, "arm
    the ladder AFTER the tickets exist" is advice to re-arm what is armed — and
    `fleet watch` states the whole watch, which is how one cap becomes two."""
    monkeypatch.setattr(client, "fleet_watch_supervision",
                        lambda t: ("policy", {"profile": "jobs"}))
    assert client.print_jobs_ticket_hint(41) is False
    assert capsys.readouterr().out == ""


def test_the_ticket_hint_still_fires_where_nothing_is_armed(monkeypatch,
                                                            capsys) -> None:
    """Including `unknown` — an unreadable fleetd state is the normal shape on
    a fresh box, and the hint is exactly right there."""
    for level in ("none", "bare", "lapsed", "unknown"):
        monkeypatch.setattr(client, "fleet_watch_supervision",
                            lambda t, _l=level: (_l, {}))
        assert client.print_jobs_ticket_hint(41) is True
        assert "no bid defense" in capsys.readouterr().out


def test_ticket_placed_is_best_effort_and_announces_only_a_real_wake(
        monkeypatch, capsys) -> None:
    """The wake is an announcement, not a request: a missing daemon, an older
    daemon that answers `unknown op`, and a box with no standing watch all leave
    the CLI's own work done and unreported."""
    calls = []
    monkeypatch.setattr(client, "fleet_request",
                        lambda op, **kw: calls.append((op, kw))
                        or result.Soft(True, {"woken": True}, None))
    client.fleet_ticket_placed(41, "j-1", source="job retarget")
    assert calls[0][0] == "ticket_placed"
    assert calls[0][1]["target"] == "41" and calls[0][1]["job_id"] == "j-1"
    assert calls[0][1]["source"] == "job retarget"
    assert "re-arms its ladder" in capsys.readouterr().out

    monkeypatch.setattr(client, "fleet_request",
                        lambda op, **kw: result.Soft(True, {"woken": False}, None))
    assert client.fleet_ticket_placed(41)["woken"] is False
    assert capsys.readouterr().out == ""

    monkeypatch.setattr(client, "fleet_request",
                        lambda op, **kw: result.Soft(False, None,
                                                     "refused:unknown op"))
    assert client.fleet_ticket_placed(41) is None
    assert capsys.readouterr().out == ""


def test_the_bare_hint_is_suppressible(monkeypatch, capsys) -> None:
    """launch_jobs_box.sh arms the ladder itself two steps later, so the hint
    would be an alarm firing on the recommended workflow. It sets this."""
    monkeypatch.setenv("HERDD_WATCH_HINT", "0")
    assert client.print_bare_watch_hint(41, "jobs") is False
    assert capsys.readouterr().out == ""


def test_call_or_die_returns_the_payload_or_exits(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: result.Soft(True, {"rows": []}, None))
    assert client._fleet_call_or_die("status") == {"rows": []}

    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: result.Soft(False, None, "refused:no"))
    with pytest.raises(SystemExit) as e:
        client._fleet_call_or_die("unwatch", target="9")
    assert str(e.value) == "error: fleet unwatch refused: no"

    monkeypatch.setenv("FLEETD_SOCK", str(tmp_path / "f.sock"))
    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: result.Soft(False, None, "nodaemon:x"))
    with pytest.raises(SystemExit) as e:
        client._fleet_call_or_die("status")
    msg = str(e.value)
    assert "fleetd is not running" in msg
    assert str(tmp_path / "f.sock") in msg, "the message names the resolved path"
    assert "fleet install" in msg and "FLEETD_DRY_RUN=1" in msg

    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: result.Soft(False, None, "timeout:x"))
    with pytest.raises(SystemExit) as e:
        client._fleet_call_or_die("status")
    assert str(e.value) == "error: fleet status failed: timeout:x"


# --------------------------------------------------------------------------- #
# 7 — fleet_follow: the frozen foreground-blocking exit codes
# --------------------------------------------------------------------------- #
def _journal(monkeypatch, tmp_path, *recs: dict) -> str:
    path = tmp_path / "journal.ndjsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    monkeypatch.setattr(client, "fleet_journal_path", lambda: str(path))
    return str(path)


def test_follow_raises_the_same_oserror_as_the_flat_copy(monkeypatch,
                                                        tmp_path) -> None:
    """FOUND DEFECT, ported unchanged — `fleet_follow` cannot read a record.

    `pos = f.tell()` is the FIRST statement of the `for line in f` body, and
    `TextIOWrapper.tell()` during iteration raises
    `OSError: telling position disabled by next() call`. So every journal line
    appended after the seek-to-EOF crashes `supervise --follow` before any
    filtering or exit-code logic runs — the S7 blocking contract has been
    unreachable since the code was written (`2c05a009`, the fleetd B1-B4
    review), and nothing covers it: `fleet_follow` had ZERO test references
    before this file.

    Plan §7.4 says a port is behavior-preserving and a difference is a found
    drift, not a fix opportunity — so the defect is ported verbatim and pinned
    here, twin against twin, rather than quietly repaired inside a refactor.
    Repairing it (`f.tell()` -> tracking the offset, or `readline()`) is a
    behavior change and belongs in its own commit with its own reasoning.
    """
    rec = {"target": "41", "event": "watch_finished", "verdict": "unrecoverable"}
    path = _journal(monkeypatch, tmp_path, rec)
    monkeypatch.setattr(client.os.path, "getsize", lambda _p: 0)
    with pytest.raises(OSError) as ported:
        client.fleet_follow("41")
    # The twin arm — the same OSError raised out of `herdd.fleet_follow`,
    # with `herdd.fleet_journal_path` and `herdd.os.path.getsize` patched —
    # went at step 6d. It was doubly dead: one body, and those two patches
    # rebound names in the launcher's namespace that `fleet.client` never
    # consults (a re-export is not a patch point), so the "flat" arm was really
    # this arm with a differently-spelled stub.
    assert "telling position" in str(ported.value)
    del path


def test_follow_seeks_to_eof_so_history_is_never_replayed(monkeypatch,
                                                          tmp_path,
                                                          capsys) -> None:
    """The one path that does work: nothing new, so nothing is printed and the
    bounded wait returns 0. It is also what keeps a 32 MiB journal from being
    dumped to the terminal on every `--follow`."""
    _journal(monkeypatch, tmp_path,
             {"target": "41", "event": "watch_started"},
             {"target": "41", "event": "watch_finished", "verdict": "drained"})
    monkeypatch.setattr(client.time, "sleep", lambda _s: None)
    clock = iter([0.0, 0.5, 99.0, 99.0])
    monkeypatch.setattr(client.time, "time", lambda: next(clock))
    assert client.fleet_follow("41", timeout=5.0) == 0
    assert capsys.readouterr().out == ""


def test_follow_is_bounded_by_its_timeout(monkeypatch, tmp_path) -> None:
    """Every wait here has a deadline; a journal that does not exist yet must
    not park the caller forever."""
    monkeypatch.setattr(client, "fleet_journal_path",
                        lambda: str(tmp_path / "not-yet.ndjsonl"))
    monkeypatch.setattr(client.time, "sleep", lambda _s: None)
    clock = iter([0.0, 0.5, 1.0, 99.0, 99.0, 99.0])
    monkeypatch.setattr(client.time, "time", lambda: next(clock))
    assert client.fleet_follow("41", timeout=5.0) == 0


# --------------------------------------------------------------------------- #
# 8 — the two state.json readers (submit path + `fleet restart` guard)
# --------------------------------------------------------------------------- #
def _state(monkeypatch, tmp_path, doc: object) -> str:
    monkeypatch.setenv("FLEETD_STATE_DIR", str(tmp_path))
    path = tmp_path / client.STATE_NAME
    path.write_text(doc if isinstance(doc, str) else json.dumps(doc))
    return str(path)


_WATCH_POLICY = {"w1": {"iid": 41, "profile": "jobs", "budget_usd": 10.0,
                        "spend_usd": 1.0, "ceiling_source": "explicit"}}
_WATCH_LAPSED = {"w1": {"iid": 41, "profile": "bare", "budget_usd": 10.0,
                        "spend_usd": 0.0, "adopted": True,
                        "ceiling_source": "inherited"}}
_WATCH_BARE = {"w1": {"iid": 41, "profile": "bare", "budget_usd": 5.0,
                      "adopted": True, "ceiling_source": "provisional"}}


# The daemon's real shape, read off a live state.json 2026-08-17: box -> ceiling
# ID (a string), with the money in `ceilings[<id>]`. The old fixture here modeled
# `ceiling_by_box` as box -> float and was self-consistent with the bug.
_CEILINGS = {"ceiling_by_box": {"41": "38"},
             "ceilings": {"38": {"cap_usd": 5.0, "spend_usd": 4.25,
                                 "members": ["38", "41"]}}}


@pytest.mark.parametrize(("watches", "level"), [
    (_WATCH_POLICY, "policy"),
    (_WATCH_LAPSED, "lapsed"),
    (_WATCH_BARE, "bare"),
    ({"w1": {"iid": 99, "profile": "jobs"}}, "none"),
    ({}, "none"),
])
def test_watch_supervision_levels(monkeypatch, tmp_path,
                                  watches, level) -> None:
    _state(monkeypatch, tmp_path,
           {"watches": watches, **_CEILINGS})
    got = client.fleet_watch_supervision(41)
    assert got[0] == level


def test_watch_supervision_prefers_the_ceilings_cumulative_spend(monkeypatch,
                                                                 tmp_path) -> None:
    _state(monkeypatch, tmp_path, {"watches": _WATCH_POLICY, **_CEILINGS})
    _lvl, d = client.fleet_watch_supervision("41")
    assert d["spend_usd"] == 4.25, "the watch's own counter reads 1.0 here"
    assert d["standing"] is False and d["standing_dormant"] is False


def test_watch_supervision_never_reports_a_ceiling_id_as_dollars(monkeypatch,
                                                                 tmp_path) -> None:
    """Regression: the box id was printed as the spend, so a watch $0.0001 in
    read as "$47939448.00 of $1.50 spent ($0.00 left)" — a blown budget."""
    _state(monkeypatch, tmp_path,
           {"watches": _WATCH_POLICY,
            "ceiling_by_box": {"41": "41"},
            "ceilings": {"41": {"cap_usd": 1.5, "spend_usd": 6.5e-05}}})
    _lvl, d = client.fleet_watch_supervision("41")
    assert d["spend_usd"] == 6.5e-05
    assert d["spend_usd"] < d["budget_usd"]


def test_watch_supervision_falls_back_when_the_ceiling_is_missing(monkeypatch,
                                                                 tmp_path) -> None:
    """No ceiling record => the watch's own counter, not None."""
    _state(monkeypatch, tmp_path,
           {"watches": _WATCH_POLICY, "ceiling_by_box": {"41": "38"}})
    _lvl, d = client.fleet_watch_supervision("41")
    assert d["spend_usd"] == 1.0


@pytest.mark.parametrize("doc", ["{not json", '"a string, not a dict"'])
def test_watch_supervision_is_unknown_not_a_guess(monkeypatch, tmp_path,
                                                  doc) -> None:
    _state(monkeypatch, tmp_path, doc)
    assert client.fleet_watch_supervision(41)[0] == "unknown"


def test_watch_supervision_never_raises_on_a_missing_file(monkeypatch,
                                                          tmp_path) -> None:
    """It runs on the SUBMIT path: it advises, it never blocks."""
    monkeypatch.setenv("FLEETD_STATE_DIR", str(tmp_path / "nothing-here"))
    assert client.fleet_watch_supervision(41) == ("unknown", {})


def test_recoveries_in_flight_delegates_to_the_pure_fold(monkeypatch,
                                                         tmp_path) -> None:
    from vastlib.fleet import rows
    seen: list[object] = []

    def _fold(state):
        seen.append(state)
        return [{"target": "w1", "iid": 41, "kind": "rebid_ladder",
                 "detail": "2 rungs"}]

    monkeypatch.setattr(rows, "recoveries_in_flight", _fold)
    _state(monkeypatch, tmp_path, {"watches": _WATCH_POLICY})
    got = client.fleet_recoveries_in_flight()
    assert got == [{"target": "w1", "iid": 41, "kind": "rebid_ladder",
                    "detail": "2 rungs"}]
    assert seen and seen[0]["watches"] == _WATCH_POLICY


def test_recoveries_in_flight_matches_the_flat_copy(monkeypatch,
                                                    tmp_path) -> None:
    """Was `…_matches_the_flat_copy`: the ported fold vs the one `herdd`
    lazily imported from `fleetd` — the import cycle the package broke, so the
    two answers had to stay identical across it. Since step 6d there is no
    cycle and no second fold; what is left is that the fold reads the state
    file at all, which is the fixture assertion below."""
    doc = {"watches": {"run:R1": {"iid": 41, "profile": "run",
                                  "replacement": {"rebid_rungs": 2}},
                       "42": {"iid": 42, "profile": "jobs",
                              "unrecoverable_since": 1000.0}},
           "destroys": {"9": {"when": "drained"}}}
    _state(monkeypatch, tmp_path, doc)
    assert client.fleet_recoveries_in_flight(), "the fixture must not be empty"


@pytest.mark.parametrize("doc", ["[1, 2, 3]", "{bad", '"str"'])
def test_recoveries_in_flight_is_empty_on_anything_unreadable(monkeypatch,
                                                              tmp_path,
                                                              doc) -> None:
    """`[]` on a parse error is deliberate: refusing a restart because the state
    file is corrupt makes the corruption unrecoverable-by-restart."""
    _state(monkeypatch, tmp_path, doc)
    assert client.fleet_recoveries_in_flight() == []


def test_recoveries_in_flight_swallows_a_raising_fold(monkeypatch,
                                                      tmp_path) -> None:
    from vastlib.fleet import rows

    def _boom(_state):
        raise RuntimeError("bad row")

    monkeypatch.setattr(rows, "recoveries_in_flight", _boom)
    _state(monkeypatch, tmp_path, {"watches": _WATCH_POLICY})
    assert client.fleet_recoveries_in_flight() == []


# --------------------------------------------------------------------------- #
# 9 — the dashboard's fleet section: READ OPS ONLY, and no PII
# --------------------------------------------------------------------------- #
_FLEET_STATUS = {
    "tick_age_s": 12.0, "dry_run": False, "api_ok": True, "rev": "abc1234",
    "version": 1, "spend_total_usd": 11.5,
    "rows": [
        {"iid": 46246859, "target": "run:R1", "profile": "run",
         "state": "WATCHED", "spend_usd": 0.0, "ceiling_spend_usd": 4.9,
         "budget_usd": 10.0, "paused": True, "pause_left_s": 600,
         "pause_reason": "operator: B2_APPLICATION_KEY=deadbeefcafe",
         "dormant": False, "adopted": True, "last_action": "park",
         "requester": "leakeduser@workstation"},
        {"iid": 46193810, "target": "46193810", "profile": "bare",
         "state": "UNWATCHED", "spend_usd": 8.3, "budget_usd": None,
         "paused": False, "last_action": None,
         "requester": "leakeduser@workstation"},
    ],
    "alarm_records": [
        {"key": "stray:46193810", "iid": 46193810, "sticky": False,
         "msg": "unwatched box billing", "since_ts": 1.0, "age_s": 300.0},
        {"key": "budget:x", "sticky": True, "msg": "budget 80% consumed",
         "since_ts": 2.0, "age_s": 90.0, "count": 3},
    ],
}


def _fleet_ok(op, **kw):                                           # noqa: ANN202
    assert op in ("status", "spend"), f"dash-cache used a non-read op: {op}"
    if op == "status":
        return result.Soft(True, _FLEET_STATUS, None)
    return result.Soft(True, {"by_box": {"46246859": 3.0, "46193810": 8.37}},
                       None)


@pytest.fixture
def deps():                                                        # noqa: ANN201
    return dashcache.DashDeps(
        is_secret_env=herdd._is_secret_env,
        secret_val_re=herdd._SECRET_VAL_RE,
        reap_idle_h_default=herdd.REAP_IDLE_H_DEFAULT,
        gather_ls_data=lambda **kw: {},
        job_cell=herdd._job_cell,
        active_job_states=herdd._ACTIVE_JOB_STATES,
    )


@pytest.fixture
def conn(tmp_path):                                                # noqa: ANN001, ANN201
    c = sqlite3.connect(str(tmp_path / "infra-metadata.db"), timeout=5)
    c.executescript(dashcache._INFRA_CACHE_SCHEMA)
    yield c
    c.close()


def test_dash_fleet_uses_only_the_two_read_ops(monkeypatch, conn, deps) -> None:
    """Re-anchored from `test_dash_cache.py:409`. The same transport accepts
    `destroy` and `pause`; these two literals are the only thing between an
    unauthenticated dashboard refresh and a mutating verb."""
    ops: list[str] = []

    def _spy(op, **kw):
        ops.append(op)
        return _fleet_ok(op, **kw)

    monkeypatch.setattr(client, "fleet_request", _spy)
    n, err = client._dash_write_fleet(conn, deps=deps)
    assert (n, err) == (2, None)
    assert ops == ["status", "spend"]
    src = Path(client.__file__).read_text()
    body = src[src.index("def _dash_write_fleet"):src.index("def _dash_offer_query")]
    assert 'fleet_request("status"' in body and 'fleet_request("spend"' in body, \
        "the op strings must stay frozen literals, never built from a variable"


def test_dash_fleet_projection_fills_every_table(monkeypatch, conn, deps) -> None:
    """One write, four tables, none of them empty.

    Was a row-for-row diff against `herdd._dash_write_fleet` while both
    copies existed; step 6d made that name THIS function, and the flat call
    shape (no injected `deps`) raises inside `dashcache._dash_scrub`. What the
    diff was really carrying is that a single call populates all four
    projections — the column-level properties are pinned by the three tests
    below (no requester, scrub through the injected deps, frozen op strings).
    """
    monkeypatch.setattr(client, "fleet_request", _fleet_ok)
    assert client._dash_write_fleet(conn, deps=deps) == (2, None)
    for table in ("fleet", "fleet_watches", "fleet_alarms", "fleet_spend"):
        got = conn.execute(f"SELECT * FROM {table}").fetchall()
        assert got, f"{table} must not be empty for this fixture"


def test_dash_fleet_never_publishes_the_requester(monkeypatch, conn,
                                                  deps) -> None:
    """`requester` is user@hostname on a page with no authentication."""
    monkeypatch.setattr(client, "fleet_request", _fleet_ok)
    client._dash_write_fleet(conn, deps=deps)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(fleet_watches)")]
    assert "requester" not in cols
    dumped = "\n".join(str(r) for t in ("fleet", "fleet_watches", "fleet_alarms")
                       for r in conn.execute(f"SELECT * FROM {t}").fetchall())
    assert "leakeduser" not in dumped


def test_dash_fleet_scrubs_through_the_injected_deps(monkeypatch, conn,
                                                     deps) -> None:
    monkeypatch.setattr(client, "fleet_request", _fleet_ok)
    client._dash_write_fleet(conn, deps=deps)
    reason = conn.execute("SELECT pause_reason FROM fleet_watches "
                          "WHERE iid=46246859").fetchone()[0]
    assert "deadbeefcafe" not in reason and "<redacted>" in reason


def test_dash_fleet_prefers_the_ceilings_spend(monkeypatch, conn, deps) -> None:
    monkeypatch.setattr(client, "fleet_request", _fleet_ok)
    client._dash_write_fleet(conn, deps=deps)
    spend, budget, frac, stray = conn.execute(
        "SELECT spend_usd,budget_usd,budget_frac,stray FROM fleet_watches "
        "WHERE iid=46246859").fetchone()
    assert (spend, budget, stray) == (4.9, 10.0, 0)
    assert frac == pytest.approx(0.49)
    # the UNWATCHED row is a stray: no profile, and no budget_frac to divide by
    prof, frac2, stray2 = conn.execute(
        "SELECT profile,budget_frac,stray FROM fleet_watches WHERE iid=46193810"
    ).fetchone()
    assert (prof, frac2, stray2) == (None, None, 1)
    n_watches, n_strays = conn.execute(
        "SELECT n_watches,n_strays FROM fleet").fetchone()
    assert (n_watches, n_strays) == (1, 1)


def test_dash_fleet_down_daemon_clears_instead_of_serving_stale(monkeypatch,
                                                                conn,
                                                                deps) -> None:
    """"unknown" must never be served as a watch table claiming boxes are being
    babysat when nothing is."""
    monkeypatch.setattr(client, "fleet_request", _fleet_ok)
    client._dash_write_fleet(conn, deps=deps)
    assert conn.execute("SELECT count(*) FROM fleet_watches").fetchone()[0] == 2

    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: result.Soft(False, None, "nodaemon:x"))
    n, err = client._dash_write_fleet(conn, deps=deps)
    assert (n, err) == (0, "nodaemon:x")
    row = conn.execute("SELECT daemon_up,api_ok,rev,tick_age_s,n_watches "
                       "FROM fleet").fetchone()
    assert row == (0, None, None, None, None)
    for table in ("fleet_watches", "fleet_alarms", "fleet_spend"):
        assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_dash_fleet_is_the_only_two_tuple_section(monkeypatch, conn,
                                                  deps) -> None:
    """`cmd_dash_cache` special-cases this return shape; every other section
    returns a bare row count."""
    monkeypatch.setattr(client, "fleet_request", _fleet_ok)
    out = client._dash_write_fleet(conn, deps=deps)
    assert isinstance(out, tuple) and len(out) == 2
    assert isinstance(out[0], int) and out[1] is None


def test_dash_fleet_tick_staleness_uses_the_shared_threshold(monkeypatch, conn,
                                                             deps) -> None:
    stale = dict(_FLEET_STATUS, tick_age_s=dashcache.DASH_TICK_STALE_S + 1)
    monkeypatch.setattr(client, "fleet_request",
                        lambda op, **kw: (result.Soft(True, stale, None)
                                          if op == "status"
                                          else result.Soft(True, {}, None)))
    client._dash_write_fleet(conn, deps=deps)
    assert conn.execute("SELECT tick_stale FROM fleet").fetchone()[0] == 1


def test_dash_offer_query_is_the_permissive_survey() -> None:
    """Was `…_matches_the_flat_copy`; one builder since step 6d."""
    for kind in ("bid", "on-demand"):
        q = client._dash_offer_query("RTX_4090", 2, kind)
        assert q["limit"] == dashcache.DASH_OFFER_LIMIT
        assert q["verified"] == {"eq": True}, "a market survey stays permissive"


# --------------------------------------------------------------------------- #
# 10 — attribution: every daemon action is journaled with a requester
# --------------------------------------------------------------------------- #
def test_requester_is_user_at_hostname(monkeypatch) -> None:
    monkeypatch.setenv("USER", "someone")
    monkeypatch.setattr(client.socket, "gethostname", lambda: "workstation")
    assert client._fleet_requester() == "someone@workstation"
    monkeypatch.delenv("USER", raising=False)
    assert client._fleet_requester() == "user@workstation"


def test_requester_falls_back_rather_than_raising(monkeypatch) -> None:
    """A destroy must stay attributable-ish even when the hostname lookup dies;
    it must never be the reason a control-plane call fails."""
    def _boom():
        raise OSError("no resolver")

    monkeypatch.setattr(client.socket, "gethostname", _boom)
    assert client._fleet_requester() == "cli"


# --------------------------------------------------------------------------- #
# 11 — rev comparison: abbreviated hashes are prefixes, never equalities
# --------------------------------------------------------------------------- #
def test_rev_matches_accepts_differing_abbreviation_lengths() -> None:
    """The measured false alarm: one commit, two lengths, `==` calls it skew."""
    assert client.rev_matches("38e76425", "38e76425e")
    assert client.rev_matches("38e76425e", "38e76425")
    assert client.rev_matches("38e76425e6bd7a12627bdd4877d913fb6bfa19c3", "38e76425")


def test_rev_matches_still_rejects_a_real_skew() -> None:
    """Prefix semantics must not blur two different commits together."""
    assert not client.rev_matches("38e76425", "3edf1aa6")
    assert not client.rev_matches("38e76425", "38e76426")


def test_rev_matches_treats_absent_as_no_answer_not_as_agreement() -> None:
    """An unreadable rev is unknown, and unknown must never read as 'matches' —
    that would silently retire the skew check instead of reporting it."""
    for a, b in (("", "38e76425"), ("38e76425", ""), (None, None),
                 (None, "38e76425"), ("   ", "38e76425")):
        assert not client.rev_matches(a, b)
