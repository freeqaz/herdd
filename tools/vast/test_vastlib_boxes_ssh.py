"""`vastlib.boxes.ssh` — the ported ssh reach/diagnose layer, held to its traps.

Why this file exists
--------------------
Three properties of this module survive a move only if something checks them,
and the existing `test_ssh_access.py` cannot check them for the ported copy
(it drives `herdd`'s, which stays live through the add-only phase and is
deliberately left untouched here):

1. **The repo-root depth.** `_tunnel_background` resolved `out/` with exactly
   three `os.path.dirname` calls from `tools/vast/herdd.py`. From
   `tools/vast/vastlib/boxes/ssh.py` that expression yields `tools/vast`, and
   NOTHING FAILS — the tunnel still comes up, the pidfile/log just quietly move
   and the printed teardown line points at a path the argparse help does not
   describe. `test_repo_root_matches_herdd_computation` pins the ported
   constant against the expression applied to `herdd.py`'s own path, and
   `test_naive_file_arithmetic_here_would_be_wrong` proves the trap is real
   rather than theoretical.
2. **The module-attribute call form** (plan §8b). Six existing patch sites
   steer `pub_key_text`, two steer `_pick_ssh_endpoint`. A `from .x import fn`
   anywhere in the chain makes all of them vacuous — green tests steering
   nothing. Every seam here is exercised by patching the MODULE ATTRIBUTE and
   asserting the patch was actually taken.
3. **The stdout/stderr split.** `_print_ssh` writes exactly one line — the ssh
   command — to stdout, and every diagnosis to stderr. That is a
   machine-readability contract (`--print` shares it), and it is one accidental
   `print(..., file=None)` away from breaking.

Twin identity, and where it went (plan §8 step 6d)
--------------------------------------------------
This file used to assert the snippet bytes and the two operator strings EQUAL
to `herdd`'s live copies: during the add-only phase both were emitted into
real onstarts, so a divergence would have meant two different boxes depending
on which code path launched them — the same reasoning
`test_vastlib_core_models.py` uses for `SSH_INJECT_MARKER`. The thinning ended
that: `herdd.py` is a launcher whose `ssh_authorized_keys_snippet`,
`SSH_STRICTMODES_HINT` and `ssh_access_warning` are re-export bindings to THIS
module's objects, so those comparisons became `x == x`. They are deleted; what
replaces them is `test_the_launcher_re_exports_rather_than_redefines`, which
fails if a second body ever reappears under any of the three names — the only
way the two-boxes hazard can come back.

What is deliberately NOT here
-----------------------------
* No re-testing of the StrictModes repair semantics (`test_ssh_access.py` runs
  the emitted shell against a fake root and owns that).
* No repoint of any existing test. All 22 own-symbol patch sites still target
  `herdd.<name>` and still steer `herdd`'s callers; they migrate with their
  callers at plan steps 6-7.
* No network, no box, no live API. `attach_ssh_key_soft` is a POST, so it is
  only ever called here with `vastlib.core.api.request_soft` stubbed — plus one
  test that deliberately does NOT stub it, to prove the conftest guard is what
  answers.

Provenance: created 2026-08-16 alongside `vastlib/boxes/ssh.py`, plan §8 step 3.
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

import herdd                                         # noqa: E402  twin, still live

from vastlib.boxes import lifecycle, ssh               # noqa: E402
from vastlib.core import api, models                   # noqa: E402

_PUB = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAItest test@workstation"

_V2 = {"id": 41, "onstart": models.SSH_INJECT_MARKER + ": x\n"}
_LEGACY = {"id": 42, "onstart": "echo k >> /root/.ssh/authorized_keys\n"}
_NONE = {"id": 43, "onstart": "echo hi\n"}


# --------------------------------------------------------------------------- #
# H1 — the repo root, which is the one line of the port that is not verbatim
# --------------------------------------------------------------------------- #
def test_repo_root_matches_herdd_computation():
    """The ported constant == what `herdd.py` computes for itself at runtime."""
    herdd_py = os.path.abspath(str(VAST_DIR / "herdd.py"))
    expected = os.path.dirname(os.path.dirname(os.path.dirname(herdd_py)))
    assert ssh._REPO_ROOT == expected
    assert os.path.isdir(os.path.join(ssh._REPO_ROOT, "tools", "vast"))


def test_naive_file_arithmetic_here_would_be_wrong():
    """Copying the three-dirname expression verbatim lands in tools/vast.

    This is the whole hazard: it is not an error, it is a silently different
    directory, so only a comparison can catch it.
    """
    naive = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(ssh.__file__))))
    assert naive.endswith(os.path.join("tools", "vast"))
    assert naive != ssh._REPO_ROOT


def test_tunnel_background_defaults_live_under_the_repo_out_dir(monkeypatch,
                                                                tmp_path):
    """The pidfile/logfile defaults quoted in `herdd tunnel --help`."""
    seen = {}

    class _Proc:
        pid = 4242
        returncode = 0

        def poll(self):
            return None

    monkeypatch.setattr(ssh.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(ssh.subprocess, "Popen",
                        lambda *a, **k: seen.setdefault("proc", _Proc()))
    monkeypatch.setattr(ssh.socket, "create_connection",
                        lambda *a, **k: io.BytesIO())
    opened = []
    real_open = open

    def _fake_open(path, mode="r", *a, **k):
        opened.append((str(path), mode))
        return real_open(os.devnull if "b" in mode or "w" in mode else path,
                         mode, *a, **k)

    monkeypatch.setitem(ssh.__dict__, "open", _fake_open)
    a = argparse.Namespace(id=41, local=8000, remote=8000,
                           pidfile=None, logfile=None)
    out = io.StringIO()
    with redirect_stdout(out):
        ssh._tunnel_background(a, ["ssh"], "h", 22)
    want = os.path.join(ssh._REPO_ROOT, "out", "vast_tunnel_41_8000.pid")
    assert f"pidfile : {want}" in out.getvalue()
    assert any(p == want for p, _ in opened)


# --------------------------------------------------------------------------- #
# the snippet — the pinned constants, and the launcher's re-export identity
# --------------------------------------------------------------------------- #
def test_the_launcher_re_exports_rather_than_redefines():
    """Post-6d replacement for the three byte-identity parity assertions.

    `herdd.py` re-exports these three names from `vastlib.boxes.ssh` (rule 1
    of its docstring: identity bindings only), so comparing the values is
    comparing an object with itself. What still has teeth is the binding: a
    peer re-adding a body to the launcher recreates the two-different-onstarts
    hazard the deleted parity tests existed to catch.
    """
    for name in ("ssh_authorized_keys_snippet", "SSH_STRICTMODES_HINT",
                 "ssh_access_warning"):
        assert getattr(herdd, name) is getattr(ssh, name), (
            f"herdd.{name} is a second body again — the launcher must "
            f"re-export vastlib.boxes.ssh's object, never redefine it")


def test_snippet_carries_the_marker_and_the_bounded_loop():
    s = ssh.ssh_authorized_keys_snippet(_PUB)
    assert s.startswith(models.SSH_INJECT_MARKER)
    assert f"_n -lt {ssh.SSH_FIX_TRIES}" in s
    assert ssh.SSH_FIX_TRIES * ssh.SSH_FIX_SLEEP_S <= 900      # product, not each
    assert s.rstrip().endswith("&")


# --------------------------------------------------------------------------- #
# with_ssh_inject — idempotence, and that pub_key_text is reached by ATTRIBUTE
# --------------------------------------------------------------------------- #
def test_with_ssh_inject_is_idempotent_and_passes_none_through(monkeypatch):
    monkeypatch.setattr(ssh, "pub_key_text", lambda *a, **k: _PUB)
    once = ssh.with_ssh_inject("echo hi")
    assert once.startswith(models.SSH_INJECT_MARKER)
    assert ssh.with_ssh_inject(once) == once                   # no second copy
    monkeypatch.setattr(ssh, "pub_key_text", lambda *a, **k: None)
    assert ssh.with_ssh_inject(None) is None                   # unchanged, incl. None
    assert ssh.with_ssh_inject("echo hi") == "echo hi"


def test_with_ssh_inject_reads_pub_key_text_as_a_module_attribute(monkeypatch):
    """The pin for six existing patch sites: a from-import would ignore this."""
    monkeypatch.setattr(ssh, "pub_key_text", lambda *a, **k: "SENTINEL-KEY")
    assert "SENTINEL-KEY" in ssh.with_ssh_inject("echo hi")


def test_pub_key_text_prefers_an_explicit_path(tmp_path):
    p = tmp_path / "k.pub"
    p.write_text(_PUB + "\n")
    assert ssh.pub_key_text(str(p)) == _PUB
    assert ssh.pub_key_text(str(tmp_path / "absent.pub")) is None


# --------------------------------------------------------------------------- #
# attach_ssh_key_soft — a POST, so it meets the conftest guard head on (H4)
# --------------------------------------------------------------------------- #
def test_attach_ssh_key_soft_posts_the_key(monkeypatch):
    calls = []

    def _stub(method, path, body=None, *a, **k):
        calls.append((method, path, body))
        return True, {}, None

    monkeypatch.setattr(api, "request_soft", _stub)
    monkeypatch.setattr(ssh, "pub_key_text", lambda *a, **k: _PUB)
    assert ssh.attach_ssh_key_soft(41) is True
    assert calls == [("POST", "v0/instances/41/ssh/", {"ssh_key": _PUB})]


def test_attach_ssh_key_soft_is_false_without_a_key_or_an_id(monkeypatch):
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: pytest.fail("must not be reached"))
    monkeypatch.setattr(ssh, "pub_key_text", lambda *a, **k: None)
    assert ssh.attach_ssh_key_soft(41) is False
    monkeypatch.setattr(ssh, "pub_key_text", lambda *a, **k: _PUB)
    assert ssh.attach_ssh_key_soft(None) is False


def test_unstubbed_post_is_refused_by_the_conftest_guard(monkeypatch):
    """Not a test of ssh.py: a test that the live-fleet guard covers vastlib.

    `attach_ssh_key_soft` swallows the refusal into `False`, so the guard is
    invisible from the return value — assert on the message the wrapped
    `request_soft` produces instead.
    """
    monkeypatch.setattr(ssh, "pub_key_text", lambda *a, **k: _PUB)
    ok, _d, err = api.request_soft("POST", "v0/instances/41/ssh/", {"x": 1})
    assert ok is False and "test isolation" in (err or "")
    assert ssh.attach_ssh_key_soft(41) is False


# --------------------------------------------------------------------------- #
# diagnosis
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("inst", [_V2, _LEGACY])
def test_ssh_access_warning_is_quiet_for_an_installing_box(inst):
    assert ssh.ssh_access_warning(inst) is None


def test_ssh_access_warning_names_the_marker_and_the_instance():
    msg = ssh.ssh_access_warning(_NONE)
    assert msg is not None
    assert repr(models.SSH_INJECT_MARKER) in msg
    assert "43" in msg and "StrictModes" in msg


def test_auth_preflight_skips_the_probe_entirely_on_v2(monkeypatch):
    monkeypatch.setattr(ssh.subprocess, "run",
                        lambda *a, **k: pytest.fail("v2 must not probe"))
    ssh._ssh_auth_preflight(_V2, "h", 22)                      # no network, no cost


def test_auth_preflight_prints_the_hint_on_a_publickey_denial(monkeypatch):
    class _R:
        returncode = 255
        stderr = "root@h: Permission denied (publickey)."
        stdout = ""

    monkeypatch.setattr(ssh.subprocess, "run", lambda *a, **k: _R())
    err = io.StringIO()
    with redirect_stderr(err):
        ssh._ssh_auth_preflight(_LEGACY, "h", 22)
    assert "StrictModes" in err.getvalue()


def test_auth_preflight_is_silent_when_the_probe_succeeds(monkeypatch):
    class _R:
        returncode = 0
        stderr = ""
        stdout = ""

    monkeypatch.setattr(ssh.subprocess, "run", lambda *a, **k: _R())
    err = io.StringIO()
    with redirect_stderr(err):
        ssh._ssh_auth_preflight(_LEGACY, "h", 22)
    assert err.getvalue() == ""


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #
def test_endpoints_put_the_direct_mapping_first_and_dedup():
    i = {"public_ipaddr": "1.2.3.4",
         "ports": {"22/tcp": [{"HostPort": "40001"}]},
         "ssh_host": "1.2.3.4", "ssh_port": "40001"}
    assert ssh._ssh_endpoints(i) == [("1.2.3.4", 40001, "direct")]


def test_endpoints_keep_a_distinct_api_endpoint_second():
    i = {"public_ipaddr": "1.2.3.4",
         "ports": {"22/tcp": [{"HostPort": "40001"}]},
         "ssh_host": "ssh5.vast.ai", "ssh_port": "22"}
    assert ssh._ssh_endpoints(i) == [("1.2.3.4", 40001, "direct"),
                                     ("ssh5.vast.ai", 22, "api")]


def test_endpoints_are_empty_when_the_box_answers_neither():
    assert ssh._ssh_endpoints({"id": 1}) == []


def test_pick_falls_back_to_the_first_candidate_unprobed(monkeypatch):
    monkeypatch.setattr(ssh.socket, "create_connection",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("refused")))
    i = {"public_ipaddr": "1.2.3.4",
         "ports": {"22/tcp": [{"HostPort": "40001"}]},
         "ssh_host": "ssh5.vast.ai", "ssh_port": "22"}
    assert ssh._pick_ssh_endpoint(i) == ("1.2.3.4", 40001, "direct")
    assert ssh._pick_ssh_endpoint({}, probe_timeout=0.01) == (None, None, None)


def test_pick_returns_the_endpoint_that_answers(monkeypatch):
    class _S:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _conn(addr, timeout=None):
        if addr[0] == "1.2.3.4":
            raise OSError("stale direct mapping")
        return _S()

    monkeypatch.setattr(ssh.socket, "create_connection", _conn)
    i = {"public_ipaddr": "1.2.3.4",
         "ports": {"22/tcp": [{"HostPort": "40001"}]},
         "ssh_host": "ssh5.vast.ai", "ssh_port": "22"}
    assert ssh._pick_ssh_endpoint(i) == ("ssh5.vast.ai", 22, "api")


# --------------------------------------------------------------------------- #
# _print_ssh — stdout is machine-readable, everything else is stderr
# --------------------------------------------------------------------------- #
def test_print_ssh_writes_one_line_to_stdout_and_diagnosis_to_stderr(monkeypatch):
    monkeypatch.setattr(lifecycle, "_get_instance", lambda iid: dict(_NONE))
    monkeypatch.setattr(ssh, "_pick_ssh_endpoint",
                        lambda i, **k: ("1.2.3.4", 40001, "direct"))
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        ssh._print_ssh(43)
    assert out.getvalue() == "ssh -p 40001 root@1.2.3.4\n"
    assert "!! ssh: instance 43" in err.getvalue()             # the warning
    assert "FAIL_HOLD_MINUTES" in err.getvalue()               # the hold reminder


def test_print_ssh_reaches_get_instance_through_lifecycle(monkeypatch):
    """Module-attribute pin for the cross-module edge into boxes.lifecycle."""
    seen = []
    monkeypatch.setattr(lifecycle, "_get_instance",
                        lambda iid: seen.append(iid) or dict(_V2))
    monkeypatch.setattr(ssh, "_pick_ssh_endpoint",
                        lambda i, **k: (None, None, None))
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        ssh._print_ssh(41)
    assert seen == [41]


def test_no_subprocess_or_socket_escapes_this_module_by_import():
    """`subprocess` and `socket` are module attributes here, not from-imports.

    Four existing patch sites and two of the tests above bind
    `ssh.subprocess.run` / `ssh.socket.create_connection`; a
    `from subprocess import run` would leave them steering nothing.
    """
    assert ssh.subprocess is subprocess
    assert getattr(ssh.socket, "create_connection", None) is not None
