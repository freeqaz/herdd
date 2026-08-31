"""Portable tests for SSH access on every box we rent (no network, no box).

Root-caused live on 2026-07-31 against instance 46449950 (spot RTX PRO 6000,
run t211-vet-1). `herdd ssh` and `herdd metrics` both died on a bare
`root@203.0.113.99: Permission denied (publickey)` while the vast API happily
reported the key attached ("SSH key already associated with instance."). Two
independent defects stacked:

  1. **Coverage.** `_do_launch` is the ONLY path that installs the pubkey, and
     the box was not created by it: the supervisor's eviction relaunch builds
     its create body from `runs/<RUN>/spec.json` via `_relaunch_body` and PUTs
     it straight to `/asks/`. The spec records the PRE-inject wire, so every
     relaunched box, every run-lane handoff understudy, and (explicitly,
     `ssh=False`) every jobs-lane understudy was born un-ssh-able. `herdd
     train` DID pass `ssh=True` — the box that outlived the eviction just never
     went through that code.

  2. **The install itself was not enough.** The container's
     /root/.ssh/authorized_keys held the right key, byte for byte, but was
     owned `vastai_kaalia:docker` mode 644 (vast's host daemon wrote it), and
     sshd runs `StrictModes yes` — which refuses an authorized_keys owned by
     neither root nor the target user. A working sibling box had `root:root
     0600`. The old snippet only appended and chmod'd, so it could not repair
     ownership, and on a resume it appended a duplicate line every boot.

These tests pin both halves plus the legibility layer (`ssh_access_warning`,
the `ls` footer) so the class cannot silently come back on a new launch path.
"""
import argparse
import contextlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imageref  # noqa: E402
from vastlib.boxes import lifecycle  # noqa: E402
from vastlib.boxes import ssh as ssh_mod  # noqa: E402
from vastlib.cli import _ls_render  # noqa: E402
from vastlib.cli import ls as cli_ls  # noqa: E402
from vastlib.cli import ssh as cli_ssh  # noqa: E402
from vastlib.core import api, models  # noqa: E402
from vastlib.jobs import view as jobs_view  # noqa: E402
from vastlib.launch import launch as launch_mod  # noqa: E402
from vastlib.launch import spec as launch_spec  # noqa: E402
from vastlib.market import pricing  # noqa: E402
from vastlib.supervise import replacement  # noqa: E402

_PUB = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAItest test@workstation"


@pytest.fixture
def local_key(monkeypatch):
    """A deterministic workstation pubkey (these tests must not depend on
    whether the machine running them happens to have ~/.ssh/id_ed25519.pub).

    ONE namespace now (step 6e): every subject below is a vastlib copy and
    reads `boxes.ssh.pub_key_text` — including `_relaunch_body`, which landed at
    `vastlib.supervise.replacement` and reaches the injector as
    `ssh.with_ssh_inject`. The flat `herdd` line is gone with the last flat
    subject in this file.
    """
    monkeypatch.setattr(ssh_mod, "pub_key_text", lambda *a, **k: _PUB)
    return _PUB


# =============================================================================
# the snippet itself — what actually lands on the box
# =============================================================================
def test_snippet_is_valid_bash():
    r = subprocess.run(["bash", "-n", "-c",
                        ssh_mod.ssh_authorized_keys_snippet(_PUB)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_snippet_repairs_ownership_and_mode_not_just_appends():
    s = ssh_mod.ssh_authorized_keys_snippet(_PUB)
    # the actual root cause: a correct key in a file sshd's StrictModes refuses
    assert "chown root:root /root/.ssh /root/.ssh/authorized_keys" in s
    assert "chmod 600 /root/.ssh/authorized_keys" in s
    assert "chmod 700 /root/.ssh" in s


def test_snippet_dedupes_so_a_resume_does_not_grow_the_file():
    # onstart re-runs on EVERY resume; a bare `>>` added a duplicate line each
    # time. The guard must test the exact line (-x), literally (-F).
    assert "grep -qxF " in ssh_mod.ssh_authorized_keys_snippet(_PUB)


def test_snippet_reassert_loop_is_bounded_and_backgrounded():
    # vast's own key push races onstart and can re-create the file with host
    # ownership AFTER we fixed it — so we re-assert. But a watcher that never
    # exits is its own footgun: bound it, and never block the real onstart.
    s = ssh_mod.ssh_authorized_keys_snippet(_PUB)
    assert f"_n -lt {ssh_mod.SSH_FIX_TRIES}" in s
    assert "while true" not in s
    assert s.rstrip().endswith("&")
    assert ssh_mod.SSH_FIX_TRIES * ssh_mod.SSH_FIX_SLEEP_S <= 900   # <= 15 min


def test_snippet_quotes_the_key():
    s = ssh_mod.ssh_authorized_keys_snippet("key with $(rm -rf /) spaces")
    assert "$(rm -rf /)" not in s.replace("'key with $(rm -rf /) spaces'", "")


def test_snippet_actually_produces_a_root_600_file(tmp_path):
    """Run the foreground half against a fake root and check the end state —
    twice, to prove idempotence (the resume case)."""
    fake = tmp_path / "root"
    fake.mkdir()
    body = ssh_mod.ssh_authorized_keys_snippet(_PUB) \
        .replace("/root/", str(fake) + "/") \
        .replace("chown root:root", "chown $(id -un):$(id -gn)")
    body = "\n".join(l for l in body.splitlines() if not l.startswith("( _n=0"))
    subprocess.run(["bash", "-c", body + "\n" + body], check=True)
    ak = fake / ".ssh" / "authorized_keys"
    assert ak.read_text() == _PUB + "\n"                    # deduped
    assert stat.S_IMODE(ak.stat().st_mode) == 0o600
    assert stat.S_IMODE((fake / ".ssh").stat().st_mode) == 0o700


def test_snippet_repairs_a_preexisting_bad_file(tmp_path):
    """The 46449950 shape exactly: the key is ALREADY there, in a file sshd
    will not read. The snippet must fix the metadata and not duplicate."""
    fake = tmp_path / "root"
    (fake / ".ssh").mkdir(parents=True)
    ak = fake / ".ssh" / "authorized_keys"
    ak.write_text(_PUB + "\n")
    ak.chmod(0o644)
    body = ssh_mod.ssh_authorized_keys_snippet(_PUB) \
        .replace("/root/", str(fake) + "/") \
        .replace("chown root:root", "chown $(id -un):$(id -gn)")
    body = "\n".join(l for l in body.splitlines() if not l.startswith("( _n=0"))
    subprocess.run(["bash", "-c", body], check=True)
    assert ak.read_text() == _PUB + "\n"
    assert stat.S_IMODE(ak.stat().st_mode) == 0o600


# =============================================================================
# with_ssh_inject / instance_has_ssh_inject
# =============================================================================
def test_with_ssh_inject_prepends_and_preserves_the_wire(local_key):
    out = ssh_mod.with_ssh_inject("#!/bin/bash\necho hi\n")
    assert out.endswith("#!/bin/bash\necho hi\n")
    assert models.SSH_INJECT_MARKER in out


def test_with_ssh_inject_is_idempotent(local_key):
    once = ssh_mod.with_ssh_inject("echo hi\n")
    assert ssh_mod.with_ssh_inject(once) == once
    assert once.count(models.SSH_INJECT_MARKER) == 1


def test_with_ssh_inject_noop_without_a_local_key(monkeypatch):
    monkeypatch.setattr(ssh_mod, "pub_key_text", lambda *a, **k: None)
    assert ssh_mod.with_ssh_inject("echo hi\n") == "echo hi\n"
    assert ssh_mod.with_ssh_inject(None) is None


def test_instance_has_ssh_inject_reads_the_stored_onstart(local_key):
    assert models.instance_has_ssh_inject(
        {"onstart": ssh_mod.with_ssh_inject("echo hi\n")})
    assert not models.instance_has_ssh_inject({"onstart": "echo hi\n"})
    assert not models.instance_has_ssh_inject({})
    assert not models.instance_has_ssh_inject(None)


# =============================================================================
# launch — ssh is ON BY DEFAULT (it costs nothing and every box needs a shell)
# =============================================================================
def _launch_ns(**over):
    base = dict(offer=None, type="ondemand", price=None, env=None, port=None,
                jupyter=False, onstart="echo hi\n", no_hf_token=True,
                hf_token=None, ssh_key_file=None, jobs=False, image="img:tag",
                disk=40, runtype="ssh_direct", label=None, template_id=None,
                no_registry_login=True, login=None, dry_run=True, wait=None,
                force=False)
    base.update(over)
    return argparse.Namespace(**base)


def _dry_body(monkeypatch, ns):
    monkeypatch.setattr(imageref, "image_tag_digest", lambda img: None)
    monkeypatch.setattr(pricing, "_offer_pricing_soft",
                        lambda oid: (0.10, 1.00, 1))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        launch_mod._do_launch(ns)
    out = buf.getvalue()
    return json.loads(out[out.index("{"):])["body"]


def test_launch_installs_the_key_by_default(monkeypatch, local_key):
    body = _dry_body(monkeypatch, _launch_ns(offer=1, ssh=True))
    assert models.SSH_INJECT_MARKER in body["onstart"]


def test_launch_no_ssh_opts_out(monkeypatch, local_key):
    body = _dry_body(monkeypatch, _launch_ns(offer=1, ssh=False))
    assert models.SSH_INJECT_MARKER not in body["onstart"]


def test_launch_warns_loudly_when_no_local_pubkey_exists(monkeypatch, capsys):
    monkeypatch.setattr(ssh_mod, "pub_key_text", lambda *a, **k: None)
    _dry_body(monkeypatch, _launch_ns(offer=1, ssh=True))
    err = capsys.readouterr().err
    assert "NOT be ssh-able" in err and "ssh-keygen" in err


def test_launch_flag_defaults_and_no_ssh_exist():
    """`--ssh` must stay accepted (back-compat) and `--no-ssh` must exist."""
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, os.path.join(here, "herdd.py"),
                        "launch", "--help"], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0
    assert "--ssh " in r.stdout or "--ssh\n" in r.stdout
    assert "--no-ssh" in r.stdout


# =============================================================================
# the create paths that BYPASS _do_launch — the actual 46449950 regression
# =============================================================================
# MIGRATED (step 6e): `_relaunch_body` is ported — the integrator ruling landed
# the body in `vastlib.supervise.replacement` beside the rest of the effectful
# drivers rather than in `run_lane`, and the raising SEAM stub is gone. Subject
# and seams both repoint: the injector is reached as `ssh.with_ssh_inject`, so
# `pub_key_text` is stubbed at `boxes.ssh` (the `local_key` fixture), and the
# marker constant is read from `core.models`.
def _spec_state(onstart="#!/bin/bash\ntrain\n"):
    return {"run_id": "r1",
            "launch_spec": {"image": "img:1", "disk": 64,
                            "runtype": "ssh_direct", "env": {"RUN_ID": "r1"},
                            "secret_env_keys": [], "onstart": onstart}}


def test_relaunch_body_installs_the_key(local_key):
    """The eviction relaunch PUTs this body straight to /asks/ — if the ssh
    install is not in it, the replacement box is un-debuggable for its whole
    life (the onstart is fixed at create time)."""
    body, missing = replacement._relaunch_body(
        _spec_state(), argparse.Namespace(dry_run=True), 0.5)
    assert missing == []
    assert body["onstart"].endswith("#!/bin/bash\ntrain\n")
    assert models.SSH_INJECT_MARKER in body["onstart"]


def test_relaunch_body_does_not_double_inject(local_key):
    """A spec captured from an onstart we already composed replays once."""
    spec_onstart = ssh_mod.with_ssh_inject("#!/bin/bash\ntrain\n")
    body, _ = replacement._relaunch_body(
        _spec_state(spec_onstart), argparse.Namespace(dry_run=True), 0.5)
    assert body["onstart"].count(models.SSH_INJECT_MARKER) == 1


def test_relaunch_body_without_a_local_key_keeps_the_wire(monkeypatch):
    monkeypatch.setattr(ssh_mod, "pub_key_text", lambda *a, **k: None)
    body, _ = replacement._relaunch_body(
        _spec_state(), argparse.Namespace(dry_run=True), 0.5)
    assert body["onstart"] == "#!/bin/bash\ntrain\n"


def test_relaunch_body_with_no_spec_onstart_still_installs_the_key(local_key):
    st = _spec_state()
    st["launch_spec"].pop("onstart")
    body, _ = replacement._relaunch_body(st, argparse.Namespace(dry_run=True), 0.5)
    assert models.SSH_INJECT_MARKER in body["onstart"]


# MIGRATED (was MIGRATION-BLOCKED, plan §7 batch B2): `_relaunch_body` landed in
# `vastlib.supervise.replacement`, the same module as `_handoff_understudy_body`,
# so the :2643 call reaches a real body and nothing has to be stubbed to make it
# run — which matters here, because stubbing it would have deleted the property
# under test (the understudy INHERITS the relaunch body's ssh install).
# `_ship_b2_pair` is stubbed at `launch.spec`, the module `_resolve_secret`
# resolves it through.
def test_handoff_understudy_body_installs_the_key(monkeypatch, local_key):
    """The run-lane handoff reuses _relaunch_body, so it inherits the fix —
    pin it, because an understudy is a full production replacement box."""
    monkeypatch.setattr(launch_spec, "_ship_b2_pair",
                        lambda name, hours=None, dry_run=False: ("KID", "SEC"))
    st = {"run_id": "r1", "dph_total": 0.60, "last_bid": 0.60,
          "on_demand": 0.50, "remaining_wall_h": 10.0,
          "launch_spec": {"image": "reg/img:tag", "disk": 100,
                          "runtype": "ssh_direct", "env": {"RUN_ID": "r1"},
                          "runset": "rs1", "secret_env_keys": [],
                          "onstart": "#!/bin/bash\ntrain\n"}}
    body, bid, missing = replacement._handoff_understudy_body(
        st, argparse.Namespace(dry_run=True),
        {"id": 999, "min_bid": 0.10, "dph_total": 0.50})
    assert missing == [] and body is not None
    assert models.SSH_INJECT_MARKER in body["onstart"]


def test_jobs_handoff_understudy_does_not_opt_out_of_ssh():
    """The jobs-lane understudy builds its Namespace by hand and passed
    ssh=False until 2026-07-31 — a source-level pin, because the launch itself
    needs a live market to exercise.

    Class-C repoint (plan §7, batch B2): an ABSENCE-assert over a single file
    goes vacuously green the moment the code moves out of that file, which is
    the silent failure mode — so it now scans the flat launcher AND every
    vastlib module. Plan §8 step 6d emptied the launcher: its scan is now a
    tripwire (a launch path re-appearing there is caught) rather than the
    primary arm, and the vastlib walk is what carries the property.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    files = [os.path.join(here, "herdd.py")]
    for root, _dirs, names in os.walk(os.path.join(here, "vastlib")):
        files += [os.path.join(root, n) for n in names if n.endswith(".py")]
    for f in files:
        assert "ssh=False" not in open(f).read(), (
            f"a launch path opted out of ssh in {os.path.relpath(f, here)} — "
            "every box we rent must be ssh-able by default (see "
            "ssh_authorized_keys_snippet)")


# =============================================================================
# legibility: say WHY, instead of leaking `Permission denied (publickey)`
# =============================================================================
def test_ssh_access_warning_silent_on_a_healthy_box(local_key):
    assert ssh_mod.ssh_access_warning(
        {"id": 1, "onstart": ssh_mod.with_ssh_inject("echo hi\n")}) is None


def test_ssh_access_warning_explains_the_real_cause():
    msg = ssh_mod.ssh_access_warning({"id": 46449950, "onstart": "echo hi\n"})
    assert msg is not None
    assert "46449950" in msg
    assert "StrictModes" in msg                  # the actual mechanism
    assert "resume cannot repair" in msg         # what NOT to waste time on
    assert "relaunch" in msg                     # what to do instead


def test_instance_ssh_install_classifies_the_three_populations(local_key):
    assert models.instance_ssh_install(
        {"onstart": ssh_mod.with_ssh_inject("echo hi\n")}) == "v2"
    # a pre-2026-07-31 append-only inject: installs, but cannot repair ownership
    assert models.instance_ssh_install(
        {"onstart": "mkdir -p /root/.ssh && echo K >> "
                    "/root/.ssh/authorized_keys\n"}) == "legacy"
    assert models.instance_ssh_install({"onstart": "echo hi\n"}) == "none"
    assert models.instance_ssh_install({}) == "none"


def test_ssh_access_warning_quiet_on_a_legacy_box():
    # a legacy box usually works — the loud warning is for boxes that CANNOT
    # work. Nagging on every older box would train people to ignore it.
    assert ssh_mod.ssh_access_warning(
        {"id": 1, "onstart": "echo K >> /root/.ssh/authorized_keys\n"}) is None


def _ssh_ns_instance(onstart):
    return {"id": 9, "onstart": onstart, "public_ipaddr": "1.2.3.4",
            "ports": {"22/tcp": [{"HostPort": "222"}]}}


def _run_cmd_ssh(monkeypatch, onstart):
    monkeypatch.setattr(lifecycle, "_get_instance",
                        lambda iid: _ssh_ns_instance(onstart))
    monkeypatch.setattr(ssh_mod, "_pick_ssh_endpoint",
                        lambda i, **k: ("1.2.3.4", 222, "direct"))
    execd = {}
    monkeypatch.setattr(os, "execvp",
                        lambda f, cmd: execd.setdefault("cmd", cmd))
    cli_ssh.run(argparse.Namespace(id=9, exec=None, print=False))
    return execd


def test_cmd_ssh_warns_before_handing_off_to_ssh(monkeypatch, capsys):
    execd = _run_cmd_ssh(monkeypatch, "echo hi\n")             # no install
    assert execd["cmd"][:2] == ["ssh", "-p"]                   # still connects
    assert "StrictModes" in capsys.readouterr().err


def test_cmd_ssh_preflight_is_free_on_a_current_box(monkeypatch, capsys,
                                                    local_key):
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: pytest.fail("must not probe a v2 box"))
    _run_cmd_ssh(monkeypatch, ssh_mod.with_ssh_inject("echo hi\n"))
    err = capsys.readouterr().err            # the debug-hold note still prints
    assert "ssh:" not in err and "StrictModes" not in err


def test_cmd_ssh_probes_a_legacy_box_and_names_strictmodes(monkeypatch, capsys):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a[0], 255, "", "root@1.2.3.4: Permission denied (publickey)."))
    _run_cmd_ssh(monkeypatch, "echo K >> /root/.ssh/authorized_keys\n")
    err = capsys.readouterr().err
    assert "StrictModes" in err and "vastai_kaalia" in err
    assert "relaunch on a different host" in err       # the folk remedy, named


def test_cmd_ssh_probe_silent_when_auth_succeeds(monkeypatch, capsys):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "", ""))
    _run_cmd_ssh(monkeypatch, "echo K >> /root/.ssh/authorized_keys\n")
    assert "StrictModes" not in capsys.readouterr().err


def test_cmd_ssh_probe_failure_never_blocks_the_connection(monkeypatch):
    def boom(*a, **k):
        raise OSError("ssh binary missing")
    monkeypatch.setattr(subprocess, "run", boom)
    execd = _run_cmd_ssh(monkeypatch, "echo K >> /root/.ssh/authorized_keys\n")
    assert execd["cmd"][:2] == ["ssh", "-p"]


def test_cmd_ssh_print_stays_machine_readable(monkeypatch, capsys):
    """--print feeds other scripts: the diagnosis must never reach stdout."""
    monkeypatch.setattr(lifecycle, "_get_instance",
                        lambda iid: {"id": iid, "onstart": "echo hi\n"})
    monkeypatch.setattr(ssh_mod, "_pick_ssh_endpoint",
                        lambda i, **k: ("1.2.3.4", 222, "direct"))
    cli_ssh.run(argparse.Namespace(id=9, exec=None, print=True))
    assert capsys.readouterr().out.strip() == \
        "ssh -p 222 root@1.2.3.4 -o StrictHostKeyChecking=accept-new " \
        "-o LogLevel=ERROR"


# =============================================================================
# attach_ssh_key_soft — belt to the snippet's braces, and never fatal
# =============================================================================
def test_attach_ssh_key_soft_posts_the_key(monkeypatch, local_key):
    seen = {}
    monkeypatch.setattr(api, "request_soft",
                        lambda m, p, b=None, **k:
                        (seen.update(method=m, path=p, body=b), (True, {}, None))[1])
    assert ssh_mod.attach_ssh_key_soft(77) is True
    assert seen["method"] == "POST" and seen["path"] == "v0/instances/77/ssh/"
    assert seen["body"] == {"ssh_key": _PUB}


def test_attach_ssh_key_soft_never_raises(monkeypatch, local_key):
    def boom(*a, **k):
        raise RuntimeError("network is on fire")
    monkeypatch.setattr(api, "request_soft", boom)
    assert ssh_mod.attach_ssh_key_soft(77) is False


def test_attach_ssh_key_soft_noop_without_a_key(monkeypatch):
    monkeypatch.setattr(ssh_mod, "pub_key_text", lambda *a, **k: None)
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: pytest.fail("must not POST"))
    assert ssh_mod.attach_ssh_key_soft(77) is False


# =============================================================================
# `ls` footer — surface un-ssh-able boxes before you need a shell, not after
# =============================================================================
def test_ls_footers_only_boxes_with_no_ssh_install(monkeypatch, capsys,
                                                   local_key):
    boxes = [
        # current install — silent
        {"id": 1, "actual_status": "running", "num_gpus": 1, "gpu_name": "X",
         "dph_total": 0.1, "label": "run:a",
         "onstart": ssh_mod.with_ssh_inject("echo hi\n")},
        # pre-2026-07-31 append-only inject — usually fine, must NOT nag
        {"id": 2, "actual_status": "running", "num_gpus": 1, "gpu_name": "X",
         "dph_total": 0.1, "label": "run:b",
         "onstart": "echo K >> /root/.ssh/authorized_keys\n"},
        # the 46449950 class: nothing installs the key, and nothing can
        {"id": 3, "actual_status": "exited", "num_gpus": 1, "gpu_name": "X",
         "dph_total": 0.1, "label": "run:c", "onstart": "#!/bin/bash\ntrain\n"},
    ]
    monkeypatch.setattr(lifecycle, "_instances", lambda: boxes)
    monkeypatch.setattr(imageref, "image_tag_digest", lambda img: None)
    monkeypatch.setattr(_ls_render, "_market_map",
                        lambda ins, enabled=True, prog=None: {})
    monkeypatch.setattr(jobs_view, "_fold_fleet_jobs",
                        lambda live, prog=None: {})
    cli_ls.run(argparse.Namespace(json=False, minimal=False,
                                      cached=False, no_spot=True))
    out = capsys.readouterr().out
    assert "install NO ssh key" in out
    warn = [l for l in out.splitlines() if "install NO ssh key" in l][0]
    assert "1 box(es)" in warn and "(3)" in warn
