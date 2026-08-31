"""`vastlib.cli.job` — the `herdd job` group, held to its surface and its two bodies.

Why this file exists
--------------------
The group is 14 subcommands, 330 lines of parser, and exactly two handler
bodies that had nowhere below `cli` to land (`attach`, `supervise`). Three
different kinds of regression are possible here and each needs its own kind of
check:

1. **Surface drift.** Every help string, default, flag order, exclusive group
   and the post-hoc `--local` loop must reproduce the flat parser exactly. That
   is checked against the frozen capture in
   `testfixtures/cli_surface_flat_7a177e2a.json` — the SAME fixture the
   whole-CLI gate uses — but restricted to the `job` subtree, so this file goes
   red for a defect in THIS assignment while `cli/main.py` is still being
   built. The walker, the normalizer and the differ are imported from
   `test_vastlib_cli_surface.py` rather than re-implemented: a second walker
   would be a second definition of "the surface".

2. **Wiring.** `jobfunc` on every leaf, `func=runlocal.cmd_job` on the group
   (it is NOT a pure dispatcher — it activates the local lane first), and
   `--local` on exactly `_JOB_LOCAL_SUBCOMMANDS`, LAST in the option list.

3. **The two bodies.** `cmd_job_attach`'s checked start step (an unchecked ssh
   prints a success banner over a box where no daemon ever started — observed
   2026-08-01) and `cmd_job_supervise`'s budget refusal, its
   delegate-before-any-work order, and the `finally` that reaps a pre-cutover
   understudy on the `sys.exit(3)` path.

What is deliberately NOT here
-----------------------------
* No re-testing of the twelve handlers that live below `cli`
  (`test_vastlib_jobs_{submit,view,control,runlocal}.py` own them). This file
  checks that the parser reaches them, not what they do.
* No network, no ssh, no B2, no vast API. Every seam is patched as a MODULE
  ATTRIBUTE, which is also the property being asserted: the port kept the
  patch idiom alive by never doing `from … import fn`.
* No assertion on which function object a `jobfunc` default holds. The
  cross-arm compare deliberately normalizes callable names away
  (`cmd_job_status` legitimately becomes `vastlib.cli.job.status.run`); what is
  asserted is that a callable is there and that calling it reaches the ported
  body.

Provenance: created 2026-08-16 alongside `vastlib/cli/job/`, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

import test_vastlib_cli_surface as surface             # noqa: E402  the shared walker

from vastlib.boxes import lifecycle, remote            # noqa: E402
from vastlib.boxes import ssh as boxes_ssh             # noqa: E402
from vastlib.cli import _args, _docs                   # noqa: E402
from vastlib.cli import job as cli_job                 # noqa: E402
from vastlib.cli.job import attach as cli_attach       # noqa: E402
from vastlib.cli.job import supervise as cli_supervise  # noqa: E402
from vastlib.core import config                        # noqa: E402
from vastlib.fleet import client                       # noqa: E402
from vastlib.jobs import bundle, runlocal              # noqa: E402
from vastlib.launch import spec                        # noqa: E402
from vastlib.storage import b2                         # noqa: E402
from vastlib.supervise import handoff, job_lane        # noqa: E402


# --------------------------------------------------------------------------- #
# Parser construction — the same root `main()` builds, with only this group on
# it. Node `prog`s derive from the root prog, so it has to be the real one.
# --------------------------------------------------------------------------- #

def build_job_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="herdd", description="Vast.ai control CLI for upstream-monorepo",
        epilog=_args._docs_epilog(_docs.DOC_SKILL, _docs.DOC_README, _docs.DOC_TRAINING),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cli_job.add_parser(sub, _args._add_cmd)
    return ap


@pytest.fixture
def job_nodes() -> dict[str, dict[str, object]]:
    """The `job` subtree of the vastlib parser, canonicalized like the fixture."""
    with surface._pinned_environment():
        tree = surface.build_tree(build_job_parser(),
                                  source="vastlib.cli.job", seam="add_parser")
    return surface.strip_callable_names(tree["nodes"])


# --------------------------------------------------------------------------- #
# 1. Surface parity against the frozen flat capture
# --------------------------------------------------------------------------- #

def test_job_subtree_matches_the_frozen_flat_surface(job_nodes) -> None:
    """THE acceptance bar for this assignment (plan §8 step 6).

    Every prog, usage line, epilog, flag, default, metavar, choice, help
    string, subcommand order and mutually-exclusive group of all 20 `job` nodes,
    compared byte-for-byte against the flat parser's frozen capture.

    20, not the original 15: `job dlq` and its three verbs (add/ls/restore)
    joined the surface on 2026-08-26 and `job flush` on 2026-08-27, each frozen
    through the fixture's own `--add-command` path rather than hand-spliced.
    """
    want_all = surface.strip_callable_names(surface.load_fixture()["nodes"])
    paths = [p for p in want_all if p == "job" or p.startswith("job ")]
    assert len(paths) == 20, f"fixture holds {len(paths)} job nodes, expected 20"

    missing = [p for p in paths if p not in job_nodes]
    assert not missing, f"subcommands not built by vastlib.cli.job: {missing}"
    extra = [p for p in job_nodes if p and p not in want_all]
    assert not extra, f"subcommands the flat surface does not have: {extra}"

    findings = surface.diff_trees(
        {"nodes": {p: want_all[p] for p in paths}},
        {"nodes": {p: job_nodes[p] for p in paths}},
    )
    assert not findings, surface.format_findings(findings)


def test_subcommand_order_is_the_flat_order(job_nodes) -> None:
    """argparse lists subcommands in INSERTION order, so the order is printed
    output — and `run-local`/`supervise` come after the `--local` loop."""
    names = [s["name"] for s in job_nodes["job"]["subcommands"]]
    # `dlq` sits next to `orphans` on purpose: they are the two ticket-hygiene
    # commands, and `job orphans --resolve` routes through the DLQ. `flush` sits
    # next to `cancel`: same marker mechanism, same poll, opposite intent.
    assert names == ["submit", "status", "wait", "logs", "pull", "ls", "defs",
                     "orphans", "dlq", "attach", "retarget", "requeue",
                     "cancel", "flush", "run-local", "supervise"]


# --------------------------------------------------------------------------- #
# 2. Wiring
# --------------------------------------------------------------------------- #

def test_local_flag_is_the_roster_and_is_added_last(job_nodes) -> None:
    """The post-hoc loop, both halves.

    WHICH parsers get `--local` is `runlocal._JOB_LOCAL_SUBCOMMANDS` and
    nothing else (a re-literalled list is the drift this catches), and WHERE it
    lands is the end of the option list — because the flat file adds it after
    every subcommand is built. Both are printed output.
    """
    with_local = sorted(
        path.split(" ", 1)[1]
        for path, node in job_nodes.items()
        if path.startswith("job ")
        and any("--local" in a["option_strings"] for a in node["actions"])
    )
    assert with_local == sorted(runlocal._JOB_LOCAL_SUBCOMMANDS)
    for name in runlocal._JOB_LOCAL_SUBCOMMANDS:
        actions = job_nodes[f"job {name}"]["actions"]
        assert actions[-1]["option_strings"] == ["--local"], (
            f"job {name}: --local is not the last option; the post-hoc loop moved")


def test_every_leaf_carries_a_jobfunc_and_the_group_carries_func() -> None:
    """`a.func(a)` at the seam, `a.jobfunc(a)` one level down — and the group's
    `func` is `runlocal.cmd_job`, which activates the LOCAL lane before
    dispatching. A pure-dispatcher port would silently drop `--local`."""
    ap = build_job_parser()
    pj = ap._subparsers._group_actions[0].choices["job"]      # type: ignore[union-attr]
    assert pj._defaults["func"] is runlocal.cmd_job
    jsub = [a for a in pj._actions if isinstance(a, argparse._SubParsersAction)][0]

    def _check(name: str, sp: argparse.ArgumentParser) -> None:
        """The invariant is about DISPATCHABLE LEAVES. `job dlq` is a nested
        group (add/ls/restore) with `required=True`, so it is never dispatched
        itself — argparse rejects it before `jobfunc` would be read. Recursing
        keeps the contract honest instead of parking a dead default on the
        group to satisfy the assert, and it newly covers the dlq leaves."""
        nested = [x for x in sp._actions
                  if isinstance(x, argparse._SubParsersAction)]
        if nested:
            for sub_name, sub_sp in nested[0].choices.items():
                _check(f"{name} {sub_name}", sub_sp)
            return
        assert callable(sp._defaults.get("jobfunc")), f"job {name}: no jobfunc default"

    for name, sp in jsub.choices.items():
        _check(name, sp)


@pytest.mark.parametrize("name,mod_attr", [
    ("submit", "vastlib.jobs.submit.cmd_job_submit"),
    ("status", "vastlib.jobs.view.cmd_job_status"),
    ("wait", "vastlib.jobs.view.cmd_job_wait"),
    ("logs", "vastlib.jobs.view.cmd_job_logs"),
    ("pull", "vastlib.jobs.view.cmd_job_pull"),
    ("ls", "vastlib.jobs.view.cmd_job_ls"),
    ("defs", "vastlib.jobs.view.cmd_job_defs"),
    ("orphans", "vastlib.jobs.control.cmd_job_orphans"),
    ("retarget", "vastlib.jobs.control.cmd_job_retarget"),
    ("requeue", "vastlib.jobs.control.cmd_job_requeue"),
    ("cancel", "vastlib.jobs.control.cmd_job_cancel"),
    ("run-local", "vastlib.jobs.runlocal.cmd_job_run_local"),
])
def test_shim_dispatch_reaches_the_ported_body_by_module_attribute(
        name, mod_attr, monkeypatch) -> None:
    """The twelve argparse-only shims.

    Patching the LOWER RING's module attribute must steer the dispatch — that
    is the whole reason the shims call `view.cmd_job_status(a)` rather than
    binding the function at import. A `from … import` port passes every surface
    test and fails exactly here.
    """
    import importlib
    mod_name, fn_name = mod_attr.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    seen: list[object] = []
    monkeypatch.setattr(mod, fn_name, lambda a: seen.append(a))

    ap = build_job_parser()
    pj = ap._subparsers._group_actions[0].choices["job"]      # type: ignore[union-attr]
    jsub = [a for a in pj._actions if isinstance(a, argparse._SubParsersAction)][0]
    ns = argparse.Namespace()
    jsub.choices[name]._defaults["jobfunc"](ns)
    assert seen == [ns], f"job {name}: dispatch did not reach {mod_attr}"


def test_exclusive_ceiling_group_still_rejects_the_pair() -> None:
    """Hazard H6: exclusivity is invisible to `--help`, so only a parse proves
    it survived. All three pairings must be rejected with argparse's exit 2."""
    ap = build_job_parser()
    for pair in (["--handoff", "--no-handoff"],
                 ["--strict-ceiling", "--handoff"],
                 ["--strict-ceiling", "--no-handoff"]):
        with pytest.raises(SystemExit) as e:
            ap.parse_args(["job", "supervise", "1", *pair])
        assert e.value.code == 2, pair
    # the singletons still parse
    assert ap.parse_args(["job", "supervise", "1", "--no-handoff"]).handoff is False


def test_handoff_default_is_the_config_switch(monkeypatch) -> None:
    """SAFE-OFF on the jobs lane is `config.jobs_handoff_enabled()`, read at
    parser-BUILD time exactly as the flat `set_defaults` does. A hard-coded
    False renders identically and ignores the switch."""
    for value in (True, False):
        monkeypatch.setattr(config, "jobs_handoff_enabled", lambda: value)
        ap = build_job_parser()
        assert ap.parse_args(["job", "supervise", "7"]).handoff is value
        # an explicit flag still wins — this is a default, not a prohibition
        assert ap.parse_args(["job", "supervise", "7", "--handoff"]).handoff is True


def test_importing_the_group_touches_no_network() -> None:
    """`cli/` modules must be import-cheap: no API call, no socket, at import.

    Run in a fresh interpreter with `socket.socket` poisoned, because an import
    side effect only fires once per process and this process already imported
    the package at collection time.
    """
    # The connect PATHS are poisoned, not the socket CLASS: `ssl` subclasses
    # `socket.socket` at import, so replacing the class breaks the stdlib
    # instead of catching the defect.
    code = (
        "import socket, urllib.request\n"
        "def _boom(*a, **k):\n"
        "    raise AssertionError('vastlib.cli.job reached the network at import')\n"
        "socket.socket.connect = _boom\n"
        "socket.create_connection = _boom\n"
        "urllib.request.urlopen = _boom\n"
        "import vastlib.cli.job\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=str(VAST_DIR), timeout=120)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# --------------------------------------------------------------------------- #
# 3a. `cmd_job_attach` — the body that had nowhere below `cli` to land
# --------------------------------------------------------------------------- #

@pytest.fixture
def attach_env(monkeypatch, tmp_path):
    """Every seam of `cmd_job_attach` stubbed as a module attribute."""
    f = tmp_path / "jobd.sh"
    f.write_text("#!/bin/sh\n")
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setenv("B2_S3_ENDPOINT", "https://s3.example")
    monkeypatch.delenv("CRED_BROKER_URL", raising=False)
    monkeypatch.setattr(bundle, "_job_attach_files", lambda: [str(f)])
    monkeypatch.setattr(bundle, "_jobd_import_gate", lambda files, **kw: None)
    monkeypatch.setattr(bundle, "_stage_jobd_bootstrap", lambda *a, **k: "0" * 64)
    monkeypatch.setattr(spec, "_ephemeral_hours", lambda *a, **k: 24.0)
    monkeypatch.setattr(spec, "_ship_b2_env", lambda *a, **k: [("B2_KEY_ID", "k")])
    monkeypatch.setattr(spec, "_minted_expiry", lambda *a, **k: None)
    monkeypatch.setattr(spec, "_b2_eu_pairs", lambda: [])
    monkeypatch.setattr(spec, "_r2_tc_pairs", lambda: [])
    monkeypatch.setattr(lifecycle, "_get_instance", lambda iid: {"actual_status": "running"})
    monkeypatch.setattr(boxes_ssh, "_pick_ssh_endpoint", lambda i, **k: ("h", 22, None))
    monkeypatch.setattr(boxes_ssh, "_warn_ssh_access", lambda i: None)
    registered: list[tuple[object, str]] = []
    monkeypatch.setattr(remote, "_broker_register",
                        lambda iid, nonce: registered.append((iid, nonce)))
    watched: list[object] = []
    monkeypatch.setattr(client, "fleet_watch_best_effort",
                        lambda *a, **k: watched.append(a) or True)
    return {"registered": registered, "watched": watched}


def _ns_attach(**kw) -> argparse.Namespace:
    base = dict(id=42, dry_run=False, no_idle_park=False, idle_park_grace=None,
                no_job_deadline=None, fleet_watch=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_attach_dry_run_spends_nothing(attach_env, monkeypatch, capsys) -> None:
    """`--dry-run` returns before the instance lookup and before any ssh."""
    def _no_run(*a, **k):
        raise AssertionError("dry-run reached subprocess")
    monkeypatch.setattr(cli_attach.subprocess, "run", _no_run)
    monkeypatch.setattr(lifecycle, "_get_instance",
                        lambda iid: pytest.fail("dry-run hit the vast API"))
    cli_attach.cmd_job_attach(_ns_attach(dry_run=True))
    out = capsys.readouterr().out
    assert "[dry-run/attach]" in out and "NO ssh, NO spend" in out


def test_attach_refuses_when_the_start_step_fails(attach_env, monkeypatch,
                                                  capsys) -> None:
    """The 2026-08-01 defect, pinned.

    B2 staging succeeds whether or not the box is reachable, so an unchecked
    start prints the success banner over a box with no daemon. A non-zero rc on
    the start ssh must exit, and must NOT print `jobd attached`.
    """
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        rc = 1 if any("nohup bash" in str(c) for c in cmd) else 0
        return subprocess.CompletedProcess(cmd, rc)

    monkeypatch.setattr(cli_attach.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as e:
        cli_attach.cmd_job_attach(_ns_attach())
    assert "could not start jobd" in str(e.value)
    assert "jobd attached" not in capsys.readouterr().out


def test_attach_happy_path_registers_the_nonce_and_repoints_the_boot_sha(
        attach_env, monkeypatch, capsys) -> None:
    """The success path: env pushed, daemon started, broker told, onstart hook
    installed, LAUNCH-pinned boot sha repointed (the 2026-07-31 rollback)."""
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli_attach.subprocess, "run", fake_run)
    cli_attach.cmd_job_attach(_ns_attach(fleet_watch=True))

    joined = "\n".join(str(c) for c in calls)
    assert "jobd-autostart" in joined, "onstart persistence hook not installed"
    assert "jobs/jobd-boot/" + "0" * 64 in joined, "boot sha not repointed"
    assert len(attach_env["registered"]) == 1
    iid, nonce = attach_env["registered"][0]
    assert iid == 42 and len(nonce) == 32           # 16 bytes hex, never the raw key
    assert attach_env["watched"], "--fleet-watch did not register the box"
    assert ">> jobd attached to 42" in capsys.readouterr().out


def test_attach_survives_a_failed_boot_repoint_but_says_so(attach_env, monkeypatch,
                                                           capsys) -> None:
    """Best-effort stays best-effort — and stays LOUD: a failed re-stage means
    the next resume rolls this attach back."""
    monkeypatch.setattr(cli_attach.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0))
    monkeypatch.setattr(bundle, "_stage_jobd_bootstrap",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("b2 down")))
    cli_attach.cmd_job_attach(_ns_attach())
    out = capsys.readouterr().out
    assert "boot-bundle repoint FAILED" in out
    assert ">> jobd attached to 42" in out


def test_attach_reads_the_idle_park_knobs_with_getattr(attach_env, monkeypatch) -> None:
    """`supervise`'s `_reattach` builds a MINIMAL Namespace; the knobs must be
    optional, not AttributeErrors."""
    monkeypatch.setattr(cli_attach.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0))
    cli_attach.cmd_job_attach(argparse.Namespace(id=7, dry_run=False))


def test_broker_register_is_a_silent_noop_without_broker_env(monkeypatch,
                                                             capsys) -> None:
    """It landed in `boxes.remote` (three command modules reach it). Absence of
    either broker variable is a no-op, and nothing raises — attach must never
    fail on broker absence."""
    monkeypatch.delenv("CRED_BROKER_URL", raising=False)
    monkeypatch.delenv("CRED_BROKER_ADMIN_TOKEN", raising=False)
    remote._broker_register(1, "deadbeef")
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------- #
# 3b. `cmd_job_supervise` — the legacy inline driver
# --------------------------------------------------------------------------- #

def _ns_sup(**kw) -> argparse.Namespace:
    base = dict(id=9, budget=5.0, dry_run=False, max_bid=None, strict_ceiling=False)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def sup_env(monkeypatch):
    monkeypatch.setattr(client, "fleet_delegate_job_supervise", lambda a: False)
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda *a, **k: None)
    monkeypatch.setattr(cli_supervise.time, "sleep", lambda s: None)
    state = {"reaped": 0, "ticks": 0}
    monkeypatch.setattr(handoff, "_job_handoff_reap_on_exit",
                        lambda jc, hf: state.__setitem__("reaped", state["reaped"] + 1))
    monkeypatch.setattr(job_lane, "job_supervise_init",
                        lambda a: ({"iid": 9, "handoff_on": True}, {}))
    return state


def test_supervise_requires_a_budget(monkeypatch) -> None:
    """The only thing between an unattended babysitter and an open invoice."""
    monkeypatch.setattr(client, "fleet_delegate_job_supervise",
                        lambda a: pytest.fail("refusal must precede delegation"))
    with pytest.raises(SystemExit) as e:
        cli_supervise.cmd_job_supervise(_ns_sup(budget=None))
    assert "--budget USD is required" in str(e.value)


def test_supervise_dry_run_is_exempt_from_the_budget(sup_env, monkeypatch) -> None:
    monkeypatch.setattr(job_lane, "job_supervise_tick", lambda jc, hf: "drained")
    cli_supervise.cmd_job_supervise(_ns_sup(budget=None, dry_run=True))


def test_supervise_delegates_to_fleetd_before_any_work(monkeypatch) -> None:
    """FLEETD_DESIGN §6: when the daemon is up the watch is HANDED OVER, and
    nothing else happens — no B2 remote, no init, no loop."""
    monkeypatch.setattr(client, "fleet_delegate_job_supervise", lambda a: True)
    monkeypatch.setattr(b2, "_ensure_b2_remote",
                        lambda *a, **k: pytest.fail("ran work after delegating"))
    monkeypatch.setattr(job_lane, "job_supervise_init",
                        lambda a: pytest.fail("ran work after delegating"))
    cli_supervise.cmd_job_supervise(_ns_sup())


def test_supervise_sets_handoff_can_complete(sup_env, monkeypatch) -> None:
    """Defect #61: this driver keeps ticking until a verdict, so unlike fleetd
    it CAN complete a handoff — and it says so on the namespace."""
    seen: list[bool] = []
    monkeypatch.setattr(job_lane, "job_supervise_init",
                        lambda a: (seen.append(a.handoff_can_complete),
                                   ({"iid": 9, "handoff_on": False}, {}))[1])
    monkeypatch.setattr(job_lane, "job_supervise_tick", lambda jc, hf: "drained")
    cli_supervise.cmd_job_supervise(_ns_sup())
    assert seen == [True]


def test_supervise_loops_until_a_verdict(sup_env, monkeypatch) -> None:
    ticks = iter([None, None, "self_parked"])
    monkeypatch.setattr(job_lane, "job_supervise_tick", lambda jc, hf: next(ticks))
    cli_supervise.cmd_job_supervise(_ns_sup())
    assert sup_env["reaped"] == 1


def test_supervise_reaps_the_understudy_on_the_unrecoverable_exit(sup_env,
                                                                  monkeypatch) -> None:
    """F3: `sys.exit(3)` raises SystemExit, which runs `finally` — the reap must
    fire on THAT path too, not just on the budget branch."""
    monkeypatch.setattr(job_lane, "job_supervise_tick", lambda jc, hf: "unrecoverable")
    with pytest.raises(SystemExit) as e:
        cli_supervise.cmd_job_supervise(_ns_sup())
    assert e.value.code == 3
    assert sup_env["reaped"] == 1


def test_supervise_does_not_reap_when_handoff_is_off(sup_env, monkeypatch) -> None:
    monkeypatch.setattr(job_lane, "job_supervise_init",
                        lambda a: ({"iid": 9, "handoff_on": False}, {}))
    monkeypatch.setattr(job_lane, "job_supervise_tick", lambda jc, hf: "drained")
    cli_supervise.cmd_job_supervise(_ns_sup())
    assert sup_env["reaped"] == 0


# --------------------------------------------------------------------------- #
# Provenance markers — the rename table is generated from them
# --------------------------------------------------------------------------- #

def test_ported_bodies_carry_their_moved_from_markers() -> None:
    """`.port_manifests/gen_rename_table.py` reads these, and 72 seam sites
    read the table. A missing marker is a symbol the test migration cannot
    find (README §2 rule 1: directly above the def, no blank line)."""
    for path, symbol in (
        (VAST_DIR / "vastlib/cli/job/attach.py", "cmd_job_attach"),
        (VAST_DIR / "vastlib/cli/job/supervise.py", "cmd_job_supervise"),
        (VAST_DIR / "vastlib/boxes/remote.py", "_broker_register"),
    ):
        lines = path.read_text(encoding="utf-8").splitlines()
        idx = [n for n, ln in enumerate(lines) if ln.startswith(f"def {symbol}(")]
        assert idx, f"{path.name}: no top-level def {symbol}"
        assert lines[idx[0] - 1] == f"# moved-from: herdd.{symbol}", (
            f"{path.name}: {symbol} has no moved-from marker on the line above it")
