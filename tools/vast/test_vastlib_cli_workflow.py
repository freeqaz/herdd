"""`vastlib.cli.workflow` vs the flat `herdd.py` workflow group — the port's acceptance bar.

Why this file exists
--------------------
Step 6 of the vast-tooling refactor (plan §8) moves the CLI into `vastlib.cli`
while `herdd.py` is still alive and still owns the same parser tree. That
overlap is a one-time opportunity: for the length of this wave BOTH parser
trees exist in one process, so "did the port change the CLI" is answerable by
construction rather than by review. Every test below either diffs the two trees
or pins something the diff cannot see.

What is checked, and why each one is here
-----------------------------------------
1. **`format_help()` bytes**, group + all seven subcommands. Prog names, flag
   strings, defaults, wrapped help text and the `docs:` epilog are all in
   there; a moved comma fails.
2. **Structural action-by-action compare.** `--help` prints neither `dest` nor
   `required` nor `default` for a store_true, and prints nothing at all for
   the `wfcmd` subparsers action. Those are the parts a help diff cannot see.
3. **Subcommand ORDER.** argparse prints subcommands in insertion order, so
   the order is printed output — but it is also the thing a mechanical port is
   most likely to permute, and `choices` is a dict, so equality of the SET
   would pass while the page changed.
4. **The parser-build-time constant read.** `--takeover`'s help interpolates
   `POLL_INTERVAL_S * HEARTBEAT_STALE_MULT`, which is evaluated on EVERY
   `herdd` invocation of EVERY command. The port must read those constants
   from `vastlib.workflows.ctl` (the cycle stays broken) and the product must
   still be the number the flat file printed.
5. **The Zone E re-exec literal.** `--detach` re-execs
   `<python> tools/vast/herdd.py workflow run|resume …`. In the flat module
   that path was `os.path.abspath(__file__)`; inside the package `__file__` is
   four levels deeper, so the anchor is re-derived — and re-derived wrong is a
   detached controller that either crash-loops or drives the wrong tree.
6. **The dispatch shape**, `a.func(a)` -> `a.wffunc(a)`, and that each
   subparser binds ITS module's handler.
7. **`# moved-from:` markers**, because plan §7.1 regenerates the rename table
   from them and a missing marker is a symbol the test migration cannot find.

What is deliberately NOT here
-----------------------------
* Reconcile-loop behavior, event folding, spec validation. That is
  `test_workflow.py` / `test_workflowmeta.py` against `workflows/ctl.py`; this
  file is about the CLI seam only.
* Any assertion that would still pass after `herdd.py` is thinned. The
  comparison arm disappears at wave 6d by design; when it does, the flat-arm
  tests here become the fixture baseline's job, not this file's.
"""
import argparse
import importlib
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import herdd  # noqa: E402
# `workflowctl` is a re-export shim over `vastlib.workflows.ctl` since step 7,
# so it is imported here only to prove the shim still resolves — every
# assertion below reads the port directly.
import workflowctl as flat_wc  # noqa: E402,F401
from vastlib.cli import _args, _docs  # noqa: E402
from vastlib.cli import workflow as wfcli  # noqa: E402
from vastlib.storage import b2  # noqa: E402
from vastlib.workflows import ctl as workflowctl  # noqa: E402

_SUBCOMMANDS = ["plan", "run", "status", "logs", "pull", "cancel", "resume"]
_PKG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "vastlib", "cli", "workflow")


def _module(name):
    """The subcommand MODULE, always by `sys.modules` lookup.

    `from vastlib.cli.workflow import run` does NOT give the module: the group
    package defines a handler called `run` (the `cli/` convention) which
    overwrites the submodule attribute the import machinery set. That shadow is
    deliberate and pinned by `test_the_run_attribute_is_the_group_handler`;
    this helper is what every test uses so no other test depends on it.
    """
    return importlib.import_module(f"vastlib.cli.workflow.{name}")


class _Captured(Exception):
    """Carries the top-level parser out of `herdd.main()` before dispatch."""

    def __init__(self, parser):
        self.parser = parser


def _subparsers_action(parser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise AssertionError(f"{parser.prog} has no subparsers action")


@pytest.fixture()
def flat_group(monkeypatch):
    """The `workflow` parser as `herdd.main()` builds it.

    `main()` has no seam that returns the parser, so `parse_args` is patched
    to raise with `self` — the same method the port manifest used to read the
    live flag table out of the flat file, and the only way to get the REAL
    tree (reading the source would re-implement argparse).

    SINCE STEP 6d THIS IS NO LONGER A SECOND TREE. `herdd.main` IS
    `vastlib.cli.main.main`, so every `*_matches_flat` test below now compares
    the ported tree with itself and can only fail if the interception breaks.
    They are kept because that interception is still worth exercising and
    costs nothing; the byte-level reference that actually guards the surface is
    the FROZEN fixture (`.port_manifests/cli-surface.json`, captured at rev
    7a177e2a and asserted by `test_vastlib_cli_surface.py`), which is where a
    real diff would show up. Do not read a green here as independent evidence.
    """
    def _boom(self, *a, **kw):
        raise _Captured(self)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _boom)
    with pytest.raises(_Captured) as excinfo:
        herdd.main()
    top = excinfo.value.parser
    return _subparsers_action(top).choices["workflow"]


@pytest.fixture()
def ported_group():
    """The same group, built by `vastlib.cli.workflow.add_parser`.

    `prog="herdd"` and `dest="cmd"` mirror `main()`'s own root parser, since
    prog names propagate into every child's help page.
    """
    ap = argparse.ArgumentParser(
        prog="herdd", description="Vast.ai control CLI for upstream-monorepo",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    return wfcli.add_parser(sub, _args._add_cmd)


# --------------------------------------------------------------------------- #
# 1-3. the help tree
# --------------------------------------------------------------------------- #
def test_group_help_is_byte_identical(flat_group, ported_group):
    assert ported_group.format_help() == flat_group.format_help()
    assert ported_group.format_usage() == flat_group.format_usage()


@pytest.mark.parametrize("name", _SUBCOMMANDS)
def test_subcommand_help_is_byte_identical(name, flat_group, ported_group):
    flat = _subparsers_action(flat_group).choices[name]
    mine = _subparsers_action(ported_group).choices[name]
    assert mine.format_help() == flat.format_help()


def test_subcommand_order_matches_flat(flat_group, ported_group):
    flat_order = list(_subparsers_action(flat_group).choices)
    assert flat_order == _SUBCOMMANDS, "flat file changed; update this port"
    assert list(_subparsers_action(ported_group).choices) == flat_order


def _action_shape(action):
    return {
        "class": type(action).__name__,
        "opts": list(action.option_strings),
        "dest": action.dest,
        "default": action.default,
        "nargs": action.nargs,
        "choices": (list(action.choices) if isinstance(action.choices, dict)
                    else action.choices),
        "required": action.required,
        "help": action.help,
        "metavar": action.metavar,
    }


def test_group_actions_match_flat_structurally(flat_group, ported_group):
    flat = [_action_shape(a) for a in flat_group._actions]
    mine = [_action_shape(a) for a in ported_group._actions]
    assert mine == flat
    # dest/required of the nested dispatcher are invisible to --help and are
    # what `a.wffunc` hangs off.
    assert _subparsers_action(ported_group).dest == "wfcmd"
    assert _subparsers_action(ported_group).required is True


@pytest.mark.parametrize("name", _SUBCOMMANDS)
def test_subcommand_actions_match_flat_structurally(name, flat_group, ported_group):
    flat = _subparsers_action(flat_group).choices[name]
    mine = _subparsers_action(ported_group).choices[name]
    assert [_action_shape(a) for a in mine._actions] == [_action_shape(a) for a in flat._actions]


def test_group_epilog_uses_the_ported_doc_pointers(flat_group, ported_group):
    # H4/MED-H7: the epilog is printed output built from constants that now
    # exist in both trees. Equal today; this pins WHICH copy the port reads.
    for doc in (_docs.DOC_WORKFLOW, _docs.DOC_JOBS, _docs.DOC_SKILL):
        assert doc in ported_group.epilog
    assert ported_group.epilog == flat_group.epilog


def test_no_mutually_exclusive_groups_were_lost(flat_group, ported_group):
    # H6: the three groups in the surface belong to supervise/train/job
    # supervise, none to workflow — pinned so a later edit cannot add one to
    # only one arm unnoticed.
    def groups(p):
        return [[a.option_strings for a in g._group_actions]
                for g in p._mutually_exclusive_groups]

    assert groups(ported_group) == groups(flat_group) == []
    for name in _SUBCOMMANDS:
        assert groups(_subparsers_action(ported_group).choices[name]) == \
               groups(_subparsers_action(flat_group).choices[name])


# --------------------------------------------------------------------------- #
# 4. the parser-build-time constant read
# --------------------------------------------------------------------------- #
def test_staleness_constants_are_the_frozen_product_the_help_text_quotes():
    """The `== flat_wc.X` half was dropped at step 7: `workflowctl.py` became a
    re-export shim, so it compared each constant with itself. The product is
    what actually matters — the `--takeover` help text interpolates it."""
    assert flat_wc.POLL_INTERVAL_S is workflowctl.POLL_INTERVAL_S   # one object
    assert workflowctl.POLL_INTERVAL_S == 30
    assert workflowctl.HEARTBEAT_STALE_MULT == 3
    assert workflowctl.POLL_INTERVAL_S * workflowctl.HEARTBEAT_STALE_MULT == 90


def test_takeover_help_interpolates_the_live_product(ported_group):
    (takeover,) = [a for a in _subparsers_action(ported_group).choices["run"]._actions
                   if a.option_strings == ["--takeover"]]
    product = workflowctl.POLL_INTERVAL_S * workflowctl.HEARTBEAT_STALE_MULT
    assert f"older than {product}s" in takeover.help


def test_the_interpolation_reads_the_vastlib_constants(monkeypatch, ported_group):
    """Not a tautology check: it proves the f-string resolves through
    `vastlib.workflows.ctl` (so the cycle stays broken) rather than through a
    re-literalled 90 or the flat module."""
    monkeypatch.setattr(workflowctl, "POLL_INTERVAL_S", 7)
    ap = argparse.ArgumentParser(prog="herdd")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rebuilt = wfcli.add_parser(sub, _args._add_cmd)
    (takeover,) = [a for a in _subparsers_action(rebuilt).choices["run"]._actions
                   if a.option_strings == ["--takeover"]]
    assert "older than 21s" in takeover.help


# --------------------------------------------------------------------------- #
# 5. the Zone E re-exec literal
# --------------------------------------------------------------------------- #
def test_herdd_script_anchor_matches_the_flat_module():
    _entry = _module("_entry")
    assert _entry.HERDD_SCRIPT == os.path.abspath(herdd.__file__)
    assert os.path.isfile(_entry.HERDD_SCRIPT)


@pytest.mark.parametrize("verb,positional", [
    ("run", "wf/e2.py"),
    ("resume", "20260713T000000-e2-3553"),
])
def test_detach_reexec_argv_is_the_frozen_literal(monkeypatch, verb, positional):
    mod = _module(verb)
    seen = {}

    def _fake(*a, **kw):
        seen["argv"] = list(kw["argv"])
        return workflowctl.EXIT_OK, {"wf_id": "wf-1", "status": "detached"}

    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(workflowctl, f"{verb}_workflow", _fake)
    ns = argparse.Namespace(path=positional, wf_id=positional, detach=True,
                            takeover=True, json=False, detached_controller=False)
    with pytest.raises(SystemExit) as e:
        mod.run(ns)
    assert e.value.code == workflowctl.EXIT_OK
    assert seen["argv"] == [sys.executable, os.path.abspath(herdd.__file__),
                            "workflow", verb, positional, "--takeover"]


def test_detach_argv_omits_takeover_when_not_asked(monkeypatch):
    run_mod = _module("run")
    seen = {}
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(workflowctl, "run_workflow",
                        lambda *a, **kw: (seen.setdefault("argv", list(kw["argv"])),
                                          (workflowctl.EXIT_OK, {"wf_id": "x"}))[1])
    ns = argparse.Namespace(path="wf/e2.py", wf_id=None, detach=True, takeover=False,
                            json=False, detached_controller=False)
    with pytest.raises(SystemExit):
        run_mod.run(ns)
    assert seen["argv"][-1] == "wf/e2.py"


# --------------------------------------------------------------------------- #
# 6. dispatch
# --------------------------------------------------------------------------- #
def test_group_handler_dispatches_to_wffunc():
    called = []
    ns = argparse.Namespace(wffunc=called.append)
    wfcli.run(ns)
    assert called == [ns]


def test_each_subparser_binds_its_own_module_handler(ported_group):
    for name in _SUBCOMMANDS:
        sp = _subparsers_action(ported_group).choices[name]
        mod = _module(name)
        assert sp.get_default("wffunc") is mod.run, name
    assert ported_group.get_default("func") is wfcli.run


def test_each_subcommand_dispatches_to_its_own_modules_run(ported_group):
    """What the `cmd_workflow_<name>` name-check was really pinning: `wffunc`
    is the handler OF THAT SUBCOMMAND and not a neighbour's.

    The old spelling compared the flat handler's `__name__` against
    `cmd_workflow_<name>`; the port renamed every one of them to `<module>.run`
    (recorded in `_EXPECTED_MARKERS` above as `herdd.cmd_workflow_X -> run`),
    so after 6d that assertion checked a name that no longer exists on a tree
    that is the ported one anyway. Identity against the module object is
    stronger than the name ever was — a copy-paste that wired `cancel` to
    `logs.run` passes a name check on neither spelling but was invisible to the
    flat-vs-ported diff, which saw two identical trees."""
    for name in _SUBCOMMANDS:
        fn = _subparsers_action(ported_group).choices[name].get_default("wffunc")
        assert fn is _module(name).run, name


# --------------------------------------------------------------------------- #
# 7. handlers delegate to workflows.ctl, and exit with its rc
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _no_b2(monkeypatch):
    """Every handler starts with `_ensure_b2_remote()`; patching it HERE (on
    `vastlib.storage.b2`, its ported home) is also the assertion that the port
    repointed that call — a handler still reaching `herdd._ensure_b2_remote`
    would shell out to rclone in a unit test."""
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)


def test_plan_delegates_and_exits_with_ctl_rc(monkeypatch, capsys):
    plan = _module("plan")
    seen = {}

    def _fake_plan(path, **kw):
        seen.update(path=path, kw=kw)
        return workflowctl.EXIT_OK, {"wf_id": "wf-7"}

    monkeypatch.setattr(workflowctl, "plan_workflow", _fake_plan)
    with pytest.raises(SystemExit) as e:
        plan.run(argparse.Namespace(path="wf/e2.py", online=False, json=False))
    assert e.value.code == workflowctl.EXIT_OK
    assert seen["path"] == "wf/e2.py"
    assert seen["kw"]["actor"].startswith("cli:")
    assert ">> planned workflow wf-7" in capsys.readouterr().out


def test_plan_online_passes_the_image_resolver(monkeypatch):
    plan = _module("plan")
    import imageref
    seen = {}
    monkeypatch.setattr(workflowctl, "plan_workflow",
                        lambda path, **kw: (seen.update(kw),
                                            (workflowctl.EXIT_OK, {"wf_id": "w"}))[1])
    with pytest.raises(SystemExit):
        plan.run(argparse.Namespace(path="wf/e2.py", online=True, json=True))
    assert seen["online"] is True
    assert seen["image_resolver"] is imageref.image_ref_digest
    assert isinstance(seen["now_epoch"], float)


def test_plan_error_path_prints_to_stderr_and_exits_nonzero(monkeypatch, capsys):
    plan = _module("plan")
    monkeypatch.setattr(workflowctl, "plan_workflow",
                        lambda *a, **kw: (workflowctl.EXIT_INVALID, {"error": "bad stage"}))
    with pytest.raises(SystemExit) as e:
        plan.run(argparse.Namespace(path="wf/e2.py", online=False, json=False))
    assert e.value.code == workflowctl.EXIT_INVALID
    assert "error: bad stage" in capsys.readouterr().err


def test_status_renders_the_meta_table_and_degrades_without_extras(monkeypatch, capsys):
    status = _module("status")
    from vastlib.workflows import meta as workflowmeta
    view = {"wf_id": "wf-1", "status": "running", "stages": {}}
    monkeypatch.setattr(workflowctl, "status_workflow",
                        lambda wf_id, **kw: (workflowctl.EXIT_OK, view))
    monkeypatch.setattr(workflowctl, "status_extras",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no creds")))
    monkeypatch.setattr(workflowmeta, "format_status_table",
                        lambda v, extras=None: f"TABLE extras={extras!r}")
    with pytest.raises(SystemExit) as e:
        status.run(argparse.Namespace(wf_id="wf-1", json=False))
    assert e.value.code == workflowctl.EXIT_OK
    assert "TABLE extras=None" in capsys.readouterr().out


def test_status_json_prints_the_folded_view(monkeypatch, capsys):
    status = _module("status")
    view = {"wf_id": "wf-1", "status": "failed"}
    monkeypatch.setattr(workflowctl, "status_workflow",
                        lambda wf_id, **kw: (workflowctl.EXIT_FAILED, view))
    with pytest.raises(SystemExit) as e:
        status.run(argparse.Namespace(wf_id="wf-1", json=True))
    assert e.value.code == workflowctl.EXIT_FAILED
    assert json.loads(capsys.readouterr().out) == view


def test_logs_filters_by_stage_and_says_so_when_empty(monkeypatch, capsys):
    logs = _module("logs")
    events = [{"stage": "gen", "kind": "a"}, {"stage": "score", "kind": "b"}]
    monkeypatch.setattr(workflowctl, "logs_workflow",
                        lambda wf_id, **kw: (workflowctl.EXIT_OK, {"events": events}))
    with pytest.raises(SystemExit):
        logs.run(argparse.Namespace(wf_id="wf-1", stage="gen"))
    out = capsys.readouterr().out
    assert json.loads(out.strip()) == events[0]

    with pytest.raises(SystemExit):
        logs.run(argparse.Namespace(wf_id="wf-1", stage="nope"))
    assert "(no events for workflow wf-1 stage=nope)" in capsys.readouterr().out


def test_pull_dest_defaults_under_the_repo_root(monkeypatch):
    pull = _module("pull")
    from vastlib.jobs import submit
    monkeypatch.setattr(submit, "_repo_root", lambda: "/repo")
    ns = argparse.Namespace(wf_id="wf-1", dest=None)
    assert pull._workflow_pull_dest(ns) == os.path.join("/repo", "out", "workflows", "wf-1")
    assert pull._workflow_pull_dest(argparse.Namespace(wf_id="wf-1", dest="/tmp/x")) == "/tmp/x"


def test_pull_maps_zone_s_artifact_errors_to_exit_artifact(monkeypatch, capsys):
    pull = _module("pull")
    import jobmeta
    monkeypatch.setattr(workflowctl, "read_spec", lambda wf_id, **kw: object())
    monkeypatch.setattr(workflowctl, "pull_workflow",
                        lambda *a, **kw: (_ for _ in ()).throw(jobmeta.JobmetaError("gone")))
    with pytest.raises(SystemExit) as e:
        pull.run(argparse.Namespace(wf_id="wf-1", dest="/tmp/x"))
    assert e.value.code == workflowctl.EXIT_ARTIFACT
    assert "error: gone" in capsys.readouterr().err


def test_cancel_prints_the_terminal_wording(monkeypatch, capsys):
    cancel = _module("cancel")
    seen = {}
    monkeypatch.setattr(workflowctl, "cancel_workflow",
                        lambda wf_id, **kw: (seen.update(kw),
                                             (workflowctl.EXIT_CANCELLED, {}))[1])
    with pytest.raises(SystemExit) as e:
        cancel.run(argparse.Namespace(wf_id="wf-1", reason="operator"))
    assert e.value.code == workflowctl.EXIT_CANCELLED
    assert seen["reason"] == "operator"
    assert seen["actor"].startswith("cli:")
    assert "cancelled (terminal, non-resumable)" in capsys.readouterr().out


@pytest.mark.parametrize("module", ["run", "resume"])
def test_detach_unavailable_prints_the_foreground_command(monkeypatch, capsys, module):
    mod = _module(module)

    def _raise(*a, **kw):
        raise workflowctl.DetachUnavailable("python herdd.py workflow run wf.py")

    monkeypatch.setattr(workflowctl, f"{module}_workflow", _raise)
    ns = argparse.Namespace(path="wf.py", wf_id="wf-1", detach=True, takeover=False,
                            json=False, detached_controller=False)
    with pytest.raises(SystemExit) as e:
        mod.run(ns)
    assert e.value.code == workflowctl.EXIT_INVALID
    assert "python herdd.py workflow run wf.py" in capsys.readouterr().err


@pytest.mark.parametrize("module", ["run", "resume"])
def test_refused_controller_claim_exits_credential(monkeypatch, capsys, module):
    mod = _module(module)

    def _raise(*a, **kw):
        raise workflowctl.WorkflowCtlError("a live controller holds wf-1")

    monkeypatch.setattr(workflowctl, f"{module}_workflow", _raise)
    ns = argparse.Namespace(path="wf.py", wf_id="wf-1", detach=False, takeover=True,
                            json=True, detached_controller=False)
    with pytest.raises(SystemExit) as e:
        mod.run(ns)
    assert e.value.code == workflowctl.EXIT_CREDENTIAL
    assert json.loads(capsys.readouterr().out)["error"] == "a live controller holds wf-1"


def test_run_surfaces_the_error_behind_a_nonzero_rc(monkeypatch, capsys):
    """The 2026-07-15 regression this line was added for: without it a
    config/spec failure printed a bare `rc=1` and the --detach crash-loop was
    undiagnosable from the unit journal."""
    run_mod = _module("run")
    monkeypatch.setattr(workflowctl, "run_workflow",
                        lambda *a, **kw: (workflowctl.EXIT_INVALID,
                                          {"wf_id": "wf-1", "error": "stage cfg invalid"}))
    with pytest.raises(SystemExit) as e:
        run_mod.run(argparse.Namespace(path="wf.py", wf_id=None, detach=False,
                                       takeover=False, json=False,
                                       detached_controller=False))
    assert e.value.code == workflowctl.EXIT_INVALID
    assert "error: stage cfg invalid" in capsys.readouterr().err


def test_run_passes_a_live_controller_deps_builder(monkeypatch):
    run_mod = _module("run")
    seen = {}
    monkeypatch.setattr(workflowctl, "run_workflow",
                        lambda *a, **kw: (seen.update(kw), (workflowctl.EXIT_OK, {}))[1])
    monkeypatch.setattr(workflowctl, "build_live_controller_deps",
                        lambda wf, wf_id, **kw: {"wf": wf, "wf_id": wf_id, "actor": kw["actor"]})
    with pytest.raises(SystemExit):
        run_mod.run(argparse.Namespace(path="wf.py", wf_id=None, detach=False,
                                       takeover=False, json=False,
                                       detached_controller=False))
    deps = seen["controller_deps"]("WF", "wf-9")
    assert deps["wf"] == "WF" and deps["wf_id"] == "wf-9"
    assert deps["actor"].startswith("cli:")


# --------------------------------------------------------------------------- #
# 8. the rename table's inputs
# --------------------------------------------------------------------------- #
_EXPECTED_MARKERS = {
    "__init__.py": ["herdd.cmd_workflow -> run"],
    "plan.py": ["herdd.cmd_workflow_plan -> run"],
    "run.py": ["herdd.cmd_workflow_run -> run"],
    "status.py": ["herdd.cmd_workflow_status -> run"],
    "logs.py": ["herdd.cmd_workflow_logs -> run"],
    "pull.py": ["herdd._workflow_pull_dest", "herdd.cmd_workflow_pull -> run"],
    "cancel.py": ["herdd.cmd_workflow_cancel -> run"],
    "resume.py": ["herdd.cmd_workflow_resume -> run"],
}


def test_every_ported_symbol_carries_its_moved_from_marker():
    for fname, expected in _EXPECTED_MARKERS.items():
        text = open(os.path.join(_PKG_DIR, fname), encoding="utf-8").read()
        found = re.findall(r"^# moved-from: (.+)$", text, flags=re.M)
        assert found == expected, fname


def test_markers_sit_directly_above_a_definition():
    """Grammar rule 1 (vastlib/README.md §2): no blank line between a marker
    and the def/class/assignment it labels — the table generator reads the
    NEXT line."""
    for fname in _EXPECTED_MARKERS:
        lines = open(os.path.join(_PKG_DIR, fname), encoding="utf-8").read().splitlines()
        for i, line in enumerate(lines):
            if line.startswith("# moved-from: "):
                assert re.match(r"^(def |class |[A-Za-z_][A-Za-z0-9_]* =)", lines[i + 1]), \
                    f"{fname}:{i + 2}"


def test_the_frozen_surface_fixture_covers_every_workflow_subcommand():
    """The successor to `test_flat_handlers_still_exist_...`, which demanded
    `herdd.cmd_workflow_<name>` so the flat diff arm could not quietly become
    a self-comparison. Wave 6d thinned `herdd.py` exactly as that test
    predicted, so its own instruction is followed here: the reference moves to
    the FROZEN CLI-surface fixture — captured from the flat parser at rev
    7a177e2a, before the thinning, and compared against the vastlib tree by
    `test_vastlib_cli_surface.py`.

    What this asserts is the precondition that test cannot assert about itself:
    the fixture still CONTAINS the whole `workflow` subtree. A fixture that
    lost these nodes would let the surface test pass while checking nothing,
    and every `*_matches_flat` diff in this file is a self-comparison now.
    """
    fixture = json.loads(
        open(os.path.join(os.path.dirname(_PKG_DIR), "..", "..",
                          "testfixtures",
                          "cli_surface_flat_7a177e2a.json"),
             encoding="utf-8").read())
    nodes = fixture["nodes"]
    assert "workflow" in nodes, "the frozen surface fixture lost `workflow`"
    missing = [n for n in _SUBCOMMANDS if f"workflow {n}" not in nodes]
    assert not missing, f"frozen fixture is missing workflow subcommands: {missing}"


def test_the_run_attribute_is_the_group_handler():
    """The deliberate shadow, stated once so it is a decision and not a
    surprise: `workflow.run` is the group's dispatcher; the `run` SUBCOMMAND
    module is reached through `sys.modules`. `job run-local` lands in
    `run_local.py`, so no other group has this collision."""
    assert wfcli.run is not _module("run")
    assert callable(wfcli.run) and not hasattr(wfcli.run, "add_parser")
    assert sys.modules["vastlib.cli.workflow.run"].run.__module__ == \
        "vastlib.cli.workflow.run"


def test_import_is_cheap_no_module_level_work():
    """A `cli/` module is imported on EVERY `herdd` invocation, including
    `herdd --help`. The only module-level calls allowed here are the path
    arithmetic in `_entry.py`; an API call, an `.env` read or a config load at
    import time would tax every command in the surface."""
    import ast
    for fname in sorted(os.listdir(_PKG_DIR)):
        if not fname.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(_PKG_DIR, fname), encoding="utf-8").read())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import,
                                 ast.ImportFrom, ast.Expr, ast.If)):
                continue
            for call in [n for n in ast.walk(node) if isinstance(n, ast.Call)]:
                fn = call.func
                assert (isinstance(fn, ast.Attribute)
                        and isinstance(fn.value, ast.Attribute)
                        and isinstance(fn.value.value, ast.Name)
                        and fn.value.value.id == "os"
                        and fn.value.attr == "path"), f"{fname}: module-level call"
