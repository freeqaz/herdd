"""`vastlib.cli.main` — the registry loop is the composition, so test the registry.

`test_vastlib_cli_surface.py` proves the two parser TREES render identically.
This file proves the things a byte diff of `--help` cannot see:

* a command registered twice, or missing, or in the wrong slot — the diff would
  catch the last one but reports it as a wall of text; here it is one name;
* `a.func(a)` still being the dispatch — argparse prints nothing about `func`,
  so a composition root that forgot to dispatch, or dispatched `a.cmd`, looks
  identical in help;
* a mutually-exclusive group that stopped being exclusive. **`--help` renders a
  mutex group and three loose flags exactly the same way** (argparse only shows
  the grouping in `usage:`, and only when the group is on the same line), so
  losing the group is a silent behaviour change: `--handoff --no-handoff`
  starts being accepted;
* the post-hoc `--local` loop covering exactly the seven job subcommands
  `jobs.runlocal._JOB_LOCAL_SUBCOMMANDS` names — a flag-by-flag port turns one
  loop into seven copies, and the eighth subcommand then quietly acquires a
  half-wired local mode.

Mid-wave tolerance
------------------
`cli/` lands one command module per sub-agent. Until every registry entry
exists, importing `vastlib.cli.main` raises — deliberately, because a
silently-dropped subcommand is the failure this refactor must not be able to
produce. The tests that need the composed tree therefore `skip` with the list of
missing modules rather than failing; the registry-shape tests below need no
sibling module at all and run from the first commit.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from typing import Any

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from vastlib.cli import _args, main as cli_main  # noqa: E402

#: The three flags every supervision lane makes mutually exclusive, and the
#: commands that build that group (cli-surface.json `argparse_groups`).
CEILING_HANDOFF_FLAGS = frozenset({"--strict-ceiling", "--handoff", "--no-handoff"})
MUTEX_COMMANDS = ("supervise", "train", "job supervise")


# ------------------------------------------------------------------ helpers


def _subparser_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction[Any] | None:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _command_names(parser: argparse.ArgumentParser) -> list[str]:
    """Canonical subcommand names, in argparse insertion order, aliases excluded.

    `_choices_actions` is the list argparse itself prints, one entry per
    `add_parser` call; `choices` would also contain every alias as its own key
    and in an order that is not the printed one.
    """
    action = _subparser_action(parser)
    if action is None:
        return []
    return [choice.dest for choice in action._choices_actions]


def _aliases(parser: argparse.ArgumentParser, name: str) -> set[str]:
    action = _subparser_action(parser)
    assert action is not None
    target = action.choices[name]
    return {key for key, value in action.choices.items() if value is target} - {name}


def _module_name(command: str) -> str:
    return command.replace("-", "_")


def _missing_registry_modules() -> list[str]:
    return [name for name in cli_main._REGISTRY
            if importlib.util.find_spec(f"vastlib.cli.{name}") is None]


def _vastlib_parser() -> argparse.ArgumentParser:
    """Build the real tree by driving `main()` and intercepting `parse_args`."""
    missing = _missing_registry_modules()
    if missing:
        pytest.skip(f"mid-wave: cli command modules not landed yet: {missing}")
    surface = pytest.importorskip("test_vastlib_cli_surface")
    with surface._pinned_environment():
        return surface._drive_main_for_parser(cli_main.main)  # type: ignore[no-any-return]


def _flat_parser() -> argparse.ArgumentParser:
    surface = pytest.importorskip("test_vastlib_cli_surface")
    herdd = pytest.importorskip("herdd")
    if not hasattr(herdd, "_add_cmd"):
        pytest.skip("herdd.py has been thinned (plan §8 step 6d) — no flat arm left")
    with surface._pinned_environment():
        return surface._drive_main_for_parser(herdd.main)  # type: ignore[no-any-return]


# ------------------------------------------------------- registry shape


def test_registry_has_no_duplicates() -> None:
    assert len(cli_main._REGISTRY) == len(set(cli_main._REGISTRY))


def test_registry_entries_are_module_names_not_command_names() -> None:
    """`dash-cache` is a command; `dash_cache` is the module. Only the latter
    may appear here — `importlib` would fail on the former at startup."""
    for name in cli_main._REGISTRY:
        assert name.isidentifier(), name


def test_registry_covers_every_flat_command_exactly_once() -> None:
    flat = _flat_parser()
    expected = [_module_name(name) for name in _command_names(flat)]
    assert list(cli_main._REGISTRY) == expected


def test_registry_order_is_the_printed_order() -> None:
    """argparse lists subcommands in insertion order, so `_REGISTRY`'s order IS
    printed output. Compared against the composed tree rather than the flat one
    so this still binds after the thinning."""
    parser = _vastlib_parser()
    assert [_module_name(name) for name in _command_names(parser)] == list(cli_main._REGISTRY)


def test_registry_modules_all_expose_add_parser() -> None:
    missing = _missing_registry_modules()
    if missing:
        pytest.skip(f"mid-wave: cli command modules not landed yet: {missing}")
    for name in cli_main._REGISTRY:
        module = importlib.import_module(f"vastlib.cli.{name}")
        assert callable(getattr(module, "add_parser", None)), name


# ------------------------------------------------------------- dispatch


def test_every_top_level_command_sets_func() -> None:
    parser = _vastlib_parser()
    action = _subparser_action(parser)
    assert action is not None
    for name in _command_names(parser):
        func = action.choices[name].get_default("func")
        assert callable(func), f"{name} registered no `func` default"


def test_main_dispatches_a_func_a(monkeypatch: pytest.MonkeyPatch) -> None:
    """The seam plan §5 freezes: `a = ap.parse_args(); a.func(a)`.

    Patched at `ArgumentParser.parse_args` so the whole real tree is still
    built first — this is a test of the composition root, not of a stub.
    """
    missing = _missing_registry_modules()
    if missing:
        pytest.skip(f"mid-wave: cli command modules not landed yet: {missing}")
    seen: list[argparse.Namespace] = []
    namespace = argparse.Namespace(cmd="whoami", func=seen.append)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args",
                        lambda self, *a, **k: namespace)
    monkeypatch.setattr("vastlib.core.config.load_env", lambda: None)
    cli_main.main()

    assert seen == [namespace], "main() did not dispatch exactly `a.func(a)`"


def test_main_loads_env_before_building_parsers(monkeypatch: pytest.MonkeyPatch) -> None:
    """`.env` populates `os.environ`, and several parser defaults read it — so
    the order in the prologue is load-bearing, not cosmetic."""
    missing = _missing_registry_modules()
    if missing:
        pytest.skip(f"mid-wave: cli command modules not landed yet: {missing}")
    order: list[str] = []
    monkeypatch.setattr("vastlib.core.config.load_env", lambda: order.append("load_env"))
    original = _args._add_cmd
    monkeypatch.setattr(_args, "_add_cmd",
                        lambda *a, **k: (order.append("add_cmd"), original(*a, **k))[1])
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args",
                        lambda self, *a, **k: argparse.Namespace(func=lambda _a: None))
    cli_main.main()

    assert order[0] == "load_env"
    assert "add_cmd" in order


# --------------------------------------------------------------- aliases


@pytest.mark.parametrize(("command", "alias"), [("stop", "park"), ("start", "resume")])
def test_the_only_two_aliases_survive(command: str, alias: str) -> None:
    """`stop (park)` and `start (resume)` are the whole alias surface. Dropping
    one changes the printed listing AND 404s a spelling that appears in runbooks
    (cli-surface.json hazard H11)."""
    parser = _vastlib_parser()
    assert _aliases(parser, command) == {alias}


def test_no_other_command_grew_an_alias() -> None:
    parser = _vastlib_parser()
    aliased = {name for name in _command_names(parser) if _aliases(parser, name)}
    assert aliased == {"stop", "start"}


# ----------------------------------------------- mutually-exclusive groups


@pytest.mark.parametrize("path", MUTEX_COMMANDS)
def test_ceiling_handoff_group_is_still_exclusive(path: str) -> None:
    """`--strict-ceiling` / `--handoff` / `--no-handoff` must remain ONE
    `add_mutually_exclusive_group`. A flag-by-flag port renders identically and
    silently starts accepting `--handoff --no-handoff` together."""
    parser: argparse.ArgumentParser = _vastlib_parser()
    for part in path.split():
        action = _subparser_action(parser)
        assert action is not None, f"{path}: no subparsers below {parser.prog}"
        assert part in action.choices, f"{path}: `{part}` is not registered"
        parser = action.choices[part]

    groups = [{option for act in group._group_actions for option in act.option_strings}
              for group in parser._mutually_exclusive_groups]
    assert CEILING_HANDOFF_FLAGS in [g & CEILING_HANDOFF_FLAGS for g in groups
                                     if g & CEILING_HANDOFF_FLAGS], (
        f"{path}: the ceiling/handoff flags are no longer one exclusive group "
        f"(groups seen: {groups})")


# ------------------------------------------------- the post-hoc --local loop


def test_local_flag_covers_exactly_the_declared_job_subcommands() -> None:
    """`--local` is hung on seven `job` subparsers by ONE loop over
    `jobs.runlocal._JOB_LOCAL_SUBCOMMANDS`. Assert the wiring against that
    single source of truth, from the outside."""
    from vastlib.jobs import runlocal

    parser = _vastlib_parser()
    top = _subparser_action(parser)
    assert top is not None
    job = _subparser_action(top.choices["job"])
    assert job is not None

    with_local = {name for name in _command_names(top.choices["job"])
                  if any("--local" in act.option_strings for act in job.choices[name]._actions)}
    assert with_local == set(runlocal._JOB_LOCAL_SUBCOMMANDS)


def test_local_flag_is_the_last_flag_on_each_of_them() -> None:
    """Order within a parser is printed output too. The flat loop runs AFTER all
    seven subparsers are built, so `--local` is last on each; a port that added
    it inline would move it up the help page."""
    from vastlib.jobs import runlocal

    parser = _vastlib_parser()
    top = _subparser_action(parser)
    assert top is not None
    job = _subparser_action(top.choices["job"])
    assert job is not None

    for name in runlocal._JOB_LOCAL_SUBCOMMANDS:
        options = [act.option_strings for act in job.choices[name]._actions]
        assert options[-1] == ["--local"], f"job {name}: --local is not the last flag"
