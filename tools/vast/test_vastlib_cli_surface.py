"""CLI-surface parity harness — the §8 step-6 cutover gate, implemented.

Why this file exists
--------------------
`herdd`'s help tree is a public contract. ~30 callers bind the literal
`tools/vast/herdd.py` path, the `herdd-reaper.timer` systemd unit runs
`reap -y` every 15 minutes, four dashboard spawn sites hold a frozen
`dash-cache` argv, the `herdd` skill and ~550 markdown references name flags
by hand. Plan §8 step 6 therefore requires the flat parser tree and the
`vastlib.cli` parser tree to be **byte-identical**: every prog name, usage
string, flag, default, help string, subcommand order, alias, mutually-exclusive
group, and the post-hoc `--local` loop over `_JOB_LOCAL_SUBCOMMANDS`.

This module is that comparison. It has three parts:

  1. a **walker** (`walk_parser`) that renders any `argparse.ArgumentParser`
     into a canonical, deterministic JSON tree — one node per subcommand path,
     each node carrying its structured actions *and* its rendered `--help`
     text, so a mismatch can be reported both as "flag `--cuda` default 13.0
     != 12.0" and as a line diff of the help page;
  2. a **capture** of the flat `herdd.py` tree, frozen into the committed
     fixture `testfixtures/cli_surface_flat_<rev>.json`;
  3. the **comparison** of `vastlib.cli.main`'s tree against that fixture,
     reported per command so a half-built `cli/` names exactly which commands
     are missing rather than collapsing to one opaque failure.

Two lives, one file
-------------------
*Before the thinning (plan §8 step 6d)* both arms are alive: the fixture is
re-proved against a fresh capture of the flat parser on every run
(`test_fixture_matches_live_flat_parser`), so it cannot rot while `herdd.py`
still owns the surface, and the vastlib arm is compared against it.

*After the thinning* the flat side is **gone** — `tools/vast/herdd.py`
becomes a `sys.path` bootstrap that calls `vastlib.cli.main:main`, so
intercepting its parser would capture the vastlib parser and the comparison
would be tautological. The live-flat test detects that state (the parser
plumbing `_add_cmd` / `add_fleet_parser` no longer exists on the module) and
skips with that reason. From then on **the committed fixture is the frozen
reference** and `test_vastlib_cli_surface_matches_fixture` is the only arm:
the fixture is the record of what the surface was at rev 7a177e2a, and any
intended change to it is a deliberate, reviewed fixture edit.

Regenerating the fixture
------------------------
    cd tools/vast && python3 test_vastlib_cli_surface.py --write

Only legitimate while the flat arm is alive. Do not regenerate to make a red
test green — a diff here is either a port defect or a CLI change that needs an
owner decision.

Amending the fixture, post-thinning (added 2026-08-18)
-----------------------------------------------------
    cd tools/vast && python3 test_vastlib_cli_surface.py --amend dash-cache

`--write` is dead once the flat arm is, and that left an intended CLI change
with no landing path at all: the surface could not move without a red test, so
`DASH_GPUS_DEFAULT` sat un-editable for want of a way to re-freeze one help
string. That is the freeze failing safe in the wrong direction — it stopped
being a detector of unintended drift and became a ban on intended change.

`--amend` restores the intended half WITHOUT weakening the detector, because
it re-freezes only the commands you NAME and refuses if anything else moved
(and equally if a named command did not move, or if a command was added or
removed). You cannot use it to make a surprise go away: a surprise is by
definition in a command you did not name, and it aborts and prints it.

The freeze still means what it meant. An unreviewed diff is still a failure.

Adding a command, post-thinning (added 2026-08-20)
--------------------------------------------------
    cd tools/vast && python3 test_vastlib_cli_surface.py --add-command "fleet hosts"

`--amend` deliberately refuses a STRUCTURAL change ("a command was added or
removed"), and post-thinning that left the same gap one level up: the surface
could not GROW without a red test and a hand-edited JSON fixture. Adding
`fleet hosts` hit exactly that. Hand-splicing the fixture is the outcome to
avoid — it is unreviewable, it silently skips the manifest's node count, and
it teaches the next person that the freeze is a formality.

`--add-command` is the narrow, auditable version of that edit. It re-freezes
the NAMED new nodes and their ancestor chain — a parent's `subcommands` list,
usage line and rendered help move as a mechanical consequence of the child
existing — and refuses if anything OUTSIDE that closure moved, if a named path
is not actually new, or if any command was REMOVED (removal stays an owner
decision with no tool; it is the direction that breaks callers). It also bumps
`counts.command_nodes` in `.port_manifests/cli-surface.json`, because the
inventory cross-check ties the two together and a green fixture with a stale
manifest is the second red test nobody was warned about.

What it cannot do is hide a surprise, for the same reason `--amend` cannot: a
surprise is by definition outside the closure you named, and it aborts and
prints it.

Environment dependence (manifest hazard H5)
-------------------------------------------
Three things make a raw help tree machine-dependent: `main()` reads
`herdd.yaml` (repo file + `~/.config/herdd/herdd.yaml` override +
`HERDD_CONFIG`) for several defaults, `load_env()` populates `os.environ`
from the nearest `.env`, and `format_help()` wraps to `$COLUMNS`. So:

* **Pin.** Every capture runs inside `_pinned_environment()` — fixed
  `COLUMNS`, no user config, no `HERDD_CONFIG`, `os.environ` restored
  afterwards (measured: capturing without the restore leaks the developer's
  `.env` into the rest of the pytest process). What remains is the committed
  `tools/vast/herdd.yaml`, identical for both arms and for every checkout.
* **Normalize** the genuinely host-derived strings — the configured image,
  `$HOME`, the repo root, the cwd — to placeholders, scoped by field so the
  `launch --image` *default* (yaml) and its *help text* (the
  `_EXPECTED_DEFAULT_IMAGE` source constant) get different placeholders even
  though they hold the same bytes today. `normalizations.applied` records
  every place a substitution fired.
* **Record, don't hide, the rest.** `config_dependent` lists every help-tree
  value that moves when `herdd.yaml` moves — *measured* by perturbing
  `load_herdd_config` and re-walking, not hand-listed. Those values stay
  literal in the fixture: normalizing them would also hide a port that
  hardcoded what the flat code reads from config, which is precisely the
  defect class this gate exists for. `capture_env` records the one residual
  (PyYAML presence changes how the yaml parses), and a parity failure on one
  of these entries prints a "check the config environment first" note.

What is deliberately NOT here
-----------------------------
* No behavior testing. This compares the *surface*, never what a command does.
* No mutation of either arm. The flat capture restores `parse_args` and
  `os.environ`; nothing here writes to the source tree except the explicit
  `--write` regeneration path.
* No assertion that a command's `func` default points at any particular
  function. The dispatch *shape* (`a.func(a)`, a callable on every leaf) is
  checked; the callable's name is normalized away in the cross-arm compare
  because `cmd_train` legitimately becomes `vastlib.cli.train.run`.

Provenance: created 2026-08-16, plan §8 step 6, rev 7a177e2a. Manifest:
`.port_manifests/cli-surface.json` (hazards H4/H5/H6/H11 are the reason this
file records constants-in-help, the yaml-dependent default, the exclusive
groups and the aliases explicitly).
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import importlib
import importlib.util
import json
import os
import re
import sys
from collections.abc import Iterator, Sequence
from typing import Any

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
FIXTURE_REV = "7a177e2a"
FIXTURE_DIR = os.path.join(HERE, "testfixtures")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, f"cli_surface_flat_{FIXTURE_REV}.json")
MANIFEST_PATH = os.path.join(HERE, ".port_manifests", "cli-surface.json")

#: Terminal width pinned for `format_help()`. argparse asks
#: `shutil.get_terminal_size()`, which honours `$COLUMNS` — without this the
#: rendered help wraps differently under pytest-xdist, a CI runner and a human
#: terminal, and every node would diff.
HELP_COLUMNS = "100"

SCHEMA_VERSION = 2

#: Parser plumbing that only exists while `herdd.py` still owns the surface.
#: Their absence is how this file detects the post-6d thin launcher.
FLAT_PLUMBING_ATTRS = ("_add_cmd", "add_search_filters", "add_fleet_parser", "main")

#: Seams tried, in order, to obtain the vastlib parser without dispatching.
#: A dedicated builder is preferred; the `main()` interception is the fallback
#: for the same reason it works on the flat side.
VASTLIB_PARSER_SEAMS = ("build_parser", "make_parser", "_build_parser", "build_root_parser")

_ADDR_RE = re.compile(r"0x[0-9a-fA-F]{6,}")


# --------------------------------------------------------------- JSON atoms

def _sanitize_repr(text: str) -> str:
    """Strip `at 0x7f...` addresses so a repr fallback stays deterministic."""
    return _ADDR_RE.sub("0xADDR", text)


def _callable_name(fn: Any) -> str:
    return str(getattr(fn, "__name__", None) or _sanitize_repr(repr(fn)))


def _jsonable(value: Any) -> Any:
    """Faithful, deterministic JSON for an argparse default/const/metavar.

    Container *type* is preserved (a tuple default is not a list default), sets
    are ordered by repr, callables collapse to their name, and anything else
    falls back to an address-scrubbed repr.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_jsonable(v) for v in value]}
    if isinstance(value, (set, frozenset)):
        items = sorted((_jsonable(v) for v in value), key=repr)
        return {"__type__": type(value).__name__, "items": items}
    if isinstance(value, dict):
        return {"__type__": "dict", "items": {str(k): _jsonable(v) for k, v in value.items()}}
    if callable(value):
        return {"__callable__": _callable_name(value)}
    return {"__repr__": _sanitize_repr(repr(value))}


# ------------------------------------------------------------------ walker

def _action_record(action: argparse.Action) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "cls": type(action).__name__,
        "option_strings": list(action.option_strings),
        "dest": action.dest,
        "required": bool(action.required),
        "nargs": _jsonable(action.nargs),
        "const": _jsonable(action.const),
        "default": _jsonable(action.default),
        "type": None if action.type is None else _callable_name(action.type),
        "metavar": _jsonable(action.metavar),
        "help": action.help,
    }
    if isinstance(action, argparse._SubParsersAction):
        # `choices` here maps names (including aliases) to parser objects; the
        # names are recorded as the node's subcommand order instead. What the
        # top-level listing actually renders is the pseudo-action row, which
        # carries the alias-decorated metavar ("stop (park)") and per-command
        # help — H11's alias contract lives in exactly this list.
        rec["choices"] = None
        rec["choice_rows"] = [
            {"name": ch.dest, "metavar": _jsonable(ch.metavar), "help": ch.help}
            for ch in action._choices_actions
        ]
    else:
        rec["choices"] = None if action.choices is None else [_jsonable(c) for c in action.choices]
    for extra in ("version", "deprecated"):
        if hasattr(action, extra):
            rec[extra] = _jsonable(getattr(action, extra))
    return rec


def _action_key(rec: dict[str, Any]) -> str:
    """Stable identity for an action inside one node, for the flag-named diff."""
    opts = rec["option_strings"]
    return "/".join(opts) if opts else f"<positional {rec['dest']}>"


def _group_records(parser: argparse.ArgumentParser) -> list[dict[str, Any]]:
    return [
        {
            "title": g.title,
            "description": g.description,
            "members": [_action_key(_action_record(a)) for a in g._group_actions],
        }
        for g in parser._action_groups
    ]


def _mutually_exclusive_records(parser: argparse.ArgumentParser) -> list[dict[str, Any]]:
    """H6: exclusivity is invisible to `--help`, so it is recorded structurally.

    Porting the three `--strict-ceiling | --handoff | --no-handoff` groups as
    plain flags renders identically and silently stops argparse rejecting the
    pair. That regression is only catchable here.
    """
    return [
        {
            "required": bool(g.required),
            "members": [_action_key(_action_record(a)) for a in g._group_actions],
        }
        for g in parser._mutually_exclusive_groups
    ]


def _rendered_help(parser: argparse.ArgumentParser) -> list[str]:
    with _pinned_columns():
        return parser.format_help().splitlines()


def _rendered_usage(parser: argparse.ArgumentParser) -> list[str]:
    with _pinned_columns():
        return parser.format_usage().splitlines()


def _node_record(parser: argparse.ArgumentParser, aliases: list[str]) -> dict[str, Any]:
    return {
        "prog": parser.prog,
        "aliases": aliases,
        "usage_attr": parser.usage,
        "description": parser.description,
        "epilog_lines": None if parser.epilog is None else parser.epilog.splitlines(),
        "formatter_class": _callable_name(parser.formatter_class),
        "prefix_chars": parser.prefix_chars,
        "add_help": bool(parser.add_help),
        "allow_abbrev": bool(parser.allow_abbrev),
        "conflict_handler": parser.conflict_handler,
        "argument_default": _jsonable(parser.argument_default),
        "parser_defaults": {k: _jsonable(v) for k, v in sorted(parser._defaults.items())},
        "argument_groups": _group_records(parser),
        "mutually_exclusive": _mutually_exclusive_records(parser),
        "actions": [_action_record(a) for a in parser._actions],
        "subcommands": [],
        "usage_lines": _rendered_usage(parser),
        "help_lines": _rendered_help(parser),
    }


def _subparser_actions(parser: argparse.ArgumentParser) -> list[argparse._SubParsersAction]:
    return [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]


def walk_parser(parser: argparse.ArgumentParser) -> dict[str, dict[str, Any]]:
    """Depth-first canonical tree: `{"": root, "job": …, "job submit": …}`.

    Key insertion order is the traversal order, which is the order argparse
    itself lists the subcommands — so `json.dumps(..., sort_keys=False)` keeps
    subcommand order inside the compared bytes rather than only inside a
    per-node list.
    """
    nodes: dict[str, dict[str, Any]] = {}

    def visit(p: argparse.ArgumentParser, path: str, aliases: list[str]) -> None:
        if path in nodes:  # pragma: no cover - a duplicated path is a port bug
            raise AssertionError(f"duplicate command path in parser tree: {path!r}")
        node = _node_record(p, aliases)
        nodes[path] = node
        for action in _subparser_actions(p):
            # One parser object may be registered under several names; the
            # first is the command, the rest are aliases (H11: only stop/park
            # and start/resume today, and losing them 404s `herdd park`).
            by_id: dict[int, list[str]] = {}
            for name, sub in action.choices.items():
                by_id.setdefault(id(sub), []).append(name)
            seen: set[int] = set()
            for name, sub in action.choices.items():
                if id(sub) in seen:
                    continue
                seen.add(id(sub))
                names = by_id[id(sub)]
                child_path = f"{path} {name}".strip()
                node["subcommands"].append(
                    {"name": name, "aliases": names[1:], "path": child_path}
                )
                visit(sub, child_path, names[1:])

    visit(parser, "", [])
    return nodes


# ------------------------------------------------------- environment pinning

@contextlib.contextmanager
def _pinned_columns() -> Iterator[None]:
    prev = os.environ.get("COLUMNS")
    os.environ["COLUMNS"] = HELP_COLUMNS
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("COLUMNS", None)
        else:
            os.environ["COLUMNS"] = prev


@contextlib.contextmanager
def _pinned_environment() -> Iterator[None]:
    """Pin every host input the help tree can see, and restore afterwards.

    `main()` calls `load_env()`, which walks up to six directories looking for
    a `.env` and `setdefault`s every key into `os.environ` — capturing the
    parser would otherwise leak the developer's `.env` into the rest of the
    pytest process. The user-level `herdd.yaml` override and `HERDD_CONFIG`
    are removed so both arms read the committed `tools/vast/herdd.yaml` only
    (manifest H5).
    """
    saved_env = dict(os.environ)
    patched: list[tuple[Any, str, Any]] = []
    try:
        os.environ["COLUMNS"] = HELP_COLUMNS
        os.environ.pop("HERDD_CONFIG", None)
        missing = os.path.join(HERE, "testfixtures", "_no_such_user_config.yaml")
        for mod_name in ("vastconf", "vastlib.core.config"):
            try:
                mod = importlib.import_module(mod_name)
            except ImportError:
                continue
            if hasattr(mod, "_USER_CONFIG"):
                patched.append((mod, "_USER_CONFIG", mod._USER_CONFIG))
                mod._USER_CONFIG = missing
        yield
    finally:
        for mod, attr, old in patched:
            setattr(mod, attr, old)
        os.environ.clear()
        os.environ.update(saved_env)


class _ParserCaptured(Exception):
    """Carries the fully-built parser out of `main()` before dispatch."""

    def __init__(self, parser: argparse.ArgumentParser) -> None:
        super().__init__("parser captured")
        self.parser = parser


@contextlib.contextmanager
def _parse_args_intercepted() -> Iterator[None]:
    """Make `ArgumentParser.parse_args` raise instead of parsing/exiting.

    `main()` builds the whole tree and only then calls `ap.parse_args()`, so
    raising there hands back a complete parser without executing a command and
    without `--help`'s `sys.exit`. Restored unconditionally.
    """
    original = argparse.ArgumentParser.parse_args

    def _raise(self: argparse.ArgumentParser, *a: Any, **k: Any) -> Any:
        raise _ParserCaptured(self)

    argparse.ArgumentParser.parse_args = _raise  # type: ignore[method-assign]
    try:
        yield
    finally:
        argparse.ArgumentParser.parse_args = original  # type: ignore[method-assign]


def _drive_main_for_parser(main_fn: Any) -> argparse.ArgumentParser:
    with _parse_args_intercepted():
        try:
            main_fn()
        except _ParserCaptured as captured:
            return captured.parser
        except SystemExit as exc:  # pragma: no cover - a broken composition root
            raise AssertionError(
                f"{main_fn!r} exited (code {exc.code}) before building its parser"
            ) from exc
    raise AssertionError(f"{main_fn!r} returned without ever calling parse_args()")


# ---------------------------------------------------------- normalization

#: Node fields that hold a *value* rather than rendered prose. The image
#: substitution is scoped by this set because the same string arrives from two
#: different places (see `_substitutions`).
VALUE_FIELDS = frozenset({"default", "const", "parser_defaults", "argument_default"})


def _first_attr(modules: tuple[str, ...], attr: str) -> tuple[Any, str | None]:
    for name in modules:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(mod, attr):
            return getattr(mod, attr), f"{name}.{attr}"
    return None, None


def load_pinned_config(prefer: str = "flat") -> tuple[dict[str, Any], str | None]:
    """The `herdd.yaml` mapping, read through the arm's own config module."""
    pair = ("vastconf", "vastlib.core.config")
    order = pair[::-1] if prefer == "vastlib" else pair
    fn, source = _first_attr(order, "load_herdd_config")
    if fn is None:  # pragma: no cover - vastconf is a flat sibling, always there
        return {}, None
    return dict(fn()), source


def _substitutions(prefer: str = "flat") -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    """`[(literal, placeholder, scope)]` + the record of where each came from.

    Scope is `"value"` (defaults only), `"text"` (help/usage prose only) or
    `"any"`. The split exists because `launch --image` shows the SAME image
    string from two different sources: its `default=` is
    `herdd.yaml:default_image` (H5, machine config) while its help text
    interpolates the source constant `_EXPECTED_DEFAULT_IMAGE` (H1/H4). Scoping
    keeps the record honest about which is which.

    Blind spot, stated plainly: `test_rehearse.py` pins those two values EQUAL,
    so at any healthy rev a port that read one where the flat code read the
    other renders identical bytes — invisible to this gate with or without
    normalization. That equality guard, not this file, is what covers it.

    Longest literal first: the repo root contains `$HOME`, so substituting
    `$HOME` first would leave `<<home>>/code/...` and the repo rule never fires.
    """
    sources: dict[str, str] = {}
    raw: list[tuple[str, str, str]] = []

    cfg, cfg_source = load_pinned_config(prefer)
    image = cfg.get("default_image")
    if isinstance(image, str) and image:
        raw.append((image, "<<cfg:default_image>>", "value"))
        sources["<<cfg:default_image>>"] = f"{cfg_source}()['default_image']"

    # H1: the constant is expected to land in vastlib/launch/spec.py when the
    # thinning ports it; until then only the flat copy exists.
    homes = ("herdd", "vastlib.launch.spec")
    order = homes[::-1] if prefer == "vastlib" else homes
    const, const_source = _first_attr(order, "_EXPECTED_DEFAULT_IMAGE")
    if isinstance(const, str) and const:
        raw.append((const, "<<const:_EXPECTED_DEFAULT_IMAGE>>", "text"))
        sources["<<const:_EXPECTED_DEFAULT_IMAGE>>"] = str(const_source)

    for literal, placeholder in (
        (REPO_ROOT, "<<repo>>"),
        (os.getcwd(), "<<cwd>>"),
        (os.path.expanduser("~"), "<<home>>"),
    ):
        raw.append((literal, placeholder, "any"))
        sources[placeholder] = "host path"

    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for literal, placeholder, scope in sorted(raw, key=lambda kv: -len(kv[0])):
        if not literal or literal == "/" or (literal, scope) in seen:
            continue
        seen.add((literal, scope))
        out.append((literal, placeholder, scope))
    return out, sources


def _trail_is_value(trail: str) -> bool:
    segments = {seg.split("[", 1)[0] for seg in trail.split(".")}
    return bool(segments & VALUE_FIELDS)


def normalize_tree(
    nodes: dict[str, dict[str, Any]], subs: list[tuple[str, str, str]]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Replace host-derived literals, recording every place one fired.

    The returned `applied` list is the committed record of which entries of the
    surface are environment-dependent (manifest H5) — sorted, so a newly
    environment-dependent flag lands as a fixture diff, never as a silent
    normalization.
    """
    applied: set[str] = set()

    def walk(value: Any, trail: str) -> Any:
        if isinstance(value, str):
            out = value
            is_value = _trail_is_value(trail)
            for literal, placeholder, scope in subs:
                if scope == "value" and not is_value:
                    continue
                if scope == "text" and is_value:
                    continue
                if literal in out:
                    out = out.replace(literal, placeholder)
                    applied.add(f"{trail} -> {placeholder}")
            return out
        if isinstance(value, list):
            return [walk(v, f"{trail}[{i}]") for i, v in enumerate(value)]
        if isinstance(value, dict):
            return {k: walk(v, f"{trail}.{k}" if trail else str(k)) for k, v in value.items()}
        return value

    normalized = {path: walk(node, path or "<root>") for path, node in nodes.items()}
    return normalized, sorted(applied)


def strip_callable_names(nodes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Collapse `{"__callable__": "cmd_train"}` to a presence marker.

    The cross-arm compare must not require `vastlib.cli.train.run` to still be
    named `cmd_train`; what it requires is that the dispatch default is present
    and callable, which is exactly what the marker preserves.
    """

    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {"__callable__"}:
                return {"__callable__": "<callable>"}
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    return {path: walk(node) for path, node in nodes.items()}


# ------------------------------------------------------------- tree build

def build_tree(
    parser: argparse.ArgumentParser,
    *,
    source: str,
    seam: str,
    prefer: str = "flat",
    normalize: bool = True,
) -> dict[str, Any]:
    """Walk + normalize into the canonical tree.

    `normalize=False` is used only by the config-dependence discovery, which
    must see raw values to tell a config-derived default from a literal one.
    """
    nodes = walk_parser(parser)
    subs, sources = _substitutions(prefer)
    applied: list[str] = []
    if normalize:
        nodes, applied = normalize_tree(nodes, subs)
    return {
        "schema": SCHEMA_VERSION,
        "source": source,
        "seam": seam,
        "rev": FIXTURE_REV,
        "columns": int(HELP_COLUMNS),
        "root_prog": nodes[""]["prog"],
        "command_count": len(nodes),
        "normalizations": {
            "placeholders": {
                "<<cfg:default_image>>": "herdd.yaml key `default_image`, in value fields (H5)",
                "<<const:_EXPECTED_DEFAULT_IMAGE>>":
                    "source constant, in help/usage prose (H1/H4)",
                "<<repo>>": "repository root",
                "<<cwd>>": "working directory at capture time",
                "<<home>>": "$HOME",
            },
            "sources": sources,
            "applied": applied,
        },
        "nodes": nodes,
    }


def dumps_tree(tree: dict[str, Any]) -> str:
    """The one canonical serialization. `sort_keys=False` is load-bearing."""
    return json.dumps(tree, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def capture_flat_tree(*, normalize: bool = True) -> dict[str, Any]:
    """Build `herdd.py`'s parser tree without dispatching a command."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    with _pinned_environment():
        herdd = importlib.import_module("herdd")
        parser = _drive_main_for_parser(herdd.main)
        return build_tree(
            parser,
            source="herdd.py",
            seam="main()+parse_args interception",
            prefer="flat",
            normalize=normalize,
        )


# ------------------------------------------------- config-dependence probe

#: Modules whose `load_herdd_config` must be swapped together: `main()` holds
#: a from-import binding, while `default_disk_gb` (and friends) call through the
#: defining module's globals. Patching one and not the other measures nothing.
CONFIG_READER_MODULES = ("vastconf", "herdd", "vastlib.core.config")


def _perturb_config_value(value: Any) -> Any:
    """Type-preserving perturbation: flip bools, shift numbers, suffix strings."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1000
    if isinstance(value, float):
        return value + 1000.0
    if isinstance(value, str):
        return value + "-PERTURBED"
    return value


@contextlib.contextmanager
def _perturbed_config() -> Iterator[None]:
    base, _ = load_pinned_config("flat")
    perturbed = {k: _perturb_config_value(v) for k, v in base.items()}
    patched: list[tuple[Any, Any]] = []
    try:
        for name in CONFIG_READER_MODULES:
            try:
                mod = importlib.import_module(name)
            except ImportError:
                continue
            if hasattr(mod, "load_herdd_config"):
                patched.append((mod, mod.load_herdd_config))
                mod.load_herdd_config = lambda: dict(perturbed)
        yield
    finally:
        for mod, original in patched:
            mod.load_herdd_config = original


def _value_entries(nodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Flatten every *value* field of the tree to `"path :: flag :: field"`."""
    out: dict[str, Any] = {}
    for path, node in nodes.items():
        label = path or "<root>"
        for key, value in node["parser_defaults"].items():
            out[f"{label} :: <parser default> :: {key}"] = value
        for action in node["actions"]:
            for field in ("default", "const"):
                out[f"{label} :: {_action_key(action)} :: {field}"] = action[field]
    return out


def discover_config_dependent_entries() -> list[str]:
    """Which help-tree values move when `herdd.yaml` moves — measured.

    Manifest H5 names one (`launch --image`); measuring rather than listing
    catches the others (today: `job supervise --handoff`, whose default is the
    `jobs_handoff_unsafe_enable` key). The list is committed into the fixture,
    so a newly config-derived default is a reviewable diff and an operator
    seeing one of these in a parity failure knows to check the config
    environment (PyYAML present? user override pinned away?) before suspecting
    the port.
    """
    baseline = _value_entries(capture_flat_tree(normalize=False)["nodes"])
    with _perturbed_config():
        moved = _value_entries(capture_flat_tree(normalize=False)["nodes"])
    return sorted(k for k, v in baseline.items() if moved.get(k) != v)


def flat_arm_is_alive() -> bool:
    """False once `herdd.py` is the post-6d thin launcher.

    After the thinning the module still has `main`, but it is
    `vastlib.cli.main:main` — intercepting it would capture the vastlib parser
    and compare it to itself. The parser plumbing that dies with the thinning
    is the honest tell.
    """
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    try:
        herdd = importlib.import_module("herdd")
    except ImportError:  # pragma: no cover
        return False
    return all(hasattr(herdd, attr) for attr in FLAT_PLUMBING_ATTRS)


def capture_vastlib_tree() -> tuple[dict[str, Any] | None, str]:
    """`(tree, note)` — tree is None while `vastlib.cli` has no buildable parser.

    Three not-yet states are folded into `None`, all with the reason attached:
    `vastlib.cli.main` missing, no parser seam on it, and a seam that raises
    (mid-wave, `main()` imports command modules that a sibling has not written
    yet — a `ModuleNotFoundError` there is "not built", not a surface delta).
    `VASTLIB_CLI_SURFACE_STRICT=1` turns all three into failures.
    """
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    try:
        mod = importlib.import_module("vastlib.cli.main")
    except Exception as exc:
        return None, f"vastlib.cli.main is not importable yet ({type(exc).__name__}: {exc})"
    try:
        return _capture_vastlib_tree_inner(mod)
    except Exception as exc:
        return None, f"vastlib.cli parser build raised {type(exc).__name__}: {exc}"


def _capture_vastlib_tree_inner(mod: Any) -> tuple[dict[str, Any] | None, str]:
    with _pinned_environment():
        for seam in VASTLIB_PARSER_SEAMS:
            fn = getattr(mod, seam, None)
            if callable(fn):
                return (
                    build_tree(
                        fn(), source="vastlib.cli.main", seam=f"{seam}()", prefer="vastlib"
                    ),
                    seam,
                )
        main_fn = getattr(mod, "main", None)
        if callable(main_fn):
            parser = _drive_main_for_parser(main_fn)
            return (
                build_tree(
                    parser,
                    source="vastlib.cli.main",
                    seam="main()+parse_args interception",
                    prefer="vastlib",
                ),
                "main()",
            )
    return None, "vastlib.cli.main exposes no parser seam (build_parser/main) yet"


# ----------------------------------------------------------------- diffing

def _diff_actions(path: str, want: dict[str, Any], got: dict[str, Any]) -> list[str]:
    out: list[str] = []
    label = path or "<root>"
    want_actions = {_action_key(a): a for a in want["actions"]}
    got_actions = {_action_key(a): a for a in got["actions"]}
    for key in want_actions:
        if key not in got_actions:
            out.append(f"{label}: MISSING flag {key}")
    for key in got_actions:
        if key not in want_actions:
            out.append(f"{label}: EXTRA flag {key}")
    for key, want_a in want_actions.items():
        got_a = got_actions.get(key)
        if got_a is None:
            continue
        for field in want_a:
            if field not in got_a:
                out.append(f"{label}: flag {key}: field {field!r} absent")
            elif want_a[field] != got_a[field]:
                out.append(
                    f"{label}: flag {key}: {field} "
                    f"{json.dumps(want_a[field], ensure_ascii=False)} != "
                    f"{json.dumps(got_a[field], ensure_ascii=False)}"
                )
    want_order = [_action_key(a) for a in want["actions"]]
    got_order = [_action_key(a) for a in got["actions"]]
    if want_order != got_order and sorted(want_order) == sorted(got_order):
        out.append(f"{label}: flag ORDER differs: {want_order} != {got_order}")
    return out


def _diff_node(path: str, want: dict[str, Any], got: dict[str, Any]) -> list[str]:
    out: list[str] = []
    label = path or "<root>"
    scalar_fields = (
        "prog", "aliases", "usage_attr", "description", "epilog_lines", "formatter_class",
        "prefix_chars", "add_help", "allow_abbrev", "conflict_handler", "argument_default",
        "parser_defaults", "argument_groups", "mutually_exclusive", "subcommands",
        "usage_lines",
    )
    for field in scalar_fields:
        if want.get(field) != got.get(field):
            out.append(
                f"{label}: {field} differs: "
                f"{json.dumps(want.get(field), ensure_ascii=False)[:400]} != "
                f"{json.dumps(got.get(field), ensure_ascii=False)[:400]}"
            )
    out.extend(_diff_actions(path, want, got))
    if want.get("help_lines") != got.get("help_lines"):
        delta = list(
            difflib.unified_diff(
                want.get("help_lines", []),
                got.get("help_lines", []),
                fromfile=f"fixture:{label}",
                tofile=f"vastlib:{label}",
                lineterm="",
                n=1,
            )
        )
        out.append(f"{label}: rendered --help differs:\n    " + "\n    ".join(delta[:40]))
    return out


def diff_trees(want: dict[str, Any], got: dict[str, Any]) -> dict[str, list[str]]:
    """Per-command findings, `{}` when the two surfaces are identical.

    Keyed by command path so a half-built `cli/` reports "these 41 commands do
    not exist yet" plus the exact flag deltas of the ones that do, instead of
    one wall of JSON.
    """
    findings: dict[str, list[str]] = {}
    want_nodes: dict[str, Any] = want["nodes"]
    got_nodes: dict[str, Any] = got["nodes"]
    for path in want_nodes:
        if path not in got_nodes:
            findings[path or "<root>"] = ["MISSING: command not present in vastlib.cli"]
    for path in got_nodes:
        if path not in want_nodes:
            findings[path or "<root>"] = ["EXTRA: command not present in the flat surface"]
    for path, want_node in want_nodes.items():
        got_node = got_nodes.get(path)
        if got_node is None:
            continue
        deltas = _diff_node(path, want_node, got_node)
        if deltas:
            findings[path or "<root>"] = deltas
    return findings


def format_findings(
    findings: dict[str, list[str]],
    limit: int = 30,
    config_dependent: list[str] | None = None,
) -> str:
    lines = [f"CLI-surface parity: {len(findings)} command(s) differ"]
    for path, deltas in list(findings.items())[:limit]:
        lines.append(f"  [{path}]")
        lines.extend(f"    - {d}" for d in deltas[:12])
        if len(deltas) > 12:
            lines.append(f"    - … {len(deltas) - 12} more delta(s)")
    if len(findings) > limit:
        lines.append(f"  … {len(findings) - limit} more command(s)")
    # Only worth saying when the command is present and a *value* moved; on a
    # missing command the config environment is irrelevant noise.
    value_deltas = {
        path
        for path, deltas in findings.items()
        if not deltas[0].startswith(("MISSING", "EXTRA"))
        and any(": default " in d or ": const " in d or "parser_defaults" in d for d in deltas)
    }
    hits = sorted({e for e in (config_dependent or []) if e.split(" :: ", 1)[0] in value_deltas})
    if hits:
        lines.append(
            "  note: the following differing commands carry herdd.yaml-derived defaults — "
            "check the config environment (PyYAML installed? user override pinned away?) "
            "before suspecting the port:"
        )
        lines.extend(f"    * {h}" for h in hits)
    return "\n".join(lines)


# ------------------------------------------------------------------- fixture

def load_fixture() -> dict[str, Any]:
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def write_fixture() -> str:
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    tree = capture_flat_tree()
    nodes = tree.pop("nodes")
    tree["capture_env"] = {
        # PyYAML's presence changes how `herdd.yaml` parses (the stdlib
        # fallback yields strings where PyYAML yields bools/floats), which moves
        # every config-derived default below. Recorded, not normalized.
        "pyyaml": importlib.util.find_spec("yaml") is not None,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
    tree["config_dependent"] = discover_config_dependent_entries()
    tree["nodes"] = nodes
    with open(FIXTURE_PATH, "w", encoding="utf-8") as fh:
        fh.write(dumps_tree(tree))
    return FIXTURE_PATH


# ------------------------------------------------------- scoped amendment

def _restore_callable_names(live: Any, frozen: Any) -> Any:
    """`live`, but with every dispatch callable NAMED as the fixture named it.

    The fixture records the flat surface, where the handler was `cmd_dash_cache`;
    the vastlib arm calls it `run`. The compare strips both to a presence marker
    (`strip_callable_names`), so that rename is not a surface change — and an
    amendment must not smuggle it in as one.
    """
    if isinstance(live, dict) and isinstance(frozen, dict):
        if set(live) == {"__callable__"} and set(frozen) == {"__callable__"}:
            return frozen
        return {k: _restore_callable_names(v, frozen.get(k)) for k, v in live.items()}
    if isinstance(live, list) and isinstance(frozen, list) and len(live) == len(frozen):
        return [_restore_callable_names(a, b) for a, b in zip(live, frozen)]
    return live


def amend_fixture(paths: Sequence[str]) -> dict[str, list[str]]:
    """Re-freeze ONLY the named commands, and only if nothing else moved.

    This is the deliberate-edit path, and it is not `--write` with a smaller
    blast radius: `--write` re-captures all 70 commands from a flat parser that
    no longer exists, which is why it refuses post-thinning. This one splices
    the named nodes out of the LIVE `vastlib.cli` tree into the committed
    fixture and leaves every other byte alone.

    Intent is the whole mechanism, so it is checked in both directions:

      * anything differing OUTSIDE `paths` aborts — that is the unintended
        surface drift the freeze exists to catch, and an amendment must never
        sweep it up silently;
      * a named path that does NOT differ aborts — a stale intent means the
        operator is describing a change that is not there, and quietly
        succeeding would teach them the flag is a no-op;
      * an added or removed COMMAND aborts — the freeze is per-command, so a
        structural change is an owner decision, not a help-string edit.

    Raises `SystemExit` with the reason on every refusal. Returns the findings
    it applied.
    """
    tree, note = capture_vastlib_tree()
    if tree is None:
        raise SystemExit(f"refusing to amend: {note}")
    fixture = load_fixture()
    findings = diff_trees(
        {**fixture, "nodes": strip_callable_names(fixture["nodes"])},
        {**tree, "nodes": strip_callable_names(tree["nodes"])},
    )
    named = list(dict.fromkeys(paths))
    unknown = [p for p in named if p not in fixture["nodes"]]
    if unknown:
        raise SystemExit(f"refusing to amend: not a command in the fixture: {unknown}")
    stray = {p: d for p, d in findings.items() if p not in named}
    if stray:
        raise SystemExit(
            "refusing to amend: the surface also moved in command(s) you did not "
            "name, which is exactly what this freeze is for. Name them too if the "
            "change is intended, or fix the regression:\n"
            + format_findings(stray, config_dependent=fixture.get("config_dependent"))
        )
    quiet = [p for p in named if p not in findings]
    if quiet:
        raise SystemExit(
            f"refusing to amend: no surface change in {quiet} — the fixture already "
            "matches. Nothing to re-freeze."
        )
    structural = [p for p, d in findings.items() if d and d[0].startswith(("MISSING", "EXTRA"))]
    if structural:
        raise SystemExit(
            f"refusing to amend: {structural} add or remove a COMMAND, not a flag. "
            "The fixture is frozen per command; a structural change is an owner "
            "decision, not an amendment."
        )
    for path in named:
        fixture["nodes"][path] = _restore_callable_names(
            tree["nodes"][path], fixture["nodes"][path])
    with open(FIXTURE_PATH, "w", encoding="utf-8") as fh:
        fh.write(dumps_tree(fixture))
    return findings


def _ancestors(path: str) -> list[str]:
    """Every enclosing command node of `path`, root ("") last. `fleet hosts` ->
    ["fleet", ""]."""
    parts = path.split()
    return [" ".join(parts[:i]) for i in range(len(parts) - 1, -1, -1)]


def add_command_to_fixture(paths: Sequence[str]) -> dict[str, list[str]]:
    """Re-freeze NEW command nodes and the ancestors they mechanically move.

    The structural sibling of `amend_fixture`, and it keeps the same contract:
    you name your intent, and anything outside it aborts. The closure is the
    named paths plus their ancestor chain, because a parent's subcommand list,
    usage line and rendered help all change the moment a child exists — that
    drift is implied by the addition, not independent evidence of anything.

    Refuses a REMOVAL in any case. Growing a surface is additive and safe;
    dropping a command breaks callers that already exist, so it keeps having no
    tool and stays an owner decision.

    Also bumps `counts.command_nodes` in the port manifest, whose inventory
    cross-check would otherwise fail immediately afterwards with no hint that
    the two files are coupled.
    """
    tree, note = capture_vastlib_tree()
    if tree is None:
        raise SystemExit(f"refusing to add: {note}")
    fixture = load_fixture()
    findings = diff_trees(
        {**fixture, "nodes": strip_callable_names(fixture["nodes"])},
        {**tree, "nodes": strip_callable_names(tree["nodes"])},
    )
    named = list(dict.fromkeys(paths))
    already = [p for p in named if p in fixture["nodes"]]
    if already:
        raise SystemExit(
            f"refusing to add: {already} already in the fixture. A command that "
            f"exists moves with --amend; --add-command is for new ones.")
    missing = [p for p in named if p not in tree["nodes"]]
    if missing:
        raise SystemExit(
            f"refusing to add: {missing} is not in the live vastlib parser "
            f"either. Register the command first, then re-freeze.")
    removed = [p for p, d in findings.items()
               if d and d[0].startswith("MISSING")]
    if removed:
        raise SystemExit(
            f"refusing to add: {removed} would be REMOVED from the surface. "
            f"Removal breaks callers and has no tool — it is an owner decision.")
    closure = set(named)
    for p in named:
        closure.update(_ancestors(p))
    # findings are keyed with the root's DISPLAY spelling ("<root>"), nodes
    # with "". A nested add never moves the root, so the mismatch was
    # unreachable until the first TOP-LEVEL add; compare on display labels.
    closure_labels = {p or "<root>" for p in closure}
    stray = {p: d for p, d in findings.items() if p not in closure_labels}
    if stray:
        raise SystemExit(
            "refusing to add: the surface also moved outside the new commands "
            "and their parents, which is exactly what this freeze is for. Amend "
            "those separately if intended, or fix the regression:\n"
            + format_findings(stray, config_dependent=fixture.get("config_dependent"))
        )
    for path in sorted(closure, key=lambda p: (p.count(" "), p)):
        if path not in tree["nodes"]:
            continue
        fixture["nodes"][path] = _restore_callable_names(
            tree["nodes"][path], fixture["nodes"].get(path))
    fixture["command_count"] = len(fixture["nodes"])
    with open(FIXTURE_PATH, "w", encoding="utf-8") as fh:
        fh.write(dumps_tree(fixture))
    # a TOP-LEVEL add must also enter the manifest's `commands` inventory (the
    # inventory cross-check compares name-for-name, in order) — nested adds
    # never touch it, which is why `fleet hosts` did not need this
    top = [s["name"] for s in fixture["nodes"][""]["subcommands"]]
    new_top = [(top.index(p), p) for p in named if " " not in p and p in top]
    _bump_manifest_nodes(len(named), new_top_level=new_top)
    return findings


def _bump_manifest_nodes(added: int,
                         new_top_level: Sequence[tuple[int, str]] = ()) -> None:
    """Keep `.port_manifests/cli-surface.json`'s node count in step.

    The manifest is a PORT RECORD of rev 7a177e2a and its other numbers stay
    frozen at what the port measured. `command_nodes` is the one field a live
    test compares against the fixture, so it alone tracks the surface; the note
    written beside it says so, and names what has been added since, rather than
    letting a bumped number read as a re-measurement of the port.

    `new_top_level` [(fixture_index, name), ...]: top-level additions also
    enter the `commands` inventory, as minimal stubs — they have no
    rev-7a177e2a port record and must not fake one.
    """
    if not os.path.exists(MANIFEST_PATH):  # pragma: no cover - manifest is committed
        return
    with open(MANIFEST_PATH, encoding="utf-8") as fh:
        manifest = json.load(fh)
    counts = manifest.setdefault("counts", {})
    counts["command_nodes"] = int(counts.get("command_nodes", 0)) + int(added)
    cmds = manifest.get("commands")
    if isinstance(cmds, list):
        for idx, name in sorted(new_top_level):
            cmds.insert(idx, {
                "name": name, "aliases": [],
                "post_port_addition": "added by --add-command; not part of "
                                      "the rev-7a177e2a port record"})
    notes = manifest.setdefault("notes", [])
    if isinstance(notes, list):
        notes.append(
            f"counts.command_nodes bumped +{added} by --add-command; every other "
            f"count remains the rev-7a177e2a port measurement. This field alone "
            f"tracks the live surface (test_fixture_matches_manifest_command_"
            f"inventory compares it against the fixture).")
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")


# --------------------------------------------------------------------- tests

@pytest.fixture(scope="module")
def flat_tree() -> dict[str, Any]:
    if not flat_arm_is_alive():
        pytest.skip(
            "herdd.py is the thin launcher (plan §8 step 6d) — the flat arm is gone "
            "and the committed fixture is now the frozen reference"
        )
    return capture_flat_tree()


def test_walker_is_deterministic(flat_tree: dict[str, Any]) -> None:
    """Two independent captures must serialize to identical bytes.

    This is the property the whole harness rests on: a walker with any
    set-iteration, address-repr or terminal-width dependence would make every
    later diff untrustworthy noise.
    """
    again = capture_flat_tree()
    first, second = dumps_tree(flat_tree), dumps_tree(again)
    if first != second:
        delta = list(
            difflib.unified_diff(
                first.splitlines(), second.splitlines(),
                fromfile="capture-1", tofile="capture-2", lineterm="", n=1,
            )
        )
        raise AssertionError("walker is not deterministic:\n" + "\n".join(delta[:60]))


def test_capture_leaves_no_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both capture-time mutations must be undone: env and `parse_args`.

    `main()` calls `load_env()`, which `setdefault`s the developer's `.env`
    into `os.environ` — leaking that would silently reconfigure every test
    that runs after this file. The leak is simulated inside the block rather
    than relied upon, so the check does not depend on which capture ran first.
    """
    monkeypatch.setenv("HERDD_CONFIG", "/nonexistent/override.yaml")
    before = dict(os.environ)
    with _pinned_environment():
        assert "HERDD_CONFIG" not in os.environ, "user config override not pinned away"
        assert os.environ["COLUMNS"] == HELP_COLUMNS
        os.environ["A_CAPTURE_LEAKED_THIS"] = "1"  # stands in for load_env()
    assert dict(os.environ) == before, "capture leaked environment into the test process"

    parser = argparse.ArgumentParser(prog="probe")
    parser.add_argument("--x", default=1)
    assert parser.parse_args(["--x", "2"]).x == "2", "parse_args was not restored"


def test_fixture_exists_and_is_canonical() -> None:
    """The committed bytes must be exactly what `--write` produces.

    If the file is hand-edited into a non-canonical shape, every later
    regeneration shows a spurious whole-file diff and the record stops being
    reviewable.
    """
    assert os.path.exists(FIXTURE_PATH), (
        f"missing CLI-surface fixture {FIXTURE_PATH} — regenerate with "
        f"`cd tools/vast && python3 {os.path.basename(__file__)} --write`"
    )
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        text = fh.read()
    assert dumps_tree(json.loads(text)) == text, (
        "fixture is not in canonical serialization; regenerate it with --write"
    )


def test_fixture_matches_live_flat_parser(flat_tree: dict[str, Any]) -> None:
    """While both arms are alive, the fixture cannot rot.

    A red here means `herdd.py`'s surface moved (a peer commit added a flag)
    — regenerate the fixture in the same commit that ports the change into
    `vastlib.cli`, never on its own.
    """
    fixture = load_fixture()
    findings = diff_trees(fixture, flat_tree)
    assert not findings, (
        "committed fixture no longer matches the live herdd.py help tree:\n"
        + format_findings(findings, config_dependent=fixture.get("config_dependent"))
    )


def test_fixture_matches_manifest_command_inventory() -> None:
    """Cross-check against `.port_manifests/cli-surface.json`.

    An independent inventory catches a capture that silently truncated — e.g.
    an interception that fired on an inner parser and froze a partial tree as
    the reference.
    """
    if not os.path.exists(MANIFEST_PATH):  # pragma: no cover - manifest is committed
        pytest.skip(f"manifest {MANIFEST_PATH} absent")
    with open(MANIFEST_PATH, encoding="utf-8") as fh:
        manifest = json.load(fh)
    fixture = load_fixture()
    top_level = [s["name"] for s in fixture["nodes"][""]["subcommands"]]
    expected = [c["name"] for c in manifest["commands"]]
    assert top_level == expected, (
        f"top-level command order/inventory differs from the manifest:\n"
        f"  fixture:  {top_level}\n  manifest: {expected}"
    )
    assert fixture["command_count"] == manifest["counts"]["command_nodes"] + 1, (
        f"node count {fixture['command_count']} != manifest command_nodes + root "
        f"({manifest['counts']['command_nodes']} + 1)"
    )
    # H11: the only two aliases in the whole surface.
    aliases = {
        s["name"]: s["aliases"] for s in fixture["nodes"][""]["subcommands"] if s["aliases"]
    }
    assert aliases == {"stop": ["park"], "start": ["resume"]}, aliases


def test_fixture_normalizes_the_yaml_derived_image_default(flat_tree: dict[str, Any]) -> None:
    """H5: the yaml-derived `launch --image` default is a placeholder.

    If this stops firing, either the capture saw no `herdd.yaml` or `--image`
    stopped defaulting from config — both make the fixture host-specific, which
    is exactly the failure this normalization exists to prevent. The help
    string beside it keeps the source constant literal (`<<const:…>>`), so the
    record still distinguishes the two sources (H1/H4).
    """
    fixture = load_fixture()
    applied = fixture["normalizations"]["applied"]
    assert any("<<cfg:default_image>>" in entry for entry in applied), applied
    image = [
        a for a in fixture["nodes"]["launch"]["actions"] if "--image" in a["option_strings"]
    ]
    assert image and image[0]["default"] == "<<cfg:default_image>>", image
    assert "<<const:_EXPECTED_DEFAULT_IMAGE>>" in (image[0]["help"] or ""), image[0]["help"]
    assert not any(
        ".help " in entry or "help_lines" in entry
        for entry in applied
        if "<<cfg:default_image>>" in entry
    ), "the config placeholder leaked into prose — value/text scoping is broken"


def test_config_dependent_record_is_live(flat_tree: dict[str, Any]) -> None:
    """The recorded set of yaml-derived defaults is re-measured, not asserted.

    A newly config-derived default (or one that quietly stopped reading the
    config) shows up here as a diff against the committed record. Measured by
    perturbing `load_herdd_config` and re-walking, so it cannot drift out of
    date the way a hand-written list would.
    """
    fixture = load_fixture()
    recorded = fixture.get("config_dependent")
    assert recorded is not None, "fixture predates the config-dependence record; regenerate"
    live = discover_config_dependent_entries()
    assert live == recorded, (
        "herdd.yaml-derived help-tree entries changed:\n"
        f"  gone: {[e for e in recorded if e not in live]}\n"
        f"  new:  {[e for e in live if e not in recorded]}"
    )
    assert any("--image" in e for e in recorded), recorded  # H5's named case


def test_fixture_preserves_dispatch_and_groups() -> None:
    """The two structural contracts `--help` cannot show.

    (H6) the three `--strict-ceiling|--handoff|--no-handoff` exclusive groups,
    and the `a.func(a)` dispatch default on every leaf command.
    """
    fixture = load_fixture()
    nodes = fixture["nodes"]
    for path in ("supervise", "train", "job supervise"):
        groups = nodes[path]["mutually_exclusive"]
        assert groups, f"{path}: mutually-exclusive group lost"
        members = [m for g in groups for m in g["members"]]
        assert set(members) == {"--strict-ceiling", "--handoff", "--no-handoff"}, members
    dispatch_attrs = ("func", "jobfunc", "fleetfunc", "wffunc", "notifyfunc")
    for path, node in nodes.items():
        if path == "" or node["subcommands"]:
            continue
        defaults = node["parser_defaults"]
        hit = [k for k in dispatch_attrs if isinstance(defaults.get(k), dict)]
        assert hit, f"{path}: no dispatch callable among {dispatch_attrs}: {list(defaults)}"


def test_local_flag_loop_is_present() -> None:
    """H6's second half: the post-hoc `--local` loop over job subcommands.

    `main()` builds seven job subparsers and only afterwards walks
    `_JOB_LOCAL_SUBCOMMANDS` adding `--local` to each. A port that re-literals
    the list, or adds the flag to the wrong seven, shows up right here.
    """
    fixture = load_fixture()
    with_local = sorted(
        path.split(" ", 1)[1]
        for path, node in fixture["nodes"].items()
        if path.startswith("job ")
        and any("--local" in a["option_strings"] for a in node["actions"])
    )
    try:
        from vastlib.jobs import runlocal
    except ImportError:  # pragma: no cover
        pytest.skip("vastlib.jobs.runlocal not importable")
    expected = sorted(runlocal._JOB_LOCAL_SUBCOMMANDS)
    assert with_local == expected, (
        f"`--local` is on {with_local} but _JOB_LOCAL_SUBCOMMANDS is {expected}"
    )


def test_vastlib_cli_surface_matches_fixture() -> None:
    """THE CUTOVER GATE (plan §8 step 6).

    Skips while `vastlib.cli.main` has no parser seam — `cli/` modules land
    incrementally and a hard red on every intermediate commit would train the
    branch to ignore this file. Set `VASTLIB_CLI_SURFACE_STRICT=1` (and the
    step-6 completion commit does) to turn the skip into a failure.
    """
    strict = os.environ.get("VASTLIB_CLI_SURFACE_STRICT") == "1"
    tree, note = capture_vastlib_tree()
    if tree is None:
        if strict:
            raise AssertionError(f"VASTLIB_CLI_SURFACE_STRICT=1 but {note}")
        pytest.skip(f"vastlib.cli parser not built yet: {note}")
    fixture = load_fixture()
    # Callable *names* legitimately change (`cmd_train` -> `vastlib.cli.train.run`);
    # the dispatch shape does not, so both sides are stripped to a presence marker.
    fixture_cmp = {**fixture, "nodes": strip_callable_names(fixture["nodes"])}
    tree_cmp = {**tree, "nodes": strip_callable_names(tree["nodes"])}
    findings = diff_trees(fixture_cmp, tree_cmp)
    assert not findings, (
        f"vastlib.cli help tree differs from the frozen flat surface (seam: {note})\n"
        + format_findings(findings, config_dependent=fixture.get("config_dependent"))
    )


def test_partial_cli_reports_per_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """A half-built `cli/` must name what is missing, command by command.

    Driven through the real `capture_vastlib_tree` seam selection (a stub
    module standing in for `vastlib.cli.main`) so the reporting path is
    exercised now, not first discovered at cutover.
    """
    import types

    stub = types.ModuleType("vastlib.cli.main")

    def build_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(prog="herdd")
        sub = p.add_subparsers(dest="cmd", required=True)
        sub.add_parser("whoami", help="show account credit/balance")
        return p

    stub.build_parser = build_parser  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vastlib.cli.main", stub)

    tree, seam = capture_vastlib_tree()
    assert tree is not None and seam == "build_parser", seam
    findings = diff_trees(load_fixture(), tree)
    missing = [p for p, d in findings.items() if d and d[0].startswith("MISSING")]
    assert "guard" in missing and "job submit" in missing, sorted(missing)[:10]
    assert "whoami" in findings, "the one present command must still be flag-diffed"
    assert len(findings) > 50, len(findings)


def test_strict_mode_turns_a_missing_arm_into_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The skip above must be an opt-out, not an escape hatch.

    A gate that skips forever is a gate nobody notices is gone; the step-6
    completion commit runs the suite with `VASTLIB_CLI_SURFACE_STRICT=1`, and
    this proves that switch actually bites while `cli/` is incomplete.
    """
    tree, _note = capture_vastlib_tree()
    if tree is not None:
        pytest.skip("vastlib.cli builds a parser; strict mode has nothing to refuse")
    monkeypatch.setenv("VASTLIB_CLI_SURFACE_STRICT", "1")
    with pytest.raises(AssertionError, match="STRICT"):
        test_vastlib_cli_surface_matches_fixture()


# ---------------------------------------------- the scoped amendment path
#
# `--amend` is the one writer that survives the thinning, so its REFUSALS are
# the freeze. Each is unit-tested on a synthetic pair of trees: a real capture
# would make these tests pass or fail on whatever the live surface happens to
# be, which is the opposite of what a gate test should depend on.

def _amend_fixture_double(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    live_help: str,
    also_move: str | None = None,
) -> str:
    """Freeze a two-command tree to disk, then offer a mutated live capture."""
    parser = argparse.ArgumentParser(prog="t")
    sub = parser.add_subparsers(dest="cmd")
    a = sub.add_parser("alpha", help="alpha")
    a.add_argument("--gpus", help="frozen help")
    a.set_defaults(func=lambda _n: None)
    b = sub.add_parser("beta", help="beta")
    b.add_argument("--other", help="frozen other")
    b.set_defaults(func=lambda _n: None)
    frozen = {"schema": SCHEMA_VERSION, "nodes": walk_parser(parser)}

    live = json.loads(json.dumps(frozen))
    for action in live["nodes"]["alpha"]["actions"]:
        if action["dest"] == "gpus":
            action["help"] = live_help
    # The vastlib arm renames handlers; an amendment must not record that.
    live["nodes"]["alpha"]["parser_defaults"]["func"] = {"__callable__": "run"}
    if also_move is not None:
        for action in live["nodes"]["beta"]["actions"]:
            if action["dest"] == "other":
                action["help"] = also_move

    path = os.path.join(str(tmp_path), "fixture.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dumps_tree(frozen))
    monkeypatch.setattr(sys.modules[__name__], "FIXTURE_PATH", path)
    monkeypatch.setattr(sys.modules[__name__], "capture_vastlib_tree", lambda: (live, "test"))
    return path


def test_amend_rewrites_only_the_named_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    path = _amend_fixture_double(monkeypatch, tmp_path, live_help="new help")
    before = json.loads(open(path, encoding="utf-8").read())
    findings = amend_fixture(["alpha"])
    after = json.loads(open(path, encoding="utf-8").read())
    assert list(findings) == ["alpha"]
    assert after["nodes"]["beta"] == before["nodes"]["beta"], "an unnamed command moved"
    helps = {a["dest"]: a["help"] for a in after["nodes"]["alpha"]["actions"]}
    assert helps["gpus"] == "new help"
    # The handler rename is normalized away by the compare, so it must not be
    # smuggled into the record by the amendment either.
    assert after["nodes"]["alpha"]["parser_defaults"]["func"] == {"__callable__": "<lambda>"}
    assert dumps_tree(after) == open(path, encoding="utf-8").read(), "not canonical"


def test_amend_refuses_when_an_unnamed_command_also_moved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    """The detector, in one test: a surprise rides in and the write aborts."""
    path = _amend_fixture_double(
        monkeypatch, tmp_path, live_help="new help", also_move="SURPRISE")
    before = open(path, encoding="utf-8").read()
    with pytest.raises(SystemExit, match="you did not name"):
        amend_fixture(["alpha"])
    assert open(path, encoding="utf-8").read() == before, "aborted amend still wrote"


def test_amend_refuses_a_named_command_that_did_not_move(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    """A stale intent is a refusal, not a silent no-op."""
    _amend_fixture_double(monkeypatch, tmp_path, live_help="new help")
    with pytest.raises(SystemExit, match="no surface change"):
        amend_fixture(["alpha", "beta"])


def test_amend_refuses_a_command_absent_from_the_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    _amend_fixture_double(monkeypatch, tmp_path, live_help="new help")
    with pytest.raises(SystemExit, match="not a command in the fixture"):
        amend_fixture(["gamma"])


def test_add_command_accepts_a_top_level_addition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    """Regression: `diff_trees` keys the root node "<root>" while the ancestor
    closure spells it "" — a NESTED add never moves the root, so the mismatch
    stayed unreachable until the first top-level add (`box`, 2026-08-26),
    which was refused as a stray root movement."""
    def build(with_gamma: bool) -> dict[str, Any]:
        parser = argparse.ArgumentParser(prog="t")
        sub = parser.add_subparsers(dest="cmd")
        names = ("alpha", "beta", "gamma") if with_gamma else ("alpha", "beta")
        for name in names:
            p = sub.add_parser(name, help=name)
            p.set_defaults(func=lambda _n: None)
        return {"schema": SCHEMA_VERSION, "nodes": walk_parser(parser)}

    frozen, live = build(False), build(True)
    path = os.path.join(str(tmp_path), "fixture.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dumps_tree(frozen))
    monkeypatch.setattr(sys.modules[__name__], "FIXTURE_PATH", path)
    monkeypatch.setattr(sys.modules[__name__], "MANIFEST_PATH",
                        os.path.join(str(tmp_path), "absent.json"))
    monkeypatch.setattr(sys.modules[__name__], "capture_vastlib_tree",
                        lambda: (live, "test"))
    findings = add_command_to_fixture(["gamma"])
    after = json.loads(open(path, encoding="utf-8").read())
    assert "gamma" in after["nodes"]
    assert "<root>" in findings, "the root moves WITH a top-level addition"


def test_diff_report_names_the_exact_flag() -> None:
    """The diff must be actionable, not "trees differ".

    Unit-tested on a synthetic mutation rather than on the real tree, so it
    keeps working after the flat arm is deleted.
    """
    parser = argparse.ArgumentParser(prog="t")
    sub = parser.add_subparsers(dest="cmd")
    child = sub.add_parser("go", help="go")
    child.add_argument("--cuda", type=float, default=13.0, help="min cuda")
    want = build_tree(parser, source="synthetic", seam="direct")

    mutant = argparse.ArgumentParser(prog="t")
    msub = mutant.add_subparsers(dest="cmd")
    mchild = msub.add_parser("go", help="go")
    mchild.add_argument("--cuda", type=float, default=12.0, help="min cuda")
    got = build_tree(mutant, source="synthetic", seam="direct")

    findings = diff_trees(want, got)
    assert set(findings) == {"go"}, findings
    report = format_findings(findings)
    assert "--cuda" in report and "13.0" in report and "12.0" in report, report

    # …and a dropped command is named as missing, not swallowed.
    bare = argparse.ArgumentParser(prog="t")
    bare.add_subparsers(dest="cmd")
    missing = diff_trees(want, build_tree(bare, source="synthetic", seam="direct"))
    assert "go" in missing and "MISSING" in missing["go"][0], missing


def test_exclusive_group_loss_is_detected() -> None:
    """A regression that `--help` renders identically must still be caught.

    Porting `add_mutually_exclusive_group()` flags as three plain flags is
    invisible in the rendered page (H6). If this ever passes-through, the
    parity gate is decorative for the supervise/train money paths.
    """
    def build(exclusive: bool) -> dict[str, Any]:
        p = argparse.ArgumentParser(prog="t")
        target: Any = p.add_mutually_exclusive_group() if exclusive else p
        target.add_argument("--handoff", action="store_true", help="h")
        target.add_argument("--no-handoff", action="store_true", help="n")
        return build_tree(p, source="synthetic", seam="direct")

    with_group, without = build(True), build(False)
    findings = diff_trees(with_group, without)
    assert "<root>" in findings, findings
    assert any("mutually_exclusive" in d for d in findings["<root>"]), findings["<root>"]


if __name__ == "__main__":  # pragma: no cover - regeneration entry point
    if "--write" in sys.argv:
        if not flat_arm_is_alive():
            raise SystemExit(
                "refusing to regenerate: herdd.py no longer builds the parser "
                "(post-thinning the fixture is the frozen reference)"
            )
        print(f"wrote {write_fixture()}")
    elif "--amend" in sys.argv:
        _named = [a for a in sys.argv[sys.argv.index("--amend") + 1:] if not a.startswith("-")]
        if not _named:
            raise SystemExit(
                f"usage: python3 {os.path.basename(__file__)} --amend <command> [<command>...]\n"
                "Name every command whose surface you intend to move. It refuses if "
                "anything else moved, or if a named command did not."
            )
        _applied = amend_fixture(_named)
        print(f"amended {FIXTURE_PATH}\n" + format_findings(_applied))
    elif "--add-command" in sys.argv:
        _named = [a for a in sys.argv[sys.argv.index("--add-command") + 1:]
                  if not a.startswith("-")]
        if not _named:
            raise SystemExit(
                f"usage: python3 {os.path.basename(__file__)} --add-command "
                f"'<command path>' [...]\n"
                "Quote a nested path as one argument (\"fleet hosts\"). Parents "
                "are re-frozen with it; anything else moving aborts."
            )
        _applied = add_command_to_fixture(_named)
        print(f"added {_named} to {FIXTURE_PATH}\n" + format_findings(_applied))
    else:
        raise SystemExit(
            f"usage: python3 {os.path.basename(__file__)} --write\n"
            f"       python3 {os.path.basename(__file__)} --amend <command> [<command>...]\n"
            f"       python3 {os.path.basename(__file__)} --add-command '<command path>' [...]"
        )
