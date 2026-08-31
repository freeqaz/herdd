"""vastlib.cli.main — the composition root: build the parser tree, dispatch, done.

Why this module exists
----------------------
`herdd.main()` was 1,137 lines: an argparse prologue, then 29 hand-inlined
parser blocks (473 flags, four nested groups), then two lines of dispatch. The
blocks were the file's single largest cluster and its least reviewable one —
adding a flag meant editing the same function every other command lives in, and
nothing but a careful read could tell you whether a subparser had been built
twice or a `set_defaults(func=…)` forgotten.

What replaces it is a REGISTRY LOOP over one module per command (plan §5). The
loop is the whole of the composition: for each name in `_REGISTRY`, import
`vastlib.cli.<name>` and call its `add_parser(sub, add_cmd)`. Everything that
used to be a parser block is now that module's own business, including its
`set_defaults(func=…)`.

The three things that must not move
-----------------------------------
1. **Order is printed output.** argparse lists subcommands in insertion order,
   so `_REGISTRY` is ordered exactly as `main()`'s blocks were and the tuple is
   the frozen spelling of that order. `test_vastlib_cli_main.py` compares it
   against the flat surface name-for-name.
2. **Dispatch is unchanged.** `a = <parser>.parse_args(); a.func(a)` (plan §5:
   "the `a.func(a)` shape preserved at the seam"). The only edit to the flat
   shape is that the tree now arrives from `build_parser()` instead of being
   composed inline — a seam the CLI-surface gate asks for by name, and the one
   thing `main()` does that a parser walker must not have to run. Every
   command module sets `func` itself; this module never does. Nothing here
   inspects `a.cmd`, and no `else: parser.error(...)` arm was added — argparse's
   `required=True` on the subparsers is what rejects a bare `herdd`, exactly
   as before.
3. **`load_env()` first.** The flat prologue loaded `.env` before anything
   else, so a parser default that reads the environment sees the same values.

What is deliberately NOT here
-----------------------------
* **Flags.** Not one. If a flag string appears in this module, the port went
  wrong. The top-level parser's own `prog` / `description` / `epilog` are the
  only argparse arguments this file may hold.
* **The `herdd.yaml` `--image` default.** The flat prologue read
  `load_herdd_config()["default_image"]` and closed over it for the two
  parsers that bake it into a flag default (`launch`, `supervise`). It now
  lives in `cli/launch.default_image()`, next to the help text that quotes it:
  the registry keeps one two-argument call shape for all 29 entries, and the
  value cannot drift from the sentence describing it. Same read, same source,
  same bytes (cli-surface.json hazard H5 — this default is environment
  dependent, so the CLI-surface diff pins `herdd.yaml` on both arms).
* **The `--local` post-hoc loop.** `--local` is hung on seven of the fourteen
  `job` subparsers by one loop over `jobs.runlocal._JOB_LOCAL_SUBCOMMANDS`,
  which lives inside `cli/job/__init__.py` where the flat code has it — between
  the `cancel` and `run-local` blocks. Keeping it there keeps the ordering
  argument local: `--local` must be the LAST flag added to each of those seven
  parsers, which is true because nothing touches them after that point.
* **Anything conditional.** No command is registered behind a feature flag, an
  env read, or a `try: import`. A missing command module is an ImportError at
  startup and is meant to be: a silently-dropped subcommand is exactly the
  failure this refactor must not be able to produce.

Provenance: moved from `tools/vast/herdd.py` `main()` (rev 7a177e2a),
plan §8 step 6, 2026-08-16, behavior-preserving. Step 6 is ADD-ONLY at this
commit: `herdd.py` still owns the live `main()` and the CLI-surface byte diff
is what proves the two parser trees identical.
"""

from __future__ import annotations

import argparse
import importlib
from types import ModuleType

from vastlib.cli import _args, _compose, _docs
from vastlib.core import config

#: Every top-level command, in argparse insertion order — which is the order
#: `herdd --help` prints. Entries are MODULE names under `vastlib.cli`, so the
#: one command whose name is not a Python identifier appears here with an
#: underscore (`dash-cache` -> `dash_cache`); the command name itself is the
#: module's own business, declared in its `add_parser`.
#:
#: The four groups (`job`, `notify`, `fleet`, `workflow`) are subpackages and
#: sit at the end because that is where the flat file put them.
#:
#: DELIBERATELY MARKER-LESS (ruled 2026-08-16, wave 6a). This tuple is the frozen
#: ORDER of `main()`'s hand-inlined `_add_cmd(sub, …)` blocks — a partial port of
#: a function body, not a symbol anyone reaches by name. It carried a
#: `# moved-from: herdd.main (…)` marker, which was MALFORMED (the grammar is
#: `<module>.<name>[ -> <new>]`, no prose), and `herdd.main -> _REGISTRY` would
#: have given `herdd.main` a second rename target alongside `main()` below —
#: which is the one an external caller or a `monkeypatch.setattr` site reaches.
#: Recorded in `gen_rename_table.py::KNOWN_MARKERLESS`. The order itself is
#: pinned by `test_vastlib_cli_main.py` against the flat surface, name for name.
_REGISTRY: tuple[str, ...] = (
    "whoami",
    "search",
    "launch",
    "ls",
    "box",
    "show",
    "ssh",
    "tunnel",
    "debug",
    "metrics",
    "wait",
    "stop",
    "start",
    "sync",
    "shipcheck",
    "label",
    "destroy",
    "guard",
    "reap",
    "salvage",
    "logs",
    "bid",
    "runs",
    "dash_cache",
    "supervise",
    "train",
    "job",
    "notify",
    "fleet",
    "workflow",
)


def _command_module(name: str) -> ModuleType:
    """Import one registry entry.

    Imported lazily (by name) rather than with 29 `from … import` lines at the
    top of this file for one reason: this module is the only place that knows
    the whole roster, and a static import list would have to be kept in sync
    with `_REGISTRY` by hand — two spellings of one fact. The failure mode of a
    typo'd or missing entry is an ImportError naming the module, at startup,
    before any parser is built.
    """
    return importlib.import_module(f"vastlib.cli.{name}")


def build_parser() -> argparse.ArgumentParser:
    """Compose the whole parser tree and return it. Builds, never dispatches.

    Split out of `main()` for the CLI-surface gate: `test_vastlib_cli_surface.py`
    prefers a dedicated builder (`VASTLIB_PARSER_SEAMS`) and falls back to
    driving `main()` with `parse_args` intercepted — the same trick it needs on
    the flat side, where no such seam exists. A builder is the better seam
    because the fallback proves the tree only as a side effect of a function
    whose real job is to run a command; here the tree IS the return value.

    `load_env()` stays FIRST and stays inside the builder, not above the call
    site. It is the flat prologue's first statement, and a parser default that
    reads `os.environ` must see the same values on both arms or the two help
    trees legitimately disagree. It is `setdefault`-based and therefore
    idempotent, so calling it here costs `main()` nothing.
    """
    config.load_env()
    ap = argparse.ArgumentParser(
        prog="herdd", description="herdd — vast.ai control CLI",
        epilog=_args._docs_epilog(_docs.DOC_SKILL, _docs.DOC_README, _docs.DOC_TRAINING),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for _name in _REGISTRY:
        _command_module(_name).add_parser(sub, _args._add_cmd)

    return ap


# moved-from: herdd.main
def main() -> None:
    # The composition root's one non-parser job: point the three cross-ring
    # seams at their real implementations (`cli/_compose.py` explains why they
    # cannot be closed where they are called, and why this is a call-time bind
    # rather than three assignments at import). Before `parse_args`, so a
    # command whose parser DEFAULT reaches one of them is wired too, and
    # unconditional, because a bind behind an `if a.cmd == …` is how one
    # command ends up with a jobd-less `--jobs` box.
    _compose.bind()
    a = build_parser().parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
