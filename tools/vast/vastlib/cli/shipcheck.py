"""`herdd shipcheck [imports|env|all]` — pre-flight on what actually reaches a box.

Two derived guards, one implementation. `tools/vast/shipcheck.py` owns both —
the ship-manifest import closure and baked-eval-env staleness — and three
callers share it: this command, `cli/sync.py` (through
`jobs.bundle._sync_import_gate`, the imports half) and `bake.sh`. That is the
whole design: a guard with two implementations is a guard that disagrees with
itself on the box, after the meter has started.

`sys.exit(shipcheck.main(argv))` — the exit code IS the product
---------------------------------------------------------------
Exit 1 on a finding, so this composes into a shell `&&` chain in front of a wave
submit. `--warn-only` forces exit 0 for the case where the operator wants the
report but has already decided to ship.

Why `import shipcheck` is bare-name, function-local, and preceded by a
`sys.path.insert`
------------------------------------------------------------------
`shipcheck.py` is an absorbed sibling that is still a FLAT file at
`tools/vast/shipcheck.py` (it lands inside the package at plan step 7). vastlib
reaches it the way Zone S leaves are reached — bare name, with `tools/vast` on
`sys.path` — and the insert is what makes that resolvable from a caller that did
not bootstrap `tools/vast` itself. All three lines are ported verbatim from the
flat `cmd_shipcheck` for exactly that reason; `jobs/bundle.py::_sync_import_gate`
does the same thing for the same reason.

What is deliberately NOT here
-----------------------------
* Either check. Import-closure parsing, the manifest reader and the baked-env
  MANIFEST comparison are all `shipcheck.py`'s.
* Any argv beyond translating this parser's flags. The flag names are the
  script's own, one-for-one, so a `shipcheck.py` flag can never mean one thing
  here and another under `bake.sh`.

Provenance: moved from `tools/vast/herdd.py` (`cmd_shipcheck`, parser block in
`main()`), plan §8 step 6, 2026-08-16, behavior-preserving. The one mechanical
change: the `sys.path` entry comes from `_TOOLS_VAST_DIR` below rather than from
this module's own `__file__`, which is three directories deeper.
"""

from __future__ import annotations

import argparse
import os
import sys

from vastlib.cli import _args, _docs

# `tools/vast/` — three dirnames up from `tools/vast/vastlib/cli/`. The flat
# `cmd_shipcheck` spelled this `os.path.dirname(os.path.abspath(__file__))`
# inline; a wrong depth here would make the bare-name import unresolvable, so
# the arithmetic lives in one named constant (as in `cli/_runsets.py::_HERE`).
_TOOLS_VAST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# moved-from: herdd.cmd_shipcheck
def run(a: argparse.Namespace) -> None:
    """Thin wrapper over tools/vast/shipcheck.py — the ONE implementation of both
    derived ship-manifest guards, shared with `sync` (imports) and bake.sh.
    Exit 1 on a finding so it can gate a submit in a shell `&&` chain."""
    sys.path.insert(0, _TOOLS_VAST_DIR)
    import shipcheck
    argv = [a.check]
    for flag, val in (("--env-manifest", a.env_manifest),
                      ("--env-version", a.env_version), ("--box", a.box)):
        if val:
            argv += [flag, val]
    if a.warn_only:
        argv.append("--warn-only")
    sys.exit(shipcheck.main(argv))


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    pshc = add_cmd(sub, "shipcheck",
                   "$0 pre-flight on what reaches a box: ship-manifest import "
                   "closure + baked-env staleness",
                   _docs.DOC_README, _docs.DOC_EVALS,
                   "run before a wave submit; `sync` runs the import half itself")
    pshc.add_argument("check", nargs="?", default="all",
                      choices=("imports", "env", "all"))
    pshc.add_argument("--env-manifest", help="baked eval-env MANIFEST json "
                                            "(default: newest in out/eval-env/dist)")
    pshc.add_argument("--env-version", help="env version, resolved in out/eval-env/dist")
    pshc.add_argument("--box", default="<box>", help="box id, for the fix hint")
    pshc.add_argument("--warn-only", action="store_true", help="always exit 0")
    pshc.set_defaults(func=run)
    return pshc
