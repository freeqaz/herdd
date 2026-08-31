"""vastlib.cli.workflow._entry — the Zone E script path a detached controller re-execs.

Why this module exists
----------------------
`workflow run --detach` and `workflow resume --detach` hand the reconcile loop
to `systemd-run --user` by building the exact FOREGROUND argv the same command
would have run, and `workflowctl.spawn_detached` turns that argv into a unit
(and, when `systemd-run` is missing, into the `DetachUnavailable` message the
operator is told to paste). Both sites spell the argv the same way:

    [sys.executable, os.path.abspath(__file__), "workflow", "run", path]

where `__file__` was `tools/vast/herdd.py` — the Zone E entry script whose
path is frozen (plan §3: ~30 callers, the reaper unit, the dashboard spawn
sites, ~550 markdown references). Inside the package `__file__` is four levels
deeper and points at a module that is not executable, so the expression has to
be re-anchored to keep producing the same string. That is the whole job of this
file, and it is deliberately one line of arithmetic with a name on it rather
than a repeated `dirname(dirname(dirname(dirname(...))))` at two call sites —
the same treatment `core/config.py::_HERE`, `boxes/ssh.py::_REPO_ROOT` and
`workflows/ctl.py::_TOOLS_VAST_DIR` got, and for the same reason: a depth
change is a silent wrong answer, so it lives in exactly one place.

What is deliberately NOT here
-----------------------------
* The argv itself. Which subcommand and which arguments a detached controller
  re-execs is the command module's business (`run.py` / `resume.py` build it
  verbatim); this module only answers "where is the entry script".
* `sys.executable`. The interpreter is whatever is running now, by design —
  a detached controller must not silently switch venvs.
* Any other Zone E path. `fleetd.py`'s anchor already lives in
  `fleet/deploy.py`; if a second CLI re-exec site lands (`herdd.py:9630`,
  the `train --detach` lane), this constant is what it should move up to
  `cli/_entry.py` to share — not a second copy.

Provenance: re-anchored (not textually verbatim, deliberately so — see above)
from `tools/vast/herdd.py`'s `cmd_workflow_run` / `cmd_workflow_resume`,
plan §8 step 6, 2026-08-16.
"""

from __future__ import annotations

import os

# `tools/vast/` — four dirnames up from `tools/vast/vastlib/cli/workflow/`.
_TOOLS_VAST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# What `os.path.abspath(__file__)` evaluated to in the flat module. Pinned
# against the flat file's own resolution by
# `test_vastlib_cli_workflow.py::test_herdd_script_anchor_matches_the_flat_module`.
HERDD_SCRIPT = os.path.join(_TOOLS_VAST_DIR, "herdd.py")
