"""vastlib — the vast.ai tooling package (Zone P of the three-zone layout).

Why this package exists
-----------------------
`tools/vast/herdd.py` reached 19,931 lines, 467 top-level functions, 65
`cmd_*` entry points and ~1% type coverage, with `fleetd.py` (4,176 lines)
importing it as its policy source and nine other siblings importing it too.
The 2026-07-30 strangler plan lost the race — the file grew 9,200 lines while
three of its seven increments landed. `vastlib` is the designed replacement:
one package, a strictly downward dependency DAG, strict typing, and a single
coordinated test migration instead of thirty more increments of the same audit.

The three zones (full statement in `README.md`, doctrine in the plan doc §3)
-----------------------------------------------------------------------------
  ZONE S  shipped flat leaves (`jobmeta`, `runmeta`, `bidpolicy`,
          `metrics_probe`, `gemm_probe`, `parse_vllm_mem`, `triton_cache`,
          `onstart/*`) — stdlib only, bare-name imports, copied into job
          bundles by basename. `vastlib` MAY import them; they may NEVER
          import `vastlib`.
  ZONE P  this package — workstation-only, strict-typed, pydantic allowed.
  ZONE E  `tools/vast/herdd.py` and `tools/vast/fleetd.py` stay as thin
          real scripts at their exact current paths (the reaper systemd unit,
          fleetd's deploy audit, `wave_driver.py`, the dashboard argv
          literals, the skills and ~550 doc references all keep working).

The dependency rule (enforced, not merely documented)
-----------------------------------------------------
    core  ->  {market, boxes, launch, storage}
          ->  {supervise, jobs, fleet, workflows}
          ->  cli

`core` imports stdlib + Zone S only. Nothing imports `cli`. `fleet.daemon`
and `cli.main` are the only composition roots. `import-linter` checks the DAG
from `tools/vast/vastlib/importlinter.ini`, wrapped by
`tools/vast/test_vastlib_static.py` because tools/vast has no CI.

What is deliberately NOT here
-----------------------------
* **No `sys.path` manipulation, anywhere inside this package.** The package is
  written installable-clean from day one (owner decision, plan §0.1) so the
  eventual promotion to a real pyproject package is a move, not a rewrite. The
  Zone E entry scripts own the bootstrap.
* **No re-export shims for `herdd.<name>`.** A `setattr(herdd, name)`
  steers execution only while the *caller* also lives in `herdd.py`; in a
  big-bang cutover the callers move, so tests are repointed to the new homes
  (plan §7) rather than propped up by a shim that quietly guards nothing.
* **No Zone S code.** Shipped leaves stay flat files forever — the bundle
  copies by basename and always will.
* **No behavior changes.** Ports are verbatim moves plus annotations. The
  parked fixes (`_put_label_soft`'s duplicate definition, the ~70 bare
  `os.environ.get` reads, lane unification) are plan §9, not this refactor.

Provenance: created 2026-08-16 as step 1 of `docs/plans/vast-tooling-refactor-v2.md`
§8 (branch `vast-refactor-v2`, worktree-only). No code has moved yet — this
commit is the skeleton plus the static lane that every later port is judged by.
Ported symbols carry a `# moved-from:` marker; grammar in `README.md`.
"""

from __future__ import annotations
