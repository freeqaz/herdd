#!/usr/bin/env python3
"""`jobmatrix.py` — deprecation shim over `vastlib.workflows.matrix` + frozen CLI.

Why this exists
---------------
The expander moved to `vastlib/workflows/matrix.py` (plan §8 step 5); this file
became a re-export shim at step 7. Two things keep it alive permanently:

1. **The bare name `jobmatrix` is a frozen AUTHORING contract.** Every authored
   `matrix.py` says `from jobmatrix import Experiment, Variant`, is executed by
   `runpy.run_path`, and the resulting `EXPERIMENT` is `isinstance`-checked by
   the loader. Matrices exist on laptops no grep can reach, so **plain deletion
   is UNSAFE — permanently. Escalate to the owner before proposing removal.**
2. **`python3 tools/vast/jobmatrix.py expand|submit|status` is a frozen CLI**
   (`MATRIX_DESIGN.md`, the `herdd` skill, `launch_jobs_box.sh:239`). The
   `__main__` guard therefore lives HERE; the package module deliberately has
   none, because it may not bootstrap `sys.path` for its own bare-name Zone S
   imports and Zone E is the only place a bootstrap may live (plan §3).

The `sys.modules.setdefault` below is load-bearing, not decoration: on the CLI
path this file executes as `__main__`, so the name `jobmatrix` is unregistered
and an authored matrix's `from jobmatrix import Experiment` would import the
file a SECOND time under a second module object with a second `Experiment`
class. The flat file carried that alias for exactly this reason since `:69-72`;
the shim keeps it.

What is deliberately NOT here
-----------------------------
* No class or function bodies — every name below is a re-export, so
  `jobmatrix.Experiment is vastlib.workflows.matrix.Experiment`.
* No re-statement of the expander's behavior. The `os.getcwd()` staging/manifest
  anchoring the flat copy used to carry is GONE with the body: this shim runs
  `vastlib.workflows.matrix`, which anchors both to `_REPO_ROOT` and creates the
  staging dir.
* The `sys.path` bootstrap IS kept (unlike the other shims): this file is
  executable, and both `vastlib` and the Zone S modules `matrix` imports bare
  (`jobmeta`, `disksize`) live in this directory. Under a wrapper, `-m`, or a
  console script, `sys.path[0]` is not `tools/vast`.

Provenance: body ported to `vastlib.workflows.matrix` @ `ea8360dc` (manifest
`workflows-core.json`); shim shape per plan §3 and the recipe in that module's
docstring, extended with the six private names tests read through this name,
audit `.port_manifests/step7-shims.json`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vastlib.workflows.matrix as _m  # noqa: E402
from vastlib.workflows.matrix import (  # noqa: E402,F401  re-export
    MANIFEST_VERSION,
    MATRIX_FILENAME,
    RESERVED_ENV,
    Arm,
    Experiment,
    MatrixError,
    Variant,
    _as_variant,
    _cmd_expand,
    _cmd_status,
    _cmd_submit,
    _repo_root,
    _resolve,
    exp_status,
    expand,
    load_experiment,
    main,
    read_manifest,
    submit,
    validate_experiment,
)

# Namespace parity with the pre-shim module: the two Zone S / sibling seams it
# bound as MODULE objects (`monkeypatch.setattr(jobmatrix.jobmeta, …)` is the
# idiom the suites rest on, and these are the same module objects the port
# patches through).
from vastlib.workflows.matrix import disksize, jobmeta  # noqa: E402,F401  re-export

# When executed as a script (python3 tools/vast/jobmatrix.py …), make the
# matrix file's `from jobmatrix import Experiment` resolve to THIS module
# instance — otherwise isinstance(EXPERIMENT, Experiment) sees two classes.
sys.modules.setdefault("jobmatrix", sys.modules[__name__])
sys.modules.setdefault("vastlib.workflows.matrix", _m)

if __name__ == "__main__":            # frozen CLI surface — see the docstring
    main()
