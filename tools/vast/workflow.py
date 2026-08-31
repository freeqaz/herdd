#!/usr/bin/env python3
"""`workflow.py` — deprecation shim over `vastlib.workflows.spec`.

Why this exists
---------------
The code that used to live here moved to `vastlib/workflows/spec.py` (plan §8
step 5); this file became a re-export shim at step 7. It is NOT a transitional
scaffold with an expiry date — **the bare name `workflow` is a frozen AUTHORING
contract**. Every workflow spec ever written says `from workflow import
Workflow, JobStage, ...`, a bare-name import resolved off `sys.path` when
`workflowctl.load_workflow_module` path-loads the authored file; the loader then
`isinstance()`-checks the result against its own `Workflow`. Specs exist outside
this repo and cannot be grepped, so there is no deprecation window that finds
them. **Plain deletion is permanently UNSAFE — escalate to the owner before
proposing removal** (`vastlib/workflows/spec.py`, "the exact step-7 shim
recipe").

Because this module only re-exports, `workflow.Workflow is
vastlib.workflows.spec.Workflow` — one class object, so `isinstance` holds
whichever side imported it, and the identity split the port opened is closed.

What is deliberately NOT here
-----------------------------
* No class or function bodies. A shim that redefines anything forks the class
  identity it exists to preserve; every name below is a re-export.
* No `sys.path` bootstrap. This file is import-only (no `__main__` guard), so a
  caller that resolved the bare name `workflow` already has `tools/vast` on
  `sys.path` and therefore reaches `vastlib` too. (`jobmatrix.py`'s shim does
  keep one — it is executable as a frozen CLI.)
* No new API. Extend `vastlib.workflows.spec`, never this file.

Provenance: body ported to `vastlib.workflows.spec` @ `ea8360dc` (manifest
`workflows-core.json`); shim shape per plan §3 and the recipe in that module's
docstring, audit `.port_manifests/step7-shims.json`.
"""
from __future__ import annotations

import sys

import vastlib.workflows.spec as _spec
from vastlib.workflows.spec import (  # noqa: F401  re-export
    RENTAL_CHOICES,
    RETRY_CLASSES,
    STAGE_NAME_RE,
    TEARDOWN_CHOICES,
    ArtifactContract,
    InputRef,
    JobStage,
    ResourceProfile,
    RetryPolicy,
    Workflow,
    WorkflowError,
    _non_negative,
    _slug,
    _tuple_of_str,
)

# An authored spec resolving the bare name must get the SAME module object the
# controller imported, or isinstance() sees two classes. On the normal
# `import workflow` path this is a no-op (the name is registered while the
# module executes); it earns its keep if this file is ever loaded under a
# synthetic name (spec_from_file_location) — cheap insurance, and the recipe.
sys.modules.setdefault("workflow", sys.modules[__name__])
sys.modules.setdefault("vastlib.workflows.spec", _spec)
