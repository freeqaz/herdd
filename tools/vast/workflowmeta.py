#!/usr/bin/env python3
"""`workflowmeta.py` — deprecation shim over `vastlib.workflows.meta`.

Why this exists
---------------
Spec validation, workflow IDs, the frozen event vocabulary and the pure fold
moved to `vastlib/workflows/meta.py` (plan §8 step 5); this file became a
re-export shim at step 7. Unlike `workflow.py`/`jobmatrix.py` there is no
bare-name AUTHORING contract on `workflowmeta` — nothing on disk imports it by
name outside this tree — so the shim exists purely so in-repo callers and the
flat owner suite (`test_workflowmeta.py`) keep resolving one set of objects
with the port: `workflowmeta.WorkflowSpecError is
vastlib.workflows.meta.WorkflowSpecError`, which is what makes
`pytest.raises` and every `except` clause agree across the two names.

What is deliberately NOT here
-----------------------------
* No class or function bodies — re-exports only. A redefined exception class
  would silently stop being caught by callers that imported the other name.
* **No `sys.path` bootstrap.** The flat file used to self-bootstrap (`_HERE` +
  `sys.path.insert`, `:36-38`), which made it importable from anywhere; the
  port dropped that on purpose (a bootstrap may only live in Zone E for a file
  that is actually executable, plan §3, and this one is import-only). After the
  shim, `import workflowmeta` requires `tools/vast` on `sys.path` — true for
  every in-repo caller, all of which are `tools/vast`-rooted pytest runs whose
  rootdir conftest puts the directory there.
* No `sys.modules` aliasing — there is no path-loaded file resolving this name.
* No new API. Extend `vastlib.workflows.meta`.

The `runmeta` primitives and the `workflow` DSL names below are re-exported
because the flat module's namespace carried them (`from runmeta import ...`,
`from workflow import ...`) and callers read them through this name — e.g.
`wm.now_ts` / `wm.event_key` from the controller, `wm.WorkflowError` from
`test_workflowmeta.py:467`. They are the same objects `vastlib.workflows.meta`
exposes, not a second binding of their own.

Provenance: body ported to `vastlib.workflows.meta` @ `ea8360dc` (manifest
`workflows-core.json`); shim shape per plan §3 and the recipe in
`vastlib/workflows/spec.py`, audit `.port_manifests/step7-shims.json`.
"""
from __future__ import annotations

from vastlib.workflows.meta import (  # noqa: F401  re-export
    CONTROLLER_EVENTS,
    EVENTS,
    FAILURE_CLASSES,
    SCHEMA_VERSION,
    STAGE_TERMINAL,
    WF_ID_RE,
    WF_SLUG_RE,
    WORKFLOW_TERMINAL,
    WorkflowIdError,
    WorkflowSpecError,
    _ASSET_NAME_RE,
    _CORE_KEYS,
    _FAILURE_CLASS_EXIT_CODE,
    _STAGE_TERMINAL_RANK,
    _TS_RE,
    _WORKFLOW_TERMINAL_RANK,
    _cell,
    _check_acyclic,
    _coerce,
    _num,
    _parse_ts,
    _profile_from_dict,
    _profile_to_dict,
    _stage_from_dict,
    _stage_to_dict,
    _ts_diff_seconds,
    canonical_spec_json,
    controller_is_stale,
    decide_retry,
    failure_class_exit_code,
    fold_workflow_events,
    format_status_table,
    input_ref_asset,
    make_event,
    mint_wf_id,
    next_ready_stage,
    ready_stages,
    require_from_manifest,
    spec_from_dict,
    spec_to_dict,
    stage_job_id,
    status_table_rows,
    validate_wf_id,
    validate_workflow_spec,
    wf_slugify,
    wf_ts,
)

# Namespace parity with the pre-shim module: names it re-exported from its own
# imports, read by callers through this one (see the docstring).
from vastlib.workflows.meta import (  # noqa: F401  re-export
    RETRY_CLASSES,
    STAGE_NAME_RE,
    ArtifactContract,
    InputRef,
    JobStage,
    ResourceProfile,
    RetryPolicy,
    Workflow,
    WorkflowError,
    _actor_slug,
    event_key,
    nonce,
    now_ts,
    runmeta,
)
