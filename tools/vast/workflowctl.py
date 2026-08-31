#!/usr/bin/env python3
"""`workflowctl.py` — deprecation shim over `vastlib.workflows.ctl`.

Why this exists
---------------
The workflow controller — spec/event transport, the pure-per-call
`reconcile_tick`, box acquisition and teardown, the local claim lock, cost
accrual, and the CLI verb bodies — moved to `vastlib/workflows/ctl.py` (plan §8
step 5); this file became a re-export shim at step 7. It has no `__main__`
guard and never had one: the operator surface is `herdd workflow <verb>`, not
`python3 workflowctl.py`. The shim exists so in-repo importers (`test_workflow.py`,
`test_workflow_preflight.py`, `test_vram_gate.py`, the twin suites) keep
resolving ONE set of objects with the port — `workflowctl.WorkflowCtlError is
vastlib.workflows.ctl.WorkflowCtlError`, so `pytest.raises` and every `except`
clause agree across both names.

Read this before patching through this module
---------------------------------------------
**A re-export is not a steering seam.** `monkeypatch.setattr(workflowctl,
"reconcile_tick", spy)` rebinds THIS module's attribute; `run_controller` and
every other ported caller resolves `reconcile_tick` in `vastlib.workflows.ctl`'s
globals and will never see the patch. A test that patches here and then calls a
function here gets the REAL implementation and stays green while steering
nothing. Patch `vastlib.workflows.ctl` instead — that is why
`test_workflow.py`/`test_vram_gate.py` were repointed at the port in the same
commit as this shim (audit `.port_manifests/step7-shims.json`,
`gate_inventory.workflowctl_patch_vacuity`).

What is deliberately NOT here
-----------------------------
* No class or function bodies — re-exports only.
* No `sys.path` bootstrap (the flat file's `_HERE` + `sys.path.insert`, `:40-42`).
  This module is import-only, so a caller that resolved the bare name already
  has `tools/vast` on `sys.path`. `_HERE` survives below only as an alias of the
  port's `_TOOLS_VAST_DIR` (same value: the `tools/vast` directory), which the
  port had to compute differently because it sits two levels deeper.
* No `sys.modules` aliasing — nothing path-loaded resolves the name
  `workflowctl` (the AUTHORING contract is on `workflow`, see `workflow.py`).
* No new API. Extend `vastlib.workflows.ctl`.

Provenance: body ported to `vastlib.workflows.ctl` @ `ea8360dc` (manifest
`workflows-ctl.json`); shim shape per plan §3 and the recipe in
`vastlib/workflows/spec.py`, export list from `.port_manifests/rename_table.json`.
"""
from __future__ import annotations

from vastlib.workflows.ctl import (  # noqa: F401  re-export
    BOOT_DEADLINE_S,
    BOOT_MIN_MBPS,
    CANARY_RECEIPT_TTL_S,
    DetachUnavailable,
    EXIT_ARTIFACT,
    EXIT_CANCELLED,
    EXIT_CREDENTIAL,
    EXIT_FAILED,
    EXIT_INVALID,
    EXIT_OK,
    EXIT_TIMEOUT,
    HEARTBEAT_STALE_MULT,
    IMAGE_GATE_ENFORCE,
    IMAGE_GATE_FAILURE_CLASS,
    JOB_HEARTBEAT_STALE_S,
    LAUNCH_INET_DOWN_FLOOR_MBPS,
    LiveCostObserver,
    POLL_INTERVAL_S,
    REHEARSAL_DISCLAIMER,
    STAGE_INFLIGHT,
    STOPPED_STATES,
    TAKEOVER_WAIT_GRACE_S,
    TEARDOWN_MAX_ATTEMPTS,
    WorkflowCtlError,
    _FAILURE_CLASS_MAP,
    _TOOLS_VAST_DIR as _HERE,
    _LIVE_DEP_KEYS,
    _RUNNING_JOB_STATUSES,
    _accept_stage_artifacts,
    _artifact_key,
    _attempt_teardown,
    _build_default_stage_rehearser,
    _build_stage_config,
    _canary_receipt_key_ref,
    _canonical_sha256,
    _check_credential_horizon,
    _check_stage_inputs,
    _classify_job_failure,
    _default_asset_checker,
    _default_box_resolver,
    _default_box_starter,
    _default_canary_checker,
    _default_runner,
    _ensure_bundle_uploaded,
    _image_gate_refuses,
    _instance_image_verdict,
    _lock_dir,
    _owned_instance_ids,
    _plan_and_submit_stage,
    _plan_online,
    _plan_online_canary,
    _plan_online_credentials,
    _prior_stage_machines,
    _provenance_key,
    _q,
    _read_stage_manifest,
    _reconcile_completion,
    _released_instance_ids,
    _render_provenance,
    _render_report_md,
    _repo_root,
    _report_key,
    _resolve_controller_deps,
    _resolve_runner,
    _rotate_credential,
    _seconds_between,
    _stable_manifest_sha256,
    _stage_bundle_dir,
    _stage_job_extras,
    _stage_manifest_path,
    _teardown_attempts_seen,
    _teardown_boxes,
    _teardown_failed_recorded,
    _terminal_exit_code,
    _topo_order_stages,
    _validate_stage_bundle,
    _verdict_key,
    accrue_and_persist_cost,
    acquire_local_lock,
    budget_exhausted,
    build_box_observer,
    build_box_resolver,
    build_box_teardown,
    build_cost_observer,
    build_image_state_observer,
    build_live_controller_deps,
    canary_launch_env,
    canary_receipt_key,
    canary_receipt_status,
    cancel_workflow,
    claim_controller,
    credential_horizon_ok,
    emit,
    folded_spend,
    heartbeat,
    load_workflow_module,
    logs_workflow,
    owned_boxes_remaining,
    plan_workflow,
    pull_workflow,
    read_accepted_artifact,
    read_canary_receipt,
    read_events,
    read_spec,
    reconcile_active_box,
    reconcile_tick,
    record_box_cost,
    rehearse_workflow,
    release_local_lock,
    remaining_wall_s,
    resume_workflow,
    run_controller,
    run_workflow,
    spawn_detached,
    stage_canary_key,
    status_extras,
    status_workflow,
    view,
    write_canary_receipt,
    write_spec,
)
