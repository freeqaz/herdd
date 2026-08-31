"""`vastlib.workflows.meta` — spec validation, IDs, events, and the pure workflow fold.

Why this exists
---------------
The workflow controller's DECISIONS must be testable without a network, a
bucket, or a clock it does not own. Everything in this module is pure: given a
`Workflow` (from the sibling `workflow` DSL) and a bag of raw event bodies, it
answers what the workflow's state IS, which stage is ready next, whether a
failed attempt may retry, whether the controller heartbeat has gone stale, and
what bytes `spec.json` must contain. The transport that reads and writes those
bytes lives one module up in `ctl.py`, and it never re-implements a decision
made here.

It is a SIBLING of Zone S's `runmeta.py`/`jobmeta.py`, not a copy: it imports
their clock/nonce/event-key primitives so there is ONE implementation of that
discipline, and adds only what genuinely differs for workflows — the
`workflow_id` field and its own frozen event vocabulary, cross-object spec
validation for the DSL, canonical `spec.json` round-tripping, the fold, and the
deterministic stage-attempt JOB_ID.

Frozen wire contracts carried by named symbols here
---------------------------------------------------
Objects on B2 are immutable once written, so several names in this file are
FROZEN and may only ever grow (`docs/plans/herdd-workflow-v1-interfaces.md`):

* `EVENTS` — the 17 shipped V1 event names. The fold tolerates any name NOT in
  the set (counted as parse info, never dropped), which is exactly why
  `ctl.py`'s own local events (`box_cost`, `teardown_attempt`) are deliberately
  NOT added here: this set is `meta`'s alone to grow.
* `make_event` — the `v`/`ts`/`actor`/`event`/`workflow_id`/`nonce` envelope
  every emitted object passes through.
* `stage_job_id` — the deterministic `<ts15>-<hash8>-<stage>-a<n>` JOB_ID.
  Idempotent resubmission of a reconcile tick depends on it being stable;
  changing the hash or the layout orphans in-flight jobs.
* `canonical_spec_json` + `spec_to_dict`/`spec_from_dict` — the exact bytes
  `ctl.write_spec` stores and byte-compares on resubmit. A key-order or
  separator change reads as a spec conflict on every live workflow.
* `FAILURE_CLASSES` + `_FAILURE_CLASS_EXIT_CODE` — the class vocabulary and its
  exit-code map, which MUST equal `ctl.EXIT_*`. Two tables, one contract:
  `test_vastlib_workflows_ctl.py::test_exit_code_tables_agree` asserts the
  correspondence rather than either file duplicating the other's constants.

What is deliberately NOT here
-----------------------------
* **Transport.** No subprocess, no rclone, no HTTP, no filesystem. Reading
  `spec.json`/events from B2 and submitting the deterministic JOB_ID belong to
  `ctl.py` and `jobmeta.submit_with_id`.
* **`jobmeta`.** This module never imports it — which is why the job-vocabulary
  translation `_classify_job_failure` lives in `ctl.py` instead of here, and why
  `_ASSET_NAME_RE` is duplicated rather than imported.
* **The staleness INTERVAL.** `controller_is_stale` does the pure age
  comparison; `POLL_INTERVAL_S * HEARTBEAT_STALE_MULT` is `ctl.py`'s knob.
* **`mint_wf_id` is the one impurity** — it calls `os.urandom(2)` when no
  `nonce4` is injected. Left as found (plan §7.4).

Provenance: `tools/vast/workflowmeta.py`, moved verbatim in plan §8 step 5
(behavior-preserving; the only non-textual changes are the dropped
`sys.path.insert` — forbidden inside Zone P by plan §3 — and the type
annotations mypy strict requires).

The DSL import is `vastlib.workflows.spec` — and that is now safe
-----------------------------------------------------------------
It was BARE-NAME (`from workflow import ...`) until step 7, because
`ctl.load_workflow_module` path-loads an AUTHORED spec file that says
`from workflow import Workflow` and then `isinstance()`-checks it against the
class this package imported: two different resolutions means two different class
objects and every spec fails to load. Step 7 turned the flat `workflow.py` into
a pure re-export shim over `spec`, so both spellings now yield ONE set of class
objects and the import below can name its real home. See `spec.py`'s header for
the shim recipe and why deleting the flat file stays unsafe.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
from collections.abc import Iterable
from typing import Any

from vastlib.workflows.spec import (
    RETRY_CLASSES,
    STAGE_NAME_RE,
    ArtifactContract,
    InputRef,
    JobStage,
    ResourceProfile,
    RetryPolicy,
    Workflow,
    WorkflowError,
)

import runmeta  # noqa: F401  module object bound as an attribute (patch seam)
from runmeta import _actor_slug as _actor_slug  # noqa: F401  explicit re-export
from runmeta import event_key as event_key  # noqa: F401  explicit re-export
from runmeta import nonce as nonce  # noqa: F401  explicit re-export
from runmeta import now_ts as now_ts  # noqa: F401  explicit re-export

# The four `runmeta` names above are re-exported with the redundant `as` form
# ON PURPOSE. `ctl.py` calls `wm.now_ts()` / `wm.event_key(ev)` (and
# `test_workflowmeta.py` reaches the same way), and mypy's strict
# `no_implicit_reexport` would otherwise reject every one of those reads. The
# plain `import runmeta` above it is also load-bearing: it binds the MODULE
# object as an attribute, which is the patch seam tests use.

# --- frozen schema (v1) -------------------------------------------------------
# moved-from: workflowmeta.SCHEMA_VERSION
SCHEMA_VERSION = 1

# Frozen V1 event names (roadmap "Workflow state and events"). Objects are
# immutable once shipped, so any name here lives forever; the fold below
# tolerates any name NOT in this set (unknown events are counted as parse
# info, never dropped as errors, and never crash the fold).
# moved-from: workflowmeta.EVENTS
EVENTS = frozenset({
    "submitted", "controller_started", "controller_heartbeat", "takeover",
    "stage_planned", "box_acquired", "stage_submitted", "stage_started",
    "artifact_accepted", "stage_succeeded", "stage_failed", "stage_cancelled",
    "teardown_started", "box_released",
    "workflow_succeeded", "workflow_failed", "workflow_cancelled",
})

# moved-from: workflowmeta.STAGE_TERMINAL
STAGE_TERMINAL = frozenset({"stage_succeeded", "stage_failed", "stage_cancelled"})
# moved-from: workflowmeta.WORKFLOW_TERMINAL
WORKFLOW_TERMINAL = frozenset({"workflow_succeeded", "workflow_failed", "workflow_cancelled"})
# moved-from: workflowmeta.CONTROLLER_EVENTS
CONTROLLER_EVENTS = frozenset({"controller_started", "controller_heartbeat", "takeover"})

# Terminal workflow precedence (roadmap): an observed real outcome outranks a
# late operator action. Higher rank wins when more than one terminal workflow
# event is present (a race between the reconciler and an operator cancel).
# moved-from: workflowmeta._WORKFLOW_TERMINAL_RANK
_WORKFLOW_TERMINAL_RANK = {
    "workflow_failed": 3,
    "workflow_succeeded": 2,
    "workflow_cancelled": 1,
}
# Per-stage-attempt terminal precedence mirrors the same principle.
# moved-from: workflowmeta._STAGE_TERMINAL_RANK
_STAGE_TERMINAL_RANK = {
    "stage_failed": 3,
    "stage_succeeded": 2,
    "stage_cancelled": 1,
}

# Frozen V1 terminal-workflow failure classes (roadmap "Failure classes and
# exit codes"). A `workflow_failed` event's `failure_class` field is drawn
# from this set; the CLI exit-code mapping below is derived from it.
# moved-from: workflowmeta.FAILURE_CLASSES
FAILURE_CLASSES = frozenset({
    "CONFIG_INVALID", "ASSET_STALE", "ARTIFACT_INVALID", "IMAGE_DRIFT",
    "CREDENTIAL_EXPIRES", "ENV_CANARY_FAILED", "INFRASTRUCTURE_FAILED",
    "ENTRYPOINT_FAILED", "CHECKPOINT_STALLED", "POSTCONDITION_FAILED",
    "BUDGET_EXHAUSTED", "WALL_EXHAUSTED", "RETRY_EXHAUSTED", "TEARDOWN_FAILED",
})

# moved-from: workflowmeta._FAILURE_CLASS_EXIT_CODE
_FAILURE_CLASS_EXIT_CODE = {
    "CONFIG_INVALID": 1, "ASSET_STALE": 1, "IMAGE_DRIFT": 1,
    "ARTIFACT_INVALID": 4, "POSTCONDITION_FAILED": 4,
    "CREDENTIAL_EXPIRES": 5,
    "WALL_EXHAUSTED": 124,
    "ENV_CANARY_FAILED": 2, "INFRASTRUCTURE_FAILED": 2,
    "ENTRYPOINT_FAILED": 2, "CHECKPOINT_STALLED": 2,
    "BUDGET_EXHAUSTED": 2, "RETRY_EXHAUSTED": 2, "TEARDOWN_FAILED": 2,
}


# moved-from: workflowmeta.failure_class_exit_code
def failure_class_exit_code(failure_class: str | None) -> int:
    """Map a TERMINAL workflow `failure_class` (roadmap "Failure classes and
    exit codes") to a CLI exit code. These ints are the single source of
    truth for the class->code mapping and MUST match `workflowctl.EXIT_*`;
    an unknown or missing `failure_class` (including `None`) maps to the
    generic terminal-failure code 2."""
    return _FAILURE_CLASS_EXIT_CODE.get(failure_class, 2)  # type: ignore[arg-type]


# moved-from: workflowmeta._CORE_KEYS
_CORE_KEYS = ("ts", "event", "workflow_id")   # required for a valid event

# WF_ID = <YYYYMMDDTHHMMSS>-<workflow-slug>-<nonce4>; validated with the same
# charset rule runmeta/jobmeta use for RUN_ID/JOB_ID (raw in object keys and a
# vast label), capped at 64 characters.
# moved-from: workflowmeta.WF_ID_RE
WF_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
# name -> slug (becomes part of WF_ID): lowercase alnum + dashes, bounded.
# moved-from: workflowmeta.WF_SLUG_RE
WF_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")


# moved-from: workflowmeta.WorkflowSpecError
class WorkflowSpecError(WorkflowError):
    """A structurally-valid `Workflow` violates a cross-object V1 spec rule
    (uniqueness, acyclic `after`, dependency-declared inputs, profile
    pinning, `retry_on` vocabulary) — as opposed to `workflow.WorkflowError`,
    which is a single field/dataclass shape violation."""


# moved-from: workflowmeta.WorkflowIdError
class WorkflowIdError(ValueError):
    pass


# --- id / slug -----------------------------------------------------------------
# moved-from: workflowmeta.validate_wf_id
def validate_wf_id(wf_id: str) -> str:
    if not isinstance(wf_id, str) or not WF_ID_RE.match(wf_id):
        raise WorkflowIdError(
            f"invalid WF_ID {wf_id!r}: must match {WF_ID_RE.pattern}")
    return wf_id


# moved-from: workflowmeta.wf_slugify
def wf_slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")[:40]
    if not s or not WF_SLUG_RE.match(s):
        raise WorkflowIdError(
            f"invalid workflow name {name!r}: slug must match {WF_SLUG_RE.pattern} "
            f"(lowercase letters/digits/dashes, <=40 chars)")
    return s


# moved-from: workflowmeta.wf_ts
def wf_ts() -> str:
    """Compact UTC stamp for the WF_ID prefix: YYYYMMDDTHHMMSS (no millis/Z —
    that is the event-key clock; this is a human-facing id prefix)."""
    return now_ts()[:15]


# moved-from: workflowmeta.mint_wf_id
def mint_wf_id(name: str, *, ts: str | None = None, nonce4: str | None = None) -> str:
    slug = wf_slugify(name)
    ts = ts or wf_ts()
    n4 = nonce4 or os.urandom(2).hex()
    return validate_wf_id(f"{ts}-{slug}-{n4}")


# moved-from: workflowmeta.stage_job_id
def stage_job_id(workflow_id: str, stage: str, attempt: int) -> str:
    """Deterministic stage-attempt JOB_ID from WF_ID hash + stage + attempt
    (roadmap: `20260713T120000-a1b2c3d4-score-a0`). NOT random — the same
    (workflow_id, stage, attempt) always yields the same id, so submitting a
    stage attempt twice (retry-safe reconcile tick) targets one ticket.
    """
    validate_wf_id(workflow_id)
    if not isinstance(stage, str) or not STAGE_NAME_RE.match(stage):
        raise WorkflowIdError(f"invalid stage name for stage_job_id: {stage!r}")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
        raise WorkflowIdError(f"attempt must be a non-negative int, got {attempt!r}")
    ts15 = workflow_id[:15]
    hash8 = hashlib.sha256(workflow_id.encode("utf-8")).hexdigest()[:8]
    out = f"{ts15}-{hash8}-{stage}-a{attempt}"
    validate_wf_id(out)          # same charset/length rule as any JOB_ID
    return out


# --- canonical spec JSON -------------------------------------------------------
# moved-from: workflowmeta._profile_to_dict
def _profile_to_dict(p: ResourceProfile) -> dict[str, Any]:
    return {
        "image": p.image,
        "image_digest": p.image_digest,
        "gpu": list(p.gpu),
        "num_gpus": p.num_gpus,
        "gpu_ram_gb": p.gpu_ram_gb,
        "disk_gb": p.disk_gb,
        "rental": p.rental,
        "max_bid": p.max_bid,
        "budget_usd": p.budget_usd,
        "max_wall_s": p.max_wall_s,
        "geo": list(p.geo),
    }


# moved-from: workflowmeta._profile_from_dict
def _profile_from_dict(d: dict[str, Any]) -> ResourceProfile:
    return ResourceProfile(
        image=d.get("image"),          # type: ignore[arg-type]
        image_digest=d.get("image_digest"),
        gpu=tuple(d.get("gpu") or ()),
        num_gpus=d.get("num_gpus", 1),
        gpu_ram_gb=d.get("gpu_ram_gb"),
        disk_gb=d.get("disk_gb"),
        rental=d.get("rental", "bid"),
        max_bid=d.get("max_bid"),
        budget_usd=d.get("budget_usd", 0.0),
        max_wall_s=d.get("max_wall_s", 0),
        geo=tuple(d.get("geo") or ()),
    )


# moved-from: workflowmeta._stage_to_dict
def _stage_to_dict(s: JobStage) -> dict[str, Any]:
    return {
        "name": s.name,
        "bundle": s.bundle,
        "profile": s.profile,
        "after": list(s.after),
        "inputs": {
            k: {"stage": v.stage, "artifact": v.artifact, "dest": v.dest}
            for k, v in sorted(s.inputs.items())
        },
        "outputs": {
            k: {"kind": v.kind, "manifest_path": v.manifest_path}
            for k, v in sorted(s.outputs.items())
        },
        "retry": {
            "max_attempts": s.retry.max_attempts,
            "retry_on": list(s.retry.retry_on),
        },
        "secrets": list(s.secrets),
    }


# moved-from: workflowmeta._stage_from_dict
def _stage_from_dict(d: dict[str, Any]) -> JobStage:
    inputs = {
        k: InputRef(stage=v.get("stage"), artifact=v.get("artifact"), dest=v.get("dest"))
        for k, v in (d.get("inputs") or {}).items()
    }
    outputs = {
        k: ArtifactContract(kind=v.get("kind"), manifest_path=v.get("manifest_path"))
        for k, v in (d.get("outputs") or {}).items()
    }
    retry_d = d.get("retry") or {}
    retry = RetryPolicy(
        max_attempts=retry_d.get("max_attempts", 1),
        retry_on=tuple(retry_d.get("retry_on") or ()),
    )
    return JobStage(
        name=d.get("name"), bundle=d.get("bundle"),  # type: ignore[arg-type]
        profile=d.get("profile"),  # type: ignore[arg-type]
        after=tuple(d.get("after") or ()), inputs=inputs, outputs=outputs,
        retry=retry, secrets=tuple(d.get("secrets") or ()),
    )


# moved-from: workflowmeta.spec_to_dict
def spec_to_dict(wf: Workflow) -> dict[str, Any]:
    """The canonical (round-trippable) dict for `workflows/<WF_ID>/spec.json`.
    Stage order is semantically meaningful (declared pipeline order) and is
    preserved as a list; `profiles` is a name-keyed mapping, sorted for a
    deterministic byte-identical JSON encoding."""
    return {
        "v": SCHEMA_VERSION,
        "version": wf.version,
        "name": wf.name,
        "budget_usd": wf.budget_usd,
        "max_wall_s": wf.max_wall_s,
        "teardown": wf.teardown,
        "profiles": {k: _profile_to_dict(v) for k, v in sorted(wf.profiles.items())},
        "stages": [_stage_to_dict(s) for s in wf.stages],
    }


# moved-from: workflowmeta.spec_from_dict
def spec_from_dict(d: dict[str, Any]) -> Workflow:
    """Reconstruct a `Workflow` from a spec dict. Tolerant of extra/unknown
    top-level and nested keys (forward-compatible readers) — only the field
    names `workflow.py`'s dataclasses know about are consumed."""
    if not isinstance(d, dict):
        raise WorkflowSpecError("workflow spec must be a JSON object")
    profiles = {
        k: _profile_from_dict(v) for k, v in (d.get("profiles") or {}).items()
    }
    stages = tuple(_stage_from_dict(s) for s in (d.get("stages") or ()))
    return Workflow(
        version=d.get("version", 1),
        name=d.get("name"),          # type: ignore[arg-type]
        budget_usd=d.get("budget_usd", 0.0),
        max_wall_s=d.get("max_wall_s", 0),
        teardown=d.get("teardown", "stop"),
        profiles=profiles,
        stages=stages,
    )


# moved-from: workflowmeta.canonical_spec_json
def canonical_spec_json(wf: Workflow) -> str:
    """Deterministic (sorted-key, compact) JSON — stable across processes for
    hashing/diffing a spec, e.g. detecting resubmission with identical bytes
    vs a hard conflict (M2-T2)."""
    return json.dumps(spec_to_dict(wf), sort_keys=True, separators=(",", ":"))


# --- spec validation (cross-object V1 rules) ----------------------------------
# moved-from: workflowmeta._check_acyclic
def _check_acyclic(stages_by_name: dict[str, JobStage]) -> list[str]:
    """DFS cycle detection over the `after` dependency graph. Returns a list
    of error strings (missing-dependency + cycle findings); never raises
    directly so callers can aggregate with the other spec-rule violations."""
    errors: list[str] = []
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in stages_by_name}

    def visit(n: str, stack: list[str]) -> None:
        color[n] = GRAY
        for dep in stages_by_name[n].after:
            if dep not in stages_by_name:
                errors.append(
                    f"stage {n!r} declares after={dep!r}, which is not a "
                    f"declared stage")
                continue
            if color[dep] == GRAY:
                cyc = "->".join(stack + [dep])
                errors.append(f"cycle in stage dependencies: {cyc}")
                continue
            if color[dep] == WHITE:
                visit(dep, stack + [dep])
        color[n] = BLACK

    for n in stages_by_name:
        if color[n] == WHITE:
            visit(n, [n])
    return errors


# moved-from: workflowmeta.validate_workflow_spec
def validate_workflow_spec(wf: Workflow, *, of_record: bool = True) -> None:
    """Enforce the V1 cross-object spec rules (roadmap "Workflow DSL"):

      * stage names unique (guaranteed by `workflow.Workflow`'s dict-shaped
        collapse being checked here for actual duplicates before collapse);
      * every stage's `profile` references a declared profile;
      * `after` is acyclic and every entry names a declared stage;
      * every `InputRef.stage` is a declared dependency (`after`) of the
        owning stage, references a real stage, and a real output of it;
      * profiles are fully pinned (`image_digest` set) when `of_record`;
      * `retry_on` is a subset of `workflow.RETRY_CLASSES`.

    Raises `WorkflowSpecError` with every violation joined, or returns None.
    """
    errors: list[str] = []

    seen_names: dict[str, JobStage] = {}
    for s in wf.stages:
        if s.name in seen_names:
            errors.append(f"duplicate stage name: {s.name!r}")
        else:
            seen_names[s.name] = s
    stages_by_name = seen_names

    for s in wf.stages:
        if s.profile not in wf.profiles:
            errors.append(
                f"stage {s.name!r} references undeclared profile {s.profile!r}")

    errors.extend(_check_acyclic(stages_by_name))

    for s in wf.stages:
        for input_name, ref in s.inputs.items():
            if ref.stage not in stages_by_name:
                errors.append(
                    f"stage {s.name!r} input {input_name!r} references "
                    f"unknown stage {ref.stage!r}")
                continue
            if ref.stage not in s.after:
                errors.append(
                    f"stage {s.name!r} input {input_name!r} references stage "
                    f"{ref.stage!r}, which is not in its declared after={s.after!r} "
                    f"(input without dependency)")
            upstream = stages_by_name[ref.stage]
            if ref.artifact not in upstream.outputs:
                errors.append(
                    f"stage {s.name!r} input {input_name!r} references "
                    f"artifact {ref.artifact!r}, which stage {ref.stage!r} "
                    f"does not declare as an output")

    if of_record:
        for name, p in wf.profiles.items():
            if not p.image_digest:
                errors.append(
                    f"profile {name!r} has no image_digest (image={p.image!r}); "
                    f"an of-record workflow must pin a digest, not a bare tag")

    for s in wf.stages:
        bad = set(s.retry.retry_on) - RETRY_CLASSES
        if bad:
            errors.append(
                f"stage {s.name!r} retry_on has values outside {sorted(RETRY_CLASSES)}: "
                f"{sorted(bad)}")

    if errors:
        raise WorkflowSpecError("; ".join(errors))


# --- artifact binding (roadmap "Artifact binding"; pure, no I/O) --------------
# jobmeta asset-name slug rule, duplicated here (not imported — this module
# never imports jobmeta, per the no-transport module boundary above).
# moved-from: workflowmeta._ASSET_NAME_RE
_ASSET_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")


# moved-from: workflowmeta.require_from_manifest
def require_from_manifest(manifest: dict[str, Any], *,
                          manifest_rel: str = "results/artifact-manifest.json") -> list[str]:
    """Derive the asset `require` list from an accepted generation manifest
    dict: the manifest path itself, followed by the sorted set of each arm's
    `path` (skipping arms with no truthy `path`).

    `manifest_rel` is WORKDIR-RELATIVE (the producing bundle's `results:`
    frame), i.e. relative to the asset's `jobs/<gen_job_id>/results` b2
    prefix — pass the upstream stage's `ArtifactContract.manifest_path`; the
    default matches the e2 bundles (which write into a local `results/`
    dir) and `jobmeta.validate_generation_artifact`'s default.

    Each arm's require glob is anchored to `manifest_rel`'s directory frame
    (`results/<basename>`), NOT the manifest's `arm['path']` verbatim. The
    generation manifest records `path` as an ABSOLUTE box path
    (`/workspace/jobs/<gen_job>/work/results/gens_*.jsonl`), but the score
    stage's `input-generate` asset pulls the `jobs/<gen_job>/results/` b2
    prefix into a cache and jobd checks each require glob RELATIVE TO THAT
    CACHE — where the arm lands at `results/gens_*.jsonl`. Using the absolute
    path never matches (live 2026-07-20, run d8e9:
    `asset_stage_failed: require globs unmatched`)."""
    arms = (manifest or {}).get("arms") or {}
    frame = manifest_rel.rsplit("/", 1)[0] if "/" in manifest_rel else ""
    def _rel(p: object) -> str:
        base = str(p).replace("\\", "/").rsplit("/", 1)[-1]
        return f"{frame}/{base}" if frame else base
    paths = {_rel(arm["path"]) for arm in arms.values() if arm.get("path")}
    return [manifest_rel] + sorted(paths)


# moved-from: workflowmeta.input_ref_asset
def input_ref_asset(ref: InputRef, *, gen_job_id: str, manifest_sha256: str,
                    require: Iterable[str]) -> dict[str, Any]:
    """Materialize an `InputRef` as the ordinary asset block (roadmap
    "Artifact binding") the score ticket appends for the generate stage's
    accepted artifact.

    The SHA-prefixed `name` is load-bearing: jobd caches an asset by NAME, not
    by its B2 source, so a fixed name could silently reuse bytes staged for a
    different workflow/attempt. `dest` stays the stable stage-local path
    (`ref.dest`) regardless of which manifest SHA is bound.
    """
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) < 12:
        raise WorkflowError("manifest_sha256 must be >=12 hex chars")
    name = f"input-{ref.stage}-{manifest_sha256[:12]}"
    if not _ASSET_NAME_RE.match(name):
        raise WorkflowError(
            f"invalid asset name {name!r}: must match {_ASSET_NAME_RE.pattern}")
    return {
        "name": name,
        "b2": f"jobs/{gen_job_id}/results",
        "dest": ref.dest,
        "mode": "copy",
        "optional": False,
        "require": list(require),
    }


# --- events --------------------------------------------------------------------
# moved-from: workflowmeta.make_event
def make_event(workflow_id: str, event: str, actor: str, *, ts: str | None = None,
               **fields: Any) -> dict[str, Any]:  # noqa: ANN401 — free-form event body
    """Build a workflow event dict with the v1 envelope. Unknown `event`
    values are allowed (the fold tolerates them) — reuses runmeta's
    `now_ts`/`nonce` for the clock/nonce discipline."""
    validate_wf_id(workflow_id)
    ev = {
        "v": SCHEMA_VERSION,
        "ts": ts or now_ts(),
        "actor": actor,
        "event": event,
        "workflow_id": workflow_id,
        "nonce": fields.pop("nonce", None) or nonce(),
    }
    ev.update({k: v for k, v in fields.items() if v is not None})
    return ev


# moved-from: workflowmeta._coerce
def _coerce(raw: object) -> dict[str, Any] | None:
    """Return a valid workflow event dict, or None (counted as a parse
    error). Mirrors runmeta._coerce's tolerance: one bad object never breaks
    a fold."""
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return None
    if not isinstance(raw, dict):
        return None
    if not all(raw.get(k) for k in _CORE_KEYS):
        return None
    return raw


# moved-from: workflowmeta._num
def _num(x: object) -> Any:  # noqa: ANN401 — returns its argument UNCHANGED (int stays int)
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


# --- ts parsing (pure; runmeta.now_ts format: YYYYMMDDTHHMMSSmmmZ) -----------
# moved-from: workflowmeta._TS_RE
_TS_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})(\d{3})Z$")


# moved-from: workflowmeta._parse_ts
def _parse_ts(ts: str) -> datetime.datetime:
    m = _TS_RE.match(ts or "")
    if not m:
        raise WorkflowIdError(f"not a valid runmeta timestamp: {ts!r}")
    y, mo, d, h, mi, s, ms = (int(g) for g in m.groups())
    return datetime.datetime(y, mo, d, h, mi, s, ms * 1000, tzinfo=datetime.timezone.utc)


# moved-from: workflowmeta._ts_diff_seconds
def _ts_diff_seconds(later: str, earlier: str) -> float:
    return (_parse_ts(later) - _parse_ts(earlier)).total_seconds()


# --- the fold (PURE — no I/O) --------------------------------------------------
# moved-from: workflowmeta.fold_workflow_events
def fold_workflow_events(raw_events: Iterable[object]) -> dict[str, Any]:
    """Fold an unordered multiset of workflow events into a view:

      * top-level workflow status (submitted/running/succeeded/failed/
        cancelled/unknown) by TERMINAL PRECEDENCE, never last-event-wins;
      * a controller sub-view (current actor, started/heartbeat timestamps);
      * a per-stage sub-view keyed by stage name: latest attempt number,
        that attempt's status (also by the stage terminal precedence), and
        the most recently known job_id/instance_id for it.

    Tolerant of unknown event names/fields (I1-style: one bad or unknown
    object is counted, never dropped in a way that breaks the fold; unknown
    EVENT NAMES fold as inert/no-op contributions beyond the raw counters).
    """
    evs: list[dict[str, Any]] = []
    parse_errors, unknown_events = 0, 0
    for r in raw_events:
        e = _coerce(r)
        if e is None:
            parse_errors += 1
            continue
        evs.append(e)
        if e.get("event") not in EVENTS:
            unknown_events += 1

    evs.sort(key=lambda e: (e.get("ts", ""), e.get("nonce", "")))

    view: dict[str, Any] = {
        "workflow_id": evs[-1]["workflow_id"] if evs else None,
        "status": "unknown",
        "terminal": False,
        "failure_class": None,
        "n_events": len(evs),
        "parse_errors": parse_errors,
        "unknown_events": unknown_events,
        "controller": {
            "actor": None, "started_ts": None, "last_heartbeat_ts": None,
        },
        "stages": {},
    }
    if not evs:
        return view

    # --- controller sub-fold (any actor emitting a controller_* event) -----
    ctrl_evs = [e for e in evs if e.get("event") in CONTROLLER_EVENTS]
    if ctrl_evs:
        latest = ctrl_evs[-1]
        view["controller"]["actor"] = latest.get("actor")
        view["controller"]["last_heartbeat_ts"] = latest.get("ts")
        started = [e for e in evs if e.get("event") in ("controller_started", "takeover")]
        if started:
            view["controller"]["started_ts"] = started[-1].get("ts")

    # --- per-stage-attempt sub-fold -----------------------------------------
    stage_evs: dict[str, list[dict[str, Any]]] = {}
    for e in evs:
        stage = e.get("stage")
        if not stage:
            continue
        stage_evs.setdefault(stage, []).append(e)

    for stage, s_evs in stage_evs.items():
        attempts = sorted({
            _num(e.get("attempt")) for e in s_evs
            if _num(e.get("attempt")) is not None
        })
        cur_attempt = attempts[-1] if attempts else None
        cur_evs = [e for e in s_evs if _num(e.get("attempt")) == cur_attempt] \
            if cur_attempt is not None else s_evs

        terminal_evs = [e for e in cur_evs if e.get("event") in STAGE_TERMINAL]
        status = None
        stage_failure_class = None
        if terminal_evs:
            best = max(terminal_evs,
                       key=lambda e: _STAGE_TERMINAL_RANK.get(e.get("event"), 0))  # type: ignore[arg-type]
            status = best.get("event")
            stage_failure_class = best.get("failure_class")
        else:
            for name in ("stage_started", "stage_submitted", "box_acquired", "stage_planned"):
                if any(e.get("event") == name for e in cur_evs):
                    status = name
                    break

        job_id = None
        instance_id = None
        box_acquired_ts = None
        for e in cur_evs:                       # cur_evs preserves evs' ts sort
            if e.get("job_id"):
                job_id = e.get("job_id")
            if e.get("instance_id"):
                instance_id = e.get("instance_id")
            if e.get("event") == "box_acquired" and e.get("ts"):
                box_acquired_ts = e.get("ts")   # latest box_acquired for this attempt

        view["stages"][stage] = {
            "status": status,
            "attempt": cur_attempt,
            "attempts_seen": len(attempts) if attempts else (1 if s_evs else 0),
            "job_id": job_id,
            "instance_id": instance_id,
            "box_acquired_ts": box_acquired_ts,
            "failure_class": stage_failure_class,
        }

    # --- top-level workflow status: terminal precedence, then activity -----
    terms = [e for e in evs if e.get("event") in WORKFLOW_TERMINAL]
    if terms:
        best = max(
            terms,
            key=lambda e: _WORKFLOW_TERMINAL_RANK.get(e.get("event"), 0))  # type: ignore[arg-type]
        ev_name = best.get("event")
        view["terminal"] = True
        view["failure_class"] = best.get("failure_class")
        view["status"] = {
            "workflow_succeeded": "succeeded",
            "workflow_failed": "failed",
            "workflow_cancelled": "cancelled",
        }[ev_name]  # type: ignore[index]
    elif any(e.get("stage") for e in evs) or any(
            e.get("event") in ("teardown_started", "box_released") for e in evs):
        view["status"] = "running"
    elif any(e.get("event") == "submitted" for e in evs):
        view["status"] = "submitted"
    else:
        view["status"] = "unknown"

    return view


# --- ready-stage selection / retry decisions -----------------------------------
# moved-from: workflowmeta.ready_stages
def ready_stages(wf: Workflow, view: dict[str, Any]) -> list[str]:
    """Stages with no attempt yet whose every `after` dependency has
    SUCCEEDED, in declared order. The controller submits `ready_stages[0]`
    (roadmap step 6: "the first ready stage in declared order"); a stage
    whose dependency failed/was cancelled never becomes ready (the
    "failed dependency" case) — it stays out of this list forever unless the
    failed upstream is itself later retried to success.
    """
    stages_view = view.get("stages", {})
    out: list[str] = []
    for s in wf.stages:
        sv = stages_view.get(s.name)
        if sv is not None and sv.get("status") is not None:
            continue          # already planned/attempted at least once
        deps_ok = all(
            stages_view.get(dep, {}).get("status") == "stage_succeeded"
            for dep in s.after
        )
        if deps_ok:
            out.append(s.name)
    return out


# moved-from: workflowmeta.next_ready_stage
def next_ready_stage(wf: Workflow, view: dict[str, Any]) -> str | None:
    rs = ready_stages(wf, view)
    return rs[0] if rs else None


# moved-from: workflowmeta.decide_retry
def decide_retry(stage: JobStage, *, attempts_used: int, failure_class: str) -> str:
    """Pure retry decision for a terminal-failed stage attempt. Returns
    `"retry"` when `failure_class` is in the stage's `retry_on` set AND
    another attempt remains under `max_attempts`; else `"fail"`. A brand new
    attempt gets a new deterministic JOB_ID (`stage_job_id(..., attempt+1)`)
    only on `"retry"` — this function makes no I/O decision, only the
    allow/deny call."""
    if attempts_used < stage.retry.max_attempts and failure_class in stage.retry.retry_on:
        return "retry"
    return "fail"


# moved-from: workflowmeta.controller_is_stale
def controller_is_stale(view: dict[str, Any], *, now: str, stale_after_s: float) -> bool:
    """Pure staleness check for the controller heartbeat (M2-T3's
    `--takeover` needs "heartbeat older than three poll intervals" — the
    interval math is the CLI's concern; this is the pure age comparison).
    No heartbeat ever recorded is treated as stale (safe to take over)."""
    last = view.get("controller", {}).get("last_heartbeat_ts")
    if not last:
        return True
    # `now` is the caller's own clock (always runmeta now_ts()); a malformed
    # value there is a caller bug and must surface. The recorded heartbeat ts,
    # by contrast, came from an event the fold tolerated (`_coerce` checks only
    # that ts is truthy, not well-formed): an unparseable one cannot establish
    # liveness, so treat it as stale — safe to take over — rather than let a
    # single corrupt event crash the takeover decision path.
    now_dt = _parse_ts(now)
    try:
        last_dt = _parse_ts(last)
    except WorkflowIdError:
        return True
    return (now_dt - last_dt).total_seconds() > stale_after_s


# --- status table (roadmap M4-T3 "concise stage table"; pure formatting) ------
# moved-from: workflowmeta.status_table_rows
def status_table_rows(view: dict[str, Any], *,
                      extras: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """One row per stage in `view['stages']` (insertion order preserved),
    WITHOUT reading raw logs. `extras` is an optional caller-supplied dict of
    higher-layer (I/O-derived) per-stage figures keyed
    `extras['stages'][stage] = {'progress', 'spend_usd', 'checkpoint_age_s'}`
    — this function stays pure and merely reads it; a missing/partial
    `extras` yields `None` cells rather than raising."""
    extras = extras or {}
    stage_extras = extras.get("stages") or {}
    rows: list[dict[str, Any]] = []
    for name, sv in (view.get("stages") or {}).items():
        sv = sv or {}
        e = stage_extras.get(name) or {}
        rows.append({
            "stage": name,
            "state": sv.get("status"),
            "attempt": sv.get("attempt"),
            "job": sv.get("job_id"),
            "box": sv.get("instance_id"),
            "failure": sv.get("failure_class"),
            "progress": e.get("progress"),
            "spend": e.get("spend_usd"),
            "checkpoint_age": e.get("checkpoint_age_s"),
        })
    return rows


# moved-from: workflowmeta._cell
def _cell(v: object) -> str:
    return "-" if v is None else str(v)


# moved-from: workflowmeta.format_status_table
def format_status_table(view: dict[str, Any], *,
                        extras: dict[str, Any] | None = None) -> str:
    """Render `status_table_rows(view, extras=extras)` as a concise, fixed-
    width text table (roadmap M4-T3: "print a concise stage table ... WITHOUT
    reading raw logs") preceded by a one-line workflow header. Dependency-
    free (no I/O, no third-party table lib) so it stays importable/testable
    without a terminal or network."""
    extras = extras or {}
    header = (
        f"workflow={view.get('workflow_id')} "
        f"status={_cell(view.get('status'))} "
        f"terminal={view.get('terminal')} "
        f"spend={_cell(extras.get('spend_usd'))} "
        f"budget={_cell(extras.get('budget_usd'))}"
    )

    cols = ["STAGE", "STATE", "ATTEMPT", "JOB", "BOX",
            "PROGRESS", "SPEND", "CKPT_AGE", "FAILURE"]
    keys = ["stage", "state", "attempt", "job", "box",
            "progress", "spend", "checkpoint_age", "failure"]
    rows = status_table_rows(view, extras=extras)
    str_rows = [[_cell(r.get(k)) for k in keys] for r in rows]

    widths = [len(c) for c in cols]
    for r in str_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [header, fmt_row(cols)]
    lines.extend(fmt_row(r) for r in str_rows)
    return "\n".join(lines)
