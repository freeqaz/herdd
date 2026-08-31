"""`vastlib.workflows.spec` — the immutable V1 workflow DSL (ported `workflow.py`).

Why this exists
---------------
A `Workflow` is a small, pinned, multi-stage job pipeline: a DAG of `JobStage`s,
each running one existing `jobmeta` bundle on a pinned `ResourceProfile`, wired
together by `InputRef`/`ArtifactContract` so a later stage can consume an
earlier stage's accepted output. V1 deliberately has NO `LocalStage` — every
stage is a box job.

This module holds ONLY the dataclasses + the per-field shape checks a dataclass
cannot avoid (empty/wrong-typed fields). Cross-object spec rules — unique names,
acyclic `after`, `InputRef` pointing at a declared dependency, profile pinning
for an of-record spec, the `retry_on` vocabulary, canonical JSON, IDs, events,
the pure fold, ready-stage selection, retry decisions, and terminal precedence —
live in the sibling `workflowmeta` module, because those rules must apply
equally to a spec authored in Python and one round-tripped from
`workflows/<WF_ID>/spec.json` on B2.

No network/API/subprocess here — construction is pure and instantaneous. It is
the bottom of the `workflows` ring: it imports nothing from `vastlib` and must
keep it that way (`workflowmeta` and `workflowctl` sit above it).

THIS FILE OWNS THE CLASSES — and the bare name still resolves to them
---------------------------------------------------------------------
Since plan step 7, `tools/vast/workflow.py` is a pure re-export shim over this
module, so `workflow.Workflow is spec.Workflow` and there is exactly ONE class
object per name. The hazard that shaped both files is worth keeping in view,
because reintroducing a body in the flat file brings it straight back
(`workflows-core.json` H1):

* Every authored workflow spec on disk says `from workflow import Workflow, ...`
  — a BARE-NAME import resolved off `sys.path`. In-repo example:
  `tools/witness/workflows/e2-paired/workflow.py:60-63`. Out-of-repo copies
  exist and cannot be grepped.
* `ctl.load_workflow_module` path-loads that authored file
  (`importlib.util.spec_from_file_location`) and then `isinstance(wf, Workflow)`
  against **its own** `Workflow` class object (`vastlib/workflows/ctl.py:412`).
* If the loader's class came from `vastlib.workflows.spec` while the authored
  file's came from a flat `workflow.py` with a BODY of its own, `isinstance`
  would see TWO DIFFERENT CLASS OBJECTS and every spec would fail to load with
  the maximally misleading "must define a module-level WORKFLOW". The shim's
  re-export is the only thing standing between here and that.

The alias lives in the SHIM, never here. `sys.modules.setdefault("workflow",
...)` executed from inside a `vastlib` import would hijack the bare name for the
whole process at import time — a Zone P module has no business having that side
effect, and it would break any consumer that legitimately wants the flat file.
This copy stays inert with respect to `sys.modules` and `sys.path`, and
`test_vastlib_workflows_spec.py` pins both halves: that the split is CLOSED
(one class object through either name) and that importing this module does not
hijack the bare name.

The shim recipe, as landed at step 7 (do not improvise a replacement)
---------------------------------------------------------------------
`tools/vast/workflow.py` is a re-export shim, not a deletion:

    # tools/vast/workflow.py  (Zone E-adjacent compatibility shim)
    from vastlib.workflows.spec import (          # noqa: F401  re-export
        RENTAL_CHOICES, RETRY_CLASSES, STAGE_NAME_RE, TEARDOWN_CHOICES,
        ArtifactContract, InputRef, JobStage, ResourceProfile, RetryPolicy,
        Workflow, WorkflowError, _non_negative, _slug, _tuple_of_str,
    )
    import sys, vastlib.workflows.spec as _spec
    # An authored spec resolving the bare name must get the SAME module object
    # the controller imported, or isinstance() sees two classes.
    sys.modules.setdefault("workflow", sys.modules[__name__])
    sys.modules.setdefault("vastlib.workflows.spec", _spec)

`ctl.py` and `meta.py` switched their `from workflow import ...` to
`vastlib.workflows.spec` in the SAME commit. Because the shim only re-exports,
both names resolve to one set of class objects and `isinstance` holds from
either side — the repoint is hygiene, and doing it WITHOUT the shim is the one
ordering that breaks every authored spec. (`jobmatrix.py:72`'s `sys.modules.setdefault` is the
in-tree precedent for the aliasing half — it exists for exactly this reason.)

**Plain deletion of `tools/vast/workflow.py` is UNSAFE — permanently, not just
for one release.** The bare name is a frozen AUTHORING contract, not an
implementation detail: every spec file ever written (including ones outside
this repo, and the generated fixture strings in `test_workflow.py:3157` and
`test_workflow_preflight.py:90-92`) contains the literal `from workflow import`.
Deleting the flat file breaks all of them with an ImportError at path-load time,
and there is no deprecation window that finds specs living on someone's laptop.
The shim is the terminal state. Escalate to the owner before proposing removal.

What is deliberately NOT here
-----------------------------
* No cross-object validation (`workflowmeta` owns it, so the rules bind equally
  on B2-round-tripped JSON specs).
* No `sys.path` / `sys.modules` manipulation — Zone P forbids the first
  outright (plan §3) and the second belongs in the flat shim, above.
* No `LocalStage`, no scheduling, no I/O. Construction stays pure.

Provenance: verbatim port of `tools/vast/workflow.py` @ `ea8360dc`, plan §8
step 5, manifest `workflows-core.json`. Only the annotations required by Zone P
strict typing (plan §6) were added; no check, message, or default changed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple


# moved-from: workflow.WorkflowError
class WorkflowError(ValueError):
    """A dataclass was constructed with a shape violation."""


# Stage / profile / artifact names are embedded raw into B2 keys and into the
# deterministic stage-attempt JOB_ID (workflowmeta.stage_job_id), so they are
# bounded, lowercase slugs — never arbitrary strings.
# moved-from: workflow.STAGE_NAME_RE
STAGE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")

# V1 retry-class vocabulary (frozen; see roadmap "Failure classes and exit
# codes" / "Workflow DSL"). Shared with workflowmeta's retry_on subset check.
# FROZEN BY VALUE: workflowmeta does the subset check against it and specs
# round-trip through B2 JSON with no schema negotiation.
# moved-from: workflow.RETRY_CLASSES
RETRY_CLASSES = frozenset({"infrastructure", "entrypoint", "postcondition"})

# V1 teardown vocabulary (frozen).
# moved-from: workflow.TEARDOWN_CHOICES
TEARDOWN_CHOICES = ("stop", "destroy")

# V1 rental vocabulary (frozen). THE HYPHEN IS LOAD-BEARING: `vastlib.market.
# offers.pick_offer` accepts the vast-native "ondemand" AND this module's
# "on-demand" and normalizes both (offers.py:447-456, pinned by
# test_vastlib_market.py:206). The coupling is by VALUE — there is no import
# edge, and there must not be one: market sits BELOW workflows in the ring DAG,
# so an import would invert it. Never normalize this to "on_demand"/"ondemand".
# moved-from: workflow.RENTAL_CHOICES
RENTAL_CHOICES = ("bid", "on-demand")


# moved-from: workflow._slug
def _slug(value: Any, field_name: str) -> str:  # noqa: ANN401 — validates arbitrary user input
    if not isinstance(value, str) or not STAGE_NAME_RE.match(value):
        raise WorkflowError(
            f"{field_name} must be a lowercase slug matching "
            f"{STAGE_NAME_RE.pattern} (<=32 chars), got {value!r}")
    return value


# moved-from: workflow._tuple_of_str
def _tuple_of_str(value: Any, field_name: str) -> Tuple[str, ...]:  # noqa: ANN401 — same
    if isinstance(value, (str, bytes)):
        raise WorkflowError(
            f"{field_name} must be a sequence of str, not a bare str/bytes")
    try:
        items = tuple(value)
    except TypeError:
        raise WorkflowError(f"{field_name} must be an iterable of str")
    for it in items:
        if not isinstance(it, str) or not it:
            raise WorkflowError(f"{field_name} entries must be non-empty str: {it!r}")
    return items


# moved-from: workflow._non_negative
def _non_negative(value: Any, field_name: str) -> Any:  # noqa: ANN401 — passes the value through
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise WorkflowError(f"{field_name} must be a non-negative number, got {value!r}")
    return value


# moved-from: workflow.ResourceProfile
@dataclass(frozen=True)
class ResourceProfile:
    """A pinned rental shape. `image_digest` is required to submit an
    of-record workflow (workflowmeta.validate_workflow_spec enforces this) —
    a bare tag can drift under you between planning and launch."""

    image: str
    image_digest: Optional[str] = None
    gpu: Tuple[str, ...] = ()
    num_gpus: int = 1
    gpu_ram_gb: Optional[int] = None
    disk_gb: Optional[int] = None
    rental: str = "bid"
    max_bid: Optional[float] = None
    budget_usd: float = 0.0
    max_wall_s: int = 0
    # Optional 2-letter host-country allowlist, e.g. ("US",). EMPTY IS THE
    # RECOMMENDED VALUE (owner directive 2026-08-05).
    #
    # History, because the reversal matters: 2026-07-20 pinned the e2-paired
    # profiles to ("US",) after run 5819 — far-flung cheapest offers paired
    # shaped per-flow bandwidth (47-min image pulls -> boot-deadline kills) with
    # contested spot floors. That pin was a PROXY for "a host that can pull the
    # image fast", chosen because nothing measured it directly. Now something
    # does: every auto-pick carries `inet_down >= LAUNCH_INET_DOWN_MBPS`
    # (1000 Mb/s, landed 2026-08-03), the image is small and per-flow-layered
    # (t211), and the 600 s boot SLA rehosts a host that slips through. Set this
    # only for a REAL geography constraint (B2 locality, data residency) — a
    # narrower pool also means a thinner, more contested spot market, which is
    # what the pin was supposed to avoid.
    # Supersession: docs/plans/witness/g2_push/FLEETD_AUTOREPLACE_2026-08-05.md
    geo: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.image, str) or not self.image:
            raise WorkflowError("ResourceProfile.image is required")
        if self.image_digest is not None and not isinstance(self.image_digest, str):
            raise WorkflowError("ResourceProfile.image_digest must be str or None")
        object.__setattr__(self, "gpu", _tuple_of_str(self.gpu, "ResourceProfile.gpu"))
        object.__setattr__(self, "geo", _tuple_of_str(self.geo, "ResourceProfile.geo"))
        if self.rental not in RENTAL_CHOICES:
            raise WorkflowError(
                f"ResourceProfile.rental must be one of {RENTAL_CHOICES}, got {self.rental!r}")
        if not isinstance(self.num_gpus, int) or isinstance(self.num_gpus, bool) \
                or self.num_gpus < 1:
            raise WorkflowError("ResourceProfile.num_gpus must be an int >= 1")
        _non_negative(self.budget_usd, "ResourceProfile.budget_usd")
        _non_negative(self.max_wall_s, "ResourceProfile.max_wall_s")
        if self.max_bid is not None:
            _non_negative(self.max_bid, "ResourceProfile.max_bid")
        if self.gpu_ram_gb is not None:
            _non_negative(self.gpu_ram_gb, "ResourceProfile.gpu_ram_gb")
        if self.disk_gb is not None:
            _non_negative(self.disk_gb, "ResourceProfile.disk_gb")


# moved-from: workflow.ArtifactContract
@dataclass(frozen=True)
class ArtifactContract:
    """A named output a stage promises: an artifact-manifest `kind` plus the
    in-results path of its manifest (matches the E2 manifest convention)."""

    kind: str
    manifest_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise WorkflowError("ArtifactContract.kind is required")
        if not isinstance(self.manifest_path, str) or not self.manifest_path:
            raise WorkflowError("ArtifactContract.manifest_path is required")
        # manifest_path is interpolated RAW into the jobs/<job_id>/results/
        # frame (jobmeta.validate_generation_artifact) and joined under
        # locally-captured rehearsal results dirs -- a '..' segment or an
        # absolute path escapes that frame entirely. This check also binds
        # on specs round-tripped from B2 JSON (workflowmeta.spec_from_dict
        # reconstructs ArtifactContract, so a foreign spec.json cannot smuggle
        # an escaping path past it).
        if self.manifest_path.startswith("/") or \
                ".." in self.manifest_path.split("/"):
            raise WorkflowError(
                f"ArtifactContract.manifest_path must be a relative path with "
                f"no '..' segments (it must stay inside the results frame), "
                f"got {self.manifest_path!r}")


# moved-from: workflow.InputRef
@dataclass(frozen=True)
class InputRef:
    """A stage's reference to an upstream stage's accepted artifact. The
    referenced `stage` must be a declared dependency (`after`) of the owning
    stage — workflowmeta.validate_workflow_spec enforces this cross-object
    rule (the "input without dependency" failure mode)."""

    stage: str
    artifact: str
    dest: str

    def __post_init__(self) -> None:
        for f in ("stage", "artifact", "dest"):
            v = getattr(self, f)
            if not isinstance(v, str) or not v:
                raise WorkflowError(f"InputRef.{f} is required")


# moved-from: workflow.RetryPolicy
@dataclass(frozen=True)
class RetryPolicy:
    """`retry_on` values are checked against `RETRY_CLASSES` only for their
    TYPE here (non-empty strings); the controlled-vocabulary subset check is
    a spec-validation rule in workflowmeta (applies uniformly to specs
    reloaded from JSON, where a stale/foreign value could appear)."""

    max_attempts: int = 1
    retry_on: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.max_attempts, int) or isinstance(self.max_attempts, bool) \
                or self.max_attempts < 1:
            raise WorkflowError("RetryPolicy.max_attempts must be an int >= 1")
        object.__setattr__(
            self, "retry_on", _tuple_of_str(self.retry_on, "RetryPolicy.retry_on"))


# moved-from: workflow.JobStage
@dataclass(frozen=True)
class JobStage:
    """One DAG node: run `bundle` (an existing jobmeta bundle directory) on
    `profile`, after every stage named in `after`, materializing `inputs` from
    upstream accepted artifacts and promising `outputs`.

    `secrets` names credentials the entrypoint needs injected at run time by
    NAME only (e.g. a B2 write-key name) — values never enter this spec or
    its canonical JSON; that is enforced structurally by the field being a
    tuple of str, never a mapping."""

    name: str
    bundle: str
    profile: str
    after: Tuple[str, ...] = ()
    inputs: Mapping[str, InputRef] = field(default_factory=dict)
    outputs: Mapping[str, ArtifactContract] = field(default_factory=dict)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    secrets: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _slug(self.name, "JobStage.name")
        if not isinstance(self.bundle, str) or not self.bundle:
            raise WorkflowError("JobStage.bundle is required")
        _slug(self.profile, "JobStage.profile")
        object.__setattr__(self, "after", _tuple_of_str(self.after, "JobStage.after"))
        for dep in self.after:
            _slug(dep, "JobStage.after entry")

        inputs = dict(self.inputs or {})
        for k, v in inputs.items():
            _slug(k, "JobStage.inputs key")
            if not isinstance(v, InputRef):
                raise WorkflowError(f"JobStage.inputs[{k!r}] must be an InputRef")
        object.__setattr__(self, "inputs", inputs)

        outputs = dict(self.outputs or {})
        # `out_v`, not `v`: the flat copy reuses the `v` of the inputs loop
        # above, which strict mypy reads as rebinding an `InputRef` local to an
        # `ArtifactContract`. Loop-local rename only — same iteration, same
        # checks, same messages (the name never reaches an error string).
        for k, out_v in outputs.items():
            _slug(k, "JobStage.outputs key")
            if not isinstance(out_v, ArtifactContract):
                raise WorkflowError(f"JobStage.outputs[{k!r}] must be an ArtifactContract")
        object.__setattr__(self, "outputs", outputs)

        if not isinstance(self.retry, RetryPolicy):
            raise WorkflowError("JobStage.retry must be a RetryPolicy")

        object.__setattr__(
            self, "secrets", _tuple_of_str(self.secrets, "JobStage.secrets"))
        for s in self.secrets:
            if "=" in s or any(c.isspace() for c in s):
                raise WorkflowError(
                    f"JobStage.secrets entries are NAMES only, not KEY=VALUE "
                    f"pairs or values: {s!r}")


# moved-from: workflow.Workflow
@dataclass(frozen=True)
class Workflow:
    """The whole spec: one module-level `WORKFLOW` per roadmap convention."""

    version: int
    name: str
    budget_usd: float
    max_wall_s: int
    teardown: str
    profiles: Mapping[str, ResourceProfile]
    stages: Tuple[JobStage, ...]

    def __post_init__(self) -> None:
        if self.version != 1:
            raise WorkflowError(f"Workflow.version must be 1, got {self.version!r}")
        if not isinstance(self.name, str) or not self.name:
            raise WorkflowError("Workflow.name is required")
        _non_negative(self.budget_usd, "Workflow.budget_usd")
        _non_negative(self.max_wall_s, "Workflow.max_wall_s")
        if self.teardown not in TEARDOWN_CHOICES:
            raise WorkflowError(
                f"Workflow.teardown must be one of {TEARDOWN_CHOICES}, got {self.teardown!r}")

        profiles = dict(self.profiles or {})
        for k, v in profiles.items():
            _slug(k, "Workflow.profiles key")
            if not isinstance(v, ResourceProfile):
                raise WorkflowError(f"Workflow.profiles[{k!r}] must be a ResourceProfile")
        object.__setattr__(self, "profiles", profiles)

        stages = tuple(self.stages or ())
        for s in stages:
            if not isinstance(s, JobStage):
                raise WorkflowError("Workflow.stages entries must be JobStage")
        object.__setattr__(self, "stages", stages)
