"""`vastlib.workflows.ctl` — the workflow controller's transport + reconcile layer.

Why this exists
---------------
This is the module the whole package was rearranged around. `herdd.py` and
`workflowctl.py` imported each other at module top level — a real cycle, latent
only because neither dereferenced the other at import time — and 37 of those 38
edges pointed THIS way: the controller reaching into the CLI for a box
primitive. Living here, one ring above `core`/`boxes`/`market`/`launch` and
below `cli`, breaks the cycle **by direction**: every one of those 27 names now
has a `vastlib` home at or below this tier, each is called in module-attribute
form, and none of them resolves from `cli`. If a future edit makes one of them
resolve from `cli`, the cycle re-forms as a layer violation and `lint-imports`
says so.

What this module actually does: load a `Workflow` from an authored `.py`,
write/read `spec.json`, emit/read/fold the append-only event log, hold the local
flock + the remote controller claim, and perform AT MOST ONE idempotent
reconcile action per tick in the frozen 8-step order. Every PURE decision it
drives — spec validation, canonical JSON, the fold, ready-stage selection, retry
policy, terminal precedence, heartbeat staleness — belongs to the sibling
`meta` module and is never re-implemented here.

Five designs here that look like waste and are not
--------------------------------------------------
1. **`build_box_observer` does not call `lifecycle._instances_soft`.** That
   primitive returns `[]` on an API error, which would read every owned box as
   `gone`, trigger a false retarget, and DOUBLE SPEND. It reads through
   `api.request_soft` so a failure is visible as `unknown`, and `gone` requires
   a SECOND independent HEALTHY read to also omit the box.
2. **`LiveCostObserver._snapshot` keeps the PRIOR snapshot on a read failure**
   rather than zeroing `present`, which would silently drop accrual.
3. **Idempotency is B2 OBJECT EXISTENCE, never memory.** `verdict.json`'s
   existence is `_reconcile_completion`'s "has step 7 happened" check;
   `read_accepted_artifact` is `_accept_stage_artifacts`' check; `write_spec`'s
   byte-comparison is the resubmit check; `_teardown_attempts_seen` and
   `folded_spend` RE-READ the whole event log on every call. Memoizing any of
   them "for efficiency" breaks restart durability in a way no test that never
   restarts a controller can see.
4. **`_owned_instance_ids` `str()`-normalizes.** An adopted box records a str id
   and a fresh launch records an int; `sorted({str, int})` crashed the live
   controller on every cost tick (run 2ed9, 2026-07-30) and the same mix
   double-counts one box as two cost keys.
5. **`_prior_stage_machines` re-lists the whole event log.** It is the
   host-rotation seed for a retry, and it must see what a PRIOR attempt did, not
   what this process remembers.

What is deliberately NOT here
-----------------------------
* **Pure decisions.** They are `meta`'s (see above).
* **A second box-acquisition path.** `reconcile_active_box` REUSES
  `build_box_resolver`'s adopt/launch primitives rather than forking one.
* **`b2_mint_key`.** The credential seam is an injected `cred_provider`; this
  module never imports the minter.
* **`box_cost` / `teardown_attempt` in `meta.EVENTS`.** They are ctl-LOCAL
  events; that frozen V1 set is `meta`'s alone to grow, and the fold already
  tolerates unknown names as inert.
* **Any `sys.path` manipulation.** Forbidden inside Zone P (plan §3). The flat
  file's `sys.path.insert(0, _HERE)` is dropped; the Zone S bare names below
  resolve because every entry script puts `tools/vast` on the path.

The seven import-time env reads STAY module constants
------------------------------------------------------
`BOOT_DEADLINE_S`, `JOB_HEARTBEAT_STALE_S`, `BOOT_MIN_MBPS` and
`IMAGE_GATE_ENFORCE` read `os.environ` at IMPORT time, and `_lock_dir` /
`_instance_image_verdict` read `XDG_CACHE_HOME` at call time (the registry-host
question moved to `imageref.is_our_registry`, 2026-08-21).
Routing the first four through `core.config` would change the test idiom
(a test must monkeypatch the MODULE ATTRIBUTE, which is what
`test_workflow.py` does for `IMAGE_GATE_ENFORCE`), and behavior change is out of
scope for this port. Noted for the later knob unification, not done here.

The DSL import is `vastlib.workflows.spec` — and that is now safe
-----------------------------------------------------------------
`load_workflow_module` path-loads an AUTHORED spec file whose own
`from workflow import Workflow` resolves the flat `tools/vast/workflow.py` off
`sys.path`, then `isinstance()`-checks the result against the class THIS module
imported. Both sides must resolve the same class object or every spec fails to
load with the maximally misleading "WORKFLOW is a Workflow, not a Workflow" —
which is why this file kept the BARE name `workflow` until step 7. At step 7
the flat file became a pure re-export shim over `spec`, so
`workflow.Workflow is spec.Workflow` and the import below states the truth
instead of routing through Zone E. **Do not reintroduce a second definition of
these classes anywhere** — the identity, not the spelling, is the contract.

Provenance: `tools/vast/workflowctl.py`, moved in plan §8 step 5,
behavior-preserving. The only non-textual changes are the dropped `sys.path`
mutation, the `herdd.*` -> `vastlib.*` repointing listed above, the two
`__file__`-anchored path constants (recomputed depth, pinned by a test), and the
type annotations mypy strict requires.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from typing import IO, Any, Callable, Protocol

import imageref

from vastlib.boxes import health, lifecycle, ssh
from vastlib.core import api, config, models
from vastlib.fleet import client as fleet_client
from vastlib.jobs import bundle, risk
from vastlib.jobs import view as jobs_view
from vastlib.launch import launch
from vastlib.market import offers
from vastlib.supervise import run_lane
from vastlib.workflows import meta as wm
from vastlib.workflows.spec import JobStage, ResourceProfile, Workflow

import bidpolicy
import jobmeta
import runmeta

# WHERE THE 27 CYCLE EDGES WENT (the rename table reads the `moved-from:`
# markers; this is the same map for a human, OLD name on the left):
#
#   stop_box / destroy_box / launch_instance     -> boxes.lifecycle
#   find_matching_instance                       -> boxes.lifecycle
#   _get_instance_soft                           -> boxes.health
#   build_throughput_observer                    -> boxes.health
#   pub_key_text / ssh_authorized_keys_snippet   -> boxes.ssh
#   request_soft                                 -> core.api
#   _instance_image / _instance_env / _num_dph   -> core.models
#   vastconf.DISK_DEFAULT_WORKFLOW_GB            -> core.config
#   pick_cheapest_offer                          -> market.offers
#   image_login_arg / hf_token_text              -> launch.launch
#   hf_login_snippet                             -> launch.launch
#   compose_jobs_launch_env                      -> jobs.bundle
#   _stage_jobd_bootstrap                        -> jobs.bundle
#   _job_progress                                -> jobs.view
#   _ckpt_watchdog_alarm                         -> jobs.risk
#   _accrue_cost                                 -> supervise.run_lane
#   fleet_watch_best_effort                      -> fleet.client
#   LIVE_STATES / BID_TARGET_MULT                -> bidpolicy (Zone S: herdd only
#                                                   ever re-exported these)
#   IMAGE_DIGEST_ENV / image_tag_digest /
#   image_ref_digest                             -> imageref (same: re-exports)
#
# `jobs`, `fleet` and `supervise` are the SAME ring as `workflows`, and the §5
# contract joins that ring with `:` (siblings may import each other), so those
# four edges are layer-legal. Everything else points strictly downward.
#
# Five more `herdd.` names appear in the prose below and are NOT edges — the
# docstrings cite them as roads deliberately NOT taken (`_instances_soft`,
# `_put_state_soft`, `_emit_cost`, `_repo_root`) or as a sibling's own
# implementation. A grep-based port would have moved them; the manifest's
# tokenizer count (37 dereferences over 27 names) is the one to trust.

# --- seam types ---------------------------------------------------------------
# Every transport in this module is INJECTABLE and every default is a module
# attribute, which is what keeps `monkeypatch.setattr` steering a caller. The
# aliases below name the two shapes; they deliberately do not narrow the
# call signature, because several seams are called with different keyword sets
# by design (`runner` takes `input=`, `box_resolver` takes three positionals)
# and a narrower type would force a cast at every fake in the test suite.
Runner = Callable[..., Any]        # runner(args, input=None) -> (rc, stdout, stderr)
Seam = Callable[..., Any]          # any other injected transport/decision seam


class CredProvider(Protocol):
    """The injected credential seam (`cred_provider`), typed structurally.

    `None` on every production `reconcile_tick` today; production wiring is a
    caller closure over `b2_mint_key.mint`/`mint_pair`. This module never
    imports the minter — it only ever calls these two methods, and both return
    an epoch that `_check_credential_horizon` re-validates with `isinstance`
    before trusting (a provider that answers with junk is TRANSIENT, not a
    workflow terminal)."""

    def current_expiry(self, name: object) -> object: ...

    def rotate(self, name: object) -> object: ...


# --- __file__-anchored paths: depth RECOMPUTED, not copied --------------------
# The flat `workflowctl.py` computed both of these inline as 3x-dirname of
# `__file__`, correct only from `tools/vast/`. This file sits two levels deeper
# (`tools/vast/vastlib/workflows/ctl.py`), so a verbatim copy would silently
# return `tools/vast` where the repo root belongs — and NOTHING would raise:
# `jobmeta.check_asset_staleness` would compare a mutable B2 asset against
# sources that do not exist at that root, and the pre-spend staleness gate would
# pass vacuously. Same defect class as `boxes/ssh.py:_REPO_ROOT` and
# `core/config.py:_HERE`; same fix — one module constant, one place to correct
# if the package ever moves again, pinned against the flat file's own resolution
# by `test_vastlib_workflows_ctl.py::test_path_anchors_match_the_flat_module`.
#
# `tools/vast/` — three dirnames up. `rehearse.sh` STAYS there (it is not part
# of the package), so `rehearse_workflow`'s default script path resolves through
# this constant, not through the package directory.
# moved-from: workflowctl._HERE
_TOOLS_VAST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The repo checkout root — five dirnames up (`.../repo/tools/vast/vastlib/workflows`).
_REPO_ROOT = os.path.dirname(os.path.dirname(_TOOLS_VAST_DIR))


# --- frozen exit codes (roadmap "Failure classes and exit codes") ------------
# moved-from: workflowctl.EXIT_OK
EXIT_OK = 0
# moved-from: workflowctl.EXIT_INVALID
EXIT_INVALID = 1
# moved-from: workflowctl.EXIT_FAILED
EXIT_FAILED = 2
# moved-from: workflowctl.EXIT_CANCELLED
EXIT_CANCELLED = 3
# moved-from: workflowctl.EXIT_ARTIFACT
EXIT_ARTIFACT = 4
# moved-from: workflowctl.EXIT_CREDENTIAL
EXIT_CREDENTIAL = 5
# moved-from: workflowctl.EXIT_TIMEOUT
EXIT_TIMEOUT = 124

# Fixed operator-facing disclaimer (roadmap M4-T1 "Print a fixed disclaimer
# that rehearsal does not certify CUDA/live artifacts"). `plan_workflow`
# (online) and the later rehearse driver both surface this VERBATIM -- never
# reworded per call site -- so an operator sees the identical caveat whether
# they ran `workflow plan --online` or `workflow rehearse`.
# moved-from: workflowctl.REHEARSAL_DISCLAIMER
REHEARSAL_DISCLAIMER = (
    "REHEARSAL DISCLAIMER: this validates PLUMBING ONLY -- bundle/config "
    "shape, asset staging, and entrypoint wiring. It does NOT certify "
    "CUDA/vLLM ABI compatibility, the CONTENTS of live B2 assets, or real "
    "model/adapter artifacts. A real-GPU canary (M4-T2, deferred to the "
    "operator) is required before any live spend."
)

# Reconcile poll cadence + the "how many missed heartbeats is stale" rule
# `--takeover` needs (wm.controller_is_stale's `stale_after_s` is
# POLL_INTERVAL_S * HEARTBEAT_STALE_MULT — the interval math is this CLI
# layer's concern; workflowmeta only does the pure age comparison).
# moved-from: workflowctl.POLL_INTERVAL_S
POLL_INTERVAL_S = 30
# moved-from: workflowctl.HEARTBEAT_STALE_MULT
HEARTBEAT_STALE_MULT = 3

# Extra grace `claim_controller(takeover=True)` waits BEYOND the staleness
# window before giving up on an incumbent that neither heartbeats again nor
# goes stale (clock skew / a slow event listing). See the takeover dead-zone
# fix below (2026-07-15): takeover WAITS for staleness instead of refusing.
# moved-from: workflowctl.TAKEOVER_WAIT_GRACE_S
TAKEOVER_WAIT_GRACE_S = 30

# Boot/claim watchdog (Gap D, 2026-07-15): a box that is acquired but whose
# child job is still 'submitted' (never claimed by jobd) this many seconds
# after `box_acquired` is an infra failure — a box that never booted jobd,
# hung in `loading`, or otherwise idle-bills without ever taking its job. Past
# this deadline `reconcile_active_box` tears the box down and retargets the
# SAME deterministic job_id onto a replacement, routed through the ordinary
# stage retry policy so repeated boot hangs terminate as an infra stage_failed
# rather than looping teardown+relaunch forever. 1500s (25 min) tolerates a
# cold multilayer image pull on a moderately-slow host; a genuinely-stuck box
# is replaced on the next attempt (retry_on infrastructure) rather than billed.
# Env-overridable (default unchanged): a HEAVY-boot stage — e.g. the e2 `score`
# stage's heavy image + large B2-fetched /workspace/eval/dc3 tree — legit
# runs ~1520-1560s to first claim (found live 2026-07-20, run d8e9: two score
# attempts boot-deadlined at 1527s/1557s, ~30-60s over), so a slow eval host
# needs a longer budget than a lean gen box. Raise via WORKFLOW_BOOT_DEADLINE_S.
# moved-from: workflowctl.BOOT_DEADLINE_S
BOOT_DEADLINE_S = int(os.environ.get("WORKFLOW_BOOT_DEADLINE_S", "1500"))

# Mid-run liveness: once a job is claimed/started, the ONLY dead-box signal the
# controller previously acted on was the vast API listing the instance as
# 'gone' — a preempted/reclaimed spot box that lingers listed ('stopped', or
# stale-'running') left the workflow blind for as long as the listing lingered
# (found live 2026-07-20, run 5819: a2 box outbid at 13:11, billing flatlined,
# but retarget only fired at 18:20 when the instance finally left the list —
# 5h of dead air). jobd heartbeats every ~60s while a job runs; a claimed/
# started job with no heartbeat for this many seconds is presumed dead
# REGARDLESS of the box's observed power state, and is torn down + retargeted
# (same-attempt, same deterministic job_id, resuming from jobs/<id>/checkpoints/).
# 900s = 15 consecutive missed beats — far beyond any B2 event-listing jitter;
# a live box whose event PUBLISH transport died (rotated/dead key) also can
# never publish results, so replacing it is correct there too.
# moved-from: workflowctl.JOB_HEARTBEAT_STALE_S
JOB_HEARTBEAT_STALE_S = int(os.environ.get("WORKFLOW_JOB_HEARTBEAT_STALE_S", "900"))

# Download-bandwidth floor (Mb/s) for workflow box launches. Excludes the
# pathologically slow-flow hosts whose cold image pull blows the boot deadline
# (found live 2026-07-15). Advisory: paired with retry so a slow host that
# still slips through is replaced, not fatal.
# 2026-08-03: the pick-time floor moved into herdd (LAUNCH_INET_DOWN_MBPS
# knob, default 1000 — relaxed from 2000 the same day, since the rating is a
# weak predictor in both directions and 2000 excluded too much supply — with an
# automatic unfloored fallback pass when the floor empties the market). None
# here = defer to that knob; a number would override.
# 2026-08-05: this floor is now the ONLY host-quality prior at pick time on the
# workflow lane — the ResourceProfile.geo pins it used to be paired with were
# opened (see tools/vast/workflow.py's `geo` comment).
# moved-from: workflowctl.LAUNCH_INET_DOWN_FLOOR_MBPS
LAUNCH_INET_DOWN_FLOOR_MBPS = None

# Sustained image-pull throughput floor (MB/s) for the boot-throughput watchdog
# (BOOT_HEALTHCHECK phase P0). Only used to render the `failed`-event reason
# string here; the sampler math + window/poll knobs live in herdd
# (BootThroughputSampler / _boot_knob). Env-overridable to match the herdd
# side's BOOT_MIN_MBPS precedence.
# moved-from: workflowctl.BOOT_MIN_MBPS
BOOT_MIN_MBPS = float(os.environ.get("BOOT_MIN_MBPS", "5"))

# --- stale-image scheduling gate (velvet P3; docs/plans/stale-image-gate.md) --
# The incident, 2026-07-30: three frontier-wave jobs died within seconds on box
# 46240842 because its baked env predated a script they needed. The staleness
# SIGNAL already existed; no scheduling path consulted it. P1 landed the pure
# tri-state classifier (`imageref.classify_image_staleness`) and wired it into
# alarms only; this is the enforcement half for the workflow controller.
#
# Enforced ONLY on the box-REUSE paths, which checked nothing at all: the
# fresh-launch branch of `build_box_resolver` has always been fail-closed on the
# profile's pinned `image_digest`, and a launch bakes the CURRENT image by
# construction. Reuse is where an old env survives — an adopted box, or a
# resume (vast keeps the disk, so a park/resume can NEVER refresh an image).
#
# `<= 0` = ALARM ONLY, the `FLEETD_DESIGN.md` escape-hatch convention: the gate
# still classifies and still records its verdict, it just stops refusing. A
# knob that softens a gate is why the gate stays armed instead of deleted.
# moved-from: workflowctl.IMAGE_GATE_ENFORCE
IMAGE_GATE_ENFORCE = int(os.environ.get("WORKFLOW_IMAGE_GATE", "1"))

# Named failure classes for the refusals (never a bare exception): `stale` is a
# CONFIRMED mismatch and a fresh box fixes it; `unresolved` means we could not
# compare at all, and the owner's rule there is HOLD — auto-launching on
# unresolved would turn a transient registry/API outage into an unbounded
# box-rental loop. Deliberately NOT added to `wm._FAILURE_CLASS_EXIT_CODE`:
# these refusals are per-tick reconcile actions, not terminal workflow verdicts.
# moved-from: workflowctl.IMAGE_GATE_FAILURE_CLASS
IMAGE_GATE_FAILURE_CLASS = {
    imageref.IMG_STALE: "STALE_IMAGE",
    imageref.IMG_UNRESOLVED: "IMAGE_UNRESOLVED",
}


# moved-from: workflowctl.WorkflowCtlError
class WorkflowCtlError(Exception):
    """Any operator-facing failure in this I/O/reconcile layer: a bad/missing
    `WORKFLOW` module, a spec-validation error, a conflicting spec.json byte
    mismatch, a refused second-live-controller claim, a transport failure a
    caller must not silently swallow. The CLI layer (a later subtask) maps
    this into a stable --json error response + one of the exit codes above."""


# moved-from: workflowctl.DetachUnavailable
class DetachUnavailable(Exception):
    """Raised by the (strictly later) --detach path when `systemd-run --user`
    is not on PATH. `str()` on the raised instance is the EXACT foreground
    command the operator must run instead — --detach must never fall back to
    a hidden `nohup`. The message content is set by the reconcile/detach
    subtask's raiser; this module only defines the class shape so this
    foundation and that subtask agree on the exception type up front."""


# --- transport seam (injectable; mirrors jobmeta/runmeta's runner contract) --
# `runner(args, input=None) -> (rc, stdout, stderr)`. Every I/O function below
# accepts `runner=None, bucket=None` and resolves them to these defaults
# inside the function body (not as the parameter default itself) so a caller
# can always tell "nothing was passed" apart from "the default was passed
# explicitly" — same effect as jobmeta's `runner=_default_runner` default,
# spelled so `None` is the uniform sentinel across this module's functions.
# moved-from: workflowctl._default_runner
def _default_runner(args: list[str], input: str | None = None) -> Any:  # noqa: ANN401 — Zone S `jobmeta._default_runner`'s (rc, out, err) is untyped
    """Delegates to jobmeta._default_runner — ONE rclone subprocess
    implementation shared by runmeta/jobmeta/workflowctl."""
    return jobmeta._default_runner(args, input=input)  # type: ignore[no-untyped-call]


# moved-from: workflowctl._resolve_runner
def _resolve_runner(runner: Runner | None) -> Runner:
    return runner if runner is not None else _default_runner


# moved-from: workflowctl._q
def _q(bucket: str | None, key: str) -> str:
    """b2:<bucket>/<key>, with bucket resolved via jobmeta._bucket (env
    B2_BUCKET fallback, raises RunmetaError if neither is set)."""
    return f"b2:{jobmeta._bucket(bucket)}/{key}"  # type: ignore[no-untyped-call]


# --- loading a Workflow module ------------------------------------------------
# moved-from: workflowctl.load_workflow_module
def load_workflow_module(path: str) -> Workflow:
    """Load a `WORKFLOW = Workflow(...)` module-level object from a `.py`
    file at `path` and validate it as a V1 spec (`wm.validate_workflow_spec`,
    `of_record=True` — a workflow submitted for real must be fully pinned).

    Raises `WorkflowCtlError` for: the file failing to import, no
    module-level `WORKFLOW` attribute, `WORKFLOW` not a `Workflow` instance,
    or a `WorkflowSpecError` from cross-object validation (wrapped, not
    propagated raw, so every caller of this function catches one exception
    type)."""
    spec = importlib.util.spec_from_file_location("_workflowctl_loaded_wf", path)
    if spec is None or spec.loader is None:
        raise WorkflowCtlError(f"cannot load workflow module from {path!r}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        raise WorkflowCtlError(
            f"error executing workflow module {path!r}: {e}") from e

    wf = getattr(mod, "WORKFLOW", None)
    if wf is None:
        raise WorkflowCtlError(
            f"{path!r} does not define a module-level WORKFLOW = Workflow(...)")
    if not isinstance(wf, Workflow):
        raise WorkflowCtlError(
            f"{path!r}: WORKFLOW is a {type(wf).__name__}, not a Workflow")

    try:
        wm.validate_workflow_spec(wf, of_record=True)
    except wm.WorkflowSpecError as e:
        raise WorkflowCtlError(f"invalid workflow spec in {path!r}: {e}") from e
    return wf


# --- spec.json (write-once; idempotent on identical bytes) -------------------
# moved-from: workflowctl.write_spec
def write_spec(wf: Workflow, wf_id: str, *, runner: Runner | None = None,
               bucket: str | None = None) -> dict[str, Any]:
    """Write `workflows/<WF_ID>/spec.json`. A deterministic spec must not
    silently change: if an object already exists there with byte-identical
    canonical JSON this is a no-op (safe to retry `plan`/`run` from a crashed
    controller); if it exists with DIFFERENT bytes this raises
    `WorkflowCtlError` rather than overwrite it (spec.json is never
    overwritten by this function)."""
    runner = _resolve_runner(runner)
    wm.validate_wf_id(wf_id)
    body = wm.canonical_spec_json(wf)
    key = f"workflows/{wf_id}/spec.json"

    rc, out, _ = runner(["cat", _q(bucket, key)])
    if rc == 0 and (out or "").strip():
        if out.strip() == body.strip():
            return {"status": "noop", "wf_id": wf_id, "key": key}
        raise WorkflowCtlError(
            f"spec.json for {wf_id!r} already exists with different bytes; "
            f"a deterministic workflow spec must not silently change "
            f"(mint a new WF_ID for a changed spec)")

    rc, _, err = runner(["rcat", _q(bucket, key)], input=body + "\n")
    if rc != 0:
        raise WorkflowCtlError(
            f"write_spec {wf_id!r}: rcat failed: {(err or '').strip()}")
    return {"status": "written", "wf_id": wf_id, "key": key}


# moved-from: workflowctl.read_spec
def read_spec(wf_id: str, *, runner: Runner | None = None,
              bucket: str | None = None) -> Workflow:
    """Read+reconstruct the `Workflow` written by `write_spec`."""
    runner = _resolve_runner(runner)
    wm.validate_wf_id(wf_id)
    key = f"workflows/{wf_id}/spec.json"
    rc, out, err = runner(["cat", _q(bucket, key)])
    if rc != 0 or not (out or "").strip():
        raise WorkflowCtlError(
            f"read_spec {wf_id!r}: no spec.json found ({(err or '').strip()})")
    try:
        d = json.loads(out)
    except (ValueError, TypeError) as e:
        raise WorkflowCtlError(
            f"read_spec {wf_id!r}: spec.json is not valid JSON: {e}") from e
    return wm.spec_from_dict(d)


# --- events --------------------------------------------------------------------
# moved-from: workflowctl.emit
def emit(wf_id: str, event: str, actor: str, *, runner: Runner | None = None,
         bucket: str | None = None,
         **fields: Any) -> dict[str, Any]:  # noqa: ANN401 — free-form event body
    """Append one immutable event object to `workflows/<WF_ID>/events/`,
    same discipline as `runmeta.emit_event`/`jobmeta.emit_event`: unique
    nonce-bearing key (safe for any number of concurrent writers), best-effort
    (`_emitted=False` + `_error` on a transport failure rather than raising —
    a dying controller's final emit can't crash the caller)."""
    runner = _resolve_runner(runner)
    ev = wm.make_event(wf_id, event, actor, **fields)
    key = f"workflows/{wf_id}/events/{wm.event_key(ev)}"
    body = json.dumps(ev, separators=(",", ":")) + "\n"
    rc, _, err = runner(["rcat", _q(bucket, key)], input=body)
    ev["_key"] = key
    ev["_emitted"] = (rc == 0)
    if rc != 0:
        ev["_error"] = (err or "").strip()
    return ev


# moved-from: workflowctl.read_events
def read_events(wf_id: str, *, runner: Runner | None = None,
                bucket: str | None = None) -> list[str]:
    """List + read every object under `workflows/<WF_ID>/events/`. Tolerant
    of a per-object read failure (transient transport hiccup skips that one
    object this tick — its key is immutable so a later tick's `lsf`+`cat`
    picks it up again; never raises). Returns raw bodies (str); parsing is
    `wm.fold_workflow_events`'s `_coerce` job, not this function's."""
    runner = _resolve_runner(runner)
    rc, out, _ = runner(["lsf", _q(bucket, f"workflows/{wf_id}/events/")])
    if rc != 0:
        return []
    events = []
    for name in (out or "").splitlines():
        name = name.strip()
        if not name:
            continue
        rc2, body, _ = runner(["cat", _q(bucket, f"workflows/{wf_id}/events/{name}")])
        if rc2 != 0:
            continue
        events.append(body)
    return events


# moved-from: workflowctl.view
def view(wf_id: str, *, runner: Runner | None = None,
         bucket: str | None = None) -> dict[str, Any]:
    """The current pure-folded view of a workflow (`wm.fold_workflow_events`
    over every event object currently on B2)."""
    return wm.fold_workflow_events(read_events(wf_id, runner=runner, bucket=bucket))


# moved-from: workflowctl._prior_stage_machines
def _prior_stage_machines(wf_id: str, stage_name: str, *, runner: Runner | None = None,
                          bucket: str | None = None) -> set[Any]:
    """The set of vast machine_ids that PRIOR box_acquired events for this
    stage recorded — the `exclude_machines` seed for host rotation. Read
    straight off the immutable event log (box_acquired now carries machine_id),
    so a retry after a boot kill / infra failure picks a DIFFERENT host instead
    of looping onto the same slow one. A box_acquired without a machine_id
    (older event, or an API read that lacked it) simply doesn't contribute."""
    machines = set()
    for body in read_events(wf_id, runner=runner, bucket=bucket):
        try:
            e = json.loads(body)
        except (ValueError, TypeError):
            continue
        if (isinstance(e, dict) and e.get("event") == "box_acquired"
                and e.get("stage") == stage_name
                and e.get("machine_id") is not None):
            machines.add(e["machine_id"])
    return machines


# --- local flock: one live controller PROCESS per host ------------------------
# This is a same-host guard only (a crashed process's fd closes -> the lock
# releases immediately) — the REMOTE controller-heartbeat check below is what
# refuses a second live controller across different hosts/restarts.
# moved-from: workflowctl._lock_dir
def _lock_dir() -> str:
    root = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    d = os.path.join(root, "vast-workflowctl")
    os.makedirs(d, exist_ok=True)
    return d


# moved-from: workflowctl.acquire_local_lock
def acquire_local_lock(wf_id: str, *, takeover: bool = False) -> IO[Any]:
    """`flock(LOCK_EX | LOCK_NB)` on `<cache>/vast-workflowctl/<wf_id>.lock`.
    Records the holder pid in the file body. Returns an open file handle to
    pass to `release_local_lock`.

    Normal (`takeover=False`): raises `WorkflowCtlError` if another process
    already holds the flock.

    Takeover (`takeover=True`): mirrors `claim_controller`'s stale-supersede
    pattern for the LOCAL lock — the `--takeover` incident (2026-07-15) was a
    controller that cleared the REMOTE heartbeat but then died locally on
    'another controller holds the local lock'. On `BlockingIOError` we read the
    recorded pid and probe it: a dead/unparseable holder (`ProcessLookupError`
    or empty/garbage pid) is force-broken (re-attempt the flock, restamp our
    pid); a still-alive holder is never stomped (re-raise)."""
    wm.validate_wf_id(wf_id)
    path = os.path.join(_lock_dir(), f"{wf_id}.lock")
    fh = open(path, "a+")

    def _stamp_pid() -> None:
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()

    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        if not takeover:
            fh.close()
            raise WorkflowCtlError(
                f"another controller holds the local lock for {wf_id!r} ({path})")
        # takeover: only steal from a demonstrably dead holder.
        try:
            fh.seek(0)
            raw = fh.read().strip()
            holder_pid = int(raw)
        except (ValueError, OSError):
            holder_pid = None
        holder_alive = False
        if holder_pid is not None:
            try:
                os.kill(holder_pid, 0)
                holder_alive = True
            except ProcessLookupError:
                holder_alive = False
            except PermissionError:
                holder_alive = True   # exists, owned by another user — do not steal
        if holder_alive:
            fh.close()
            raise WorkflowCtlError(
                f"--takeover refused for {wf_id!r}: a live controller (pid "
                f"{holder_pid}) holds the local lock ({path})")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.close()
            raise WorkflowCtlError(
                f"--takeover could not break the local lock for {wf_id!r} "
                f"({path}) — holder pid {holder_pid} left an unreleasable flock")
        _stamp_pid()
        return fh
    _stamp_pid()
    return fh


# moved-from: workflowctl.release_local_lock
def release_local_lock(handle: IO[Any]) -> None:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


# --- controller liveness (remote; across hosts/restarts) ---------------------
# moved-from: workflowctl.claim_controller
def claim_controller(wf_id: str, actor: str, *, runner: Runner | None = None,
                     bucket: str | None = None,
                     now: str, takeover: bool = False, clock: Seam | None = None,
                     sleep_fn: Seam | None = None) -> dict[str, Any]:
    """Claim (or take over) the controller role for `wf_id`.

    Normal claim (`takeover=False`): refuses with `WorkflowCtlError` if a
    DIFFERENT actor is recorded as controller and its heartbeat is not yet
    stale (`wm.controller_is_stale`, `stale_after_s=POLL_INTERVAL_S *
    HEARTBEAT_STALE_MULT`) — "refuse a 2nd live controller". Re-claiming as
    the SAME actor (a restarted process resuming its own run) is always
    allowed. Emits `controller_started`.

    Takeover (`takeover=True`): WAITS for staleness instead of refusing (the
    2026-07-15 dead-zone incident: 'stop old controller, immediately resume
    --takeover' ALWAYS hit a refusal because the dead controller's last
    heartbeat was <90s old — the refusal went to a detached log and the
    workflow ran controller-less for an hour). If the recorded heartbeat is
    not yet stale, poll the fold (one `sleep_fn(POLL_INTERVAL_S)` +
    re-`view` per poll, up to `stale window + TAKEOVER_WAIT_GRACE_S` from
    `now`) until it goes stale — a genuinely-dead incumbent goes stale
    within the wait budget by construction. Only if the incumbent
    heartbeats AGAIN during the wait (a genuinely live controller), or the
    wait budget is exhausted without staleness, raise the refusal. A
    takeover where the recorded incumbent IS the claiming actor skips the
    wait entirely (same-actor fast path — the non-takeover claim path
    already readmits the same actor instantly, so waiting out one's own
    heartbeat adds only a stall). Emits `takeover` instead of
    `controller_started`. `clock`/`sleep_fn` default to
    `wm.now_ts`/`time.sleep` (injectable, same seams as
    `run_controller`)."""
    runner = _resolve_runner(runner)
    v = view(wf_id, runner=runner, bucket=bucket)
    cur_actor = v.get("controller", {}).get("actor")
    stale_after_s = POLL_INTERVAL_S * HEARTBEAT_STALE_MULT
    stale = wm.controller_is_stale(v, now=now, stale_after_s=stale_after_s)

    if takeover:
        # Same-actor fast path (2026-07-15 adversarial review): an operator
        # stop -> `resume --takeover` cycle re-claims as the SAME actor. The
        # non-takeover claim path below already readmits the same actor
        # instantly (no staleness gate), so skipping the wait here introduces
        # no new risk class — waiting out one's OWN last heartbeat only added
        # a 60-150s stall to the exact flow the takeover wait was built for.
        if not stale and cur_actor == actor:
            return emit(wf_id, "takeover", actor, runner=runner, bucket=bucket)
        if not stale:
            clock = clock or wm.now_ts
            sleep_fn = sleep_fn or time.sleep
            wait_budget_s = stale_after_s + TAKEOVER_WAIT_GRACE_S
            baseline_hb = v.get("controller", {}).get("last_heartbeat_ts")
            # bounded by poll COUNT too, so a frozen injected clock can never
            # spin this loop forever in a test.
            max_polls = int(wait_budget_s // POLL_INTERVAL_S) + 1
            # ACCEPTED split-brain window: two waiters that both observe
            # staleness inside the same poll window will BOTH claim (each
            # emits `takeover`). A successful takeover advances the fold's
            # controller last_heartbeat_ts (`takeover` is a CONTROLLER_EVENT),
            # so a third-party claim IS caught on the NEXT poll — the residual
            # window is one poll interval, identical to the pre-existing
            # non-takeover stale-claim race, and accepted as-is.
            for _ in range(max_polls):
                sleep_fn(POLL_INTERVAL_S)
                poll_now = clock()
                v = view(wf_id, runner=runner, bucket=bucket)
                cur_actor = v.get("controller", {}).get("actor")
                hb = v.get("controller", {}).get("last_heartbeat_ts")
                # runmeta ts strings are fixed-width — lexicographic order IS
                # chronological order (same discipline as the fold's ts sort).
                if hb and baseline_hb and hb > baseline_hb:
                    # the incumbent heartbeated AGAIN during the wait: a
                    # genuinely live controller — never stomp it.
                    raise WorkflowCtlError(
                        f"--takeover refused for {wf_id!r}: controller "
                        f"{cur_actor!r} heartbeat is not yet stale (must be "
                        f"idle for more than {stale_after_s}s)")
                if wm.controller_is_stale(v, now=poll_now,
                                           stale_after_s=stale_after_s):
                    stale = True
                    break
                try:
                    waited_s = wm._ts_diff_seconds(poll_now, now)
                except wm.WorkflowIdError:
                    waited_s = None
                if waited_s is not None and waited_s >= wait_budget_s:
                    break
            if not stale:
                raise WorkflowCtlError(
                    f"--takeover refused for {wf_id!r}: controller {cur_actor!r} "
                    f"heartbeat is not yet stale (must be idle for more than "
                    f"{stale_after_s}s)")
        return emit(wf_id, "takeover", actor, runner=runner, bucket=bucket)

    if cur_actor and cur_actor != actor and not stale:
        raise WorkflowCtlError(
            f"refusing second live controller for {wf_id!r}: {cur_actor!r} "
            f"is live (use --takeover once its heartbeat is stale)")
    return emit(wf_id, "controller_started", actor, runner=runner, bucket=bucket)


# moved-from: workflowctl.heartbeat
def heartbeat(wf_id: str, actor: str, *, runner: Runner | None = None,
              bucket: str | None = None) -> dict[str, Any]:
    """Emit one `controller_heartbeat` event (the reconcile loop calls this
    once per tick to keep `claim_controller`'s staleness check honest)."""
    return emit(wf_id, "controller_heartbeat", actor, runner=runner, bucket=bucket)


# --- reconcile engine + controller loop + detach: added by wfctl-reconcile ---
# A stage-attempt status that means "something is out on a box right now" —
# no new submission may start for that stage, and step 3 (cancellation) must
# propagate a kill rather than just wait, while step 6/7/8 must NOT treat the
# stage as resolved. Mirrors `wm.STAGE_TERMINAL`'s role for the opposite case.
# moved-from: workflowctl.STAGE_INFLIGHT
STAGE_INFLIGHT = frozenset(
    {"stage_planned", "stage_submitted", "stage_started", "box_acquired"})

# Translation from the workflow retry_on vocabulary (`workflow.RETRY_CLASSES`)
# to the roadmap's stable reporting vocabulary ("Failure classes and exit
# codes") for a stage that has exhausted its retries or was never retryable.
# moved-from: workflowctl._FAILURE_CLASS_MAP
_FAILURE_CLASS_MAP = {
    "infrastructure": "INFRASTRUCTURE_FAILED",
    "entrypoint": "ENTRYPOINT_FAILED",
    "postcondition": "POSTCONDITION_FAILED",
}


# moved-from: workflowctl._artifact_key
def _artifact_key(wf_id: str, stage: str, artifact: str) -> str:
    return f"workflows/{wf_id}/artifacts/{stage}/{artifact}.json"


# moved-from: workflowctl._verdict_key
def _verdict_key(wf_id: str) -> str:
    return f"workflows/{wf_id}/verdict.json"


# moved-from: workflowctl._provenance_key
def _provenance_key(wf_id: str) -> str:
    return f"workflows/{wf_id}/provenance.json"


# moved-from: workflowctl._report_key
def _report_key(wf_id: str) -> str:
    return f"workflows/{wf_id}/report.md"


# moved-from: workflowctl.read_accepted_artifact
def read_accepted_artifact(wf_id: str, stage: str, artifact: str, *,
                           runner: Runner | None = None,
                           bucket: str | None = None) -> Any:  # noqa: ANN401 — the accepted-artifact record is whatever json.loads returned
    """Read a previously-accepted stage artifact record (written by
    reconcile_tick's artifact-acceptance action). Returns None if this
    (stage, artifact) has never been accepted — a probe, not a validator,
    same contract as `jobmeta.read_results_done`. This is also the
    idempotency check `reconcile_tick` uses to skip a re-acceptance."""
    runner = _resolve_runner(runner)
    rc, out, _ = runner(["cat", _q(bucket, _artifact_key(wf_id, stage, artifact))])
    if rc != 0 or not (out or "").strip():
        return None
    try:
        return json.loads(out)
    except (ValueError, TypeError):
        return None


# --- M4-T2 canary receipt store: content-keyed, TTL-bounded, reusable --------
# A canary receipt is NOT a per-workflow event: it is reusable ACROSS workflows
# by its composite content key, so it lives at a flat `workflow-canary/receipts/`
# prefix (mirrors the jobd-bootstrap / bundle content-addressed stores), never
# under workflows/<WF_ID>/. B2 objects never auto-expire, so the TTL is a field
# in the body enforced at READ time (same pure age-compare posture as
# wm.controller_is_stale).
# moved-from: workflowctl.CANARY_RECEIPT_TTL_S
CANARY_RECEIPT_TTL_S = 86400  # default receipt validity window (24h)


# moved-from: workflowctl._canonical_sha256
def _canonical_sha256(obj: object) -> str:
    """SHA-256 over canonical JSON (sorted keys, tight separators, no NaN) —
    the one content-hash discipline used everywhere in this tree."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode("utf-8")).hexdigest()


# moved-from: workflowctl.canary_receipt_key
def canary_receipt_key(*, image_digest: str | None, jobd_sha: str | None,
                       model_manifest_sha: str | None,
                       adapter_manifest_sha: str | None,
                       recipe_sha: str | None) -> str:
    """Composite content key for a canary receipt (roadmap M4-T2): SHA-256 over
    the canonical JSON of the frozen 5-tuple. Absent components normalize to ''
    so the key is stable and reproducible by both the producer and the gate."""
    return _canonical_sha256({
        "image_digest": image_digest or "", "jobd_sha": jobd_sha or "",
        "model_manifest_sha": model_manifest_sha or "",
        "adapter_manifest_sha": adapter_manifest_sha or "",
        "recipe_sha": recipe_sha or ""})


# moved-from: workflowctl._canary_receipt_key_ref
def _canary_receipt_key_ref(key: str) -> str:
    return f"workflow-canary/receipts/{key}.json"


# moved-from: workflowctl.canary_receipt_status
def canary_receipt_status(key: str, *, runner: Runner | None = None,
                          bucket: str | None = None,
                          now_epoch: float | None = None) -> tuple[str, Any]:
    """Probe a stored canary receipt by key. Returns (status, receipt|None) with
    status in {'valid','missing','expired','failed'}: 'missing' if never written
    or unreadable, 'expired' if now >= expires_ts, 'failed' if the receipt
    records rc != 0 or a non-passing GPU step, else 'valid'. A probe, not a
    validator — never raises."""
    runner = _resolve_runner(runner)
    rc, out, _ = runner(["cat", _q(bucket, _canary_receipt_key_ref(key))])
    if rc != 0 or not (out or "").strip():
        return "missing", None
    try:
        receipt = json.loads(out)
    except (ValueError, TypeError):
        return "missing", None
    now = now_epoch if now_epoch is not None else time.time()
    exp = receipt.get("expires_ts")
    if not isinstance(exp, (int, float)) or isinstance(exp, bool) or now >= exp:
        return "expired", receipt
    if receipt.get("rc") != 0 or not receipt.get("gpu_step_ok"):
        return "failed", receipt
    return "valid", receipt


# moved-from: workflowctl.read_canary_receipt
def read_canary_receipt(key: str, *, runner: Runner | None = None,
                        bucket: str | None = None,
                        now_epoch: float | None = None) -> Any:  # noqa: ANN401 — the receipt body is whatever json.loads returned
    """Return the receipt dict ONLY if it exists, is unexpired, and passed;
    else None (missing/expired/failed collapse to None for callers that only
    need pass/fail). Use canary_receipt_status for the precise reason."""
    status, receipt = canary_receipt_status(
        key, runner=runner, bucket=bucket, now_epoch=now_epoch)
    return receipt if status == "valid" else None


# moved-from: workflowctl.write_canary_receipt
def write_canary_receipt(key: str, receipt: dict[str, Any], *,
                         runner: Runner | None = None,
                         bucket: str | None = None) -> dict[str, Any]:
    """Persist a receipt at its content key. A fresh canary run for the same key
    overwrites (refreshes expiry) — rcat is a PUT. Best-effort like emit():
    returns the receipt with _stored/_error, never raises."""
    runner = _resolve_runner(runner)
    body = json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    rc, _, err = runner(["rcat", _q(bucket, _canary_receipt_key_ref(key))],
                        input=body)
    out = dict(receipt)
    out["_stored"] = (rc == 0)
    if rc != 0:
        out["_error"] = (err or "").strip()
    return out


# moved-from: workflowctl._owned_instance_ids
def _owned_instance_ids(v: dict[str, Any]) -> list[str]:
    """Every distinct box instance_id any stage attempt has recorded in the
    fold. Empty in every M2 test path — no box_acquired event carries an
    instance_id until M3-T1 wires real box acquisition — so the teardown
    steps below are a no-op today by construction, not by a special case.

    STR-normalized: a fresh launch records the vast API's int id while the
    ADOPT re-acquire path records the label lookup's str — one workflow fold
    can hold both shapes at once (found live 2026-07-30, run 2ed9: generate
    adopted as '46216906', score fresh as int 46249864 → `sorted({str,int})`
    TypeError crashed the controller on every cost tick; the same mix would
    also double-count one box as two cost-observer keys)."""
    return sorted({
        str(sv.get("instance_id")) for sv in v.get("stages", {}).values()
        if sv.get("instance_id")
    })


# moved-from: workflowctl._classify_job_failure
def _classify_job_failure(job_view: dict[str, Any]) -> str:
    """Translate a folded child-job view's `status`/`fail_reason` into one
    of `workflow.RETRY_CLASSES` ('infrastructure'|'entrypoint'|
    'postcondition'). This is a translation between jobmeta's job-event
    vocabulary and the workflow spec's `retry_on` vocabulary — two sibling
    modules' vocabularies — not a workflow-spec PURE decision itself, so it
    stays in this I/O layer rather than in workflowmeta (which never imports
    jobmeta). `wm.decide_retry` remains the one place that turns this
    classification + the stage's retry policy into retry/fail."""
    if job_view.get("status") == "cancelled":
        return "infrastructure"
    reason = (job_view.get("fail_reason") or "").lower()
    if any(k in reason for k in
           ("preempt", "evict", "lost", "timeout", "connection", "ssh", "spot",
            "throughput")):        # boot-throughput condemn: a slow HOST = infra
        return "infrastructure"
    if any(k in reason for k in
           ("artifact", "manifest", "postcondition", "sha256", "validate")):
        return "postcondition"
    return "entrypoint"


# --- box acquisition: a deliberate M3 seam ------------------------------------
# moved-from: workflowctl._default_box_resolver
def _default_box_resolver(stage: JobStage, wf: Workflow, attempt: int = 0) -> str | None:
    """M3-T1 ("resource profiles and reusable launch primitives") owns real
    box acquisition/launch and the budget/credential checks that must gate
    it. THIS default never picks or launches a real vast box — always
    returns None, so `reconcile_tick` reports `{'action': 'need_box', ...}`
    instead of fabricating a box id or spending anything. Every PRODUCTION
    `reconcile_tick`/`run_controller` call that passes no `box_resolver=`
    keeps hitting this no-op; `build_box_resolver` below is the real M3-T1
    resolver a caller opts into explicitly. Never call herdd launch from
    here."""
    return None


# moved-from: workflowctl.build_box_teardown
def build_box_teardown(*, stopper: Seam | None = None,
                       destroyer: Seam | None = None) -> Seam:
    """M3-T1's real `box_teardown` builder: plugs straight into
    `reconcile_tick(..., box_teardown=build_box_teardown())`'s
    `_teardown_boxes` call site. `stopper`/`destroyer` default to
    `lifecycle.stop_box`/`lifecycle.destroy_box` (both argparse-free, soft
    `(ok, err)` primitives -- never sys.exit, never raise on a normal API
    error) so a test can inject fakes with the same `(iid) -> (ok, err)`
    shape without importing real vast transport.

    Returns `box_teardown(iid, mode) -> bool`: `mode == 'destroy'` routes to
    `destroyer`, anything else (in practice always `'stop'`, `Workflow.
    teardown`'s only other TEARDOWN_CHOICES member) routes to `stopper`.
    Any exception from the injected callable is swallowed to `False` --
    `_teardown_boxes` already does the same belt-and-suspenders catch, but
    this builder's own contract (per the roadmap packet) is to never let a
    single box's teardown failure raise into the reconcile loop itself."""
    stopper = stopper if stopper is not None else lifecycle.stop_box
    destroyer = destroyer if destroyer is not None else lifecycle.destroy_box

    def box_teardown(iid: object, mode: str) -> bool:
        try:
            ok, _err = destroyer(iid) if mode == "destroy" else stopper(iid)
        except Exception:
            return False
        return bool(ok)

    return box_teardown


# --- velvet P3 stale-image gate: gather inputs here, DECIDE in imageref ------
# moved-from: workflowctl._instance_image_verdict
def _instance_image_verdict(inst: Mapping[str, Any], *, digest_verifier: Seam,
                            fallback_image: str | None = None,
                            pinned_digest: str | None = None) -> Any:  # noqa: ANN401 — `imageref.classify_image_staleness` is an untyped flat sibling
    """`(state, reason)` for ONE vast instance record, straight out of
    `imageref.classify_image_staleness`. This function only GATHERS that pure
    classifier's inputs — the box's launch-time `HERDD_IMAGE_DIGEST` stamp,
    and the digest the box OUGHT to be running. It decides nothing: any `if` on
    a digest belongs in `imageref`, so the CLI, the daemon and this controller
    can never drift into three subtly different staleness policies.

    WHICH "ought" — the pin wins when there is one. For a profile carrying an
    of-record `image_digest`, that pin IS the workflow's declared truth and the
    fresh-launch branch already refuses to launch anything else, so a box
    stamped with the pin runs exactly the declared env and MUST stay adoptable
    even after the tag moves. Comparing it against the moved tag instead would
    refuse the adopt and then fail its replacement launch with IMAGE_DRIFT — a
    workflow that can neither reuse nor launch, killed by someone else's push.
    Pinning exists precisely to be immune to that. Only an UNPINNED profile has
    no declared truth, and there the live tag is the only reference.

    A pinned comparison needs no network at all. For the unpinned case the
    lookup is SKIPPED for exactly the shapes the classifier short-circuits
    before consulting a digest (unstamped box / foreign registry /
    `@sha256:`-pinned ref) — the same no-lookup rule `herdd.
    _fleet_image_states` applies, so a public-registry serve box costs zero
    registry calls and can never be held up by one.

    Resolution wraps the injected `digest_verifier` in
    `imageref.resolve_tag_digest_ttl` rather than calling it bare. That is
    mandatory, not an optimization: a controller lives for hours, and BOTH
    `image_tag_digest` and `image_ref_digest` cache a success for the life of
    the process. Bare, this would compare the box's stamp against whatever the
    tag pointed at when the controller started and therefore never notice the
    push that made the box stale — precisely the 2026-07-30 footgun.

    A verifier that raises, or that returns None, leaves `current_digest=None`
    -> IMG_UNRESOLVED -> the callers HOLD. Never fail-open."""
    image = models._instance_image(inst) or (fallback_image or "")
    stamped = models._instance_env(inst).get(imageref.IMAGE_DIGEST_ENV)
    current = pinned_digest or None
    if current is None and stamped and image and "@sha256:" not in image:
        host, _path, _tag = imageref._split_image(image)
        if imageref.is_our_registry(host):
            try:
                current, _cache_state = imageref.resolve_tag_digest_ttl(
                    image, resolver=digest_verifier)
            except Exception:
                current = None
    return imageref.classify_image_staleness(
        image=image, stamped_digest=stamped, current_digest=current)


# moved-from: workflowctl._image_gate_refuses
def _image_gate_refuses(state: object) -> bool:
    """ENFORCEMENT, kept separate from classification: `stale` and
    `unresolved` refuse a box REUSE, `fresh` and `not_applicable` proceed
    silently. `IMAGE_GATE_ENFORCE <= 0` (WORKFLOW_IMAGE_GATE) demotes the whole
    gate to alarm-only — the verdict is still computed and still recorded, it
    just stops refusing."""
    return IMAGE_GATE_ENFORCE > 0 and state in (imageref.IMG_STALE,
                                                imageref.IMG_UNRESOLVED)


# moved-from: workflowctl.build_image_state_observer
def build_image_state_observer(*, instance_reader: Seam | None = None,
                               digest_verifier: Seam | None = None) -> Seam:
    """velvet P3 `image_state_observer(instance_id, pinned_digest=None) ->
    (state, reason)`: the seam `reconcile_active_box` consults before RESUMING
    a stopped box. `pinned_digest` is the active stage profile's of-record
    `image_digest` (see `_instance_image_verdict` for why the pin outranks the
    live tag) — the caller supplies it because only the caller knows the
    profile; this builder never reads a Workflow.

    Default reader is `health._get_instance_soft` (never raises, never
    sys.exits — returns None on any API error). An unreadable record
    classifies IMG_UNRESOLVED and therefore HOLDs: we cannot prove the box runs
    current code, and "could not compare" is exactly the state the owner ruled
    must not auto-spend. Default verifier matches the one
    `build_live_controller_deps` hands `build_box_resolver`, so the adopt gate
    and the resume gate resolve through ONE function and ONE TTL cache entry
    per image and can never return opposite verdicts in the same tick."""
    reader = (instance_reader if instance_reader is not None
              else health._get_instance_soft)
    verifier = (digest_verifier if digest_verifier is not None
                else imageref.image_ref_digest)

    def image_state_observer(instance_id: object,
                             pinned_digest: str | None = None) -> Any:  # noqa: ANN401 — same untyped (state, reason) pair as `_instance_image_verdict`
        try:
            inst = reader(instance_id)
        except Exception:
            inst = None
        if not isinstance(inst, dict) or not inst:
            return imageref.IMG_UNRESOLVED, (
                "instance record unreadable — cannot compare the box's image "
                "stamp against what it ought to be running")
        return _instance_image_verdict(inst, digest_verifier=verifier,
                                        pinned_digest=pinned_digest)

    return image_state_observer


# moved-from: workflowctl.build_box_resolver
def build_box_resolver(*, wf_id: str, actor: str, runner: Runner | None = None,
                       bucket: str | None = None,
                       offer_picker: Seam | None = None, launcher: Seam | None = None,
                       digest_verifier: Seam | None = None,
                       bootstrap_stager: Seam | None = None,
                       instance_finder: Seam | None = None,
                       image_login_provider: Seam | None = None,
                       jobs_composer: Seam | None = None,
                       ssh_pubkey_provider: Seam | None = None,
                       hf_token_provider: Seam | None = None,
                       dry_run: bool = False) -> Seam:
    """M3-T1's real `box_resolver` builder: plugs straight into
    `reconcile_tick(..., box_resolver=build_box_resolver(...))`'s
    `_plan_and_submit_stage` call site. Every transport primitive defaults to
    the real `herdd` function it's named after, all overridable so a test
    (or a future dry-run CLI flag) drives this with fakes/closures and never
    touches the network.

    The returned `box_resolver(stage, wf, attempt) -> str|None` runs, per
    call:
      1. ADOPT: an existing vast instance labelled the deterministic
         `'run:' + wm.stage_job_id(wf_id, stage.name, attempt)` box label is
         reused (preferring a running one) instead of launching a duplicate
         -- emits `box_acquired(adopted=True)` and returns its id WITHOUT
         spending anything new. Gated by the velvet P3 stale-image gate: a
         candidate whose baked env predates the current tag is REFUSED
         (`box_adopt_refused`), never handed a job.
      2. DIGEST VERIFY (fail-closed): a profile with a pinned
         `image_digest` must re-resolve to that SAME digest right now, or
         this raises `WorkflowCtlError` rather than launch a drifted image
         (an of-record profile that can't be verified at all -- no
         `digest_verifier` result -- is ALSO a hard refusal, not a silent
         skip).
      3. Stage the content-addressed jobd bootstrap.
      4. Pick the cheapest matching offer; `None` (no match) returns `None`
         so `reconcile_tick` reports `need_box` -- no spend, no event.
      5. Launch, stamping the verified digest into the box env, and emit
         `box_acquired(adopted=False)` carrying the offer/price/digest/
         bootstrap provenance the roadmap packet requires."""
    offer_picker = offer_picker if offer_picker is not None else offers.pick_cheapest_offer
    launcher = launcher if launcher is not None else lifecycle.launch_instance
    digest_verifier = (digest_verifier if digest_verifier is not None
                        else imageref.image_tag_digest)
    bootstrap_stager = (bootstrap_stager if bootstrap_stager is not None
                         else bundle._stage_jobd_bootstrap)
    instance_finder = (instance_finder if instance_finder is not None
                        else lifecycle.find_matching_instance)
    image_login_provider = (image_login_provider if image_login_provider is not None
                            else launch.image_login_arg)
    # The ONE shared jobs-launch-body builder (M3-T1 extraction, defect #6 fix):
    # composes the jobd-boot onstart + scoped B2 cred env EXACTLY like
    # `herdd launch --jobs`, so a workflow-launched box actually runs jobd and
    # claims its queued job instead of booting bare and idling forever.
    jobs_composer = (jobs_composer if jobs_composer is not None
                     else bundle.compose_jobs_launch_env)
    # Gap B (parity): ssh-authorized_keys inject + hf_login prelude, mirroring
    # `_do_launch` (which prepends them BEFORE the jobs composer). Best-effort,
    # injectable for hermetic tests; both defaults return None when absent.
    ssh_pubkey_provider = (ssh_pubkey_provider if ssh_pubkey_provider is not None
                           else (lambda: ssh.pub_key_text(None)))
    hf_token_provider = (hf_token_provider if hf_token_provider is not None
                         else launch.hf_token_text)
    # One durable `box_adopt_refused` per (instance, verdict) per controller.
    # A HOLD re-enters this resolver every POLL_INTERVAL_S, and one immutable B2
    # object per tick for the length of a registry outage is a log flood, not a
    # record. The per-tick action dict + the journald tick line carry the
    # repetition; B2 carries the transition.
    _refusals_emitted = set()

    def box_resolver(stage: JobStage, wf: Workflow, attempt: int) -> Any:  # noqa: ANN401 — a vast instance id: str from ADOPT, int from a fresh launch
        profile = wf.profiles[stage.profile]
        label = "run:" + wm.stage_job_id(wf_id, stage.name, attempt)

        existing = instance_finder(label)
        if existing:
            # velvet P3 stale-image gate. Adoption is the fully-AUTOMATIC reuse
            # path (`reconcile_active_box` -> `_retarget` reaches it with no
            # operator in the loop), so it hard-refuses rather than warning: a
            # box whose baked env predates the current tag is exactly box
            # 46240842, which took three frontier-wave jobs down in seconds.
            # Classification is per CANDIDATE, not per label — a label can hold
            # both a refused box and its replacement, and the fresh one must
            # still be adoptable rather than the whole label being poisoned.
            adoptable: list[Any] = []
            refused: list[Any] = []
            for cand in existing:
                st, why = _instance_image_verdict(
                    cand, digest_verifier=digest_verifier,
                    fallback_image=profile.image,
                    pinned_digest=profile.image_digest)
                (refused if _image_gate_refuses(st) else adoptable).append(
                    (cand, st, why))
            for cand, st, why in refused:
                cid = str(cand.get("id"))
                if (cid, st) not in _refusals_emitted:
                    _refusals_emitted.add((cid, st))
                    emit(wf_id, "box_adopt_refused", actor, runner=runner,
                         bucket=bucket, ts=wm.now_ts(), stage=stage.name,
                         attempt=attempt, instance_id=cid, profile=stage.profile,
                         image=models._instance_image(cand) or profile.image,
                         image_state=st, image_reason=why,
                         failure_class=IMAGE_GATE_FAILURE_CLASS.get(st))
            if adoptable:
                inst, st, _why = next(
                    ((i, s, w) for i, s, w in adoptable
                     if (i.get("actual_status") or "").lower() in bidpolicy.LIVE_STATES),
                    adoptable[0])
                iid = str(inst.get("id"))
                emit(wf_id, "box_acquired", actor, runner=runner, bucket=bucket,
                     ts=wm.now_ts(), stage=stage.name, attempt=attempt,
                     instance_id=iid, adopted=True, profile=stage.profile,
                     image=profile.image, machine_id=inst.get("machine_id"),
                     image_state=st)
                return iid
            if any(s == imageref.IMG_UNRESOLVED for _c, s, _w in refused):
                # HOLD, do NOT fall through and launch. `unresolved` means the
                # registry/API could not be reached, so "launch a fresh one" is
                # unbounded: every tick of an outage would rent another box on
                # a guess. Returning None costs one `need_box` tick and retries
                # for free when the registry answers again (owner ruling; see
                # docs/plans/stale-image-gate.md §5).
                return None
            # Every candidate was CONFIRMED stale: a fresh launch bakes the
            # current image by construction, so falling through to the launch
            # path below FIXES the condition rather than holding forever. The
            # refused box keeps its label and is left for the idle reaper /
            # `guard` — this builder has no teardown seam, and the next tick
            # adopts the fresh box because the stale one stays refused.

        # A RETIRED registry is refused before anything is resolved or rented.
        # This path does not go through `spec._require_image`, and the checks
        # below cannot stand in for it: an UNPINNED profile with a digest that
        # will not resolve proceeds by design (`verified_digest = None`), which
        # is right for "could not ask" and wrong for "this host is gone". The
        # cost of getting it wrong is a rented box in `loading` on
        # `denied: access forbidden` for the whole boot deadline.
        _host, _p, _t = imageref._split_image(str(profile.image or ""))
        if imageref.is_retired_registry(_host):
            raise WorkflowCtlError(
                f"{profile.image!r} (profile {stage.profile!r}) is on {_host}, "
                f"a RETIRED registry — cut 2026-08-22. It can never pull. "
                f"Repoint the profile at {imageref.R2_REGISTRY_HOST}")
        dg = digest_verifier(profile.image)  # type: ignore[no-untyped-call]
        if profile.image_digest is not None:
            if dg is None:
                raise WorkflowCtlError(
                    f"cannot verify pinned image digest for {profile.image!r} "
                    f"(profile {stage.profile!r}); refusing to launch an "
                    f"unverifiable of-record image")
            if dg != profile.image_digest:
                raise WorkflowCtlError(
                    f"IMAGE_DRIFT: {profile.image!r} resolved to digest "
                    f"{dg!r}, pinned to {profile.image_digest!r} "
                    f"(profile {stage.profile!r})")
        verified_digest = dg if dg is not None else profile.image_digest

        # inet_down floor biases toward hosts that can actually pull a cold
        # multilayer image inside BOOT_DEADLINE_S (found live 2026-07-15: a
        # slow-flow host never finished the train-image pull in 1200s). Paired
        # with retry_on=("infrastructure",) so a slow host that slips through is
        # replaced on a fresh attempt rather than killing the workflow.
        # Host rotation: skip every machine a PRIOR attempt of this stage already
        # landed on (recorded on box_acquired, below) so a retry after a boot
        # kill / infra failure never loops back onto the same slow/broken host
        # (pick_cheapest_offer's `exclude_machines` -> machine_id notin filter).
        exclude = _prior_stage_machines(wf_id, stage.name, runner=runner, bucket=bucket)
        offer = offer_picker(gpu=profile.gpu, num_gpus=profile.num_gpus,
                              gpu_ram_gb=profile.gpu_ram_gb,
                              max_dph=profile.max_bid, rental=profile.rental,
                              inet_down=LAUNCH_INET_DOWN_FLOOR_MBPS,
                              exclude_machines=(sorted(exclude) or None),
                              geo=(profile.geo or None))
        if offer is None:
            return None

        # Compose a REAL jobs box, not a bare one (defect #6, roadmap
        # 2026-07-15): the jobd-boot onstart prelude + scoped B2 cred env — the
        # identical `--jobs` composition the manual CLI path uses — so the box
        # runs jobd at boot and CLAIMS its deterministic queued job. Without
        # this the old body launched a daemon-less box: no jobd, no job ever
        # claimed, billing until teardown. `compose_jobs_launch_env` mutates
        # `env` in place and stages the content-addressed jobd bundle via
        # `bootstrap_stager`, returning the onstart-with-prelude + bundle sha.
        # The score stage (needs.venv:eval) needs nothing more at boot: jobd's
        # check_venv self-provisions /workspace/eval/dc3 via fetch_eval_env.sh
        # on claim, using these same B2 creds — no eval-env onstart wiring here.
        env = {}
        if verified_digest:
            env[imageref.IMAGE_DIGEST_ENV] = verified_digest
        # Gap B parity prelude: build ssh + hf exactly like `_do_launch`, in the
        # SAME order (ssh then hf), then hand it to the composer which prepends
        # jobd-boot -> final onstart is jobd_boot + ssh + hf. Parity-only for this
        # workflow (SSH via runtype=ssh_direct, weights B2-staged) and MUST NOT
        # raise when a key/token is absent (keeps the fakes-based tests green).
        prelude = ""
        try:
            tok = hf_token_provider()
        except Exception:
            tok = None
        if tok:
            env.setdefault("HF_TOKEN", tok)
            prelude = launch.hf_login_snippet() + prelude
        try:
            pub = ssh_pubkey_provider()
        except Exception:
            pub = None
        if pub:
            # ONE snippet for every launch path (2026-07-31): a bare append left
            # the file owned by vast's host user and sshd's StrictModes then
            # refused it — see ssh.ssh_authorized_keys_snippet.
            prelude = ssh.ssh_authorized_keys_snippet(pub) + prelude
        onstart, bootstrap_sha = jobs_composer(
            env, prelude or None, dry_run=dry_run, bootstrap_stager=bootstrap_stager)

        body = {
            "image": profile.image,
            "disk": profile.disk_gb or config.DISK_DEFAULT_WORKFLOW_GB,
            "runtype": "ssh_direct",
            "label": label,
            "env": env,
            "onstart": onstart,
        }
        price = offer.get("dph_total")
        if profile.rental == "bid":
            # NEVER bid the bare floor: a floor bid is preempted by any
            # competing bid one grid-step higher, and nothing in this layer
            # defends a standing bid afterward (found live 2026-07-20: the
            # e2-paired a2 box, bid exactly min_bid, was outbid ~1h into
            # generation). And do NOT clamp below the offer's `dph_total` the
            # way `_auto_bid_price` does — on a bid-type OFFER that field is
            # just floor+storage adders, not the machine's on-demand price, so
            # the clamp silently squashed the 1.2x headroom back to floor+~1¢
            # and boxes kept getting outbid mid-image-pull (2026-07-30, run
            # 2ed9: two consecutive score boxes lost during their pulls). The
            # profile's max_bid IS the deliberate ceiling here; the offer
            # search already filters min_bid <= max_bid, so the cap can never
            # fall below the floor.
            # WORKFLOW_BID_MULT: operator escape hatch for bid-sniped classes.
            # 2026-07-30, run 2ed9: three consecutive score boxes on the
            # razor-thin 3090/4090 class (floor $0.107) were outbid MID-PULL
            # even at 1.2x — a penny-sniper follows small headroom instantly,
            # while 3x floor (~$0.32/hr) is still pocket change for a
            # bounded scoring stage. Default unchanged.
            mult = float(os.environ.get("WORKFLOW_BID_MULT",
                                        bidpolicy.BID_TARGET_MULT))
            mb = models._num_dph(offer.get("min_bid"))
            if mb is not None:
                price = round(mb * mult, 3)
                if profile.max_bid is not None:
                    price = min(price, profile.max_bid)
            else:                           # unpriceable offer: legacy posture
                price = offer.get("min_bid")
            body["price"] = price
        # Private-registry pull auth: a privately-hosted of-record image (e.g.
        # the generate stage's train image) is FORBIDDEN without image_login, and
        # the box hangs in `loading` billing the whole time (found live
        # 2026-07-15: generate box stuck 1h05m on `denied: access forbidden`).
        # image_login_arg returns None for a PUBLIC image, so this used to be a
        # no-op for the score stage. It no longer is: since 2026-08-02 the score
        # stage runs the same private t211 image as generate (the eval env was
        # unified onto it), so BOTH stages now need auth. Derived from the
        # signing secret in the env (a REGISTRY_AUTH_SECRET mint for the R2
        # registry, the only one left), same as the `herdd launch` path —
        # never stored in the durable spec.
        login = image_login_provider(profile.image)
        if login:
            body["image_login"] = login

        ok, cid, err = launcher(offer["id"], body)
        if not ok:
            raise WorkflowCtlError(f"launch failed: {err}")

        emit(wf_id, "box_acquired", actor, runner=runner, bucket=bucket,
             ts=wm.now_ts(), stage=stage.name, attempt=attempt, instance_id=cid,
             adopted=False, profile=stage.profile, image=profile.image,
             image_digest=verified_digest, offer=offer["id"], price=price,
             bootstrap_sha=bootstrap_sha, machine_id=offer.get("machine_id"))
        # fleetd registration (FLEETD_DESIGN §6, review B1c): a workflow stage
        # box is babysat IN-PROCESS by this controller, so it would otherwise
        # look "unwatched" to the daemon. Registering it as `bare` (observation
        # + alarms, no money moves) means fleetd never fights the controller,
        # and the box is still visible in `fleet status`/`fleet spend`. Chosen
        # over a label-token exemption because the stage label is parsed
        # elsewhere (`run:<stage_job_id>` must stay exactly that). Best-effort:
        # no daemon, no problem — this must never fail a stage launch.
        try:
            fleet_client.fleet_watch_best_effort(
                cid, "bare", policy={"workflow": wf_id, "stage": stage.name,
                                     "in_process_supervisor": "workflowctl"})
        except Exception:
            pass
        return cid

    return box_resolver


# moved-from: workflowctl._default_box_starter
def _default_box_starter(iid: object) -> tuple[bool, Any]:
    """M3-T2's default `box_starter(iid) -> (ok, err)`: resume a parked/
    outbid instance in place, PUT v0/instances/{iid}/ {"state": "running"} —
    the exact shape `herdd._put_state_soft(iid, "running")` already uses
    (that primitive is intentionally NOT called here: the roadmap packet asks
    for a closure over `api.request_soft` directly so this seam never
    depends on a private herdd name). Soft by contract: never raises, never
    sys.exits; a caller (`reconcile_active_box`) treats a False `ok` as an
    advisory failure, not a terminal."""
    ok, d, err = api.request_soft("PUT", f"v0/instances/{iid}/",
                                      {"state": "running"})
    if not ok:
        return False, err
    if isinstance(d, dict) and d.get("success") is False:
        return False, str(d.get("msg") or d)
    return True, None


# moved-from: workflowctl.STOPPED_STATES
STOPPED_STATES = {"exited", "stopped"}   # local: parked/outbid, resumable


# moved-from: workflowctl.build_box_observer
def build_box_observer(*, instances_reader: Seam | None = None) -> Seam:
    """M3-T2 `box_observer(instance_id) -> 'live'|'stopped'|'gone'|'unknown'`.

    CRITICAL: reads the vast instance list via a reader that SURFACES API
    failure — NOT `herdd._instances_soft` (which returns [] on error and
    would masquerade every owned box as 'gone', triggering a false retarget
    -> DOUBLE SPEND). The default reader closes over `api.request_soft`
    and returns `(ok, list)`; `not ok` maps to 'unknown' (transient !=
    eviction). `instances_reader` is injectable for tests."""
    def _default_reader() -> tuple[bool, list[Any]]:
        ok, d, _err = api.request_soft("GET", "v1/instances/", retries=2)
        if not ok:
            return False, []
        insts = d.get("instances", d) if isinstance(d, dict) else d
        return True, (insts if isinstance(insts, list) else [])

    reader = instances_reader if instances_reader is not None else _default_reader

    def _find(insts: list[Any], instance_id: object) -> Any:  # noqa: ANN401 — one raw vast instance record
        return next((i for i in insts
                     if str(i.get("id")) == str(instance_id)), None)

    def box_observer(instance_id: object) -> str:
        ok, insts = reader()
        if not ok:
            return "unknown"                       # transient != eviction
        inst = _find(insts, instance_id)
        if inst is None:
            # CONFIRM before declaring 'gone'. A single eventually-consistent
            # list omission of a still-live box would otherwise trigger a
            # false retarget -> a DUPLICATE launch (double-spend) plus the
            # original orphaned from teardown (billing leak). 'gone' drives an
            # irreversible relaunch, so require a SECOND independent HEALTHY
            # read to ALSO omit the box; a transient failure or a reappearance
            # on the confirm read falls back to the safe non-acting states
            # (`unknown`/`live`/`stopped`). This layers atop box_resolver's own
            # ADOPT re-read, so a false retarget now needs three consecutive
            # stale reads rather than one.
            ok2, insts2 = reader()
            if not ok2:
                return "unknown"                   # transient on confirm read
            inst = _find(insts2, instance_id)
            if inst is None:
                return "gone"
        status = (inst.get("actual_status") or "").lower()
        if status in bidpolicy.LIVE_STATES:
            return "live"
        if status in STOPPED_STATES:
            return "stopped"
        return "unknown"                            # boot/None/other: don't act

    return box_observer


# moved-from: workflowctl.LiveCostObserver
class LiveCostObserver:
    """Caller-owned `{instance_id: st}` object `accrue_and_persist_cost`
    consumes UNCHANGED: it calls `.get(iid)`, then `run_lane._accrue_cost(st)`,
    then `self[iid] = st`. `.get(iid)` OBSERVES (refreshes `st` fields
    `_accrue_cost` reads) but does NOT accrue — the accrue/observe split
    matches `_accrue_cost`'s contract.

    A per-tick snapshot of the instance list is memoized (one GET, short TTL
    via injected clock) so N owned boxes cost ~1 GET/tick. `spend_usd`
    persists across ticks (monotonic: only a live tick with dt>0 adds
    dph/3600*dt), so the summed cost is monotonic and `folded_spend`'s
    fold-takes-MAX survives a controller restart. A parked/not-live box adds
    nothing (consistent with `_accrue_cost` §5). `dt=0` on first observe -> no
    phantom spend."""

    def __init__(self, *, instances_reader: Seam | None = None,
                 clock: Seam | None = None, ttl_s: float | None = None) -> None:
        self._st: dict[Any, Any] = {}
        self._clock = clock or time.time
        self._ttl_s = ttl_s if ttl_s is not None else (POLL_INTERVAL_S / 3.0)
        self._snap: list[Any] | None = None
        self._snap_t: float | None = None
        self._reader = (instances_reader if instances_reader is not None
                         else self._default_reader)

    @staticmethod
    def _default_reader() -> tuple[bool, list[Any]]:
        ok, d, _err = api.request_soft("GET", "v1/instances/", retries=2)
        if not ok:
            return False, []
        insts = d.get("instances", d) if isinstance(d, dict) else d
        return True, (insts if isinstance(insts, list) else [])

    def _snapshot(self) -> list[Any]:
        now = self._clock()
        if (self._snap is not None and self._snap_t is not None
                and (now - self._snap_t) < self._ttl_s):
            return self._snap
        ok, insts = self._reader()
        if not ok:
            # transient read failure: keep any prior snapshot rather than
            # zeroing every box present->False (which would drop accrual).
            if self._snap is not None:
                return self._snap
            return []
        self._snap = insts
        self._snap_t = now
        return insts

    def get(self, iid: object) -> dict[str, Any]:
        st = self._st.get(iid)
        if st is None:
            st = {"spend_usd": 0.0, "_last_obs_t": None, "present": False}
            self._st[iid] = st
        insts = self._snapshot()
        inst = next((i for i in insts if str(i.get("id")) == str(iid)), None)
        now = self._clock()
        last = st.get("_last_obs_t")
        st["dt"] = 0.0 if last is None else max(0.0, now - last)
        st["_last_obs_t"] = now
        if inst is None:
            st["present"] = False
            st["actual_status"] = None
            st["dph_total"] = None
        else:
            st["present"] = True
            st["actual_status"] = (inst.get("actual_status") or "").lower()
            st["dph_total"] = models._num_dph(inst.get("dph_total"))
        return st

    def __setitem__(self, iid: object, st: Any) -> None:  # noqa: ANN401 — caller-owned st dict
        self._st[iid] = st

    def __contains__(self, iid: object) -> bool:
        return iid in self._st


# moved-from: workflowctl.build_cost_observer
def build_cost_observer(*, instances_reader: Seam | None = None,
                        clock: Seam | None = None,
                        ttl_s: float | None = None) -> LiveCostObserver:
    """Construct the `LiveCostObserver` `accrue_and_persist_cost` feeds on.
    Pure — no network I/O at build time; the first GET happens on the first
    `.get(iid)` inside a reconcile tick."""
    return LiveCostObserver(instances_reader=instances_reader, clock=clock,
                             ttl_s=ttl_s)


# moved-from: workflowctl.build_live_controller_deps
def build_live_controller_deps(wf: Workflow, wf_id: str, *, actor: str,
                               runner: Runner | None = None, bucket: str | None = None,
                               dry_run: bool = False) -> dict[str, Any]:
    """Construct the live dependency bundle for `run_controller`.
    PURE closure construction — performs ZERO network I/O at build time (so
    the CLI resume path can build deps before `claim_controller` decides
    whether it will even run). `cred_provider` stays None: no production
    B2-horizon provider exists in-repo, so rotate-on-resume remains an
    intentional no-op (out of scope)."""
    return {
        "box_resolver": build_box_resolver(
            wf_id=wf_id, actor=actor, runner=runner, bucket=bucket,
            digest_verifier=imageref.image_ref_digest, dry_run=dry_run),
        "box_teardown": build_box_teardown(),
        "box_observer": build_box_observer(),
        "box_starter": _default_box_starter,
        "cost_observer": build_cost_observer(),
        "cred_provider": None,
        # Boot-throughput watchdog: a per-tick sampler holding per-instance
        # state across ticks (BOOT_HEALTHCHECK phase P0). Its own deadline is
        # left effectively infinite (deadline_s default) so it only ever
        # surfaces the 'slow' condemn verdict — the fixed BOOT_DEADLINE_S
        # backstop stays owned by reconcile_active_box's _boot_deadline_action.
        "throughput_observer": health.build_throughput_observer(),
        # velvet P3 stale-image gate (docs/plans/stale-image-gate.md). Given the
        # SAME digest function the box resolver above uses, so both gates read
        # one TTL cache entry per image and can never disagree within a tick.
        "image_state_observer": build_image_state_observer(
            digest_verifier=imageref.image_ref_digest),
    }


# moved-from: workflowctl._stage_bundle_dir
def _stage_bundle_dir(stage: JobStage) -> str:
    """CWD-relative (or already-absolute) resolution of `stage.bundle`, the
    same convention `herdd.py`'s plain `job submit --dir` uses for a
    bundle directory — no new repo-root-relative scheme invented here."""
    return os.path.abspath(stage.bundle)


# --- M4-T1 plan preflight: offline bundle/wiring validation + online seams ---
# moved-from: workflowctl._repo_root
def _repo_root() -> str:
    """The repo checkout root — what the default `asset_checker` needs to locate
    the local sources `jobmeta.check_asset_staleness` compares a MUTABLE B2 asset
    against. Reads the module constant; see `_REPO_ROOT` for why the depth is not
    the flat file's."""
    return _REPO_ROOT


# moved-from: workflowctl._default_asset_checker
def _default_asset_checker(assets: list[Any], *, runner: Runner | None,
                           bucket: str | None) -> Any:  # noqa: ANN401 — `jobmeta.check_asset_staleness` findings are untyped
    """The real (non-test) `asset_checker`: `jobmeta.check_asset_staleness`
    bound to this repo checkout's root."""
    return jobmeta.check_asset_staleness(  # type: ignore[no-untyped-call]
        assets, repo_root=_repo_root(), runner=runner, bucket=bucket)


# moved-from: workflowctl._validate_stage_bundle
def _validate_stage_bundle(stage: JobStage) -> dict[str, Any]:
    """Offline plan preflight (roadmap "offline plan validates every
    expanded child config and bundle"): load+validate `stage`'s bundle
    config the SAME way `herdd.py cmd_job_submit`/`_build_stage_config` do
    (`jobmeta.load_job_config` + `jobmeta.validate_job_config`), but WITHOUT
    `_build_stage_config`'s upstream-artifact wiring — at plan time no stage
    has run yet, so `read_accepted_artifact` would always be None and that
    function would always raise. Returns the validated `cfg` dict; raises
    `WorkflowCtlError` on any bundle problem (unreadable dir, bad job.json,
    a `validate_job_config` rejection)."""
    bundle_dir = _stage_bundle_dir(stage)
    try:
        raw = jobmeta.load_job_config(bundle_dir)
        cfg, _warnings = jobmeta.validate_job_config(raw, bundle_dir)
    except jobmeta.JobmetaError as e:
        raise WorkflowCtlError(
            f"stage {stage.name!r}: invalid bundle config at {bundle_dir!r}: {e}") from e
    # VRAM sizing, HERE and not at stage submit. `job submit` refuses a
    # provably-undersized bundle before spending; a workflow's stages are
    # submitted one at a time, hours in, unattended — refusing there strands a
    # run mid-flight over a number that was already wrong at plan time. `plan`
    # is the workflow's own pre-spend gate ($0, no box), so it is where the
    # refusal belongs and where the operator can still fix the bundle.
    try:
        _vram = jobmeta.vram_gate_findings(cfg)
    except Exception:
        _vram = None                       # advice must never break a plan
    lines, refuse = jobmeta.vram_gate_report(_vram)  # type: ignore[no-untyped-call]
    if refuse:
        raise WorkflowCtlError(
            f"stage {stage.name!r}: needs.gpu_ram_gb is below a peak this shape "
            f"has already measured — " + "; ".join(ln.strip() for ln in lines))
    return cfg


# moved-from: workflowctl._check_stage_inputs
def _check_stage_inputs(stage: JobStage, stages_by_name: dict[str, JobStage]) -> None:
    """Confirm every `InputRef` in `stage.inputs` points at a stage listed
    in `stage.after` whose `artifact` is a declared `output` of that
    upstream stage. This duplicates (deliberately) the same cross-object
    rule `wm.validate_workflow_spec` already enforced inside
    `load_workflow_module` — re-run here so `plan_workflow`'s own
    per-stage `CONFIG_INVALID` reporting owns its error shape rather than
    depending on that earlier, differently-shaped `WorkflowSpecError`.
    Raises `WorkflowCtlError` on the first violation."""
    for input_name, ref in sorted(stage.inputs.items()):
        if ref.stage not in stage.after:
            raise WorkflowCtlError(
                f"stage {stage.name!r} input {input_name!r} references stage "
                f"{ref.stage!r}, which is not in its declared after={stage.after!r} "
                f"(input without dependency)")
        upstream = stages_by_name.get(ref.stage)
        if upstream is None or ref.artifact not in upstream.outputs:
            raise WorkflowCtlError(
                f"stage {stage.name!r} input {input_name!r} references artifact "
                f"{ref.artifact!r}, which stage {ref.stage!r} does not declare "
                f"as an output")


# moved-from: workflowctl._build_stage_config
def _build_stage_config(wf_id: str, stage: JobStage, *, runner: Runner | None,
                        bucket: str | None) -> tuple[dict[str, Any], str]:
    """Load + validate `stage`'s job-config from its bundle directory
    (`jobmeta.load_job_config`/`validate_job_config` — the SAME validation
    `herdd.py cmd_job_submit` runs, never duplicated), then wire every
    declared `InputRef` as the ordinary asset block (roadmap "Artifact
    binding") from the upstream ACCEPTED artifact record. Returns
    (config, bundle_dir). Raises WorkflowCtlError on a bad bundle or a
    missing upstream acceptance (the latter should be impossible — ready-
    stage selection already required every `after` dependency to have
    reached `stage_succeeded` before this stage can be planned)."""
    bundle_dir = _stage_bundle_dir(stage)
    try:
        raw = jobmeta.load_job_config(bundle_dir)
        cfg, _warnings = jobmeta.validate_job_config(raw, bundle_dir)
    except jobmeta.JobmetaError as e:
        raise WorkflowCtlError(
            f"stage {stage.name!r}: invalid bundle config at {bundle_dir!r}: {e}") from e

    assets = list(cfg.get("assets") or [])
    for input_name, ref in sorted(stage.inputs.items()):
        art = read_accepted_artifact(wf_id, ref.stage, ref.artifact,
                                      runner=runner, bucket=bucket)
        if art is None:
            raise WorkflowCtlError(
                f"stage {stage.name!r} input {input_name!r}: upstream artifact "
                f"{ref.stage}/{ref.artifact} was never accepted")
        # `manifest_path` rode into the accepted record when the upstream
        # contract was accepted (workdir-relative, same frame as the asset's
        # `jobs/<gen_job_id>/results` b2 prefix); older records without it
        # keep the e2 default byte-identically.
        require = wm.require_from_manifest(
            art["manifest"],
            manifest_rel=art.get("manifest_path") or "results/artifact-manifest.json")
        assets.append(wm.input_ref_asset(
            ref, gen_job_id=art["job_id"], manifest_sha256=art["manifest_sha256"],
            require=require))
    if assets:
        cfg = dict(cfg)
        cfg["assets"] = assets
    return cfg, bundle_dir


# moved-from: workflowctl._ensure_bundle_uploaded
def _ensure_bundle_uploaded(bundle_dir: str, *, runner: Runner | None,
                            bucket: str | None) -> str:
    """Content-address `bundle_dir` (`jobmeta.bundle_sha256`, pure/local) and
    upload it if the dedupe check (`jobmeta.bundle_exists`) says B2 doesn't
    already have it — same dedupe-then-upload sequence as
    `herdd.py cmd_job_submit`, staged under this module's own local cache
    dir rather than the CLI's `out/jobs/_bundles`."""
    sha = jobmeta.bundle_sha256(bundle_dir)
    try:
        exists = jobmeta.bundle_exists(sha, runner=runner, bucket=bucket)
    except runmeta.RunmetaError as e:
        raise WorkflowCtlError(f"bundle_exists check failed: {e}") from e
    if not exists:
        tmp_dir = os.path.join(_lock_dir(), "bundles")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_out = os.path.join(tmp_dir, f"{sha}.tar.zst")
        jobmeta.write_bundle(bundle_dir, tmp_out)
        ok, err = jobmeta.upload_bundle(tmp_out, sha, runner=runner, bucket=bucket)  # type: ignore[no-untyped-call]
        if not ok:
            raise WorkflowCtlError(f"bundle upload failed for {bundle_dir!r}: {err}")
    return sha


# moved-from: workflowctl._plan_and_submit_stage
def _plan_and_submit_stage(wf: Workflow, wf_id: str, stage: JobStage, attempt: int,
                           *, runner: Runner | None, bucket: str | None, actor: str,
                           now: str, box_resolver: Seam,
                           cred_provider: CredProvider | None = None) -> dict[str, Any]:
    """Roadmap step 6 ("plan and submit the first ready stage") — also
    reused by step 5's retry path (a new attempt is planned+submitted the
    same way a fresh stage is; both mint a deterministic JOB_ID and call
    `jobmeta.submit_with_id`, which is itself idempotent). Returns
    `{'action': 'need_box', ...}` without touching config/bundle I/O at all
    when `box_resolver` (default `_default_box_resolver`, the M3 seam)
    has no box to offer — never a live vast spend from this layer.

    `cred_provider` (M3-T2, default `None` -> skipped entirely) gates BOTH
    call sites (step 6 and the retry path) via `_check_credential_horizon`
    BEFORE `box_resolver` is ever called — no box acquisition, no job
    ticket, on a confirmed-insufficient credential horizon."""
    if cred_provider is not None:
        gate = _check_credential_horizon(wf, wf_id, stage, now, cred_provider,
                                          actor=actor, runner=runner, bucket=bucket)
        if gate is not None:
            return gate
    box = box_resolver(stage, wf, attempt)
    if not box:
        return {"action": "need_box", "stage": stage.name, "attempt": attempt}

    job_id = wm.stage_job_id(wf_id, stage.name, attempt)
    cfg, bundle_dir = _build_stage_config(wf_id, stage, runner=runner, bucket=bucket)

    # B2 write-scope gate — same seam as `herdd job submit` / `jobmatrix
    # submit` / rehearse.sh (jobmeta.b2_write_preflight). The workflow lane has
    # its own submit path, so without this the ONE surface that runs unattended
    # for hours would be the one surface that cannot see an unentitled write.
    # Pure and offline: it happens before the bundle upload and before any
    # ticket, so refusing costs nothing.
    _wf_lines, _wf_refuse = jobmeta.b2_write_scope_report(  # type: ignore[no-untyped-call]
        jobmeta.b2_write_preflight(cfg, bundle_dir))
    if _wf_refuse:
        raise WorkflowCtlError(
            f"stage {stage.name!r}: B2 write scope — " + "; ".join(_wf_lines))

    sha = _ensure_bundle_uploaded(bundle_dir, runner=runner, bucket=bucket)

    # Write the durable job ticket FIRST (idempotent for this deterministic
    # job_id), THEN record the workflow-level plan/submit events. A controller
    # crash — or a `submit_with_id` conflict raise — AFTER a `stage_planned`
    # emit but BEFORE the ticket is durable would strand a `stage_planned`
    # marker (job_id set, status in STAGE_INFLIGHT) pointing at a job that was
    # never queued: step 4 then read_job-folds it as `unknown` and returns
    # `noop_running` forever — a deadlock. With the ticket written first, a
    # resume finds no workflow stage events yet, re-enters via next_ready_stage,
    # and re-submits idempotently (the ticket identity matches -> no-op).
    submit_result = jobmeta.submit_with_id(job_id, cfg, box, bundle_sha256=sha,
                                            actor=actor, runner=runner, bucket=bucket)
    plan_ev = emit(wf_id, "stage_planned", actor, runner=runner, bucket=bucket, ts=now,
                   stage=stage.name, attempt=attempt, job_id=job_id, box=box)
    submit_ev = emit(wf_id, "stage_submitted", actor, runner=runner, bucket=bucket, ts=now,
                     stage=stage.name, attempt=attempt, job_id=job_id, box=box)
    return {"action": "stage_submitted", "stage": stage.name, "attempt": attempt,
            "job_id": job_id, "box": box, "submit": submit_result,
            "plan_event": plan_ev, "submit_event": submit_ev}


# moved-from: workflowctl._accept_stage_artifacts
def _accept_stage_artifacts(wf_id: str, stage: JobStage, stage_view: dict[str, Any],
                            job_id: str, *, runner: Runner, bucket: str | None,
                            actor: str, now: str) -> dict[str, Any]:
    """Roadmap step 4's artifact-postcondition case: `stage`'s job just went
    `done`. Accept each declared output ONE AT A TIME (one idempotent action
    per tick — `read_accepted_artifact` is the idempotency check, so
    re-ticking after a killed/recreated controller resumes from whichever
    output was accepted last, never re-validates or re-emits one already
    written). Once every declared output is accepted (vacuously true for a
    stage with none), emits `stage_succeeded`."""
    for art_name in sorted(stage.outputs):
        if read_accepted_artifact(wf_id, stage.name, art_name,
                                  runner=runner, bucket=bucket) is not None:
            continue    # already accepted this tick's predecessor -- skip
        contract = stage.outputs[art_name]
        try:
            # contract.manifest_path is WORKDIR-RELATIVE (the bundle's
            # `results:` frame) — jobmeta resolves it under
            # jobs/<job_id>/results/. The e2 default
            # 'results/artifact-manifest.json' reproduces the historical
            # double-'results' key byte-identically.
            result = jobmeta.validate_generation_artifact(
                job_id, expect_kind=contract.kind, runner=runner, bucket=bucket,
                manifest_path=contract.manifest_path)
        except jobmeta.JobmetaError as e:
            ev = emit(wf_id, "stage_failed", actor, runner=runner, bucket=bucket, ts=now,
                     stage=stage.name, attempt=stage_view.get("attempt"), job_id=job_id,
                     failure_class="ARTIFACT_INVALID", reason=str(e))
            return {"action": "artifact_rejected", "stage": stage.name,
                    "artifact": art_name, "event": ev}
        record = {
            "v": 1, "stage": stage.name, "artifact": art_name, "job_id": job_id,
            "manifest_sha256": result["manifest_sha256"], "manifest": result["manifest"],
            "manifest_path": contract.manifest_path,
            "accepted_ts": now,
        }
        body = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        rc, _, err = runner(
            ["rcat", _q(bucket, _artifact_key(wf_id, stage.name, art_name))], input=body)
        if rc != 0:
            raise WorkflowCtlError(
                f"stage {stage.name!r}: writing accepted artifact {art_name!r} "
                f"failed: {(err or '').strip()}")
        ev = emit(wf_id, "artifact_accepted", actor, runner=runner, bucket=bucket, ts=now,
                 stage=stage.name, attempt=stage_view.get("attempt"), job_id=job_id,
                 artifact=art_name, manifest_sha256=result["manifest_sha256"])
        return {"action": "artifact_accepted", "stage": stage.name,
                "artifact": art_name, "event": ev}

    ev = emit(wf_id, "stage_succeeded", actor, runner=runner, bucket=bucket, ts=now,
             stage=stage.name, attempt=stage_view.get("attempt"), job_id=job_id)
    return {"action": "stage_succeeded", "stage": stage.name, "event": ev}


# moved-from: workflowctl._teardown_boxes
def _teardown_boxes(owned: list[str], mode: str, box_teardown: Seam, wf_id: str,
                    actor: str, now: str, runner: Runner | None,
                    bucket: str | None) -> tuple[list[str], list[str]]:
    """Call the injected `box_teardown(instance_id, mode) -> bool` for every
    owned box, emitting `box_released` for each success. `box_teardown`
    itself is the M3-owned real stop/destroy transport (never invoked from
    here in production today — `owned` is always empty until M3-T1 wires
    `box_acquired` events with a real `instance_id`)."""
    released, failed = [], []
    for iid in owned:
        try:
            ok = bool(box_teardown(iid, mode))
        except Exception:
            ok = False
        if ok:
            emit(wf_id, "box_released", actor, runner=runner, bucket=bucket, ts=now,
                 instance_id=iid, mode=mode)
            released.append(iid)
        else:
            failed.append(iid)
    return released, failed


# --- M3-T2 teardown reconcile: retry-until-done + bounded TEARDOWN_FAILED ----
# `box_released` (frozen name, emitted by `_teardown_boxes`) never carries a
# `stage` field, so it is invisible to `wm.fold_workflow_events`'s per-stage
# sub-fold -- `owned_boxes_remaining` below does its own small local scan of
# `read_events` (per the roadmap packet) rather than teaching `workflowmeta`
# a new fold shape. This is also how a self-parked box (its OWN `box_released`
# landing concurrently, e.g. from jobd) is recognized as already-released:
# same event name, same fold, no special case.
# moved-from: workflowctl.TEARDOWN_MAX_ATTEMPTS
TEARDOWN_MAX_ATTEMPTS = 5

# workflowctl-LOCAL event ("teardown_attempt") -- same posture as `box_cost`:
# deliberately NOT added to `workflowmeta.EVENTS` (that frozen V1 set is
# `wm`'s alone to grow); the fold already tolerates any unknown event name as
# inert, so this needs no `workflowmeta.py` change. One per tick where
# `_teardown_boxes` was actually invoked against a nonempty remaining set --
# the durable, restart-safe attempt counter `TEARDOWN_MAX_ATTEMPTS` is judged
# against.


# moved-from: workflowctl._released_instance_ids
def _released_instance_ids(wf_id: str, *, runner: Runner | None = None,
                           bucket: str | None = None) -> set[str]:
    """Every instance_id that already has a `box_released` event in the
    fold -- a small local scan, not part of `wm.fold_workflow_events` (see
    module note above)."""
    runner = _resolve_runner(runner)
    released = set()
    for raw in read_events(wf_id, runner=runner, bucket=bucket):
        ev = wm._coerce(raw)
        if ev is not None and ev.get("event") == "box_released" and ev.get("instance_id"):
            released.add(str(ev["instance_id"]))    # match _owned_instance_ids'
    return released                                 # str normalization


# moved-from: workflowctl.owned_boxes_remaining
def owned_boxes_remaining(v: dict[str, Any], wf_id: str, *, runner: Runner | None = None,
                          bucket: str | None = None) -> list[str]:
    """`_owned_instance_ids(v)` MINUS every instance_id that already has a
    `box_released` event -- the durable "what still needs tearing down" set.
    A box released by ANY actor (this controller's own `_teardown_boxes`, or
    a racing self-park from jobd) is excluded identically: `_teardown_boxes`
    must never be called on it again (idempotent by construction, since
    every caller below passes this function's output, never raw `owned`)."""
    runner = _resolve_runner(runner)
    owned = set(_owned_instance_ids(v))
    if not owned:
        return []
    return sorted(owned - _released_instance_ids(wf_id, runner=runner, bucket=bucket))


# moved-from: workflowctl._teardown_attempts_seen
def _teardown_attempts_seen(wf_id: str, *, runner: Runner | None = None,
                            bucket: str | None = None) -> int:
    """Restart-durable count of `teardown_attempt` events emitted so far for
    `wf_id` -- a FRESH `read_events` scan every call (same discipline as
    `folded_spend`), never an in-memory counter, so a killed/recreated
    controller resumes the bound exactly where the store left it."""
    runner = _resolve_runner(runner)
    return sum(
        1 for raw in read_events(wf_id, runner=runner, bucket=bucket)
        if (ev := wm._coerce(raw)) is not None and ev.get("event") == "teardown_attempt")


# moved-from: workflowctl._teardown_failed_recorded
def _teardown_failed_recorded(wf_id: str, *, runner: Runner | None = None,
                              bucket: str | None = None) -> bool:
    """True once a `workflow_failed`/`TEARDOWN_FAILED` terminal supplement has
    already been emitted for `wf_id` -- guards against re-emitting it every
    subsequent tick while still best-effort retrying the actual box
    stop/destroy calls forever (a stuck box that later frees up should still
    get released; the roadmap's "bounded" language is about the ONE failure
    verdict, not about giving up on the underlying teardown)."""
    runner = _resolve_runner(runner)
    for raw in read_events(wf_id, runner=runner, bucket=bucket):
        ev = wm._coerce(raw)
        if (ev is not None and ev.get("event") == "workflow_failed"
                and ev.get("failure_class") == "TEARDOWN_FAILED"):
            return True
    return False


# moved-from: workflowctl._attempt_teardown
def _attempt_teardown(wf: Workflow, wf_id: str, remaining: list[str], box_teardown: Seam,
                      *, actor: str, now: str, runner: Runner | None,
                      bucket: str | None) -> tuple[list[str], list[str], list[str], int]:
    """Call `_teardown_boxes` (never forked) over exactly `remaining` — always
    `owned_boxes_remaining`'s output, never the raw `owned` set, so an
    already-released box (this controller's or a racing self-park's) is
    never passed to `box_teardown` again. Durably records one
    `teardown_attempt` event so `TEARDOWN_MAX_ATTEMPTS` survives a controller
    restart. Returns `(remaining_after, released, failed, attempts_seen)`."""
    released, failed = _teardown_boxes(
        remaining, wf.teardown, box_teardown, wf_id, actor, now, runner, bucket)
    emit(wf_id, "teardown_attempt", actor, runner=runner, bucket=bucket, ts=now,
         attempted=remaining, released=released, failed=failed)
    remaining_after = [iid for iid in remaining if iid not in released]
    attempts_seen = _teardown_attempts_seen(wf_id, runner=runner, bucket=bucket)
    return remaining_after, released, failed, attempts_seen


# moved-from: workflowctl._render_provenance
def _render_provenance(wf: Workflow, wf_id: str, verdict: dict[str, Any], *,
                       now: str) -> dict[str, Any]:
    """Build the `provenance.json` body (roadmap M4-T3): the verdict's
    outcome plus a spec-content hash (`wm.canonical_spec_json`), so a
    provenance record can later be checked against the spec it was produced
    from without re-reading the full spec.json."""
    spec_sha256 = hashlib.sha256(wm.canonical_spec_json(wf).encode("utf-8")).hexdigest()
    stages = {
        name: {"job_id": sv.get("job_id"), "instance_id": sv.get("instance_id"),
               "status": sv.get("status")}
        for name, sv in (verdict.get("stages") or {}).items()
    }
    return {"v": 1, "workflow_id": wf_id, "ts": now, "spec_sha256": spec_sha256,
            "outcome": verdict.get("outcome"), "stages": stages}


# moved-from: workflowctl._render_report_md
def _render_report_md(wf_id: str, verdict: dict[str, Any]) -> str:
    """Short human-readable summary (roadmap M4-T3 `report.md`): workflow id,
    outcome, one line per stage. Pure string formatting, no I/O."""
    lines = [f"# Workflow {wf_id}", "", f"Outcome: {verdict.get('outcome')}", "", "## Stages"]
    stages = verdict.get("stages") or {}
    if not stages:
        lines.append("(no stages)")
    for name, sv in stages.items():
        sv = sv or {}
        lines.append(
            f"- {name}: status={sv.get('status')} attempt={sv.get('attempt')} "
            f"job={sv.get('job_id')} box={sv.get('instance_id')} "
            f"failure={sv.get('failure_class')}")
    return "\n".join(lines) + "\n"


# moved-from: workflowctl._reconcile_completion
def _reconcile_completion(wf: Workflow, wf_id: str, v: dict[str, Any], *,
                          all_succeeded: bool, actor: str, now: str, runner: Runner,
                          bucket: str | None,
                          box_teardown: Seam | None) -> dict[str, Any]:
    """Roadmap steps 7+8. Step 7 (write `verdict.json` + emit
    `teardown_started`) is gated on `verdict.json` NOT existing yet — the
    write-once existence of that object IS this function's idempotency
    check for "has step 7 already happened", so a killed/recreated
    controller replays step 7 exactly once. Once it exists, step 8
    reconciles `owned_boxes_remaining` (idempotent: never re-stops an
    already-released box) and only decides the workflow's REAL terminal
    (succeeded/RETRY_EXHAUSTED) once every owned box is released — a
    bounded teardown failure (`TEARDOWN_MAX_ATTEMPTS`) instead emits
    `workflow_failed`/`TEARDOWN_FAILED` as a terminal supplement while
    RETAINING the already-written verdict (never overwritten, never
    deleted). Boxes still stuck under the bound return `teardown_retrying`
    so a LATER tick gets another chance — never a premature terminal."""
    vkey = _verdict_key(wf_id)
    rc, existing, _ = runner(["cat", _q(bucket, vkey)])
    verdict_written = rc == 0 and bool((existing or "").strip())

    if not verdict_written:
        verdict = {
            "v": 1, "workflow_id": wf_id, "ts": now,
            "outcome": "succeeded" if all_succeeded else "failed",
            "stages": v.get("stages", {}),
        }
        body = json.dumps(verdict, sort_keys=True, separators=(",", ":")) + "\n"
        rc2, _, err = runner(["rcat", _q(bucket, vkey)], input=body)
        if rc2 != 0:
            raise WorkflowCtlError(f"writing verdict.json failed: {(err or '').strip()}")

        provenance = _render_provenance(wf, wf_id, verdict, now=now)
        pbody = json.dumps(provenance, sort_keys=True, separators=(",", ":")) + "\n"
        rc3, _, perr = runner(["rcat", _q(bucket, _provenance_key(wf_id))], input=pbody)
        if rc3 != 0:
            raise WorkflowCtlError(f"writing provenance.json failed: {(perr or '').strip()}")

        rbody = _render_report_md(wf_id, verdict)
        rc4, _, rerr = runner(["rcat", _q(bucket, _report_key(wf_id))], input=rbody)
        if rc4 != 0:
            raise WorkflowCtlError(f"writing report.md failed: {(rerr or '').strip()}")

        ev = emit(wf_id, "teardown_started", actor, runner=runner, bucket=bucket, ts=now)
        return {"action": "teardown_started", "verdict": verdict,
                "provenance": provenance, "event": ev}

    remaining = owned_boxes_remaining(v, wf_id, runner=runner, bucket=bucket)
    if remaining:
        if box_teardown is None:
            return {"action": "need_box_teardown", "boxes": remaining}
        remaining, released, failed, attempts_seen = _attempt_teardown(
            wf, wf_id, remaining, box_teardown, actor=actor, now=now,
            runner=runner, bucket=bucket)
        if remaining:
            if attempts_seen >= TEARDOWN_MAX_ATTEMPTS:
                # bounded terminal supplement -- verdict.json (written above,
                # this call or an earlier one) is NEVER touched here.
                ev = emit(wf_id, "workflow_failed", actor, runner=runner, bucket=bucket,
                         ts=now, failure_class="TEARDOWN_FAILED", boxes=remaining)
                return {"action": "workflow_failed", "failure_class": "TEARDOWN_FAILED",
                        "boxes": remaining, "event": ev}
            return {"action": "teardown_retrying", "remaining": remaining,
                    "released": released, "failed": failed}

    if all_succeeded:
        ev = emit(wf_id, "workflow_succeeded", actor, runner=runner, bucket=bucket, ts=now)
        return {"action": "workflow_succeeded", "event": ev}
    ev = emit(wf_id, "workflow_failed", actor, runner=runner, bucket=bucket, ts=now,
             failure_class="RETRY_EXHAUSTED")
    return {"action": "workflow_failed", "failure_class": "RETRY_EXHAUSTED", "event": ev}


# --- M3-T2 budget seam: durable cost accrual + enforcement -------------------
# workflowctl-LOCAL event pair -- deliberately NOT added to workflowmeta.EVENTS
# (that frozen V1 set is `wm`'s alone to grow). `box_cost` rides the same
# append-only `workflows/<wf_id>/events/` store every other event uses, and
# `wm.fold_workflow_events` already tolerates any unknown event name as inert
# (never breaks the status fold), so this needs no `workflowmeta.py` change.
# moved-from: workflowctl.record_box_cost
def record_box_cost(wf_id: str, cost_usd: float, *, actor: str,
                    runner: Runner | None = None, bucket: str | None = None,
                    **fields: Any) -> dict[str, Any]:  # noqa: ANN401 — free-form event body
    """Emit one `box_cost` event. `cost_usd` is always the CUMULATIVE-to-date
    spend (never a delta) -- mirrors `herdd._emit_cost`'s run-level `cost`
    event (SUPERVISE_DESIGN section 5: fold takes the max, so a late or
    duplicate emit from a restarted controller can never lower the folded
    budget)."""
    runner = _resolve_runner(runner)
    return emit(wf_id, "box_cost", actor, runner=runner, bucket=bucket,
                cost_usd=round(float(cost_usd), 4), **fields)


# moved-from: workflowctl.folded_spend
def folded_spend(wf_id: str, *, runner: Runner | None = None,
                 bucket: str | None = None) -> float:
    """The restart-durable budget source: RE-READ `read_events` fresh (never
    from in-memory state) and return the MAX `cost_usd` seen across every
    `box_cost` event -- cumulative-to-date, fold-takes-max, same discipline
    `record_box_cost` writes under. 0.0 when no `box_cost` event exists yet."""
    runner = _resolve_runner(runner)
    best = 0.0
    for raw in read_events(wf_id, runner=runner, bucket=bucket):
        ev = wm._coerce(raw)
        if ev is None or ev.get("event") != "box_cost":
            continue
        v = wm._num(ev.get("cost_usd"))
        if v is not None and v > best:
            best = v
    return best


# moved-from: workflowctl.budget_exhausted
def budget_exhausted(wf: Workflow, spent: float, *,
                     profile: ResourceProfile | None = None) -> bool:
    """Pure gate: true once cumulative `spent` has reached the workflow's
    budget cap, or (when a narrower `profile` is given) that profile's own
    cap. No I/O -- callers supply `spent` from `folded_spend`.

    A falsy (0/unset) cap means "NO cap" at BOTH levels -- matching the
    profile branch below and herdd's `budget_usd=None` sentinel. Critical:
    a workflow spec that omits `budget_usd` deserializes to 0.0
    (`workflowmeta`'s default), so an unguarded `spent >= wf.budget_usd`
    would fire `BUDGET_EXHAUSTED` on the FIRST tick of every unbudgeted
    workflow -- before any box, and in production where `folded_spend` is
    always 0.0 because cost accrual is not yet wired. Guard the workflow cap
    the same way the profile cap is already guarded."""
    if wf.budget_usd and spent >= wf.budget_usd:
        return True
    if profile is not None and profile.budget_usd and spent >= profile.budget_usd:
        return True
    return False


# moved-from: workflowctl.accrue_and_persist_cost
def accrue_and_persist_cost(wf: Workflow, wf_id: str, v: dict[str, Any], *, actor: str,
                            runner: Runner | None = None, bucket: str | None = None,
                            cost_observer: Any = None,  # noqa: ANN401 — caller-owned dict-like
                            ) -> dict[str, Any] | None:
    """Per-tick accrual seam. For each box this workflow currently owns
    (`_owned_instance_ids`), REUSES `run_lane._accrue_cost` (never
    reimplements the dph/3600*dt math) against that box's entry in the
    injected `cost_observer` -- a caller-owned `{instance_id: st}` dict kept
    warm across ticks by a real vast-observe loop, `st` shaped exactly as
    `run_lane._accrue_cost` expects (`present`/`actual_status`/`dph_total`/
    `dt`/`spend_usd`). Emits ONE cumulative `box_cost` event: the durable
    prior cumulative (`folded_spend`) PLUS this tick's per-owned-box accrual
    DELTA -- restart/retarget-durable, never a sum of the observer's in-memory
    absolute `spend_usd` (see the body comment for why the absolute form
    under-counts across box churn).

    `cost_observer=None` -- the default on every call today, same posture as
    `box_resolver`/`box_teardown` -- is a strict no-op: no box is touched, no
    event is emitted, no vast API is ever called from here. Wiring a real
    observer in is a later subtask's concern."""
    if cost_observer is None:
        return None
    owned = _owned_instance_ids(v)
    if not owned:
        return None
    # DURABLE-CUMULATIVE accrual: emit `prior_cumulative + this tick's per-box
    # DELTA`, never the sum of the live observer's in-memory ABSOLUTE
    # `spend_usd`. The absolute-sum form under-counts whenever a box leaves the
    # owned set between ticks: (a) a controller restart/resume rebuilds a fresh
    # observer with every `spend_usd` reseeded to 0, and (b) a retarget drops
    # the replaced box out of `_owned_instance_ids` (the fold keeps only the
    # newest instance_id per stage). folded_spend's fold-takes-MAX would then
    # STALL at the pre-churn peak while real billing keeps climbing on the
    # surviving/replacement boxes -- defeating the budget cap by up to the
    # pre-churn accrual, exactly the spot churn E2 is built to tolerate. Adding
    # only the per-tick delta onto the durable prior keeps the cumulative
    # monotonic and churn-proof: `_accrue_cost` yields dt=0 (delta 0) on any
    # box's FIRST observe, so a restart-reseeded or freshly-adopted box is
    # never double-counted against the prior it already contributed to.
    prior = folded_spend(wf_id, runner=runner, bucket=bucket)
    tick_delta = 0.0
    for iid in owned:
        st = cost_observer.get(iid)
        if st is None:
            continue
        before = st.get("spend_usd", 0.0) or 0.0
        st = run_lane._accrue_cost(st)
        cost_observer[iid] = st
        tick_delta += max(0.0, (st.get("spend_usd", 0.0) or 0.0) - before)
    return record_box_cost(wf_id, prior + tick_delta, actor=actor,
                            runner=runner, bucket=bucket)


# --- M3-T2 credential seam: expiry gate + rotate-on-resume -------------------
# `cred_provider` is the injected object/closure exposing:
#   current_expiry(instance_id_or_stage) -> epoch (float seconds)
#   rotate(name, ...) -> new_expiry_epoch (float seconds)
# `cred_provider=None` on every production reconcile_tick call today (same
# posture as box_resolver/box_teardown/cost_observer): a strict no-op, no B2
# minter call, existing tests byte-identical. Production wiring is a caller's
# closure over `b2_mint_key.mint`/`mint_pair` (never forked here — this
# module only calls the injected `cred_provider`, never `b2_mint_key`
# directly).
# moved-from: workflowctl.remaining_wall_s
def remaining_wall_s(profile: ResourceProfile, elapsed_s: float) -> float:
    """Pure: the stage's remaining wall-clock bound `elapsed_s` seconds into
    its run -- `max(0, profile.max_wall_s - elapsed_s)`. `elapsed_s=0` at
    plan/launch time (nothing spent yet) is `profile.max_wall_s` in full."""
    return max(0.0, float(profile.max_wall_s or 0) - float(elapsed_s))


# moved-from: workflowctl.credential_horizon_ok
def credential_horizon_ok(*, now_epoch: float, cred_expiry_epoch: float,
                           remaining_wall_s: float) -> bool:
    """Pure gate: True only when the credential outlives the stage's
    remaining wall bound (`cred_expiry_epoch - now_epoch >= remaining_wall_s`).
    No I/O -- callers resolve `cred_expiry_epoch` via the injected
    `cred_provider` and `now_epoch` via `wm._parse_ts(now).timestamp()`."""
    return (cred_expiry_epoch - now_epoch) >= remaining_wall_s


# moved-from: workflowctl._rotate_credential
def _rotate_credential(cred_provider: CredProvider, name: object) -> object:
    """Rotate the credential for `name` (production: a closure over
    `b2_mint_key.mint`/`mint_pair` -- never forked here) BEFORE continuing
    paid work on a resume/retarget. A genuine rotation failure (minter auth
    refusal, minter unreachable, ...) is a HARD STOP: raises
    `WorkflowCtlError` rather than being silently swallowed into a
    fabricated workflow terminal -- the CLI layer's existing
    `except WorkflowCtlError: sys.exit(EXIT_CREDENTIAL)` wrapping
    `run_controller`/`resume_workflow` is the operator-facing surface for
    this (roadmap "B2-auth-failure resilience")."""
    try:
        return cred_provider.rotate(name)
    except Exception as e:
        raise WorkflowCtlError(f"credential rotation failed for {name!r}: {e}") from e


# moved-from: workflowctl._check_credential_horizon
def _check_credential_horizon(wf: Workflow, wf_id: str, stage: JobStage, now: str,
                              cred_provider: CredProvider, *, actor: str,
                              runner: Runner | None,
                              bucket: str | None) -> dict[str, Any] | None:
    """Roadmap "credential-horizon gate before launch": consulted by
    `_plan_and_submit_stage` right before it would call `box_resolver` (step
    6 / the retry-submit path). Returns `None` when the credential clears the
    gate (proceed with acquisition), or a terminal `reconcile_tick`-shaped
    action dict otherwise.

    A transient failure reading the credential's expiry (the injected
    `current_expiry` raises, or returns something that isn't a plain number
    -- an auth/network hiccup, indistinguishable from a real B2 401/403 at
    this layer) is deliberately NOT a workflow terminal: it returns a
    `noop_credential_transient` action so the NEXT tick gets another chance
    (same "transient != eviction" posture `reconcile_active_box`'s
    `box_observer` already uses). Only a CONFIRMED insufficient horizon
    (the provider answered, and the answer says the credential won't
    outlast the stage) fails the workflow closed with `CREDENTIAL_EXPIRES`
    -- fail-closed, no box acquired, no job ticket written."""
    profile = wf.profiles[stage.profile]
    try:
        expiry = cred_provider.current_expiry(stage.name)
    except Exception:
        expiry = None
    if not isinstance(expiry, (int, float)) or isinstance(expiry, bool):
        return {"action": "noop_credential_transient", "stage": stage.name}

    now_epoch = wm._parse_ts(now).timestamp()
    remaining = remaining_wall_s(profile, 0.0)
    if credential_horizon_ok(now_epoch=now_epoch, cred_expiry_epoch=float(expiry),
                              remaining_wall_s=remaining):
        return None
    ev = emit(wf_id, "workflow_failed", actor, runner=runner, bucket=bucket, ts=now,
             failure_class="CREDENTIAL_EXPIRES", stage=stage.name)
    return {"action": "workflow_failed", "failure_class": "CREDENTIAL_EXPIRES",
            "event": ev}


# moved-from: workflowctl._RUNNING_JOB_STATUSES
_RUNNING_JOB_STATUSES = ("submitted", "claimed", "started", "unknown")


# moved-from: workflowctl._seconds_between
def _seconds_between(earlier: str | None, later: str | None) -> float | None:
    """Age in seconds between two runmeta timestamps, or `None` when either is
    missing/unparseable (the Gap D boot watchdog must NEVER fire on a bad or
    absent `box_acquired_ts` — a missing ts is 'don't know', i.e. don't act)."""
    if not earlier or not later:
        return None
    try:
        return wm._ts_diff_seconds(later, earlier)
    except Exception:
        return None


# moved-from: workflowctl.reconcile_active_box
def reconcile_active_box(wf: Workflow, wf_id: str, stage: JobStage,
                         active_sv: dict[str, Any],
                         *, actor: str, runner: Runner | None = None,
                         bucket: str | None = None,
                         box_observer: Seam | None = None, box_resolver: Seam | None = None,
                         box_starter: Seam | None = None,
                         cred_provider: CredProvider | None = None,
                         box_teardown: Seam | None = None,
                         throughput_observer: Seam | None = None,
                         image_state_observer: Seam | None = None,
                         now: str | None = None) -> dict[str, Any] | None:
    """M3-T2 in-flight box reconciliation (roadmap step 4 of `reconcile_tick`):
    resume a stopped box, replace a gone box under the SAME deterministic
    JOB_ID/attempt, or refuse to act on a transient observation failure —
    REUSING `build_box_resolver`'s existing ADOPT/launch primitives rather
    than forking a second acquisition path. Only meaningful when the active
    stage already owns an `instance_id` and its child job is still running
    (`status` in `submitted`/`claimed`/`started`/`unknown`); returns `None`
    in every other case so the caller falls through to its own ordinary
    `noop_running` handling.

    `box_observer(instance_id) -> 'live'|'stopped'|'gone'|'unknown'` is the
    M3-T2 seam (default `None` -> treated as `'unknown'`, i.e. a transient
    observation is indistinguishable from an unwired observer -- both are
    the safe "don't act" posture per SUPERVISE_DESIGN's transient-!=-eviction
    invariant).

    `image_state_observer(instance_id) -> (state, reason)` (velvet P3, default
    `None` -> gate unwired, the same opt-in posture every watchdog seam here
    uses; `build_live_controller_deps` wires the real one so production is
    armed). Consulted before a RESUME because a resume is exactly the moment an
    image cannot refresh itself — vast keeps the box's disk, so a stale box
    comes back just as stale as it parked.

    `cred_provider` (M3-T2, default `None` -> skipped entirely): a genuine
    resume ('stopped') or retarget ('gone') rotates the credential via
    `_rotate_credential` BEFORE calling `box_starter`/`box_resolver` -- never
    resumes/relaunches paid work on a stale key. A rotation failure raises
    `WorkflowCtlError` (propagates out of this call, uncaught, same as
    `claim_controller`'s refusal) rather than being folded into `action` --
    the CLI layer maps it to `EXIT_CREDENTIAL`.
    """
    runner = _resolve_runner(runner)
    instance_id = active_sv.get("instance_id")
    job_id = active_sv.get("job_id")
    if not instance_id or not job_id:
        return None

    jv = jobmeta.read_job(job_id, runner=runner, bucket=bucket)
    if jv.get("status") not in _RUNNING_JOB_STATUSES:
        return None    # done/failed/cancelled -- steps 4/5's own logic owns this

    state = "unknown"
    if box_observer is not None:
        try:
            observed = box_observer(instance_id)
        except Exception:
            observed = None
        if observed in ("live", "stopped", "gone", "unknown"):
            state = observed

    # Gap D boot/claim watchdog (defect #6 cost backstop): a box whose child
    # job is STILL 'submitted' (jobd never claimed it) more than BOOT_DEADLINE_S
    # after `box_acquired` is an infra failure -- the box never ran jobd, hung
    # in `loading`, or otherwise idle-bills without taking its job. This fires
    # for BOTH a box observed 'live' (running/loading/created -- the PRIMARY
    # booted-but-unclaimed signature) AND an 'unknown'/transient observation:
    # the discriminator is the never-claimed job + deadline, not the box's
    # observed power state. Tear it down and fail THIS attempt as an
    # infrastructure failure so the ordinary retry/stage_failed policy (`failed`
    # job -> _classify_job_failure 'infrastructure' -> wm.decide_retry) bounds
    # repeated boot hangs: a retryable stage gets a fresh attempt+box, an
    # exhausted one terminates as INFRASTRUCTURE_FAILED rather than looping
    # teardown+relaunch forever. Gated strictly on status=='submitted'
    # (never-claimed) + an injected box_teardown, so a briefly-unobservable-but-
    # already-claimed box (status 'claimed'/'started'/'unknown') is never torn
    # down. Returns the box_boot_failed action, or None when the box is healthy.
    def _boot_deadline_action() -> dict[str, Any] | None:
        acq_ts = active_sv.get("box_acquired_ts")
        age = _seconds_between(acq_ts, now)
        if not (jv.get("status") == "submitted" and box_teardown is not None
                and age is not None and age > BOOT_DEADLINE_S):
            return None
        try:
            box_teardown(instance_id, "destroy")
        except Exception:
            pass
        jobmeta.emit_event(
            job_id, "failed", actor=actor, runner=runner, bucket=bucket,
            reason=(f"boot deadline exceeded ({int(age)}s > {BOOT_DEADLINE_S}s): "
                    "jobd never claimed the job (box hung/loading timeout)"))
        return {"action": "box_boot_failed", "stage": stage.name,
                "instance_id": instance_id, "job_id": job_id,
                "age_s": int(age), "boot_deadline_s": BOOT_DEADLINE_S}

    # Boot THROUGHPUT watchdog (BOOT_HEALTHCHECK phase P0) — sibling to the
    # fixed-deadline action, sharing its contract and its retry-bounding path,
    # but firing EARLY (t >= one BOOT_MBPS_WINDOW_S, ~5-6 min) on a provably
    # starved image pull instead of waiting out the whole BOOT_DEADLINE_S. Gated
    # identically: status=='submitted' (jobd never claimed) + an injected
    # box_teardown + a `throughput_observer` that returns a 'slow' verdict. The
    # observer holds the per-instance sampler state across ticks (build_
    # throughput_observer); here we only ACT on its condemn verdict. Emits the
    # same `failed` job event -> _classify_job_failure infra -> wm.decide_retry
    # so repeats terminate as INFRASTRUCTURE_FAILED, never loop teardown+launch.
    # Composes with, never replaces, _boot_deadline_action (both stay armed).
    def _boot_throughput_action() -> dict[str, Any] | None:
        if not (jv.get("status") == "submitted" and box_teardown is not None
                and throughput_observer is not None):
            return None
        try:
            res = throughput_observer(instance_id)
        except Exception:
            res = None
        if not (isinstance(res, dict) and res.get("verdict") == "slow"):
            return None
        mbps = res.get("mbps")
        window_s = res.get("window_s")
        try:
            box_teardown(instance_id, "destroy")
        except Exception:
            pass
        mbps_txt = f"{mbps:.2f}" if isinstance(mbps, (int, float)) else "?"
        jobmeta.emit_event(
            job_id, "failed", actor=actor, runner=runner, bucket=bucket,
            reason=(f"boot throughput floor ({mbps_txt} MB/s < "
                    f"{int(BOOT_MIN_MBPS)} MB/s over {window_s}s): slow host"))
        return {"action": "box_boot_failed", "stage": stage.name,
                "instance_id": instance_id, "job_id": job_id,
                "mbps": mbps, "window_s": window_s,
                "machine_id": res.get("machine_id"), "reason": "boot_throughput_floor"}

    box_starter = box_starter if box_starter is not None else _default_box_starter
    box_resolver = box_resolver if box_resolver is not None else _default_box_resolver

    def _retarget(reason: str | None = None,
                  **extra: Any) -> dict[str, Any]:  # noqa: ANN401 — free-form event body
        """Shared 'replace the box under the SAME attempt' move ('gone' path +
        the heartbeat watchdog): rotate cred, resolve a replacement, and move
        the job ticket to the new box's queue. Launching the replacement box is
        NOT enough: jobd claims per-box from jobs/queue/<iid>/, so without the
        retarget the ticket is stranded in the dead box's queue, the new box's
        jobd idles forever, and the next tick sees a LIVE box + a
        still-'submitted' job and reports noop_running — a silent deadlock
        (found live 2026-07-15, generate stage). No-op if box_resolver adopted
        the same box; idempotent if a prior tick already moved it. Durably
        emitted (box_retargeted) so the WHY of a replacement is forensically
        visible next to its box_acquired — the 2026-07-20 5h-blind window left
        no durable trace of what the controller observed."""
        if cred_provider is not None:
            _rotate_credential(cred_provider, instance_id)
        attempt = active_sv.get("attempt") or 0
        new_iid = box_resolver(stage, wf, attempt)
        if new_iid and str(new_iid) != str(instance_id):
            jobmeta.retarget_ticket(job_id, instance_id, new_iid, actor=actor,
                                     runner=runner, bucket=bucket)
        emit(wf_id, "box_retargeted", actor, runner=runner, bucket=bucket,
             ts=wm.now_ts(), stage=stage.name, attempt=attempt,
             old_instance_id=instance_id, instance_id=new_iid,
             reason=reason, **extra)
        action = {"action": "box_retargeted", "stage": stage.name,
                  "instance_id": new_iid, "job_id": job_id, "attempt": attempt}
        if reason:
            action["reason"] = reason
        action.update(extra)
        return action

    # Mid-run liveness watchdog (the 2026-07-20 5h-blind fix): a STARTED job
    # whose jobd heartbeats (~60s cadence) have gone silent past
    # JOB_HEARTBEAT_STALE_S is presumed dead REGARDLESS of the box's observed
    # power state — a preempted spot box can linger listed as 'stopped' (resume
    # attempts can never win while outbid) or stale-'running' for hours, and
    # the 'gone'-only trigger left the workflow blind the whole time. Teardown
    # + retarget under the SAME attempt (job resumes from checkpoints/).
    # STARTED only, never 'claimed': jobd's heartbeat loop launches only after
    # it emits `started` — the claimed->started window is silent-by-design
    # asset/venv staging (the score stage legitimately stages for 10+ min), so
    # a claimed-but-quiet job is indistinguishable from a healthy slow stage.
    # Gated on an injected box_teardown (same opt-in posture as the boot
    # watchdogs) and CONFIRMED by a second independent fold read — a
    # transient/partial event listing must never trigger an irreversible
    # teardown+retarget (same two-read posture as box_observer's 'gone').
    def _heartbeat_stale(view: dict[str, Any]) -> float | None:
        if view.get("status") != "started":
            return None
        base = (view.get("last_heartbeat_ts") or view.get("last_resumed_ts")
                or view.get("started_at"))
        age = _seconds_between(base, now)
        if age is not None and age > JOB_HEARTBEAT_STALE_S:
            return age
        return None

    def _heartbeat_dead_action() -> dict[str, Any] | None:
        if box_teardown is None:
            return None
        age = _heartbeat_stale(jv)
        if age is None:
            return None
        jv2 = jobmeta.read_job(job_id, runner=runner, bucket=bucket)
        age2 = _heartbeat_stale(jv2)
        if age2 is None:
            return None
        try:
            box_teardown(instance_id, "destroy")
        except Exception:
            pass
        return _retarget(reason="job_heartbeat_stale",
                         heartbeat_age_s=int(age), observed_state=state)

    def _stale_image_resume_action() -> dict[str, Any] | None:
        """velvet P3: refuse to RESUME a box whose baked env predates the
        current image tag. A resume is the one moment an image provably cannot
        refresh itself — vast keeps the disk, so the box comes back exactly as
        stale as it parked (imageref: "a park/resume will NOT refresh it"), and
        the job it then claims dies the way the three frontier-wave jobs died
        on box 46240842 on 2026-07-30.

        Fully automatic path, so no `--allow-stale-image` warn-and-proceed
        here; the two refusals split on whether we KNOW:
          * `stale` (confirmed mismatch) -> `_retarget`: a fresh box bakes the
            current image, and holding a box that can never become fresh is
            strictly worse than replacing it. The stopped box is left to the
            idle reaper rather than destroyed — the workflow's ticket has moved
            off it, and no teardown seam is required to make the gate correct.
          * `unresolved` (could not compare) -> HOLD: no resume AND no launch.
            Auto-launching here would turn a registry/API outage into one
            rented box per tick. Returning the held action re-checks for free
            on the next tick.
        Unwired observer (`None`) -> no verdict -> unchanged behavior."""
        if image_state_observer is None:
            return None
        try:
            # The profile pin, when the workflow declared one, is the reference
            # this box OUGHT to match — never the live tag (_instance_image_
            # verdict's docstring: comparing an of-record box against a moved
            # tag refuses the resume and then IMAGE_DRIFTs its replacement).
            profile = wf.profiles.get(stage.profile)
            verdict = image_state_observer(
                instance_id,
                pinned_digest=getattr(profile, "image_digest", None))
        except Exception:
            verdict = (imageref.IMG_UNRESOLVED,
                       "image-state observer raised — cannot compare")
        if isinstance(verdict, tuple):
            img_state, why = (list(verdict) + [None])[:2]
        else:
            img_state, why = verdict, None
        if not _image_gate_refuses(img_state):
            return None
        if img_state == imageref.IMG_UNRESOLVED:
            # Deliberately NOT a durable event: this repeats every tick for the
            # length of an outage. The action dict lands in the controller's
            # journald tick line, which is where per-tick observations belong.
            return {"action": "box_resume_held", "stage": stage.name,
                    "instance_id": instance_id, "job_id": job_id,
                    "failure_class": IMAGE_GATE_FAILURE_CLASS[img_state],
                    "image_state": img_state, "image_reason": why}
        return _retarget(reason="stale_image", image_state=img_state,
                         image_reason=why,
                         failure_class=IMAGE_GATE_FAILURE_CLASS[img_state])

    if state == "live":
        # A booted container that idle-bills but never claimed its job reports
        # 'live' (running/loading), NOT 'unknown' -- so the watchdog MUST run
        # here or the primary defect-#6 leak is never bounded (was: an
        # unconditional `return None`, leaving the watchdog unreachable for the
        # exact booted-but-unclaimed state it was written to catch).
        # Throughput watchdog first (condemns EARLIER than the fixed deadline),
        # then the fixed-deadline backstop, then the mid-run heartbeat watchdog
        # (a host-reclaimed box can keep reporting 'running' while dead).
        boot_failed = _boot_throughput_action() or _boot_deadline_action()
        if boot_failed is not None:
            return boot_failed
        dead = _heartbeat_dead_action()
        if dead is not None:
            return dead
        return None    # box healthy -- fall through to the existing noop_running

    if state == "stopped":
        # Heartbeats stale past the threshold: this parked box is NOT coming
        # back (an outbid spot box's resume re-enters at the losing bid and
        # parks again — the exact resume-forever loop of 2026-07-20). Replace
        # it instead of poking it.
        dead = _heartbeat_dead_action()
        if dead is not None:
            return dead
        # Stale-image gate BEFORE the credential rotation and the resume PUT:
        # rotating a credential for a box we are about to refuse spends a mint
        # for nothing, and the resume itself is the money move being gated.
        gated = _stale_image_resume_action()
        if gated is not None:
            return gated
        if cred_provider is not None:
            _rotate_credential(cred_provider, instance_id)
        ok, err = box_starter(instance_id)
        action = {"action": "box_resumed", "stage": stage.name,
                  "instance_id": instance_id, "job_id": job_id, "ok": ok, "err": err}
    elif state == "gone":
        action = _retarget()
    else:   # "unknown" -- explicit-None observer OR a transient/attach failure
        # Same boot/claim watchdog as the 'live' path above: an 'unknown'
        # observation past the deadline with a never-claimed job is equally a
        # hung box (this branch remains reachable on a transient instances-API
        # read failure). Fire the throughput watchdog then the fixed-deadline
        # backstop, then the mid-run heartbeat watchdog, else fall through to
        # the unchanged noop_running.
        boot_failed = _boot_throughput_action() or _boot_deadline_action()
        if boot_failed is not None:
            return boot_failed
        dead = _heartbeat_dead_action()
        if dead is not None:
            return dead
        action = {"action": "noop_running", "stage": stage.name,
                  "job_status": jv.get("status")}

    alarm = risk._ckpt_watchdog_alarm(jv, time.time())
    if alarm:
        action["ckpt_alarm"] = alarm    # advisory only -- never changes `action`
    return action


# moved-from: workflowctl.reconcile_tick
def reconcile_tick(wf: Workflow, wf_id: str, *, runner: Runner | None = None,
                   bucket: str | None = None, actor: str,
                   now: str, box_resolver: Seam | None = None,
                   box_teardown: Seam | None = None,
                   cost_observer: Any = None,  # noqa: ANN401 — caller-owned dict-like
                   box_observer: Seam | None = None, box_starter: Seam | None = None,
                   cred_provider: CredProvider | None = None,
                   throughput_observer: Seam | None = None,
                   image_state_observer: Seam | None = None) -> dict[str, Any]:
    """Perform AT MOST ONE idempotent reconcile action for `wf_id`, in the
    frozen 8-step order (roadmap "Workflow state and events"). Returns
    `{'action': <name>, ...}`.

    V1 simplification (documented, not hidden): at most one job is ever "in
    flight" at a time — matches every "the active stage" (singular)
    reference in the roadmap step list. An independent DAG branch that is
    already ready still waits until nothing else is in flight, even though
    nothing here forbids it in principle; lifting that is a straightforward
    but out-of-scope-for-M2-T3 follow-up.

    `box_resolver`/`box_teardown` are the M3 seam (see `_default_box_resolver`):
    left uninjected in every production call today, so a stage that needs a
    box reports `{'action': 'need_box', ...}` and a completed workflow that
    still owns boxes (always none today) reports `{'action':
    'need_box_teardown', ...}` rather than fabricating success.

    `cost_observer` (M3-T2 budget seam) is `None` on every production call
    today (same posture as `box_resolver`/`box_teardown`): a strict no-op
    that neither touches a box nor calls a real vast API. `folded_spend`
    still re-reads whatever `box_cost` events already exist on B2 every
    tick, so budget enforcement is durable across a controller restart even
    while `cost_observer` stays uninjected.

    `box_observer`/`box_starter` (M3-T2 recovery seam, `reconcile_active_box`)
    are `None` on every production call today, same posture as every other
    M3 seam: step 4's in-flight branch below only calls
    `reconcile_active_box` when `box_observer` is actually injected, so a
    caller that never passes it gets the byte-identical pre-M3-T2 behavior.

    `cred_provider` (M3-T2 credential seam) is `None` on every production
    call today, same posture: skipped entirely by `_plan_and_submit_stage`
    (step 6 / the retry path) and by `reconcile_active_box` (resume/retarget
    rotation) unless actually injected -- existing tests stay byte-identical.
    """
    runner = _resolve_runner(runner)
    box_resolver = box_resolver or _default_box_resolver

    # --- Step 1: fold workflow events + per-stage-attempt state -------------
    v = view(wf_id, runner=runner, bucket=bucket)
    stages_view = v.get("stages", {})
    stages_by_name = {s.name: s for s in wf.stages}

    # --- Step 1.5: per-tick cost accrual + a fresh restart-durable spend read
    accrue_and_persist_cost(wf, wf_id, v, actor=actor, runner=runner, bucket=bucket,
                             cost_observer=cost_observer)
    spent = folded_spend(wf_id, runner=runner, bucket=bucket)

    # --- Step 2: terminal workflow -> reconcile teardown only ----------------
    # Reconciles REPEATEDLY across ticks (`owned_boxes_remaining` -- never a
    # box that already has a `box_released` event, whether released by THIS
    # controller's own prior attempt or a racing self-park) until every
    # owned box is released, or `TEARDOWN_MAX_ATTEMPTS` is exhausted -- a
    # bounded `workflow_failed`/`TEARDOWN_FAILED` terminal supplement, never
    # re-emitted once recorded (best-effort stop/destroy attempts keep going
    # regardless -- only the ONE failure verdict is bounded).
    if v.get("terminal"):
        remaining = owned_boxes_remaining(v, wf_id, runner=runner, bucket=bucket)
        if not remaining:
            return {"action": "noop_terminal", "status": v.get("status")}
        if box_teardown is None:
            return {"action": "need_box_teardown", "boxes": remaining}
        remaining, released, failed, attempts_seen = _attempt_teardown(
            wf, wf_id, remaining, box_teardown, actor=actor, now=now,
            runner=runner, bucket=bucket)
        if not remaining:
            return {"action": "teardown_reconciled", "released": released, "failed": failed}
        if (attempts_seen >= TEARDOWN_MAX_ATTEMPTS
                and not _teardown_failed_recorded(wf_id, runner=runner, bucket=bucket)):
            ev = emit(wf_id, "workflow_failed", actor, runner=runner, bucket=bucket, ts=now,
                     failure_class="TEARDOWN_FAILED", boxes=remaining)
            return {"action": "workflow_failed", "failure_class": "TEARDOWN_FAILED",
                    "boxes": remaining, "event": ev}
        return {"action": "teardown_retrying", "remaining": remaining,
                "released": released, "failed": failed}

    # `active_sv` starts as `{}` where the flat module started it at `None`.
    # Not a behavior change: every read below is either guarded by
    # `active_name is not None` (which is set in the same assignment) or
    # inside the `if active_name is not None:` block, so the None-typed
    # sentinel was unreachable. The empty dict is what lets the annotation
    # say so instead of scattering eight `union-attr` waivers.
    active_name: str | None = None
    active_sv: dict[str, Any] = {}
    for s in wf.stages:
        sv = stages_view.get(s.name)
        if sv is not None and sv.get("status") not in (None, "stage_succeeded"):
            active_name, active_sv = s.name, sv
            break

    # --- Step 3: honor cancellation (stop new submissions) -------------------
    cancel_requested = jobmeta.has_cancel_marker(wf_id, runner=runner, bucket=bucket)
    in_flight = active_name is not None and active_sv.get("status") in STAGE_INFLIGHT
    if cancel_requested and not in_flight:
        ev = emit(wf_id, "workflow_cancelled", actor, runner=runner, bucket=bucket, ts=now)
        return {"action": "workflow_cancelled", "event": ev}
    if cancel_requested:
        job_id = active_sv.get("job_id")
        if job_id and not jobmeta.has_cancel_marker(job_id, runner=runner, bucket=bucket):
            ok, err = jobmeta.write_cancel_marker(  # type: ignore[no-untyped-call]
                job_id, actor=actor, reason="workflow cancelled",
                runner=runner, bucket=bucket)
            return {"action": "stage_cancel_propagated", "stage": active_name,
                    "job_id": job_id, "ok": ok, "err": err}
        # kill already propagated to the child job -- fall through to step 4
        # so its eventual done/failed/cancelled outcome still gets reconciled
        # instead of stalling forever waiting on a cancellation-only tick.

    # --- Steps 4/5: reconcile the active stage -------------------------------
    if active_name is not None:
        stage = stages_by_name[active_name]
        status = active_sv.get("status")
        if status in STAGE_INFLIGHT:
            job_id = active_sv.get("job_id")
            if not job_id:
                # `box_acquired` is the ONLY STAGE_INFLIGHT status carrying no
                # job_id: box_resolver emits it (with the launched instance_id)
                # BEFORE the job ticket + stage_planned/stage_submitted are
                # durable. A controller death in that window strands the stage
                # here with a real, owned, BILLING box. Do NOT noop-await
                # forever (the box would idle-bill until a manual cancel) --
                # re-plan the SAME attempt: box_resolver ADOPTs the already-
                # labelled box (no duplicate launch), submit_with_id is
                # idempotent under the deterministic job_id, so the stranded
                # box is put to work and the stage falls through to normal
                # in-flight reconciliation on the next tick.
                return _plan_and_submit_stage(
                    wf, wf_id, stage, active_sv.get("attempt") or 0,
                    runner=runner, bucket=bucket, actor=actor, now=now,
                    box_resolver=box_resolver, cred_provider=cred_provider)
            jv = jobmeta.read_job(job_id, runner=runner, bucket=bucket)
            jstatus = jv.get("status")
            if jstatus in ("submitted", "claimed", "started", "unknown"):
                # IN-RUN budget gate (M3-T2): a still-running child job over
                # budget gets a cooperative kill via the SAME cancel-marker
                # path step 3 uses for an operator cancel -- its eventual
                # terminal (failed/cancelled) outcome folds through the
                # ordinary retry/fail logic below on a later tick.
                if budget_exhausted(wf, spent, profile=wf.profiles.get(stage.profile)):
                    ok, err = jobmeta.write_cancel_marker(  # type: ignore[no-untyped-call]
                        job_id, actor=actor, reason="workflow budget exhausted",
                        runner=runner, bucket=bucket)
                    return {"action": "stage_cancel_budget", "stage": active_name,
                            "job_id": job_id, "ok": ok, "err": err}
                if box_observer is not None:
                    recovery = reconcile_active_box(
                        wf, wf_id, stage, active_sv, actor=actor, runner=runner,
                        bucket=bucket, box_observer=box_observer,
                        box_resolver=box_resolver, box_starter=box_starter,
                        cred_provider=cred_provider, box_teardown=box_teardown,
                        throughput_observer=throughput_observer,
                        image_state_observer=image_state_observer, now=now)
                    if recovery is not None:
                        return recovery
                # Status observability (2026-07-15 audit): `stage_started` is
                # in the frozen V1 vocabulary and the fold's status ladder,
                # but nothing ever emitted it — `workflow status` showed a
                # 4h-RUNNING job as stage_submitted forever. Once the child
                # job is observed claimed/started, advance the folded stage
                # status ONCE (idempotent: after this emit the folded status
                # is 'stage_started', so this branch never re-fires).
                # `stage_planned` is included too (2026-07-15 adversarial
                # review): a controller death between the stage_planned and
                # stage_submitted emits leaves a claimable ticket whose fold
                # sits at stage_planned forever — a resume must still advance
                # it once the job is observed claimed/started. Both statuses
                # rank below stage_started in the fold ladder, so the
                # idempotency argument is unchanged.
                if (jstatus in ("claimed", "started")
                        and status in ("stage_planned", "stage_submitted")):
                    ev = emit(wf_id, "stage_started", actor, runner=runner,
                             bucket=bucket, ts=now, stage=active_name,
                             attempt=active_sv.get("attempt"), job_id=job_id)
                    return {"action": "stage_started", "stage": active_name,
                            "job_status": jstatus, "event": ev}
                return {"action": "noop_running", "stage": active_name,
                        "job_status": jstatus}
            if jstatus == "done":
                return _accept_stage_artifacts(
                    wf_id, stage, active_sv, job_id, runner=runner, bucket=bucket,
                    actor=actor, now=now)
            # jstatus in ("failed", "cancelled"): terminal-failed -> retry/fail
            retry_class = _classify_job_failure(jv)
            attempts_used = active_sv.get("attempts_seen") or \
                ((active_sv.get("attempt") or 0) + 1)
            decision = wm.decide_retry(
                stage, attempts_used=attempts_used, failure_class=retry_class)
            if decision == "retry":
                new_attempt = (active_sv.get("attempt") or 0) + 1
                return _plan_and_submit_stage(
                    wf, wf_id, stage, new_attempt, runner=runner, bucket=bucket,
                    actor=actor, now=now, box_resolver=box_resolver,
                    cred_provider=cred_provider)
            workflow_class = ("RETRY_EXHAUSTED" if retry_class in stage.retry.retry_on
                               else _FAILURE_CLASS_MAP[retry_class])
            ev = emit(wf_id, "stage_failed", actor, runner=runner, bucket=bucket, ts=now,
                     stage=active_name, attempt=active_sv.get("attempt"), job_id=job_id,
                     failure_class=workflow_class, fail_reason=jv.get("fail_reason"))
            return {"action": "stage_failed", "stage": active_name, "event": ev}
        # status is already stage-terminal (stage_failed/stage_cancelled) with
        # no further work possible for THIS stage -- fall through to steps
        # 6/7/8 so the rest of the workflow (or its completion) can resolve.

    # --- Step 6: plan+submit the first ready stage ---------------------------
    # cancel_requested implies an early return above on every reachable path
    # (see step 3): either finalized directly (not in_flight) or resolved via
    # noop_running/artifact-acceptance/retry/stage_failed above (in_flight).
    # So by construction we only reach here with cancellation not requested.
    next_stage_name = wm.next_ready_stage(wf, v)
    if next_stage_name is not None:
        stage = stages_by_name[next_stage_name]
        # PRE-LAUNCH budget gate (M3-T2): a stage that needs a fresh box
        # never gets one once the workflow (or the stage's own profile) is
        # over budget -- no job ticket is ever written on this path. Drives
        # the workflow straight to its BUDGET_EXHAUSTED terminal; step 2's
        # terminal->teardown-only path then parks any owned boxes on
        # subsequent ticks, same as every other workflow_failed route.
        if budget_exhausted(wf, spent, profile=wf.profiles.get(stage.profile)):
            ev = emit(wf_id, "workflow_failed", actor, runner=runner, bucket=bucket,
                     ts=now, failure_class="BUDGET_EXHAUSTED")
            return {"action": "workflow_failed", "failure_class": "BUDGET_EXHAUSTED",
                    "event": ev}
        return _plan_and_submit_stage(
            wf, wf_id, stage, 0, runner=runner, bucket=bucket, actor=actor, now=now,
            box_resolver=box_resolver, cred_provider=cred_provider)

    # --- Steps 7/8: completion -------------------------------------------------
    all_succeeded = all(
        stages_view.get(s.name, {}).get("status") == "stage_succeeded" for s in wf.stages)
    return _reconcile_completion(
        wf, wf_id, v, all_succeeded=all_succeeded, actor=actor, now=now,
        runner=runner, bucket=bucket, box_teardown=box_teardown)


# moved-from: workflowctl._terminal_exit_code
def _terminal_exit_code(status: str | None, failure_class: str | None = None) -> int:
    """The one place a folded terminal workflow `status` becomes a CLI exit
    code (roadmap "Failure classes and exit codes"): succeeded->0,
    cancelled->3. A nonterminal/unknown status maps to 0 too ("0 ...
    nonterminal status read") — callers check `terminal` themselves before
    deciding whether this mapping even applies to a wait/read.

    `failed` with a `failure_class` routes through `wm.failure_class_exit_code`
    (ARTIFACT_INVALID/POSTCONDITION_FAILED->4, CREDENTIAL_EXPIRES->5,
    WALL_EXHAUSTED->124, everything else incl. RETRY_EXHAUSTED->2); `failed`
    with no `failure_class` keeps the EXACT prior behavior (`EXIT_FAILED`)."""
    if status == "failed" and failure_class:
        return wm.failure_class_exit_code(failure_class)
    return {"succeeded": EXIT_OK, "failed": EXIT_FAILED,
            "cancelled": EXIT_CANCELLED}.get(status, EXIT_OK)  # type: ignore[arg-type]


# moved-from: workflowctl.spawn_detached
def spawn_detached(argv: Sequence[Any] | None, *,
                   wf_id: str | None = None) -> dict[str, Any]:
    """Build (and, unless probing fails, launch) a `systemd-run --user` unit
    that re-execs `argv` (the exact FOREGROUND command this same run would
    need) with `Restart=on-failure`, so a killed/rebooted session resumes
    the controller automatically. NEVER falls back to a hidden `nohup`: if
    `systemd-run` is unavailable, raises `DetachUnavailable` whose message
    IS that exact foreground command string, verbatim, for the operator to
    run instead.

    `argv` is the CLI layer's responsibility to build (e.g. `[sys.executable,
    ".../herdd.py", "workflow", "run", path]` minus `--detach`) — this
    function only turns it into a systemd unit."""
    if not argv:
        raise WorkflowCtlError("spawn_detached: argv must be a non-empty list")
    fg_cmd = " ".join(str(a) for a in argv)

    if shutil.which("systemd-run") is None:
        raise DetachUnavailable(fg_cmd)
    probe = subprocess.run(["systemd-run", "--user", "--version"],
                            capture_output=True, text=True)
    if probe.returncode != 0:
        raise DetachUnavailable(fg_cmd)

    unit = re.sub(r"[^A-Za-z0-9_.-]", "-", f"wfctl-{wf_id}" if wf_id else "wfctl")[:255]
    # --same-dir: transient units default WorkingDirectory to $HOME, which
    # breaks the repo-root-relative module path and .env resolution the
    # foreground command depends on (found live 2026-07-15: 5x crash-loop to
    # start-limit-hit, each restart exiting rc=1 before writing any event).
    cmd = (["systemd-run", "--user", "--same-dir", f"--unit={unit}",
            "--property=Restart=on-failure", "--"]
           + [str(a) for a in argv])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise WorkflowCtlError(
            f"systemd-run failed (rc={proc.returncode}): {proc.stderr.strip()}")
    return {"status": "detached", "unit": unit, "cmd": cmd}


# moved-from: workflowctl.run_controller
def run_controller(wf: Workflow, wf_id: str, *, runner: Runner | None = None,
                   bucket: str | None = None, actor: str,
                   detach: bool = False, takeover: bool = False,
                   max_ticks: int | None = None,
                   clock: Seam | None = None, sleep_fn: Seam | None = None,
                   box_resolver: Seam | None = None, box_teardown: Seam | None = None,
                   box_observer: Seam | None = None, box_starter: Seam | None = None,
                   cost_observer: Any = None,  # noqa: ANN401 — caller-owned dict-like
                   cred_provider: CredProvider | None = None,
                   throughput_observer: Seam | None = None,
                   image_state_observer: Seam | None = None,
                   argv: Sequence[Any] | None = None,
                   notifier: Seam | None = None,
                   detached_controller: bool = False) -> int:
    """Foreground reconcile loop for one workflow (or, `detach=True`, hand it
    off to `spawn_detached` and return immediately). Claims the controller
    role (`claim_controller`, refusing a second live controller unless
    `takeover`), then heartbeats + reconciles once per `POLL_INTERVAL_S`
    until the workflow is terminal or `max_ticks` bounds the loop (tests
    inject `clock`/`sleep_fn`/`max_ticks` so nothing really sleeps/loops).
    Always releases the local lock, even on an exception.

    `cred_provider` (M3-T2, default `None` -> skipped, same posture as
    `box_resolver`/`box_teardown`) is threaded into every `reconcile_tick`
    call so a resume/retarget mid-loop rotates its credential; `resume_
    workflow` additionally rotates ONCE up front before entering this loop.

    `notifier` (M4-T3, default `None` -> skipped) is called `notifier(wf_id,
    v)` exactly once, right before this function returns, ONLY after the
    terminal view/verdict is already durable on B2 — a POST-terminal,
    best-effort hook. Any exception it raises is swallowed; it can never
    change the returned exit code or the workflow's verdict.

    `detached_controller` (the `--detached-controller` flag the detach argv
    pinning appends to the systemd re-exec): this process IS the detached
    controller, so a TERMINAL workflow — success OR failure — returns
    EXIT_OK. The unit is `Restart=on-failure`; a real terminal exit code
    from it just flap-restarts a controller with nothing left to drive
    (live 2026-07-16, E2 run 0d9d: 57 `controller_started` events over
    9.5h, ~one per 4-6 min, all re-reading the same terminal-FAILED spec).
    The verdict is already durable in the B2 event log — the exit code is
    not an operator-visible signal for a detached run. The FOREGROUND
    run/resume path keeps returning the real terminal exit code."""
    if detach:
        spawn_detached(argv or [], wf_id=wf_id)
        return EXIT_OK

    runner = _resolve_runner(runner)
    clock = clock or wm.now_ts
    sleep_fn = sleep_fn or time.sleep

    handle = acquire_local_lock(wf_id, takeover=takeover)
    try:
        claim_controller(wf_id, actor, runner=runner, bucket=bucket,
                          now=clock(), takeover=takeover, clock=clock,
                          sleep_fn=sleep_fn)
        ticks = 0
        while True:
            heartbeat(wf_id, actor, runner=runner, bucket=bucket)
            act = reconcile_tick(wf, wf_id, runner=runner, bucket=bucket, actor=actor,
                           now=clock(), box_resolver=box_resolver,
                           box_teardown=box_teardown, box_observer=box_observer,
                           box_starter=box_starter, cost_observer=cost_observer,
                           cred_provider=cred_provider,
                           throughput_observer=throughput_observer,
                           image_state_observer=image_state_observer)
            # Every non-noop action to stdout (journald under a detached unit):
            # the 2026-07-20 5h-blind forensics found the reconcile action was
            # DISCARDED here — the durable event log alone showed nothing but
            # flat box_cost reads, and what the observer/resumer actually did
            # each tick was unrecoverable. One line per acting tick is the
            # cheapest possible flight recorder.
            if isinstance(act, dict) and act.get("action") not in (None, "noop_running"):
                print(f"[{clock()}] tick: "
                      f"{json.dumps(act, sort_keys=True, default=str)}", flush=True)
            v = view(wf_id, runner=runner, bucket=bucket)
            if v.get("terminal"):
                if notifier is not None:
                    try:
                        notifier(wf_id, v)
                    except Exception:
                        pass
                if detached_controller:
                    return EXIT_OK      # don't feed Restart=on-failure (docstring)
                return _terminal_exit_code(v.get("status"), v.get("failure_class"))
            ticks += 1
            if max_ticks is not None and ticks >= max_ticks:
                return EXIT_OK
            sleep_fn(POLL_INTERVAL_S)
    finally:
        release_local_lock(handle)


# --- M4-T2 canary key derivation: ONE place, shared by the gate + the launcher
# so a produced receipt lands exactly where the online gate looks it up. v1 sets
# model/adapter SHAs to '' (the image-probe canary stages no weights). Any change
# to this derivation must stay in this function or the two halves diverge.
# moved-from: workflowctl.stage_canary_key
def stage_canary_key(wf: Workflow, stage: JobStage, *, jobd_sha: str | None,
                     digests: dict[str, Any] | None = None,
                     stage_cfgs: dict[str, Any] | None = None) -> str:
    profile = wf.profiles[stage.profile]
    img = (digests or {}).get(stage.profile) or profile.image_digest
    recipe_sha = _canonical_sha256((stage_cfgs or {}).get(stage.name) or {})
    return canary_receipt_key(
        image_digest=img, jobd_sha=jobd_sha or "", model_manifest_sha="",
        adapter_manifest_sha="", recipe_sha=recipe_sha)


# moved-from: workflowctl.canary_launch_env
def canary_launch_env(wf: Workflow, stage: JobStage, *, jobd_sha: str | None,
                      digests: dict[str, Any] | None = None,
                      stage_cfgs: dict[str, Any] | None = None,
                      ttl_s: int = CANARY_RECEIPT_TTL_S) -> tuple[str, dict[str, str]]:
    """Return (key, env) for launching the workflow-canary bundle against
    `stage`. The env's CANARY_KEY is derived by `stage_canary_key` — the SAME
    function the online gate uses — so the receipt the bundle writes under that
    key is exactly the one `_plan_online` will later read as valid. This is the
    store-population half of the M4-T2 loop (the launcher/collector caller)."""
    key = stage_canary_key(wf, stage, jobd_sha=jobd_sha, digests=digests,
                           stage_cfgs=stage_cfgs)
    profile = wf.profiles[stage.profile]
    img = (digests or {}).get(stage.profile) or profile.image_digest
    recipe_sha = _canonical_sha256((stage_cfgs or {}).get(stage.name) or {})
    env = {"CANARY_KEY": key, "CANARY_IMAGE_DIGEST": img or "",
           "CANARY_JOBD_SHA": jobd_sha or "", "CANARY_MODEL_SHA": "",
           "CANARY_ADAPTER_SHA": "", "CANARY_RECIPE_SHA": recipe_sha,
           "CANARY_TTL_S": str(int(ttl_s))}
    return key, env


# --- M4-T1 online plan: B2/registry/credential/spend resolution (read-only) --
# moved-from: workflowctl._default_canary_checker
def _default_canary_checker(wf: Workflow, digests: dict[str, Any],
                            stage_cfgs: dict[str, Any], *, runner: Runner | None,
                            bucket: str | None, jobd_sha: str | None,
                            now_epoch: float | None) -> Seam:
    """Build the per-stage canary check used by `_plan_online`. Returns
    `check(stage) -> (status, key)`, deriving the key via the shared
    `stage_canary_key`. jobd_sha is computed once via the pure dry-run bootstrap
    with stdout suppressed (its progress line would corrupt `plan --online`
    JSON)."""
    if jobd_sha is None:
        with contextlib.redirect_stdout(io.StringIO()):
            jobd_sha = bundle._stage_jobd_bootstrap(dry_run=True)

    def check(stage: JobStage) -> tuple[str, str]:
        key = stage_canary_key(wf, stage, jobd_sha=jobd_sha, digests=digests,
                               stage_cfgs=stage_cfgs)
        status, _ = canary_receipt_status(
            key, runner=runner, bucket=bucket, now_epoch=now_epoch)
        return status, key

    return check


# moved-from: workflowctl._plan_online_canary
def _plan_online_canary(wf: Workflow, checker: Seam) -> tuple[int, dict[str, Any]]:
    """Require a valid, unexpired, passing canary receipt for every stage before
    live spend (roadmap M4-T2; the REHEARSAL_DISCLAIMER names this gate). First
    non-valid stage is surfaced, same fail-fast posture as the rest of
    `_plan_online`. failure_class distinguishes missing / expired / failed."""
    _FC = {"missing": "CANARY_MISSING", "expired": "CANARY_EXPIRED",
           "failed": "CANARY_FAILED"}
    canary = {}
    for stage in wf.stages:
        status, key = checker(stage)
        canary[stage.name] = {"status": status, "key": key}
        if status != "valid":
            fc = _FC.get(status, "CANARY_MISSING")
            return EXIT_ARTIFACT, {
                "canary": canary, "stage": stage.name, "key": key,
                "error": (f"{fc}: no valid canary receipt for stage "
                          f"{stage.name!r} (key {key}); run the workflow-canary "
                          f"bundle for this image+recipe before live spend"),
                "failure_class": fc}
    return EXIT_OK, {"canary": canary}


# moved-from: workflowctl._plan_online
def _plan_online(wf: Workflow, stage_cfgs: dict[str, Any], *, runner: Runner | None,
                 bucket: str | None, asset_checker: Seam | None,
                 image_resolver: Seam | None, cred_provider: CredProvider | None,
                 now_epoch: float | None, canary_checker: Seam | None = None,
                 jobd_sha: str | None = None) -> tuple[int, dict[str, Any]]:
    """`plan_workflow(..., online=True)`'s resolution body. Returns
    `(exit_code, report_dict)` — never raises; every resolver call is
    injected so this function itself never touches subprocess/rclone/HTTP
    directly. Order matches the roadmap bullet order (assets, image digests,
    credential horizon, spend) so the FIRST failure encountered is always
    the one surfaced, same "fail fast, one action at a time" posture as
    `reconcile_tick`."""
    runner = _resolve_runner(runner)
    checker = asset_checker if asset_checker is not None else _default_asset_checker
    resolver = image_resolver if image_resolver is not None else imageref.image_tag_digest

    assets_report = {}
    for stage in wf.stages:
        assets = list((stage_cfgs.get(stage.name) or {}).get("assets") or [])
        findings = checker(assets, runner=runner, bucket=bucket)
        lines, refuse = jobmeta.asset_preflight_report(findings, strict=True)  # type: ignore[no-untyped-call]
        assets_report[stage.name] = {"findings": findings, "lines": lines}
        if refuse:
            return EXIT_INVALID, {
                "assets": assets_report,
                "error": "; ".join(lines) or f"stage {stage.name!r}: stale asset(s) on B2",
                "stage": stage.name, "failure_class": "ASSET_STALE"}

    digests = {}
    for name, profile in sorted(wf.profiles.items()):
        dg = resolver(profile.image)  # type: ignore[no-untyped-call]
        digests[name] = dg
        if dg is None or dg != profile.image_digest:
            return EXIT_INVALID, {
                "assets": assets_report, "digests": digests,
                "error": (f"IMAGE_DRIFT: {profile.image!r} resolved to digest {dg!r}, "
                          f"pinned to {profile.image_digest!r} (profile {name!r})"),
                "profile": name, "failure_class": "IMAGE_DRIFT"}

    # M4-T2 canary gate: after digests (the key binds the resolved digest),
    # before credential/spend — no live spend without a valid canary receipt.
    checker = canary_checker if canary_checker is not None else _default_canary_checker(
        wf, digests, stage_cfgs, runner=runner, bucket=bucket,
        jobd_sha=jobd_sha, now_epoch=now_epoch)
    canary_rc, canary_report = _plan_online_canary(wf, checker)
    if canary_rc != EXIT_OK:
        return canary_rc, {
            "assets": assets_report, "digests": digests, **canary_report}

    cred_report = _plan_online_credentials(wf, cred_provider, now_epoch)
    if cred_report.get("failure_class") == "CREDENTIAL_EXPIRES":
        return EXIT_CREDENTIAL, {
            "assets": assets_report, "digests": digests,
            "canary": canary_report["canary"], "credential": cred_report,
            "error": cred_report["error"], "stage": cred_report.get("stage"),
            "failure_class": "CREDENTIAL_EXPIRES"}

    worst_case_spend_usd = sum(
        (wf.profiles[s.profile].budget_usd or 0.0) for s in wf.stages)
    report = {"assets": assets_report, "digests": digests,
              "canary": canary_report["canary"], "credential": cred_report,
              "worst_case_spend_usd": worst_case_spend_usd, "budget_usd": wf.budget_usd}
    if wf.budget_usd and worst_case_spend_usd > wf.budget_usd:
        report["error"] = (f"BUDGET_EXHAUSTED: worst-case spend ${worst_case_spend_usd:.2f} "
                           f"exceeds workflow budget_usd ${wf.budget_usd:.2f}")
        report["failure_class"] = "BUDGET_EXHAUSTED"
        return EXIT_INVALID, report
    return EXIT_OK, report


# moved-from: workflowctl._plan_online_credentials
def _plan_online_credentials(wf: Workflow, cred_provider: CredProvider | None,
                             now_epoch: float | None) -> dict[str, Any]:
    """Per-stage credential-horizon check (`credential_horizon_ok`, pure).
    `cred_provider=None` (no real production credential-horizon provider
    exists in this module — every OTHER `cred_provider` seam in this file
    is also `None`-skips-entirely) reports `{'checked': False}`, never a
    failure. A provider read error (raises, or returns a non-numeric expiry)
    is NON-terminal: recorded as a `'transient': True` note so planning can
    still proceed — same "transient != confirmed-bad" posture
    `_check_credential_horizon` already uses for the live reconcile path.
    Only a CONFIRMED insufficient horizon sets `failure_class:
    CREDENTIAL_EXPIRES`."""
    if cred_provider is None:
        return {"checked": False}
    now_epoch = now_epoch if now_epoch is not None else time.time()
    notes = []
    for stage in wf.stages:
        profile = wf.profiles[stage.profile]
        try:
            expiry = cred_provider.current_expiry(stage.name)
        except Exception as e:
            notes.append(f"stage {stage.name!r}: credential horizon UNVERIFIED ({e})")
            continue
        if not isinstance(expiry, (int, float)) or isinstance(expiry, bool):
            notes.append(f"stage {stage.name!r}: credential horizon UNVERIFIED "
                        f"(provider returned {expiry!r})")
            continue
        remaining = remaining_wall_s(profile, 0.0)
        if not credential_horizon_ok(now_epoch=now_epoch, cred_expiry_epoch=float(expiry),
                                      remaining_wall_s=remaining):
            return {"checked": True, "stage": stage.name, "expiry": expiry,
                    "error": (f"CREDENTIAL_EXPIRES: credential for stage {stage.name!r} "
                              f"would expire before its remaining wall bound "
                              f"({remaining:.0f}s)"),
                    "failure_class": "CREDENTIAL_EXPIRES"}
    return {"checked": True, "transient": bool(notes), "notes": notes}


# --- thin CLI wrappers (optional; keep herdd.py's argparse handlers tiny) ---
# Every wrapper returns (exit_code, json_dict) — the CLI subtask's argparse
# handler need only `json.dumps(...)`/`sys.exit(...)` this tuple. Signatures
# are the contract downstream (herdd.py, test_workflow.py) code against;
# see this subtask's final report for the frozen list.
# moved-from: workflowctl.plan_workflow
def plan_workflow(path: str, *, online: bool = False, wf_id: str | None = None,
                  runner: Runner | None = None, bucket: str | None = None,
                  actor: str | None = None, asset_checker: Seam | None = None,
                  image_resolver: Seam | None = None,
                  cred_provider: CredProvider | None = None,
                  now_epoch: float | None = None, canary_checker: Seam | None = None,
                  jobd_sha: str | None = None) -> tuple[int, dict[str, Any]]:
    """Roadmap M4-T1 "workflow plan". Offline (`online=False`, the default):
    load + write_spec, then validate EVERY expanded child bundle config
    (`_validate_stage_bundle`) and EVERY `InputRef` wiring
    (`_check_stage_inputs`) — a bad stage fails closed with
    `failure_class: CONFIG_INVALID` before anything is submitted.

    Online (`online=True`): additionally resolves B2 asset staleness (strict),
    per-profile image digests, credential horizon, and worst-case spend via
    `_plan_online` — every resolver there is injectable
    (`asset_checker`/`image_resolver`/`cred_provider`), defaulting to the
    real `jobmeta`/`herdd` helpers so a caller that injects nothing gets
    real (but still no-live-spend: read-only) B2/registry resolution; a test
    always injects fakes instead."""
    actor = actor or jobmeta._default_actor()  # type: ignore[no-untyped-call]
    try:
        wf = load_workflow_module(path)
        wf_id = wf_id or wm.mint_wf_id(wf.name)
        spec_result = write_spec(wf, wf_id, runner=runner, bucket=bucket)
    except WorkflowCtlError as e:
        return EXIT_INVALID, {"error": str(e)}

    stages_by_name = {s.name: s for s in wf.stages}
    stage_cfgs, stage_reports = {}, []
    for stage in wf.stages:
        try:
            cfg = _validate_stage_bundle(stage)
            _check_stage_inputs(stage, stages_by_name)
        except WorkflowCtlError as e:
            return EXIT_INVALID, {"error": str(e), "stage": stage.name,
                                   "failure_class": "CONFIG_INVALID"}
        stage_cfgs[stage.name] = cfg
        stage_reports.append({"name": stage.name, "bundle_ok": True, "inputs_ok": True})

    result: dict[str, Any] = {"wf_id": wf_id, "spec": spec_result,
                              "stages": stage_reports}
    if not online:
        return EXIT_OK, result

    rc, online_report = _plan_online(
        wf, stage_cfgs, runner=runner, bucket=bucket, asset_checker=asset_checker,
        image_resolver=image_resolver, cred_provider=cred_provider, now_epoch=now_epoch,
        canary_checker=canary_checker, jobd_sha=jobd_sha)
    result["online"] = online_report
    result["disclaimer"] = REHEARSAL_DISCLAIMER
    if rc != EXIT_OK:
        result["error"] = online_report.get("error")
        result["failure_class"] = online_report.get("failure_class")
        return rc, result
    return EXIT_OK, result


# --- M4-T1 rehearse: dependency-ordered stage driver over FAKE B2 ------------
# moved-from: workflowctl._topo_order_stages
def _topo_order_stages(wf: Workflow) -> list[JobStage]:
    """A plain Kahn sort over `wf.stages`'s `after` graph, declared order as
    the tiebreak among simultaneously-ready stages (same "first ready in
    declared order" convention `wm.ready_stages` uses for the live reconcile
    path — this is the static, event-free counterpart: `rehearse_workflow`
    has no event log to fold a view from, so it walks the DAG directly).
    `load_workflow_module` already ran `wm.validate_workflow_spec`
    (`_check_acyclic`) before a `Workflow` reaches here, so the `else`
    branch below is unreachable in practice — kept as a hard-fail rather
    than a silent infinite loop if that invariant is ever violated."""
    remaining = list(wf.stages)
    resolved = set()
    order = []
    while remaining:
        for i, s in enumerate(remaining):
            if all(dep in resolved for dep in s.after):
                order.append(s)
                resolved.add(s.name)
                del remaining[i]
                break
        else:
            raise WorkflowCtlError(
                "rehearse_workflow: no ready stage found among "
                f"{[s.name for s in remaining]!r} -- cyclic or unresolved "
                "dependency (should be impossible after load_workflow_module's "
                "acyclic validation)")
    return order


# moved-from: workflowctl._stable_manifest_sha256
def _stable_manifest_sha256(manifest: dict[str, Any]) -> str:
    """Deterministic sha256 over a parsed manifest dict's CANONICAL bytes
    (sort_keys, compact separators) — same discipline every other manifest
    hash in this module already uses (`_accept_stage_artifacts`'s accepted-
    artifact record). Hashing the re-serialized dict, not the raw file
    bytes, means whitespace/formatting differences between what a stage
    produced and what a later read sees never cause a false mismatch."""
    body = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# moved-from: workflowctl._read_stage_manifest
def _read_stage_manifest(results_dir: str,
                         manifest_path: str = "results/artifact-manifest.json") -> Any:  # noqa: ANN401 — a parsed manifest is whatever json.load returned
    """Tolerant read of `<results_dir>/` + `manifest_path`. `manifest_path`
    is WORKDIR-RELATIVE (the producing stage's
    `ArtifactContract.manifest_path`), resolved under the locally-captured
    `results_dir` (rehearse.sh's `REHEARSE_RESULTS_OUT`) exactly the way
    `jobmeta.validate_generation_artifact` resolves it under a live job's
    `jobs/<job_id>/results/` prefix; the default reproduces the historical
    e2 read (`<results_dir>/results/artifact-manifest.json`). Returns None
    (never raises) when the file is missing/unparseable — a probe, not a
    validator, same contract as `read_accepted_artifact`;
    `rehearse_workflow` itself turns a None into a LOUD failure whenever the
    stage declared outputs (a manifest that cannot be read must never pass
    the binding check vacuously)."""
    path = os.path.join(results_dir, manifest_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# moved-from: workflowctl._stage_manifest_path
def _stage_manifest_path(stage: JobStage) -> str | None:
    """The ONE workdir-relative manifest path `stage`'s rehearsal capture
    reads (None for a stage declaring no outputs). The rehearsal seam
    captures a single produced manifest per stage (`stage_rehearser` returns
    one `(rc, manifest)`), so a stage whose output contracts declare more
    than one DISTINCT `manifest_path` cannot be captured faithfully —
    hard-fail rather than silently reading one path and vacuously passing
    the binding check on the others."""
    paths = sorted({c.manifest_path for c in stage.outputs.values()})
    if not paths:
        return None
    if len(paths) > 1:
        raise WorkflowCtlError(
            f"rehearse_workflow: stage {stage.name!r} declares {len(paths)} "
            f"distinct output manifest_paths {paths!r} — the rehearsal lane "
            f"captures exactly one manifest per stage")
    return paths[0]


# moved-from: workflowctl._build_default_stage_rehearser
def _build_default_stage_rehearser(rehearse_script: str) -> Seam:
    """The real (non-test) `stage_rehearser(stage, bundle_dir, asset_overrides,
    results_out) -> (rc, produced_manifest_dict)`: one `rehearse_script`
    subprocess per stage, DRY_RUN=1 (so the bundle's own entrypoint
    fabricates its outputs the same way the e2-paired-gen/-score DRY_RUN
    lanes already do — never a real CUDA/vLLM load) plus
    `REHEARSE_RESULTS_OUT=<results_out>` (rehearse.sh's existing results-
    capture seam) and one `--asset NAME=DIR` per injected upstream input.
    NEVER invoked by a test — every test in this module injects a fake
    `stage_rehearser` instead, so this closure (and the `subprocess.run` it
    wraps) is dead code on the portable pytest lane by construction.

    Never raises on a nonzero rehearse.sh exit — `rehearse_workflow` reads
    `rc` itself and turns a nonzero rc into the stage's own failure report,
    same posture as `jobmeta`/`herdd`'s soft `(ok, err)` transport
    primitives (a rehearsal FAIL is expected, ordinary control flow, not an
    exceptional condition worth a traceback)."""
    def stage_rehearser(stage: JobStage, bundle_dir: str,
                        asset_overrides: dict[str, str],
                        results_out: str) -> tuple[int, Any]:
        cmd = [rehearse_script, bundle_dir]
        for name, seed_dir in sorted(asset_overrides.items()):
            cmd += ["--asset", f"{name}={seed_dir}"]
        env = dict(os.environ)
        env["DRY_RUN"] = "1"
        env["REHEARSE_RESULTS_OUT"] = results_out
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        manifest = None
        if proc.returncode == 0:
            # read at the CONTRACT's declared manifest_path (None for a stage
            # with no outputs) — never the hardcoded e2 default.
            manifest_rel = _stage_manifest_path(stage)
            if manifest_rel is not None:
                manifest = _read_stage_manifest(results_out, manifest_rel)
        return proc.returncode, manifest

    return stage_rehearser


# moved-from: workflowctl.rehearse_workflow
def rehearse_workflow(path: str, *, wf_id: str | None = None,
                      runner: Runner | None = None, bucket: str | None = None,
                      rehearse_script: str | None = None, workdir: str | None = None,
                      stage_rehearser: Seam | None = None) -> tuple[int, dict[str, Any]]:
    """Roadmap M4-T1 "workflow rehearse": run every stage's bundle through a
    LOCAL rehearsal (`rehearse.sh` DRY_RUN=1 against ITS OWN fake bucket —
    never real B2/vast) in dependency order, injecting each stage's
    captured produced artifact as the next stage's `--asset` seed and
    verifying the manifest bytes a downstream stage would consume are
    byte-identical to what its upstream produced.

    `stage_rehearser` is the injectable transport seam
    (`_build_default_stage_rehearser`'s real shape); every test in this
    module passes a fake one, so this function never imports torch/vLLM,
    never requires a real bundle checkout, and never shells out in the
    portable pytest lane. `runner`/`bucket` are accepted for signature
    parity with every other `*_workflow` entry point in this module (a
    rehearsal never reads/writes real B2 today — nothing here calls them);
    kept so a future subtask can record a rehearsal summary durably without
    another signature change.

    Returns `(exit_code, dict)`. The dict ALWAYS carries `disclaimer`
    (`REHEARSAL_DISCLAIMER`, verbatim) on every path — success, a stage
    failure, an artifact mismatch, or a bad workflow module — so a caller
    that prints this result can never surface a rehearsal outcome without
    the caveat that it does not certify CUDA/live artifacts."""
    try:
        wf = load_workflow_module(path)
    except WorkflowCtlError as e:
        return EXIT_INVALID, {"error": str(e), "disclaimer": REHEARSAL_DISCLAIMER}
    wf_id = wf_id or wm.mint_wf_id(wf.name)

    script = rehearse_script or os.path.join(_TOOLS_VAST_DIR, "rehearse.sh")
    rehearser = stage_rehearser or _build_default_stage_rehearser(script)

    root = workdir or os.path.join(_lock_dir(), "rehearse", wf_id)
    os.makedirs(root, exist_ok=True)

    order = _topo_order_stages(wf)
    produced: dict[str, dict[str, Any]] = {}
    stage_reports: list[dict[str, Any]] = []
    for stage in order:
        asset_overrides = {}
        for input_name, ref in sorted(stage.inputs.items()):
            upstream = produced.get(ref.stage)
            if upstream is None:
                return EXIT_ARTIFACT, {
                    "wf_id": wf_id, "stage": stage.name, "input": input_name,
                    "error": (
                        f"stage {stage.name!r} input {input_name!r} references "
                        f"upstream stage {ref.stage!r}, which has no captured "
                        f"rehearsal results (dependency did not run first)"),
                    "failure_class": "ARTIFACT_INVALID", "stages": stage_reports,
                    "disclaimer": REHEARSAL_DISCLAIMER}
            # re-read at the PRODUCING stage's declared manifest_path (the
            # same path its capture read) — a hardcoded default here made
            # both sides None for a non-default contract and the check below
            # passed vacuously (None == None).
            fresh = _read_stage_manifest(
                upstream["results_dir"],
                upstream["manifest_path"] or "results/artifact-manifest.json")
            fresh_sha = _stable_manifest_sha256(fresh) if fresh is not None else None
            if fresh_sha != upstream["manifest_sha256"]:
                return EXIT_ARTIFACT, {
                    "wf_id": wf_id, "stage": stage.name, "input": input_name,
                    "error": (
                        f"stage {stage.name!r} input {input_name!r}: the manifest "
                        f"the downstream would consume ({fresh_sha!r}) does not "
                        f"match what stage {ref.stage!r} produced "
                        f"({upstream['manifest_sha256']!r})"),
                    "failure_class": "ARTIFACT_INVALID", "stages": stage_reports,
                    "disclaimer": REHEARSAL_DISCLAIMER}
            asset_overrides[input_name] = upstream["results_dir"]

        bundle_dir = _stage_bundle_dir(stage)
        results_out = os.path.join(root, stage.name)
        manifest_rel = _stage_manifest_path(stage)
        rc, manifest = rehearser(stage, bundle_dir, asset_overrides, results_out)
        if rc != 0:
            return EXIT_FAILED, {
                "wf_id": wf_id, "stage": stage.name, "rc": rc,
                "error": f"stage {stage.name!r} rehearsal failed (rc={rc})",
                "failure_class": "ENTRYPOINT_FAILED", "stages": stage_reports,
                "disclaimer": REHEARSAL_DISCLAIMER}
        if manifest_rel is not None and manifest is None:
            # LOUD fail, never a None-pass: a stage that declares outputs but
            # whose rehearsal manifest cannot be read must fail the binding
            # check here — letting it through stores manifest_sha256=None and
            # the downstream byte-identity check would compare None == None.
            return EXIT_ARTIFACT, {
                "wf_id": wf_id, "stage": stage.name,
                "error": (
                    f"stage {stage.name!r} declares outputs but its rehearsal "
                    f"manifest at {manifest_rel!r} (under {results_out!r}) "
                    f"could not be read — refusing to pass the artifact "
                    f"binding check vacuously"),
                "failure_class": "ARTIFACT_INVALID", "stages": stage_reports,
                "disclaimer": REHEARSAL_DISCLAIMER}

        manifest_sha256 = _stable_manifest_sha256(manifest) if manifest is not None else None
        produced[stage.name] = {"results_dir": results_out, "manifest": manifest,
                                 "manifest_sha256": manifest_sha256,
                                 "manifest_path": manifest_rel}
        stage_reports.append({"name": stage.name, "rc": rc, "results_dir": results_out,
                              "manifest_sha256": manifest_sha256})

    return EXIT_OK, {"wf_id": wf_id, "stages": stage_reports,
                      "disclaimer": REHEARSAL_DISCLAIMER}


# moved-from: workflowctl._LIVE_DEP_KEYS
_LIVE_DEP_KEYS = ("box_resolver", "box_teardown", "box_observer",
                  "box_starter", "cost_observer", "cred_provider",
                  "throughput_observer", "image_state_observer")


# moved-from: workflowctl._resolve_controller_deps
def _resolve_controller_deps(controller_deps: dict[str, Any] | Seam | None,
                             wf: Workflow, wf_id: str) -> dict[str, Any]:
    """`controller_deps` is EITHER a ready dict of the `_LIVE_DEP_KEYS` OR a
    callable `(wf, wf_id) -> dict` (invoked AFTER wf_id is resolved so the
    factory has the freshly-minted/read wf_id `build_box_resolver`'s
    deterministic run label needs; the callable closes over actor/runner/
    bucket itself — see the CLI's `build_live_controller_deps` lambda).
    Returns the {key: value} subset to forward into `run_controller`; `None`
    -> `{}` -> control-plane behavior preserved."""
    if controller_deps is None:
        return {}
    if callable(controller_deps):
        controller_deps = controller_deps(wf, wf_id)
    return {k: controller_deps[k]  # type: ignore[index]
            for k in _LIVE_DEP_KEYS
            if k in controller_deps}  # type: ignore[operator]


# moved-from: workflowctl.run_workflow
def run_workflow(path: str, *, wf_id: str | None = None, actor: str,
                 detach: bool = False, takeover: bool = False,
                 max_ticks: int | None = None, runner: Runner | None = None,
                 bucket: str | None = None, clock: Seam | None = None,
                 sleep_fn: Seam | None = None, argv: Sequence[Any] | None = None,
                 notifier: Seam | None = None,
                 controller_deps: dict[str, Any] | Seam | None = None,
                 detached_controller: bool = False) -> tuple[int, dict[str, Any]]:
    try:
        wf = load_workflow_module(path)
        wf_id = wf_id or wm.mint_wf_id(wf.name)
        write_spec(wf, wf_id, runner=runner, bucket=bucket)
    except WorkflowCtlError as e:
        return EXIT_INVALID, {"error": str(e)}
    if detach and argv:
        # Pin the detached re-exec (and every Restart=on-failure re-run) to
        # THIS wf_id. Without the pin the child's bare `workflow run <path>`
        # re-plans and mints a SECOND id, so the id this call returns (the
        # printed id, the unit name, the spec.json just written) is an orphan
        # while the real controller drives an unadvertised one (found live
        # 2026-07-15: --detach printed ...-e2-paired-3553, the controller ran
        # ...-e2-paired-2c10). The pinned child's own write_spec is then a
        # byte-identical noop, not a second plan. `--detached-controller`
        # marks the child (and every Restart=on-failure re-run) as THE
        # detached controller so a terminal workflow exits 0 instead of
        # flap-restarting the unit (see run_controller's docstring).
        argv = list(argv) + ["--detached-controller", "--wf-id", wf_id]
    deps = _resolve_controller_deps(controller_deps, wf, wf_id)
    rc = run_controller(wf, wf_id, runner=runner, bucket=bucket, actor=actor,
                         detach=detach, takeover=takeover, max_ticks=max_ticks,
                         clock=clock, sleep_fn=sleep_fn, argv=argv, notifier=notifier,
                         detached_controller=detached_controller, **deps)
    v = view(wf_id, runner=runner, bucket=bucket)
    return rc, {"wf_id": wf_id, "view": v}


# moved-from: workflowctl._stage_job_extras
def _stage_job_extras(job_id: str, *, runner: Runner | None,
                      bucket: str | None) -> dict[str, Any]:
    """Best-effort per-stage figures folded from one child job's own event
    log (roadmap M4-T3 `status_extras`). Never raises: any exception folding
    the child job (missing job, partial/corrupt events, unparseable
    checkpoint ts) yields every field `None` rather than breaking the
    caller's status read."""
    out: dict[str, Any] = {"progress": None, "spend_usd": None,
                           "checkpoint_age_s": None}
    try:
        jv = jobmeta.read_job(job_id, runner=runner, bucket=bucket)
    except Exception:
        return out
    try:
        pg = jobs_view._job_progress(jv)
        if pg.get("step") is not None and pg.get("total") is not None:
            out["progress"] = f"{pg['step']}/{pg['total']}"
        elif pg.get("pct") is not None:
            out["progress"] = f"{pg['pct']}%"
    except Exception:
        pass
    try:
        last_ckpt = jv.get("last_checkpoint_ts")
        if last_ckpt:
            out["checkpoint_age_s"] = wm._ts_diff_seconds(wm.now_ts(), last_ckpt)
    except Exception:
        pass
    return out


# moved-from: workflowctl.status_extras
def status_extras(wf_id: str, *, wf: Workflow | None = None,
                  runner: Runner | None = None,
                  bucket: str | None = None) -> dict[str, Any]:
    """Read-only, best-effort I/O-derived figures for `format_status_table`'s
    `extras` param (pure formatting lives in `wm.status_table_rows` /
    `wm.format_status_table` — this is the I/O side that fills it in).
    MUST NOT raise for a partial/empty/brand-new workflow: every unknown
    datum comes back `None` rather than propagating a fold/network error."""
    runner = _resolve_runner(runner)
    out: dict[str, Any] = {
        "spend_usd": None, "budget_usd": (wf.budget_usd if wf is not None else None),
        "stages": {}}
    try:
        out["spend_usd"] = folded_spend(wf_id, runner=runner, bucket=bucket)
    except Exception:
        pass
    try:
        v = view(wf_id, runner=runner, bucket=bucket)
    except Exception:
        v = {}
    for name, sv in (v.get("stages") or {}).items():
        sv = sv or {}
        job_id = sv.get("job_id")
        if job_id:
            out["stages"][name] = _stage_job_extras(job_id, runner=runner, bucket=bucket)
        else:
            out["stages"][name] = {"progress": None, "spend_usd": None,
                                    "checkpoint_age_s": None}
    return out


# moved-from: workflowctl.status_workflow
def status_workflow(wf_id: str, *, runner: Runner | None = None,
                    bucket: str | None = None) -> tuple[int, dict[str, Any]]:
    v = view(wf_id, runner=runner, bucket=bucket)
    rc = (_terminal_exit_code(v.get("status"), v.get("failure_class"))
          if v.get("terminal") else EXIT_OK)
    return rc, v


# moved-from: workflowctl.logs_workflow
def logs_workflow(wf_id: str, *, runner: Runner | None = None,
                  bucket: str | None = None) -> tuple[int, dict[str, Any]]:
    raw = read_events(wf_id, runner=runner, bucket=bucket)
    events = []
    for r in raw:
        try:
            events.append(json.loads(r))
        except (ValueError, TypeError):
            continue
    events.sort(key=lambda e: (e.get("ts", ""), e.get("nonce", "")))
    return EXIT_OK, {"wf_id": wf_id, "events": events}


# moved-from: workflowctl.pull_workflow
def pull_workflow(wf: Workflow, wf_id: str, dest: str, *, stage: str | None = None,
                  runner: Runner | None = None,
                  bucket: str | None = None) -> tuple[int, dict[str, Any]]:
    """`stage` defaults to the LAST declared stage (the pipeline's terminal
    stage) — needs `wf`, not just `wf_id`, to know declared order."""
    v = view(wf_id, runner=runner, bucket=bucket)
    stage_name = stage or (wf.stages[-1].name if wf.stages else None)
    if stage_name is None:
        return EXIT_INVALID, {"error": "workflow has no stages to pull"}
    job_id = v.get("stages", {}).get(stage_name, {}).get("job_id")
    if not job_id:
        return EXIT_INVALID, {"error": f"stage {stage_name!r} has no job_id yet"}
    files = jobmeta.pull_results(job_id, dest, runner=runner, bucket=bucket)
    return EXIT_OK, {"wf_id": wf_id, "stage": stage_name, "job_id": job_id, "files": files}


# moved-from: workflowctl.cancel_workflow
def cancel_workflow(wf_id: str, *, actor: str, reason: str | None = None,
                    runner: Runner | None = None,
                    bucket: str | None = None) -> tuple[int, dict[str, Any]]:
    # jobmeta defaults runner=_default_runner as the PARAMETER default, so
    # passing an explicit runner=None clobbers it into a TypeError — resolve.
    ok, err = jobmeta.write_cancel_marker(wf_id, actor=actor, reason=reason,  # type: ignore[no-untyped-call]
                                          runner=_resolve_runner(runner),
                                          bucket=bucket)
    if not ok:
        return EXIT_INVALID, {"error": err}
    ev = emit(wf_id, "workflow_cancelled", actor, runner=runner, bucket=bucket,
             reason=reason or "cancelled by operator")
    return EXIT_CANCELLED, {"wf_id": wf_id, "event": ev}


# moved-from: workflowctl.resume_workflow
def resume_workflow(wf_id: str, *, actor: str, takeover: bool = True,
                    detach: bool = False, max_ticks: int | None = None,
                    runner: Runner | None = None, bucket: str | None = None,
                    clock: Seam | None = None, sleep_fn: Seam | None = None,
                    cred_provider: CredProvider | None = None,
                    argv: Sequence[Any] | None = None, notifier: Seam | None = None,
                    controller_deps: dict[str, Any] | Seam | None = None,
                    detached_controller: bool = False) -> tuple[int, dict[str, Any]]:
    """Reattach a controller to an EXISTING `wf_id` (reads `spec.json` from
    B2 rather than a local `.py` path). Defaults `takeover=True` since a
    `resume` is, by definition, expected to replace whatever controller last
    held this workflow.

    `cred_provider` (M3-T2, default `None` -> skipped): a `resume` is exactly
    the "recovery path" the roadmap packet calls out for rotate-on-resume --
    rotates ONCE via `_rotate_credential` here, BEFORE any paid work
    continues, in addition to being threaded into `run_controller`'s loop for
    any in-run resume/retarget. A rotation failure raises `WorkflowCtlError`
    UNCAUGHT (never folded into the `(rc, dict)` return below) so the CLI
    layer's existing `except WorkflowCtlError: sys.exit(EXIT_CREDENTIAL)`
    around this call is the one place it becomes operator-facing."""
    try:
        wf = read_spec(wf_id, runner=runner, bucket=bucket)
    except WorkflowCtlError as e:
        return EXIT_INVALID, {"error": str(e)}
    if cred_provider is not None:
        _rotate_credential(cred_provider, wf_id)
    if detach and argv:
        # mark the detached re-exec (and every Restart=on-failure re-run) as
        # THE detached controller — a terminal workflow exits 0 instead of
        # flap-restarting the unit (run_workflow appends the same flag; see
        # run_controller's docstring). Resume argv already carries wf_id.
        argv = list(argv) + ["--detached-controller"]
    deps = _resolve_controller_deps(controller_deps, wf, wf_id)
    # an explicit cred_provider kwarg still wins if the deps bundle didn't set
    # one (production bundle leaves cred_provider=None -> harmless override).
    deps.setdefault("cred_provider", cred_provider)
    rc = run_controller(wf, wf_id, runner=runner, bucket=bucket, actor=actor,
                         detach=detach, takeover=takeover, max_ticks=max_ticks,
                         clock=clock, sleep_fn=sleep_fn, argv=argv, notifier=notifier,
                         detached_controller=detached_controller, **deps)
    v = view(wf_id, runner=runner, bucket=bucket)
    return rc, {"wf_id": wf_id, "view": v}
