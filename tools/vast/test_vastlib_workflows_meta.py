"""`vastlib.workflows.meta` — the ported pure fold, held to its FROZEN wire shapes.

Why this file exists
--------------------
`test_workflowmeta.py` drives the flat `workflowmeta.py`, which stays live
through the add-only phase and is deliberately left untouched (plan §8). So it
cannot say anything about the ported copy — and this module is almost entirely
FROZEN WIRE CONTRACT. Objects on B2 are immutable once written, so a port that
changed one byte of `canonical_spec_json`, one name in `EVENTS`, or one field of
the `make_event` envelope would not fail loudly: it would read as a spec
conflict on every live workflow, or as an event the fold silently counts as
unknown.

The strategy WAS twin identity (`ported == flat`, computed from the same input
in the same process). **Step 7 retired it**: `workflowmeta.py` is now a
re-export shim over this module, so `wm.X == flat.X` compares an object with
itself — green forever, evidence of nothing. Every such comparison was replaced
by the value it was standing in for: the frozen vocabularies spelled out, and
GOLDEN full outputs for the four that must be compared whole
(`canonical_spec_json`'s bytes, `stage_job_id`, the fold's view dict,
`format_status_table`'s rendering) — the only way to catch a field that quietly
stopped being emitted. A golden literal cannot co-drift with its subject, which
is strictly stronger than the twin was.

`flat` is still imported: it is what proves the two names resolve to ONE set of
objects (`test_runmeta_primitives_are_the_same_objects_not_copies`,
`test_workflow_spec_error_is_a_workflow_error`).

What is deliberately NOT here
-----------------------------
* No re-testing of the DSL's own field validation — `vastlib.workflows.spec`'s
  `__post_init__` owns that and `test_vastlib_workflows_spec.py` covers it.
* No transport. Every function under test here is pure by module contract; the
  I/O that drives them is `ctl.py`'s and is covered next door.

Provenance: created 2026-08-16 alongside `vastlib/workflows/meta.py`, plan §8
step 5; de-tautologized at step 7 when the flat sibling became a shim
(`.port_manifests/step7-shims.json`, `parity_suites_that_go_vacuous`).
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

import runmeta                                          # noqa: E402
import workflowmeta as flat                             # noqa: E402  twin, still live
from workflow import (                                  # noqa: E402
    ArtifactContract,
    InputRef,
    JobStage,
    ResourceProfile,
    RetryPolicy,
    Workflow,
)

from vastlib.workflows import meta as wm                # noqa: E402


def _wf(**over: object) -> Workflow:
    """A two-stage generate->score workflow, the shape the e2 lane actually
    runs: pinned digests, one InputRef, one output contract, a retry policy."""
    profiles = {
        "gen": ResourceProfile(image="reg/x:t211", image_digest="sha256:aaa",
                               gpu=("RTX 4090",), num_gpus=1, disk_gb=40,
                               rental="bid", max_bid=0.5, budget_usd=3.0,
                               max_wall_s=7200),
        "cpu": ResourceProfile(image="reg/x:t211", image_digest="sha256:bbb",
                               rental="on-demand", budget_usd=1.0, max_wall_s=600),
    }
    stages = (
        JobStage(name="generate", bundle="b/gen", profile="gen",
                 outputs={"gens": ArtifactContract(
                     kind="generation", manifest_path="results/artifact-manifest.json")},
                 retry=RetryPolicy(max_attempts=2, retry_on=("infrastructure",))),
        JobStage(name="score", bundle="b/score", profile="cpu", after=("generate",),
                 inputs={"input-generate": InputRef(
                     stage="generate", artifact="gens", dest="/workspace/in")}),
    )
    kw: dict[str, object] = dict(version=1, name="e2 paired", budget_usd=10.0,
                                 max_wall_s=14400, teardown="stop",
                                 profiles=profiles, stages=stages)
    kw.update(over)
    return Workflow(**kw)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The frozen vocabularies — spelled out, and counted
#
# These read `== flat.X` until step 7. The flat module is now a re-export of
# THIS one, so such a comparison is `x == x` — green forever, evidence of
# nothing. They are pinned against their literals instead: the vocabularies are
# a frozen wire contract that may only ever GROW, so a spelled-out set is a
# stronger gate than the twin ever was.
# --------------------------------------------------------------------------- #
def test_event_vocabulary_is_the_frozen_v1_set():
    assert wm.EVENTS == frozenset({
        "submitted", "controller_started", "controller_heartbeat", "takeover",
        "stage_planned", "box_acquired", "stage_submitted", "stage_started",
        "artifact_accepted", "stage_succeeded", "stage_failed",
        "stage_cancelled", "teardown_started", "box_released",
        "workflow_succeeded", "workflow_failed", "workflow_cancelled",
    })
    assert len(wm.EVENTS) == 17            # the shipped V1 set; it may only GROW
    assert wm.STAGE_TERMINAL == frozenset({
        "stage_succeeded", "stage_failed", "stage_cancelled"})
    assert wm.WORKFLOW_TERMINAL == frozenset({
        "workflow_succeeded", "workflow_failed", "workflow_cancelled"})
    assert wm.CONTROLLER_EVENTS == frozenset({
        "controller_started", "controller_heartbeat", "takeover"})
    assert wm.STAGE_TERMINAL <= wm.EVENTS and wm.WORKFLOW_TERMINAL <= wm.EVENTS


def test_ctl_local_events_are_not_in_the_frozen_set():
    """`box_cost` and `teardown_attempt` are `ctl.py`'s, and the fold tolerates
    them as inert. Tidying them in here would change nothing functionally and
    break the stated ownership rule."""
    assert "box_cost" not in wm.EVENTS
    assert "teardown_attempt" not in wm.EVENTS
    assert "box_retargeted" not in wm.EVENTS
    assert "box_adopt_refused" not in wm.EVENTS


def test_failure_class_tables_are_the_frozen_ones():
    assert wm._FAILURE_CLASS_EXIT_CODE == {
        "CONFIG_INVALID": 1, "ASSET_STALE": 1, "IMAGE_DRIFT": 1,
        "ARTIFACT_INVALID": 4, "POSTCONDITION_FAILED": 4,
        "CREDENTIAL_EXPIRES": 5, "WALL_EXHAUSTED": 124,
        "ENV_CANARY_FAILED": 2, "INFRASTRUCTURE_FAILED": 2,
        "ENTRYPOINT_FAILED": 2, "CHECKPOINT_STALLED": 2,
        "BUDGET_EXHAUSTED": 2, "RETRY_EXHAUSTED": 2, "TEARDOWN_FAILED": 2,
    }
    assert set(wm._FAILURE_CLASS_EXIT_CODE) == set(wm.FAILURE_CLASSES)
    assert len(wm.FAILURE_CLASSES) == 14


def test_terminal_ranks_are_the_frozen_precedence():
    """Higher rank wins when a log carries more than one terminal event:
    an observed failure outranks a success, which outranks a late cancel."""
    assert wm._WORKFLOW_TERMINAL_RANK == {
        "workflow_failed": 3, "workflow_succeeded": 2, "workflow_cancelled": 1}
    assert wm._STAGE_TERMINAL_RANK == {
        "stage_failed": 3, "stage_succeeded": 2, "stage_cancelled": 1}


@pytest.mark.parametrize("fc,code", [
    ("CONFIG_INVALID", 1), ("ARTIFACT_INVALID", 4), ("CREDENTIAL_EXPIRES", 5),
    ("WALL_EXHAUSTED", 124), ("RETRY_EXHAUSTED", 2), (None, 2), ("NOT_A_CLASS", 2),
])
def test_failure_class_exit_code(fc, code):
    assert wm.failure_class_exit_code(fc) == code


# --------------------------------------------------------------------------- #
# IDs — deterministic, and identical to the flat module's
# --------------------------------------------------------------------------- #
def test_stage_job_id_is_deterministic_and_structured():
    """Was `== flat.stage_job_id(...)`; the flat name re-exports this function
    since step 7, so the id is pinned to its GOLDEN value instead — a
    submitted JOB_ID is what makes resubmission idempotent across restarts, so
    a hash-input change must fail here rather than silently double-submit."""
    wf_id = "20260713T120000-e2-paired-ab12"
    got = wm.stage_job_id(wf_id, "score", 0)
    assert got == "20260713T120000-25f61bf5-score-a0"
    assert got == wm.stage_job_id(wf_id, "score", 0)          # same input, same id
    ts15, hash8, stage, attempt = got.split("-")
    assert ts15 == wf_id[:15] and len(hash8) == 8
    assert stage == "score" and attempt == "a0"
    assert wm.stage_job_id(wf_id, "score", 1).endswith("-a1")


def test_stage_job_id_rejects_a_bad_attempt_or_stage():
    wf_id = "20260713T120000-e2-paired-ab12"
    for bad in (-1, True, "0"):
        with pytest.raises(wm.WorkflowIdError):
            wm.stage_job_id(wf_id, "score", bad)
    with pytest.raises(wm.WorkflowIdError):
        wm.stage_job_id(wf_id, "Score Stage", 0)


def test_mint_wf_id_shape_and_injected_nonce():
    got = wm.mint_wf_id("E2 Paired!", ts="20260713T120000", nonce4="beef")
    assert got == "20260713T120000-e2-paired-beef"


# --------------------------------------------------------------------------- #
# spec.json — the byte contract
# --------------------------------------------------------------------------- #
CANONICAL_SPEC_JSON_GOLDEN = (
    '{"budget_usd":10.0,"max_wall_s":14400,"name":"e2 paired","profiles":'
    '{"cpu":{"budget_usd":1.0,"disk_gb":null,"geo":[],"gpu":[],'
    '"gpu_ram_gb":null,"image":"reg/x:t211","image_digest":"sha256:bbb",'
    '"max_bid":null,"max_wall_s":600,"num_gpus":1,"rental":"on-demand"},'
    '"gen":{"budget_usd":3.0,"disk_gb":40,"geo":[],"gpu":["RTX 4090"],'
    '"gpu_ram_gb":null,"image":"reg/x:t211","image_digest":"sha256:aaa",'
    '"max_bid":0.5,"max_wall_s":7200,"num_gpus":1,"rental":"bid"}},'
    '"stages":[{"after":[],"bundle":"b/gen","inputs":{},"name":"generate",'
    '"outputs":{"gens":{"kind":"generation",'
    '"manifest_path":"results/artifact-manifest.json"}},"profile":"gen",'
    '"retry":{"max_attempts":2,"retry_on":["infrastructure"]},"secrets":[]},'
    '{"after":["generate"],"bundle":"b/score","inputs":{"input-generate":'
    '{"artifact":"gens","dest":"/workspace/in","stage":"generate"}},'
    '"name":"score","outputs":{},"profile":"cpu",'
    '"retry":{"max_attempts":1,"retry_on":[]},"secrets":[]}],'
    '"teardown":"stop","v":1,"version":1}'
)


def test_canonical_spec_json_bytes_are_the_frozen_bytes():
    """`ctl.write_spec` byte-compares this against what is already on B2. A
    key-order or separator drift reads as a spec CONFLICT on a live workflow,
    not as a test failure, which is why this compares whole bytes.

    Was `== flat.canonical_spec_json(wf)`; step 7 made the flat name a
    re-export of this very function, so the byte contract is pinned against a
    GOLDEN string instead — a literal cannot co-drift with its subject."""
    wf = _wf()
    body = wm.canonical_spec_json(wf)
    assert body == CANONICAL_SPEC_JSON_GOLDEN
    assert ", " not in body and '": ' not in body      # compact separators
    assert body == json.dumps(json.loads(body), sort_keys=True,
                              separators=(",", ":"))   # sorted keys


def test_spec_round_trips_through_the_dict_form():
    wf = _wf()
    back = wm.spec_from_dict(wm.spec_to_dict(wf))
    assert wm.canonical_spec_json(back) == wm.canonical_spec_json(wf)
    assert back.stages[1].inputs["input-generate"].dest == "/workspace/in"
    assert back.stages[0].outputs["gens"].kind == "generation"
    assert back.stages[0].retry.retry_on == ("infrastructure",)


def test_spec_from_dict_tolerates_unknown_keys():
    d = wm.spec_to_dict(_wf())
    d["a_field_from_the_future"] = {"x": 1}
    assert wm.spec_from_dict(d).name == "e2 paired"


# --------------------------------------------------------------------------- #
# Cross-object validation
# --------------------------------------------------------------------------- #
def test_validate_workflow_spec_accepts_the_pinned_shape():
    assert wm.validate_workflow_spec(_wf(), of_record=True) is None


def test_validate_workflow_spec_aggregates_every_violation():
    """One raise carrying ALL findings — the aggregating contract `plan` needs
    so an operator fixes a spec in one pass, not five."""
    bad = Workflow(
        version=1, name="bad", budget_usd=0.0, max_wall_s=0, teardown="stop",
        profiles={"gen": ResourceProfile(image="reg/x:t")},          # no digest
        stages=(
            JobStage(name="a", bundle="b", profile="nope", after=("b",)),
            JobStage(name="b", bundle="b", profile="gen", after=("a",),
                     inputs={"in": InputRef(stage="a", artifact="ghost",
                                            dest="/d")},
                     retry=RetryPolicy(max_attempts=1, retry_on=("meteor",))),
        ))
    with pytest.raises(wm.WorkflowSpecError) as e:
        wm.validate_workflow_spec(bad, of_record=True)
    msg = str(e.value)
    assert "undeclared profile" in msg
    assert "cycle in stage dependencies" in msg
    assert "does not declare as an output" in msg
    assert "no image_digest" in msg
    assert "retry_on has values outside" in msg
    # (the `== str(flat_error)` half of this test was dropped at step 7: the
    # flat name re-exports this validator, so it compared a string to itself.)


def test_of_record_false_permits_an_unpinned_profile():
    wf = Workflow(version=1, name="draft", budget_usd=0.0, max_wall_s=0,
                  teardown="stop",
                  profiles={"gen": ResourceProfile(image="reg/x:t")},
                  stages=(JobStage(name="a", bundle="b", profile="gen"),))
    assert wm.validate_workflow_spec(wf, of_record=False) is None
    with pytest.raises(wm.WorkflowSpecError):
        wm.validate_workflow_spec(wf, of_record=True)


def test_workflow_spec_error_is_a_workflow_error():
    """`ctl.load_workflow_module` catches `wm.WorkflowSpecError` specifically;
    the flat DSL's `WorkflowError` is its base so a caller can catch either."""
    from workflow import WorkflowError
    assert issubclass(wm.WorkflowSpecError, WorkflowError)


# --------------------------------------------------------------------------- #
# The event envelope
# --------------------------------------------------------------------------- #
def test_make_event_envelope_is_the_frozen_v1_envelope():
    """The `== flat.make_event(...)` half was dropped at step 7 (self-comparison
    once the flat name re-exports this function); the literal envelope below is
    what actually pins the wire shape."""
    kw = dict(ts="20260713T120000000Z", nonce="cafe", stage="score", attempt=1)
    got = wm.make_event("20260713T120000-e2-ab12", "stage_started", "me", **kw)
    assert got == {
        "v": 1, "ts": "20260713T120000000Z", "actor": "me",
        "event": "stage_started", "workflow_id": "20260713T120000-e2-ab12",
        "nonce": "cafe", "stage": "score", "attempt": 1,
    }


def test_make_event_drops_none_valued_fields():
    ev = wm.make_event("20260713T120000-e2-ab12", "stage_failed", "me",
                       ts="20260713T120000000Z", nonce="cafe",
                       failure_class=None, reason="boom")
    assert "failure_class" not in ev and ev["reason"] == "boom"


def test_make_event_validates_the_workflow_id():
    with pytest.raises(wm.WorkflowIdError):
        wm.make_event("not a valid id!", "submitted", "me")


def test_runmeta_primitives_are_the_same_objects_not_copies():
    """ONE clock/nonce/event-key implementation. `ctl.py` reaches three of these
    as `wm.now_ts` / `wm.event_key`; if the port had re-implemented them the
    event KEY discipline would fork silently."""
    assert wm.now_ts is runmeta.now_ts
    assert wm.nonce is runmeta.nonce
    assert wm.event_key is runmeta.event_key
    assert wm._actor_slug is runmeta._actor_slug


# --------------------------------------------------------------------------- #
# The fold
# --------------------------------------------------------------------------- #
def _ev(event, ts, **f):
    d = {"v": 1, "ts": ts, "actor": "me", "event": event,
         "workflow_id": "20260713T120000-e2-ab12", "nonce": ts[-4:]}
    d.update(f)
    return json.dumps(d)


def test_fold_emits_the_WHOLE_view_on_a_realistic_log():
    """Was `== flat.fold_workflow_events(evs)`; the flat name re-exports this
    fold since step 7, so the view is compared against a GOLDEN dict — the only
    way to catch a field that quietly stopped being emitted."""
    evs = [
        _ev("submitted", "20260713T120000000Z"),
        _ev("controller_started", "20260713T120001000Z"),
        _ev("stage_planned", "20260713T120002000Z", stage="generate", attempt=0,
            job_id="j0"),
        _ev("box_acquired", "20260713T120003000Z", stage="generate", attempt=0,
            instance_id=4711, machine_id=9),
        _ev("stage_started", "20260713T120004000Z", stage="generate", attempt=0,
            job_id="j0"),
        _ev("stage_succeeded", "20260713T120500000Z", stage="generate", attempt=0,
            job_id="j0"),
        _ev("controller_heartbeat", "20260713T120600000Z"),
    ]
    v = wm.fold_workflow_events(evs)
    assert v == {
        "workflow_id": "20260713T120000-e2-ab12",
        "status": "running", "terminal": False, "failure_class": None,
        "n_events": 7, "parse_errors": 0, "unknown_events": 0,
        "controller": {"actor": "me",
                       "started_ts": "20260713T120001000Z",
                       "last_heartbeat_ts": "20260713T120600000Z"},
        "stages": {"generate": {"status": "stage_succeeded", "attempt": 0,
                                "attempts_seen": 1, "job_id": "j0",
                                "instance_id": 4711, "failure_class": None,
                                "box_acquired_ts": "20260713T120003000Z"}},
    }
    assert v["status"] == "running" and v["terminal"] is False
    assert v["stages"]["generate"]["status"] == "stage_succeeded"
    assert v["stages"]["generate"]["instance_id"] == 4711
    assert v["stages"]["generate"]["box_acquired_ts"] == "20260713T120003000Z"
    assert v["controller"]["last_heartbeat_ts"] == "20260713T120600000Z"
    assert v["parse_errors"] == 0 and v["unknown_events"] == 0


def test_fold_counts_unknown_events_and_never_drops_the_view():
    evs = [_ev("submitted", "20260713T120000000Z"),
           _ev("box_cost", "20260713T120100000Z", cost_usd=0.42),
           _ev("teardown_attempt", "20260713T120200000Z")]
    v = wm.fold_workflow_events(evs)
    assert v["unknown_events"] == 2          # ctl-local names, inert but counted
    assert v["status"] == "submitted"
    assert v["n_events"] == 3


def test_fold_counts_a_malformed_object_instead_of_crashing():
    evs = ["{not json", "", b"\xff\xfe", json.dumps({"ts": "x"}),
           _ev("submitted", "20260713T120000000Z")]
    v = wm.fold_workflow_events(evs)
    assert v["parse_errors"] == 4 and v["n_events"] == 1
    assert v["status"] == "submitted"


def test_terminal_precedence_prefers_the_observed_outcome():
    """failed > succeeded > cancelled — a real outcome outranks a late operator
    action, and it is NEVER last-event-wins."""
    late_cancel = [
        _ev("workflow_failed", "20260713T120000000Z", failure_class="RETRY_EXHAUSTED"),
        _ev("workflow_cancelled", "20260713T130000000Z"),
    ]
    v = wm.fold_workflow_events(late_cancel)
    assert v["status"] == "failed" and v["terminal"] is True
    assert v["failure_class"] == "RETRY_EXHAUSTED"


def test_stage_terminal_precedence_and_latest_attempt_win():
    evs = [
        _ev("stage_failed", "20260713T120000000Z", stage="s", attempt=0,
            failure_class="INFRASTRUCTURE_FAILED"),
        _ev("stage_planned", "20260713T120100000Z", stage="s", attempt=1, job_id="j1"),
    ]
    sv = wm.fold_workflow_events(evs)["stages"]["s"]
    assert sv["attempt"] == 1 and sv["status"] == "stage_planned"
    assert sv["attempts_seen"] == 2


def test_ready_stages_is_declared_order_and_respects_dependencies():
    wf = _wf()
    empty = wm.fold_workflow_events([])
    assert wm.ready_stages(wf, empty) == ["generate"]     # score waits on generate
    assert wm.next_ready_stage(wf, empty) == "generate"
    done = wm.fold_workflow_events([
        _ev("stage_succeeded", "20260713T120000000Z", stage="generate", attempt=0)])
    assert wm.ready_stages(wf, done) == ["score"]
    failed = wm.fold_workflow_events([
        _ev("stage_failed", "20260713T120000000Z", stage="generate", attempt=0)])
    assert wm.ready_stages(wf, failed) == []               # a failed dep never readies
    assert wm.next_ready_stage(wf, failed) is None


def test_decide_retry_needs_both_the_class_and_a_remaining_attempt():
    stage = _wf().stages[0]                                # 2 attempts, infra only
    assert wm.decide_retry(stage, attempts_used=1, failure_class="infrastructure") \
        == "retry"
    assert wm.decide_retry(stage, attempts_used=2, failure_class="infrastructure") \
        == "fail"
    assert wm.decide_retry(stage, attempts_used=1, failure_class="entrypoint") \
        == "fail"


def test_controller_is_stale_treats_missing_and_unparseable_as_stale():
    now = "20260713T120000000Z"
    assert wm.controller_is_stale({}, now=now, stale_after_s=90) is True
    junk = {"controller": {"last_heartbeat_ts": "not-a-ts"}}
    assert wm.controller_is_stale(junk, now=now, stale_after_s=90) is True
    fresh = {"controller": {"last_heartbeat_ts": "20260713T115959000Z"}}
    assert wm.controller_is_stale(fresh, now=now, stale_after_s=90) is False
    old = {"controller": {"last_heartbeat_ts": "20260713T115000000Z"}}
    assert wm.controller_is_stale(old, now=now, stale_after_s=90) is True


def test_a_malformed_caller_clock_still_raises():
    """The recorded heartbeat is tolerated; the CALLER's own `now` is not — a
    bad value there is a caller bug and must surface."""
    with pytest.raises(wm.WorkflowIdError):
        wm.controller_is_stale({"controller": {"last_heartbeat_ts": "20260713T120000000Z"}},
                               now="garbage", stale_after_s=90)


# --------------------------------------------------------------------------- #
# Artifact binding
# --------------------------------------------------------------------------- #
def test_require_from_manifest_anchors_arms_to_the_manifest_frame():
    """The 2026-07-20 run-d8e9 fix: the manifest records an ABSOLUTE box path,
    and jobd matches require globs relative to the asset cache."""
    manifest = {"arms": {"a": {"path": "/workspace/jobs/g/work/results/gens_a.jsonl"},
                         "b": {"path": "/workspace/jobs/g/work/results/gens_b.jsonl"},
                         "c": {}}}
    got = wm.require_from_manifest(manifest)
    assert got == ["results/artifact-manifest.json",
                   "results/gens_a.jsonl", "results/gens_b.jsonl"]


def test_input_ref_asset_name_carries_the_manifest_sha():
    ref = InputRef(stage="generate", artifact="gens", dest="/workspace/in")
    a = wm.input_ref_asset(ref, gen_job_id="j0", manifest_sha256="0123456789abcdef",
                           require=["results/artifact-manifest.json"])
    assert a["name"] == "input-generate-0123456789ab"     # jobd caches BY NAME
    assert a["b2"] == "jobs/j0/results" and a["dest"] == "/workspace/in"
    assert a["mode"] == "copy" and a["optional"] is False
    with pytest.raises(Exception):
        wm.input_ref_asset(ref, gen_job_id="j0", manifest_sha256="short",
                           require=[])


# --------------------------------------------------------------------------- #
# The operator-facing table (herdd cmd_workflow_status prints it verbatim)
# --------------------------------------------------------------------------- #
def test_status_table_renders_verbatim_and_survives_a_partial_view():
    """Was `== flat.format_status_table(...)`; that comparison became a
    self-comparison at step 7, so the operator-facing rendering `herdd
    workflow status` prints is pinned against its GOLDEN text."""
    v = wm.fold_workflow_events([
        _ev("stage_planned", "20260713T120000000Z", stage="generate", attempt=0,
            job_id="j0")])
    extras = {"spend_usd": 1.5, "budget_usd": 10.0,
              "stages": {"generate": {"progress": "3/10"}}}
    out = wm.format_status_table(v, extras=extras)
    assert out == (
        "workflow=20260713T120000-e2-ab12 status=running terminal=False "
        "spend=1.5 budget=10.0\n"
        "STAGE     STATE          ATTEMPT  JOB  BOX  PROGRESS  SPEND  "
        "CKPT_AGE  FAILURE\n"
        "generate  stage_planned  0        j0   -    3/10      -      "
        "-         -      "
    )
    head, cols, row = out.splitlines()
    assert "status=running" in head and "spend=1.5" in head and "budget=10.0" in head
    assert cols.split()[:5] == ["STAGE", "STATE", "ATTEMPT", "JOB", "BOX"]
    assert row.split()[:4] == ["generate", "stage_planned", "0", "j0"]
    assert "-" in row                                     # missing cells render as -


def test_status_table_on_an_empty_view_is_header_plus_columns_only():
    out = wm.format_status_table(wm.fold_workflow_events([]))
    assert len(out.splitlines()) == 2


# --------------------------------------------------------------------------- #
# Zone P hygiene
# --------------------------------------------------------------------------- #
def test_module_does_not_mutate_sys_path():
    """The flat file did `sys.path.insert(0, _HERE)` at import; plan §3 forbids
    that anywhere inside the package. Checked over the AST, not the text, so a
    docstring that NAMES the forbidden call cannot pass or fail it."""
    tree = ast.parse(Path(wm.__file__).read_text())
    imported = {n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)}
    assert "sys" not in imported
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("insert", "append"):
            val = node.value
            assert not (isinstance(val, ast.Attribute) and val.attr == "path")


def test_module_imports_no_transport_and_no_cli():
    """The pure-lane boundary, over the import graph rather than the prose (the
    docstring cites `herdd` and `jobmeta` by name, deliberately)."""
    tree = ast.parse(Path(wm.__file__).read_text())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module)
    # `workflow` (the bare Zone E name) was replaced by `vastlib.workflows.spec`
    # at step 7, once the flat file became a re-export shim and both spellings
    # resolved to one set of classes. Neither is transport or CLI.
    assert mods <= {"__future__", "collections.abc", "datetime", "hashlib", "json",
                    "os", "re", "typing", "runmeta",
                    "vastlib.workflows.spec"}, sorted(mods)
