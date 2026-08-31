"""Portable tests for `workflowctl.py`'s M4-T1 surface: offline/online
`plan_workflow` preflight and the `rehearse_workflow` dependency-ordered
stage driver (roadmap "M4-T1: workflow plan and rehearsal").

Runs in the toolchain-free lane (`pytest -m "not integration"`): no rclone, no
real B2, no network, no torch/vLLM/CUDA, and NEVER shells out to the real
`rehearse.sh`. Every transport call goes through `test_jobmeta.FakeB2` (same
in-memory rclone-shaped runner `test_workflow.py` uses); every online resolver
(`asset_checker`/`image_resolver`/`cred_provider`) and the rehearsal driver's
`stage_rehearser` are hand-rolled fakes passed in explicitly. An autouse guard
fixture makes `jobmeta._default_runner`/`subprocess.run` raise if anything in
this file ever falls through to a real transport call.
"""
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobmeta  # noqa: E402
import workflowctl as wc  # noqa: E402
from test_jobmeta import FakeB2  # noqa: E402

BUCKET = "bkt"


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Same isolation discipline as test_workflow.py's `_isolated_env`."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("B2_BUCKET", BUCKET)
    monkeypatch.delenv("B2_WRITE_KEY_ID", raising=False)
    monkeypatch.delenv("VAST_INSTANCE_ID", raising=False)
    monkeypatch.delenv("INSTANCE_ID", raising=False)


@pytest.fixture(autouse=True)
def _no_real_transport(monkeypatch):
    """Hard guard (packet constraint: no real network/subprocess anywhere in
    this file): every real transport seam this module could accidentally
    fall through to (the real rclone runner, any subprocess) raises instead
    of running. Every test below injects a `runner`/`asset_checker`/
    `image_resolver`/`cred_provider`/`stage_rehearser` fake explicitly, so
    none of these should ever fire; if one does, that is itself the bug this
    guard exists to catch."""
    def _boom(*a, **kw):
        raise AssertionError(
            "real transport (rclone runner or subprocess) invoked in "
            "test_workflow_preflight.py -- inject a fake instead")
    monkeypatch.setattr(jobmeta, "_default_runner", _boom)
    monkeypatch.setattr(subprocess, "run", _boom)


# --- bundle-dir + workflow-module fixture builders ---------------------------
def _write_bundle_dir(tmp_path, name, *, broken=False):
    """A minimal real bundle dir: job-config.yaml + entrypoint, same shape
    test_workflow.py's `_write_bundle_dir` uses. `broken=True` omits the
    required `entrypoint:` key so `jobmeta.validate_job_config` rejects it —
    the fixture for the "invalid child bundle config" case."""
    d = tmp_path / f"bundle-{name}"
    d.mkdir()
    (d / "run.sh").write_text("#!/bin/sh\necho hi\n")
    lines = ["version: 1", f"name: {name}"]
    if not broken:
        lines.append("entrypoint: run.sh")
    lines.append("timeout_s: 60")
    (d / "job-config.yaml").write_text("\n".join(lines) + "\n")
    return str(d)


_GEN_DIGEST = "sha256:" + "a" * 64
_SCORE_DIGEST = "sha256:" + "b" * 64


def _workflow_source(gen_bundle, score_bundle, *, budget_usd=10.0,
                      gen_image="repo/image:tag", gen_digest=_GEN_DIGEST,
                      score_image="repo/eval:tag", score_digest=_SCORE_DIGEST,
                      gen_budget=6.0, score_budget=4.0,
                      input_stage="generate", input_artifact="generations",
                      wf_name="e2-paired-toy"):
    """Renders a `WORKFLOW = Workflow(...)` module source: a generate->score
    two-stage spec, matching test_workflow.py's `two_stage_workflow` shape but
    as a real `.py` FILE (`load_workflow_module` loads a path, not an
    in-memory object). Every knob defaults to a valid, fully-pinned spec;
    individual tests override exactly the one knob under test (`input_stage`/
    `input_artifact` for a bad InputRef, `*_digest` for image drift, ...)."""
    return f'''\
from workflow import (
    ArtifactContract, InputRef, JobStage, ResourceProfile, RetryPolicy, Workflow,
)

WORKFLOW = Workflow(
    version=1, name={wf_name!r}, budget_usd={budget_usd!r}, max_wall_s=3600,
    teardown="stop",
    profiles={{
        "generate": ResourceProfile(
            image={gen_image!r}, image_digest={gen_digest!r},
            gpu=("RTX 5090",), num_gpus=1, gpu_ram_gb=32, disk_gb=160,
            rental="bid", max_bid=1.0, budget_usd={gen_budget!r}, max_wall_s=3600),
        "score": ResourceProfile(
            image={score_image!r}, image_digest={score_digest!r},
            gpu=("RTX 3090",), num_gpus=1, gpu_ram_gb=24, disk_gb=160,
            rental="bid", max_bid=0.6, budget_usd={score_budget!r}, max_wall_s=3600),
    }},
    stages=(
        JobStage(
            name="generate", bundle={gen_bundle!r}, profile="generate",
            after=(), inputs={{}},
            outputs={{"generations": ArtifactContract(
                kind="e2-generations",
                manifest_path="results/artifact-manifest.json")}},
            retry=RetryPolicy(max_attempts=1, retry_on=())),
        JobStage(
            name="score", bundle={score_bundle!r}, profile="score",
            after=("generate",),
            inputs={{"generations": InputRef(
                stage={input_stage!r}, artifact={input_artifact!r},
                dest="inputs/generate")}},
            outputs={{"scores": ArtifactContract(
                kind="e2-scores",
                manifest_path="results/artifact-manifest.json")}},
            retry=RetryPolicy(max_attempts=1, retry_on=())),
    ),
)
'''


def _write_workflow_module(tmp_path, gen_bundle, score_bundle, *, filename="workflow.py",
                            **kw):
    path = tmp_path / filename
    path.write_text(_workflow_source(gen_bundle, score_bundle, **kw))
    return str(path)


# --- offline `workflow plan` --------------------------------------------------
def test_plan_offline_happy_path_two_stage(tmp_path):
    gen_bundle = _write_bundle_dir(tmp_path, "generate")
    score_bundle = _write_bundle_dir(tmp_path, "score")
    path = _write_workflow_module(tmp_path, gen_bundle, score_bundle)
    fake = FakeB2(bucket=BUCKET)

    rc, result = wc.plan_workflow(path, online=False, runner=fake, bucket=BUCKET)

    assert rc == wc.EXIT_OK
    assert "error" not in result
    assert result["wf_id"]
    assert [s["name"] for s in result["stages"]] == ["generate", "score"]
    assert all(s["bundle_ok"] and s["inputs_ok"] for s in result["stages"])
    # offline plan never touches the online-only report/disclaimer.
    assert "online" not in result
    assert "disclaimer" not in result


def test_plan_offline_bad_input_ref_undeclared_artifact_rejected(tmp_path):
    """An InputRef whose `artifact` was never declared as an output of its
    (correctly-`after`-listed) upstream stage. This cross-object rule is
    enforced by `wm.validate_workflow_spec` INSIDE `load_workflow_module`
    (before `plan_workflow` ever reaches its own per-stage
    `_check_stage_inputs` duplicate of the same check) -- so the failure
    surfaces as `plan_workflow`'s outer WorkflowCtlError catch: EXIT_INVALID
    with a descriptive `error`, no bundle/box ever touched."""
    gen_bundle = _write_bundle_dir(tmp_path, "generate")
    score_bundle = _write_bundle_dir(tmp_path, "score")
    path = _write_workflow_module(
        tmp_path, gen_bundle, score_bundle, input_artifact="bogus-artifact")
    fake = FakeB2(bucket=BUCKET)

    rc, result = wc.plan_workflow(path, online=False, runner=fake, bucket=BUCKET)

    assert rc == wc.EXIT_INVALID
    assert "bogus-artifact" in result["error"]
    assert "does not declare as an output" in result["error"]
    # nothing was written to the fake bucket -- rejected before write_spec.
    assert fake.store == {}


def test_plan_offline_bad_input_ref_non_after_stage_rejected(tmp_path):
    """Same load-time rejection, the other named case: InputRef.stage is not
    in the owning stage's declared `after=`."""
    gen_bundle = _write_bundle_dir(tmp_path, "generate")
    score_bundle = _write_bundle_dir(tmp_path, "score")
    path = _write_workflow_module(
        tmp_path, gen_bundle, score_bundle, input_stage="nonexistent-stage")
    fake = FakeB2(bucket=BUCKET)

    rc, result = wc.plan_workflow(path, online=False, runner=fake, bucket=BUCKET)

    assert rc == wc.EXIT_INVALID
    assert "nonexistent-stage" in result["error"]


def test_plan_offline_invalid_bundle_config_names_offending_stage(tmp_path):
    """`generate`'s bundle is valid; `score`'s bundle is missing the required
    `entrypoint:` key. Offline plan must fail closed with `CONFIG_INVALID`
    and name the SPECIFIC broken stage, not just "something is wrong"."""
    gen_bundle = _write_bundle_dir(tmp_path, "generate")
    score_bundle = _write_bundle_dir(tmp_path, "score", broken=True)
    path = _write_workflow_module(tmp_path, gen_bundle, score_bundle)
    fake = FakeB2(bucket=BUCKET)

    rc, result = wc.plan_workflow(path, online=False, runner=fake, bucket=BUCKET)

    assert rc == wc.EXIT_INVALID
    assert result["failure_class"] == "CONFIG_INVALID"
    assert result["stage"] == "score"
    assert "entrypoint" in result["error"]


# --- online `workflow plan --online` (every resolver faked) ------------------
def _ok_asset_checker(assets, *, runner, bucket):
    return []


def _stale_asset_checker(assets, *, runner, bucket):
    return [{"name": "some-asset", "b2": "jobs/x/results", "status": "stale",
             "sentinel": "manifest.txt", "detail": "local source changed",
             "src": "/repo/local"}]


def _matching_image_resolver(image):
    return {"repo/image:tag": _GEN_DIGEST, "repo/eval:tag": _SCORE_DIGEST}.get(image)


def _drifted_image_resolver(image):
    if image == "repo/eval:tag":
        return "sha256:" + "c" * 64  # deliberately wrong
    return _matching_image_resolver(image)


class FakeCredProvider:
    """`.current_expiry(stage_name) -> epoch_seconds`, the only method
    `_plan_online_credentials` calls. `expiries` maps stage name -> a plain
    epoch float (or the string 'raise' to simulate a transient read error,
    which `_plan_online_credentials` treats as non-terminal)."""
    def __init__(self, expiries):
        self.expiries = expiries
        self.calls = []

    def current_expiry(self, stage_name):
        self.calls.append(stage_name)
        v = self.expiries[stage_name]
        if v == "raise":
            raise RuntimeError("credential horizon read blip")
        return v


NOW_EPOCH = 1_000_000.0


def _valid_canary_checker(stage):
    # M4-T2 gate seam: check(stage) -> (status, key). Default = every stage has
    # a valid receipt, so the OTHER online checks (credential, spend) are reached.
    return ("valid", "key-" + stage.name)


def _canary_checker_returning(status):
    return lambda stage: (status, "key-" + stage.name)


def _online_ok_kwargs(**overrides):
    kw = dict(
        asset_checker=_ok_asset_checker, image_resolver=_matching_image_resolver,
        cred_provider=FakeCredProvider(
            {"generate": NOW_EPOCH + 999999, "score": NOW_EPOCH + 999999}),
        canary_checker=_valid_canary_checker,
        now_epoch=NOW_EPOCH)
    kw.update(overrides)
    return kw


def test_plan_online_happy_path_reports_digests_credential_and_spend(tmp_path):
    gen_bundle = _write_bundle_dir(tmp_path, "generate")
    score_bundle = _write_bundle_dir(tmp_path, "score")
    path = _write_workflow_module(tmp_path, gen_bundle, score_bundle)
    fake = FakeB2(bucket=BUCKET)

    rc, result = wc.plan_workflow(
        path, online=True, runner=fake, bucket=BUCKET, **_online_ok_kwargs())

    assert rc == wc.EXIT_OK
    assert "error" not in result
    online = result["online"]
    assert online["digests"] == {"generate": _GEN_DIGEST, "score": _SCORE_DIGEST}
    assert online["credential"]["checked"] is True
    assert online["credential"]["transient"] is False
    assert online["canary"]["generate"]["status"] == "valid"
    assert online["canary"]["score"]["status"] == "valid"
    assert online["worst_case_spend_usd"] == pytest.approx(10.0)
    assert result["disclaimer"] == wc.REHEARSAL_DISCLAIMER


def test_plan_online_image_drift_rejected(tmp_path):
    gen_bundle = _write_bundle_dir(tmp_path, "generate")
    score_bundle = _write_bundle_dir(tmp_path, "score")
    path = _write_workflow_module(tmp_path, gen_bundle, score_bundle)
    fake = FakeB2(bucket=BUCKET)

    rc, result = wc.plan_workflow(
        path, online=True, runner=fake, bucket=BUCKET,
        **_online_ok_kwargs(image_resolver=_drifted_image_resolver))

    assert rc == wc.EXIT_INVALID
    assert result["failure_class"] == "IMAGE_DRIFT"
    assert result["online"]["profile"] == "score"
    assert result["disclaimer"] == wc.REHEARSAL_DISCLAIMER


def test_plan_online_asset_stale_rejected(tmp_path):
    gen_bundle = _write_bundle_dir(tmp_path, "generate")
    score_bundle = _write_bundle_dir(tmp_path, "score")
    path = _write_workflow_module(tmp_path, gen_bundle, score_bundle)
    fake = FakeB2(bucket=BUCKET)

    rc, result = wc.plan_workflow(
        path, online=True, runner=fake, bucket=BUCKET,
        **_online_ok_kwargs(asset_checker=_stale_asset_checker))

    assert rc == wc.EXIT_INVALID
    assert result["failure_class"] == "ASSET_STALE"
    assert result["online"]["stage"] == "generate"
    assert result["disclaimer"] == wc.REHEARSAL_DISCLAIMER


def test_plan_online_credential_expires_inside_stage_wall(tmp_path):
    gen_bundle = _write_bundle_dir(tmp_path, "generate")
    score_bundle = _write_bundle_dir(tmp_path, "score")
    path = _write_workflow_module(tmp_path, gen_bundle, score_bundle)
    fake = FakeB2(bucket=BUCKET)
    # "score" profile's max_wall_s=3600; an expiry only 1800s out does not
    # outlast the stage's remaining wall bound -> CREDENTIAL_EXPIRES.
    cred = FakeCredProvider(
        {"generate": NOW_EPOCH + 999999, "score": NOW_EPOCH + 1800})

    rc, result = wc.plan_workflow(
        path, online=True, runner=fake, bucket=BUCKET,
        **_online_ok_kwargs(cred_provider=cred))

    assert rc == wc.EXIT_CREDENTIAL
    assert result["failure_class"] == "CREDENTIAL_EXPIRES"
    assert result["online"]["stage"] == "score"
    assert result["disclaimer"] == wc.REHEARSAL_DISCLAIMER


def test_plan_online_budget_exhausted_over_workflow_budget(tmp_path):
    gen_bundle = _write_bundle_dir(tmp_path, "generate")
    score_bundle = _write_bundle_dir(tmp_path, "score")
    # stage budgets sum to 10.0 (6.0 + 4.0, the template defaults); pin the
    # WORKFLOW budget below that worst-case sum.
    path = _write_workflow_module(tmp_path, gen_bundle, score_bundle, budget_usd=5.0)
    fake = FakeB2(bucket=BUCKET)

    rc, result = wc.plan_workflow(
        path, online=True, runner=fake, bucket=BUCKET, **_online_ok_kwargs())

    assert rc == wc.EXIT_INVALID
    assert result["failure_class"] == "BUDGET_EXHAUSTED"
    assert result["online"]["worst_case_spend_usd"] == pytest.approx(10.0)
    assert result["disclaimer"] == wc.REHEARSAL_DISCLAIMER


# --- M4-T2 canary gate (online plan refuses live spend without a receipt) -----
@pytest.mark.parametrize("status,fclass", [
    ("missing", "CANARY_MISSING"),
    ("expired", "CANARY_EXPIRED"),
    ("failed", "CANARY_FAILED"),
])
def test_plan_online_canary_absent_rejects_before_spend(tmp_path, status, fclass):
    gen_bundle = _write_bundle_dir(tmp_path, "generate")
    score_bundle = _write_bundle_dir(tmp_path, "score")
    path = _write_workflow_module(tmp_path, gen_bundle, score_bundle)
    fake = FakeB2(bucket=BUCKET)

    rc, result = wc.plan_workflow(
        path, online=True, runner=fake, bucket=BUCKET,
        **_online_ok_kwargs(canary_checker=_canary_checker_returning(status)))

    # EXIT_ARTIFACT (4), fails at the FIRST stage (generate), before cred/spend.
    assert rc == wc.EXIT_ARTIFACT
    assert result["failure_class"] == fclass
    assert result["online"]["stage"] == "generate"
    assert result["online"]["canary"]["generate"]["status"] == status
    assert "score" not in result["online"]["canary"]   # fail-fast: never checked
    assert result["disclaimer"] == wc.REHEARSAL_DISCLAIMER


def test_plan_online_canary_gate_ordering_after_digest(tmp_path):
    # A drifted digest must win over a missing canary: the canary key binds the
    # resolved digest, so digest verification is the earlier, more-specific fault.
    gen_bundle = _write_bundle_dir(tmp_path, "generate")
    score_bundle = _write_bundle_dir(tmp_path, "score")
    path = _write_workflow_module(tmp_path, gen_bundle, score_bundle)
    fake = FakeB2(bucket=BUCKET)

    rc, result = wc.plan_workflow(
        path, online=True, runner=fake, bucket=BUCKET,
        **_online_ok_kwargs(image_resolver=_drifted_image_resolver,
                            canary_checker=_canary_checker_returning("missing")))

    assert rc == wc.EXIT_INVALID
    assert result["failure_class"] == "IMAGE_DRIFT"   # not CANARY_MISSING


def test_plan_online_real_default_canary_checker_fails_closed_on_empty_b2(tmp_path):
    # Exercises the REAL _default_canary_checker (no injected checker): with an
    # empty receipt store it must compute a key and fail CANARY_MISSING, proving
    # the gate is fail-closed by default, not opt-in.
    gen_bundle = _write_bundle_dir(tmp_path, "generate")
    score_bundle = _write_bundle_dir(tmp_path, "score")
    path = _write_workflow_module(tmp_path, gen_bundle, score_bundle)
    fake = FakeB2(bucket=BUCKET)

    kw = _online_ok_kwargs()
    kw.pop("canary_checker")                  # use the real default
    rc, result = wc.plan_workflow(
        path, online=True, runner=fake, bucket=BUCKET, jobd_sha="test-jobd-sha", **kw)

    assert rc == wc.EXIT_ARTIFACT
    assert result["failure_class"] == "CANARY_MISSING"
    key = result["online"]["key"]
    assert isinstance(key, str) and len(key) == 64   # a real composite SHA-256


# --- M4-T2 canary receipt store (key determinism + TTL/status via FakeB2) -----
def test_canary_receipt_key_is_deterministic_and_component_sensitive():
    base = dict(image_digest="sha256:a", jobd_sha="jb",
                model_manifest_sha="m", adapter_manifest_sha="ad", recipe_sha="r")
    k = wc.canary_receipt_key(**base)
    assert k == wc.canary_receipt_key(**base)          # deterministic
    assert len(k) == 64
    for field in base:
        changed = dict(base, **{field: base[field] + "X"})
        assert wc.canary_receipt_key(**changed) != k   # every component matters


def test_canary_receipt_store_roundtrip_and_ttl():
    fake = FakeB2(bucket=BUCKET)
    key = wc.canary_receipt_key(image_digest="sha256:a", jobd_sha="jb",
                                model_manifest_sha="", adapter_manifest_sha="",
                                recipe_sha="r")
    now = 1_000.0
    good = {"expires_ts": now + 100, "rc": 0, "gpu_step_ok": True, "gpu_model": "P4000"}
    out = wc.write_canary_receipt(key, good, runner=fake, bucket=BUCKET)
    assert out["_stored"] is True

    assert wc.canary_receipt_status(key, runner=fake, bucket=BUCKET, now_epoch=now)[0] == "valid"
    assert wc.read_canary_receipt(key, runner=fake, bucket=BUCKET, now_epoch=now)["gpu_model"] == "P4000"
    # past expiry -> expired -> read returns None
    assert wc.canary_receipt_status(key, runner=fake, bucket=BUCKET, now_epoch=now + 200)[0] == "expired"
    assert wc.read_canary_receipt(key, runner=fake, bucket=BUCKET, now_epoch=now + 200) is None
    # never written -> missing
    assert wc.canary_receipt_status("no-such-key", runner=fake, bucket=BUCKET, now_epoch=now)[0] == "missing"


def test_canary_launch_env_key_matches_gate_and_pins_v1_contract(tmp_path):
    # F2 guard: the key the LAUNCHER mints must equal the key the GATE looks up,
    # and the v1 image-probe contract (model/adapter SHA == '') must hold — else
    # a produced receipt lands where the gate never reads it.
    gen_bundle = _write_bundle_dir(tmp_path, "generate")
    score_bundle = _write_bundle_dir(tmp_path, "score")
    path = _write_workflow_module(tmp_path, gen_bundle, score_bundle)
    wf = wc.load_workflow_module(path)
    stage_cfgs = {s.name: wc._validate_stage_bundle(s) for s in wf.stages}
    digests = {"generate": _GEN_DIGEST, "score": _SCORE_DIGEST}
    JOBD = "jobd-sha-under-test"

    for stage in wf.stages:
        key, env = wc.canary_launch_env(
            wf, stage, jobd_sha=JOBD, digests=digests, stage_cfgs=stage_cfgs)
        gate_key = wc.stage_canary_key(
            wf, stage, jobd_sha=JOBD, digests=digests, stage_cfgs=stage_cfgs)
        assert key == gate_key                       # launcher == gate
        assert env["CANARY_KEY"] == key
        assert env["CANARY_MODEL_SHA"] == ""         # v1 contract pinned
        assert env["CANARY_ADAPTER_SHA"] == ""
        assert env["CANARY_IMAGE_DIGEST"] == digests[stage.profile]

    # and a receipt stored at the launcher's key is what the online gate accepts
    fake = FakeB2(bucket=BUCKET)
    now = NOW_EPOCH
    for stage in wf.stages:
        key, _ = wc.canary_launch_env(
            wf, stage, jobd_sha=JOBD, digests=digests, stage_cfgs=stage_cfgs)
        wc.write_canary_receipt(
            key, {"expires_ts": now + 10_000, "rc": 0, "gpu_step_ok": True},
            runner=fake, bucket=BUCKET)
    kw = _online_ok_kwargs()
    kw.pop("canary_checker")                         # exercise the REAL checker
    rc, result = wc.plan_workflow(
        path, online=True, runner=fake, bucket=BUCKET, jobd_sha=JOBD, **kw)
    assert rc == wc.EXIT_OK
    assert result["online"]["canary"]["generate"]["status"] == "valid"
    assert result["online"]["canary"]["score"]["status"] == "valid"


def test_canary_receipt_status_failed_when_rc_or_step_bad():
    fake = FakeB2(bucket=BUCKET)
    now = 1_000.0
    for body in ({"expires_ts": now + 100, "rc": 21, "gpu_step_ok": True},
                 {"expires_ts": now + 100, "rc": 0, "gpu_step_ok": False}):
        key = wc.canary_receipt_key(image_digest="sha256:" + str(body["rc"]),
                                    jobd_sha="jb", model_manifest_sha="",
                                    adapter_manifest_sha="", recipe_sha="r")
        wc.write_canary_receipt(key, body, runner=fake, bucket=BUCKET)
        assert wc.canary_receipt_status(key, runner=fake, bucket=BUCKET, now_epoch=now)[0] == "failed"
        assert wc.read_canary_receipt(key, runner=fake, bucket=BUCKET, now_epoch=now) is None


# --- `workflow rehearse` (FAKE stage_rehearser -- real rehearse.sh NEVER run) -
def _make_fake_rehearser(write_manifests, *, return_manifests=None, rc_overrides=None):
    """A fake `stage_rehearser(stage, bundle_dir, asset_overrides, results_out)
    -> (rc, manifest)`. `write_manifests[stage.name]` is persisted to
    `<results_out>/results/artifact-manifest.json` (what `_read_stage_manifest`
    would later re-read from disk); `return_manifests[stage.name]` (defaults to
    the SAME object as written) is what this call returns directly. Diverging
    the two lets a test simulate a downstream manifest mismatch without ever
    touching the real rehearse.sh. `calls` records every invocation in order
    for dependency-order/injection assertions."""
    return_manifests = return_manifests or {}
    rc_overrides = rc_overrides or {}
    calls = []

    def _rehearser(stage, bundle_dir, asset_overrides, results_out):
        calls.append({"stage": stage.name, "bundle_dir": bundle_dir,
                       "asset_overrides": dict(asset_overrides),
                       "results_out": results_out})
        rc = rc_overrides.get(stage.name, 0)
        if rc != 0:
            return rc, None
        manifest = write_manifests.get(stage.name)
        if manifest is not None:
            results_dir = os.path.join(results_out, "results")
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, "artifact-manifest.json"), "w") as fh:
                json.dump(manifest, fh)
        return 0, return_manifests.get(stage.name, manifest)

    return _rehearser, calls


_GEN_MANIFEST = {"v": 1, "kind": "e2-generations", "rows": 400}
_SCORE_MANIFEST = {"v": 1, "kind": "e2-scores", "rows": 400}


def test_rehearse_runs_dependency_order_and_injects_upstream_artifact(tmp_path):
    gen_bundle = _write_bundle_dir(tmp_path, "generate")
    score_bundle = _write_bundle_dir(tmp_path, "score")
    path = _write_workflow_module(tmp_path, gen_bundle, score_bundle)
    rehearser, calls = _make_fake_rehearser(
        {"generate": _GEN_MANIFEST, "score": _SCORE_MANIFEST})
    workdir = str(tmp_path / "rehearse-work")

    rc, result = wc.rehearse_workflow(
        path, wf_id="20260713T000000-e2-paired-toy-a0a0", workdir=workdir,
        stage_rehearser=rehearser)

    assert rc == wc.EXIT_OK
    assert [c["stage"] for c in calls] == ["generate", "score"]
    # generate has no inputs -> no asset overrides at all.
    assert calls[0]["asset_overrides"] == {}
    # score's declared InputRef("generations") was injected as its asset
    # override, pointing at generate's OWN captured results_dir.
    gen_results_dir = os.path.join(workdir, "generate")
    assert calls[1]["asset_overrides"] == {"generations": gen_results_dir}
    names = [s["name"] for s in result["stages"]]
    assert names == ["generate", "score"]
    gen_report = result["stages"][0]
    assert gen_report["manifest_sha256"] == wc._stable_manifest_sha256(_GEN_MANIFEST)
    assert result["disclaimer"] == wc.REHEARSAL_DISCLAIMER


def test_rehearse_mismatched_downstream_manifest_sha_fails(tmp_path):
    """`generate`'s rehearser call RETURNS one manifest (what gets hashed and
    stored as "produced") but WRITES a different one to disk (what a
    downstream stage would actually read fresh) -- simulating a corrupted/
    non-deterministic rehearsal artifact. `score`'s manifest-consistency
    check must catch this and fail closed, never silently proceed."""
    gen_bundle = _write_bundle_dir(tmp_path, "generate")
    score_bundle = _write_bundle_dir(tmp_path, "score")
    path = _write_workflow_module(tmp_path, gen_bundle, score_bundle)
    corrupted_on_disk = dict(_GEN_MANIFEST, rows=1)
    rehearser, calls = _make_fake_rehearser(
        {"generate": corrupted_on_disk, "score": _SCORE_MANIFEST},
        return_manifests={"generate": _GEN_MANIFEST})
    workdir = str(tmp_path / "rehearse-work")

    rc, result = wc.rehearse_workflow(
        path, wf_id="20260713T000000-e2-paired-toy-b1b1", workdir=workdir,
        stage_rehearser=rehearser)

    assert rc == wc.EXIT_ARTIFACT
    assert result["failure_class"] == "ARTIFACT_INVALID"
    assert result["stage"] == "score"
    assert result["input"] == "generations"
    assert result["disclaimer"] == wc.REHEARSAL_DISCLAIMER
    # score's own rehearser call never ran -- the mismatch is caught BEFORE
    # the downstream stage is dispatched.
    assert [c["stage"] for c in calls] == ["generate"]


# --- fixed disclaimer ----------------------------------------------------------
def test_rehearsal_disclaimer_is_fixed_and_names_cuda_vllm():
    assert isinstance(wc.REHEARSAL_DISCLAIMER, str)
    assert wc.REHEARSAL_DISCLAIMER.strip()
    assert "CUDA" in wc.REHEARSAL_DISCLAIMER
    assert "vLLM" in wc.REHEARSAL_DISCLAIMER
