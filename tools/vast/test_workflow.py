"""Portable tests for workflowctl.py — fake-I/O reconcile idempotency, local
+ remote controller liveness, and CLI exit-code mapping (roadmap "M2-T3:
workflow CLI + reconciler").

Runs in the toolchain-free lane (`pytest -m "not integration"`): no rclone,
no real B2, no network, no torch/vLLM/CUDA. Every transport call goes through
`test_jobmeta.FakeB2`, an in-memory rclone-shaped runner (rcat/cat/lsf/copy).

The heart of this file (per the packet): drive a toy generate->score workflow
through `workflowctl.reconcile_tick` and, BETWEEN EVERY action, call it again
fresh (a new Python call re-folding the SAME fake B2 store — `reconcile_tick`
is a pure-per-call function of that store, so "recreate the reconciler" is
exactly what happens on every call already) and assert the action sequence
never re-submits an already-submitted stage, never skips a ready stage, and
never accepts a downstream stage before its dependency's artifact is in.
"""
import argparse
import hashlib
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataclasses  # noqa: E402
import imageref  # noqa: E402
import jobmeta  # noqa: E402
import bidpolicy  # noqa: E402
import herdd  # noqa: E402
# THE SUBJECT is the ported controller. `workflowctl.py` is a re-export shim
# since plan step 7, and a re-export is NOT a steering seam: patching
# `workflowctl.reconcile_tick` would rebind the shim's attribute while every
# ported caller resolves the name in `vastlib.workflows.ctl`'s globals — 14
# monkeypatch sites in this file would have gone vacuous (green, steering
# nothing). Hence `wc` IS the port; the separate `vl_ctl` alias this file
# carried through step 6d collapsed into it in the same commit.
from vastlib.workflows import ctl as wc  # noqa: E402
from vastlib.launch import launch as vl_launch  # noqa: E402
from vastlib.launch import spec as launch_spec  # noqa: E402
from vastlib.storage import b2  # noqa: E402
import workflowmeta as wm  # noqa: E402
from vastlib.core import api  # noqa: E402
from vastlib.market import offers  # noqa: E402
from test_jobmeta import FakeB2  # noqa: E402
from workflow import (  # noqa: E402
    ArtifactContract, InputRef, JobStage, ResourceProfile, RetryPolicy, Workflow,
    WorkflowError,
)

BUCKET = "bkt"
ACTOR = "cli:controller-host"


def T(n: int) -> str:
    """A valid runmeta timestamp (YYYYMMDDTHHMMSSmmmZ) whose HHMMSS digits
    are `n`, zero-padded — same convention as test_workflowmeta.py's `T`, so
    `T(300) - T(0) == 180s` by wall-clock arithmetic, useful for staleness
    math without a real clock."""
    return f"20260713T{n:06d}000Z"


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Every test gets its own local-cache dir (flock lockfile, jobmeta's
    incremental event cache, this module's bundle staging dir) and a clean
    B2_WRITE_KEY_ID/B2_BUCKET so `jobmeta._wq`/`_bucket` behave predictably
    against a `FakeB2` whose bucket is always `BUCKET`."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("B2_BUCKET", BUCKET)
    monkeypatch.delenv("B2_WRITE_KEY_ID", raising=False)
    monkeypatch.delenv("VAST_INSTANCE_ID", raising=False)
    monkeypatch.delenv("INSTANCE_ID", raising=False)
    # The velvet P3 gate resolves tag digests through imageref's TTL'd cache,
    # which is PROCESS-global by design (a controller must not re-hit the
    # registry per tick). Left uncleared, one test's fake digest would answer
    # the next test's lookup for 15 minutes of wall clock.
    imageref.clear_ttl_cache()
    imageref._digest_cache.clear()
    imageref._ref_digest_cache.clear()


def _fake_jobs_composer(env, onstart, *, dry_run=False, key_base=None,
                        bootstrap_stager=None, **kw):
    """Hermetic stand-in for `herdd.compose_jobs_launch_env` used by the
    fake-transport box_resolver tests: mutates `env` with the jobs cred keys +
    prepends a jobd-boot marker onstart WITHOUT minting a B2 key or reading
    B2_S3_ENDPOINT (the real composer `sys.exit`s without one). Honors
    `bootstrap_stager` so the `box_acquired` event's `bootstrap_sha` still comes
    from the injected stager. The REAL composer path (jobd onstart + scoped B2
    env + runtype actually reaching the launch body) is covered separately by
    `test_build_box_resolver_composes_real_jobd_body`."""
    env.setdefault("CRED_ROLE", "jobs")
    env.setdefault("B2_KEY_ID", "ro-key")
    env.setdefault("B2_BUCKET", BUCKET)
    sha = bootstrap_stager(dry_run=dry_run) if bootstrap_stager is not None else "boot-sha"
    return "#jobd-boot\n" + (onstart or ""), sha


# --- fixture builders (mirrors test_workflowmeta.py's pinned_profile) --------
def pinned_profile(**overrides):
    kw = dict(image="repo/image:tag", image_digest="sha256:" + "a" * 64,
              gpu=("RTX 5090",), num_gpus=1, gpu_ram_gb=32, disk_gb=160,
              rental="bid", max_bid=1.0, budget_usd=6.0, max_wall_s=6 * 3600)
    kw.update(overrides)
    return ResourceProfile(**kw)


def _write_bundle_dir(tmp_path, name):
    """A minimal real bundle dir: job-config.yaml + entrypoint, exactly what
    `jobmeta.load_job_config`/`validate_job_config` need (no assets — this
    subtask's reconcile test appends the InputRef asset dynamically, the same
    way `_build_stage_config` does for a real score stage)."""
    d = tmp_path / f"bundle-{name}"
    d.mkdir()
    (d / "run.sh").write_text("#!/bin/sh\necho hi\n")
    (d / "job-config.yaml").write_text(
        "version: 1\n"
        f"name: {name}\n"
        "entrypoint: run.sh\n"
        "timeout_s: 60\n")
    return str(d)


def two_stage_workflow(tmp_path, *, retry_on=(), max_attempts=1):
    """generate -> score, real bundle dirs under tmp_path so
    `workflowctl._build_stage_config`/`_ensure_bundle_uploaded` can actually
    run against them (no torch/CUDA/network — just tar+zstd+FakeB2)."""
    gen_profile = pinned_profile()
    score_profile = pinned_profile(image="repo/eval:tag", max_bid=0.6, budget_usd=4.0)
    generate = JobStage(
        name="generate", bundle=_write_bundle_dir(tmp_path, "generate"),
        profile="generate", after=(), inputs={},
        outputs={"generations": ArtifactContract(
            kind="e2-generations", manifest_path="results/artifact-manifest.json")},
        retry=RetryPolicy(max_attempts=max_attempts, retry_on=retry_on),
    )
    score = JobStage(
        name="score", bundle=_write_bundle_dir(tmp_path, "score"),
        profile="score", after=("generate",),
        inputs={"generations": InputRef(
            stage="generate", artifact="generations", dest="inputs/generate")},
        outputs={"scores": ArtifactContract(
            kind="e2-scores", manifest_path="results/artifact-manifest.json")},
        retry=RetryPolicy(max_attempts=1, retry_on=()),
    )
    return Workflow(
        version=1, name="e2-paired-toy", budget_usd=10.0, max_wall_s=12 * 3600,
        teardown="stop", profiles={"generate": gen_profile, "score": score_profile},
        stages=(generate, score),
    )


def fixed_box_resolver(stage, wf, attempt=0):
    """Always offers box "44" — a fixed id is enough to drive plan+submit
    without exercising the real M3-T1 `build_box_resolver` acquisition path
    (covered separately below)."""
    return "44"


def finish_job(fake, job_id, *, kind, arm_rel="out.txt", arm_body="hello",
               box="44"):
    """Simulate a box's jobd running a submitted job to completion: claimed
    -> started -> done, plus the DONE marker + a valid artifact manifest so
    `jobmeta.validate_generation_artifact` accepts it."""
    for ev, kw in (("claimed", {}), ("started", {})):
        jobmeta.emit_event(job_id, ev, actor=f"box:{box}", runner=fake,
                           bucket=BUCKET, instance_id=box, **kw)
    jobmeta.emit_event(job_id, "done", actor=f"box:{box}", runner=fake,
                       bucket=BUCKET, instance_id=box, rc=0)
    fake.store[f"jobs/{job_id}/results.DONE.json"] = json.dumps({"v": 1, "rc": 0})
    sha = hashlib.sha256(arm_body.encode("utf-8")).hexdigest()
    manifest = {"v": 1, "kind": kind,
                "arms": {"a0": {"path": arm_rel, "sha256": sha}}}
    fake.store[f"jobs/{job_id}/results/results/artifact-manifest.json"] = \
        json.dumps(manifest)
    fake.store[f"jobs/{job_id}/results/{arm_rel}"] = arm_body


def _event_names(fake, wf_id):
    """Every workflow event name currently on B2 for `wf_id` (file order, not
    the fold's ts-sorted order) — used to assert an action never duplicates
    or silently skips a workflow event."""
    return [json.loads(r).get("event")
            for r in wc.read_events(wf_id, runner=fake, bucket=BUCKET)]


# --- 1. reconcile idempotency: no duplicate, no skip (the heart of M2-T3) ----
def test_reconcile_idempotent_no_duplicate_no_skip(tmp_path):
    """Drive generate->score through `reconcile_tick`, calling it FRESH every
    single time. `reconcile_tick` is a pure function of the fake B2 store (no
    reconciler object carries state between calls) — so "kill and recreate
    the reconciler between every action" is exactly what already happens on
    every one of these calls. Assert the action sequence never re-submits an
    already-submitted stage, never skips a ready stage, and never starts
    `score` before `generate`'s artifact is accepted."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    actions = []

    def tick(n):
        r = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                               now=T(n), box_resolver=fixed_box_resolver)
        actions.append(r["action"])
        return r

    # 1) generate has no `after` -- ready immediately, submitted exactly once
    r1 = tick(10)
    assert (r1["action"], r1["stage"]) == ("stage_submitted", "generate")
    gen_job_id = r1["job_id"]
    assert f"jobs/queue/44/{gen_job_id}.json" in fake.store

    # explicit primitive-level idempotency: re-submitting the SAME
    # deterministic ticket is a no-op (jobmeta.submit_with_id's own
    # contract), independent of whether reconcile_tick would ever re-enter
    # the submit branch for an already-submitted stage
    cfg, bundle_dir = wc._build_stage_config(wf_id, wf.stages[0], runner=fake, bucket=BUCKET)
    sha = wc._ensure_bundle_uploaded(bundle_dir, runner=fake, bucket=BUCKET)
    resubmit = jobmeta.submit_with_id(gen_job_id, cfg, "44", bundle_sha256=sha,
                                       actor=ACTOR, runner=fake, bucket=BUCKET)
    assert resubmit["status"] == "noop"

    # 2) re-ticking before the box has done anything is a pure no-op: no
    # duplicate `stage_submitted`, no premature `score`
    events_after_1 = _event_names(fake, wf_id)
    r2 = tick(20)
    assert r2["action"] == "noop_running"
    assert _event_names(fake, wf_id) == events_after_1        # zero new events
    v2 = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert "score" not in v2["stages"]                        # never skipped-ahead

    # box finishes the generate job
    finish_job(fake, gen_job_id, kind="e2-generations")

    # 3) artifact acceptance is its OWN action (not folded into stage_succeeded)
    r3 = tick(30)
    assert r3["action"] == "artifact_accepted" and r3["artifact"] == "generations"
    v3 = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert "score" not in v3["stages"]      # still not ready: generate's own
                                             # artifact is accepted, but it is
                                             # not `stage_succeeded` yet

    # 4) THEN stage_succeeded, exactly once
    r4 = tick(40)
    assert r4["action"] == "stage_succeeded" and r4["stage"] == "generate"

    # 5) NOW (and only now) score becomes ready and is submitted
    v4 = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert wm.next_ready_stage(wf, v4) == "score"
    r5 = tick(50)
    assert (r5["action"], r5["stage"]) == ("stage_submitted", "score")
    score_job_id = r5["job_id"]
    assert score_job_id != gen_job_id

    r6 = tick(60)
    assert r6["action"] == "noop_running"

    finish_job(fake, score_job_id, kind="e2-scores")

    r7 = tick(70)
    assert r7["action"] == "artifact_accepted" and r7["artifact"] == "scores"
    r8 = tick(80)
    assert r8["action"] == "stage_succeeded" and r8["stage"] == "score"

    # 6) completion: verdict written once, then workflow_succeeded, then a
    # terminal workflow reconciles to teardown-only forever after (no owned
    # boxes in this M2 lane -> immediately `noop_terminal`)
    r9 = tick(90)
    assert r9["action"] == "teardown_started"
    assert fake.store.get(f"workflows/{wf_id}/verdict.json")
    r10 = tick(100)
    assert r10["action"] == "workflow_succeeded"
    r11 = tick(110)
    assert r11["action"] == "noop_terminal" and r11["status"] == "succeeded"

    assert actions == [
        "stage_submitted", "noop_running", "artifact_accepted", "stage_succeeded",
        "stage_submitted", "noop_running", "artifact_accepted", "stage_succeeded",
        "teardown_started", "workflow_succeeded", "noop_terminal",
    ]
    # exactly one PER STAGE across the whole run for each milestone event --
    # never duplicated, never skipped.
    names = _event_names(fake, wf_id)
    for ev_name in ("stage_submitted", "artifact_accepted", "stage_succeeded"):
        assert names.count(ev_name) == 2, (ev_name, names)


# --- 1b. crash/recover: a ticket written but NO workflow event yet ----------
def test_reconcile_recovers_partial_submit_no_double_ticket(tmp_path):
    """Atomicity guard for `_plan_and_submit_stage`: it now writes the durable
    job ticket (`submit_with_id`) BEFORE emitting the `stage_planned` /
    `stage_submitted` workflow events. Simulate a controller crash in that
    window — a ticket already on B2 for generate's deterministic job_id, but
    zero workflow stage events — and assert the next reconcile tick RECOVERS
    (re-enters via next_ready_stage, re-submits idempotently, emits the stage
    events) rather than either deadlocking or writing a SECOND ticket."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    # pre-write the ticket exactly as a crashed-mid-submit controller would
    # have left it (deterministic job_id, identical config+bundle), with NO
    # workflow stage events emitted yet.
    gen_job_id = wm.stage_job_id(wf_id, "generate", 0)
    cfg, bundle_dir = wc._build_stage_config(wf_id, wf.stages[0], runner=fake, bucket=BUCKET)
    sha = wc._ensure_bundle_uploaded(bundle_dir, runner=fake, bucket=BUCKET)
    jobmeta.submit_with_id(gen_job_id, cfg, "44", bundle_sha256=sha, actor=ACTOR,
                           runner=fake, bucket=BUCKET)
    assert f"jobs/queue/44/{gen_job_id}.json" in fake.store
    assert _event_names(fake, wf_id) == []          # no workflow events yet

    r = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                          now=T(10), box_resolver=fixed_box_resolver)
    assert (r["action"], r["stage"], r["job_id"]) == ("stage_submitted", "generate", gen_job_id)
    assert r["submit"]["status"] == "noop"          # idempotent: no second ticket
    tickets = [k for k in fake.store if k.startswith("jobs/queue/")]
    assert tickets == [f"jobs/queue/44/{gen_job_id}.json"], tickets
    names = _event_names(fake, wf_id)
    assert names.count("stage_submitted") == 1 and names.count("stage_planned") == 1


# --- 1c. retryable failure -> new attempt -> success ------------------------
def test_reconcile_infrastructure_failure_retries_then_succeeds(tmp_path):
    """An infrastructure-class (retryable) generate failure with budget left
    mints attempt+1 under a NEW deterministic job_id and drives to success —
    the retry path (`wm.decide_retry` -> re-plan+submit) that the primary
    idempotency test never exercises."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path, retry_on=("infrastructure",), max_attempts=2)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    def tick(n):
        return wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                                 now=T(n), box_resolver=fixed_box_resolver)

    r1 = tick(10)
    assert (r1["action"], r1["attempt"]) == ("stage_submitted", 0)
    a0_job = r1["job_id"]

    # box dies with a preemption (infrastructure) reason -- retryable
    for ev in ("claimed", "started"):
        jobmeta.emit_event(a0_job, ev, actor="box:44", runner=fake, bucket=BUCKET,
                           instance_id="44")
    jobmeta.emit_event(a0_job, "failed", actor="box:44", runner=fake, bucket=BUCKET,
                       instance_id="44", rc=1, reason="spot instance preempted")

    r2 = tick(20)
    assert (r2["action"], r2["stage"], r2["attempt"]) == ("stage_submitted", "generate", 1)
    a1_job = r2["job_id"]
    assert a1_job != a0_job                          # fresh deterministic id

    # the retried attempt now completes cleanly
    finish_job(fake, a1_job, kind="e2-generations")
    assert tick(30)["action"] == "artifact_accepted"
    assert tick(40)["action"] == "stage_succeeded"
    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert v["stages"]["generate"]["status"] == "stage_succeeded"
    assert v["stages"]["generate"]["attempt"] == 1


# --- 1d. run_controller: loop + local-lock lifecycle + exit-code mapping -----
def test_run_controller_terminal_status_maps_exit_code_and_releases_lock(tmp_path):
    """`run_controller` on an already-terminal (failed) workflow claims the
    role, ticks once (teardown-only), maps the folded terminal status to the
    frozen exit code (failed->EXIT_FAILED), and ALWAYS releases the local flock
    in `finally` (re-acquirable afterward)."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.emit(wf_id, "workflow_failed", ACTOR, runner=fake, bucket=BUCKET, ts=T(0),
            failure_class="RETRY_EXHAUSTED")

    rc = wc.run_controller(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                           clock=lambda: T(500), sleep_fn=lambda *_: None, max_ticks=5)
    assert rc == wc.EXIT_FAILED == 2
    # lock was released in finally -> a fresh acquire succeeds and does not raise
    h = wc.acquire_local_lock(wf_id)
    wc.release_local_lock(h)


def test_run_controller_detached_terminal_failed_exits_ok_foreground_keeps_code(tmp_path):
    """Regression for the 2026-07-16 detached flap (E2 run 0d9d): the systemd
    unit is `Restart=on-failure`, so a detached controller that mapped a
    terminal-FAILED workflow to EXIT_FAILED was restarted forever (57
    `controller_started` events over 9.5h, ~one per 4-6 min, every restart
    re-reading the same terminal spec). Under `detached_controller=True` (the
    `--detached-controller` re-exec flag) a terminal workflow — success OR
    failure — returns EXIT_OK so systemd lets the unit finish (the verdict is
    already durable in the B2 event log); the FOREGROUND path on the very
    same terminal workflow keeps returning the real failure code (operator-
    visible pass/fail). The split is on detach, not on outcome."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.emit(wf_id, "workflow_failed", ACTOR, runner=fake, bucket=BUCKET, ts=T(0),
            failure_class="RETRY_EXHAUSTED")

    rc = wc.run_controller(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                           clock=lambda: T(500), sleep_fn=lambda *_: None,
                           max_ticks=5, detached_controller=True)
    assert rc == wc.EXIT_OK == 0            # no restart signal to systemd

    # foreground on the SAME terminal workflow (same-actor re-claim is
    # instant): the real terminal exit code stays operator-visible
    rc_fg = wc.run_controller(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                              clock=lambda: T(600), sleep_fn=lambda *_: None,
                              max_ticks=5)
    assert rc_fg == wc.EXIT_FAILED == 2


def test_run_controller_detached_crash_still_signals_restart(tmp_path, monkeypatch):
    """The other half of the flap fix's contract: `detached_controller=True`
    maps ONLY a cleanly-reached terminal workflow to EXIT_OK. A genuine
    controller CRASH (unhandled exception mid-tick, before any terminal) must
    still propagate — the CLI exits non-zero and the systemd unit's
    `Restart=on-failure` restarts it (crash recovery stays intact). The lock
    is still released in `finally` so the restarted controller can re-claim."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    def boom(*a, **kw):
        raise RuntimeError("boom")
    monkeypatch.setattr(wc, "reconcile_tick", boom)

    with pytest.raises(RuntimeError, match="boom"):
        wc.run_controller(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                          clock=lambda: T(500), sleep_fn=lambda *_: None,
                          max_ticks=5, detached_controller=True)
    # finally released the flock -> the restarted unit can re-claim
    h = wc.acquire_local_lock(wf_id)
    wc.release_local_lock(h)


def test_resume_workflow_detach_appends_detached_controller_flag(tmp_path, monkeypatch):
    """`resume --detach`'s re-exec argv (and so every Restart=on-failure
    re-run) must carry `--detached-controller`, same as `run --detach` —
    resume is exactly the path the flapping unit re-executes."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)
    runs = []
    _fake_systemd(monkeypatch, runs)

    argv = [sys.executable, "/abs/herdd.py", "workflow", "resume", wf_id]
    rc, result = wc.resume_workflow(wf_id, actor=ACTOR, detach=True,
                                    argv=argv, runner=fake, bucket=BUCKET)
    assert rc == wc.EXIT_OK
    launch = runs[-1]
    child = launch[launch.index("--") + 1:]
    assert child == [str(x) for x in argv] + ["--detached-controller"]


def test_run_controller_max_ticks_bound_submits_and_releases_lock(tmp_path):
    """A non-terminal workflow: `run_controller` submits the first ready stage,
    honors `max_ticks` (no real sleeping via injected `sleep_fn`), returns
    EXIT_OK on the bound, and releases the lock."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    slept = []
    rc = wc.run_controller(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                           clock=lambda: T(300), sleep_fn=lambda s: slept.append(s),
                           max_ticks=1, box_resolver=fixed_box_resolver)
    assert rc == wc.EXIT_OK == 0
    assert slept == []                               # bound hit before any sleep
    gen_job_id = wm.stage_job_id(wf_id, "generate", 0)
    assert f"jobs/queue/44/{gen_job_id}.json" in fake.store
    h = wc.acquire_local_lock(wf_id)                 # lock released in finally
    wc.release_local_lock(h)


def test_run_controller_refuses_second_live_controller(tmp_path):
    """`run_controller` surfaces `claim_controller`'s refusal (a live 2nd
    controller) as a raised `WorkflowCtlError`, and still releases the local
    lock so a later (post-staleness) attempt can proceed."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)
    wc.emit(wf_id, "controller_started", "cli:other", runner=fake, bucket=BUCKET, ts=T(0))
    wc.emit(wf_id, "controller_heartbeat", "cli:other", runner=fake, bucket=BUCKET, ts=T(30))

    with pytest.raises(wc.WorkflowCtlError):
        wc.run_controller(wf, wf_id, runner=fake, bucket=BUCKET, actor="cli:me",
                          clock=lambda: T(40), sleep_fn=lambda *_: None, max_ticks=1,
                          box_resolver=fixed_box_resolver)
    h = wc.acquire_local_lock(wf_id)                 # released despite the raise
    wc.release_local_lock(h)


# --- 2. terminal precedence: failed beats succeeded, teardown-only after ----
def test_terminal_precedence_and_teardown_only_tick(tmp_path):
    """A store carrying both `workflow_succeeded` and `workflow_failed` folds
    to `failed` (workflowmeta's terminal-precedence rule: an observed real
    failure outranks a stray/late success). A reconcile tick on an already-
    terminal workflow is teardown-only -- never a new submission."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.emit(wf_id, "workflow_succeeded", ACTOR, runner=fake, bucket=BUCKET, ts=T(0))
    wc.emit(wf_id, "workflow_failed", ACTOR, runner=fake, bucket=BUCKET, ts=T(1),
            failure_class="RETRY_EXHAUSTED")

    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert v["terminal"] is True
    assert v["status"] == "failed"          # terminal precedence: failed > succeeded

    result = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                                now=T(2), box_resolver=fixed_box_resolver)
    assert result == {"action": "noop_terminal", "status": "failed"}
    # no job/ticket was ever written -- teardown-only, no new submission
    assert not any(k.startswith("jobs/queue/") for k in fake.store)


# --- 3. a failed dependency blocks the downstream stage forever -------------
def test_dependency_failure_blocks_downstream(tmp_path):
    """`generate` fails (no retry budget) -> `score` never becomes ready
    (`wm.next_ready_stage` stays None), the fold reports the causal stage,
    and no score ticket is ever written."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path, retry_on=(), max_attempts=1)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    r1 = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                            now=T(10), box_resolver=fixed_box_resolver)
    assert (r1["action"], r1["stage"]) == ("stage_submitted", "generate")
    gen_job_id = r1["job_id"]

    # box reports the job dead (no DONE marker, no artifact) -- an entrypoint
    # failure, not a retryable infra class
    jobmeta.emit_event(gen_job_id, "claimed", actor="box:44", runner=fake,
                        bucket=BUCKET, instance_id="44")
    jobmeta.emit_event(gen_job_id, "started", actor="box:44", runner=fake,
                        bucket=BUCKET, instance_id="44")
    jobmeta.emit_event(gen_job_id, "failed", actor="box:44", runner=fake,
                        bucket=BUCKET, instance_id="44", rc=1,
                        reason="entrypoint exited nonzero")

    r2 = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                            now=T(20), box_resolver=fixed_box_resolver)
    assert r2["action"] == "stage_failed" and r2["stage"] == "generate"
    assert r2["event"]["failure_class"] == "ENTRYPOINT_FAILED"

    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert v["stages"]["generate"]["status"] == "stage_failed"     # the causal stage
    assert wm.next_ready_stage(wf, v) is None                      # score never ready
    assert "score" not in v["stages"]

    # the reconciler still drains to a terminal workflow_failed (dependency
    # failure, not a retry-exhausted stage) -- but at no point is a score
    # ticket written
    for n in (30, 40, 50):
        r = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                               now=T(n), box_resolver=fixed_box_resolver)
        if r["action"] == "noop_terminal":
            break
    final = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert final["terminal"] is True and final["status"] == "failed"
    assert not any("score" in k for k in fake.store if k.startswith("jobs/queue/"))


# --- 4/5. controller liveness: refuse a 2nd live controller; stale takeover -
# POLL_INTERVAL_S * HEARTBEAT_STALE_MULT = 30 * 3 = 90s -- pick T() offsets
# that are unambiguously inside/outside that window (T(n) encodes n as literal
# HHMMSS digits, see the T() docstring above: T(30) - T(0) == 30s,
# T(140) - T(0) == 100s).
def test_second_live_controller_refusal():
    fake = FakeB2(bucket=BUCKET)
    wf_id = "20260713T000000-toy-a1b2"
    # Seed a controller_heartbeat from actor A directly via `wc.emit(...,
    # ts=T(0))` rather than `claim_controller(now=T(0))`: `claim_controller`'s
    # `now=` kwarg feeds ONLY the staleness comparison against whatever is
    # already recorded -- the event it appends is stamped with the real wall
    # clock (`wm.make_event` defaults `ts` to `now_ts()`, and `claim_controller`
    # never forwards its `now` through to `emit`). Seeding via a direct `emit`
    # call is the only way to pin the recorded heartbeat's `ts` for this test.
    wc.emit(wf_id, "controller_started", "cli:hostA", runner=fake, bucket=BUCKET,
            ts=T(0))

    # actor B, heartbeat still fresh (30s old) -- refused without --takeover
    with pytest.raises(wc.WorkflowCtlError):
        wc.claim_controller(wf_id, "cli:hostB", runner=fake, bucket=BUCKET,
                             now=T(30), takeover=False)

    # now advance past the staleness window (100s > 90s): a plain (non
    # --takeover) claim from a DIFFERENT actor is no longer refused
    ev_b = wc.claim_controller(wf_id, "cli:hostB", runner=fake, bucket=BUCKET,
                                now=T(140), takeover=False)
    assert ev_b["event"] == "controller_started"
    assert ev_b["actor"] == "cli:hostB"


def test_stale_heartbeat_takeover_immediate_when_already_stale():
    fake = FakeB2(bucket=BUCKET)
    wf_id = "20260713T000000-toy-c3d4"
    # see test_second_live_controller_refusal for why this seeds via a direct
    # `emit(..., ts=T(0))` rather than `claim_controller(now=T(0))`
    wc.emit(wf_id, "controller_started", "cli:hostA", runner=fake, bucket=BUCKET,
            ts=T(0))

    def _never_sleep(_s):
        raise AssertionError("an already-stale takeover must not wait at all")

    # heartbeat already stale (100s > 90s): takeover is immediate (zero
    # sleeps) and emits `takeover` (not `controller_started`)
    ev = wc.claim_controller(wf_id, "cli:hostB", runner=fake, bucket=BUCKET,
                              now=T(140), takeover=True, sleep_fn=_never_sleep)
    assert ev["event"] == "takeover"
    assert ev["actor"] == "cli:hostB"
    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert v["controller"]["actor"] == "cli:hostB"


def test_takeover_waits_out_fresh_heartbeat_of_dead_controller():
    """The 2026-07-15 dead zone: 'stop old controller, immediately resume
    --takeover' used to ALWAYS refuse (the dead controller's last heartbeat
    was <90s old), leaving the workflow controller-less. Takeover now WAITS
    for staleness: a fresh-but-never-advancing heartbeat (the incumbent is
    dead) goes stale within the wait budget and the takeover succeeds."""
    fake = FakeB2(bucket=BUCKET)
    wf_id = "20260713T000000-toy-d4e5"
    wc.emit(wf_id, "controller_started", "cli:hostA", runner=fake, bucket=BUCKET,
            ts=T(0))
    wc.emit(wf_id, "controller_heartbeat", "cli:hostA", runner=fake, bucket=BUCKET,
            ts=T(0))

    slept = []
    # T() encodes literal HHMMSS: T(105) = 1m05s = 65s after T(0) (heartbeat
    # still fresh), T(205) = 2m05s = 125s (past the 90s staleness window).
    polls = iter([T(105), T(205)])

    ev = wc.claim_controller(wf_id, "cli:hostB", runner=fake, bucket=BUCKET,
                              now=T(30), takeover=True,
                              clock=lambda: next(polls),
                              sleep_fn=lambda s: slept.append(s))
    assert ev["event"] == "takeover" and ev["actor"] == "cli:hostB"
    # waited (never refused): one sleep per poll, at the module's poll cadence
    assert slept == [wc.POLL_INTERVAL_S, wc.POLL_INTERVAL_S]


def test_takeover_refuses_when_incumbent_heartbeats_during_wait():
    """A GENUINELY live incumbent (its heartbeat ADVANCES during the takeover
    wait) is never stomped — the wait refuses with the not-yet-stale error.
    The single-sleep assertion pins the FAST refusal branch (heartbeat
    advance observed on the first poll): budget exhaustion raises the same
    message only after max_polls sleeps, so without it this test could not
    tell the two branches apart."""
    fake = FakeB2(bucket=BUCKET)
    wf_id = "20260713T000000-toy-e5f6"
    wc.emit(wf_id, "controller_started", "cli:hostA", runner=fake, bucket=BUCKET,
            ts=T(0))
    wc.emit(wf_id, "controller_heartbeat", "cli:hostA", runner=fake, bucket=BUCKET,
            ts=T(0))

    slept = []

    def live_incumbent_sleep(s):
        # the incumbent heartbeats again while we wait
        slept.append(s)
        wc.emit(wf_id, "controller_heartbeat", "cli:hostA", runner=fake,
                bucket=BUCKET, ts=T(45))

    with pytest.raises(wc.WorkflowCtlError, match="not yet stale"):
        wc.claim_controller(wf_id, "cli:hostB", runner=fake, bucket=BUCKET,
                             now=T(30), takeover=True,
                             clock=lambda: T(100),
                             sleep_fn=live_incumbent_sleep)
    # early-exit on the FIRST poll that observes the advanced heartbeat —
    # never the max_polls fallthrough.
    assert slept == [wc.POLL_INTERVAL_S]


def test_takeover_same_actor_fast_path_zero_sleeps():
    """A takeover where the recorded incumbent IS the claiming actor (the
    operator stop -> `resume --takeover` cycle) skips the staleness wait
    entirely: the non-takeover claim path already readmits the same actor
    instantly, so waiting out one's OWN fresh heartbeat was a pure
    60-150s stall."""
    fake = FakeB2(bucket=BUCKET)
    wf_id = "20260713T000000-toy-a7b8"
    wc.emit(wf_id, "controller_started", "cli:hostA", runner=fake, bucket=BUCKET,
            ts=T(0))
    wc.emit(wf_id, "controller_heartbeat", "cli:hostA", runner=fake, bucket=BUCKET,
            ts=T(0))

    def _never_sleep(_s):
        raise AssertionError("a same-actor takeover must not wait at all")

    # heartbeat still FRESH (30s < 90s window) — a different actor would wait
    ev = wc.claim_controller(wf_id, "cli:hostA", runner=fake, bucket=BUCKET,
                              now=T(30), takeover=True, sleep_fn=_never_sleep)
    assert ev["event"] == "takeover" and ev["actor"] == "cli:hostA"
    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert v["controller"]["actor"] == "cli:hostA"


def test_takeover_wait_is_bounded_even_with_frozen_clock():
    """The wait loop is bounded by poll COUNT as well as elapsed time: a
    clock that never advances (heartbeat neither re-fires nor goes stale)
    still terminates in a refusal instead of spinning forever."""
    fake = FakeB2(bucket=BUCKET)
    wf_id = "20260713T000000-toy-f6a7"
    wc.emit(wf_id, "controller_heartbeat", "cli:hostA", runner=fake, bucket=BUCKET,
            ts=T(0))

    slept = []
    with pytest.raises(wc.WorkflowCtlError, match="not yet stale"):
        wc.claim_controller(wf_id, "cli:hostB", runner=fake, bucket=BUCKET,
                             now=T(30), takeover=True,
                             clock=lambda: T(30),
                             sleep_fn=lambda s: slept.append(s))
    budget = (wc.POLL_INTERVAL_S * wc.HEARTBEAT_STALE_MULT
              + wc.TAKEOVER_WAIT_GRACE_S)
    assert len(slept) == budget // wc.POLL_INTERVAL_S + 1


# --- 6. --detach with no systemd-run: print the exact foreground command ----
def test_spawn_detached_unavailable_prints_foreground_command(monkeypatch):
    monkeypatch.setattr(wc.shutil, "which", lambda name: None)

    def _must_not_run(*a, **k):
        raise AssertionError(
            "spawn_detached must never invoke subprocess when systemd-run "
            "is unavailable -- no hidden nohup fallback")
    monkeypatch.setattr(wc.subprocess, "run", _must_not_run)

    argv = [sys.executable, "/abs/herdd.py", "workflow", "run", "/abs/wf.py"]
    with pytest.raises(wc.DetachUnavailable) as ei:
        wc.spawn_detached(argv, wf_id="20260713T000000-toy-a1b2")
    assert str(ei.value) == " ".join(str(a) for a in argv)


def _fake_systemd(monkeypatch, runs):
    """systemd-run present + succeeding, every invocation recorded into
    `runs` (the `--user --version` probe first, then the unit launch) —
    nothing really spawns."""
    monkeypatch.setattr(
        wc.shutil, "which",
        lambda name: "/usr/bin/systemd-run" if name == "systemd-run" else None)

    def _run(cmd, capture_output=True, text=True):
        runs.append(list(cmd))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(wc.subprocess, "run", _run)


def test_run_workflow_detach_pins_wf_id_into_child_argv(tmp_path, monkeypatch):
    """Regression for the 2026-07-15 --detach id fork: the parent minted one
    wf_id (the printed id, the unit name, the spec.json) while the detached
    child's bare `workflow run <path>` re-planned and drove a DIFFERENT id
    (printed ...-e2-paired-3553 stayed an orphan; the controller ran
    ...-e2-paired-2c10). The detach argv must carry `--wf-id <parent id>` so
    the id `run_workflow` returns == the systemd unit name == the id the
    child (and every Restart=on-failure re-run) actually drives, off ONE
    spec write."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    monkeypatch.setattr(wc, "load_workflow_module", lambda path: wf)
    runs = []
    _fake_systemd(monkeypatch, runs)

    argv = [sys.executable, "/abs/herdd.py", "workflow", "run", "/abs/wf.py"]
    rc, result = wc.run_workflow("/abs/wf.py", actor=ACTOR, detach=True,
                                 argv=argv, runner=fake, bucket=BUCKET)
    assert rc == wc.EXIT_OK
    wf_id = result["wf_id"]

    launch = runs[-1]                       # runs[0] is the --version probe
    assert f"--unit=wfctl-{wf_id}" in launch
    child = launch[launch.index("--") + 1:]
    # `--detached-controller` marks the child as THE detached controller
    # (terminal workflow -> exit 0, no Restart=on-failure flap)
    assert child == [str(x) for x in argv] + ["--detached-controller",
                                              "--wf-id", wf_id]

    # exactly ONE spec.json exists, under that same id — and the pinned
    # child's own write_spec is a byte-identical noop, not a second plan.
    assert ([k for k in fake.store if k.endswith("/spec.json")]
            == [f"workflows/{wf_id}/spec.json"])
    assert wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)["status"] == "noop"


def test_cli_workflow_run_detach_printed_id_is_the_child_id(
        _fake_cli, tmp_path, monkeypatch, capsys):
    """Same invariant end-to-end through `herdd.cmd_workflow_run --detach`:
    the id in the `>> workflow <id> detached` line is byte-for-byte the id
    in the child argv's `--wf-id` and the systemd unit name (an operator's
    `workflow status/cancel` against the printed id must target the run that
    actually exists)."""
    fake = _fake_cli
    wf = two_stage_workflow(tmp_path)
    # the SUBJECT is `vastlib.cli.workflow.run.run`, so the module loader it
    # calls is `vastlib.workflows.ctl`'s — which is what `wc` now names.
    monkeypatch.setattr(wc, "load_workflow_module", lambda path: wf)
    runs = []
    _fake_systemd(monkeypatch, runs)

    a = argparse.Namespace(path="/abs/wf.py", wf_id=None, detach=True,
                           takeover=False, json=False)
    with pytest.raises(SystemExit) as ei:
        herdd.cmd_workflow_run(a)
    assert ei.value.code == wc.EXIT_OK

    out = capsys.readouterr().out
    assert ">> workflow " in out and " detached" in out
    printed_id = out.split(">> workflow ")[1].split(" detached")[0]

    launch = runs[-1]
    assert f"--unit=wfctl-{printed_id}" in launch
    child = launch[launch.index("--") + 1:]
    assert child[-2:] == ["--wf-id", printed_id]
    assert ([k for k in fake.store if k.endswith("/spec.json")]
            == [f"workflows/{printed_id}/spec.json"])


# --- 7. exit-code mapping through the CLI handlers ---------------------------
# Light per the packet ("the reconcile tests above are the primary
# evidence"): call the herdd `cmd_workflow_*` handlers directly (no real
# subprocess) with `_ensure_b2_remote` and the shared default runner both
# monkeypatched to the in-memory fake -- no rclone, no B2, no network.
@pytest.fixture
def _fake_cli(monkeypatch):
    fake = FakeB2(bucket=BUCKET)
    # ONE line since step 7: this file's setup helpers (`wc.emit`,
    # `wc.write_spec`) and the subject (`vastlib.cli.workflow.<verb>.run`, which
    # reads `vastlib.workflows.ctl`) are now the same module object, so a single
    # `_default_runner` patch points both at the same FakeB2. Through step 6d
    # these were two modules with two runners and this needed two lines.
    monkeypatch.setattr(wc, "_default_runner", fake)
    # `_ensure_b2_remote` is called by the cli verb through `storage.b2`, not
    # through the `herdd` re-export.
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    return fake


def test_cli_exit_code_invalid_workflow_module(_fake_cli, tmp_path):
    bad = tmp_path / "bad_workflow.py"
    bad.write_text("# does not define a module-level WORKFLOW\nx = 1\n")
    a = argparse.Namespace(path=str(bad), online=False, json=True)
    with pytest.raises(SystemExit) as ei:
        herdd.cmd_workflow_plan(a)
    assert ei.value.code == wc.EXIT_INVALID == 1


def test_cli_exit_code_status_read_nonterminal_is_ok(_fake_cli):
    """'0 success / requested nonterminal status read' (roadmap exit-code
    table, literal case): a status read of a running (non-terminal) workflow
    never itself fails."""
    fake = _fake_cli
    wf_id = wm.mint_wf_id("toy")
    wc.emit(wf_id, "controller_started", ACTOR, runner=fake, bucket=BUCKET, ts=T(0))
    a = argparse.Namespace(wf_id=wf_id, json=True)
    with pytest.raises(SystemExit) as ei:
        herdd.cmd_workflow_status(a)
    assert ei.value.code == wc.EXIT_OK == 0


def test_cli_exit_code_status_read_terminal_failed(_fake_cli):
    """Interface note (not a symbol gap -- a spec-vs-impl mismatch surfaced
    while writing this test): this subtask's instructions describe a
    terminal-FAILED status read as exit 0 ("reading a status is success").
    The already-committed `workflowctl.status_workflow`/`_terminal_exit_code`
    (wfctl-reconcile subtask, c9fe702) deliberately mirrors `run`'s exit code
    for `status` too -- failed -> EXIT_FAILED -- consistent with the
    roadmap's own "2 terminal workflow failure" table row carrying no
    `status`-specific carve-out, and with `_terminal_exit_code`'s docstring
    ("callers check `terminal` themselves before deciding whether this
    mapping even applies"). This test asserts the REAL, already-shipped
    behavior rather than force an incorrect assertion to match the packet's
    shorthand; flagged for the coordinator rather than edited here (out of
    this subtask's declared files)."""
    fake = _fake_cli
    wf_id = wm.mint_wf_id("toy")
    wc.emit(wf_id, "workflow_failed", ACTOR, runner=fake, bucket=BUCKET, ts=T(0),
            failure_class="RETRY_EXHAUSTED")
    a = argparse.Namespace(wf_id=wf_id, json=True)
    with pytest.raises(SystemExit) as ei:
        herdd.cmd_workflow_status(a)
    assert ei.value.code == wc.EXIT_FAILED == 2


def _ts_plus(ts: str, seconds: int) -> str:
    """A runmeta ts strictly `seconds` after `ts` (test helper for faking a
    heartbeat that ADVANCES past a real-wall-clock baseline)."""
    import datetime
    d = wm._parse_ts(ts) + datetime.timedelta(seconds=seconds)
    return f"{d:%Y%m%dT%H%M%S}{d.microsecond // 1000:03d}Z"


def test_cli_exit_code_genuinely_refused_controller(_fake_cli, tmp_path, monkeypatch):
    """`workflow resume --takeover` against a controller that HEARTBEATS
    AGAIN during the takeover wait (a genuinely live controller) is a
    genuine refusal -> EXIT_CREDENTIAL (5). The CLI path injects no
    clock/sleep seams, so the wait's default `time.sleep` is monkeypatched
    to (a) not really sleep and (b) simulate the live incumbent's next
    heartbeat landing."""
    fake = _fake_cli
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)
    now = wm.now_ts()
    wc.emit(wf_id, "controller_started", "cli:other-host", runner=fake,
            bucket=BUCKET, ts=now)
    wc.emit(wf_id, "controller_heartbeat", "cli:other-host", runner=fake,
            bucket=BUCKET, ts=now)

    slept = []

    def live_incumbent_sleep(s):
        slept.append(s)
        wc.emit(wf_id, "controller_heartbeat", "cli:other-host", runner=fake,
                bucket=BUCKET, ts=_ts_plus(now, 5))

    monkeypatch.setattr(wc.time, "sleep", live_incumbent_sleep)

    a = argparse.Namespace(wf_id=wf_id, takeover=True, detach=False, json=True)
    with pytest.raises(SystemExit) as ei:
        herdd.cmd_workflow_resume(a)
    assert ei.value.code == wc.EXIT_CREDENTIAL == 5
    # fast refusal branch, not budget exhaustion (see the non-CLI variant).
    assert slept == [wc.POLL_INTERVAL_S]


# --- M3-T1: real box_resolver/box_teardown builders (digest-verify,
# bootstrap, adopt, box_acquired event) ---------------------------------------
# Every transport primitive here is a fake closure -- `build_box_resolver`/
# `build_box_teardown` never touch a real vast API or B2 object beyond the
# `FakeB2` fixture's in-memory store.
def _acquired_events(fake, wf_id):
    return [json.loads(r) for r in wc.read_events(wf_id, runner=fake, bucket=BUCKET)
            if json.loads(r).get("event") == "box_acquired"]


def test_build_box_resolver_fresh_launch(tmp_path):
    """No adoption match, digest verifies clean -> picks the offer, launches,
    and emits ONE `box_acquired(adopted=False)` carrying offer/price/digest/
    bootstrap provenance."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    stage = wf.stages[0]                       # "generate"
    profile = wf.profiles[stage.profile]

    launched = []

    def launcher(offer_id, body):
        launched.append((offer_id, body))
        return True, "inst-1", None

    resolver = wc.build_box_resolver(
        wf_id=wf_id, actor=ACTOR, runner=fake, bucket=BUCKET,
        offer_picker=lambda **kw: {"id": 7, "min_bid": 0.4, "dph_total": 0.9},
        launcher=launcher,
        digest_verifier=lambda image: profile.image_digest,
        bootstrap_stager=lambda dry_run=False: "boot-sha",
        instance_finder=lambda label: [],
        jobs_composer=_fake_jobs_composer,
    )

    result = resolver(stage, wf, 0)
    assert result == "inst-1"
    assert len(launched) == 1                  # launcher called exactly once
    # defect #6 regression: the launch body is a REAL jobs box — jobd-boot
    # onstart + B2 cred env + runtype — not the old bare {image,disk,label}.
    _body = launched[0][1]
    assert "jobd-boot" in (_body.get("onstart") or "")
    assert _body.get("env", {}).get("B2_KEY_ID")
    assert _body.get("runtype")

    acquired = _acquired_events(fake, wf_id)
    assert len(acquired) == 1
    ev = acquired[0]
    assert ev["instance_id"] == "inst-1"
    assert ev["offer"] == 7
    # Bid rental prices at BID_TARGET_MULT x floor capped ONLY by the profile
    # max_bid, NEVER the bare floor — a floor bid is preempted by any competing
    # bid one grid-step higher (found live 2026-07-20: the e2-paired a2 box,
    # bid exactly min_bid, was outbid ~1h into generation) — and NEVER clamped
    # below the offer's dph_total (on a bid offer that is floor+adders, not
    # on-demand; the clamp squashed headroom to floor+1¢ and two 2026-07-30
    # score boxes were outbid mid-pull). Launch body and the durable
    # box_acquired event carry the SAME defended price.
    expect_price = round(0.4 * bidpolicy.BID_TARGET_MULT, 3)
    assert expect_price > 0.4                   # headroom above the floor
    assert launched[0][1]["price"] == expect_price
    assert ev["price"] == expect_price
    assert ev["image_digest"] == profile.image_digest
    assert ev["bootstrap_sha"] == "boot-sha"
    assert ev["adopted"] is False
    assert ev["stage"] == "generate"
    assert ev["attempt"] == 0


def test_build_box_resolver_composes_real_jobd_body(tmp_path, monkeypatch):
    """Defect #6, REAL composer path (not the fake): the DEFAULT
    `jobs_composer` (`herdd.compose_jobs_launch_env`) must turn the launch
    body into a genuine jobs box — the actual `onstart/jobd_boot.sh` prelude,
    the scoped B2 cred env (B2_KEY_ID/B2_BUCKET/B2_S3_ENDPOINT/CRED_ROLE), the
    stamped image digest, and a runtype — so a workflow-launched box runs jobd
    and CLAIMS its queued job. A bare `{image,disk,label}` body (the pre-fix
    bug) never boots the daemon and the job idles forever.

    B2_S3_ENDPOINT is set and `_ship_b2_env` is stubbed so the real composer
    runs WITHOUT minting a live B2 key; `bootstrap_stager` supplies the sha so
    no bundle is staged to B2."""
    monkeypatch.setenv("B2_S3_ENDPOINT", "https://s3.example.invalid")
    # The default `jobs_composer` is still spelled `herdd.compose_jobs_launch_env`
    # in workflowctl.py, but since step 6d that attribute IS
    # `vastlib.jobs.bundle.compose_jobs_launch_env`, whose body resolves
    # `spec._ship_b2_env`. Patching the `herdd` re-export would let the real
    # minter run and the composer would exit on missing B2 credentials.
    monkeypatch.setattr(launch_spec, "_ship_b2_env",
                        lambda *a, **k: [("B2_KEY_ID", "ro"), ("B2_APPLICATION_KEY", "sec")])
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    stage = wf.stages[0]
    profile = wf.profiles[stage.profile]

    launched = []
    resolver = wc.build_box_resolver(
        wf_id=wf_id, actor=ACTOR, runner=fake, bucket=BUCKET,
        offer_picker=lambda **kw: {"id": 7, "min_bid": 0.4, "dph_total": 0.9},
        launcher=lambda oid, body: (launched.append(body), (True, "inst-1", None))[1],
        digest_verifier=lambda image: profile.image_digest,
        bootstrap_stager=lambda dry_run=False: "boot-sha",
        instance_finder=lambda label: [],
        # jobs_composer intentionally NOT injected -> real compose_jobs_launch_env
    )

    assert resolver(stage, wf, 0) == "inst-1"
    body = launched[0]
    onstart = body.get("onstart") or ""
    assert "jobd" in onstart.lower()                    # real jobd_boot.sh prelude
    assert "@JOBD_BUNDLE_SHA@" not in onstart           # sha substituted, not literal
    env = body.get("env") or {}
    assert env.get("B2_KEY_ID") == "ro"
    assert env.get("B2_BUCKET") == BUCKET
    assert env.get("B2_S3_ENDPOINT") == "https://s3.example.invalid"
    assert env.get("CRED_ROLE") == "jobs"
    assert env.get(imageref.IMAGE_DIGEST_ENV) == profile.image_digest
    assert body.get("runtype")


def test_build_box_resolver_attaches_image_login_for_private_pull(tmp_path):
    """A private-registry image whose login provider yields creds -> the launch
    body carries `image_login` so the box can pull it (regression: a private
    of-record image without it hangs `loading` on `denied: access forbidden`,
    found live 2026-07-15). A public image (provider returns None) launches
    with NO image_login key."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    stage = wf.stages[0]
    profile = wf.profiles[stage.profile]
    login_str = "-u vast -p TOK registry.example.com"

    def _run(login_provider, cid):
        launched = []
        resolver = wc.build_box_resolver(
            wf_id=wm.mint_wf_id(wf.name), actor=ACTOR, runner=fake, bucket=BUCKET,
            offer_picker=lambda **kw: {"id": 7, "min_bid": 0.4, "dph_total": 0.9},
            launcher=lambda oid, body: (launched.append(body), (True, cid, None))[1],
            digest_verifier=lambda image: profile.image_digest,
            bootstrap_stager=lambda dry_run=False: "boot-sha",
            instance_finder=lambda label: [],
            image_login_provider=login_provider,
            jobs_composer=_fake_jobs_composer,
        )
        assert resolver(stage, wf, 0) == cid
        return launched[0]

    # private: provider yields a docker-login string -> present in body verbatim
    assert _run(lambda image: login_str, "inst-p").get("image_login") == login_str
    # public: provider returns None -> no image_login key at all
    assert "image_login" not in _run(lambda image: None, "inst-q")


def test_build_box_resolver_refuses_a_retired_registry_image(tmp_path):
    """A workflow profile on `registry.gitlab.com` must be refused BEFORE an
    offer is picked or a box is rented.

    This path does not go through `spec._require_image`, and the digest checks
    below it cannot stand in: an UNPINNED profile whose digest does not resolve
    proceeds by design (`verified_digest = None`), which is correct for "could
    not ask" and wrong for "this registry no longer exists". The difference is
    a rented box billing in `loading` on `denied: access forbidden` for the
    whole boot deadline (measured live 2026-07-15, 1h05m)."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    stage = wf.stages[0]
    profiles = {k: dataclasses.replace(
        p, image="registry.gitlab.com/example/project:train-t211-latest",
        image_digest=None) for k, p in wf.profiles.items()}
    wf = dataclasses.replace(wf, profiles=profiles)

    resolver = wc.build_box_resolver(
        wf_id=wm.mint_wf_id(wf.name), actor=ACTOR, runner=fake, bucket=BUCKET,
        offer_picker=lambda **kw: pytest.fail("picked an offer for a dead registry"),
        launcher=lambda oid, body: pytest.fail("rented a box for a dead registry"),
        digest_verifier=lambda image: None,
        bootstrap_stager=lambda dry_run=False: "boot-sha",
        instance_finder=lambda label: [],
    )
    with pytest.raises(wc.WorkflowCtlError) as e:
        resolver(stage, wf, 0)
    assert "RETIRED" in str(e.value) and imageref.R2_REGISTRY_HOST in str(e.value)


def test_build_box_resolver_adopts_existing_instance(tmp_path):
    """A matching `run:<job_id>` instance already exists (e.g. a controller
    crashed and resumed) -> ADOPT it instead of launching a duplicate:
    launcher is never called, and `box_acquired(adopted=True)` carries the
    adopted id."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    stage = wf.stages[0]
    label = "run:" + wm.stage_job_id(wf_id, stage.name, 0)

    def launcher(offer_id, body):
        raise AssertionError("launcher must not be called on adoption")

    def instance_finder(lbl):
        assert lbl == label
        return [{"id": "inst-9", "label": lbl, "actual_status": "running"}]

    resolver = wc.build_box_resolver(
        wf_id=wf_id, actor=ACTOR, runner=fake, bucket=BUCKET,
        offer_picker=lambda **kw: {"id": 7, "min_bid": 0.4, "dph_total": 0.9},
        launcher=launcher,
        digest_verifier=lambda image: "sha256:" + "a" * 64,
        bootstrap_stager=lambda dry_run=False: "boot-sha",
        instance_finder=instance_finder,
    )

    result = resolver(stage, wf, 0)
    assert result == "inst-9"

    acquired = _acquired_events(fake, wf_id)
    assert len(acquired) == 1
    assert acquired[0]["instance_id"] == "inst-9"
    assert acquired[0]["adopted"] is True


def test_build_box_resolver_digest_drift_raises(tmp_path):
    """A pinned of-record profile whose image tag has since moved -> a
    fail-closed `WorkflowCtlError`, never a silent launch onto a drifted
    env (this subtask's DIGEST VERIFY requirement)."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    stage = wf.stages[0]
    profile = wf.profiles[stage.profile]
    assert profile.image_digest is not None      # pinned in `pinned_profile`

    def launcher(offer_id, body):
        raise AssertionError("launcher must not be called on digest drift")

    resolver = wc.build_box_resolver(
        wf_id=wf_id, actor=ACTOR, runner=fake, bucket=BUCKET,
        offer_picker=lambda **kw: {"id": 7, "min_bid": 0.4, "dph_total": 0.9},
        launcher=launcher,
        digest_verifier=lambda image: "sha256:" + "b" * 64,     # != pinned digest
        bootstrap_stager=lambda dry_run=False: "boot-sha",
        instance_finder=lambda label: [],
    )

    with pytest.raises(wc.WorkflowCtlError, match="IMAGE_DRIFT"):
        resolver(stage, wf, 0)
    assert not _acquired_events(fake, wf_id)


def test_build_box_resolver_no_offer_returns_none(tmp_path):
    """No matching offer -> resolver returns None (`reconcile_tick` then
    reports `need_box`); no launch, no `box_acquired` event, no spend."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    stage = wf.stages[0]
    profile = wf.profiles[stage.profile]

    def launcher(offer_id, body):
        raise AssertionError("launcher must not be called when no offer exists")

    resolver = wc.build_box_resolver(
        wf_id=wf_id, actor=ACTOR, runner=fake, bucket=BUCKET,
        offer_picker=lambda **kw: None,
        launcher=launcher,
        digest_verifier=lambda image: profile.image_digest,
        bootstrap_stager=lambda dry_run=False: "boot-sha",
        instance_finder=lambda label: [],
    )

    result = resolver(stage, wf, 0)
    assert result is None
    assert not _acquired_events(fake, wf_id)


def test_build_box_teardown_routes_stop_and_destroy():
    """`mode='stop'` routes to `stopper`, `mode='destroy'` routes to
    `destroyer`; both return the injected `(ok, err)` tuple's `ok` bool."""
    calls = []

    def stopper(iid):
        calls.append(("stop", iid))
        return True, None

    def destroyer(iid):
        calls.append(("destroy", iid))
        return True, None

    teardown = wc.build_box_teardown(stopper=stopper, destroyer=destroyer)
    assert teardown("44", "stop") is True
    assert teardown("45", "destroy") is True
    assert calls == [("stop", "44"), ("destroy", "45")]


def test_build_box_teardown_false_on_failure_or_exception():
    """A failed `(ok=False, err)` tuple and a raised exception both collapse
    to `False`, never propagate -- `_teardown_boxes` treats this as one
    failed-to-release box, not a crashed reconcile tick."""
    def failing_stopper(iid):
        return False, "busy"

    def raising_destroyer(iid):
        raise RuntimeError("boom")

    teardown = wc.build_box_teardown(stopper=failing_stopper, destroyer=raising_destroyer)
    assert teardown("44", "stop") is False
    assert teardown("45", "destroy") is False


def test_reconcile_end_to_end_real_box_resolver_and_teardown(tmp_path):
    """Drive the SAME generate->score toy (as the M2-T3 idempotency test)
    through `reconcile_tick`, but with a REAL `build_box_resolver`/
    `build_box_teardown` (every vast transport call faked): each stage
    acquires its own box via the real acquisition path, and completion
    releases every box it acquired."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    launched = []

    def launcher(offer_id, body):
        cid = f"inst-{len(launched)}"
        launched.append(cid)
        return True, cid, None

    stopped = []

    def stopper(iid):
        stopped.append(iid)
        return True, None

    resolver = wc.build_box_resolver(
        wf_id=wf_id, actor=ACTOR, runner=fake, bucket=BUCKET,
        offer_picker=lambda **kw: {"id": 7, "min_bid": 0.4, "dph_total": 0.9},
        launcher=launcher,
        digest_verifier=lambda image: "sha256:" + "a" * 64,
        bootstrap_stager=lambda dry_run=False: "boot-sha",
        instance_finder=lambda label: [],
        jobs_composer=_fake_jobs_composer,
    )
    teardown = wc.build_box_teardown(stopper=stopper, destroyer=lambda iid: (True, None))

    def tick(n):
        return wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                                  now=T(n), box_resolver=resolver, box_teardown=teardown)

    r1 = tick(10)
    assert (r1["action"], r1["stage"]) == ("stage_submitted", "generate")
    gen_job_id = r1["job_id"]
    assert tick(20)["action"] == "noop_running"

    finish_job(fake, gen_job_id, kind="e2-generations")
    assert tick(30)["action"] == "artifact_accepted"
    assert tick(40)["action"] == "stage_succeeded"

    r5 = tick(50)
    assert (r5["action"], r5["stage"]) == ("stage_submitted", "score")
    score_job_id = r5["job_id"]
    assert tick(60)["action"] == "noop_running"

    finish_job(fake, score_job_id, kind="e2-scores")
    assert tick(70)["action"] == "artifact_accepted"
    assert tick(80)["action"] == "stage_succeeded"

    assert tick(90)["action"] == "teardown_started"     # verdict written
    r10 = tick(100)
    assert r10["action"] == "workflow_succeeded"

    # both stages acquired a DISTINCT box via the real resolver path
    assert len(launched) == 2 and len(set(launched)) == 2
    # both were torn down as part of reaching workflow_succeeded
    assert sorted(stopped) == sorted(launched)

    events = [json.loads(r) for r in wc.read_events(wf_id, runner=fake, bucket=BUCKET)]
    acquired_ids = sorted(e["instance_id"] for e in events if e.get("event") == "box_acquired")
    released_ids = sorted(e["instance_id"] for e in events if e.get("event") == "box_released")
    assert acquired_ids == sorted(launched)
    assert released_ids == sorted(launched)


# --- M3-T1 integration regression: rental vocabulary at the herdd/workflow
# seam. `pick_cheapest_offer` (herdd-native) must accept the workflow's
# FROZEN `workflow.RENTAL_CHOICES` spelling ("on-demand"), not only the
# vast-native "ondemand" — `build_box_resolver` passes `profile.rental`
# straight through, so a divergent vocabulary would send an invalid API
# `type=` and sort on the wrong price field, silently never acquiring an
# on-demand box.
@pytest.mark.parametrize("rental,exp_type,exp_price_field", [
    ("bid", "bid", "min_bid"),
    ("ondemand", "ondemand", "dph_total"),
    ("on-demand", "ondemand", "dph_total"),      # workflow.RENTAL_CHOICES spelling
])
def test_pick_cheapest_offer_normalizes_rental_vocab(
        monkeypatch, rental, exp_type, exp_price_field):
    from workflow import RENTAL_CHOICES
    # guard: the workflow layer really does use the hyphenated spelling
    assert "on-demand" in RENTAL_CHOICES and "ondemand" not in RENTAL_CHOICES

    captured = {}

    def fake_request_soft(method, path, body=None, **kw):
        captured["method"], captured["path"], captured["body"] = method, path, body
        return True, {"offers": [{"id": 1}]}, None

    monkeypatch.setattr(api, "request_soft", fake_request_soft)
    offer = offers.pick_cheapest_offer(rental=rental, max_dph=1.0)
    assert offer == {"id": 1}
    body = captured["body"]
    assert body["type"] == exp_type                        # valid vast API type
    assert body["order"] == [[exp_price_field, "asc"]]      # correct sort field
    assert exp_price_field in body                          # max_dph on right field


# --- M3-T2 budget: durable cost accrual + workflow/stage budget enforcement --
# `record_box_cost`/`folded_spend` are workflowctl-LOCAL (never touch
# `workflowmeta.EVENTS`) -- the restart-durable budget source `reconcile_tick`
# reads fresh every tick, never from any in-memory total.
def test_cost_persists_across_controller_restart(tmp_path):
    """`record_box_cost` twice with increasing cumulative values; a FRESH
    `folded_spend()` call -- simulating a restarted controller re-reading the
    SAME fake B2 store, no in-memory total carried over -- returns the MAX.
    A later, lower/older value never lowers it."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    wc.record_box_cost(wf_id, 1.5, actor=ACTOR, runner=fake, bucket=BUCKET)
    assert wc.folded_spend(wf_id, runner=fake, bucket=BUCKET) == 1.5

    wc.record_box_cost(wf_id, 3.25, actor=ACTOR, runner=fake, bucket=BUCKET)
    assert wc.folded_spend(wf_id, runner=fake, bucket=BUCKET) == 3.25

    # a stray older/lower cumulative snapshot (e.g. a straggler emit from a
    # since-replaced controller) can never lower the folded budget
    wc.record_box_cost(wf_id, 0.1, actor=ACTOR, runner=fake, bucket=BUCKET)
    assert wc.folded_spend(wf_id, runner=fake, bucket=BUCKET) == 3.25


def test_budget_exhausted_before_launch_writes_no_ticket(tmp_path):
    """Cost already at/over the workflow's `budget_usd` BEFORE any stage ever
    needs a box: `reconcile_tick` refuses to acquire one, drives the workflow
    straight to a `workflow_failed`/`BUDGET_EXHAUSTED` terminal, and never
    writes a job ticket ("budget exhausted stops new actions")."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    wc.record_box_cost(wf_id, wf.budget_usd, actor=ACTOR, runner=fake, bucket=BUCKET)

    r = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                           now=T(10), box_resolver=fixed_box_resolver)
    assert r["action"] == "workflow_failed"
    assert r["failure_class"] == "BUDGET_EXHAUSTED"
    assert not any(k.startswith("jobs/queue/") for k in fake.store)

    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert v["terminal"] is True and v["status"] == "failed"


def test_budget_exhausted_in_run_propagates_cancel(tmp_path):
    """Cost crosses the budget WHILE `generate` is already `stage_submitted`
    (in flight): the next tick propagates a cancel to the active child job
    via the SAME `jobmeta.write_cancel_marker` primitive the operator-cancel
    path (step 3) uses, under a distinct action -- then, once the box
    reports its cooperative-kill outcome, the workflow reaches its ordinary
    terminal failed (no retry budget on `two_stage_workflow`'s default
    `retry_on=()`)."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    r1 = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                            now=T(10), box_resolver=fixed_box_resolver)
    assert r1["action"] == "stage_submitted"
    gen_job_id = r1["job_id"]
    assert not jobmeta.has_cancel_marker(gen_job_id, runner=fake, bucket=BUCKET)

    wc.record_box_cost(wf_id, wf.budget_usd, actor=ACTOR, runner=fake, bucket=BUCKET)

    r2 = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                            now=T(20), box_resolver=fixed_box_resolver)
    assert r2["action"] == "stage_cancel_budget"
    assert r2["stage"] == "generate" and r2["job_id"] == gen_job_id and r2["ok"] is True
    assert jobmeta.has_cancel_marker(gen_job_id, runner=fake, bucket=BUCKET)
    # no NEW score/generate ticket was fabricated by this action
    assert [k for k in fake.store if k.startswith("jobs/queue/")] == \
        [f"jobs/queue/44/{gen_job_id}.json"]

    # box observes CANCEL, kills the job, reports the cooperative outcome
    jobmeta.emit_event(gen_job_id, "cancelled", actor="box:44", runner=fake,
                        bucket=BUCKET, instance_id="44",
                        reason="workflow budget exhausted")

    for n in (30, 40, 50):
        r = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                               now=T(n), box_resolver=fixed_box_resolver)
        if r["action"] == "noop_terminal":
            break
    final = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert final["terminal"] is True and final["status"] == "failed"
    # score never got a chance to become ready/submitted
    assert not any("score" in k for k in fake.store if k.startswith("jobs/queue/"))


def test_budget_gate_is_noop_when_under_budget(tmp_path):
    """Zero cost (no `box_cost` event ever emitted) or cost well under
    `budget_usd`: the budget gate never fires and the ordinary
    generate -> submit path is byte-identical to the no-budget-seam
    behavior the pre-M3-T2 tests already cover."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    # under budget, but non-zero -- exercises folded_spend()'s real value,
    # not just the 0.0-default no-event path
    wc.record_box_cost(wf_id, wf.budget_usd / 2, actor=ACTOR, runner=fake, bucket=BUCKET)

    r1 = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                            now=T(10), box_resolver=fixed_box_resolver)
    assert (r1["action"], r1["stage"]) == ("stage_submitted", "generate")
    gen_job_id = r1["job_id"]

    r2 = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                            now=T(20), box_resolver=fixed_box_resolver)
    assert r2["action"] == "noop_running"        # NOT stage_cancel_budget
    assert not jobmeta.has_cancel_marker(gen_job_id, runner=fake, bucket=BUCKET)

    finish_job(fake, gen_job_id, kind="e2-generations")
    assert wc.reconcile_tick(
        wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR, now=T(30),
        box_resolver=fixed_box_resolver)["action"] == "artifact_accepted"
    assert wc.reconcile_tick(
        wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR, now=T(40),
        box_resolver=fixed_box_resolver)["action"] == "stage_succeeded"


# --- M3-T2 recovery: resume stopped / replace-gone+retarget / transient-safe -
# `reconcile_active_box` reuses `build_box_resolver`'s existing ADOPT/launch
# primitives (never a second acquisition path) -- these tests drive it via
# `reconcile_tick`'s new `box_observer`/`box_starter` seam, a REAL
# `build_box_resolver` for the launch/adopt side (fake transport only), and a
# scripted `box_observer` closure for the box-state observation itself.
def _real_resolver(fake, wf_id, *, launcher, instance_finder=None):
    return wc.build_box_resolver(
        wf_id=wf_id, actor=ACTOR, runner=fake, bucket=BUCKET,
        offer_picker=lambda **kw: {"id": 7, "min_bid": 0.4, "dph_total": 0.9},
        launcher=launcher,
        digest_verifier=lambda image: "sha256:" + "a" * 64,
        bootstrap_stager=lambda dry_run=False: "boot-sha",
        instance_finder=instance_finder if instance_finder is not None else (lambda label: []),
        jobs_composer=_fake_jobs_composer,
    )


def _acquired(fake, wf_id):
    return [e for e in [json.loads(r) for r in wc.read_events(wf_id, runner=fake, bucket=BUCKET)]
            if e.get("event") == "box_acquired"]


def test_recover_stopped_box_resumes_same_job_id(tmp_path):
    """An active `generate` job whose box_observer reports 'stopped' resumes
    IN PLACE via the injected `box_starter` -- same JOB_ID/attempt, no new
    `box_acquired` attempt (a resume, never a retarget)."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    launched = []

    def launcher(offer_id, body):
        cid = f"inst-{len(launched)}"
        launched.append(cid)
        return True, cid, None

    resolver = _real_resolver(fake, wf_id, launcher=launcher)
    r1 = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                            now=T(10), box_resolver=resolver)
    assert r1["action"] == "stage_submitted"
    gen_job_id, gen_attempt = r1["job_id"], r1["attempt"]
    iid = r1["box"]

    started = []

    def box_starter(box_iid):
        started.append(box_iid)
        return True, None

    r2 = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                            now=T(20), box_resolver=resolver,
                            box_observer=lambda box_iid: "stopped",
                            box_starter=box_starter)
    assert r2["action"] == "box_resumed"
    assert r2["instance_id"] == iid
    assert started == [iid]                     # box_starter called exactly once

    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    gsv = v["stages"]["generate"]
    assert gsv["job_id"] == gen_job_id and gsv["attempt"] == gen_attempt
    assert len(_acquired(fake, wf_id)) == 1      # no new box_acquired attempt


def test_recover_gone_box_retargets_same_job_id(tmp_path):
    """An active `generate` job whose box_observer reports 'gone' RETARGETS:
    the REAL `build_box_resolver` launches a fresh instance under the SAME
    deterministic label/JOB_ID/attempt -- never a new attempt or a second
    job_id."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    launched = []

    def launcher(offer_id, body):
        cid = f"inst-{len(launched)}"
        launched.append(cid)
        return True, cid, None

    resolver = _real_resolver(fake, wf_id, launcher=launcher)
    r1 = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                            now=T(10), box_resolver=resolver)
    gen_job_id, gen_attempt = r1["job_id"], r1["attempt"]
    assert launched == ["inst-0"]

    r2 = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                            now=T(20), box_resolver=resolver,
                            box_observer=lambda iid: "gone")
    assert r2["action"] == "box_retargeted"
    assert r2["job_id"] == gen_job_id == wm.stage_job_id(wf_id, "generate", 0)
    assert r2["attempt"] == 0 == gen_attempt
    assert launched == ["inst-0", "inst-1"]      # a genuinely NEW instance
    assert r2["instance_id"] == "inst-1"

    acquired = _acquired(fake, wf_id)
    assert [e["instance_id"] for e in acquired] == ["inst-0", "inst-1"]
    assert all(e["attempt"] == 0 for e in acquired)   # same attempt, both times

    # The ticket must MOVE to the new box's queue, else jobd on inst-1 idles
    # forever with an empty queue while the job stays 'submitted' on the dead
    # inst-0 (silent deadlock found live 2026-07-15).
    assert jobmeta.read_ticket("inst-0", gen_job_id, runner=fake, bucket=BUCKET) is None
    moved = jobmeta.read_ticket("inst-1", gen_job_id, runner=fake, bucket=BUCKET)
    assert moved is not None and moved["box"] == "inst-1"
    assert moved["retargeted_from"] == "inst-0"


def test_attach_failure_is_noop_not_terminal(tmp_path):
    """A box_observer that reports 'unknown' (a transient API/rclone/attach
    read failure) NEVER fabricates a terminal and NEVER relaunches -- mirrors
    SUPERVISE_DESIGN's 'transient != eviction' invariant."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    launched = []

    def launcher(offer_id, body):
        cid = f"inst-{len(launched)}"
        launched.append(cid)
        return True, cid, None

    resolver = _real_resolver(fake, wf_id, launcher=launcher)
    r1 = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                            now=T(10), box_resolver=resolver)
    assert r1["action"] == "stage_submitted"

    seen = []

    def box_observer(iid):
        seen.append(iid)
        return "unknown"

    r2 = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                            now=T(20), box_resolver=resolver, box_observer=box_observer)
    assert r2["action"] == "noop_running"
    assert seen == [launched[0]]

    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert v["terminal"] is False
    assert len(_acquired(fake, wf_id)) == 1      # no relaunch


def test_controller_death_during_retarget_adopts_not_duplicates(tmp_path):
    """A prior tick's `box_resolver` call for the replacement box already
    succeeded (a `box_acquired` for it exists), but the controller died
    before doing anything else. The NEXT tick's `reconcile_active_box` ->
    `build_box_resolver` ADOPTS that surviving box (its existing adopt path)
    instead of launching a duplicate."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    launched = []

    def launcher(offer_id, body):
        cid = f"inst-{len(launched)}"
        launched.append(cid)
        return True, cid, None

    # instance_finder starts blind (nothing to adopt); a replacement is
    # "discovered" once we register it below -- exactly what a live vast API
    # label lookup would surface after a controller crash mid-retarget.
    registry = {}

    def instance_finder(label):
        return [registry[label]] if label in registry else []

    resolver = _real_resolver(fake, wf_id, launcher=launcher, instance_finder=instance_finder)
    r1 = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                            now=T(10), box_resolver=resolver)
    gen_job_id = r1["job_id"]
    assert launched == ["inst-0"]

    label = "run:" + gen_job_id
    registry[label] = {"id": "inst-1", "actual_status": "running"}

    r2 = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                            now=T(20), box_resolver=resolver,
                            box_observer=lambda iid: "gone")
    assert r2["action"] == "box_retargeted"
    assert r2["instance_id"] == "inst-1"
    assert launched == ["inst-0"]                # launcher NOT called -- adopted

    acquired = _acquired(fake, wf_id)
    adopted = [e for e in acquired if e.get("adopted") is True]
    assert len(adopted) == 1 and adopted[0]["instance_id"] == "inst-1"

    # exactly one net box for this attempt
    owned = wc._owned_instance_ids(wc.view(wf_id, runner=fake, bucket=BUCKET))
    assert owned == ["inst-1"]


def test_checkpoint_silence_alarm_is_advisory(tmp_path, monkeypatch):
    """A folded job view that's gone stale past `mult * checkpoint_s` yields
    a `ckpt_alarm` on the returned action, but the action itself is still
    `noop_running` -- no stop, no cancel, workflow stays non-terminal
    (`_ckpt_watchdog_alarm` is ADVISORY ONLY, reused verbatim)."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    r1 = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                            now=T(10), box_resolver=fixed_box_resolver)
    assert r1["action"] == "stage_submitted"
    gen_job_id = r1["job_id"]
    wc.emit(wf_id, "box_acquired", ACTOR, runner=fake, bucket=BUCKET, ts=T(11),
            stage="generate", attempt=0, instance_id="44", adopted=False)

    stale_view = {
        "job_id": gen_job_id, "status": "started", "display_status": "running",
        "checkpoint_s": 60, "last_checkpoint_ts": None, "last_resumed_ts": None,
        "started_at": "20200101T000000000Z", "n_checkpoints": 0,
    }
    monkeypatch.setattr(jobmeta, "read_job", lambda *a, **kw: dict(stale_view))

    r2 = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                            now=T(20), box_resolver=fixed_box_resolver,
                            box_observer=lambda iid: "unknown")
    assert r2["action"] == "noop_running"
    assert isinstance(r2.get("ckpt_alarm"), str) and r2["ckpt_alarm"]

    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert v["terminal"] is False


# --- M3-T2 creds: credential-horizon gate before launch + rotate-on-resume ---
# `cred_provider` is a fake double: `current_expiry(name) -> epoch`
# (or raises/returns non-numeric to simulate a transient/auth read failure)
# and `rotate(name) -> new_expiry_epoch` (recording every call). Never a real
# `b2_mint_key` network call -- these tests only exercise the injected seam.
class FakeCredProvider:
    def __init__(self, expiry_epoch=None, fail_expiry=False,
                 rotate_to=None, fail_rotate=False):
        self.expiry_epoch = expiry_epoch
        self.fail_expiry = fail_expiry
        self.rotate_to = rotate_to
        self.fail_rotate = fail_rotate
        self.expiry_calls = []
        self.rotate_calls = []

    def current_expiry(self, name):
        self.expiry_calls.append(name)
        if self.fail_expiry:
            raise RuntimeError("401 (simulated B2 auth failure)")
        return self.expiry_epoch

    def rotate(self, name):
        self.rotate_calls.append(name)
        if self.fail_rotate:
            raise RuntimeError("403 (simulated B2 auth refusal)")
        return self.rotate_to


def test_credential_horizon_refuses_launch_before_spend(tmp_path):
    """A `cred_provider` reporting an expiry earlier than the stage's
    profile's `max_wall_s` (the full remaining wall bound at launch time,
    nothing spent yet) refuses to acquire a box -- `workflow_failed`/
    `CREDENTIAL_EXPIRES`, no job ticket, fail-closed."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    now = T(10)
    now_epoch = wm._parse_ts(now).timestamp()
    gen_max_wall_s = wf.profiles["generate"].max_wall_s
    assert gen_max_wall_s > 100          # sanity: pinned_profile default is 6h
    cred = FakeCredProvider(expiry_epoch=now_epoch + 100)   # expires way short

    r = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                           now=now, box_resolver=fixed_box_resolver,
                           cred_provider=cred)
    assert r["action"] == "workflow_failed"
    assert r["failure_class"] == "CREDENTIAL_EXPIRES"
    assert not any(k.startswith("jobs/queue/") for k in fake.store)

    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert v["terminal"] is True and v["status"] == "failed"


def test_credential_horizon_ok_allows_launch(tmp_path):
    """An expiry that comfortably outlasts the stage's remaining wall bound
    lets the ordinary generate->submit path proceed byte-identically to the
    no-`cred_provider` behavior."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    now = T(10)
    now_epoch = wm._parse_ts(now).timestamp()
    cred = FakeCredProvider(expiry_epoch=now_epoch + 10 * 24 * 3600)   # 10d out

    r = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                           now=now, box_resolver=fixed_box_resolver,
                           cred_provider=cred)
    assert r["action"] == "stage_submitted" and r["stage"] == "generate"
    assert cred.expiry_calls == ["generate"]

    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert v["terminal"] is False


def test_rotate_on_resume_mints_fresh_credential(tmp_path):
    """Driving the `box_observer='stopped'` resume path (M3-T1's recovery
    seam) with a `cred_provider` injected rotates the credential EXACTLY
    ONCE, before `box_starter` is called, reusing `b2_mint_key.mint`-shaped
    production wiring in spirit (never forked here) -- and the run continues
    (`box_resumed`, no `CREDENTIAL_EXPIRES` anywhere)."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    launched = []

    def launcher(offer_id, body):
        cid = f"inst-{len(launched)}"
        launched.append(cid)
        return True, cid, None

    resolver = _real_resolver(fake, wf_id, launcher=launcher)
    r1 = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                            now=T(10), box_resolver=resolver)
    assert r1["action"] == "stage_submitted"
    iid = r1["box"]

    started = []

    def box_starter(box_iid):
        started.append(box_iid)
        return True, None

    cred = FakeCredProvider(rotate_to=wm._parse_ts(T(20)).timestamp() + 999999)
    r2 = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                            now=T(20), box_resolver=resolver,
                            box_observer=lambda box_iid: "stopped",
                            box_starter=box_starter, cred_provider=cred)
    assert r2["action"] == "box_resumed"
    assert cred.rotate_calls == [iid]            # rotated exactly once
    assert started == [iid]                       # AFTER rotation succeeded

    names = _event_names(fake, wf_id)
    assert "workflow_failed" not in names
    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert v["terminal"] is False


def test_b2_auth_failure_is_transient_not_terminal(tmp_path):
    """`cred_provider.current_expiry` raising (a transient 401/403-shaped
    read failure indistinguishable from a network hiccup at this layer)
    NEVER fabricates a `workflow_failed` terminal from `reconcile_tick` --
    it's a noop, same "transient != eviction" posture as `box_observer`
    reporting 'unknown'. A genuine, non-transient auth REFUSAL (rotation
    itself failing, e.g. on a resume) is the distinct case that must surface
    operator-facing: it raises `WorkflowCtlError`, which the ALREADY-SHIPPED
    CLI wrapper (`herdd.cmd_workflow_run`/`cmd_workflow_resume`, see
    `test_cli_exit_code_genuinely_refused_controller`) maps to
    `EXIT_CREDENTIAL == 5` for ANY `WorkflowCtlError` raised out of
    `run_controller`/`resume_workflow` -- no new CLI wiring needed for this
    subtask's seam to reach that surface."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    cred = FakeCredProvider(fail_expiry=True)
    r = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                           now=T(10), box_resolver=fixed_box_resolver,
                           cred_provider=cred)
    assert r["action"] != "workflow_failed"
    assert cred.expiry_calls == ["generate"]
    assert not any(k.startswith("jobs/queue/") for k in fake.store)

    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert v["terminal"] is False
    names = _event_names(fake, wf_id)
    assert "workflow_failed" not in names

    # separately: a genuine (non-transient) auth REFUSAL on rotate() is a
    # hard stop -> WorkflowCtlError -> the existing EXIT_CREDENTIAL surface.
    refusing = FakeCredProvider(fail_rotate=True)
    with pytest.raises(wc.WorkflowCtlError):
        wc._rotate_credential(refusing, "some-instance-or-wf-id")


# --- M3-T2 teardown: reconcile-until-done + bounded TEARDOWN_FAILED ----------
# `owned_boxes_remaining`/`_attempt_teardown` are the shared seam driving BOTH
# `reconcile_tick` step 2 (already-terminal workflow) and `_reconcile_
# completion` step 8 (just became ready to complete) -- these tests exercise
# both entry points with fake `box_teardown` closures, never real vast
# transport, mirroring `build_box_teardown`'s own soft `(iid, mode) -> bool`
# contract.
def test_teardown_retries_until_all_released(tmp_path):
    """A terminal workflow owning two boxes, whose `box_teardown` fails the
    FIRST call for each box then succeeds the second: successive
    `reconcile_tick` calls report `teardown_retrying` then finally
    `teardown_reconciled` -- never a premature `workflow_failed`/
    `TEARDOWN_FAILED`, and never more than `TEARDOWN_MAX_ATTEMPTS` attempts
    needed."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.emit(wf_id, "box_acquired", ACTOR, runner=fake, bucket=BUCKET, ts=T(0),
            stage="generate", attempt=0, instance_id="b1", adopted=False)
    wc.emit(wf_id, "box_acquired", ACTOR, runner=fake, bucket=BUCKET, ts=T(1),
            stage="score", attempt=0, instance_id="b2", adopted=False)
    wc.emit(wf_id, "workflow_succeeded", ACTOR, runner=fake, bucket=BUCKET, ts=T(2))

    calls = {}

    def flaky_teardown(iid, mode):
        calls[iid] = calls.get(iid, 0) + 1
        return calls[iid] >= 2          # fails once, then succeeds

    actions = []
    for n in range(10, 10 + 10 * wc.TEARDOWN_MAX_ATTEMPTS, 10):
        r = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                               now=T(n), box_teardown=flaky_teardown)
        actions.append(r["action"])
        if r["action"] in ("teardown_reconciled", "noop_terminal"):
            break

    assert "teardown_retrying" in actions
    assert actions[-1] == "teardown_reconciled"
    names = _event_names(fake, wf_id)
    assert "workflow_failed" not in names
    assert calls == {"b1": 2, "b2": 2}

    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert wc.owned_boxes_remaining(v, wf_id, runner=fake, bucket=BUCKET) == []

    # a further tick is a clean teardown-only noop -- never re-attempts
    r_final = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                                 now=T(999), box_teardown=flaky_teardown)
    assert r_final == {"action": "noop_terminal", "status": "succeeded"}
    assert calls == {"b1": 2, "b2": 2}          # unchanged -- no re-stop


def test_teardown_failed_after_bound_retains_verdict(tmp_path):
    """`box_teardown` that ALWAYS fails: driving through completion (step 7
    writes `verdict.json` + `teardown_started`), then `TEARDOWN_MAX_ATTEMPTS`+1
    more ticks (step 8 retrying), eventually emits `workflow_failed`/
    `TEARDOWN_FAILED` listing the still-owned boxes -- and `verdict.json` on
    the fake store is BYTE-IDENTICAL to what step 7 originally wrote (the
    scientific verdict is retained, never overwritten or deleted)."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    launched = []

    def launcher(offer_id, body):
        cid = f"inst-{len(launched)}"
        launched.append(cid)
        return True, cid, None

    resolver = _real_resolver(fake, wf_id, launcher=launcher)
    always_fails = wc.build_box_teardown(
        stopper=lambda iid: (False, "busy"), destroyer=lambda iid: (False, "busy"))

    def tick(n, **kw):
        return wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                                  now=T(n), box_resolver=resolver, **kw)

    r1 = tick(10)
    gen_job_id = r1["job_id"]
    assert tick(20)["action"] == "noop_running"
    finish_job(fake, gen_job_id, kind="e2-generations")
    assert tick(30)["action"] == "artifact_accepted"
    assert tick(40)["action"] == "stage_succeeded"

    r5 = tick(50)
    score_job_id = r5["job_id"]
    assert tick(60)["action"] == "noop_running"
    finish_job(fake, score_job_id, kind="e2-scores")
    assert tick(70)["action"] == "artifact_accepted"
    assert tick(80)["action"] == "stage_succeeded"

    r_verdict = tick(90, box_teardown=always_fails)
    assert r_verdict["action"] == "teardown_started"
    verdict_key = f"workflows/{wf_id}/verdict.json"
    verdict_before = fake.store[verdict_key]
    assert verdict_before

    final = None
    n = 100
    for _ in range(wc.TEARDOWN_MAX_ATTEMPTS + 1):
        final = tick(n, box_teardown=always_fails)
        n += 10
        if final["action"] == "workflow_failed":
            break

    assert final["action"] == "workflow_failed"
    assert final["failure_class"] == "TEARDOWN_FAILED"
    assert sorted(final["boxes"]) == sorted(launched)

    # the scientific verdict is untouched -- same bytes, never overwritten.
    assert fake.store[verdict_key] == verdict_before

    names = _event_names(fake, wf_id)
    assert names.count("teardown_started") == 1
    assert names.count("workflow_succeeded") == 0
    assert sum(1 for e in [json.loads(r) for r in
                            wc.read_events(wf_id, runner=fake, bucket=BUCKET)]
               if e.get("event") == "workflow_failed"
               and e.get("failure_class") == "TEARDOWN_FAILED") == 1


def test_self_park_race_does_not_double_teardown(tmp_path):
    """A box that self-parks (jobd emits its OWN `box_released`, racing this
    controller's own reconcile loop) the instant `generate`'s job finishes
    must not block or duplicate artifact acceptance/`stage_succeeded`
    (`_accept_stage_artifacts` never looks at box state), must be excluded
    from `owned_boxes_remaining` immediately, and must NEVER be passed to
    `box_teardown` again at completion -- only `score`'s own box is."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    launched = []

    def launcher(offer_id, body):
        cid = f"inst-{len(launched)}"
        launched.append(cid)
        return True, cid, None

    resolver = _real_resolver(fake, wf_id, launcher=launcher)

    def tick(n, **kw):
        return wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                                  now=T(n), box_resolver=resolver, **kw)

    r1 = tick(10)
    assert r1["action"] == "stage_submitted"
    gen_job_id, gen_iid = r1["job_id"], r1["box"]

    finish_job(fake, gen_job_id, kind="e2-generations")
    # jobd self-parks the generate box the instant it finishes -- races with
    # this controller's own artifact-acceptance tick
    wc.emit(wf_id, "box_released", f"jobd:{gen_iid}", runner=fake, bucket=BUCKET,
            ts=T(11), instance_id=gen_iid, mode="stop")

    r2 = tick(20)
    assert r2["action"] == "artifact_accepted" and r2["artifact"] == "generations"
    r3 = tick(30)
    assert r3["action"] == "stage_succeeded" and r3["stage"] == "generate"

    r5 = tick(50)
    assert r5["action"] == "stage_submitted" and r5["stage"] == "score"
    score_job_id, score_iid = r5["job_id"], r5["box"]
    assert score_iid != gen_iid

    finish_job(fake, score_job_id, kind="e2-scores")
    assert tick(60)["action"] == "artifact_accepted"
    assert tick(70)["action"] == "stage_succeeded"

    names = _event_names(fake, wf_id)
    assert names.count("artifact_accepted") == 2       # exactly once each
    assert names.count("stage_succeeded") == 2

    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert wc._owned_instance_ids(v) == sorted([gen_iid, score_iid])
    assert wc.owned_boxes_remaining(v, wf_id, runner=fake, bucket=BUCKET) == [score_iid]

    stopped = []

    def stopper(iid):
        stopped.append(iid)
        return True, None

    teardown = wc.build_box_teardown(stopper=stopper, destroyer=lambda x: (True, None))
    assert tick(80, box_teardown=teardown)["action"] == "teardown_started"
    r9 = tick(90, box_teardown=teardown)
    assert r9["action"] == "workflow_succeeded"

    # the self-parked box was NEVER re-stopped -- only score's own box was
    assert stopped == [score_iid]
    released_ids = sorted(
        e["instance_id"] for e in [json.loads(r) for r in
                                    wc.read_events(wf_id, runner=fake, bucket=BUCKET)]
        if e.get("event") == "box_released")
    assert released_ids == sorted([gen_iid, score_iid])   # jobd's + the reconciled one


def test_already_released_box_is_idempotent(tmp_path):
    """An owned box with an existing `box_released` event is never passed to
    `box_teardown` again on any subsequent terminal tick."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.emit(wf_id, "box_acquired", ACTOR, runner=fake, bucket=BUCKET, ts=T(0),
            stage="generate", attempt=0, instance_id="b1", adopted=False)
    wc.emit(wf_id, "box_released", ACTOR, runner=fake, bucket=BUCKET, ts=T(1),
            instance_id="b1", mode="stop")
    wc.emit(wf_id, "workflow_succeeded", ACTOR, runner=fake, bucket=BUCKET, ts=T(2))

    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert wc._owned_instance_ids(v) == ["b1"]
    assert wc.owned_boxes_remaining(v, wf_id, runner=fake, bucket=BUCKET) == []

    calls = []

    def box_teardown(iid, mode):
        calls.append(iid)
        return True

    for n in (10, 20, 30):
        r = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                               now=T(n), box_teardown=box_teardown)
        assert r == {"action": "noop_terminal", "status": "succeeded"}
    assert calls == []          # never re-passed to box_teardown


def test_zero_budget_workflow_is_uncapped_not_instantly_exhausted(tmp_path):
    """A workflow whose `budget_usd` is 0 -- the `workflowmeta` deserialization
    default for a spec that OMITS the field -- means "no workflow-level cap",
    exactly like `ResourceProfile.budget_usd == 0` and herdd's
    `budget_usd=None`. It must NOT fire the `BUDGET_EXHAUSTED` pre-launch gate
    on its first tick before any spend; the ordinary generate->submit path
    proceeds. Regression guard: an unguarded `spent >= wf.budget_usd` failed
    EVERY unbudgeted workflow instantly, even in production where cost accrual
    is not yet wired and `folded_spend` is always 0.0."""
    import dataclasses
    fake = FakeB2(bucket=BUCKET)
    wf = dataclasses.replace(two_stage_workflow(tmp_path), budget_usd=0.0)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    # pure gate: a falsy cap is "no cap" -- never exhausted regardless of spend
    assert wc.budget_exhausted(wf, 0.0) is False
    assert wc.budget_exhausted(wf, 999.0) is False

    r = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                           now=T(10), box_resolver=fixed_box_resolver)
    assert r["action"] == "stage_submitted" and r["stage"] == "generate"
    assert "workflow_failed" not in _event_names(fake, wf_id)
    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert v["terminal"] is False


# --- M4-T3: failure classes + exit codes, status table, provenance/report,
# post-terminal notifier (roadmap "Failure classes and exit codes") ----------
def test_failure_class_exit_code_mapping():
    """Pure `wm.failure_class_exit_code` matches the frozen roadmap table
    exactly, and `wm.FAILURE_CLASSES` has exactly those 14 members -- no more,
    no fewer. An unknown string and `None` both fall back to the generic
    terminal-failure code 2."""
    expected = {
        "CONFIG_INVALID": 1, "ASSET_STALE": 1, "IMAGE_DRIFT": 1,
        "ARTIFACT_INVALID": 4, "POSTCONDITION_FAILED": 4,
        "CREDENTIAL_EXPIRES": 5,
        "WALL_EXHAUSTED": 124,
        "ENV_CANARY_FAILED": 2, "INFRASTRUCTURE_FAILED": 2,
        "ENTRYPOINT_FAILED": 2, "CHECKPOINT_STALLED": 2,
        "BUDGET_EXHAUSTED": 2, "RETRY_EXHAUSTED": 2, "TEARDOWN_FAILED": 2,
    }
    assert len(expected) == 14
    assert wm.FAILURE_CLASSES == frozenset(expected)
    for fc, code in expected.items():
        assert wm.failure_class_exit_code(fc) == code
    assert wm.failure_class_exit_code("NOT_A_REAL_FAILURE_CLASS") == 2
    assert wm.failure_class_exit_code(None) == 2


def test_fold_exposes_terminal_and_stage_failure_class():
    """A workflow ending in `workflow_failed`/`failure_class` folds with
    `view['failure_class']` set to that class, AND the causal stage's own
    `stage_failed`/`failure_class` is separately exposed on
    `view['stages'][name]['failure_class']` -- the "causal failure" the
    status table needs, distinct from the top-level terminal class."""
    fake = FakeB2(bucket=BUCKET)
    wf_id = wm.mint_wf_id("toy")
    wc.emit(wf_id, "stage_failed", ACTOR, runner=fake, bucket=BUCKET, ts=T(0),
            stage="generate", attempt=0, failure_class="ARTIFACT_INVALID")
    wc.emit(wf_id, "workflow_failed", ACTOR, runner=fake, bucket=BUCKET, ts=T(1),
            failure_class="ARTIFACT_INVALID")

    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert v["terminal"] is True
    assert v["failure_class"] == "ARTIFACT_INVALID"
    assert v["stages"]["generate"]["failure_class"] == "ARTIFACT_INVALID"

    # same assertion via the pure fold directly (no I/O), per the subtask
    raw = wc.read_events(wf_id, runner=fake, bucket=BUCKET)
    v2 = wm.fold_workflow_events([json.loads(r) for r in raw])
    assert v2["failure_class"] == "ARTIFACT_INVALID"
    assert v2["stages"]["generate"]["failure_class"] == "ARTIFACT_INVALID"


def test_status_table_rows_and_format():
    """`wm.status_table_rows`/`format_status_table`: one row per stage, the
    frozen field set (state/attempt/job/box/progress/spend/checkpoint_age/
    failure), `extras`-derived cells merged in, missing cells -> `None` in
    the row dict and `'-'` in the rendered text, and the rendered table
    contains the column headers and every stage name."""
    fake = FakeB2(bucket=BUCKET)
    wf_id = wm.mint_wf_id("toy")
    wc.emit(wf_id, "stage_submitted", ACTOR, runner=fake, bucket=BUCKET, ts=T(0),
            stage="generate", attempt=0, job_id="job-gen", instance_id="44")
    wc.emit(wf_id, "stage_failed", ACTOR, runner=fake, bucket=BUCKET, ts=T(1),
            stage="score", attempt=0, failure_class="ENTRYPOINT_FAILED")
    v = wc.view(wf_id, runner=fake, bucket=BUCKET)

    extras = {
        "spend_usd": 1.23, "budget_usd": 10.0,
        "stages": {"generate": {"progress": 0.5, "spend_usd": 1.0,
                                 "checkpoint_age_s": 30}},
    }
    rows = wm.status_table_rows(v, extras=extras)
    by_stage = {r["stage"]: r for r in rows}
    assert set(by_stage) == {"generate", "score"}

    g = by_stage["generate"]
    assert g["state"] == "stage_submitted"
    assert g["attempt"] == 0
    assert g["job"] == "job-gen"
    assert g["box"] == "44"
    assert g["progress"] == 0.5
    assert g["spend"] == 1.0
    assert g["checkpoint_age"] == 30
    assert g["failure"] is None

    s = by_stage["score"]
    assert s["state"] == "stage_failed"
    assert s["failure"] == "ENTRYPOINT_FAILED"
    # no extras entry for "score" -> every extras-derived cell is None
    assert s["progress"] is None and s["spend"] is None and s["checkpoint_age"] is None

    # extras entirely omitted -> pure-view cells still populate, extras None
    rows_no_extras = wm.status_table_rows(v)
    assert {r["stage"]: r["state"] for r in rows_no_extras} == \
        {"generate": "stage_submitted", "score": "stage_failed"}

    text = wm.format_status_table(v, extras=extras)
    for col in ("STAGE", "STATE", "ATTEMPT", "JOB", "BOX",
                "PROGRESS", "SPEND", "CKPT_AGE", "FAILURE"):
        assert col in text
    assert "generate" in text and "score" in text
    assert "-" in text          # missing cells (score's extras) rendered as '-'


def test_status_workflow_exit_code_end_to_end():
    """`wc.status_workflow` routes a terminal `workflow_failed`'s
    `failure_class` through `wm.failure_class_exit_code`: CREDENTIAL_EXPIRES
    -> 5, ARTIFACT_INVALID -> 4, WALL_EXHAUSTED -> 124, and (regression guard
    against the existing CLI test) RETRY_EXHAUSTED still -> 2."""
    fake = FakeB2(bucket=BUCKET)
    for fc, expected_rc in (
        ("CREDENTIAL_EXPIRES", 5),
        ("ARTIFACT_INVALID", 4),
        ("WALL_EXHAUSTED", 124),
        ("RETRY_EXHAUSTED", 2),
    ):
        wf_id = wm.mint_wf_id("toy")
        wc.emit(wf_id, "workflow_failed", ACTOR, runner=fake, bucket=BUCKET, ts=T(0),
                failure_class=fc)
        rc, v = wc.status_workflow(wf_id, runner=fake, bucket=BUCKET)
        assert rc == expected_rc, fc
        assert v["failure_class"] == fc


def test_provenance_and_report_written_before_success_event(tmp_path):
    """Step 7 of `_reconcile_completion` (roadmap M4-T3 "write provenance.json
    / verdict.json / report.md ATOMICALLY before the terminal success event")
    -- for an all-succeeded workflow with no owned boxes, the FIRST completion
    tick writes all three objects and emits `teardown_started`, WITHOUT yet
    emitting `workflow_succeeded`; only the NEXT tick emits it. `provenance.
    json` parses and carries workflow_id + spec_sha256 + a job_id per stage."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.emit(wf_id, "stage_succeeded", ACTOR, runner=fake, bucket=BUCKET, ts=T(10),
            stage="generate", attempt=0, job_id="job-gen")
    wc.emit(wf_id, "stage_succeeded", ACTOR, runner=fake, bucket=BUCKET, ts=T(20),
            stage="score", attempt=0, job_id="job-score")

    verdict_key = f"workflows/{wf_id}/verdict.json"
    provenance_key = f"workflows/{wf_id}/provenance.json"
    report_key = f"workflows/{wf_id}/report.md"
    assert verdict_key not in fake.store
    assert provenance_key not in fake.store
    assert report_key not in fake.store

    r1 = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR, now=T(30))
    assert r1["action"] == "teardown_started"

    # all three objects durable on THIS tick ...
    assert verdict_key in fake.store
    assert provenance_key in fake.store
    assert report_key in fake.store
    # ... strictly BEFORE any workflow_succeeded event exists.
    assert "workflow_succeeded" not in _event_names(fake, wf_id)

    provenance = json.loads(fake.store[provenance_key])
    assert provenance["workflow_id"] == wf_id
    assert isinstance(provenance.get("spec_sha256"), str) and len(provenance["spec_sha256"]) == 64
    assert provenance["stages"]["generate"]["job_id"] == "job-gen"
    assert provenance["stages"]["score"]["job_id"] == "job-score"
    assert fake.store[report_key].startswith(f"# Workflow {wf_id}")

    r2 = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR, now=T(40))
    assert r2["action"] == "workflow_succeeded"
    assert "workflow_succeeded" in _event_names(fake, wf_id)
    # provenance/report/verdict are never rewritten by the succeeded tick.
    assert fake.store[provenance_key] == json.dumps(provenance, sort_keys=True,
                                                      separators=(",", ":")) + "\n"


def test_notifier_is_best_effort_and_never_changes_verdict(tmp_path):
    """`run_controller`'s post-terminal `notifier` (M4-T3) never changes the
    returned exit code or the folded verdict/status, whether it raises or is
    omitted entirely -- and, when it doesn't raise, is invoked exactly once
    with `(wf_id, view)`."""
    wf = two_stage_workflow(tmp_path)

    def raising_notifier(wf_id, v):
        raise RuntimeError("notifier boom")

    def run_one(notifier):
        fake = FakeB2(bucket=BUCKET)
        wf_id = wm.mint_wf_id(wf.name)
        wc.emit(wf_id, "workflow_failed", ACTOR, runner=fake, bucket=BUCKET, ts=T(0),
                failure_class="RETRY_EXHAUSTED")
        rc = wc.run_controller(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                                clock=lambda: T(500), sleep_fn=lambda *_: None,
                                max_ticks=5, notifier=notifier)
        v = wc.view(wf_id, runner=fake, bucket=BUCKET)
        return rc, v

    rc_raise, v_raise = run_one(raising_notifier)
    rc_none, v_none = run_one(None)

    assert rc_raise == rc_none == wc.EXIT_FAILED == 2
    assert v_raise["status"] == v_none["status"] == "failed"
    assert v_raise["failure_class"] == v_none["failure_class"] == "RETRY_EXHAUSTED"

    calls = []

    def recording_notifier(wf_id, v):
        calls.append((wf_id, v.get("status"), v.get("failure_class")))

    rc_rec, v_rec = run_one(recording_notifier)
    assert rc_rec == wc.EXIT_FAILED == 2
    assert v_rec["status"] == "failed"
    assert len(calls) == 1
    assert calls[0][1:] == ("failed", "RETRY_EXHAUSTED")


# --- M5-T3 live-plane wiring: observer / deps factory / cost / forwarding ----
def test_build_box_observer_mapping_guards_double_spend():
    """The observer maps a fresh `(ok, list)` reader to the four recovery
    states. The double-spend footgun: a transient read (ok=False) must be
    'unknown' (NOT 'gone' -> false retarget), and a present-but-unknown-status
    box must be 'unknown' too, never 'gone'."""
    insts = [{"id": 44, "actual_status": "running"},
             {"id": 45, "actual_status": "exited"},
             {"id": 46, "actual_status": None}]

    obs_ok = wc.build_box_observer(instances_reader=lambda: (True, insts))
    assert obs_ok("44") == "live"
    assert obs_ok("45") == "stopped"
    assert obs_ok("46") == "unknown"        # present, boot/None -> don't act
    assert obs_ok("99") == "gone"           # id absent from a HEALTHY read

    # transient API failure: NOT 'gone' (transient != eviction)
    obs_fail = wc.build_box_observer(instances_reader=lambda: (False, []))
    assert obs_fail("44") == "unknown"


def test_build_live_controller_deps_keys_no_io(tmp_path, monkeypatch):
    """The factory returns all live dep keys (incl. the boot-throughput
    watchdog `throughput_observer`) and does ZERO network I/O at construction
    (so the CLI resume path can build deps BEFORE claim_controller decides to
    run). box_resolver/box_teardown are the real builders (callable closures)."""
    def boom(*a, **k):
        pytest.fail("network I/O at dep-construction time")

    # GUARD raisers over `wc.build_live_controller_deps`. They must live in the
    # namespaces the SUBJECT actually consults, or the guard is vacuous (plan
    # §7.3 H2): through step 6d that was `herdd.*`, because the flat
    # workflowctl read `herdd.pick_cheapest_offer` / `launch_instance` /
    # `image_tag_digest` by name. The ported builder reads none of those — it
    # goes through `imageref.image_ref_digest` and the vastlib market/launch/api
    # modules — so the five moved with it at step 7.
    monkeypatch.setattr(api, "request_soft", boom)
    monkeypatch.setattr(imageref, "_skopeo_digest", boom)
    monkeypatch.setattr(imageref, "image_ref_digest", boom)
    monkeypatch.setattr(imageref, "image_tag_digest", boom)
    monkeypatch.setattr(offers, "pick_cheapest_offer", boom)
    monkeypatch.setattr(vl_launch, "launch_instance", boom)

    wf = two_stage_workflow(tmp_path)
    deps = wc.build_live_controller_deps(wf, "wf-1", actor=ACTOR)
    assert set(deps) == {"box_resolver", "box_teardown", "box_observer",
                         "box_starter", "cost_observer", "cred_provider",
                         "throughput_observer", "image_state_observer"}
    assert deps["cred_provider"] is None
    assert callable(deps["box_resolver"]) and callable(deps["box_teardown"])
    assert callable(deps["box_observer"])
    assert callable(deps["throughput_observer"])
    # velvet P3: production is ARMED — an unwired image_state_observer would
    # leave the resume gate fail-open on every real controller.
    assert callable(deps["image_state_observer"])
    assert isinstance(deps["cost_observer"], wc.LiveCostObserver)
    assert deps["box_starter"] is wc._default_box_starter


def test_live_cost_observer_accrues_monotonic_and_persists(tmp_path):
    """Two ticks of `accrue_and_persist_cost` fed a real `LiveCostObserver`
    over a fake instances_reader (live box, dph set) + an injected clock:
    dt=0 on the FIRST observe (no phantom spend), then a monotonically
    increasing `box_cost` whose `folded_spend` fold takes the MAX. A
    parked/not-live box adds nothing."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    clk = {"t": 1000.0}
    insts = [{"id": "inst-0", "actual_status": "running", "dph_total": 3.6},
             {"id": "inst-1", "actual_status": "exited", "dph_total": 3.6}]
    obs = wc.build_cost_observer(instances_reader=lambda: (True, insts),
                                 clock=lambda: clk["t"], ttl_s=0.0)

    # a view owning BOTH a live and a parked box
    v = {"stages": {"generate": {"instance_id": "inst-0"},
                    "score": {"instance_id": "inst-1"}}}

    # tick 1: first observe -> dt=0 -> no phantom spend
    wc.accrue_and_persist_cost(wf, wf_id, v, actor=ACTOR, runner=fake,
                               bucket=BUCKET, cost_observer=obs)
    assert wc.folded_spend(wf_id, runner=fake, bucket=BUCKET) == 0.0

    # tick 2: 100s later; only the LIVE box accrues (3.6/3600*100 = 0.1)
    clk["t"] = 1100.0
    wc.accrue_and_persist_cost(wf, wf_id, v, actor=ACTOR, runner=fake,
                               bucket=BUCKET, cost_observer=obs)
    spend2 = wc.folded_spend(wf_id, runner=fake, bucket=BUCKET)
    assert spend2 == pytest.approx(0.1, abs=1e-6)

    # tick 3: another 100s -> cumulative 0.2, strictly monotonic
    clk["t"] = 1200.0
    wc.accrue_and_persist_cost(wf, wf_id, v, actor=ACTOR, runner=fake,
                               bucket=BUCKET, cost_observer=obs)
    spend3 = wc.folded_spend(wf_id, runner=fake, bucket=BUCKET)
    assert spend3 == pytest.approx(0.2, abs=1e-6)
    assert spend3 > spend2


def test_live_cost_observer_transient_read_keeps_prior_snapshot(tmp_path):
    """A transient read failure mid-run must not zero every box's `present`
    (which would silently drop accrual): the observer keeps its prior
    snapshot. Only a HEALTHY read that omits the box yields present=False."""
    obs = wc.build_cost_observer(
        instances_reader=lambda: (True, [{"id": "inst-0", "actual_status":
                                          "running", "dph_total": 3.6}]),
        clock=lambda: 0.0, ttl_s=0.0)
    st = obs.get("inst-0")
    assert st["present"] is True and st["actual_status"] == "running"

    obs._reader = lambda: (False, [])       # transient outage
    st2 = obs.get("inst-0")
    assert st2["present"] is True           # kept prior snapshot, not dropped


def test_run_controller_threads_recovery_seams_into_reconcile_tick(tmp_path, monkeypatch):
    """`run_controller` forwards box_observer/box_starter/cost_observer into
    every `reconcile_tick` call (making the box-loss recovery path reachable);
    a controller that passes none of them gets all-None kwargs (control-plane
    parity)."""
    wf = two_stage_workflow(tmp_path)
    seen = []

    def spy_tick(w, wid, **kw):
        seen.append({k: kw.get(k) for k in
                     ("box_observer", "box_starter", "cost_observer")})
        # drive to terminal so run_controller returns after one tick
        wc.emit(wid, "workflow_failed", ACTOR, runner=kw["runner"],
                bucket=kw["bucket"], ts=T(1), failure_class="RETRY_EXHAUSTED")
        return {"action": "noop"}

    monkeypatch.setattr(wc, "reconcile_tick", spy_tick)

    sentinel_obs, sentinel_start, sentinel_cost = object(), object(), object()
    fake = FakeB2(bucket=BUCKET)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)
    wc.run_controller(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                      clock=lambda: T(5), sleep_fn=lambda *_: None, max_ticks=1,
                      box_observer=sentinel_obs, box_starter=sentinel_start,
                      cost_observer=sentinel_cost)
    assert seen[0] == {"box_observer": sentinel_obs, "box_starter": sentinel_start,
                       "cost_observer": sentinel_cost}

    # control-plane parity: no seams passed -> all None
    seen.clear()
    fake2 = FakeB2(bucket=BUCKET)
    wf_id2 = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id2, runner=fake2, bucket=BUCKET)
    wc.run_controller(wf, wf_id2, runner=fake2, bucket=BUCKET, actor=ACTOR,
                      clock=lambda: T(5), sleep_fn=lambda *_: None, max_ticks=1)
    assert seen[0] == {"box_observer": None, "box_starter": None, "cost_observer": None}


def test_run_workflow_callable_deps_forwarded_to_controller(tmp_path, monkeypatch):
    """A `controller_deps` callable is invoked with the RESOLVED wf_id (what
    build_box_resolver's run label needs) and its six keys are forwarded into
    `run_controller`."""
    wf = two_stage_workflow(tmp_path)
    monkeypatch.setattr(wc, "load_workflow_module", lambda path: wf)
    monkeypatch.setattr(wc, "write_spec", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(wc, "view", lambda *a, **k: {"terminal": True, "status": "succeeded"})

    got = {}

    def spy_run_controller(w, wid, **kw):
        got["wf_id"] = wid
        got["kw"] = kw
        return wc.EXIT_OK

    monkeypatch.setattr(wc, "run_controller", spy_run_controller)

    seen_ids = []

    def deps_factory(w, wid):
        seen_ids.append(wid)
        return {"box_observer": "OBS", "box_starter": "START",
                "cost_observer": "COST", "box_resolver": "RES",
                "box_teardown": "TD", "cred_provider": None}

    rc, result = wc.run_workflow("ignored.py", wf_id="wf-forward-1", actor=ACTOR,
                                 controller_deps=deps_factory)
    assert rc == wc.EXIT_OK
    assert seen_ids == ["wf-forward-1"]         # callable saw the resolved id
    for k in ("box_observer", "box_starter", "cost_observer", "box_resolver",
              "box_teardown", "cred_provider"):
        assert k in got["kw"]
    assert got["kw"]["box_observer"] == "OBS"
    assert got["kw"]["cost_observer"] == "COST"


def test_resume_workflow_callable_deps_forwarded(tmp_path, monkeypatch):
    """`resume_workflow` forwards a `controller_deps` bundle the same way
    `run_workflow` does (reading the spec instead of a path)."""
    wf = two_stage_workflow(tmp_path)
    monkeypatch.setattr(wc, "read_spec", lambda *a, **k: wf)
    monkeypatch.setattr(wc, "view", lambda *a, **k: {"terminal": True, "status": "succeeded"})

    got = {}
    monkeypatch.setattr(wc, "run_controller",
                        lambda w, wid, **kw: got.update(kw) or wc.EXIT_OK)

    rc, _ = wc.resume_workflow(
        "wf-resume-1", actor=ACTOR,
        controller_deps=lambda w, wid: {
            "box_observer": "OBS", "box_starter": "START", "cost_observer": "COST",
            "box_resolver": "RES", "box_teardown": "TD", "cred_provider": None})
    assert rc == wc.EXIT_OK
    assert got["box_observer"] == "OBS" and got["cost_observer"] == "COST"


def test_reconcile_live_cost_observer_trips_in_run_budget(tmp_path):
    """FULL round-trip: a real `build_box_resolver` acquires a box (its
    box_acquired event lands the instance_id in the fold), then a real
    `LiveCostObserver` fed through `reconcile_tick`'s step-1.5 accrual drives
    `folded_spend` across the workflow budget WHILE the child job is in
    flight, firing the in-run cancel-marker gate -- no hand-written
    record_box_cost anywhere."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    resolver = wc.build_box_resolver(
        wf_id=wf_id, actor=ACTOR, runner=fake, bucket=BUCKET,
        offer_picker=lambda **kw: {"id": 7, "min_bid": 0.4, "dph_total": 0.9},
        launcher=lambda offer_id, body: (True, "inst-0", None),
        digest_verifier=lambda image: "sha256:" + "a" * 64,
        bootstrap_stager=lambda dry_run=False: "boot-sha",
        instance_finder=lambda label: [],
        jobs_composer=_fake_jobs_composer)

    clk = {"t": 1000.0}
    # dph 360/hr -> 360/3600 * 100s = $10.0 == wf.budget_usd, crosses in one tick
    live = [{"id": "inst-0", "actual_status": "running", "dph_total": 360.0}]
    obs = wc.build_cost_observer(instances_reader=lambda: (True, live),
                                 clock=lambda: clk["t"], ttl_s=0.0)

    def tick(n):
        return wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                                 now=T(n), box_resolver=resolver, cost_observer=obs)

    r1 = tick(10)                                   # acquire + submit generate
    assert r1["action"] == "stage_submitted"
    gen_job_id = r1["job_id"]

    clk["t"] = 1100.0
    r2 = tick(20)                                   # first observe: dt=0, no spend
    assert r2["action"] == "noop_running"
    assert wc.folded_spend(wf_id, runner=fake, bucket=BUCKET) == 0.0

    clk["t"] = 1200.0
    r3 = tick(30)                                   # dt=100 -> $10 -> over budget
    assert r3["action"] == "stage_cancel_budget"
    assert r3["job_id"] == gen_job_id
    assert jobmeta.has_cancel_marker(gen_job_id, runner=fake, bucket=BUCKET)
    assert wc.folded_spend(wf_id, runner=fake, bucket=BUCKET) >= wf.budget_usd


# --- fixer regressions: restart/retarget-durable cost + strand recovery ------
def test_cost_accrual_survives_controller_restart(tmp_path):
    """Finding-1 regression: a restarted controller rebuilds a FRESH
    LiveCostObserver (every in-memory spend_usd reseeded to 0). The durable
    cumulative must CONTINUE from the prior folded_spend -- never restart the
    accrual baseline at ~0, which folded-takes-MAX would then stall at the
    pre-restart peak while real billing keeps climbing (a budget-defeating
    under-count)."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    v = {"stages": {"generate": {"instance_id": "inst-0"}}}
    live = [{"id": "inst-0", "actual_status": "running", "dph_total": 3.6}]
    clk = {"t": 1000.0}

    def observer():
        return wc.build_cost_observer(instances_reader=lambda: (True, live),
                                      clock=lambda: clk["t"], ttl_s=0.0)

    def accrue(obs):
        wc.accrue_and_persist_cost(wf, wf_id, v, actor=ACTOR, runner=fake,
                                   bucket=BUCKET, cost_observer=obs)

    obs1 = observer()
    accrue(obs1)                                     # dt=0, no phantom spend
    clk["t"] = 1100.0
    accrue(obs1)                                     # +0.1
    assert wc.folded_spend(wf_id, runner=fake, bucket=BUCKET) == pytest.approx(0.1, abs=1e-6)

    # restart: FRESH observer, all in-memory spend_usd reseeded to 0
    obs2 = observer()
    clk["t"] = 1200.0
    accrue(obs2)                                     # first observe dt=0 -> no double-count
    assert wc.folded_spend(wf_id, runner=fake, bucket=BUCKET) == pytest.approx(0.1, abs=1e-6)
    clk["t"] = 1300.0
    accrue(obs2)                                     # +0.1 ONTO the durable prior
    assert wc.folded_spend(wf_id, runner=fake, bucket=BUCKET) == pytest.approx(0.2, abs=1e-6)


def test_cost_accrual_survives_retarget_box_swap(tmp_path):
    """Finding-2 regression (cost portion): a retarget swaps the owned box
    (inst-0 -> inst-1; the fold keeps only the newest instance_id per stage).
    The replaced box's accrued cost must stay in the durable cumulative and the
    replacement's fresh accrual ADDS on top -- never stalls folded_spend at the
    pre-retarget peak."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    clk = {"t": 1000.0}
    insts = [{"id": "inst-0", "actual_status": "running", "dph_total": 3.6},
             {"id": "inst-1", "actual_status": "running", "dph_total": 3.6}]
    obs = wc.build_cost_observer(instances_reader=lambda: (True, insts),
                                 clock=lambda: clk["t"], ttl_s=0.0)

    v0 = {"stages": {"generate": {"instance_id": "inst-0"}}}
    wc.accrue_and_persist_cost(wf, wf_id, v0, actor=ACTOR, runner=fake,
                               bucket=BUCKET, cost_observer=obs)       # dt=0
    clk["t"] = 1100.0
    wc.accrue_and_persist_cost(wf, wf_id, v0, actor=ACTOR, runner=fake,
                               bucket=BUCKET, cost_observer=obs)       # inst-0 +0.1
    assert wc.folded_spend(wf_id, runner=fake, bucket=BUCKET) == pytest.approx(0.1, abs=1e-6)

    v1 = {"stages": {"generate": {"instance_id": "inst-1"}}}   # inst-0 dropped
    clk["t"] = 1200.0
    wc.accrue_and_persist_cost(wf, wf_id, v1, actor=ACTOR, runner=fake,
                               bucket=BUCKET, cost_observer=obs)       # inst-1 first observe
    assert wc.folded_spend(wf_id, runner=fake, bucket=BUCKET) == pytest.approx(0.1, abs=1e-6)
    clk["t"] = 1300.0
    wc.accrue_and_persist_cost(wf, wf_id, v1, actor=ACTOR, runner=fake,
                               bucket=BUCKET, cost_observer=obs)       # inst-1 +0.1 onto prior
    assert wc.folded_spend(wf_id, runner=fake, bucket=BUCKET) == pytest.approx(0.2, abs=1e-6)


def test_box_observer_confirms_gone_defeats_eventual_consistency():
    """Finding-2 regression (false-retarget guard): a single eventually-
    consistent list omission of a still-live box must NOT map to 'gone' (which
    drives a duplicate launch + orphans the original). A second HEALTHY read
    that surfaces the box wins; only two consecutive healthy reads that BOTH
    omit it yield 'gone'; a transient failure on the confirm read is 'unknown',
    never a false 'gone'."""
    present = [{"id": "inst-0", "actual_status": "running"}]

    seq = [(True, []), (True, present)]           # omit, then surface
    obs = wc.build_box_observer(instances_reader=lambda: seq.pop(0))
    assert obs("inst-0") == "live"                # confirm read caught it

    obs_gone = wc.build_box_observer(instances_reader=lambda: (True, []))
    assert obs_gone("inst-0") == "gone"           # both healthy reads omit -> gone

    seq2 = [(True, []), (False, [])]              # omit, then transient failure
    obs_flap = wc.build_box_observer(instances_reader=lambda: seq2.pop(0))
    assert obs_flap("inst-0") == "unknown"        # never a false 'gone'


def test_stranded_box_acquired_replans_not_noop_forever(tmp_path):
    """Finding-3 regression: a controller death AFTER box_resolver emits
    box_acquired but BEFORE the job ticket strands the stage at status
    'box_acquired' (job_id=None). The next tick must RE-PLAN the same attempt
    -- ADOPTing the already-labelled box (no duplicate launch) and submitting
    the job -- instead of noop_awaiting_job_id forever while the owned box
    idle-bills."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    label = "run:" + wm.stage_job_id(wf_id, "generate", 0)
    wc.emit(wf_id, "box_acquired", ACTOR, runner=fake, bucket=BUCKET, ts=T(1),
            stage="generate", attempt=0, instance_id="inst-0", adopted=False)

    sv = wc.view(wf_id, runner=fake, bucket=BUCKET)["stages"]["generate"]
    assert sv["status"] == "box_acquired" and sv["job_id"] is None

    launched = []

    def launcher(offer_id, body):
        cid = f"new-{len(launched)}"
        launched.append(cid)
        return True, cid, None

    def instance_finder(lbl):
        return [{"id": "inst-0", "actual_status": "running"}] if lbl == label else []

    resolver = _real_resolver(fake, wf_id, launcher=launcher,
                              instance_finder=instance_finder)
    r = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                          now=T(10), box_resolver=resolver)
    assert r["action"] == "stage_submitted"
    assert r["job_id"] == wm.stage_job_id(wf_id, "generate", 0)
    assert r["attempt"] == 0
    assert launched == []                          # ADOPTed inst-0, no duplicate
    v2 = wc.view(wf_id, runner=fake, bucket=BUCKET)["stages"]["generate"]
    assert v2["job_id"] == wm.stage_job_id(wf_id, "generate", 0)


# --- Gap C: local-lock pid record + --takeover force-break -------------------
def test_acquire_local_lock_second_without_takeover_raises(tmp_path):
    """A second acquire while the flock is held (no takeover) raises, unchanged
    -- the live-controller guard. The holder's pid is recorded in the file."""
    wf_id = wm.mint_wf_id("wf")
    h1 = wc.acquire_local_lock(wf_id)
    try:
        h1.seek(0)
        assert h1.read().strip() == str(os.getpid())     # pid recorded on acquire
        with pytest.raises(wc.WorkflowCtlError):
            wc.acquire_local_lock(wf_id)                  # no takeover -> refuse
    finally:
        wc.release_local_lock(h1)


def test_acquire_local_lock_takeover_steals_dead_pid(tmp_path, monkeypatch):
    """`--takeover` force-breaks a lock whose recorded holder pid is DEAD (the
    2026-07-15 incident: --takeover cleared the remote heartbeat but died on the
    local lock). A real dead holder's flock auto-releases, so we model a
    leaked/inherited fd that is held for exactly the first probe then gone, plus
    a dead recorded pid (os.kill -> ProcessLookupError)."""
    wf_id = wm.mint_wf_id("wf")
    lock_path = os.path.join(wc._lock_dir(), f"{wf_id}.lock")
    with open(lock_path, "w") as fh:
        fh.write("999999")                               # stale holder pid
    real_flock, calls = wc.fcntl.flock, {"n": 0}

    def fake_flock(fd, op):
        calls["n"] += 1
        if calls["n"] == 1 and (op & wc.fcntl.LOCK_NB):
            raise BlockingIOError()                       # first probe: held
        return real_flock(fd, op)                         # then free

    def dead_kill(pid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr(wc.fcntl, "flock", fake_flock)
    monkeypatch.setattr(wc.os, "kill", dead_kill)
    h = wc.acquire_local_lock(wf_id, takeover=True)
    try:
        h.seek(0)
        assert h.read().strip() == str(os.getpid())      # our pid now stamped
    finally:
        wc.release_local_lock(h)


def test_acquire_local_lock_takeover_refuses_live_pid(tmp_path, monkeypatch):
    """`--takeover` must NEVER stomp a genuinely LIVE holder: a recorded pid
    that os.kill(pid, 0) confirms alive re-raises rather than steal."""
    wf_id = wm.mint_wf_id("wf")
    lock_path = os.path.join(wc._lock_dir(), f"{wf_id}.lock")
    with open(lock_path, "w") as fh:
        fh.write(str(os.getpid()))                        # a LIVE pid (ours)
    real_flock, calls = wc.fcntl.flock, {"n": 0}

    def fake_flock(fd, op):
        calls["n"] += 1
        if calls["n"] == 1 and (op & wc.fcntl.LOCK_NB):
            raise BlockingIOError()
        return real_flock(fd, op)

    monkeypatch.setattr(wc.fcntl, "flock", fake_flock)
    with pytest.raises(wc.WorkflowCtlError):
        wc.acquire_local_lock(wf_id, takeover=True)


# --- Gap D: boot/claim deadline watchdog (never-claimed box teardown) ---------
def _boot_deadline_setup(tmp_path):
    """A `generate` job stuck 'submitted' (jobd never claimed it) on inst-0,
    with a box_acquired_ts of T(10) folded into the active stage view -- the
    exact state a box that boots but never runs jobd leaves behind."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    stage = wf.stages[0]
    job_id = wm.stage_job_id(wf_id, "generate", 0)
    jobmeta.emit_event(job_id, "submitted", actor=ACTOR, runner=fake,
                       bucket=BUCKET, box="inst-0")
    active_sv = {"instance_id": "inst-0", "job_id": job_id, "attempt": 0,
                 "box_acquired_ts": T(10)}
    return fake, wf, wf_id, stage, job_id, active_sv


def test_reconcile_active_box_boot_deadline_tears_down_and_fails(tmp_path):
    """Past BOOT_DEADLINE_S with a never-claimed ('submitted') job and an
    'unknown' box_observer: tear the hung box down (destroy) and fail THIS
    attempt as infrastructure so the ordinary retry/stage_failed policy bounds
    it. (Deviation from the plan's literal retarget: routing through the retry
    machinery -- the plan's stated preferred bound -- means a fresh attempt
    resubmits a fresh ticket, so no retarget of the dead box's ticket.)"""
    fake, wf, wf_id, stage, job_id, active_sv = _boot_deadline_setup(tmp_path)
    torn = []
    now = T(2700)                                    # 1610s since T(10) > 1500
    assert wc._seconds_between(T(10), now) > wc.BOOT_DEADLINE_S
    r = wc.reconcile_active_box(
        wf, wf_id, stage, active_sv, actor=ACTOR, runner=fake, bucket=BUCKET,
        box_observer=lambda iid: "unknown",
        box_teardown=lambda iid, mode: (torn.append((iid, mode)), True)[1],
        now=now)
    assert r["action"] == "box_boot_failed"
    assert r["instance_id"] == "inst-0" and r["job_id"] == job_id
    assert torn == [("inst-0", "destroy")]           # hung box destroyed once
    # the attempt is now a terminal infrastructure failure the retry path owns
    jv = jobmeta.read_job(job_id, runner=fake, bucket=BUCKET)
    assert jv["status"] == "failed"
    assert wc._classify_job_failure(jv) == "infrastructure"


def test_reconcile_active_box_within_boot_deadline_noop(tmp_path):
    """Within the deadline: the SAME setup returns the unchanged noop_running
    and NEVER tears the box down (a box that just hasn't claimed yet is fine)."""
    fake, wf, wf_id, stage, job_id, active_sv = _boot_deadline_setup(tmp_path)
    torn = []
    r = wc.reconcile_active_box(
        wf, wf_id, stage, active_sv, actor=ACTOR, runner=fake, bucket=BUCKET,
        box_observer=lambda iid: "unknown",
        box_teardown=lambda iid, mode: (torn.append((iid, mode)), True)[1],
        now=T(1000))                                 # 590s since T(10) < 1500
    assert r["action"] == "noop_running"
    assert torn == []                                # box left alone
    assert jobmeta.read_job(job_id, runner=fake, bucket=BUCKET)["status"] == "submitted"


def test_reconcile_active_box_boot_deadline_needs_teardown_seam(tmp_path):
    """No box_teardown injected (a caller that never wired it) -> the watchdog
    stays dormant and the tick is the unchanged noop_running, even past the
    deadline. The watchdog is strictly opt-in on the teardown seam."""
    fake, wf, wf_id, stage, job_id, active_sv = _boot_deadline_setup(tmp_path)
    r = wc.reconcile_active_box(
        wf, wf_id, stage, active_sv, actor=ACTOR, runner=fake, bucket=BUCKET,
        box_observer=lambda iid: "unknown", now=T(2700))
    assert r["action"] == "noop_running"
    assert jobmeta.read_job(job_id, runner=fake, bucket=BUCKET)["status"] == "submitted"


def test_reconcile_active_box_boot_deadline_fires_on_live_observer(tmp_path):
    """PRIMARY defect-#6 signature: a booted container that idle-bills but never
    claimed its job reports actual_status 'running'/'loading' -> box_observer
    returns 'live', NOT 'unknown'. Past BOOT_DEADLINE_S with a still-'submitted'
    job the watchdog MUST tear the box down and fail the attempt here too --
    regression guard for the gap where the 'live' branch returned None
    unconditionally and idle-billed forever (only the effectively-unreachable
    'unknown' branch ran the watchdog)."""
    fake, wf, wf_id, stage, job_id, active_sv = _boot_deadline_setup(tmp_path)
    torn = []
    now = T(2700)                                    # 1610s since T(10) > 1500
    r = wc.reconcile_active_box(
        wf, wf_id, stage, active_sv, actor=ACTOR, runner=fake, bucket=BUCKET,
        box_observer=lambda iid: "live",
        box_teardown=lambda iid, mode: (torn.append((iid, mode)), True)[1],
        now=now)
    assert r["action"] == "box_boot_failed"
    assert r["instance_id"] == "inst-0" and r["job_id"] == job_id
    assert torn == [("inst-0", "destroy")]           # hung 'live' box destroyed
    jv = jobmeta.read_job(job_id, runner=fake, bucket=BUCKET)
    assert jv["status"] == "failed"
    assert wc._classify_job_failure(jv) == "infrastructure"


def test_reconcile_active_box_live_within_deadline_noop(tmp_path):
    """A 'live' box whose job just hasn't been claimed yet (within the deadline)
    is healthy: no teardown, and the tick falls through to the caller's ordinary
    noop_running handling (reconcile returns None)."""
    fake, wf, wf_id, stage, job_id, active_sv = _boot_deadline_setup(tmp_path)
    torn = []
    r = wc.reconcile_active_box(
        wf, wf_id, stage, active_sv, actor=ACTOR, runner=fake, bucket=BUCKET,
        box_observer=lambda iid: "live",
        box_teardown=lambda iid, mode: (torn.append((iid, mode)), True)[1],
        now=T(1000))                                 # 590s since T(10) < 1500
    assert r is None                                 # healthy live -> fall through
    assert torn == []
    assert jobmeta.read_job(job_id, runner=fake, bucket=BUCKET)["status"] == "submitted"


# --- boot THROUGHPUT watchdog (BOOT_HEALTHCHECK phase P0) ---------------------
def _slow_observer(iid, *, mbps=1.3, machine_id=140087):
    return {"verdict": "slow", "mbps": mbps, "window_s": 300,
            "phase": "downloading", "machine_id": machine_id}


def test_reconcile_active_box_boot_throughput_condemns(tmp_path):
    """A `throughput_observer` returning a 'slow' verdict on a never-claimed
    ('submitted') box: tear the box down (destroy) and fail THIS attempt as
    infrastructure, EARLY -- no BOOT_DEADLINE_S wait. The failed-event reason
    carries the measured MB/s + the floor so the retry policy bounds repeats."""
    fake, wf, wf_id, stage, job_id, active_sv = _boot_deadline_setup(tmp_path)
    torn = []
    r = wc.reconcile_active_box(
        wf, wf_id, stage, active_sv, actor=ACTOR, runner=fake, bucket=BUCKET,
        box_observer=lambda iid: "live",
        box_teardown=lambda iid, mode: (torn.append((iid, mode)), True)[1],
        throughput_observer=_slow_observer,
        now=T(400))                                   # 390s since T(10): WELL under deadline
    assert r["action"] == "box_boot_failed"
    assert r["instance_id"] == "inst-0" and r["job_id"] == job_id
    assert r["reason"] == "boot_throughput_floor"
    assert r["machine_id"] == 140087 and r["mbps"] == 1.3
    assert torn == [("inst-0", "destroy")]
    jv = jobmeta.read_job(job_id, runner=fake, bucket=BUCKET)
    assert jv["status"] == "failed"
    assert wc._classify_job_failure(jv) == "infrastructure"
    assert "boot throughput floor" in (jv.get("fail_reason") or "")
    assert "1.30 MB/s" in jv["fail_reason"] and "over 300s" in jv["fail_reason"]


def test_reconcile_active_box_boot_throughput_healthy_noop(tmp_path):
    """A throughput_observer that returns None (fast enough / window not full)
    NEVER condemns: the fixed-deadline path still governs, so within the deadline
    the tick is the unchanged noop (via the 'unknown' branch) / fall-through."""
    fake, wf, wf_id, stage, job_id, active_sv = _boot_deadline_setup(tmp_path)
    torn = []
    r = wc.reconcile_active_box(
        wf, wf_id, stage, active_sv, actor=ACTOR, runner=fake, bucket=BUCKET,
        box_observer=lambda iid: "unknown",
        box_teardown=lambda iid, mode: (torn.append((iid, mode)), True)[1],
        throughput_observer=lambda iid: None,
        now=T(400))                                   # under the fixed deadline
    assert r["action"] == "noop_running"
    assert torn == []
    assert jobmeta.read_job(job_id, runner=fake, bucket=BUCKET)["status"] == "submitted"


def test_reconcile_active_box_boot_throughput_needs_seams(tmp_path):
    """The throughput watchdog is strictly opt-in on BOTH the teardown seam and
    the injected observer: with a 'slow' observer but NO box_teardown it stays
    dormant (never destroys blind); with no observer at all it never fires."""
    fake, wf, wf_id, stage, job_id, active_sv = _boot_deadline_setup(tmp_path)
    # slow observer, but no teardown seam -> dormant (under the deadline -> noop)
    r = wc.reconcile_active_box(
        wf, wf_id, stage, active_sv, actor=ACTOR, runner=fake, bucket=BUCKET,
        box_observer=lambda iid: "unknown",
        throughput_observer=_slow_observer, now=T(400))
    assert r["action"] == "noop_running"
    assert jobmeta.read_job(job_id, runner=fake, bucket=BUCKET)["status"] == "submitted"


def test_reconcile_active_box_boot_throughput_gated_on_submitted(tmp_path):
    """A CLAIMED job (jobd already took it) is never condemned for a slow pull:
    the box is past the boot phase. The gate is status=='submitted', identical
    to _boot_deadline_action -- a slow observer on a claimed job is inert."""
    fake, wf, wf_id, stage, job_id, active_sv = _boot_deadline_setup(tmp_path)
    jobmeta.emit_event(job_id, "claimed", actor=ACTOR, runner=fake,
                       bucket=BUCKET, box="inst-0")
    torn = []
    r = wc.reconcile_active_box(
        wf, wf_id, stage, active_sv, actor=ACTOR, runner=fake, bucket=BUCKET,
        box_observer=lambda iid: "live",
        box_teardown=lambda iid, mode: (torn.append((iid, mode)), True)[1],
        throughput_observer=_slow_observer, now=T(400))
    assert r is None                                  # healthy claimed live -> fall through
    assert torn == []


def test_boot_throughput_fires_before_fixed_deadline(tmp_path):
    """Composition proof: with BOTH a slow throughput_observer AND a time well
    under BOOT_DEADLINE_S, the throughput action fires (early kill) -- the two
    gates compose, the throughput one just condemns sooner. It also fires when
    the deadline is ALSO exceeded (throughput evaluated first)."""
    fake, wf, wf_id, stage, job_id, active_sv = _boot_deadline_setup(tmp_path)
    assert wc._seconds_between(T(10), T(400)) < wc.BOOT_DEADLINE_S
    torn = []
    r = wc.reconcile_active_box(
        wf, wf_id, stage, active_sv, actor=ACTOR, runner=fake, bucket=BUCKET,
        box_observer=lambda iid: "live",
        box_teardown=lambda iid, mode: (torn.append((iid, mode)), True)[1],
        throughput_observer=_slow_observer, now=T(400))
    assert r["reason"] == "boot_throughput_floor"     # threw the EARLY kill, not deadline
    assert torn == [("inst-0", "destroy")]


# --- host rotation: exclude_machines wiring (integration point 3) --------------
def test_box_acquired_records_machine_id_and_excludes_on_retry(tmp_path):
    """box_resolver records the launched offer's machine_id on box_acquired, and
    a LATER attempt of the same stage passes every prior machine_id into the
    offer_picker as exclude_machines (host rotation -- pick_cheapest_offer's dead
    `exclude_machines` parameter finally has a caller)."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    stage = wf.stages[0]
    profile = wf.profiles[stage.profile]
    picker_calls = []

    def picker(**kw):
        picker_calls.append(kw)
        return {"id": 7, "min_bid": 0.4, "dph_total": 0.9, "machine_id": 555}

    resolver = wc.build_box_resolver(
        wf_id=wf_id, actor=ACTOR, runner=fake, bucket=BUCKET,
        offer_picker=picker,
        launcher=lambda oid, body: (True, "inst-A", None),
        digest_verifier=lambda image: profile.image_digest,
        bootstrap_stager=lambda dry_run=False: "boot-sha",
        instance_finder=lambda label: [],
        jobs_composer=_fake_jobs_composer)

    resolver(stage, wf, 0)                             # attempt 0 -> lands on machine 555
    assert picker_calls[0]["exclude_machines"] is None
    # the box_acquired event carries machine_id
    assert 555 in wc._prior_stage_machines(wf_id, stage.name, runner=fake, bucket=BUCKET)

    resolver(stage, wf, 1)                             # attempt 1 -> excludes 555
    assert picker_calls[1]["exclude_machines"] == [555]


# --- Gap B: ssh/hf parity prelude reaches the launch onstart ------------------
def _prelude_resolver(fake, wf_id, profile, launched, **prelude_kw):
    return wc.build_box_resolver(
        wf_id=wf_id, actor=ACTOR, runner=fake, bucket=BUCKET,
        offer_picker=lambda **kw: {"id": 7, "min_bid": 0.4, "dph_total": 0.9},
        launcher=lambda oid, body: (launched.append(body), (True, "inst-1", None))[1],
        digest_verifier=lambda image: profile.image_digest,
        bootstrap_stager=lambda dry_run=False: "boot-sha",
        instance_finder=lambda label: [],
        jobs_composer=_fake_jobs_composer,
        **prelude_kw)


def test_build_box_resolver_prepends_ssh_hf_prelude_when_present(tmp_path):
    """Gap B parity: a resolvable ssh pubkey + hf token prepend the SAME
    authorized_keys inject + hf_login prelude `_do_launch` uses, reaching the
    composer's onstart (and HF_TOKEN into env), matching the manual launch."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    stage = wf.stages[0]
    profile = wf.profiles[stage.profile]
    launched = []
    resolver = _prelude_resolver(
        fake, wm.mint_wf_id(wf.name), profile, launched,
        ssh_pubkey_provider=lambda: "ssh-ed25519 AAAAKEY test@host",
        hf_token_provider=lambda: "hf_TESTTOKEN")
    assert resolver(stage, wf, 0) == "inst-1"
    body = launched[0]
    onstart = body["onstart"]
    assert "authorized_keys" in onstart and "ssh-ed25519 AAAAKEY" in onstart
    assert "HF_TOKEN" in onstart                      # hf_login_snippet prelude
    assert body["env"]["HF_TOKEN"] == "hf_TESTTOKEN"


def test_build_box_resolver_absent_ssh_hf_does_not_raise(tmp_path):
    """No key/token (providers return None): the resolver launches cleanly with
    NO ssh/hf prelude and no HF_TOKEN env -- absence must never raise (keeps the
    fakes-based path green)."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    stage = wf.stages[0]
    profile = wf.profiles[stage.profile]
    launched = []
    resolver = _prelude_resolver(
        fake, wm.mint_wf_id(wf.name), profile, launched,
        ssh_pubkey_provider=lambda: None,
        hf_token_provider=lambda: None)
    assert resolver(stage, wf, 0) == "inst-1"
    body = launched[0]
    assert "authorized_keys" not in body["onstart"]
    assert "HF_TOKEN" not in body.get("env", {})

# --- FIX B (2026-07-15 audit): ArtifactContract.manifest_path is honored ------
def _custom_manifest_workflow(tmp_path, *, manifest_path):
    """two_stage_workflow variant whose generate contract declares a
    NON-default (workdir-relative) manifest_path."""
    gen_profile = pinned_profile()
    score_profile = pinned_profile(image="repo/eval:tag", max_bid=0.6, budget_usd=4.0)
    generate = JobStage(
        name="generate", bundle=_write_bundle_dir(tmp_path, "generate"),
        profile="generate", after=(), inputs={},
        outputs={"generations": ArtifactContract(
            kind="e2-generations", manifest_path=manifest_path)},
        retry=RetryPolicy(max_attempts=1, retry_on=()),
    )
    score = JobStage(
        name="score", bundle=_write_bundle_dir(tmp_path, "score"),
        profile="score", after=("generate",),
        inputs={"generations": InputRef(
            stage="generate", artifact="generations", dest="inputs/generate")},
        outputs={"scores": ArtifactContract(
            kind="e2-scores", manifest_path="results/artifact-manifest.json")},
        retry=RetryPolicy(max_attempts=1, retry_on=()),
    )
    return Workflow(
        version=1, name="e2-paired-toy", budget_usd=10.0, max_wall_s=12 * 3600,
        teardown="stop", profiles={"generate": gen_profile, "score": score_profile},
        stages=(generate, score),
    )


def finish_job_at(fake, job_id, *, kind, manifest_rel, arm_rel="out.txt",
                  arm_body="hello", box="44"):
    """`finish_job` with the manifest at a caller-chosen WORKDIR-RELATIVE
    path (results land under jobs/<id>/results/<workdir-relative-path>)."""
    for ev in ("claimed", "started"):
        jobmeta.emit_event(job_id, ev, actor=f"box:{box}", runner=fake,
                           bucket=BUCKET, instance_id=box)
    jobmeta.emit_event(job_id, "done", actor=f"box:{box}", runner=fake,
                       bucket=BUCKET, instance_id=box, rc=0)
    fake.store[f"jobs/{job_id}/results.DONE.json"] = json.dumps({"v": 1, "rc": 0})
    sha = hashlib.sha256(arm_body.encode("utf-8")).hexdigest()
    manifest = {"v": 1, "kind": kind,
                "arms": {"a0": {"path": arm_rel, "sha256": sha}}}
    fake.store[f"jobs/{job_id}/results/{manifest_rel}"] = json.dumps(manifest)
    fake.store[f"jobs/{job_id}/results/{arm_rel}"] = arm_body


def test_accept_stage_artifacts_honors_nondefault_manifest_path(tmp_path):
    """A generate contract with manifest_path='out/custom-manifest.json' is
    accepted from jobs/<id>/results/out/custom-manifest.json (NOT the e2
    default double-'results' key), the accepted record carries the
    manifest_path, and the downstream score asset's require list leads with
    the same workdir-relative path."""
    fake = FakeB2(bucket=BUCKET)
    wf = _custom_manifest_workflow(tmp_path, manifest_path="out/custom-manifest.json")
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    def tick(n):
        return wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                                 now=T(n), box_resolver=fixed_box_resolver)

    r1 = tick(10)
    assert (r1["action"], r1["stage"]) == ("stage_submitted", "generate")
    gen_job_id = r1["job_id"]

    finish_job_at(fake, gen_job_id, kind="e2-generations",
                  manifest_rel="out/custom-manifest.json", arm_rel="out/a.txt")
    # the e2 default key does NOT exist -- acceptance passing proves the
    # contract's manifest_path was actually threaded through
    assert f"jobs/{gen_job_id}/results/results/artifact-manifest.json" not in fake.store

    r2 = tick(20)
    assert r2["action"] == "artifact_accepted" and r2["artifact"] == "generations"
    rec = json.loads(
        fake.store[f"workflows/{wf_id}/artifacts/generate/generations.json"])
    assert rec["manifest_path"] == "out/custom-manifest.json"

    assert tick(30)["action"] == "stage_succeeded"

    r4 = tick(40)
    assert (r4["action"], r4["stage"]) == ("stage_submitted", "score")
    ticket = json.loads(fake.store[f"jobs/queue/44/{r4['job_id']}.json"])
    asset = ticket["config"]["assets"][-1]
    assert asset["require"][0] == "out/custom-manifest.json"
    assert "out/a.txt" in asset["require"]


def test_accept_stage_artifacts_default_manifest_path_byte_identical(tmp_path):
    """The e2 default ('results/artifact-manifest.json') still resolves to the
    historical jobs/<id>/results/results/... key and the require list still
    leads with the default rel path -- byte-identical pre-fix behavior."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    def tick(n):
        return wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                                 now=T(n), box_resolver=fixed_box_resolver)

    gen_job_id = tick(10)["job_id"]
    finish_job(fake, gen_job_id, kind="e2-generations")   # default key seeder
    assert tick(20)["action"] == "artifact_accepted"
    rec = json.loads(
        fake.store[f"workflows/{wf_id}/artifacts/generate/generations.json"])
    assert rec["manifest_path"] == "results/artifact-manifest.json"
    assert tick(30)["action"] == "stage_succeeded"
    r4 = tick(40)
    ticket = json.loads(fake.store[f"jobs/queue/44/{r4['job_id']}.json"])
    assert ticket["config"]["assets"][-1]["require"][0] == \
        "results/artifact-manifest.json"


# --- FIX A (2026-07-15 adversarial review): the REHEARSAL lane honors
# ArtifactContract.manifest_path too. Pre-fix, `_read_stage_manifest`
# hardcoded the e2 default: for a non-default contract BOTH the produced
# capture and the downstream byte-identity re-read returned None and the
# binding check — the pre-live-spend gate — passed VACUOUSLY (None == None).
def _write_custom_manifest_workflow_module(tmp_path, *, manifest_path):
    """A generate->score workflow MODULE file (`rehearse_workflow` loads a
    path, not an in-memory Workflow — same fixture shape as
    test_workflow_preflight.py's `_write_workflow_module`) whose generate
    contract declares a NON-default workdir-relative manifest_path."""
    gen_bundle = _write_bundle_dir(tmp_path, "generate")
    score_bundle = _write_bundle_dir(tmp_path, "score")
    gen_digest = "sha256:" + "a" * 64
    score_digest = "sha256:" + "b" * 64
    src = f'''\
from workflow import (
    ArtifactContract, InputRef, JobStage, ResourceProfile, RetryPolicy, Workflow,
)

_PROFILE = dict(
    gpu=("RTX 5090",), num_gpus=1, gpu_ram_gb=32, disk_gb=160,
    rental="bid", max_bid=1.0, budget_usd=5.0, max_wall_s=3600)

WORKFLOW = Workflow(
    version=1, name="e2-paired-toy", budget_usd=10.0, max_wall_s=3600,
    teardown="stop",
    profiles={{
        "generate": ResourceProfile(
            image="repo/image:tag", image_digest={gen_digest!r}, **_PROFILE),
        "score": ResourceProfile(
            image="repo/eval:tag", image_digest={score_digest!r}, **_PROFILE),
    }},
    stages=(
        JobStage(
            name="generate", bundle={gen_bundle!r}, profile="generate",
            after=(), inputs={{}},
            outputs={{"generations": ArtifactContract(
                kind="e2-generations", manifest_path={manifest_path!r})}},
            retry=RetryPolicy(max_attempts=1, retry_on=())),
        JobStage(
            name="score", bundle={score_bundle!r}, profile="score",
            after=("generate",),
            inputs={{"generations": InputRef(
                stage="generate", artifact="generations",
                dest="inputs/generate")}},
            outputs={{"scores": ArtifactContract(
                kind="e2-scores",
                manifest_path="results/artifact-manifest.json")}},
            retry=RetryPolicy(max_attempts=1, retry_on=())),
    ),
)
'''
    path = tmp_path / "wf_custom_manifest.py"
    path.write_text(src)
    return str(path)


_CUSTOM_GEN_MANIFEST_REL = "out/custom-manifest.json"


def _rehearser_writing_at(paths_by_stage):
    """A fake `stage_rehearser` that writes each stage's manifest at ITS OWN
    workdir-relative path (mirroring the real rehearser, which reads back at
    the contract's declared manifest_path) and returns the same dict."""
    def _rehearser(stage, bundle_dir, asset_overrides, results_out):
        rel = paths_by_stage[stage.name]
        manifest = {"v": 1, "kind": f"e2-{stage.name}", "rows": 1}
        dest = os.path.join(results_out, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        return 0, manifest
    return _rehearser


def test_rehearse_honors_nondefault_manifest_path(tmp_path):
    """The rehearsal PASSES when the manifest is present at the CUSTOM path:
    the produced capture and the downstream byte-identity re-read both
    resolve `<results_dir>/` + the contract's manifest_path (pre-fix, the
    re-read at the hardcoded default returned None against a real produced
    sha and the rehearsal failed spuriously — or, with the default rehearser,
    passed vacuously; see the absent-manifest test below)."""
    path = _write_custom_manifest_workflow_module(
        tmp_path, manifest_path=_CUSTOM_GEN_MANIFEST_REL)
    rehearser = _rehearser_writing_at(
        {"generate": _CUSTOM_GEN_MANIFEST_REL,
         "score": "results/artifact-manifest.json"})
    workdir = str(tmp_path / "rehearse-ok")

    rc, result = wc.rehearse_workflow(
        path, wf_id="20260713T000000-e2-paired-toy-c2c2", workdir=workdir,
        stage_rehearser=rehearser)

    assert rc == wc.EXIT_OK
    assert [s["name"] for s in result["stages"]] == ["generate", "score"]
    expected = wc._stable_manifest_sha256(
        {"v": 1, "kind": "e2-generate", "rows": 1})
    assert result["stages"][0]["manifest_sha256"] == expected
    # direct probe of the capture-read helper at the custom path
    assert wc._read_stage_manifest(
        os.path.join(workdir, "generate"), _CUSTOM_GEN_MANIFEST_REL) == \
        {"v": 1, "kind": "e2-generate", "rows": 1}


def test_rehearse_fails_loudly_when_declared_manifest_absent(tmp_path):
    """The vacuous-pass regression proper: a stage that DECLARES outputs but
    produces no readable manifest must FAIL the rehearsal — pre-fix this
    exact case sailed through to EXIT_OK (produced sha None, downstream
    re-read None, None == None)."""
    path = _write_custom_manifest_workflow_module(
        tmp_path, manifest_path=_CUSTOM_GEN_MANIFEST_REL)

    def rehearser_writing_nothing(stage, bundle_dir, asset_overrides, results_out):
        # mirrors the real rehearser when the manifest file is absent:
        # rc 0, manifest None
        return 0, None

    rc, result = wc.rehearse_workflow(
        path, wf_id="20260713T000000-e2-paired-toy-d3d3",
        workdir=str(tmp_path / "rehearse-absent"),
        stage_rehearser=rehearser_writing_nothing)

    assert rc == wc.EXIT_ARTIFACT
    assert result["failure_class"] == "ARTIFACT_INVALID"
    assert result["stage"] == "generate"
    assert _CUSTOM_GEN_MANIFEST_REL in result["error"]
    assert result["disclaimer"] == wc.REHEARSAL_DISCLAIMER


# --- FIX B' (2026-07-15 adversarial review): manifest_path stays in-frame -----
@pytest.mark.parametrize("bad", [
    "/etc/passwd",
    "/results/artifact-manifest.json",
    "../outside.json",
    "results/../../outside.json",
    "results/..",
])
def test_artifact_contract_rejects_frame_escaping_manifest_path(bad):
    """manifest_path is interpolated RAW into the jobs/<job_id>/results/
    frame (jobmeta.validate_generation_artifact) and joined under local
    rehearsal results dirs — a leading '/' or a '..' segment escapes it."""
    with pytest.raises(WorkflowError, match="manifest_path"):
        ArtifactContract(kind="e2-generations", manifest_path=bad)


def test_artifact_contract_allows_dotted_filenames():
    """Only '..' path SEGMENTS are rejected — dots INSIDE a segment stay
    legal (versioned filenames and the like)."""
    c = ArtifactContract(kind="e2-generations",
                          manifest_path="out/v1..final/manifest.json")
    assert c.manifest_path == "out/v1..final/manifest.json"


# --- FIX C (2026-07-15 audit): observed claim advances stage_submitted --------
def test_stage_started_emitted_once_on_observed_claim(tmp_path):
    """`workflow status` used to show a 4h-RUNNING job as stage_submitted:
    `stage_started` was in the frozen vocabulary + fold ladder but nothing
    emitted it. Once the controller observes the child job claimed, ONE
    `stage_started` event advances the folded stage status; subsequent ticks
    are plain noop_running (idempotent, no duplicate emit)."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    def tick(n):
        return wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                                 now=T(n), box_resolver=fixed_box_resolver)

    r1 = tick(10)
    assert r1["action"] == "stage_submitted"
    gen_job_id = r1["job_id"]

    # still only 'submitted' on the child job -> unchanged noop_running,
    # folded status stays stage_submitted
    assert tick(20)["action"] == "noop_running"
    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert v["stages"]["generate"]["status"] == "stage_submitted"

    # jobd claims the job -> next tick emits stage_started ONCE
    jobmeta.emit_event(gen_job_id, "claimed", actor="box:44", runner=fake,
                        bucket=BUCKET, instance_id="44")
    r3 = tick(30)
    assert r3["action"] == "stage_started" and r3["stage"] == "generate"
    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert v["stages"]["generate"]["status"] == "stage_started"   # fold surfaces it

    # idempotent: the folded status is no longer stage_submitted -> noop
    assert tick(40)["action"] == "noop_running"
    assert _event_names(fake, wf_id).count("stage_started") == 1

    # the run still drains normally to acceptance + success
    finish_job(fake, gen_job_id, kind="e2-generations")
    assert tick(50)["action"] == "artifact_accepted"
    assert tick(60)["action"] == "stage_succeeded"


def test_stage_started_emitted_after_stage_planned_crash(tmp_path):
    """FIX C extension (2026-07-15 adversarial review): a controller death
    BETWEEN the stage_planned and stage_submitted emits leaves a claimable
    ticket whose stage folds at stage_planned. Pre-fix the stage_started
    branch fired only from stage_submitted, so a resumed controller
    noop_running'ed the stage's fold at stage_planned forever; the observed
    claim now advances stage_planned too (both rank below stage_started in
    the fold ladder, so the emit stays idempotent)."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)

    # crash simulation: durable ticket + stage_planned event only — the
    # controller died before the stage_submitted emit (same ticket-first
    # ordering `_plan_and_submit_stage` writes).
    job_id = wm.stage_job_id(wf_id, "generate", 0)
    cfg, bundle_dir = wc._build_stage_config(wf_id, wf.stages[0], runner=fake,
                                             bucket=BUCKET)
    sha = wc._ensure_bundle_uploaded(bundle_dir, runner=fake, bucket=BUCKET)
    jobmeta.submit_with_id(job_id, cfg, "44", bundle_sha256=sha, actor=ACTOR,
                           runner=fake, bucket=BUCKET)
    wc.emit(wf_id, "stage_planned", ACTOR, runner=fake, bucket=BUCKET, ts=T(10),
            stage="generate", attempt=0, job_id=job_id, box="44")
    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert v["stages"]["generate"]["status"] == "stage_planned"

    def tick(n):
        return wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET,
                                 actor=ACTOR, now=T(n),
                                 box_resolver=fixed_box_resolver)

    # jobd claims the stranded-but-claimable job; the resumed controller's
    # next tick advances the fold past stage_planned exactly ONCE
    jobmeta.emit_event(job_id, "claimed", actor="box:44", runner=fake,
                        bucket=BUCKET, instance_id="44")
    r = tick(20)
    assert r["action"] == "stage_started" and r["stage"] == "generate"
    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert v["stages"]["generate"]["status"] == "stage_started"
    assert tick(30)["action"] == "noop_running"
    assert _event_names(fake, wf_id).count("stage_started") == 1


# --- regression: cancel_workflow with an explicit runner=None (commit 31047519)
def test_cancel_workflow_runner_none_resolves_default_runner(monkeypatch):
    """cancel_workflow(runner=None) used to clobber jobmeta.write_cancel_marker's
    `runner=_default_runner` PARAMETER default with a literal None -> TypeError
    (None is not callable). Fixed by wrapping with _resolve_runner; assert the
    resolved default runner is actually used: with `wc._default_runner`
    monkeypatched to the fake, the CANCEL marker + workflow_cancelled event
    land in the fake store."""
    fake = FakeB2(bucket=BUCKET)
    monkeypatch.setattr(wc, "_default_runner", fake)

    wf_id = wm.mint_wf_id("toy")
    rc, result = wc.cancel_workflow(wf_id, actor=ACTOR, reason="op says stop",
                                    runner=None, bucket=BUCKET)
    assert rc == wc.EXIT_CANCELLED == 3
    assert f"jobs/{wf_id}/CANCEL" in fake.store          # marker via resolved runner
    assert jobmeta.has_cancel_marker(wf_id, runner=fake, bucket=BUCKET)
    assert result["event"]["_emitted"] is True
    v = wc.view(wf_id, runner=fake, bucket=BUCKET)
    assert v["status"] == "cancelled" and v["terminal"] is True


# --- mid-run job-heartbeat liveness watchdog (2026-07-20 5h-blind fix) --------
# Run 5819's a2 box was outbid at 13:11; the instance lingered LISTED (a
# preempted spot box parks 'stopped', or a host-reclaimed one stays stale-
# 'running') so the 'gone'-only trigger never fired and the workflow sat blind
# for 5h. The watchdog presumes a CLAIMED/STARTED job dead once jobd's ~60s
# heartbeats have been silent past JOB_HEARTBEAT_STALE_S, regardless of the
# box's observed power state, and tears down + retargets under the same
# attempt (the job resumes from jobs/<id>/checkpoints/).
def _started_job_setup(tmp_path, hb_ts=None):
    """Drive a REAL first tick (plan+submit+acquire inst-0), then emit the
    box-side claimed/started(/heartbeat) lifecycle so the generate job is
    mid-run, its newest liveness signal at `hb_ts` (None: claimed only — no
    started/heartbeat, i.e. no staleness baseline at all)."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    wc.write_spec(wf, wf_id, runner=fake, bucket=BUCKET)
    launched = []

    def launcher(offer_id, body):
        cid = f"inst-{len(launched)}"
        launched.append(cid)
        return True, cid, None

    resolver = _real_resolver(fake, wf_id, launcher=launcher)
    r1 = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                            now=T(10), box_resolver=resolver)
    job_id = r1["job_id"]
    lifecycle = [("claimed", T(20))]
    if hb_ts is not None:
        lifecycle += [("started", T(30)), ("heartbeat", hb_ts)]
    for ev, ts in lifecycle:
        jobmeta.emit_event(job_id, ev, actor="box:inst-0", runner=fake,
                           bucket=BUCKET, ts=ts, instance_id="inst-0")
    return fake, wf, wf_id, resolver, launched, job_id


def test_heartbeat_stale_stopped_box_retargets_not_resumes(tmp_path):
    """A started job with its last heartbeat 1140s old (> 900s) on a box
    observed 'stopped' is NOT resumed (an outbid spot box's resume re-enters
    at the losing bid and parks again — the 5h resume-forever loop): the box
    is destroyed and the job retargeted under the SAME attempt, ticket moved,
    with a durable box_retargeted event recording WHY."""
    fake, wf, wf_id, resolver, launched, job_id = _started_job_setup(tmp_path, T(100))
    torn = []
    r = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                           now=T(2000), box_resolver=resolver,
                           box_observer=lambda iid: "stopped",
                           box_starter=lambda iid: (True, None),
                           box_teardown=lambda iid, mode: (torn.append((iid, mode)), True)[1])
    assert r["action"] == "box_retargeted"
    assert r["reason"] == "job_heartbeat_stale"
    assert r["job_id"] == job_id and r["attempt"] == 0
    assert torn == [("inst-0", "destroy")]           # dead box destroyed, not poked
    assert launched == ["inst-0", "inst-1"]          # replacement under same attempt
    assert jobmeta.read_ticket("inst-0", job_id, runner=fake, bucket=BUCKET) is None
    moved = jobmeta.read_ticket("inst-1", job_id, runner=fake, bucket=BUCKET)
    assert moved is not None and moved["retargeted_from"] == "inst-0"
    evs = [json.loads(x) for x in wc.read_events(wf_id, runner=fake, bucket=BUCKET)]
    ret = [e for e in evs if e.get("event") == "box_retargeted"]
    assert ret and ret[-1]["reason"] == "job_heartbeat_stale"
    assert ret[-1]["heartbeat_age_s"] > wc.JOB_HEARTBEAT_STALE_S
    assert ret[-1]["old_instance_id"] == "inst-0"


def test_heartbeat_fresh_stopped_box_still_resumes(tmp_path):
    """A stopped box whose job heartbeated 60s ago is a briefly-parked box:
    the cheap resume path is preserved — no teardown, no relaunch."""
    fake, wf, wf_id, resolver, launched, job_id = _started_job_setup(tmp_path, T(1900))
    started, torn = [], []
    r = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                           now=T(2000), box_resolver=resolver,
                           box_observer=lambda iid: "stopped",
                           box_starter=lambda iid: (started.append(iid), (True, None))[1],
                           box_teardown=lambda iid, mode: (torn.append(iid), True)[1])
    assert r["action"] == "box_resumed" and r["ok"] is True
    assert started == ["inst-0"] and torn == []
    assert launched == ["inst-0"]                    # no replacement launched


def test_heartbeat_stale_fires_on_stale_live_listing(tmp_path):
    """A host-reclaimed box can keep LISTING as 'running' while dead — the
    watchdog must fire on observed 'live' too (heartbeats are ground truth)."""
    fake, wf, wf_id, resolver, launched, job_id = _started_job_setup(tmp_path, T(100))
    torn = []
    r = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                           now=T(2000), box_resolver=resolver,
                           box_observer=lambda iid: "live",
                           box_teardown=lambda iid, mode: (torn.append((iid, mode)), True)[1])
    assert r["action"] == "box_retargeted"
    assert r["reason"] == "job_heartbeat_stale"
    assert torn == [("inst-0", "destroy")]
    assert launched == ["inst-0", "inst-1"]


def test_heartbeat_watchdog_needs_teardown_seam(tmp_path):
    """No box_teardown injected: the watchdog stays dormant (same opt-in
    posture as the boot watchdogs) and a stopped box gets the legacy resume."""
    fake, wf, wf_id, resolver, launched, job_id = _started_job_setup(tmp_path, T(100))
    started = []
    r = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                           now=T(2000), box_resolver=resolver,
                           box_observer=lambda iid: "stopped",
                           box_starter=lambda iid: (started.append(iid), (True, None))[1])
    assert r["action"] == "box_resumed"
    assert started == ["inst-0"] and launched == ["inst-0"]


def test_heartbeat_watchdog_silent_on_claimed_staging(tmp_path):
    """A CLAIMED job is silent-by-design: jobd's heartbeat loop only launches
    after `started`, and the claimed->started window is asset/venv staging
    that legitimately runs 10+ min (score stage). The watchdog must never
    presume a claimed-but-quiet job dead — resume stays the action."""
    fake, wf, wf_id, resolver, launched, job_id = _started_job_setup(tmp_path, None)
    started = []
    r = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                           now=T(2000), box_resolver=resolver,
                           box_observer=lambda iid: "stopped",
                           box_starter=lambda iid: (started.append(iid), (True, None))[1],
                           box_teardown=lambda iid, mode: True)
    assert r["action"] == "box_resumed"
    assert started == ["inst-0"] and launched == ["inst-0"]


# --- workflow bid pricing + geo pinning (2026-07-20 outbid/slow-host fixes) ---
def test_bid_price_capped_at_profile_max_bid(tmp_path):
    """_auto_bid_price headroom never exceeds the profile's max_bid: score
    profile max_bid=0.6, floor 0.55 -> 1.2x = 0.66 -> capped to 0.60 (the
    offer search already filters min_bid <= max_bid, so the cap stays >= floor)."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    launched = []
    resolver = wc.build_box_resolver(
        wf_id=wf_id, actor=ACTOR, runner=fake, bucket=BUCKET,
        offer_picker=lambda **kw: {"id": 7, "min_bid": 0.55, "dph_total": 0.9},
        launcher=lambda oid, body: (launched.append(body), (True, "inst-1", None))[1],
        digest_verifier=lambda image: "sha256:" + "a" * 64,
        bootstrap_stager=lambda dry_run=False: "boot-sha",
        instance_finder=lambda label: [],
        jobs_composer=_fake_jobs_composer)
    resolver(wf.stages[1], wf, 0)                    # "score": max_bid=0.6
    assert launched[0]["price"] == 0.6


def test_bid_price_ignores_razor_thin_dph_total(tmp_path):
    """dph_total on a bid-type offer is floor+storage adders, NOT the
    on-demand price — clamping below it squashes the 1.2x headroom back to
    floor+1¢ (two consecutive 2026-07-30 score boxes outbid mid-pull). The
    resolver must price 1.2x floor even when dph_total sits 1¢ above floor."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    launched = []
    resolver = wc.build_box_resolver(
        wf_id=wf_id, actor=ACTOR, runner=fake, bucket=BUCKET,
        offer_picker=lambda **kw: {"id": 7, "min_bid": 0.293, "dph_total": 0.303},
        launcher=lambda oid, body: (launched.append(body), (True, "inst-1", None))[1],
        digest_verifier=lambda image: "sha256:" + "a" * 64,
        bootstrap_stager=lambda dry_run=False: "boot-sha",
        instance_finder=lambda label: [],
        jobs_composer=_fake_jobs_composer)
    resolver(wf.stages[0], wf, 0)
    assert launched[0]["price"] == round(0.293 * bidpolicy.BID_TARGET_MULT, 3) > 0.303


def test_pick_cheapest_offer_geo_filter(monkeypatch):
    """geo=("us",) adds an uppercased geolocation-in filter to the bundles
    query; absent geo adds no geolocation key (unchanged global search)."""
    calls = []

    def fake_request(method, path, body=None, **kw):
        calls.append(body)
        return True, {"offers": [{"id": 1}]}, None

    monkeypatch.setattr(api, "request_soft", fake_request)
    assert offers.pick_cheapest_offer(gpu=("RTX 5090",), geo=("us",)) == {"id": 1}
    assert calls[0]["geolocation"] == {"in": ["US"]}
    assert offers.pick_cheapest_offer(gpu=("RTX 5090",)) == {"id": 1}
    assert "geolocation" not in calls[1]


def test_profile_geo_spec_roundtrip():
    """ResourceProfile.geo survives spec_to_dict/from_dict; an OLD spec.json
    with no geo key folds to () (forward-compatible reader)."""
    p = pinned_profile(geo=("US", "CA"))
    d = wm._profile_to_dict(p)
    assert d["geo"] == ["US", "CA"]
    assert wm._profile_from_dict(d).geo == ("US", "CA")
    d.pop("geo")
    assert wm._profile_from_dict(d).geo == ()


def test_owned_instance_ids_mixed_int_str(tmp_path):
    """One fold can hold an int instance_id (fresh launch: vast API int) AND a
    str one (ADOPT re-acquire: label lookup) — found live 2026-07-30, run
    2ed9: sorted({str, int}) TypeError crashed the controller on every cost
    tick. Both normalize to str, and int/str duplicates of the SAME box
    collapse to one entry (a mixed pair would double-count in the cost
    observer). box_released subtraction matches across shapes too."""
    v = {"stages": {
        "generate": {"instance_id": "46216906"},
        "score": {"instance_id": 46249864},
        "extra": {"instance_id": 46216906},          # same box, int shape
    }}
    assert wc._owned_instance_ids(v) == ["46216906", "46249864"]
    # released as INT (a jobd self-park emit) still subtracts from str-owned
    fake = FakeB2(bucket=BUCKET)
    wf_id = wm.mint_wf_id("wf")
    wc.emit(wf_id, "box_released", ACTOR, runner=fake, bucket=BUCKET,
            instance_id=46216906)
    assert wc.owned_boxes_remaining(v, wf_id, runner=fake, bucket=BUCKET) == ["46249864"]


def test_resolver_passes_profile_geo_to_picker(tmp_path):
    """build_box_resolver forwards the profile's geo allowlist into the offer
    search (None when unset — unchanged global search)."""
    fake = FakeB2(bucket=BUCKET)
    wf = two_stage_workflow(tmp_path)
    wf = dataclasses.replace(
        wf, profiles={**wf.profiles, "generate": pinned_profile(geo=("US",))})
    picker_calls = []

    def picker(**kw):
        picker_calls.append(kw)
        return {"id": 7, "min_bid": 0.4, "dph_total": 0.9}

    resolver = wc.build_box_resolver(
        wf_id=wm.mint_wf_id(wf.name), actor=ACTOR, runner=fake, bucket=BUCKET,
        offer_picker=picker,
        launcher=lambda oid, body: (True, "inst-1", None),
        digest_verifier=lambda image: "sha256:" + "a" * 64,
        bootstrap_stager=lambda dry_run=False: "boot-sha",
        instance_finder=lambda label: [],
        jobs_composer=_fake_jobs_composer)
    resolver(wf.stages[0], wf, 0)
    assert picker_calls[0]["geo"] == ("US",)
    resolver(wf.stages[1], wf, 0)                    # score profile: geo unset
    assert picker_calls[1]["geo"] is None


# --- velvet P3: the stale-image scheduling gate ------------------------------
# Incident, 2026-07-30: three frontier-wave jobs died within seconds on box
# 46240842 whose baked env predated a script they needed. P1 landed the pure
# tri-state classifier and alarmed only; these tests cover the enforcement half
# on the workflow controller's two box-REUSE paths (adopt, resume), which
# checked nothing at all. Per site: refuses on `stale`, proceeds on `fresh` and
# `not_applicable`, and on `unresolved` refuses WITHOUT launching (the HOLD).
OUR_IMAGE = "registry.example.com/train:t215-latest"
BAKED = "sha256:" + "a" * 64      # what the box was launched with (= the pin)
PUSHED = "sha256:" + "d" * 64     # what the tag points at after an env push


def _our_registry_workflow(tmp_path, *, pinned=True):
    """`two_stage_workflow` with both profiles on OUR registry — the only refs
    that carry a drift signal. `pinned=False` drops `image_digest`, the regime
    where the LIVE TAG is the reference (a pinned profile compares against its
    pin instead, and needs no lookup at all)."""
    wf = two_stage_workflow(tmp_path)
    profiles = {k: dataclasses.replace(
        p, image=OUR_IMAGE, image_digest=(BAKED if pinned else None))
        for k, p in wf.profiles.items()}
    return dataclasses.replace(wf, profiles=profiles)


def _stamped_box(iid, stamp, *, status="running"):
    """A vast instance record shaped like the wire: `extra_env` as [K, V] pairs
    (what `herdd._instance_env` reads back). `stamp=None` = a legacy box
    launched before/outside the digest-stamping path."""
    return {"id": iid, "actual_status": status, "image_uuid": OUR_IMAGE,
            "extra_env": ([[imageref.IMAGE_DIGEST_ENV, stamp]] if stamp else [])}


def _gate_resolver(fake, wf_id, *, existing, launcher, verifier):
    return wc.build_box_resolver(
        wf_id=wf_id, actor=ACTOR, runner=fake, bucket=BUCKET,
        offer_picker=lambda **kw: {"id": 7, "min_bid": 0.4, "dph_total": 0.9},
        launcher=launcher, digest_verifier=verifier,
        bootstrap_stager=lambda dry_run=False: "boot-sha",
        instance_finder=lambda label: [dict(i, label=label) for i in existing],
        jobs_composer=_fake_jobs_composer)


def _refusals(fake, wf_id):
    return [json.loads(r) for r in wc.read_events(wf_id, runner=fake, bucket=BUCKET)
            if json.loads(r).get("event") == "box_adopt_refused"]


def test_adopt_gate_refuses_stale_pinned_box_and_launches_fresh(tmp_path):
    """A labelled box whose stamp is NOT the profile's pin is confirmed stale:
    refuse the adopt, emit `box_adopt_refused(failure_class=STALE_IMAGE)`, and
    fall through to a fresh launch — a launch bakes the current image by
    construction, so replacing beats holding a box that can never refresh."""
    fake = FakeB2(bucket=BUCKET)
    wf = _our_registry_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)
    launched, verifier_calls = [], []

    def verifier(image):
        verifier_calls.append(image)
        return BAKED                              # tag still == the pin

    r = _gate_resolver(fake, wf_id, existing=[_stamped_box("inst-old", PUSHED)],
                       launcher=lambda oid, body: (launched.append(body), (True, "inst-new", None))[1],
                       verifier=verifier)(wf.stages[0], wf, 0)

    assert r == "inst-new" and len(launched) == 1
    ref = _refusals(fake, wf_id)
    assert len(ref) == 1
    assert ref[0]["instance_id"] == "inst-old"
    assert ref[0]["image_state"] == imageref.IMG_STALE
    assert ref[0]["failure_class"] == "STALE_IMAGE"
    assert ref[0]["image_reason"]                 # a refusal always says WHY
    assert [e for e in _acquired_events(fake, wf_id) if e["adopted"]] == []
    # the PIN was the reference: the gate itself cost no registry call (the one
    # call is the launch path's own fail-closed digest verify).
    assert verifier_calls == [OUR_IMAGE]


def test_adopt_gate_adopts_pin_matching_box_after_the_tag_moved(tmp_path):
    """The box is stamped with the profile's PIN while the tag has since moved.
    That box runs exactly the image the workflow declared, so it stays
    adoptable — comparing it against the moved tag would refuse the adopt and
    then IMAGE_DRIFT its replacement launch, killing an of-record workflow with
    someone else's push."""
    fake = FakeB2(bucket=BUCKET)
    wf = _our_registry_workflow(tmp_path)
    wf_id = wm.mint_wf_id(wf.name)

    def launcher(offer_id, body):
        raise AssertionError("a pin-matching box must be adopted, not replaced")

    r = _gate_resolver(fake, wf_id, existing=[_stamped_box("inst-9", BAKED)],
                       launcher=launcher, verifier=lambda image: PUSHED)(
                           wf.stages[0], wf, 0)

    assert r == "inst-9"
    assert _refusals(fake, wf_id) == []
    ev = _acquired_events(fake, wf_id)[0]
    assert ev["adopted"] is True and ev["image_state"] == imageref.IMG_FRESH


def test_adopt_gate_unpinned_stale_box_launches_fresh(tmp_path):
    """UNPINNED profile: the live tag is the only reference, so a box stamped
    with the pre-push digest is stale and refused (same fall-through to a fresh
    launch, which is unconstrained here because there is no pin to verify)."""
    fake = FakeB2(bucket=BUCKET)
    wf = _our_registry_workflow(tmp_path, pinned=False)
    wf_id = wm.mint_wf_id(wf.name)
    launched = []

    r = _gate_resolver(fake, wf_id, existing=[_stamped_box("inst-old", BAKED)],
                       launcher=lambda oid, body: (launched.append(body), (True, "inst-new", None))[1],
                       verifier=lambda image: PUSHED)(wf.stages[0], wf, 0)

    assert r == "inst-new" and len(launched) == 1
    assert _refusals(fake, wf_id)[0]["failure_class"] == "STALE_IMAGE"


def test_adopt_gate_unresolved_holds_without_launching(tmp_path):
    """THE HOLD, and the whole point of the four-state classifier: our registry
    + a stamp + a resolution that failed = `unresolved`. Refuse the adopt AND
    do not launch — auto-launching on "could not compare" turns a transient
    registry/API outage into one rented box per tick. Returning None costs a
    `need_box` tick and retries for free."""
    fake = FakeB2(bucket=BUCKET)
    wf = _our_registry_workflow(tmp_path, pinned=False)
    wf_id = wm.mint_wf_id(wf.name)

    def launcher(offer_id, body):
        raise AssertionError("HOLD means NO launch — this is the owner ruling")

    r = _gate_resolver(fake, wf_id, existing=[_stamped_box("inst-old", BAKED)],
                       launcher=launcher, verifier=lambda image: None)(
                           wf.stages[0], wf, 0)

    assert r is None                              # -> reconcile_tick: need_box
    ref = _refusals(fake, wf_id)
    assert len(ref) == 1
    assert ref[0]["image_state"] == imageref.IMG_UNRESOLVED
    assert ref[0]["failure_class"] == "IMAGE_UNRESOLVED"
    assert _acquired_events(fake, wf_id) == []    # nothing acquired, no spend


def test_adopt_gate_not_applicable_adopts_without_any_lookup(tmp_path):
    """An UNSTAMPED box (legacy, or a launch path that skipped the stamp) is
    `not_applicable` BY CONSTRUCTION: adopt it silently, and never pay a
    registry call to decide that — the classifier short-circuits before the
    digest is consulted and this caller must not defeat that."""
    fake = FakeB2(bucket=BUCKET)
    wf = _our_registry_workflow(tmp_path, pinned=False)
    wf_id = wm.mint_wf_id(wf.name)

    def verifier(image):
        pytest.fail("an unstamped box must cost ZERO registry lookups")

    r = _gate_resolver(fake, wf_id, existing=[_stamped_box("inst-5", None)],
                       launcher=lambda oid, body: (True, "x", None),
                       verifier=verifier)(wf.stages[0], wf, 0)

    assert r == "inst-5"
    assert _refusals(fake, wf_id) == []
    assert _acquired_events(fake, wf_id)[0]["image_state"] == imageref.IMG_NOT_APPLICABLE


def test_adopt_gate_prefers_the_fresh_twin_over_the_refused_one(tmp_path):
    """One label can hold BOTH a refused box and its replacement (the stale one
    keeps its label after the fall-through launch). Classification is per
    CANDIDATE, so the fresh twin is still adoptable — the label is not
    poisoned, and the workflow does not launch a third box every tick."""
    fake = FakeB2(bucket=BUCKET)
    wf = _our_registry_workflow(tmp_path, pinned=False)
    wf_id = wm.mint_wf_id(wf.name)

    def launcher(offer_id, body):
        raise AssertionError("the fresh twin must be adopted, not replaced")

    r = _gate_resolver(
        fake, wf_id,
        existing=[_stamped_box("inst-stale", BAKED),
                  _stamped_box("inst-fresh", PUSHED)],
        launcher=launcher, verifier=lambda image: PUSHED)(wf.stages[0], wf, 0)

    assert r == "inst-fresh"
    assert [e["instance_id"] for e in _refusals(fake, wf_id)] == ["inst-stale"]


def test_adopt_gate_alarm_only_knob_adopts_and_still_records(tmp_path,
                                                             monkeypatch):
    """WORKFLOW_IMAGE_GATE <= 0 (FLEETD_DESIGN's escape-hatch convention):
    the verdict is still computed and still recorded on the durable
    `box_acquired`, the gate just stops REFUSING. A gate an operator can soften
    is one that stays armed instead of being deleted."""
    monkeypatch.setattr(wc, "IMAGE_GATE_ENFORCE", 0)
    fake = FakeB2(bucket=BUCKET)
    wf = _our_registry_workflow(tmp_path, pinned=False)
    wf_id = wm.mint_wf_id(wf.name)

    r = _gate_resolver(fake, wf_id, existing=[_stamped_box("inst-old", BAKED)],
                       launcher=lambda oid, body: (True, "inst-new", None),
                       verifier=lambda image: PUSHED)(wf.stages[0], wf, 0)

    assert r == "inst-old"                        # adopted despite being stale
    assert _refusals(fake, wf_id) == []
    assert _acquired_events(fake, wf_id)[0]["image_state"] == imageref.IMG_STALE


def test_adopt_gate_tag_lookup_goes_through_the_ttl_cache(tmp_path):
    """The tag resolution wraps the injected verifier in
    `imageref.resolve_tag_digest_ttl`, not a bare call: a controller lives for
    hours and `image_tag_digest`/`image_ref_digest` cache a success for the life
    of the process, so a bare call would compare against the digest the tag had
    when the controller STARTED and never notice the push. Two resolves inside
    one TTL window = one lookup; a cleared cache resolves again."""
    fake = FakeB2(bucket=BUCKET)
    wf = _our_registry_workflow(tmp_path, pinned=False)
    wf_id = wm.mint_wf_id(wf.name)
    calls = []

    def verifier(image):
        calls.append(image)
        return PUSHED

    resolver = _gate_resolver(
        fake, wf_id, existing=[_stamped_box("inst-5", PUSHED)],
        launcher=lambda oid, body: (True, "x", None), verifier=verifier)
    assert resolver(wf.stages[0], wf, 0) == "inst-5"
    assert resolver(wf.stages[0], wf, 0) == "inst-5"
    assert calls == [OUR_IMAGE]                    # second resolve was a TTL hit
    imageref.clear_ttl_cache()
    assert resolver(wf.stages[0], wf, 0) == "inst-5"
    assert calls == [OUR_IMAGE, OUR_IMAGE]


# --- velvet P3, second gated site: the RESUME of a stopped box ---------------
# A resume is the one moment an image provably cannot refresh itself — vast
# keeps the box's disk, so a stale box comes back exactly as stale as it parked.
def _verdict_observer(verdict, seen=None):
    """`image_state_observer(instance_id, pinned_digest=None)` fake returning a
    fixed `(state, reason)`; `seen` collects the call kwargs."""
    def observer(instance_id, pinned_digest=None):
        if seen is not None:
            seen.append((instance_id, pinned_digest))
        if isinstance(verdict, Exception):
            raise verdict
        return verdict
    return observer


def _resume_tick(tmp_path, observer):
    """A started `generate` job on a FRESH heartbeat (so the mid-run liveness
    watchdog stays out of it) whose box is observed 'stopped' — the cheap
    resume path, now gated. Returns (action, started, launched)."""
    fake, wf, wf_id, resolver, launched, job_id = _started_job_setup(tmp_path, T(1900))
    started = []
    r = wc.reconcile_tick(wf, wf_id, runner=fake, bucket=BUCKET, actor=ACTOR,
                           now=T(2000), box_resolver=resolver,
                           box_observer=lambda iid: "stopped",
                           box_starter=lambda iid: (started.append(iid), (True, None))[1],
                           image_state_observer=observer)
    return r, started, launched


def test_resume_gate_stale_box_retargets_instead_of_resuming(tmp_path):
    """CONFIRMED stale: never PUT the box back to `running` — it would come
    back with the same old env and claim the job that dies on it (box 46240842,
    three frontier-wave jobs, seconds). Replace it under the SAME attempt; a
    fresh launch bakes the current image by construction."""
    r, started, launched = _resume_tick(
        tmp_path, _verdict_observer((imageref.IMG_STALE, "tag moved since launch")))

    assert r["action"] == "box_retargeted"
    assert r["reason"] == "stale_image"
    assert r["failure_class"] == "STALE_IMAGE"
    assert started == []                          # the resume PUT never happened
    assert launched == ["inst-0", "inst-1"]       # replacement under same attempt


def test_resume_gate_unresolved_holds_without_resuming_or_launching(tmp_path):
    """THE HOLD at the resume site: `unresolved` refuses the resume AND does not
    launch anything. Auto-launching on a registry/API outage would rent a box
    per tick; the held action re-checks for free next tick."""
    r, started, launched = _resume_tick(
        tmp_path, _verdict_observer((imageref.IMG_UNRESOLVED, "registry unreachable")))

    assert r["action"] == "box_resume_held"
    assert r["failure_class"] == "IMAGE_UNRESOLVED"
    assert r["image_state"] == imageref.IMG_UNRESOLVED and r["image_reason"]
    assert started == []                          # not resumed
    assert launched == ["inst-0"]                 # and NOT replaced — no spend


@pytest.mark.parametrize("state", [imageref.IMG_FRESH, imageref.IMG_NOT_APPLICABLE])
def test_resume_gate_proceeds_on_fresh_and_not_applicable(tmp_path, state):
    """The two proceed states are silent: the cheap in-place resume is
    preserved exactly, no retarget, no extra box."""
    r, started, launched = _resume_tick(tmp_path, _verdict_observer((state, "why")))
    assert r["action"] == "box_resumed" and r["ok"] is True
    assert started == ["inst-0"] and launched == ["inst-0"]


def test_resume_gate_unwired_observer_preserves_the_resume(tmp_path):
    """No `image_state_observer` injected (a caller that never wired it) -> the
    gate is dormant and behavior is byte-identical to before P3. Production is
    armed by `build_live_controller_deps`, not by this default."""
    r, started, launched = _resume_tick(tmp_path, None)
    assert r["action"] == "box_resumed"
    assert started == ["inst-0"] and launched == ["inst-0"]


def test_resume_gate_observer_exception_holds(tmp_path):
    """A raising observer is 'could not compare', not 'fine': fail CLOSED to the
    HOLD rather than resuming a box we failed to classify."""
    r, started, launched = _resume_tick(
        tmp_path, _verdict_observer(RuntimeError("registry blew up")))
    assert r["action"] == "box_resume_held"
    assert started == [] and launched == ["inst-0"]


def test_resume_gate_alarm_only_knob_resumes_stale(tmp_path, monkeypatch):
    """WORKFLOW_IMAGE_GATE <= 0 softens this site too — same escape hatch, same
    convention, so the gate can be turned down instead of ripped out."""
    monkeypatch.setattr(wc, "IMAGE_GATE_ENFORCE", 0)
    r, started, launched = _resume_tick(
        tmp_path, _verdict_observer((imageref.IMG_STALE, "tag moved")))
    assert r["action"] == "box_resumed"
    assert started == ["inst-0"] and launched == ["inst-0"]


def test_resume_gate_passes_the_active_profile_pin_to_the_observer(tmp_path):
    """The observer is handed the stage profile's of-record `image_digest`: the
    pin, not the live tag, is what an of-record box OUGHT to match (else a push
    by someone else refuses the resume and IMAGE_DRIFTs the replacement)."""
    seen = []
    r, _started, _launched = _resume_tick(
        tmp_path, _verdict_observer((imageref.IMG_FRESH, "match"), seen))
    assert r["action"] == "box_resumed"
    assert seen == [("inst-0", "sha256:" + "a" * 64)]   # pinned_profile's digest


# --- the observer builder itself (production default is a soft reader) -------
def test_image_state_observer_unreadable_record_holds():
    """`_get_instance_soft`-shaped None (any API error) is UNRESOLVED, never a
    silent proceed: we cannot prove the box runs current code."""
    obs = wc.build_image_state_observer(
        instance_reader=lambda iid: None,
        digest_verifier=lambda image: pytest.fail("no lookup without a record"))
    state, reason = obs("inst-0")
    assert state == imageref.IMG_UNRESOLVED and "unreadable" in reason


def test_image_state_observer_classifies_stamp_against_tag_then_pin():
    """Delegates to the ONE classifier: stamp vs tag when unpinned (stale), and
    stamp vs PIN when the caller supplies one (fresh, no lookup)."""
    box = _stamped_box("inst-0", BAKED, status="exited")
    calls = []

    def verifier(image):
        calls.append(image)
        return PUSHED

    obs = wc.build_image_state_observer(instance_reader=lambda iid: box,
                                        digest_verifier=verifier)
    assert obs("inst-0")[0] == imageref.IMG_STALE
    assert calls == [OUR_IMAGE]
    assert obs("inst-0", pinned_digest=BAKED)[0] == imageref.IMG_FRESH
    assert calls == [OUR_IMAGE]                    # the pin cost no new lookup


R2_IMAGE = f"{imageref.R2_REGISTRY_HOST}/train:t215-latest"


def test_image_state_observer_resolves_the_R2_DEFAULT_IMAGE(monkeypatch):
    """THE 2026-08-19 DEFECT, at the workflow resume gate. `_instance_image_
    verdict` gated its lookup on its OWN registry-host read, so every
    registry.example.com box skipped the lookup, classified `not_applicable`
    and the gate waved through a box running an image N pushes old."""
    monkeypatch.delenv("GITLAB_REGISTRY", raising=False)
    imageref.clear_ttl_cache()
    box = {"id": "inst-r2", "actual_status": "exited", "image_uuid": R2_IMAGE,
           "extra_env": [[imageref.IMAGE_DIGEST_ENV, BAKED]]}
    calls = []

    def verifier(image):
        calls.append(image)
        return PUSHED

    obs = wc.build_image_state_observer(instance_reader=lambda _i: box,
                                        digest_verifier=verifier)
    assert obs("inst-r2")[0] == imageref.IMG_STALE
    assert calls == [R2_IMAGE]
    imageref.clear_ttl_cache()
