"""Tests for the submit-time `needs.gpu_ram_gb` gate (jobmeta.vram_gate_*).

The gate's design is an ASYMMETRY, and these tests exist mostly to pin it:
refuse only on what measurement PROVES wrong, advise on everything the estimate
merely suggests. A gate that blocked on the estimate would block roughly a third
of submits for being one card class off, which is worse than the mis-sizing it
was built to catch.

CPU-only, stdlib + pytest. Run: pytest tools/vast/test_vram_gate.py
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import jobmeta as jm  # noqa: E402
import vram_facts as vf  # noqa: E402


def _cfg(gpu_ram_gb=48, **env_over):
    # CE_CHUNK_MATMUL is pinned fp32 here even though the trainer's default
    # flipped to bf16 on 2026-08-10: these tests assert against the 9B anchor
    # GROUP, and the fp32 group is the populated one (17 anchors vs 1). The
    # default itself is asserted separately, in
    # test_trainer_defaults_are_filled_in — that is the drift tripwire.
    env = {"BASE_SLUG": "qwen35-9b", "MAX_SEQ": "12288", "BATCH": "1",
           "QUANT": "bf16", "GRAD_CKPT": "on", "LORA_R": "32",
           "CE_CHUNK_MATMUL": "fp32"}
    env.update({k: str(v) for k, v in env_over.items()})
    return {"needs": {"gpu": True, "gpu_ram_gb": gpu_ram_gb, "gpus": 1},
            "env": env}


# --- shape reading ------------------------------------------------------------

def test_non_training_bundles_are_left_alone():
    """An eval or a generation sweep has no predictable training footprint. It
    must pass through untouched rather than be sized by a table that knows
    nothing about it."""
    assert jm.vram_gate_findings({"needs": {"gpu_ram_gb": 24}, "env": {}}) is None
    assert jm.vram_gate_findings({"needs": {}, "env": {"BASE_SLUG": "x",
                                                       "MAX_SEQ": "4096"}}) is None


def test_base_is_recovered_from_assets_when_env_omits_it():
    """Most training bundles never set BASE_SLUG — they name the model once as
    their `base` asset's B2 prefix. Reading only the env missed five of them,
    v7 included."""
    assert jm.base_slug_from_assets(
        [{"name": "runset", "b2": "runsets/x"},
         {"name": "base", "b2": "base-models/qwen25-coder-7b-instruct"}]
    ) == "qwen25-coder-7b-instruct"
    assert jm.base_slug_from_assets([{"name": "runset", "b2": "runsets/x"}]) == ""

    cfg = {"needs": {"gpu_ram_gb": 48}, "env": {"MAX_SEQ": "12288"},
           "assets": [{"name": "base", "b2": "base-models/qwen35-9b"}]}
    shape = jm.vram_shape_from_env(cfg["env"], cfg["assets"])
    assert shape["base_slug"] == "qwen35-9b"


def test_trainer_defaults_are_filled_in():
    """An anchor records what the trainer RESOLVED; a bundle's env records only
    what it overrode. Without the defaults every bundle reads as an unmeasured
    shape — which is exactly what happened on the first run of this gate: all
    six training bundles came back UNCHECKED purely for leaving
    CE_CHUNK_MATMUL and PACKING alone.

    `ce_chunk_matmul` is bf16 here because the TRAINER's default flipped to
    bf16 on 2026-08-10. This assertion is the tripwire for the two drifting
    apart — it is not a statement about which dtype is better."""
    s = jm.vram_shape_from_env({"BASE_SLUG": "qwen35-9b", "MAX_SEQ": "12288"})
    assert s["ce_chunk_matmul"] == "bf16"
    assert s["packing"] == "off"
    assert s["grad_checkpointing"] is True
    assert s["target_modules"] == list(jm._TRAINER_DEFAULT_TARGETS)


def test_grad_ckpt_auto_resolves_to_on():
    """`--grad-checkpointing auto` -> on, in the trainer. The gate must agree,
    or an `auto` bundle lands in the (much larger) grad-ckpt-off group."""
    for v in ("auto", "", "on", "true"):
        s = jm.vram_shape_from_env({"BASE_SLUG": "q", "MAX_SEQ": "1", "GRAD_CKPT": v})
        assert s["grad_checkpointing"] is True, v
    s = jm.vram_shape_from_env({"BASE_SLUG": "q", "MAX_SEQ": "1", "GRAD_CKPT": "off"})
    assert s["grad_checkpointing"] is False


# --- the asymmetry ------------------------------------------------------------

def test_refuses_only_below_an_already_measured_peak():
    """The one blocking case, and it is a fact rather than a prediction: this
    exact shape has been observed at 26+ GB, so a declared floor of 16 would
    schedule it onto a card that cannot hold it."""
    f = jm.vram_gate_findings(_cfg(gpu_ram_gb=16))
    assert f and f["status"] == "ok"
    lines, refuse = jm.vram_gate_report(f)
    assert refuse
    assert "BELOW a peak" in lines[0]
    # and the message must name runs, so the reader can check the claim
    assert any("fit-ladder" in ln or "fla-probe" in ln for ln in lines)


def test_allow_drift_downgrades_the_refusal():
    f = jm.vram_gate_findings(_cfg(gpu_ram_gb=16))
    _, refuse = jm.vram_gate_report(f, allow_drift=True)
    assert not refuse


def test_oversized_advises_but_never_refuses():
    """Over-sizing costs money, not correctness. The estimate picks the right
    card class about two thirds of the time — enough to advise with, nowhere
    near enough to block on."""
    f = jm.vram_gate_findings(_cfg(gpu_ram_gb=96))
    lines, refuse = jm.vram_gate_report(f)
    assert not refuse
    assert "bigger card than the shape needs" in lines[0]


def test_inside_the_headroom_band_advises_but_never_refuses():
    """Declared above every measured peak but below peak+headroom: every run on
    record fits, there is just less reserved-pool margin. Not a refusal."""
    est = vf.estimate_peak_gb(base_slug="qwen35-9b", quant_mode="bf16",
                              max_seq=12288, batch=1, grad_checkpointing=True,
                              ce_chunk_matmul="fp32", lora_r=32, packing="off",
                              target_modules=list(jm._TRAINER_DEFAULT_TARGETS),
                              world_size=1)
    declared = int(est["gb"]) + 1          # above the peak, inside the headroom
    f = jm.vram_gate_findings(_cfg(gpu_ram_gb=declared))
    lines, refuse = jm.vram_gate_report(f)
    assert not refuse
    assert "headroom band" in lines[0]


def test_unmeasured_shape_notes_and_names_the_probe_but_never_refuses():
    f = jm.vram_gate_findings(_cfg(gpu_ram_gb=48, BASE_SLUG="nonexistent-model-9z"))
    assert f["status"] == "unmeasured"
    lines, refuse = jm.vram_gate_report(f)
    assert not refuse
    assert "UNCHECKED" in lines[0]
    assert any("fit-ladder" in ln for ln in lines)


def test_a_broken_facts_file_never_blocks_a_submit():
    lines, refuse = jm.vram_gate_report(
        {"status": "skipped", "declared": 48, "detail": "boom"})
    assert not refuse
    assert lines and lines[0].startswith("note:")


# --- against the bundles we actually ship -------------------------------------

# --- shapes the env spells differently ----------------------------------------

def test_window_ladder_is_read_when_no_max_seq_is_authored():
    """v10 authors WINDOW_LADDER instead of MAX_SEQ and was invisible to the
    gate — the largest training run in the campaign, 90 GB declared,
    unchecked."""
    env = dict(_cfg()["env"])
    env.pop("MAX_SEQ")
    env["WINDOW_LADDER"] = "20480,16384"
    shape = jm.vram_shape_from_env(env)
    assert shape is not None
    assert shape["max_seq"] == 16384, "must size the SMALLEST rung"


def test_an_authored_max_seq_wins_over_a_ladder():
    """v12 carries both (it pins the window and leaves the ladder inert)."""
    env = dict(_cfg()["env"], WINDOW_LADDER="20480,16384", MAX_SEQ="12288")
    assert jm.vram_shape_from_env(env)["max_seq"] == 12288


def test_an_unreadable_ladder_yields_no_opinion():
    env = dict(_cfg()["env"], WINDOW_LADDER="wide,wider")
    env.pop("MAX_SEQ")
    assert jm.vram_shape_from_env(env) is None
    assert jm.window_ladder_from_env({"WINDOW_LADDER": ""}) == []


def test_the_report_names_the_rung_it_sized():
    env = dict(_cfg()["env"], WINDOW_LADDER="20480,16384")
    env.pop("MAX_SEQ")
    f = jm.vram_gate_findings({"needs": {"gpu_ram_gb": 48, "gpus": 1}, "env": env})
    lines, refuse = jm.vram_gate_report(f)
    assert any("SMALLEST rung" in ln and "16384" in ln for ln in lines), lines


# --- knobs the BOX re-decides -------------------------------------------------

def test_autotune_with_unset_grad_ckpt_is_flagged_as_uncovered():
    """The gate sizes `on` (the trainer default it fills in) while autotune may
    pick `off` — measured 20.87 -> 52.20 GB. Nothing in the tree does this
    today; the guard is for the next fork that drops the key."""
    env = dict(_cfg()["env"], MODE="autotune")
    env.pop("GRAD_CKPT")
    f = jm.vram_gate_findings({"needs": {"gpu_ram_gb": 48, "gpus": 1}, "env": env})
    assert f["autotune_may_disable_grad_ckpt"] is True
    lines, refuse = jm.vram_gate_report(f)
    assert any("may pick `off`" in ln for ln in lines), lines
    assert not refuse, "an uncovered knob is an advisory, never a refusal"


@pytest.mark.parametrize("gc", ["on", "off"])
def test_an_explicit_grad_ckpt_is_not_flagged(gc):
    """launch_plan.sh only overrides grad-ckpt when it is unset/auto, and its
    only unrequested flip is off -> on (the VRAM-safety floor, the safe
    direction). Flagging those would be noise."""
    f = jm.vram_gate_findings(_cfg(MODE="autotune", GRAD_CKPT=gc))
    assert not f.get("autotune_may_disable_grad_ckpt")


# --- WHERE the refusal is allowed to fire ------------------------------------
# `job submit`/`jobmatrix` refuse before spending. The other paths do not get
# to, and the reason differs per path:
#   * requeue/retarget are RECOVERY. The bundle is already fixed (requeue
#     refuses any edit), so a refusal leaves the operator nothing to change
#     except abandoning the rescue — and fleetd drives retarget automatically
#     on an eviction, where refusing loses the job outright.
#   * a workflow's stages submit one at a time, hours in, unattended. The
#     refusal belongs at `workflow plan`, which is $0 and pre-spend.

def _undersized_bundle(tmp_path):
    """A real bundle dir whose declared floor is below a measured peak."""
    d = tmp_path / "under"
    d.mkdir()
    (d / "run.sh").write_text("#!/usr/bin/env bash\necho hi\n")
    env = "\n".join(f"  {k}: {v!r}" for k, v in _cfg()["env"].items())
    (d / "job-config.yaml").write_text(
        "version: 1\nname: under\nentrypoint: run.sh\n"
        "needs:\n  gpus: 1\n  gpu_ram_gb: 16\n"
        f"env:\n{env}\n"
        "results:\n  - \"out/**\"\n")
    return str(d)


def test_the_bundle_fixture_really_is_refused(tmp_path):
    """Guards the two tests below: if this bundle ever stopped being refused,
    they would both pass by measuring nothing."""
    d = _undersized_bundle(tmp_path)
    cfg, _ = jm.validate_job_config(jm.load_job_config(d), d)
    assert jm.vram_gate_report(jm.vram_gate_findings(cfg))[1] is True


def test_workflow_plan_refuses_an_undersized_stage(tmp_path):
    """Pre-spend, offline, no box — so the workflow's refusal goes here."""
    import types
    # the port, not the `workflowctl` shim: this calls the refusal directly, and
    # a shim re-export would leave the assertion pointed at a name whose body
    # lives elsewhere (plan step 7).
    from vastlib.workflows import ctl as wc
    stage = types.SimpleNamespace(name="train", bundle=_undersized_bundle(tmp_path))
    with pytest.raises(wc.WorkflowCtlError, match="already measured"):
        wc._validate_stage_bundle(stage)


def test_workflow_plan_passes_a_correctly_sized_stage(tmp_path):
    import types
    from vastlib.workflows import ctl as wc
    d = _undersized_bundle(tmp_path)
    p = os.path.join(d, "job-config.yaml")
    body = open(p).read().replace("gpu_ram_gb: 16", "gpu_ram_gb: 48")
    open(p, "w").write(body)
    stage = types.SimpleNamespace(name="train", bundle=d)
    assert wc._validate_stage_bundle(stage)["name"] == "under"


def test_the_recovery_advisory_prints_but_never_exits(tmp_path, capsys):
    """`_vram_advisory` must surface the same finding and swallow the refusal:
    the facts table can move under a fixed bundle, but a rescue must not be
    blocked by it."""
    import herdd as vc
    d = _undersized_bundle(tmp_path)
    cfg, _ = jm.validate_job_config(jm.load_job_config(d), d)
    vc._vram_advisory(cfg, where="requeue")          # must not raise/SystemExit
    err = capsys.readouterr().err
    assert "vram (requeue)" in err and "BELOW a peak" in err


def test_the_recovery_advisory_survives_a_junk_config(capsys):
    """Advice must never break a recovery, whatever it is handed."""
    import herdd as vc
    for junk in (None, {}, {"needs": None}, {"env": "not-a-dict"}):
        vc._vram_advisory(junk, where="retarget")
    assert capsys.readouterr().err == ""


def test_no_shipped_bundle_is_provably_undersized():
    """Nothing in the tree declares a floor below a peak its own shape has
    measured. This is the landing state; a bundle that breaks it is either
    mis-sized or has genuinely changed shape and needs a fresh anchor."""
    import glob
    bad = []
    for p in sorted(glob.glob(os.path.join(
            os.path.dirname(os.path.dirname(HERE)), "tools", "*", "jobs", "*",
            "job-config.yaml"))):
        d = os.path.dirname(p)
        try:
            cfg, _ = jm.validate_job_config(jm.load_job_config(d), d)
        except jm.JobmetaError:
            continue                      # not this test's business
        _, refuse = jm.vram_gate_report(jm.vram_gate_findings(cfg))
        if refuse:
            bad.append(os.path.basename(d))
    assert not bad, f"provably under-sized bundles: {bad}"
