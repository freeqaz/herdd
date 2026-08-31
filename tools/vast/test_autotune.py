"""Tests for autotune.py (canonical planner) and its bash mirror
(tools/vast/jobcommon/launch_plan.sh).

The mirror used to live in each bundle, and this parity check read exactly ONE
of the nine copies — so eight were identical by luck, not by test. It is now a
shared `includes:` file, so checking the one copy checks every bundle.

CPU-only, stdlib + pytest. Run: pytest tools/vast/test_autotune.py
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import autotune     # noqa: E402
import vram_facts   # noqa: E402

LAUNCH_PLAN = os.path.join(HERE, "jobcommon", "launch_plan.sh")


# --- pure-function unit tests -------------------------------------------------

def test_resolve_mode():
    assert autotune.resolve_mode(None) == "pinned"
    assert autotune.resolve_mode("") == "pinned"
    assert autotune.resolve_mode("  ") == "pinned"
    assert autotune.resolve_mode("pinned") == "pinned"
    assert autotune.resolve_mode("AUTOTUNE") == "autotune"
    with pytest.raises(autotune.AutotuneError):
        autotune.resolve_mode("auto-tune")   # a typo must never land in pinned silently


def test_rebalance_grad_accum():
    assert autotune.rebalance_grad_accum(32, 1) == 32
    assert autotune.rebalance_grad_accum(32, 2) == 16
    assert autotune.rebalance_grad_accum(32, 4) == 8
    with pytest.raises(autotune.AutotuneError):
        autotune.rebalance_grad_accum(32, 3)   # not divisible -> refuse, never round
    with pytest.raises(autotune.AutotuneError):
        autotune.rebalance_grad_accum(0, 1)
    with pytest.raises(autotune.AutotuneError):
        autotune.rebalance_grad_accum(32, 0)


def test_pick_num_workers():
    # clamp(cores/(2*nproc), 2, 8)
    assert autotune.pick_num_workers(32, 1) == 8    # 16 -> cap 8
    assert autotune.pick_num_workers(32, 2) == 8
    assert autotune.pick_num_workers(16, 2) == 4
    assert autotune.pick_num_workers(8, 2) == 2
    assert autotune.pick_num_workers(2, 4) == 2     # floor
    assert autotune.pick_num_workers(1, 1) == 2     # floor even on a potato


@pytest.mark.parametrize("ram", [8, 16, 24, 32, 40, 47.5, 48, 96, 192])
@pytest.mark.parametrize("resumable", [False, True])
def test_quant_default_is_bf16_at_every_card_size(ram, resumable):
    """Owner standing rule 2026-08-04: always train bf16 LoRAs.

    THE REGRESSION THIS PINS. quant_for_vram used to return 4bit at <=32 GB and
    8bit at 33-47 GB for every autotuned run, so the recipe a job trained was
    decided by whichever card the rental market handed out — a 40 GB box and an
    80 GB box ran different precisions from the identical bundle, and neither
    was chosen by anyone. The VRAM tiers were authored for the local 2x3090s;
    under the 2026-07-30 rent-the-big-box posture that constraint is gone."""
    assert autotune.quant_for_vram(ram, resumable=resumable) == "bf16"


def test_quantized_table_is_still_reachable_by_asking():
    """Opt-in must still work — the small-card table is deprecated, not removed."""
    assert autotune.quant_for_vram(24, allow_quantized=True) == "4bit"
    assert autotune.quant_for_vram(32, allow_quantized=True) == "4bit"
    assert autotune.quant_for_vram(40, allow_quantized=True) == "8bit"
    assert autotune.quant_for_vram(47.5, allow_quantized=True) == "8bit"
    assert autotune.quant_for_vram(48, allow_quantized=True) == "bf16"
    assert autotune.quant_for_vram(96, allow_quantized=True) == "bf16"


def test_quant_for_vram_resumable_prefers_bf16():
    # Even under the opt-in table, resumable spot runs avoid bnb (4/8-bit can't
    # reload optimizer state on --resume): bf16 wherever a 7B QLoRA fits
    # (>=24 GB), else 4bit, NEVER 8bit. This is an independent CORRECTNESS
    # argument for bf16, not a VRAM one — it is why the new default is safe.
    q = lambda gb: autotune.quant_for_vram(gb, resumable=True, allow_quantized=True)
    assert q(24) == "bf16"
    assert q(32) == "bf16"      # was 4bit
    assert q(40) == "bf16"      # was 8bit
    assert q(47.5) == "bf16"    # never 8bit
    assert q(96) == "bf16"
    assert q(16) == "4bit"      # too small for bf16
    assert q(23.9) == "4bit"


def test_plan_resumable_quant_suggestion():
    # a 40 GB card: default -> bf16 (2026-08-04 rule); opt-in table -> 8bit;
    # opt-in + resumable -> bf16 (dodges the bnb paged-optimizer resume crash)
    assert autotune.plan(mode="autotune", gpus=1, batch=1, grad_accum=32,
                         cpu_cores=8, gpu_ram_gb=40)["PLAN_QUANT"] == "bf16"
    assert autotune.plan(mode="autotune", gpus=1, batch=1, grad_accum=32,
                         cpu_cores=8, gpu_ram_gb=40,
                         allow_quantized=True)["PLAN_QUANT"] == "8bit"
    assert autotune.plan(mode="autotune", gpus=1, batch=1, grad_accum=32,
                         cpu_cores=8, gpu_ram_gb=40, allow_quantized=True,
                         resumable=True)["PLAN_QUANT"] == "bf16"
    # pinned never suggests a precision, resumable or not
    assert "PLAN_QUANT" not in autotune.plan(mode="pinned", gpus=1, batch=1,
                                             grad_accum=32, gpu_ram_gb=40,
                                             resumable=True)


def test_grad_ckpt_off_constants_are_the_measured_anchors():
    """The fit rule's two constants must equal what vram_facts.json measures.

    THE REGRESSION THIS PINS. `GRAD_CKPT_OFF_REF_VRAM_GB` read **32** from
    2026-07-15 to 2026-08-11. It was derived from AUTOTUNE_DESIGN §4's per-card
    table, never reconciled against a run, and the run that measured the exact
    shape it describes — 7B-class LoRA, BATCH 1 x seq 4096, grad-ckpt OFF —
    peaked at **52.20 GB**. The 20 GB gap is not academic: at 96 GB x 12288
    tokens the rule computed `32*12288/4096 = 96` and `96 >= 96` passed, so it
    licensed grad-ckpt OFF for precisely the arm that OOMed a 94.97 GiB card
    (docs/plans/witness/perf/TRAINING_DEFAULTS_REVIEW_2026-08-09.md §2-§3).

    Binding the shipped scalars to the anchor table means a future re-harvest
    that moves either number fails here loudly, rather than leaving the planner
    calibrated on a value measurement has already refuted. If this fails after a
    `harvest_vram.py --write`, update BOTH constants AND the four bash mirrors
    (jobcommon/launch_plan.sh + the three runsets' train.sh)."""
    cal = vram_facts.grad_ckpt_off_calibration()
    assert autotune.GRAD_CKPT_REF_TOKENS == cal["ref_tokens"]
    assert autotune.GRAD_CKPT_OFF_REF_VRAM_GB == cal["ref_gb"], (
        f"shipped OFF reference {autotune.GRAD_CKPT_OFF_REF_VRAM_GB} != measured "
        f"{cal['ref_gb']} GB ({cal['ref_base']} {cal['ref_quant']}, run "
        f"{cal['ref_run']})")
    assert autotune.GRAD_CKPT_OFF_MIN_VRAM_GB == cal["floor_gb"], (
        f"shipped OFF floor {autotune.GRAD_CKPT_OFF_MIN_VRAM_GB} != measured "
        f"{cal['floor_gb']} GB at {cal['floor_tokens']} tokens "
        f"(run {cal['floor_run']})")
    # and the refuted value must not come back by any route
    assert autotune.GRAD_CKPT_OFF_REF_VRAM_GB > 32


def test_pick_grad_ckpt_agrees_with_every_measured_point():
    """Every grad-ckpt-OFF measurement we hold, run through the shipped rule.

    Rows 2-4 are the ones the old `32` anchor got WRONG (it said `off` at
    32/48 GB, and `off` for the 12288 arm that OOMed)."""
    # 7B bf16 B1 x 4096 OFF measured 52.20 GB -> anything under it must say 'on'
    assert autotune.pick_grad_ckpt(24, batch=1, max_seq=4096) == "on"
    assert autotune.pick_grad_ckpt(32, batch=1, max_seq=4096) == "on"
    assert autotune.pick_grad_ckpt(48, batch=1, max_seq=4096) == "on"
    assert autotune.pick_grad_ckpt(96, batch=1, max_seq=4096) == "off"   # fits
    # the arm that OOMed a 94.97 GiB card: 12288 tokens needs 156.6 GB, so the
    # 96 GB card is refused and so is the next class up (141 GB H200). 192 GB is
    # above the rule's threshold and no measurement contradicts that.
    assert autotune.pick_grad_ckpt(96, batch=1, max_seq=12288) == "on"
    assert autotune.pick_grad_ckpt(141, batch=1, max_seq=12288) == "on"
    # 8-bit B1 x 1024 OFF measured 21.87 GB: a 24 GB card fits it, a 16 does not.
    # A rule proportional through the origin would have said 13.05 GB and put
    # grad-ckpt OFF on the 16 GB card -> the OFF_MIN floor term exists for this.
    assert autotune.pick_grad_ckpt(24, batch=1, max_seq=1024) == "off"
    assert autotune.pick_grad_ckpt(16, batch=1, max_seq=1024) == "on"


def test_pick_grad_ckpt_batch_and_seq_scale_into_on():
    # activations scale with tokens-in-flight (batch*seq): a bigger batch or a
    # longer seq needs more VRAM to keep grad-ckpt OFF, wrapping back to ON.
    assert autotune.pick_grad_ckpt(96, batch=2, max_seq=4096) == "on"    # 8192 tok needs 104.4
    assert autotune.pick_grad_ckpt(96, batch=8, max_seq=4096) == "on"    # design ">=48 -> on + big batch"
    # long-seq run (the reasoning-sft-4b 32k case) keeps ckpt ON even on a 32 GB card
    assert autotune.pick_grad_ckpt(32, batch=1, max_seq=32768) == "on"   # needs 417.6
    # the fit rule is monotone in both card size and tokens-in-flight
    prev = 0.0
    for tok in (512, 1024, 2048, 4096, 8192, 16384, 32768):
        need = autotune.grad_ckpt_off_vram_gb(1, tok)
        assert need >= prev, (tok, need, prev)
        prev = need
    assert autotune.grad_ckpt_off_vram_gb(1, 4096) == pytest.approx(52.2)
    assert autotune.grad_ckpt_off_vram_gb(2, 2048) == pytest.approx(52.2)  # tokens, not seq


def test_plan_grad_ckpt_suggestion():
    # autotune + both inputs -> PLAN_GRAD_CKPT. Since 2026-08-28 the
    # throughput suggestion is 'hybrid' regardless of the fit rule: it
    # self-calibrates in-process (never OOMs where full GC fits) and
    # subsumes the on/off guess on every card size.
    p = autotune.plan(mode="autotune", gpus=1, batch=1, grad_accum=32,
                      cpu_cores=8, gpu_ram_gb=96, max_seq=4096)
    assert p["PLAN_GRAD_CKPT"] == "hybrid"
    p = autotune.plan(mode="autotune", gpus=1, batch=1, grad_accum=32,
                      cpu_cores=8, gpu_ram_gb=32, max_seq=32768)
    assert p["PLAN_GRAD_CKPT"] == "hybrid"
    p = autotune.plan(mode="autotune", gpus=1, batch=1, grad_accum=32,
                      cpu_cores=8, gpu_ram_gb=48, max_seq=4096)
    assert p["PLAN_GRAD_CKPT"] == "hybrid"
    # missing either input -> no suggestion (fit rule needs both)
    assert "PLAN_GRAD_CKPT" not in autotune.plan(
        mode="autotune", gpus=1, batch=1, grad_accum=32, gpu_ram_gb=96)
    assert "PLAN_GRAD_CKPT" not in autotune.plan(
        mode="autotune", gpus=1, batch=1, grad_accum=32, max_seq=4096)
    # pinned never emits a THROUGHPUT suggestion; with no requested grad_ckpt
    # there is nothing to safety-floor either, so still nothing on a big card.
    assert "PLAN_GRAD_CKPT" not in autotune.plan(
        mode="pinned", gpus=1, batch=1, grad_accum=32, gpu_ram_gb=96, max_seq=4096)


def test_grad_ckpt_vram_safe_floor():
    # only ever flips a requested 'off' -> 'on', and ONLY when off won't fit.
    assert autotune.grad_ckpt_vram_safe("off", 24, 1, 4096) == "on"    # 24GB @ B1/4096 OOMs off
    assert autotune.grad_ckpt_vram_safe("off", 32, 1, 4096) == "on"    # measured 52.20 GB
    assert autotune.grad_ckpt_vram_safe("off", 96, 1, 4096) == "off"   # fits -> unchanged
    assert autotune.grad_ckpt_vram_safe("off", 96, 8, 4096) == "on"    # big batch needs 417.6GB
    # a caller's 'on'/'auto' is never touched (no off->off->on churn)
    assert autotune.grad_ckpt_vram_safe("on", 24, 1, 4096) == "on"
    assert autotune.grad_ckpt_vram_safe("auto", 24, 1, 4096) == "auto"


# The live bundles whose EFFECTIVE grad-ckpt request is 'off' — either pinned
# explicitly, or left unset, which run.sh resolves as ${GRAD_CKPT:-off}. These
# are the only paths the VRAM-safety floor can act on, and each of them was
# licensed 'off' by the old 32 GB anchor on at least one rentable card class.
# (bundle, batch, max_seq, declared needs.gpu_ram_gb, requested)
_OFF_REQUESTING_BUNDLES = [
    ("repair-lifter-train", 1, 4096, 32, "off"),
    ("phase1-cot-train", 1, 16384, 48, ""),
    ("tooltrace-sft", 1, 16384, 48, ""),
]
_RENTABLE_CARDS = (16, 24, 32, 48, 80, 96, 141, 192)


@pytest.mark.parametrize("bundle,batch,seq,needs_gb,req", _OFF_REQUESTING_BUNDLES)
def test_live_off_bundles_are_floored_wherever_measurement_says_off_will_not_fit(
        bundle, batch, seq, needs_gb, req):
    """The floor must fire for every live grad-ckpt-OFF bundle on every card the
    measured rule says is too small — including the card its own
    `needs.gpu_ram_gb` asks for.

    Under the refuted 32 GB anchor: repair-lifter-train ran OFF on its declared
    32 GB floor (needs 52.2 measured), and phase1-cot-train / tooltrace-sft ran
    OFF at 16384 tokens on a 141 or 192 GB card (needs 208.8). All three are
    `MODE: pinned`, so the throughput picker never sees them — this floor is the
    only thing between them and the 2026-08-02 double-arm OOM."""
    need = autotune.grad_ckpt_off_vram_gb(batch, seq)
    assert need > needs_gb, (
        f"{bundle}: declared floor {needs_gb} GB now exceeds the measured OFF "
        f"need {need:.1f} GB — re-check this test, not just the constant")
    for card in _RENTABLE_CARDS:
        if card < needs_gb:
            continue                       # jobd would not schedule it here
        p = autotune.plan(mode="pinned", gpus=1, batch=batch, grad_accum=8,
                          cpu_cores=8, gpu_ram_gb=card, max_seq=seq,
                          grad_ckpt=(req or None))
        want = "on" if card < need else None
        got = p.get("PLAN_GRAD_CKPT")
        assert got == want, (bundle, card, need, got, want)


def test_plan_grad_ckpt_vram_safety_floor_both_modes():
    # The VRAM-safety floor fires in BOTH modes (numerically identical to off,
    # so safe for a pinned of-record run) — the whole point of the 24GB fallback.
    for mode in ("pinned", "autotune"):
        # 24GB + explicit off that won't fit -> flipped on
        p = autotune.plan(mode=mode, gpus=1, batch=1, grad_accum=32,
                          cpu_cores=8, gpu_ram_gb=24, max_seq=4096, grad_ckpt="off")
        assert p["PLAN_GRAD_CKPT"] == "on", (mode, p)
        # 96GB + explicit off that FITS -> no override (keeps the saturated shape)
        p = autotune.plan(mode=mode, gpus=1, batch=1, grad_accum=32,
                          cpu_cores=8, gpu_ram_gb=96, max_seq=4096, grad_ckpt="off")
        assert "PLAN_GRAD_CKPT" not in p, (mode, p)
        # 32GB at the same shape used to pass; measurement says 52.20 GB -> flip
        p = autotune.plan(mode=mode, gpus=1, batch=1, grad_accum=32,
                          cpu_cores=8, gpu_ram_gb=32, max_seq=4096, grad_ckpt="off")
        assert p["PLAN_GRAD_CKPT"] == "on", (mode, p)
        # explicit 'on' is never overridden (nothing to floor)
        p = autotune.plan(mode=mode, gpus=1, batch=1, grad_accum=32,
                          cpu_cores=8, gpu_ram_gb=24, max_seq=4096, grad_ckpt="on")
        assert "PLAN_GRAD_CKPT" not in p, (mode, p)


def test_plan_pinned_reproduces_historical_defaults():
    p = autotune.plan(mode=None, gpus=2, batch=1, grad_accum=32)
    assert p["PLAN_MODE"] == "pinned"
    assert p["PLAN_NPROC"] == 1            # never scales world_size
    assert p["PLAN_GRAD_ACCUM"] == 32      # verbatim
    assert p["PLAN_NUM_WORKERS"] == 2      # historical run.sh default
    assert p["PLAN_EFF_BATCH"] == 32
    assert "PLAN_QUANT" not in p           # pinned never suggests a precision
    # explicit workers still win in pinned mode
    assert autotune.plan(mode="pinned", gpus=1, batch=1, grad_accum=32,
                         num_workers=4)["PLAN_NUM_WORKERS"] == 4


def test_plan_autotune_holds_effective_batch():
    p = autotune.plan(mode="autotune", gpus=2, batch=1, grad_accum=32,
                      cpu_cores=32)
    assert p["PLAN_NPROC"] == 2
    assert p["PLAN_GRAD_ACCUM"] == 16
    assert p["PLAN_EFF_BATCH"] == 32
    assert p["PLAN_BATCH"] * p["PLAN_GRAD_ACCUM"] * p["PLAN_NPROC"] == 32
    with pytest.raises(autotune.AutotuneError):
        autotune.plan(mode="autotune", gpus=3, batch=1, grad_accum=32)


def test_plan_autotune_quant_suggestion():
    # The card size no longer decides the recipe (2026-08-04): both sizes bf16.
    assert autotune.plan(mode="autotune", gpus=1, batch=1, grad_accum=32,
                         cpu_cores=8, gpu_ram_gb=96)["PLAN_QUANT"] == "bf16"
    assert autotune.plan(mode="autotune", gpus=1, batch=1, grad_accum=32,
                         cpu_cores=8, gpu_ram_gb=24)["PLAN_QUANT"] == "bf16"
    assert autotune.plan(mode="autotune", gpus=1, batch=1, grad_accum=32,
                         cpu_cores=8, gpu_ram_gb=24,
                         allow_quantized=True)["PLAN_QUANT"] == "4bit"


def test_cache_key_stability_and_sensitivity():
    base = dict(model="qwen25-coder-7b", max_seq=4096, gpu_name="RTX PRO 6000",
                quant="bf16", world_size=2, packing=False)
    k1 = autotune.cache_key(**base)
    assert k1 == autotune.cache_key(**base)                 # stable
    assert k1 != autotune.cache_key(**{**base, "packing": True})
    assert k1 != autotune.cache_key(**{**base, "world_size": 1})
    assert k1 != autotune.cache_key(**{**base, "max_seq": 8192})


def test_cli_plan_and_refusal():
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "autotune.py"), "plan",
         "--mode", "autotune", "--gpus", "2", "--batch", "1",
         "--grad-accum", "32", "--cpu-cores", "32"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = dict(line.split("=", 1) for line in r.stdout.split())
    assert got["PLAN_NPROC"] == "2" and got["PLAN_GRAD_ACCUM"] == "16"
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "autotune.py"), "plan",
         "--mode", "autotune", "--gpus", "3", "--batch", "1",
         "--grad-accum", "32"],
        capture_output=True, text=True)
    assert r.returncode == 12
    assert "not divisible" in r.stderr


def test_cli_grad_ckpt_emitted_with_seq_and_vram():
    # --gpu-ram-gb + --max-seq together emit PLAN_GRAD_CKPT (autotune).
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "autotune.py"), "plan",
         "--mode", "autotune", "--gpus", "1", "--batch", "1",
         "--grad-accum", "32", "--gpu-ram-gb", "96", "--max-seq", "4096"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = dict(line.split("=", 1) for line in r.stdout.split())
    assert got["PLAN_GRAD_CKPT"] == "hybrid"
    # long seq on a small card: still 'hybrid' — the runtime calibrator holds
    # full GC until it measures headroom, so this is at least as safe as 'on'
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "autotune.py"), "plan",
         "--mode", "autotune", "--gpus", "1", "--batch", "1",
         "--grad-accum", "32", "--gpu-ram-gb", "32", "--max-seq", "32768"],
        capture_output=True, text=True)
    got = dict(line.split("=", 1) for line in r.stdout.split())
    assert got["PLAN_GRAD_CKPT"] == "hybrid"
    # 48 GB at the 4096 anchor -> hybrid too (the suggestion no longer
    # branches on the fit rule; the safety floor still does)
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "autotune.py"), "plan",
         "--mode", "autotune", "--gpus", "1", "--batch", "1",
         "--grad-accum", "32", "--gpu-ram-gb", "48", "--max-seq", "4096"],
        capture_output=True, text=True)
    got = dict(line.split("=", 1) for line in r.stdout.split())
    assert got["PLAN_GRAD_CKPT"] == "hybrid"
    # no --max-seq -> no PLAN_GRAD_CKPT line at all
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "autotune.py"), "plan",
         "--mode", "autotune", "--gpus", "1", "--batch", "1",
         "--grad-accum", "32", "--gpu-ram-gb", "96"],
        capture_output=True, text=True)
    assert "PLAN_GRAD_CKPT" not in r.stdout


# --- bash mirror parity (launch_plan.sh must match autotune.py exactly) --------

def _bash_plan(env_overrides: dict) -> tuple[int, dict]:
    """Source launch_plan.sh with a controlled env; return (rc, PLAN_* dict)."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    env.update({k: str(v) for k, v in env_overrides.items()})
    script = (
        f'source "{LAUNCH_PLAN}" || exit 99\n'
        'if plan_launch 2>/dev/null; then\n'
        '  echo "PLAN_MODE=$PLAN_MODE"; echo "PLAN_NPROC=$PLAN_NPROC"\n'
        '  echo "PLAN_BATCH=$PLAN_BATCH"; echo "PLAN_GRAD_ACCUM=$PLAN_GRAD_ACCUM"\n'
        '  echo "PLAN_NUM_WORKERS=$PLAN_NUM_WORKERS"; echo "PLAN_EFF_BATCH=$PLAN_EFF_BATCH"\n'
        '  echo "PLAN_QUANT=${PLAN_QUANT:-}"\n'
        '  echo "PLAN_GRAD_CKPT=${PLAN_GRAD_CKPT:-}"\n'
        'else\n  exit $?\nfi\n')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                       env=env)
    out = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return r.returncode, out


GRID = []
for mode in (None, "pinned", "autotune"):
    for batch, ga in ((1, 32), (8, 4), (2, 3)):
        for gpus in (1, 2, 4):
            for workers in (None, 4):
                GRID.append((mode, batch, ga, gpus, workers, 32))


@pytest.mark.parametrize("mode,batch,ga,gpus,workers,cores", GRID)
def test_bash_python_parity(mode, batch, ga, gpus, workers, cores):
    env = {"BATCH": batch, "GRAD_ACCUM": ga, "JOB_GPU_COUNT": gpus,
           "CPU_CORES": cores}
    if mode is not None:
        env["MODE"] = mode
    if workers is not None:
        env["NUM_WORKERS"] = workers
    rc, got = _bash_plan(env)
    try:
        want = autotune.plan(mode=mode, gpus=gpus, batch=batch, grad_accum=ga,
                             cpu_cores=cores, num_workers=workers)
    except autotune.AutotuneError:
        assert rc == 12, f"python refused but bash rc={rc} ({got})"
        return
    assert rc == 0, f"python planned but bash refused rc={rc}"
    for k in ("PLAN_MODE", "PLAN_NPROC", "PLAN_BATCH", "PLAN_GRAD_ACCUM",
              "PLAN_NUM_WORKERS", "PLAN_EFF_BATCH"):
        assert got[k] == str(want[k]), (
            f"{k}: bash={got[k]!r} python={want[k]!r} for "
            f"mode={mode} b={batch} ga={ga} gpus={gpus} w={workers}")


@pytest.mark.parametrize("ram,resumable,allow,want", [
    # DEFAULT (no ALLOW_QUANTIZED): bf16 everywhere, both resumable settings.
    (24, "0", "0", "bf16"), (32, "0", "0", "bf16"), (40, "0", "0", "bf16"),
    (48, "0", "0", "bf16"), (96, "0", "0", "bf16"), (16, "1", "0", "bf16"),
    (24, "1", "0", "bf16"), (40, "1", "0", "bf16"),
    # OPT-IN table (ALLOW_QUANTIZED=1): the historical VRAM tiers.
    (24, "0", "1", "4bit"), (32, "0", "1", "4bit"), (40, "0", "1", "8bit"),
    (48, "0", "1", "bf16"), (96, "0", "1", "bf16"),
    (24, "1", "1", "bf16"), (32, "1", "1", "bf16"), (40, "1", "1", "bf16"),
    (16, "1", "1", "4bit"), (96, "1", "1", "bf16"),
])
def test_bash_python_quant_parity(ram, resumable, allow, want):
    """launch_plan.sh PLAN_QUANT must match autotune.quant_for_vram over the
    VRAM x resumable x allow_quantized grid (the bash mirror cannot drift on
    either the 2026-08-04 bf16 default or the opt-in thresholds)."""
    rc, got = _bash_plan({"MODE": "autotune", "BATCH": 1, "GRAD_ACCUM": 32,
                          "JOB_GPU_COUNT": 1, "CPU_CORES": 8,
                          "JOB_GPU_RAM_GB": ram, "RESUMABLE": resumable,
                          "ALLOW_QUANTIZED": allow})
    assert rc == 0, got
    py = autotune.quant_for_vram(ram, resumable=(resumable == "1"),
                                 allow_quantized=(allow == "1"))
    assert got["PLAN_QUANT"] == want == py, (got, want, py)


@pytest.mark.parametrize("mode", ["pinned", "autotune"])
@pytest.mark.parametrize("ram,seq,req", [
    (24, 4096, "off"),     # safety flip -> on (both modes)
    (32, 4096, "off"),     # measured 52.20 GB -> flip (was 'fits' under the 32 anchor)
    (48, 4096, "off"),     # same
    (96, 4096, "off"),     # fits -> unchanged
    (24, 32768, "off"),    # long seq OOMs -> on
    (24, 4096, "on"),      # explicit on -> never touched
    (24, 4096, ""),        # unset: autotune suggests; pinned = effective 'off' -> flip
    (96, 4096, ""),        # unset: autotune 'off'; pinned fits -> nothing
    (96, 16384, ""),       # REGRESSION (phase1-cot 2026-08-02): pinned + unset at
                           # long seq needs 208.8 GB -> the floor MUST flip to 'on'
    (96, 16384, "off"),    # same card/seq, explicit off -> same flip
    (141, 16384, ""),      # H200: 'off' under the refuted 32 anchor -> now 'on'
    (192, 16384, "off"),   # B200: same
    (16, 1024, "off"),     # OFF_MIN floor term: 21.87 measured, 16 GB -> flip
    (24, 1024, "off"),     # ...and 24 GB clears it -> unchanged
    (256, 16384, ""),      # 256 GB clears 208.8 -> unchanged (no emit)
])
def test_bash_python_grad_ckpt_parity(mode, ram, seq, req):
    """launch_plan.sh PLAN_GRAD_CKPT must match autotune.plan over the
    (mode x VRAM x seq x requested) grid — the safety floor + throughput
    suggestion cannot drift between the bash mirror and the python."""
    env = {"MODE": mode, "BATCH": 1, "GRAD_ACCUM": 32, "JOB_GPU_COUNT": 1,
           "CPU_CORES": 8, "JOB_GPU_RAM_GB": ram, "MAX_SEQ": seq}
    if req:
        env["GRAD_CKPT"] = req
    rc, got = _bash_plan(env)
    assert rc == 0, got
    want = autotune.plan(mode=mode, gpus=1, batch=1, grad_accum=32, cpu_cores=8,
                         gpu_ram_gb=ram, max_seq=seq,
                         grad_ckpt=(req or None))
    assert got.get("PLAN_GRAD_CKPT", "") == str(want.get("PLAN_GRAD_CKPT", "")), \
        (mode, ram, seq, req, got.get("PLAN_GRAD_CKPT"), want.get("PLAN_GRAD_CKPT"))


def test_bash_pinned_emits_no_quant():
    # pinned mode never suggests a precision (run.sh falls back to ${QUANT:-4bit})
    rc, got = _bash_plan({"MODE": "pinned", "BATCH": 1, "GRAD_ACCUM": 32,
                          "JOB_GPU_COUNT": 1, "JOB_GPU_RAM_GB": 96, "RESUMABLE": "1"})
    assert rc == 0 and got.get("PLAN_QUANT", "") == "", got


def test_bash_unknown_mode_refused():
    rc, _ = _bash_plan({"MODE": "auto-tune", "BATCH": 1, "GRAD_ACCUM": 32,
                        "JOB_GPU_COUNT": 1, "CPU_CORES": 8})
    assert rc == 12


def test_launch_plan_parses():
    r = subprocess.run(["bash", "-n", LAUNCH_PLAN], capture_output=True, text=True)
    assert r.returncode == 0, f"{LAUNCH_PLAN}: {r.stderr}"


