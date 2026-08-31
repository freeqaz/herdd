"""Portable tests for gemm_probe.py — no GPU, no torch, no network.

The parts that matter on a boot path are all testable without hardware: the
busy-GPU guard, the VRAM budget planner, the wall-clock bound (exercised against
a fake child that sleeps), the record schema (which `mfu.py` has to be able to
read), and the field rendering (which jobd splices into a box event).

The one thing NOT covered here is whether the GEMMs reach peak on real silicon —
that is what the run itself measures, and nothing local can stand in for it.
"""
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemm_probe as gp  # noqa: E402
import mfu  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBE = os.path.join(_HERE, "gemm_probe.py")


def _gpu(idx=0, util=0, mem=300, **kw):
    g = {"idx": idx, "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
         "util": util, "mem_used_mb": mem, "mem_total_mb": 97887,
         "power_limit_w": 600.0, "sm_clock_mhz": 2370, "temp_c": 41,
         "throttle": ["none"]}
    g.update(kw)
    return g


def _metrics(*gpus):
    return {"gpus": list(gpus), "gpu_count": len(gpus)}


# --------------------------------------------------------------------------- #
# the guard: it must never perturb a co-tenant or an in-flight job
# --------------------------------------------------------------------------- #
def test_an_idle_box_is_probeable():
    assert gp.busy_reason([_gpu()]) is None


def test_a_loaded_card_refuses():
    assert "util" in gp.busy_reason([_gpu(util=99)])
    assert "mem" in gp.busy_reason([_gpu(mem=40000)])


def test_one_busy_card_among_idle_ones_refuses():
    """A four-card box with a co-tenant job on card 3 must not be probed: the
    GEMM would contend for host RAM bandwidth, PCIe and power headroom even on a
    'free' card (memory: cotenant-smoke-confounds-saturation-ab)."""
    r = gp.busy_reason([_gpu(0), _gpu(1), _gpu(2), _gpu(3, util=97, mem=61000)])
    assert r and "gpu3" in r


def test_an_unreadable_card_refuses_rather_than_assuming_idle():
    """Same rule metrics_probe.card_is_idle applies: an unreadable field means
    *cannot prove idle*. The conservative side is refusing."""
    assert gp.busy_reason([_gpu(util=None)]).startswith("unreadable_gpu")
    assert gp.busy_reason([_gpu(mem=None)]).startswith("unreadable_gpu")


def test_no_gpu_refuses():
    assert gp.busy_reason([]) == "no_gpu"


def test_a_running_jobd_job_refuses_even_on_an_idle_looking_card():
    """A job that has just been claimed but has not yet allocated VRAM looks
    idle to nvidia-smi. jobd's own `<jid>.running` census is the authoritative
    read and outranks the card sample."""
    r = gp.busy_reason([_gpu()], running_jobs=["20260807T0101-v12-abcd"])
    assert r.startswith("job_running:")


def test_running_job_ids_reads_the_state_dir(tmp_path):
    (tmp_path / "jobA.running").write_text("")
    (tmp_path / "jobB.running").write_text("")
    (tmp_path / "jobC.terminal").write_text("done")
    assert gp.running_job_ids(str(tmp_path)) == ["jobA", "jobB"]


def test_running_job_ids_on_a_missing_dir_is_empty_not_an_error():
    assert gp.running_job_ids("/nonexistent/jobd/state") == []
    assert gp.running_job_ids(None) == []


def test_probe_refuses_a_busy_box_and_still_records_attribution():
    """The refusal path must still carry the power cap and clock — a box too busy
    to bench is exactly the box someone will want to attribute later."""
    rec = gp.probe(metrics=_metrics(_gpu(util=100, mem=60000)))
    assert rec["status"] == "skipped:gpu_busy"
    assert rec["power_limit_w"] == 600
    assert rec["sm_clock_mhz"] == 2370
    assert "ceiling_tflops" not in rec


def test_check_only_measures_nothing_but_says_it_would():
    rec = gp.probe(metrics=_metrics(_gpu()), check_only=True)
    assert rec["status"] == "would_run"
    assert "shapes" not in rec


# --------------------------------------------------------------------------- #
# VRAM budget — the probe may not OOM a card
# --------------------------------------------------------------------------- #
def test_the_default_shape_set_fits_the_default_budget():
    kept, skipped = gp.plan_shapes(gp.GENERIC_SHAPES)
    assert kept == list(gp.GENERIC_SHAPES) and skipped == []


def test_the_default_shape_set_peaks_under_half_a_gigabyte():
    """The number that makes this safe to run beside anything: worst shape is
    8192x16384 bf16 output + its two inputs."""
    worst = max(gp.shape_bytes(*s) for s in gp.GENERIC_SHAPES)
    assert worst < 512 * 1024 ** 2


def test_an_oversized_shape_is_skipped_not_truncated():
    big = (12288, 18944, 18944)
    kept, skipped = gp.plan_shapes([*gp.GENERIC_SHAPES, big],
                                   budget_b=512 * 1024 ** 2)
    assert big not in kept
    assert skipped and skipped[0][0] == big


def test_probe_reports_when_nothing_fits_rather_than_benching_anyway():
    rec = gp.probe(metrics=_metrics(_gpu()), budget_b=1024)
    assert rec["status"] == "skipped:no_shape_fits_budget"
    assert len(rec["skipped_shapes"]) == len(gp.GENERIC_SHAPES)


def test_shape_bytes_counts_all_three_tensors():
    assert gp.shape_bytes(2, 3, 4, itemsize=2) == (6 + 12 + 8) * 2


@pytest.mark.parametrize("bad", ["8192x4096", "axbxc", "8192x0x4096", ""])
def test_parse_shape_refuses_junk(bad):
    with pytest.raises(ValueError):
        gp.parse_shape(bad)


def test_the_generic_set_covers_all_three_gemm_aspect_classes():
    """This is what licenses `mfu.harmonic_weighted` to weight a generic-shape
    ceiling over a model's own MAC mix: every class it weights has a measured
    rate (`lm_head` falls back to `mlp_up` by mfu.GEMM_CLASS_FALLBACK)."""
    classes = {mfu.classify_gemm(k, n) for _, k, n in gp.GENERIC_SHAPES}
    assert classes == {"attn_proj", "mlp_up", "mlp_down"}
    mix = mfu.mac_mix(mfu.gemma4_12b_text())
    assert set(mix) - classes == {"lm_head"}
    assert mfu.GEMM_CLASS_FALLBACK["lm_head"] in classes


# --------------------------------------------------------------------------- #
# the wall-clock bound — a probe that hangs must not wedge a boot
# --------------------------------------------------------------------------- #
def _fake_interpreter(tmp_path, name, body):
    """A stand-in 'python' that ignores its argv and does `body` instead."""
    import stat
    p = tmp_path / name
    p.write_text("#!/bin/sh\n" + body + "\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(p)


def test_run_bench_kills_a_child_that_overruns_the_deadline(tmp_path):
    """A hung CUDA call cannot be interrupted in-process, which is why the bench
    is a child at all. Stand in for it with an interpreter that sleeps."""
    blob, err = gp.run_bench(
        [(8, 8, 8)], deadline_s=0.5,
        python=_fake_interpreter(tmp_path, "python-sleeper", "exec sleep 60"))
    assert blob is None
    assert err.startswith("timeout_")


def test_a_timed_out_bench_is_recorded_as_skipped_not_failed(monkeypatch):
    """`skipped:timeout` and `failed:` are different diagnoses — a timeout is a
    box we learned nothing about, a failure is a box that tried and could not."""
    monkeypatch.setattr(gp, "run_bench",
                        lambda *a, **k: (None, "timeout_90.0s"))
    rec = gp.probe(metrics=_metrics(_gpu()))
    assert rec["status"] == "skipped:timeout"
    assert "ceiling_tflops" not in rec


def test_a_bench_that_crashes_is_recorded_and_does_not_raise(monkeypatch):
    monkeypatch.setattr(gp, "run_bench",
                        lambda *a, **k: (None, "bench_rc_1:ImportError"))
    rec = gp.probe(metrics=_metrics(_gpu()))
    assert rec["status"].startswith("failed:")
    assert "ImportError" in rec["reason"]


def test_run_bench_survives_an_unspawnable_interpreter():
    blob, err = gp.run_bench([(8, 8, 8)], python="/nonexistent/python",
                             deadline_s=5)
    assert blob is None and err.startswith("spawn_failed")


def test_run_bench_reports_unparseable_child_output(tmp_path):
    blob, err = gp.run_bench(
        [(8, 8, 8)], deadline_s=10,
        python=_fake_interpreter(tmp_path, "python-noise", "echo 'not json'"))
    assert blob is None and err == "bench_unparseable_output"


def test_probe_without_metrics_probe_refuses_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(gp, "_import_metrics_probe", lambda: None)
    rec = gp.probe()
    assert rec["status"] == "skipped:no_metrics_probe"


# --------------------------------------------------------------------------- #
# the record — mfu.py has to be able to divide by it
# --------------------------------------------------------------------------- #
_FAKE_BENCH = {
    "device": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "capability": "sm_120", "sm_count": 188, "vram_total_mb": 97887,
    "torch": "2.11.0+cu129", "cuda": "12.9", "dtype": "bf16",
    "warmup": 3, "iters": 15,
    "shapes": [{"m": 8192, "k": 4096, "n": 4096, "ms": 0.51, "tflops": 269.4},
               {"m": 8192, "k": 4096, "n": 16384, "ms": 4.1, "tflops": 268.2},
               {"m": 8192, "k": 16384, "n": 4096, "ms": 4.8, "tflops": 229.1}],
}


def test_the_record_is_readable_by_mfu_as_a_gemm_ceiling_blob():
    """The whole reason the schema is a superset: `mfu.py --ceiling-json` must
    take this file with no format branch, and the weighting must land strictly
    inside the measured min/max."""
    rec = gp.build_record(_FAKE_BENCH, attribution={"power_limit_w": 600})
    shape = mfu.gemma4_12b_text()
    c = mfu.Ceiling.from_gemm_ceiling_json(rec, weights=mfu.mac_mix(shape))
    assert 229.1 < c.tflops < 269.4
    assert c.device == _FAKE_BENCH["device"]


def test_a_bench_with_no_device_name_emits_no_tflops_at_all():
    """gemm_ceiling.py: "a TFLOP/s figure with no device attached is not
    quotable". The cheapest way to keep that true is for the unquotable number
    not to exist."""
    rec = gp.build_record({**_FAKE_BENCH, "device": "   "})
    assert rec["status"] == "refused:no_device"
    assert "ceiling_tflops" not in rec
    with pytest.raises(mfu.DenominatorError):
        mfu.Ceiling.from_gemm_ceiling_json(rec)


def test_the_record_carries_the_absolute_power_cap_not_a_percentage():
    """PERF_LEVERS §2.5: a 300 W-capped card and a 600 W card both read '100%'
    of their own limit, which is why box 46936034's 2.13x slowdown can never now
    be proven. The absolute cap rides with every ceiling."""
    a = gp.host_attribution([_gpu(0, power_limit_w=600.0),
                             _gpu(1, power_limit_w=300.0)])
    assert a["power_limit_w"] == 300          # MIN — the slowest card paces
    assert a["sm_clock_mhz"] == 2370 and a["gpu_count"] == 2


def test_host_attribution_surfaces_throttle_bits():
    a = gp.host_attribution([_gpu(throttle=["sw_power", "hw_thermal"])])
    assert a["throttle"] == ["hw_thermal", "sw_power"]


def test_host_attribution_on_a_gpuless_snapshot_is_empty_not_an_error():
    assert gp.host_attribution([]) == {"gpu_count": 0}


def test_shape_basis_is_recorded_and_defaults_to_generic():
    assert gp.probe(metrics=_metrics(_gpu()), check_only=True,
                    )["shape_basis"] == "generic"
    assert gp.probe(metrics=_metrics(_gpu()), check_only=True,
                    shapes=[(1, 1, 1)])["shape_basis"] == "model"


def test_machine_id_is_recorded_only_when_the_environment_supplies_it(monkeypatch):
    """"Inherit, never invent." vast injects no machine id, so the box must not
    fabricate one — resolving instance -> machine is hostfacts.py ingest's job,
    laptop-side, where the mapping is authoritative."""
    monkeypatch.delenv("MACHINE_ID", raising=False)
    monkeypatch.delenv("VAST_MACHINE_ID", raising=False)
    monkeypatch.setenv("INSTANCE_ID", "46947265")
    rec = gp.probe(metrics=_metrics(_gpu()), check_only=True)
    assert rec["instance_id"] == "46947265" and "machine_id" not in rec
    monkeypatch.setenv("MACHINE_ID", "140799")
    assert gp.probe(metrics=_metrics(_gpu()),
                    check_only=True)["machine_id"] == "140799"


# --------------------------------------------------------------------------- #
# field rendering — jobd splices these into a box event
# --------------------------------------------------------------------------- #
def test_rendered_fields_are_k_equals_v_with_no_whitespace():
    rec = gp.build_record(_FAKE_BENCH,
                          attribution=gp.host_attribution([_gpu()]))
    lines = gp.render_fields(rec).splitlines()
    for ln in lines:
        assert "=" in ln
        k, v = ln.split("=", 1)
        assert k and not any(c.isspace() for c in ln), ln
    d = dict(ln.split("=", 1) for ln in lines)
    assert d["status"] == "ok"
    assert d["ceiling_tflops"] == "269.4"
    assert d["power_limit_w"] == "600"
    assert d["shape_basis"] == "generic"
    # the device name loses its spaces but stays an attribution key
    assert "Server_Edition" in d["gpu"]


def test_rendered_fields_carry_every_shape_so_a_class_gap_is_visible():
    rec = gp.build_record(_FAKE_BENCH)
    d = dict(ln.split("=", 1) for ln in gp.render_fields(rec).splitlines())
    assert d["tflops_8192x16384x4096"] == "229.1"


def test_a_skipped_probe_still_renders_a_reason():
    rec = gp.probe(metrics=_metrics(_gpu(util=100)))
    d = dict(ln.split("=", 1) for ln in gp.render_fields(rec).splitlines())
    assert d["status"].startswith("skipped_gpu_busy")
    assert "util" in d["reason"]


# --------------------------------------------------------------------------- #
# CLI — the boot path invokes this, so it must not be able to fail a boot
# --------------------------------------------------------------------------- #
def _run(*argv, env=None):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run([sys.executable, _PROBE, *argv],
                          capture_output=True, text=True, timeout=180, env=e)


def test_cli_exits_zero_on_a_box_with_no_gpu():
    """The boot-path contract. A GPU-less box (or a rehearsal container, which
    has no nvidia-smi by construction) must get a recorded skip and rc=0."""
    r = _run("--check-only", env={"METRICS_NVIDIA_SMI": "/bin/false"})
    assert r.returncode == 0, r.stderr
    rec = json.loads(r.stdout)
    assert rec["status"] == "skipped:no_gpu"


def test_cli_strict_exits_nonzero_when_no_ceiling_was_produced():
    r = _run("--check-only", "--strict", env={"METRICS_NVIDIA_SMI": "/bin/false"})
    assert r.returncode == 1


def test_cli_writes_both_artifacts(tmp_path):
    out = tmp_path / "gemm.json"
    fields = tmp_path / "gemm.fields"
    r = _run("--check-only", "--quiet", "--out", str(out),
             "--fields-out", str(fields),
             env={"METRICS_NVIDIA_SMI": "/bin/false"})
    assert r.returncode == 0, r.stderr
    assert r.stdout == ""
    assert json.loads(out.read_text())["status"] == "skipped:no_gpu"
    assert "status=skipped_no_gpu" in fields.read_text()


def test_cli_refuses_a_malformed_shape_before_touching_anything():
    r = _run("--shape", "8192x4096")
    assert r.returncode == 2 and "MxKxN" in r.stderr


def _fake_nvidia_smi(tmp_path):
    """One idle RTX PRO 6000 in the CSV shape metrics_probe queries. Anything
    other than --query-gpu (the dmon PCIe sample) prints nothing."""
    import stat
    p = tmp_path / "nvidia-smi-fake"
    p.write_text(
        "#!/bin/sh\n"
        'case "$1" in --query-gpu*) ;; *) exit 0 ;; esac\n'
        "echo '0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 0, 0, 301, "
        "97887, 22.5, 600.00, 38, 405, P8, 0x0000000000000000'\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(p)


def test_cli_respects_the_state_dir_guard(tmp_path):
    """A card that reads idle is not enough: a job jobd claimed seconds ago has
    not allocated VRAM yet, and probing it would land in its first step."""
    state = tmp_path / "state"
    state.mkdir()
    (state / "somejob.running").write_text("")
    env = {"METRICS_NVIDIA_SMI": _fake_nvidia_smi(tmp_path)}
    r = _run("--check-only", "--state-dir", str(state), env=env)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["status"] == "skipped:job_running"
    # ...and with no job running the same box would be probed
    (state / "somejob.running").unlink()
    r = _run("--check-only", "--state-dir", str(state), env=env)
    rec = json.loads(r.stdout)
    assert rec["status"] == "would_run" and rec["power_limit_w"] == 600
