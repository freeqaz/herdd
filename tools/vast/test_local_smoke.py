"""Offline tests for local_smoke.py — no podman, no GPU, no image.

What these pin is the part that can be wrong SILENTLY. A smoke that runs with
the wrong env still prints "SMOKE PASSED", so the merge precedence, the forced
guards, and the ldconfig line in the bootstrap are the things worth a test; the
podman invocation itself fails loudly and needs none.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_smoke as ls  # noqa: E402


# ---------------------------------------------------------------- env merge

def test_env_precedence_job_then_smoke_then_cli():
    job = {"env": {"QUANT": "bf16", "MAX_SEQ": "32768", "EXPECT_ROWS": "4992"}}
    smoke = {"env": {"QUANT": "8bit", "MAX_SEQ": "4096"}}
    env, prov = ls.merge_env(job, smoke, ["MAX_SEQ=8192"])
    assert env["QUANT"] == "8bit"           # smoke.yaml beats job-config
    assert env["MAX_SEQ"] == "8192"         # --env beats smoke.yaml
    assert env["EXPECT_ROWS"] == "4992"     # untouched keys are INHERITED
    assert prov["EXPECT_ROWS"] == "job-config"
    assert prov["QUANT"] == "smoke.yaml"
    assert prov["MAX_SEQ"] == "--env"


def test_env_values_are_stringified_not_yaml_typed():
    # job-config scalars parse to int/bool; the container env is strings only.
    env, _ = ls.merge_env({"env": {"STEPS": 10, "TF32": True}}, {}, [])
    assert env["STEPS"] == "10"
    assert isinstance(env["TF32"], str)


def test_env_rejects_malformed_cli_override():
    with pytest.raises(ls.SmokeError):
        ls.merge_env({}, {}, ["MAX_SEQ"])


def test_identity_gate_is_inherited_by_default():
    """The whole point of inheriting job-config: a smoke still runs the
    fail-closed corpus gate, so a swapped data/ file dies locally too."""
    job = {"env": {"EXPECT_SHA256": "abc", "EXPECT_DELTA_ROWS": "0"}}
    env, _ = ls.merge_env(job, {"env": {"STEPS": "10"}}, [])
    assert env["EXPECT_SHA256"] == "abc"
    assert env["EXPECT_DELTA_ROWS"] == "0"


# ------------------------------------------------------------ asset resolution

def test_asset_resolution_prefers_cli_then_smoke_then_convention(tmp_path):
    cli_dir = tmp_path / "cli"
    cli_dir.mkdir()
    smoke_dir = tmp_path / "smoke"
    smoke_dir.mkdir()
    job = {"assets": [{"name": "base", "b2": "base-models/x", "dest": "assets/x"}]}
    got = ls.resolve_assets(job, {"assets": {"base": str(smoke_dir)}},
                            {"base": str(cli_dir)})
    assert got["base"] == (str(cli_dir), "assets/x")
    got = ls.resolve_assets(job, {"assets": {"base": str(smoke_dir)}}, {})
    assert got["base"][0] == str(smoke_dir)


def test_asset_convention_finds_runset_build(tmp_path):
    build = tmp_path / "tools" / "vast" / "runsets" / "demo" / "_build"
    build.mkdir(parents=True)
    job = {"assets": [{"name": "runset", "b2": "runsets/demo", "dest": "assets/demo"}]}
    got = ls.resolve_assets(job, {}, {}, repo_root=str(tmp_path))
    assert got["runset"][0] == str(build)


def test_unresolved_asset_raises_and_names_what_it_tried(tmp_path):
    job = {"assets": [{"name": "base", "b2": "base-models/nope", "dest": "assets/nope"}]}
    with pytest.raises(ls.SmokeError) as e:
        ls.resolve_assets(job, {}, {}, repo_root=str(tmp_path))
    assert "base-models/nope" in str(e.value)
    assert "--asset base=" in str(e.value)


def test_absolute_asset_dest_is_refused():
    """jobd refuses a dest outside its root; the local lane must agree, or a
    bundle that cannot work on a box would smoke green."""
    job = {"assets": [{"name": "base", "b2": "base-models/x", "dest": "/workspace/x"}]}
    with pytest.raises(ls.SmokeError):
        ls.resolve_assets(job, {"assets": {"base": "/tmp"}}, {})


# ----------------------------------------------------------------- bootstrap

def test_bootstrap_runs_ldconfig_after_symlinks_and_execs_entrypoint():
    """ldconfig is the load-bearing line (doc 119): without it triton JIT dies
    even though torch imports fine."""
    script = ls.bootstrap_script(
        {"libcuda.so.1": "/usr/lib/libcuda.so.610.43.03",
         "libnvidia-ml.so.1": "/usr/lib/libnvidia-ml.so.610.43.03"}, "run.sh")
    lines = [ln.strip() for ln in script.splitlines() if ln.strip()]
    assert "ldconfig" in lines
    assert lines.index("ldconfig") > lines.index(
        "ln -sf libcuda.so.610.43.03 /usr/lib/nvhost/libcuda.so.1")
    assert lines[-1] == "exec bash run.sh"
    assert f"echo {ls.CTR_LIB_DIR} > /etc/ld.so.conf.d/nvhost.conf" in lines


def test_bootstrap_uses_the_versioned_file_not_the_symlink():
    script = ls.bootstrap_script({"libcuda.so.1": "/usr/lib/libcuda.so.610.43.03"}, "e.sh")
    assert "ln -sf libcuda.so.610.43.03 /usr/lib/nvhost/libcuda.so.1" in script


def test_bootstrap_is_bare_exec_on_the_cdi_path():
    """CDI's createContainer hook already updates the ld cache — verified
    in-container: ctypes.CDLL('libcuda.so') resolves. Nothing to bootstrap."""
    assert ls.bootstrap_script({}, "run.sh") == "exec bash run.sh"
    assert "ldconfig" not in ls.bootstrap_script({}, "run.sh")


# ------------------------------------------------------------------------ CDI

def test_detect_cdi_reads_the_spec_files_not_the_binary(tmp_path):
    """podman resolves the device name against these files, so a host with
    nvidia-ctk installed but no generated spec must fall back to manual."""
    d = tmp_path / "cdi"
    d.mkdir()
    assert ls.detect_cdi((str(d),)) is False
    (d / "nvidia.yaml").write_text("cdiVersion: 0.6.0\nkind: nvidia.com/gpu\n")
    assert ls.detect_cdi((str(d),)) is True


def test_detect_cdi_ignores_unrelated_specs(tmp_path):
    d = tmp_path / "cdi"
    d.mkdir()
    (d / "other.yaml").write_text("kind: example.com/fpga\n")
    assert ls.detect_cdi((str(d),)) is False


def test_detect_cdi_tolerates_missing_dirs():
    assert ls.detect_cdi(("/nonexistent-cdi-dir",)) is False


def test_cdi_device_args_name_each_card():
    """`=all` would hand over every card and make --gpus a lie."""
    assert ls.cdi_device_args([0]) == ["--device", "nvidia.com/gpu=0"]
    assert ls.cdi_device_args([0, 1]) == [
        "--device", "nvidia.com/gpu=0", "--device", "nvidia.com/gpu=1"]
    assert "nvidia.com/gpu=all" not in ls.cdi_device_args([0, 1])


def _prov_fixture(tmp_path, staged_text, truth_text):
    staged_dir = tmp_path / "_build"
    staged_dir.mkdir()
    (staged_dir / "train.py").write_text(staged_text)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "train.py").write_text(truth_text)
    job = {"assets": [{"name": "runset", "b2": "runsets/x", "dest": "assets/x",
                       "tracks": {"train.py": "src/train.py"}}]}
    assets = {"runset": (str(staged_dir), "assets/x")}
    return job, assets


def test_provenance_flags_a_staged_copy_that_drifted(tmp_path):
    """The real defect: _build lagged the source of truth by 109 lines while B2
    matched it, so the smoke would have certified code no box runs."""
    job, assets = _prov_fixture(tmp_path, "old\n", "new\n")
    rows = ls.check_asset_provenance(job, assets, repo_root=str(tmp_path))
    assert [r[4] for r in rows] == ["drift"]


def test_provenance_matches_when_staged_equals_truth(tmp_path):
    job, assets = _prov_fixture(tmp_path, "same\n", "same\n")
    rows = ls.check_asset_provenance(job, assets, repo_root=str(tmp_path))
    assert [r[4] for r in rows] == ["match"]


def test_provenance_reports_rather_than_raises_on_missing_files(tmp_path):
    """A repo without the tracked file must not become an un-runnable smoke."""
    job, assets = _prov_fixture(tmp_path, "x\n", "x\n")
    (tmp_path / "src" / "train.py").unlink()
    assert [r[4] for r in ls.check_asset_provenance(
        job, assets, repo_root=str(tmp_path))] == ["missing-truth"]
    (tmp_path / "_build" / "train.py").unlink()
    assert [r[4] for r in ls.check_asset_provenance(
        job, assets, repo_root=str(tmp_path))] == ["missing-staged"]


def test_provenance_ignores_assets_without_tracks(tmp_path):
    """`tracks:` is optional — the base model declares none and must not warn."""
    job = {"assets": [{"name": "base", "b2": "base-models/x", "dest": "assets/x"}]}
    assert ls.check_asset_provenance(job, {"base": (str(tmp_path), "assets/x")},
                                     repo_root=str(tmp_path)) == []


def test_width_pin_refuses_a_narrower_smoke():
    """run.sh exits 13 on this; catching it here saves a container start."""
    with pytest.raises(ls.SmokeError) as e:
        ls.check_width_pin({"EXPECT_GPU_COUNT": "2", "JOB_GPU_COUNT": "1"})
    msg = str(e.value)
    assert "smoke.yaml" in msg and "exit 13" in msg
    assert "job-config.yaml" in msg          # names it only to say DON'T edit it


def test_width_pin_passes_when_satisfied_or_absent():
    """Silence is the common case: matching width, no pin, or the CPU carve-out."""
    ls.check_width_pin({"EXPECT_GPU_COUNT": "2", "JOB_GPU_COUNT": "2"})
    ls.check_width_pin({"JOB_GPU_COUNT": "1"})                   # unpinned bundle
    ls.check_width_pin({"EXPECT_GPU_COUNT": "2"})                # --no-gpu
    ls.check_width_pin({"EXPECT_GPU_COUNT": "2", "JOB_GPU_COUNT": "0"})  # rehearsal


def test_cdi_argv_mounts_no_driver_libs():
    argv = ls.podman_argv(image="img", workdir="/w", job_mounts=[], env={},
                          devices=ls.cdi_device_args([0]), libs={},
                          entrypoint="run.sh", name="s")
    assert not any("nvhost" in a for a in argv)
    assert argv[-1] == "exec bash run.sh"


# --------------------------------------------------------------- podman argv

def _argv(**kw):
    base = dict(image="img", workdir="/w", job_mounts=[("/host/base", "assets/b")],
                env={"QUANT": "8bit"}, devices=["--device", "/dev/nvidia0"],
                libs={"libcuda.so.1": "/usr/lib/libcuda.so.610.1"},
                entrypoint="run.sh", name="smoke-x")
    base.update(kw)
    return ls.podman_argv(**base)


def test_argv_mounts_workdir_rw_and_assets_ro():
    argv = _argv()
    assert "-v" in argv and "/w:/job" in argv
    assert "/host/base:/job/assets/b:ro" in argv
    assert "/usr/lib/libcuda.so.610.1:/usr/lib/nvhost/libcuda.so.610.1:ro" in argv


def test_argv_sets_shm_size_because_podman_defaults_to_64m():
    assert "--shm-size" in _argv()
    assert _argv(shm="16g")[_argv(shm="16g").index("--shm-size") + 1] == "16g"


def test_argv_passes_env_and_ends_with_the_bootstrap():
    argv = _argv()
    assert "QUANT=8bit" in argv
    assert argv[-3:-1] == ["bash", "-lc"]
    assert argv[-1].endswith("exec bash run.sh")


# ------------------------------------------------------------------- probes

SMI = "0, NVIDIA GeForce RTX 3090, 24576, 1\n1, NVIDIA GeForce RTX 3090, 24576, 20480\n"


def test_probe_gpus_parses_and_floors_ram_to_gib():
    gpus = ls.probe_gpus(SMI)
    assert [g["index"] for g in gpus] == [0, 1]
    assert gpus[0]["ram_gb"] == 24          # 24576 MiB -> 24, matching gpu_ram_gb
    assert gpus[1]["used_mb"] == 20480      # what the busy-card guard reads


def test_probe_gpus_raises_on_empty():
    with pytest.raises(ls.SmokeError):
        ls.probe_gpus("\n")


def test_find_driver_libs_resolves_from_ldconfig():
    out = ("\tlibcuda.so.1 (libc6,x86-64) => /usr/lib/libcuda.so.1\n"
           "\tlibnvidia-ml.so.1 (libc6,x86-64) => /usr/lib/libnvidia-ml.so.1\n")
    libs = ls.find_driver_libs(out)
    assert set(libs) == {"libcuda.so.1", "libnvidia-ml.so.1"}


def test_find_driver_libs_raises_when_absent(monkeypatch):
    monkeypatch.setattr(ls.glob, "glob", lambda *a, **k: [])
    with pytest.raises(ls.SmokeError):
        ls.find_driver_libs("")


# ------------------------------------------------------------------- workdir

def test_stage_workdir_copies_bundle_but_binds_data(tmp_path):
    job = tmp_path / "job"
    (job / "data").mkdir(parents=True)
    (job / "out").mkdir()
    (job / "run.sh").write_text("#!/bin/sh\n")
    (job / "data" / "corpus.jsonl").write_text("{}\n")
    (job / "out" / "stale-adapter.safetensors").write_text("x")
    wd = tmp_path / "wd"
    mounts = ls.stage_workdir(str(job), str(wd))
    assert (wd / "run.sh").is_file()
    assert not (wd / "data").exists()               # bind-mounted, not copied
    assert mounts == [(str(job / "data"), "data")]
    assert (wd / "out").is_dir()
    # a previous run's out/ must NOT ride along: run.sh purges DRY_RUN manifests
    # for the same reason, and a stale adapter would pass the post-train gate.
    assert not (wd / "out" / "stale-adapter.safetensors").exists()


def test_prune_weights_keeps_evidence(tmp_path):
    out = tmp_path / "out"
    (out / "checkpoint-10").mkdir(parents=True)
    (out / "adapter_model.safetensors").write_bytes(b"0" * 2048)
    (out / "checkpoint-10" / "optimizer.pt").write_bytes(b"0" * 4096)
    (out / "train.log").write_text("log")
    (out / "train_summary.json").write_text("{}")
    freed = ls.prune_weights(str(out))
    assert freed >= 6144
    assert (out / "train.log").is_file()
    assert (out / "train_summary.json").is_file()
    assert not (out / "adapter_model.safetensors").exists()
    assert not (out / "checkpoint-10").exists()


# --------------------------------------------------------------- smoke config

def test_load_smoke_config_absent_is_empty(tmp_path):
    assert ls.load_smoke_config(str(tmp_path / "nope.yaml")) == {}


def test_load_smoke_config_parses_without_pyyaml(tmp_path):
    """The bundle schema must survive jobmeta's stdlib fallback parser: one
    level of nesting only, which is why smoke config is its own FILE rather
    than a `smoke:` block inside job-config.yaml."""
    p = tmp_path / "smoke.yaml"
    p.write_text("version: 1\ntimeout_s: 900\n"
                 "assets:\n  base: ~/base-models/x\n"
                 "env:\n  QUANT: \"8bit\"\n  STEPS: \"10\"\n")
    cfg = ls.load_smoke_config(str(p))
    assert cfg["timeout_s"] == 900
    assert cfg["env"]["QUANT"] == "8bit"
    assert cfg["assets"]["base"] == "~/base-models/x"


def test_load_smoke_config_rejects_non_map_env(tmp_path):
    p = tmp_path / "smoke.yaml"
    p.write_text("env:\n  - QUANT=8bit\n")
    with pytest.raises(ls.SmokeError):
        ls.load_smoke_config(str(p))
