"""Tests for the LOCAL GPU LANE (joblocal.py + `herdd job --local/run-local`).

Two layers, mirroring the portable-lane discipline:

* PURE — identity/paths/env/asset-map/GPU-allow parsing and the flock liveness
  probe. No subprocess, no hardware.
* END-TO-END — real `onstart/jobd.sh` driven through the real local transport
  (`testlib/rclone_shim.sh` over a tmpdir bucket), proving the properties the
  lane exists to give: a bundle runs, its results globs are HONORED (not the
  whole workdir), a killed run RESUMES from its checkpoint under the same
  JOB_ID, a pre-seeded local asset stages with NO copy, and an unseeded one
  fails pre-entrypoint exactly as an empty B2 prefix does.

GPU scheduling is asserted under `JOBD_FAKE_GPUS` so the suite stays portable —
a real 2x3090 run is exercised by hand (see LOCAL_GPU_LANE.md / the commit).
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblocal  # noqa: E402
import jobmeta as jm  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
HERDD = os.path.join(_HERE, "herdd.py")

shell = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("timeout")),
    reason="needs bash + timeout")

# Same allowlist discipline as test_jobd.py: never `dict(os.environ)`, which
# would hand a shelled jobd the repo `.env`'s real B2 key and the real $HOME
# (jobd's park-key fallback reads ~/.vast_api_key).
_PASSTHROUGH = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM")


def _env(tmp_path, **over):
    env = {k: os.environ[k] for k in _PASSTHROUGH if k in os.environ}
    env.setdefault("PATH", os.defpath)
    home = tmp_path / "fakehome"
    home.mkdir(exist_ok=True)
    env["HOME"] = str(home)
    env["JOBLOCAL_HOME"] = str(tmp_path / "joblocal")
    # Owner ruling 2026-08-06 (CLAUDE.md "GPU jobs (vast.ai)"): `job run-local`
    # refuses to touch the local GPU unless authorized (vastconf.
    # require_local_gpu). Every `run-local` test in this file drives a fake/
    # CPU-only bundle under JOBD_FAKE_GPUS and never probes a real card (see
    # joblocal.probe_gpus/foreign_gpu_procs, which both honor JOBD_FAKE_GPUS
    # and skip nvidia-smi entirely) — so authorizing here is honest, not a
    # bypass of the policy. test_run_local_refuses_without_local_gpu_authorization
    # below explicitly withholds this to prove the gate still fires.
    env["HERDD_ALLOW_LOCAL_GPU"] = "1"
    env.update(over)
    return env


# --------------------------------------------------------------------------- #
# pure
# --------------------------------------------------------------------------- #
def test_local_box_id_is_prefixed_and_sanitized():
    assert joblocal.local_box_id("Example-Rig") == "local-example-rig"
    assert joblocal.local_box_id("box.lan_01") == "local-box-lan-01"
    assert joblocal.local_box_id("") == "local-host"
    # used raw as a bucket path segment / queue prefix
    assert "/" not in joblocal.local_box_id("a/b")


def test_local_home_honors_env(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBLOCAL_HOME", str(tmp_path / "x"))
    assert joblocal.local_home() == str(tmp_path / "x")
    monkeypatch.delenv("JOBLOCAL_HOME")
    assert joblocal.local_home().endswith("/upstream-monorepo/joblocal")
    # NEVER inside the repo: a machine path must not become committable.
    assert _HERE not in joblocal.local_home()


def test_transport_env_points_at_local_bucket_and_drops_b2_creds(tmp_path):
    home = str(tmp_path / "jl")
    env = joblocal.transport_env(home, base_env={
        "PATH": "/usr/bin", "B2_KEY_ID": "secret", "B2_APPLICATION_KEY": "s",
        "B2_BUCKET": "the-real-bucket"})
    assert env["PATH"].startswith(joblocal.bin_dir(home) + os.pathsep)
    assert env["FAKE_BUCKET"] == joblocal.bucket_dir(home)
    assert env["B2_BUCKET"] == joblocal.LOCAL_BUCKET_NAME != "the-real-bucket"
    for k in ("B2_KEY_ID", "B2_APPLICATION_KEY"):
        assert k not in env, "the local lane must never carry a B2 credential"


def test_jobd_env_disables_every_box_only_behavior(tmp_path):
    env = joblocal.jobd_env(str(tmp_path / "jl"), gpu_allow=[1], base_env={"PATH": "/usr/bin"})
    assert env["JOBD_IDLE_PARK"] == "0"      # no box to park
    assert env["JOBD_GPU_REAP"] == "0"       # would kill the operator's own procs
    assert env["JOBD_SKIP_B2CONFIG"] == "1"  # the remote IS the shim
    assert env["JOBD_GPU_ALLOW"] == "1"
    assert env["JOBD_ONCE"] == "1"
    assert env["JOBD_IID"] == joblocal.local_box_id()
    # crucially NOT set: the GPU probe must be REAL (this is the GPU lane)
    assert "JOBD_SKIP_GPU" not in env and "JOBD_FAKE_GPUS" not in env


def test_jobd_env_watch_mode_drops_once(tmp_path):
    env = joblocal.jobd_env(str(tmp_path / "jl"), once=False, base_env={"PATH": "/usr/bin"})
    assert "JOBD_ONCE" not in env


def test_parse_gpu_allow():
    assert joblocal.parse_gpu_allow("0,1") == [0, 1]
    assert joblocal.parse_gpu_allow(" 2 ") == [2]
    assert joblocal.parse_gpu_allow(None) == [] == joblocal.parse_gpu_allow("")
    with pytest.raises(joblocal.JoblocalError):
        joblocal.parse_gpu_allow("cuda:0")


def test_asset_map_roundtrip_is_atomic(tmp_path):
    home = str(tmp_path / "jl")
    os.makedirs(home)
    joblocal.save_asset_map({"base": "/models/qwen", "adapter": "/ckpt/a"}, home)
    assert joblocal.load_asset_map(home) == {"base": "/models/qwen", "adapter": "/ckpt/a"}
    assert not os.path.exists(joblocal.asset_map_path(home) + ".tmp")
    assert joblocal.load_asset_map(str(tmp_path / "nope")) == {}


def test_parse_asset_arg(tmp_path):
    d = tmp_path / "weights"
    d.mkdir()
    assert joblocal.parse_asset_arg(f"base={d}") == ("base", str(d))
    for bad in ("base", "=/tmp", f"base={tmp_path}/missing"):
        with pytest.raises(joblocal.JoblocalError):
            joblocal.parse_asset_arg(bad)


def test_seed_asset_symlinks_and_marks(tmp_path):
    root = tmp_path / "ws"
    src = tmp_path / "models"
    src.mkdir()
    (src / "config.json").write_text("{}")
    cache = joblocal.seed_asset(str(root), "base", str(src))
    assert os.path.islink(cache) and os.path.realpath(cache) == str(src)
    marker = root / "assets" / ".base.local"
    assert marker.is_file(), "jobd keys the skip-the-pull decision on this marker"
    # re-seeding is idempotent (a later run passes --asset again)
    joblocal.seed_asset(str(root), "base", str(src))
    assert os.path.realpath(cache) == str(src)


def test_seed_asset_refuses_to_clobber_a_real_pulled_cache(tmp_path):
    root = tmp_path / "ws"
    (root / "assets" / "base").mkdir(parents=True)
    (root / "assets" / "base" / "pulled.bin").write_text("x")
    with pytest.raises(joblocal.JoblocalError):
        joblocal.seed_asset(str(root), "base", str(tmp_path))


def test_daemon_liveness_probe_reads_the_jobd_flock(tmp_path):
    """`live_boxes` replaces the vast API for a local fold. It must be a REAL
    probe: no lock file / an unheld lock => not live; a held lock => live."""
    root = tmp_path / "ws"
    root.mkdir()
    assert joblocal.daemon_running(str(root)) is False       # no lock file yet
    lock = root / ".jobd.lock"
    lock.write_text("")
    assert joblocal.daemon_running(str(root)) is False       # exists, unheld
    import fcntl
    fd = os.open(str(lock), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert joblocal.daemon_running(str(root)) is True
    finally:
        os.close(fd)
    assert joblocal.daemon_running(str(root)) is False


def test_install_shim_is_the_shared_one(tmp_path):
    p = joblocal.install_shim(str(tmp_path / "jl"))
    assert os.access(p, os.X_OK)
    with open(p) as a, open(joblocal.SHIM_SRC) as b:
        assert a.read() == b.read(), \
            "the local lane's correctness rests on laptop and box running the SAME transport"


def test_differences_banner_names_the_real_gaps():
    b = joblocal.differences_banner().lower()
    for must in ("preemption", "self-park", "reap", "local disk", "rehearse.sh --image"):
        assert must in b


# --------------------------------------------------------------------------- #
# end-to-end: the real jobd, the real transport
# --------------------------------------------------------------------------- #
def _write_job(d, *, name, body, results='  - "out/**"\n', extra="",
               needs="  gpu: false\n  venv: none\n"):
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.sh").write_text(body)
    (d / "job-config.yaml").write_text(
        f"version: 1\nname: {name}\nentrypoint: run.sh\ntimeout_s: 120\n"
        f"{extra}results:\n{results}needs:\n{needs}")
    return str(d)


def _herdd(tmp_path, *args, expect_rc=0, timeout=180, **envover):
    p = subprocess.run([sys.executable, HERDD, *args], capture_output=True,
                       text=True, timeout=timeout, env=_env(tmp_path, **envover))
    if expect_rc is not None:
        assert p.returncode == expect_rc, f"rc={p.returncode}\n{p.stdout}\n{p.stderr}"
    return p


def _job_id(out):
    for line in out.splitlines():
        if line.startswith(">> JOB_ID="):
            return line.split("=", 1)[1].strip()
    raise AssertionError(f"no JOB_ID in:\n{out}")


def _bucket(tmp_path):
    return os.path.join(str(tmp_path / "joblocal"), "bucket")


@shell
def test_submit_local_queues_onto_the_local_box_and_never_touches_b2(tmp_path):
    src = _write_job(tmp_path / "j", name="hello", body="#!/bin/bash\nmkdir -p out\necho hi > out/a.txt\n")
    p = _herdd(tmp_path, "job", "submit", src, "--local")
    jid = _job_id(p.stdout)
    ticket = os.path.join(_bucket(tmp_path), "jobs", "queue",
                          joblocal.local_box_id(), f"{jid}.json")
    assert os.path.isfile(ticket)
    assert json.load(open(ticket))["box"] == joblocal.local_box_id()
    # the submit hint must point at the local executor, not `job attach`
    assert "job run-local" in p.stdout and "job attach" not in p.stdout


@shell
def test_submit_rejects_box_and_local_together(tmp_path):
    src = _write_job(tmp_path / "j", name="hello", body="#!/bin/bash\ntrue\n")
    p = _herdd(tmp_path, "job", "submit", src, "--local", "--box", "12345",
                 expect_rc=None)
    assert p.returncode != 0 and "mutually exclusive" in (p.stdout + p.stderr)


@shell
def test_submit_without_box_or_local_says_how_to_run_locally(tmp_path):
    src = _write_job(tmp_path / "j", name="hello", body="#!/bin/bash\ntrue\n")
    p = _herdd(tmp_path, "job", "submit", src, expect_rc=None)
    assert p.returncode != 0 and "--local" in (p.stdout + p.stderr)


@shell
def test_run_local_end_to_end_honors_the_results_globs(tmp_path):
    """The core promise: SAME config, SAME results globs. The workdir file that
    NO glob selects must NOT reach the bucket — for years the local shim dropped
    rclone's --include and copied the whole workdir, which made this assertion
    (and rehearse.sh's) vacuous and would have shipped GB of intermediates."""
    src = _write_job(
        tmp_path / "j", name="globs",
        body=("#!/bin/bash\nset -e\nmkdir -p out/deep\n"
              "echo keep > out/a.txt\necho keep > out/deep/b.txt\n"
              "echo DROP > scratch.bin\necho DROP > out_of_band.log\n"))
    p = _herdd(tmp_path, "job", "run-local", src, timeout=300,
                 JOBD_FAKE_GPUS="0:24,1:24")
    jid = _job_id(p.stdout)
    res = os.path.join(_bucket(tmp_path), "jobs", jid, "results")
    got = sorted(os.path.relpath(os.path.join(r, f), res)
                 for r, _, fs in os.walk(res) for f in fs)
    assert got == ["out/a.txt", "out/deep/b.txt"], got
    assert os.path.isfile(os.path.join(_bucket(tmp_path), "jobs", jid,
                                       "results.DONE.json"))
    assert "status=done" in p.stdout


@shell
def test_run_local_prints_the_differences_banner(tmp_path):
    src = _write_job(tmp_path / "j", name="banner", body="#!/bin/bash\nmkdir -p out\ntouch out/x\n")
    p = _herdd(tmp_path, "job", "run-local", src, timeout=300,
                 JOBD_FAKE_GPUS="0:24")
    assert "LOCAL LANE" in p.stdout and "no spot preemption" in p.stdout


@shell
def test_run_local_assigns_needs_gpus_via_cuda_visible_devices(tmp_path):
    """`needs.gpus` -> CUDA_VISIBLE_DEVICES + JOB_GPU_COUNT/JOB_GPUS, the same
    scheduling a box does. JOB_GPUS is asserted because it was SHADOWED by the
    daemon's associative array and silently never exported until 2026-07-30."""
    src = _write_job(
        tmp_path / "j", name="cards",
        body=("#!/bin/bash\nset -eu\nmkdir -p out\n"
              'echo "cvd=$CUDA_VISIBLE_DEVICES n=$JOB_GPU_COUNT g=$JOB_GPUS '
              'ram=$JOB_GPU_RAM_GB" > out/env.txt\n'),
        needs="  gpu: true\n  gpus: 2\n  gpu_ram_gb: 20\n  venv: none\n")
    p = _herdd(tmp_path, "job", "run-local", src, timeout=300,
                 JOBD_FAKE_GPUS="0:24,1:24,2:8")
    jid = _job_id(p.stdout)
    txt = open(os.path.join(_bucket(tmp_path), "jobs", jid,
                            "results", "out", "env.txt")).read()
    # cards 0,1 fit the 20 GB floor; card 2 (8 GB) must not be assigned
    assert "cvd=0,1" in txt and "n=2" in txt and "g=0,1" in txt and "ram=24" in txt, txt


@shell
def test_run_local_resumes_the_same_job_from_its_checkpoint(tmp_path):
    """Interruption tolerance is REAL locally, not simulated: kill the runner,
    re-run `run-local`, and jobd takes its resume path — same JOB_ID, checkpoint
    pulled back into the workdir, JOB_RESTART_COUNT bumped."""
    src = _write_job(
        tmp_path / "j", name="resume",
        body=("#!/bin/bash\nset -eu\nmkdir -p out/state\n"
              "S=0; [ -f out/state/step ] && S=$(cat out/state/step)\n"
              'echo "attempt from $S" > out/trace.txt\n'
              "while [ \"$S\" -lt 6 ]; do S=$((S+1)); echo $S > out/state/step; sleep 1; done\n"
              "echo done > out/final.txt\n"),
        extra='checkpoint_s: 1\ncheckpoints:\n  - "out/state/**"\nmax_restarts: 4\n')
    jid = _job_id(_herdd(tmp_path, "job", "submit", src, "--local").stdout)

    # pass 1: kill mid-flight (SIGTERM -> jobd's preempt trap, final flush)
    p1 = subprocess.run(["timeout", "-s", "TERM", "4", sys.executable, HERDD,
                         "job", "run-local"], capture_output=True, text=True,
                        timeout=120, env=_env(tmp_path, JOBD_FAKE_GPUS="0:24"))
    assert "start entrypoint" in p1.stdout + p1.stderr

    v = json.loads(_herdd(tmp_path, "job", "status", jid, "--local", "--json").stdout)
    assert v["display_status"] == "interrupted", v
    assert v["status"] != "failed", "an interrupted local job must stay resumable"

    # pass 2: same JOB_ID resumes from the checkpoint and finishes
    p2 = _herdd(tmp_path, "job", "run-local", timeout=300, JOBD_FAKE_GPUS="0:24")
    assert "resume" in (p2.stdout + p2.stderr).lower()
    v = json.loads(_herdd(tmp_path, "job", "status", jid, "--local", "--json").stdout)
    assert v["status"] == "done" and v["rc"] == 0, v
    assert v["attempts"] >= 2, v
    # The final publish carries attempt 2's workdir, and jobd's pull-back restores
    # the `checkpoints:` globs only — so trace.txt records where attempt 2 STARTED.
    # Anything but 0 means the checkpoint really came back and the driver
    # continued from it rather than restarting the run.
    trace = open(os.path.join(_bucket(tmp_path), "jobs", jid, "results",
                              "out", "trace.txt")).read().strip()
    assert trace != "attempt from 0", \
        "the second attempt restarted from ZERO — the checkpoint pull-back was wasted"
    assert trace.startswith("attempt from "), trace


@shell
def test_local_asset_override_stages_without_copying(tmp_path):
    """A pre-seeded local asset must be used IN PLACE (no multi-GB copy) and must
    NOT be pulled over — the cache is a symlink into the operator's model dir and
    an rclone copy would write straight through it."""
    weights = tmp_path / "models" / "base"
    weights.mkdir(parents=True)
    (weights / "config.json").write_text("{}")
    (weights / "model.safetensors").write_text("W" * 64)
    src = _write_job(
        tmp_path / "j", name="assetjob",
        body=("#!/bin/bash\nset -eu\nmkdir -p out\n"
              'cat base/config.json > out/seen.json\n'
              'readlink -f base > out/where.txt\n'),
        extra=('assets:\n  - name: base\n    b2: base-models/qwen\n'
               '    dest: base\n    require:\n      - config.json\n'
               '      - "*.safetensors"\n'))
    p = _herdd(tmp_path, "job", "run-local", src, f"--asset=base={weights}",
                 timeout=300, JOBD_FAKE_GPUS="0:24")
    jid = _job_id(p.stdout)
    res = os.path.join(_bucket(tmp_path), "jobs", jid, "results")
    assert open(os.path.join(res, "out", "seen.json")).read().strip() == "{}"
    assert open(os.path.join(res, "out", "where.txt")).read().strip() == str(weights)
    # NO copy happened: the cache is still a symlink at the source
    cache = os.path.join(str(tmp_path / "joblocal"), "workspace", "assets", "base")
    assert os.path.islink(cache) and os.path.realpath(cache) == str(weights)
    # and the source tree was not written through
    assert sorted(os.listdir(weights)) == ["config.json", "model.safetensors"]
    # the override is remembered for the next run
    assert joblocal.load_asset_map(str(tmp_path / "joblocal")) == {"base": str(weights)}


@shell
def test_unseeded_asset_fails_pre_entrypoint_like_an_empty_b2_prefix(tmp_path):
    src = _write_job(
        tmp_path / "j", name="missingasset",
        body="#!/bin/bash\nmkdir -p out\ntouch out/x\n",
        extra=('assets:\n  - name: base\n    b2: base-models/qwen\n'
               '    require:\n      - config.json\n'))
    p = _herdd(tmp_path, "job", "run-local", src, timeout=300, expect_rc=None,
                 JOBD_FAKE_GPUS="0:24")
    jid = _job_id(p.stdout)
    v = json.loads(_herdd(tmp_path, "job", "status", jid, "--local", "--json").stdout)
    assert v["status"] == "failed", v
    assert v["fail_reason"] == "asset_stage_failed:base", v
    # PRE-entrypoint: no `started` event, so nothing half-staged ever ran
    assert v["attempts"] == 0, v
    assert "no local override" in p.stderr


@shell
def test_pre_seeded_asset_still_enforces_require_globs(tmp_path):
    """Skipping the pull must NOT skip integrity: a wrong local path fails in
    exactly the place a truncated B2 pull does."""
    bad = tmp_path / "models" / "empty"
    bad.mkdir(parents=True)
    (bad / "README").write_text("not a model")
    src = _write_job(
        tmp_path / "j", name="badasset",
        body="#!/bin/bash\nmkdir -p out\ntouch out/x\n",
        extra=('assets:\n  - name: base\n    b2: base-models/qwen\n'
               '    require:\n      - config.json\n'))
    p = _herdd(tmp_path, "job", "run-local", src, f"--asset=base={bad}",
                 timeout=300, expect_rc=None, JOBD_FAKE_GPUS="0:24")
    v = json.loads(_herdd(tmp_path, "job", "status", _job_id(p.stdout),
                            "--local", "--json").stdout)
    assert v["status"] == "failed" and v["fail_reason"] == "asset_stage_failed:base", v


@shell
def test_ls_and_pull_local(tmp_path):
    src = _write_job(tmp_path / "j", name="lsjob",
                     body="#!/bin/bash\nmkdir -p out\necho R > out/r.txt\n")
    jid = _job_id(_herdd(tmp_path, "job", "run-local", src, timeout=300,
                           JOBD_FAKE_GPUS="0:24").stdout)
    ls = _herdd(tmp_path, "job", "ls", "--local").stdout
    assert jid in ls and joblocal.local_box_id() in ls
    dest = tmp_path / "pulled"
    out = _herdd(tmp_path, "job", "pull", jid, str(dest), "--local").stdout
    assert "out/r.txt" in out and (dest / "out" / "r.txt").read_text().strip() == "R"


@shell
def test_local_and_remote_buckets_cannot_see_each_other(tmp_path):
    """The separation is physical (a different filesystem root), not cosmetic."""
    src = _write_job(tmp_path / "j", name="isolated",
                     body="#!/bin/bash\nmkdir -p out\ntouch out/x\n")
    jid = _job_id(_herdd(tmp_path, "job", "submit", src, "--local").stdout)
    # every object this job produced lives under the local root, nowhere else
    root = str(tmp_path / "joblocal")
    hits = [os.path.join(r, f)
            for r, _, fs in os.walk(str(tmp_path)) for f in fs if jid in f]
    assert hits and all(h.startswith(root) for h in hits), hits


@shell
def test_run_local_refuses_when_no_gpu_is_visible(tmp_path, monkeypatch):
    """This is the GPU lane; a CPU-only box must be sent to rehearse.sh, not
    silently run without hardware."""
    fakebin = tmp_path / "nogpu"
    fakebin.mkdir()
    (fakebin / "nvidia-smi").write_text("#!/bin/sh\nexit 1\n")
    os.chmod(fakebin / "nvidia-smi", 0o755)
    src = _write_job(tmp_path / "j", name="nogpu", body="#!/bin/bash\ntrue\n")
    p = _herdd(tmp_path, "job", "run-local", src, expect_rc=None,
                 PATH=str(fakebin) + os.pathsep + os.environ.get("PATH", os.defpath))
    assert p.returncode != 0
    assert "rehearse.sh" in (p.stdout + p.stderr)


@shell
def test_run_local_refuses_without_local_gpu_authorization(tmp_path):
    """A gate nobody tests is a gate that can silently stop working. Every other
    `run-local` test in this file authorizes the local-GPU gate
    (vastconf.require_local_gpu, CLAUDE.md "GPU jobs (vast.ai)") through
    `_env()`'s HERDD_ALLOW_LOCAL_GPU=1 default, so none of them would notice if
    the gate silently stopped firing.

    It withholds authorization with `=0`, NOT by unsetting the variable. That
    distinction became load-bearing on 2026-08-11, when the owner reversed the
    2026-08-06 ban and `allow_local_gpu` shipped `true`: with the var merely
    absent, this test would now fall through to the repo config, find the
    authorization there, and pass by running the lane it exists to prove is
    refusable. `=0` closes the lane regardless of what the config says, which is
    also the documented one-off for proving a bundle has no local fallback."""
    src = _write_job(tmp_path / "j", name="gated", body="#!/bin/bash\ntrue\n")
    env = _env(tmp_path)
    env["HERDD_ALLOW_LOCAL_GPU"] = "0"
    p = subprocess.run([sys.executable, HERDD, "job", "run-local", src],
                       capture_output=True, text=True, timeout=60, env=env)
    assert p.returncode != 0
    out = p.stdout + p.stderr
    assert "LOCAL GPU" in out and "has not authorized it" in out
    assert "HERDD_ALLOW_LOCAL_GPU=1" in out
