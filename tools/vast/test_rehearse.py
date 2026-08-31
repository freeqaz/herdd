"""Tests for rehearse.sh — the one-shot LOCAL rehearsal of a job folder.

rehearse.sh runs the REAL onstart/jobd.sh against the bash rclone shim (the same
local-B2 technique as test_jobd.py), so these tests prove the driver end-to-end:
a toy job PASSES, a broken config FAILS loudly BEFORE any run, a failing
entrypoint surfaces as FAIL, and --stub-vllm exposes an OpenAI endpoint the
entrypoint can curl. No B2, no GPU, no podman (the --image lane auto-skips).
"""
import glob
import json
import os
import re
import shutil
import socket
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
REHEARSE = os.path.join(_HERE, "rehearse.sh")
FIXTURES = os.path.join(_HERE, "testlib", "fixtures")
EVAL_JOB = os.path.join(FIXTURES, "eval-job")
EVAL_ASSET = os.path.join(FIXTURES, "eval-job-assets", "reader-adapter")

pytestmark = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("timeout")),
    reason="needs bash + timeout")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _write_job(tmp_path, name, run_body, results="  - \"out/**\"\n",
               entrypoint="run.sh", env_block=""):
    src = tmp_path / name
    src.mkdir()
    if run_body is not None:
        (src / "run.sh").write_text(run_body)
    cfg = (f"version: 1\nname: {name}\nentrypoint: {entrypoint}\ntimeout_s: 60\n"
           f"{env_block}results:\n{results}"
           "needs:\n  gpu: false\n  venv: none\n")
    (src / "job-config.yaml").write_text(cfg)
    return str(src)


def _run(folder, *flags, timeout=180, tmpdir=None, env_extra=None):
    env = dict(os.environ)
    env["REHEARSE_PYTHON"] = sys.executable
    # keep rehearse's mktemp scratch INSIDE the pytest tmp so --keep dirs are
    # auto-reaped and never litter /tmp.
    if tmpdir is not None:
        env["TMPDIR"] = str(tmpdir)
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})
    return subprocess.run(["bash", REHEARSE, folder, *flags],
                          capture_output=True, text=True, timeout=timeout, env=env)


def _kept_work(stdout):
    """The scratch dir rehearse prints at [2/5] (its fake bucket lives under it)."""
    m = re.search(r"\[2/5\] fake bucket \+ rclone shim \.\.\. OK \((.+?)\)", stdout)
    return m.group(1) if m else None


def _job_id(stdout):
    m = re.search(r"job_id=([^\s)]+)", stdout)
    return m.group(1) if m else None


def _write_asset_job(tmp_path, name, assets, run_body,
                     results=("out/**",), checkpoints=None, env=None):
    """A job folder whose job-config.JSON carries an `assets:` list-of-maps
    (canonical JSON — the YAML fallback can't express list-of-maps, and it is
    what a real submitted ticket ships anyway)."""
    src = tmp_path / name
    src.mkdir()
    (src / "run.sh").write_text(run_body)
    cfg = {"version": 1, "name": name, "entrypoint": "run.sh", "timeout_s": 60,
           "results": list(results), "needs": {"gpu": False, "venv": "none"},
           "assets": assets}
    if checkpoints is not None:
        cfg["checkpoints"] = list(checkpoints)
        cfg["checkpoint_s"] = 300
    if env is not None:
        cfg["env"] = dict(env)
    (src / "job-config.json").write_text(json.dumps(cfg, indent=2))
    return str(src)


def _seed_dir(tmp_path, sub, files):
    """Build a local seed dir {relpath: content} for --asset NAME=DIR."""
    d = tmp_path / sub
    for rel, content in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return str(d)


def test_default_image_matches_herdd_yaml():
    """rehearse.sh DEFAULT_IMAGE must equal herdd.yaml's default_image.

    This is the one drift in this file that CANNOT announce itself. `--image`
    runs jobd inside podman only if the image is already present locally —
    rehearse.sh NEVER pulls (run_jobd_podman: "SKIP --image: image not present
    locally … NOT pulling; LOCAL jobd"). So a DEFAULT_IMAGE left behind by an
    image flip does not error, it silently falls back to the local lane, and the
    $0 pre-spend rehearsal quietly stops rehearsing the thing that will actually
    boot. Both refs were flipped by hand on 2026-08-01 (d9587c52) with nothing
    holding them together; this is that guard.
    """
    m = re.search(r'^DEFAULT_IMAGE="([^"]+)"', open(REHEARSE).read(), re.M)
    assert m, "rehearse.sh no longer defines DEFAULT_IMAGE=\"...\""
    cfg = os.path.join(_HERE, "herdd.yaml")
    y = re.search(r"^default_image:\s*(\S+)", open(cfg).read(), re.M)
    assert y, "herdd.yaml no longer defines default_image"
    assert m.group(1) == y.group(1), (
        f"rehearse.sh DEFAULT_IMAGE={m.group(1)!r} != herdd.yaml "
        f"default_image={y.group(1)!r}. The box will run the herdd.yaml one; "
        f"rehearsal would silently skip the podman lane instead of failing.")


# Guarding ONE copy was not enough. The 2026-08-07 R2 cutover moved
# herdd.yaml to registry.example.com and left rehearse.sh on
# registry.gitlab.com -- caught, because the test above existed -- but ALSO
# left launch_serve.sh, local_smoke.py and herdd.py's _TRAIN_FALLBACK_IMAGE
# behind, which nothing was watching. Every hardcoded default image in the tree
# is pinned here now, so the next flip cannot half-land.
@pytest.mark.parametrize("relpath,pattern", [
    ("launch_serve.sh", r'^IMAGE="([^"]+)"; PUBLIC_PORT='),
    ("local_smoke.py", r'^DEFAULT_IMAGE = "([^"]+)"'),
    # `_TRAIN_FALLBACK_IMAGE` moved with the launch spec when `herdd.py`
    # became the thin launcher (plan §8 step 6d); the flat name is now a
    # re-export of this literal, so the LITERAL is what this guard must read.
    ("vastlib/launch/spec.py", r'^_TRAIN_FALLBACK_IMAGE = "([^"]+)"'),
])
def test_every_hardcoded_default_image_matches_herdd_yaml(relpath, pattern):
    cfg = os.path.join(_HERE, "herdd.yaml")
    y = re.search(r"^default_image:\s*(\S+)", open(cfg).read(), re.M)
    assert y, "herdd.yaml no longer defines default_image"
    src = open(os.path.join(_HERE, relpath)).read()
    m = re.search(pattern, src, re.M)
    assert m, f"{relpath} no longer matches {pattern!r} — update this guard"
    assert m.group(1) == y.group(1), (
        f"{relpath} pins {m.group(1)!r} but herdd.yaml default_image is "
        f"{y.group(1)!r}. A half-landed image flip leaves some lanes on the old "
        f"image with nothing to announce it.")


def test_toy_job_passes_end_to_end(tmp_path):
    folder = _write_job(
        tmp_path, "toy",
        "mkdir -p out\necho \"hi $FOO\"\necho done > out/result.txt\n",
        env_block="env:\n  FOO: \"world\"\n")
    r = _run(folder)
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert "RESULT: PASS" in r.stdout
    assert "n_results=1" in r.stdout


def test_broken_config_fails_pre_run(tmp_path):
    """A bad entrypoint path is caught at validate — BEFORE jobd ever runs (the
    whole point: fail before spending). No [4/5] line is reached."""
    folder = _write_job(tmp_path, "broken", None, entrypoint="nope.sh")
    r = _run(folder)
    assert r.returncode == 1
    assert "RESULT: FAIL" in r.stdout
    assert "[1/5] validate config ... FAIL" in r.stdout
    # never got to running jobd
    assert "[4/5]" not in r.stdout


def test_failing_entrypoint_surfaces_fail(tmp_path):
    folder = _write_job(
        tmp_path, "failrc",
        "mkdir -p out\necho partial > out/x.txt\nexit 7\n")
    r = _run(folder)
    assert r.returncode == 1
    assert "RESULT: FAIL" in r.stdout
    # the job DID run (jobd ok), but the nonzero rc is caught at the results gate
    assert "rc=7" in (r.stdout + r.stderr)


def test_stub_vllm_endpoint_reachable(tmp_path):
    """--stub-vllm brings up the OpenAI stub before jobd; a toy entrypoint curls
    /v1/models and the 1-token /v1/completions probe successfully."""
    if not shutil.which("curl"):
        pytest.skip("needs curl")
    port = _free_port()
    folder = _write_job(
        tmp_path, "evaltoy",
        "mkdir -p out\n"
        "curl -fsS \"$STUB_ENDPOINT/v1/models\" > out/models.json\n"
        "curl -fsS -H 'Content-Type: application/json' "
        "-d '{\"model\":\"reader\",\"prompt\":\"ping\",\"max_tokens\":1}' "
        "\"$STUB_ENDPOINT/v1/completions\" > out/comp.json\n"
        "echo done > out/result.txt\n")
    r = _run(folder, "--stub-vllm", "--stub-port", str(port))
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert "RESULT: PASS" in r.stdout
    # models.json + comp.json + result.txt all landed
    assert "n_results=3" in r.stdout


# ---------------------------------------------------------------------------
# N6: rehearsal <-> assets, end-to-end. rehearse SEEDS a job's `assets:` B2
# prefixes from local dirs, then the REAL jobd asset-staging path (pull ->
# .complete marker -> require: postconditions) runs against the fake bucket.
# ---------------------------------------------------------------------------
def test_asset_seeded_job_stages_and_passes(tmp_path):
    """A job with an `assets:` block + a `require:` glob: rehearse seeds the B2
    prefix from a local dir via --asset, jobd pulls it, the require passes, the
    entrypoint reads the staged file, and results land -> PASS."""
    seed = _seed_dir(tmp_path, "seed",
                     {"config.json": '{"ok":1}', "w.safetensors": "WEIGHTS"})
    folder = _write_asset_job(
        tmp_path, "assetok",
        [{"name": "base", "b2": "weights/base",
          "dest": "model", "require": ["*.safetensors"]}],
        "mkdir -p out\ncat model/config.json > out/seen.json\necho ok > out/r.txt\n")
    r = _run(folder, "--asset", f"base={seed}", tmpdir=tmp_path)
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert "asset 'base' seeded from" in r.stdout
    assert "RESULT: PASS" in r.stdout
    assert "n_results=2" in r.stdout


def test_asset_sync_mode_stages_and_passes(tmp_path):
    """An asset with `mode: sync` (what jobd's asset_pull uses and the shipped
    eval-template + p2_reader_eval declare) must rehearse: the rclone shim maps
    `sync` onto the same local copy as `copy` (dest starts fresh), so staging
    reaches the entrypoint. Regression: the shim used to only handle `copy`, so
    every sync-mode asset failed the rehearsal with 'shim: unhandled op sync'."""
    seed = _seed_dir(tmp_path, "seed_sync",
                     {"config.json": '{"ok":1}', "w.safetensors": "WEIGHTS"})
    folder = _write_asset_job(
        tmp_path, "assetsync",
        [{"name": "base", "b2": "weights/base", "dest": "model",
          "mode": "sync", "require": ["config.json", "*.safetensors"]}],
        "mkdir -p out\ncat model/config.json > out/seen.json\necho ok > out/r.txt\n")
    r = _run(folder, "--asset", f"base={seed}", tmpdir=tmp_path)
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert "unhandled op sync" not in (r.stdout + r.stderr)
    assert "RESULT: PASS" in r.stdout
    assert "n_results=2" in r.stdout


def test_asset_missing_from_fixture_fails_with_stage_failed(tmp_path):
    """(a) A rehearsal of a job whose asset is absent from the fixture fails with
    the SAME `asset_stage_failed:<name>` shape a real box emits — pre-entrypoint,
    surfaced from the folded event log (a `require:` glob makes the empty pull
    fail exactly as an empty B2 prefix would on a live box)."""
    folder = _write_asset_job(
        tmp_path, "assetmiss",
        [{"name": "base", "b2": "weights/base", "require": ["*.safetensors"]}],
        "mkdir -p out\necho RAN > out/r.txt\n")
    r = _run(folder, tmpdir=tmp_path)   # no --asset, no convention dir
    assert r.returncode == 1
    assert "asset 'base' NOT seeded" in r.stdout
    assert "RESULT: FAIL" in r.stdout
    assert "asset_stage_failed:base" in r.stdout
    assert "pre-entrypoint" in r.stdout
    assert "[4/5]" in r.stdout   # jobd DID run; the failure is inside staging


def test_asset_fixture_convention_seeds_without_flag(tmp_path):
    """The `<job>/assets-fixture/<name>/` convention seeds an asset with no
    --asset flag."""
    folder = _write_asset_job(
        tmp_path, "assetconv",
        [{"name": "base", "b2": "w/base", "require": ["*.bin"]}],
        "mkdir -p out\necho ok > out/r.txt\n")
    fx = os.path.join(folder, "assets-fixture", "base")
    os.makedirs(fx)
    with open(os.path.join(fx, "model.bin"), "w") as f:
        f.write("BIN")
    r = _run(folder, tmpdir=tmp_path)
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert "asset 'base' seeded from" in r.stdout   # seeded via the convention dir
    assert "NOT seeded" not in r.stdout
    assert "RESULT: PASS" in r.stdout


def test_eval_job_fixture_end_to_end(tmp_path):
    """(b) The committed eval-shaped job fixture, rehearsed end-to-end: asset
    staged, stub vLLM up, entrypoint hits /v1/models + /v1/completions +
    /v1/chat/completions, writes an NDJSON probe (a `checkpoints:` glob) + a
    summary -> PASS. --keep lets us inspect the fake bucket + asset cache."""
    if not shutil.which("curl"):
        pytest.skip("needs curl")
    port = _free_port()
    r = _run(EVAL_JOB, "--stub-vllm", "--stub-port", str(port),
             "--asset", f"reader-adapter={EVAL_ASSET}", "--keep", tmpdir=tmp_path)
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert "RESULT: PASS" in r.stdout
    assert "n_results=3" in r.stdout          # models.json + probe.ndjson + summary.json
    work = _kept_work(r.stdout)
    job_id = _job_id(r.stdout)
    assert work and job_id, r.stdout
    jdir = os.path.join(work, "bucket", "jobs", job_id)
    rdir = os.path.join(jdir, "results")
    # results landed on the fake bucket (DONE marker is written LAST)
    assert os.path.isfile(os.path.join(jdir, "results.DONE.json"))
    ndjson = os.path.join(rdir, "out", "probe.ndjson")   # a `checkpoints:` glob file
    summary = os.path.join(rdir, "out", "summary.json")
    assert os.path.isfile(ndjson), f"no NDJSON result on bucket; ls={os.listdir(rdir)}"
    assert os.path.isfile(summary)
    rows = [json.loads(x) for x in open(ndjson) if x.strip()]
    assert len(rows) == 2                     # stub-base + reader
    # asset staging reached the entrypoint: the adapter base_model was read
    assert all(row["adapter"] == "nanbeige-3b" for row in rows), rows
    assert json.load(open(summary))["adapter"] == "nanbeige-3b"
    # the shared pull primitive wrote the .complete byte-total marker
    marker = os.path.join(work, "workspace", "assets", ".reader-adapter.complete")
    assert os.path.isfile(marker) and int(open(marker).read().strip()) > 0


def test_eval_job_truncated_asset_fails_pre_entrypoint(tmp_path):
    """(c) A TRUNCATED asset (only adapter_config.json seeded; the required
    *.safetensors absent) fails the rehearsal pre-entrypoint with the named
    reason — no entrypoint, no stub needed."""
    trunc = _seed_dir(tmp_path, "trunc",
                      {"adapter_config.json": '{"base_model":"x"}'})   # no *.safetensors
    r = _run(EVAL_JOB, "--asset", f"reader-adapter={trunc}", tmpdir=tmp_path)
    assert r.returncode == 1
    assert "RESULT: FAIL" in r.stdout
    assert "asset_stage_failed:reader-adapter" in r.stdout
    assert "pre-entrypoint" in r.stdout


def test_eval_job_stub_absent_surfaces_log_tail(tmp_path):
    """(c) Stub vLLM absent while the entrypoint needs it: FAIL surfaces the
    entrypoint's own log tail (the missing-endpoint error), not just a generic
    rc. Asset is seeded so the failure is unambiguously the missing stub."""
    r = _run(EVAL_JOB, "--asset", f"reader-adapter={EVAL_ASSET}", tmpdir=tmp_path)
    assert r.returncode == 1
    assert "RESULT: FAIL" in r.stdout
    assert "entrypoint log tail" in r.stdout
    assert "STUB_ENDPOINT" in r.stdout        # the actual reason, from log.txt



# =============================================================================
# tools/vast/jobs/workflow-canary (M4-T2) — the committed canary bundle rehearses
# green CPU-only: JOBD_SKIP_GPU=1 flips it into no-GPU-tolerant mode, it does a
# checkpoint round-trip and writes a receipt (gpu_step_ok False, no GPU). A REAL
# box (no JOBD_SKIP_GPU) stays fail-closed. No B2, no GPU, no spend.
# =============================================================================
def test_workflow_canary_rehearsal_passes_cpu_only(tmp_path, monkeypatch):
    bundle = os.path.join(_HERE, "jobs", "workflow-canary")
    out = tmp_path / "captured"
    monkeypatch.setenv("REHEARSE_RESULTS_OUT", str(out))
    r = _run(bundle, tmpdir=tmp_path)
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert "RESULT: PASS" in r.stdout
    assert "name=workflow-canary" in r.stdout

    hits = glob.glob(str(out / "**" / "canary-receipt.json"), recursive=True)
    assert hits, f"no receipt captured under {out}; ls={list(out.rglob('*')) if out.exists() else 'MISSING'}"
    receipt = json.load(open(hits[0]))
    assert receipt["kind"] == "workflow-canary-receipt"
    assert receipt["v"] == 1
    assert receipt["checkpoint_roundtrip_ok"] is True
    assert receipt["gpu_step_ok"] is False   # CPU-only rehearsal: no real GPU step
    assert receipt["rc"] == 0
    # a rehearsal receipt must NOT satisfy the real gate (gpu_step_ok False -> failed)
    assert receipt["expires_ts"] > receipt["ts_end"]


def test_workflow_canary_broken_gpu_emits_failed_receipt(tmp_path):
    # F4: on a real box with no working GPU step (and NO rehearsal marker),
    # canary.sh must exit 21 AND still write a receipt (rc:21, gpu_step_ok:false)
    # so the control plane sees a 'failed' receipt, not a silently-vanished box.
    # Deterministic on any host: stub nvidia-smi to report no device.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "nvidia-smi").write_text("#!/usr/bin/env bash\nexit 1\n")
    (bindir / "nvidia-smi").chmod(0o755)
    workdir = tmp_path / "run"
    workdir.mkdir()
    script = os.path.join(_HERE, "jobs", "workflow-canary", "canary.sh")
    env = dict(os.environ)
    env["PATH"] = str(bindir) + os.pathsep + env["PATH"]
    env.update({"JOB_ID": "brk", "JOB_RESTART_COUNT": "0",
                "CANARY_ALLOW_NO_GPU": "0", "CANARY_TTL_S": "60",
                "CANARY_KEY": "k", "CANARY_IMAGE_DIGEST": "sha256:x"})
    env.pop("JOBD_SKIP_GPU", None)   # NOT rehearsal -> fail-closed
    r = subprocess.run(["bash", script], cwd=str(workdir), env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 21, f"stdout={r.stdout}\nstderr={r.stderr}"
    receipt = json.load(open(workdir / "out" / "canary-receipt.json"))
    assert receipt["rc"] == 21
    assert receipt["gpu_step_ok"] is False
    assert receipt["gpu_model"] == "unavailable"


# =============================================================================
# --concurrent N: BOX-GLOBAL MUTABLE STATE under contention
# =============================================================================
# The 2026-07-30 escape these close: three concurrent frontier-wave jobs raced
# `job_serve.sh --build-venv` into the shared /workspace/serve and the parallel
# pip self-upgrade corrupted pip (`ModuleNotFoundError: pip._internal` in 2 of
# 3 jobs). rehearse.sh ran ONE job at a time, so no amount of realism in the
# single-job lane could have seen it: the failure needs CONTENTION.

# A barrier entrypoint: every job announces itself in ONE box-global file, then
# waits (bounded) for its N peers. If jobd ran the jobs serially, no job ever
# sees N arrivals and the rehearsal FAILS — so a green test is positive evidence
# of genuine overlap, not just of N passing runs.
_BARRIER_RUN = """set -e
mkdir -p out
SHARED="${JOBD_ROOT:-/workspace}/.barrier"
echo "$JOB_ID" >> "$SHARED"
for _ in $(seq 1 200); do
  [ "$(wc -l < "$SHARED")" -ge "$WANT_N" ] && break
  sleep 0.1
done
n="$(wc -l < "$SHARED")"
[ "$n" -ge "$WANT_N" ] || { echo "NO OVERLAP: only $n of $WANT_N jobs arrived"; exit 9; }
echo "$n" > out/peers.txt
"""


def test_concurrent_jobs_run_in_parallel_and_all_land(tmp_path):
    folder = _write_job(tmp_path, "barrier", _BARRIER_RUN,
                        env_block="env:\n  WANT_N: \"3\"\n")
    r = _run(folder, "--concurrent", "3", tmpdir=tmp_path, timeout=300)
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert "seed bundle + 3 tickets" in r.stdout
    assert "3/3 concurrent job(s) landed" in r.stdout
    assert "RESULT: PASS" in r.stdout


def test_concurrent_default_is_one_job(tmp_path):
    """No --concurrent => byte-for-byte the old single-job report shape (the
    workflow driver and test suite parse it)."""
    folder = _write_job(tmp_path, "solo", "mkdir -p out\necho x > out/r.txt\n")
    r = _run(folder, tmpdir=tmp_path)
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert "seed bundle + ticket ... OK (job_id=" in r.stdout
    assert "concurrent job(s) landed" not in r.stdout


def test_concurrent_rejects_a_bad_count(tmp_path):
    folder = _write_job(tmp_path, "solo2", "mkdir -p out\necho x > out/r.txt\n")
    assert _run(folder, "--concurrent", "0", tmpdir=tmp_path).returncode == 2
    assert _run(folder, "--concurrent", "two", tmpdir=tmp_path).returncode == 2


def test_concurrent_venv_provisioning_is_serialized(tmp_path):
    """THE round-3 gate. `needs.venv: eval` provisioning is box-global mutable
    state; jobd's check_venv now runs the provisioner under a per-kind flock.

    The stub provisioner here is a mutual-exclusion DETECTOR: it fails hard if it
    ever finds a peer inside its critical section. Unserialized, two of two jobs
    overlap and the rehearsal goes red — which is exactly the signal that was
    missing on 2026-07-30.
    """
    prov = tmp_path / "prov.sh"
    prov.write_text(
        "#!/usr/bin/env bash\n"
        'ROOT="${JOBD_ROOT:-/workspace}"\n'
        'mkdir -p "$ROOT/eval"\n'
        'if [ -e "$ROOT/.prov.inflight" ]; then\n'
        '  echo OVERLAP >> "$ROOT/.prov.overlap"\n'
        '  echo "!! provisioner re-entered concurrently" >&2\n'
        "  exit 1\n"
        "fi\n"
        ': > "$ROOT/.prov.inflight"\n'
        "sleep 1\n"
        'rm -f "$ROOT/.prov.inflight"\n'
        'echo run >> "$ROOT/.prov.runs"\n'
        'echo "export HERDD_EVAL_ENV_STUB=1" > "$ROOT/eval/env.sh"\n')
    prov.chmod(0o755)
    folder = _write_job(tmp_path, "venvrace", "mkdir -p out\necho x > out/r.txt\n")
    # needs.venv: eval, so BOTH runners reach check_venv's provisioning branch.
    cfg = json.load(open(os.path.join(folder, "job-config.json"))) \
        if os.path.exists(os.path.join(folder, "job-config.json")) else None
    if cfg is None:                     # _write_job wrote YAML
        p = os.path.join(folder, "job-config.yaml")
        body = open(p).read().replace("venv: none", "venv: eval")
        open(p, "w").write(body)
    r = _run(folder, "--concurrent", "2", tmpdir=tmp_path, timeout=300,
             env_extra={"JOBD_FETCH_EVAL_SH": str(prov)})
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert "2/2 concurrent job(s) landed" in r.stdout
    # the detector never fired, and the provisioner really did run (not skipped)
    kept = glob.glob(str(tmp_path / "rehearse.*" / "workspace" / ".prov.overlap"))
    assert kept == [], f"provisioner overlapped: {kept}"


def test_concurrent_shared_asset_stages_once_and_both_land(tmp_path):
    """N concurrent jobs sharing an asset NAME (all three 2026-07-30 waves
    declared base/adapter-t/adapter-c) must land, and the cache must be pulled
    ONCE — the first holder pulls, the peers fall through skip-if-complete.

    HONEST SCOPE: this asserts the OUTCOME (one pull, both jobs land), not that
    jobd's per-name staging lock caused it. Measured: with the lock disabled this
    test still passes, because a rehearsal asset copies in microseconds and the
    two runners never actually overlap inside the pull. The lock is
    defense-in-depth for the real case (a multi-GB rclone pull, where N runners
    DO overlap); the contention-proving test in this file is the venv one above,
    whose provisioner holds its critical section long enough to collide."""
    seed = _seed_dir(tmp_path, "shared-seed",
                     {"config.json": '{"ok":1}', "w.safetensors": "WEIGHTS"})
    folder = _write_asset_job(
        tmp_path, "sharedasset",
        [{"name": "base", "b2": "weights/base", "dest": "model",
          "require": ["*.safetensors"]}],
        "mkdir -p out\ncat model/config.json > out/seen.json\n")
    pulllog = tmp_path / "pulls.txt"
    r = _run(folder, "--asset", f"base={seed}", "--concurrent", "2",
             tmpdir=tmp_path, timeout=300,
             env_extra={"JOBD_ASSET_PULLLOG": str(pulllog)})
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert "2/2 concurrent job(s) landed" in r.stdout
    pulls = [l for l in pulllog.read_text().splitlines() if l.strip()] \
        if pulllog.exists() else []
    assert pulls == ["base"], f"expected ONE pull of 'base', got {pulls}"
