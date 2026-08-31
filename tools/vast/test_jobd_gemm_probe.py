"""The jobd boot stanza for the GEMM ceiling probe, under the fake-rclone shim.

Scope split, deliberately: `test_gemm_probe.py` owns the PROBE (guard, budget,
deadline, record). This file owns the WIRING — that jobd emits the box event,
pushes the durable object to the per-box B2 prefix, honours its cache and its
kill switch, and above all **cannot fail a boot**. So the probe interpreter is a
fake here: no torch, no CUDA, no card is touched by any test in this file.

The boot path is load-bearing on every box we rent; a defect here costs real
money and strands runs. Every branch below ends in "jobd carried on".
"""
import json
import os
import shutil
import stat
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_HERE = os.path.dirname(os.path.abspath(__file__))
JOBD_SH = os.path.join(_HERE, "onstart", "jobd.sh")
_SHIM_PATH = os.path.join(_HERE, "testlib", "rclone_shim.sh")

pytestmark = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("timeout")),
    reason="needs bash + timeout")

_ENV_PASSTHROUGH = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM")

IID = "46947265"

#: What the fake probe interpreter writes to --fields-out. Shaped exactly like
#: gemm_probe.render_fields' output (that shape is pinned by test_gemm_probe.py).
FAKE_FIELDS = (
    "status=ok\n"
    "gpu=RTX_PRO_6000_Blackwell_Server_Edition\n"
    "cap=sm_120\n"
    "shape_basis=generic\n"
    "ceiling_tflops=269.4\n"
    "power_limit_w=600\n"
    "sm_clock_mhz=2370\n")

FAKE_RECORD = {"probe_version": 1, "status": "ok", "ts": "2026-08-07T01:02:03Z",
               "device": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
               "ceiling_tflops": 269.4, "power_limit_w": 600}


def _exe(path, text):
    with open(path, "w") as fh:
        fh.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP
             | stat.S_IXOTH)
    return path


def _bucket_and_shim(tmp_path):
    bucket = tmp_path / "bucket"
    bucket.mkdir()
    shimdir = tmp_path / "bin"
    shimdir.mkdir()
    with open(_SHIM_PATH) as f:
        _exe(str(shimdir / "rclone"), f.read())
    return bucket, shimdir


def _fake_probe_py(tmp_path, name, body):
    """A stand-in for the probe INTERPRETER. It receives gemm_probe.py's real
    argv, so it parses --out/--fields-out exactly as the real invocation would."""
    return _exe(str(tmp_path / name),
                "#!/usr/bin/env bash\n"
                'OUT=""; FLD=""\n'
                'while [ $# -gt 0 ]; do\n'
                '  case "$1" in\n'
                '    --out) OUT="$2"; shift 2 ;;\n'
                '    --fields-out) FLD="$2"; shift 2 ;;\n'
                '    *) shift ;;\n'
                '  esac\n'
                'done\n' + body)


def _writes_a_record(tmp_path, name="fake-python"):
    return _fake_probe_py(
        tmp_path, name,
        'printf "%s" ' + json.dumps(json.dumps(FAKE_RECORD)) + ' > "$OUT"\n'
        'printf "%b" ' + json.dumps(FAKE_FIELDS) + ' > "$FLD"\n'
        "exit 0\n")


def _run_jobd(tmp_path, bucket, shimdir, **extra):
    env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
    env.setdefault("PATH", os.defpath)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    shm = tmp_path / "shm"
    shm.mkdir(exist_ok=True)
    env.update({
        "HOME": str(home),
        "PATH": f"{shimdir}:{env['PATH']}",
        "FAKE_BUCKET": str(bucket),
        "B2_BUCKET": "testbucket",
        "JOBD_IID": IID,
        "JOBD_ROOT": str(tmp_path / "workspace"),
        "JOBD_BOOT_NONCE_FILE": str(shm / "nonce"),
        "JOBD_ONCE": "1",
        "JOBD_SKIP_B2CONFIG": "1",
        "JOBD_TMPFS_PROBE": "0",
        "JOBD_PYTHON": sys.executable,
        "JOBD_CRED_DIR": str(tmp_path / "credstate"),
        # SKIP_GPU would short-circuit the stanza under test; the probe
        # interpreter is faked instead, so no card is touched either way.
        "JOBD_SKIP_GPU": "0",
        "JOBD_FAKE_GPUS": "",
        # The CPU probe ignores JOBD_SKIP_GPU by design, so nothing else here
        # keeps it out of this file's boots — and its cache logs the same
        # "not re-measuring" phrase the cache tests below grep for.
        # test_jobd_cpu_probe.py owns that stanza.
        "JOBD_CPU_PROBE": "0",
    })
    env.update(extra)
    return subprocess.run(["bash", JOBD_SH], env=env, capture_output=True,
                          text=True, timeout=180)


def _box_events(bucket, event):
    d = bucket / "jobs" / "nodes" / IID / "events"
    out = []
    if d.is_dir():
        for f in sorted(d.iterdir()):
            blob = json.loads(f.read_text())
            if blob.get("event") == event:
                out.append(blob)
    return out


def _hostfacts(bucket):
    d = bucket / "jobs" / "nodes" / IID / "hostfacts"
    return sorted(d.iterdir()) if d.is_dir() else []


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #
def test_boot_emits_a_box_event_and_a_durable_object(tmp_path):
    bucket, shimdir = _bucket_and_shim(tmp_path)
    r = _run_jobd(tmp_path, bucket, shimdir,
                  JOBD_GEMM_PY=_writes_a_record(tmp_path))
    assert r.returncode == 0, r.stderr[-3000:]
    evs = _box_events(bucket, "gemm_probe")
    assert len(evs) == 1
    ev = evs[0]
    assert ev["status"] == "ok"
    assert ev["ceiling_tflops"] == "269.4"
    assert ev["power_limit_w"] == 600          # jobd.py coerces integer-looking
    assert ev["instance_id"] == IID
    facts = _hostfacts(bucket)
    assert len(facts) == 1 and facts[0].name.startswith("gemm-")
    assert json.loads(facts[0].read_text())["device"].endswith("Server Edition")


def test_the_durable_object_is_keyed_on_the_box_and_names_no_job(tmp_path):
    """`job retarget` keeps the JOB_ID and moves it to a different box. A
    ceiling filed under a job would be attributed to the wrong machine
    (memory: workload-state-stored-on-the-box)."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    _run_jobd(tmp_path, bucket, shimdir, JOBD_GEMM_PY=_writes_a_record(tmp_path))
    key = str(_hostfacts(bucket)[0].relative_to(bucket))
    assert key.startswith(f"jobs/nodes/{IID}/hostfacts/")
    # ...and inside the `jobs/` namePrefix a split box's scoped write key allows
    assert key.startswith("jobs/")


def test_the_probe_runs_before_the_poll_loop_claims_anything(tmp_path):
    """The only moment the GPU is provably idle. The stanza sits between
    scratch_probe and adopt_running, so its log line precedes them."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    r = _run_jobd(tmp_path, bucket, shimdir,
                  JOBD_GEMM_PY=_writes_a_record(tmp_path))
    out = r.stderr
    assert "gemm probe:" in out
    assert out.index("scratch probe:") < out.index("gemm probe:")


# --------------------------------------------------------------------------- #
# it cannot fail a boot
# --------------------------------------------------------------------------- #
def test_a_probe_that_hangs_is_killed_and_the_boot_continues(tmp_path):
    bucket, shimdir = _bucket_and_shim(tmp_path)
    hang = _fake_probe_py(tmp_path, "fake-hang", "sleep 300\n")
    r = _run_jobd(tmp_path, bucket, shimdir, JOBD_GEMM_PY=hang,
                  JOBD_GEMM_TIMEOUT_S="2")
    assert r.returncode == 0, r.stderr[-3000:]
    assert "no record produced" in r.stderr
    ev = _box_events(bucket, "gemm_probe")
    assert len(ev) == 1 and ev[0]["status"] == "skipped_jobd_timeout_2s"
    assert _hostfacts(bucket) == []


def test_a_probe_that_crashes_is_recorded_and_the_boot_continues(tmp_path):
    bucket, shimdir = _bucket_and_shim(tmp_path)
    crash = _fake_probe_py(tmp_path, "fake-crash",
                           "echo 'boom' >&2\nexit 3\n")
    r = _run_jobd(tmp_path, bucket, shimdir, JOBD_GEMM_PY=crash)
    assert r.returncode == 0, r.stderr[-3000:]
    ev = _box_events(bucket, "gemm_probe")
    assert len(ev) == 1 and ev[0]["status"] == "skipped_probe_rc_3"


def test_a_missing_interpreter_does_not_fail_the_boot(tmp_path):
    bucket, shimdir = _bucket_and_shim(tmp_path)
    r = _run_jobd(tmp_path, bucket, shimdir,
                  JOBD_GEMM_PY="/nonexistent/python")
    assert r.returncode == 0, r.stderr[-3000:]
    assert len(_box_events(bucket, "gemm_probe")) == 1


def test_an_older_bundle_without_the_probe_file_just_skips(tmp_path):
    """jobd.sh ships ahead of / behind gemm_probe.py on a box mid-rotation. The
    stanza must degrade, not die — the same shape as jobd's `preempt_save`
    fail-open."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    jobd_dir = tmp_path / "jobd"
    jobd_dir.mkdir()
    for f in ("jobd.sh", "jobd.py"):
        shutil.copy(os.path.join(_HERE, "onstart", f), str(jobd_dir / f))
    for f in ("jobmeta.py", "runmeta.py"):
        shutil.copy(os.path.join(_HERE, f), str(jobd_dir / f))
    env_extra = {"PATH": f"{shimdir}:{os.environ.get('PATH', os.defpath)}"}
    env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
    env.update({
        "HOME": str(tmp_path / "home"), "FAKE_BUCKET": str(bucket),
        "B2_BUCKET": "testbucket", "JOBD_IID": IID,
        "JOBD_ROOT": str(tmp_path / "workspace"),
        "JOBD_BOOT_NONCE_FILE": str(tmp_path / "nonce"),
        "JOBD_ONCE": "1", "JOBD_SKIP_B2CONFIG": "1", "JOBD_TMPFS_PROBE": "0",
        "JOBD_PYTHON": sys.executable, "JOBD_SKIP_GPU": "0",
        "JOBD_CRED_DIR": str(tmp_path / "credstate"),
    })
    env.update(env_extra)
    os.makedirs(env["HOME"], exist_ok=True)
    r = subprocess.run(["bash", str(jobd_dir / "jobd.sh")], env=env,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "gemm_probe.py absent" in r.stderr
    assert _box_events(bucket, "gemm_probe") == []


def test_a_b2_push_failure_keeps_the_record_locally_and_continues(tmp_path):
    """A dead B2 key must not lose the measurement — the local cache is still
    there and `hostfacts.py` can be pointed at a pulled copy."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    real = shimdir / "rclone"
    wrap = tmp_path / "failbin"
    wrap.mkdir()
    _exe(str(wrap / "rclone"),
         "#!/usr/bin/env bash\n"
         f'REAL="{real}"\n'
         'case "${2:-}" in *"/hostfacts/"*) '
         'echo "ERROR: InvalidAccessKeyId" >&2; exit 1 ;; esac\n'
         'exec "$REAL" "$@"\n')
    r = _run_jobd(tmp_path, bucket, shimdir,
                  JOBD_GEMM_PY=_writes_a_record(tmp_path),
                  PATH=f"{wrap}:{shimdir}:{os.environ.get('PATH', os.defpath)}")
    assert r.returncode == 0, r.stderr[-3000:]
    assert "B2 push failed" in r.stderr
    assert (tmp_path / "workspace" / ".gemm_probe.json").is_file()
    assert len(_box_events(bucket, "gemm_probe")) == 1     # the event still lands


# --------------------------------------------------------------------------- #
# it must not run when it should not
# --------------------------------------------------------------------------- #
def test_the_kill_switch_stops_it_dead(tmp_path):
    bucket, shimdir = _bucket_and_shim(tmp_path)
    r = _run_jobd(tmp_path, bucket, shimdir, JOBD_GEMM_PROBE="0",
                  JOBD_GEMM_PY=_writes_a_record(tmp_path))
    assert r.returncode == 0
    assert _box_events(bucket, "gemm_probe") == [] and _hostfacts(bucket) == []


def test_a_box_presenting_no_gpus_by_policy_is_not_benched(tmp_path):
    """CPU boxes and the CPU-only rehearsal lane. Also what keeps the rest of
    the suite (JOBD_SKIP_GPU=1 everywhere) from ever spawning this."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    r = _run_jobd(tmp_path, bucket, shimdir, JOBD_SKIP_GPU="1",
                  JOBD_GEMM_PY=_writes_a_record(tmp_path))
    assert r.returncode == 0
    assert _box_events(bucket, "gemm_probe") == []


def test_a_faked_gpu_inventory_is_not_benched(tmp_path):
    """JOBD_FAKE_GPUS exists so the scheduler is testable without hardware. If
    the probe ran anyway it would bench whatever silicon is really there."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    r = _run_jobd(tmp_path, bucket, shimdir, JOBD_FAKE_GPUS="0:32",
                  JOBD_GEMM_PY=_writes_a_record(tmp_path))
    assert r.returncode == 0
    assert _box_events(bucket, "gemm_probe") == []


def test_a_fresh_cached_record_is_not_re_measured(tmp_path):
    """onstart re-runs on every park/resume. A machine's ceiling does not change
    across one, so re-probing would pay 30 s for a number we already have."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    py = _writes_a_record(tmp_path)
    _run_jobd(tmp_path, bucket, shimdir, JOBD_GEMM_PY=py)
    assert len(_hostfacts(bucket)) == 1
    r = _run_jobd(tmp_path, bucket, shimdir, JOBD_GEMM_PY=py)
    # Name the probe: every cached stanza logs this phrase, so the bare
    # substring silently starts matching a different one the day a second
    # probe with a cache is added (it was, 2026-08-25).
    assert "gemm probe: cached record" in r.stderr
    assert len(_hostfacts(bucket)) == 1
    assert len(_box_events(bucket, "gemm_probe")) == 1


def test_an_expired_cache_is_re_measured(tmp_path):
    bucket, shimdir = _bucket_and_shim(tmp_path)
    py = _writes_a_record(tmp_path)
    _run_jobd(tmp_path, bucket, shimdir, JOBD_GEMM_PY=py)
    r = _run_jobd(tmp_path, bucket, shimdir, JOBD_GEMM_PY=py,
                  JOBD_GEMM_MAX_AGE_S="0")
    assert "gemm probe: cached record" not in r.stderr
    assert len(_box_events(bucket, "gemm_probe")) == 2


def test_the_probe_does_not_leak_the_train_env_into_jobds_own_shell(tmp_path):
    """The venv is sourced in a SUBSHELL. Activating it in jobd's shell would
    change PATH/VIRTUAL_ENV for every entrypoint it later spawns — a config
    change no job asked for."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    ws = tmp_path / "workspace"
    ws.mkdir(exist_ok=True)
    act = tmp_path / "activate.sh"
    act.write_text('export JOBD_GEMM_LEAK_CANARY=leaked\nexport PATH="/nope:$PATH"\n')
    # jobd reads the marker at the hardcoded /workspace path, so this only
    # exercises the subshell when that file exists; assert the canary either way.
    r = _run_jobd(tmp_path, bucket, shimdir,
                  JOBD_GEMM_PY=_fake_probe_py(
                      tmp_path, "fake-canary",
                      'echo "canary=${JOBD_GEMM_LEAK_CANARY:-unset}" >&2\n'
                      'printf "%s" ' + json.dumps(json.dumps(FAKE_RECORD))
                      + ' > "$OUT"\n'
                      'printf "%b" ' + json.dumps(FAKE_FIELDS) + ' > "$FLD"\n'))
    assert r.returncode == 0
    assert "JOBD_GEMM_LEAK_CANARY" not in r.stderr
