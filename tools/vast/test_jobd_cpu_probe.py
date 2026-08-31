"""The jobd boot stanza for the CPU probe, under the fake-rclone shim.

Scope split mirrors the GEMM pair: `test_cpu_probe.py` owns the PROBE (slice
width, guard, scaling, record); this file owns the WIRING — that jobd emits the
box event, that the record reaches the per-box B2 prefix through the drain, that
the cache and kill switch are honoured, and above all that **it cannot fail a
boot**. The probe interpreter is faked here, so no test in this file depends on
how loaded the machine running it happens to be.

The one behaviour that differs from gemm_probe and is pinned hardest: a box
presenting no GPUs is still measured. That is the whole point of the probe, and
it is a one-line omission away from silently covering nothing.
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

#: Shaped exactly like cpu_probe.render_fields' output (pinned by test_cpu_probe).
FAKE_FIELDS = (
    "status=ok\n"
    "cpu=AMD_EPYC_7713_64-Core_Processor\n"
    "width_source=cgroup_v2_quota\n"
    "cores=64.0\n"
    "workers=64\n"
    "single_per_s=7100000.0\n"
    "allcore_per_s=433000000.0\n"
    "scaling=0.9534\n")

FAKE_RECORD = {"probe_version": 1, "kind": "cpu", "ts": "2026-08-25T01:02:03Z",
               "instance_id": IID, "units": "pyops", "count": 5.12e8,
               "wall_s": 1.18, "per_s": 4.339e8, "cores": 64.0,
               "per_core_s": 6779000.0, "scaling": 0.9534,
               "width_source": "cgroup_v2_quota"}


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
    """Stands in for the probe INTERPRETER. Receives cpu_probe.py's real argv
    (`drop --fields`), so the invocation under test is the real one."""
    return _exe(str(tmp_path / name), "#!/usr/bin/env bash\n" + body)


def _drops_a_record(tmp_path, name="fake-cpu-python"):
    """The real contract: fields on STDOUT, the record into the drop dir."""
    return _fake_probe_py(
        tmp_path, name,
        'D="${JOBD_HOSTFACTS_DROP:?no drop dir exported}"\n'
        'mkdir -p "$D"\n'
        "printf '%s' " + json.dumps(json.dumps(FAKE_RECORD))
        + ' > "$D/cpu-20260825T010203Z.json"\n'
        "printf '%b' " + json.dumps(FAKE_FIELDS) + "\n"
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
        # The GEMM probe is what we are NOT testing here; keep it out of the
        # drain so any hostfacts object below is unambiguously the CPU one.
        "JOBD_GEMM_PROBE": "0",
        "JOBD_SKIP_GPU": "1",
        "JOBD_FAKE_GPUS": "",
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
def test_boot_emits_a_box_event_and_drains_the_record(tmp_path):
    bucket, shimdir = _bucket_and_shim(tmp_path)
    r = _run_jobd(tmp_path, bucket, shimdir,
                  JOBD_CPU_PY=_drops_a_record(tmp_path))
    assert r.returncode == 0, r.stderr[-3000:]
    evs = _box_events(bucket, "cpu_probe")
    assert len(evs) == 1
    ev = evs[0]
    assert ev["status"] == "ok"
    assert ev["scaling"] == "0.9534"
    assert ev["width_source"] == "cgroup_v2_quota"
    assert ev["instance_id"] == IID

    facts = _hostfacts(bucket)
    assert len(facts) == 1 and facts[0].name.startswith("cpu-")
    rec = json.loads(facts[0].read_text())
    assert rec["kind"] == "cpu" and rec["units"] == "pyops"


def test_the_record_is_keyed_on_the_box_inside_the_jobs_prefix(tmp_path):
    """A split box's scoped write key only allows the `jobs/` namePrefix — a
    `hostfacts/` root would 403 quietly on exactly those boxes."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    _run_jobd(tmp_path, bucket, shimdir,
              JOBD_CPU_PY=_drops_a_record(tmp_path))
    key = str(_hostfacts(bucket)[0].relative_to(bucket))
    assert key.startswith(f"jobs/nodes/{IID}/hostfacts/")
    assert key.startswith("jobs/")


def test_it_runs_before_the_poll_loop_claims_anything(tmp_path):
    """The only moment the box is provably unclaimed. cpu_probe.py refuses a
    busy box, so a stanza that ran later would refuse forever."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    r = _run_jobd(tmp_path, bucket, shimdir,
                  JOBD_CPU_PY=_drops_a_record(tmp_path))
    assert "cpu probe:" in r.stderr
    assert r.stderr.index("scratch probe:") < r.stderr.index("cpu probe:")


# --------------------------------------------------------------------------- #
# the deliberate inversion: a GPU-less box is the one we most want measured
# --------------------------------------------------------------------------- #
def test_a_box_presenting_no_gpus_is_still_probed(tmp_path):
    """gemm_probe hard-skips on JOBD_SKIP_GPU because there is no silicon to
    bench. Copying that guard here would blind the probe to the entire CPU-only
    fleet — which is the fleet it exists for."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    r = _run_jobd(tmp_path, bucket, shimdir, JOBD_SKIP_GPU="1",
                  JOBD_CPU_PY=_drops_a_record(tmp_path))
    assert r.returncode == 0, r.stderr[-3000:]
    assert len(_box_events(bucket, "cpu_probe")) == 1
    assert len(_hostfacts(bucket)) == 1


def test_a_faked_gpu_inventory_does_not_suppress_it_either(tmp_path):
    """JOBD_FAKE_GPUS fakes the *GPU* inventory. The CPUs are still real and
    still worth measuring."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    r = _run_jobd(tmp_path, bucket, shimdir, JOBD_FAKE_GPUS="0:32",
                  JOBD_CPU_PY=_drops_a_record(tmp_path))
    assert r.returncode == 0
    assert len(_box_events(bucket, "cpu_probe")) == 1


def test_the_drop_dir_is_exported_to_the_probe(tmp_path):
    """The probe hands its record to the drain rather than rcat-ing it, so the
    stanza must tell it where the drop dir is. The fake refuses to run without
    it (`${JOBD_HOSTFACTS_DROP:?}`), which is what makes this test bite."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    r = _run_jobd(tmp_path, bucket, shimdir,
                  JOBD_CPU_PY=_drops_a_record(tmp_path))
    assert r.returncode == 0
    assert len(_hostfacts(bucket)) == 1


# --------------------------------------------------------------------------- #
# it cannot fail a boot
# --------------------------------------------------------------------------- #
def test_a_probe_that_hangs_is_killed_and_the_boot_continues(tmp_path):
    bucket, shimdir = _bucket_and_shim(tmp_path)
    hang = _fake_probe_py(tmp_path, "fake-hang", "sleep 300\n")
    r = _run_jobd(tmp_path, bucket, shimdir, JOBD_CPU_PY=hang,
                  JOBD_CPU_TIMEOUT_S="2")
    assert r.returncode == 0, r.stderr[-3000:]
    assert "no record produced" in r.stderr
    ev = _box_events(bucket, "cpu_probe")
    assert len(ev) == 1 and ev[0]["status"] == "skipped_jobd_timeout_2s"
    assert _hostfacts(bucket) == []


def test_a_probe_that_crashes_is_recorded_and_the_boot_continues(tmp_path):
    bucket, shimdir = _bucket_and_shim(tmp_path)
    crash = _fake_probe_py(tmp_path, "fake-crash", "echo boom >&2\nexit 3\n")
    r = _run_jobd(tmp_path, bucket, shimdir, JOBD_CPU_PY=crash)
    assert r.returncode == 0, r.stderr[-3000:]
    ev = _box_events(bucket, "cpu_probe")
    assert len(ev) == 1 and ev[0]["status"] == "skipped_probe_rc_3"


def test_a_missing_interpreter_does_not_fail_the_boot(tmp_path):
    bucket, shimdir = _bucket_and_shim(tmp_path)
    r = _run_jobd(tmp_path, bucket, shimdir, JOBD_CPU_PY="/nonexistent/python")
    assert r.returncode == 0, r.stderr[-3000:]
    assert len(_box_events(bucket, "cpu_probe")) == 1


def test_an_older_bundle_without_the_probe_file_just_skips(tmp_path):
    """jobd.sh ships ahead of / behind cpu_probe.py on a box mid-rotation."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    jobd_dir = tmp_path / "jobd"
    jobd_dir.mkdir()
    for f in ("jobd.sh", "jobd.py"):
        shutil.copy(os.path.join(_HERE, "onstart", f), str(jobd_dir / f))
    for f in ("jobmeta.py", "runmeta.py"):
        shutil.copy(os.path.join(_HERE, f), str(jobd_dir / f))
    env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
    env.update({
        "HOME": str(tmp_path / "home"), "FAKE_BUCKET": str(bucket),
        "B2_BUCKET": "testbucket", "JOBD_IID": IID,
        "JOBD_ROOT": str(tmp_path / "workspace"),
        "JOBD_BOOT_NONCE_FILE": str(tmp_path / "nonce"),
        "JOBD_ONCE": "1", "JOBD_SKIP_B2CONFIG": "1", "JOBD_TMPFS_PROBE": "0",
        "JOBD_PYTHON": sys.executable, "JOBD_SKIP_GPU": "1",
        "JOBD_GEMM_PROBE": "0",
        "JOBD_CRED_DIR": str(tmp_path / "credstate"),
        "PATH": f"{shimdir}:{os.environ.get('PATH', os.defpath)}",
    })
    os.makedirs(env["HOME"], exist_ok=True)
    r = subprocess.run(["bash", str(jobd_dir / "jobd.sh")], env=env,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "cpu_probe.py absent" in r.stderr
    assert _box_events(bucket, "cpu_probe") == []


# --------------------------------------------------------------------------- #
# it must not run when it should not
# --------------------------------------------------------------------------- #
def test_the_kill_switch_stops_it_dead(tmp_path):
    bucket, shimdir = _bucket_and_shim(tmp_path)
    r = _run_jobd(tmp_path, bucket, shimdir, JOBD_CPU_PROBE="0",
                  JOBD_CPU_PY=_drops_a_record(tmp_path))
    assert r.returncode == 0
    assert _box_events(bucket, "cpu_probe") == [] and _hostfacts(bucket) == []


def test_the_probe_is_ON_by_default(tmp_path):
    """The switch above defaults to 1. Several test harnesses pin it to 0 for
    determinism, and this is what keeps one of those from being mistaken for
    the production default."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    r = _run_jobd(tmp_path, bucket, shimdir,
                  JOBD_CPU_PY=_drops_a_record(tmp_path))
    assert len(_box_events(bucket, "cpu_probe")) == 1, r.stderr[-2000:]


def test_a_fresh_cached_record_is_not_re_measured(tmp_path):
    """onstart re-runs on every park/resume, and core throughput does not change
    across one."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / ".cpu_probe.json").write_text("2026-08-25T00:00:00Z")
    r = _run_jobd(tmp_path, bucket, shimdir,
                  JOBD_CPU_PY=_drops_a_record(tmp_path))
    assert r.returncode == 0
    assert "not re-measuring" in r.stderr
    assert _box_events(bucket, "cpu_probe") == []


def test_an_expired_cache_is_re_measured(tmp_path):
    bucket, shimdir = _bucket_and_shim(tmp_path)
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / ".cpu_probe.json").write_text("old")
    r = _run_jobd(tmp_path, bucket, shimdir, JOBD_CPU_MAX_AGE_S="0",
                  JOBD_CPU_PY=_drops_a_record(tmp_path))
    assert r.returncode == 0
    assert len(_box_events(bucket, "cpu_probe")) == 1


# --------------------------------------------------------------------------- #
# a refusal is telemetry, not silence
# --------------------------------------------------------------------------- #
def test_a_busy_box_refusal_still_reaches_the_event_stream(tmp_path):
    """The probe declines to measure a box that is already working, but the
    decline is worth knowing: a box busy at boot is a box that was not idle
    when we started paying for it."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    refuse = _fake_probe_py(
        tmp_path, "fake-busy",
        "printf '%b' 'status=skipped_cpu_busy_3.02_cores\\n"
        "pre_probe_busy_cores=3.017\\n'\nexit 0\n")
    r = _run_jobd(tmp_path, bucket, shimdir, JOBD_CPU_PY=refuse)
    assert r.returncode == 0
    ev = _box_events(bucket, "cpu_probe")
    assert len(ev) == 1
    assert ev[0]["status"].startswith("skipped_cpu_busy")
    assert _hostfacts(bucket) == []      # ...and nothing unquotable was banked


def test_the_local_lane_is_never_measured(tmp_path):
    """`job run-local` boots this jobd on the operator's own machine with IID
    `local-<hostname>`. A record from there would enter the host scorecard and
    the fleet median as if a laptop had been rented — and per_core_s is the
    cross-machine comparand, so one such row skews the number every real box is
    judged against. gemm_probe is spared only because run-local sets
    JOBD_FAKE_GPUS, the guard this probe deliberately drops.
    """
    bucket, shimdir = _bucket_and_shim(tmp_path)
    r = _run_jobd(tmp_path, bucket, shimdir, JOBD_IID="local-example-rig",
                  JOBD_CPU_PY=_drops_a_record(tmp_path))
    assert r.returncode == 0, r.stderr[-3000:]
    assert "local lane" in r.stderr
    d = bucket / "jobs" / "nodes" / "local-example-rig" / "hostfacts"
    assert not d.is_dir() or list(d.iterdir()) == []


def test_a_rented_box_is_still_measured(tmp_path):
    """The negative control for the guard above: an ordinary numeric IID is
    exactly what we DO want measured."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    r = _run_jobd(tmp_path, bucket, shimdir,
                  JOBD_CPU_PY=_drops_a_record(tmp_path))
    assert "local lane" not in r.stderr
    assert len(_hostfacts(bucket)) == 1
