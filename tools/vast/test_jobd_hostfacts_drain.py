"""jobd ships what a JOB harvested — the producer-agnostic half of hostfacts.

`gemm_probe` is a benchmark jobd runs itself at boot, so jobd owns the file and
uploads it in the same function. That shape cannot carry a HARVESTED fact: what
a machine's cores are worth is counted off work we were already paying for, so
it is produced by the job, mid-run, and only the job knows what it counted.

Hence the drop dir: the producer writes `<kind>-<ts>.json` and jobd drains it.
These tests own the DRAIN — that jobd finds a dropped record, PUTs it under the
per-box prefix a scoped write key allows, does not re-upload it, retries a
failure, refuses a truncated file, and above all cannot fail a boot.

`test_hostfacts.py` owns the record and the drop-side writer; this file owns the
wiring, the same split `test_jobd_gemm_probe.py` states for the probe.

Offline: a fake `rclone` shim and a tmp "bucket" dir. No B2, no vast API, $0.
"""
import json
import os
import shutil
import stat
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hostfacts as hf  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
JOBD_SH = os.path.join(_HERE, "onstart", "jobd.sh")
_SHIM_PATH = os.path.join(_HERE, "testlib", "rclone_shim.sh")

pytestmark = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("timeout")),
    reason="needs bash + timeout")

_ENV_PASSTHROUGH = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM")

IID = "46947265"


def _exe(path, text):
    with open(path, "w") as fh:
        fh.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP
             | stat.S_IXOTH)
    return path


def _bucket_and_shim(tmp_path, rclone_body=None):
    bucket = tmp_path / "bucket"
    bucket.mkdir()
    shimdir = tmp_path / "bin"
    shimdir.mkdir()
    if rclone_body is None:
        with open(_SHIM_PATH) as f:
            rclone_body = f.read()
    _exe(str(shimdir / "rclone"), rclone_body)
    return bucket, shimdir


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
        # The CPU probe deliberately ignores JOBD_SKIP_GPU (a GPU-less box is
        # the one it most wants to measure), so unlike gemm_probe nothing else
        # here suppresses it — and it drops a REAL cpu record into the same
        # drain this file is testing, which would show up as an extra object.
        "JOBD_CPU_PROBE": "0",
        "JOBD_PYTHON": sys.executable,
        "JOBD_CRED_DIR": str(tmp_path / "credstate"),
        "JOBD_SKIP_GPU": "1",     # no card, and no gemm stanza in the way
        "JOBD_FAKE_GPUS": "",
    })
    env.update(extra)
    return subprocess.run(["bash", JOBD_SH], env=env, capture_output=True,
                          text=True, timeout=180)


def _drop(tmp_path, name="cpu-2026-08-24T09_00_00Z.json", body=None):
    """Put a record in the drop dir the way a producer would."""
    d = tmp_path / "workspace" / "hostfacts.d"
    d.mkdir(parents=True, exist_ok=True)
    f = d / name
    if body is None:
        body = json.dumps(hf.cpu_record(IID, "2026-08-24T09:00:00Z",
                                        units="tu_compiles", count=1200,
                                        wall_s=300, cores=48))
    f.write_text(body)
    return d, f


def _uploaded(bucket):
    d = bucket / "jobs" / "nodes" / IID / "hostfacts"
    return sorted(p.name for p in d.iterdir()) if d.is_dir() else []


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #
def test_a_dropped_record_reaches_the_per_box_prefix(tmp_path):
    bucket, shimdir = _bucket_and_shim(tmp_path)
    _drop(tmp_path)
    r = _run_jobd(tmp_path, bucket, shimdir)
    assert r.returncode == 0, r.stderr[-3000:]
    assert _uploaded(bucket) == ["cpu-2026-08-24T09_00_00Z.json"]


def test_the_uploaded_bytes_are_the_record_verbatim(tmp_path):
    """A drain is a transfer, not a re-serialization: `ingest` reads `units`
    and the derived rates straight back out."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    _drop(tmp_path)
    _run_jobd(tmp_path, bucket, shimdir)
    got = json.loads((bucket / "jobs" / "nodes" / IID / "hostfacts"
                      / "cpu-2026-08-24T09_00_00Z.json").read_text())
    assert got["kind"] == "cpu"
    assert got["units"] == "tu_compiles"
    assert got["per_s"] == 4.0
    assert got["per_core_s"] == pytest.approx(0.08333)


def test_the_key_names_no_job_and_stays_inside_the_jobs_prefix(tmp_path):
    """A split box's write key is `namePrefix=jobs/`, so anything outside it
    403s — and the fact belongs to the MACHINE, so it must not be filed under a
    job id that `job retarget` can move to a different box."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    _drop(tmp_path)
    _run_jobd(tmp_path, bucket, shimdir)
    rel = (bucket / "jobs" / "nodes" / IID / "hostfacts").relative_to(bucket)
    assert str(rel).startswith("jobs/")
    assert str(rel) == f"jobs/nodes/{IID}/hostfacts"


def test_a_drained_record_is_not_uploaded_twice(tmp_path):
    """One immutable object per measurement. A re-PUT of the same key is at
    best wasted and at worst a silent overwrite."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    d, f = _drop(tmp_path)
    _run_jobd(tmp_path, bucket, shimdir)
    assert not f.exists(), "drained record left in the drop dir"
    assert (d / ".sent" / f.name).exists(), "drained record was not retained"
    _run_jobd(tmp_path, bucket, shimdir)
    assert _uploaded(bucket) == ["cpu-2026-08-24T09_00_00Z.json"]


def test_several_records_all_ship(tmp_path):
    bucket, shimdir = _bucket_and_shim(tmp_path)
    _drop(tmp_path, name="cpu-a.json")
    _drop(tmp_path, name="cpu-b.json")
    _drop(tmp_path, name="gemm-c.json")
    _run_jobd(tmp_path, bucket, shimdir)
    assert _uploaded(bucket) == ["cpu-a.json", "cpu-b.json", "gemm-c.json"]


# --------------------------------------------------------------------------- #
# it cannot fail a boot, and it cannot ship garbage
# --------------------------------------------------------------------------- #
def test_no_drop_dir_at_all_is_the_normal_case(tmp_path):
    bucket, shimdir = _bucket_and_shim(tmp_path)
    r = _run_jobd(tmp_path, bucket, shimdir)
    assert r.returncode == 0, r.stderr[-3000:]
    assert _uploaded(bucket) == []


def test_an_empty_drop_dir_ships_nothing_and_logs_nothing(tmp_path):
    """`*.json` unexpanded is the every-tick case; it must not read as a file
    literally named `*.json`."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    (tmp_path / "workspace" / "hostfacts.d").mkdir(parents=True)
    r = _run_jobd(tmp_path, bucket, shimdir)
    assert r.returncode == 0, r.stderr[-3000:]
    assert _uploaded(bucket) == []
    assert "*.json" not in r.stderr


def test_a_truncated_record_is_refused_rather_than_PUT(tmp_path):
    """A zero-byte file is a producer that died between create and write.
    PUTting it mints an immutable empty object nobody can later tell from a
    real record."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    _drop(tmp_path, name="cpu-empty.json", body="")
    r = _run_jobd(tmp_path, bucket, shimdir)
    assert r.returncode == 0
    assert _uploaded(bucket) == []
    assert "is empty" in r.stderr


def test_a_partial_file_is_invisible_to_the_drain(tmp_path):
    """`drop_record` renames into place, so a `.partial` is mid-write BY
    CONTRACT and the glob must not pick it up."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    d = tmp_path / "workspace" / "hostfacts.d"
    d.mkdir(parents=True)
    (d / "cpu-x.json.partial").write_text('{"half":')
    r = _run_jobd(tmp_path, bucket, shimdir)
    assert r.returncode == 0
    assert _uploaded(bucket) == []
    assert (d / "cpu-x.json.partial").exists()


def test_a_push_failure_keeps_the_record_for_the_next_drain(tmp_path):
    """The one difference from gemm_probe, which has a local cache to fall back
    on: there is no second copy of a harvested rate, so a failed PUT must leave
    the file in place to retry rather than drop it."""
    with open(_SHIM_PATH) as f:
        real = f.read()
    body = real.replace(
        "#!/usr/bin/env bash",
        '#!/usr/bin/env bash\ncase "${2:-}" in *"/hostfacts/"*) exit 7;; esac',
        1)
    bucket, shimdir = _bucket_and_shim(tmp_path, rclone_body=body)
    d, f = _drop(tmp_path)
    r = _run_jobd(tmp_path, bucket, shimdir)
    assert r.returncode == 0, r.stderr[-3000:]
    assert f.exists(), "a failed push must not lose the record"
    assert "kept for the next drain" in r.stderr


def test_the_kill_switch_stops_it_dead(tmp_path):
    bucket, shimdir = _bucket_and_shim(tmp_path)
    d, f = _drop(tmp_path)
    r = _run_jobd(tmp_path, bucket, shimdir, JOBD_HOSTFACTS_DRAIN="0")
    assert r.returncode == 0
    assert _uploaded(bucket) == []
    assert f.exists()


def test_the_drop_dir_is_overridable(tmp_path):
    bucket, shimdir = _bucket_and_shim(tmp_path)
    alt = tmp_path / "elsewhere"
    alt.mkdir()
    (alt / "cpu-alt.json").write_text('{"kind":"cpu"}')
    r = _run_jobd(tmp_path, bucket, shimdir, JOBD_HOSTFACTS_DROP=str(alt))
    assert r.returncode == 0
    assert _uploaded(bucket) == ["cpu-alt.json"]


# --------------------------------------------------------------------------- #
# bash <-> python key parity
# --------------------------------------------------------------------------- #
def test_jobds_bash_key_and_hostfacts_python_key_agree(tmp_path):
    """jobd.sh open-codes the B2 prefix as a bash string literal and
    `hostfacts.instance_prefix` builds the same path in python. Nothing tied
    them together, which is the drift class `test_autotune.py`'s bash-mirror
    grid exists to prevent — and a drift here files records where `ingest`
    does not look, silently, forever.

    Asserted against the path jobd ACTUALLY WROTE, not against a re-read of the
    literal: a test that reads the literal out of the script proves only that
    the script contains itself."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    _drop(tmp_path, name="cpu-parity.json")
    _run_jobd(tmp_path, bucket, shimdir)
    written = bucket / "jobs" / "nodes" / IID / "hostfacts" / "cpu-parity.json"
    assert written.exists(), "jobd wrote nothing to compare"
    assert str(written.parent.relative_to(bucket)) == hf.instance_prefix(IID)
    assert str(written.relative_to(bucket)) == hf.instance_key(
        IID, "parity", kind="cpu")


def test_a_dropped_record_pins_as_its_own_kind_through_ingest(tmp_path):
    """End to end across the seam: what jobd uploads is what `ingest` promotes,
    and a cpu record must not pin as a gemm one (the kind rides the KEY)."""
    bucket, shimdir = _bucket_and_shim(tmp_path)
    _drop(tmp_path)
    _run_jobd(tmp_path, bucket, shimdir)
    store = hf.LocalStore(str(bucket))
    res = hf.ingest(store, lambda iid: "140799")
    assert len(res["pinned"]) == 1
    assert res["pinned"][0].startswith("hostfacts/by-machine/140799/cpu-")
    rec = store.get(res["pinned"][0])
    assert rec["kind"] == "cpu" and rec["units"] == "tu_compiles"
