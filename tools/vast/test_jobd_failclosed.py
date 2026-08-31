"""Fail-closed semantics for a HALF-dead jobd (tools/vast/FAILCLOSED_DESIGN.md).

The incident these tests encode: box 47737955 (2026-08-13, 8xRTX4090) rented,
booted, and billed $1.742 over ~52 minutes completing zero work while reporting
itself IDLE. jobd's bash/rclone half was perfectly healthy — it wrote the
JOBD_STATUS marker and a full GEMM hostfacts record — while its python half was
100% dead, because the flat bundle shipped `jobmeta.py` (which had grown an
unguarded top-level `from bidpolicy import ...`) without `bidpolicy.py`. Every
`python3 jobd.py ...` call died ModuleNotFoundError into a `>/dev/null 2>&1 ||
true`.

So the fixture here is not a mock: it is the REAL onstart bundle in the REAL
flat `herdd job attach` layout, and the ONLY difference between the healthy
arm and the broken arm is whether `bidpolicy.py` was copied in beside the rest.
That one file's absence is the whole 2026-08-13 outage.

Runs against the shared fake-rclone shim from test_jobd.py (no network, no B2,
no vast box). `JOBD_SRC_DIR` overrides where the bundle files are copied FROM,
which is how the red half of the red/green proof is taken against the
pre-change sources without touching the working tree.
"""
import datetime
import os
import shutil
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_jobd import (  # noqa: E402  - shared fake-B2 harness, single source of truth
    _cred_hermetic,
    _fake_shm,
    _hermetic_env,
    _make_bucket,
    _stage_job,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
# Overridable so the same test can be pointed at an older checkout of the files
# under test (the "watch it fail first" half of the proof).
_SRC = os.environ.get("JOBD_SRC_DIR", _HERE)

pytestmark = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("timeout")),
    reason="needs bash + timeout")

# Exactly what `herdd job attach` pushes: onstart/ contents and the tools/vast
# python modules, FLATTENED into one directory beside each other.
_FLAT_FILES = ("onstart/jobd.sh", "onstart/jobd.py", "jobmeta.py", "runmeta.py")
_THE_MISSING_FILE = "bidpolicy.py"


def _flat_bundle(tmp_path, *, healthy):
    """Build the flat attach-layout bundle. `healthy=False` omits bidpolicy.py —
    i.e. reproduces ship_manifest.txt as it stood during the incident."""
    d = tmp_path / "jobd"
    d.mkdir()
    for rel in _FLAT_FILES:
        src = os.path.join(_SRC, rel)
        if not os.path.exists(src):
            pytest.skip(f"source bundle file missing: {src}")
        shutil.copy(src, str(d / os.path.basename(rel)))
    if healthy:
        shutil.copy(os.path.join(_SRC, _THE_MISSING_FILE),
                    str(d / _THE_MISSING_FILE))
    return d


def _run(tmp_path, bucket, shimdir, jobd_dir, iid, extra_env=None, timeout=120,
         wall=None):
    """Run the bundle's jobd.sh. `wall` bounds a LOOP-mode run with `timeout(1)`
    (rc 124 when it survives, which several tests below assert is the correct
    outcome — a daemon that must NOT park has to still be running at the end)."""
    env = _hermetic_env(tmp_path)
    env["PATH"] = f"{shimdir}:{env['PATH']}"
    env["FAKE_BUCKET"] = str(bucket)
    env["B2_BUCKET"] = "testbucket"
    env["JOBD_IID"] = str(iid)
    env["JOBD_ROOT"] = str(tmp_path / "workspace")
    env["JOBD_BOOT_NONCE_FILE"] = str(_fake_shm(tmp_path))
    env["JOBD_ONCE"] = "1"
    env["JOBD_SKIP_GPU"] = "1"
    env["JOBD_SKIP_B2CONFIG"] = "1"
    env["JOBD_HEARTBEAT_S"] = "1"
    env["JOBD_PYTHON"] = sys.executable
    env["JOBD_TMPFS_PROBE"] = "0"
    # The CPU probe deliberately ignores JOBD_SKIP_GPU (a GPU-less box is the
    # one it most wants to measure), so unlike gemm_probe nothing else here
    # suppresses it. Off by name: on an idle machine it would add seconds to
    # every boot in this file, and its own busy-box refusal would make that
    # cost depend on host load. test_jobd_cpu_probe.py owns the stanza.
    env["JOBD_CPU_PROBE"] = "0"
    _cred_hermetic(env, tmp_path)
    if extra_env:
        env.update(extra_env)
    cmd = ["bash", str(jobd_dir / "jobd.sh")]
    if wall:
        cmd = ["timeout", str(wall)] + cmd
    return subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=timeout)


def _status(bucket, iid):
    p = bucket / "jobs" / "nodes" / str(iid) / "JOBD_STATUS"
    return p.read_text().strip() if p.exists() else ""


def _hb_epoch(line):
    """First %FT%TZ-shaped token -> UTC epoch. Deliberately the same rule as
    herdd._jobd_status_hb_epoch, so this test fails if the beacon's format
    ever drifts away from what the fleet actually parses."""
    for tok in (line or "").strip().split():
        try:
            dt = datetime.datetime.strptime(tok, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
        return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return None


def _queue_files(bucket, iid):
    d = bucket / "jobs" / "queue" / str(iid)
    return sorted(f.name for f in d.iterdir()) if d.is_dir() else []


# --------------------------------------------------------------------------
# 1. The beacon must stop lying.
# --------------------------------------------------------------------------

def test_healthy_python_half_advertises_ok(tmp_path):
    """Control arm. Same bundle, bidpolicy.py PRESENT: the box says pyhalf=ok.

    Without this the broken-arm assertion below is unfalsifiable — a beacon that
    never says `ok` would pass it for the wrong reason."""
    bucket, shimdir = _make_bucket(tmp_path)
    jobd = _flat_bundle(tmp_path, healthy=True)
    r = _run(tmp_path, bucket, shimdir, jobd, 5501)
    st = _status(bucket, 5501)
    assert "pyhalf=ok" in st, f"status={st!r} stderr={r.stderr[-2000:]}"
    assert "pyhalf=broken" not in st


def test_broken_python_half_refuses_to_advertise_healthy(tmp_path):
    """THE REGRESSION TEST FOR THE INCIDENT. With the python half dead, the box
    must NOT report a bare `IDLE` — the marker that lied for 52 minutes. It must
    self-report `pyhalf=broken` on the bash/rclone channel, which is strictly
    below the half that failed and demonstrably kept working throughout."""
    bucket, shimdir = _make_bucket(tmp_path)
    jobd = _flat_bundle(tmp_path, healthy=False)
    r = _run(tmp_path, bucket, shimdir, jobd, 5502)
    st = _status(bucket, 5502)
    assert st, f"no JOBD_STATUS written at all; stderr={r.stderr[-2000:]}"
    assert "pyhalf=broken" in st, (
        f"a half-dead daemon advertised itself as healthy: {st!r}")
    # and it must name the fault, not just flag it
    assert "bidpolicy" in st or "ModuleNotFoundError" in st, st


# --------------------------------------------------------------------------
# 2. Fail CLOSED before claiming: never consume a ticket you cannot report on.
# --------------------------------------------------------------------------

def test_broken_python_half_refuses_to_claim_a_queued_ticket(tmp_path):
    """Claiming work we cannot emit a single event for is strictly worse than
    refusing it: the ticket is consumed and no observer can ever learn what
    happened. The ticket must be left queued for a box that can run it."""
    bucket, shimdir = _make_bucket(tmp_path)
    jobd = _flat_bundle(tmp_path, healthy=False)
    job_id, _ = _stage_job(tmp_path, bucket, 5503)
    _run(tmp_path, bucket, shimdir, jobd, 5503)
    assert f"{job_id}.json" in _queue_files(bucket, 5503), \
        "a box that cannot emit consumed the ticket anyway"
    # no attempt was recorded and no work dir was built for it
    state = tmp_path / "workspace" / "jobs" / ".state"
    assert not (state / f"{job_id}.attempts").exists()
    assert not (state / f"{job_id}.terminal").exists(), \
        "the ticket was failed rather than left for a healthy box"


def test_ticket_prepare_rc_is_checked_not_swallowed_by_eval(tmp_path):
    """The 14th silent call site — the one that LOOKED guarded.

    `if ! eval "$(jobd.py prepare ...)"` cannot detect this failure: command
    substitution discards the inner exit status and `eval ""` returns 0, so a
    dying prepare parsed as a successful one into empty JOB_* variables. Proven
    on this box 2026-08-14:
        $ bash -c 'eval "$(python3 -c "import nope" 2>/dev/null)"; echo $?'
        0
    Reached here by disabling the boot gate (JOBD_SELFTEST=0), so poll_once runs
    all the way to `prepare` with a broken interpreter — which is precisely the
    path the old guard was supposed to cover and did not."""
    bucket, shimdir = _make_bucket(tmp_path)
    jobd = _flat_bundle(tmp_path, healthy=False)
    job_id, _ = _stage_job(tmp_path, bucket, 5504)
    r = _run(tmp_path, bucket, shimdir, jobd, 5504,
             extra_env={"JOBD_SELFTEST": "0"})
    assert "pyhalf=broken" in _status(bucket, 5504), (
        "prepare died and nothing noticed; stderr=" + r.stderr[-3000:])
    # A box fault must not be charged to the ticket: it stays queued and
    # un-terminal so a healthy box can still run it.
    assert f"{job_id}.json" in _queue_files(bucket, 5504)
    state = tmp_path / "workspace" / "jobs" / ".state"
    assert not (state / f"{job_id}.terminal").exists()


# --------------------------------------------------------------------------
# 3. The $1.742 hole: never claimed work, cannot emit -> die fast and cheap.
# --------------------------------------------------------------------------

def test_broken_half_with_no_work_parks_instead_of_billing_forever(tmp_path):
    """A queued ticket used to pin the box alive FOREVER: an unparsed ticket
    never got a terminal marker, so maybe_idle_park read it as pending work and
    reset the idle clock on every poll. The box that most needed to park was the
    one structurally incapable of parking. Here the ticket is present and the
    box must still park.

    Runs the real poll loop (not JOBD_ONCE) with the JOBD_PARK_CMD test seam
    standing in for the vast API call — the same seam test_jobd.py's park tests
    use, so no box and no key are involved."""
    bucket, shimdir = _make_bucket(tmp_path)
    jobd = _flat_bundle(tmp_path, healthy=False)
    _stage_job(tmp_path, bucket, 5505)
    parked = tmp_path / "parked.marker"
    r = _run(tmp_path, bucket, shimdir, jobd, 5505, timeout=90, wall=25,
             extra_env={
                 "JOBD_ONCE": "0",
                 "JOBD_POLL": "1",
                 "JOBD_PY_BROKEN_PARK_S": "0",   # park on the first idle check
                 "JOBD_PARK_CMD": f"touch {parked}",
             })
    assert parked.exists(), (
        "a box that can neither claim nor report kept billing; "
        f"rc={r.returncode} stderr={r.stderr[-3000:]}")
    assert "pyhalf=broken" in _status(bucket, 5505)


def test_a_healthy_idle_box_is_not_parked_by_the_new_path(tmp_path):
    """FALSE-POSITIVE GUARD. The fail-closed park must be reachable ONLY via a
    proven capability fault. A healthy box with no work must be left entirely to
    the pre-existing idle-park deadlines (JOBD_NO_JOB_PARK_S etc.), untouched by
    anything added here — a mechanism that kills healthy rented boxes costs real
    money in the opposite direction."""
    bucket, shimdir = _make_bucket(tmp_path)
    jobd = _flat_bundle(tmp_path, healthy=True)
    parked = tmp_path / "parked.marker"
    r = _run(tmp_path, bucket, shimdir, jobd, 5506, timeout=60, wall=10,
             extra_env={
                 "JOBD_ONCE": "0",
                 "JOBD_POLL": "1",
                 "JOBD_PY_BROKEN_PARK_S": "0",   # armed, and still must not fire
                 "JOBD_NO_JOB_PARK_S": "9999",   # ordinary deadline far away
                 "JOBD_PARK_CMD": f"touch {parked}",
             })
    assert not parked.exists(), (
        "the fail-closed path parked a HEALTHY box; "
        f"rc={r.returncode} stderr={r.stderr[-3000:]}")
    assert r.returncode == 124, (
        f"the daemon exited on its own (rc={r.returncode}); it should still be "
        f"polling. stderr={r.stderr[-3000:]}")
    assert "pyhalf=ok" in _status(bucket, 5506)


# --------------------------------------------------------------------------
# 4. The beacon must be periodic, or "idle and fine" is indistinguishable
#    from "dead" and nothing downstream can ever be given teeth.
# --------------------------------------------------------------------------

def test_idle_box_re_stamps_its_beacon(tmp_path):
    """status_marker was event-driven only (boot, spawn, reap, end-of-staging),
    so an idle box wrote JOBD_STATUS exactly ONCE and let it age forever. That
    is documented in herdd._jobd_heartbeat_epoch_soft as the false half of two
    ZOMBIE_NO_JOBD alarms on 2026-08-07, and it is why 47737955's genuine
    staleness could not be acted on: the identical observation is produced by
    every healthy idle box. The marker must advance while the box sits idle."""
    bucket, shimdir = _make_bucket(tmp_path)
    jobd = _flat_bundle(tmp_path, healthy=True)
    marker = bucket / "jobs" / "nodes" / "5507" / "JOBD_STATUS"
    wall = 14
    t0 = time.time()
    r = _run(tmp_path, bucket, shimdir, jobd, 5507, timeout=60, wall=wall,
             extra_env={
                 "JOBD_ONCE": "0",
                 "JOBD_POLL": "1",
                 "JOBD_STATUS_EVERY_S": "1",
                 "JOBD_NO_JOB_PARK_S": "9999",
             })
    assert marker.exists(), f"stderr={r.stderr[-2000:]}"
    body = marker.read_text()
    assert "IDLE" in body and "pyhalf=ok" in body, body
    # The stamp is the daemon's own `date -u +%FT%TZ`. A boot-only marker carries
    # a time within a few seconds of launch no matter how long the box lives;
    # a periodic one tracks the wall clock. Parsed the same way
    # herdd._jobd_status_hb_epoch does it: first %FT%TZ-shaped token.
    stamp = _hb_epoch(body)
    assert stamp is not None, f"no parseable heartbeat stamp in {body!r}"
    age_at_exit = (t0 + wall) - stamp
    assert age_at_exit < wall / 2, (
        "an idle box stamped JOBD_STATUS once and let it age forever: the "
        f"marker was already {age_at_exit:.0f}s stale after a {wall}s run "
        f"(body={body!r})")
