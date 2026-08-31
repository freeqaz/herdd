"""Canned-box test for onstart/train_boot.sh — the tiny boot-pull wire that
replaced the over-cap inline onstart/train.sh (Vast's 16 KiB cap).

train_boot.sh configures B2, pulls b2:<bucket>/runs/<RUN_ID>/train_main.sh with a
bounded retry, then execs it. We run the REAL shipped script with a stub `rclone`
on PATH (records invocations; copyto writes the dest, or fails on demand) and the
TRAIN_BOOT_* test seams, asserting:
  * success  -> the trainer is pulled, rclone.conf is written with the B2 creds,
                and the exec hand-off is reached (TRAIN_BOOT_NO_EXEC marker).
  * failure  -> after 5 pull attempts a terminal `FAILED train_boot_pull` STATUS
                marker is rcat'd to checkpoints/<RUN_ID>/STATUS, exit 1 (the box
                never sits silent).

Skipped when bash is unavailable (portable lane runs it).
"""
import os
import shutil
import subprocess
import tempfile

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
BOOT_SH = os.path.join(_HERE, "onstart", "train_boot.sh")

pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")

_FAKE_RCLONE = """#!/usr/bin/env bash
echo "CALL $*" >> "$RCLONE_CALLS"
case "$1" in
  copyto)
    [ "${RCLONE_COPYTO_FAIL:-0}" = "1" ] && exit 1
    printf '#!/usr/bin/env bash\\necho trainer-ran\\n' > "$3"
    exit 0 ;;
  rcat)
    body="$(cat)"
    printf 'RCAT %s :: %s\\n' "$2" "$body" >> "$RCLONE_CALLS"
    exit 0 ;;
  *) exit 0 ;;
esac
"""


def _run_boot(copyto_fail=False):
    """Run the real train_boot.sh against a stub rclone. Returns
    (returncode, calls[list], workspace_dir, home_dir)."""
    d = tempfile.mkdtemp()
    bind = os.path.join(d, "bin")
    ws = os.path.join(d, "ws")
    home = os.path.join(d, "home")
    os.makedirs(bind)
    os.makedirs(ws)
    os.makedirs(home)
    calls = os.path.join(d, "calls")
    open(calls, "w").close()
    rc = os.path.join(bind, "rclone")
    with open(rc, "w") as f:
        f.write(_FAKE_RCLONE)
    os.chmod(rc, 0o755)

    env = dict(os.environ)
    env["PATH"] = bind + os.pathsep + env["PATH"]
    env["HOME"] = home
    # Asserts the DEFAULT-path branch ($HOME/.config/rclone/rclone.conf), so it
    # opts out of the suite-wide RCLONE_CONFIG redirect
    # (conftest._rclone_config_scratch) that would otherwise make the assertion
    # vacuous. Safe only because HOME above is a sandbox.
    env.pop("RCLONE_CONFIG", None)
    env["RCLONE_CALLS"] = calls
    env["RCLONE_COPYTO_FAIL"] = "1" if copyto_fail else "0"
    env["TRAIN_BOOT_WS"] = ws
    env["TRAIN_BOOT_NO_EXEC"] = "1"          # stop before exec; assert the marker
    env["TRAIN_BOOT_RETRY_SLEEP"] = "0"      # keep the failure path fast
    env["RUN_ID"] = "r1"
    env["B2_BUCKET"] = "testbucket"
    env["B2_KEY_ID"] = "KID"
    env["B2_APPLICATION_KEY"] = "APPKEY"
    env["B2_S3_ENDPOINT"] = "https://s3.example.com"
    env["B2_REGION"] = "us-west-004"

    p = subprocess.run(["bash", BOOT_SH], env=env, capture_output=True,
                       text=True, timeout=30)
    cl = [ln.strip() for ln in open(calls) if ln.strip()]
    return p.returncode, cl, ws, home


def test_success_pulls_trainer_writes_conf_and_reaches_exec():
    rc, calls, ws, home = _run_boot(copyto_fail=False)
    assert rc == 0, calls
    # pulled the per-RUN trainer from the runs/<RUN_ID>/ path
    assert any("copyto b2:testbucket/runs/r1/train_main.sh" in c for c in calls), calls
    assert os.path.exists(os.path.join(ws, "train_main.sh"))
    # rclone.conf written with the B2 creds (secret in the on-box conf, not the wire)
    conf = os.path.join(home, ".config", "rclone", "rclone.conf")
    assert os.path.exists(conf)
    body = open(conf).read()
    assert "[b2]" in body and "access_key_id = KID" in body \
        and "secret_access_key = APPKEY" in body
    # reached the exec hand-off (NO_EXEC seam wrote the would-exec marker)
    assert os.path.exists(os.path.join(ws, ".train_boot_would_exec")), calls
    # never wrote a FAILED marker on the happy path
    assert not any("RCAT" in c for c in calls), calls


def test_failure_writes_terminal_status_after_retries():
    rc, calls, ws, home = _run_boot(copyto_fail=True)
    assert rc == 1, calls
    # exactly 5 pull attempts before giving up
    assert sum(1 for c in calls if c.startswith("CALL copyto")) == 5, calls
    # a terminal FAILED* STATUS marker was rcat'd (babysit/fold glob-match FAILED*)
    rcat = [c for c in calls if c.startswith("RCAT")]
    assert any("checkpoints/r1/STATUS" in c and "FAILED train_boot_pull" in c
               for c in rcat), calls
    # did NOT reach the exec hand-off
    assert not os.path.exists(os.path.join(ws, ".train_boot_would_exec")), calls
