"""Canned-box test for onstart/serve_boot.sh — the tiny boot-pull wire that
replaced the over-cap inline onstart/serve_vllm.sh (Vast's 16 KiB cap).

serve_boot.sh configures B2, pulls b2:<bucket>/serve/<SERVE_ID>/serve_main.sh with
a bounded retry, then execs it. We run the REAL shipped script with a stub `rclone`
on PATH (records invocations; copyto writes the dest, or fails on demand) and the
SERVE_BOOT_* test seams, asserting:
  * success  -> the server is pulled, rclone.conf is written with the B2 creds,
                and the exec hand-off is reached (SERVE_BOOT_NO_EXEC marker).
  * failure  -> after 5 pull attempts a terminal `FAILED serve_boot_pull`
                SERVE_STATUS marker is rcat'd to serve/<SERVE_ID>/SERVE_STATUS
                (serve_ready.sh exits 3 on it — the box never sits silent), exit 1.
  * scoped write pair -> the FAILED marker routes via the [b2w] remote.

Skipped when bash is unavailable (portable lane runs it).
"""
import os
import shutil
import subprocess
import tempfile

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
BOOT_SH = os.path.join(_HERE, "onstart", "serve_boot.sh")

pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")

_FAKE_RCLONE = """#!/usr/bin/env bash
echo "CALL $*" >> "$RCLONE_CALLS"
case "$1" in
  copyto)
    [ "${RCLONE_COPYTO_FAIL:-0}" = "1" ] && exit 1
    printf '#!/usr/bin/env bash\\necho server-ran\\n' > "$3"
    exit 0 ;;
  rcat)
    body="$(cat)"
    printf 'RCAT %s :: %s\\n' "$2" "$body" >> "$RCLONE_CALLS"
    exit 0 ;;
  *) exit 0 ;;
esac
"""


def _run_boot(copyto_fail=False, write_pair=False):
    """Run the real serve_boot.sh against a stub rclone. Returns
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
    # This test asserts the DEFAULT-path branch — that the boot script writes
    # $HOME/.config/rclone/rclone.conf — so it must opt out of the suite-wide
    # RCLONE_CONFIG redirect (conftest._rclone_config_scratch), which would
    # otherwise send the write elsewhere and make the assertion vacuous. Safe
    # only because HOME above is a sandbox; never drop it while forwarding the
    # real one.
    env.pop("RCLONE_CONFIG", None)
    env["RCLONE_CALLS"] = calls
    env["RCLONE_COPYTO_FAIL"] = "1" if copyto_fail else "0"
    env["SERVE_BOOT_WS"] = ws
    env["SERVE_BOOT_NO_EXEC"] = "1"          # stop before exec; assert the marker
    env["SERVE_BOOT_RETRY_SLEEP"] = "0"      # keep the failure path fast
    env["SERVE_ID"] = "s1"
    env["B2_BUCKET"] = "testbucket"
    env["B2_KEY_ID"] = "KID"
    env["B2_APPLICATION_KEY"] = "APPKEY"
    env["B2_S3_ENDPOINT"] = "https://s3.example.com"
    env["B2_REGION"] = "us-west-004"
    if write_pair:
        env["B2_WRITE_KEY_ID"] = "WKID"
        env["B2_WRITE_APPLICATION_KEY"] = "WAPPKEY"

    p = subprocess.run(["bash", BOOT_SH], env=env, capture_output=True,
                       text=True, timeout=30)
    cl = [ln.strip() for ln in open(calls) if ln.strip()]
    return p.returncode, cl, ws, home


def test_success_pulls_server_writes_conf_and_reaches_exec():
    rc, calls, ws, home = _run_boot(copyto_fail=False)
    assert rc == 0, calls
    # pulled the per-SERVE server from the serve/<SERVE_ID>/ path
    assert any("copyto b2:testbucket/serve/s1/serve_main.sh" in c for c in calls), calls
    assert os.path.exists(os.path.join(ws, "serve_main.sh"))
    # rclone.conf written with the B2 creds (secret in the on-box conf, not the wire)
    conf = os.path.join(home, ".config", "rclone", "rclone.conf")
    assert os.path.exists(conf)
    body = open(conf).read()
    assert "[b2]" in body and "access_key_id = KID" in body \
        and "secret_access_key = APPKEY" in body
    # single-key box: no [b2w] remote
    assert "[b2w]" not in body
    # reached the exec hand-off (NO_EXEC seam wrote the would-exec marker)
    assert os.path.exists(os.path.join(ws, ".serve_boot_would_exec")), calls
    # never wrote a FAILED marker on the happy path
    assert not any("RCAT" in c for c in calls), calls


def test_failure_writes_terminal_status_after_retries():
    rc, calls, ws, home = _run_boot(copyto_fail=True)
    assert rc == 1, calls
    # exactly 5 pull attempts before giving up
    assert sum(1 for c in calls if c.startswith("CALL copyto")) == 5, calls
    # a terminal FAILED SERVE_STATUS marker was rcat'd (serve_ready.sh exits 3 on it)
    rcat = [c for c in calls if c.startswith("RCAT")]
    assert any("serve/s1/SERVE_STATUS" in c and "FAILED serve_boot_pull" in c
               for c in rcat), calls
    # did NOT reach the exec hand-off
    assert not os.path.exists(os.path.join(ws, ".serve_boot_would_exec")), calls


def test_scoped_write_pair_routes_failed_marker_via_b2w():
    rc, calls, ws, home = _run_boot(copyto_fail=True, write_pair=True)
    assert rc == 1, calls
    # [b2w] remote configured from the scoped write pair
    conf = os.path.join(home, ".config", "rclone", "rclone.conf")
    body = open(conf).read()
    assert "[b2w]" in body and "access_key_id = WKID" in body
    # the FAILED marker write routed via b2w: (the RO [b2] key can't write serve/)
    rcat = [c for c in calls if c.startswith("RCAT")]
    assert any("b2w:testbucket/serve/s1/SERVE_STATUS" in c for c in rcat), calls
