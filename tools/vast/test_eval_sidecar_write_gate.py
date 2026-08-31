"""Unit test for onstart/eval_sidecar.sh's B2 write gate.

On pair-key serve boxes the shipped [b2] key is bucket-wide READ-ONLY and the
write key is scoped to serve/ — every sidecar write targets evals/<RUN_ID>/ and
used to be `|| true`-swallowed, silently losing a whole session's farm output.
The gate probes the first real write (EVAL_STATUS) through [b2] then [b2w] and
refuses loudly when neither lands. Here we extract the marker-fenced gate
functions, drive them with a fake rclone whose per-remote verdicts we control,
and assert remote selection, [b2w] config, and the deny path. No network.
"""
import os
import re
import stat
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
SIDECAR = os.path.join(_HERE, "onstart", "eval_sidecar.sh")

FAKE_RCLONE = """#!/usr/bin/env bash
# fake rclone: log the call; deny when the target's remote is in FAKE_RCLONE_DENY
echo "$*" >> "${FAKE_RCLONE_LOG:?}"
cat > /dev/null 2>&1 || true
target="${2:-}"
for d in ${FAKE_RCLONE_DENY:-}; do
  case "$target" in "$d":*) exit 1 ;; esac
done
exit 0
"""


def _gate_block():
    src = open(SIDECAR).read()
    m = re.search(r"# BEGIN b2-write-gate.*?\n(.*)# END b2-write-gate", src, re.S)
    assert m, "b2-write-gate markers missing from eval_sidecar.sh"
    return m.group(1)


def _run_gate(tmp_path, deny="", write_pair=False):
    """Source the gate block under bash with a fake rclone; return (rc, out, log)."""
    home = tmp_path / "home"
    (home / ".config" / "rclone").mkdir(parents=True)
    (home / ".config" / "rclone" / "rclone.conf").write_text("[b2]\ntype = s3\n")
    fake = tmp_path / "bin" / "rclone"
    fake.parent.mkdir()
    fake.write_text(FAKE_RCLONE)
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    rlog = tmp_path / "rclone.log"
    rlog.write_text("")
    driver = tmp_path / "driver.sh"
    driver.write_text(
        "set -uo pipefail\n"
        "fail() { echo \"FAIL:$1\"; exit 9; }\n"
        + _gate_block()
        + "\npick_write_remote || fail \"denied\"\n"
        + "echo \"B2W=$B2W\"\n"
    )
    env = {
        "PATH": f"{fake.parent}:{os.environ['PATH']}",
        "HOME": str(home),
        "RUN_ID": "ridX",
        "B2_BUCKET": "bkt",
        "B2_S3_ENDPOINT": "https://s3.example",
        "B2_PROBE_BACKOFF": "0",
        "FAKE_RCLONE_LOG": str(rlog),
        "FAKE_RCLONE_DENY": deny,
    }
    if write_pair:
        env["B2_WRITE_KEY_ID"] = "wkid"
        env["B2_WRITE_APPLICATION_KEY"] = "wsec"
    p = subprocess.run(["bash", str(driver)], env=env, capture_output=True,
                       text=True, timeout=60)
    conf = (home / ".config" / "rclone" / "rclone.conf").read_text()
    return p, conf, rlog.read_text()


pytestmark = pytest.mark.skipif(not os.path.exists(SIDECAR),
                                reason="eval_sidecar.sh missing")


def test_b2_writable_selects_b2(tmp_path):
    p, conf, rlog = _run_gate(tmp_path)
    assert p.returncode == 0, p.stderr
    assert "B2W=b2:bkt" in p.stdout
    # the probe IS the status write — payload lands on EVAL_STATUS
    assert "rcat b2:bkt/evals/ridX/EVAL_STATUS" in rlog
    assert "[b2w]" not in conf  # no pair shipped -> no remote added


def test_ro_b2_falls_back_to_b2w(tmp_path):
    p, conf, _ = _run_gate(tmp_path, deny="b2", write_pair=True)
    assert p.returncode == 0, p.stderr
    assert "B2W=b2w:bkt" in p.stdout
    assert "[b2w]" in conf and "wkid" in conf


def test_both_denied_fails_loudly(tmp_path):
    # serve pair case: [b2] read-only AND [b2w] scoped to serve/ (evals/ 403s)
    p, _, rlog = _run_gate(tmp_path, deny="b2 b2w", write_pair=True)
    assert p.returncode == 9
    assert "FAIL:denied" in p.stdout
    # retried: 3 attempts x (b2 + b2w) probes
    assert rlog.count("rcat b2:bkt/evals/") == 3
    assert rlog.count("rcat b2w:bkt/evals/") == 3


def test_ro_b2_without_pair_fails(tmp_path):
    p, conf, _ = _run_gate(tmp_path, deny="b2", write_pair=False)
    assert p.returncode == 9
    assert "FAIL:denied" in p.stdout
    assert "[b2w]" not in conf


def test_no_write_bypasses_gate():
    """Every rclone write targeting evals/ must go through $B2W, not $B2."""
    bad = []
    for i, line in enumerate(open(SIDECAR), 1):
        if "rclone" in line and re.search(r'"\$B2/evals', line):
            bad.append((i, line.strip()))
    assert not bad, f"writes still routed via read-side [b2] remote: {bad}"

