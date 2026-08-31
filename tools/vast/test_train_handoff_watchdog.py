"""Canned-box test for onstart/train.sh's handoff dead-man watchdog (T6).

train.sh has no full test harness (it pulls a runset, trains, tears down a real
box), so we exercise ONLY the watchdog: the block between the
`# >>> handoff-deadman-watchdog` / `# <<< handoff-deadman-watchdog` sentinels is
extracted VERBATIM and run under bash with stubbed `rclone` / `self_park` /
`emit_event`. This tests the real shipped code, not a copy.

Guard contract (HANDOFF_DESIGN §6): an understudy launched with HANDOFF_TTL_S
self-parks at the TTL UNLESS a promotion marker (runs/<RUN_ID>/handoff/promoted)
appeared — and is a pure no-op when HANDOFF_TTL_S is unset (every normal run).

Skipped when bash is unavailable.
"""
import os
import re
import shutil
import subprocess
import tempfile

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SH = os.path.join(_HERE, "onstart", "train.sh")

pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")


def _watchdog_block():
    src = open(TRAIN_SH).read()
    m = re.search(r"# >>> handoff-deadman-watchdog.*?# <<< handoff-deadman-watchdog",
                  src, re.S)
    assert m, "handoff-deadman-watchdog sentinels not found in train.sh"
    return m.group(0)


def _run(*, ttl_env, promoted):
    """Run the extracted watchdog with a small TTL. `promoted` toggles whether the
    stub `rclone lsf ...promoted` reports the marker. Returns True iff self_park fired."""
    block = _watchdog_block()
    rclone_body = "echo promoted; return 0" if promoted else "return 0"
    with tempfile.TemporaryDirectory() as d:
        park = os.path.join(d, "parked")
        env_line = f'export HANDOFF_TTL_S="{ttl_env}"\n' if ttl_env is not None else ""
        script = (
            "set -uo pipefail\n"
            'B2="b2:testbucket"; RUN_ID="r1"\n'
            f'PARK_MARK="{park}"\n'
            f"rclone() {{ {rclone_body}; }}\n"
            'self_park() { echo parked > "$PARK_MARK"; }\n'
            "emit_event() { return 0; }\n"
            + env_line
            + block
            + "\nwait 2>/dev/null || true\n"
        )
        subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                       timeout=30)
        return os.path.exists(park)


def test_watchdog_self_parks_after_ttl_without_promotion():
    # TTL elapses, no promotion marker -> supervisor presumed dead -> self-park.
    assert _run(ttl_env="1", promoted=False) is True


def test_watchdog_stays_up_when_promoted():
    # promotion marker present at TTL -> this box is canonical -> must NOT park.
    assert _run(ttl_env="1", promoted=True) is False


def test_watchdog_is_noop_without_env():
    # HANDOFF_TTL_S unset (normal run) -> watchdog never arms -> never parks.
    assert _run(ttl_env=None, promoted=False) is False
