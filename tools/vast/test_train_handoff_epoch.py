"""Canned-box test for onstart/train.sh's `_handoff_epoch_stale` (T6 two-writer
fence predicate), promoted-keyed.

Extraction mirrors test_train_handoff_watchdog.py: the block between the
`# >>> handoff-epoch-stale` / `# <<< handoff-epoch-stale` sentinels is run
VERBATIM under bash with a stubbed `rclone` — this tests the real shipped code.

Contract: rc 0 == STALE (a `promoted` marker names an epoch strictly greater
than this box's HANDOFF_EPOCH — write ownership transferred at PROMOTION);
rc 1 == ok to push. Keyed on the promoted marker, NOT the ARM-time
<epoch>.json max, so a still-canonical primary keeps syncing through a second
handoff's ARM->fence window and an ABORTED attempt (which leaves its ARM marker
behind) never silences the survivor. FAIL-SAFE: unset HANDOFF_EPOCH, missing/
unreadable promoted marker, or unparsable epoch field => not stale.

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


def _epoch_block():
    src = open(TRAIN_SH).read()
    m = re.search(r"# >>> handoff-epoch-stale.*?# <<< handoff-epoch-stale",
                  src, re.S)
    assert m, "handoff-epoch-stale sentinels not found in train.sh"
    return m.group(0)


def _stale(*, epoch, promoted):
    """Run the extracted predicate. `epoch` -> HANDOFF_EPOCH ('' == unset case);
    `promoted` is the marker body served by the rclone-cat stub (None == marker
    absent, cat fails). Returns True iff the predicate reported STALE (rc 0)."""
    block = _epoch_block()
    with tempfile.TemporaryDirectory() as d:
        if promoted is None:
            rclone = 'rclone() { return 1; }\n'
        else:
            marker = os.path.join(d, "promoted")
            with open(marker, "w") as fh:
                fh.write(promoted)
            rclone = f'rclone() {{ cat "{marker}"; }}\n'
        script = (
            "set -uo pipefail\n"
            'B2="b2:testbucket"; RUN_ID="r1"\n'
            f'HANDOFF_EPOCH="{epoch}"\n'
            + rclone
            + block
            + "\n_handoff_epoch_stale\n"
        )
        r = subprocess.run(["bash", "-c", script], capture_output=True,
                           text=True, timeout=30)
        return r.returncode == 0


def test_unset_epoch_never_stale():
    # every normal run (and a same-RUN_ID rerun launched without the env) even
    # with a leftover promoted marker from an earlier campaign.
    assert _stale(epoch="", promoted='{"epoch":2}') is False


def test_no_promoted_marker_not_stale():
    # ARM markers alone must NOT silence the canonical primary (second-handoff
    # window + aborted-attempt leftovers).
    assert _stale(epoch="1", promoted=None) is False


def test_newer_promotion_is_stale():
    assert _stale(epoch="1", promoted='{"run_id":"r1","understudy":"999",'
                                      '"epoch":2,"promoted_at":"t",'
                                      '"reason":"post_flush"}') is True


def test_own_promotion_not_stale():
    # the promoted understudy itself (marker names ITS epoch) keeps pushing.
    assert _stale(epoch="2", promoted='{"epoch":2}') is False


def test_unparsable_marker_not_stale():
    assert _stale(epoch="1", promoted="garbage, no epoch field") is False


def test_spaced_epoch_field_parses():
    assert _stale(epoch="1", promoted='{ "epoch": 3 }') is True


# ---------------------------------------------------------------------------
# handoff-synced-marker: the understudy's box-side boot proof (D3 fix, live
# canary 2026-07-15). The block between `# >>> handoff-synced-marker` /
# `# <<< handoff-synced-marker` runs after the checkpoint resume pull and must
# rcat runs/<RUN_ID>/handoff/<HANDOFF_EPOCH>.synced — the driver's SYNCED gate
# keys on this marker, never on API liveness.
# ---------------------------------------------------------------------------
def _synced_block():
    src = open(TRAIN_SH).read()
    m = re.search(r"# >>> handoff-synced-marker.*?# <<< handoff-synced-marker",
                  src, re.S)
    assert m, "handoff-synced-marker sentinels not found in train.sh"
    return m.group(0)


def _run_synced(epoch):
    """Run the extracted block; returns (rcat_target, body) or (None, None) when
    no rcat was issued. `epoch` '' == HANDOFF_EPOCH unset (a normal primary)."""
    block = _synced_block()
    with tempfile.TemporaryDirectory() as d:
        cap = os.path.join(d, "cap")
        rclone = (
            'rclone() {\n'
            '  if [ "$1" = "rcat" ]; then\n'
            f'    echo "$2" > "{cap}.path"; cat > "{cap}.body"; return 0\n'
            '  fi\n  return 1\n}\n'
        )
        script = (
            "set -uo pipefail\n"
            'B2="b2:testbucket"; RUN_ID="r-sync"\n'
            f'HANDOFF_EPOCH="{epoch}"\n'
            + rclone + block + "\n"
        )
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        if not os.path.exists(cap + ".path"):
            return None, None
        return (open(cap + ".path").read().strip(),
                open(cap + ".body").read())


def test_synced_marker_written_for_understudy_epoch():
    path, body = _run_synced(epoch="2")
    assert path == "b2:testbucket/runs/r-sync/handoff/2.synced"
    assert '"epoch":2' in body and '"run_id":"r-sync"' in body


def test_synced_marker_skipped_without_handoff_epoch():
    # a normal primary (no HANDOFF_EPOCH) must not write understudy boot proofs
    path, body = _run_synced(epoch="")
    assert path is None
