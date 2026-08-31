"""Canned-box test for onstart/preempt_trap.sh's `final_flush` emit (run-lane
handoff cutover fence) and its epoch guard on the bounded flush.

The run-lane handoff cutover (`_handoff_run_signals`) waits for a `final_flush`
event from the primary box. Only `jobd.sh` (jobs lane) emitted it originally;
this proves the run-lane box side now emits it too, AFTER its bounded final
checkpoint flush, on the SIGTERM/preempt path (HANDOFF_DESIGN). We source the
REAL shipped `_preempt_trap` and invoke it with stubbed `emit_event`/`rclone`/
`timeout`, asserting the emit ORDER (`preempted` before the flush, `final_flush`
after) and the two-writer guard: when the caller's `_handoff_epoch_stale`
(train.sh, promoted-keyed) reports STALE, the checkpoint BYTES are skipped but
the events still fire; when not stale — or when the function is undefined
(standalone source, every non-handoff run) — the flush proceeds exactly as
before.

Assumes `/workspace/.run_terminal` is absent and `RC` unset (the genuinely
mid-training path); true in the portable lane. Skipped when bash is unavailable.
"""
import os
import shutil
import subprocess
import tempfile

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
TRAP_SH = os.path.join(_HERE, "onstart", "preempt_trap.sh")

pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")


def _run_trap(prelude=""):
    """Source the real preempt_trap.sh, call _preempt_trap with stubs, return
    (ordered emitted event names, ordered stubbed rclone invocations). `prelude`
    injects extra definitions (e.g. a canned _handoff_epoch_stale) before the
    source, mimicking train.sh's dynamic scope."""
    with tempfile.TemporaryDirectory() as d:
        events = os.path.join(d, "events")
        calls = os.path.join(d, "calls")
        ckpt = os.path.join(d, "ckpt")
        os.makedirs(ckpt)
        script = (
            "set -uo pipefail\n"
            f'CKPT_DIR="{ckpt}"; B2="b2:testbucket"; RUN_ID="r1"\n'
            f'EVENTS="{events}"; CALLS="{calls}"\n'
            'emit_event() { echo "$1" >> "$EVENTS"; return 0; }\n'
            '_emit_terminal() { echo "terminal:$1" >> "$EVENTS"; return 0; }\n'
            'rclone() { echo "rclone $*" >> "$CALLS"; return 0; }\n'
            # `timeout 45 rclone ...` must reach the rclone FUNCTION above; the
            # real /usr/bin/timeout can only exec binaries, so stub it too.
            'timeout() { shift; "$@"; }\n'
            + prelude
            + f'. "{TRAP_SH}"\n'          # defines _preempt_trap + arms the trap
            "_preempt_trap\n"             # invoke directly (it exits 143)
        )
        subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                       timeout=30)
        evs = []
        if os.path.exists(events):
            evs = [ln.strip() for ln in open(events) if ln.strip()]
        cls = []
        if os.path.exists(calls):
            cls = [ln.strip() for ln in open(calls) if ln.strip()]
        return evs, cls


def test_preempt_trap_emits_final_flush_after_preempted():
    evs, calls = _run_trap()
    assert "preempted" in evs, evs
    assert "final_flush" in evs, evs
    # order: preempted first (a preempted run is not failed), final_flush AFTER the
    # bounded flush so the newest checkpoint bytes are on B2 before the understudy pulls
    assert evs.index("final_flush") > evs.index("preempted"), evs
    # never a terminal on the mid-training preempt path
    assert not any(e.startswith("terminal:") for e in evs), evs
    # no guard function defined (standalone source) -> fail-safe: flush proceeds
    assert any(c.startswith("rclone copy") for c in calls), calls


def test_preempt_trap_stale_epoch_skips_bytes_keeps_events():
    # a superseded box (newer epoch PROMOTED over it) must not clobber the
    # understudy's checkpoints — bytes skipped, both events still emitted in order.
    evs, calls = _run_trap(prelude="_handoff_epoch_stale() { return 0; }\n")
    assert not any(c.startswith("rclone copy") for c in calls), calls
    assert "preempted" in evs and "final_flush" in evs, evs
    assert evs.index("final_flush") > evs.index("preempted"), evs


def test_preempt_trap_current_epoch_flushes():
    # not stale (the fence-park final flush happens PRE-promotion) -> bytes go out.
    evs, calls = _run_trap(prelude="_handoff_epoch_stale() { return 1; }\n")
    assert any(c.startswith("rclone copy") for c in calls), calls
    assert "final_flush" in evs, evs
