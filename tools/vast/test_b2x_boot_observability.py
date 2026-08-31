"""`b2x_boot.sh`: the transport a call site actually used is observable.

Runs the shim with a STUB `b2x` first on `PATH` that writes the `--stats-env`
file the real binary writes and exits with whatever rc the test asks for. No
network, no B2.

The property under test is not throughput, it is *legibility*. Until 2026-08-25
the shim logged only failures, and the jobs lane shipped no shim at all — so
`b2x_pull` was `return 1`, every site fell through to its rclone line, and from
off-box that was indistinguishable from b2x working. Quiet read as healthy. The
four things pinned here are what make the difference visible:

  * a SUCCESS says so, and carries the byte/rate/stream figures;
  * `B2X_LAST_STREAMS` is exported — the concurrency b2x actually used, which is
    the only witness that a `B2X_CONCURRENCY` override took, since the env var
    is parsed with a positive-int guard that silently keeps the default;
  * the tally separates `fallback` (b2x ran and failed) from `unavailable`
    (b2x was never there) — the second is a delivery bug, the first is not;
  * `B2X_DISABLE=1` announces itself ONCE, instead of returning 1 in silence.

Callers run under `set -u`, so every path is exercised with no optional args.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SHIM = HERE / "onstart" / "b2x_boot.sh"

# Mirrors stats.go's --stats-env body. `prev` walk rather than getopt: the shim
# appends the flag, so its position is not fixed.
STUB_B2X = """#!/usr/bin/env bash
# Mimics Go's flag package, which STOPS parsing at the first non-flag argument.
# A permissive stub is why the first version of this file passed while every
# real call site was exiting 2: `b2x pull SRC DST --exclude X` leaves four
# positionals, fails the arity check and never transfers a byte.
sub="$1"; shift
seen_pos=0
se=""; prev=""
for a in "$@"; do
  case "$a" in
    --*) [ "$seen_pos" = 1 ] && { echo "b2x: usage (flag after positional)" >&2; exit 2; } ;;
    *)   [ "$prev" = "--stats-env" ] || [ "$prev" = "--exclude" ] || \
         [ "$prev" = "--include" ] || [ "$prev" = "--deadline" ] || seen_pos=1 ;;
  esac
  [ "$prev" = "--stats-env" ] && se="$a"
  prev="$a"
done
[ -n "$se" ] && printf 'B2X_BYTES=1234\\nB2X_SECS=2.0\\nB2X_MBPS=617.0\\nB2X_OBJECTS=3\\nB2X_SKIPPED=0\\nB2X_SKIPPED_BYTES=0\\nB2X_STREAMS=192\\nB2X_RETRIES=0\\nB2X_VERDICT=ok\\n' > "$se"
exit "${FAKE_RC:-0}"
"""


@pytest.fixture
def run(tmp_path):
    """Source the shim with a stub b2x and run `body`. Returns (rc, out, tally)."""
    stub = tmp_path / "b2x"
    stub.write_text(STUB_B2X)
    stub.chmod(0o755)
    tally = tmp_path / "tally"

    def _run(body: str, env: dict | None = None, stub_ensure: bool = True):
        # stub_ensure skips the 4-rung install ladder: it is not what this file
        # tests and it would reach for the network. Pass False when the property
        # under test LIVES in b2x_ensure (the B2X_DISABLE branch) — stubbing it
        # there replaces the code being tested, which is how the first draft of
        # this file passed a test that exercised nothing.
        ensure = f"b2x_ensure() {{ B2X={stub}; return 0; }}\n" if stub_ensure else ""
        script = (
            "set -uo pipefail\n"
            f"B2X_TALLY={tally}\n"
            f". {SHIM}\n"
            f"{ensure}"
            f"{body}\n"
        )
        p = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                           env={"PATH": "/usr/bin:/bin", **(env or {})})
        return p.returncode, p.stdout + p.stderr, (
            tally.read_text() if tally.exists() else "")

    return _run


def test_success_is_logged_with_figures(run):
    """Quiet must not be the only signal for healthy — say so, with numbers."""
    rc, out, tally = run('b2x_pull b2:bkt/src /tmp/dst')
    assert rc == 0
    assert "OK via b2x" in out
    assert "617.0 MB/s" in out and "192 streams" in out
    assert tally.startswith("ok\t1234\t")


def test_exports_last_streams_as_the_override_witness(run):
    """B2X_CONCURRENCY is silently ignored unless positive-int; streams proves it."""
    rc, out, _ = run('b2x_pull b2:bkt/src /tmp/dst && echo "S=$B2X_LAST_STREAMS M=$B2X_LAST_MBPS"')
    assert rc == 0
    assert "S=192 M=617.0" in out


def test_failure_tallies_fallback_not_unavailable(run):
    """b2x ran and failed. Distinct from never having been delivered."""
    rc, out, tally = run('FAKE_RC=5 b2x_pull b2:bkt/src /tmp/dst; echo "rc=$?"')
    assert "falling back to rclone" in out
    assert "rc=1" in out
    assert tally.startswith("fallback\t")


def test_missing_b2x_tallies_unavailable(run, tmp_path):
    """The shape that hid the jobs-lane defect: no binary, so no transfer at all."""
    rc, out, tally = run('b2x_ensure() { return 1; }\n'
                         'b2x_pull b2:bkt/src /tmp/dst; echo "rc=$?"')
    assert "rc=1" in out
    assert tally.startswith("unavailable\t")
    assert "ok\t" not in tally


def test_disable_announces_itself_once(run):
    """Three calls, one announcement — loud enough to see, quiet enough to keep."""
    rc, out, tally = run('b2x_pull a b; b2x_pull a b; b2x_pull a b\n'
                         'b2x_tally_summary', env={"B2X_DISABLE": "1"},
                         stub_ensure=False)
    assert out.count("B2X_DISABLE=1") == 1
    assert "disabled=1 unavailable=3" in out
    assert "ok=0" in out, "the kill switch must actually stop the transfer"


def test_callers_stats_env_is_reused_not_clobbered(run, tmp_path):
    """A site that wants the file at a known path keeps it — and we still read it."""
    mine = tmp_path / "mine"
    rc, out, _ = run(f'b2x_pull b2:bkt/src /tmp/dst --stats-env {mine}')
    assert rc == 0
    assert mine.exists(), "caller's path must be the one b2x wrote"
    assert "617.0 MB/s" in out, "figures must still reach the log line"


def test_summary_reports_zeroes_when_nothing_ran(run, tmp_path):
    """Absence has to be readable, not an error — it is the alarm condition."""
    rc, out, _ = run('B2X_TALLY=%s b2x_tally_summary' % (tmp_path / "nope"))
    assert "ok=0" in out and "bytes=0" in out


def test_flags_are_passed_before_the_positionals(run):
    """Go's flag parser stops at the first positional, so order is not cosmetic.

    Every call site in the repo wrote `b2x pull SRC DST --exclude X` and every
    one of them exited 2 without transferring a byte, then fell back to rclone
    and succeeded — so the site looked healthy and simply never used b2x. The
    stub above now enforces the real parser's rule; this asserts the wrapper
    reorders on the caller's behalf.
    """
    rc, out, tally = run('b2x_pull b2:bkt/src /tmp/dst --exclude FOO --deadline 40s')
    assert rc == 0, f"wrapper must put flags first; got:\n{out}"
    assert "OK via b2x" in out
    assert tally.startswith("ok\t")
