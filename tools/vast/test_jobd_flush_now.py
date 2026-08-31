"""Unit tests for jobd.sh's CHECKPOINT_NOW marker — the operator-initiated flush.

WHY A SOURCED BLOCK, not the end-to-end harness: the same argument
test_jobd_ckpt_lifecycle.py makes. The behaviours that matter here are a
breadcrumb that did or did not appear and a B2 object that did or did not get
deleted, and both are invisible in an end-to-end run (nothing observable changes
about the job — that IS the feature). Driving `_flush_marker_consume` directly
lets each test assert exactly one of them.

jobd.sh cannot be sourced (it is a daemon), so the tests extract the block
between the FLUSHNOW_BEGIN/END sentinels and source THAT with `log` and `rclone`
stubbed. `test_sentinels_present` fails loudly if someone removes the sentinels,
so this file can never silently start testing an empty string.

The three CALL-SITE properties — the poll wiring, the flush arm's include/age
shape, and the tail-snapshot exclusion — are not reachable from the extracted
block, so they are pinned as source assertions instead. That is a weaker check
and is labelled as one.
"""
import os
import shutil
import subprocess
import textwrap

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
JOBD_SH = os.path.join(_HERE, "onstart", "jobd.sh")

pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")

_BEGIN = "# >>> FLUSHNOW_BEGIN"
_END = "# <<< FLUSHNOW_END"


def _flush_block():
    src = open(JOBD_SH).read()
    return src[src.index(_BEGIN):src.index(_END)]


def test_sentinels_present():
    """The extraction contract with jobd.sh."""
    src = open(JOBD_SH).read()
    assert src.count(_BEGIN) == 1 and src.count(_END) == 1
    assert "_flush_marker_consume() {" in _flush_block()


# --- harness -----------------------------------------------------------------
# A fake `rclone` covering the two ops the consumer uses (`lsf`, `deletefile`)
# against a directory tree standing in for B2. RCLONE_BREAK=<op> makes that one
# op fail, which is how the "delete failed" path is reached.
_RCLONE = r"""#!/usr/bin/env bash
set -u
B="$FAKE_BUCKET"
map() { case "$1" in b2*:*/*) echo "$B/${1#*:*/}" ;; *) echo "$1" ;; esac; }
op="$1"; shift
case "${RCLONE_BREAK:-}" in all) exit 7 ;; "$op") exit 7 ;; esac
t=""; for a in "$@"; do case "$a" in --*) ;; *) t="$a" ;; esac; done
p="$(map "$t")"
case "$op" in
  lsf)        [ -e "$p" ] || exit 0; basename "$p"; exit 0 ;;
  deletefile) [ -f "$p" ] || exit 1; rm -f "$p"; exit 0 ;;
esac
echo "unhandled $op" >&2; exit 2
"""

_PRELUDE = r"""
set -uo pipefail
log() { echo "LOG: $*" >&2; }
B2="b2:bkt"
B2W="b2w:bkt"
"""


def _run(tmp_path, script, env=None, timeout=30):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    rc = bindir / "rclone"
    rc.write_text(_RCLONE)
    rc.chmod(0o755)
    sf = tmp_path / "drive.sh"
    sf.write_text(_PRELUDE + _flush_block() + "\n" + textwrap.dedent(script))
    e = {"PATH": f"{bindir}:{os.environ.get('PATH', os.defpath)}",
         "FAKE_BUCKET": str(tmp_path / "bucket"), "HOME": str(tmp_path)}
    e.update(env or {})
    return subprocess.run(["bash", str(sf)], env=e, capture_output=True,
                          text=True, timeout=timeout)


def _mk_marker(tmp_path, jobid="j1"):
    d = tmp_path / "bucket" / "jobs" / jobid
    d.mkdir(parents=True, exist_ok=True)
    (d / "CHECKPOINT_NOW").write_text('{"v":1,"reason":"pre-park"}\n')
    return d / "CHECKPOINT_NOW"


_DRIVE = """
    mkdir -p "$W"
    if _flush_marker_consume j1 "$W/.checkpoint_now"; then echo RC=0; else echo RC=1; fi
    [ -f "$W/.checkpoint_now" ] && echo CRUMB=yes || echo CRUMB=no
"""


def test_no_marker_is_a_silent_no_op(tmp_path):
    """The steady state — one cheap lsf per poll, nothing else."""
    (tmp_path / "bucket").mkdir()
    r = _run(tmp_path, f'W={tmp_path}/w\n{_DRIVE}')
    assert "RC=1" in r.stdout, r.stdout + r.stderr
    assert "CRUMB=no" in r.stdout
    assert "CHECKPOINT_NOW" not in r.stderr      # nothing logged


def test_marker_drops_the_crumb_and_is_deleted(tmp_path):
    """Sighting -> flush requested -> marker consumed. The delete is what makes
    it at-most-once: the next poll must not re-fire on the same request."""
    m = _mk_marker(tmp_path)
    r = _run(tmp_path, f'W={tmp_path}/w\n{_DRIVE}')
    assert "RC=0" in r.stdout, r.stdout + r.stderr
    assert "CRUMB=yes" in r.stdout
    assert not m.exists(), "marker survived — the next poll would flush again"
    assert "CHECKPOINT_NOW seen" in r.stderr


def test_a_second_poll_after_consumption_does_not_re_fire(tmp_path):
    """At-most-once, driven twice through the real function rather than argued."""
    _mk_marker(tmp_path)
    r = _run(tmp_path, f"""
        W={tmp_path}/w
        mkdir -p "$W"
        _flush_marker_consume j1 "$W/.checkpoint_now" && echo FIRST=fired
        rm -f "$W/.checkpoint_now"
        _flush_marker_consume j1 "$W/.checkpoint_now" && echo SECOND=fired || echo SECOND=quiet
    """)
    assert "FIRST=fired" in r.stdout and "SECOND=quiet" in r.stdout, r.stdout


def test_a_failed_delete_still_requests_the_flush(tmp_path):
    """B2 has no CAS and the flush is idempotent, so a lost delete must cost an
    EXTRA flush, never a missed one. Refusing here would invert that."""
    _mk_marker(tmp_path)
    r = _run(tmp_path, f'W={tmp_path}/w\n{_DRIVE}', env={"RCLONE_BREAK": "deletefile"})
    assert "RC=0" in r.stdout and "CRUMB=yes" in r.stdout, r.stdout
    assert "DELETE failed" in r.stderr


def test_an_unreadable_b2_is_not_a_flush(tmp_path):
    """Fail-closed on the listing: an lsf that cannot answer must not be read as
    'a marker is there'."""
    _mk_marker(tmp_path)
    r = _run(tmp_path, f'W={tmp_path}/w\n{_DRIVE}', env={"RCLONE_BREAK": "lsf"})
    assert "RC=1" in r.stdout and "CRUMB=no" in r.stdout, r.stdout


def test_the_entrypoint_is_never_signalled(tmp_path):
    """A flush is not a stop. The consumer must contain no kill of any kind —
    checked on the source, because a test cannot prove the absence of a signal."""
    block = _flush_block()
    assert "kill" not in block
    assert "kill_tree" not in block


# --- call-site properties (source assertions; see the module docstring) -------

def _code():
    """jobd.sh with comment-only lines dropped — these assertions are about what
    the shell RUNS, and this file explains itself at length in prose that would
    otherwise match every one of them."""
    return "\n".join(l for l in open(JOBD_SH).read().splitlines()
                     if not l.lstrip().startswith("#"))


def test_the_poll_is_the_cancel_poll_and_does_not_break_it():
    """No second poll loop, and the flush must not `break` out of the cancel
    watch — that would stop watching for CANCEL after the first flush."""
    src = _code()
    i = src.index('if rclone lsf "$B2/jobs/$jobid/CANCEL"')
    seg = src[i:i + 1200]
    call = seg.index('_flush_marker_consume "$jobid"')
    assert "break" not in seg[call:seg.index("done ) &", call)]
    assert src.count("_flush_marker_consume") == 2      # definition + one call site


def test_the_flush_arm_ships_the_whole_glob_with_no_age_filter():
    """`page=()` is the no---min-age half and `pinc=("${inc[@]}")` the whole-glob
    half; a flush arm carrying the fire-on-arrival `pinc=()` would ship nothing."""
    src = _code()
    i = src.index('trig=flush-now')
    arm = src[i:i + 200]
    assert 'pinc=("${inc[@]}")' in arm and "page=()" in arm


def test_the_tail_snapshot_is_excluded_from_the_flush_arm():
    """The tail stage writes a cut-at-last-complete-line copy to the SAME key as
    the full push. On the age-filtered pass those files never shipped; on the
    flush arm they did, and the shorter copy would overwrite them."""
    src = _code()
    i = src.index('if [ "${JOBD_CKPT_TAIL:-1}" != "0" ]')
    assert '[ "${flushnow:-0}" != 1 ]' in src[i:i + 300]


def test_the_crumb_is_consumed_above_the_empty_glob_continue():
    """A job whose checkpoint globs match nothing hits `continue` before the arm
    selection. Reading the crumb after that point would re-arm the watch on every
    tick forever."""
    src = _code()
    consume = src.index('if [ -f "$wdir/.checkpoint_now" ]; then flushnow=1')
    guard = src.index('[ "${nmatch:-0}" -gt 0 ] || continue')
    assert consume < guard
