"""Unit tests for jobd.sh's BOX-DISK checkpoint lifecycle (delete-after-sync,
end-of-run scrub, fire-on-arrival sync trigger).

WHY A SOURCED BLOCK, not the end-to-end harness. The failure modes that matter
here are all *refusals* — "B2 could not be read, so nothing was deleted" — and a
refusal is invisible in an end-to-end run (the disk simply still has the files,
which is also what a no-op looks like). Driving the functions directly lets each
test assert the exact directory that survived and the exact reason logged.
jobd.sh cannot be sourced (it is a daemon: `set -uo pipefail`, then a poll loop),
so the tests extract the block between the CKPT_LIFECYCLE_BEGIN/END sentinels and
source THAT, with `log`, `rclone` and `_handoff_epoch_stale` stubbed. The
sentinels are declared in jobd.sh; `test_sentinels_present` fails loudly if
someone removes them, so this file can never silently start testing nothing.

The stub `rclone` is a real local-B2 (a directory tree), so `lsf -R` and
`size --json` answer from actual bytes — a truncated or missing remote file is a
truncated or missing file on disk, not a mock's opinion.

test_jobd.py covers the same code end to end (a real jobd run that checkpoints).
"""
import json
import os
import shlex
import shutil
import subprocess
import textwrap

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
JOBD_SH = os.path.join(_HERE, "onstart", "jobd.sh")

pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")

_BEGIN = "# >>> CKPT_LIFECYCLE_BEGIN"
_END = "# <<< CKPT_LIFECYCLE_END"


def _lifecycle_block():
    src = open(JOBD_SH).read()
    i = src.index(_BEGIN)
    j = src.index(_END)
    return src[i:j]


def test_sentinels_present():
    """The extraction contract with jobd.sh. If this fails, every other test in
    this file is testing an empty string."""
    src = open(JOBD_SH).read()
    assert src.count(_BEGIN) == 1 and src.count(_END) == 1
    block = _lifecycle_block()
    for fn in ("_ckpt_step", "_ckpt_quiescent", "_ckpt_write_complete",
               "_ckpt_b2_verified", "_ckpt_dirs_from_matchlist", "_ckpt_safe_rel",
               "_ckpt_prune_synced", "_ckpt_scrub_local", "_ckpt_new_ready",
               "_ckpt_all_dirs", "_ckpt_names_complete", "_ckpt_marker_rel",
               "_ckpt_marker_json", "_ckpt_publish_marker", "_ckpt_mark_complete"):
        assert f"{fn}() {{" in block, fn


# --- the harness -------------------------------------------------------------
# A fake `rclone` supporting exactly the four ops this block uses (`lsf -R`,
# `size --json`, `cat`, and `rcat` for the completion marker), mapping
# b2:<bucket>/<key> onto $FAKE_BUCKET. Kept separate from testlib/rclone_shim.sh
# on purpose: these tests need to BREAK individual ops
# (RCLONE_BREAK=lsf|size|cat|rcat|all) to prove the fail-safe path, which the
# shared shim must never learn to do.
_RCLONE = r"""#!/usr/bin/env bash
set -u
B="$FAKE_BUCKET"
map() { case "$1" in b2*:*/*) echo "$B/${1#*:*/}" ;; *) echo "$1" ;; esac; }
op="$1"; shift
case "${RCLONE_BREAK:-}" in
  all) exit 7 ;;
  "$op") exit 7 ;;
esac
case "$op" in
  cat)
    p="$(map "$1")"; [ -f "$p" ] || exit 1; cat "$p"; exit 0 ;;
  rcat)
    p="$(map "$1")"; mkdir -p "$(dirname "$p")" || exit 1; cat > "$p"; exit 0 ;;
  lsf)
    t=""; for a in "$@"; do case "$a" in --*) ;; *) t="$a" ;; esac; done
    p="$(map "$t")"; [ -d "$p" ] || exit 0
    ( cd "$p" && find . -type f -printf '%P\n' 2>/dev/null )
    exit 0 ;;
  size)
    t=""; for a in "$@"; do case "$a" in --*) ;; *) t="$a" ;; esac; done
    p="$(map "$t")"; [ -d "$p" ] || exit 1
    n="$(find "$p" -type f 2>/dev/null | wc -l | tr -d ' ')"
    b="$(find "$p" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')"
    echo "{\"count\":$n,\"bytes\":$b}"
    exit 0 ;;
esac
echo "unhandled $op" >&2; exit 2
"""

_PRELUDE = r"""
set -uo pipefail
log() { echo "LOG: $*" >&2; }
B2="b2:bkt"
B2W="b2:bkt"
HANDOFF_EPOCH="${HANDOFF_EPOCH:-}"
# Stub: STALE iff HANDOFF_STALE=1. The real predicate reads a `promoted` marker
# off B2; what these tests pin is that the prune/scrub HONOUR it, not how it is
# computed (test_jobd.py covers the real one).
_handoff_epoch_stale() { [ "${HANDOFF_STALE:-0}" = "1" ]; }
"""


def _run(tmp_path, script, env=None, timeout=60):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    rc = bindir / "rclone"
    rc.write_text(_RCLONE)
    rc.chmod(0o755)
    body = _PRELUDE + _lifecycle_block() + "\n" + textwrap.dedent(script)
    sf = tmp_path / "drive.sh"
    sf.write_text(body)
    e = {
        "PATH": f"{bindir}:{os.environ.get('PATH', os.defpath)}",
        "FAKE_BUCKET": str(tmp_path / "bucket"),
        "HOME": str(tmp_path),
        # settle window off unless a test asks for it: these fixtures write files
        # microseconds before the call and are not racing a trainer.
        "JOBD_CKPT_SETTLE_S": "0",
    }
    e.update(env or {})
    return subprocess.run(["bash", str(sf)], env=e, capture_output=True,
                          text=True, timeout=timeout)


def _mk_ckpt(root, rel, files=(("adapter_model.safetensors", 4096),
                               ("optimizer.pt", 8192),
                               ("trainer_state.json", 64))):
    d = os.path.join(root, rel)
    os.makedirs(d, exist_ok=True)
    for name, size in files:
        with open(os.path.join(d, name), "wb") as f:
            f.write(b"x" * size)
    return d


def _mk_job(tmp_path, jobid="j1", steps=(10, 20, 30, 40), on_b2=None):
    """A run dir with out/checkpoint-<step>/ dirs, a matchlist naming their files,
    and (by default) a byte-identical copy of every one of them on the fake B2."""
    run = tmp_path / "run"
    for s in steps:
        _mk_ckpt(str(run), f"out/checkpoint-{s}")
    bucket = tmp_path / "bucket" / "jobs" / jobid / "checkpoints"
    for s in (steps if on_b2 is None else on_b2):
        shutil.copytree(run / "out" / f"checkpoint-{s}",
                        bucket / "out" / f"checkpoint-{s}")
    ml = tmp_path / "matchlist"
    ml.write_text("".join(
        f"out/checkpoint-{s}/{n}\n"
        for s in steps
        for n in ("adapter_model.safetensors", "optimizer.pt", "trainer_state.json")))
    return run, ml


_REPORT = 'echo "N=$CKPT_PRUNE_N BYTES=$CKPT_PRUNE_BYTES LIST=$CKPT_PRUNE_LIST"'


def _survivors(run):
    return sorted(os.path.basename(p) for p in os.listdir(os.path.join(run, "out")))


# --- lever 1: delete-after-sync ---------------------------------------------

def test_prune_keeps_newest_two_and_deletes_the_verified_rest(tmp_path):
    run, ml = _mk_job(tmp_path)
    r = _run(tmp_path, f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}')
    assert r.returncode == 0, r.stderr
    assert "N=2 " in r.stdout, (r.stdout, r.stderr)
    # newest TWO survive — the invariant resume_pull.sh's newest-2 pull depends on
    assert _survivors(run) == ["checkpoint-30", "checkpoint-40"]
    assert "checkpoint-10" in r.stdout and "checkpoint-20" in r.stdout


def test_prune_orders_by_step_number_not_lexically(tmp_path):
    """checkpoint-100 is NEWER than checkpoint-90. A lexical sort would delete the
    newest checkpoint and keep two stale ones — the worst possible bug here."""
    run, ml = _mk_job(tmp_path, steps=(9, 90, 100, 1000))
    r = _run(tmp_path, f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}')
    assert _survivors(run) == ["checkpoint-100", "checkpoint-1000"], r.stdout


def test_prune_is_per_layout_root(tmp_path):
    """arms/<name>/checkpoint-N/ is a second root (resume_pull.sh walks both);
    newest-2 is kept in EACH, not two across the whole run."""
    run = tmp_path / "run"
    for s in (10, 20, 30):
        _mk_ckpt(str(run), f"out/checkpoint-{s}")
        _mk_ckpt(str(run), f"out/arms/hex/checkpoint-{s}")
    shutil.copytree(run, tmp_path / "bucket" / "jobs" / "j1" / "checkpoints")
    ml = tmp_path / "ml"
    ml.write_text("".join(
        f"{p}/checkpoint-{s}/trainer_state.json\n{p}/checkpoint-{s}/optimizer.pt\n"
        f"{p}/checkpoint-{s}/adapter_model.safetensors\n"
        for p in ("out", "out/arms/hex") for s in (10, 20, 30)))
    r = _run(tmp_path, f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}')
    assert "N=2 " in r.stdout, (r.stdout, r.stderr)
    assert sorted(os.listdir(run / "out" / "arms" / "hex")) == \
        ["checkpoint-20", "checkpoint-30"]
    assert sorted(p for p in os.listdir(run / "out") if p.startswith("checkpoint")) == \
        ["checkpoint-20", "checkpoint-30"]


def test_prune_refuses_when_the_dir_is_absent_from_b2(tmp_path):
    run, ml = _mk_job(tmp_path, on_b2=(20, 30, 40))     # checkpoint-10 never shipped
    r = _run(tmp_path, f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}')
    # checkpoint-20 IS durable and goes; checkpoint-10 is not and stays.
    assert "N=1 " in r.stdout and "LIST=out/checkpoint-20" in r.stdout, r.stdout
    assert "checkpoint-10" in _survivors(run)
    assert "name-set MISMATCH" in r.stderr


def test_prune_refuses_when_one_remote_file_is_missing(tmp_path):
    run, ml = _mk_job(tmp_path)
    os.remove(tmp_path / "bucket" / "jobs" / "j1" / "checkpoints" / "out" /
              "checkpoint-10" / "optimizer.pt")
    r = _run(tmp_path, f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}')
    assert "N=1 " in r.stdout, (r.stdout, r.stderr)          # only checkpoint-20 went
    assert "checkpoint-10" in _survivors(run)


def test_prune_refuses_a_truncated_remote_file(tmp_path):
    """The torn-upload case: every NAME is present on B2 but one object holds
    fewer bytes than the local file. Name-presence alone would delete it."""
    run, ml = _mk_job(tmp_path)
    tgt = (tmp_path / "bucket" / "jobs" / "j1" / "checkpoints" / "out" /
           "checkpoint-10" / "optimizer.pt")
    tgt.write_bytes(b"x" * 128)
    r = _run(tmp_path, f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}')
    assert "N=1 " in r.stdout, (r.stdout, r.stderr)
    assert "checkpoint-10" in _survivors(run)
    assert "byte-total MISMATCH" in r.stderr


def test_prune_refuses_a_same_shape_checkpoint_from_a_DIFFERENT_lineage(tmp_path):
    """Adversarial-review finding: a LoRA checkpoint's file names and per-file sizes
    are fixed by the run's CONFIG, not its weights — so two independent training
    lineages at the same step have an identical name set AND an identical byte
    total. Name+size alone would accept either as evidence for the other (reachable
    via a `job retarget` whose source box is still running with HANDOFF_EPOCH
    unset). trainer_state.json is a few KB and its loss history identifies the
    lineage, so its sha256 is read back too."""
    run, ml = _mk_job(tmp_path)
    rem = (tmp_path / "bucket" / "jobs" / "j1" / "checkpoints" / "out" /
           "checkpoint-10" / "trainer_state.json")
    local = run / "out" / "checkpoint-10" / "trainer_state.json"
    assert rem.stat().st_size == local.stat().st_size          # same SHAPE...
    rem.write_bytes(b"y" * local.stat().st_size)               # ...different RUN
    r = _run(tmp_path, f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}')
    assert "N=1 " in r.stdout and "LIST=out/checkpoint-20" in r.stdout, r.stdout
    assert "checkpoint-10" in _survivors(run)
    assert "trainer_state.json CONTENT mismatch" in r.stderr


def test_prune_deletes_nothing_when_the_content_read_back_is_unavailable(tmp_path):
    run, ml = _mk_job(tmp_path)
    r = _run(tmp_path, f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}',
             env={"RCLONE_BREAK": "cat"})
    assert "N=0 " in r.stdout, (r.stdout, r.stderr)
    assert len(_survivors(run)) == 4


def test_prune_still_works_without_a_trainer_state_marker(tmp_path):
    """A non-HF checkpointer has no trainer_state.json; the content check is then
    skipped and name-set + byte-total carry the verification, as before."""
    run = tmp_path / "run"
    for s in (10, 20, 30):
        _mk_ckpt(str(run), f"out/checkpoint-{s}", files=(("weights.bin", 4096),))
    shutil.copytree(run / "out", tmp_path / "bucket" / "jobs" / "j1" /
                    "checkpoints" / "out")
    ml = tmp_path / "ml"
    ml.write_text("".join(f"out/checkpoint-{s}/weights.bin\n" for s in (10, 20, 30)))
    r = _run(tmp_path, f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}')
    assert "N=1 " in r.stdout and "LIST=out/checkpoint-10" in r.stdout, r.stdout


def test_prune_refuses_when_the_size_probe_reports_a_ZERO_byte_prefix(tmp_path):
    """Against a real S3/B2 remote an ABSENT prefix is NOT an error: `size --json`
    answers {"count":0,"bytes":0} and exits 0. Both test shims model it as a
    non-zero exit, so this pins the production shape explicitly — the size gate
    must refuse a zero total rather than degrade to a no-op."""
    run, ml = _mk_job(tmp_path)
    # a size op that always answers the real rc-0 / zero-count shape
    zero = tmp_path / "bin" / "rclone"
    tmp_path.joinpath("bin").mkdir(exist_ok=True)
    zero.write_text(_RCLONE.replace(
        'echo "{\\"count\\":$n,\\"bytes\\":$b}"',
        'echo "{\\"count\\":0,\\"bytes\\":0}"'))
    zero.chmod(0o755)
    body = _PRELUDE + _lifecycle_block() + \
        f'\n_ckpt_b2_verified "{run}/out/checkpoint-10" "$B2/jobs/j1/checkpoints/out/checkpoint-10"; echo "RC=$?"\n'
    sf = tmp_path / "zero.sh"
    sf.write_text(body)
    r = subprocess.run(
        ["bash", str(sf)],
        env={"PATH": f"{tmp_path / 'bin'}:{os.environ.get('PATH', os.defpath)}",
             "FAKE_BUCKET": str(tmp_path / "bucket"), "HOME": str(tmp_path),
             "JOBD_CKPT_SETTLE_S": "0"},
        capture_output=True, text=True, timeout=60)
    assert "RC=1" in r.stdout, (r.stdout, r.stderr)
    assert "UNAVAILABLE or empty" in r.stderr


def test_prune_names_the_superset_case_distinctly(tmp_path):
    """`rclone copy` never deletes, so the remote accumulates the union of every
    attempt's file names — a resume at a different world size leaves extra
    rng_state_<i>.pth and WEDGES the prune for that dir. Correct, but it has to be
    diagnosable from the log without deriving it from two counts."""
    run, ml = _mk_job(tmp_path)
    extra = (tmp_path / "bucket" / "jobs" / "j1" / "checkpoints" / "out" /
             "checkpoint-10" / "rng_state_7.pth")
    extra.write_bytes(b"z" * 16)
    r = _run(tmp_path, f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}')
    assert "checkpoint-10" in _survivors(run)
    assert "strict SUPERSET" in r.stderr and "WEDGED" in r.stderr


def test_prune_deletes_nothing_when_the_listing_op_is_unavailable(tmp_path):
    """Verification UNAVAILABLE must fail safe, not fail open."""
    run, ml = _mk_job(tmp_path)
    r = _run(tmp_path, f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}',
             env={"RCLONE_BREAK": "lsf"})
    assert "N=0 " in r.stdout, (r.stdout, r.stderr)
    assert _survivors(run) == ["checkpoint-10", "checkpoint-20",
                               "checkpoint-30", "checkpoint-40"]


def test_prune_deletes_nothing_when_the_size_probe_is_unavailable(tmp_path):
    """Names line up but the byte total cannot be read back — still a refusal."""
    run, ml = _mk_job(tmp_path)
    r = _run(tmp_path, f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}',
             env={"RCLONE_BREAK": "size"})
    assert "N=0 " in r.stdout, (r.stdout, r.stderr)
    assert len(_survivors(run)) == 4
    assert "size probe UNAVAILABLE" in r.stderr


def test_prune_deletes_nothing_when_b2_is_entirely_unreachable(tmp_path):
    run, ml = _mk_job(tmp_path)
    r = _run(tmp_path, f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}',
             env={"RCLONE_BREAK": "all"})
    assert "N=0 " in r.stdout, (r.stdout, r.stderr)
    assert len(_survivors(run)) == 4


def test_prune_refuses_on_a_stale_handoff_epoch(tmp_path):
    """A superseded box must not prune on the strength of a prefix a newer handoff
    epoch owns — even though the bytes ARE all present under it."""
    run, ml = _mk_job(tmp_path)
    r = _run(tmp_path, f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}',
             env={"HANDOFF_STALE": "1", "HANDOFF_EPOCH": "3"})
    assert "N=0 " in r.stdout, (r.stdout, r.stderr)
    assert len(_survivors(run)) == 4
    assert "prune REFUSED" in r.stderr and "stale" in r.stderr


def test_prune_can_be_disabled(tmp_path):
    run, ml = _mk_job(tmp_path)
    r = _run(tmp_path, f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}',
             env={"JOBD_CKPT_PRUNE": "0"})
    assert "N=0 " in r.stdout and len(_survivors(run)) == 4


def test_keep_floor_is_two_even_if_the_operator_asks_for_zero(tmp_path):
    """JOBD_CKPT_KEEP is clamped: resume survivability is not an operator knob."""
    run, ml = _mk_job(tmp_path)
    r = _run(tmp_path, f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}',
             env={"JOBD_CKPT_KEEP": "0"})
    assert _survivors(run) == ["checkpoint-30", "checkpoint-40"], r.stdout
    r = _run(tmp_path, f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}',
             env={"JOBD_CKPT_KEEP": "banana"})
    assert len(_survivors(run)) == 2


def test_keep_larger_is_honoured(tmp_path):
    run, ml = _mk_job(tmp_path)
    r = _run(tmp_path, f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}',
             env={"JOBD_CKPT_KEEP": "3"})
    assert "N=1 " in r.stdout, (r.stdout, r.stderr)
    assert len(_survivors(run)) == 3


def test_prune_only_considers_dirs_the_sync_actually_matched(tmp_path):
    """Candidates come from the sync's OWN glob expansion. A checkpoint-N dir that
    no checkpoint glob matches (so nothing ever shipped it) is not a candidate —
    which is a second, independent barrier in front of the same `rm -rf`."""
    run, ml = _mk_job(tmp_path)
    _mk_ckpt(str(run), "scratch/checkpoint-1")     # not in the matchlist
    r = _run(tmp_path, f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}')
    assert os.path.isdir(run / "scratch" / "checkpoint-1")
    assert "scratch" not in r.stdout


def test_prune_skips_a_dir_still_being_written(tmp_path):
    """The settle window: a candidate whose files were touched inside
    JOBD_CKPT_SETTLE_S is left alone even when B2 already verifies."""
    run, ml = _mk_job(tmp_path)
    r = _run(tmp_path, f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}',
             env={"JOBD_CKPT_SETTLE_S": "3600"})
    assert "N=0 " in r.stdout, (r.stdout, r.stderr)
    assert "still being written" in r.stderr


def test_prune_never_escapes_the_run_dir(tmp_path):
    """A matchlist naming a traversal path is refused by _ckpt_safe_rel."""
    run, _ = _mk_job(tmp_path, steps=(10, 20, 30, 40))
    outside = tmp_path / "outside" / "checkpoint-1"
    outside.mkdir(parents=True)
    (outside / "trainer_state.json").write_text("x")
    ml = tmp_path / "evil"
    ml.write_text("../outside/checkpoint-1/trainer_state.json\n"
                  "../outside/checkpoint-2/trainer_state.json\n"
                  "../outside/checkpoint-3/trainer_state.json\n")
    r = _run(tmp_path, f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}')
    assert "N=0 " in r.stdout, (r.stdout, r.stderr)
    assert outside.is_dir()


# --- lever 2: end-of-run scrub ----------------------------------------------

_SCRUB_REPORT = 'echo "N=$CKPT_SCRUB_N BYTES=$CKPT_SCRUB_BYTES LIST=$CKPT_SCRUB_LIST"'


def _mk_wdir(tmp_path, run, jobid="j1", uploaded=True, steps=(10, 20, 30, 40)):
    wdir = tmp_path / "wdir"
    wdir.mkdir(exist_ok=True)
    names = ("adapter_model.safetensors", "optimizer.pt", "trainer_state.json")
    lines = "".join(f"out/checkpoint-{s}/{n}\n" for s in steps for n in names)
    (wdir / ".uploaded").write_text(lines if uploaded else "")
    (wdir / ".checkpoint.matched").write_text(lines)
    return wdir


def test_scrub_removes_every_verified_checkpoint_including_the_newest(tmp_path):
    run, _ = _mk_job(tmp_path)
    wdir = _mk_wdir(tmp_path, run)
    r = _run(tmp_path, f'_ckpt_scrub_local j1 "{run}" "{wdir}"\n{_SCRUB_REPORT}')
    assert "N=4 " in r.stdout, (r.stdout, r.stderr)
    assert _survivors(run) == []            # terminal run: no newest-2 exemption


def test_scrub_accepts_the_results_prefix_as_evidence(tmp_path):
    """The finalize publish writes jobs/<id>/results/; the mid-run sync writes
    jobs/<id>/checkpoints/. Either is durable, so either verifies."""
    run, _ = _mk_job(tmp_path, on_b2=())        # nothing under checkpoints/
    shutil.copytree(run / "out",
                    tmp_path / "bucket" / "jobs" / "j1" / "results" / "out")
    wdir = _mk_wdir(tmp_path, run)
    r = _run(tmp_path, f'_ckpt_scrub_local j1 "{run}" "{wdir}"\n{_SCRUB_REPORT}')
    assert "N=4 " in r.stdout, (r.stdout, r.stderr)
    assert _survivors(run) == []


def test_scrub_keeps_a_dir_that_is_on_neither_prefix(tmp_path):
    run, _ = _mk_job(tmp_path, on_b2=(20, 30, 40))
    wdir = _mk_wdir(tmp_path, run)
    r = _run(tmp_path, f'_ckpt_scrub_local j1 "{run}" "{wdir}"\n{_SCRUB_REPORT}')
    assert "N=3 " in r.stdout, (r.stdout, r.stderr)
    assert _survivors(run) == ["checkpoint-10"]


def test_scrub_deletes_nothing_when_b2_is_unreachable(tmp_path):
    run, _ = _mk_job(tmp_path)
    wdir = _mk_wdir(tmp_path, run)
    r = _run(tmp_path, f'_ckpt_scrub_local j1 "{run}" "{wdir}"\n{_SCRUB_REPORT}',
             env={"RCLONE_BREAK": "all"})
    assert "N=0 " in r.stdout, (r.stdout, r.stderr)
    assert len(_survivors(run)) == 4


def test_scrub_refuses_on_a_stale_handoff_epoch(tmp_path):
    run, _ = _mk_job(tmp_path)
    wdir = _mk_wdir(tmp_path, run)
    r = _run(tmp_path, f'_ckpt_scrub_local j1 "{run}" "{wdir}"\n{_SCRUB_REPORT}',
             env={"HANDOFF_STALE": "1", "HANDOFF_EPOCH": "2"})
    assert "N=0 " in r.stdout and len(_survivors(run)) == 4
    assert "scrub REFUSED" in r.stderr


def test_scrub_skips_a_dir_that_is_still_being_written(tmp_path):
    """`wait $epid` returns when timeout's DIRECT child exits — an orphaned DDP rank
    or an in-flight checkpoint rclone can outlive it (jobd says so itself in the
    publish-verify comment). The scrub gets the same quiescence gate the prune has."""
    run, _ = _mk_job(tmp_path)
    wdir = _mk_wdir(tmp_path, run)
    r = _run(tmp_path, f'_ckpt_scrub_local j1 "{run}" "{wdir}"\n{_SCRUB_REPORT}',
             env={"JOBD_CKPT_SETTLE_S": "3600"})
    assert "N=0 " in r.stdout, (r.stdout, r.stderr)
    assert len(_survivors(run)) == 4
    assert "still being written after the entrypoint exited" in r.stderr


def test_scrub_can_be_disabled(tmp_path):
    run, _ = _mk_job(tmp_path)
    wdir = _mk_wdir(tmp_path, run)
    r = _run(tmp_path, f'_ckpt_scrub_local j1 "{run}" "{wdir}"\n{_SCRUB_REPORT}',
             env={"JOBD_CKPT_SCRUB": "0"})
    assert "N=0 " in r.stdout and len(_survivors(run)) == 4


def test_scrub_call_site_is_gated_on_publish_verify_and_ordered_last(tmp_path):
    """A source-level assertion, because the ORDERING is the safety property and no
    unit test of the function itself can see it. `results:` is `out/**`, which
    CONTAINS the checkpoints, so the scrub must sit after the publish, after
    publish-verify, and after the manifest pass that stats the local files."""
    src = open(JOBD_SH).read()
    call = src.index("_ckpt_scrub_local \"$jobid\"")
    for earlier in ('b2x_push "$run" "$B2W/jobs/$jobid/results/"',   # publish
                    "publish verify: SKIP",                          # verify loop
                    'rclone rcat "$B2W/jobs/$jobid/results.DONE.json"',
                    'emit "$jobid" results_uploaded'):
        assert src.index(earlier) < call, earlier
    # the gate itself
    gate = src.rindex('if [ -s "$wdir/.uploaded" ] && [ -z "${vfail:-}" ]; then',
                      0, call)
    assert call - gate < 400, "scrub is not directly under its publish-verify gate"
    # and it is the last thing the runner does
    assert src.index('rm -f "$STATE_DIR/$jobid.running"', call) > call


# --- lever 3: fire-on-arrival ------------------------------------------------

def test_new_ready_ignores_a_dir_without_the_completion_marker(tmp_path):
    """transformers 5.13.0 writes the checkpoint dir IN PLACE (no tmp-checkpoint
    staging dir, no os.rename), and _save_rng_state can create it from a NON-ZERO
    rank before rank 0 has written a byte. So an existing dir proves nothing;
    trainer_state.json (rank 0's last write) is the marker."""
    run = tmp_path / "run"
    _mk_ckpt(str(run), "out/checkpoint-10",
             files=(("adapter_model.safetensors", 100),))     # mid-write
    seen = tmp_path / "seen"
    seen.write_text("")
    r = _run(tmp_path, f'_ckpt_new_ready "{run}" "{seen}"')
    assert r.stdout.strip() == "", r.stdout
    _mk_ckpt(str(run), "out/checkpoint-10")                   # now complete
    r = _run(tmp_path, f'_ckpt_new_ready "{run}" "{seen}"')
    assert r.stdout.strip() == "out/checkpoint-10"


def test_new_ready_honours_the_settle_window(tmp_path):
    """The marker does NOT order a non-zero rank's rng_state_<i>.pth against rank
    0's trainer_state.json, so a quiescence window backs it up."""
    run = tmp_path / "run"
    _mk_ckpt(str(run), "out/checkpoint-10")
    seen = tmp_path / "seen"
    seen.write_text("")
    r = _run(tmp_path, f'_ckpt_new_ready "{run}" "{seen}"',
             env={"JOBD_CKPT_SETTLE_S": "3600"})
    assert r.stdout.strip() == ""


def test_new_ready_skips_dirs_already_seen(tmp_path):
    run = tmp_path / "run"
    _mk_ckpt(str(run), "out/checkpoint-10")
    _mk_ckpt(str(run), "out/checkpoint-20")
    seen = tmp_path / "seen"
    seen.write_text("out/checkpoint-10\n")
    r = _run(tmp_path, f'_ckpt_new_ready "{run}" "{seen}"')
    assert r.stdout.split() == ["out/checkpoint-20"]


def test_all_dirs_seeds_the_seen_set(tmp_path):
    """On a resume the pull-back re-creates dirs that are ALREADY on B2; firing on
    them would re-push GB for nothing."""
    run = tmp_path / "run"
    _mk_ckpt(str(run), "out/checkpoint-10",
             files=(("adapter_model.safetensors", 10),))      # incomplete on disk
    _mk_ckpt(str(run), "out/arms/hex/checkpoint-20")
    r = _run(tmp_path, f'_ckpt_all_dirs "{run}"')
    assert sorted(r.stdout.split()) == ["out/arms/hex/checkpoint-20",
                                        "out/checkpoint-10"]


def test_sync_loop_drops_min_age_only_on_the_fast_path(tmp_path):
    """Source-level: the periodic pass keeps --min-age (it may catch a half-written
    file), the fire-on-arrival pass drops it (completeness is proven) and is scoped
    to the new dirs only."""
    src = open(JOBD_SH).read()
    i = src.index("pinc=(\"${inc[@]}\"); page=(--min-age")
    j = src.index("if _handoff_epoch_stale \"$jobid\"", i)
    seg = src[i:j]
    assert 'trig=new-checkpoint; pinc=(); page=()' in seg
    assert 'pinc+=(--include "$_d/**")' in seg
    # and the push uses the per-pass arrays, not the old hardcoded --min-age
    push = src.index('b2x_push "$run" "$B2W/jobs/$jobid/checkpoints/"')
    assert '${page[@]+"${page[@]}"} "${pinc[@]}"' in src[push:push + 400]


def test_fast_path_dirs_are_marked_seen_even_when_the_push_fails(tmp_path):
    """Otherwise a failing push re-arms the 5 s watcher forever. It is `fast_all`
    (every dir the watcher saw), not `fast` (the subset the bundle's checkpoint
    globs cover) — a dir outside the globs would otherwise break the wait on every
    single tick, forever. The periodic backstop re-ships the whole glob either
    way, so nothing is stranded."""
    src = open(JOBD_SH).read()
    assert ('if [ -n "$fast_all" ]; then printf \'%s\\n\' "$fast_all" >> "$seenf"; fi'
            in src)


def test_fast_path_ships_only_what_the_checkpoint_globs_declare(tmp_path):
    """A `checkpoints:` glob that does not cover checkpoint-<N>/ must not have those
    bytes shipped just because a directory-name watcher noticed them. `fast` is the
    intersection of the watcher's hits with the glob expansion."""
    src = open(JOBD_SH).read()
    i = src.index('fast="$(LC_ALL=C comm -12')
    seg = src[i:i + 300]
    assert '_ckpt_dirs_from_matchlist "$cmatch"' in seg, seg
    # and the intersection happens BEFORE the include list is built from it
    assert i < src.index('pinc+=(--include "$_d/**")')


def test_nothing_in_the_block_deletes_from_b2(tmp_path):
    """Scope boundary: B2 holds the full dose-curve grid. This change is box disk
    only — no rclone delete/purge/deletefile/sync anywhere in the block."""
    block = _lifecycle_block()
    for bad in ("rclone delete", "rclone purge", "rclone deletefile",
                "rclone sync", "rclone rmdir", "b2x_push"):
        assert bad not in block, bad
    # exactly one rm -rf, and it is guarded by _ckpt_safe_rel in both call sites
    assert block.count("rm -rf ") == 2      # prune + scrub, nothing else


# --- lever 4: publish-by-marker ----------------------------------------------
# B2 has no atomic directory rename, so a multi-GB checkpoint reaches its keys as
# N uploads and an eviction anywhere in that window leaves a directory that still
# looks like the newest checkpoint. Five v16 restarts on 2026-08-28 were lost to
# `--resume auto` selecting one. Completion is therefore a separate, single,
# small, LAST write: a `<dir>.complete.json` sibling, published only after
# `_ckpt_b2_verified` has read the exact directory back.

_MARK = 'echo "MARK_N=$CKPT_MARK_N MARK_LIST=$CKPT_MARK_LIST"'


def _marker_path(tmp_path, jobid, rel):
    return (tmp_path / "bucket" / "jobs" / jobid / "checkpoints"
            / (rel + ".complete.json"))


def test_a_verified_checkpoint_gets_a_marker_beside_it(tmp_path):
    run, ml = _mk_job(tmp_path, steps=(10, 20))
    r = _run(tmp_path,
             f'_ckpt_mark_complete j1 "{run}" "{ml}" "{tmp_path}/marked"\n{_MARK}')
    assert r.returncode == 0, r.stderr
    assert "MARK_N=2 " in r.stdout, (r.stdout, r.stderr)
    for s in (10, 20):
        p = _marker_path(tmp_path, "j1", f"out/checkpoint-{s}")
        assert p.exists(), f"no marker for checkpoint-{s}"
        doc = json.loads(p.read_text())
        assert doc["step"] == s and doc["n_files"] == 3
        assert doc["files"]["optimizer.pt"] == 8192
        assert doc["total_bytes"] == 4096 + 8192 + 64


def test_the_marker_is_a_SIBLING_so_the_delete_gate_still_verifies(tmp_path):
    """Load-bearing placement. Inside the dir the marker would join the name set
    and byte total `_ckpt_b2_verified` compares against the LOCAL copy (which
    never holds it), every later verify would read a strict SUPERSET, and the
    prune would be wedged for that dir forever — the box disk fills instead."""
    run, ml = _mk_job(tmp_path, steps=(10, 20, 30, 40))
    r = _run(tmp_path,
             f'_ckpt_mark_complete j1 "{run}" "{ml}" "{tmp_path}/marked"\n'
             f'_ckpt_prune_synced j1 "{run}" "{ml}"\n{_REPORT}')
    assert r.returncode == 0, r.stderr
    assert "N=2 " in r.stdout, (r.stdout, r.stderr)          # prune still works
    assert "SUPERSET" not in r.stderr and "MISMATCH" not in r.stderr
    assert _survivors(run) == ["checkpoint-30", "checkpoint-40"]


def test_no_marker_when_b2_does_not_hold_the_directory(tmp_path):
    """Publish is gated on the SAME read-back the delete gate uses. A checkpoint
    whose upload is still in flight has no marker, so the resume side reaches
    past it — which is the entire mechanism."""
    run, ml = _mk_job(tmp_path, steps=(10, 20), on_b2=(10,))
    r = _run(tmp_path,
             f'_ckpt_mark_complete j1 "{run}" "{ml}" "{tmp_path}/marked"\n{_MARK}')
    assert "MARK_N=1 " in r.stdout, (r.stdout, r.stderr)
    assert _marker_path(tmp_path, "j1", "out/checkpoint-10").exists()
    assert not _marker_path(tmp_path, "j1", "out/checkpoint-20").exists()
    assert "NOT published yet" in r.stderr


def test_a_TRUNCATED_remote_file_is_not_published(tmp_path):
    """The interrupted-upload shape, exactly: the directory exists on B2 and one
    object is short. The byte-total check refuses, so nothing is published."""
    run, ml = _mk_job(tmp_path, steps=(10,))
    victim = (tmp_path / "bucket" / "jobs" / "j1" / "checkpoints"
              / "out" / "checkpoint-10" / "optimizer.pt")
    victim.write_bytes(b"x" * 40)
    r = _run(tmp_path,
             f'_ckpt_mark_complete j1 "{run}" "{ml}" "{tmp_path}/marked"\n{_MARK}')
    assert "MARK_N=0 " in r.stdout, (r.stdout, r.stderr)
    assert not _marker_path(tmp_path, "j1", "out/checkpoint-10").exists()


def test_an_INCOMPLETE_local_dir_is_never_published(tmp_path):
    """The 2-object checkpoint-96 shape. `_ckpt_write_complete` gates this:
    trainer_state.json is the rank-0 written-last marker, and a dir without it
    was never finished on disk, whatever B2 holds."""
    run = tmp_path / "run"
    _mk_ckpt(str(run), "out/checkpoint-10",
             files=(("adapter_model.safetensors", 4096),))
    shutil.copytree(run / "out" / "checkpoint-10",
                    tmp_path / "bucket" / "jobs" / "j1" / "checkpoints"
                    / "out" / "checkpoint-10")
    ml = tmp_path / "matchlist"
    ml.write_text("out/checkpoint-10/adapter_model.safetensors\n")
    r = _run(tmp_path,
             f'_ckpt_mark_complete j1 "{run}" "{ml}" "{tmp_path}/marked"\n{_MARK}')
    assert "MARK_N=0 " in r.stdout, (r.stdout, r.stderr)
    assert not _marker_path(tmp_path, "j1", "out/checkpoint-10").exists()


def test_a_failed_marker_write_is_retried_next_pass_not_recorded(tmp_path):
    """A refusal must never be latched: the sync loop calls this every pass, and
    a dir recorded as marked when the write failed would never be published."""
    run, ml = _mk_job(tmp_path, steps=(10,))
    marked = tmp_path / "marked"
    r = _run(tmp_path,
             f'_ckpt_mark_complete j1 "{run}" "{ml}" "{marked}"\n{_MARK}',
             env={"RCLONE_BREAK": "rcat"})
    assert "MARK_N=0 " in r.stdout, (r.stdout, r.stderr)
    assert marked.read_text() == ""
    r2 = _run(tmp_path, f'_ckpt_mark_complete j1 "{run}" "{ml}" "{marked}"\n{_MARK}')
    assert "MARK_N=1 " in r2.stdout, (r2.stdout, r2.stderr)


def test_publishing_is_idempotent_and_costs_one_read_back_per_dir(tmp_path):
    run, ml = _mk_job(tmp_path, steps=(10,))
    marked = tmp_path / "marked"
    _run(tmp_path, f'_ckpt_mark_complete j1 "{run}" "{ml}" "{marked}"')
    before = _marker_path(tmp_path, "j1", "out/checkpoint-10").read_text()
    r = _run(tmp_path, f'_ckpt_mark_complete j1 "{run}" "{ml}" "{marked}"\n{_MARK}')
    assert "MARK_N=0 " in r.stdout                    # already marked, skipped
    assert _marker_path(tmp_path, "j1", "out/checkpoint-10").read_text() == before


def test_the_marker_body_is_DETERMINISTIC(tmp_path):
    """A resumed box re-pushes the marker it pulled back to the same key. If the
    bytes differed (a timestamp), every resume would open an overwrite
    eventual-consistency window on B2 for nothing."""
    run, ml = _mk_job(tmp_path, steps=(10,))
    out = []
    for _ in range(2):
        r = _run(tmp_path, f'_ckpt_marker_json "{run}/out/checkpoint-10" '
                           f'out/checkpoint-10')
        assert r.returncode == 0, r.stderr
        out.append(r.stdout)
    assert out[0] == out[1] and out[0].strip()


def test_JOBD_CKPT_MARK_0_disables_publishing(tmp_path):
    run, ml = _mk_job(tmp_path, steps=(10,))
    r = _run(tmp_path,
             f'_ckpt_mark_complete j1 "{run}" "{ml}" "{tmp_path}/marked"\n{_MARK}',
             env={"JOBD_CKPT_MARK": "0"})
    assert "MARK_N=0 " in r.stdout
    assert not _marker_path(tmp_path, "j1", "out/checkpoint-10").exists()


@pytest.mark.parametrize("names,ok", [
    ("trainer_state.json optimizer.pt scheduler.pt adapter_model.safetensors", 0),
    ("trainer_state.json", 1),                                  # checkpoint-96
    ("trainer_state.json scheduler.pt adapter_model.safetensors", 1),  # -112
    ("trainer_state.json optimizer.pt scheduler.pt rng_state.pth", 1),  # -176
    ("trainer_state.json optimizer.bin scheduler.pt model.safetensors", 0),
    ("trainer_state.json optimizer.pt.bnb_skipped scheduler.pt.bnb_skipped "
     "adapter_model.safetensors", 0),
    ("optimizer.pt scheduler.pt adapter_model.safetensors", 1),
])
def test_the_file_set_predicate_names_each_observed_shape(tmp_path, names, ok):
    """`_ckpt_names_complete` is the legacy/no-marker tier, and the bash spelling
    of the contract the trainer and ckpt_retention.py also implement."""
    lines = "".join(f"out/checkpoint-1/{n}\n" for n in names.split())
    r = _run(tmp_path, f"printf '%s' {shlex.quote(lines)} | _ckpt_names_complete; "
                       f'echo "RC=$?"')
    assert f"RC={ok}" in r.stdout, (names, r.stdout, r.stderr)


def test_the_publish_call_precedes_the_prune_at_the_sync_call_site(tmp_path):
    """Ordering is load-bearing: the marker's file list is built from the LOCAL
    directory, which the prune is about to delete."""
    src = open(JOBD_SH).read()
    i_mark = src.index('_ckpt_mark_complete "$jobid" "$run" "$cmatch"')
    i_prune = src.index('_ckpt_prune_synced "$jobid" "$run" "$cmatch"')
    assert i_mark < i_prune


def test_a_marker_an_EARLIER_BOX_published_is_adopted_not_re_verified(tmp_path):
    """A resumed box pulls the checkpoint back with b2x's .b2x/state.json beside
    it, so the local name set no longer equals the remote one and
    `_ckpt_b2_verified` refuses -- correctly, and forever, logging a refusal
    every 180 s pass for a checkpoint published hours ago on another box."""
    run, ml = _mk_job(tmp_path, steps=(10,))
    _run(tmp_path, f'_ckpt_mark_complete j1 "{run}" "{ml}" "{tmp_path}/m1"')
    assert _marker_path(tmp_path, "j1", "out/checkpoint-10").exists()
    # now make the local dir look pulled-back: an extra file the remote lacks
    os.makedirs(run / "out" / "checkpoint-10" / ".b2x", exist_ok=True)
    (run / "out" / "checkpoint-10" / ".b2x" / "state.json").write_text("{}")
    marked2 = tmp_path / "m2"          # a FRESH box: empty marked-set
    r = _run(tmp_path, f'_ckpt_mark_complete j1 "{run}" "{ml}" "{marked2}"\n{_MARK}')
    assert "MARK_N=0 " in r.stdout, (r.stdout, r.stderr)
    assert "NOT published yet" not in r.stderr
    assert marked2.read_text().strip() == "out/checkpoint-10"
