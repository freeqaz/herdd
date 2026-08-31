"""The resume pull-back must fetch the NEWEST checkpoint, not the whole ladder.

Regression cover for a production stall on 2026-08-07: v9-gemma4 was evicted at
step 135 of 156, and its replacement pulled back 27 checkpoints x ~1.55 GB = 41 GB
onto a 50 GB disk, leaving 9 GB against the 28 GB of assets it still needed. The
job was reaped `INSUFFICIENT DISK` with every byte of its state intact and
uncorrupted -- it simply could not fit itself. `--save-total-limit` bounds the
ladder on the BOX during a run but does not touch B2, and the pull-back ignored
it, so resume disk need grew LINEARLY with run length.

These exercise the SHIPPED function, extracted from jobd.sh by text rather than
reimplemented here -- a copy would drift and then pass while the box fails.
"""
import os
import re
import shlex
import subprocess

import pytest

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JOBD = os.path.join(REPO_ROOT, "tools", "vast", "onstart", "jobd.sh")


def _extract(fn):
    """Pull one shell function's source straight out of jobd.sh."""
    src = open(JOBD).read()
    m = re.search(rf"^{re.escape(fn)}\(\) \{{.*?^\}}", src, re.S | re.M)
    assert m, f"{fn}() not found in jobd.sh — keep this extractor in step"
    return m.group(0)


# _ckpt_latest_remote does not stand alone: since 2026-08-28 it asks whether a
# remote checkpoint is COMPLETE before selecting it, which needs the suffix
# constant and the file-set predicate. Extracting the caller without them would
# leave `_ckpt_names_complete` as a missing command -- every candidate would read
# incomplete, every test would fall through to the fail-open branch, and the
# suite would pass while testing the wrong path.
def _deps():
    src = open(JOBD).read()
    m = re.search(r"^CKPT_COMPLETE_SUFFIX=.*$", src, re.M)
    assert m, "CKPT_COMPLETE_SUFFIX is gone from jobd.sh"
    return m.group(0) + "\n" + _extract("_ckpt_names_complete") + "\n"


def _run(listing, keep=None, rclone_rc=0, files=None):
    """Run _ckpt_latest_remote with a stubbed rclone.

    `listing` answers `lsf --dirs-only`. `files` (optional) maps a
    `checkpoint-<N>` name -> its remote file list, answering the per-candidate
    `lsf -R --files-only`; the marker objects, if any, go in `listing` too since
    `lsf --files-only` on the same prefix is the call that finds them. With
    `files` unset every candidate reads incomplete and the fail-open branch runs
    -- which is exactly the pre-marker world these ladder tests describe.
    """
    # shlex.quote, NOT repr: repr renders newlines as literal backslash-n, which
    # bash single-quotes preserve verbatim -- the stub then emits one long line
    # and every ladder test silently returns empty.
    if listing is None:
        stub = f"rclone() {{ return {rclone_rc}; }}\n"
    else:
        cases = ""
        for name, flist in (files or {}).items():
            cases += (f"    *{shlex.quote('/' + name + '/')}) "
                      f"printf '%s' {shlex.quote(''.join(f'{f}\n' for f in flist))}; "
                      f"return 0 ;;\n")
        stub = (
            "rclone() {\n"
            '  local last="${@: -1}"\n'
            "  case \"$last\" in\n"
            f"{cases}"
            f"    *) printf '%s' {shlex.quote(listing)}; return {rclone_rc} ;;\n"
            "  esac\n"
            "}\n")
    env = dict(os.environ)
    if keep is not None:
        env["JOBD_CKPT_PULL_KEEP"] = str(keep)
    script = ("log() { :; }\n" + _deps() + _extract("_ckpt_latest_remote")
              + '\n_ckpt_latest_remote "b2:x/" || true\n')
    r = subprocess.run(["bash", "-c", stub + script],
                       capture_output=True, text=True, env=env)
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


# A resumable checkpoint's file set, in the shape the trainer writes it.
COMPLETE = ["trainer_state.json", "optimizer.pt", "scheduler.pt",
            "adapter_model.safetensors", "rng_state.pth", "README.md"]


LADDER = "".join(f"checkpoint-{n}/\n" for n in
                 (5, 10, 40, 60, 90, 100, 130, 135))


def test_picks_the_newest_checkpoint_only():
    assert _run(LADDER) == ["checkpoint-135"]


def test_numeric_not_alphabetical_ordering():
    """`sort -V`, never plain sort.

    Alphabetically `checkpoint-100` precedes `checkpoint-60`, so a plain sort
    selects checkpoint-90 out of this ladder. That exact confusion produced a
    false 'we only have checkpoint-90' report on 2026-08-06, and here it would
    silently resume from 45 steps earlier than the state we hold.
    """
    got = _run(LADDER)
    assert got == ["checkpoint-135"]
    assert got != ["checkpoint-90"], "plain `sort` regression"


def test_keep_count_is_tunable_and_returns_the_newest_n():
    assert _run(LADDER, keep=2) == ["checkpoint-130", "checkpoint-135"]
    assert _run(LADDER, keep=3) == ["checkpoint-100", "checkpoint-130",
                                    "checkpoint-135"]


def test_non_checkpoint_directories_are_ignored():
    listing = "out/\nruns/\ncheckpoint-abc/\ntmp/\ncheckpoint-40/\n"
    assert _run(listing) == ["checkpoint-40"]


@pytest.mark.parametrize("listing,rc", [
    (None, 1),        # rclone failed outright
    ("", 0),          # listing readable but empty
    ("out/\nruns/\n", 0),   # readable, but names no checkpoint at all
])
def test_returns_nothing_when_it_cannot_tell(listing, rc):
    """Empty output is the FAIL-OPEN signal: the caller falls back to the
    historical whole-prefix pull.

    The asymmetry is deliberate and load-bearing. A resume that fetches too much
    still resumes, just slowly. A resume that fetches too little restarts from
    step 0 silently, with no error and a normal-looking log -- the single most
    expensive failure mode in this system. So every uncertain case must degrade
    toward over-fetching.
    """
    assert _run(listing, rclone_rc=rc) == []


def test_pullback_call_site_is_bounded_and_still_fetches_non_checkpoints():
    """Guard the call site, not just the helper.

    Two things must remain true at jobd.sh's pull-back: the ladder is excluded
    from the bulk copy (else the bound does nothing), and a fallback branch to
    the whole prefix survives (else fail-open becomes fail-closed).
    """
    src = open(JOBD).read()
    assert '--exclude "out/checkpoint-*/**"' in src, \
        "bulk pull no longer excludes the ladder — the bound is inert"
    assert "_ckpt_latest_remote" in src, "helper is not wired to the call site"
    assert "could not enumerate remote checkpoints" in src, \
        "fail-open fallback branch is gone"


# --- the pull-back must not fetch a TORN checkpoint (2026-08-28) --------------
# Bounding the pull-back to newest-1 on 2026-08-07 retired the property
# CHECKPOINT_LIFECYCLE.md's newest-2 floor exists for: "the newest can be a
# partial upload from a box that died mid-push; HF resume validation then falls
# back to the complete one". With one dir pulled there is nothing to fall back
# to. Five v16 restarts on 2026-08-28 were lost that way. Asking B2 which dir is
# complete costs one LIST and pulls the RIGHT single directory.

def test_skips_a_torn_newest_and_pulls_the_newest_COMPLETE():
    """The observed shape: checkpoint-176 arrived with the adapter missing."""
    listing = "checkpoint-160/\ncheckpoint-176/\n"
    files = {"checkpoint-160": COMPLETE,
             "checkpoint-176": ["trainer_state.json", "optimizer.pt",
                                "scheduler.pt", "rng_state.pth"]}
    assert _run(listing, files=files) == ["checkpoint-160"]


def test_a_completion_marker_selects_without_reading_the_file_list():
    """Tier 1 of the contract. The marker is jobd's own read-back-verified
    publish, so it stands on its own -- note checkpoint-176 here has NO file
    list stubbed at all and is still selected."""
    listing = "checkpoint-160/\ncheckpoint-176/\ncheckpoint-176.complete.json\n"
    assert _run(listing, files={"checkpoint-160": COMPLETE}) == ["checkpoint-176"]


def test_a_marker_for_an_older_step_does_not_rescue_a_torn_newest():
    listing = ("checkpoint-160/\ncheckpoint-176/\n"
               "checkpoint-160.complete.json\n")
    files = {"checkpoint-176": ["trainer_state.json"]}     # the 2-object shape
    assert _run(listing, files=files) == ["checkpoint-160"]


def test_a_legacy_complete_checkpoint_with_no_marker_is_still_selected():
    """Backward compatibility. Everything on B2 today predates the marker, and
    the preempt trap's final flush (40 s deadline, no room for a read-back) will
    keep producing unmarked-but-complete dirs. Unmarked must not mean unusable."""
    assert _run("checkpoint-135/\n", files={"checkpoint-135": COMPLETE}) \
        == ["checkpoint-135"]


def test_every_candidate_torn_falls_back_to_the_numeric_max():
    """FAIL-OPEN is preserved exactly. A pull-back that fetches too little is
    the expensive failure; when nothing reads as complete we hand back what the
    2026-08-07 bound handed back, and the trainer's own resume guard judges it."""
    listing = "checkpoint-160/\ncheckpoint-176/\n"
    files = {"checkpoint-160": ["trainer_state.json"],
             "checkpoint-176": ["trainer_state.json"]}
    assert _run(listing, files=files) == ["checkpoint-176"]


def test_bnb_skipped_counterparts_satisfy_the_optimizer_requirement():
    """`_disable_checkpoint_optimizer_state` renames optimizer.pt aside on a
    quantized resume. That is OUR deliberate act, not a torn upload."""
    files = {"checkpoint-40": ["trainer_state.json", "optimizer.pt.bnb_skipped",
                               "scheduler.pt.bnb_skipped",
                               "adapter_model.safetensors"]}
    assert _run("checkpoint-40/\n", files=files) == ["checkpoint-40"]


def test_keep_count_greater_than_one_takes_complete_dirs_newest_first():
    listing = "checkpoint-90/\ncheckpoint-160/\ncheckpoint-176/\n"
    files = {"checkpoint-90": COMPLETE, "checkpoint-160": COMPLETE,
             "checkpoint-176": ["trainer_state.json"]}
    assert sorted(_run(listing, keep=2, files=files)) == \
        ["checkpoint-160", "checkpoint-90"]
