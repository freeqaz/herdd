"""Portable tests for the submit-time B2 WRITE-SCOPE preflight (jobmeta).

Runs in the toolchain-free lane (`pytest -m "not integration"`): the check is
PURE — it reads a bundle's own text and compares each B2 destination against the
grant table the launcher mints. NO network, NO B2, NO creds, NO vast API. Every
fixture bundle here is synthetic; the real bundles are never read (a peer agent
edits them, and a test that fails when someone else edits a bundle is a tripwire,
not a test).

The incident this closes — docs/plans/witness/g2_push/V7_TRAIN_RUN_2026-08-05.md:
v7's publish stage wrote `b2:$B2_BUCKET/checkpoints/$RUN_NAME/` while the box held
a bucket-wide READ key on `b2` and a write key scoped to `jobs/`. Both arms
trained to completion, then exited rc=15 on a 403. The rehearsal is DRY_RUN and
never touches B2, so only a static check could have seen it.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2_mint_key as bmk  # noqa: E402
import jobmeta as jm  # noqa: E402


def _bundle(tmp_path, **files):
    """Write a synthetic bundle dir; returns its path."""
    d = tmp_path / "bundle"
    d.mkdir(exist_ok=True)
    for name, body in files.items():
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return str(d)


def _status(findings, fname=None):
    return sorted(f["status"] for f in findings
                  if fname is None or f["file"] == fname)


# --------------------------------------------------------------- the defect #
V7_PUBLISH = """#!/usr/bin/env bash
set -euo pipefail
OUT=/workspace/out
RUN_NAME="${RUN_NAME:?}"
PUB_DEST="b2:${B2_BUCKET}/checkpoints/${RUN_NAME}/"
for _try in 1 2 3; do
  if rclone copy --fast-list "${PUB_INC[@]}" "$OUT" "$PUB_DEST" \\
     && rclone lsjson --hash "$PUB_DEST" > "$PUB_LS" 2>/dev/null; then
    PUB_OK=1; break
  fi
done
"""


def test_the_v7_publish_403_is_caught_statically(tmp_path):
    b = _bundle(tmp_path, **{"run.sh": V7_PUBLISH})
    findings = jm.scan_b2_writes(b)
    assert [f["status"] for f in findings] == ["uncovered"]
    f = findings[0]
    assert (f["remote"], f["prefix"], f["verb"]) == ("b2", "checkpoints/", "copy")
    assert f["file"] == "run.sh" and f["line"] == 7
    lines, refuse = jm.b2_write_scope_report(findings)
    assert refuse is True
    assert any("NOT ENTITLED" in ln and "checkpoints/" in ln for ln in lines)


def test_the_same_publish_through_b2p_passes(tmp_path):
    """The fix, from the bundle's side: route the publish through the remote that
    holds the checkpoints/ key."""
    b = _bundle(tmp_path, **{"run.sh": V7_PUBLISH.replace('"b2:${B2_BUCKET}',
                                                          '"b2p:${B2_BUCKET}')})
    findings = jm.scan_b2_writes(b)
    assert _status(findings) == ["ok"]
    lines, refuse = jm.b2_write_scope_report(findings)
    assert refuse is False
    assert any(ln.startswith(">> B2 write scope OK") for ln in lines)


def test_reads_through_the_read_key_are_never_flagged(tmp_path):
    b = _bundle(tmp_path, **{"run.sh": (
        "rclone cat b2:$B2_BUCKET/checkpoints/x/PUBLISHED.json > /tmp/p\n"
        "rclone lsjson --hash b2:$B2_BUCKET/checkpoints/x/\n"
        "rclone md5sum b2:$B2_BUCKET/base-models/qwen/\n")})
    assert jm.scan_b2_writes(b) == []


def test_a_download_is_not_a_write(tmp_path):
    """`rclone copy b2:…/assets LOCAL` — the b2 ref is the SOURCE."""
    b = _bundle(tmp_path, **{"run.sh": (
        'BASE_DIR="$WORK/base"\n'
        'rclone copy b2:$B2_BUCKET/base-models/qwen "$BASE_DIR/"\n')})
    assert jm.scan_b2_writes(b) == []


# ------------------------------------------------------------ grant table #
@pytest.mark.parametrize("dest,status", [
    ("b2w:$B2_BUCKET/jobs/$JOB_ID/results/", "ok"),
    ("b2w:$B2_BUCKET/jobs/nodes/1/", "ok"),
    ("b2w:$B2_BUCKET/checkpoints/run/", "uncovered"),   # jobs/-scoped key
    ("b2p:$B2_BUCKET/checkpoints/run/", "ok"),
    ("b2p:$B2_BUCKET/jobs/j/results/", "uncovered"),    # checkpoints/-scoped key
    ("b2:$B2_BUCKET/jobs/j/results/", "uncovered"),     # READ key
    ("b2eu:$B2_BUCKET/checkpoints/r/", "uncovered"),    # read replica
    ("b2x:$B2_BUCKET/checkpoints/r/", "uncovered"),     # no such remote on a box
])
def test_each_remote_is_checked_against_its_own_prefix(tmp_path, dest, status):
    b = _bundle(tmp_path, **{"run.sh": f'rclone copy "$OUT" "{dest}"\n'})
    findings = jm.scan_b2_writes(b)
    assert [f["status"] for f in findings] == [status], findings


def test_every_write_verb_is_scanned(tmp_path):
    body = "".join(f'rclone {v} "$OUT" "b2:$B2_BUCKET/checkpoints/r/"\n'
                   for v in sorted(jm.B2_WRITE_VERBS))
    findings = jm.scan_b2_writes(_bundle(tmp_path, **{"run.sh": body}))
    assert len(findings) == len(jm.B2_WRITE_VERBS)
    assert set(f["status"] for f in findings) == {"uncovered"}


def test_grant_table_matches_what_the_minter_actually_scopes():
    """The preflight is only worth anything if its grant table is the one the
    launcher mints. jobmeta ships to boxes and b2_mint_key does not, so the two
    cannot import each other — this is the cross-check that keeps them honest."""
    assert jm.B2_BOX_GRANTS["b2w"] == jm.JOB_WRITE_PREFIXES == ("jobs/",)
    assert jm.B2_BOX_GRANTS["b2p"] == jm.JOB_PUBLISH_PREFIXES
    assert jm.B2_BOX_GRANTS["b2p"] == (bmk.PUBLISH_PREFIX,)
    assert jm.B2_BOX_GRANTS["b2"] == () and jm.B2_BOX_GRANTS["b2eu"] == ()


# --------------------------------------------------- resolution + blind spots #
def test_variable_indirection_is_resolved_across_lines(tmp_path):
    b = _bundle(tmp_path, **{"jobd_like.sh": (
        'if [ -n "${B2_WRITE_KEY_ID:-}" ]; then B2W="b2w:${B2_BUCKET}"; '
        'else B2W="$B2"; fi\n'
        'rclone copyto "$log" "$B2W/jobs/$jobid/log.txt"\n')})
    findings = jm.scan_b2_writes(b)
    assert [(f["remote"], f["prefix"], f["status"]) for f in findings] == \
        [("b2w", "jobs/", "ok")]


def test_a_loop_variable_local_destination_is_not_noise(tmp_path):
    """`for f in …; do rclone copy … /workspace/$f; done` is a local copy. A name
    bound by `for` counts as assigned, or every such line becomes a note and the
    notes stop being read."""
    b = _bundle(tmp_path, **{"run.sh": (
        "for _f in a b c; do\n"
        '  rclone copy "$SRC" "/workspace/${_f}"\n'
        "done\n")})
    assert jm.scan_b2_writes(b) == []


def test_an_unresolvable_destination_is_a_note_not_a_refusal(tmp_path):
    b = _bundle(tmp_path, **{"run.sh": 'rclone copy "$OUT" "$DEST_FROM_ENV"\n'})
    findings = jm.scan_b2_writes(b)
    assert _status(findings) == ["unknown"]
    lines, refuse = jm.b2_write_scope_report(findings)
    assert refuse is False
    assert lines and lines[0].startswith("note: B2 write scope UNVERIFIED")


def test_an_unresolvable_PREFIX_is_a_note_too(tmp_path):
    b = _bundle(tmp_path, **{"run.sh":
                             'rclone copy "$OUT" "b2p:$B2_BUCKET/$PREFIX/x/"\n'})
    findings = jm.scan_b2_writes(b)
    assert [(f["remote"], f["prefix"], f["status"]) for f in findings] == \
        [("b2p", None, "unknown")]
    assert jm.b2_write_scope_report(findings)[1] is False


def test_python_subprocess_rclone_is_scanned_too(tmp_path):
    b = _bundle(tmp_path, **{"pub.py": (
        "import subprocess\n"
        "subprocess.run([\"rclone\", \"copy\", out,\n"
        "                f\"b2:{bucket}/checkpoints/{run}/\"], check=True)\n")})
    findings = jm.scan_b2_writes(b)
    assert [(f["remote"], f["prefix"], f["status"]) for f in findings] == \
        [("b2", "checkpoints/", "uncovered")]


def test_scan_ignores_non_code_and_never_leaves_the_bundle(tmp_path):
    outside = tmp_path / "outside.sh"
    outside.write_text('rclone copy "$OUT" "b2:$B2_BUCKET/checkpoints/x/"\n')
    b = _bundle(tmp_path, **{
        "data/rows.jsonl": '{"cmd": "rclone copy x b2:b/checkpoints/y/"}\n',
        "README.md": "rclone copy $OUT b2:$B2_BUCKET/checkpoints/x/\n",
        "run.sh": "echo hi\n"})
    os.symlink(str(outside), os.path.join(b, "link.sh"))
    assert jm.scan_b2_writes(b) == []


def test_scan_is_read_only(tmp_path):
    b = _bundle(tmp_path, **{"run.sh": V7_PUBLISH})
    before = {p: os.stat(os.path.join(b, p)).st_mtime_ns for p in os.listdir(b)}
    jm.scan_b2_writes(b)
    after = {p: os.stat(os.path.join(b, p)).st_mtime_ns for p in os.listdir(b)}
    assert before == after and sorted(os.listdir(b)) == ["run.sh"]


# ------------------------------------------------------- scope: consistency #
def test_declared_scope_write_must_cover_what_the_bundle_writes(tmp_path):
    b = _bundle(tmp_path, **{"run.sh":
                             'rclone copy "$OUT" "b2p:$B2_BUCKET/checkpoints/r/"\n'})
    cfg = {"scope": {"write": ["jobs/"]}}
    findings = jm.b2_write_preflight(cfg, b)
    assert _status(findings) == ["undeclared"]
    assert jm.b2_write_scope_report(findings)[1] is True
    ok = jm.b2_write_preflight({"scope": {"write": ["jobs/", "checkpoints/"]}}, b)
    assert _status(ok) == ["ok"]


def test_no_declared_scope_leaves_the_grant_table_as_the_only_judge(tmp_path):
    b = _bundle(tmp_path, **{"run.sh":
                             'rclone copy "$OUT" "b2p:$B2_BUCKET/checkpoints/r/"\n'})
    assert _status(jm.b2_write_preflight({}, b)) == ["ok"]


def test_scope_write_accepts_checkpoints_and_still_refuses_the_rest():
    assert jm.validate_scope({"scope": {"write": ["checkpoints/"]}}, [])["write"] \
        == ["checkpoints/"]
    assert jm.validate_scope({}, [])["write"] == ["jobs/"]      # default unchanged
    for bad in (["runsets/"], ["/"], ["evals/x/"]):
        with pytest.raises(jm.JobmetaError, match="scope.write"):
            jm.validate_scope({"scope": {"write": bad}}, [])


# ------------------------------------------------------------- report policy #
def test_report_is_silent_on_an_empty_scan():
    assert jm.b2_write_scope_report([]) == ([], False)


def test_report_confirms_each_distinct_grant_exactly_once():
    findings = [{"status": "ok", "file": "run.sh", "line": i, "verb": "copy",
                 "remote": "b2w", "prefix": "jobs/", "dest": "d", "detail": "x"}
                for i in range(5)]
    lines, refuse = jm.b2_write_scope_report(findings)
    assert refuse is False and len(lines) == 1


def test_every_pre_spend_surface_calls_the_shared_seam():
    """WIRING CHECK (textual, not behavioural): the four surfaces that can put a
    bundle on a box must all go through `b2_write_preflight`. The behaviour is
    proven by the cases above and by an end-to-end rehearsal; this only catches a
    call site being dropped — which is how the asset preflight lost the matrix
    lane once already."""
    here = os.path.dirname(os.path.abspath(__file__))
    # `herdd.py`'s submit path moved to `vastlib/jobs/submit.py` at plan §8
    # step 6d (the launcher has no bodies to check); `jobmatrix.py`'s went to
    # `vastlib/workflows/matrix.py` and `workflowctl.py`'s to
    # `vastlib/workflows/ctl.py` with the same wave. Pointing this check at the
    # thinned files would pass on nothing at all.
    for fname in ("vastlib/jobs/submit.py", "vastlib/workflows/matrix.py",
                  "rehearse.sh", "vastlib/workflows/ctl.py"):
        text = open(os.path.join(here, fname), encoding="utf-8").read()
        assert "b2_write_preflight" in text and "b2_write_scope_report" in text, \
            f"{fname} no longer runs the B2 write-scope preflight"


def test_allow_unscoped_downgrades_the_refusal_but_still_prints(tmp_path):
    findings = jm.scan_b2_writes(_bundle(tmp_path, **{"run.sh": V7_PUBLISH}))
    lines, refuse = jm.b2_write_scope_report(findings, allow_unscoped=True)
    assert refuse is False
    assert any("NOT ENTITLED" in ln for ln in lines)
