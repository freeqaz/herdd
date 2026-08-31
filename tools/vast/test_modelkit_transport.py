"""`b2_transport.sh`: write scope, the write-last marker, and pull exclusion.

Runs bash with a STUB `rclone` first on `PATH` that records its argv and returns
what the test tells it to. **No network, no B2, no uploads** — the harness is the
one `test_bake_push_latest.py` uses, for the same reason: the properties under
test are ordering and refusal, which are decided before any byte moves.

The four that have all bitten in the field:

  * a push OUTSIDE the box's grant is a 403 discovered at the END of a 52 GiB
    upload, so the refusal has to be at the start;
  * `PUSHED.json` is written LAST and only after a read-back, so `has` can never
    see a prefix whose payload is still in flight;
  * the marker is EXCLUDED from the pull, or every restore lands a file the
    merged-dir fingerprint has never seen and reports UNEXPECTED;
  * `rclone copyto` is never used — it HEADs the destination, and B2 has
    hours-long windows where HeadObject on a not-yet-existing key returns 403.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import jobmeta  # noqa: E402

TRANSPORT = HERE / "modelkit" / "b2_transport.sh"
STUB_RCLONE = """#!/usr/bin/env bash
echo "rclone $*" >> "$RCLONE_CALL_LOG"
case "$1" in
  lsf)   [ "${STUB_LSF_RC:-0}" = 0 ] || exit 1
         printf '%s\\n' ${STUB_LSF_OUT:-} ;;
  copy)  [ "${STUB_COPY_RC:-0}" = 0 ] || exit 1 ;;
  rcat)  cat >/dev/null; [ "${STUB_RCAT_RC:-0}" = 0 ] || exit 1 ;;
esac
exit 0
"""


@pytest.fixture
def sh(tmp_path):
    """Run the transport with a stub rclone and no b2x. Returns (rc, calls)."""
    binp = tmp_path / "bin"
    binp.mkdir()
    (binp / "rclone").write_text(STUB_RCLONE)
    (binp / "rclone").chmod(0o755)
    log = tmp_path / "rclone.log"
    log.write_text("")

    def run(*args, script=TRANSPORT, **env):
        log.write_text("")            # per-invocation, so `calls` is THIS run's
        e = dict(os.environ)
        e.update(PATH=f"{binp}:{os.environ['PATH']}",
                 RCLONE_CALL_LOG=str(log),
                 RCLONE_CONFIG=str(tmp_path / "rclone.conf"),
                 B2_BUCKET="DRY-RUN-BUCKET",
                 B2X_DISABLE="1")           # rclone leg only; b2x is a binary
        e.update({k: str(v) for k, v in env.items()})
        r = subprocess.run(["bash", str(script), *args], env=e,
                           capture_output=True, text=True)
        calls = [ln for ln in log.read_text().splitlines() if ln]
        return r, calls

    run.log = log
    return run


@pytest.fixture
def src(tmp_path):
    d = tmp_path / "merged"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\0" * 16)
    return d


# ------------------------------------------------------------------ write scope
@pytest.mark.parametrize("key", [
    "jobs/whatever",              # the b2w grant, not the publish grant
    "base-models/qwen36-27b",     # read-only territory
    "checkpointsX/sneaky",        # a prefix that only LOOKS like the grant
])
def test_push_outside_the_grant_is_refused_BEFORE_any_upload(sh, src, key):
    r, calls = sh("push", str(src), key)
    assert r.returncode == 2, r.stderr
    assert "REFUSING to push" in r.stderr
    assert calls == [], f"the refusal must precede every transfer: {calls}"


def test_a_missing_argument_is_a_USAGE_error_and_not_a_scope_refusal(sh, src):
    """Distinct exit codes for distinct facts: rc 2 means "you asked for a key
    this box may not write", rc 1 means "you did not name a key at all"."""
    r, calls = sh("push", str(src))
    assert r.returncode == 1 and calls == []
    assert "usage" in r.stderr


def test_push_inside_the_grant_proceeds(sh, src):
    r, calls = sh("push", str(src), "checkpoints/demo/abc/model",
                  STUB_LSF_OUT="config.json model.safetensors")
    assert r.returncode == 0, r.stderr
    assert any(c.startswith("rclone copy ") for c in calls)


def test_the_default_scope_is_exactly_the_publish_key_grant():
    """The script's default and `jobmeta.B2_BOX_GRANTS` are two statements of
    ONE grant. The table is not restated in the script — this is the test that
    keeps the default honest without a second copy of the table."""
    body = TRANSPORT.read_text()
    assert 'WRITE_PREFIXES="checkpoints/"' in body
    assert jobmeta.B2_BOX_GRANTS["b2p"] == ("checkpoints/",)
    assert jobmeta.B2_BOX_GRANTS["b2"] == ()      # the read key writes nothing


def test_the_override_can_MOVE_the_refusal_but_not_remove_it(sh, src):
    r, _ = sh("push", str(src), "models/demo",
              MODELKIT_B2_WRITE_PREFIXES="models/ checkpoints/",
              STUB_LSF_OUT="config.json model.safetensors")
    assert r.returncode == 0, r.stderr

    r, calls = sh("push", str(src), "elsewhere/demo",
                  MODELKIT_B2_WRITE_PREFIXES="models/ checkpoints/")
    assert r.returncode == 2 and calls == []

    # An EMPTY override falls back to the default rather than allowing
    # everything: a transport that accepts every prefix is not a scope check.
    r, calls = sh("push", str(src), "models/demo",
                  MODELKIT_B2_WRITE_PREFIXES="")
    assert r.returncode == 2 and calls == []


def test_the_push_remote_is_the_grant_seen_from_the_other_side(sh, src):
    r, calls = sh("push", str(src), "checkpoints/demo/abc/model",
                  MODELKIT_B2_WRITE_REMOTE="b2alt",
                  STUB_LSF_OUT="config.json model.safetensors")
    assert r.returncode == 0, r.stderr
    assert any("b2alt:DRY-RUN-BUCKET/checkpoints/demo/abc/model/" in c
               for c in calls)
    assert not any(" b2p:" in c for c in calls)


# --------------------------------------------------- the write-last publish
def test_the_marker_is_written_LAST_and_only_after_a_read_back(sh, src):
    r, calls = sh("push", str(src), "checkpoints/demo/abc/model",
                  STUB_LSF_OUT="config.json model.safetensors")
    assert r.returncode == 0, r.stderr
    verbs = [c.split()[1] for c in calls]
    assert verbs == ["copy", "lsf", "rcat"], calls
    # …and the rcat target IS the marker. `has` stats exactly this key, so a
    # marker written before the read-back would publish a prefix whose payload
    # is still in flight.
    assert calls[-1].endswith("/checkpoints/demo/abc/model/PUSHED.json")


def test_a_short_read_back_leaves_the_prefix_UNPUBLISHED(sh, src):
    """A payload nobody can see beats a marker that lies: if the listing comes
    back short of what is on disk, the marker is never written and `has` keeps
    returning 1, so the next box merges instead of restoring a truncated dir."""
    r, calls = sh("push", str(src), "checkpoints/demo/abc/model",
                  STUB_LSF_OUT="config.json")     # 1 remote vs 2 local
    assert r.returncode == 1
    assert "read-back SHORT" in r.stderr
    assert not any(" rcat " in c for c in calls)


def test_a_failed_payload_upload_never_writes_the_marker(sh, src):
    r, calls = sh("push", str(src), "checkpoints/demo/abc/model",
                  STUB_COPY_RC="1")
    assert r.returncode == 1 and "marker NOT written" in r.stderr
    assert not any(" rcat " in c for c in calls)


def test_a_failed_marker_write_fails_the_push(sh, src):
    r, _ = sh("push", str(src), "checkpoints/demo/abc/model",
              STUB_LSF_OUT="config.json model.safetensors", STUB_RCAT_RC="1")
    assert r.returncode == 1 and "marker write FAILED" in r.stderr


def test_push_never_uses_copyto(sh, src):
    """`copyto` HEADs the destination, and B2 has hours-long windows where
    HeadObject on a not-yet-existing key returns 403 — the push silently fails
    and the box reports success. The fallback is list-based `copy --include`
    plus `rcat` for the one small marker object."""
    r, calls = sh("push", str(src), "checkpoints/demo/abc/model",
                  STUB_LSF_OUT="config.json model.safetensors")
    assert r.returncode == 0
    assert not any(" copyto " in c for c in calls)
    # …and not anywhere in the file either, outside the comment that explains
    # why. Reading only the argv would miss a `copyto` on a branch this stub
    # never takes.
    code = [ln for ln in TRANSPORT.read_text().splitlines()
            if not ln.lstrip().startswith("#")]
    assert not any("copyto" in ln for ln in code)


def test_push_include_list_is_ROOT_ANCHORED(sh, src, tmp_path):
    """A stray subdirectory must not ride into the published prefix. `/name`
    anchors each include at the source root."""
    (src / "sub").mkdir()
    (src / "sub" / "config.json").write_text("{}")
    r, calls = sh("push", str(src), "checkpoints/demo/abc/model",
                  STUB_LSF_OUT="config.json model.safetensors")
    assert r.returncode == 0
    copy = next(c for c in calls if c.startswith("rclone copy "))
    assert "--include /config.json" in copy
    assert "--include /sub" not in copy


def test_push_refuses_a_source_that_is_not_a_directory(sh, tmp_path):
    f = tmp_path / "notadir"
    f.write_text("x")
    r, calls = sh("push", str(f), "checkpoints/demo/abc/model")
    assert r.returncode == 1 and calls == []


# ------------------------------------------------------------- has / pull
def test_has_reads_the_MARKER_and_nothing_else(sh):
    r, calls = sh("has", "checkpoints/demo/abc/model")
    assert r.returncode == 0
    assert len(calls) == 1 and calls[0].endswith("/model/PUSHED.json")

    r, _ = sh("has", "checkpoints/demo/abc/model", STUB_LSF_RC="1")
    assert r.returncode == 1


def test_pull_EXCLUDES_the_marker(sh, tmp_path):
    """The marker lives in the payload prefix so `has` is one stat, but it is
    transport metadata: landing it in the merged dir adds a file the merged-dir
    fingerprint has never seen and turns every restore into an UNEXPECTED-file
    failure."""
    dest = tmp_path / "dest"
    r, calls = sh("pull", "checkpoints/demo/abc/model", str(dest))
    assert r.returncode == 0 and dest.is_dir()
    copy = next(c for c in calls if c.startswith("rclone copy "))
    assert "--exclude PUSHED.json" in copy


def test_no_bucket_is_a_refusal_on_every_verb(sh, src):
    for args in (("has", "checkpoints/x"),
                 ("pull", "checkpoints/x", "/tmp/nope-dest"),
                 ("push", str(src), "checkpoints/x")):
        r, calls = sh(*args, B2_BUCKET="")
        assert r.returncode == 1 and calls == []
        assert "B2_BUCKET unset" in r.stderr
