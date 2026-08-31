"""The live rclone config must be unreachable from a fixture.

Found 2026-08-22: `test_serve_attach_model_precedence.py` hands
`launch_serve.sh --dry-run` a fake `B2_S3_ENDPOINT` and the real `$HOME`, the
disk-autosize step calls `b2_sync.sh config`, and `~/.config/rclone/rclone.conf`
comes back with `endpoint = https://example.invalid`. Every B2 read/write on the
box then fails, including fleetd's queue listing.

The gates below are behavioural: run the real writers with `HOME` pointed at
`tmp_path`, so "the live config" is a sandbox path and the guard's own predicate
is what is under test.
"""
import os
import shutil
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
B2_SYNC = os.path.join(_HERE, "b2_sync.sh")
B2_REGION = os.path.join(_HERE, "b2_region.sh")
LAUNCH_SH = os.path.join(_HERE, "launch_serve.sh")

pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")

REAL_ENDPOINT = "https://s3.us-west-004.backblazeb2.com"


def _env(home, **extra):
    """A minimal environment with $HOME redirected and no ambient B2 creds."""
    e = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "TMPDIR")}
    e.update(HOME=str(home), B2_KEY_ID="fake", B2_APPLICATION_KEY="fake",
             B2_BUCKET="fake", B2_REGION="us-west-004")
    e.update({k: v for k, v in extra.items() if v is not None})
    return e


def _live(home):
    return home / ".config" / "rclone" / "rclone.conf"


def _run_config(home, **extra):
    return subprocess.run(["bash", B2_SYNC, "config"], capture_output=True,
                          text=True, timeout=120, env=_env(home, **extra),
                          cwd=_HERE)


# --------------------------------------------------------------------------- #
# 1. the refusal
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("endpoint", [
    "https://example.invalid",          # the literal that clobbered the live file
    "https://s3.example.invalid",       # test_workflow.py's spelling
    "https://example.com",              # RFC 2606 documentation name
    "http://localhost:9000",            # a local fake-S3 fixture
])
def test_a_reserved_endpoint_cannot_write_the_live_config(tmp_path, endpoint):
    """The defect itself. Unfixed, this writes the file and exits 0."""
    r = _run_config(tmp_path, B2_S3_ENDPOINT=endpoint)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "REFUSING to write the live rclone config" in r.stderr, r.stderr
    assert not _live(tmp_path).exists(), _live(tmp_path).read_text()


def test_the_refusal_names_the_reason_and_the_way_out(tmp_path):
    """A refusal belongs where the operator can still act on it."""
    r = _run_config(tmp_path, B2_S3_ENDPOINT="https://example.invalid")
    assert "reserved test name" in r.stderr, r.stderr
    assert "RCLONE_CONFIG" in r.stderr, r.stderr


def test_a_reserved_endpoint_clobbers_an_existing_live_config_when_unguarded(tmp_path):
    """The blast radius: the write REPLACES a good [b2] block in place, so the
    failure is silent until the next B2 call resolves example.invalid."""
    live = _live(tmp_path)
    live.parent.mkdir(parents=True)
    live.write_text("[b2]\ntype = s3\nendpoint = %s\n" % REAL_ENDPOINT)
    r = _run_config(tmp_path, B2_S3_ENDPOINT="https://example.invalid")
    assert r.returncode == 3, r.stdout + r.stderr
    assert REAL_ENDPOINT in live.read_text()
    assert "example.invalid" not in live.read_text()


def test_b2_region_config_carries_the_same_guard(tmp_path):
    """The OTHER writer of [b2]. Guarding only b2_sync.sh leaves the hole open."""
    r = subprocess.run(
        ["bash", "-c", 'set -e; . "$1"; b2_region_config', "_", B2_REGION],
        capture_output=True, text=True, timeout=120, cwd=_HERE,
        env=_env(tmp_path, B2_S3_ENDPOINT="https://example.invalid"))
    assert r.returncode == 3, r.stdout + r.stderr
    assert not _live(tmp_path).exists()


# --------------------------------------------------------------------------- #
# 2. the controls — the guard must not refuse anything legitimate
# --------------------------------------------------------------------------- #

def test_a_real_endpoint_still_writes_the_live_path(tmp_path):
    """Positive control. Without this the guard could be "refuse everything"."""
    r = _run_config(tmp_path, B2_S3_ENDPOINT=REAL_ENDPOINT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[b2]" in _live(tmp_path).read_text()


def test_RCLONE_CONFIG_is_the_sanctioned_escape_for_a_fixture(tmp_path):
    """A test that genuinely needs a config written from a fake endpoint points
    RCLONE_CONFIG somewhere private — and that keeps working."""
    rc = tmp_path / "private" / "rclone.conf"
    r = _run_config(tmp_path, B2_S3_ENDPOINT="https://example.invalid",
                    RCLONE_CONFIG=str(rc))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "example.invalid" in rc.read_text()
    assert not _live(tmp_path).exists()


# --------------------------------------------------------------------------- #
# 3. the chain that actually fired
# --------------------------------------------------------------------------- #

def test_a_launch_serve_dry_run_never_touches_the_live_config(tmp_path):
    """End to end on the reproducer: `--model b2:x --dry-run` with fake creds.
    The disk-autosize step calls `b2_sync.sh config` best-effort; unfixed it
    lands in $HOME. The dry run must still succeed — the guard degrades it to
    the static disk default, it does not abort a launch."""
    e = _env(tmp_path, B2_S3_ENDPOINT="https://example.invalid")
    empty = tmp_path / "empty.env"
    empty.write_text("")
    e["_LAUNCH_SERVE_ENV"] = str(empty)
    r = subprocess.run(["bash", LAUNCH_SH, "--model", "b2:x", "--dry-run",
                        "--api-key-file", str(tmp_path / "key.txt")],
                       capture_output=True, text=True, timeout=180, env=e,
                       cwd=_HERE)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not _live(tmp_path).exists(), _live(tmp_path).read_text()


# --------------------------------------------------------------------------- #
# 4. the pins: every writer of [b2] carries the guard, and none of them acquires
#    a delivery dependency doing it
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("writer", [B2_SYNC, B2_REGION])
def test_every_shipped_writer_of_the_b2_block_calls_the_guard(writer):
    src = open(writer, encoding="utf-8").read()
    assert "\n[b2]\n" in src or "[$REMOTE]" in src   # it really is a writer
    assert "b2_guard_live_rclone_config()" in src, \
        f"{writer} writes [b2] without defining the guard"
    assert 'b2_guard_live_rclone_config "$' in src, \
        f"{writer} defines the guard but never calls it"


def test_b2_sync_stays_self_contained():
    """`vastlib/jobs/bundle.py` ships b2_sync.sh FLAT — no b2_region.sh, no
    other sibling. A guard sourced from a file that does not ride along would
    refuse on every jobs box, and no test on this workstation could see it. So
    the guard is INLINE, and the one sibling this file reads stays optional."""
    src = open(B2_SYNC, encoding="utf-8").read()
    sources = [ln.strip() for ln in src.splitlines()
               if ln.strip().startswith((". \"$_HERE", "source \"$_HERE"))]
    for ln in sources:
        assert "|| true" in ln or "[ -f " in ln, \
            f"hard dependency on a sibling that the jobd bundle does not ship: {ln}"


def test_the_two_copies_of_the_guard_agree_on_the_predicate():
    """b2_region.sh carries a standalone copy (b2_sync.sh's wins when sourced).
    Drift between them is a hole, so the reserved-name list is pinned in both."""
    for path in (B2_SYNC, B2_REGION):
        src = open(path, encoding="utf-8").read()
        for tok in ("*.invalid", "*.test", "*.localhost", "example.com",
                    "PYTEST_CURRENT_TEST", "exit 3"):
            assert tok in src, f"{path} is missing {tok}"
