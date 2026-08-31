"""`b2x_boot.sh` rung 0: the CDN tier engages exactly where it should, and a
failed CDN pull can never leave the destination in a state the fallbacks skip.

Runs the shim with a STUB `cdn_pull.py` and a STUB `b2x`; no network, no B2, no
real prefix. The stub honours `--dest`/`--only`/`--stats-env` and can be told to
fail, so both halves of the ladder are exercised for real.

Two properties, and the second is the one that matters.

**Gating.** The tier must engage only for `base-models/` sources with all three
`B2_CDN_*` vars present, and must fall THROUGH — never fail — on anything else:
a checkpoint path, a missing var, an unmapped flag, an absent worker. Fail-open
is the whole contract; a CDN that can refuse a pull is worse than no CDN.

**Atomicity.** `cdn_pull.py` PREALLOCATES every destination at full size, so a
pull that dies half way leaves full-size files full of holes. Both fallbacks
would then skip them — b2x preallocates identically, and rclone's default
size-and-modtime compare sees a size match — and the box would train or serve on
zero-filled weights with every transfer in the chain reporting success. Nothing
downstream can detect that, which is why it is pinned here: a failed CDN attempt
must leave NO file behind at the destination.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SHIM = HERE / "onstart" / "b2x_boot.sh"

# Obviously fake. The real value is a URL-bearer secret and lives only in .env.
FAKE_ENV = {
    "B2_CDN_HOST": "cdn.invalid",
    "B2_CDN_BUCKET": "test-weights-cdn",
    "B2_CDN_PREFIX": "wtestdeadbeef0000000000000000000000000",
}

# Writes one file per manifest entry, like the real thing, and reports through
# --stats-env. CDN_PULL_FAIL=1 makes it PREALLOCATE and then die, which is the
# exact hazard shape: full-size output, nonzero exit.
STUB_CDN_PULL = r"""#!/usr/bin/env python3
import os, sys
a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
dest, only = a["--dest"], a.get("--only")
files = ["config.json", "model.safetensors", "sub/extra.bin"]
if only:
    files = [f for f in files if only in f]
if not files:
    sys.exit("nothing to fetch (check --only)")
for f in files:
    p = os.path.join(dest, f)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "wb") as fh:
        fh.truncate(4096)          # PREALLOCATE, exactly like cdn_pull.py
fail = os.environ.get("CDN_PULL_FAIL") == "1"
if a.get("--stats-env"):
    with open(a["--stats-env"], "w") as fh:
        fh.write("CDN_BYTES=4096\nCDN_SECS=1.0\nCDN_MBPS=4.1\n"
                 "CDN_CHUNKS=%d\nCDN_FAILED=%d\n" % (len(files), 1 if fail else 0))
if fail:
    print("chunk 3 FAILED: 429", file=sys.stderr)
    sys.exit(1)
"""

STUB_B2X = """#!/usr/bin/env bash
prev=""; se=""
for a in "$@"; do [ "$prev" = "--stats-env" ] && se="$a"; prev="$a"; done
[ -n "$se" ] && printf 'B2X_BYTES=99\\nB2X_SECS=1.0\\nB2X_MBPS=0.1\\nB2X_VERDICT=ok\\n' > "$se"
exit "${FAKE_B2X_RC:-0}"
"""


@pytest.fixture
def run(tmp_path):
    # Source a COPY of the shim, so `_B2X_SHIM_DIR` points into tmp and the
    # worker discovery under test is the real sibling lookup (`<shim>/../`) with
    # a stub at the other end -- not the repo's own cdn_pull.py, which would
    # reach the network and make the not-found case untestable.
    shim_dir = tmp_path / "onstart"
    shim_dir.mkdir()
    (shim_dir / "b2x_boot.sh").write_text(SHIM.read_text())
    worker = tmp_path / "cdn_pull.py"
    worker.write_text(STUB_CDN_PULL)
    b2x = tmp_path / "b2x"
    b2x.write_text(STUB_B2X)
    b2x.chmod(0o755)
    tally = tmp_path / "tally"

    def _run(body: str, env: dict | None = None, worker_present: bool = True):
        if not worker_present:
            worker.unlink(missing_ok=True)
        # B2_BUCKET deliberately unset: it short-circuits the shim's last-resort
        # rclone fetch of cdn_pull.py, which has no business running in a test.
        e = {"PATH": "/usr/bin:/bin",
             "B2X_INSTALL_DIR": str(tmp_path / "bin"),
             "JOBD_DIR": str(tmp_path / "nojobd"),
             **FAKE_ENV}
        e.update(env or {})
        script = (
            "set -uo pipefail\n"
            f"B2X_TALLY={tally}\n"
            f". {shim_dir / 'b2x_boot.sh'}\n"
            # stub the install ladder: it would reach for the network and is not
            # what this file tests.
            f"b2x_ensure() {{ B2X={b2x}; return 0; }}\n"
            f"{body}\n"
        )
        p = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                           env=e)
        return p.returncode, p.stdout + p.stderr, (
            tally.read_text() if tally.exists() else "")

    return _run


# --- the tier engages, and says so ------------------------------------------

def test_a_base_model_pull_is_served_by_the_cdn(run, tmp_path):
    dst = tmp_path / "base"
    rc, out, tally = run(f"b2x_pull b2:bkt/base-models/qwen35-9b {dst}")
    assert rc == 0
    assert "via cdn (3 chunks, 4.1 MB/s)" in out
    assert "\tcdn\t" not in tally and tally.startswith("cdn\t")
    assert (dst / "config.json").is_file()
    assert (dst / "sub" / "extra.bin").is_file(), "subdirectories must survive the move"
    assert not list(tmp_path.glob("*.cdn_tmp.*")), "temp dir left behind"


@pytest.mark.parametrize("src", [
    "base-models/qwen35-9b",            # bare, no bucket
    "b2:bkt/base-models/qwen35-9b",     # the train.sh / serve_vllm spelling
    "b2:bkt/base-models/qwen35-9b/",    # jobd's asset pull adds a trailing slash
    "bkt/base-models/qwen35-9b",
])
def test_every_call_site_spelling_maps_to_the_same_model(run, tmp_path, src):
    dst = tmp_path / "base"
    rc, out, _ = run(f"b2x_pull {src} {dst}")
    assert rc == 0 and "via cdn" in out
    assert (dst / "config.json").is_file()


def test_a_deadline_is_honoured_as_a_bound_not_a_refusal(run, tmp_path):
    """jobd's asset pull passes its own ceiling; the tier must accept it."""
    dst = tmp_path / "base"
    rc, out, _ = run(f"b2x_pull b2:bkt/base-models/m/ {dst} --deadline 600s")
    assert rc == 0 and "via cdn" in out


def test_a_caller_stats_env_is_answered_in_the_b2x_shape(run, tmp_path):
    """A site that brought its own --stats-env reads B2X_* keys out of it; the
    CDN's own CDN_* spelling would read as an empty stats file."""
    dst, se = tmp_path / "base", tmp_path / "stats"
    rc, out, _ = run(
        f"b2x_pull b2:bkt/base-models/m {dst} --stats-env {se}\n"
        'echo "T=$B2X_LAST_TRANSPORT B=$B2X_LAST_BYTES"')
    assert rc == 0
    assert "T=cdn B=4096" in out
    body = se.read_text()
    assert "B2X_BYTES=4096" in body and "B2X_TRANSPORT=cdn" in body


def test_a_single_file_pull_uses_only_and_lands_at_the_file_path(run, tmp_path):
    dst = tmp_path / "cfg.json"
    rc, out, _ = run(f"b2x_pull b2:bkt/base-models/m/config.json {dst}")
    assert rc == 0 and "via cdn" in out
    assert dst.is_file() and not dst.is_dir()


# --- everything else falls THROUGH, never fails ------------------------------

@pytest.mark.parametrize("src", [
    "b2:bkt/checkpoints/run-1",                 # not mirrored, must not be tried
    "b2:bkt/eval-env/env-1.tar.zst",
    "b2:bkt/a/b/base-models/m",                 # too deep to be a bucket prefix
    "b2:bkt/base-models",                       # no slug
])
def test_a_non_mirror_source_never_reaches_the_cdn(run, tmp_path, src):
    dst = tmp_path / "d"
    rc, out, tally = run(f"b2x_pull {src} {dst}")
    assert rc == 0, "the b2x rung must still run"
    assert "via cdn" not in out and "cdn miss" not in out
    assert tally.startswith("ok\t")


@pytest.mark.parametrize("drop", sorted(FAKE_ENV))
def test_a_partial_cdn_env_falls_through_and_says_why_once(run, tmp_path, drop):
    dst = tmp_path / "d"
    rc, out, tally = run(
        f"b2x_pull b2:bkt/base-models/m {dst}\nb2x_pull b2:bkt/base-models/m {dst}",
        env={drop: ""})
    assert rc == 0 and tally.startswith("ok\t")
    assert out.count("B2_CDN_HOST/BUCKET/PREFIX not in the box env") == 1


def test_an_unmappable_flag_falls_through_rather_than_dropping_it(run, tmp_path):
    """--exclude has no manifest equivalent. Honouring it by ignoring it would
    silently widen what the caller asked for."""
    dst = tmp_path / "d"
    rc, out, tally = run(
        f"b2x_pull b2:bkt/base-models/m {dst} --exclude 'checkpoint-*/**'")
    assert rc == 0 and "via cdn" not in out
    assert tally.startswith("ok\t")


def test_no_worker_on_the_box_is_a_logged_miss_not_a_failure(run, tmp_path):
    dst = tmp_path / "d"
    rc, out, tally = run(f"b2x_pull b2:bkt/base-models/m {dst}",
                         worker_present=False)
    assert rc == 0 and tally.startswith("ok\t")
    assert "cdn_pull.py not on this box" in out


def test_the_kill_switch_is_silent_and_total(run, tmp_path):
    dst = tmp_path / "d"
    rc, out, tally = run(f"b2x_pull b2:bkt/base-models/m {dst}",
                         env={"B2X_CDN_DISABLE": "1"})
    assert rc == 0 and "via cdn" not in out and tally.startswith("ok\t")


# --- the atomicity invariant -------------------------------------------------

def test_a_failed_cdn_pull_leaves_nothing_at_the_destination(run, tmp_path):
    """THE invariant. The stub preallocates and then dies, exactly like the real
    worker losing a chunk to a 429. If those full-size holey files reached the
    destination, b2x and rclone would both skip them and the box would run on
    zero-filled weights with every layer reporting success."""
    dst = tmp_path / "base"
    rc, out, tally = run(f"b2x_pull b2:bkt/base-models/m {dst}",
                         env={"CDN_PULL_FAIL": "1"})
    assert rc == 0, "the b2x fallback must still succeed"
    assert "cdn miss -> b2x" in out
    assert tally.startswith("fallback\t") or tally.startswith("ok\t")
    assert not (dst / "model.safetensors").exists(), \
        "a preallocated, unverified file reached the destination"
    assert not list(dst.iterdir()) if dst.exists() else True
    assert not list(tmp_path.glob("*.cdn_tmp.*")), "temp dir left behind on failure"


def test_the_worker_never_writes_into_the_destination_directly(run, tmp_path):
    """Belt and braces on the same invariant, stated as a property of the argv:
    --dest must be a temp sibling, never the caller's dst."""
    dst = tmp_path / "base"
    argv = tmp_path / "argv"
    spy = tmp_path / "spy.py"
    spy.write_text(
        "import sys\n"
        f"open({str(argv)!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
        "raise SystemExit(2)\n")
    run(f"b2x_pull b2:bkt/base-models/m {dst}", env={"B2X_CDN_PULL": str(spy)})
    seen = argv.read_text()
    assert f"--dest {dst}." in seen and f"--dest {dst} " not in seen
    assert ".cdn_tmp." in seen
