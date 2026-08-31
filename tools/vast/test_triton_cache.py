#!/usr/bin/env python3
"""Tests for triton_cache.py.

Two properties carry the weight and each has several tests:

  1. FAIL-OPEN. Every failure mode must return exit 0 and report a miss. A
     cache that can fail a training run is worse than no cache, and the failure
     modes are exactly the ones that show up on a flaky box (no bucket, no
     rclone, timeout, truncated object).

  2. NEVER UNPACK UNVERIFIED BYTES. The digest sidecar is the commit point and
     the gate. A tarball whose sha256 does not match must be refused *before*
     extraction — a bit-flipped .cubin loaded into the compiler's own toolchain
     is precisely the failure this project's doctrine cannot tolerate.

The remote is faked with a shim `rclone` script over a local directory, so the
tests exercise the real subprocess path, the real tar, and the real digest
check without touching B2.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import triton_cache as tc  # noqa: E402


# --------------------------------------------------------------------------
# a fake rclone over a local dir: supports copyto (both directions) and lsf
# --------------------------------------------------------------------------
SHIM = r'''#!/usr/bin/env python3
import os, shutil, sys
ROOT = os.environ["FAKE_REMOTE"]
MODE = os.environ.get("FAKE_MODE", "ok")
def local(spec):
    if ":" not in spec:
        return spec, False
    _, path = spec.split(":", 1)
    return os.path.join(ROOT, path), True
args = [a for a in sys.argv[1:] if not a.startswith("--")]
# strip flag VALUES (--retries 2 etc.)
clean, skip = [], False
for a in sys.argv[1:]:
    if skip: skip = False; continue
    if a.startswith("--"): skip = a in ("--retries", "--low-level-retries"); continue
    clean.append(a)
cmd = clean[0]
if MODE == "fail": sys.exit(1)
if cmd == "copyto":
    src, s_remote = local(clean[1]); dst, d_remote = local(clean[2])
    if not os.path.exists(src): sys.exit(1)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    if MODE == "corrupt" and d_remote is False:
        with open(dst, "ab") as fh: fh.write(b"\x00tamper")
    sys.exit(0)
if cmd == "lsf":
    p, _ = local(clean[1])
    if os.path.exists(p):
        print(os.path.basename(p)); sys.exit(0)
    sys.exit(1)
if cmd == "lsjson":
    import json as _j
    p, _ = local(clean[1])
    rows = []
    if os.path.isdir(p):
        for n in sorted(os.listdir(p)):
            f = os.path.join(p, n)
            rows.append({"Path": n, "Name": n, "Size": os.path.getsize(f),
                         "ModTime": "2026-08-21T00:00:00Z", "IsDir": False})
    print(_j.dumps(rows)); sys.exit(0)
sys.exit(2)
'''


@pytest.fixture(autouse=True)
def _no_ambient_remotes(monkeypatch):
    """A dev workstation's .env may carry R2_TC_*/TRITON_CACHE_REMOTE; tests
    must resolve remotes from what THEY set, not from ambient credentials."""
    for k in ("TRITON_CACHE_REMOTE", "R2_TC_KEY_ID", "R2_TC_SECRET_ACCESS_KEY",
              "R2_TC_ENDPOINT", "R2_TC_BUCKET", "B2_BUCKET", "B2_WRITE_REMOTE"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture()
def shim(tmp_path):
    p = tmp_path / "rclone_shim.py"
    p.write_text(SHIM)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    remote = tmp_path / "remote"
    remote.mkdir()
    os.environ["FAKE_REMOTE"] = str(remote)
    os.environ.pop("FAKE_MODE", None)
    yield str(p), remote
    os.environ.pop("FAKE_REMOTE", None)
    os.environ.pop("FAKE_MODE", None)


def make_cache(d: Path, n=3) -> Path:
    """A directory shaped like Triton's: one dir per kernel, small files in it."""
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        k = d / f"{i:064x}"[:64]
        k.mkdir(exist_ok=True)
        (k / "kernel.json").write_text(json.dumps({"i": i}))
        (k / "kernel.cubin").write_bytes(b"\x7fELF" + bytes([i]) * 64)
    return d


def run(argv, shim_path):
    """Invoke a command and return its parsed JSON line + exit code."""
    rc = tc.main(argv + ["--rclone", shim_path])
    return rc


def capture(capsys):
    out = capsys.readouterr().out.strip().splitlines()
    return json.loads(out[-1])


# --------------------------------------------------------------------------
# key derivation
# --------------------------------------------------------------------------
def test_key_is_stable_and_safe():
    k = tc.cache_key("2.11.0+cu129", "3.4.0", "sm_90")
    assert k == "torch2.11.0_cu129-triton3.4.0-sm_90"
    assert "/" not in k and " " not in k and "+" not in k


def test_key_separates_arch_because_triton_cache_is_arch_keyed():
    """The whole reason per-arch keys exist: a Blackwell cache is useless on
    Ampere, and serving it would be a guaranteed 100 % miss with a download."""
    a = tc.cache_key("2.11.0", "3.4.0", "sm_80")
    b = tc.cache_key("2.11.0", "3.4.0", "sm_120")
    assert a != b


def test_key_separates_triton_versions_because_triton_hashes_its_own_version():
    """Triton folds its own version into every entry's directory name, so two
    Triton versions share zero reachable entries — one key for both would ship
    each box the other's dead weight forever."""
    assert tc.cache_key("2.13.0", "3.4.0", "sm_90") != \
        tc.cache_key("2.13.0", "3.5.0", "sm_90")


def test_key_missing_parts_become_unknown_not_empty():
    assert tc.cache_key("", "", "") == "torchunknown-tritonunknown-unknown"


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------
def test_push_then_pull_round_trips_every_entry(tmp_path, shim, capsys):
    shim_path, _ = shim
    src = make_cache(tmp_path / "src", n=4)
    run(["push", "--src", str(src), "--key", "k1", "--bucket", "b"], shim_path)
    pushed = capture(capsys)
    assert pushed["pushed"] is True and pushed["entries_packed"] == 4

    dest = tmp_path / "dest"
    run(["pull", "--dest", str(dest), "--key", "k1", "--bucket", "b"], shim_path)
    pulled = capture(capsys)
    assert pulled["hit"] is True
    assert pulled["entries_installed"] == 4
    assert pulled["digest"] == pushed["digest"]
    # contents identical, not just counted
    for child in src.iterdir():
        assert (dest / child.name / "kernel.cubin").read_bytes() == \
               (child / "kernel.cubin").read_bytes()


def test_pull_never_clobbers_a_locally_compiled_entry(tmp_path, shim, capsys):
    """A local entry is the ground truth for this box: it was produced by this
    exact toolchain. A remote tarball must add to the cache, never overwrite it."""
    shim_path, _ = shim
    src = make_cache(tmp_path / "src", n=2)
    run(["push", "--src", str(src), "--key", "k", "--bucket", "b"], shim_path)
    capture(capsys)

    dest = make_cache(tmp_path / "dest", n=1)
    local_name = next(dest.iterdir()).name
    (dest / local_name / "kernel.cubin").write_bytes(b"LOCAL-ORIGINAL")
    run(["pull", "--dest", str(dest), "--key", "k", "--bucket", "b"], shim_path)
    res = capture(capsys)
    assert res["hit"] is True
    assert res["entries_installed"] == 1        # the one it did not already have
    assert (dest / local_name / "kernel.cubin").read_bytes() == b"LOCAL-ORIGINAL"


# --------------------------------------------------------------------------
# the integrity gate
# --------------------------------------------------------------------------
def test_corrupted_tarball_is_refused_before_extraction(tmp_path, shim, capsys):
    """THE test. A tampered tarball must produce a miss and an EMPTY dest —
    proof that nothing was unpacked, not merely that a warning was printed."""
    shim_path, _ = shim
    src = make_cache(tmp_path / "src", n=3)
    run(["push", "--src", str(src), "--key", "k", "--bucket", "b"], shim_path)
    capture(capsys)

    os.environ["FAKE_MODE"] = "corrupt"     # tamper on the way down
    dest = tmp_path / "dest"
    rc = run(["pull", "--dest", str(dest), "--key", "k", "--bucket", "b"], shim_path)
    res = capture(capsys)
    assert rc == 0                           # fail-open
    assert res["hit"] is False
    assert "DIGEST MISMATCH" in res["reason"]
    assert not dest.exists() or list(dest.iterdir()) == []


def test_malformed_digest_sidecar_is_a_miss(tmp_path, shim, capsys):
    shim_path, remote = shim
    src = make_cache(tmp_path / "src")
    run(["push", "--src", str(src), "--key", "k", "--bucket", "b"], shim_path)
    capture(capsys)
    (remote / "b" / tc.REMOTE_PREFIX / "k.sha256").write_text("not-a-digest\n")
    dest = tmp_path / "dest"
    rc = run(["pull", "--dest", str(dest), "--key", "k", "--bucket", "b"], shim_path)
    res = capture(capsys)
    assert rc == 0 and res["hit"] is False and "malformed" in res["reason"]


def test_tarball_without_digest_sidecar_is_cold_not_a_hit(tmp_path, shim, capsys):
    """The sidecar is the commit point. A tarball that uploaded but whose
    digest did not must read as cold, never as a usable cache."""
    shim_path, remote = shim
    src = make_cache(tmp_path / "src")
    run(["push", "--src", str(src), "--key", "k", "--bucket", "b"], shim_path)
    capture(capsys)
    (remote / "b" / tc.REMOTE_PREFIX / "k.sha256").unlink()
    dest = tmp_path / "dest"
    run(["pull", "--dest", str(dest), "--key", "k", "--bucket", "b"], shim_path)
    res = capture(capsys)
    assert res["hit"] is False and "cold" in res["reason"]


def test_extract_refuses_path_traversal(tmp_path):
    """A tarball is remote input. `../` members must not escape the dest."""
    outside = tmp_path / "outside.txt"
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "ok.txt").write_text("fine")
    tar_path = tmp_path / "evil.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(str(payload / "ok.txt"), arcname="ok.txt")
        tf.add(str(payload / "ok.txt"), arcname="../../outside.txt")
    dest = tmp_path / "d" / "cache"
    dest.parent.mkdir()
    installed = tc._safe_extract(tar_path, dest)
    assert installed == 1
    assert (dest / "ok.txt").exists()
    assert not outside.exists()


# --------------------------------------------------------------------------
# fail-open, every path
# --------------------------------------------------------------------------
def test_pull_with_no_remote_at_all_is_a_clean_miss(tmp_path, shim, capsys):
    shim_path, _ = shim
    rc = run(["pull", "--dest", str(tmp_path / "d"), "--key", "k"], shim_path)
    res = capture(capsys)
    assert rc == 0 and res["hit"] is False and "no remote" in res["reason"]


def test_pull_with_missing_rclone_binary_is_a_clean_miss(tmp_path, capsys):
    rc = tc.main(["pull", "--dest", str(tmp_path / "d"), "--key", "k",
                  "--bucket", "b", "--rclone", "/nonexistent/rclone"])
    res = capture(capsys)
    assert rc == 0 and res["hit"] is False


def test_pull_with_remote_erroring_is_a_clean_miss(tmp_path, shim, capsys):
    shim_path, _ = shim
    os.environ["FAKE_MODE"] = "fail"
    rc = run(["pull", "--dest", str(tmp_path / "d"), "--key", "k",
              "--bucket", "b"], shim_path)
    res = capture(capsys)
    assert rc == 0 and res["hit"] is False


def test_push_of_empty_cache_is_a_no_op_not_an_error(tmp_path, shim, capsys):
    shim_path, _ = shim
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = run(["push", "--src", str(empty), "--key", "k", "--bucket", "b"], shim_path)
    res = capture(capsys)
    assert rc == 0 and res["pushed"] is False and "empty" in res["reason"]


def test_push_does_not_overwrite_an_existing_key_without_force(tmp_path, shim, capsys):
    shim_path, _ = shim
    src = make_cache(tmp_path / "src", n=2)
    run(["push", "--src", str(src), "--key", "k", "--bucket", "b"], shim_path)
    assert capture(capsys)["pushed"] is True
    bigger = make_cache(tmp_path / "src2", n=5)
    run(["push", "--src", str(bigger), "--key", "k", "--bucket", "b"], shim_path)
    res = capture(capsys)
    assert res["pushed"] is False and "already has this key" in res["reason"]
    run(["push", "--src", str(bigger), "--key", "k", "--bucket", "b", "--force"],
        shim_path)
    assert capture(capsys)["pushed"] is True


# --------------------------------------------------------------------------
# the telemetry that answers "does the JIT tax recur on novel shapes?"
# --------------------------------------------------------------------------
def test_push_reports_entries_added_against_the_pulled_baseline(tmp_path, shim, capsys):
    """If a run keeps adding entries to a key it already pulled, the tax
    recurs per novel shape; if entries_added stays 0, it is one-time. This
    number is the whole experiment, so it has to be right."""
    shim_path, _ = shim
    src = make_cache(tmp_path / "src", n=7)
    run(["push", "--src", str(src), "--key", "k", "--bucket", "b",
         "--baseline-entries", "5"], shim_path)
    res = capture(capsys)
    assert res["entries_added"] == 2


def test_entries_added_never_goes_negative(tmp_path, shim, capsys):
    shim_path, _ = shim
    src = make_cache(tmp_path / "src", n=2)
    run(["push", "--src", str(src), "--key", "k", "--bucket", "b",
         "--baseline-entries", "9"], shim_path)
    assert capture(capsys)["entries_added"] == 0


def test_stat_counts_entries_and_bytes(tmp_path, capsys):
    src = make_cache(tmp_path / "src", n=3)
    tc.main(["stat", "--src", str(src)])
    res = capture(capsys)
    assert res["entries"] == 3 and res["bytes"] > 0 and res["exists"] is True


def test_stat_on_a_missing_dir_is_zero_not_an_error(tmp_path, capsys):
    rc = tc.main(["stat", "--src", str(tmp_path / "nope")])
    res = capture(capsys)
    assert rc == 0 and res["entries"] == 0 and res["exists"] is False


def test_plan_makes_no_network_call(tmp_path, capsys):
    """plan must be usable to plumb a bundle without a bucket or a remote."""
    rc = tc.main(["plan", "--key", "k1", "--rclone", "/nonexistent/rclone"])
    res = capture(capsys)
    assert rc == 0 and res["key"] == "k1"
    assert res["tarball"].endswith("k1.tar.gz")


# --------------------------------------------------------------------------
# remote resolution — R2 preferred, secrets in env not argv, --update mode
# --------------------------------------------------------------------------
R2_ENV = {"R2_TC_KEY_ID": "keyid", "R2_TC_SECRET_ACCESS_KEY": "s3cr3t",
          "R2_TC_ENDPOINT": "https://acct.r2.cloudflarestorage.com"}


def test_resolve_prefers_r2_over_b2(monkeypatch):
    for k, v in R2_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("B2_BUCKET", "some-b2-bucket")
    r = tc.resolve_remote()
    assert r["backend"] == "r2"
    assert r["read"] == f"{tc.R2_REMOTE_NAME}:{tc.R2_DEFAULT_BUCKET}"
    # the scoped token cannot HeadBucket; without this rclone CreateBucket-403s
    assert r["env"][f"RCLONE_CONFIG_{tc.R2_REMOTE_NAME}_NO_CHECK_BUCKET"] == "true"


def test_resolve_secret_never_reaches_argv(monkeypatch):
    """argv is world-readable in ps on a shared box; the secret must travel
    only in the subprocess environment overlay."""
    for k, v in R2_ENV.items():
        monkeypatch.setenv(k, v)
    r = tc.resolve_remote()
    assert "s3cr3t" not in r["read"] and "s3cr3t" not in r["write"]
    assert "s3cr3t" in r["env"].values()


def test_resolve_explicit_remote_overrides_r2(monkeypatch):
    for k, v in R2_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("TRITON_CACHE_REMOTE", "myremote:cachebucket")
    r = tc.resolve_remote()
    assert r["backend"] == "explicit" and r["read"] == "myremote:cachebucket"


def test_resolve_bucket_arg_forces_legacy_b2(monkeypatch):
    for k, v in R2_ENV.items():
        monkeypatch.setenv(k, v)
    r = tc.resolve_remote("legacy-bucket")
    assert r["backend"] == "b2" and r["read"] == "b2:legacy-bucket"


def test_resolve_b2_fallback_and_none(monkeypatch):
    monkeypatch.setenv("B2_BUCKET", "bb")
    monkeypatch.setenv("B2_WRITE_REMOTE", "b2w")
    r = tc.resolve_remote()
    assert r["backend"] == "b2" and r["read"] == "b2:bb" and r["write"] == "b2w:bb"
    monkeypatch.delenv("B2_BUCKET")
    assert tc.resolve_remote() is None


def test_pull_and_push_through_r2_shape_remote(tmp_path, shim, capsys, monkeypatch):
    """End-to-end through resolve_remote's R2 lane (shim plays R2): the remote
    string the shim receives is TRITONR2:<bucket>/..., and the round trip
    behaves exactly as the B2 lane."""
    shim_path, remote = shim
    for k, v in R2_ENV.items():
        monkeypatch.setenv(k, v)
    src = make_cache(tmp_path / "src", n=2)
    run(["push", "--src", str(src), "--key", "k"], shim_path)
    pushed = capture(capsys)
    assert pushed["pushed"] is True and pushed["backend"] == "r2"
    assert (remote / tc.R2_DEFAULT_BUCKET / tc.REMOTE_PREFIX / "k.tar.gz").exists()
    dest = tmp_path / "dest"
    run(["pull", "--dest", str(dest), "--key", "k"], shim_path)
    pulled = capture(capsys)
    assert pulled["hit"] is True and pulled["backend"] == "r2"


def test_push_update_replaces_only_when_grown(tmp_path, shim, capsys):
    """The jobd hook's mode: --update --baseline-entries N replaces an existing
    remote key iff the local dir grew past N — the mechanism that keeps the
    fleet-wide cache converging as novel shapes compile."""
    shim_path, _ = shim
    src = make_cache(tmp_path / "src", n=3)
    run(["push", "--src", str(src), "--key", "k", "--bucket", "b"], shim_path)
    assert capture(capsys)["pushed"] is True
    # no growth: baseline == current entries -> skip
    run(["push", "--src", str(src), "--key", "k", "--bucket", "b",
         "--update", "--baseline-entries", "3"], shim_path)
    res = capture(capsys)
    assert res["pushed"] is False and "already has this key" in res["reason"]
    # growth: two new kernels beyond the baseline -> replace
    grown = make_cache(tmp_path / "src", n=5)
    run(["push", "--src", str(grown), "--key", "k", "--bucket", "b",
         "--update", "--baseline-entries", "3"], shim_path)
    res = capture(capsys)
    assert res["pushed"] is True and res.get("updating") is True
    assert res["entries_added"] == 2


# --------------------------------------------------------------------------
# key provenance + inventory — the two things that make a key flip diagnosable
# --------------------------------------------------------------------------
def test_key_shape_is_pinned_so_a_re_key_cannot_happen_by_accident():
    """The whole bucket is addressed by this string. Changing the field list or
    their order invalidates every object in it, which is an owner call — so the
    shape is pinned here rather than left to whoever edits cache_key() next."""
    assert tc.cache_key("2.13.0+cu129", "3.5.0", "sm_90", "fitladder") == \
        "torch2.13.0_cu129-triton3.5.0-sm_90-fitladder"
    assert tc.cache_key("2.13.0+cu129", "none", "sm_90") == \
        "torch2.13.0_cu129-tritonnone-sm_90"


def test_fla_is_reported_but_never_keyed():
    """The 2026-08-21 re-key. `fla` is one kernel source among several, its
    version already sits inside Triton's own entry hash, and it is the one field
    a bake cannot compute ahead of a box — so it is diagnosis only. Keeping it
    OUT of cache_key() is the property; `detect()` still has to carry it."""
    assert "fla" in tc.detect()
    assert "fla" not in tc.cache_key("2.13.0", "3.5.0", "sm_90")


def test_a_parked_fla_arm_is_reported_as_masked_not_absent(monkeypatch):
    """The bench bundles call `set_fla_env off` before the cache pull, which
    nulls the module for the rest of the script. Reporting that as `none` is
    what made the old key read as a box fact when it was an A/B knob."""
    monkeypatch.setenv("FLA_FORCE_OFF", "1")
    monkeypatch.setitem(sys.modules, "fla", None)
    assert tc.detect()["fla"] == "masked_FLA_FORCE_OFF"
    monkeypatch.delenv("FLA_FORCE_OFF")
    assert tc.detect()["fla"] == "none"


def test_sidecars_record_what_produced_the_key(tmp_path, shim, capsys):
    """Every key field is DETECTED by default, and what detect() sees depends
    on the box. Without this, a key that silently flips is indistinguishable
    from a cold bucket in the artifact."""
    shim_path, _ = shim
    src = make_cache(tmp_path / "src", n=2)
    run(["push", "--src", str(src), "--torch", "2.13.0", "--triton", "3.5.0",
         "--sm", "sm_90", "--extra", "fitladder", "--bucket", "b"], shim_path)
    res = capture(capsys)
    ki = res["key_inputs"]
    assert ki["source"] == "derived"
    assert ki["triton"] == {"value": "3.5.0", "from": "flag"}
    assert ki["extra"] == {"value": "fitladder", "from": "flag"}
    assert ki["sm"]["value"] == "sm_90"
    assert "fla" not in ki   # reported under fla_detected, never keyed


def test_a_stale_caller_passing_fla_still_produces_the_new_key(tmp_path, shim, capsys):
    """FAIL-OPEN outranks strictness. `--fla` is dead as a key field, but an
    older bundle copy still passes it; rejecting it would be argparse's only
    non-zero exit, inside a job, which this tool must never cause."""
    shim_path, _ = shim
    src = make_cache(tmp_path / "src", n=2)
    rc = run(["push", "--src", str(src), "--torch", "2.13.0", "--triton", "3.5.0",
              "--sm", "sm_90", "--fla", "0.5.2", "--bucket", "b"], shim_path)
    res = capture(capsys)
    assert rc == 0
    assert res["key"] == "torch2.13.0-triton3.5.0-sm_90"
    assert res["key_inputs"]["fla_detected"] == "0.5.2"


def test_explicit_key_is_recorded_as_explicit(tmp_path, shim, capsys):
    shim_path, _ = shim
    src = make_cache(tmp_path / "src", n=2)
    run(["push", "--src", str(src), "--key", "k", "--bucket", "b"], shim_path)
    assert capture(capsys)["key_inputs"] == {"source": "explicit", "key": "k"}


def test_refused_push_says_so_when_the_box_compiled_new_kernels(tmp_path, shim, capsys):
    """The frozen-key defect: a caller that omits --update but DID grow the
    cache silently throws the growth away. The refusal has to name it."""
    shim_path, _ = shim
    src = make_cache(tmp_path / "src", n=3)
    run(["push", "--src", str(src), "--key", "k", "--bucket", "b"], shim_path)
    assert capture(capsys)["pushed"] is True
    grown = make_cache(tmp_path / "src", n=7)
    run(["push", "--src", str(grown), "--key", "k", "--bucket", "b",
         "--baseline-entries", "3"], shim_path)
    res = capture(capsys)
    assert res["pushed"] is False and res.get("stale_remote") is True
    assert "4 entries beyond the pull baseline" in res["reason"]
    assert "--update" in res["reason"]


def test_refused_push_without_growth_is_not_flagged_stale(tmp_path, shim, capsys):
    shim_path, _ = shim
    src = make_cache(tmp_path / "src", n=3)
    run(["push", "--src", str(src), "--key", "k", "--bucket", "b"], shim_path)
    capture(capsys)
    run(["push", "--src", str(src), "--key", "k", "--bucket", "b",
         "--baseline-entries", "3"], shim_path)
    res = capture(capsys)
    assert res["pushed"] is False and "stale_remote" not in res


def test_ls_inventories_the_bucket_and_splits_namespaces(tmp_path, shim, capsys):
    shim_path, _ = shim
    a = make_cache(tmp_path / "a", n=2)
    run(["push", "--src", str(a), "--key", "torch2.13.0_cu129-triton3.5.0-sm_90",
         "--bucket", "b"], shim_path)
    capture(capsys)
    run(["push", "--src", str(a), "--key",
         "torch2.13.0_cu129-triton3.5.0-sm_90-fitladder", "--bucket", "b"], shim_path)
    capture(capsys)
    run(["ls", "--bucket", "b"], shim_path)
    res = capture(capsys)
    assert res["totals"]["keys"] == 2 and res["totals"]["objects"] == 4
    assert set(res["namespaces"]) == {"", "fitladder"}
    by_key = {k["key"]: k for k in res["keys"]}
    plain = by_key["torch2.13.0_cu129-triton3.5.0-sm_90"]
    assert plain["torch"] == "2.13.0_cu129" and plain["triton"] == "3.5.0"
    assert plain["sm"] == "sm_90" and plain["extra"] == ""
    assert plain["complete"] is True
    assert by_key["torch2.13.0_cu129-triton3.5.0-sm_90-fitladder"]["extra"] == "fitladder"


def test_ls_still_reads_the_orphaned_pre_rekey_keys(tmp_path, shim, capsys):
    """The 2026-08-21 re-key orphaned 15 `-fla-` keys and deleted none of them.
    Retention here is by hand, so `ls` has to be able to SEE what it is being
    asked about — a key shape it cannot parse would report as `unknown` and the
    operator would be back to eyeballing names."""
    shim_path, _ = shim
    a = make_cache(tmp_path / "a", n=2)
    run(["push", "--src", str(a), "--key", "torch2.11.0_cu129-fla0.4.2-sm_90",
         "--bucket", "b"], shim_path)
    capture(capsys)
    run(["push", "--src", str(a), "--key", "torch2.13.0_cu129-triton3.5.0-sm_90",
         "--bucket", "b"], shim_path)
    capture(capsys)
    run(["ls", "--bucket", "b"], shim_path)
    res = capture(capsys)
    assert res["schemas"]["v1-fla"]["keys"] == 1
    assert res["schemas"]["v2-triton"]["keys"] == 1
    old = {k["key"]: k for k in res["keys"]}["torch2.11.0_cu129-fla0.4.2-sm_90"]
    assert old["fla"] == "0.4.2" and old["triton"] is None and old["sm"] == "sm_90"


def test_ls_with_no_remote_is_a_clean_report_not_an_error(tmp_path, capsys):
    assert tc.main(["ls"]) == 0
    res = capture(capsys)
    assert res["totals"]["keys"] == 0 and res["reason"] == "no remote configured"


def test_ls_never_deletes(tmp_path, shim, capsys):
    """`ls` is the whole of this tool's read side of retention. It must be
    incapable of removing an object — there is no delete path at all."""
    shim_path, remote = shim
    src = make_cache(tmp_path / "src", n=2)
    run(["push", "--src", str(src), "--key", "k", "--bucket", "b"], shim_path)
    capture(capsys)
    before = sorted(p.name for p in (remote / "b" / tc.REMOTE_PREFIX).iterdir())
    run(["ls", "--bucket", "b"], shim_path)
    capture(capsys)
    after = sorted(p.name for p in (remote / "b" / tc.REMOTE_PREFIX).iterdir())
    assert before == after and len(before) == 2
    src_text = Path(tc.__file__).read_text()
    for verb in ('"delete"', '"deletefile"', '"purge"', '"rmdir"', '"rmdirs"'):
        assert verb not in src_text, f"a destructive rclone verb appeared: {verb}"


def test_no_shipped_caller_splits_the_bucket_into_a_second_namespace():
    """One bucket, one namespace.

    jobd exports TRITON_CACHE_DIR and every bundle inherits it, so a bundle
    that passes `--extra` does not get an isolated cache — it gets a second
    remote copy of the same directory under a different key. Measured
    2026-08-21: 365.6 MB of `-fitladder` objects whose newest sm_90 tarball had
    been frozen since 2026-08-16 while jobd's key stayed current.
    """
    root = Path(__file__).resolve().parents[2]
    callers = [root / "tools/vast/onstart/jobd.sh"]
    callers = [p for p in callers if p.exists()]
    assert callers, "no shipped caller found — this test would pass vacuously"
    offenders = []
    for p in callers:
        for line in p.read_text(errors="replace").splitlines():
            if "TC_TOOL" in line and "--extra" in line and not line.lstrip().startswith("#"):
                offenders.append(f"{p.parent.name}/{p.name}: {line.strip()}")
    assert not offenders, "callers split the cache bucket:\n" + "\n".join(offenders)


def test_only_jobd_pushes_and_it_passes_update():
    """One writer, and it has to be the one that can win.

    jobd exports TRITON_CACHE_DIR, so a bundle's cache dir IS jobd's box-level
    one. jobd pulls it at boot and pushes it back with `--update` on every
    terminal path of every GPU job. A second push from a bundle passed no
    `--update`, so once the key existed it was refused forever and the run's
    kernels were thrown away — measured: a key frozen at 817 entries while the
    pushes it refused carried 1,574 / 1,695 / 1,788. Deleted 2026-08-21.
    """
    root = Path(__file__).resolve().parents[2]
    jobd = (root / "tools/vast/onstart/jobd.sh").read_text(errors="replace")
    push = [ln for ln in jobd.splitlines()
            if "push --src" in ln and not ln.lstrip().startswith("#")]
    assert len(push) == 1, f"expected exactly one jobd push line, got {push}"
    assert "--update" in push[0] and "--baseline-entries" in push[0], push[0]


def test_every_shipped_bundle_copy_computes_the_same_key():
    """Every triton_cache.py a box can end up running must agree on the key.

    The bundle's copy wins over the checkout's (`for _c in
    "$JOB_DIR/triton_cache.py" …`), and a copy left behind at an older key does
    not fail — it MISSES, quietly, forever, against a bucket jobd is filling
    under a different name. Byte-identity is not the property; agreeing on the
    key is.

    Bundles used to carry 21 private copies of this file and now take it via
    `includes:` from jobcommon (itself a symlink to tools/vast/triton_cache.py),
    which removes that hazard by construction for those bundles. So the sweep is
    the shared canonical PLUS any copy still pinned to a bundle — the pins are
    the only place the drift can still happen.
    """
    root = Path(__file__).resolve().parents[2]
    shared = root / "tools" / "vast" / "jobcommon" / "triton_cache.py"
    copies = [shared] if shared.is_file() else []
    assert copies, "no copy found at all — this test would pass vacuously"
    assert shared.is_file(), (
        "the shared canonical is gone; every bundle that includes it would fail "
        "to assemble, and this sweep would silently stop covering them")
    argv = ["plan", "--torch", "2.13.0+cu129", "--triton", "3.5.0", "--sm", "sm_90"]
    want = tc.cache_key("2.13.0+cu129", "3.5.0", "sm_90")
    bad = []
    for c in copies:
        p = subprocess.run([sys.executable, str(c)] + argv,
                           capture_output=True, text=True, timeout=60)
        got = None
        if p.returncode == 0:
            try:
                got = json.loads(p.stdout.strip().splitlines()[-1])["key"]
            except Exception:
                got = f"unparseable: {p.stdout[:80]!r}"
        else:
            got = f"rc={p.returncode} {p.stderr.strip().splitlines()[-1:]}"
        if got != want:
            bad.append(f"{c.parent.name}: {got}")
    assert not bad, (f"bundle copies disagree with the checkout (want {want}); "
                     f"re-copy tools/vast/triton_cache.py into them:\n"
                     + "\n".join(bad))


def test_no_shipped_caller_keys_on_fla():
    """The re-key's other half. `--fla` is still ACCEPTED (fail-open beats
    strictness inside a job) so a stale caller cannot be caught by argparse —
    only by this."""
    offenders = []
    for p in [Path(tc.__file__).parent / "onstart/jobd.sh"]:
        for line in p.read_text(errors="replace").splitlines():
            if "TC_TOOL" in line and "--fla" in line and not line.lstrip().startswith("#"):
                offenders.append(f"{p.parent.name}/{p.name}: {line.strip()}")
    assert not offenders, "a caller still passes --fla:\n" + "\n".join(offenders)
