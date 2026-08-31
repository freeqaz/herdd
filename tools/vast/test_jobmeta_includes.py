"""Tests for `includes:` — shared bundle files overlaid at bundle time.

Two things are being defended here, and they fail in different ways:

  * TRANSPORT. A bundle that declares an include must arrive at a box WITH that
    file. The two entry points that build a bundle (`bundle_sha256`,
    `write_bundle`) are called separately by workflowctl, so both must
    materialize or the address won't name the bytes.
  * MIGRATION. Moving a file out of nine bundles into one shared dir must not
    change what any box receives.

CPU-only, stdlib + pytest. Run: pytest tools/vast/test_jobmeta_includes.py
"""
from __future__ import annotations

import os
import sys
import tarfile
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import jobmeta as jm  # noqa: E402

def _bundle(tmp_path, *, includes=("launch_plan.sh",), local_copies=None):
    d = tmp_path / "bundle"
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.sh").write_text("#!/usr/bin/env bash\necho hi\n")
    inc = "\n".join(f"  - {n}" for n in includes)
    (d / "job-config.yaml").write_text(textwrap.dedent(f"""\
        version: 1
        name: t
        entrypoint: run.sh
        includes:
        {inc}
        results:
          - "out/**"
        """))
    for name, body in (local_copies or {}).items():
        (d / name).write_text(body)
    return str(d)


# --- resolution ---------------------------------------------------------------

def test_unknown_include_is_refused_by_name(tmp_path):
    b = _bundle(tmp_path, includes=("not_a_real_file.sh",))
    with pytest.raises(jm.JobmetaError) as e:
        jm.resolve_includes(b)
    # The message must name the file AND where to look — a bare "not found"
    # sends the reader hunting through six call sites.
    assert "not_a_real_file.sh" in str(e.value)
    assert "jobcommon" in str(e.value)


def test_include_plus_local_copy_is_refused_even_when_identical(tmp_path):
    """The un-migrated state. Identical content is the DANGEROUS case, not the
    safe one: it validates, ships, and leaves the next edit with two homes."""
    shared = open(os.path.join(jm.JOBCOMMON_DIR, "launch_plan.sh")).read()
    b = _bundle(tmp_path, local_copies={"launch_plan.sh": shared})
    with pytest.raises(jm.JobmetaError) as e:
        jm.resolve_includes(b)
    assert "byte-identical" in str(e.value)

    b2 = _bundle(tmp_path / "x", local_copies={"launch_plan.sh": "# different\n"})
    with pytest.raises(jm.JobmetaError) as e:
        jm.resolve_includes(b2)
    assert "DIFFERENT content" in str(e.value)


def test_include_escaping_the_shared_dir_is_refused():
    for bad in ("../herdd.py", "/etc/passwd", "a\\b"):
        with pytest.raises(jm.JobmetaError):
            jm._normalize_includes([bad])


def test_duplicate_include_is_refused():
    with pytest.raises(jm.JobmetaError):
        jm._normalize_includes(["launch_plan.sh", "launch_plan.sh"])


# --- materialization ----------------------------------------------------------

def test_bundle_without_includes_yields_itself_unchanged(tmp_path):
    """No includes -> no copy, no temp dir, and the SAME content address as
    before this feature existed. Every un-migrated bundle depends on this."""
    d = tmp_path / "plain"
    d.mkdir()
    (d / "run.sh").write_text("#!/usr/bin/env bash\ntrue\n")
    (d / "job-config.yaml").write_text(
        "version: 1\nname: t\nentrypoint: run.sh\nresults:\n  - \"out/**\"\n")
    with jm.materialize_bundle(str(d)) as staged:
        assert staged == str(d)


def test_materialize_overlays_the_shared_file(tmp_path):
    b = _bundle(tmp_path)
    with jm.materialize_bundle(b) as staged:
        got = open(os.path.join(staged, "launch_plan.sh"), "rb").read()
    want = open(os.path.join(jm.JOBCOMMON_DIR, "launch_plan.sh"), "rb").read()
    assert got == want
    # and the staging dir is gone once the context closes
    assert not os.path.exists(staged)


def test_both_bundle_entry_points_agree(tmp_path):
    """workflowctl calls bundle_sha256 and write_bundle SEPARATELY. If only one
    materialized, it would upload a blob whose sha is not its name."""
    b = _bundle(tmp_path)
    addressed = jm.bundle_sha256(b)
    written = jm.write_bundle(b, str(tmp_path / "out.tar.zst"))["sha256"]
    assert addressed == written

    # and the tar actually contains the shared file
    blob = open(tmp_path / "out.tar.zst", "rb").read()
    tar = jm.decompress_zst(blob)
    with tarfile.open(fileobj=__import__("io").BytesIO(tar)) as tf:
        names = {os.path.basename(n) for n in tf.getnames()}
    assert "launch_plan.sh" in names


def test_sha_is_stable_across_materializations(tmp_path):
    """The temp dir's name must not leak into the address — otherwise every
    submit uploads a 'new' bundle and dedupe never hits."""
    b = _bundle(tmp_path)
    assert jm.bundle_sha256(b) == jm.bundle_sha256(b)


# --- gates stay wired ---------------------------------------------------------

def test_validate_accepts_an_included_entrypoint(tmp_path):
    """A bundle may name a shared file as its entrypoint. Without this the
    include mechanism could never absorb the duplicated run.sh."""
    d = tmp_path / "epinc"
    d.mkdir()
    (d / "job-config.yaml").write_text(
        "version: 1\nname: t\nentrypoint: launch_plan.sh\n"
        "includes:\n  - launch_plan.sh\nresults:\n  - \"out/**\"\n")
    cfg, _ = jm.validate_job_config(jm.load_job_config(str(d)), str(d))
    assert cfg["includes"] == ["launch_plan.sh"]


# --- the extracted bundle (recovery paths) ------------------------------------
# A bundle that has been through materialize_bundle carries its includes as
# FILES. Validating such a tree with the authoring-tree rules refuses it —
# "declared in includes: AND present in the bundle" — which is exactly the
# un-migrated state the rule exists to catch, and exactly NOT what this is.
# It shipped that way and broke `job retarget --reconstruct` for every migrated
# bundle: the one command whose whole job is recovering a lost ticket.

def test_round_trip_write_extract_validate(tmp_path):
    """THE regression. write -> extract -> validate, the literal sequence
    `_retarget_reconstruct` runs."""
    b = _bundle(tmp_path)
    sha = jm.write_bundle(b, str(tmp_path / "o.tar.zst"))["sha256"]
    dest = str(tmp_path / "x")
    jm.extract_bundle(str(tmp_path / "o.tar.zst"), dest, expect_sha=sha)
    cfg, _ = jm.validate_job_config(jm.load_job_config(dest), dest,
                                    materialized=True)
    assert cfg["includes"] == ["launch_plan.sh"]
    assert os.path.isfile(os.path.join(dest, "launch_plan.sh"))


def test_extracted_bundle_is_still_refused_without_the_flag(tmp_path):
    """The strict rule must survive — `materialized` is a caller assertion
    about which kind of tree it holds, not a general loosening."""
    b = _bundle(tmp_path)
    sha = jm.write_bundle(b, str(tmp_path / "o.tar.zst"))["sha256"]
    dest = str(tmp_path / "x")
    jm.extract_bundle(str(tmp_path / "o.tar.zst"), dest, expect_sha=sha)
    with pytest.raises(jm.JobmetaError, match="AND present"):
        jm.validate_job_config(jm.load_job_config(dest), dest)


def test_materialized_mode_refuses_a_declared_include_that_is_absent(tmp_path):
    """In an extracted tree the include's PRESENCE is the success condition, so
    absence means the tar is incomplete — the one thing this mode must catch."""
    b = _bundle(tmp_path)                    # authoring tree: no local copy
    with pytest.raises(jm.JobmetaError, match="MISSING"):
        jm.validate_job_config(jm.load_job_config(b), b, materialized=True)


def test_b2_scan_sees_text_inside_an_included_file(tmp_path, monkeypatch):
    """The B2 write preflight walks shipped files, not the folder. Moving a
    B2-writing script into the shared dir must not make it invisible."""
    fake = tmp_path / "jobcommon"
    fake.mkdir()
    (fake / "pusher.sh").write_text(
        "#!/usr/bin/env bash\nrclone copy out b2read:example-runs-bucket/x\n")
    monkeypatch.setattr(jm, "JOBCOMMON_DIR", str(fake))
    b = _bundle(tmp_path, includes=("pusher.sh",))
    findings = jm.scan_b2_writes(b)
    assert any(f["file"] == "pusher.sh" for f in findings), findings
