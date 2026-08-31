"""The two hash grades: what each claims, and that it refuses what it must.

Portable (`pytest -m "not integration"`): no network, no B2, no model. Every
fixture is a handful of small files in tmp_path — the rules under test are
`os.scandir` + `hashlib`, and a 52 GiB dir would exercise nothing extra.

The property that matters and is easy to lose: **grade A sees SIZE, grade B sees
CONTENT**. A same-size edit is invisible to grade A by design (which is why a
restored dir is gated on grade B) and a same-name/different-size file is caught
by both. A test suite that only ever checked "the digests differ when I change
something" would pass with the two grades swapped, so both directions are
asserted.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from modelkit import dirhash, merged_fingerprint as mfp  # noqa: E402

PY = sys.executable
CLI = HERE / "modelkit" / "merged_fingerprint.py"


@pytest.fixture
def merged(tmp_path):
    """A stand-in merged dir: two shards, an index, a wrapper config."""
    d = tmp_path / "merged"
    d.mkdir()
    (d / "model-00001-of-00002.safetensors").write_bytes(b"A" * 4096)
    (d / "model-00002-of-00002.safetensors").write_bytes(b"B" * 2048)
    (d / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    (d / "config.json").write_text('{"model_type": "test"}')
    return d


# ------------------------------------------------------------ grade A: name/size
def test_fingerprint_round_trips(merged):
    fp = mfp.fingerprint_dir(str(merged))
    assert fp["n_files"] == 4
    assert [r["f"] for r in fp["files"]] == sorted(r["f"] for r in fp["files"])
    assert mfp.compare_fingerprint(fp, mfp.fingerprint_dir(str(merged))) == []


def test_fingerprint_is_stable_under_mtime_and_order(merged):
    """The digest is over the sorted NAME/SIZE list, so touching a file or
    rewriting it with identical bytes cannot move it. A fingerprint that drifted
    on mtime would fail every honest restore."""
    before = mfp.fingerprint_dir(str(merged))
    p = merged / "config.json"
    body = p.read_bytes()
    p.unlink()
    p.write_bytes(body)
    os.utime(p, (0, 0))
    assert mfp.fingerprint_dir(str(merged)) == before


def test_fingerprint_detects_missing_unexpected_and_size(merged):
    frozen = mfp.fingerprint_dir(str(merged))

    (merged / "config.json").unlink()
    assert any("MISSING" in p for p in
               mfp.compare_fingerprint(frozen, mfp.fingerprint_dir(str(merged))))

    (merged / "config.json").write_text('{"model_type": "test"}')
    (merged / "surprise.bin").write_bytes(b"x")
    probs = mfp.compare_fingerprint(frozen, mfp.fingerprint_dir(str(merged)))
    assert any("UNEXPECTED" in p and "surprise.bin" in p for p in probs)

    (merged / "surprise.bin").unlink()
    (merged / "model-00002-of-00002.safetensors").write_bytes(b"B" * 2049)
    probs = mfp.compare_fingerprint(frozen, mfp.fingerprint_dir(str(merged)))
    assert probs == ["SIZE model-00002-of-00002.safetensors: 2049 != recorded 2048"]


def test_grade_a_is_BLIND_to_a_same_size_content_swap(merged):
    """Not a bug — the documented limit. The merge is not bit-reproducible
    across hosts, so grade A cannot make a content claim; grade B is what gates
    a dir that was PULLED. Pinned so nobody 'fixes' it into a lie."""
    frozen = mfp.fingerprint_dir(str(merged))
    (merged / "model-00001-of-00002.safetensors").write_bytes(b"Z" * 4096)
    assert mfp.compare_fingerprint(frozen, mfp.fingerprint_dir(str(merged))) == []


def test_size_exempt_forgives_size_but_not_absence(merged):
    """The merge marker embeds a host path and a timestamp, so its SIZE is
    host-dependent by construction. The exemption is from the SIZE comparison
    only: the file must still be PRESENT (its content is the marker check's
    job), and an exemption that also forgave absence would silently accept a
    merged dir carrying no record of the merge that produced it."""
    marker = merged / ".merge_ok.json"
    marker.write_text(json.dumps({"adapter": "/short"}))
    frozen = mfp.fingerprint_dir(str(merged))

    marker.write_text(json.dumps({"adapter": "/a/much/longer/host/path"}))
    assert mfp.compare_fingerprint(frozen, mfp.fingerprint_dir(str(merged)),
                                   size_exempt=(".merge_ok.json",)) == []

    marker.unlink()
    probs = mfp.compare_fingerprint(frozen, mfp.fingerprint_dir(str(merged)),
                                    size_exempt=(".merge_ok.json",))
    assert any("MISSING" in p and ".merge_ok.json" in p for p in probs)


def test_exclude_defaults_to_the_bundle_era_rule(merged):
    """`fingerprint_dir`'s default exclusion set is EMPTY, matching the bundle
    copies bit for bit. Only the CLI opts in to dropping transport artefacts."""
    (merged / "PUSHED.json").write_text('{"complete": true, "files": 4}')
    assert mfp.fingerprint_dir(str(merged))["n_files"] == 5
    assert mfp.fingerprint_dir(
        str(merged), exclude=mfp.DEFAULT_EXCLUDE)["n_files"] == 4


# ---------------------------------------------------------- the PUSHED receipt
def test_receipt_corroborates_the_payload_count(merged):
    (merged / "PUSHED.json").write_text(json.dumps({"complete": True, "files": 4}))
    rec, why = mfp.read_pushed_receipt(str(merged))
    assert why is None and rec["files"] == 4
    fp = mfp.fingerprint_dir(str(merged), exclude=mfp.DEFAULT_EXCLUDE)
    assert mfp.corroborate_receipt(rec, fp) == []


def test_receipt_catches_a_short_restore(merged):
    """The failure `has` cannot see: the marker exists (so the prefix reads as
    published) and the payload is short. `has` only stats the marker."""
    (merged / "PUSHED.json").write_text(json.dumps({"complete": True, "files": 4}))
    (merged / "config.json").unlink()
    rec, _ = mfp.read_pushed_receipt(str(merged))
    fp = mfp.fingerprint_dir(str(merged), exclude=mfp.DEFAULT_EXCLUDE)
    probs = mfp.corroborate_receipt(rec, fp)
    assert len(probs) == 1 and "pushed 4 payload files" in probs[0]


@pytest.mark.parametrize("body,expect", [
    ({"complete": False, "files": 4}, "complete"),
    ({"files": 4}, "complete"),
    ({"complete": True}, "not an integer count"),
    ({"complete": True, "files": "4"}, "not an integer count"),
    ({"complete": True, "files": True}, "not an integer count"),
])
def test_receipt_refuses_every_unusable_shape(merged, body, expect):
    (merged / "PUSHED.json").write_text(json.dumps(body))
    rec, _ = mfp.read_pushed_receipt(str(merged))
    fp = mfp.fingerprint_dir(str(merged), exclude=mfp.DEFAULT_EXCLUDE)
    probs = mfp.corroborate_receipt(rec, fp)
    assert probs and any(expect in p for p in probs)


def test_an_unreadable_receipt_is_an_error_but_an_absent_one_is_not(merged):
    """Absence is a FACT about a freshly merged dir, which never had a receipt.
    A receipt that exists and cannot be parsed is a different thing: it is a
    claim that cannot be believed, and only the caller knows which it expected
    — so the module reports and the CLI decides."""
    rec, why = mfp.read_pushed_receipt(str(merged))
    assert rec is None and "absent" in why

    (merged / "PUSHED.json").write_text("{not json")
    rec, why = mfp.read_pushed_receipt(str(merged))
    assert rec is None and "unreadable" in why

    (merged / "PUSHED.json").write_text("[1, 2]")
    rec, why = mfp.read_pushed_receipt(str(merged))
    assert rec is None and "not a JSON object" in why


# -------------------------------------------------------------------- the CLI
def _cli(*args):
    return subprocess.run([PY, str(CLI), *args], capture_output=True, text=True)


def test_cli_emits_then_verifies(merged, tmp_path):
    out = tmp_path / "fp.json"
    r = _cli("--dir", str(merged), "--emit", "--out", str(out))
    assert r.returncode == 0 and out.is_file()

    r = _cli("--dir", str(merged), "--frozen", str(out))
    assert r.returncode == 0
    assert r.stdout.strip().splitlines()[-1] == mfp.VERDICT_OK

    (merged / "config.json").write_text("{}")
    r = _cli("--dir", str(merged), "--frozen", str(out))
    assert r.returncode == 1                      # checked, and it differs
    assert r.stdout.strip().splitlines()[-1] == mfp.VERDICT_REFUSED


@pytest.mark.parametrize("args", [
    (),                                           # nothing to compare against
    ("--frozen", "/nonexistent/fp.json"),         # cannot read the comparand
    ("--receipt",),                               # receipt demanded, absent
])
def test_cli_fails_CLOSED_with_exit_2_when_it_cannot_check(merged, args):
    """2 is 'could not check', 1 is 'checked and it differs'. Collapsing them
    is how a broken gate reads as a passing one to a caller that only tests
    `rc != 0` on the happy path."""
    r = _cli("--dir", str(merged), *args)
    assert r.returncode == 2, r.stderr
    assert r.stdout.strip().splitlines()[-1] == mfp.VERDICT_REFUSED


def test_the_gate_script_is_SELF_CONTAINED(merged, tmp_path):
    """The serve lane stages this ONE file to a box. Copy it somewhere with no
    package around it and it must still run — a `from . import` here produces a
    gate that cannot start, discovered at the moment it was supposed to refuse
    something."""
    alone = tmp_path / "staged" / "merged_fingerprint.py"
    alone.parent.mkdir()
    alone.write_bytes(CLI.read_bytes())
    r = subprocess.run([PY, str(alone), "--dir", str(merged), "--emit"],
                       capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["fingerprint"]["n_files"] == 4


# ------------------------------------------------------------ grade B: content
def test_grade_b_sees_what_grade_a_cannot(merged):
    frozen = dirhash.manifest(merged)
    a_frozen = mfp.fingerprint_dir(str(merged))
    (merged / "model-00001-of-00002.safetensors").write_bytes(b"Z" * 4096)

    assert mfp.compare_fingerprint(a_frozen, mfp.fingerprint_dir(str(merged))) == []
    probs = dirhash.compare_manifest(frozen, dirhash.manifest(merged))
    assert len(probs) == 1 and probs[0].startswith("SHA256 model-00001")


def test_rollup_is_a_single_pin_over_names_sizes_and_content(merged):
    man = dirhash.manifest(merged)
    pin = dirhash.rollup(man)
    assert len(pin) == 64 and dirhash.rollup(dirhash.manifest(merged)) == pin

    renamed = dict(man)
    renamed["renamed.json"] = renamed.pop("config.json")
    assert dirhash.rollup(renamed) != pin        # the NAME is inside the pin


def test_grade_b_ignores_transport_artefacts_by_default(merged):
    pin = dirhash.rollup(dirhash.manifest(merged))
    (merged / "PUSHED.json").write_text('{"complete": true, "files": 4}')
    (merged / ".complete").write_text("4096")
    assert dirhash.rollup(dirhash.manifest(merged)) == pin
    assert dirhash.rollup(dirhash.manifest(merged, ignore=())) != pin


def test_grade_b_reports_size_instead_of_sha_when_both_moved(merged):
    """A truncated shard differs in both, and the SIZE is the actionable half.
    Printing a sha for every shard of a 52 GiB model buries it."""
    frozen = dirhash.manifest(merged)
    (merged / "model-00002-of-00002.safetensors").write_bytes(b"B" * 7)
    probs = dirhash.compare_manifest(frozen, dirhash.manifest(merged))
    assert probs == ["SIZE model-00002-of-00002.safetensors: 7 != 2048"]
