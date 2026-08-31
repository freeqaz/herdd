"""The model-artifact registry: schema, derivation, lookups, and the seeds.

Portable: the registry is committed JSON and the checks are pure. Nothing here
reaches B2 — the point of the file is precisely that a consumer can learn where
an artifact lives WITHOUT asking B2, because asking B2 costs a listing and
answers a different question ("is there something at this path") than the one
that matters ("is this the artifact I named").
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from modelkit import merge_guard, registry as reg  # noqa: E402

PY = sys.executable
CLI = HERE / "modelkit" / "registry.py"

MERGED = {
    "schema_version": 1,
    "id": "demo",
    "kind": "merged",
    "family": "qwen36-27b",
    "base": "qwen36-27b",
    "adapter_ident": "ab" * 32,
    "b2_root": "checkpoints/demo-merged",
    "b2_model": "checkpoints/demo-merged/ababababab" + "ab" + "/model",
    "b2_guards": "checkpoints/demo-merged/ababababab" + "ab" + "/guards",
    "fingerprint_sha256": "cd" * 32,
    "n_files": 27,
    "content_sha256": None,
    "serve": {"served_name": "DEMO", "dtype": "bfloat16", "max_len": 20480,
              "min_vram_gb": 96, "tp": 1, "lora_forbidden": True},
}
BASE = {
    "schema_version": 1, "id": "qwen36-27b", "kind": "base",
    "family": "qwen36-27b", "b2_root": "base-models/qwen36-27b",
}


def _reg(tmp_path, *entries):
    d = tmp_path / "registry"
    d.mkdir(exist_ok=True)
    for e in entries:
        (d / f"{e['id']}.json").write_text(json.dumps(e))
    return str(d)


# ------------------------------------------------------------------- the module
def test_the_module_wins_over_the_same_named_data_directory():
    """`modelkit/registry.py` and `modelkit/registry/` share a name, and Python
    resolves the MODULE (a directory with no `__init__.py` is only a namespace
    package, which loses to a file loader). Pinned because adding an
    `__init__.py` to the data dir would silently turn every
    `from modelkit import registry` into an empty namespace package."""
    assert Path(reg.__file__).name == "registry.py"
    assert Path(reg.REGISTRY_DIR).is_dir()


# ------------------------------------------------------------------ the schema
def test_a_well_formed_entry_validates():
    assert reg.validate(MERGED, stem="demo") == []
    assert reg.validate(BASE, stem="qwen36-27b") == []


def test_the_b2_prefixes_are_DERIVED_and_a_typed_one_is_refused():
    """An artifact published under a name that does not carry its adapter's
    identity is one rename away from serving the wrong weights under the right
    label — so the prefix is recomputed, never read."""
    e = dict(MERGED, b2_model="checkpoints/demo-merged/deadbeefcafe/model")
    probs = reg.validate(e)
    assert len(probs) == 1 and probs[0].startswith("b2_model:")
    assert "DERIVED" in probs[0]

    assert reg.merged_prefixes("checkpoints/x/", "f" * 64) == {
        "b2_model": "checkpoints/x/ffffffffffff/model",
        "b2_guards": "checkpoints/x/ffffffffffff/guards"}


@pytest.mark.parametrize("mutate,expect", [
    (dict(kind="adapter"), "kind"),
    (dict(schema_version=2), "schema_version"),
    (dict(id="Demo"), "id"),
    (dict(b2_root="/checkpoints/demo"), "b2_root"),
    (dict(adapter_ident="abc"), "adapter_ident"),
    (dict(fingerprint_sha256="nothex"), "fingerprint_sha256"),
    (dict(fingerprint_sha256=None), "fingerprint_sha256"),
    (dict(content_sha256="short"), "content_sha256"),
    (dict(n_files=0), "n_files"),
    (dict(n_files=None), "n_files"),
    (dict(surprise=1), "unknown key"),
    (dict(serve="V10"), "serve"),
])
def test_a_malformed_entry_is_refused(mutate, expect):
    probs = reg.validate(dict(MERGED, **mutate))
    assert probs and any(expect in p for p in probs), probs


@pytest.mark.parametrize("mutate,expect", [
    ({"served_name": None}, "serve.served_name"),
    ({"served_name": ""}, "serve.served_name"),
    ({"dtype": 16}, "serve.dtype"),
    ({"max_len": 0}, "serve.max_len"),
    ({"tp": "1"}, "serve.tp"),
    ({"min_vram_gb": True}, "serve.min_vram_gb"),
    ({"lora_forbidden": "yes"}, "serve.lora_forbidden"),
])
def test_the_serve_block_is_type_checked(mutate, expect):
    serve = dict(MERGED["serve"])
    serve.update(mutate)
    probs = reg.validate(dict(MERGED, serve=serve))
    assert probs and any(expect.split(":")[0] in p for p in probs), probs


def test_a_missing_serve_key_is_reported_by_name():
    serve = {k: v for k, v in MERGED["serve"].items() if k != "min_vram_gb"}
    probs = reg.validate(dict(MERGED, serve=serve))
    assert probs == ["serve.min_vram_gb: missing"]


def test_the_filename_and_the_id_may_not_disagree():
    """Two names for one artifact, and a lookup by either finds a different
    thing."""
    probs = reg.validate(MERGED, stem="mergeddemoa")
    assert probs == ["id: 'demo' != its filename stem 'mergeddemoa'"]


def test_null_content_sha_means_UNMEASURED_and_still_validates():
    """The field is optional-by-null on purpose. What must never happen is a
    consumer reading a null as a pass; the schema records the absence rather
    than hiding it behind a default."""
    assert reg.validate(dict(MERGED, content_sha256=None)) == []
    assert reg.validate(dict(MERGED, content_sha256="ef" * 32)) == []


# ------------------------------------------------------------------ load / check
def test_load_and_lookup(tmp_path):
    d = _reg(tmp_path, BASE, MERGED)
    r = reg.load(d)
    assert sorted(r) == ["demo", "qwen36-27b"]
    assert reg.get("demo", d)["serve"]["served_name"] == "DEMO"
    assert reg.resolve_base(reg.get("demo", d), d)["id"] == "qwen36-27b"
    with pytest.raises(reg.RegistryError, match="no artifact"):
        reg.get("nope", d)


def test_strict_load_refuses_a_partially_valid_registry(tmp_path):
    """A partially-loaded registry is the shape that serves the wrong model: a
    consumer that gets a KeyError at least fails."""
    d = _reg(tmp_path, BASE, dict(MERGED, n_files=0))
    with pytest.raises(reg.RegistryError):
        reg.load(d)
    assert sorted(reg.load(d, strict=False)) == ["qwen36-27b"]


def test_check_catches_cross_entry_breakage(tmp_path):
    """Well-formed in isolation, broken in context — and context is where it is
    read. This is why `check` is not `validate` in a loop."""
    d = _reg(tmp_path, MERGED)                       # base not present
    assert any("base 'qwen36-27b' is not an entry" in p for p in reg.check(d))

    d = _reg(tmp_path, BASE, dict(MERGED, family="no-such-family"))
    assert any("family 'no-such-family' has no family_specs" in p
               for p in reg.check(d))

    d2 = tmp_path / "r2"
    d2.mkdir()
    (d2 / "stray.txt").write_text("x")
    assert any("not a .json file" in p for p in reg.check(str(d2)))

    d3 = tmp_path / "r3"
    d3.mkdir()
    (d3 / "broken.json").write_text("{oops")
    assert any("unparseable" in p for p in reg.check(str(d3)))


def test_a_merged_entry_may_not_point_its_base_at_another_merged_entry(tmp_path):
    d = _reg(tmp_path, BASE, MERGED,
             dict(MERGED, id="demo2", base="demo",
                  b2_root="checkpoints/demo2-merged",
                  b2_model="checkpoints/demo2-merged/ababababab" + "ab" + "/model",
                  b2_guards="checkpoints/demo2-merged/ababababab" + "ab" + "/guards"))
    assert any("has kind 'merged', not 'base'" in p for p in reg.check(d))


# ------------------------------------------------- env composition (--artifact)
def test_env_exports_names_the_payload_prefix_for_both_kinds():
    """`<PREFIX>_B2` is what a `${<PREFIX>_B2}` asset template consumes, so it
    must be the dir holding the payload: the derived `model` leaf for a merged
    artifact, the root for a base."""
    assert reg.env_exports(MERGED, "ADAPTER")["ADAPTER_B2"] == MERGED["b2_model"]
    assert reg.env_exports(BASE, "BASE")["BASE_B2"] == BASE["b2_root"]


def test_env_exports_stringifies_and_never_guesses_an_absent_field():
    m = reg.env_exports(MERGED, "M")
    assert m["M_N_FILES"] == "27" and m["M_MAX_LEN"] == "20480"
    assert m["M_LORA_FORBIDDEN"] == "1"
    assert m["M_CONTENT_SHA"] == ""            # null is UNMEASURED, not clean
    b = reg.env_exports(BASE, "B")
    # a base carries no serve block and no adapter identity: "" everywhere,
    # never a plausible default the box would then act on
    assert b["B_SERVED_NAME"] == "" and b["B_ADAPTER_IDENT"] == ""
    assert b["B_LORA_FORBIDDEN"] == ""
    assert set(b) == {f"B_{s}" for s in reg.ENV_SUFFIXES}
    assert all(isinstance(v, str) for v in {**m, **b}.values())


@pytest.mark.parametrize("prefix", ["", "lower", "1ST", "A-B", "A B"])
def test_env_exports_refuses_a_prefix_that_is_not_a_shell_identifier(prefix):
    with pytest.raises(reg.RegistryError):
        reg.env_exports(MERGED, prefix)


def test_env_exports_and_the_serve_lane_agree_on_the_shared_facts():
    """One definition of "which prefix holds the weights". `serve_artifact`
    delegates to `registry.model_prefix`; this pins that the two lanes cannot
    drift apart on the facts they both publish."""
    import serve_artifact as sa
    ar = dict(ln.split("=", 1) for ln in sa.resolve_lines(MERGED))
    env = reg.env_exports(MERGED, "M")
    for ar_key, env_key in (("AR_MODEL_B2", "M_B2"), ("AR_ID", "M_ID"),
                            ("AR_SERVED_NAME", "M_SERVED_NAME"),
                            ("AR_MAX_LEN", "M_MAX_LEN"), ("AR_TP", "M_TP"),
                            ("AR_MIN_VRAM_GB", "M_MIN_VRAM_GB"),
                            ("AR_FINGERPRINT", "M_FINGERPRINT")):
        assert ar[ar_key].strip("'") == env[env_key], ar_key


# ------------------------------------------------------------------- the seeds
def test_the_committed_registry_is_valid():
    assert reg.check() == []


def test_the_mergeddemoa_seed_says_what_the_serve_lane_needs():
    e = reg.get("mergeddemoa")
    assert e["b2_model"] == "checkpoints/mergeddemoa-merged/fdfa492a959d/model"
    assert e["b2_guards"] == "checkpoints/mergeddemoa-merged/fdfa492a959d/guards"
    assert e["n_files"] == 27
    assert e["serve"] == {"served_name": "MERGEDDEMOA", "dtype": "bfloat16",
                          "max_len": 20480, "min_vram_gb": 96, "tp": 1,
                          "lora_forbidden": True}
    # lora_forbidden is not decoration: this artifact is a MERGED dir, and
    # mounting an adapter over it applies the adapter a second time
    # and serves the base under the adapter's label.
    assert e["serve"]["lora_forbidden"] is True


def test_every_seed_entry_names_a_family_that_actually_exists():
    families = set(merge_guard.known_families())
    for ident, e in reg.load().items():
        assert e["family"] in families, ident


def test_the_qwen38_base_seed_describes_the_snapshot_it_names():
    """The hand-written half of a base entry, which no run mints: kind, the B2
    root, and the family whose spec drives its merges. Its hash fields are the
    measured half and are rostered in PINNED_BASE_SEEDS instead — `n_files` is
    not the B2 object count, which includes the `.complete` transport marker."""
    e = reg.get("qwen38-27b")
    assert e["kind"] == "base"
    assert e["b2_root"] == "base-models/qwen38-27b"
    assert e["family"] == "qwen38-27b"
    assert reg.model_prefix(e) == "base-models/qwen38-27b"


def test_every_base_seed_that_cites_pins_cites_a_file_that_EXISTS():
    """`pins` is not validated by the schema, so a stale path is a snapshot
    identity check the operator believes is armed and is not."""
    repo = HERE.parent.parent
    for ident, e in reg.load().items():
        pins = e.get("pins")
        if pins is None:
            continue
        p = repo / pins
        assert p.is_file(), f"{ident}: pins {pins} does not exist"
        assert not pins.startswith("/"), f"{ident}: pins must be repo-relative"
        body = json.loads(p.read_text())
        assert any(not k.startswith("_") for k in body), f"{ident}: empty pins"


def test_the_seed_b2_prefixes_carry_their_own_adapter_sha():
    e = reg.get("mergeddemoa")
    assert e["adapter_ident"].startswith("fdfa492a959d")
    assert reg.merged_prefixes(e["b2_root"], e["adapter_ident"]) == {
        "b2_model": e["b2_model"], "b2_guards": e["b2_guards"]}


# ------------------------------------------------------------------------- CLI
def _cli(*args):
    return subprocess.run([PY, str(CLI), *args], capture_output=True, text=True)


def test_cli_check_ls_show_path():
    assert _cli("check").returncode == 0
    assert "mergeddemoa" in _cli("ls").stdout
    assert json.loads(_cli("show", "mergeddemoa").stdout)["id"] == "mergeddemoa"
    r = _cli("path", "mergeddemoa")
    assert r.returncode == 0
    assert r.stdout.strip() == "checkpoints/mergeddemoa-merged/fdfa492a959d/model"
    assert _cli("path", "mergeddemoa", "--leaf", "guards").stdout.strip().endswith(
        "/guards")


def test_cli_check_exits_1_on_an_invalid_registry(tmp_path):
    d = _reg(tmp_path, dict(MERGED, n_files=0))
    r = _cli("--dir", d, "check")
    assert r.returncode == 1 and "REGISTRY INVALID" in r.stderr


# ------------------------------------------------------ pin-base (the promotion)
#: What the publisher measures about the BASE it merged against. The end-to-end
#: leg — mint real guards, compose, promote — is in the bundle's own test file,
#: beside the fixtures that make those numbers real.
BASE_SHA = "1a" * 32
PINS = {"config.json": {"size": 3, "sha256": "aa" * 32},
        "m.safetensors": {"size": 9, "sha256": "bb" * 32}}


def _cand(**over):
    prov = {"merge_job_id": "j1", "base_rollup_sha256": BASE_SHA,
            "base_n_files": 5}
    prov.update(over.pop("provenance", {}))
    return {"candidate_schema_version": 1, "entry": dict(MERGED, **over),
            "provenance": prov}


def test_base_pin_reads_all_three_published_shapes():
    """A run may leave a foldable candidate, only its guards summary, or only
    the pins manifest. All three name the same bytes."""
    m, probs = reg.base_pin(_cand())
    assert probs == [] and m["content_sha256"] == BASE_SHA
    assert m["n_files"] == 5 and m["base"] == "qwen36-27b"

    m, probs = reg.base_pin({"base_rollup_sha256": BASE_SHA, "base_n_files": 5})
    assert probs == [] and m["n_files"] == 5 and m["base"] is None

    m, probs = reg.base_pin(PINS)
    assert probs == [] and m["n_files"] == 2 and m["base"] is None
    assert reg.HEX64_RE.match(m["content_sha256"])


def test_base_pin_RECOMPUTES_a_pins_manifest_rather_than_reading_a_number():
    """The manifest states no rollup, so promoting it must measure the file —
    and must agree with what the publisher computed from the same manifest."""
    from modelkit import dirhash
    m, _ = reg.base_pin(PINS)
    assert m["content_sha256"] == dirhash.rollup(PINS)


@pytest.mark.parametrize("source,expect", [
    ({"candidate_schema_version": 1, "entry": MERGED,
      "provenance": {"merge_job_id": "j"}}, "base_rollup_sha256: missing"),
    ({"nothing": "here"}, "unrecognised shape"),
    ([], "not a JSON object"),
    # the merged dir's own guards summary: right file, wrong dir
    ({"merged_dir": "/x", "content_sha256": "ab" * 32, "n_files": 27},
     "MERGED dir's hashes, not the base's"),
])
def test_base_pin_refuses_a_source_with_nothing_to_promote(source, expect):
    m, probs = reg.base_pin(source)
    assert m == {} and any(expect in p for p in probs), probs


def test_pin_base_fills_the_hole_and_leaves_the_registry_valid(tmp_path):
    d = _reg(tmp_path, BASE, MERGED)
    path, probs, note = reg.pin_base(_cand(), None, d)
    assert probs == [] and "1a1a1a1a1a1a" in note
    assert reg.check(d) == []
    e = reg.get("qwen36-27b", d)
    assert e["content_sha256"] == BASE_SHA and e["n_files"] == 5


def test_pin_base_is_idempotent_and_reports_the_no_op(tmp_path):
    """The publisher measures the base on EVERY merge, so folding a second
    artifact off one snapshot must not have to remember the first."""
    d = _reg(tmp_path, BASE, MERGED)
    reg.pin_base(_cand(), None, d)
    before = Path(d, "qwen36-27b.json").read_text()
    path, probs, note = reg.pin_base(_cand(), None, d)
    assert probs == [] and note == ""            # empty note IS the no-op
    assert Path(d, "qwen36-27b.json").read_text() == before


def test_pin_base_REFUSES_a_rollup_that_disagrees_with_the_pin(tmp_path):
    """The guard this feature exists for. A base snapshot is immutable, so a
    changed rollup is not a stale pin — it says the prefix runbooks point at
    now holds different bytes, or the merge used a different snapshot."""
    d = _reg(tmp_path, BASE, MERGED)
    reg.pin_base(_cand(), None, d)
    before = Path(d, "qwen36-27b.json").read_text()

    moved = _cand(provenance={"base_rollup_sha256": "ff" * 32})
    _, probs, _ = reg.pin_base(moved, None, d)
    assert any("DISAGREES" in p and "IMMUTABLE" in p for p in probs), probs
    assert any(p.startswith("content_sha256: registry") for p in probs)
    assert Path(d, "qwen36-27b.json").read_text() == before

    # and it cannot be forced by accident — only by naming the flag
    _, probs, note = reg.pin_base(moved, None, d, force=True)
    assert probs == [] and note
    assert reg.get("qwen36-27b", d)["content_sha256"] == "ff" * 32


def test_pin_base_REFUSES_a_file_count_that_disagrees(tmp_path):
    """Same snapshot, different listing: the rollup covers the file SET, so a
    count that moved without it means the two halves describe different dirs."""
    d = _reg(tmp_path, BASE, MERGED)
    reg.pin_base(_cand(), None, d)
    _, probs, _ = reg.pin_base(_cand(provenance={"base_n_files": 6}), None, d)
    assert any("n_files: registry 5 != measured 6" in p for p in probs), probs


def test_pin_base_leaves_n_files_UNMEASURED_when_the_source_has_no_count(tmp_path):
    """A candidate published before the count was threaded into provenance
    carries only the rollup. Deriving the count from anything else would pin
    two halves that describe different listings."""
    d = _reg(tmp_path, BASE, MERGED)
    cand = _cand()
    del cand["provenance"]["base_n_files"]
    _, probs, note = reg.pin_base(cand, None, d)
    assert probs == [] and "UNMEASURED file count" in note
    e = reg.get("qwen36-27b", d)
    assert e["content_sha256"] == BASE_SHA and e.get("n_files") is None


def test_the_pins_manifest_completes_a_candidate_that_had_no_count(tmp_path):
    """The shape a run published before `base_n_files` was threaded into
    provenance: the candidate pins the rollup, and `guards/base_pins.json`
    beside the weights fills the count without re-measuring anything."""
    d = _reg(tmp_path, BASE, MERGED)
    cand = _cand()
    del cand["provenance"]["base_n_files"]
    reg.pin_base(cand, None, d)
    assert reg.get("qwen36-27b", d).get("n_files") is None

    # the same snapshot's manifest: the rollup AGREES, so only the count moves
    pins = dict(PINS)
    _, probs, note = reg.pin_base(
        {"base_rollup_sha256": BASE_SHA, "base_n_files": len(pins)},
        "qwen36-27b", d)
    assert probs == [] and note
    e = reg.get("qwen36-27b", d)
    assert e["content_sha256"] == BASE_SHA and e["n_files"] == len(pins)


def test_pin_base_needs_a_base_id_for_a_source_that_carries_no_name(tmp_path):
    d = _reg(tmp_path, BASE, MERGED)
    _, probs, _ = reg.pin_base(PINS, None, d)
    assert any("--base-id" in p for p in probs), probs
    _, probs, note = reg.pin_base(PINS, "qwen36-27b", d)
    assert probs == [] and note


def test_pin_base_refuses_a_base_id_the_candidate_disowns(tmp_path):
    """Pinning one artifact's rollup onto another gives a base entry an
    identity nothing was ever published against."""
    d = _reg(tmp_path, BASE, MERGED, dict(BASE, id="other-base"))
    _, probs, _ = reg.pin_base(_cand(), "other-base", d)
    assert any("!= the candidate's own base" in p for p in probs), probs


@pytest.mark.parametrize("target,expect", [
    ("demo", "not 'base'"),                      # a merged entry
    ("nope", "does not mint one"),               # absent
])
def test_pin_base_only_edits_an_existing_base_entry(tmp_path, target, expect):
    d = _reg(tmp_path, BASE, MERGED)
    _, probs, _ = reg.pin_base(PINS, target, d)
    assert any(expect in p for p in probs), probs


def test_pin_base_refuses_a_registry_directory_that_is_not_there(tmp_path):
    _, probs, _ = reg.pin_base(PINS, "qwen36-27b", str(tmp_path / "nope"))
    assert any("no registry directory" in p for p in probs)


def test_pin_base_REVERTS_rather_than_leaving_a_registry_that_will_not_load(
        tmp_path):
    d = _reg(tmp_path, BASE, MERGED,
             {"schema_version": 1, "id": "junk", "kind": "base",
              "family": "no-such-family", "b2_root": "x/"})
    before = Path(d, "qwen36-27b.json").read_text()
    _, probs, _ = reg.pin_base(_cand(), None, d)
    assert any(p.startswith("REVERTED") for p in probs)
    assert Path(d, "qwen36-27b.json").read_text() == before


def test_pin_base_preserves_the_files_own_shape(tmp_path):
    """A two-field pin must read as a two-field diff. Re-emitting a
    hand-written entry sorted and re-indented buries the change it made."""
    d = tmp_path / "shape"
    d.mkdir()
    raw = json.dumps(dict(BASE, description="d", content_sha256=None,
                          n_files=None), indent=2) + "\n"
    (d / "qwen36-27b.json").write_text(raw)
    (d / "demo.json").write_text(json.dumps(MERGED))
    reg.pin_base(_cand(), None, str(d))
    after = (d / "qwen36-27b.json").read_text()
    assert '\n  "id"' in after                   # indent 2 kept, not fold's 1
    assert after.index('"kind"') < after.index('"b2_root"')   # order kept
    assert [ln for ln in after.splitlines()
            if ln not in raw.splitlines()] == [
        f'  "content_sha256": "{BASE_SHA}",', '  "n_files": 5']


def test_a_candidate_whose_provenance_disowns_its_own_base_is_REFUSED():
    """`base_artifact_id` restates `entry.base`; the same box wrote both, and
    `pin-base` resolves through the entry — so a disagreement must not simply
    lose to it."""
    cand = _cand(provenance={"base_artifact_id": "some-other-base"})
    assert any("provenance.base_artifact_id" in p and "entry.base" in p
               for p in reg.validate_candidate(cand))
    assert reg.validate_candidate(
        _cand(provenance={"base_artifact_id": "qwen36-27b"})) == []


# ------------------------------------------------------------- pin-base, the CLI
def test_cli_pin_base_writes_dry_runs_and_reports_the_no_op(tmp_path):
    d = _reg(tmp_path, BASE, MERGED)
    src = tmp_path / "cand.json"
    src.write_text(json.dumps(_cand()))

    r = _cli("--dir", d, "pin-base", str(src), "--dry-run")
    assert r.returncode == 0 and "would pin" in r.stdout
    assert reg.get("qwen36-27b", d).get("content_sha256") is None

    r = _cli("--dir", d, "pin-base", str(src))
    assert r.returncode == 0 and "COMMIT IT" in r.stdout
    assert reg.get("qwen36-27b", d)["content_sha256"] == BASE_SHA

    r = _cli("--dir", d, "pin-base", str(src))
    assert r.returncode == 0 and "nothing to do" in r.stdout


def test_cli_pin_base_exits_1_on_the_disagreement(tmp_path):
    """A guard nobody has watched fail is not a guard — including at the exit
    code, which is what a runbook or a wrapper script actually reads."""
    d = _reg(tmp_path, BASE, MERGED)
    src = tmp_path / "cand.json"
    src.write_text(json.dumps(_cand()))
    _cli("--dir", d, "pin-base", str(src))
    src.write_text(json.dumps(_cand(provenance={"base_rollup_sha256": "ff" * 32})))
    r = _cli("--dir", d, "pin-base", str(src))
    assert r.returncode == 1
    assert "DISAGREES" in r.stderr and "nothing was written" in r.stderr


def test_cli_pin_base_reports_an_unreadable_source(tmp_path):
    d = _reg(tmp_path, BASE, MERGED)
    r = _cli("--dir", d, "pin-base", str(tmp_path / "absent.json"))
    assert r.returncode == 1 and "unreadable/unparseable" in r.stderr


#: Base seeds a merge run has MEASURED, and the pin it produced. An entry here
#: is a serve gate somebody arms; reverting one to null silently un-gates it.
PINNED_BASE_SEEDS = {
    "qwen36-27b": "6ea978e4a7a17608c842b8d2db37eb36ff2ccaa1217500d48724601b6c2bfda1",
    "qwen38-27b": "c289648aaf2e10415d2d7f44392c9a1774d30bdcc7cfbb00eef8bb7834a53227",
    # Measured at acquisition 2026-08-26, and measured at a STRONGER grade than
    # its bf16 sibling: the whole 28.77 GiB was materialised on the workstation
    # before upload, so all 78 files including every one of the 66 shards carry
    # a local sha256 rather than size + B2's stored md5.
    "qwen38-27b-fp8": "62fc8a2d79cbcff10206392b870401efec9a459bcc2b5d61620c15c7e72c2a2b",
}


def test_the_committed_base_seeds_say_measured_or_unmeasured_and_mean_it():
    """Two directions, because both are silent failures. A seed that gained a
    pin needs MERGED_MODEL_ARTIFACTS.md §7 revisited (its refusal claim goes
    stale); a seed that LOST one un-gates every serve that trusted it."""
    for ident, e in reg.load().items():
        if e["kind"] != "base":
            continue
        want = PINNED_BASE_SEEDS.get(ident)
        got = e.get("content_sha256")
        if want is None:
            assert got is None, (
                f"{ident} is newly pinned — update MERGED_MODEL_ARTIFACTS.md §7 "
                f"and add it to PINNED_BASE_SEEDS")
        else:
            assert got == want, (
                f"{ident} was measured at {want} and now reads {got!r} — a base "
                f"snapshot is immutable, so this is a regression, not a refresh")


def test_cli_path_refuses_a_base_entry(tmp_path):
    """A base has no `<sha12>/model` prefix — it is a snapshot root. Printing
    one anyway would hand a caller a path that does not exist."""
    d = _reg(tmp_path, BASE)
    r = _cli("--dir", d, "path", "qwen36-27b")
    assert r.returncode == 1 and "only a merged artifact" in r.stderr
