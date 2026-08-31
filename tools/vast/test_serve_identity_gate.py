"""The ON-BOX half: the gate itself, the READY marker's grammar, serve_ready.

Split from `test_serve_model_artifact.py` by WHERE the code runs, because the
threat models are different. The launcher's gates protect money and can be
argued with; this one protects the number an eval produces and cannot.

The gate is driven directly against fixture directories — it is a standalone
staged script by design, so testing it that way is testing what ships. The
marker grammar is pinned from both ends: the writer (`onstart/serve_vllm.sh`)
and the reader (`serve_ready.sh`), because a field appended by one and parsed
positionally by the other is exactly where an off-by-one column lives.
"""
import json
import os
import shutil
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(_HERE, "serve_identity_gate.py")
FINGERPRINT = os.path.join(_HERE, "modelkit", "merged_fingerprint.py")
DIRHASH = os.path.join(_HERE, "modelkit", "dirhash.py")
SERVE_SH = os.path.join(_HERE, "onstart", "serve_vllm.sh")
READY_SH = os.path.join(_HERE, "serve_ready.sh")

bash_only = pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _model_dir(tmp_path, name="model", *, files=None, receipt=True):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    for fn, body in (files or {"a.safetensors": "aaaa",
                               "b.safetensors": "bbbbbb",
                               "config.json": '{"x":1}'}).items():
        (d / fn).write_text(body)
    if receipt:
        n = len(files or {}) or 3
        (d / "PUSHED.json").write_text(
            json.dumps({"complete": True, "files": n, "ts_utc": "x"}))
    return d


def _fingerprint(d):
    out = subprocess.run(["python3", FINGERPRINT, "--dir", str(d), "--emit"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)["fingerprint"]


def _expect(tmp_path, d, *, grade="A", content=None, artifact="mergeddemoa",
            n_files=None, sha=None, name="expect.json", schema=1):
    fp = _fingerprint(d)
    doc = {"schema_version": schema, "artifact": artifact, "kind": "merged",
           "b2_model": "checkpoints/x/model", "served_name": "MERGEDDEMOA",
           "grade": grade,
           "fingerprint_sha256": sha if sha is not None else fp["sha256"],
           "n_files": n_files if n_files is not None else fp["n_files"],
           "content_sha256": content, "composed_utc": "2026-08-24T00:00:00Z",
           "source": "tools/vast/modelkit/registry/mergeddemoa.json"}
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return p


def _run_gate(d, expect, *extra):
    return subprocess.run(
        ["python3", GATE, "--dir", str(d), "--expect", str(expect),
         "--fingerprint-tool", FINGERPRINT, *extra],
        capture_output=True, text=True, timeout=120)


# --------------------------------------------------------------------------- #
# 1. PASS
# --------------------------------------------------------------------------- #

def test_a_matching_dir_verifies_and_prints_a_parseable_verdict(tmp_path):
    d = _model_dir(tmp_path)
    p = _run_gate(d, _expect(tmp_path, d))
    assert p.returncode == 0, p.stdout + p.stderr
    tok = p.stdout.strip().splitlines()[-1].split()
    assert tok[0] == "IDENTITY_VERIFIED"
    assert tok[1] == "A"
    assert len(tok[2]) == 12 and tok[3].startswith(tok[2]) and len(tok[3]) == 64


def test_the_receipt_is_excluded_from_the_fingerprint(tmp_path):
    """`PUSHED.json` is written AFTER the content it attests to, so it can never
    be inside the fingerprint of that content. A pull brings it along, and a
    gate that counted it would refuse every honestly restored dir."""
    d = _model_dir(tmp_path)
    exp = _expect(tmp_path, d)
    assert (d / "PUSHED.json").exists()
    assert json.loads(exp.read_text())["n_files"] == 3
    assert _run_gate(d, exp).returncode == 0


def test_grade_a_says_out_loud_what_it_did_not_check(tmp_path):
    """`content_sha256: null` is UNMEASURED, never clean. An operator reading a
    serve log must be able to tell 'checked the bytes' from 'checked the
    shape'."""
    d = _model_dir(tmp_path)
    p = _run_gate(d, _expect(tmp_path, d))
    assert "GRADE A ONLY" in p.stderr
    assert "same-size content swap is INVISIBLE" in p.stderr
    assert "gate_dir.py" in p.stderr


# --------------------------------------------------------------------------- #
# 2. FAIL — each of the shapes that scores like a baseline
# --------------------------------------------------------------------------- #

def test_a_short_restore_is_refused(tmp_path):
    """20-of-27 shards loads, serves, answers, and scores as the base."""
    d = _model_dir(tmp_path)
    exp = _expect(tmp_path, d)
    (d / "b.safetensors").unlink()
    p = _run_gate(d, exp)
    assert p.returncode == 1, p.stdout + p.stderr
    assert "IDENTITY_MISMATCH" in p.stdout
    assert "short or fat restore" in p.stderr


def test_a_same_shape_dir_with_different_sizes_is_refused(tmp_path):
    d = _model_dir(tmp_path)
    exp = _expect(tmp_path, d)
    (d / "b.safetensors").write_text("bbbbbbbbbbbbbbbb")
    p = _run_gate(d, exp)
    assert p.returncode == 1
    assert "fingerprint_sha256" in p.stderr


def test_a_wholly_different_artifact_is_refused(tmp_path):
    """The 2026-08-21 shape: the box pulled some other model entirely."""
    right = _model_dir(tmp_path, "right")
    wrong = _model_dir(tmp_path, "wrong",
                       files={"z.safetensors": "zzzz", "config.json": "{}"})
    p = _run_gate(wrong, _expect(tmp_path, right))
    assert p.returncode == 1
    assert "IDENTITY_MISMATCH" in p.stdout


def test_a_receipt_contradicting_the_dir_is_refused(tmp_path):
    """Independent of the fingerprint: the publisher's own count against what
    is on disk. Corroboration, never authority — it can only contradict."""
    d = _model_dir(tmp_path)
    exp = _expect(tmp_path, d)
    (d / "PUSHED.json").write_text(json.dumps({"complete": True, "files": 99}))
    p = _run_gate(d, exp)
    assert p.returncode == 1
    assert "publisher pushed 99" in p.stderr


# --------------------------------------------------------------------------- #
# 3. CANNOT CHECK — a distinct verdict, because the remedy is opposite
# --------------------------------------------------------------------------- #

def test_a_missing_fingerprint_tool_cannot_check_and_says_so(tmp_path):
    """A gate that cannot run is discovered at the moment it was supposed to
    refuse something. Reporting that as a MISMATCH would send an operator
    hunting a model defect that does not exist."""
    d = _model_dir(tmp_path)
    exp = _expect(tmp_path, d)
    # copy the gate somewhere with no modelkit/ sibling so the ladder misses
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    shutil.copy(GATE, lonely / "serve_identity_gate.py")
    p = subprocess.run(["python3", str(lonely / "serve_identity_gate.py"),
                        "--dir", str(d), "--expect", str(exp),
                        "--fingerprint-tool", str(tmp_path / "nope.py")],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 2, p.stdout + p.stderr
    assert "IDENTITY_CANNOT_CHECK" in p.stdout
    assert "REFUSAL and not a skip" in p.stderr


def test_an_unparseable_expectation_cannot_check(tmp_path):
    """The shape a truncated pull of the expectation itself takes. It must not
    read as a pass."""
    d = _model_dir(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    p = _run_gate(d, bad)
    assert p.returncode == 2
    assert "IDENTITY_CANNOT_CHECK" in p.stdout


@pytest.mark.parametrize("mutate,why", [
    ({"schema": 7}, "schema_version"),
    ({"grade": "Z"}, "grade"),
    ({"sha": 42}, "fingerprint_sha256"),
    ({"n_files": "three"}, "n_files"),
])
def test_a_malformed_expectation_cannot_check(mutate, why, tmp_path):
    d = _model_dir(tmp_path)
    p = _run_gate(d, _expect(tmp_path, d, **mutate))
    assert p.returncode == 2, p.stdout + p.stderr
    assert why in p.stderr


def test_a_missing_dir_cannot_check(tmp_path):
    d = _model_dir(tmp_path)
    p = _run_gate(tmp_path / "gone", _expect(tmp_path, d))
    assert p.returncode == 2
    assert "not a directory" in p.stderr


# --------------------------------------------------------------------------- #
# 4. GRADE B — the stronger claim, and the refusal when it cannot be made
# --------------------------------------------------------------------------- #

def test_grade_b_verifies_the_bytes(tmp_path):
    d = _model_dir(tmp_path)
    out = subprocess.run(["python3", DIRHASH, str(d), "--rollup"],
                         capture_output=True, text=True, timeout=60)
    rollup = out.stdout.strip()
    p = _run_gate(d, _expect(tmp_path, d, grade="B", content=rollup),
                  "--dirhash-tool", DIRHASH)
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.strip().splitlines()[-1].split()[1] == "B"
    assert "GRADE A ONLY" not in p.stderr


def test_grade_b_catches_the_swap_grade_a_is_blind_to(tmp_path):
    """A same-SIZE content swap. Grade A passes it by construction; this is the
    whole reason both grades exist."""
    d = _model_dir(tmp_path)
    out = subprocess.run(["python3", DIRHASH, str(d), "--rollup"],
                         capture_output=True, text=True, timeout=60)
    exp_b = _expect(tmp_path, d, grade="B", content=out.stdout.strip(),
                    name="expect_b.json")
    exp_a = _expect(tmp_path, d, name="expect_a.json")
    (d / "b.safetensors").write_text("cccccc")       # same 6 bytes, other bytes
    # The control, and it is the point: grade A PASSES this dir. Without it,
    # "grade B refused" says nothing about whether grade B saw anything.
    assert _run_gate(d, exp_a).returncode == 0, "control void: grade A must PASS"
    p = _run_gate(d, exp_b, "--dirhash-tool", DIRHASH)
    assert p.returncode == 1
    assert "different BYTES" in p.stderr


def test_grade_b_pinned_with_no_tool_is_a_refusal(tmp_path):
    """The stronger claim was requested and cannot be made. Falling back to
    grade A here would silently answer a question nobody asked."""
    d = _model_dir(tmp_path)
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    shutil.copy(GATE, lonely / "serve_identity_gate.py")
    exp = _expect(tmp_path, d, grade="B", content="0" * 64)
    p = subprocess.run(["python3", str(lonely / "serve_identity_gate.py"),
                        "--dir", str(d), "--expect", str(exp),
                        "--fingerprint-tool", FINGERPRINT,
                        "--dirhash-tool", str(tmp_path / "nope.py")],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 2, p.stdout + p.stderr
    assert "dirhash.py is not on this box" in p.stderr


# --------------------------------------------------------------------------- #
# 5. the staged gate runs ALONE — the property merged_fingerprint's
#    self-contained-by-contract rule exists to give the serve lane
# --------------------------------------------------------------------------- #

def test_the_gate_and_its_tool_run_copied_into_an_empty_dir(tmp_path):
    """Exactly what a box gets: three files in /workspace and no repo, no
    package, no `modelkit/` sibling. If loading by path ever regressed to an
    intra-package import, this is where it shows."""
    box = tmp_path / "workspace"
    box.mkdir()
    for src in (GATE, FINGERPRINT):
        shutil.copy(src, box / os.path.basename(src))
    d = _model_dir(tmp_path)
    exp = _expect(tmp_path, d)
    shutil.copy(exp, box / "identity_expect.json")
    p = subprocess.run(["python3", "serve_identity_gate.py",
                        "--dir", str(d), "--expect", "identity_expect.json"],
                       capture_output=True, text=True, timeout=120, cwd=box)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "IDENTITY_VERIFIED" in p.stdout


def test_the_report_records_which_tool_actually_ran(tmp_path):
    d = _model_dir(tmp_path)
    rep = tmp_path / "report.json"
    _run_gate(d, _expect(tmp_path, d), "--out", str(rep))
    doc = json.loads(rep.read_text())
    assert doc["ok"] is True
    assert doc["fingerprint_tool"] == FINGERPRINT
    assert doc["ident_grade"] == "A"
    assert len(doc["ident_sha256"]) == 64


# --------------------------------------------------------------------------- #
# 6. the READY marker's grammar, pinned from BOTH ends
# --------------------------------------------------------------------------- #

def _marker_fields(line, fn):
    """Run one of serve_ready.sh's own parse helpers against a marker line."""
    src = open(READY_SH, encoding="utf-8").read()
    start = src.index("marker_models() {")
    end = src.index("# --- poll SERVE_STATUS")
    prog = "#!/usr/bin/env bash\nset -euo pipefail\n" + src[start:end] \
        + '\n%s "$1"\n' % fn
    p = subprocess.run(["bash", "-c", prog, "_", line],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


@bash_only
@pytest.mark.parametrize("line,models,ident", [
    # legacy: every marker written before the gate existed
    ("READY 2026-08-24T00:00:00Z MERGEDDEMOA", "MERGEDDEMOA", ""),
    ("READY 2026-08-24T00:00:00Z a,b,c", "a,b,c", ""),
    ("READY 2026-08-24T00:00:00Z", "", ""),
    # gated
    ("READY 2026-08-24T00:00:00Z MERGEDDEMOA ident=ad65f40a677e", "MERGEDDEMOA",
     "ad65f40a677e"),
    ("READY 2026-08-24T00:00:00Z a,b ident=0123456789ab", "a,b",
     "0123456789ab"),
    # gated with no parseable id list: `-` is a placeholder, not a model
    ("READY 2026-08-24T00:00:00Z - ident=0123456789ab", "", "0123456789ab"),
])
def test_the_reader_parses_every_marker_shape(line, models, ident):
    assert _marker_fields(line, "marker_models") == models
    assert _marker_fields(line, "marker_ident") == ident


def test_the_writer_appends_and_never_reorders():
    """`poll_marker` has read the id CSV from field 3 since this marker existed,
    so a new field could only ever go AFTER it."""
    src = open(SERVE_SH, encoding="utf-8").read()
    assert '_mk="$_ids"' in src
    assert '_mk="${_ids:--} ident=$SERVE_IDENT_SHA12"' in src
    assert 'status READY "$_mk"' in src


def test_an_ungated_serve_writes_the_byte_identical_legacy_marker():
    """The compatibility floor. With no identity verified, `_mk` IS `_ids` —
    not `_ids` plus an empty suffix, which would leave a trailing space and a
    4th field of whitespace for someone to trip over."""
    src = open(SERVE_SH, encoding="utf-8").read()
    i = src.index('_mk="$_ids"')
    j = src.index('status READY "$_mk"')
    between = src[i:j]
    assert between.count("_mk=") == 2          # the base and the guarded append
    assert '[ -n "$SERVE_IDENT_SHA12" ] && _mk=' in between


# --------------------------------------------------------------------------- #
# 6b. the WIRING in serve_vllm.sh — armed, unarmed, and armed-but-unresolvable
#
# Driven as a FRAGMENT of the shipped payload rather than a paraphrase of it:
# the block between its own two banner comments is extracted verbatim and run
# with `status` stubbed. A `DRY_RUN=1` whole-script run cannot reach it (nothing
# was pulled, so there is nothing to fingerprint), and a real run ends in
# `exec vllm`.
# --------------------------------------------------------------------------- #
def _gate_block():
    src = open(SERVE_SH, encoding="utf-8").read()
    start = src.index("# --- ON-BOX IDENTITY GATE")
    end = src.index("# --- MTP speculative decoding")
    return src[start:end]


def _run_block(tmp_path, **env):
    """The shipped block, with `status` captured to a file instead of B2."""
    marks = tmp_path / "status.log"
    prog = ("#!/usr/bin/env bash\nset -euo pipefail\n"
            "DRY_RUN=${DRY_RUN:-0}\n"
            f'status() {{ echo "$@" >> {marks}; return 0; }}\n'
            + _gate_block()
            + '\necho "SHA12=$SERVE_IDENT_SHA12 GRADE=$SERVE_IDENT_GRADE"\n')
    e = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "TMPDIR")}
    e["HOME"] = str(tmp_path / "home")
    os.makedirs(e["HOME"], exist_ok=True)
    e.update({k: str(v) for k, v in env.items()})
    p = subprocess.run(["bash", "-c", prog, str(tmp_path / "serve_main.sh")],
                       capture_output=True, text=True, timeout=180, env=e)
    p.marks = marks.read_text() if marks.exists() else ""   # type: ignore[attr-defined]
    return p


@bash_only
def test_an_unarmed_serve_skips_with_one_loud_line(tmp_path):
    """Every pre-artifact caller. The skip must be LOUD — the whole reason this
    lane needed a gate is that nothing else can see the failure."""
    p = _run_block(tmp_path, MODEL_ID="/workspace/base-model")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "identity gate: SKIPPED" in p.stderr
    assert "proves the LABEL, never the WEIGHTS" in p.stderr
    assert "SHA12= GRADE=" in p.stdout
    assert p.marks == ""


@bash_only
def test_an_armed_serve_verifies_and_exports_the_identity(tmp_path):
    d = _model_dir(tmp_path)
    exp = _expect(tmp_path, d)
    fp = _fingerprint(d)
    p = _run_block(tmp_path, MODEL_ID=str(d), SERVE_IDENT_REQUIRED="1",
                   SERVE_IDENT_ARTIFACT="mergeddemoa", SERVE_IDENT_EXPECT=str(exp),
                   SERVE_IDENT_GATE=GATE, SERVE_IDENT_FINGERPRINT=FINGERPRINT)
    assert p.returncode == 0, p.stdout + p.stderr
    assert f"SHA12={fp['sha256'][:12]} GRADE=A" in p.stdout
    assert "PULLING identity_gate" in p.marks


@bash_only
def test_an_armed_serve_refuses_a_mismatched_dir(tmp_path):
    """`status FAILED identity_mismatch`, and no serve. The marker reason is
    what fleetd and an operator read; it must name the defect."""
    d = _model_dir(tmp_path)
    exp = _expect(tmp_path, d)
    (d / "b.safetensors").unlink()
    p = _run_block(tmp_path, MODEL_ID=str(d), SERVE_IDENT_REQUIRED="1",
                   SERVE_IDENT_ARTIFACT="mergeddemoa", SERVE_IDENT_EXPECT=str(exp),
                   SERVE_IDENT_GATE=GATE, SERVE_IDENT_FINGERPRINT=FINGERPRINT)
    assert p.returncode == 1, p.stdout + p.stderr
    assert "FAILED identity_mismatch" in p.marks
    assert "identity gate REFUSED — not serving" in p.stderr


@bash_only
def test_a_broken_gate_is_reported_apart_from_a_wrong_model(tmp_path):
    """`identity_cannot_check` vs `identity_mismatch`. Same refusal, opposite
    remedies: one means the WEIGHTS are wrong, the other that the GATE is."""
    d = _model_dir(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{truncated")
    p = _run_block(tmp_path, MODEL_ID=str(d), SERVE_IDENT_REQUIRED="1",
                   SERVE_IDENT_EXPECT=str(bad), SERVE_IDENT_GATE=GATE,
                   SERVE_IDENT_FINGERPRINT=FINGERPRINT)
    assert p.returncode == 1, p.stdout + p.stderr
    assert "FAILED identity_cannot_check" in p.marks


@bash_only
def test_an_armed_serve_that_cannot_find_its_expectation_FAILS(tmp_path):
    """The property `SERVE_IDENT_REQUIRED` exists for. A transient B2 read that
    degraded the gate to a skip would be a gate you cannot rely on having run —
    and the runs it skipped are exactly the ones it was needed for."""
    d = _model_dir(tmp_path)
    p = _run_block(tmp_path, MODEL_ID=str(d), SERVE_IDENT_REQUIRED="1",
                   SERVE_IDENT_ARTIFACT="mergeddemoa")
    assert p.returncode == 1, p.stdout + p.stderr
    assert "FAILED identity_expect_missing" in p.marks
    assert "REFUSAL, not a skip" in p.stderr


@bash_only
def test_a_leftover_expectation_on_an_unarmed_run_is_not_used(tmp_path):
    """The same stale-inheritance shape as the 2026-08-21 MODEL_B2 bug, one
    layer up: an expectation left by an EARLIER serve on this box would gate
    THIS model against the PREVIOUS one's fingerprint. Unarmed means unarmed."""
    d = _model_dir(tmp_path)
    exp = _expect(tmp_path, d)
    p = _run_block(tmp_path, MODEL_ID=str(d), SERVE_IDENT_EXPECT=str(exp))
    assert p.returncode == 0, p.stdout + p.stderr
    assert "NOT this" in p.stderr and "run's expectation" in p.stderr
    assert "SHA12= GRADE=" in p.stdout


def test_the_env_persistence_filter_covers_the_identity_variables():
    """`/etc/environment` is rewritten key-by-key, and an identity expectation
    that outlived its run and was inherited by a later attach would gate the
    NEXT model against the PREVIOUS one's pin. The `SERVE_` prefix in the
    persistence grep is what stops that — pinned because the filter is one
    regex nobody re-reads."""
    src = open(SERVE_SH, encoding="utf-8").read()
    grep = [ln for ln in src.splitlines()
            if ln.startswith("env | grep -E") and "_SERVE_ENV_SNAP" in ln]
    assert len(grep) == 1, grep
    assert "SERVE_" in grep[0]
    for var in ("SERVE_IDENT_REQUIRED", "SERVE_IDENT_EXPECT",
                "SERVE_IDENT_ARTIFACT", "SERVE_DTYPE"):
        assert var.startswith("SERVE_"), var


# --------------------------------------------------------------------------- #
# 7. serve_ready.sh --expect-ident
# --------------------------------------------------------------------------- #
def _ready(tmp_path, marker_line, *args):
    """Run serve_ready.sh against a fixture rclone serving one marker line."""
    bindir = tmp_path / "rbin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "rclone"
    stub.write_text("#!/usr/bin/env bash\n"
                    "case \"$1\" in cat) cat <<'M'\n" + marker_line + "\nM\n"
                    "  ;; *) exit 0;; esac\n")
    stub.chmod(0o755)
    e = {k: v for k, v in os.environ.items() if k in ("LANG", "TMPDIR")}
    e["PATH"] = str(bindir) + os.pathsep + os.environ.get("PATH", "")
    e["HOME"] = str(tmp_path / "home")
    os.makedirs(e["HOME"], exist_ok=True)
    e.update(B2_BUCKET="fake", B2_KEY_ID="fake")
    return subprocess.run(["bash", READY_SH, "sid", *args],
                          capture_output=True, text=True, timeout=120, env=e,
                          cwd=_HERE)


@bash_only
def test_expect_ident_passes_on_a_match(tmp_path):
    p = _ready(tmp_path, "READY 2026-08-24T00:00:00Z MERGEDDEMOA ident=ad65f40a677e",
               "--status-only")
    assert "ident=ad65f40a677e" in p.stdout, p.stdout


@bash_only
def test_expect_ident_fails_on_a_mismatch(tmp_path):
    """The box proved it serves SOMETHING coherently — just not the artifact
    about to be scored."""
    p = _ready(tmp_path, "READY 2026-08-24T00:00:00Z MERGEDDEMOA ident=ffffffffffff",
               "--expect-ident", "ad65f40a677e", "--timeout", "1", "--poll", "1")
    assert p.returncode == 9, p.stdout + p.stderr
    assert "IDENTITY MISMATCH" in p.stderr


@bash_only
def test_expect_ident_fails_on_ABSENT(tmp_path):
    """Absent is a failure on purpose: 'no claim' is not a passing claim. A box
    running a pre-gate serve payload, or launched without --model-artifact,
    never verified its own weights and must not read as verified."""
    p = _ready(tmp_path, "READY 2026-08-24T00:00:00Z MERGEDDEMOA",
               "--expect-ident", "ad65f40a677e", "--timeout", "1", "--poll", "1")
    assert p.returncode == 9, p.stdout + p.stderr
    assert "carries NO ident= field" in p.stderr
    assert "not a passing claim" in p.stderr


@bash_only
def test_without_expect_ident_a_gated_marker_is_business_as_usual(tmp_path):
    """The 4th field must not disturb a caller that does not ask about it —
    including the model list, which is still field 3."""
    p = _ready(tmp_path, "READY 2026-08-24T00:00:00Z MERGEDDEMOA ident=ad65f40a677e",
               "--status-only")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "models=MERGEDDEMOA" in p.stdout


@bash_only
def test_expect_ident_is_refused_in_verify_only_mode(tmp_path):
    """`--base-url` reads no marker, so there is nothing to check it against.
    Silently ignoring the flag is worse than not having it: the caller believes
    a gate ran."""
    e = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "TMPDIR")}
    e["HOME"] = str(tmp_path)
    p = subprocess.run(
        ["bash", READY_SH, "--base-url", "http://127.0.0.1:1/v1",
         "--expect-models", "X", "--expect-ident", "ad65f40a677e"],
        capture_output=True, text=True, timeout=60, env=e, cwd=_HERE)
    assert p.returncode == 2, p.stdout + p.stderr
    assert "--expect-ident needs the SERVE_STATUS marker" in p.stderr
