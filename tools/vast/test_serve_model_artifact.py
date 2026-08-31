"""`launch_serve.sh --model-artifact`: resolution, refusals, and what is staged.

The class of failure behind all of it: a serve box that pulls the wrong weights
is INVISIBLE to every check a caller runs. The process is up, `/v1/models` lists
the name that was asked for, `serve_ready.sh --expect-models` passes, and the
eval scores like the baseline. Measured live 2026-08-21 (a stale `MODEL_B2`
inherited through `/etc/environment`). Names cannot catch it; only a claim about
bytes can.

These gates run the REAL `launch_serve.sh` under `--dry-run` with a fixture
`rclone` on PATH, so the verdicts are the script's and not a restatement of it.
Every subprocess gets a scratch `HOME` and an empty `_LAUNCH_SERVE_ENV`: the
script sources `$REPO_ROOT/.env` itself, and a checkout that has one takes
different branches than one that does not — and a shipped writer handed the real
`$HOME` rewrites the operator's live rclone config (conftest's
`_protect_operator_rclone_conf` turns that into a red test).
"""
import json
import os
import re
import shutil
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCH_SH = os.path.join(_HERE, "launch_serve.sh")

pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")

_FAKE_B2 = {"B2_BUCKET": "fake", "B2_KEY_ID": "fake",
            "B2_APPLICATION_KEY": "fake",
            "B2_S3_ENDPOINT": "https://example.invalid"}

#: mergeddemoa's registry pins, restated so a seed edit that changes them is a red
#: test naming the number rather than a fixture that quietly stops matching.
MERGEDDEMOA_FILES = 27
MERGEDDEMOA_PREFIX = "checkpoints/mergeddemoa-merged/fdfa492a959d/model"

#: A MERGED entry pinned at grade A only, for the same reason UNPINNED_BASE
#: exists below: the grade-A-only case has no permanent live example either.
#: mergeddemoa was that example until 2026-08-25, when its bytes were measured and
#: three tests here went red for describing the registry's roster instead of
#: the launcher's behaviour. `fingerprint_sha256` + `n_files` and a NULL
#: `content_sha256` is the whole shape — the pins are a fixture's, not a real
#: artifact's, so nothing can be gated on them by accident.
GRADE_A_ONLY_MERGED = {
    "schema_version": 1,
    "id": "gradea-fixture",
    "kind": "merged",
    "family": "qwen36-27b",
    "base": "qwen36-27b",
    "adapter_ident": "ad" * 32,
    "b2_root": "checkpoints/gradea-fixture",
    "b2_model": "checkpoints/gradea-fixture/adadadadadad/model",
    "b2_guards": "checkpoints/gradea-fixture/adadadadadad/guards",
    "fingerprint_sha256": "fe" * 32,
    "n_files": 27,
    "content_sha256": None,
    "serve": {"served_name": "GRADEAFIX", "dtype": "bfloat16",
              "max_len": 20480, "min_vram_gb": 96, "tp": 1,
              "lora_forbidden": True},
}

#: A base entry with NULL pins. Committed seeds get measured as merges publish
#: them, so the unpinned case has no permanent live example — pinning this
#: refusal to whichever seed happened to be unmeasured is a test that expires.
#: `fingerprint_sha256` is not a key a `base` entry may carry, so grade B (the
#: `content_sha256` below) is the only pin that can move this fixture.
UNPINNED_BASE = {
    "schema_version": 1,
    "id": "unpinned-fixture",
    "kind": "base",
    "b2_root": "base-models/unpinned-fixture",
    "content_sha256": None,
    "n_files": None,
}
#: The real python3 `launch_serve.sh` would find, resolved before the shim
#: below shadows it.
_REAL_PY = shutil.which("python3")


def _fixture_registry(tmp_path, entry):
    """A one-entry registry directory, plus a `python3` on PATH that points
    `serve_artifact.py` at it.

    The committed registry is the TRUST ANCHOR for artifact identity — the
    expectation is composed from git precisely so that B2 cannot corroborate
    B2 — so the shipped script offers no flag or env var for reading a
    different one. This seam is a stub binary in `tmp_path`: it exists only
    inside one test's PATH and no operator invocation can reach it.
    """
    d = tmp_path / "fixture-registry"
    d.mkdir(exist_ok=True)
    (d / f"{entry['id']}.json").write_text(json.dumps(entry))
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    shim = bindir / "python3"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        "args=()\n"
        'for a in "$@"; do\n'
        '  args+=("$a")\n'
        f'  case "$a" in */serve_artifact.py) args+=(--dir {d!s});; esac\n'
        "done\n"
        f'exec {_REAL_PY} "${{args[@]}}"\n')
    shim.chmod(0o755)
    return str(bindir)


def _rclone_stub(tmp_path, *, names=None, receipt=None, size_bytes=55834574848,
                 lsf_rc=0):
    """A fixture `rclone` on PATH. NO network, and no real config is consulted.

    Only the four verbs the artifact gate and the disk sizer use are answered;
    anything else exits 0 silently, which is what `rcat` staging needs.
    """
    if names is None:
        names = [f"f{i}.safetensors" for i in range(1, MERGEDDEMOA_FILES)] \
            + ["config.json", "PUSHED.json"]
    if receipt is None:
        receipt = {"complete": True, "files": MERGEDDEMOA_FILES, "ts_utc": "x"}
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "rclone"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "case \"$1\" in\n"
        f"  lsf) [ {lsf_rc} -ne 0 ] && exit {lsf_rc}; "
        f"cat <<'LSF'\n" + "\n".join(names) + "\nLSF\n    ;;\n"
        "  cat) cat <<'REC'\n" + json.dumps(receipt) + "\nREC\n    ;;\n"
        f"  size) echo '{{\"count\": 28, \"bytes\": {size_bytes}}}';;\n"
        "  rcat) cat > /dev/null;;\n"
        "  *) exit 0;;\n"
        "esac\n")
    stub.chmod(0o755)
    return str(bindir)


def _scratch_home(tmp_path):
    h = tmp_path / "home"
    (h / ".config" / "rclone").mkdir(parents=True, exist_ok=True)
    return str(h)


def _launch(tmp_path, *args, stub=True, **env):
    e = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "TMPDIR")}
    if stub:
        e["PATH"] = _rclone_stub(tmp_path) + os.pathsep + e.get("PATH", "")
    e["HOME"] = _scratch_home(tmp_path)
    empty = tmp_path / "empty.env"
    empty.write_text("")
    e["_LAUNCH_SERVE_ENV"] = str(empty)
    e["RCLONE_CONFIG"] = str(tmp_path / "rclone.conf")
    e.update(_FAKE_B2)
    e.update(env)
    return subprocess.run(
        ["bash", LAUNCH_SH, "--dry-run",
         "--api-key-file", str(tmp_path / "key.txt"), *args],
        capture_output=True, text=True, timeout=180, env=e, cwd=_HERE)


def _envmap(out):
    """`--env K=V` pairs from the printed herdd argv."""
    return dict(m.groups() for m in
                re.finditer(r"--env\s+(\w+)=(\S*)", out))


# --------------------------------------------------------------------------- #
# 1. resolution — and that an explicit flag still wins
# --------------------------------------------------------------------------- #

def test_the_artifact_resolves_every_serve_field_from_the_registry(tmp_path):
    p = _launch(tmp_path, "--model-artifact", "mergeddemoa")
    assert p.returncode == 0, p.stdout + p.stderr
    env = _envmap(p.stdout)
    assert env["MODEL_B2"] == MERGEDDEMOA_PREFIX
    assert env["SERVED_NAME"] == "MERGEDDEMOA"
    assert env["MAX_LEN"] == "20480"
    assert env["SERVE_DTYPE"] == "bfloat16"
    assert "MODEL_ID" not in env


def test_min_vram_gb_becomes_the_host_floor(tmp_path):
    """A 27B bf16 merged model rented onto a 24 GB card dies at engine init —
    after the pull, on a box that is already billing."""
    p = _launch(tmp_path, "--model-artifact", "mergeddemoa")
    assert re.search(r"--gpu-ram\s+96", p.stdout), p.stdout


@pytest.mark.parametrize("flag,value,var", [
    ("--served-name", "FOO", "SERVED_NAME"),
    ("--max-len", "4096", "MAX_LEN"),
    ("--dtype", "float16", "SERVE_DTYPE"),
])
def test_an_explicit_flag_wins_over_the_registry(flag, value, var, tmp_path):
    """The bakeoff-row precedence rule, unchanged. The registry is a default
    source, not an override."""
    p = _launch(tmp_path, "--model-artifact", "mergeddemoa", flag, value)
    assert p.returncode == 0, p.stdout + p.stderr
    assert _envmap(p.stdout)[var] == value


def test_an_explicit_gpu_ram_wins_over_min_vram_gb(tmp_path):
    p = _launch(tmp_path, "--model-artifact", "mergeddemoa", "--gpu-ram", "48")
    assert re.search(r"--gpu-ram\s+48", p.stdout), p.stdout
    assert not re.search(r"--gpu-ram\s+96", p.stdout), p.stdout


def test_an_explicit_model_is_refused_not_merged(tmp_path):
    """The ONE flag that cannot win. An identity expectation naming artifact X
    shipped beside a pull of model Y can only fail — after the box is paid
    for — or, if the gate is ever skipped, serve Y under X's name."""
    p = _launch(tmp_path, "--model-artifact", "mergeddemoa", "--model", "b2:other/x")
    assert p.returncode == 2, p.stdout + p.stderr
    assert "with an explicit --model: refusing" in p.stderr


def test_bakeoff_and_artifact_are_two_resolvers_for_one_serve(tmp_path):
    """A RESOLVABLE bakeoff slug, so the refusal under test is the mutual
    exclusion and not the bakeoff resolver's own not-found (which fires
    first) — hence the throwaway manifest rather than a bare slug."""
    mj = tmp_path / "models.json"
    mj.write_text(json.dumps({"models": [
        {"slug": "toy-8b", "b2_subpath": "base-models/toy-8b",
         "served_name": "toy-8b", "max_model_len": 16384, "quant": "bf16"}]}))
    p = _launch(tmp_path, "--model-artifact", "mergeddemoa",
                "--bakeoff", "toy-8b", "--models-json", str(mj))
    assert p.returncode == 2, p.stdout + p.stderr
    assert "two model resolvers" in p.stderr


def test_an_unknown_slug_names_what_the_registry_holds(tmp_path):
    p = _launch(tmp_path, "--model-artifact", "no-such-artifact")
    assert p.returncode == 2, p.stdout + p.stderr
    assert "no artifact 'no-such-artifact'" in p.stderr
    assert "mergeddemoa" in p.stderr           # ...and says what IS there


# --------------------------------------------------------------------------- #
# 2. the lora_forbidden refusal — the double-apply trap
# --------------------------------------------------------------------------- #

def test_lora_against_a_merged_artifact_is_a_hard_refusal(tmp_path):
    """Mounting an adapter over a merged dir applies it a second time; every
    readiness gate stays green and the eval scores a model nobody trained, so
    this cannot be a warning."""
    p = _launch(tmp_path, "--model-artifact", "mergeddemoa", "--lora", "v10=ad/v10")
    assert p.returncode == 8, p.stdout + p.stderr
    assert "lora_forbidden" in p.stderr
    assert "applies it" in p.stderr and "twice" in p.stderr


def test_the_lora_refusal_fires_before_anything_is_staged(tmp_path):
    """Pre-spend means pre-spend: no key mint, no marker, no B2 write."""
    p = _launch(tmp_path, "--model-artifact", "mergeddemoa", "--lora", "v10=ad/v10")
    assert p.returncode == 8
    assert "would stage" not in p.stdout
    assert not (tmp_path / "key.txt").exists()


def test_lora_without_an_artifact_is_untouched(tmp_path):
    """The legacy path the refusal must not reach — a base model plus adapters
    is the shape most of this lane serves."""
    p = _launch(tmp_path, "--model", "b2:base-models/qwen35-9b",
                "--lora", "v10=ad/v10")
    assert p.returncode == 0, p.stdout + p.stderr
    assert _envmap(p.stdout)["LORA_SPECS"] == "v10=ad/v10"


# --------------------------------------------------------------------------- #
# 3. the pre-spend B2 gate
# --------------------------------------------------------------------------- #

def _launch_with(tmp_path, **stub_kw):
    e = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "TMPDIR")}
    e["PATH"] = _rclone_stub(tmp_path, **stub_kw) + os.pathsep + e.get("PATH", "")
    return _launch(tmp_path, "--model-artifact", "mergeddemoa", stub=False, **e)


def test_a_complete_artifact_passes_and_says_what_it_checked(tmp_path):
    p = _launch_with(tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "PUSHED.json present" in p.stderr
    assert f"{MERGEDDEMOA_FILES} objects == registry n_files" in p.stderr
    assert "PASSED pre-spend" in p.stderr


def test_an_absent_artifact_refuses_before_spending(tmp_path):
    p = _launch_with(tmp_path, names=[])
    assert p.returncode == 9, p.stdout + p.stderr
    assert "ABSENT (no objects)" in p.stderr
    assert "REFUSING to launch — nothing has been spent" in p.stderr


def test_a_prefix_with_no_completion_marker_refuses(tmp_path):
    """`PUSHED.json` is written LAST, after a read-back, so its absence is
    exactly the half-published prefix the write ordering exists to expose."""
    p = _launch_with(tmp_path, names=["a.safetensors", "config.json"])
    assert p.returncode == 9, p.stdout + p.stderr
    assert "no PUSHED.json" in p.stderr


def test_a_short_publish_refuses_on_the_object_count(tmp_path):
    """The failure the marker cannot see: it only stats itself. A restore that
    lands 20 of 27 shards serves, answers, and scores like the baseline."""
    short = [f"f{i}.safetensors" for i in range(1, 20)] + ["PUSHED.json"]
    p = _launch_with(tmp_path, names=short)
    assert p.returncode == 9, p.stdout + p.stderr
    assert f"registry says {MERGEDDEMOA_FILES}" in p.stderr


def test_a_receipt_that_contradicts_the_registry_refuses(tmp_path):
    """Publisher and registry disagreeing about what was published is not a
    detail to reconcile at serve time."""
    p = _launch_with(tmp_path, receipt={"complete": True, "files": 12})
    assert p.returncode == 9, p.stdout + p.stderr
    assert "publisher and the registry disagree" in p.stderr


def test_an_incomplete_receipt_refuses(tmp_path):
    p = _launch_with(tmp_path, receipt={"complete": False,
                                        "files": MERGEDDEMOA_FILES})
    assert p.returncode == 9, p.stdout + p.stderr
    assert "complete=False != True" in p.stderr


def test_an_unreachable_b2_is_a_refusal_not_a_skip(tmp_path):
    """Cannot check is a refusal. A gate that shrugs when it cannot read is a
    gate that reports success on exactly the runs it was needed for."""
    p = _launch_with(tmp_path, lsf_rc=1)
    assert p.returncode == 9, p.stdout + p.stderr
    assert "cannot LIST" in p.stderr
    assert "cannot check is a REFUSAL" in p.stderr


def test_the_gate_bytes_size_the_disk_without_a_second_listing(tmp_path):
    """The gate already measured the prefix; sizing off its number means the
    52 GiB prefix is listed once instead of twice, and the two cannot disagree."""
    p = _launch_with(tmp_path)
    assert re.search(r">> disk: \d+GB \(auto: model 52\.0GB", p.stdout), p.stdout


def test_a_plain_b2_model_gains_no_new_requirements(tmp_path):
    """The compatibility floor. No registry entry, no gate, no expectation —
    every pre-artifact caller behaves exactly as it did."""
    p = _launch(tmp_path, "--model", "b2:base-models/qwen35-9b")
    assert p.returncode == 0, p.stdout + p.stderr
    env = _envmap(p.stdout)
    assert "SERVE_IDENT_REQUIRED" not in env
    assert "SERVE_DTYPE" not in env
    assert "artifact-gate" not in p.stderr


# --------------------------------------------------------------------------- #
# 4. the identity expectation and its staging
# --------------------------------------------------------------------------- #

def test_the_identity_payload_is_staged_to_the_per_serve_prefix(tmp_path):
    """None of it fits the 16 KiB onstart wire (merged_fingerprint.py alone is
    ~11.7 KiB), so it rides the same per-SERVE B2 prefix as parse_vllm_mem.py."""
    p = _launch(tmp_path, "--model-artifact", "mergeddemoa")
    assert p.returncode == 0, p.stdout + p.stderr
    for name in ("identity_expect.json", "serve_identity_gate.py",
                 "merged_fingerprint.py"):
        assert re.search(
            r"would stage %s \(\d+B\) -> b2:fake/serve/\S+/%s" % (name, name),
            p.stdout), (name, p.stdout)


def test_dirhash_is_staged_only_when_a_grade_b_pin_exists(tmp_path):
    """Staging it unconditionally would put a tool on the box for a check no
    expectation asks for — and NOT staging it when a grade-B pin exists would
    silently drop the byte check to a name/size one. Both directions, because
    only asserting the negative is how this passed for a grade-A-only roster
    and would have kept passing after the pin landed."""
    _fixture_registry(tmp_path, GRADE_A_ONLY_MERGED)
    p = _launch(tmp_path, "--model-artifact", "gradea-fixture")
    # The launch must actually REACH staging: "dirhash.py not in stdout" is
    # trivially true of an aborted run, so assert the run happened first.
    assert p.returncode == 0, p.stdout + p.stderr
    assert "merged_fingerprint.py" in p.stdout, "grade A did not stage either"
    assert "dirhash.py" not in p.stdout, p.stdout


def test_dirhash_is_staged_for_a_grade_b_artifact(tmp_path):
    """The positive half, on the live mergeddemoa seed: its bytes were measured
    2026-08-25, so the box must receive the tool that checks them."""
    p = _launch(tmp_path, "--model-artifact", "mergeddemoa")
    assert "dirhash.py" in p.stdout, p.stdout


def test_the_null_grade_b_pin_degrades_loudly_and_distinctly(tmp_path):
    """`content_sha256: null` is UNMEASURED, never clean. The operator must be
    able to read, from the launch log, that the bytes were NOT checked.

    On a FIXTURE, not on whichever seed happens to be unmeasured: mergeddemoa was
    this test's subject until its bytes were measured, at which point the test
    was asserting a fact about the roster rather than about the launcher."""
    _fixture_registry(tmp_path, GRADE_A_ONLY_MERGED)
    p = _launch(tmp_path, "--model-artifact", "gradea-fixture")
    assert "grade-B pin is NULL" in p.stderr
    assert "UNMEASURED, not clean" in p.stderr
    assert "gate_dir.py --emit" in p.stderr        # ...and how to fix it


def test_an_artifact_with_no_identity_pin_is_refused(tmp_path):
    """An artifact whose pins are all null is refused pre-spend, and the refusal
    names the UNGATED door: `--model-artifact` is the gated one, and an
    expectation that cannot fail is not a gate."""
    _fixture_registry(tmp_path, UNPINNED_BASE)
    p = _launch(tmp_path, "--model-artifact", "unpinned-fixture")
    assert p.returncode == 9, p.stdout + p.stderr
    # the fixture registry really was read, so the refusal is the expectation's
    # and not a slug that resolved to nothing
    assert "artifact 'unpinned-fixture' (kind=base" in p.stderr
    assert "carries NO identity pin" in p.stderr
    assert "--model b2:base-models/unpinned-fixture" in p.stderr
    assert "PASSED pre-spend" not in p.stderr


def test_a_content_rollup_ALONE_is_still_no_identity_pin(tmp_path):
    """Grade A is the on-box gate's floor and grade B is layered on it, so a
    base seed whose content rollup a merge run measured still has no
    expectation any box can verify — and must still refuse here, at $0."""
    _fixture_registry(tmp_path, dict(UNPINNED_BASE, content_sha256="ab" * 32,
                                     n_files=26))
    p = _launch(tmp_path, "--model-artifact", "unpinned-fixture")
    assert p.returncode == 9, p.stdout + p.stderr
    assert "carries NO identity pin" in p.stderr
    assert "PASSED pre-spend" not in p.stderr


def test_the_box_is_armed_by_one_variable(tmp_path):
    """`SERVE_IDENT_REQUIRED` is what makes an unresolvable expectation a
    FAILURE on the box rather than a skip."""
    env = _envmap(_launch(tmp_path, "--model-artifact", "mergeddemoa").stdout)
    assert env["SERVE_IDENT_REQUIRED"] == "1"
    assert env["SERVE_IDENT_ARTIFACT"] == "mergeddemoa"


def test_the_expectation_is_composed_from_git_not_from_b2(tmp_path):
    """The load-bearing property. An expectation read from the guards published
    beside the weights would have B2 corroborating B2, and a renamed or
    re-published prefix would agree with itself. The composer's rclone-free-ness
    is asserted BEHAVIOURALLY: with no rclone at all it still produces the file,
    and only the (later) B2 check fails."""
    e = {k: v for k, v in os.environ.items() if k in ("LANG", "TMPDIR")}
    e["PATH"] = str(tmp_path / "emptybin")
    (tmp_path / "emptybin").mkdir()
    for tool in ("bash", "python3", "openssl", "sed", "grep", "cat", "wc",
                 "date", "mktemp", "dirname", "basename", "tr", "rm", "mkdir",
                 "chmod", "awk", "head", "seq", "printf", "cut", "sort",
                 "paste", "env", "readlink", "ls"):
        src = shutil.which(tool)
        if src:
            os.symlink(src, tmp_path / "emptybin" / tool)
    p = _launch(tmp_path, "--model-artifact", "mergeddemoa", stub=False, **e)
    assert p.returncode == 9, p.stdout + p.stderr
    # the EXPECTATION step passed (it never needed B2); the B2 CHECK is what failed
    assert "carries NO identity pin" not in p.stderr
    assert "cannot LIST" in p.stderr


def test_the_frozen_expectation_pins_the_registrys_own_numbers():
    """Composed directly, so the JSON the box is gated on is inspectable without
    a launch. `serve_artifact.py` is the composer; nothing else may mint one."""
    out = subprocess.run(
        ["python3", os.path.join(_HERE, "serve_artifact.py"), "expect",
         "mergeddemoa"], capture_output=True, text=True, timeout=60, cwd=_HERE)
    assert out.returncode == 0, out.stderr
    doc = json.loads(out.stdout)
    seed = json.load(open(os.path.join(_HERE, "modelkit", "registry",
                                       "mergeddemoa.json")))
    assert doc["fingerprint_sha256"] == seed["fingerprint_sha256"]
    assert doc["n_files"] == seed["n_files"] == MERGEDDEMOA_FILES
    assert doc["b2_model"] == seed["b2_model"]
    # Whatever the seed pins, the composer must carry it through and grade it
    # accordingly — asserting a literal "A" here made this a test of the
    # roster, and it went red the day mergeddemoa's bytes were measured.
    assert doc["content_sha256"] == seed["content_sha256"]
    assert doc["grade"] == ("B" if seed["content_sha256"] else "A")


# --------------------------------------------------------------------------- #
# 5. the boot-SLA relaunch spec
# --------------------------------------------------------------------------- #

def test_the_relaunch_spec_round_trips_the_artifact_flag():
    """A condemned serve is re-fired from its saved argv on a different host. If
    `--model-artifact` were dropped there, the replacement would resolve nothing,
    gate nothing and verify nothing — an ungated serve produced by the recovery
    path for a gated one, which nobody would look at again.

    `write_sla_spec` allows by OMISSION (only three flags are stripped), so this
    holds for free; the gate is here so that adding a fourth entry is a red test.
    """
    src = open(LAUNCH_SH, encoding="utf-8").read()
    skip = re.search(r"^skip_valued = \{(.*)\}$", src, re.M).group(1)
    assert "--model-artifact" not in skip
    assert "--dtype" not in skip
    # and the flag really is two tokens the passthrough keeps intact
    assert "--model-artifact) MODEL_ARTIFACT=\"$2\"; shift 2;;" in src
