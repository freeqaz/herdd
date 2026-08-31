"""The durable model flip: `serve_flip.sh` writes it, `serve_vllm.sh` obeys it.

The failure it closes (2026-08-26, box 48737856). A serve box's model arrives as
vast `extra_env`, which is fixed at create time. A flip done at runtime — kill
the base vLLM, hand-start one on a merged dir — is process state and nothing
else, so when a spot eviction's rescue re-created the container, PID 1 re-ran the
stored onstart against that same env and the box came back serving
`/workspace/base-model` under the ratified served-model id. Renaming the base dir
did not help: the boot re-pulls it from B2 before serving it.

THE PROPERTY THAT MATTERS IS THE REFUSAL. A flipped box that cannot serve the
override must go DOWN, not back to the launch model: `Connection refused` is a
state the eval gates already handle, and the launch weights answering under the
override's label is the exact failure the ratification record exists to prevent.
So every unusable-override case below asserts BOTH halves — non-zero exit AND
that no `vllm serve` argv was produced at all.

Behavioural throughout: the real shipped scripts, `serve_vllm.sh` under its own
`DRY_RUN=1` preview, with `/etc/environment` redirected into tmp_path (the
script persists env there on every run, and the real file is not this suite's to
rewrite).

Skipped when bash is unavailable (the portable lane runs it).
"""
import json
import os
import shutil
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
SERVE_SH = os.path.join(_HERE, "onstart", "serve_vllm.sh")
FLIP_SH = os.path.join(_HERE, "serve_flip.sh")

pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")


def _merged_dir(tmp_path, name="merged", marker=".v4_relayout_ok.json"):
    """A servable-looking model dir, optionally with its completion marker."""
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text("{}")
    if marker:
        (d / marker).write_text("{}")
    return d


def _serve(tmp_path, **env):
    """`serve_vllm.sh` DRY_RUN=1 -> CompletedProcess, with `.argv`.

    MAX_HOURS=0 disarms the watchdog and SERVE_DP=1 skips the nvidia-smi probe;
    MNBT_DEVICE_TOTAL_MIB pins the card the prefill default is derived from so
    the argv reads the same on a GPU box as in the portable lane.
    """
    home = tmp_path / "home"
    (home / ".config" / "rclone").mkdir(parents=True, exist_ok=True)
    e = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "TMPDIR")}
    e["HOME"] = str(home)
    e.update({"DRY_RUN": "1", "MAX_HOURS": "0", "SERVE_DP": "1",
              "MNBT_DEVICE_TOTAL_MIB": "97887",
              "SERVE_ENV_FILE": str(tmp_path / "environment"),
              # default to a path nothing wrote, so a test that means "no
              # override" cannot accidentally read a real /workspace one
              "SERVE_MODEL_OVERRIDE": str(tmp_path / "no-override.json"),
              "SERVE_FLIP_EVIDENCE": str(tmp_path / "no-evidence")})
    e.update({k: v for k, v in env.items() if v is not None})
    p = subprocess.run(["bash", SERVE_SH], capture_output=True, text=True,
                       timeout=120, env=e, cwd=_HERE)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("vllm serve")]
    p.argv = lines[0].split() if lines else None
    return p


def _flip(tmp_path, *args):
    e = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "TMPDIR")}
    e["HOME"] = str(tmp_path / "home")
    return subprocess.run(["bash", FLIP_SH, *args], capture_output=True,
                          text=True, timeout=60, env=e, cwd=_HERE)


def _write_override(path, **fields):
    doc = {"schema_version": 1}
    doc.update({k: v for k, v in fields.items() if v is not None})
    path.write_text(json.dumps(doc))
    return str(path)


# --------------------------------------------------------------------------- #
# 1. the control: nothing changes for a box nobody flipped
# --------------------------------------------------------------------------- #

def test_no_override_serves_the_launch_model(tmp_path):
    """The compatibility floor. Every serve that predates the flip file behaves
    exactly as it did — MODEL_B2 still resolves to the local pull."""
    p = _serve(tmp_path, MODEL_B2="base-models/qwen35-9b",
               SERVED_NAME="qwen35-9b")
    assert p.returncode == 0, p.stderr
    assert p.argv[2] == "/workspace/base-model"
    assert "LAUNCH DEFAULT" in p.stdout
    assert "OVERRIDE ACTIVE" not in p.stdout


def test_every_start_says_which_model_it_resolved_and_why(tmp_path):
    """The incident had to be reconstructed from the instance record because no
    log line said which of MODEL_B2 / MODEL_ID / a flip had won."""
    p = _serve(tmp_path, MODEL_B2="base-models/qwen35-9b",
               SERVED_NAME="qwen35-9b")
    line = [ln for ln in p.stdout.splitlines()
            if ln.startswith(">> serve model RESOLVED:")]
    assert len(line) == 1, p.stdout
    assert "/workspace/base-model" in line[0]
    assert "MODEL_B2=base-models/qwen35-9b" in line[0]
    assert "qwen35-9b" in line[0]


# --------------------------------------------------------------------------- #
# 2. a valid override wins, on every start, over the immutable launch env
# --------------------------------------------------------------------------- #

def test_the_override_beats_MODEL_B2(tmp_path):
    """MODEL_B2 is what the resumed container gets handed, and it takes
    precedence over MODEL_ID — so the override has to clear it, not merely
    out-rank it."""
    d = _merged_dir(tmp_path)
    ov = _write_override(tmp_path / "ov.json", model_path=str(d),
                         marker=".v4_relayout_ok.json", reason="stage 1 flip")
    p = _serve(tmp_path, MODEL_B2="base-models/qwen35-9b",
               SERVED_NAME="qwen35-9b", SERVE_MODEL_OVERRIDE=ov)
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.argv[2] == str(d)
    assert "/workspace/base-model" not in " ".join(p.argv)
    assert "OVERRIDE ACTIVE" in p.stdout
    assert "stage 1 flip" in p.stdout


def test_the_launch_endpoint_label_is_kept_unless_the_override_renames_it(tmp_path):
    """Serving merged weights under the ratified id is usually the whole point
    of a flip, so `served_name` is opt-in, not defaulted from the dir name."""
    d = _merged_dir(tmp_path)
    ov = _write_override(tmp_path / "ov.json", model_path=str(d))
    p = _serve(tmp_path, MODEL_B2="base-models/qwen35-9b",
               SERVED_NAME="qwen35-9b", SERVE_MODEL_OVERRIDE=ov)
    assert p.argv[p.argv.index("--served-model-name") + 1] == "qwen35-9b"

    ov2 = _write_override(tmp_path / "ov2.json", model_path=str(d),
                          served_name="v4-merged", max_len="65536")
    q = _serve(tmp_path, MODEL_B2="base-models/qwen35-9b",
               SERVED_NAME="qwen35-9b", SERVE_MODEL_OVERRIDE=ov2)
    assert q.argv[q.argv.index("--served-model-name") + 1] == "v4-merged"
    assert q.argv[q.argv.index("--max-model-len") + 1] == "65536"


# --------------------------------------------------------------------------- #
# 3. unusable override => DOWN AND LOUD, never a fallback to the launch model
# --------------------------------------------------------------------------- #

def _assert_refused(p, *needles):
    assert p.returncode != 0, p.stdout + p.stderr
    assert p.argv is None, "produced a vllm argv on a refusal: %s" % p.argv
    assert "/workspace/base-model" not in p.stdout, p.stdout
    for n in needles:
        assert n in p.stderr, (n, p.stderr)


def test_a_missing_completion_marker_refuses(tmp_path):
    """The half-written merge. The marker is written last, so its absence is
    the one signal that separates a finished merge from a directory that
    already has a config.json in it."""
    d = _merged_dir(tmp_path, marker=None)
    ov = _write_override(tmp_path / "ov.json", model_path=str(d),
                         marker=".v4_relayout_ok.json")
    p = _serve(tmp_path, MODEL_B2="base-models/qwen35-9b",
               SERVED_NAME="qwen35-9b", SERVE_MODEL_OVERRIDE=ov)
    _assert_refused(p, ".v4_relayout_ok.json", "has not finished",
                    "NOT falling back to the launch model")


def test_a_target_that_does_not_exist_refuses(tmp_path):
    ov = _write_override(tmp_path / "ov.json",
                         model_path=str(tmp_path / "gone"))
    p = _serve(tmp_path, MODEL_B2="base-models/qwen35-9b",
               SERVED_NAME="qwen35-9b", SERVE_MODEL_OVERRIDE=ov)
    _assert_refused(p, "is not a directory on this box")


def test_a_target_with_no_config_json_refuses(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    ov = _write_override(tmp_path / "ov.json", model_path=str(d))
    p = _serve(tmp_path, MODEL_B2="base-models/qwen35-9b",
               SERVED_NAME="qwen35-9b", SERVE_MODEL_OVERRIDE=ov)
    _assert_refused(p, "no config.json")


def test_an_unparseable_override_refuses(tmp_path):
    """Truncated by an interrupted write, or hand-edited. Either way it is not
    an instruction to serve the launch model."""
    ov = tmp_path / "ov.json"
    ov.write_text('{"model_path": "/workspace/merg')
    p = _serve(tmp_path, MODEL_B2="base-models/qwen35-9b",
               SERVED_NAME="qwen35-9b", SERVE_MODEL_OVERRIDE=str(ov))
    _assert_refused(p, "not a usable override document")


def test_an_override_with_no_model_path_refuses(tmp_path):
    ov = _write_override(tmp_path / "ov.json", reason="oops")
    p = _serve(tmp_path, MODEL_B2="base-models/qwen35-9b",
               SERVED_NAME="qwen35-9b", SERVE_MODEL_OVERRIDE=ov)
    _assert_refused(p, "not a usable override document")


# --------------------------------------------------------------------------- #
# 4. the fail-closed guard: a FLIPPED box may not quietly re-serve the launch
#    model, and this is the shape the incident actually took
# --------------------------------------------------------------------------- #

def test_flip_evidence_without_an_override_refuses_the_launch_model(tmp_path):
    """The exact 2026-08-26 boot: the flip lived only in a process, the rescue
    re-ran the onstart against the immutable env, and MODEL_B2 was re-pulled and
    served under the ratified label. Evidence that a flip happened must make
    that boot refuse. It also makes the `park the base dir` defence work, which
    on its own is undone by the very re-pull it is meant to defeat."""
    ev = tmp_path / "base-model.parked"
    ev.mkdir()
    p = _serve(tmp_path, MODEL_B2="base-models/qwen35-9b",
               SERVED_NAME="qwen35-9b", SERVE_FLIP_EVIDENCE=str(ev))
    assert p.returncode != 0, p.stdout + p.stderr
    assert p.argv is None
    assert "was FLIPPED away from the launch model" in p.stderr
    assert "REFUSING" in p.stderr


def test_evidence_is_moot_once_the_override_is_installed(tmp_path):
    """The guard is a backstop for a flip the box cannot express, not a second
    thing to keep in sync: a valid override serves, evidence or no evidence."""
    d = _merged_dir(tmp_path)
    ev = tmp_path / ".serve_flipped"
    ev.write_text("flipped\n")
    ov = _write_override(tmp_path / "ov.json", model_path=str(d))
    p = _serve(tmp_path, MODEL_B2="base-models/qwen35-9b",
               SERVED_NAME="qwen35-9b", SERVE_MODEL_OVERRIDE=ov,
               SERVE_FLIP_EVIDENCE=str(ev))
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.argv[2] == str(d)


# --------------------------------------------------------------------------- #
# 5. the two refusals a flip inherits from the launch it is replacing
# --------------------------------------------------------------------------- #

def test_the_launchs_adapter_is_not_mounted_over_the_override(tmp_path):
    """`lora_forbidden`, one lane over. When the override IS the merge, the
    launch's LORA_SPECS applies the adapter a second time — the server boots,
    /v1/models lists what you asked for, and the eval scores a model nobody
    trained."""
    d = _merged_dir(tmp_path)
    ov = _write_override(tmp_path / "ov.json", model_path=str(d))
    p = _serve(tmp_path, MODEL_B2="base-models/qwen35-9b",
               SERVED_NAME="qwen35-9b", SERVE_MODEL_OVERRIDE=ov,
               LORA_SPECS="v4=ad/v4")
    _assert_refused(p, "applies it TWICE")

    ov2 = _write_override(tmp_path / "ov2.json", model_path=str(d),
                          allow_lora=True)
    q = _serve(tmp_path, MODEL_B2="base-models/qwen35-9b",
               SERVED_NAME="qwen35-9b", SERVE_MODEL_OVERRIDE=ov2,
               LORA_SPECS="v4=ad/v4")
    assert q.returncode == 0, q.stdout + q.stderr
    assert "--enable-lora" in q.argv


def test_an_armed_identity_gate_refuses_an_override_that_brings_no_expectation(tmp_path):
    """The launch expectation describes the LAUNCH artifact, so checking a
    flipped box against it can only fail — and skipping it would disarm the one
    gate that can see wrong weights. The override must carry its own."""
    d = _merged_dir(tmp_path)
    ov = _write_override(tmp_path / "ov.json", model_path=str(d))
    p = _serve(tmp_path, MODEL_B2="base-models/qwen35-9b",
               SERVED_NAME="qwen35-9b", SERVE_MODEL_OVERRIDE=ov,
               SERVE_IDENT_REQUIRED="1", SERVE_IDENT_ARTIFACT="mergeddemoa")
    _assert_refused(p, "points serving elsewhere", "REFUSING")


# --------------------------------------------------------------------------- #
# 6. serve_flip.sh — the writer, and that the two halves agree
# --------------------------------------------------------------------------- #

def test_the_writer_and_the_reader_agree(tmp_path):
    """Round trip. A flip file the shipped writer produces is one the shipped
    serve path serves from — asserted end to end rather than by two literals."""
    d = _merged_dir(tmp_path)
    ovf = tmp_path / "ov.json"
    w = _flip(tmp_path, "write", "--model-path", str(d),
              "--marker", ".v4_relayout_ok.json", "--reason", "stage 1",
              "--file", str(ovf), "--sentinel", str(tmp_path / ".flipped"))
    assert w.returncode == 0, w.stdout + w.stderr
    p = _serve(tmp_path, MODEL_B2="base-models/qwen35-9b",
               SERVED_NAME="qwen35-9b", SERVE_MODEL_OVERRIDE=str(ovf))
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.argv[2] == str(d)
    assert "stage 1" in p.stdout


def test_the_write_is_atomic_and_leaves_no_partial_file(tmp_path):
    """Temp-in-the-same-dir plus rename: a boot that reads this file mid-write
    must see the old document or the new one, never half of one."""
    d = _merged_dir(tmp_path)
    dest = tmp_path / "ovdir" / "ov.json"
    assert _flip(tmp_path, "write", "--model-path", str(d),
                 "--file", str(dest),
                 "--sentinel", str(tmp_path / ".flipped")).returncode == 0
    assert json.loads(dest.read_text())["model_path"] == str(d)
    assert sorted(os.listdir(dest.parent)) == ["ov.json"], \
        "a temp file survived the write"


def test_the_writer_refuses_a_target_the_box_would_refuse(tmp_path):
    """Installing an override that is guaranteed to fail at boot is a down
    endpoint you find out about after the next eviction."""
    d = _merged_dir(tmp_path, marker=None)
    w = _flip(tmp_path, "write", "--model-path", str(d),
              "--marker", ".v4_relayout_ok.json",
              "--file", str(tmp_path / "ov.json"),
              "--sentinel", str(tmp_path / ".flipped"))
    assert w.returncode == 1, w.stdout + w.stderr
    assert "has not finished" in w.stderr
    assert not (tmp_path / "ov.json").exists()


def test_write_drops_the_fail_closed_sentinel_and_clear_removes_both(tmp_path):
    """The sentinel is what makes a LOST override file a refusal instead of a
    silent return to the launch weights."""
    d = _merged_dir(tmp_path)
    ovf, sent = tmp_path / "ov.json", tmp_path / ".flipped"
    assert _flip(tmp_path, "write", "--model-path", str(d), "--file", str(ovf),
                 "--sentinel", str(sent)).returncode == 0
    assert sent.exists()
    # override gone, sentinel left: the box must NOT fall back
    ovf.unlink()
    p = _serve(tmp_path, MODEL_B2="base-models/qwen35-9b",
               SERVED_NAME="qwen35-9b", SERVE_MODEL_OVERRIDE=str(ovf),
               SERVE_FLIP_EVIDENCE=str(sent))
    assert p.returncode != 0 and p.argv is None
    assert "was FLIPPED away" in p.stderr

    assert _flip(tmp_path, "clear", "--file", str(ovf),
                 "--sentinel", str(sent)).returncode == 0
    assert not sent.exists()


def test_the_restart_pattern_cannot_match_its_own_shell():
    """`pkill -f 'vllm serve'` is present in the argv of the shell running it,
    so it kills the script that is doing the flip."""
    src = open(FLIP_SH, encoding="utf-8").read()
    assert "pkill -f '[v]llm serve'" in src
    assert "pkill -f 'vllm serve'" not in src
