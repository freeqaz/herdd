"""Shell-level test for the serve_summary.json capture wired into
onstart/serve_vllm.sh.

We cannot run vLLM here (no GPU, and the local GPU is off limits anyway), so
what is under test is the WIRING, not the engine: the real
`_resolve_mem_parser` / `capture_serve_summary` functions are lifted verbatim
out of the shipped serve_vllm.sh and driven with a stub `rclone` on PATH and a
canned vLLM startup log on disk.

The properties that matter, in order of how much they would cost to get wrong:

  1. It is INERT when there is nothing to parse. No serve log, no parser, no
     B2 — every one of those must be a note and exit 0, because this runs on the
     serve path of a paid box right after the READY marker.
  2. It writes BESIDE THE LOG. That is what makes it ride the existing results
     plumbing (the jobs lane points SERVE_LOG inside the synced output dir).
  3. It ships to serve/<SERVE_ID>/serve_summary.json through the SAME remote as
     SERVE_STATUS/METRICS ([b2w] when a scoped write pair was shipped), and
     touches rclone at all ONLY when a marker is configured.

Skipped when bash is unavailable (portable lane runs it).
"""
import json
import os
import re
import shutil
import subprocess
import tempfile

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
SERVE_SH = os.path.join(_HERE, "onstart", "serve_vllm.sh")
PARSER = os.path.join(_HERE, "parse_vllm_mem.py")

pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")

#: Verbatim from a real vLLM 0.24.0 serve log (see test_parse_vllm_mem.py).
VLLM_LOG = """\
(EngineCore pid=1878) INFO 07-09 21:27:35 [core.py:114] Initializing a V1 LLM engine (v0.24.0) with config: model='/workspace/base', dtype=torch.bfloat16, max_seq_len=8192, tensor_parallel_size=1, data_parallel_size=1, quantization=None, kv_cache_dtype=auto, seed=0, served_model_name=base
(EngineCore pid=1878) INFO 07-09 21:28:26 [gpu_model_runner.py:5255] Model loading took 14.37 GiB memory and 13.381314 seconds
(EngineCore pid=1878) INFO 07-09 21:29:51 [gpu_worker.py:508] Available KV cache memory: 11.92 GiB
(EngineCore pid=1878) INFO 07-09 21:29:51 [kv_cache_utils.py:2146] GPU KV cache size: 223,152 tokens
(EngineCore pid=1878) INFO 07-09 21:30:13 [gpu_worker.py:597] CUDA graph pool memory: 0.65 GiB (actual), 0.63 GiB (estimated), difference: 0.02 GiB (2.7%).
"""

_FAKE_RCLONE = """#!/usr/bin/env bash
echo "CALL $*" >> "$RCLONE_CALLS"
case "$1" in
  rcat)
    [ "${RCLONE_RCAT_FAIL:-0}" = "1" ] && exit 1
    body="$(cat)"
    printf '%s' "$body" > "$RCLONE_RCAT_BODY"
    exit 0 ;;
  copyto) exit 1 ;;
  *) exit 0 ;;
esac
"""


def _extract_capture_block():
    """Lift the two real functions out of the shipped serve_vllm.sh.

    Anchored on the function name and the next section header. If either moves,
    this raises rather than silently testing nothing — a test that quietly stops
    covering its subject is worse than one that fails.
    """
    src = open(SERVE_SH).read()
    m = re.search(r"^_resolve_mem_parser\(\) \{.*?^# --- readiness poll",
                  src, re.S | re.M)
    assert m, ("could not find the capture block in serve_vllm.sh — did "
               "_resolve_mem_parser/capture_serve_summary move or get renamed?")
    block = m.group(0).rsplit("# --- readiness poll", 1)[0]
    assert "capture_serve_summary()" in block
    return block


def _run(tmp, *, log_text=VLLM_LOG, write_log=True, marker=True,
         parser=PARSER, b2w="b2", extra_env=None):
    """Drive capture_serve_summary once. Returns (proc, paths dict)."""
    bind = os.path.join(tmp, "bin")
    logdir = os.path.join(tmp, "out", "eval")
    os.makedirs(bind, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    log_path = os.path.join(logdir, "serve.log")
    if write_log:
        with open(log_path, "w") as fh:
            fh.write(log_text)

    calls = os.path.join(tmp, "calls")
    open(calls, "w").close()
    rcat_body = os.path.join(tmp, "rcat_body")
    rc = os.path.join(bind, "rclone")
    with open(rc, "w") as fh:
        fh.write(_FAKE_RCLONE)
    os.chmod(rc, 0o755)
    # nvidia-smi must be ABSENT so --nvidia-smi exercises its fail-soft path.
    harness = os.path.join(tmp, "harness.sh")
    with open(harness, "w") as fh:
        fh.write("#!/usr/bin/env bash\nset -euo pipefail\nB2W=%s\n%s\n"
                 "capture_serve_summary 'base,reader' || echo HARNESS_NONZERO\n"
                 % (b2w, _extract_capture_block()))

    env = dict(os.environ)
    env["PATH"] = bind + os.pathsep + env["PATH"]
    env["RCLONE_CALLS"] = calls
    env["RCLONE_RCAT_BODY"] = rcat_body
    env["SERVE_LOG"] = log_path
    env["MODEL_B2"] = "base-models/qwen3-8b"
    env["MODEL_ID"] = ""
    env["SERVED_NAME"] = "qwen3-8b"
    env["MAX_LEN"] = "16384"
    env["GPU_UTIL"] = "0.90"
    env["SERVE_DP"] = "1"
    env["SERVE_TP"] = "1"
    env["SERVE_REPLICAS"] = "1"
    env["QUANTIZATION"] = ""
    env["KV_CACHE_DTYPE"] = "fp8"
    env["LORA_SPECS"] = "reader=artifacts/x/serve/gen"
    env["MAX_LORA_RANK"] = "32"
    if parser:
        env["VLLM_MEM_PARSER"] = parser
    else:
        env.pop("VLLM_MEM_PARSER", None)
    if marker:
        env["SERVE_ID"] = "serve-260809-1200-ab12"
        env["B2_KEY_ID"] = "k"
        env["B2_BUCKET"] = "bucket"
    else:
        for k in ("SERVE_ID", "B2_KEY_ID", "B2_BUCKET"):
            env.pop(k, None)
    env.update(extra_env or {})

    proc = subprocess.run(["bash", harness], capture_output=True, text=True,
                          timeout=180, env=env, cwd=tmp)
    return proc, {"log": log_path, "logdir": logdir, "calls": calls,
                  "rcat_body": rcat_body,
                  "summary": os.path.join(logdir, "serve_summary.json")}


@pytest.fixture()
def tmp():
    d = tempfile.mkdtemp()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #

def test_summary_lands_beside_the_serve_log(tmp):
    proc, p = _run(tmp)
    assert proc.returncode == 0, proc.stderr
    assert "HARNESS_NONZERO" not in proc.stdout
    assert os.path.exists(p["summary"]), proc.stdout + proc.stderr
    doc = json.load(open(p["summary"]))
    assert doc["schema"] == "serve_memory/v1"
    assert doc["parsed_ok"] is True
    assert doc["memory"]["kv_cache_gib"] == 11.92
    assert doc["memory"]["model_load_gib"] == 14.37
    assert doc["kv"]["tokens"] == 223152
    assert doc["log_path"] == p["log"]


def test_serve_shape_is_recorded_from_the_env_the_serve_actually_used(tmp):
    proc, p = _run(tmp)
    assert proc.returncode == 0, proc.stderr
    shape = json.load(open(p["summary"]))["shape"]
    assert shape["serve_id"] == "serve-260809-1200-ab12"
    assert shape["served_name"] == "qwen3-8b"
    assert shape["model"] == "base-models/qwen3-8b"
    assert shape["max_len"] == 16384
    assert shape["gpu_util"] == 0.9
    assert shape["serve_tp"] == 1
    assert shape["serve_mode"] == "single"
    assert shape["kv_cache_dtype"] == "fp8"
    assert shape["quantization"] is None
    assert shape["served_ids"] == "base,reader"


def test_summary_is_rcat_to_the_serve_prefix(tmp):
    proc, p = _run(tmp)
    assert proc.returncode == 0, proc.stderr
    calls = open(p["calls"]).read()
    assert "b2:bucket/serve/serve-260809-1200-ab12/serve_summary.json" in calls
    shipped = json.loads(open(p["rcat_body"]).read())
    assert shipped["memory"]["kv_cache_gib"] == 11.92
    assert ">> serve summary -> b2:bucket/serve/" in proc.stdout


def test_scoped_write_pair_routes_through_b2w(tmp):
    proc, p = _run(tmp, b2w="b2w")
    assert proc.returncode == 0, proc.stderr
    assert "b2w:bucket/serve/serve-260809-1200-ab12/serve_summary.json" \
        in open(p["calls"]).read()


def test_replica_mode_is_labelled(tmp):
    proc, p = _run(tmp, extra_env={"SERVE_REPLICAS": "4"})
    assert proc.returncode == 0, proc.stderr
    assert json.load(open(p["summary"]))["shape"]["serve_mode"] == "haproxy"


# --------------------------------------------------------------------------- #
# Inert / degraded paths — none of these may be fatal
# --------------------------------------------------------------------------- #

def test_inert_when_the_log_has_no_vllm_memory_lines(tmp):
    """The shell-level dry test the task asks for: wiring fires, nothing
    crashes, and the artifact says WHY it is empty."""
    proc, p = _run(tmp, log_text=">> job_serve: starting serve_vllm.sh\n")
    assert proc.returncode == 0, proc.stderr
    assert "HARNESS_NONZERO" not in proc.stdout
    doc = json.load(open(p["summary"]))
    assert doc["parsed_ok"] is False
    assert "parse" in doc["unavailable"]


def test_inert_when_no_serve_log_exists_at_all(tmp):
    proc, p = _run(tmp, write_log=False)
    assert proc.returncode == 0, proc.stderr
    assert "HARNESS_NONZERO" not in proc.stdout
    assert not os.path.exists(p["summary"])
    assert "parser wrote nothing" in proc.stderr
    # nothing to ship => rclone was never asked to
    assert "serve_summary.json" not in open(p["calls"]).read()


def test_inert_when_the_parser_is_not_on_the_box(tmp):
    proc, p = _run(tmp, parser=None)
    assert proc.returncode == 0, proc.stderr
    assert "HARNESS_NONZERO" not in proc.stdout
    assert "parse_vllm_mem.py not on this box" in proc.stderr
    assert not os.path.exists(p["summary"])


def test_no_marker_means_local_only_and_no_rclone(tmp):
    proc, p = _run(tmp, marker=False)
    assert proc.returncode == 0, proc.stderr
    assert os.path.exists(p["summary"])
    assert open(p["calls"]).read().strip() == ""


def test_a_failed_b2_write_keeps_the_local_copy_and_does_not_fail(tmp):
    proc, p = _run(tmp, extra_env={"RCLONE_RCAT_FAIL": "1"})
    assert proc.returncode == 0, proc.stderr
    assert "HARNESS_NONZERO" not in proc.stdout
    assert os.path.exists(p["summary"])
    assert "B2 write failed" in proc.stderr


def test_capture_is_invoked_only_after_the_ready_marker():
    """Ordering is the whole safety argument: an eval driver blocks on READY,
    so the capture must never sit in front of it."""
    src = open(SERVE_SH).read()
    # `$_mk` is the READY line's payload — the id CSV, plus the `ident=` field
    # when the on-box identity gate ran. The ordering property is unchanged.
    ready = src.index('status READY "$_mk"')
    capture = src.index('capture_serve_summary "$_ids"')
    assert ready < capture
    assert '|| true' in src[capture:capture + 120]
