"""Offline tests for tools/vast/parse_vllm_mem.py.

No GPU, no vLLM, no box. Every fixture below is COPIED VERBATIM from a real
vLLM serve log banked in this repo, so the parser is tested against wording we
have actually observed rather than wording we imagined:

  * `V024` — vLLM 0.24.0, from
    `out/modelzoo-reader-06-eval/eval/serve.log` and
    `out/jobs/20260711T051057-waveb-bakeoff05-149d/out/bakeoff/serve_logs/qwen36-27b-fp8.full.log`
  * `V019` — vLLM 0.19.1, from
    `out/jobs/20260713T025157-p2-reader-eval-01-ep1p0-f710/out/eval/serve.log`
    (different `gpu_worker.py` line numbers, different CUDA-graph advisory
    wording, and it DOES carry the `Graph capturing finished` line)
  * `V08_DEBUG` — the older `logger.debug` memory-profile sentence, kept so a
    `VLLM_LOGGING_LEVEL=DEBUG` serve yields the full per-term split.

The negative cases matter as much as the positive ones: the wiring has to be
inert — not fatal — when the log is a wrapper log, an empty file, a missing
path, or a vLLM whose wording this parser predates.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import parse_vllm_mem as P  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures — verbatim lines from real serve logs
# --------------------------------------------------------------------------- #

V024 = """\
(APIServer pid=1148) INFO 07-09 21:27:30 [api_server.py:1978] vLLM API server version 0.24.0
(EngineCore pid=1878) INFO 07-09 21:27:35 [core.py:114] Initializing a V1 LLM engine (v0.24.0) with config: model='/workspace/base', speculative_config=None, tokenizer='/workspace/base', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, disable_custom_all_reduce=False, quantization=None, quantization_config=None, enforce_eager=False, kv_cache_dtype=auto, device_config=cuda, seed=0, served_model_name=base, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None
(EngineCore pid=1878) INFO 07-09 21:27:38 [flash_attn.py:670] Using FlashAttention version 2
(EngineCore pid=1878) INFO 07-09 21:28:12 [weight_utils.py:849] Filesystem type for checkpoints: OVERLAY. Checkpoint size: 14.19 GiB. Available RAM: 121.01 GiB.
(EngineCore pid=1878) INFO 07-09 21:28:25 [default_loader.py:430] Loading weights took 12.26 seconds
(EngineCore pid=1878) INFO 07-09 21:28:26 [gpu_model_runner.py:5255] Model loading took 14.37 GiB memory and 13.381314 seconds
(EngineCore pid=1878) INFO 07-09 21:29:50 [gpu_model_runner.py:6588] Estimated CUDA graph memory: 0.63 GiB total
(EngineCore pid=1878) INFO 07-09 21:29:51 [gpu_worker.py:508] Available KV cache memory: 11.92 GiB
(EngineCore pid=1878) INFO 07-09 21:29:51 [gpu_worker.py:523] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.9000 is equivalent to --gpu-memory-utilization=0.8799 without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.9201. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
(EngineCore pid=1878) INFO 07-09 21:29:51 [kv_cache_utils.py:2146] GPU KV cache size: 223,152 tokens
(EngineCore pid=1878) INFO 07-09 21:29:51 [kv_cache_utils.py:2147] Maximum concurrency for 8,192 tokens per request: 27.24x
(EngineCore pid=1878) INFO 07-09 21:30:13 [gpu_model_runner.py:7331] Graph capturing finished in 22 secs, took 0.65 GiB
(EngineCore pid=1878) INFO 07-09 21:30:13 [gpu_worker.py:597] CUDA graph pool memory: 0.65 GiB (actual), 0.63 GiB (estimated), difference: 0.02 GiB (2.7%).
(EngineCore pid=1878) INFO 07-09 21:30:16 [core.py:283] init engine (profile, create kv cache, warmup model) took 41.20 seconds
(APIServer pid=1148) INFO 07-09 21:30:17 [api_server.py:592] Supported tasks: ['generate']
"""

V019 = """\
(EngineCore pid=1629) INFO 07-13 02:52:20 [core.py:96] Initializing a V1 LLM engine (v0.19.1) with config: model='/workspace/base', dtype=torch.bfloat16, max_seq_len=32768, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, quantization=None, quantization_config=None, enforce_eager=False, kv_cache_dtype=auto, seed=0, served_model_name=base, enable_prefix_caching=True, enable_chunked_prefill=True
(EngineCore pid=1629) INFO 07-13 02:52:45 [flash_attn.py:596] Using FlashAttention version 2
(EngineCore pid=1629) INFO 07-13 02:52:55 [default_loader.py:384] Loading weights took 10.49 seconds
(EngineCore pid=1629) INFO 07-13 02:52:56 [gpu_model_runner.py:4820] Model loading took 7.52 GiB memory and 10.887501 seconds
(EngineCore pid=1629) WARNING 07-13 02:53:13 [utils.py:267] Using default LoRA kernel configs
(EngineCore pid=1629) INFO 07-13 02:53:15 [gpu_model_runner.py:5955] Estimated CUDA graph memory: 0.61 GiB total
(EngineCore pid=1629) INFO 07-13 02:53:15 [gpu_worker.py:436] Available KV cache memory: 19.84 GiB
(EngineCore pid=1629) INFO 07-13 02:53:15 [gpu_worker.py:470] In v0.19, CUDA graph memory profiling will be enabled by default (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1), which more accurately accounts for CUDA graph memory during KV cache allocation. To try it now, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 and increase --gpu-memory-utilization from 0.9000 to 0.9196 to maintain the same effective KV cache size.
(EngineCore pid=1629) INFO 07-13 02:53:15 [kv_cache_utils.py:1319] GPU KV cache size: 324,976 tokens
(EngineCore pid=1629) INFO 07-13 02:53:15 [kv_cache_utils.py:1324] Maximum concurrency for 32,768 tokens per request: 9.92x
(EngineCore pid=1629) INFO 07-13 02:53:29 [gpu_model_runner.py:6046] Graph capturing finished in 12 secs, took 0.69 GiB
(EngineCore pid=1629) INFO 07-13 02:53:29 [gpu_worker.py:597] CUDA graph pool memory: 0.69 GiB (actual), 0.61 GiB (estimated), difference: 0.08 GiB (11.3%).
(EngineCore pid=1629) INFO 07-13 02:53:32 [core.py:283] init engine (profile, create kv cache, warmup model) took 35.83 seconds
"""

#: The DEBUG-only per-term split (vLLM 0.8-era wording), plus the V0 block line.
V08_DEBUG = """\
INFO 05-02 10:00:00 [llm_engine.py:234] Initializing a V0 LLM engine (v0.8.5) with config: model='Qwen/Qwen2.5-Coder-7B', dtype=torch.bfloat16, max_seq_len=16384, tensor_parallel_size=2, quantization=None, kv_cache_dtype=auto, seed=0, served_model_name=qwen
DEBUG 05-02 10:01:00 [worker.py:267] Memory profiling takes 4.32 seconds
DEBUG 05-02 10:01:00 [worker.py:268] the current vLLM instance can use total_gpu_memory (79.15GiB) x gpu_memory_utilization (0.90) = 71.23GiB
DEBUG 05-02 10:01:00 [worker.py:269] model weights take 14.99GiB; non_torch_memory takes 0.11GiB; PyTorch activation peak memory takes 1.20GiB; the rest of the memory reserved for KV Cache is 54.93GiB.
INFO 05-02 10:01:01 [executor_base.py:112] # GPU blocks: 28123, # CPU blocks: 2048
"""

#: The two lines that carry the throughput levers the engine banner does NOT.
#: Both COPIED VERBATIM from banked logs under `out/` (2026-08-09 sweep):
#:   * `scheduler.py:252` — from every 0.24.0 HTTP serve log (8192 on a <70 GiB
#:     card) and from `jobs/20260730T031619-frontier-wave-b79d/results/gen_C.log`
#:     (16384 on the in-process lane);
#:   * `api_utils.py:273` — from `out/v4_followup/r2_dose/ck50/serve.log` and
#:     siblings, where `--max-num-seqs 16` was pinned by serve_v4.sh.
#: Neither string appears in ANY banner we hold, which is why they are parsed
#: from their own lines rather than whitelisted into `_CONFIG_KEYS`.
V024_LEVERS = """\
(APIServer pid=1027546) INFO 07-27 23:50:25 [api_utils.py:273] non-default args: {'model_tag': '/home/someone/.cache/v4_merge/checkpoint-50-merged', 'model': '/home/someone/.cache/v4_merge/checkpoint-50-merged', 'dtype': 'bfloat16', 'max_model_len': 8192, 'served_model_name': ['tuner-v4-qwen35'], 'gpu_memory_utilization': 0.9, 'max_num_seqs': 16}
(APIServer pid=15592) INFO 07-11 11:19:20 [scheduler.py:252] Chunked prefill is enabled with max_num_batched_tokens=8192.
"""


def _tmp_log(tmp_path, text, name="serve.log"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# --------------------------------------------------------------------------- #
# Positive parses
# --------------------------------------------------------------------------- #

def test_v024_memory_profile():
    rep = P.parse_text(V024)
    assert rep["parsed_ok"] is True
    assert rep["schema"] == "serve_memory/v1"
    assert rep["vllm_version"] == "0.24.0"
    assert rep["engine"]["api_version"] == "V1"
    assert rep["memory"]["model_load_gib"] == 14.37
    assert rep["memory"]["kv_cache_gib"] == 11.92
    assert rep["memory"]["cudagraph_actual_gib"] == 0.65
    assert rep["memory"]["cudagraph_estimate_gib"] == 0.63
    assert rep["memory"]["cudagraph_estimate_error_pct"] == 2.7
    assert rep["kv"]["tokens"] == 223152
    assert rep["kv"]["tokens_per_request"] == 8192
    assert rep["kv"]["max_concurrency_x"] == 27.24
    assert rep["timing"]["weights_load_secs"] == 12.26
    assert rep["timing"]["graph_capture_secs"] == 22.0
    assert rep["timing"]["init_engine_secs"] == 41.20
    assert rep["host"]["checkpoint_gib"] == 14.19
    assert rep["host"]["ram_available_gib"] == 121.01
    assert rep["engine"]["attention"] == "FlashAttention version 2"


def test_v024_engine_config_whitelist():
    cfg = P.parse_text(V024)["engine_config"]
    assert cfg["max_seq_len"] == 8192
    assert cfg["dtype"] == "torch.bfloat16"
    assert cfg["tensor_parallel_size"] == 1
    assert cfg["data_parallel_size"] == 1
    assert cfg["served_model_name"] == "base"
    assert cfg["enforce_eager"] is False
    assert cfg["enable_prefix_caching"] is True
    # `quantization=None` must NOT be shadowed by `quantization_config=None`
    assert cfg["quantization"] is None
    assert cfg["kv_cache_dtype"] == "auto"
    assert cfg["model"] == "/workspace/base"


def test_v024_banner_carries_speculative_config():
    """`speculative_config=None` is in the real 0.24.0 banner and is the one
    throughput lever that was printed there and never lifted. None here means
    spec decode OFF — a real reading, so it must NOT also be named absent."""
    rep = P.parse_text(V024)
    assert "speculative_config" in rep["engine_config"]
    assert rep["engine_config"]["speculative_config"] is None
    assert "engine_config.speculative_config" not in rep["unavailable"]


def test_the_widths_are_named_absences_when_the_log_only_has_the_banner():
    """The banner cannot answer either width. A bare null would read as
    'unknown'; the point is that it means 'not pinned, engine default'."""
    rep = P.parse_text(V024)
    for key in ("max_num_seqs", "max_num_batched_tokens"):
        assert key in rep["engine_config"] and rep["engine_config"][key] is None
        assert "engine_config.%s" % key in rep["unavailable"]
    assert "NOT PINNED" in rep["unavailable"]["engine_config.max_num_seqs"]


def test_v019_banner_has_no_speculative_config_and_says_so():
    rep = P.parse_text(V019)
    assert rep["engine_config"]["speculative_config"] is None
    assert "engine_config.speculative_config" in rep["unavailable"]


def test_scheduler_line_yields_max_num_batched_tokens():
    rep = P.parse_text(V024 + V024_LEVERS)
    assert rep["engine_config"]["max_num_batched_tokens"] == 8192
    assert "engine_config.max_num_batched_tokens" not in rep["unavailable"]
    assert any("max_num_batched_tokens=8192" in ln for ln in rep["source_lines"])


def test_in_process_lane_16384_also_parses():
    """Same wording, the other card class — verbatim from a banked gen log."""
    rep = P.parse_text(
        "INFO 07-30 03:28:09 [scheduler.py:252] Chunked prefill is enabled "
        "with max_num_batched_tokens=16384.\n")
    assert rep["engine_config"]["max_num_batched_tokens"] == 16384
    # the sentence asserts the flag in its own words
    assert rep["engine_config"]["enable_chunked_prefill"] is True


def test_non_default_args_supplies_the_pinned_width_with_a_note():
    rep = P.parse_text(V024 + V024_LEVERS)
    assert rep["cli_args"]["max_num_seqs"] == 16
    assert rep["cli_args"]["max_model_len"] == 8192
    assert rep["cli_args"]["gpu_memory_utilization"] == 0.9
    # promoted into engine_config, and the provenance is recorded
    assert rep["engine_config"]["max_num_seqs"] == 16
    assert "engine_config.max_num_seqs" not in rep["unavailable"]
    assert any("non-default args" in n and "max_num_seqs" in n
               for n in rep["notes"])


def test_non_default_args_lift_is_whitelisted_not_a_dump():
    cli = P.parse_text(V024_LEVERS)["cli_args"]
    assert "model_tag" not in cli and "model" not in cli
    assert cli["dtype"] == "bfloat16"
    # `dtype` must not be matched inside `kv_cache_dtype`
    assert "kv_cache_dtype" not in cli


def test_structured_values_are_not_cut_at_their_first_inner_comma():
    """Unit test of the scanner, NOT a claim about vLLM's wording: we hold no
    banner with spec decode ON, so the string below is synthetic. It exists
    because a value truncated to `SpeculativeConfig(method='mtp'` would look
    like a complete reading and would not be one."""
    assert P._scan_value("x=Foo(a=1, b=[2, 3]), y=4", 2) == "Foo(a=1, b=[2, 3])"
    assert P._scan_value("x='a, b', y=4", 2) == "'a, b'"
    assert P._scan_value("x=1, y=2", 2) == "1"
    # unbalanced => None, so the caller falls back to cut-at-first-comma
    assert P._scan_value("x=Foo(a=1, b=2", 2) is None


def test_v024_gpu_util_falls_back_to_the_advisory_line():
    """0.24.0's engine banner omits gpu_memory_utilization; the CUDA-graph
    advisory quotes the value actually in force. Taking it must be RECORDED."""
    rep = P.parse_text(V024)
    assert rep["engine_config"]["gpu_memory_utilization"] == 0.9
    assert any("advisory" in n for n in rep["notes"])


def test_v019_wording_still_parses():
    rep = P.parse_text(V019)
    assert rep["parsed_ok"] is True
    assert rep["vllm_version"] == "0.19.1"
    assert rep["memory"]["model_load_gib"] == 7.52
    assert rep["memory"]["kv_cache_gib"] == 19.84
    assert rep["memory"]["cudagraph_actual_gib"] == 0.69
    assert rep["kv"]["tokens"] == 324976
    assert rep["kv"]["max_concurrency_x"] == 9.92
    assert rep["timing"]["init_engine_secs"] == 35.83
    # max_seq_len 32768, 324,976 KV tokens -> 9.917 full-length sequences
    assert rep["derived"]["full_length_seqs"] == pytest.approx(9.917, abs=1e-3)


def test_debug_level_yields_the_full_per_term_split():
    rep = P.parse_text(V08_DEBUG)
    assert rep["memory"]["weights_gib"] == 14.99
    assert rep["memory"]["non_torch_gib"] == 0.11
    assert rep["memory"]["activation_peak_gib"] == 1.20
    assert rep["memory"]["kv_cache_gib"] == 54.93
    assert rep["memory"]["total_gpu_gib"] == 79.15
    assert rep["engine_config"]["gpu_memory_utilization"] == 0.90
    assert rep["kv"]["gpu_blocks"] == 28123
    assert rep["kv"]["cpu_blocks"] == 2048
    # the whole point: with DEBUG on, the V10 section-7 overhead band closes.
    # This 0.8-era sentence has no CUDA-graph term, so the sum names its terms
    # and the absence is a note rather than a silent zero.
    assert rep["derived"]["overhead_gib"] == pytest.approx(1.31, abs=1e-6)
    assert rep["derived"]["overhead_terms"] == "non_torch + activation_peak"
    assert any("CUDA-graph term" in n for n in rep["notes"])
    assert "derived.overhead_gib" not in rep["unavailable"]
    assert rep["derived"]["budget_gib"] == pytest.approx(71.235, abs=1e-2)


def test_024_actual_usage_debug_summary():
    text = ("DEBUG 08-09 00:00:00 [gpu_worker.py:731] Actual usage is 51.75 GiB for "
            "weight, 1.02 GiB for peak activation, 0.94 GiB for non-torch, and "
            "2.37 GiB for CUDAGraph memory\n")
    rep = P.parse_text(text)
    assert rep["memory"]["weights_gib"] == 51.75
    assert rep["memory"]["activation_peak_gib"] == 1.02
    assert rep["memory"]["non_torch_gib"] == 0.94
    assert rep["memory"]["cudagraph_actual_gib"] == 2.37
    assert rep["derived"]["overhead_gib"] == pytest.approx(4.33, abs=1e-6)


def test_source_lines_are_verbatim_and_deduped():
    rep = P.parse_text(V024)
    assert rep["source_lines"], "raw provenance must travel with the numbers"
    assert len(rep["source_lines"]) == len(set(rep["source_lines"]))
    assert any("Available KV cache memory: 11.92 GiB" in ln
               for ln in rep["source_lines"])


# --------------------------------------------------------------------------- #
# No silent nulls
# --------------------------------------------------------------------------- #

def test_debug_only_terms_are_named_absences_not_silent_nulls():
    rep = P.parse_text(V024)
    for key in ("weights_gib", "non_torch_gib", "activation_peak_gib",
                "total_gpu_gib"):
        assert key in rep["memory"] and rep["memory"][key] is None
        assert "memory.%s" % key in rep["unavailable"]
    assert "DEBUG" in rep["unavailable"]["memory.weights_gib"]
    assert rep["derived"]["overhead_gib"] is None
    assert "derived.overhead_gib" in rep["unavailable"]


def test_model_load_stands_in_for_weights_with_a_note():
    rep = P.parse_text(V024)
    # 14.37 (load) + 11.92 (kv)
    assert rep["derived"]["weights_plus_kv_gib"] == pytest.approx(26.29, abs=1e-6)
    assert any("Model loading took" in n for n in rep["notes"])


# --------------------------------------------------------------------------- #
# Fail-soft / inert cases
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text", ["", "\n\n", "some unrelated wrapper output\n"])
def test_unrecognised_log_is_not_fatal(text):
    rep = P.parse_text(text)
    assert rep["parsed_ok"] is False
    assert "parse" in rep["unavailable"]
    assert rep["schema"] == "serve_memory/v1"


def test_unrecognised_but_memory_shaped_log_keeps_candidate_lines():
    text = ("INFO some future vLLM 9.9: reserved 12.5 GiB for the KV cache\n"
            "INFO unrelated line with no numbers\n")
    rep = P.parse_text(text)
    assert rep["parsed_ok"] is False
    assert any("12.5 GiB" in ln for ln in rep["candidate_lines"])
    assert "unrelated line" not in " ".join(rep["candidate_lines"])


def test_giant_progress_bar_lines_are_skipped_not_choked_on():
    text = V024 + "Capturing CUDA graphs: " + ("x" * 50000) + "\n"
    rep = P.parse_text(text)
    assert rep["parsed_ok"] is True
    assert all(len(ln) <= P.MAX_LINE_CHARS for ln in rep["source_lines"])


def test_the_line_ceiling_clears_a_real_engine_banner_by_a_wide_margin():
    """The longest engine-config banner measured over every banked log is 3,969
    characters and the ceiling used to be 4,000 — 31 characters from silently
    dropping `engine_config` on every serve. Keep the margin visible."""
    assert P.MAX_LINE_CHARS >= 4 * 3969
    long_banner = V024.replace(
        "pooler_config=None",
        "pooler_config=None, compilation_config={%s}" % ("'k': 'v', " * 500))
    assert len(long_banner.splitlines()[1]) > 4000
    rep = P.parse_text(long_banner)
    assert rep["engine_config"]["max_seq_len"] == 8192
    assert rep["vllm_version"] == "0.24.0"


def test_missing_log_paths_yield_a_named_absence(tmp_path):
    rep = P.build([str(tmp_path / "nope.log"), str(tmp_path / "also-nope.log")])
    assert rep["parsed_ok"] is False
    assert "log" in rep["unavailable"]
    assert [c["verdict"] for c in rep["log_candidates"]] == ["unreadable"] * 2


# --------------------------------------------------------------------------- #
# Candidate-log selection (content, not existence)
# --------------------------------------------------------------------------- #

def test_choose_log_skips_an_existing_wrapper_log_for_the_real_one(tmp_path):
    wrapper = _tmp_log(tmp_path, ">> job_serve: starting serve_vllm.sh\n",
                       "onstart.log")
    real = _tmp_log(tmp_path, V024, "vllm-0.log")
    path, text, tried = P.choose_log([wrapper, real])
    assert path == real
    assert "no vLLM startup marker" in tried[0]["verdict"]
    assert tried[1]["verdict"] == "selected"
    assert "Available KV cache memory" in text


def test_choose_log_falls_back_to_the_first_readable_candidate(tmp_path):
    a = _tmp_log(tmp_path, "wrapper only\n", "a.log")
    b = _tmp_log(tmp_path, "also wrapper\n", "b.log")
    path, _text, tried = P.choose_log([None, "", a, b])
    assert path == a
    assert "fallback" in tried[0]["verdict"]


def test_build_records_the_selection_trail(tmp_path):
    real = _tmp_log(tmp_path, V024)
    rep = P.build([str(tmp_path / "missing.log"), real])
    assert rep["log_path"] == real
    assert rep["memory"]["kv_cache_gib"] == 11.92
    assert [c["verdict"] for c in rep["log_candidates"]] == ["unreadable", "selected"]


def test_read_log_is_bounded(tmp_path):
    p = _tmp_log(tmp_path, "A" * 5000)
    assert len(P.read_log(p, max_bytes=100)) == 100


# --------------------------------------------------------------------------- #
# Shape fields + output placement
# --------------------------------------------------------------------------- #

def test_shape_fields_are_coerced():
    shape = P._parse_fields(["serve_id=serve-260809-1200-ab12", "tp=2", "dp=1",
                             "gpu_util=0.9", "quantization=", "lora=r1=sub/a",
                             "novalue"])
    assert shape["serve_id"] == "serve-260809-1200-ab12"
    assert shape["tp"] == 2
    assert shape["gpu_util"] == 0.9
    assert shape["quantization"] is None
    assert shape["lora"] == "r1=sub/a"        # only the FIRST '=' splits
    assert "novalue" not in shape


def test_default_out_path_lands_beside_the_log(tmp_path):
    log = _tmp_log(tmp_path, V024)
    assert P.default_out_path(log) == str(tmp_path / "serve_summary.json")
    assert P.default_out_path(None) is None


# --------------------------------------------------------------------------- #
# CLI — the shape the on-box shell actually invokes
# --------------------------------------------------------------------------- #

def _cli(*args):
    return subprocess.run([sys.executable, str(HERE / "parse_vllm_mem.py"), *args],
                          capture_output=True, text=True, timeout=120)


def test_cli_writes_serve_summary_beside_the_log(tmp_path):
    log = _tmp_log(tmp_path, V024)
    r = _cli("--log", log, "--field", "serve_id=serve-x", "--field", "tp=1")
    assert r.returncode == 0, r.stderr
    out = tmp_path / "serve_summary.json"
    assert r.stdout.strip() == str(out)
    doc = json.loads(out.read_text())
    assert doc["schema"] == "serve_memory/v1"
    assert doc["parsed_ok"] is True
    assert doc["memory"]["kv_cache_gib"] == 11.92
    assert doc["shape"] == {"serve_id": "serve-x", "tp": 1}
    assert doc["captured_at_utc"].endswith("Z")


def test_cli_is_inert_when_no_candidate_log_exists(tmp_path):
    """The shell-level dry test: wiring fires, nothing crashes, and the
    artifact says WHY it is empty rather than being absent."""
    r = _cli("--log", str(tmp_path / "absent.log"), "--out",
             str(tmp_path / "serve_summary.json"))
    assert r.returncode == 0, r.stderr
    doc = json.loads((tmp_path / "serve_summary.json").read_text())
    assert doc["parsed_ok"] is False
    assert "log" in doc["unavailable"]


def test_cli_with_no_log_flag_at_all_still_exits_zero():
    r = _cli("--out", "-")
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert doc["parsed_ok"] is False


def test_cli_stdout_json_when_out_is_dash(tmp_path):
    log = _tmp_log(tmp_path, V019)
    r = _cli("--log", log, "--out", "-", "--print")
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert doc["vllm_version"] == "0.19.1"
    assert "serve memory:" in r.stderr
    assert not (tmp_path / "serve_summary.json").exists()


def test_cli_unwritable_out_falls_back_to_stdout(tmp_path):
    log = _tmp_log(tmp_path, V024)
    r = _cli("--log", log, "--out", str(tmp_path / "serve.log" / "nested.json"))
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["parsed_ok"] is True
    assert "could not write" in r.stderr


# --------------------------------------------------------------------------- #
# Secret redaction — the banked-report leak class (2026-08-28)
# --------------------------------------------------------------------------- #

# Same real api_utils.py:273 wording as V024_LEVERS, with the two api_key
# spellings observed in banked logs: the quoted-hex serve bearer and the
# list-form token. Keys here are SYNTHETIC.
_FAKE_HEX = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
V_SECRET = (
    "(APIServer pid=284) INFO 08-27 06:18:06 [api_utils.py:273] non-default "
    "args: {'model_tag': '/workspace/base-model', 'api_key': '%s', "
    "'max_num_seqs': 16}\n" % _FAKE_HEX
)
V_SECRET_LIST = (
    "(APIServer pid=6503) INFO 08-17 16:31:17 [api_utils.py:273] non-default "
    "args: {'host': '0.0.0.0', 'api_key': ['w5wave-1786984050-local'], "
    "'max_num_seqs': 16}\n"
)


def test_secret_values_never_reach_the_report():
    """A serve bearer rode source_lines[0] into a banked serve_summary.json;
    this pins that no api_key VALUE survives anywhere in the report."""
    for fixture, token in ((V_SECRET, _FAKE_HEX),
                           (V_SECRET_LIST, "w5wave-1786984050-local")):
        rep = P.parse_text(V024 + fixture)
        blob = json.dumps(rep)
        assert token not in blob
        # the line itself still travels (rule 3), with the value redacted and
        # the key NAME kept so a reader can see the arg was pinned
        assert any("api_key" in ln and "[REDACTED]" in ln
                   for ln in rep["source_lines"])
        # the whitelist lift is untouched
        assert rep["cli_args"]["max_num_seqs"] == 16
        assert "api_key" not in rep["cli_args"]


def test_candidate_lines_are_redacted_too():
    """The parsed_ok:false diagnostic dump must not smuggle the value either."""
    line = "startup wrapper: memory guard armed, api_key=%s, 12 GiB free\n" % _FAKE_HEX
    rep = P.parse_text(line)
    assert rep["parsed_ok"] is False
    blob = json.dumps(rep)
    assert _FAKE_HEX not in blob
    assert any("[REDACTED]" in ln for ln in rep["candidate_lines"])


def test_redact_leaves_ordinary_lines_alone():
    assert P._redact("Available KV cache memory: 10.5 GiB") == \
        "Available KV cache memory: 10.5 GiB"
