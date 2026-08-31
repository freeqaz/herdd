#!/usr/bin/env python3
"""parse_vllm_mem.py — turn a vLLM server startup log into `serve_summary.json`.

WHY THIS EXISTS. Serve/eval boxes had no measured VRAM anchor. Training boxes
size off `vram_facts.py`; serve boxes were sized by hand-arithmetic over the
model config, and the one term nobody ever recorded was vLLM's own memory
profile — named in `docs/plans/witness/V10_SPOT_PROVISIONING_2026-08-08.md`
section 7 as "the bundle never captures vLLM's own memory profiler output",
which is why the overhead band there is quoted as an ESTIMATED 3-5 GiB.

This closes the SERVE half of that gap. The GEN half (an in-process `LLM()`,
where the per-term split can be read off the Worker by `collective_rpc`) is
already covered by `tools/witness/gen_probe_resumable.py`'s
`gates/gpu_memory_<arm>.json`, schema `gpu_memory/v1`. A `vllm serve` process
is a different shape: the numbers are only ever visible as log lines, so this
module parses them. The two artifacts are meant to be read side by side, and
the field names below match `gpu_memory/v1` wherever the same quantity exists.

WHAT A SERVE LOG ACTUALLY OFFERS (verified against real logs banked in this
repo — `out/modelzoo-reader-06-eval/eval/serve.log`,
`out/jobs/20260711T051057-waveb-bakeoff05-149d/out/bakeoff/serve_logs/`,
`out/jobs/20260713T025157-p2-reader-eval-01-ep1p0-f710/out/eval/serve.log`):

    Model loading took 7.52 GiB memory and 10.887501 seconds
    Estimated CUDA graph memory: 0.61 GiB total
    Available KV cache memory: 19.84 GiB
    GPU KV cache size: 324,976 tokens
    Maximum concurrency for 32,768 tokens per request: 9.92x
    Graph capturing finished in 12 secs, took 0.69 GiB
    CUDA graph pool memory: 0.69 GiB (actual), 0.61 GiB (estimated), ...
    init engine (profile, create kv cache, warmup model) took 35.83 seconds
    Chunked prefill is enabled with max_num_batched_tokens=8192.
    non-default args: {'model': '...', 'max_model_len': 8192, 'max_num_seqs': 16}

THE THROUGHPUT LEVERS (added 2026-08-09, cell E0 of
`docs/plans/witness/EVAL_THROUGHPUT_AUDIT_2026-08-09.md`). The audit asked
for `max_num_batched_tokens` and `speculative_config` on the grounds that both
were already in the engine-config banner. Checking that against every banked
serve/gen log under `out/` found it half true:

  * `speculative_config` IS in the banner (`speculative_config=None` on 0.24.0)
    and is now lifted;
  * `max_num_batched_tokens` is NOT, at any version we hold. It comes from the
    scheduler's own line above instead;
  * `max_num_seqs` was already whitelisted but is not in the banner EITHER, so
    that whitelist entry had never once fired. It is only ever printed in the
    API server's `non-default args` echo, i.e. only when somebody pinned it.

Which is why the CLI echo is parsed into its own `cli_args` section, and why an
absent width is a NAMED absence meaning "not pinned, engine default in force"
rather than a null meaning "unknown".

The finer weights / non-torch / activation-peak split is NOT in a default-level
serve log: vLLM 0.24.0 emits it at `logger.debug` (`v1/worker/gpu_worker.py`
:507 and :731). That is recorded as a NAMED absence in `unavailable`, never as
a silent null — the parser still matches the DEBUG wording (both the 0.8-era
"model weights take X GiB; non_torch_memory takes ..." sentence and the 0.24
"Actual usage is ..." summary) so a run launched with `VLLM_LOGGING_LEVEL=DEBUG`
gets the full split for free.

DESIGN RULES.
  1. FAIL-SOFT, ALWAYS. This runs on the serve path of a paid box. Every entry
     point catches `Exception`; an unrecognised log yields `parsed_ok: false`
     plus the candidate lines that looked memory-shaped, and exit status 0.
     It must never be the reason a serve dies.
  2. NO SILENT NULLS. A field we could not read is null AND named in
     `unavailable` with the reason (same rule as `gpu_memory/v1`).
  3. RAW LINES TRAVEL, MINUS SECRETS. Every line a number came from is kept
     in `source_lines` so a future reader can re-derive or dispute the parse
     without the box — but secret VALUES (api_key and friends) are redacted
     first: these reports get banked into run archives, and the CLI-echo line
     carries the serve bearer verbatim.
  4. STDLIB ONLY, and no import of anything else in this repo — it is staged
     to boxes standalone.

CLI:
    parse_vllm_mem.py --log /workspace/serve.log [--log ...] \\
        [--out serve_summary.json] [--field k=v]... [--nvidia-smi] [--print]

Several `--log` candidates may be given; the FIRST one that exists and looks
like a vLLM startup log wins (a serve box has several plausible log paths and
only one of them holds the engine's output). With no `--out`, the artifact is
written next to the chosen log as `serve_summary.json`, and its path is printed
on stdout so a shell caller can ship it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

#: Schema id. ADDITIVE-WITHIN-A-VERSION is the discipline here (same as the
#: training-telemetry precedent): new optional fields may appear inside
#: `serve_memory/v1` without a bump, because every field is optional by
#: construction and a reader that does not know a key simply does not read it.
#: Bump only on a BREAKING change — a key renamed, removed, or its meaning or
#: units changed under a reader's feet.
SCHEMA = "serve_memory/v1"

#: Cap on how much of a log we read. Serve logs carry CUDA-graph progress bars
#: (one enormous line) and then every request; the startup block is the first
#: few hundred lines. 8 MiB is far past any startup and bounds a runaway file.
MAX_READ_BYTES = 8 * 1024 * 1024

#: Per-line ceiling: above this a line is skipped unread, because CUDA-graph
#: progress bars arrive as one enormous line and regexing them is wasted work.
#: This was 4000 and that was nearly a silent failure: measured 2026-08-09 over
#: every banked serve/gen log, the LONGEST real engine-config banner is 3,969
#: characters — 31 characters of headroom. One more config field upstream and
#: the banner would have been skipped whole, taking `engine_config` with it,
#: silently, on every serve. Raised to 20,000, which still skips the progress
#: bars (tens of thousands of characters) by an order of magnitude.
MAX_LINE_CHARS = 20000

#: Lines that mark a file as "a vLLM startup log" for candidate selection.
_IS_VLLM_LOG = re.compile(
    r"LLM engine|vLLM API server version|Available KV cache memory|"
    r"GPU KV cache size|Model loading took|# GPU blocks|starting vLLM:",
)

#: Anything memory-shaped, for the `parsed_ok: false` diagnostic dump.
_MEMORY_SHAPED = re.compile(
    r"\b(GiB|GB|MiB)\b|KV cache|GPU blocks|memory|CUDA graph", re.IGNORECASE)

#: A printed number, thousands-separated or not, always ENDING in a digit — a
#: trailing-comma-greedy variant silently swallowed the separator in
#: "# GPU blocks: 28123, # CPU blocks: 2048" and dropped the second field.
_NUM = r"((?:[0-9]+,)*[0-9]+(?:\.[0-9]+)?)"

#: Secret values ride in vLLM's own CLI echo (`non-default args: {...
#: 'api_key': '<token>' ...}`) and would otherwise travel verbatim into
#: banked run artifacts via `source_lines`/`candidate_lines` — a live serve
#: bearer was found committed that way (2026-08-28). Redaction keeps the key
#: NAME so a reader can still see the arg was pinned; only the value goes.
_SECRET_RX = re.compile(
    r"((?:api[_-]?key|hf[_-]?token|auth[_-]?token|authorization|bearer)"
    r"['\"\]]*\s*[:=]\s*\[?['\"]?)([A-Za-z0-9][A-Za-z0-9_\-\.]{7,})",
    re.IGNORECASE)


def _redact(line):
    """Strip secret VALUES from a raw log line before it enters the report."""
    return _SECRET_RX.sub(lambda m: m.group(1) + "[REDACTED]", line)


def _f(text):
    """`"324,976"` -> 324976.0. Returns None on anything unparseable."""
    try:
        return float(str(text).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _i(text):
    v = _f(text)
    return None if v is None else int(v)


def _to_gib(value, unit):
    """Normalise a printed size to GiB. vLLM has used both `GiB` and `GB` for
    the same quantity across versions; `GB` there has always meant 2**30 bytes,
    so both map to GiB and the unit seen is recorded in `notes`."""
    v = _f(value)
    if v is None:
        return None
    unit = (unit or "GiB").strip()
    if unit in ("GiB", "GB"):
        return v
    if unit in ("MiB", "MB"):
        return v / 1024.0
    return None


# --------------------------------------------------------------------------- #
# Line patterns.
#
# Each entry is (key, compiled regex, handler). The handler receives the match
# and the in-progress report; it stores whatever it found. A pattern that does
# not match is simply absent — every field is optional by construction, which
# is what makes an unknown vLLM version degrade instead of crash.
# --------------------------------------------------------------------------- #

_PATTERNS = []


def _pattern(name, regex, flags=0):
    def register(fn):
        _PATTERNS.append((name, re.compile(regex, flags), fn))
        return fn
    return register


@_pattern("model_load", r"Model loading took %s\s*(GiB|GB)(?: memory)? and %s seconds"
                        % (_NUM, _NUM))
def _h_model_load(m, rep):
    rep["memory"]["model_load_gib"] = _to_gib(m.group(1), m.group(2))
    rep["timing"]["model_load_secs"] = _f(m.group(3))


@_pattern("weights_load_secs", r"Loading weights took %s seconds" % _NUM)
def _h_weights_load(m, rep):
    rep["timing"]["weights_load_secs"] = _f(m.group(1))


@_pattern("kv_cache", r"Available KV cache memory:\s*%s\s*(GiB|GB)" % _NUM)
def _h_kv(m, rep):
    rep["memory"]["kv_cache_gib"] = _to_gib(m.group(1), m.group(2))


@_pattern("kv_tokens", r"GPU KV cache size:\s*%s tokens" % _NUM)
def _h_kv_tokens(m, rep):
    rep["kv"]["tokens"] = _i(m.group(1))


@_pattern("gpu_blocks", r"#\s*GPU blocks:\s*%s(?:,\s*#\s*CPU blocks:\s*%s)?"
                        % (_NUM, _NUM))
def _h_blocks(m, rep):
    # V0-era wording, and still what the v10 question asked for by name.
    rep["kv"]["gpu_blocks"] = _i(m.group(1))
    if m.group(2) is not None:
        rep["kv"]["cpu_blocks"] = _i(m.group(2))


@_pattern("max_concurrency",
          r"Maximum concurrency for %s tokens per request:\s*%s\s*x" % (_NUM, _NUM))
def _h_concurrency(m, rep):
    rep["kv"]["tokens_per_request"] = _i(m.group(1))
    rep["kv"]["max_concurrency_x"] = _f(m.group(2))


@_pattern("cudagraph_estimate",
          r"Estimated CUDA graph memory:\s*%s\s*(GiB|GB)" % _NUM)
def _h_cg_est(m, rep):
    rep["memory"]["cudagraph_estimate_gib"] = _to_gib(m.group(1), m.group(2))


@_pattern("graph_capture",
          r"Graph capturing finished in %s secs?, took %s\s*(GiB|GB)" % (_NUM, _NUM))
def _h_graph_capture(m, rep):
    rep["timing"]["graph_capture_secs"] = _f(m.group(1))
    rep["memory"]["cudagraph_actual_gib"] = _to_gib(m.group(2), m.group(3))


@_pattern("cudagraph_pool",
          r"CUDA graph pool memory:\s*%s\s*(?:GiB|GB) \(actual\),\s*%s\s*(?:GiB|GB) "
          r"\(estimated\), difference:\s*%s\s*(?:GiB|GB) \(%s%%\)" % (_NUM, _NUM, _NUM, _NUM))
def _h_cg_pool(m, rep):
    rep["memory"]["cudagraph_actual_gib"] = _f(m.group(1))
    rep["memory"]["cudagraph_estimate_gib"] = _f(m.group(2))
    rep["memory"]["cudagraph_estimate_error_gib"] = _f(m.group(3))
    rep["memory"]["cudagraph_estimate_error_pct"] = _f(m.group(4))


@_pattern("init_engine",
          r"init engine \([^)]*\) took %s seconds" % _NUM)
def _h_init(m, rep):
    rep["timing"]["init_engine_secs"] = _f(m.group(1))


@_pattern("checkpoint_size",
          r"Filesystem type for checkpoints:\s*(\S+?)\.\s*Checkpoint size:\s*%s\s*(GiB|GB)\."
          r"\s*Available RAM:\s*%s\s*(GiB|GB)" % (_NUM, _NUM))
def _h_ckpt(m, rep):
    rep["host"]["checkpoint_fs"] = m.group(1)
    rep["host"]["checkpoint_gib"] = _to_gib(m.group(2), m.group(3))
    rep["host"]["ram_available_gib"] = _to_gib(m.group(4), m.group(5))


@_pattern("util_advisory",
          # 0.21+: "The current --gpu-memory-utilization=0.9000 is equivalent to ..."
          # 0.19 : "increase --gpu-memory-utilization from 0.9000 to 0.9196 ..."
          # DEBUG: "total_gpu_memory (79.15GiB) x gpu_memory_utilization (0.90) = ..."
          # In every wording the FIRST number is the value actually in force.
          r"(?:--gpu-memory-utilization(?:=|\s+from\s+)|gpu_memory_utilization\s*\()"
          r"([0-9.]+)")
def _h_util(m, rep):
    # Only a fallback: the engine-config banner is authoritative when it carries
    # the key at all (0.24.0's banner does not).
    rep["_util_from_advisory"] = _f(m.group(1))


@_pattern("engine_banner",
          r"Initializing a (V[01]) LLM engine \(v([^)]+)\) with config:(.*)$")
def _h_engine(m, rep):
    rep["engine"]["api_version"] = m.group(1)
    rep["vllm_version"] = m.group(2)
    values, seen = _parse_engine_config(m.group(3))
    rep["engine_config"].update(values)
    rep.setdefault("_cfg_seen", set()).update(seen)


@_pattern("api_server_version",
          r"vLLM API server version (\S+)")
def _h_api_version(m, rep):
    rep.setdefault("vllm_version", None)
    if not rep.get("vllm_version"):
        rep["vllm_version"] = m.group(1)


@_pattern("attention_backend",
          r"Using (FlashAttention version \d+|\S+ backend)")
def _h_attn(m, rep):
    rep["engine"]["attention"] = m.group(1)


# --- the two throughput levers the banner does NOT carry -------------------- #
#
# Verified 2026-08-09 against every banked serve/gen log under `out/` (vLLM
# 0.19.1 / 0.24.0, HTTP and in-process): `max_num_batched_tokens` and
# `max_num_seqs` appear NOWHERE in the "Initializing a V1 LLM engine ... with
# config:" line. Whitelisting them in `_CONFIG_KEYS` alone would therefore never
# fire. They ARE printed, but on two other lines:
#
#   INFO ... [scheduler.py:252] Chunked prefill is enabled with max_num_batched_tokens=8192.
#   INFO ... [api_utils.py:273] non-default args: {'model': '...', 'max_num_seqs': 16}
#
# The first is the RESOLVED value and is emitted whenever chunked prefill is on
# (which is vLLM's default for generate models). The second is the API server's
# echo of the CLI args that differ from the defaults — so it carries
# `max_num_seqs` if and only if somebody pinned it, which is exactly the
# distinction a reader of this artifact needs.

@_pattern("chunked_prefill_batch",
          r"Chunked prefill is enabled with max_num_batched_tokens=%s" % _NUM)
def _h_chunked_prefill_batch(m, rep):
    # The scheduler's own resolved number — authoritative, so plain assignment.
    rep["engine_config"]["max_num_batched_tokens"] = _i(m.group(1))
    # The sentence asserts the flag in its own wording; the banner still wins.
    rep["engine_config"].setdefault("enable_chunked_prefill", True)


@_pattern("non_default_args", r"non-default args:\s*\{(.*)\}\s*$")
def _h_non_default_args(m, rep):
    """The API server's dict of CLI args that differ from vLLM's defaults.

    Kept in its OWN section rather than merged into `engine_config`: these are
    what the launcher asked for, not what the engine resolved, and conflating
    the two is how "unset" starts looking like "set to the default"."""
    rep["cli_args"].update(_parse_non_default_args(m.group(1)))


# --- the DEBUG-only per-term split (both wordings) -------------------------- #

@_pattern("debug_split_sentence",
          r"model weights take %s\s*(GiB|GB)" % _NUM)
def _h_dbg_weights(m, rep):
    rep["memory"]["weights_gib"] = _to_gib(m.group(1), m.group(2))


@_pattern("debug_split_nontorch",
          r"non_torch_memory takes %s\s*(GiB|GB)" % _NUM)
def _h_dbg_nontorch(m, rep):
    rep["memory"]["non_torch_gib"] = _to_gib(m.group(1), m.group(2))


@_pattern("debug_split_activation",
          r"PyTorch activation peak memory takes %s\s*(GiB|GB)" % _NUM)
def _h_dbg_activation(m, rep):
    rep["memory"]["activation_peak_gib"] = _to_gib(m.group(1), m.group(2))


@_pattern("debug_split_kv",
          r"the rest of the memory reserved for KV Cache is %s\s*(GiB|GB)" % _NUM)
def _h_dbg_kv(m, rep):
    rep["memory"].setdefault("kv_cache_gib", _to_gib(m.group(1), m.group(2)))


@_pattern("debug_total_gpu",
          r"total_gpu_memory \(%s\s*(GiB|GB)\)" % _NUM)
def _h_dbg_total(m, rep):
    rep["memory"]["total_gpu_gib"] = _to_gib(m.group(1), m.group(2))


@_pattern("debug_actual_usage",
          r"Actual usage is %s\s*(?:GiB|GB) for weight, %s\s*(?:GiB|GB) for peak "
          r"activation, %s\s*(?:GiB|GB) for non-torch, and %s\s*(?:GiB|GB) for "
          r"CUDAGraph" % (_NUM, _NUM, _NUM, _NUM))
def _h_dbg_actual(m, rep):
    rep["memory"]["weights_gib"] = _f(m.group(1))
    rep["memory"]["activation_peak_gib"] = _f(m.group(2))
    rep["memory"]["non_torch_gib"] = _f(m.group(3))
    rep["memory"]["cudagraph_actual_gib"] = _f(m.group(4))


#: Scalars lifted out of the engine-config banner. Whitelisted rather than
#: parsed wholesale — that line is thousands of characters of nested dicts, and
#: an over-eager parse of it is how a "fail-soft" parser starts raising.
_CONFIG_KEYS = (
    "model", "dtype", "max_seq_len", "tensor_parallel_size",
    "pipeline_parallel_size", "data_parallel_size", "quantization",
    "kv_cache_dtype", "enforce_eager", "served_model_name",
    "enable_prefix_caching", "enable_chunked_prefill", "seed",
    "gpu_memory_utilization", "max_num_seqs", "trust_remote_code",
    # `speculative_config` IS in the banner (`speculative_config=None` in every
    # 0.24.0 log we hold) and was the one throughput lever printed there and
    # never lifted. `max_num_batched_tokens` is NOT in any banner we have seen —
    # it is listed anyway so a future vLLM that starts printing it is picked up
    # for free, but the value normally arrives via `_h_chunked_prefill_batch`.
    "speculative_config", "max_num_batched_tokens",
)

#: Whitelisted keys lifted out of the API server's `non-default args:` dict.
#: Deliberately narrow — this is the throughput/serving-shape set, not a dump.
_CLI_KEYS = (
    "max_num_seqs", "max_num_batched_tokens", "max_model_len",
    "gpu_memory_utilization", "enable_prefix_caching",
    "enable_chunked_prefill", "kv_cache_dtype", "quantization",
    "tensor_parallel_size", "data_parallel_size", "speculative_config",
    "enforce_eager", "dtype", "served_model_name",
)

_OPEN_BRACKETS, _CLOSE_BRACKETS = "([{", ")]}"


def _scan_value(text, start):
    """Slice `text[start:]` up to the first TOP-LEVEL `,` (or closing bracket).

    Nested `()`/`[]`/`{}` and quoted strings are stepped over, so a structured
    value survives whole instead of being cut at its first inner comma:
    `speculative_config=SpeculativeConfig(method='mtp', num_speculative_tokens=1)`
    truncated to `SpeculativeConfig(method='mtp'` would be a WORSE artifact than
    no field at all — it reads like a complete value and is not one. That shape
    is unwitnessed here (every banked banner says `speculative_config=None`),
    which is exactly why the scanner has to be right before the first MTP serve
    rather than after it.

    Returns None when the brackets or quotes never balance (a truncated line),
    so the caller can fall back to the old cut-at-first-comma behaviour instead
    of swallowing thousands of characters of banner.
    """
    depth, quote, i, n = 0, None, start, len(text)
    while i < n:
        ch = text[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in _OPEN_BRACKETS:
            depth += 1
        elif ch in _CLOSE_BRACKETS:
            if depth == 0:
                break          # a bracket closing an OUTER container: value ends
            depth -= 1
        elif ch == "," and depth == 0:
            break
        i += 1
    if depth != 0 or quote is not None:
        return None
    return text[start:i]


def _coerce(token):
    token = token.strip().strip(",").strip()
    if token.startswith(("'", '"')) and token.endswith(("'", '"')) and len(token) >= 2:
        return token[1:-1]
    if token in ("None", ""):
        return None
    if token == "True":
        return True
    if token == "False":
        return False
    num = _f(token)
    if num is not None and re.fullmatch(r"-?[0-9]+(\.[0-9]+)?", token):
        return int(num) if num.is_integer() and "." not in token else num
    return token


def _lift(tail, keys, assign):
    """Whitelisted `<key><assign><value>` lift out of one long log-line tail.

    Returns `(values, seen)`. `seen` is the set of keys that were PRESENT, which
    is not the same question as "is the value non-null": `speculative_config`
    is legitimately `None` when spec decode is off, and rule 2 (no silent nulls)
    needs to tell that apart from a banner we never parsed.
    """
    out, seen = {}, set()
    for key in keys:
        m = re.search(r"(?<![\w.])%s%s" % (re.escape(key), assign), tail)
        if not m:
            continue
        raw = _scan_value(tail, m.end())
        if raw is None:                       # unbalanced/truncated: old behaviour
            raw = re.match(r"[^,]*", tail[m.end():]).group(0)
        out[key] = _coerce(raw)
        seen.add(key)
    return out, seen


def _parse_engine_config(tail):
    """Whitelisted `key=value` lift out of the engine-config banner tail."""
    return _lift(tail, _CONFIG_KEYS, r"=")


def _parse_non_default_args(body):
    """Whitelisted `'key': value` lift out of the `non-default args:` dict."""
    return _lift(body, _CLI_KEYS, r"'\s*:\s*")[0]


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #

def _blank_report():
    return {
        "schema": SCHEMA,
        "source": "vllm_startup_log",
        "parsed_ok": False,
        "captured_at_utc": None,
        "vllm_version": None,
        "engine": {},
        "engine_config": {},
        "cli_args": {},
        "shape": {},
        "memory": {},
        "kv": {},
        "timing": {},
        "host": {},
        "gpus": None,
        "derived": {},
        "unavailable": {},
        "notes": [],
        "log_path": None,
        "log_candidates": [],
        "source_lines": [],
        "candidate_lines": [],
    }


#: Memory terms we always report on, so an absent one is a NAMED absence.
_EXPECTED_MEMORY = (
    ("model_load_gib", "vLLM never printed 'Model loading took N GiB memory'"),
    ("kv_cache_gib", "vLLM never printed 'Available KV cache memory: N GiB'"),
    ("cudagraph_actual_gib",
     "no 'Graph capturing finished ... took N GiB' / 'CUDA graph pool memory' "
     "line (expected when CUDA graphs are disabled, e.g. --enforce-eager)"),
    ("weights_gib",
     "vLLM 0.24.0 logs the weights/non-torch/activation split at logger.DEBUG "
     "(v1/worker/gpu_worker.py:507,731), so it is absent from a default-level "
     "serve log; relaunch with VLLM_LOGGING_LEVEL=DEBUG to capture it"),
    ("non_torch_gib", "DEBUG-only in vLLM 0.24.0 (see weights_gib)"),
    ("activation_peak_gib", "DEBUG-only in vLLM 0.24.0 (see weights_gib)"),
    ("total_gpu_gib",
     "DEBUG-only ('total_gpu_memory (N GiB) x gpu_memory_utilization'); the "
     "card's capacity is otherwise only visible via nvidia-smi"),
)

#: The throughput levers we always report on, so an absent one is a NAMED
#: absence with the reason it is absent — which for these three is the whole
#: point of the field: "we could not read it" and "the engine default is in
#: force" are different facts and a bare null cannot tell them apart.
_EXPECTED_CONFIG = (
    ("max_num_batched_tokens",
     "not in vLLM's engine-config banner at any version we hold; it is printed "
     "only by 'Chunked prefill is enabled with max_num_batched_tokens=N.', "
     "which is absent when chunked prefill is off — and it is NOT echoed in "
     "'non-default args' unless it was set on the CLI"),
    ("max_num_seqs",
     "vLLM never prints the RESOLVED max_num_seqs; it appears only in the API "
     "server's 'non-default args' dict, and only when it was set explicitly. "
     "Absent here therefore means NOT PINNED — the engine's card-dependent "
     "default is in force (at 0.26.0: 256 on <70 GiB, 1024 on >=70 GiB)"),
    ("speculative_config",
     "no engine-config banner line was parsed (truncated or pre-engine-init "
     "log); note that a PARSED banner with spec decode off records the value "
     "as null WITHOUT this entry"),
)


def parse_text(text, shape=None, gpus=None, max_candidates=40, now=None):
    """Parse a vLLM startup log body. Never raises on bad input."""
    rep = _blank_report()
    rep["captured_at_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(now if now is not None else time.time()))
    rep["shape"] = dict(shape or {})
    if gpus:
        rep["gpus"] = gpus

    seen = []
    try:
        for line in (text or "").splitlines():
            # CUDA-graph progress bars are one enormous line; regex over them is
            # wasted work and they never carry a number we want.
            if len(line) > MAX_LINE_CHARS:
                continue
            for _name, rx, handler in _PATTERNS:
                m = rx.search(line)
                if not m:
                    continue
                try:
                    handler(m, rep)
                except Exception as exc:                       # pragma: no cover
                    rep["notes"].append("handler %s failed: %r" % (_name, exc))
                    continue
                red = _redact(line)
                if red not in seen:
                    seen.append(red)
    except Exception as exc:                                   # pragma: no cover
        rep["notes"].append("scan aborted: %r" % (exc,))

    rep["source_lines"] = seen

    util_fallback = rep.pop("_util_from_advisory", None)
    if rep["engine_config"].get("gpu_memory_utilization") is None and util_fallback is not None:
        rep["engine_config"]["gpu_memory_utilization"] = util_fallback
        rep["notes"].append(
            "gpu_memory_utilization read from the CUDA-graph advisory line, not "
            "the engine-config banner")

    # Same fallback shape as the advisory line above: the banner cannot answer
    # these two, so the CLI echo does — and the substitution is RECORDED, never
    # silent, because "explicitly pinned" and "engine default" are the exact
    # distinction the width levers turn on.
    cfg_seen = rep.pop("_cfg_seen", set())
    for key in ("max_num_seqs", "max_num_batched_tokens"):
        if rep["engine_config"].get(key) is None and rep["cli_args"].get(key) is not None:
            rep["engine_config"][key] = rep["cli_args"][key]
            rep["notes"].append(
                "%s read from the API server's 'non-default args' line, not the "
                "engine-config banner (vLLM does not print the resolved value); "
                "its presence there means it was PINNED on the CLI" % key)

    rep["parsed_ok"] = bool(rep["memory"] or rep["kv"])
    if not rep["parsed_ok"]:
        rep["unavailable"]["parse"] = (
            "no recognised vLLM memory-profile line — the log may be truncated, "
            "pre-engine-init, or from a vLLM whose wording this parser predates")
        cands = []
        for line in (text or "").splitlines():
            if len(line) > MAX_LINE_CHARS:
                continue
            if _MEMORY_SHAPED.search(line):
                cands.append(_redact(line))
                if len(cands) >= max_candidates:
                    break
        rep["candidate_lines"] = cands

    for key, why in _EXPECTED_MEMORY:
        if rep["memory"].get(key) is None:
            rep["memory"].setdefault(key, None)
            rep["unavailable"]["memory.%s" % key] = why

    for key, why in _EXPECTED_CONFIG:
        if key in cfg_seen or rep["engine_config"].get(key) is not None:
            continue
        rep["engine_config"].setdefault(key, None)
        rep["unavailable"]["engine_config.%s" % key] = why

    _derive(rep)
    return rep


def _derive(rep):
    """Cheap arithmetic over what we did read. Every term is None-guarded: a
    derived number is only ever as present as its inputs."""
    mem, kv, cfg = rep["memory"], rep["kv"], rep["engine_config"]

    weights = mem.get("weights_gib")
    if weights is None:
        weights = mem.get("model_load_gib")
        if weights is not None:
            rep["notes"].append(
                "weights term taken from 'Model loading took N GiB' (the torch "
                "allocation delta across load), the INFO-level stand-in for the "
                "DEBUG-only weights_gib")
    kvg = mem.get("kv_cache_gib")

    # THE V10 section-7 band. `non_torch` and `activation_peak` are mandatory
    # (both DEBUG-only at 0.24.0 — which is exactly why the band was never
    # measured). `cudagraph_actual` is summed WHEN PRESENT and named when not:
    # the 0.8-era profile sentence has no CUDA-graph term at all, and folding a
    # missing term in as zero would understate the overhead silently.
    base_parts = {"non_torch": mem.get("non_torch_gib"),
                  "activation_peak": mem.get("activation_peak_gib")}
    if any(v is None for v in base_parts.values()):
        rep["derived"]["overhead_gib"] = None
        rep["derived"]["overhead_terms"] = None
        rep["unavailable"]["derived.overhead_gib"] = (
            "needs non_torch + activation_peak (+ cudagraph); the first two are "
            "DEBUG-only at vLLM 0.24.0, so a default-level serve log cannot "
            "close the V10 section-7 band from the log alone")
    else:
        terms = dict(base_parts)
        if mem.get("cudagraph_actual_gib") is not None:
            terms["cudagraph"] = mem["cudagraph_actual_gib"]
        else:
            rep["notes"].append(
                "overhead excludes a CUDA-graph term: the log carried no "
                "'Graph capturing finished'/'CUDA graph pool memory' line")
        rep["derived"]["overhead_gib"] = round(sum(terms.values()), 4)
        rep["derived"]["overhead_terms"] = " + ".join(terms)

    if weights is not None and kvg is not None:
        rep["derived"]["weights_plus_kv_gib"] = round(weights + kvg, 4)
    else:
        rep["derived"]["weights_plus_kv_gib"] = None

    total = mem.get("total_gpu_gib")
    util = cfg.get("gpu_memory_utilization")
    if total is not None and util is not None:
        rep["derived"]["budget_gib"] = round(total * float(util), 4)
    else:
        rep["derived"]["budget_gib"] = None

    tokens, max_len = kv.get("tokens"), cfg.get("max_seq_len")
    if tokens and max_len:
        try:
            rep["derived"]["full_length_seqs"] = round(tokens / float(max_len), 3)
        except Exception:                                      # pragma: no cover
            rep["derived"]["full_length_seqs"] = None
    else:
        rep["derived"]["full_length_seqs"] = None


# --------------------------------------------------------------------------- #
# Log selection + IO
# --------------------------------------------------------------------------- #

def read_log(path, max_bytes=MAX_READ_BYTES):
    """Read at most `max_bytes` of `path`, tolerating binary junk. Returns None
    when the path is unreadable — never raises."""
    try:
        with open(path, "rb") as fh:
            data = fh.read(max_bytes)
    except OSError:
        return None
    return data.decode("utf-8", "replace")


def choose_log(candidates, max_bytes=MAX_READ_BYTES):
    """First candidate that exists AND looks like a vLLM startup log.

    A serve box has several plausible log paths at once — the job lane's
    `$SERVE_LOG`, the boot-pull lane's tee'd `onstart.log`, the HAProxy lane's
    per-replica `vllm-N.log` — and only one of them holds the engine's output.
    Picking "the first that exists" would happily settle on a wrapper log with
    no memory lines in it, so existence is not the test: content is.

    Returns `(path, text, tried)` where `tried` records the verdict per
    candidate. Falls back to the first READABLE candidate (so an unrecognised
    vLLM still yields a `parsed_ok: false` artifact with candidate lines
    attached) and finally to `(None, None, tried)`.
    """
    tried, fallback = [], None
    for path in candidates:
        if not path:
            continue
        text = read_log(path, max_bytes)
        if text is None:
            tried.append({"path": path, "verdict": "unreadable"})
            continue
        if _IS_VLLM_LOG.search(text):
            tried.append({"path": path, "verdict": "selected"})
            return path, text, tried
        tried.append({"path": path, "verdict": "no vLLM startup marker"})
        if fallback is None:
            fallback = (path, text)
    if fallback is not None:
        for row in tried:
            if row["path"] == fallback[0]:
                row["verdict"] = "selected (fallback: no candidate had a marker)"
                break
        return fallback[0], fallback[1], tried
    return None, None, tried


def probe_gpus():
    """`nvidia-smi` GPU names + memory, or None. Fail-soft by contract: no GPU,
    no nvidia-smi, and a hung nvidia-smi all yield None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    gpus = []
    for line in (out.stdout or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        row = {"name": parts[0]}
        if len(parts) > 1:
            mib = _f(parts[1])
            row["memory_total_mib"] = None if mib is None else int(mib)
            row["memory_total_gib"] = None if mib is None else round(mib / 1024.0, 2)
        gpus.append(row)
    return gpus or None


def default_out_path(log_path):
    """`serve_summary.json` beside the log it was parsed from.

    Beside-the-log is what makes this ride the EXISTING results plumbing: the
    jobs lane points `SERVE_LOG` inside the job's synced output dir, so the
    summary lands in the same place the serve log already comes back from.
    """
    if not log_path:
        return None
    return os.path.join(os.path.dirname(os.path.abspath(log_path)),
                        "serve_summary.json")


def _parse_fields(pairs):
    shape = {}
    for item in pairs or []:
        key, sep, value = item.partition("=")
        if not sep:
            continue
        key = key.strip()
        if key:
            shape[key] = _coerce(value) if value.strip() else None
    return shape


def build(log_candidates, shape=None, gpus=None, max_bytes=MAX_READ_BYTES, now=None):
    """Choose a log, parse it, and stamp the selection trail onto the report."""
    path, text, tried = choose_log(log_candidates, max_bytes)
    rep = parse_text(text or "", shape=shape, gpus=gpus, now=now)
    rep["log_path"] = path
    rep["log_candidates"] = tried
    if path is None:
        rep["unavailable"]["log"] = (
            "none of the candidate serve-log paths were readable: %s"
            % ", ".join(str(c) for c in log_candidates if c))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Parse a vLLM serve startup log into serve_summary.json")
    ap.add_argument("--log", action="append", default=[], metavar="PATH",
                    help="candidate serve-log path (repeatable; first with a "
                         "vLLM startup marker wins)")
    ap.add_argument("--out", metavar="PATH",
                    help="output JSON path (default: serve_summary.json beside "
                         "the chosen log; '-' writes to stdout)")
    ap.add_argument("--field", action="append", default=[], metavar="K=V",
                    help="serve-shape key/value to record (repeatable)")
    ap.add_argument("--nvidia-smi", action="store_true",
                    help="also record GPU names/VRAM via nvidia-smi (fail-soft)")
    ap.add_argument("--print", dest="do_print", action="store_true",
                    help="print a one-line human summary to stderr")
    ap.add_argument("--max-bytes", type=int, default=MAX_READ_BYTES,
                    help="max bytes read per candidate log (default %(default)s)")
    args = ap.parse_args(argv)

    try:
        rep = build(args.log or [],
                    shape=_parse_fields(args.field),
                    gpus=probe_gpus() if args.nvidia_smi else None,
                    max_bytes=args.max_bytes)
    except Exception as exc:                                   # pragma: no cover
        # Rule 1: never be the reason a serve dies.
        rep = _blank_report()
        rep["unavailable"]["parse"] = "parser raised: %r" % (exc,)

    blob = json.dumps(rep, indent=1, sort_keys=False, default=str)

    out = args.out or default_out_path(rep.get("log_path"))
    if out == "-" or out is None:
        sys.stdout.write(blob + "\n")
    else:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out, "w") as fh:
                fh.write(blob + "\n")
            print(out)
        except OSError as exc:
            print("!! parse_vllm_mem: could not write %s: %s" % (out, exc),
                  file=sys.stderr)
            sys.stdout.write(blob + "\n")

    if args.do_print:
        mem, kv = rep["memory"], rep["kv"]
        print(">> serve memory: weights/load=%s GiB kv=%s GiB kv_tokens=%s "
              "cudagraph=%s GiB (vllm %s, parsed_ok=%s)"
              % (mem.get("model_load_gib"), mem.get("kv_cache_gib"),
                 kv.get("tokens"), mem.get("cudagraph_actual_gib"),
                 rep.get("vllm_version"), rep.get("parsed_ok")), file=sys.stderr)
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    sys.exit(main())
