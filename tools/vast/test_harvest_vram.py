"""Tests for harvest_vram.py — summaries -> committed VRAM anchors.

Two things are pinned here that are cheap to regress and expensive to notice:

* ABSENT MEANS ABSENT. Every summary on disk when the schema-2 telemetry blocks
  landed is v1 and carries none of them. A harvester that filled missing blocks
  with zeros or nulls would hand the knob report a table full of runs that
  "measured" 0 tokens/s, and the report would dutifully print deltas against
  them.
* NO ABSOLUTE MACHINE PATH reaches `vram_facts.json`. It is committed
  (CLAUDE.md), and the new telemetry carries free-form provenance strings —
  `token_stats.source` is an on-box path on a local run.

CPU-only, stdlib + pytest. Run: pytest tools/vast/test_harvest_vram.py
"""
from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import harvest_vram as hv  # noqa: E402

V1_SUMMARY = {
    "base": "/home/someone/assets/qwen35-9b",
    "quant_mode": "bf16", "max_seq": 12288, "batch": 1, "grad_accum": 4,
    "world_size": 1, "grad_checkpointing": True, "ce_chunk_matmul": "fp32",
    "target_modules": ["q_proj", "k_proj"], "lora_r": 32, "packing": "off",
    "n_rows": 61, "global_steps": 3, "step_time_seconds": 20.9,
    "peak_vram_alloc_gb": 26.34, "peak_vram_reserved_gb": 27.1,
    "out": "/home/someone/out/jobs/x", "data": "/home/someone/data/x.jsonl",
}

V2_BLOCKS = {
    "summary_schema": 2,
    "hardware": {
        "gpu_names": ["NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 5090"],
        "gpu_total_mem_gb": [31.4, 31.4], "driver_version": "580.65.06",
        "cuda_version": "12.8", "torch_version": "2.8.0+cu128",
        "transformers_version": "4.55.0", "trl_version": "0.21.0",
        "peft_version": "0.17.0", "flash_attn_version": None,
        "bitsandbytes_version": "0.47.0",
        "pytorch_cuda_alloc_conf": "expandable_segments:True",
        "platform": "Linux-6.16.0-arch1-1-x86_64-with-glibc2.42",
    },
    "token_stats": {
        "row_tokens_max": 5120, "row_tokens_p99": 4900, "row_tokens_p95": 4100,
        "row_tokens_p50": 1800, "row_tokens_mean": 2100.5, "n_rows": 61,
        "n_rows_at_max_seq": 0,
        "source": "/home/someone/out/jobs/x/runset/train.jsonl",
    },
    "throughput": {
        "tokens_seen": 128000, "tokens_per_second": 1450.5,
        "samples_per_second": 0.7, "step_time_p50_s": 20.4,
        "step_time_p95_s": 24.1, "step_time_max_s": 31.0, "n_steps_timed": 3,
        "source": "trainer_state", "scope": "post-warmup",
    },
    "gpu_power": {
        "interval_s": 5, "n_samples": 240,
        "per_gpu": [
            {"index": 0, "power_mean_w": 420.0, "power_p95_w": 510.0,
             "power_max_w": 545.0, "power_limit_w": 600.0,
             "util_mean_pct": 99.0, "mem_used_max_gb": 28.9},
            {"index": 1, "power_mean_w": 400.0, "power_p95_w": 495.0,
             "power_max_w": 520.0, "power_limit_w": 600.0,
             "util_mean_pct": 97.0, "mem_used_max_gb": 27.5},
        ],
    },
    "alloc_stats": {"per_gpu": [{"num_alloc_retries": 3, "num_ooms": 0},
                                {"num_alloc_retries": 1, "num_ooms": 0}]},
}


def _v2_summary(**over):
    s = dict(V1_SUMMARY)
    s.update(V2_BLOCKS)
    s.update(over)
    return s


# --- absent means absent ------------------------------------------------------

def test_v1_summary_yields_no_telemetry_at_all():
    """The common case, and the one that must not invent anything: 167 of 167
    harvested summaries are v1."""
    assert hv.telemetry_of(V1_SUMMARY) == {}


def test_a_missing_block_leaves_the_key_absent_not_null():
    """A null would read downstream as "measured, and it was nothing"."""
    s = _v2_summary()
    del s["throughput"]
    del s["gpu_power"]
    t = hv.telemetry_of(s)
    assert "throughput" not in t and "gpu_power" not in t
    assert "token_stats" in t and "hardware" in t


def test_only_allow_listed_keys_survive():
    """The anchor file is committed and reviewed; a summary is free to grow
    fields, the anchor is not free to grow unreviewed ones."""
    s = _v2_summary()
    s["token_stats"] = dict(s["token_stats"], secret_token="hunter2")
    s["hardware"] = dict(s["hardware"], hostname="box-46193810")
    t = hv.telemetry_of(s)
    assert "secret_token" not in t["token_stats"]
    assert "hostname" not in t["hardware"]
    assert t["token_stats"]["row_tokens_max"] == 5120


def test_partial_block_copies_only_what_is_there():
    s = _v2_summary(token_stats={"row_tokens_max": 900})
    assert hv.telemetry_of(s)["token_stats"] == {"row_tokens_max": 900}


# --- aggregation --------------------------------------------------------------

def test_gpu_power_is_aggregated_not_copied_per_card():
    t = hv.telemetry_of(_v2_summary())["gpu_power"]
    assert t["n_gpus"] == 2 and t["n_samples"] == 240
    assert t["power_mean_w"] == 410.0          # mean across cards
    assert t["power_max_w"] == 545.0           # worst card
    assert t["power_limit_w"] == 600.0
    assert t["power_mean_pct_of_limit"] == 68.3
    assert t["mem_used_max_gb"] == 28.9
    assert "per_gpu" not in t


def test_util_is_kept_but_power_is_the_saturation_signal():
    """util reads ~100% for anything resident on the card (measured: +36% power
    at a flat util 100%), so both are kept and the percentage-of-limit is the
    one derived for the reader."""
    t = hv.telemetry_of(_v2_summary())["gpu_power"]
    assert t["util_mean_pct"] == 98.0
    assert t["power_mean_pct_of_limit"] < 100.0


def test_alloc_retries_are_summed_across_cards():
    t = hv.telemetry_of(_v2_summary())["alloc_stats"]
    assert t == {"num_alloc_retries": 4, "num_ooms": 0, "n_gpus": 2}


def test_empty_power_block_produces_no_key():
    s = _v2_summary(gpu_power={"interval_s": 5, "per_gpu": []})
    assert "gpu_power" not in hv.telemetry_of(s)


# --- redaction ----------------------------------------------------------------

def test_path_like_telemetry_fields_are_redacted():
    t = hv.telemetry_of(_v2_summary())
    assert t["token_stats"]["source"] == os.path.join("runset", "train.jsonl")
    assert "/home/" not in json.dumps(t)


def test_non_path_strings_pass_through_untouched():
    """The scrub is applied to whole blocks, so it must be a no-op on the
    version strings and alloc-conf values that make up most of `hardware`."""
    hw = hv.telemetry_of(_v2_summary())["hardware"]
    assert hw["torch_version"] == "2.8.0+cu128"
    assert hw["pytorch_cuda_alloc_conf"] == "expandable_segments:True"
    assert hw["platform"].startswith("Linux-")
    assert hw["gpu_names"][0] == "NVIDIA GeForce RTX 5090"
    assert hw["flash_attn_version"] is None    # present-and-null is a fact


def test_scrub_reaches_nested_lists_and_dicts():
    assert hv._scrub_paths({"a": ["/home/u/x/y.jsonl", "plain"],
                            "b": {"c": "~/checkpoints/run/adapter"}}) == {
        "a": [os.path.join("x", "y.jsonl"), "plain"],
        "b": {"c": os.path.join("run", "adapter")}}


# --- end to end ---------------------------------------------------------------

def _write_run(root, name, summary):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "train_summary.json"), "w") as fh:
        json.dump(summary, fh)
    return d


def test_collect_attaches_telemetry_only_to_the_v2_run(tmp_path):
    root = str(tmp_path)
    _write_run(root, "run-v1", dict(V1_SUMMARY, n_rows=7))
    _write_run(root, "run-v2", _v2_summary(n_rows=8))
    anchors, stats = hv.collect([("t", root)])
    by_run = {a["run"]: a for a in anchors}
    assert stats["files"] == 2 and stats["telemetry"] == 1
    assert "telemetry" not in by_run["run-v1"]
    tel = by_run["run-v2"]["telemetry"]
    assert tel["schema"] == 2
    assert tel["throughput"]["tokens_per_second"] == 1450.5
    assert tel["token_stats"]["row_tokens_max"] == 5120


def test_collect_still_dedupes_by_content_with_telemetry_present(tmp_path):
    """A finished run is copied into archive/runs/, so the same summary is seen
    twice. Adding blocks to the summary must not defeat that — both copies hash
    the same, and the second only adds a source."""
    root = str(tmp_path)
    s = _v2_summary()
    _write_run(root, "live", s)
    _write_run(root, "archived", dict(s, out="/somewhere/else",
                                      data="/elsewhere/x.jsonl"))
    anchors, stats = hv.collect([("t", root)])
    assert len(anchors) == 1 and stats["duplicates"] == 1
    assert len(anchors[0]["sources"]) == 2


def test_the_committed_facts_file_carries_no_absolute_path():
    """The rule that made `_redact_path` exist, asserted against the artifact
    rather than the helper."""
    with open(hv.os.path.join(HERE, "vram_facts.json")) as fh:
        blob = fh.read()
    for needle in ("/home/", "/Users/", "/root/", os.path.expanduser("~")):
        assert needle not in blob, f"absolute path {needle!r} in vram_facts.json"


def test_build_reports_the_telemetry_census(tmp_path):
    root = str(tmp_path)
    _write_run(root, "run-v2", _v2_summary())
    doc = hv.build([("t", root)])
    assert doc["stats"]["telemetry"] == 1
    assert doc["stats"]["anchors"] == 1


# --- the B2 published-checkpoints source --------------------------------------

def test_sync_b2_pulls_only_summaries_and_configs(tmp_path):
    """A rented-box run's `train_summary.json` exists only under
    b2:<bucket>/checkpoints/, so the harvest has to reach it. The rclone
    invocation is asserted, not the network."""
    calls = []

    def runner(args):
        calls.append(args)
        return 0, "", ""

    st = hv.sync_b2(dest=str(tmp_path), runner=runner)
    assert st["rc"] == 0 and st["dest"] == str(tmp_path)
    argv = calls[0]
    assert argv[0] == "copy"
    assert argv[1] == f"{hv.B2_REMOTE}:{hv.B2_BUCKET}/{hv.B2_PREFIX}"
    assert "--include" in argv and "**/train_summary.json" in argv
    # read-only against B2: never sync, never delete
    assert not {"sync", "delete", "purge", "rcat", "move"} & set(argv)


def test_sync_b2_reports_a_failed_pull_rather_than_an_empty_success(tmp_path):
    st = hv.sync_b2(dest=str(tmp_path), runner=lambda a: (1, "", "no such remote"))
    assert st["rc"] == 1 and "no such remote" in st["error"]
    assert st["summaries"] == 0


def test_sync_b2_prunes_adapter_configs_with_no_summary_beside_them(tmp_path):
    """`adapter_config.json` is pulled for the base-name recovery, and every
    `checkpoint-NN/` has one. Only the ones next to a summary are of any use."""
    root = str(tmp_path)
    _write_run(root, "published", dict(V1_SUMMARY))
    for sub in ("published", os.path.join("published", "checkpoint-20")):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
        with open(os.path.join(root, sub, "adapter_config.json"), "w") as fh:
            json.dump({"base_model_name_or_path": "x"}, fh)
    st = hv.sync_b2(dest=root, runner=lambda a: (0, "", ""))
    assert st["pruned"] == 1 and st["summaries"] == 1
    assert os.path.isfile(os.path.join(root, "published", "adapter_config.json"))
    assert not os.path.isfile(
        os.path.join(root, "published", "checkpoint-20", "adapter_config.json"))


def test_the_b2_mirror_is_a_default_root():
    """Synced once, read every harvest — otherwise the next plain `--write`
    silently drops the anchors the sync was run for."""
    assert any(label == "b2-published" and path == hv.b2_cache_dir()
               for label, path in hv.default_roots())


def test_the_partial_gc_fraction_is_kept_as_a_shape_dimension():
    """`grad_checkpointing: true` with `grad_checkpointing_flag: 0.0` is a
    partial-GC cell that measures ~4x the memory of the same declared shape at
    1.0. Dropping the flag put both in one group."""
    assert "grad_checkpointing_flag" in hv.SHAPE_KEYS
