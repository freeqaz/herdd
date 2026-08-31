#!/usr/bin/env python3
"""harvest_vram.py — turn past runs' train_summary.json into VRAM anchors.

The trainer has always recorded what a run actually cost (`peak_vram_alloc_gb`,
`peak_vram_reserved_gb`, and the per-GPU lists) next to every shape dimension
that could explain it. Nothing read them back, so every bundle's
`needs.gpu_ram_gb` stayed a hand-authored guess while the answer sat on disk.
This walks those summaries and writes `vram_facts.json`, which
`vram_facts.py` then answers sizing questions from.

    python3 tools/vast/harvest_vram.py --write
    python3 tools/vast/harvest_vram.py --root ~/checkpoints --print

TWO THINGS THIS MUST GET RIGHT, both learned from the data:

1. DEDUPE BY CONTENT, NOT PATH. A finished run is copied from `out/jobs/<RUN>/`
   into `upstream-bench/archive/runs/<name>/`, so a naive walk counts every run
   twice and reports an `n` that is pure double-vision. The dedupe key is the
   summary JSON minus the two path-valued keys (`out`, `data`) — everything
   else, down to wall_seconds and the loss series, is identical for a copy and
   different for a rerun.

2. RECORD THE CORPUS AND THE TRAINER, NOT JUST THE SHAPE. Peak VRAM is set by
   the LONGEST ROW ACTUALLY PROCESSED, not by `--max-seq`
   (`FITTING_9B_ON_A_5090_2026-08-06.md` §8.2). Two runs with byte-identical
   declared shape measured 20.77 and 24.56 GB — a 3.8 GB spread explained by
   neither seq, batch, quant, target_modules, lora_r, packing, liger, nor
   attention impl. They differ by bundle and date (`perf-levers`, 2026-08-05 vs
   `fit-ladder`, 2026-08-06/07), i.e. by which rows survived the trainer's
   drop policy. So the anchor carries `dataset_content_sha256`, `n_rows` and
   its source run, and the estimator takes a group's MAX rather than its mean.

3. INGEST THE TELEMETRY BLOCKS WHEN THEY EXIST, FABRICATE NOTHING WHEN THEY DO
   NOT. `train_summary.json` grew a `summary_schema: 2` with `hardware`,
   `token_stats`, `throughput`, `gpu_power` and `alloc_stats` blocks; every one
   of the 167 summaries already on disk is v1 and carries none of them. So each
   block is copied only if present, key by key, and an absent block leaves the
   anchor's key ABSENT rather than null — a null would be indistinguishable
   downstream from "measured and zero", which is how a knob report starts
   quoting throughput deltas that were never measured.

NO ABSOLUTE PATHS land in the output: sources are recorded as (root-label,
relative path) so the file is committable and reads the same on any box. That
rule extends to every ingested telemetry field — `token_stats.source` and
`throughput.source` are on-box paths on a local run — via `_scrub_paths`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def _main_checkout() -> str:
    """The primary worktree. Run output lands in the MAIN checkout's out/jobs;
    harvesting from inside a worktree would otherwise walk an empty dir and
    report a confident zero."""
    import subprocess
    try:
        common = subprocess.run(
            ["git", "-C", REPO, "rev-parse", "--path-format=absolute",
             "--git-common-dir"], capture_output=True, text=True, timeout=10)
        if common.returncode == 0 and common.stdout.strip():
            return os.path.dirname(common.stdout.strip().rstrip("/"))
    except Exception:
        pass
    return REPO


def _bench_dir() -> str:
    """upstream-bench, via the repo's ONE resolver (scripts/grind/bench_paths.py)
    rather than a second hard-coded `../upstream-bench` guess."""
    for cand in (os.path.join(REPO, "scripts", "grind"),
                 os.path.join(_main_checkout(), "scripts", "grind")):
        if os.path.isdir(cand):
            sys.path.insert(0, cand)
            try:
                import bench_paths  # noqa
                return str(bench_paths.bench_dir())
            except Exception:
                pass
            finally:
                sys.path.remove(cand)
    return os.path.join(os.path.dirname(_main_checkout()), "upstream-bench")


# --- B2 published checkpoints -------------------------------------------------
# A run that trained on a RENTED box leaves no `train_summary.json` on this
# machine: `archive/runs/<name>/` keeps the gates and the manifest, the publish
# stage puts the summary at b2:<bucket>/checkpoints/<RUN_NAME>/, and the three
# roots above walk local disk only. That is how the sole 27B measurement we own
# (tuner-v10-qwen36-27b-dec, 82.77 GB at max_seq 20480) stayed invisible to
# a harvest that reported 360 files and 255 anchors. `--b2-sync` mirrors the
# published summaries into a local root so the ordinary walk can see them; the
# mirror is a default root, so a later plain harvest does not lose them again.
B2_REMOTE = os.environ.get("B2_REMOTE", "b2")
B2_BUCKET = os.environ.get("B2_BUCKET", "example-runs-bucket")
B2_PREFIX = os.environ.get("B2_PREFIX", "checkpoints")


def b2_cache_dir() -> str:
    """The mirror lives in the MAIN checkout's `out/` (gitignored, and where
    run output already lives) so a harvest from a worktree reads one copy."""
    return os.path.join(_main_checkout(), "out", "vram-b2-summaries")


def default_roots() -> list:
    """Where finished runs land. Labels (not paths) are what gets recorded, so
    the committed facts file carries no absolute machine path."""
    return [
        ("checkpoints", os.path.expanduser("~/checkpoints")),
        ("out-jobs", os.path.join(_main_checkout(), "out", "jobs")),
        ("archive-runs", os.path.join(_bench_dir(), "archive", "runs")),
        ("b2-published", b2_cache_dir()),
    ]


def _rclone(args) -> tuple:
    """(rc, stdout, stderr). A missing rclone is rc=127, never an exception —
    the harvester must still walk local roots on a box without it."""
    import subprocess
    try:
        p = subprocess.run(["rclone", *args], capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", "rclone not found on PATH"


def _prune_orphan_adapter_configs(dest: str) -> int:
    """Drop `adapter_config.json` files with no `train_summary.json` beside them.

    It is pulled for the base-name recovery `_base_slug` does, but every
    `checkpoint-NN/` subdir has one too and none of those sits next to a
    summary. Pruning keeps the mirror shaped like a run dir, not like a bucket.
    """
    n = 0
    for dirpath, _dirs, files in os.walk(dest):
        if "adapter_config.json" in files and "train_summary.json" not in files:
            try:
                os.remove(os.path.join(dirpath, "adapter_config.json"))
                n += 1
            except OSError:
                pass
    return n


def sync_b2(dest=None, remote=B2_REMOTE, bucket=B2_BUCKET, prefix=B2_PREFIX,
            runner=None) -> dict:
    """Mirror published `train_summary.json` (+ its `adapter_config.json`) into
    the local B2 root. Read-only against B2: `rclone copy`, never sync/delete.

    `runner` is injectable so the tests drive this without a network.
    """
    dest = dest or b2_cache_dir()
    run = runner or _rclone
    os.makedirs(dest, exist_ok=True)
    rc, _out, err = run(["copy", f"{remote}:{bucket}/{prefix}", dest,
                         "--include", "**/train_summary.json",
                         "--include", "**/adapter_config.json",
                         "--transfers", "16", "--checkers", "16"])
    pruned = _prune_orphan_adapter_configs(dest)
    found = sum(1 for _d, _s, files in os.walk(dest)
                if "train_summary.json" in files)
    return {"rc": rc, "dest": dest, "summaries": found, "pruned": pruned,
            "error": (err or "").strip()}

# Copied verbatim from train_summary.json. Split into the dimensions that
# SELECT an anchor group (vram_facts.GROUP_KEYS) and the ones that only
# describe it — the split lives in vram_facts.py so there is one definition.
SHAPE_KEYS = [
    "base", "quant_mode", "quant_skip_modules", "max_seq", "batch", "grad_accum",
    # `grad_checkpointing_flag` is the FRACTION of layers checkpointed, and it
    # is a setting, not a detail: a partial-GC cell records
    # grad_checkpointing=true with flag 0.0 and measures 147 GB where the same
    # declared shape at flag 1.0 measures 29. Without it those cells land in the
    # full-GC group and drive its max, which is how a re-harvest can make a
    # correctly-sized bundle read as provably under-sized.
    "eff_batch", "world_size", "grad_checkpointing", "grad_checkpointing_flag",
    "ce_chunk_matmul",
    "target_modules", "lora_r", "lora_alpha", "packing", "attn_impl",
    "sdpa_backends", "use_liger", "fsdp", "device_map_mode", "pipeline_split",
    "activation_offloading", "num_workers", "tf32",
]
CONTEXT_KEYS = ["n_rows", "dataset_content_sha256", "global_steps",
                "step_time_seconds", "resumed"]
MEASURE_KEYS = ["peak_vram_alloc_gb", "peak_vram_reserved_gb",
                "peak_vram_alloc_gb_per_gpu", "peak_vram_reserved_gb_per_gpu"]

# --- summary_schema 2 telemetry -----------------------------------------------
# Copied key-by-key from an allow-list rather than wholesale, for the same
# reason SHAPE_KEYS is a list: a summary is free to grow fields, the committed
# anchor file is not free to grow unreviewed ones (a stray absolute path, a
# token, a megabyte of per-step series). A block the trainer has not written
# yet simply produces no key.
HARDWARE_KEYS = ["gpu_names", "gpu_total_mem_gb", "driver_version",
                 "cuda_version", "torch_version", "transformers_version",
                 "trl_version", "peft_version", "flash_attn_version",
                 "bitsandbytes_version", "pytorch_cuda_alloc_conf", "platform",
                 # software-epoch keys: which of OUR software produced the
                 # anchor. A rate derived across an epoch boundary is not one
                 # measurement, so the boundary has to be IN the anchor.
                 "trainer_rev", "image"]
TOKEN_STAT_KEYS = ["row_tokens_max", "row_tokens_p99", "row_tokens_p95",
                   "row_tokens_p50", "row_tokens_mean", "n_rows",
                   "n_rows_at_max_seq", "source"]
THROUGHPUT_KEYS = ["tokens_seen", "tokens_per_second", "samples_per_second",
                   "step_time_p50_s", "step_time_p95_s", "step_time_max_s",
                   "n_steps_timed", "source", "scope"]

# Path-valued keys, dropped before hashing: they are exactly what differs
# between a run and its archived copy.
_VOLATILE = ("out", "data")


def _dedupe_key(summary: dict) -> str:
    stable = {k: v for k, v in summary.items() if k not in _VOLATILE}
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()


# Model names reach us three ways — an asset dir (`assets/qwen35-9b`), an HF
# repo id (`Qwen/Qwen2.5-Coder-7B-Instruct`), and an HF cache path
# (`models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/<sha>`). They are the same
# model and must land in ONE anchor group, so names are canonicalized by
# lowercasing and dropping punctuation. An explicit table, not a fuzzy match:
# two models silently merged would average incompatible measurements.
_ALIASES = {
    "qwen25coder7binstruct": "qwen25-coder-7b-instruct",
    "qwen2coder7binstruct": "qwen25-coder-7b-instruct",
    "qwen359b": "qwen35-9b",
    "qwen354b": "qwen35-4b",
    "gemma412btext": "gemma4-12b-text",
    "qwen3627b": "qwen36-27b",
}
_GENERIC = ("base", "model", "assets", "out", "adapter")


def _canon(name: str) -> str:
    flat = "".join(c for c in name.lower() if c.isalnum())
    return _ALIASES.get(flat, name)


def _from_hf_path(text: str) -> str:
    """`.../models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/<sha>` -> the
    model name. The HF cache encodes it; a content-addressed snapshot dir does
    not, which is why reading only the basename lost 22 of 25 anchors."""
    for part in str(text or "").split(os.sep):
        if part.startswith("models--"):
            return part.split("--")[-1]
    return ""


def _base_slug(raw, summary_dir: str = "") -> str:
    """The base model's canonical name, or '' when it cannot be identified.

    `--base` is often an on-box asset path, and some are content-addressed
    (`.../c202236235762e1c871ad0ccb60c8ee5ba337b9a`) or generically named
    (`assets/base`). Two recoveries before giving up: the HF cache path encodes
    the name, and the run's own `adapter_config.json` records
    `base_model_name_or_path`. Only then return '' — the estimator refuses to
    match those, because a sizing answer keyed on a name that identifies no
    model is worse than no answer."""
    raw = str(raw or "").rstrip("/")
    hf = _from_hf_path(raw)
    if hf:
        return _canon(hf)
    name = os.path.basename(raw)
    unusable = (not name or name.lower() in _GENERIC
                or (len(name) >= 32
                    and all(c in "0123456789abcdef" for c in name.lower())))
    if not unusable:
        return _canon(name)
    cfg = os.path.join(summary_dir, "adapter_config.json")
    if os.path.isfile(cfg):
        try:
            with open(cfg) as fh:
                ref = json.load(fh).get("base_model_name_or_path") or ""
            hf = _from_hf_path(ref) or os.path.basename(str(ref).rstrip("/"))
            if hf and hf.lower() not in _GENERIC and not (
                    len(hf) >= 32 and all(c in "0123456789abcdef" for c in hf.lower())):
                return _canon(hf)
        except (OSError, ValueError):
            pass
    return ""


def _redact_path(raw) -> str:
    """Keep the last two path segments of a model path, drop everything above.

    `/home/<user>/.cache/huggingface/.../models--Qwen--X/snapshots/<sha>` ->
    `models--Qwen--X/snapshots/<sha>`. Enough to see WHAT was loaded, with no
    machine path committed to git."""
    parts = [p for p in str(raw or "").rstrip("/").split(os.sep) if p]
    return os.sep.join(parts[-2:]) if parts else ""


def _scrub_paths(value):
    """`_redact_path` applied to every path-like string in a nested structure.

    The telemetry blocks carry free-form provenance strings — `token_stats.
    source` and `throughput.source` name the file the numbers were computed
    from, which on a local run is `/home/<user>/…`. The anchor file is
    COMMITTED, so no absolute machine path may reach it (CLAUDE.md), and the
    rule has to survive a block gaining a field nobody here reviewed. Hence a
    recursive scrub of the whole block rather than a per-field call: strings
    with no separator pass through untouched (`2.8.0+cu128`,
    `expandable_segments:True`), so the cost of over-applying it is nil."""
    if isinstance(value, dict):
        return {k: _scrub_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_paths(v) for v in value]
    if isinstance(value, str) and (os.sep in value or value.startswith("~")):
        return _redact_path(value)
    return value


def _gpu_power_aggregate(block: dict) -> dict:
    """Per-GPU power samples -> the fleet-level numbers worth committing.

    Aggregated, not copied: the per-GPU list is 8 dicts on a big box and the
    anchor file is read by a submit-path lookup. What survives is the pair that
    answers "was this shape actually saturating the card": `power_max_w` against
    `power_limit_w`, and the mean as a percentage of the limit.

    UTIL IS KEPT BUT IS NOT THE SATURATION SIGNAL. `nvidia-smi` utilization
    reads 100% for any kernel resident on the device; a contended GEMM measured
    +36% power at a flat util 100%. So a knob that changes power at constant
    util has changed the work, and a report that looked only at util would call
    that a null."""
    per = [p for p in (block.get("per_gpu") or []) if isinstance(p, dict)]

    def _vals(key):
        return [p[key] for p in per
                if isinstance(p.get(key), (int, float))]

    def _mean(key):
        v = _vals(key)
        return round(sum(v) / len(v), 1) if v else None

    def _max(key):
        v = _vals(key)
        return round(max(v), 1) if v else None

    out, measured = {"n_gpus": len(per)}, {}
    for k in ("interval_s", "n_samples"):
        if block.get(k) is not None:
            out[k] = block[k]
    for name, fn, key in (("power_mean_w", _mean, "power_mean_w"),
                          ("power_p95_w", _max, "power_p95_w"),
                          ("power_max_w", _max, "power_max_w"),
                          ("power_limit_w", _max, "power_limit_w"),
                          ("util_mean_pct", _mean, "util_mean_pct"),
                          ("mem_used_max_gb", _max, "mem_used_max_gb")):
        v = fn(key)
        if v is not None:
            out[name] = measured[name] = v
    if out.get("power_mean_w") and out.get("power_limit_w"):
        out["power_mean_pct_of_limit"] = round(
            100.0 * out["power_mean_w"] / out["power_limit_w"], 1)
    # A sampler that ran but observed nothing (an empty `per_gpu`) is not a
    # measurement, and an anchor carrying `{"n_gpus": 0, "interval_s": 5}` would
    # read to a later census as an instrumented run.
    return out if measured else {}


def _alloc_aggregate(block: dict) -> dict:
    """Allocator retries/OOMs summed across cards.

    A nonzero `num_alloc_retries` is the caching allocator having to free and
    re-request — the state a peak measured just under the card's capacity is
    actually in, and the reason two runs of one declared shape can differ. Summed
    because any card thrashing makes the run a thrashing run; the per-card
    breakdown is not worth committing."""
    per = [p for p in (block.get("per_gpu") or []) if isinstance(p, dict)]
    out = {}
    for key in ("num_alloc_retries", "num_ooms"):
        vals = [p[key] for p in per if isinstance(p.get(key), (int, float))]
        if vals:
            out[key] = sum(vals)
    if out:
        out["n_gpus"] = len(per)
    return out


def telemetry_of(summary: dict) -> dict:
    """The schema-2 blocks an anchor keeps, or `{}` for a v1 summary.

    ABSENT MEANS ABSENT. Every block, and every key inside it, is copied only
    when the summary actually has it: no defaults, no nulls, no zeros. All 167
    summaries on disk when this landed are v1, so the common case is `{}` and
    the anchor gains no `telemetry` key at all — which is what lets a reader
    distinguish "this run was not instrumented" from "this run measured zero
    tokens/s"."""
    if not isinstance(summary, dict):
        return {}
    t = {}
    for name, keys in (("hardware", HARDWARE_KEYS),
                       ("token_stats", TOKEN_STAT_KEYS),
                       ("throughput", THROUGHPUT_KEYS)):
        block = summary.get(name)
        if isinstance(block, dict):
            kept = {k: block[k] for k in keys if k in block}
            if kept:
                t[name] = kept
    for name, fn in (("gpu_power", _gpu_power_aggregate),
                     ("alloc_stats", _alloc_aggregate)):
        block = summary.get(name)
        if isinstance(block, dict):
            kept = fn(block)
            if kept:
                t[name] = kept
    if t:
        schema = summary.get("summary_schema")
        if schema is not None:
            t["schema"] = schema
    return _scrub_paths(t)


def _run_id(rel: str) -> str:
    """First path segment — the run/experiment folder. Best available id."""
    return rel.split(os.sep, 1)[0] if rel else ""


# A resume that re-enters a finished segment still writes a train_summary, and
# its peak is a model sitting in memory, not a training peak. Such rows cannot
# lift a group's estimate (the estimate is the group MAX) but they land in the
# spread, and spread is `risk_gb` -- the band jobmeta widens its refusal by --
# so one no-op resume silently buys a wrong declaration ~13 GB of tolerance.
# Sub-threshold step time is the one signal both observed no-ops carry (2-3 ms;
# the smallest real training step on file is 0.484 s, a 160x gap). A summary
# with NO step_time predates the instrument and is admitted as before.
MIN_LIVE_STEP_TIME_S = 0.1


def collect(roots=None, verbose=False) -> tuple[list, dict]:
    """Walk `roots`, return (anchors, stats). Pure except for reading files."""
    roots = roots if roots is not None else default_roots()
    anchors, seen = [], {}
    stats = {"files": 0, "no_peak": 0, "duplicates": 0, "unresolved_base": 0,
             "idle": 0, "telemetry": 0}
    for label, root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            if "train_summary.json" not in filenames:
                continue
            path = os.path.join(dirpath, "train_summary.json")
            stats["files"] += 1
            try:
                with open(path) as fh:
                    s = json.load(fh)
            except (OSError, ValueError):
                continue
            if not s.get("peak_vram_alloc_gb"):
                stats["no_peak"] += 1          # older runs predate the instrument
                continue
            st = s.get("step_time_seconds")
            if isinstance(st, (int, float)) and st < MIN_LIVE_STEP_TIME_S:
                stats["idle"] += 1             # no-op resume: a model sitting in
                continue                       # memory, not a training peak
            k = _dedupe_key(s)
            rel = os.path.relpath(os.path.dirname(path), root)
            if k in seen:
                stats["duplicates"] += 1
                seen[k]["sources"].append({"root": label, "path": rel})
                continue
            slug = _base_slug(s.get("base"), os.path.dirname(path))
            if not slug:
                stats["unresolved_base"] += 1
            shape = {k2: s.get(k2) for k2 in SHAPE_KEYS if k2 in s}
            # `base` is whatever --base was pointed at, which on a local run is
            # an absolute home path. The slug is the identity we key on and the
            # only part worth keeping; committing the rest would put machine
            # paths in git (CLAUDE.md) for no analytical gain.
            shape["base"] = _redact_path(shape.get("base"))
            a = {
                "base_slug": slug,
                "run": _run_id(rel),
                "sources": [{"root": label, "path": rel}],
                "shape": shape,
                "context": {k2: s.get(k2) for k2 in CONTEXT_KEYS if k2 in s},
                "measured": {k2: s.get(k2) for k2 in MEASURE_KEYS if k2 in s},
            }
            tel = telemetry_of(s)
            if tel:
                a["telemetry"] = tel
                stats["telemetry"] += 1
            seen[k] = a
            anchors.append(a)
            if verbose:
                print(f"  + {slug or '(unresolved)':28s} "
                      f"{a['measured']['peak_vram_alloc_gb']:6.2f} GB  {label}/{rel}",
                      file=sys.stderr)
    # Deterministic order so a re-harvest produces a reviewable diff, not a
    # reshuffle. Sorted on identity, never on a float.
    anchors.sort(key=lambda a: (a["base_slug"], str(a["shape"].get("quant_mode")),
                                a["shape"].get("max_seq") or 0,
                                a["run"], a["sources"][0]["path"]))
    for a in anchors:
        a["sources"].sort(key=lambda s: (s["root"], s["path"]))
    return anchors, stats


def build(roots=None, verbose=False) -> dict:
    anchors, stats = collect(roots, verbose=verbose)
    usable = [a for a in anchors if a["base_slug"]]
    return {
        "schema": 1,
        "note": ("Measured VRAM anchors harvested from train_summary.json. "
                 "Regenerate with tools/vast/harvest_vram.py --write; read "
                 "with tools/vast/vram_facts.py. Method and caveats: "
                 "tools/vast/VRAM_SIZING.md."),
        "stats": dict(stats, anchors=len(anchors), usable=len(usable)),
        "anchors": anchors,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", action="append", default=[],
                   help="extra root to walk (repeatable); defaults still apply")
    p.add_argument("--only-root", action="append", default=[],
                   help="walk ONLY these roots (repeatable)")
    p.add_argument("--write", action="store_true",
                   help=f"write {os.path.join('tools', 'vast', 'vram_facts.json')}")
    p.add_argument("--out", default=os.path.join(HERE, "vram_facts.json"))
    p.add_argument("--print", dest="show", action="store_true",
                   help="print the anchor table instead of writing")
    p.add_argument("--force", action="store_true",
                   help="write even when the harvest found FEWER anchors than "
                        "the existing file — i.e. run history really was removed")
    p.add_argument("--b2-sync", action="store_true",
                   help="mirror published train_summary.json from "
                        "b2:<bucket>/checkpoints/ before walking — the only "
                        "place a rented-box run's summary exists")
    p.add_argument("--b2-remote", default=B2_REMOTE)
    p.add_argument("--b2-bucket", default=B2_BUCKET)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    if a.b2_sync:
        st = sync_b2(remote=a.b2_remote, bucket=a.b2_bucket)
        print(f">> b2 sync: {st['summaries']} published summaries in "
              f"{st['dest']} ({st['pruned']} orphan adapter_config pruned)",
              file=sys.stderr)
        if st["rc"]:
            # A failed pull that harvested anyway would write a file missing
            # exactly the anchors the sync was asked for, and report success.
            print(f"error: b2 sync failed (rc={st['rc']}): {st['error'][:400]}",
                  file=sys.stderr)
            return 2

    roots = ([(os.path.basename(r.rstrip("/")) or "root", os.path.expanduser(r))
              for r in a.only_root] if a.only_root else default_roots())
    roots += [(os.path.basename(r.rstrip("/")) or "root", os.path.expanduser(r))
              for r in a.root]

    doc = build(roots, verbose=a.verbose)
    st = doc["stats"]
    print(f">> {st['files']} summaries, {st['no_peak']} without a peak, "
          f"{st['duplicates']} archive duplicates collapsed, "
          f"{st['unresolved_base']} with an unidentifiable base "
          f"-> {st['anchors']} anchors ({st['usable']} usable, "
          f"{st['telemetry']} with schema-2 telemetry)", file=sys.stderr)

    if a.show or not a.write:
        print(f"{'base':26s} {'quant':6s} {'seq':>6s} {'B':>2s} {'ws':>2s} "
              f"{'gck':5s} {'ce':5s} {'peakGB':>7s}  run")
        for an in doc["anchors"]:
            s, m = an["shape"], an["measured"]
            print(f"{(an['base_slug'] or '(unresolved)')[:26]:26s} "
                  f"{str(s.get('quant_mode'))[:6]:6s} {str(s.get('max_seq')):>6s} "
                  f"{str(s.get('batch')):>2s} {str(s.get('world_size')):>2s} "
                  f"{str(s.get('grad_checkpointing'))[:5]:5s} "
                  f"{str(s.get('ce_chunk_matmul'))[:5]:5s} "
                  f"{m['peak_vram_alloc_gb']:7.2f}  {an['run'][:34]}")
    if a.write:
        # The anchor file is COMMITTED, and a harvest on a box with no local run
        # history finds nothing. Without this, `--write` there would silently
        # replace every anchor with an empty list and the estimator would start
        # refusing shapes it has measured for months — a destructive no-op that
        # reads as success. Losing anchors is always a fact to confirm, never a
        # default.
        prior = 0
        if os.path.isfile(a.out):
            try:
                with open(a.out) as fh:
                    prior = len(json.load(fh).get("anchors") or [])
            except (OSError, ValueError):
                prior = 0
        if prior and len(doc["anchors"]) < prior and not a.force:
            print(f"error: refusing to write — this harvest found "
                  f"{len(doc['anchors'])} anchors but {a.out} already has "
                  f"{prior}. Roots walked: "
                  f"{', '.join(label for label, _ in roots)}. If run history "
                  f"really was removed, re-run with --force.", file=sys.stderr)
            return 2
        with open(a.out, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=False)
            fh.write("\n")
        print(f">> wrote {a.out}"
              + (f" ({len(doc['anchors']) - prior:+d} anchors)" if prior else ""),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
