#!/usr/bin/env python3
"""disksize — how much disk does this job actually need?

Owner question this answers: *"How do we best calculate the disk a run
requires? Right now I see agents picking an arbitrary number, but we should
compute this based on the configuration of the job instead."*

The arbitrary numbers are measurable and expensive. `--disk` has six
hand-typed defaults across the tree carrying three different values with no
shared constant and no comment explaining any of them, runset READMEs duplicate
a fourth set in prose (60/80/120/200), and `jobmatrix.py` has **zero** disk
references, so an N-arm matrix cannot express disk at all. What that produced,
measured: box 46234244 billed **$4.62/day on a 160 GB allocation with 17 GB
used** (8.9x oversized), and box 46256890 launched at 160 GB again the same
night. Storage bills on the **allocated** disk, so the waste is charged whether
or not a byte is written.

WHAT IS KNOWABLE, AND WHAT IS NOT. This module is deliberately honest about the
boundary, because a confident wrong number is worse than a declared unknown:

  * KNOWABLE pre-launch, exactly — every `assets[].b2` prefix size, the base
    model's bytes, the bundle tarball, the declared checkpoint retention.
  * A POLICY BOUND ONLY, never a prediction — `results:`/`checkpoints:` output
    volume (only the path *set* is declared, never its size) and a `venv: serve`
    live pip resolve. Those are measured once per pin and carried as constants,
    not derived.
  * NOT KNOWABLE AT ALL WITHOUT A DECLARATION — whatever the ENTRYPOINT writes.
    A ninja build tree, per-worker PCHs, a merged model. `needs.scratch_gb` is
    the only channel, and its ABSENCE is indistinguishable from a job that
    writes nothing. So `complete: true` means "every asset I tried to size, I
    sized" and never "this number is the requirement" — `scratch_declared` in
    the breakdown is the other half, and `format_breakdown` prints it.

    Cost of conflating the two, measured 2026-08-25: `screenv1-e3-evals` sized
    at 40 G — complete, evidenced, every asset measured — and its entrypoint's
    two-stage LoRA merge needs ~2x the 18 GiB base co-resident. rc 5 on the
    bundle's own pre-merge guard, AFTER the rent, a ~19 GB image pull, the base
    pull, a vLLM boot and a 12/12 positive control. The fix is on both sides:
    the bundle now declares `scratch_gb`, and nothing here calls a bare
    lower bound "recommended" any more.

So `estimate_disk_gb` returns a floor plus explicit slack, and every term is
itemized in the returned breakdown so a human can see which one dominates
rather than trusting a single scalar.

TWO STRUCTURAL FACTS THE FORMULA TURNS ON

1. **Assets dedupe across arms.** jobd's asset cache is keyed on `name`,
   survives park/resume, and is shared by every job on the box; `dest` is a
   SYMLINK into that cache (JOBS_DESIGN.md), so it adds zero bytes. The floor is
   therefore `sum(distinct assets)`, NOT a sum over arms — a naive per-arm sum
   massively over-allocates exactly the matrix jobs that most need this right.
2. **Peak > steady state — but only for an ASSET THAT IS ACTUALLY AN ARCHIVE.**
   The tree's reclaim pattern is `unpack && rm -f "$TARBALL"`, so a formula
   built on FINAL usage under-sizes every box that unpacks an archive: for one
   moment the tarball and its expansion are both on disk. That peak gets an
   explicit term rather than being folded into slack, so it stays visible.

   CORRECTED 2026-08-17 — this term used to be applied to the largest asset
   UNCONDITIONALLY, and that was wrong twice over. The `unpack && rm` pattern
   lives in `onstart/fetch_eval_env.sh`, which fetches the eval-env tarball
   (`env-<ver>.tar.zst`) and is already priced by `VENV_GB["eval"]` — it is not
   the `assets:` path at all. `asset_pull()` (`onstart/jobd.sh`) is a plain
   `rclone copy|sync` of a B2 PREFIX into the asset cache: no tar, no zstd, no
   unpack step, so the archive-plus-expansion moment cannot occur for a
   directory asset. Charging it anyway added a full duplicate of the base model
   to every training box — measured on `k4-qla-steadystate`, 18.0 GB of a
   70 GB request (26%), against boxes whose real usage was 20-22 GB.

   That was not merely wasted storage. `replacement_disk_gb()` takes the
   launch figure as a floor, so the inflation propagates into the REPLACEMENT
   SEARCH FILTER: on 2026-08-17 fleetd condemned a stalled box and then failed
   every rescue with `no qualifying replacement offer (after exclusions; the
   search requires >= 70G of container disk)`. An over-sized disk shrinks the
   offer pool exactly when a rescue needs it widest.

   So the peak is now charged only to assets that are archives — declared
   `archive: true`, or named with an archive suffix. The direction of the
   residual error is deliberate: an unflagged tarball still gets the term via
   its suffix, so the failure mode is over-sizing a rare case rather than
   under-sizing it.

Leaf module: stdlib only, imports nothing from tools/vast, no I/O. Byte
resolution belongs to the caller's existing runner seam
(`jobmeta.check_asset_staleness`), which already holds a bucket + runner and
already runs at submit — before any upload or spend.
"""
from __future__ import annotations

GB = float(1 << 30)

# Fixed floor for the OS, logs and jobd's own working state. A box that fits its
# payload exactly is a box that dies on the first log rotation.
#
# NOT an image allowance, measured 2026-08-17 from jobd's own `scratch_probe`
# boot events (n=3: boxes 47922911/47921947/47928229 at 80/50/38 GB): every one
# reports `workspace_size_mb` equal to its allocation exactly and 85 MB used at
# boot. The ~18 GB unpacked image is therefore OUTSIDE the container quota — an
# 80 GB box would show ~62 GB free if it were not. So these 12 GB are pure
# headroom, generous rather than dangerously thin, and adding image bytes on top
# of an estimate is double-counting. `disk_usage_from_event` collects the other
# half (peak-vs-allocated per job) so this constant can eventually be argued
# from a distribution instead of from prudence.
BASE_OVERHEAD_GB = 12.0

# `venv: serve` is a LIVE pip resolve (vLLM + torch + CUDA wheels), with an
# ensurepip fallback that relocates bytes. Measured once per pin and carried;
# predicting it from a config is not possible and pretending otherwise is how
# you get a box that dies 40 minutes in.
VENV_GB = {"none": 0.0, "eval": 6.0, "serve": 22.0}

# Multiplier applied to the largest ARCHIVE asset to cover the
# archive-plus-expansion moment (fact 2 above). 1.0 = room for the archive
# alongside its own expansion. Charged ONLY to archive assets — a directory
# asset is rclone-copied and never has both forms on disk at once.
UNPACK_PEAK_FACTOR = 1.0

# An asset counts as an archive when it declares `archive: true` OR its B2 path
# ends in one of these. The suffix fallback exists so an author who ships a
# tarball without the flag is over-sized rather than under-sized; the flag
# exists because a B2 prefix carries no reliable type information.
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.zst", ".tzst",
                    ".tar.bz2", ".tar.xz", ".zip", ".7z", ".gz", ".zst")


def _is_archive_asset(a):
    """True when this asset is unpacked on the box, so its expansion may be
    co-resident with the archive itself. PURE, and deliberately conservative:
    unknown shapes are NOT archives, because charging every asset the peak is
    the bug this predicate was introduced to fix."""
    if not isinstance(a, dict):
        return False
    if a.get("archive"):
        return True
    b2 = str(a.get("b2") or "").strip().lower().rstrip("/")
    return b2.endswith(ARCHIVE_SUFFIXES)

# Trainer default (`runspec_proposal.py`): how many checkpoint dirs live on disk
# at once. Retention is a MULTIPLIER on checkpoint size, and it is the term that
# most often dominates a training box.
DEFAULT_SAVE_TOTAL_LIMIT = 7

# Round the answer up to a tidy allocation. Vast bills the allocated size, so
# the step is deliberately small — the old habit of reaching for 160 is what
# this module exists to replace.
ROUND_STEP_GB = 10.0


def _gb(nbytes):
    try:
        return max(0.0, float(nbytes) / GB)
    except (TypeError, ValueError):
        return 0.0


def _pos(gb):
    """A declared GB figure, or 0.0 for absent/garbage/non-positive. Never
    raises: a malformed declaration must not take down a submit, and jobmeta
    already rejects one at parse time."""
    try:
        v = float(gb)
    except (TypeError, ValueError):
        return 0.0
    return v if v > 0 else 0.0


def _round_up(gb, step=ROUND_STEP_GB):
    if step <= 0:
        return float(gb)
    return float(step) * ((float(gb) + float(step) - 1e-9) // float(step) +
                          (0.0 if abs(gb % step) < 1e-9 and gb > 0 else 0.0))


def estimate_disk_gb(cfg, sizes, *, save_total_limit=None, ckpt_bytes=None,
                     results_bytes=None, base_model_bytes=None,
                     bundle_bytes=None):
    """PURE. `(gb, breakdown)` — the disk this job config needs.

    `cfg`    a parsed job-config (jobmeta.parse_job_config's dict), or anything
             with the same `assets` / `needs` shape.
    `sizes`  {asset_name: bytes} resolved by the caller (rclone size, or the
             `.complete` marker's `bytes` for a base model). A name absent from
             `sizes` is UNKNOWN, not zero: it is reported in
             `breakdown["unknown_assets"]` so the caller can say so out loud
             rather than silently under-allocating.

    Everything else is optional and defaults to a documented constant. Returns
    the rounded GB and an itemized breakdown; the caller decides whether to
    warn, refuse, or just print.
    """
    cfg = cfg or {}
    sizes = sizes or {}
    assets = list(cfg.get("assets") or [])
    needs = cfg.get("needs") or {}

    # (1) distinct assets — keyed on NAME, which is exactly jobd's cache key, so
    # this mirrors what the box will actually store. Duplicate names cannot
    # occur (jobmeta rejects them), but arms sharing a name legitimately do, and
    # they cost the bytes ONCE.
    per_asset, unknown, optional_gb = {}, [], 0.0
    archive_assets = {}
    for a in assets:
        name = str(a.get("name") or "")
        if not name:
            continue
        if name in per_asset:
            continue
        if name not in sizes:
            unknown.append(name)
            continue
        g = _gb(sizes[name])
        per_asset[name] = g
        if _is_archive_asset(a):
            archive_assets[name] = g
        if a.get("optional"):
            optional_gb += g
    assets_gb = sum(per_asset.values())

    # (2) the unpack peak: the largest single ARCHIVE asset may transiently
    # exist as archive + expansion. Keyed off the LARGEST, not the sum — they
    # are staged one at a time. A directory asset contributes NOTHING here:
    # `asset_pull()` rclone-copies a B2 prefix straight into the cache, so
    # there is no second copy at any instant (fact 2 in the header, corrected
    # 2026-08-17 — charging every asset this term was adding a full duplicate
    # of the base model to every training box).
    largest = max(archive_assets.values(), default=0.0)
    unpack_gb = largest * UNPACK_PEAK_FACTOR

    # (3) base model and bundle, when the caller resolved them.
    base_gb = _gb(base_model_bytes) if base_model_bytes is not None else 0.0
    bundle_gb = _gb(bundle_bytes) if bundle_bytes is not None else 0.0

    # (4) venv, by declared kind.
    venv = str(needs.get("venv", "none") or "none")
    venv_gb = VENV_GB.get(venv, VENV_GB["none"])

    # (5) checkpoint retention. Only counts when the job actually checkpoints;
    # `checkpoint_s` is what makes a job resumable and is what puts N copies on
    # disk at once.
    limit = (DEFAULT_SAVE_TOTAL_LIMIT if save_total_limit is None
             else max(0, int(save_total_limit)))
    ckpt_gb = 0.0
    if cfg.get("checkpoint_s") and ckpt_bytes:
        ckpt_gb = _gb(ckpt_bytes) * limit
    # A job that checkpoints but whose checkpoint SIZE nobody resolved is an
    # unpriced term, not a zero one — and it must say so. This was masked until
    # 2026-08-17: the unconditional unpack peak happened to add roughly a base
    # model of slack to every training box, which quietly absorbed checkpoints
    # the estimator had never counted. Removing that padding is correct, and it
    # makes this omission load-bearing, so it is now surfaced rather than
    # rounded into silence. `launch_jobs_box.sh` resolves no ckpt bytes today,
    # so every checkpointing bundle trips this.
    ckpt_unpriced = bool(cfg.get("checkpoint_s")) and not ckpt_bytes

    # (6) results the job writes locally before publishing. A bound the caller
    # supplies; never predicted from the glob set.
    results_gb = _gb(results_bytes) if results_bytes is not None else 0.0

    # (7) declared scratch — working state the job CREATES, which no amount of
    # reading the config can reveal: a ninja build tree of object files, N
    # per-worker compile worktrees each with its own PCH, an unpacked toolchain.
    # Every other term here is measured; this one is the author telling us what
    # their entrypoint makes. It is additive on purpose, so the measured terms
    # keep tracking reality as assets grow underneath it.
    scratch_gb = _pos(needs.get("scratch_gb"))

    subtotal = (assets_gb + unpack_gb + base_gb + bundle_gb + venv_gb +
                ckpt_gb + results_gb + scratch_gb + BASE_OVERHEAD_GB)
    gb = _round_up(subtotal)

    # A declared needs.disk_gb OVERRIDES the derivation outright — the escape
    # hatch for "the estimate is known wrong here". The derived number is still
    # computed and kept in the breakdown, because the interesting failure is a
    # declaration that has drifted far from what the config now implies, and
    # dropping the estimate would hide exactly that.
    declared = _pos(needs.get("disk_gb"))
    derived_gb = gb
    if declared:
        gb = float(declared)

    breakdown = {
        "scratch_gb": round(scratch_gb, 2),
        "declared_disk_gb": declared or None,
        "derived_gb": derived_gb,
        "assets_gb": round(assets_gb, 2),
        "assets_distinct": len(per_asset),
        "assets_optional_gb": round(optional_gb, 2),
        "unpack_peak_gb": round(unpack_gb, 2),
        # Which assets were treated as archives — empty is the common and
        # correct case, and printing it is what makes a non-zero peak
        # auditable instead of mysterious.
        "archive_assets": sorted(archive_assets),
        "base_model_gb": round(base_gb, 2),
        "bundle_gb": round(bundle_gb, 2),
        "venv_gb": round(venv_gb, 2),
        "venv": venv,
        "checkpoints_gb": round(ckpt_gb, 2),
        "save_total_limit": limit if ckpt_gb else 0,
        "results_gb": round(results_gb, 2),
        "base_overhead_gb": BASE_OVERHEAD_GB,
        "subtotal_gb": round(subtotal, 2),
        "recommended_gb": gb,
        "unknown_assets": sorted(unknown),
        "checkpoints_unpriced": ckpt_unpriced,
        "complete": not unknown,
        # Whether the author told us what the ENTRYPOINT makes. `complete`
        # above is about assets and says nothing about this: a config can
        # measure every asset it declares, report complete, and still be short
        # by everything the job writes. Measured the expensive way 2026-08-25 —
        # screenv1-e3-evals sized at a `complete: true` 40 G and died on its own
        # pre-merge guard after the rent, the image pull, the base pull and a
        # 12/12 positive control, because a two-stage LoRA merge holds ~2x the
        # base on disk and no amount of reading `assets:` can see it. A declared
        # `needs.disk_gb` is the author speaking too, so it counts as declared.
        "scratch_declared": bool(scratch_gb or declared),
    }
    # `complete` means "nothing I tried to measure failed" — NOT "I measured
    # everything that matters". A config that declares no assets, no base model,
    # no bundle and no scratch gives the estimator nothing but the fixed
    # overhead, and it would then confidently call a 160 GB box 8x oversized on
    # the strength of having looked at nothing. Real bundles hit this: a trainer
    # whose entrypoint pulls its own base model from B2/HF declares no `assets:`
    # at all, while the audited box held 15 GB of weights and a 9.3 GB serve
    # venv. So an estimate with no variable evidence is flagged, and
    # oversize_finding treats it like an incomplete one — it may still warn
    # undersized, but it may never claim oversized.
    breakdown["evidence"] = bool(
        per_asset or unknown or base_gb or bundle_gb or ckpt_gb or results_gb
        or scratch_gb or venv_gb or declared)
    return gb, breakdown


# Weight-bytes multiplier for a serve box: the B2 pull lands the safetensors
# once (no unpack peak), but vLLM adds compile/torch caches, tokenizer dupes,
# and log growth proportional to nothing we can measure pre-launch — 1.2x the
# weights plus the fixed overhead absorbed both on every measured serve box.
SERVE_MODEL_FACTOR = 1.2


def serve_disk_gb(model_bytes, *, lora_bytes=0, extra_gb=0.0):
    """PURE. `(gb, breakdown)` — disk for a SERVE box (launch_serve.sh).

    `model_bytes` is the measured byte total of the served base (rclone size of
    the b2: subpath, or an HF API sum). None/0 means UNMEASURED — the caller
    must fall back to its static default and say so out loud, never treat the
    result as a measurement (`complete` is False in that case). The t211 image
    does not count against the allocation (vast stores image layers host-side);
    BASE_OVERHEAD_GB covers OS/logs/caches, same constant as the jobs formula.
    """
    model_gb = _gb(model_bytes)
    lora_gb = _gb(lora_bytes)
    subtotal = (BASE_OVERHEAD_GB + model_gb * SERVE_MODEL_FACTOR
                + lora_gb + _pos(extra_gb))
    gb = _round_up(subtotal)
    return gb, {
        "model_gb": round(model_gb, 2),
        "model_factor": SERVE_MODEL_FACTOR,
        "lora_gb": round(lora_gb, 2),
        "extra_gb": round(_pos(extra_gb), 2),
        "base_overhead_gb": BASE_OVERHEAD_GB,
        "subtotal_gb": round(subtotal, 2),
        "recommended_gb": gb,
        "complete": model_gb > 0,
    }


def oversize_finding(*, declared_gb, recommended_gb, storage_day_usd=None,
                     breakdown=None, tolerance=1.5):
    """PURE. `(severity, message)` for a declared `--disk` vs the estimate.

    severity: `ok` | `unknown` | `oversized` | `undersized`.

    Priced in DOLLARS, not GB, because GB is not what anyone decides on: the
    2026-07-21 audit found a 160 GB allocation billing ~$4.60/day while using
    18, and it stayed invisible for hours precisely because `ls` reported the
    disk's cost but never its utilization.

    `undersized` outranks `oversized`: too small is a job that dies mid-run on a
    rented box, too large is money. An INCOMPLETE estimate can only ever be a
    lower bound, so it may warn undersized but must never claim oversized —
    that direction would talk someone into shrinking a box on missing data.
    """
    if declared_gb is None or recommended_gb is None:
        return "unknown", "no declared or recommended size to compare"
    dec, rec = float(declared_gb), float(recommended_gb)
    complete = (breakdown or {}).get("complete", True)

    if dec < rec:
        gap = rec - dec
        return "undersized", (
            f"--disk {dec:g}G is BELOW the estimated need of {rec:g}G "
            f"(short {gap:.0f}G) — the job dies mid-stage on a rented box, "
            f"not here")
    if not complete:
        miss = ", ".join((breakdown or {}).get("unknown_assets") or [])
        return "unknown", (
            f"estimate is a LOWER BOUND — could not size: {miss}. "
            f"Not claiming {dec:g}G is oversized on incomplete data")
    if breakdown is not None and not breakdown.get("evidence", True):
        return "unknown", (
            f"this config declares no assets, base model, checkpoints or "
            f"needs.scratch_gb, so the estimate is the fixed overhead and "
            f"nothing else — an entrypoint that fetches its own weights is "
            f"invisible here. Not claiming {dec:g}G is oversized on that")
    if dec > rec * float(tolerance):
        msg = (f"--disk {dec:g}G is {dec / max(rec, 1e-9):.1f}x the estimated "
               f"need of {rec:g}G")
        if storage_day_usd:
            waste = float(storage_day_usd) * (dec - rec) / max(dec, 1e-9)
            msg += (f" — about ${waste:.2f}/day of the ${storage_day_usd:.2f}/day "
                    f"storage bill is allocation you will not use "
                    f"(${waste * 30:.0f}/month)")
        return "oversized", msg
    return "ok", f"--disk {dec:g}G is within {tolerance:g}x of the {rec:g}G estimate"


def format_breakdown(breakdown, *, indent="    "):
    """Itemized lines, biggest term first — so a reader sees WHICH term
    dominates instead of arguing with a scalar."""
    b = dict(breakdown or {})
    terms = [("assets", b.get("assets_gb")),
             ("base-model", b.get("base_model_gb")),
             ("checkpoints", b.get("checkpoints_gb")),
             ("venv:" + str(b.get("venv", "?")), b.get("venv_gb")),
             ("unpack-peak", b.get("unpack_peak_gb")),
             ("results", b.get("results_gb")),
             ("scratch (declared)", b.get("scratch_gb")),
             ("bundle", b.get("bundle_gb")),
             ("os+image overhead", b.get("base_overhead_gb"))]
    out = []
    for label, gb in sorted(((l, g or 0.0) for l, g in terms),
                            key=lambda t: -t[1]):
        if gb <= 0:
            continue
        out.append(f"{indent}{gb:7.1f}G  {label}")
    if b.get("unknown_assets"):
        out.append(f"{indent}      ?  UNSIZED: "
                   + ", ".join(b["unknown_assets"]) + "  (lower bound only)")
    if b.get("checkpoints_unpriced"):
        out.append(f"{indent}      ?  checkpoint_s is set but no checkpoint "
                   f"size was resolved — retention is UNPRICED here "
                   f"(lower bound only)")
    # The blind spot that is NOT an unknown: nothing failed to measure, the
    # author simply never said what the entrypoint writes. Printed because the
    # reader's question is "can I trust this number", and a clean itemization
    # of assets answers it wrongly for any job that builds, merges or unpacks.
    if not b.get("scratch_declared", True):
        out.append(f"{indent}      ?  no needs.scratch_gb — working state the "
                   f"ENTRYPOINT makes (build trees, a merged model, an "
                   f"unpacked toolchain) is not in this number")
    out.append(f"{indent}{'-' * 30}")
    # FLOOR, not "recommended": every unresolved term above can only push the
    # real need UP, so the honest reading of the total is a lower bound. It was
    # called "recommended" until 2026-08-25 and that word is what an operator
    # took at face value into a box 40 G too small.
    out.append(f"{indent}{b.get('recommended_gb', 0):7.1f}G  FLOOR / recommended"
               f" minimum (subtotal {b.get('subtotal_gb', 0):.1f}G, rounded up)")
    # A needs.disk_gb declaration wins, but the derived figure stays visible:
    # the interesting failure mode is a declaration that has drifted away from
    # what the config now implies, and printing only the winner hides it.
    dec = b.get("declared_disk_gb")
    if dec:
        drift = "" if not b.get("derived_gb") else (
            f", config now implies {b['derived_gb']:.0f}G")
        out.append(f"{indent}{' ' * 7}   ^ declared by needs.disk_gb{drift}")
    return out


# --------------------------------------------------------------------------- #
# RAM-backed scratch (velvet P4d)
#
# The owner's observation is right and the obvious implementation of it is
# wrong, so the numbers are worth stating: on the audited box
# (BOX_SATURATION_AUDIT_2026-07-30.md §3.1-3.2) RAM was 503 GiB visible with a
# 366.9 GB cgroup limit and only 34-37 GB in use all run — hundreds of idle
# gigabytes — while the 120 GB disk held a 31.3 GB steady state.
#
# But `/tmp` on that box is on the OVERLAY, not a tmpfs. Moving scratch to
# /tmp therefore moves nothing: those bytes land on the same allocated disk,
# and if we had shrunk the disk to pay for the move, the job would die. The
# RAM-backed filesystem that actually exists there is `/dev/shm` — a 125 GB
# tmpfs that no script in the tree uses.
#
# So the placement rule below turns on MEASURED facts about a specific box and
# refuses to act on anything else. Two hard constraints:
#
#   1. tmpfs pages are charged to the cgroup's memory limit. Filling a 125 GB
#      /dev/shm on a box whose trainer wants RAM does not cost disk, it OOM-kills
#      the job. The RAM budget is bounded by FREE memory, never by shm size alone.
#   2. Scratch only moves off disk when the job author declared it
#      reconstructible. Data we cannot rebuild does not belong on a filesystem
#      that a reboot empties.
# --------------------------------------------------------------------------- #

SHM_HEADROOM_GB = 8.0        # never hand out the last of a tmpfs
RAM_HEADROOM_FRAC = 0.35     # of free cgroup memory, leave this much to the job


def plan_scratch_placement(*, scratch_gb, volatile=False, shm_gb=None,
                           mem_limit_gb=None, mem_used_gb=None):
    """PURE. `(ram_gb, disk_gb, reason)` — where does declared scratch live?

    `shm_gb` / `mem_limit_gb` / `mem_used_gb` are MEASURED box facts (jobd's
    boot probe). Any of them missing means the whole thing degrades to disk:
    an unverified assumption about a box's filesystems must never be the reason
    its disk allocation shrank, because if the assumption is wrong the bytes
    land on the disk we just made too small. That is the same shape as the house
    rule that unreadable evidence never accelerates a destructive action.

    Returns disk_gb == scratch_gb in every uncertain case, so a caller that
    ignores `ram_gb` entirely still gets today's correct answer.
    """
    want = _pos(scratch_gb)
    if want <= 0:
        return 0.0, 0.0, "no scratch declared"
    if not volatile:
        return 0.0, want, ("scratch is not declared volatile — data we cannot "
                           "rebuild does not go on a filesystem a reboot empties")
    shm, lim, used = _pos(shm_gb), _pos(mem_limit_gb), _pos(mem_used_gb)
    if shm <= 0 or lim <= 0:
        return 0.0, want, ("no measured tmpfs/memory facts for this box — "
                           "keeping scratch on disk rather than shrinking the "
                           "allocation on an assumption")
    by_shm = shm - SHM_HEADROOM_GB
    by_mem = (lim - used) * (1.0 - RAM_HEADROOM_FRAC)
    budget = max(0.0, min(by_shm, by_mem))
    ram = min(want, budget)
    if ram <= 0:
        return 0.0, want, (f"no RAM budget for scratch (tmpfs {shm:.0f}G, free "
                           f"memory {max(0.0, lim - used):.0f}G) — staying on disk")
    binding = "tmpfs size" if by_shm <= by_mem else "free memory"
    return ram, want - ram, (
        f"{ram:.0f}G of {want:.0f}G scratch fits in RAM (bound by {binding}; "
        f"tmpfs {shm:.0f}G, free memory {max(0.0, lim - used):.0f}G) — tmpfs "
        f"pages are charged to the cgroup, so this is capped by memory, not by "
        f"the tmpfs alone")


def matrix_disk_shape(cfgs, *, concurrency=None):
    """PURE. What an N-arm matrix costs a single box, structurally (no I/O).

    Returns a dict; the two numbers that matter pull in OPPOSITE directions and
    getting either backwards is a real over/under-allocation:

      * **assets DEDUPE across arms.** jobd's cache is keyed on `name` and every
        job on the box shares it, so four arms naming the same 15 GB base model
        cost 15 GB, not 60. Summing per arm over-allocates precisely the matrix
        jobs this module exists to fix.
      * **scratch does NOT dedupe.** jobd schedules arms onto free cards and runs
        them CONCURRENTLY, so each running arm builds its own tree in its own
        workdir. Four concurrent arms declaring 20 GB of scratch need 80 GB at
        once. Deduping it the way assets dedupe would under-allocate, which is
        the worse direction — that is a job that dies on a rented box.

    `concurrency` caps how many arms are in flight at once (jobd assigns cards,
    so it is usually the GPU count). None = assume every arm runs at once, the
    conservative reading.
    """
    cfgs = list(cfgs or [])
    names, scratches, volatile_only = [], [], True
    for c in cfgs:
        for a in (c or {}).get("assets") or []:
            n = str(a.get("name") or "")
            if n and n not in names:
                names.append(n)
        needs = (c or {}).get("needs") or {}
        s = _pos(needs.get("scratch_gb"))
        if s:
            scratches.append(s)
            if not needs.get("scratch_volatile"):
                volatile_only = False
    naive = sum(len((c or {}).get("assets") or []) for c in cfgs)
    inflight = len(scratches) if concurrency is None else min(
        len(scratches), max(1, int(concurrency)))
    # the largest declarations are the ones that will be co-resident in the worst
    # case, so take them from the top rather than averaging
    peak_scratch = sum(sorted(scratches, reverse=True)[:inflight])
    return {
        "arms": len(cfgs),
        "distinct_assets": names,
        "asset_entries_naive": naive,
        "asset_dedup_saving": max(0, naive - len(names)),
        "scratch_arms": len(scratches),
        "scratch_concurrent": inflight,
        "scratch_peak_gb": round(peak_scratch, 2),
        "scratch_all_volatile": bool(scratches) and volatile_only,
    }


# Headroom multiplier on a box's MEASURED usage when sizing its replacement.
# The replacement re-stages the same assets, and the measurement is a snapshot
# that may predate the last of them.
UNDERSTUDY_HEADROOM_FACTOR = 1.4


def understudy_disk_gb(*, allocated_gb, used_gb):
    """PURE. `(gb, reason)` — disk for a box that REPLACES a live one.

    Today the handoff/understudy path copies the primary's ALLOCATED
    `disk_space`, which propagates whatever guess the primary launched with —
    the only "derived" size in the tree, deriving the wrong thing. A primary at
    160 GB holding 17 GB mints another 160 GB box.

    This can only ever shrink, never grow, and only on a real measurement:
    `max(used * headroom + overhead)` clamped to the allocated size, and it
    returns the allocated size untouched whenever usage is unreadable (a booting
    box reports `disk_usage: -1`). A handoff is time-critical and a too-small
    replacement loses the run, so every uncertain path keeps today's behaviour.
    """
    alloc = _pos(allocated_gb)
    used = _pos(used_gb)
    if alloc <= 0:
        return None, "no allocated size to copy — caller keeps its own default"
    if used <= 0:
        return alloc, (f"primary usage unreadable — copying its {alloc:g}G "
                       f"allocation unchanged")
    want = used * UNDERSTUDY_HEADROOM_FACTOR + BASE_OVERHEAD_GB
    gb = min(alloc, _round_up(want))
    if gb >= alloc:
        return alloc, (f"primary is using {used:.0f}G of {alloc:g}G — its "
                       f"allocation is already right-sized")
    return gb, (f"primary is using {used:.0f}G of {alloc:g}G, so the understudy "
                f"gets {gb:g}G ({UNDERSTUDY_HEADROOM_FACTOR:g}x usage + "
                f"{BASE_OVERHEAD_GB:g}G overhead) instead of copying the "
                f"allocation")


# Box-env key carrying the `--disk` GB the LAUNCH ASKED VAST FOR, stamped at
# create time and read back off `extra_env` by the supervise lanes.
#
# It exists because `disk_space` on the instance body is what vast DELIVERED,
# and the two are not the same number: a host advertising less than the request
# hands back a smaller container instead of refusing the rental. Sizing a
# replacement off the delivered figure therefore inherits the shortfall — a job
# whose assets need 19.3 GB was rehosted onto 10 GB (2026-08-18, box 48005604
# -> 48006140). The request is the statement about the WORKLOAD; the allocation
# is a fact about one box.
LAUNCH_DISK_ENV = "LAUNCH_DISK_GB"


def launch_disk_gb_from_env(env, allocated_gb=None):
    """PURE. `(gb, shortfall_gb)` from a box's env stamp + its allocation.

    `gb` is the REQUESTED size (None when the box predates the stamp, so the
    caller keeps its old behaviour), `shortfall_gb` the positive gap when vast
    delivered less than was asked — the number a caller must say out loud."""
    want = _pos((env or {}).get(LAUNCH_DISK_ENV))
    if want <= 0:
        return None, 0.0
    got = _pos(allocated_gb)
    return want, (want - got if 0 < got < want else 0.0)


# Disk for a forced rehost when NOTHING is knowable — no launch anchor, no
# readable primary. Deliberately generous (a too-small box loses the run, a
# too-large one costs ~$2/day) and deliberately NOT a vastconf launch default:
# reaching it means the inheritance chain broke, which the caller must SAY.
REPLACEMENT_FALLBACK_GB = 120.0


def replacement_disk_gb(*, launch_gb, allocated_gb, used_gb):
    """PURE. `(gb, why)` — disk for a box that REPLACES a condemned or EVICTED
    one, or `(None, why)` when nothing is knowable and the caller must fall
    back to `REPLACEMENT_FALLBACK_GB` *out loud*.

    `max` of three terms, every one of which is allowed to be unknown:

      * `launch_gb`  — what the WORKLOAD was sized at (the watch's
        `launch_disk_gb` anchor). This is the term the driftr3 H200 lane was
        missing: `--disk 110`, sized for a 56.8 GB two-stage merge transient,
        and the job ended up on a 60 GB box that died on its own disk guard
        (DRIFT_ROSTER_R3_H200_COHORT_2026-08-06.md §8).
      * `allocated_gb` — what the box being replaced holds NOW. Never below it
        (R7, 2026-08-05, box 46914272): on a forced rehost the box was
        condemned or evicted, often mid-boot, so it may be the only evidence
        there is — but it is evidence about the box, not about the job, so once
        any hop lands smaller, flooring at it propagates the shrink for the
        rest of the chain. That is why `launch_gb` outranks nothing and simply
        joins the max.
      * `used_gb` — measured usage plus headroom, the one term that may exceed
        both of the others. A box 92% full is evidence its allocation was
        WRONG, and a rehost is the only moment that can be corrected (a running
        instance's disk cannot be resized). Bounded by construction: usage can
        never exceed the allocation, so a single hop can grow the box by at
        most `UNDERSTUDY_HEADROOM_FACTOR` x + `BASE_OVERHEAD_GB`.

    Contrast `understudy_disk_gb`, which may only SHRINK: that is the right rule
    for the ECONOMIC handoff lane (a steady-state box using 17 of 160 GB should
    not mint another 160 GB box) and the wrong one here, because on this lane
    the usage snapshot measures how far the restage got, not what the job needs.
    """
    terms = []
    launch, alloc, used = _pos(launch_gb), _pos(allocated_gb), _pos(used_gb)
    if launch > 0:
        terms.append((launch, f"the {launch:g}G the job was LAUNCHED at"))
    if alloc > 0:
        terms.append((alloc, f"the replaced box's {alloc:g}G allocation"))
    if used > 0:
        terms.append((_round_up(used * UNDERSTUDY_HEADROOM_FACTOR +
                                BASE_OVERHEAD_GB),
                      f"its measured usage ({used:.0f}G x "
                      f"{UNDERSTUDY_HEADROOM_FACTOR:g} + "
                      f"{BASE_OVERHEAD_GB:g}G overhead)"))
    if not terms:
        return None, ("no launch anchor, no allocation and no usage — nothing "
                      "in this watch knows how big the job's disk was")
    gb, why = max(terms, key=lambda t: t[0])
    others = [w for g, w in terms if w != why]
    return gb, (f"{why} wins"
                + (f" over {', '.join(others)}" if others else
                   " (the only figure available)"))


def scratch_facts_from_probe(view):
    """PURE. jobd's `scratch_probe` event -> `plan_scratch_placement` kwargs.

    The probe reports megabytes and writes the literal string `"unknown"` for
    anything it could not read, so this is where those become the `None`s that
    make the placement rule degrade to disk. Silently coercing `"unknown"` to 0
    would read as "no tmpfs, no memory" and is harmless here only by luck —
    doing it explicitly keeps it that way.

    Uses the CGROUP pair, not `/proc/meminfo`: meminfo reports the HOST's memory
    (503 GiB on the audited box) while tmpfs pages are charged against the
    container's limit (366.9 GB there). Sizing RAM scratch off the host figure
    would over-commit by whatever the host is oversubscribed.
    """
    v = dict(view or {})

    def mb(key):
        raw = v.get(key)
        if raw in (None, "", "unknown"):
            return None
        try:
            n = float(raw)
        except (TypeError, ValueError):
            return None
        return (n / 1024.0) if n > 0 else None

    return {
        "shm_gb": mb("shm_size_mb"),
        "mem_limit_gb": mb("cgroup_mem_limit_mb"),
        "mem_used_gb": mb("cgroup_mem_current_mb") or 0.0,
    }


def disk_usage_from_event(view):
    """PURE. jobd's `disk_usage` event -> the allocated-vs-used record.

    This estimator has never had a measured distribution to check itself
    against: `scratch_probe` fires at boot, so it sees the allocation against an
    empty disk and can only ever report the floor (85 MB on every box measured,
    at 38/50/80 GB — the unpacked image lives outside the container quota).
    jobd's `disk_usage` event is the other half, emitted at a job's terminal
    before the checkpoint scrub, i.e. at the job's real peak footprint.

    `slack_gb` is the number to fold across many jobs before touching a
    constant here: it is what the launch over-bought. It can be NEGATIVE only
    if a box reports usage above its own allocation (an overlay accounting
    quirk), which is worth seeing rather than clamping away.

    The high-water is BOX-scoped by construction — concurrent arms share the
    filesystem — so on a multi-job box read it as the box peak, not this job's.
    """
    v = dict(view or {})

    def gb(key):
        raw = v.get(key)
        if raw in (None, "", "unknown"):
            return None
        try:
            n = float(raw)
        except (TypeError, ValueError):
            return None
        return n / 1024.0 if n >= 0 else None

    size, used, hw = gb("workspace_size_mb"), gb("workspace_used_mb"), \
        gb("box_high_water_mb")
    peak = max([x for x in (used, hw) if x is not None], default=None)
    return {
        "job": v.get("job") or None,
        "allocated_gb": size,
        "used_gb": used,
        "high_water_gb": hw,
        "job_dir_gb": gb("job_dir_mb"),
        "peak_gb": peak,
        "slack_gb": None if (size is None or peak is None) else size - peak,
    }
