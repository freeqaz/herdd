"""The config-derived disk estimator (velvet plan P2).

What these guard is a MONEY bug with a measured history: box 46234244 billed
$4.62/day on a 160 GB allocation with 17 GB used (8.9x), and box 46256890
launched at 160 GB again the same night. Storage bills on the ALLOCATED disk,
so the waste is charged whether or not a byte is written.

Two rules carry most of the value and are easy to regress into a plausible
wrong answer:

  * assets dedupe on NAME (jobd's cache key) and `dest` is a symlink into that
    cache, so a matrix of arms sharing an asset pays for it ONCE. Summing per
    arm is the naive mistake, and it over-allocates exactly the jobs that most
    need this right.
  * an INCOMPLETE estimate is a lower bound and must never be allowed to say
    "oversized" — that direction talks someone into shrinking a box on data we
    do not have.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import disksize as ds  # noqa: E402

GB = ds.GB


def cfg(assets=None, **kw):
    c = {"version": 1, "name": "j", "needs": {"venv": "none"}}
    if assets is not None:
        c["assets"] = assets
    c.update(kw)
    return c


def A(name, optional=False, b2=None, archive=None):
    a = {"name": name, "b2": b2 if b2 is not None else f"x/{name}",
         "optional": optional}
    if archive is not None:
        a["archive"] = archive
    return a


# --- the dedup rule ----------------------------------------------------------

def test_assets_are_counted_once_per_NAME_not_per_arm():
    """jobd's cache is keyed on name and shared across concurrent jobs, so N
    arms declaring the same asset cost its bytes once. A per-arm sum here would
    over-allocate an N-arm matrix by N-fold."""
    one = ds.estimate_disk_gb(cfg([A("base")]), {"base": 20 * GB})[1]
    # the same asset named by many arms of one bundle
    many = ds.estimate_disk_gb(
        cfg([A("base"), A("base"), A("base")]), {"base": 20 * GB})[1]
    assert one["assets_gb"] == many["assets_gb"] == 20.0
    assert many["assets_distinct"] == 1


def test_distinct_assets_do_add_up():
    b = ds.estimate_disk_gb(
        cfg([A("a"), A("b")]), {"a": 10 * GB, "b": 5 * GB})[1]
    assert b["assets_gb"] == 15.0 and b["assets_distinct"] == 2


# --- unknown sizes are UNKNOWN, never zero -----------------------------------

def test_an_unsized_asset_is_reported_not_silently_zero():
    """Treating a missing size as 0 is how you ship a confidently undersized
    box. It must surface as an explicit incompleteness."""
    gb, b = ds.estimate_disk_gb(cfg([A("a"), A("ghost")]), {"a": 10 * GB})
    assert b["unknown_assets"] == ["ghost"] and b["complete"] is False


def test_a_fully_sized_config_is_complete():
    _gb, b = ds.estimate_disk_gb(cfg([A("a")]), {"a": 1 * GB})
    assert b["complete"] is True and b["unknown_assets"] == []


# --- the unpack peak ---------------------------------------------------------

def test_unpack_peak_keys_off_the_LARGEST_ARCHIVE_not_the_sum():
    """Archives stage one at a time, so the transient archive+expansion moment
    is bounded by the biggest single one. Using the sum would re-inflate the
    over-allocation this module exists to remove."""
    _gb, b = ds.estimate_disk_gb(
        cfg([A("big", archive=True), A("small", archive=True)]),
        {"big": 30 * GB, "small": 2 * GB})
    assert b["unpack_peak_gb"] == 30.0


def test_no_assets_means_no_unpack_peak():
    _gb, b = ds.estimate_disk_gb(cfg([]), {})
    assert b["unpack_peak_gb"] == 0.0


def test_a_DIRECTORY_asset_is_charged_NO_unpack_peak():
    """THE 2026-08-17 FIX. `asset_pull()` rclone-copies a B2 prefix into the
    cache — no tar, no zstd, no second copy at any instant — so a directory
    asset cannot produce the archive+expansion moment. Charging it anyway put
    a full duplicate of the 18 GB base model into every training box's disk
    request (k4-qla-steadystate: 18 of 70 GB, against 20-22 GB really used),
    and because replacement_disk_gb() floors on the launch figure it also
    narrowed the rescue search until no offer qualified."""
    _gb, b = ds.estimate_disk_gb(cfg([A("base")]), {"base": 18 * GB})
    assert b["unpack_peak_gb"] == 0.0
    assert b["archive_assets"] == []


def test_the_base_model_shaped_config_lands_at_the_measured_size():
    """End-to-end guard on the real k4 shape: an 18 GiB base-model prefix, a
    tiny runset, 15 GB declared scratch. 70 GB was the old answer; the boxes
    that ran it used 20-22 GB, and a peer box ran the same image + 9B model
    on a 38 GB allocation."""
    c = cfg([A("base"), A("runset")], needs={"venv": "none", "scratch_gb": 15})
    gb, b = ds.estimate_disk_gb(c, {"base": 18 * GB, "runset": 1024})
    assert b["unpack_peak_gb"] == 0.0
    assert gb == 50.0, b


def test_an_explicit_archive_flag_restores_the_peak():
    _gb, b = ds.estimate_disk_gb(cfg([A("env", archive=True)]), {"env": 6 * GB})
    assert b["unpack_peak_gb"] == 6.0 and b["archive_assets"] == ["env"]


def test_an_archive_SUFFIX_is_enough_without_the_flag():
    """Direction of the residual error is deliberate: an unflagged tarball is
    over-sized by its suffix rather than under-sized by its silence."""
    for path in ("eval-env/env-9.tar.zst", "x/toolchain.tgz", "x/w.zip"):
        _gb, b = ds.estimate_disk_gb(
            cfg([A("t", b2=path)]), {"t": 4 * GB})
        assert b["unpack_peak_gb"] == 4.0, path


def test_a_prefix_that_merely_CONTAINS_a_dot_is_not_an_archive():
    _gb, b = ds.estimate_disk_gb(
        cfg([A("m", b2="base-models/qwen3.5-9b")]), {"m": 18 * GB})
    assert b["unpack_peak_gb"] == 0.0


def test_declared_checkpoints_with_no_resolved_size_are_flagged_UNPRICED():
    """Removing the blanket unpack peak removed slack that had been quietly
    absorbing uncounted checkpoints, so the omission has to become visible.
    An unpriced term must never read as a zero one."""
    c = cfg([A("base")], checkpoint_s=600)
    _gb, b = ds.estimate_disk_gb(c, {"base": 18 * GB})
    assert b["checkpoints_unpriced"] is True and b["checkpoints_gb"] == 0.0
    assert "UNPRICED" in "\n".join(ds.format_breakdown(b))


def test_priced_checkpoints_are_not_flagged():
    c = cfg([A("base")], checkpoint_s=600)
    _gb, b = ds.estimate_disk_gb(c, {"base": 18 * GB}, ckpt_bytes=GB,
                                 save_total_limit=3)
    assert b["checkpoints_unpriced"] is False and b["checkpoints_gb"] == 3.0


def test_a_job_that_never_checkpoints_is_not_flagged():
    _gb, b = ds.estimate_disk_gb(cfg([A("base")]), {"base": 18 * GB})
    assert b["checkpoints_unpriced"] is False


# --- venv / checkpoints / overhead -------------------------------------------

def test_serve_venv_dominates_a_bare_job():
    bare = ds.estimate_disk_gb(cfg([]), {})[1]["venv_gb"]
    serve = ds.estimate_disk_gb(cfg([], needs={"venv": "serve"}), {})[1]["venv_gb"]
    assert bare == 0.0 and serve == ds.VENV_GB["serve"] > 0


def test_unknown_venv_kind_falls_back_to_zero_not_a_crash():
    assert ds.estimate_disk_gb(cfg([], needs={"venv": "wat"}), {})[1]["venv_gb"] == 0.0


def test_checkpoint_retention_multiplies_and_only_when_checkpointing():
    """Retention is the term that most often dominates a training box:
    SAVE_TOTAL_LIMIT copies live on disk at once."""
    no_ckpt = ds.estimate_disk_gb(cfg([]), {}, ckpt_bytes=3 * GB)[1]
    assert no_ckpt["checkpoints_gb"] == 0.0, "no checkpoint_s -> no retention cost"
    with_ckpt = ds.estimate_disk_gb(
        cfg([], checkpoint_s=300), {}, ckpt_bytes=3 * GB, save_total_limit=4)[1]
    assert with_ckpt["checkpoints_gb"] == 12.0 and with_ckpt["save_total_limit"] == 4


def test_default_retention_is_the_trainers_own():
    b = ds.estimate_disk_gb(cfg([], checkpoint_s=300), {}, ckpt_bytes=1 * GB)[1]
    assert b["checkpoints_gb"] == float(ds.DEFAULT_SAVE_TOTAL_LIMIT)


def test_overhead_floor_applies_even_to_an_empty_job():
    """A box sized to its payload exactly dies on the first log rotation."""
    gb, b = ds.estimate_disk_gb(cfg([]), {})
    assert b["base_overhead_gb"] == ds.BASE_OVERHEAD_GB and gb >= ds.BASE_OVERHEAD_GB


def test_recommendation_rounds_UP_never_down():
    gb, b = ds.estimate_disk_gb(cfg([A("a")]), {"a": 1 * GB})
    assert gb >= b["subtotal_gb"]
    assert gb % ds.ROUND_STEP_GB == 0


# --- the finding: direction, dollars, and the incompleteness guard -----------

def test_undersized_is_flagged_and_outranks_everything():
    sev, msg = ds.oversize_finding(declared_gb=20, recommended_gb=80)
    assert sev == "undersized" and "60G" in msg


def test_the_real_46234244_shape_reads_oversized_in_dollars():
    """160G allocated, ~20G needed, $4.62/day — the measured incident."""
    sev, msg = ds.oversize_finding(declared_gb=160, recommended_gb=20,
                                   storage_day_usd=4.62,
                                   breakdown={"complete": True})
    assert sev == "oversized"
    assert "8.0x" in msg and "$" in msg and "/month" in msg


def test_an_INCOMPLETE_estimate_may_warn_undersized_but_never_oversized():
    """THE asymmetry. A lower bound can prove a box is too small; it can never
    prove one is too big, and saying so would talk someone into shrinking on
    data we do not have."""
    incomplete = {"complete": False, "unknown_assets": ["ghost"]}
    sev, msg = ds.oversize_finding(declared_gb=400, recommended_gb=20,
                                   breakdown=incomplete)
    assert sev == "unknown" and "LOWER BOUND" in msg and "ghost" in msg
    # ...but too-small is still actionable on a lower bound
    sev, _ = ds.oversize_finding(declared_gb=5, recommended_gb=20,
                                 breakdown=incomplete)
    assert sev == "undersized"


def test_a_reasonable_margin_is_ok_not_nagged():
    sev, _ = ds.oversize_finding(declared_gb=60, recommended_gb=50,
                                 breakdown={"complete": True})
    assert sev == "ok"


def test_missing_inputs_are_unknown_not_a_crash():
    assert ds.oversize_finding(declared_gb=None, recommended_gb=10)[0] == "unknown"
    assert ds.oversize_finding(declared_gb=10, recommended_gb=None)[0] == "unknown"


# --- the human-facing breakdown ---------------------------------------------

def test_breakdown_is_ordered_biggest_term_first():
    """The point of an itemized breakdown is seeing WHICH term dominates."""
    _gb, b = ds.estimate_disk_gb(
        cfg([A("a")], needs={"venv": "serve"}, checkpoint_s=300),
        {"a": 40 * GB}, ckpt_bytes=1 * GB, save_total_limit=2)
    lines = ds.format_breakdown(b)
    # terms only: the rule separator ends them, the total follows it
    terms = lines[:next(i for i, l in enumerate(lines) if set(l.strip()) == {"-"})]
    vals = [float(l.split("G")[0].strip()) for l in terms
            if l.strip() and l.strip()[0].isdigit()]
    assert vals and vals == sorted(vals, reverse=True), lines
    assert "recommended" in lines[-1]


def test_breakdown_says_so_when_it_is_only_a_lower_bound():
    _gb, b = ds.estimate_disk_gb(cfg([A("ghost")]), {})
    text = "\n".join(ds.format_breakdown(b))
    assert "UNSIZED" in text and "lower bound" in text


# --- the impure sizer: jobmeta.measure_asset_bytes ---------------------------
# Injected-runner, never raises. The contract that matters is what it does on a
# FAILED read: OMIT the name, so the estimator reports it unsized and degrades
# to a lower bound. Returning 0 would look like a confident tiny answer.

def _sizer_runner(sizes, *, fail=(), garbage=()):
    """rclone-shaped stub for `size --json`."""
    import json as _json

    def run(args, input=None):
        assert args[0] == "size", args
        prefix = args[-1].split("/", 1)[1]
        if prefix in fail:
            return 1, "", "transport blip"
        if prefix in garbage:
            return 0, "not json at all", ""
        if prefix not in sizes:
            return 0, "", ""
        return 0, _json.dumps({"count": 3, "bytes": sizes[prefix]}), ""
    return run


def test_measure_asset_bytes_maps_prefix_sizes_onto_names():
    import jobmeta
    assets = [{"name": "base", "b2": "base-models/m"},
              {"name": "rows", "b2": "corpora/v4"}]
    got = jobmeta.measure_asset_bytes(
        assets, bucket="bkt",
        runner=_sizer_runner({"base-models/m": 8 * GB, "corpora/v4": 2 * GB}))
    assert got == {"base": 8 * (1 << 30), "rows": 2 * (1 << 30)}


def test_measure_asset_bytes_OMITS_a_failed_read_rather_than_zeroing_it():
    """The whole honesty chain: omitted -> unsized -> lower bound -> the finding
    refuses to claim 'oversized'. A 0 here would break all three links."""
    import jobmeta
    assets = [{"name": "ok", "b2": "a/ok"}, {"name": "blip", "b2": "a/blip"},
              {"name": "junk", "b2": "a/junk"}]
    got = jobmeta.measure_asset_bytes(
        assets, bucket="bkt",
        runner=_sizer_runner({"a/ok": GB}, fail=("a/blip",), garbage=("a/junk",)))
    assert got == {"ok": 1 << 30}
    assert "blip" not in got and "junk" not in got

    # ...and that omission propagates all the way to the verdict
    c = cfg([A("ok"), A("blip"), A("junk")])
    _g, b = ds.estimate_disk_gb(c, got)
    assert b["complete"] is False
    assert ds.oversize_finding(declared_gb=500, recommended_gb=_g,
                               breakdown=b)[0] == "unknown"


def test_measure_asset_bytes_never_raises_on_a_throwing_runner():
    import jobmeta

    def boom(args, input=None):
        raise RuntimeError("rclone exploded")

    assert jobmeta.measure_asset_bytes(
        [{"name": "a", "b2": "x/a"}], bucket="bkt", runner=boom) == {}


def test_measure_asset_bytes_reads_each_distinct_prefix_once():
    import jobmeta
    seen = []

    def counting(args, input=None):
        seen.append(args[-1])
        import json as _json
        return 0, _json.dumps({"bytes": GB}), ""

    jobmeta.measure_asset_bytes(
        [{"name": "a", "b2": "x/shared"}, {"name": "b", "b2": "x/shared"}],
        bucket="bkt", runner=counting)
    assert len(seen) == 1, seen


# --- declared scratch and the disk_gb override (velvet P4b/P4c) --------------
# Two knobs that are deliberately NOT interchangeable. scratch_gb ADDS to a live
# derivation; disk_gb REPLACES it. Confusing them is how a config goes stale
# without anyone noticing.

def test_scratch_gb_adds_to_the_derived_estimate():
    """The owner's case: leave room for compiler build output. `assets:` sizes
    are measurable, but whatever the entrypoint MAKES — a ninja tree of object
    files, N per-worker worktrees and their PCHs — is invisible to the config."""
    plain = ds.estimate_disk_gb(cfg([A("a")]), {"a": 10 * GB})[1]
    with_s = ds.estimate_disk_gb(
        cfg([A("a")], needs={"venv": "none", "scratch_gb": 25}),
        {"a": 10 * GB})[1]
    assert with_s["scratch_gb"] == 25.0
    assert with_s["subtotal_gb"] == plain["subtotal_gb"] + 25.0


def test_scratch_does_NOT_freeze_the_measured_terms():
    """Why scratch_gb is additive rather than a total: the asset terms must keep
    tracking reality underneath it. A hand-typed total silently would not."""
    small = ds.estimate_disk_gb(
        cfg([A("a")], needs={"scratch_gb": 25}), {"a": 10 * GB})[0]
    grown = ds.estimate_disk_gb(
        cfg([A("a")], needs={"scratch_gb": 25}), {"a": 60 * GB})[0]
    assert grown > small


def test_disk_gb_overrides_but_keeps_the_derived_number_visible():
    """The escape hatch wins the answer, but hiding the derivation would hide
    the interesting failure: a declaration that has drifted from the config."""
    gb, b = ds.estimate_disk_gb(
        cfg([A("a")], needs={"venv": "none", "disk_gb": 200}), {"a": 10 * GB})
    assert gb == 200.0
    assert b["declared_disk_gb"] == 200.0
    assert 0 < b["derived_gb"] < 200.0, "derivation must survive the override"
    assert "declared by needs.disk_gb" in "\n".join(ds.format_breakdown(b))


def test_a_declaration_that_drifted_is_visible_in_the_breakdown():
    _gb, b = ds.estimate_disk_gb(
        cfg([A("a")], needs={"disk_gb": 160}), {"a": 5 * GB})
    assert "config now implies" in "\n".join(ds.format_breakdown(b))


def test_garbage_scratch_declarations_do_not_crash_a_submit():
    """jobmeta rejects these at parse time; the estimator still must not be the
    thing that explodes if one reaches it by another route."""
    for bad in (None, "lots", -5, float("nan")):
        b = ds.estimate_disk_gb(cfg([], needs={"scratch_gb": bad}), {})[1]
        assert b["scratch_gb"] >= 0.0


def test_scratch_appears_in_the_ordered_breakdown():
    _gb, b = ds.estimate_disk_gb(
        cfg([A("a")], needs={"scratch_gb": 40}), {"a": 5 * GB})
    assert any("scratch (declared)" in l for l in ds.format_breakdown(b))


# --- RAM-backed scratch placement (velvet P4d) ------------------------------ #
# The owner's observation is right, the obvious implementation is wrong, and
# BOX_SATURATION_AUDIT_2026-07-30 §3.1 has the measurements: 503 GiB RAM with a
# 366.9 GB cgroup limit and 34-37 GB in use, a 125 GB /dev/shm tmpfs nothing
# uses — and `/tmp` on the OVERLAY, so "just use /tmp" moves nothing and costs
# the same disk. These pin the two safety properties that make it usable.

AUDITED = dict(shm_gb=125.0, mem_limit_gb=366.9, mem_used_gb=37.0)


def test_the_audited_box_can_hold_a_compile_scratch_set_in_RAM():
    ram, disk, why = ds.plan_scratch_placement(
        scratch_gb=60, volatile=True, **AUDITED)
    assert (ram, disk) == (60.0, 0.0)
    assert "tmpfs" in why


def test_scratch_stays_on_DISK_without_measured_box_facts():
    """THE safety property. If we shrink an allocation because we assumed a
    tmpfs and the assumption is wrong, the bytes land on the disk we just made
    too small — an unverified assumption must never be the reason a box shrank."""
    ram, disk, why = ds.plan_scratch_placement(scratch_gb=60, volatile=True)
    assert (ram, disk) == (0.0, 60.0)
    assert "assumption" in why
    # partial facts are still no facts
    for partial in ({"shm_gb": 125.0}, {"mem_limit_gb": 366.9}):
        assert ds.plan_scratch_placement(
            scratch_gb=60, volatile=True, **partial)[1] == 60.0


def test_non_volatile_scratch_never_moves_however_much_RAM_exists():
    ram, disk, why = ds.plan_scratch_placement(scratch_gb=60, **AUDITED)
    assert (ram, disk) == (0.0, 60.0)
    assert "reboot empties" in why


def test_the_budget_is_bounded_by_FREE_MEMORY_not_by_the_tmpfs():
    """tmpfs pages are charged to the cgroup: filling a 125G /dev/shm on a box
    whose trainer wants RAM does not cost disk, it OOM-kills the job."""
    ram, disk, why = ds.plan_scratch_placement(
        scratch_gb=60, volatile=True, shm_gb=125.0,
        mem_limit_gb=64.0, mem_used_gb=60.0)
    assert ram < 5.0 and disk > 55.0, (ram, disk)
    assert "free memory" in why


def test_an_oversized_ask_SPLITS_rather_than_failing():
    """Partial placement is the useful answer: take what RAM can hold, leave the
    rest on disk, and keep the disk term honest about the remainder."""
    ram, disk, _ = ds.plan_scratch_placement(
        scratch_gb=200, volatile=True, **AUDITED)
    assert ram > 0 and disk > 0 and abs((ram + disk) - 200.0) < 1e-6


def test_placement_conserves_the_declared_total_in_every_branch():
    for kw in ({}, {"volatile": True}, dict(volatile=True, **AUDITED),
               dict(volatile=True, shm_gb=1.0, mem_limit_gb=8.0, mem_used_gb=7.9)):
        ram, disk, _ = ds.plan_scratch_placement(scratch_gb=60, **kw)
        assert abs((ram + disk) - 60.0) < 1e-6, kw


def test_a_full_tmpfs_degrades_to_disk_rather_than_handing_out_its_last_byte():
    ram, disk, why = ds.plan_scratch_placement(
        scratch_gb=60, volatile=True, shm_gb=4.0,
        mem_limit_gb=366.9, mem_used_gb=37.0)
    assert (ram, disk) == (0.0, 60.0) and "no RAM budget" in why


# --- matrix shape (velvet P4b) ---------------------------------------------- #
# The two rules pull in OPPOSITE directions, and each has a distinct failure:
# summing assets over-allocates (money), deduping scratch under-allocates (a
# dead job on a rented box). Both are easy to write the wrong way round.

def _arm(scratch=None, volatile=False, assets=("base", "rows")):
    c = {"assets": [{"name": n} for n in assets], "needs": {}}
    if scratch:
        c["needs"]["scratch_gb"] = scratch
        c["needs"]["scratch_volatile"] = volatile
    return c


def test_shared_assets_are_counted_once_across_the_whole_matrix():
    s = ds.matrix_disk_shape([_arm() for _ in range(4)])
    assert s["distinct_assets"] == ["base", "rows"]
    assert s["asset_entries_naive"] == 8 and s["asset_dedup_saving"] == 6


def test_scratch_is_SUMMED_across_concurrent_arms_not_deduped():
    """The opposite rule from assets, and the more dangerous one to get wrong:
    jobd runs arms concurrently on separate cards and each builds its own tree,
    so deduping here under-allocates and kills the job mid-run."""
    s = ds.matrix_disk_shape([_arm(scratch=20) for _ in range(4)])
    assert s["scratch_peak_gb"] == 80.0 and s["scratch_concurrent"] == 4


def test_concurrency_caps_the_scratch_peak():
    s = ds.matrix_disk_shape([_arm(scratch=20) for _ in range(4)], concurrency=2)
    assert s["scratch_peak_gb"] == 40.0


def test_the_LARGEST_declarations_are_the_ones_assumed_co_resident():
    """Worst case, not average: if only 2 of 4 arms run at once, assume the two
    hungriest are the pair."""
    arms = [_arm(scratch=n) for n in (5, 50, 10, 40)]
    assert ds.matrix_disk_shape(arms, concurrency=2)["scratch_peak_gb"] == 90.0


def test_arms_without_scratch_do_not_inflate_the_concurrency_count():
    s = ds.matrix_disk_shape([_arm(scratch=20), _arm(), _arm()])
    assert s["scratch_arms"] == 1 and s["scratch_peak_gb"] == 20.0


def test_all_volatile_is_only_true_when_EVERY_declaring_arm_says_so():
    assert ds.matrix_disk_shape(
        [_arm(scratch=10, volatile=True)] * 3)["scratch_all_volatile"] is True
    assert ds.matrix_disk_shape(
        [_arm(scratch=10, volatile=True), _arm(scratch=10)]
    )["scratch_all_volatile"] is False


def test_an_empty_matrix_is_not_a_crash():
    s = ds.matrix_disk_shape([])
    assert s["arms"] == 0 and s["scratch_peak_gb"] == 0.0


# --- "complete" is not "informative" ---------------------------------------- #
# Caught on real bundles: out/jobs-bundles/g4fit-liger-32k and
# out/jobs_stage/h1_paired_eval declare NO `assets:` at all — their entrypoints
# fetch their own weights — so nothing failed to measure and `complete` was
# True. The estimator then called a 160G box "8.0x oversized" on the strength of
# having looked at nothing, while the audited box of that shape held 15 GB of
# weights and a 9.3 GB serve venv.

def test_a_config_with_no_variable_terms_is_flagged_as_evidence_free():
    _gb, b = ds.estimate_disk_gb(cfg([]), {})
    assert b["complete"] is True, "nothing FAILED to measure"
    assert b["evidence"] is False, "but nothing was measured either"


def test_an_evidence_free_estimate_may_never_claim_oversized():
    """The same asymmetry as the incomplete case, through a different door."""
    _gb, b = ds.estimate_disk_gb(cfg([]), {})
    sev, msg = ds.oversize_finding(declared_gb=160, recommended_gb=20,
                                   storage_day_usd=4.62, breakdown=b)
    assert sev == "unknown" and "fetches its own weights" in msg


def test_an_evidence_free_estimate_still_warns_UNDERSIZED():
    """A floor is still a floor: a box below the fixed overhead is too small
    whatever else we failed to see."""
    _gb, b = ds.estimate_disk_gb(cfg([]), {})
    assert ds.oversize_finding(declared_gb=5, recommended_gb=20,
                               breakdown=b)[0] == "undersized"


def test_any_single_declared_term_counts_as_evidence():
    for kw in ({"needs": {"venv": "serve"}},
               {"needs": {"scratch_gb": 10}},
               {"needs": {"disk_gb": 50}}):
        assert ds.estimate_disk_gb(cfg([], **kw), {})[1]["evidence"] is True
    # a sized asset, and an UNSIZED one, both count
    assert ds.estimate_disk_gb(cfg([A("a")]), {"a": GB})[1]["evidence"] is True
    assert ds.estimate_disk_gb(cfg([A("ghost")]), {})[1]["evidence"] is True


def test_the_real_46234244_shape_still_reads_oversized():
    """The regression guard for the fix above: a config with real evidence must
    keep producing the actionable verdict."""
    _gb, b = ds.estimate_disk_gb(cfg([A("a")]), {"a": 8 * GB})
    assert ds.oversize_finding(declared_gb=160, recommended_gb=20,
                               storage_day_usd=4.62, breakdown=b)[0] == "oversized"


# --- understudy sizing (velvet P4) ------------------------------------------ #
# Replacing a live box was the ONE place the tree derived a size, and it derived
# the wrong thing: it copied the primary's ALLOCATED disk, so a 160G box holding
# 17G minted another 160G box. Every property here is about only ever shrinking.

def test_understudy_sizes_from_usage_on_the_measured_incident_shape():
    gb, why = ds.understudy_disk_gb(allocated_gb=160, used_gb=17)
    assert gb == 40.0 and "instead of copying the allocation" in why


def test_understudy_keeps_the_allocation_when_usage_is_unknown():
    """A booting box reports disk_usage -1, which _disk_gb maps to None. A
    handoff is time-critical: a too-small replacement loses the run, so an
    unreadable measurement keeps today's behaviour."""
    for unknown in (None, 0, -1):
        assert ds.understudy_disk_gb(allocated_gb=160, used_gb=unknown)[0] == 160.0


def test_understudy_never_grows_the_allocation():
    """Monotone-safe: this can only ever reduce what we would have launched."""
    for alloc, used in ((40, 35), (40, 39), (20, 18), (120, 110)):
        gb, _ = ds.understudy_disk_gb(allocated_gb=alloc, used_gb=used)
        assert gb <= alloc, (alloc, used)


def test_understudy_keeps_real_headroom_over_measured_usage():
    gb, _ = ds.understudy_disk_gb(allocated_gb=500, used_gb=100)
    assert gb >= 100 * ds.UNDERSTUDY_HEADROOM_FACTOR + ds.BASE_OVERHEAD_GB


def test_understudy_with_no_allocation_defers_to_the_caller():
    gb, why = ds.understudy_disk_gb(allocated_gb=None, used_gb=10)
    assert gb is None and "caller keeps its own default" in why


# --- probe -> placement (velvet P4d, closing the loop) ---------------------- #

def test_probe_megabytes_become_placement_gigabytes():
    f = ds.scratch_facts_from_probe({
        "shm_size_mb": 128000, "cgroup_mem_limit_mb": 350000,
        "cgroup_mem_current_mb": 36000})
    assert f["shm_gb"] == 125.0
    ram, disk, _ = ds.plan_scratch_placement(scratch_gb=60, volatile=True, **f)
    assert (ram, disk) == (60.0, 0.0)


def test_the_probes_literal_unknown_becomes_None_not_zero():
    """jobd writes the STRING "unknown" for anything it could not read. Coercing
    that to 0 would read as 'no tmpfs, no memory' — right answer, wrong reason,
    and only right by luck."""
    f = ds.scratch_facts_from_probe({
        "shm_size_mb": "unknown", "cgroup_mem_limit_mb": "unknown",
        "cgroup_mem_current_mb": "unknown"})
    assert f["shm_gb"] is None and f["mem_limit_gb"] is None
    assert ds.plan_scratch_placement(scratch_gb=60, volatile=True, **f)[1] == 60.0


def test_a_missing_probe_keeps_everything_on_disk():
    """A box we cannot interrogate must never get a SMALLER allocation than one
    we can."""
    for absent in (None, {}, {"event": "something_else"}):
        f = ds.scratch_facts_from_probe(absent)
        assert ds.plan_scratch_placement(scratch_gb=60, volatile=True, **f)[1] == 60.0


def test_facts_use_the_CGROUP_limit_not_host_meminfo():
    """meminfo reports the HOST's memory (503 GiB on the audited box) while
    tmpfs pages are charged to the container limit (366.9 GB). Sizing off the
    host figure over-commits by whatever the host is oversubscribed."""
    f = ds.scratch_facts_from_probe({
        "shm_size_mb": 128000, "mem_total_mb": 515000,
        "cgroup_mem_limit_mb": 20000, "cgroup_mem_current_mb": 1000})
    assert f["mem_limit_gb"] < 20.0
    ram, _d, _w = ds.plan_scratch_placement(scratch_gb=200, volatile=True, **f)
    assert ram < 20.0, "host memory must not leak into the budget"


def test_garbage_probe_values_do_not_crash_a_submit():
    f = ds.scratch_facts_from_probe({"shm_size_mb": "12x", "cgroup_mem_limit_mb": -1})
    assert f["shm_gb"] is None and f["mem_limit_gb"] is None


def test_a_REAL_jobd_box_event_feeds_the_placement_rule():
    """Built through jobmeta.make_box_event rather than a hand-written dict, so
    the two modules cannot drift apart silently: jobd emits flat k=v, jobd.py
    coerces integer-looking values to ints and leaves "unknown" a STRING, and
    that is exactly what the converter is written against."""
    import jobmeta
    ev = jobmeta.make_box_event(
        "12345", "scratch_probe", shm_size_mb=128000, shm_fs="tmpfs",
        tmp_fs="overlay", cgroup_mem_limit_mb=350000,
        cgroup_mem_current_mb=36000, mem_total_mb=515000, tmpfs_mount="ok")
    assert ev["event"] == "scratch_probe", "the reader filters on this"
    ram, disk, _ = ds.plan_scratch_placement(
        scratch_gb=60, volatile=True, **ds.scratch_facts_from_probe(ev))
    assert (ram, disk) == (60.0, 0.0)


# --- serve_disk_gb: the launch_serve.sh sizing seam (2026-08-02) --------------

def test_serve_disk_measured_model():
    """15.2 GB of weights (qwen2.5-coder-7b bf16): 12 overhead + 1.2x weights
    ~= 30.2 -> rounds to 40, replacing the hand-typed --disk 60 habit."""
    gb, b = ds.serve_disk_gb(int(15.2 * GB))
    assert gb == 40.0
    assert b["complete"] is True
    assert b["model_gb"] == 15.2


def test_serve_disk_unmeasured_is_declared_incomplete():
    """None/0 bytes is UNKNOWN, not zero-cost: the caller must fall back to its
    static default and say so — a bare-overhead number must never masquerade
    as a measurement."""
    for empty in (None, 0):
        gb, b = ds.serve_disk_gb(empty)
        assert b["complete"] is False
        assert b["model_gb"] == 0.0
        assert gb == 20.0                       # overhead-only floor, rounded


def test_serve_disk_lora_and_extra_count():
    gb0, _ = ds.serve_disk_gb(int(10 * GB))
    gb1, b = ds.serve_disk_gb(int(10 * GB), lora_bytes=int(2 * GB), extra_gb=5)
    assert gb1 >= gb0
    assert b["lora_gb"] == 2.0 and b["extra_gb"] == 5.0


# --- disk_usage_from_event: the measured half (2026-08-17) -------------------

def test_disk_usage_event_yields_allocated_used_and_slack():
    """The record the estimator has never had: what a job actually peaked at
    against what its launch bought."""
    r = ds.disk_usage_from_event({
        "job": "k4-qla", "workspace_size_mb": 51200,
        "workspace_free_mb": 30720, "workspace_used_mb": 20480,
        "box_high_water_mb": 22528, "job_dir_mb": 19456})
    assert r["allocated_gb"] == 50.0
    assert r["used_gb"] == 20.0
    assert r["high_water_gb"] == 22.0
    assert r["job_dir_gb"] == 19.0
    assert r["peak_gb"] == 22.0, "the high-water outranks the terminal snapshot"
    assert r["slack_gb"] == 28.0


def test_disk_usage_unknown_stays_unknown_never_zero():
    """jobd writes the literal string "unknown" when df could not answer.
    Coercing that to 0 would read as "the job needed nothing"."""
    r = ds.disk_usage_from_event({
        "job": "j", "workspace_size_mb": "unknown",
        "workspace_used_mb": "unknown", "box_high_water_mb": "unknown",
        "job_dir_mb": "unknown"})
    assert r["allocated_gb"] is None and r["used_gb"] is None
    assert r["peak_gb"] is None and r["slack_gb"] is None


def test_disk_usage_peak_survives_a_missing_high_water():
    """A job that finished inside one sample interval has no high-water file;
    the terminal snapshot must still produce a usable record."""
    r = ds.disk_usage_from_event({
        "workspace_size_mb": 38912, "workspace_used_mb": 20480,
        "box_high_water_mb": "unknown"})
    assert r["peak_gb"] == 20.0
    assert r["slack_gb"] == 18.0


def test_disk_usage_over_allocation_is_reported_not_clamped():
    """Overlay accounting can report usage above the nominal quota. A negative
    slack is a fact worth seeing, not something to floor at 0."""
    r = ds.disk_usage_from_event(
        {"workspace_size_mb": 51200, "workspace_used_mb": 52224})
    assert r["slack_gb"] == -1.0


def test_a_REAL_jobd_disk_usage_event_round_trips():
    """Same anti-drift guard as the scratch_probe case: built through
    jobmeta.make_box_event so jobd.py's int coercion is in the path."""
    import jobmeta
    ev = jobmeta.make_box_event(
        "12345", "disk_usage", job="k4-qla", phase="terminal",
        workspace_fs="overlay", workspace_size_mb=51200,
        workspace_free_mb=30720, workspace_used_mb=20480,
        box_high_water_mb=22528, job_dir_mb=19456)
    assert ev["event"] == "disk_usage", "the reader filters on this"
    r = ds.disk_usage_from_event(ev)
    assert r["job"] == "k4-qla" and r["allocated_gb"] == 50.0
    assert r["slack_gb"] == 28.0


# --- the blind spot that is NOT an unknown ---------------------------------- #
# `complete` is about ASSETS. It says nothing about what the entrypoint writes,
# and until 2026-08-25 nothing said that out loud: screenv1-e3-evals sized at a
# `complete: true` 40 G, every asset measured, and died on its own pre-merge
# guard because a two-stage LoRA merge holds ~2x the base on disk. The rent, the
# ~19 GB image pull, the base pull, a vLLM boot and a 12/12 positive control were
# all paid before the refusal.

def test_complete_does_not_mean_the_number_is_the_requirement():
    """Every asset sized, nothing failed — and the job still writes a merged
    model the estimate never counted. The two facts must be separately
    readable."""
    _gb, b = ds.estimate_disk_gb(cfg([A("base")]), {"base": 18 * GB})
    assert b["complete"] is True
    assert b["scratch_declared"] is False


def test_declaring_either_disk_knob_answers_the_scratch_question():
    """scratch_gb is the additive channel and disk_gb the override, but both are
    the author stating what the job needs — neither leaves the hole."""
    for needs in ({"scratch_gb": 42}, {"disk_gb": 130}):
        _gb, b = ds.estimate_disk_gb(cfg([A("base")], needs=needs),
                                     {"base": 18 * GB})
        assert b["scratch_declared"] is True, needs


def test_the_breakdown_names_the_missing_declaration_and_the_fix():
    """A reader deciding whether to trust the number needs the blind spot
    printed next to it, and the key to fix it by name."""
    _gb, b = ds.estimate_disk_gb(cfg([A("base")]), {"base": 18 * GB})
    text = "\n".join(ds.format_breakdown(b))
    assert "needs.scratch_gb" in text and "ENTRYPOINT" in text


def test_the_breakdown_stays_quiet_once_scratch_is_declared():
    """No nagging: the line is a hole report, not a banner. A bundle that has
    answered must not keep being asked."""
    _gb, b = ds.estimate_disk_gb(cfg([A("base")], needs={"scratch_gb": 42}),
                                 {"base": 18 * GB})
    assert "needs.scratch_gb" not in "\n".join(ds.format_breakdown(b))


def test_the_total_is_labelled_a_floor_not_a_recommendation():
    """Every unresolved term above the rule can only push the real need UP, so
    the total is a lower bound. `recommended` is the word an operator took at
    face value into a box 40 G too small."""
    _gb, b = ds.estimate_disk_gb(cfg([A("base")]), {"base": 18 * GB})
    assert "FLOOR" in ds.format_breakdown(b)[-1]


def test_scratch_is_additive_so_the_estimate_still_tracks_the_assets():
    """The reason a bundle should reach for scratch_gb over disk_gb: the derived
    terms keep moving underneath the declaration. disk_gb replaces them and
    would silently keep sizing for the old base."""
    c = cfg([A("base")], needs={"scratch_gb": 42})
    small = ds.estimate_disk_gb(c, {"base": 18 * GB})[0]
    big = ds.estimate_disk_gb(c, {"base": 36 * GB})[0]
    assert big > small
    frozen = cfg([A("base")], needs={"disk_gb": 80})
    assert (ds.estimate_disk_gb(frozen, {"base": 18 * GB})[0]
            == ds.estimate_disk_gb(frozen, {"base": 36 * GB})[0])
