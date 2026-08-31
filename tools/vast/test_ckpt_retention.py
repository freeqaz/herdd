"""Portable tests for the B2 intermediate-checkpoint retention sweep.

This file is weighted deliberately: most of it is about what the sweep must
NEVER do. B2 deletes are irreversible and two of the failure modes are
catastrophic rather than annoying —

  * deleting `checkpoints/<RUN_NAME>/`, the PUBLISHED FINAL ADAPTERS (the actual
    product of every training run we have ever done), and
  * deleting a live or requeue-able job's `jobs/<JOB_ID>/checkpoints/`, which IS
    that job's resume state.

So `_assert_sweepable` is exercised from every angle a path could arrive from,
`delete_objects` is asserted to delete NOTHING when any single path in a plan is
unsafe, and each of the five gates is asserted to KEEP on an unknown answer
("verification unavailable => do nothing").

Toolchain-free: no B2, no rclone — the transport is an injected runner.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckpt_retention as R      # noqa: E402

NOW = 1_800_000_000.0
DAY = 86400.0
GB = 1_000_000_000

CKPT_FILE_SIZES = {"adapter_model.safetensors": 323_000_000,
                   "optimizer.pt": 646_000_000,
                   "tokenizer.json": 11_000_000}
CKPT_BYTES = sum(CKPT_FILE_SIZES.values())


def _objs(job="J1", steps=(50,), age_days=30.0, root="out"):
    """The REAL bucket layout: jobs/<JOB_ID>/checkpoints/out/checkpoint-<N>/...

    jobd pushes the job's whole `work/` tree, so the training output dir sits
    between `checkpoints/` and `checkpoint-<N>`. Measured 2026-08-05: 22,577 of
    22,898 live objects have this shape and ZERO have `checkpoints/checkpoint-N`.
    Fixtures that used the flat shape hid a keying bug that made every job look
    like it had no steps at all — i.e. nothing for --keep-first/--keep-last to
    protect."""
    out = []
    pre = f"{root}/" if root else ""
    for s in steps:
        for name, size in CKPT_FILE_SIZES.items():
            out.append(R.CkptObj(
                f"jobs/{job}/checkpoints/{pre}checkpoint-{s}/{name}",
                size, NOW - age_days * DAY))
    return out


def _classify(**kw):
    kw.setdefault("job_id", "J1")
    kw.setdefault("objects", _objs())
    kw.setdefault("status", "done")
    kw.setdefault("published", True)
    kw.setdefault("now", NOW)
    return R.classify_job(**kw)


# --------------------------------------------------------------------------- #
# _assert_sweepable — the blast door
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", [
    # THE one that must never be reachable: the published final adapters.
    "checkpoints/20260805-v7-dec/adapter_model.safetensors",
    "checkpoints/20260805-v7-dec/PUBLISHED.json",
    "/checkpoints/run/adapter.safetensors",
    # other bucket tenants
    "adapters/x/adapter_model.safetensors",
    "runs/RID/events/e.json",
    "base-models/qwen/model.safetensors",
    "eval-fixtures/kr1/rows.jsonl",
    # inside jobs/, but NOT intermediate checkpoints
    "jobs/J1/results/gens.jsonl",
    "jobs/J1/results.DONE.json",
    "jobs/J1/events/e.json",
    "jobs/J1/log.txt",
    "jobs/queue/4685/J1.json",
    "jobs/nodes/4685/JOBD_STATUS",
    "jobs/bundles/abc.tar.zst",
    # shape violations
    "jobs/J1/checkpoints/",          # a prefix, not an object
    "jobs/J1/checkpoints",           # ditto
    "jobs//checkpoints/x",           # empty job id
    "jobs/J1/../../checkpoints/x",   # traversal
    "",
    None,
])
def test_assert_sweepable_refuses_everything_that_is_not_an_intermediate(path):
    with pytest.raises(R.UnsafePath):
        R._assert_sweepable(path)


@pytest.mark.parametrize("path", [
    "jobs/J1/checkpoints/checkpoint-50/optimizer.pt",
    "jobs/J1/checkpoints/checkpoint-50/extra/shard.bin",
    "jobs/20260710T103032-qwen4b/checkpoints/trainer_state.json",
    "/jobs/J1/checkpoints/checkpoint-50/optimizer.pt",
])
def test_assert_sweepable_accepts_real_intermediates(path):
    assert R._assert_sweepable(path).startswith("jobs/")


def test_published_adapter_prefix_is_not_merely_excluded_but_unreachable():
    """`checkpoints/<RUN_NAME>/` fails the check no matter how it is spelled —
    there is no encoding of it that reaches a delete call."""
    for p in ("checkpoints/r/a.bin", "./checkpoints/r/a.bin",
              "jobs/../checkpoints/r/a.bin"):
        with pytest.raises(R.UnsafePath):
            R._assert_sweepable(p)


def test_ckpt_step_and_job_id_go_through_the_same_door():
    assert R.ckpt_step("jobs/J1/checkpoints/checkpoint-50/a.bin") == 50
    assert R.ckpt_step("jobs/J1/checkpoints/trainer_state.json") is None
    assert R.job_id_of("jobs/J1/checkpoints/checkpoint-50/a.bin") == "J1"
    with pytest.raises(R.UnsafePath):
        R.ckpt_step("checkpoints/RUN/a.bin")


# --------------------------------------------------------------------------- #
# gate 1 — terminal and NON-resumable
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", ["queued", "claimed", "started", "running",
                                    "resumed", "reopened"])
def test_live_job_is_never_swept(status):
    vd = _classify(status=status)
    assert vd.action == "keep_live"
    assert vd.paths == () and vd.bytes == 0


def test_terminal_FAILED_is_kept_because_requeue_resumes_from_this_prefix():
    vd = _classify(status="failed")
    assert vd.action == "keep_resumable"
    assert "requeue" in vd.reason


def test_include_failed_opts_in_explicitly():
    vd = _classify(status="failed", include_failed=True)
    assert vd.action == "sweep"


@pytest.mark.parametrize("status", ["done", "cancelled"])
def test_done_and_cancelled_are_the_only_default_candidates(status):
    assert _classify(status=status).action == "sweep"


def test_unknown_status_is_kept():
    """'could not read the event stream' is the one state with no evidence at
    all — and the sweep resolves no-evidence to KEEP."""
    vd = _classify(status=None)
    assert vd.action == "keep_unknown_status"


# --------------------------------------------------------------------------- #
# gate 2 — a published artifact must exist
# --------------------------------------------------------------------------- #
def test_unpublished_job_is_never_swept():
    vd = _classify(published=False)
    assert vd.action == "keep_unpublished"
    assert "nowhere else" in vd.reason


def test_unverifiable_publication_is_kept():
    assert _classify(published=None).action == "keep_unpublished"


def test_is_published_requires_marker_AND_nonempty_results():
    """jobd writes results.DONE.json even when publish-verify FAILED, so the
    marker alone can sit over an empty results tree."""
    def runner(args):
        if args[0] == "lsf" and args[-1].endswith("results.DONE.json"):
            return 0, "results.DONE.json\n", ""
        if args[0] == "lsf" and args[-1].endswith("/results/"):
            return 0, "", ""            # marker present, nothing published
        return 1, "", "not found"
    assert R.is_published("bkt", "J1", runner=runner) is False


def test_is_published_true_on_marker_plus_results():
    def runner(args):
        if args[-1].endswith("results.DONE.json"):
            return 0, "results.DONE.json\n", ""
        if args[-1].endswith("/results/"):
            return 0, "gens.jsonl\n", ""
        return 1, "", "not found"
    assert R.is_published("bkt", "J1", runner=runner) is True


def test_is_published_true_on_the_PUBLISHED_json_marker():
    calls = []

    def runner(args):
        calls.append(args)
        if args[-1].endswith("checkpoints/RUN7/PUBLISHED.json"):
            return 0, "PUBLISHED.json\n", ""
        return 1, "", "not found"
    assert R.is_published("bkt", "J1", run_name="RUN7", runner=runner) is True
    # probed by EXACT KEY — the published-adapter prefix is never enumerated
    assert all("-R" not in c for c in calls)


def test_is_published_None_on_a_transport_error_not_a_miss():
    """A network blip must not read as 'no published artifact' — that would flip
    a KEEP into a SWEEP."""
    assert R.is_published("bkt", "J1",
                          runner=lambda a: (1, "", "connection reset")) is None


# --------------------------------------------------------------------------- #
# gate 3 — age window
# --------------------------------------------------------------------------- #
def test_young_checkpoints_are_kept():
    vd = _classify(objects=_objs(age_days=2.0), window_days=7.0)
    assert vd.action == "keep_young"


def test_window_is_a_tunable_not_a_constant():
    objs = _objs(age_days=10.0)
    assert _classify(objects=objs, window_days=7.0).action == "sweep"
    assert _classify(objects=objs, window_days=30.0).action == "keep_young"


def test_age_is_measured_on_the_NEWEST_object():
    """One recently-touched object keeps the whole job: a job still being written
    to is not old, however old its earliest bytes are."""
    objs = _objs(steps=(50,), age_days=90.0) + _objs(steps=(100,), age_days=1.0)
    assert _classify(objects=objs, window_days=7.0).action == "keep_young"


# --------------------------------------------------------------------------- #
# keep policy — the dose curve
# --------------------------------------------------------------------------- #
def test_keep_first_preserves_the_dose_curve_head():
    """v4's dose curve was U-shaped: the EARLY steps carry the signal, so
    keep-first is a first-class knob, not an afterthought behind keep-last."""
    assert R.steps_to_keep([10, 20, 30, 40], keep_first=2) == {10, 20}
    assert R.steps_to_keep([10, 20, 30, 40], keep_last=1) == {40}
    assert R.steps_to_keep([10, 20, 30, 40], keep_stride=2) == {10, 30}
    assert R.steps_to_keep([10, 20, 30, 40], keep_first=1,
                           keep_last=1) == {10, 40}


def test_kept_steps_are_excluded_from_the_delete_list():
    objs = _objs(steps=(10, 20, 30), age_days=30.0)
    vd = _classify(objects=objs, keep_first=1, keep_last=1)
    assert vd.action == "sweep"
    assert vd.kept_steps == (("out", 10), ("out", 30))
    assert all("checkpoint-20/" in p for p in vd.paths)
    assert vd.bytes == CKPT_BYTES


def test_ckpt_key_finds_the_checkpoint_at_ANY_depth():
    """The real layout nests the training output dir in between. A fixed-depth
    key finds nothing on the live bucket, and a job that appears to have zero
    steps has nothing for --keep-first/--keep-last to protect."""
    assert R.ckpt_key("jobs/J1/checkpoints/out/checkpoint-50/optimizer.pt") == \
        ("out", 50)
    assert R.ckpt_key("jobs/J1/checkpoints/arms/hex/checkpoint-50/a.bin") == \
        ("arms/hex", 50)
    assert R.ckpt_key("jobs/J1/checkpoints/checkpoint-50/a.bin") == ("", 50)
    assert R.ckpt_key("jobs/J1/checkpoints/out/trainer_state.json") == \
        (None, None)


def test_keep_policy_is_applied_PER_LAYOUT_ROOT():
    """A multi-arm bundle writes arms/a/checkpoint-N and arms/b/checkpoint-N.
    Merging them would let --keep-last 1 protect one arm's newest step and
    delete the other's, which is not what 'keep the last checkpoint' means."""
    objs = (_objs(steps=(10, 20), age_days=30.0, root="arms/a")
            + _objs(steps=(10, 20), age_days=30.0, root="arms/b"))
    vd = _classify(objects=objs, keep_last=1)
    assert set(vd.kept_steps) == {("arms/a", 20), ("arms/b", 20)}
    assert all("checkpoint-10/" in p for p in vd.paths)
    assert len(vd.paths) == 2 * len(CKPT_FILE_SIZES)


def test_an_object_outside_any_checkpoint_dir_is_never_protected():
    """A stray file directly under checkpoints/ has no step, so no keep policy
    covers it — it is swept with the rest rather than silently retained."""
    objs = _objs(steps=(10,), age_days=30.0) + [
        R.CkptObj("jobs/J1/checkpoints/out/trainer_state.json", 100,
                  NOW - 30 * DAY)]
    vd = _classify(objects=objs, keep_last=1)
    assert vd.action == "sweep"
    assert vd.paths == ("jobs/J1/checkpoints/out/trainer_state.json",)


def test_keeping_everything_yields_nothing_to_do():
    vd = _classify(objects=_objs(steps=(10, 20), age_days=30.0), keep_stride=1)
    assert vd.action == "nothing"
    assert vd.paths == ()


def test_no_objects_is_a_no_op():
    assert _classify(objects=[]).action == "nothing"


# --------------------------------------------------------------------------- #
# delete_objects — the last door before an irreversible call
# --------------------------------------------------------------------------- #
def test_delete_objects_deletes_nothing_if_ANY_path_is_unsafe():
    """A plan containing one unsafe path is a plan we do not understand, and
    partially executing a plan we do not understand is the worst option."""
    calls = []
    paths = ["jobs/J1/checkpoints/checkpoint-50/a.bin",
             "checkpoints/PUBLISHED_RUN/adapter_model.safetensors",
             "jobs/J1/checkpoints/checkpoint-50/b.bin"]
    with pytest.raises(R.UnsafePath):
        R.delete_objects("bkt", paths,
                         runner=lambda a: calls.append(a) or (0, "", ""),
                         log=lambda *a: None)
    assert calls == []


def test_delete_objects_issues_one_deletefile_per_object():
    calls = []
    d, f = R.delete_objects(
        "bkt", ["jobs/J1/checkpoints/checkpoint-50/a.bin"],
        runner=lambda a: calls.append(a) or (0, "", ""), log=lambda *a: None)
    assert calls == [["deletefile", "b2:bkt/jobs/J1/checkpoints/checkpoint-50/a.bin"]]
    assert d and not f


def test_delete_objects_never_uses_purge_or_a_recursive_delete():
    calls = []
    R.delete_objects("bkt", ["jobs/J1/checkpoints/checkpoint-50/a.bin"],
                     runner=lambda a: calls.append(a) or (0, "", ""),
                     log=lambda *a: None)
    flat = " ".join(" ".join(c) for c in calls)
    for banned in ("purge", "rmdir", "delete ", "-R", "--rmdirs"):
        assert banned not in flat


def test_delete_objects_reports_failures_rather_than_claiming_success():
    d, f = R.delete_objects("bkt", ["jobs/J1/checkpoints/checkpoint-50/a.bin"],
                            runner=lambda a: (1, "", "403"),
                            log=lambda *a: None)
    assert d == [] and len(f) == 1


# --------------------------------------------------------------------------- #
# sweep() — dry-run is the default
# --------------------------------------------------------------------------- #
def _sweep_verdicts():
    return [R.Verdict("J1", "sweep", "old+published", CKPT_BYTES,
                      ("jobs/J1/checkpoints/checkpoint-50/a.bin",), "done",
                      (50,), (), NOW - 30 * DAY),
            R.Verdict("J2", "keep_live", "running", 0, (), "running", (10,), (),
                      NOW)]


def test_sweep_is_dry_run_unless_apply():
    calls = []
    out = R.sweep("bkt", _sweep_verdicts(),
                  runner=lambda a: calls.append(a) or (0, "", ""),
                  log=lambda *a: None)
    assert out["applied"] is False
    assert calls == []
    assert out["objects"] == 1


def test_sweep_apply_only_touches_the_sweep_verdicts():
    calls = []
    out = R.sweep("bkt", _sweep_verdicts(), apply=True,
                  runner=lambda a: calls.append(a) or (0, "", ""),
                  log=lambda *a: None)
    assert out["applied"] is True and out["deleted"] == 1
    assert len(calls) == 1
    assert "J2" not in calls[0][-1]


def test_sweep_apply_refuses_a_plan_with_an_unsafe_path():
    bad = [R.Verdict("J1", "sweep", "x", 1, ("checkpoints/RUN/a.bin",), "done",
                     (50,), (), NOW)]
    calls = []
    with pytest.raises(R.UnsafePath):
        R.sweep("bkt", bad, apply=True,
                runner=lambda a: calls.append(a) or (0, "", ""),
                log=lambda *a: None)
    assert calls == []


# --------------------------------------------------------------------------- #
# listing + build_verdicts — "verification unavailable => do nothing"
# --------------------------------------------------------------------------- #
def test_listing_is_anchored_so_published_adapters_are_never_enumerated():
    seen = {}

    def runner(args):
        seen["args"] = args
        return 0, "[]", ""
    R.list_checkpoint_objects("bkt", runner=runner)
    assert "--include" in seen["args"]
    assert seen["args"][seen["args"].index("--include") + 1] == "/*/checkpoints/**"
    assert seen["args"][-1] == "b2:bkt/jobs/"


def test_listing_error_makes_build_verdicts_do_NOTHING():
    vds, err = R.build_verdicts("bkt", now=NOW,
                                runner=lambda a: (1, "", "b2 unreachable"))
    assert vds == [] and "unreachable" in err


def test_unparseable_listing_is_an_error_not_an_empty_bucket():
    jobs, err = R.list_checkpoint_objects("bkt",
                                          runner=lambda a: (0, "not json", ""))
    assert jobs == {} and err


def test_listing_drops_rows_that_are_not_intermediate_checkpoints():
    rows = ('[{"Path":"J1/checkpoints/checkpoint-50/a.bin","Size":10,'
            '"ModTime":"2026-07-01T00:00:00Z"},'
            '{"Path":"J1/results/gens.jsonl","Size":99,'
            '"ModTime":"2026-07-01T00:00:00Z"}]')
    jobs, err = R.list_checkpoint_objects("bkt", runner=lambda a: (0, rows, ""))
    assert err is None
    assert list(jobs) == ["J1"]
    assert [o.path for o in jobs["J1"]] == \
        ["jobs/J1/checkpoints/checkpoint-50/a.bin"]


def test_build_verdicts_uses_injected_status_and_publication():
    jobs = {"J1": _objs(job="J1", age_days=30.0)}
    vds, err = R.build_verdicts("bkt", now=NOW, jobs=jobs,
                                status_of=lambda j: "done",
                                published_of=lambda j: True)
    assert err is None and vds[0].action == "sweep"


def test_build_verdicts_skips_the_publication_probe_for_a_live_job():
    """Two B2 round trips per probe, and the answer is irrelevant for a job whose
    status already keeps it."""
    probed = []
    jobs = {"J1": _objs(job="J1", age_days=30.0)}
    R.build_verdicts("bkt", now=NOW, jobs=jobs, status_of=lambda j: "running",
                     published_of=lambda j: probed.append(j) or True)
    assert probed == []


def test_job_status_reads_no_events_as_UNKNOWN_not_as_terminal():
    """`jobmeta.read_job_events` returns [] for both 'unreadable' and 'empty'.
    For a deletion decision those must not look alike."""
    assert R.job_status("bkt", "J1", events=[]) is None


def test_job_status_reports_a_reopened_job_as_reopened():
    import jobmeta
    orig = jobmeta.fold_events
    try:
        jobmeta.fold_events = lambda evs, *a, **k: {"status": "done",
                                                    "reopened": True}
        assert R.job_status("bkt", "J1", events=[{"event": "x"}]) == "reopened"
    finally:
        jobmeta.fold_events = orig


def test_run_name_of_pulls_the_strongest_publication_witness():
    assert R.run_name_of([{"event": "submitted",
                           "config_echo": {"run_name": "RUN7"}}]) == "RUN7"
    assert R.run_name_of([{"event": "x", "run_name": "RUN8"}]) == "RUN8"
    assert R.run_name_of([{"event": "x"}]) is None
    assert R.run_name_of(None) is None


# --------------------------------------------------------------------------- #
# end-to-end through the real gates, with only the transport faked
# --------------------------------------------------------------------------- #
def _bucket_runner(*, status_events, done=True, results=True):
    """An rclone stand-in for one job J1 with two old checkpoints."""
    listing = ('[' + ",".join(
        f'{{"Path":"J1/checkpoints/checkpoint-{s}/{n}","Size":{sz},'
        f'"ModTime":"2026-06-01T00:00:00Z"}}'
        for s in (50, 100) for n, sz in CKPT_FILE_SIZES.items()) + ']')

    def runner(args):
        tail = args[-1]
        if args[0] == "lsjson":
            return 0, listing, ""
        if args[0] == "lsf" and tail.endswith("/events/"):
            return 0, "e1.json\n", ""
        if args[0] == "cat" and "events/e1.json" in tail:
            import json as _j
            return 0, _j.dumps(status_events), ""
        if args[0] == "lsf" and tail.endswith("results.DONE.json"):
            return (0, "results.DONE.json\n", "") if done else (1, "", "not found")
        if args[0] == "lsf" and tail.endswith("/results/"):
            return (0, "gens.jsonl\n", "") if results else (0, "", "")
        return 1, "", "not found"
    return runner


def test_end_to_end_done_and_published_sweeps():
    runner = _bucket_runner(status_events={"event": "done", "ts": 1,
                                           "job_id": "J1", "actor": "box"})
    vds, err = R.build_verdicts("bkt", now=NOW, window_days=7.0, runner=runner)
    assert err is None
    assert [vd.action for vd in vds] == ["sweep"]
    assert all(p.startswith("jobs/J1/checkpoints/") for p in vds[0].paths)


def test_end_to_end_done_but_UNPUBLISHED_keeps():
    runner = _bucket_runner(status_events={"event": "done", "ts": 1,
                                           "job_id": "J1", "actor": "box"},
                            done=False)
    vds, _ = R.build_verdicts("bkt", now=NOW, window_days=7.0, runner=runner)
    assert vds[0].action == "keep_unpublished"


def test_inventory_is_read_only():
    """The inventory command must not be able to delete anything, ever."""
    calls = []

    def runner(args):
        calls.append(args[0])
        return _bucket_runner(status_events={"event": "done", "ts": 1,
                                             "job_id": "J1",
                                             "actor": "box"})(args)
    rep = R.inventory("bkt", now=NOW, windows=(7.0, 30.0), runner=runner)
    assert rep["total_bytes"] == 2 * CKPT_BYTES
    assert set(rep["windows"]) == {"7.0", "30.0"}
    assert "deletefile" not in calls and "purge" not in calls
    assert rep["monthly_usd"] >= 0


def test_inventory_probes_status_once_per_job_across_windows():
    """Three candidate windows must not cost three passes of B2 probes."""
    seen, pubs = [], []
    runner = _bucket_runner(status_events={"event": "done", "ts": 1,
                                           "job_id": "J1", "actor": "box"})
    rep = R.inventory("bkt", now=NOW, windows=(7.0, 14.0, 30.0), runner=runner,
                      status_of=lambda j: seen.append(j) or "done",
                      published_of=lambda j: pubs.append(j) or True)
    assert seen == ["J1"] and pubs == ["J1"]
    assert rep["status_counts"] == {"done": 1}
    for w in ("7.0", "14.0", "30.0"):
        assert rep["windows"][w]["sweep_jobs"] == 1


def test_main_plan_is_never_destructive(monkeypatch, capsys):
    monkeypatch.setattr(R, "build_verdicts",
                        lambda *a, **k: (_sweep_verdicts(), None))
    calls = []
    monkeypatch.setattr(R, "_rclone", lambda args, runner=None:
                        calls.append(args) or (0, "", ""))
    assert R.main(["plan", "--bucket", "bkt"]) == 0
    assert calls == []


def test_main_sweep_without_apply_deletes_nothing(monkeypatch):
    monkeypatch.setattr(R, "build_verdicts",
                        lambda *a, **k: (_sweep_verdicts(), None))
    calls = []
    monkeypatch.setattr(R, "_rclone", lambda args, runner=None:
                        calls.append(args) or (0, "", ""))
    assert R.main(["sweep", "--bucket", "bkt"]) == 0
    assert calls == []


def test_main_refuses_to_act_when_the_plan_could_not_be_built(monkeypatch):
    monkeypatch.setattr(R, "build_verdicts", lambda *a, **k: ([], "b2 down"))
    calls = []
    monkeypatch.setattr(R, "_rclone", lambda args, runner=None:
                        calls.append(args) or (0, "", ""))
    assert R.main(["sweep", "--bucket", "bkt", "--apply"]) == 1
    assert calls == []


def test_parse_ts_shapes():
    assert R._parse_ts("2026-06-01T00:00:00Z") > 0
    assert R._parse_ts("2026-06-01T00:00:00.123456789Z") > 0
    assert R._parse_ts("") is None
    assert R._parse_ts("nonsense") is None


# --------------------------------------------------------------------------- #
# gate 5 — the box prune makes checkpoints/ the SOLE copy
# --------------------------------------------------------------------------- #
def test_box_pruned_job_is_never_swept():
    """Since 2026-08-05 jobd prunes checkpoint dirs off the BOX disk after
    verifying them on B2. `checkpoints:` and `results:` are the same `out/**`
    glob, so a pruned job's results/ is NOT a second copy of the grid and this
    prefix holds the only copy of the pruned steps (CHECKPOINT_LIFECYCLE.md)."""
    vd = _classify(pruned=True)
    assert vd.action == "keep_box_pruned"
    assert vd.paths == ()


def test_box_pruned_opt_in_without_a_keep_policy_is_still_kept():
    """The opt-in alone is not enough. A marked prefix is the SOLE copy of the
    pruned steps, so sweeping it with an empty keep policy deletes the dose
    curve outright — the one combination that loses data with nothing left."""
    vd = _classify(pruned=True, sweep_box_pruned=True)
    assert vd.action == "keep_box_pruned_no_policy"
    assert vd.paths == ()


@pytest.mark.parametrize("policy", [
    {"keep_first": 1}, {"keep_last": 1}, {"keep_stride": 2},
])
def test_box_pruned_sweeps_only_with_an_explicit_skeleton(policy):
    vd = _classify(objects=_objs(steps=(50, 100, 150, 200)),
                   pruned=True, sweep_box_pruned=True, **policy)
    assert vd.action == "sweep"
    assert vd.kept_steps, \
        "a swept sole-copy prefix must retain a dose-curve skeleton"
    assert vd.paths, "nothing was actually selected for deletion"


def test_box_pruned_opt_in_does_not_bypass_the_earlier_gates():
    """--sweep-box-pruned relaxes gate 5 and nothing else: an unpublished or
    still-live job stays kept however the flag is set."""
    assert _classify(pruned=True, sweep_box_pruned=True, keep_first=1,
                     published=False).action == "keep_unpublished"
    assert _classify(pruned=True, sweep_box_pruned=True, keep_first=1,
                     status="running").action == "keep_live"


def test_unknown_prune_marker_is_kept():
    assert _classify(pruned=None).action == "keep_pruned_unknown"
    # the opt-in is about a KNOWN marker; an unreadable one is still unknown
    assert _classify(pruned=None, sweep_box_pruned=True,
                     keep_first=1).action == "keep_pruned_unknown"


def test_unpruned_job_still_sweeps():
    assert _classify(pruned=False).action == "sweep"


def test_is_box_pruned_trichotomy():
    hit = lambda a: (0, "CHECKPOINTS_PRUNED.json\n", "")        # noqa: E731
    miss = lambda a: (1, "", "directory not found")             # noqa: E731
    blip = lambda a: (1, "", "connection reset by peer")        # noqa: E731
    assert R.is_box_pruned("bkt", "J1", runner=hit) is True
    assert R.is_box_pruned("bkt", "J1", runner=miss) is False
    assert R.is_box_pruned("bkt", "J1", runner=blip) is None


def test_build_verdicts_probes_the_prune_marker_before_sweeping():
    probed = []
    jobs = {"J1": _objs(job="J1", age_days=30.0)}
    vds, _ = R.build_verdicts("bkt", now=NOW, jobs=jobs,
                              status_of=lambda j: "done",
                              published_of=lambda j: True,
                              pruned_of=lambda j: probed.append(j) or True)
    assert probed == ["J1"]
    assert vds[0].action == "keep_box_pruned"


def test_end_to_end_pruned_job_keeps(monkeypatch):
    """Through the REAL gates with only the transport faked."""
    base = _bucket_runner(status_events={"event": "done", "ts": 1,
                                         "job_id": "J1", "actor": "box"})

    def runner(args):
        if args[-1].endswith("CHECKPOINTS_PRUNED.json"):
            return 0, "CHECKPOINTS_PRUNED.json\n", ""
        return base(args)
    vds, _ = R.build_verdicts("bkt", now=NOW, window_days=7.0, runner=runner)
    assert vds[0].action == "keep_box_pruned"


# --------------------------------------------------------------------------- #
# gate 3 — an UNREADABLE age must not read as "maximally old"
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("modtime", [
    None,                                   # key absent from the lsjson row
    "",                                     # present but empty
    "Wed, 05 Aug 2026 07:00:00 GMT",        # a shape _parse_ts cannot read
    "nonsense",
])
def test_an_unreadable_ModTime_is_KEPT_not_treated_as_1970(modtime):
    """Mapping an unparseable timestamp to epoch 0 makes it older than every
    window, so gate 3 passes and objects written MINUTES ago get deleted. It is
    the one unknown in this module that would resolve to DELETE."""
    row = {"Path": "J1/checkpoints/out/checkpoint-50/optimizer.pt", "Size": 10}
    if modtime is not None:
        row["ModTime"] = modtime
    import json as _j
    jobs, err = R.list_checkpoint_objects(
        "bkt", runner=lambda a: (0, _j.dumps([row]), ""))
    assert err is None
    assert jobs["J1"][0].mtime is None       # preserved, not coerced
    vd = R.classify_job(job_id="J1", objects=jobs["J1"], status="done",
                        published=True, pruned=False, now=NOW, window_days=7.0)
    assert vd.action == "keep_unknown_age"
    assert vd.paths == ()


def test_one_unaged_object_keeps_the_WHOLE_job():
    objs = _objs(steps=(10,), age_days=90.0) + [
        R.CkptObj("jobs/J1/checkpoints/out/checkpoint-20/a.bin", 5, None)]
    assert _classify(objects=objs, pruned=False).action == "keep_unknown_age"


def test_a_valid_ModTime_still_flows_through():
    import json as _j
    row = {"Path": "J1/checkpoints/out/checkpoint-50/optimizer.pt", "Size": 10,
           "ModTime": "2020-01-01T00:00:00Z"}
    jobs, _ = R.list_checkpoint_objects(
        "bkt", runner=lambda a: (0, _j.dumps([row]), ""))
    assert jobs["J1"][0].mtime > 0
    assert R.classify_job(job_id="J1", objects=jobs["J1"], status="done",
                          published=True, pruned=False, now=NOW,
                          window_days=7.0).action == "sweep"


def test_inventory_tolerates_an_unaged_job():
    import json as _j
    rows = _j.dumps([{"Path": "J1/checkpoints/out/checkpoint-50/a.bin",
                      "Size": 10}])
    rep = R.inventory("bkt", now=NOW, windows=(7.0,),
                      runner=lambda a: (0, rows, ""),
                      status_of=lambda j: "done", published_of=lambda j: True,
                      pruned_of=lambda j: False)
    assert rep["per_job"]["J1"]["age_days"] is None
    assert rep["age_bytes"]["unknown"] == 10
    assert rep["windows"]["7.0"]["sweep_jobs"] == 0


def test_assert_sweepable_refuses_a_dot_segment():
    with pytest.raises(R.UnsafePath):
        R._assert_sweepable("jobs/./checkpoints/RUN/a.bin")
    with pytest.raises(R.UnsafePath):
        R._assert_sweepable("jobs/J1/checkpoints/./a.bin")


# --------------------------------------------------------------------------- #
# torn uploads must not occupy a --keep-* slot (2026-08-28)
# --------------------------------------------------------------------------- #
# B2 has no atomic directory rename, so an eviction mid-sync leaves a real
# `checkpoint-<N>/` prefix that is not resumable state. Letting it consume a
# `--keep-last 1` slot is a corruption evicting a good checkpoint — and this
# module's next step is an irreversible delete.

RESUMABLE = {"trainer_state.json": 60_000, "optimizer.pt": 646_000_000,
             "scheduler.pt": 1_000, "adapter_model.safetensors": 323_000_000}


def _ckpts(spec, job="J1", root="out", age_days=30.0):
    """`spec` maps step -> {filename: size}. Sizes are per-object, as on B2."""
    out = []
    pre = f"{root}/" if root else ""
    for step, files in spec.items():
        for name, size in files.items():
            out.append(R.CkptObj(
                f"jobs/{job}/checkpoints/{pre}checkpoint-{step}/{name}",
                size, NOW - age_days * DAY))
    return out


def _marker(step, job="J1", root="out", age_days=30.0):
    pre = f"{root}/" if root else ""
    return R.CkptObj(
        f"jobs/{job}/checkpoints/{pre}checkpoint-{step}.complete.json",
        400, NOW - age_days * DAY)


def test_a_torn_step_is_named_when_a_sibling_proves_the_layout():
    """The v16 shapes: a 2-object checkpoint-96 beside complete neighbours."""
    objs = _ckpts({160: RESUMABLE,
                   176: {"trainer_state.json": 60_000},
                   96: {k: v for k, v in RESUMABLE.items() if k != "optimizer.pt"}})
    assert R.incomplete_checkpoint_keys(o.path for o in objs) == \
        {("out", 176), ("out", 96)}


def test_a_marker_makes_a_step_complete_whatever_its_file_set():
    """Tier 1. jobd publishes the marker only after reading the exact directory
    back off B2, so it outranks our guess about the file set."""
    objs = _ckpts({176: {"trainer_state.json": 60_000}}) + [_marker(176)]
    objs += _ckpts({160: RESUMABLE})
    assert R.incomplete_checkpoint_keys(o.path for o in objs) == set()


def test_an_unmodelled_layout_keeps_its_protection():
    """SELF-CALIBRATING, and this is the safety property. No sibling holds the
    files these lack, so nothing is called incomplete — a checkpointer we do not
    model (or a caller handing us a filtered object list) must not have its keep
    protection stripped by a module whose next step is a delete."""
    objs = _ckpts({10: {"model.pt": 5}, 20: {"model.pt": 5}})
    assert R.incomplete_checkpoint_keys(o.path for o in objs) == set()


def test_calibration_is_PER_LAYOUT_ROOT():
    """arms/a's complete checkpoints say nothing about arms/b's layout."""
    objs = _ckpts({10: RESUMABLE}, root="arms/a")
    objs += _ckpts({10: {"model.pt": 5}}, root="arms/b")
    assert R.incomplete_checkpoint_keys(o.path for o in objs) == set()


def test_a_torn_step_does_not_consume_a_keep_last_slot():
    """The whole point. --keep-last 1 must protect checkpoint-160, not the
    2-object checkpoint-176 that happens to be the numeric max."""
    objs = _ckpts({100: RESUMABLE, 160: RESUMABLE,
                   176: {"trainer_state.json": 60_000}})
    vd = R.classify_job(job_id="J1", objects=objs, status="done", published=True,
                        now=NOW, keep_last=1)
    assert vd.action == "sweep"
    assert vd.kept_steps == (("out", 160),)
    assert any("checkpoint-176/" in p for p in vd.paths)
    assert not any("checkpoint-160/" in p for p in vd.paths)
    assert "withheld from the keep policy" in vd.reason


def test_a_kept_checkpoint_keeps_its_completion_marker():
    """`ckpt_key` looks for a checkpoint-<N> path SEGMENT and the marker is a
    SIBLING, so without `marker_key` the sweep would delete a kept checkpoint's
    proof of completeness and the next resume would fall back to guessing."""
    objs = _ckpts({100: RESUMABLE, 160: RESUMABLE}) + [_marker(100), _marker(160)]
    vd = R.classify_job(job_id="J1", objects=objs, status="done", published=True,
                        now=NOW, keep_last=1)
    assert vd.action == "sweep"
    assert not any("checkpoint-160.complete.json" in p for p in vd.paths)
    assert any("checkpoint-100.complete.json" in p for p in vd.paths)


def test_marker_key_only_matches_the_marker_spelling():
    assert R.marker_key("jobs/J1/checkpoints/out/checkpoint-96.complete.json") \
        == ("out", 96)
    for p in ("jobs/J1/checkpoints/out/checkpoint-96/trainer_state.json",
              "jobs/J1/checkpoints/out/checkpoint-abc.complete.json",
              "jobs/J1/checkpoints/out/notes.complete.json",
              "jobs/J1/checkpoints/out/checkpoint-96/other.complete.json"):
        assert R.marker_key(p) == (None, None), p


def test_bnb_skipped_counterparts_satisfy_the_contract():
    objs = _ckpts({10: RESUMABLE})
    objs += _ckpts({20: {"trainer_state.json": 60_000,
                         "optimizer.pt.bnb_skipped": 646_000_000,
                         "scheduler.pt.bnb_skipped": 1_000,
                         "adapter_model.safetensors": 323_000_000}})
    assert R.incomplete_checkpoint_keys(o.path for o in objs) == set()


def test_the_contract_matches_the_box_side_spelling():
    """Three languages, one contract (jobd.sh `_ckpt_names_complete`,
    train_proposer_lora.py `_CKPT_REQUIRED_FILES`, here). Pinned so an edit to
    one cannot silently diverge — a resume-side answer that disagrees with the
    write side is how a partial gets published or a complete one gets skipped."""
    jobd = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "onstart", "jobd.sh")).read()
    assert f'CKPT_COMPLETE_SUFFIX="{R.CKPT_COMPLETE_SUFFIX}"' in jobd
    for group in R.CKPT_REQUIRED_FILES:
        assert any(f'f["{n}"]' in jobd for n in group), group
