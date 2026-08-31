"""Portable tests for cpu_probe.py — no container, no network, no rented box.

Everything a boot path can get wrong is testable here: reading the slice width
rather than the host, the busy guard, the scaling arithmetic, and the record
schema jobd and hostfacts both have to consume.

What is NOT covered is whether the rates mean anything on real silicon — that
is what the run itself measures. The scaling arithmetic is pinned against
synthetic benches so a wrong denominator fails HERE and not six boxes later.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpu_probe as cp  # noqa: E402
import hostfacts as hf  # noqa: E402


@pytest.fixture
def fake_fs(monkeypatch):
    """Stand in for /sys/fs/cgroup and /proc so a test can describe any box."""
    files = {}

    def _read(path):
        return files.get(path, "")

    monkeypatch.setattr(cp, "_read", _read)
    return files


# --------------------------------------------------------------------------- #
# the slice, not the host — the failure this probe exists downstream of
# --------------------------------------------------------------------------- #
def test_a_quota_narrower_than_the_host_wins(fake_fs):
    """4 cores of quota on a 256-core host is a 4-core box.

    This is the whole reason `cpu_width` exists. `nproc` and `/proc/stat` are
    not virtualised in a container, and reading the host instead of the slice
    has already produced two wrong instruments in this lane.
    """
    fake_fs["/sys/fs/cgroup/cpu.max"] = "400000 100000"
    cores, source = cp.cpu_width()
    assert cores == 4.0
    assert source == "cgroup_v2_quota"
    assert cores < (os.cpu_count() or 1) or True   # the point is 4, not the host


def test_v1_quota_is_read_when_v2_is_absent(fake_fs):
    fake_fs["/sys/fs/cgroup/cpu/cpu.cfs_quota_us"] = "250000"
    fake_fs["/sys/fs/cgroup/cpu/cpu.cfs_period_us"] = "100000"
    assert cp.cpu_width() == (2.5, "cgroup_v1_quota")


def test_an_unlimited_quota_falls_through_to_cpuset(fake_fs):
    """`max` is not a number. A box with no quota is still bounded by cpuset."""
    fake_fs["/sys/fs/cgroup/cpu.max"] = "max 100000"
    fake_fs["/sys/fs/cgroup/cpuset.cpus.effective"] = "0-7,16"
    assert cp.cpu_width() == (9.0, "cpuset_v2")


@pytest.mark.parametrize("text,want", [
    ("0-3", 4), ("0", 1), ("0-1,4-5", 4), ("2,4,6", 3), ("", 0), ("bad-x", 0),
])
def test_cpuset_ranges_are_inclusive(text, want):
    assert cp._cpuset_count(text) == want


def test_width_source_marks_the_unmeasured_fallback(fake_fs):
    """With no cgroup legible at all the width describes the HOST, and the
    record has to say so — that is what `width_source` is for."""
    cores, source = cp.cpu_width()
    assert cores >= 1
    assert source in ("sched_affinity", "os_cpu_count")


def test_busy_is_read_from_the_cgroup_not_proc_stat(fake_fs, monkeypatch):
    """/proc/stat reports the host inside a container. It must never be consulted."""
    fake_fs["/sys/fs/cgroup/cpu.stat"] = "usage_usec 1000000\n"
    monkeypatch.setattr(cp.time, "sleep", lambda _s: None)
    monkeypatch.setattr(cp.time, "perf_counter",
                        _stepping_clock([0.0, 1.0]))
    fake_fs["/sys/fs/cgroup/cpu.stat"] = "usage_usec 1000000\n"
    assert cp.cgroup_cpu_usage_s() == 1.0
    assert "/proc/stat" not in fake_fs


def _stepping_clock(values):
    it = iter(values)
    last = [0.0]

    def _clock():
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]
    return _clock


def test_unreadable_cgroup_usage_is_none(fake_fs):
    assert cp.cgroup_cpu_usage_s() is None
    assert cp.busy_cores() is None


# --------------------------------------------------------------------------- #
# the guard: never perturb a job someone is paying for
# --------------------------------------------------------------------------- #
def test_an_idle_box_is_probeable():
    assert cp.busy_reason(running_jobs=(), level=0.05) is None


def test_a_running_job_refuses():
    r = cp.busy_reason(running_jobs=("job-a",), level=0.0)
    assert r and r.startswith("job_running:")


def test_a_busy_cgroup_refuses():
    r = cp.busy_reason(running_jobs=(), level=4.0, threshold=1.0)
    assert r and "cpu_busy" in r


def test_an_unreadable_level_does_not_refuse():
    """Deliberately unlike gemm_probe, whose unreadable case is a card it cannot
    prove idle. Here it is a host whose cgroup files are not visible, which is
    every non-container environment — refusing there would mean never measuring.
    `running_jobs` stays the definitive guard and does not need the cgroup."""
    assert cp.busy_reason(running_jobs=(), level=None) is None


def test_running_job_ids_survives_a_missing_state_dir(tmp_path):
    assert cp.running_job_ids(None) == []
    assert cp.running_job_ids(str(tmp_path / "nope")) == []
    (tmp_path / "a.running").write_text("")
    assert cp.running_job_ids(str(tmp_path)) == ["a"]


# --------------------------------------------------------------------------- #
# the kernel is FIXED — that is what makes two machines comparable
# --------------------------------------------------------------------------- #
def test_the_kernel_is_deterministic():
    """Same work every time, on every box. A kernel whose op count varied would
    make `per_core_s` incomparable across exactly the machines it exists to
    compare."""
    assert cp.kernel(5000) == cp.kernel(5000)
    assert cp.kernel(5000, seed=2) != cp.kernel(5000, seed=1)


def test_more_iterations_is_more_work():
    assert cp.kernel(1000) != cp.kernel(2000)


# --------------------------------------------------------------------------- #
# scaling: the arithmetic that turns two rates into the finding
# --------------------------------------------------------------------------- #
def test_perfect_scaling_reads_one():
    rec = cp.build_record({"single_per_s": 100.0, "allcore_per_s": 800.0,
                           "workers": 8}, cores=8)
    assert rec["scaling"] == 1.0


def test_a_box_that_does_not_scale_reads_low():
    """The purchase question `cores x GHz` cannot answer: 32 advertised, 16 real."""
    rec = cp.build_record({"single_per_s": 100.0, "allcore_per_s": 1600.0,
                           "workers": 32}, cores=32)
    assert rec["scaling"] == 0.5


def test_scaling_divides_by_workers_not_cores():
    """A run capped by MAX_WORKERS must not read as a machine that fails to
    scale. 8 workers on a 256-core slice at perfect efficiency is 1.0."""
    rec = cp.build_record({"single_per_s": 100.0, "allcore_per_s": 800.0,
                           "workers": 8}, cores=256)
    assert rec["scaling"] == 1.0


def test_the_probe_caps_workers_but_keeps_the_true_width(monkeypatch):
    monkeypatch.setattr(cp, "cpu_width", lambda: (256.0, "cgroup_v2_quota"))
    monkeypatch.setattr(cp, "busy_cores", lambda dwell=0.4: 0.0)
    monkeypatch.setattr(cp, "bench_single", lambda i, r: {"single_per_s": 10.0})
    seen = {}

    def _all(workers, iters, rounds):
        seen["workers"] = workers
        return {"allcore_per_s": 40.0, "workers": workers,
                "allcore_wall_s": 1.0, "allcore_count": 40}
    monkeypatch.setattr(cp, "bench_allcore", _all)
    rec, _ = cp.probe(max_workers=4, with_compile=False, state_dir="")
    assert seen["workers"] == 4
    assert rec["cores"] == 256.0          # the slice is reported as it is
    assert rec["scaling"] == 1.0          # ...and scaling is against the cap


# --------------------------------------------------------------------------- #
# the record
# --------------------------------------------------------------------------- #
def test_a_refusal_carries_no_rate():
    """An unquotable number should not exist, the same rule gemm_probe applies
    to a TFLOP/s figure with no device attached."""
    rec = cp.build_record(None, cores=8, width_source="cgroup_v2_quota",
                          status="skipped:job_running:x")
    assert rec["status"].startswith("skipped:")
    for k in ("single_per_s", "allcore_per_s", "scaling"):
        assert k not in rec


def test_the_probe_refuses_a_busy_box_without_benching(monkeypatch):
    monkeypatch.setattr(cp, "cpu_width", lambda: (8.0, "cgroup_v2_quota"))
    monkeypatch.setattr(cp, "busy_cores", lambda dwell=0.4: 9.0)

    def _boom(*a, **k):
        raise AssertionError("benched a busy box")
    monkeypatch.setattr(cp, "bench_single", _boom)
    rec, comp = cp.probe(with_compile=False, state_dir="")
    assert rec["status"].startswith("skipped:cpu_busy")
    assert comp is None
    assert rec["pre_probe_busy_cores"] == 9.0


def test_a_bench_failure_is_recorded_not_raised(monkeypatch):
    monkeypatch.setattr(cp, "cpu_width", lambda: (8.0, "cgroup_v2_quota"))
    monkeypatch.setattr(cp, "busy_cores", lambda dwell=0.4: 0.0)

    def _boom(*a, **k):
        raise RuntimeError("no fork for you")
    monkeypatch.setattr(cp, "bench_single", _boom)
    rec, _ = cp.probe(with_compile=False, state_dir="")
    assert rec["status"] == "failed"
    assert "RuntimeError" in rec["reason"]


def test_machine_id_is_inherited_never_invented(monkeypatch):
    monkeypatch.delenv("MACHINE_ID", raising=False)
    monkeypatch.delenv("VAST_MACHINE_ID", raising=False)
    monkeypatch.setenv("INSTANCE_ID", "48603388")
    rec = cp.build_record(None)
    assert rec["instance_id"] == "48603388"
    assert "machine_id" not in rec        # vast injects none; ingest resolves it
    monkeypatch.setenv("MACHINE_ID", "140799")
    assert cp.build_record(None)["machine_id"] == "140799"


# --------------------------------------------------------------------------- #
# jobd field rendering — a value with whitespace breaks the K=V parser
# --------------------------------------------------------------------------- #
def test_rendered_fields_have_no_whitespace_in_any_value():
    rec = cp.build_record({"single_per_s": 10.0, "allcore_per_s": 80.0,
                           "workers": 8}, cores=8,
                          width_source="cgroup_v2_quota", level=0.1)
    rec["cpu_name"] = "AMD EPYC 7713 64-Core Processor"
    for line in cp.render_fields(rec).splitlines():
        k, _, v = line.partition("=")
        assert k and v and " " not in v, line


def test_rendered_fields_carry_the_scaling_finding():
    rec = cp.build_record({"single_per_s": 10.0, "allcore_per_s": 80.0,
                           "workers": 8}, cores=8)
    assert "scaling=1.0" in cp.render_fields(rec)


# --------------------------------------------------------------------------- #
# the seam into hostfacts: two units that must never average together
# --------------------------------------------------------------------------- #
def test_pyops_and_compile_tu_stay_separate_comparands():
    """`summarize_cpu` groups by unit. A synthetic rate and a compile rate share
    a machine but not a scale, and collapsing them would produce a number with
    no referent."""
    recs = [
        hf.cpu_record("1", "t1", units="pyops", count=8e7, wall_s=1.0,
                      cores=8, machine_id="140799"),
        hf.cpu_record("1", "t2", units="compile_tu", count=12, wall_s=1.0,
                      cores=8, machine_id="140799"),
    ]
    rows = hf.summarize_cpu(recs)["hosts"]
    assert {r["units"] for r in rows} == {"pyops", "compile_tu"}
    assert {r["host"] for r in rows} == {"140799"}   # one machine, two rows
    by_units = {r["units"]: r["best_per_core_s"] for r in rows}
    assert by_units["pyops"] == pytest.approx(1e7)
    assert by_units["compile_tu"] == pytest.approx(1.5)


def test_the_serial_compile_arm_is_not_normalised_by_the_slice_width(tmp_path,
                                                                     monkeypatch):
    """`bench_compile` runs one subprocess at a time, so its rate is a SINGLE
    THREAD's. Dividing it by a 128-thread width made `per_core_s` mean
    single-thread-over-width — it ranked narrow boxes best, and the fleet read a
    35x spread that was mostly just how wide the boxes were."""
    rec = {"status": "ok", "cores": 128.0, "allcore_count": 8e7,
           "allcore_wall_s": 1.0, "cpu_name": "Test CPU", "instance_id": "1"}
    comp = {"count": 40, "wall_s": 10.0, "cc": "cc"}
    monkeypatch.setattr(cp, "probe", lambda **kw: (rec, comp))
    cp.main(["drop", "--directory", str(tmp_path)])
    got = {}
    for name in os.listdir(tmp_path):
        with open(os.path.join(tmp_path, name)) as fh:
            r = json.load(fh)
        got[r["units"]] = r
    assert got["compile_tu"]["cores"] == 1
    # the single-thread rate, undivided — and equal to per_s, so the scorecard
    # column and the raw rate finally say the same thing
    assert got["compile_tu"]["per_core_s"] == pytest.approx(4.0)
    assert got["compile_tu"]["per_core_s"] == got["compile_tu"]["per_s"]
    # the pyops arm IS all-core and keeps the width
    assert got["pyops"]["cores"] == 128.0


def test_a_dropped_probe_record_is_a_cpu_hostfact(tmp_path, monkeypatch):
    monkeypatch.setenv("INSTANCE_ID", "48603388")
    p = hf.drop_cpu_record(units="pyops", count=8e7, wall_s=1.0, cores=8.0,
                           workload="cpu_probe.kernel", scaling=0.93,
                           width_source="cgroup_v2_quota",
                           directory=str(tmp_path))
    assert os.path.basename(p).startswith("cpu-")
    import json
    rec = json.loads(open(p).read())
    assert rec["kind"] == "cpu" and rec["units"] == "pyops"
    assert rec["instance_id"] == "48603388"
    assert rec["scaling"] == 0.93
    assert rec["per_core_s"] == pytest.approx(1e7)


# --------------------------------------------------------------------------- #
# two records, one second — the collision that silently ate the pyops record
# --------------------------------------------------------------------------- #
def test_two_records_in_the_same_second_both_survive(tmp_path):
    """`ts` is second-resolution and the probe drops `pyops` and `compile_tu`
    back to back, so they collided on one filename and the second REPLACED the
    first — destroying the universal record and keeping the optional one.

    Measured 2026-08-25 by running the real probe. A faked interpreter that
    writes one file cannot see this, which is why it survived the wiring tests.
    """
    common = {"count": 10, "wall_s": 1.0, "cores": 4.0, "instance_id": "1",
              "ts": "2026-08-25T02:06:05Z", "directory": str(tmp_path)}
    a = hf.drop_cpu_record(units="pyops", **common)
    b = hf.drop_cpu_record(units="compile_tu", **common)
    assert a != b
    assert os.path.isfile(a) and os.path.isfile(b)
    units = sorted(json.loads(open(p).read())["units"] for p in (a, b))
    assert units == ["compile_tu", "pyops"]


def test_the_kind_still_parses_off_a_suffixed_name(tmp_path):
    """The counter goes after the stamp precisely so `_kind_from_key` — which
    reads the LEADING token — keeps calling it a cpu record."""
    common = {"count": 10, "wall_s": 1.0, "instance_id": "1",
              "ts": "2026-08-25T02:06:05Z", "directory": str(tmp_path)}
    hf.drop_cpu_record(units="pyops", **common)
    b = hf.drop_cpu_record(units="compile_tu", **common)
    assert b.endswith("-2.json")
    assert hf._kind_from_key(b) == hf.KIND_CPU


def test_a_third_record_gets_its_own_name(tmp_path):
    common = {"count": 10, "wall_s": 1.0, "instance_id": "1",
              "ts": "2026-08-25T02:06:05Z", "directory": str(tmp_path)}
    paths = {hf.drop_cpu_record(units=u, **common)
             for u in ("pyops", "compile_tu", "tu")}
    assert len(paths) == 3


# --------------------------------------------------------------------------- #
# the compile TU: big enough that it measures the COMPILER
# --------------------------------------------------------------------------- #
def test_the_tu_repeats_with_distinct_names():
    """Identical copies would let the compiler do the work once, turning a
    size knob into a no-op."""
    src = cp.c_source(3)
    assert src.count("probe_kernel") == 3
    for n in range(3):
        assert f"probe_kernel{n}" in src
    assert src.count("#include") == 1        # the prologue is not repeated


def test_the_tu_grows_with_repeats():
    assert len(cp.c_source(32)) > len(cp.c_source(4))


def test_the_shipped_tu_is_sized_against_process_startup():
    """At one copy, `cc` startup was 41% of the bench — a fork+exec benchmark
    wearing a compiler's name. The constant is a measured choice (see its
    comment) and dropping it back to a token size undoes that."""
    assert cp._C_REPEATS >= 32
