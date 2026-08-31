"""Portable tests for hostfacts.py — no GPU, no B2, no vast API.

Every command runs against a LocalStore laid out with the real B2 key structure,
so what is tested is the key convention and the resolution semantics, not a
mock's idea of them.
"""
import argparse
import json
import os
import subprocess
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemm_probe as gp  # noqa: E402
import hostfacts as hf  # noqa: E402
import mfu  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_HF = os.path.join(_HERE, "hostfacts.py")


def _rec(iid, ts, tflops=(269.4, 268.2, 229.1),
         device="NVIDIA RTX PRO 6000 Blackwell Server Edition", plim=600,
         status="ok", **kw):
    shapes = [{"m": 8192, "k": k, "n": n, "ms": 1.0, "tflops": t}
              for (k, n), t in zip([(4096, 4096), (4096, 16384), (16384, 4096)],
                                   tflops)]
    r = {"probe_version": 1, "ts": ts, "status": status, "instance_id": iid,
         "shape_basis": "generic", "device": device, "capability": "sm_120",
         "sm_count": 188, "torch": "2.11.0+cu129", "cuda": "12.9",
         "dtype": "bf16", "shapes": shapes, "power_limit_w": plim,
         "sm_clock_mhz": 2370, "gpu_count": 1}
    if tflops:
        r["ceiling_tflops"] = max(tflops)
        r["min_tflops"] = min(tflops)
    r.update(kw)
    return r


@pytest.fixture()
def store(tmp_path):
    return hf.LocalStore(str(tmp_path / "b2"))


# --------------------------------------------------------------------------- #
# keys: the fact belongs to the machine, not to the job
# --------------------------------------------------------------------------- #
def test_keys_are_host_scoped_and_carry_no_job_id():
    """The defect class this design exists to avoid (memory:
    workload-state-stored-on-the-box): `job retarget` keeps the JOB_ID and moves
    to a different box, so a ceiling filed under a job is attributed to the wrong
    machine. Neither key contains a job id — the box tier keys on the INSTANCE
    under jobd's per-box `nodes/` segment, the pinned tier on the MACHINE."""
    ik = hf.instance_key("46947265", "2026-08-07T01:02:03Z")
    mk = hf.machine_key("140799", "46947265", "2026-08-07T01:02:03Z")
    assert ik.startswith("jobs/nodes/46947265/hostfacts/")
    assert mk.startswith("hostfacts/by-machine/140799/")


def test_the_box_tier_stays_inside_the_scoped_write_keys_prefix():
    """A B2 key carries a SINGLE namePrefix and a split box's write key is
    `namePrefix=jobs/` (CREDENTIAL_LIFECYCLE §2). A record written outside it
    403s — verbatim the B2_PUBLISH_KEY_SCOPE_FIX incident, where both v7 arms
    trained to completion and only then failed to publish."""
    assert hf.instance_key("46947265", "t").startswith("jobs/")
    assert hf.instance_prefix("46947265").startswith("jobs/")


def test_every_measurement_is_its_own_immutable_object():
    """B2 has no compare-and-set; a shared mutable `latest.json` under concurrent
    writers is the hazard the runs/ event log is built around."""
    a = hf.instance_key("46947265", "2026-08-07T01:02:03Z")
    b = hf.instance_key("46947265", "2026-08-07T09:00:00Z")
    assert a != b and "latest" not in a


def test_machine_key_keeps_the_instance_so_re_rentals_stay_distinguishable():
    """One machine rented twice yields two records. Collapsing them would hide
    exactly the case worth seeing — throttled on one rental, healthy on the next
    (PERF_LEVERS_INVESTIGATION §2.4)."""
    a = hf.machine_key("140799", "46947265", "t1")
    b = hf.machine_key("140799", "46999999", "t2")
    assert a != b and a.rsplit("/", 1)[0] == b.rsplit("/", 1)[0]


# --------------------------------------------------------------------------- #
# store round-trip
# --------------------------------------------------------------------------- #
def test_local_store_round_trip(store):
    key = hf.instance_key("46947265", "2026-08-07T01:02:03Z")
    assert store.put(key, _rec("46947265", "2026-08-07T01:02:03Z"))
    assert store.get(key)["ceiling_tflops"] == 269.4
    assert store.keys(hf.NODES) == [key]


def test_a_missing_object_reads_as_none_not_an_exception(store):
    assert store.get(hf.instance_key("nope", "t1")) is None


def _write_raw(store, key, text):
    p = os.path.join(store.root, *key.split("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        fh.write(text)


def test_load_records_skips_unparseable_rows_without_losing_the_good_ones(store):
    store.put(hf.instance_key("1", "t1"), _rec("1", "t1"))
    _write_raw(store, hf.instance_key("2", "t2"), "{not json")
    recs = hf.load_records(store)
    assert [r["instance_id"] for r in recs] == ["1"]


def test_the_box_tier_is_not_enumerated_by_recursing_over_jobs_nodes(store):
    """`jobs/nodes/<IID>/` also holds every box's whole lifecycle event stream.
    Listing it recursively would pull thousands of unrelated objects, so the
    enumeration is dirs(nodes) x keys(<iid>/hostfacts)."""
    store.put(hf.instance_key("46947265", "t1"), _rec("46947265", "t1"))
    _write_raw(store, "jobs/nodes/46947265/events/20260807-box-abcd.json",
               json.dumps({"event": "jobd_up"}))
    assert hf.node_keys(store) == [hf.instance_key("46947265", "t1")]
    assert len(hf.load_records(store)) == 1


def test_instance_id_is_recovered_from_the_key_when_the_record_omits_it(store):
    blob = _rec("1", "t1")
    del blob["instance_id"]
    store.put(hf.instance_key("46947265", "t1"), blob)
    assert hf.load_records(store)[0]["instance_id"] == "46947265"


# --------------------------------------------------------------------------- #
# ingest: eager resolution, and a promotion that can never lose a record
# --------------------------------------------------------------------------- #
def test_ingest_pins_a_resolvable_record_to_by_machine(store):
    store.put(hf.instance_key("46947265", "t1"), _rec("46947265", "t1"))
    res = hf.ingest(store, lambda iid: "140799" if iid == "46947265" else None)
    assert len(res["pinned"]) == 1
    pinned = store.get(res["pinned"][0])
    assert pinned["machine_id"] == "140799"
    assert pinned["pinned_from"].startswith("jobs/nodes/46947265/")


def test_ingest_leaves_an_unresolvable_record_alone_rather_than_dropping_it(store):
    """The instance -> machine mapping dies with the instance. An ingest that ran
    too late must degrade to less-queryable, never to lost."""
    key = hf.instance_key("46936034", "t1")
    store.put(key, _rec("46936034", "t1"))
    res = hf.ingest(store, lambda iid: None)
    assert res["pinned"] == [] and res["unresolved"] == [key]
    assert store.get(key) is not None                     # still there, intact


def test_ingest_is_idempotent(store):
    store.put(hf.instance_key("46947265", "t1"), _rec("46947265", "t1"))
    r1 = hf.ingest(store, lambda _iid: "140799")
    r2 = hf.ingest(store, lambda _iid: "140799")
    assert len(r1["pinned"]) == 1 and r2["pinned"] == [] and r2["already"] == 1


def test_ingest_dry_run_writes_nothing(store):
    store.put(hf.instance_key("46947265", "t1"), _rec("46947265", "t1"))
    res = hf.ingest(store, lambda _iid: "140799", dry_run=True)
    assert len(res["pinned"]) == 1
    assert store.keys(hf.BY_MACHINE) == []


def test_a_self_declared_machine_id_is_honoured_without_the_api(store):
    """If a future launch injects MACHINE_ID the box records it, and ingest must
    not need the API at all for those."""
    store.put(hf.instance_key("46947265", "t1"),
              _rec("46947265", "t1", machine_id="140799"))
    res = hf.ingest(store, lambda _iid: None)
    assert len(res["pinned"]) == 1 and res["unresolved"] == []


def test_a_pinned_record_does_not_double_count_in_the_rollup(store):
    store.put(hf.instance_key("46947265", "t1"), _rec("46947265", "t1"))
    hf.ingest(store, lambda _iid: "140799")
    recs = hf.load_records(store)
    assert len(recs) == 1 and recs[0]["machine_id"] == "140799"


# --------------------------------------------------------------------------- #
# the rollup an agent reads when a run is slow
# --------------------------------------------------------------------------- #
def _fleet(store):
    """Three machines, one of them the degraded box: the shape of the finding in
    PERF_LEVERS_INVESTIGATION §2.3 (2.13x at the same price)."""
    store.put(hf.machine_key("140799", "46947265", "t1"),
              _rec("46947265", "t1", tflops=(269.4, 268.2, 229.1)))
    store.put(hf.machine_key("37777", "46950000", "t1"),
              _rec("46950000", "t1", tflops=(262.0, 259.0, 224.0)))
    store.put(hf.machine_key("144366", "46936034", "t1"),
              _rec("46936034", "t1", tflops=(127.0, 124.0, 108.0), plim=300,
                   device="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation "
                          "Edition"))


def test_summary_ranks_hosts_and_reports_the_spread(store):
    _fleet(store)
    s = hf.summarize(hf.load_records(store))
    assert [h["host"] for h in s["hosts"]] == ["140799", "37777", "144366"]
    assert s["fleet_median_tflops"] == 262.0
    assert s["spread"] == pytest.approx(269.4 / 127.0, abs=0.01)
    slow = s["hosts"][-1]
    assert slow["ratio_to_median"] == pytest.approx(0.485, abs=0.005)
    assert slow["power_limit_w"] == [300]


def test_the_fleet_median_is_over_hosts_not_over_records(store):
    """Three rentals of one fast machine must not drag the median that every
    other host is compared against."""
    for i, iid in enumerate(("a1", "a2", "a3")):
        store.put(hf.machine_key("fast", iid, f"t{i}"),
                  _rec(iid, f"t{i}", tflops=(300.0, 299.0, 280.0)))
    store.put(hf.machine_key("slow", "b1", "t1"),
              _rec("b1", "t1", tflops=(100.0, 99.0, 90.0)))
    s = hf.summarize(hf.load_records(store))
    assert s["n_hosts"] == 2
    assert s["fleet_median_tflops"] == pytest.approx(200.0)   # (300+100)/2


def test_an_unresolved_record_groups_under_its_instance_not_a_guessed_machine(store):
    store.put(hf.instance_key("46936034", "t1"), _rec("46936034", "t1"))
    s = hf.summarize(hf.load_records(store))
    assert s["hosts"][0]["host"] == "iid:46936034"
    assert s["hosts"][0]["resolved"] is False


def test_a_record_with_no_device_name_is_not_quotable(store):
    """gemm_ceiling.py's rule, enforced at READ time too: a TFLOP/s figure with
    no device attached is not quotable."""
    store.put(hf.instance_key("x", "t1"), _rec("x", "t1", device=""))
    s = hf.summarize(hf.load_records(store))
    assert s["hosts"][0]["ceiling_tflops"] is None
    assert s["fleet_median_tflops"] is None


def test_a_skipped_probe_is_recorded_and_reported_but_never_counted(store):
    skipped = gp.probe(metrics={"gpus": [{"idx": 0, "util": 99,
                                          "mem_used_mb": 61000,
                                          "power_limit_w": 600.0,
                                          "sm_clock_mhz": 2370, "temp_c": 71,
                                          "throttle": ["sw_power"]}]})
    store.put(hf.instance_key("busybox", skipped["ts"]), skipped)
    s = hf.summarize(hf.load_records(store))
    h = s["hosts"][0]
    assert h["n_records"] == 1 and h["n_quotable"] == 0
    assert h["statuses"] == ["skipped:gpu_busy"]


def test_render_table_names_the_missing_policy_half(store):
    _fleet(store)
    out = hf.render_table(hf.summarize(hf.load_records(store)))
    assert "fleet median" in out
    assert "Observation only" in out or "observation only" in out


def test_render_table_on_an_empty_store_says_so_rather_than_dividing_by_zero(store):
    out = hf.render_table(hf.summarize([]))
    assert "no quotable ceiling yet" in out


# --------------------------------------------------------------------------- #
# handing the ceiling to mfu.py
# --------------------------------------------------------------------------- #
def test_weighted_ceiling_uses_the_models_own_mac_mix(store):
    rec = _rec("x", "t1")
    mix = mfu.mac_mix(mfu.gemma4_12b_text())
    tf = hf.weighted_ceiling(rec, mix)
    assert 229.1 < tf < 269.4          # strictly inside, never the headline max


def test_weighted_ceiling_of_a_deviceless_record_is_none():
    assert hf.weighted_ceiling(_rec("x", "t1", device=""),
                               mfu.mac_mix(mfu.gemma4_12b_text())) is None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _run(store, *argv):
    return subprocess.run([sys.executable, _HF, "--local", store.root, *argv],
                          capture_output=True, text=True, timeout=120)


def test_cli_list_json(store):
    _fleet(store)
    r = _run(store, "list", "--json")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["n_hosts"] == 3


def test_cli_list_table_on_an_empty_store_is_not_an_error(store):
    r = _run(store, "list")
    assert r.returncode == 0, r.stderr
    assert "no quotable ceiling yet" in r.stdout


def test_cli_ceiling_emits_a_blob_mfu_can_divide_by(store, tmp_path):
    _fleet(store)
    r = _run(store, "ceiling", "--machine", "140799")
    assert r.returncode == 0, r.stderr
    blob = tmp_path / "ceiling.json"
    blob.write_text(r.stdout)
    m = subprocess.run(
        [sys.executable, os.path.join(_HERE, "mfu.py"),
         "--model", "gemma-4-12b-text", "--tok-s", "1430",
         "--device", "NVIDIA RTX PRO 6000 Blackwell Server Edition",
         "--ceiling-json", str(blob), "--json"],
        capture_output=True, text=True, timeout=120)
    assert m.returncode == 0, m.stderr
    u = json.loads(m.stdout)["utilisation"]
    assert u["provisional"] is False           # measured on the run's own device
    assert 0 < u["mfu_raw"] < 1


def test_cli_ceiling_for_an_unknown_host_fails_with_a_next_step(store):
    r = _run(store, "ceiling", "--machine", "999")
    assert r.returncode == 1
    assert "gemm_probe.py" in r.stderr


def test_cli_ceiling_needs_an_identifier(store):
    r = _run(store, "ceiling")
    assert r.returncode != 0 and "--machine" in r.stderr


def test_cli_show(store):
    _fleet(store)
    r = _run(store, "show", "46936034")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)[0]["machine_id"] == "144366"


def test_ingest_survives_a_resolver_that_cannot_reach_the_api(monkeypatch, store):
    """`vast_machine_resolver` degrades to "resolve nothing" on any API failure,
    and `ingest` treats that as "leave it unresolved" — never a traceback and
    never a dropped record."""
    import builtins
    real_import = builtins.__import__

    def boom(name, *a, **k):
        if name == "herdd":
            raise ImportError("no credentials on this host")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    resolver = hf.vast_machine_resolver()
    monkeypatch.undo()
    assert resolver("46947265") is None
    store.put(hf.instance_key("46947265", "t1"), _rec("46947265", "t1"))
    res = hf.ingest(store, resolver)
    assert res["pinned"] == [] and len(res["unresolved"]) == 1


def test_cli_ingest_dry_run_is_offline_safe(store, tmp_path):
    """Run from a directory with no `.env` and with no API key in the
    environment, so `_api_key_soft` returns fatal-config and no request is ever
    made. rc must still be 0 — an ingest with nothing resolvable is a normal
    outcome, not a failure."""
    store.put(hf.instance_key("46947265", "t1"), _rec("46947265", "t1"))
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items()
           if k not in ("VASTAI_API_KEY", "CONTAINER_API_KEY")}
    env["HOME"] = str(home)
    r = subprocess.run([sys.executable, _HF, "--local", store.root,
                        "ingest", "--dry-run"],
                       capture_output=True, text=True, timeout=180,
                       env=env, cwd=str(cwd))
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["unresolved"] == 1


# --------------------------------------------------------------------------- #
# the `cpu` fact kind — same mechanism, one hardware axis over
# --------------------------------------------------------------------------- #

def _cpurec(iid, ts, count=1200, wall_s=600.0, cores=64, **kw):
    return hf.cpu_record(iid, ts, units="tu", count=count, wall_s=wall_s,
                         cores=cores, workload="mwcceppc", **kw)


def _cpu_at(iid, units, per_core_s, cores=64, ts="t1"):
    """A cpu record whose per_core_s is exactly `per_core_s`."""
    return hf.cpu_record(iid, ts, units=units, count=per_core_s * cores,
                         wall_s=1.0, cores=cores)

def test_cpu_and_gemm_facts_do_not_collide_in_the_key_space():
    """Both kinds share the two tiers, so the filename token is the only thing
    keeping one machine's two measurements apart."""
    g = hf.instance_key("46947265", "t1")
    c = hf.instance_key("46947265", "t1", hf.KIND_CPU)
    assert g != c
    assert hf._kind_from_key(g) == "gemm" and hf._kind_from_key(c) == "cpu"
    gm = hf.machine_key("140799", "46947265", "t1")
    cm = hf.machine_key("140799", "46947265", "t1", hf.KIND_CPU)
    assert gm != cm
    assert hf._kind_from_key(gm) == "gemm" and hf._kind_from_key(cm) == "cpu"


def test_a_pre_kinds_record_still_reads_as_gemm():
    """Records written before kinds existed carry no token this knows, and each
    cost a rental. Defaulting them to gemm is what stops a rename orphaning
    them."""
    assert hf._kind_from_key("jobs/nodes/1/hostfacts/gemm-t.json") == "gemm"
    assert hf._kind_from_key("jobs/nodes/1/hostfacts/weird-t.json") == "gemm"
    assert hf.kind_of({}) == "gemm"
    assert hf.kind_of({"kind": "bogus"}) == "gemm"
    assert hf.kind_of({"kind": "cpu"}) == "cpu"


def test_cpu_record_requires_a_unit_and_a_positive_wall():
    """A bare rate with no unit is the thing that gets quoted years later
    against a different workload."""
    with pytest.raises(ValueError):
        hf.cpu_record("1", "t", units="", count=10, wall_s=1.0)
    with pytest.raises(ValueError):
        hf.cpu_record("1", "t", units="tu", count=10, wall_s=0)


def test_cpu_record_derives_both_rates_once():
    r = _cpurec("1", "t1", count=1200, wall_s=600.0, cores=64)
    assert r["per_s"] == 2.0
    assert r["per_core_s"] == round(2.0 / 64, 5)
    assert r["kind"] == "cpu" and r["units"] == "tu"


def test_the_gemm_scorecard_does_not_count_cpu_records(store):
    """A cpu record has no ceiling_tflops, so it would not CRASH the GEMM
    rollup — it would quietly land as a host with zero quotable measurements
    and understate that machine. Kind filtering, not a shape test."""
    recs = [_rec("46947265", "t1", machine_id="140799"),
            _cpurec("46947265", "t2", machine_id="140799")]
    s = hf.summarize(recs)
    assert s["n_hosts"] == 1
    assert s["hosts"][0]["n_records"] == 1        # the cpu row is not in here
    assert s["hosts"][0]["n_quotable"] == 1


def test_cpu_rollup_keeps_different_units_in_different_rows():
    """Averaging two units together yields a number with no referent."""
    recs = [_cpurec("1", "t1", machine_id="140799"),
            hf.cpu_record("1", "t2", units="objs", count=50, wall_s=10.0,
                          cores=64, machine_id="140799")]
    s = hf.summarize_cpu(recs)
    assert {r["units"] for r in s["hosts"]} == {"tu", "objs"}
    assert s["n_hosts"] == 1                      # one host, two unit rows


def test_cpu_ingest_pins_as_a_cpu_record(store):
    """Ingest is kind-agnostic: the kind rides across from the source key, so a
    cpu record pins as one without ingest knowing what is inside it."""
    store.put(hf.instance_key("46947265", "t1", hf.KIND_CPU),
              _cpurec("46947265", "t1"))
    out = hf.ingest(store, lambda iid: "140799")
    assert len(out["pinned"]) == 1
    dest = out["pinned"][0]
    assert hf._kind_from_key(dest) == "cpu"
    assert hf._machine_from_key(dest) == "140799"


# --------------------------------------------------------------------------- #
# the same-second pair: cpu_probe drops two records back to back, and BOTH
# halves of the pipeline used to collapse them into one
# --------------------------------------------------------------------------- #
def _pair(store, iid="46861146", ts="2026-08-27T07_18_27Z"):
    """The shape a box actually writes: `drop_record` disambiguates the second
    same-second record with a `-N` suffix, and jobd uploads by BASENAME."""
    base = f"{hf.NODES}/{iid}/{hf.NODE_LEAF}"
    store.put(f"{base}/cpu-{ts}.json", _cpu_at(iid, "pyops", 5e6, ts=ts))
    store.put(f"{base}/cpu-{ts}-2.json",
              _cpu_at(iid, "compile_tu", 0.2, ts=ts))
    return iid, ts


def test_two_same_second_records_of_different_units_both_survive_the_read(store):
    """The dedup exists so a pinned copy shadows its node-tier original. Keyed
    on (instance, ts) alone it also ate the SECOND MEASUREMENT: cpu_probe drops
    `pyops` and `compile_tu` back to back at second resolution, so one of the
    two was discarded on the way in."""
    _pair(store)
    got = sorted(r["units"] for r in hf.load_records(store))
    assert got == ["compile_tu", "pyops"]


def test_pinning_two_same_second_records_keeps_them_apart(store):
    """`machine_key` is built from (machine, instance, ts, kind) — no term of
    which separates the pair — and `existing` is snapshotted BEFORE the loop, so
    the second put overwrote the first in an immutable store."""
    _pair(store)
    out = hf.ingest(store, lambda iid: "140799")
    assert len(out["pinned"]) == 2
    assert len(set(out["pinned"])) == 2
    units = sorted(store.get(k)["units"] for k in out["pinned"])
    assert units == ["compile_tu", "pyops"]


def test_ingest_is_idempotent_over_a_same_second_pair(store):
    """Order-independence is the reason the suffix is carried across from the
    source key rather than assigned on collision: a re-run must re-derive the
    same two names, not mint a third."""
    _pair(store)
    first = hf.ingest(store, lambda iid: "140799")
    again = hf.ingest(store, lambda iid: "140799")
    assert again["pinned"] == [] and again["already"] == 2
    assert sorted(first["pinned"]) == sorted(
        k for k in store.keys(hf.BY_MACHINE) if hf._kind_from_key(k) == "cpu")


def test_parallel_reads_come_back_in_the_order_they_were_asked_for():
    """The scorecard's determinism comes from sorted keys, so the fan-out has
    to preserve order however the threads interleave."""
    import time as _t

    def _slow(i):
        _t.sleep(0.02 if i % 2 else 0.0)   # invert the natural finish order
        return i * 2

    assert hf._map_reads(_slow, range(24)) == [i * 2 for i in range(24)]
    assert hf._map_reads(_slow, []) == []
    assert hf._map_reads(_slow, [7], workers=1) == [14]


def test_a_record_with_no_duplicate_suffix_pins_where_it_always_did(store):
    """The 82 objects already in by-machine were named without a suffix. The
    fix has to be ADDITIVE or a re-ingest mints a duplicate of every one."""
    store.put(hf.instance_key("46947265", "t1", hf.KIND_CPU),
              _cpurec("46947265", "t1"))
    out = hf.ingest(store, lambda iid: "140799")
    assert out["pinned"] == [hf.machine_key("140799", "46947265", "t1",
                                            hf.KIND_CPU)]


# --------------------------------------------------------------------------- #
# calibration: the scorecard frozen into what the offer lane joins against
# --------------------------------------------------------------------------- #
def _cal_rec(iid, machine, cpu_name, per_core_s, cores=64, ts="t1",
             units="pyops"):
    return hf.cpu_record(iid, ts, units=units, count=per_core_s * cores,
                         wall_s=1.0, cores=cores, cpu_name=cpu_name,
                         machine_id=machine)


def test_calibration_keys_a_machine_and_a_model_the_two_ways_an_offer_joins():
    """A vast offer carries `machine_id` AND a `cpu_name` byte-identical to the
    one the probe banks, so both tiers are a string join rather than an
    inference."""
    t = hf.calibration([
        _cal_rec("1", "140799", "AMD EPYC 7713P 64-Core Processor", 4.6e6),
        _cal_rec("2", "140800", "AMD EPYC 7713P 64-Core Processor", 4.8e6),
    ])
    assert t["n_machines"] == 2 and t["n_models"] == 1
    assert t["by_machine"]["140799"]["rate"] == pytest.approx(4.6e6)
    model = t["by_model"]["AMD EPYC 7713P 64-Core Processor"]
    assert model["n_machines"] == 2
    assert model["rate"] == pytest.approx(4.7e6)      # median of the two
    assert model["spread"] == pytest.approx(4.8 / 4.6, rel=1e-3)


def test_a_model_measured_once_reports_no_spread_rather_than_a_fake_one():
    """`spread` is None, not 1.0 — a caller must be able to tell "one machine,
    so we cannot know" from "two machines that agreed"."""
    t = hf.calibration([_cal_rec("1", "140799", "Intel(R) Xeon(R) 6767P", 6.8e6)])
    assert t["by_model"]["Intel(R) Xeon(R) 6767P"]["spread"] is None


def test_an_unresolved_host_still_informs_the_model_tier():
    """by_machine needs machine grain; a MODEL estimate does not need identity,
    so a record from a box we could never resolve is not thrown away."""
    orphan = hf.cpu_record("999", "t1", units="pyops", count=3.2e8, wall_s=1.0,
                           cores=64, cpu_name="AMD EPYC 7452 32-Core Processor")
    t = hf.calibration([orphan])
    assert t["n_machines"] == 0
    assert t["by_model"]["AMD EPYC 7452 32-Core Processor"]["n_machines"] == 1


def test_a_serial_unit_calibrates_on_per_s_not_on_per_core():
    """`compile_tu` runs one compile at a time. Dividing it by the slice width
    answers "single thread over 128 threads" — it ranks narrow boxes best, and
    it is why that cohort read a 35x fleet spread that was mostly just width."""
    wide = hf.cpu_record("1", "t1", units="compile_tu", count=40, wall_s=10.0,
                         cores=128, cpu_name="Wide", machine_id="1")
    narrow = hf.cpu_record("2", "t1", units="compile_tu", count=40, wall_s=10.0,
                           cores=8, cpu_name="Narrow", machine_id="2")
    t = hf.calibration([wide, narrow], units="compile_tu")
    assert t["rate_is"] == "per_s"
    # identical compile rates, so the table must not separate them by width
    assert t["by_machine"]["1"]["rate"] == t["by_machine"]["2"]["rate"] == 4.0
    assert t["fleet_spread"] is None or t["fleet_spread"] == 1.0


def test_the_fleet_median_is_over_the_same_rate_the_tiers_use():
    """The offer-side floor is a RATIO of this median, so reading it off a
    different derivation than the tiers would set the floor in another unit."""
    t = hf.calibration([
        hf.cpu_record("1", "t1", units="compile_tu", count=40, wall_s=10.0,
                      cores=128, cpu_name="A", machine_id="1"),
        hf.cpu_record("2", "t1", units="compile_tu", count=80, wall_s=10.0,
                      cores=8, cpu_name="B", machine_id="2"),
    ], units="compile_tu")
    assert t["fleet_median"] == pytest.approx(6.0)     # median of 4.0 and 8.0


def test_a_host_reporting_two_cpu_names_is_left_out_rather_than_guessed():
    """The model tier is a string join. A host that named itself two different
    ways cannot be joined to either without inventing which one an offer means."""
    recs = [_cal_rec("1", "140799", "CPU A", 4e6, ts="t1"),
            _cal_rec("1", "140799", "CPU B", 5e6, ts="t2")]
    t = hf.calibration(recs)
    assert t["by_machine"] == {} and t["by_model"] == {}


def test_a_missing_or_corrupt_calibration_reads_as_nothing_measured(tmp_path):
    """Same state as an offer we have never seen, which every caller handles.
    A search must never fail because a table is absent."""
    assert hf.load_calibration(str(tmp_path / "nope.json")) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert hf.load_calibration(str(bad)) is None
    empty = tmp_path / "empty.json"
    empty.write_text('{"by_machine": {}}')
    assert hf.load_calibration(str(empty)) is None


def test_the_shipped_calibration_table_loads_and_is_self_consistent():
    """The tracked table is what the offer lane actually reads."""
    blob = hf.load_calibration()
    assert blob is not None, "tools/vast/cpu_calibration.json missing"
    for arm in hf.CALIBRATION_ARMS:
        t = hf.calibration_arm(blob, arm)
        assert t is not None, f"shipped table has no {arm} arm"
        assert t["units"] == arm and t["fleet_median"] > 0
        assert len(t["by_machine"]) == t["n_machines"] >= 1
        assert all(v["rate"] > 0 for v in t["by_machine"].values())
        assert all(v["spread"] is None or v["spread"] >= 1.0
                   for v in t["by_model"].values())


def test_the_shipped_table_carries_both_arms_over_the_same_fleet():
    """Both arms measuring the same machines is what lets the floor move to
    `compile_tu` for free. If the compile arm ever falls behind, the floor
    silently narrows to the machines that still have one — so pin the parity."""
    blob = hf.load_calibration()
    arms = {a: hf.calibration_arm(blob, a) for a in hf.CALIBRATION_ARMS}
    assert all(v for v in arms.values())
    assert (set(arms["pyops"]["by_machine"])
            == set(arms["compile_tu"]["by_machine"]))
    # The serial arm is a rate as measured; the all-core arm is per core. Their
    # units differ, so only the KEYS are comparable -- never the rates.
    assert arms["compile_tu"]["rate_is"] == "per_s"
    assert arms["pyops"]["rate_is"] == "per_core_s"


def test_calibration_arm_reads_a_pre_arms_table_as_any_arm():
    """A schema-1 table predates the split and every consumer read it for every
    purpose, so it must answer for whichever arm is asked rather than read as
    unmeasured -- old table, old behaviour."""
    legacy = {"units": "pyops", "fleet_median": 5.0,
              "by_machine": {"1": {"rate": 5.0}}, "by_model": {}}
    for arm in (*hf.CALIBRATION_ARMS, "something-else"):
        assert hf.calibration_arm(legacy, arm) is legacy


def test_calibration_table_omits_an_arm_nothing_measured():
    """An unmeasured arm is absent, not present-and-empty: a consumer's
    fallback should be exercised by a real absence, and an empty table would
    otherwise report a fleet median of None as if it were a measurement."""
    recs = [hf.cpu_record("i1", "2026-08-28T00:00:00Z", units="pyops",
                          count=100, wall_s=1.0, cores=2, cpu_name="Fake X")]
    t = hf.calibration_table(recs)
    assert "pyops" in t["arms"] and "compile_tu" not in t["arms"]
    assert t["schema"] == 2


# --------------------------------------------------------------------------- #
# the box-side drop dir: how a producer hands a harvested fact to jobd
# --------------------------------------------------------------------------- #
def test_drop_dir_prefers_the_explicit_env_then_jobd_root(monkeypatch):
    monkeypatch.setenv("JOBD_HOSTFACTS_DROP", "/somewhere/else")
    monkeypatch.setenv("JOBD_ROOT", "/workspace")
    assert hf.drop_dir() == "/somewhere/else"
    monkeypatch.delenv("JOBD_HOSTFACTS_DROP")
    assert hf.drop_dir() == "/workspace/hostfacts.d"
    monkeypatch.delenv("JOBD_ROOT")
    assert hf.drop_dir() == hf.DEFAULT_DROP_DIR


def test_the_dropped_filename_carries_the_kind_and_the_ts(tmp_path):
    """Both are load-bearing: `ingest` reads the KIND off the key, and one
    immutable object per measurement is the whole storage model."""
    p = hf.drop_cpu_record(units="tu", count=10, wall_s=5.0, instance_id="1",
                           ts="2026-08-24T09:00:00Z", directory=str(tmp_path))
    assert os.path.basename(p) == "cpu-2026-08-24T09_00_00Z.json"
    assert hf._kind_from_key(p) == "cpu"


def test_dropping_is_atomic(tmp_path):
    """jobd's drain runs on a timer, so a half-written file would be PUT as an
    immutable object that outlives the run that made it. Nothing but the final
    name may ever appear."""
    seen = []
    real = os.replace

    def _watch(src, dst):
        seen.append(sorted(os.listdir(tmp_path)))
        return real(src, dst)

    with mock.patch("os.replace", _watch):
        hf.drop_cpu_record(units="tu", count=10, wall_s=5.0, instance_id="1",
                           ts="t", directory=str(tmp_path))
    assert seen == [["cpu-t.json.partial"]]
    assert sorted(os.listdir(tmp_path)) == ["cpu-t.json"]


def test_a_dropped_record_is_a_cpu_record(tmp_path):
    """One factory, not two. A producer computing its own per-core rate is how
    two definitions of "per core" end up in one scorecard."""
    p = hf.drop_cpu_record(units="tu_compiles", count=1200, wall_s=300.0,
                           cores=48, instance_id="9", ts="t",
                           directory=str(tmp_path))
    with open(p) as fh:
        got = json.load(fh)
    assert got == hf.cpu_record("9", "t", units="tu_compiles", count=1200,
                                wall_s=300.0, cores=48)


def test_a_box_that_cannot_name_itself_still_drops_the_record(tmp_path,
                                                              monkeypatch):
    """"Inherit, never invent" — but refusing to write would LOSE a measurement
    that cost real work, and jobd's drain re-keys on $IID anyway."""
    monkeypatch.delenv("INSTANCE_ID", raising=False)
    monkeypatch.delenv("CONTAINER_ID", raising=False)
    p = hf.drop_cpu_record(units="tu", count=1, wall_s=1.0, ts="t",
                           directory=str(tmp_path))
    with open(p) as fh:
        assert json.load(fh)["instance_id"] == "unknown"


def test_the_write_side_still_refuses_an_unlabelled_rate(tmp_path):
    """`units` stays mandatory across the drop helper — a bare rate with no
    unit is the thing that gets quoted years later against a different
    workload."""
    with pytest.raises(ValueError):
        hf.drop_cpu_record(units="", count=1, wall_s=1.0,
                           directory=str(tmp_path))


# --------------------------------------------------------------------------- #
# `list --kind cpu`
# --------------------------------------------------------------------------- #
def test_list_defaults_to_gemm_so_every_runbook_still_works(store, capsys):
    store.put(hf.instance_key("1", "t1"), _rec("1", "t1"))
    assert hf.main(["--local", store.root, "list"]) == 0
    assert "TFLOP/s" in capsys.readouterr().out


def test_list_kind_cpu_scores_the_harvested_records(store, capsys):
    store.put(hf.instance_key("1", "t1", hf.KIND_CPU), _cpurec("1", "t1"))
    assert hf.main(["--local", store.root, "list", "--kind", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "per core/s" in out and "tu" in out
    assert "TFLOP/s" not in out, "the two scorecards share no unit"


def test_the_two_scorecards_do_not_see_each_others_records(store, capsys):
    store.put(hf.instance_key("1", "t1"), _rec("1", "t1"))
    store.put(hf.instance_key("1", "t2", hf.KIND_CPU), _cpurec("1", "t2"))
    hf.main(["--local", store.root, "list", "--kind", "cpu", "--json"])
    cpu = json.loads(capsys.readouterr().out)
    hf.main(["--local", store.root, "list", "--json"])
    gemm = json.loads(capsys.readouterr().out)
    assert len(cpu["hosts"]) == 1 and cpu["hosts"][0]["units"] == "tu"
    assert all("units" not in h for h in gemm["hosts"])


def test_the_b2_store_reads_the_bucket_out_of_dot_env(tmp_path, monkeypatch):
    """The unattended callers get no exported B2_BUCKET — the .env IS the config.

    `B2Store.__init__` reads os.environ and its failure message already claimed
    "(env or .env)", so a store that never consulted .env failed while telling
    you it had. That shape is invisible to every test using --local.
    """
    monkeypatch.delenv("B2_BUCKET", raising=False)
    (tmp_path / ".env").write_text("B2_BUCKET=env-derived-bucket\n")
    monkeypatch.chdir(tmp_path)

    st = hf._store(argparse.Namespace(local=""))
    assert st.ok, f"store refused a bucket that was in .env: {st.reason}"
    assert st.bucket == "env-derived-bucket"


def test_an_exported_bucket_still_wins_over_dot_env(tmp_path, monkeypatch):
    """load_env is setdefault, so an operator's explicit export is not clobbered."""
    monkeypatch.setenv("B2_BUCKET", "explicit")
    (tmp_path / ".env").write_text("B2_BUCKET=env-derived-bucket\n")
    monkeypatch.chdir(tmp_path)
    assert hf._store(argparse.Namespace(local="")).bucket == "explicit"


def test_the_fleet_median_never_averages_across_units():
    """A pyops rate and a merge_tensors rate share no scale.

    The rollup keeps units apart per ROW but used to pool every row into one
    median and one spread, which on the first real probe data printed
    `spread 5.7e8x` and `vs med 0.00x` for every merge_tensors host.
    """
    s = hf.summarize_cpu([_cpu_at("1", "pyops", 8.0e6),
                          _cpu_at("2", "pyops", 4.0e6),
                          _cpu_at("3", "merge_tensors", 0.1)])
    assert set(s["by_units"]) == {"pyops", "merge_tensors"}
    assert s["by_units"]["pyops"]["fleet_median_per_core_s"] == 6.0e6
    assert s["by_units"]["pyops"]["spread"] == 2.0
    assert s["by_units"]["merge_tensors"]["spread"] is None, "one host, no spread"
    assert s["fleet_median_per_core_s"] is None, (
        "a cross-unit median has no referent and must not be invented")
    assert s["spread"] is None


def test_a_single_unit_fleet_still_reports_one_median():
    s = hf.summarize_cpu([_cpu_at("1", "pyops", 8.0e6),
                          _cpu_at("2", "pyops", 4.0e6)])
    assert s["fleet_median_per_core_s"] == 6.0e6 and s["spread"] == 2.0


def test_the_table_scores_each_row_against_its_own_unit(store, capsys):
    store.put(hf.instance_key("1", "t1", hf.KIND_CPU),
              _cpu_at("1", "pyops", 8.0e6))
    store.put(hf.instance_key("3", "t1", hf.KIND_CPU),
              _cpu_at("3", "merge_tensors", 0.1))
    hf.main(["--local", store.root, "list", "--kind", "cpu"])
    out = capsys.readouterr().out
    assert "0.00x" not in out, "a row scored against another unit's median"
    assert out.count("fleet median") == 2, "one median line per unit"
