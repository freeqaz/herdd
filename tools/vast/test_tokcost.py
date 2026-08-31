"""test_tokcost.py — $/Mtok arithmetic, the gen_stats_v1 scan, and the OFFLINE
cost lookup. No network, no live fleet: every fixture is a tmp dir plus a tmp
sqlite cache built with the same schema herdd writes.

The properties worth protecting here are the ones that fake success:
  * an absent price must render as absent, never as $0.00/Mtok;
  * a scan spanning two boxes at different rates must NOT collapse to one
    rollup price;
  * `--box` must resolve from the local cache with the network untouched (the
    tool is offline by default, and --live is the only door out);
  * a destroyed box is the NORMAL case — its rate lives in `runs`, not
    `instances`, and an instances-only lookup would silently price nothing.
"""
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokcost  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _stats(**over):
    """A schema-exact gen_stats_v1 record (the sidecar gen_client.py writes)."""
    d = {
        "schema": "gen_stats_v1", "model": "qwen3-8b", "prompts": 40,
        "requests": 40, "prompt_tokens": 100_000, "completion_tokens": 200_000,
        "total_tokens": 300_000, "wall_s": 3600.0, "gen_tok_per_s": 55.5,
        "k": 5, "concurrency": 8, "max_new": 2048,
        "t_start_unix": 1_760_000_000.0, "t_end_unix": 1_760_003_600.0,
        "resumed": False,
    }
    d.update(over)
    return d


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    return path


def _tree(tmp_path, run="job-A", cells=("cell1", "cell2"), **over):
    root = tmp_path / "jobs"
    for c in cells:
        _write(str(root / run / c / "gens.stats.json"), _stats(**over))
    return str(root)


def _cache(tmp_path, instances=(), runs=()):
    """A tmp infra-metadata.db with just the columns tokcost reads, in the same
    shape herdd's _INFRA_CACHE_SCHEMA declares."""
    db = str(tmp_path / "infra-metadata.db")
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE instances(iid INTEGER PRIMARY KEY, machine_id INTEGER,
          gpu TEXT, hourly REAL, geo TEXT, run_id TEXT);
        CREATE TABLE runs(run TEXT PRIMARY KEY, gpu TEXT, dph REAL,
          instance_id TEXT);
    """)
    conn.executemany("INSERT INTO instances VALUES(?,?,?,?,?,?)", instances)
    conn.executemany("INSERT INTO runs VALUES(?,?,?,?)", runs)
    conn.commit()
    conn.close()
    return db


# --------------------------------------------------------------------------- #
# the arithmetic
# --------------------------------------------------------------------------- #
def test_usd_per_mtok_is_the_documented_formula():
    # $1.00/hr for exactly one hour producing exactly 1M tokens = $1.00/Mtok.
    assert tokcost.usd_per_mtok(1.0, 3600.0, 1_000_000) == 1.0
    # half the tokens in the same hour is twice the unit cost
    assert tokcost.usd_per_mtok(1.0, 3600.0, 500_000) == 2.0
    # twice the rate is twice the unit cost
    assert tokcost.usd_per_mtok(2.0, 3600.0, 1_000_000) == 2.0


def test_missing_or_zero_inputs_are_none_never_zero():
    """A $0.00/Mtok cell reads as 'free' and would rank first in every host
    comparison. Absent must stay absent."""
    for bad in (
        (None, 3600.0, 1_000_000),      # no price
        (1.0, None, 1_000_000),         # no wall
        (1.0, 3600.0, None),            # no tokens
        (1.0, 3600.0, 0),               # produced nothing
        (0.0, 3600.0, 1_000_000),       # free box? not a real rate
        (1.0, 0.0, 1_000_000),          # no time passed
    ):
        assert tokcost.usd_per_mtok(*bad) is None, bad


# --------------------------------------------------------------------------- #
# the scan
# --------------------------------------------------------------------------- #
def test_scan_finds_sidecars_and_ignores_other_json(tmp_path):
    root = _tree(tmp_path)
    _write(os.path.join(root, "job-A", "results.json"), {"schema": "other"})
    _write(os.path.join(root, "job-A", "cell1", "other.stats.json"),
           {"schema": "not_gen_stats", "x": 1})
    rep = tokcost.build_report([root], dph=1.0)
    assert rep["rollup"]["files"] == 2
    assert rep["rollup"]["skipped_files"] == 1     # the wrong-schema *stats.json
    assert rep["rollup"]["completion_tokens"] == 400_000


def test_unparseable_and_oversized_sidecars_degrade_not_crash(tmp_path):
    root = _tree(tmp_path, cells=("cell1",))
    with open(os.path.join(root, "job-A", "broken.stats.json"), "w") as fh:
        fh.write("{not json")
    big = os.path.join(root, "job-A", "huge.stats.json")
    with open(big, "w") as fh:
        fh.write(" " * (tokcost.MAX_STATS_BYTES + 1))
    rep = tokcost.build_report([root], dph=1.0)
    assert rep["rollup"]["files"] == 1
    assert rep["rollup"]["skipped_files"] == 2


def test_missing_root_is_reported_not_raised(tmp_path):
    rep = tokcost.build_report([str(tmp_path / "nope")])
    assert rep["missing_roots"] and rep["rollup"]["files"] == 0
    assert "does not exist" in tokcost.render(rep)


def test_unknown_keys_never_break_the_scan(tmp_path):
    """gen_stats_v1 grows fields (concurrency_mode landed after the first
    sidecars were written). A reader that trips over an unrecognised key turns
    every future producer change into a silent data outage here."""
    root = str(tmp_path / "jobs")
    _write(os.path.join(root, "job-A", "c1", "gens.stats.json"),
           _stats(concurrency_mode="auto", some_field_from_2027={"a": [1, 2]}))
    _write(os.path.join(root, "job-A", "c2", "gens.stats.json"), _stats())
    rep = tokcost.build_report([root], dph=1.0)
    assert rep["rollup"]["files"] == 2
    modes = {r["concurrency_mode"] for r in rep["rows"]}
    assert modes == {"auto", None}       # absent stays absent, never invented


def test_concurrency_is_reported_as_resolved_not_configured(tmp_path):
    """The box sizes concurrency at runtime off its own cpuset; the sidecar
    records what it RESOLVED to plus how. Both travel through to the JSON."""
    root = str(tmp_path / "jobs")
    _write(os.path.join(root, "job-A", "c", "gens.stats.json"),
           _stats(concurrency=48, concurrency_mode="pinned"))
    r = tokcost.build_report([root], dph=1.0)["rows"][0]
    assert r["concurrency"] == 48 and r["concurrency_mode"] == "pinned"


def test_run_id_is_the_first_segment_under_the_root(tmp_path):
    root = _tree(tmp_path, run="20260816T010203-e3-evals")
    rep = tokcost.build_report([root], dph=1.0)
    assert {r["run"] for r in rep["rows"]} == {"20260816T010203-e3-evals"}


# --------------------------------------------------------------------------- #
# the two dollar figures
# --------------------------------------------------------------------------- #
def test_gen_attributed_and_box_amortized_are_separate_figures(tmp_path):
    # one cell: 1 hour of gen, 1M completion tokens, $2/hr box billed 4 hours.
    root = _tree(tmp_path, cells=("cell1",), completion_tokens=1_000_000,
                 wall_s=3600.0)
    rep = tokcost.build_report([root], dph=2.0, box_wall_s=4 * 3600.0)
    roll = rep["rollup"]
    assert roll["gen_attributed_usd_per_mtok"] == 2.0
    assert roll["box_amortized_usd_per_mtok"] == 8.0     # 4x the billed wall
    assert abs(roll["gen_wall_frac_of_box"] - 0.25) < 1e-9


def test_box_amortized_is_absent_without_box_wall(tmp_path):
    """Silence, not a guess: with no billed wall the whole-rental figure does
    not exist and the renderer says how to get it."""
    rep = tokcost.build_report([_tree(tmp_path)], dph=2.0)
    assert rep["rollup"]["box_amortized_usd_per_mtok"] is None
    assert "--box-wall-s" in tokcost.render(rep)


def test_unpriced_scan_still_reports_tokens_and_rate(tmp_path):
    rep = tokcost.build_report([_tree(tmp_path)],
                               cache_db=str(tmp_path / "absent.db"))
    roll = rep["rollup"]
    assert roll["completion_tokens"] == 400_000
    assert roll["tok_per_s_derived"] is not None
    assert roll["dph"] is None
    assert roll["gen_attributed_usd_per_mtok"] is None
    text = tokcost.render(rep)
    assert "$0.00" not in text and "$0.000" not in text


# --------------------------------------------------------------------------- #
# the OFFLINE cost lookup
# --------------------------------------------------------------------------- #
def test_box_rate_from_live_instances_row(tmp_path, monkeypatch):
    _no_home_caches(tmp_path, monkeypatch)
    db = _cache(tmp_path, instances=[(47018759, 142018, "RTX 5090", 1.21,
                                      "US-West", "run-x")])
    f = tokcost.box_rate_cached("47018759", db)
    assert f["dph"] == 1.21
    assert f["source"] == "cache:instances.hourly"
    assert f["machine_id"] == 142018 and f["gpu"] == "RTX 5090"


def test_box_rate_falls_back_to_runs_for_a_destroyed_box(tmp_path, monkeypatch):
    """The instances table only holds boxes that exist RIGHT NOW. Every finished
    eval ran on a box that no longer does, so runs.dph is the normal path, not
    the exotic one."""
    _no_home_caches(tmp_path, monkeypatch)
    db = _cache(tmp_path, instances=[], runs=[("job-A", "H100", 2.5, "44409354")])
    f = tokcost.box_rate_cached("44409354", db)
    assert f["dph"] == 2.5 and f["source"] == "cache:runs.dph"
    assert f["run"] == "job-A"


def test_box_rate_missing_cache_is_a_blank_price_not_an_error(tmp_path,
                                                              monkeypatch):
    _no_home_caches(tmp_path, monkeypatch)
    f = tokcost.box_rate_cached("123", str(tmp_path / "no-such.db"))
    assert f["dph"] is None and f["source"] is None


# --- the four-source chain -------------------------------------------------- #
def _no_home_caches(tmp_path, monkeypatch):
    """Point every home-dir cache tokcost consults at an empty tmp tree, so a
    test asserting a MISS cannot be rescued by this workstation's real caches."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("FLEETD_STATE_DIR", str(tmp_path / "fleetd"))


def _ls_snapshot(tmp_path, instances):
    d = tmp_path / "xdg" / "herdd"
    d.mkdir(parents=True, exist_ok=True)
    (d / "ls-snapshot.json").write_text(json.dumps({"instances": instances}))


def _runmeta_event(tmp_path, run, name, ev):
    d = tmp_path / "xdg" / "vast-runmeta" / run / "events"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(ev))


def _journal(tmp_path, rows):
    d = tmp_path / "fleetd"
    d.mkdir(parents=True, exist_ok=True)
    (d / "journal.ndjsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))


def test_ls_snapshot_is_the_first_source_and_carries_the_billed_rate(
        tmp_path, monkeypatch):
    _no_home_caches(tmp_path, monkeypatch)
    _ls_snapshot(tmp_path, [{"id": 47018759, "dph_total": 1.21,
                             "machine_id": 142018, "gpu_name": "RTX 5090",
                             "geolocation": ", US-West",
                             "label": "run:e3-evals keep:why"}])
    db = _cache(tmp_path, runs=[("r", "H100", 9.99, "47018759")])
    f = tokcost.box_rate_cached("47018759", db)
    assert f["dph"] == 1.21 and f["source"] == "cache:ls-snapshot.dph_total"
    assert f["machine_id"] == 142018 and f["geo"] == "US-West"
    assert f["run"] == "e3-evals"


def test_ls_snapshot_matches_the_exact_iid_only(tmp_path, monkeypatch):
    """The vast suite's own tests clobber this file with fixture boxes 1/2/3 at
    $0.10. Exact-id matching is what keeps that from pricing a real box."""
    _no_home_caches(tmp_path, monkeypatch)
    _ls_snapshot(tmp_path, [{"id": 1, "dph_total": 0.1, "label": "run:a"},
                            {"id": 2, "dph_total": 0.1, "label": "run:b"}])
    assert tokcost.box_rate_cached("47018759",
                                   str(tmp_path / "none.db"))["dph"] is None


def test_runmeta_launched_event_prices_a_destroyed_box(tmp_path, monkeypatch):
    """No live instance row, no runs row — just the immutable launched event
    already mirrored to disk. This is the common case for a finished eval."""
    _no_home_caches(tmp_path, monkeypatch)
    _runmeta_event(tmp_path, "e3-run", "20260816T000000000Z-cli_a-n1.json", {
        "event": "launched", "instance_id": 47018759, "machine_id": 142018,
        "gpu": "RTX 5090", "dph": 0.8888, "run_id": "e3-run",
        "ts": "20260816T000000000Z"})
    f = tokcost.box_rate_cached("47018759", str(tmp_path / "none.db"))
    assert f["dph"] == 0.8888 and f["source"] == "cache:runmeta.launched.dph"
    assert f["machine_id"] == 142018 and f["run"] == "e3-run"


def test_fleetd_journal_is_the_last_local_source(tmp_path, monkeypatch):
    _no_home_caches(tmp_path, monkeypatch)
    _journal(tmp_path, [
        {"event": "tick", "iid": "47018759", "spend_usd": 1.0},
        {"event": "unwatched", "iid": "47018759", "dph": 0.6888,
         "dph_known": True},
    ])
    f = tokcost.box_rate_cached("47018759", str(tmp_path / "none.db"))
    assert f["dph"] == 0.6888 and f["source"] == "cache:fleetd-journal.dph"


def test_journal_dph_known_false_is_not_a_free_box(tmp_path, monkeypatch):
    """fleetd journals an unreadable rate as 0.0 with dph_known:false. Honouring
    the flag is the difference between 'unknown' and 'free'."""
    _no_home_caches(tmp_path, monkeypatch)
    _journal(tmp_path, [{"event": "unwatched", "iid": "47018759", "dph": 0.0,
                         "dph_known": False}])
    assert tokcost.box_rate_cached("47018759",
                                   str(tmp_path / "none.db"))["dph"] is None


def test_host_facts_carry_forward_from_a_priceless_earlier_source(
        tmp_path, monkeypatch):
    """The infra-db knows the machine but not the price; the journal knows the
    price but not the machine. The chain must end up with both."""
    _no_home_caches(tmp_path, monkeypatch)
    db = _cache(tmp_path, instances=[(47018759, 142018, "RTX 5090", None,
                                      "US-West", None)])
    _journal(tmp_path, [{"event": "unwatched", "iid": "47018759", "dph": 0.5,
                         "dph_known": True}])
    f = tokcost.box_rate_cached("47018759", db)
    assert f["dph"] == 0.5 and f["machine_id"] == 142018
    assert f["gpu"] == "RTX 5090"


def test_a_corrupt_local_cache_is_a_miss_not_a_crash(tmp_path, monkeypatch):
    _no_home_caches(tmp_path, monkeypatch)
    d = tmp_path / "xdg" / "herdd"
    d.mkdir(parents=True)
    (d / "ls-snapshot.json").write_text("{not json")
    _journal(tmp_path, [{"event": "unwatched", "iid": "7", "dph": 0.25,
                         "dph_known": True}])
    assert tokcost.box_rate_cached("7", str(tmp_path / "none.db"))["dph"] == 0.25


def test_auto_resolves_the_rate_from_the_run_id_with_no_flags(tmp_path):
    db = _cache(tmp_path, instances=[(5001, 777, "RTX 5090", None, "EU", None)],
                runs=[("job-A", "RTX 5090", 0.5, "5001")])
    rep = tokcost.build_report([_tree(tmp_path)], cache_db=db)
    assert rep["rollup"]["dph"] == 0.5
    r = rep["rows"][0]
    assert r["dph_source"] == "cache:runs.dph"
    assert r["machine_id"] == 777 and r["host"] == "m777"


def test_explicit_dph_outranks_every_lookup(tmp_path):
    db = _cache(tmp_path, runs=[("job-A", "H100", 9.99, "5001")])
    rep = tokcost.build_report([_tree(tmp_path)], dph=1.0, cache_db=db)
    assert rep["rollup"]["dph"] == 1.0
    assert rep["rows"][0]["dph_source"] == "flag:--dph"


def test_box_flag_outranks_the_per_run_auto_lookup(tmp_path, monkeypatch):
    _no_home_caches(tmp_path, monkeypatch)
    db = _cache(tmp_path,
                instances=[(5002, 888, "B200", 3.0, "US", None)],
                runs=[("job-A", "H100", 9.99, "5001")])
    rep = tokcost.build_report([_tree(tmp_path)], box="5002", cache_db=db)
    assert rep["rollup"]["dph"] == 3.0
    assert rep["rows"][0]["machine_id"] == 888


def test_lookup_never_touches_the_network_without_live(tmp_path, monkeypatch):
    """Offline by default is the contract, so make the live door explode and
    assert the default path never opens it."""
    _no_home_caches(tmp_path, monkeypatch)

    def boom(_iid):
        raise AssertionError("network lookup attempted without --live")
    monkeypatch.setattr(tokcost, "box_rate_live", boom)
    rep = tokcost.build_report([_tree(tmp_path)], box="99999999",
                               cache_db=str(tmp_path / "absent.db"))
    assert rep["rollup"]["dph"] is None


def test_live_flag_is_only_reached_after_the_cache_misses(tmp_path, monkeypatch):
    _no_home_caches(tmp_path, monkeypatch)
    calls = []

    def fake(iid):
        calls.append(iid)
        return {"dph": 1.5, "source": "live:dph_total", "machine_id": 42,
                "gpu": "H200 NVL", "geo": None, "run": None, "iid": iid}
    monkeypatch.setattr(tokcost, "box_rate_live", fake)
    # cache HIT -> no live call
    db = _cache(tmp_path, instances=[(7, 1, "X", 0.25, None, None)])
    rep = tokcost.build_report([_tree(tmp_path)], box="7", cache_db=db,
                               live=True)
    assert rep["rollup"]["dph"] == 0.25 and calls == []
    # cache MISS -> exactly one live call
    rep = tokcost.build_report([_tree(tmp_path)], box="8", cache_db=db,
                               live=True)
    assert rep["rollup"]["dph"] == 1.5 and calls == ["8"]


# --------------------------------------------------------------------------- #
# rollup / host grouping
# --------------------------------------------------------------------------- #
def test_two_boxes_at_different_rates_have_no_single_rollup_price(tmp_path):
    """Averaging two rates would invent a price nobody was billed."""
    db = _cache(tmp_path, runs=[("job-A", "H100", 1.0, "1"),
                                ("job-B", "RTX 5090", 4.0, "2")])
    root = str(tmp_path / "jobs")
    _write(os.path.join(root, "job-A", "c", "gens.stats.json"), _stats())
    _write(os.path.join(root, "job-B", "c", "gens.stats.json"), _stats())
    rep = tokcost.build_report([root], cache_db=db)
    assert rep["rollup"]["dph"] is None
    assert rep["rollup"]["gen_attributed_usd_per_mtok"] is None
    # ... but each HOST still has its own priced unit cost
    per_host = {h["gpu"]: h["gen_attributed_usd_per_mtok"] for h in rep["hosts"]}
    assert per_host["H100"] is not None and per_host["RTX 5090"] is not None
    assert per_host["RTX 5090"] > per_host["H100"]


def test_hosts_sort_cheapest_gen_attributed_first(tmp_path):
    db = _cache(tmp_path, runs=[("job-A", "H100", 4.0, "1"),
                                ("job-B", "RTX 5090", 1.0, "2")])
    root = str(tmp_path / "jobs")
    _write(os.path.join(root, "job-A", "c", "gens.stats.json"), _stats())
    _write(os.path.join(root, "job-B", "c", "gens.stats.json"), _stats())
    rep = tokcost.build_report([root], cache_db=db)
    assert [h["gpu"] for h in rep["hosts"]] == ["RTX 5090", "H100"]


def test_rollup_sums_tokens_and_wall_across_cells(tmp_path):
    root = _tree(tmp_path, cells=("a", "b", "c"))
    rep = tokcost.build_report([root], dph=1.0)
    roll = rep["rollup"]
    assert roll["files"] == 3
    assert roll["prompt_tokens"] == 300_000
    assert roll["completion_tokens"] == 600_000
    assert roll["wall_s"] == 3 * 3600.0
    # derived rate is completion/wall, NOT the mean of the reported rates
    assert abs(roll["tok_per_s_derived"] - 600_000 / (3 * 3600.0)) < 1e-9


def test_json_output_is_parseable_and_text_output_renders(tmp_path, capsys):
    root = _tree(tmp_path)
    assert tokcost.main([root, "--dph", "1.0", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == "tokcost_v1" and len(doc["rows"]) == 2
    assert tokcost.main([root, "--dph", "1.0", "--box-wall-s", "7200"]) == 0
    text = capsys.readouterr().out
    assert "gen-attributed $/Mtok" in text
    assert "box-amortized $/Mtok" in text


def test_empty_scan_explains_itself(tmp_path):
    d = str(tmp_path / "empty")
    os.makedirs(d)
    rep = tokcost.build_report([d])
    text = tokcost.render(rep)
    assert "no gen_stats_v1 sidecars found" in text


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
