"""`herdd dash-cache` — the dashboard's read-only /admin snapshot writer.

These are the guarantees the INTERNET-FACING dashboard rests on
(tools/vast/dashboard/DESIGN_V5_ADMIN.md §3, §10), so they are tests rather
than a review note:

  1. **No credential or reachability field can reach sqlite.** Every INSERT is
     a positive allowlist; `extra_env` alone carries `HF_TOKEN` and the B2 EU
     keys verbatim, and it rides every `GET v1/instances/` response. The leak
     test below feeds a record stuffed with every hard-excluded field and
     asserts none of it appears anywhere in the written rows.
  2. **Exit is 0 or 1, never 2.** `reap --json` / `guard --json` exit 2 when
     they have findings; the Node caller treats any nonzero as failure, so that
     convention must not leak into this command. A failed section is a SKIP.
  3. **The journal mode stays `delete`.** The dashboard reads with
     `sqlite3 -readonly`, which cannot open a WAL database — a stray
     `journal_mode=WAL` would silently break every page, runs view included.

Hermetic: no network, no fleetd socket, no vast API.
"""
import argparse
import dataclasses
import io
import os
import re
import sqlite3
import sys
import time
import tokenize

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import herdd  # noqa: E402
from vastlib.boxes import health as boxes_health  # noqa: E402
from vastlib.cli import _ls_render  # noqa: E402
from vastlib.cli import dash_cache as cli_dash_cache  # noqa: E402
from vastlib.core import api, models  # noqa: E402
from vastlib.fleet import client as fleet_client  # noqa: E402
from vastlib.storage import dashcache  # noqa: E402


def _deps():
    """The composition root's own `DashDeps`, rebuilt per call.

    `storage/dashcache.py` sits BELOW `cli` and `fleet` in the plan §5 DAG, so
    the reads this file stubs (`_gather_ls_data`, `_dash_write_fleet`, the
    `launch.spec` secret matchers) reach it by injection rather than by import.
    `cli/dash_cache._deps()` reads each one as a module attribute at call time,
    which is exactly what keeps the `monkeypatch.setattr(<owning module>, ...)`
    idiom below steering the ported code. Wiring only — no expectation here.
    """
    return cli_dash_cache._deps()


# Every field the contract hard-excludes, with a value we can grep for.
SECRET_MARKERS = {
    "extra_env_token": "hf_LEAKED_TOKEN_VALUE",
    "extra_env_b2": "K005LEAKEDB2KEY",
    "image_login": "glpat-LEAKEDREGISTRYPAT",
    "onstart": "curl-http-leaked-onstart",
    "ssh_host": "ssh9.leaked.vast.ai",
    "public_ipaddr": "203.0.113.77",
    "hostname": "node-leaked.provider.example",
    "requester": "leakeduser@leakedhost",
    "last_tail": "LEAKEDCONTAINERTAIL",
}


def _instance(now):
    return {
        "id": 46246859, "machine_id": 24815, "actual_status": "running",
        "status_msg": "boot ok, log /home/leakeduser/.local/state/vast/boot.log "
                      f"HF_TOKEN={SECRET_MARKERS['extra_env_token']}",
        "is_bid": True, "num_gpus": 2, "gpu_name": "RTX 5090", "gpu_util": 97.5,
        "dph_total": 0.55, "dph_base": 0.45, "min_bid": 0.33,
        "storage_total_cost": 0.0888,
        "disk_space": 160.0, "disk_usage": 18.0,          # 11% of 160 GB
        "start_date": now - 7200, "geolocation": ", US",
        "label": "wave:rb3-wide-A keep:FLOOR-repair-pending",
        "image_uuid": "registry.gitlab.com/acme/trainer:train-latest",
        "extra_env": [["HF_TOKEN", SECRET_MARKERS["extra_env_token"]],
                      ["B2_APPLICATION_KEY_EU", SECRET_MARKERS["extra_env_b2"]]],
        "image_login": SECRET_MARKERS["image_login"],
        "onstart": SECRET_MARKERS["onstart"],
        "ssh_host": SECRET_MARKERS["ssh_host"], "ssh_port": 12345,
        "public_ipaddr": SECRET_MARKERS["public_ipaddr"],
        "hostname": SECRET_MARKERS["hostname"], "host_id": 987654,
        "requester": SECRET_MARKERS["requester"],
    }


def _stopped(now):
    return {
        "id": 46193810, "machine_id": 41526, "actual_status": "stopped",
        "is_bid": False, "num_gpus": 1, "gpu_name": "RTX 3090",
        "dph_total": 0.2, "dph_base": 0.15,
        # a loading/stopped box reports -1 == UNKNOWN, and it must not read as 0
        "disk_space": 120.0, "disk_usage": -1,
        "start_date": now - 86400, "label": "serve:eval", "geolocation": ", DE",
        "image_uuid": "pytorch/pytorch@sha256:b85566342b8612abcdef",
        "extra_env": [],
    }


@pytest.fixture
def gathered(monkeypatch):
    """A fake `_gather_ls_data` payload: one live bid box + one idle stopped
    box past the reap deadline, plus a job view whose raw tail is a leak
    marker."""
    now = time.time()
    data = {
        "ts": now, "no_spot": False,
        "instances": [_instance(now), _stopped(now)],
        "live_ids": [46246859],
        "jobs_by_box": {"46246859": [{
            "job_id": "j1", "name": "train-a", "display_status": "running",
            "n_checkpoints": 3,
            "last_tail": SECRET_MARKERS["last_tail"] +
                         " 83%|xx| 572/688 [1:55:27<23:29, 12.15s/it]"}]},
        "market": {}, "stale_ids": [46246859],
        "idle_secs": {"46193810": 9000.0},
        "health": {"46246859": {
            "verdict": boxes_health.GUARD_STALE_IMAGE,
            "reason": "tag moved since launch (/home/leakeduser/img.txt)"}},
    }
    monkeypatch.setattr(_ls_render, "_gather_ls_data", lambda **kw: data)
    return data


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "infra-metadata.db")
    conn = sqlite3.connect(path, timeout=5)
    conn.execute("PRAGMA busy_timeout=3000")
    conn.executescript(dashcache._INFRA_CACHE_SCHEMA)
    yield path, conn
    conn.close()


def _flatten(conn, table):
    return " ".join(str(x) for row in conn.execute(f"SELECT * FROM {table}")
                    for x in row)


# --------------------------------------------------------------------------- #
# 1. the allowlist: nothing hard-excluded can reach the table
# --------------------------------------------------------------------------- #
def test_instances_write_leaks_no_credential_or_reachability_field(gathered, db):
    path, conn = db
    assert dashcache._dash_write_instances(conn, deps=_deps()) == 2
    blob = _flatten(conn, "instances")
    leaked = sorted(k for k, v in SECRET_MARKERS.items() if v in blob)
    assert leaked == [], f"hard-excluded field(s) reached sqlite: {leaked}"


def test_instances_write_publishes_no_absolute_machine_path(gathered, db):
    _path, conn = db
    dashcache._dash_write_instances(conn, deps=_deps())
    blob = _flatten(conn, "instances")
    assert "/home/" not in blob and "/Users/" not in blob
    # ...but the useful tail of the path survives, so the field stays legible
    row = conn.execute("SELECT status, health_reason FROM instances "
                       "WHERE iid=46246859").fetchone()
    assert "boot.log" in row[0]
    assert "img.txt" in row[1]


def test_instances_insert_columns_match_the_ddl_exactly(gathered, db):
    """The INSERT names its columns on purpose (no dict splat). If the DDL and
    the INSERT ever drift, the allowlist has a hole — so pin them together."""
    _path, conn = db
    ddl = [r[1] for r in conn.execute("PRAGMA table_info(instances)")]
    named = dashcache._DASH_INSTANCES_INSERT.split("(", 1)[1].split(")", 1)[0]
    assert [c.strip() for c in named.split(",")] == ddl


def test_no_dashboard_table_has_a_requester_or_ssh_column(db):
    """`fleet status` rows carry `requester` (user@hostname) and instances carry
    ssh/ip reachability. Neither may EXIST as a column — the guarantee is
    structural, not a filter someone can forget."""
    _path, conn = db
    banned = {"requester", "hostname", "host_id", "ssh_host", "ssh_port",
              "public_ipaddr", "extra_env", "image_login", "onstart",
              "last_tail", "email"}
    for (table,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"):
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        assert not (cols & banned), f"{table} exposes {cols & banned}"


# --------------------------------------------------------------------------- #
# 2. projections that are easy to get wrong
# --------------------------------------------------------------------------- #
def test_projection_derivations(gathered, db):
    _path, conn = db
    dashcache._dash_write_instances(conn, deps=_deps())
    conn.row_factory = sqlite3.Row
    live = conn.execute("SELECT * FROM instances WHERE iid=46246859").fetchone()
    idle = conn.execute("SELECT * FROM instances WHERE iid=46193810").fetchone()

    # A bid box is billed ITS OWN BID (`dph_total`), never the market floor.
    # This fixture is the stuck-high shape the check exists for: we pay $0.55
    # while the floor sits at $0.33 + $0.10 disk = $0.43. Publishing the floor
    # as `hourly` understated the burn 1.3x here (6.3x on the real J1 box) and
    # made /market's `vs_floor` compare the floor against itself.
    assert live["mode"] == "bid"
    assert live["hourly"] == pytest.approx(0.55)          # dph_total == our bid
    assert live["spot"] == pytest.approx(0.33 + 0.10)     # counterfactual floor
    assert live["hourly"] > live["spot"]
    assert live["ondemand"] == pytest.approx(0.55)        # no market -> dph_total
    # 18 GB used of a 160 GB allocation -> the oversizing flag
    assert live["disk_frac"] == pytest.approx(18.0 / 160.0)
    assert live["disk_oversized"] == 1
    # `keep:<why>` as an appended GROUP still opts the box out of the reaper
    assert live["keep"] == 1
    # ", US" -> "US"
    assert live["geo"] == "US" and idle["geo"] == "DE"
    assert live["image_short"] == "trainer:train-latest"
    assert idle["image_short"] == "pytorch@b85566342b86"
    assert live["stale_image"] == 1
    assert live["health_verdict"] == boxes_health.GUARD_STALE_IMAGE
    # the jobs cell carries the PARSED scalars only, never the tail it came from
    # …and the rate is LABELLED. One tqdm bar in the tail means there is no
    # second elapsed stamp to subtract, so this is tqdm's own attempt-wide
    # aggregate, not a step time — `~…(avg)` says so. With two bars the cell
    # reads a bare `101s/it`, the consecutive-step delta (test_jobprogress_rate).
    assert live["jobs"] == "train-a:running:83%:~12.2s/it(avg):ckpt3"
    assert live["n_jobs"] == 1
    # disk_usage: -1 is UNKNOWN, not 0 — otherwise every booting box reads as
    # 100% wasted allocation
    assert idle["disk_used_gb"] is None and idle["disk_frac"] is None
    assert idle["disk_oversized"] == 0


def test_reap_verdict_is_a_preview_never_a_destroy(gathered, db, monkeypatch):
    """The verdict is recomputed in-process from the same idle ledger `reap`
    reads. A live box is not an idle-lane candidate and gets NULL."""
    _path, conn = db
    monkeypatch.setenv("HERDD_REAP_IDLE_H", "2")
    dashcache._dash_write_instances(conn, deps=_deps())
    rows = dict(conn.execute("SELECT iid, reap_verdict FROM instances"))
    assert rows[46246859] is None               # live: idle lane skips it
    assert rows[46193810] == "REAP"             # stopped 2.5 h, no keep token

    monkeypatch.setenv("HERDD_REAP_IDLE_H", "4")
    dashcache._dash_write_instances(conn, deps=_deps())
    r = conn.execute("SELECT reap_verdict, reap_wait_s FROM instances "
                     "WHERE iid=46193810").fetchone()
    assert r[0] == "WAIT" and r[1] == pytest.approx(4 * 3600 - 9000, abs=2)


@pytest.mark.parametrize("raw,expect", [
    ("/home/someone/code/repo/out/train.log", "train.log"),
    ("~/.local/state/vast-fleetd/fleetd.sock", "fleetd.sock"),
    ("started HF_TOKEN=hf_realtokenvalue ok", "started HF_TOKEN=<redacted> ok"),
    # `GITHUB_PAT` is a real launch env here and is OUTSIDE the B2-spec secret
    # family (TOKEN|KEY|SECRET|PASS|PWD|CRED|AUTH|PRIVATE|SIGNATURE|SESSION);
    # a create failure echoes the whole `docker run` into `status_msg`.
    ("create failed: docker run -e GITHUB_PAT=ghp_realpatvalue img",
     "create failed: docker run -e GITHUB_PAT=<redacted> img"),
    # ...and a token with no `NAME=` in front of it at all.
    ("pull denied for glpat-abcd1234EFGH5678", "pull denied for <redacted>"),
    # PATH must survive: widening the SHARED family would strand it out of the
    # durable launch spec, which is why the widening is dash-side only.
    ("env PATH=/usr/bin ok", "env PATH=/usr/bin ok"),
    ("rclone https://kid:secret@host/p", "rclone https://<redacted>@host/p"),
    ("waiting ~2h for the reaper", "waiting ~2h for the reaper"),
    ("   ", None),
    (None, None),
])
def test_scrub(raw, expect):
    assert dashcache._dash_scrub(raw, deps=_deps()) == expect


def test_scrub_truncates():
    assert len(dashcache._dash_scrub("x" * 900, dashcache.DASH_STATUS_MAX,
                                     deps=_deps())) == 200


def test_verified_reads_the_verification_field_not_the_filter_key():
    """The bundles response spells it `verification: "verified"`; the SEARCH
    FILTER spells it `verified`. Reading the filter's key off a response row
    yields None for every offer — measured live 2026-08-01, 469/469 rows."""
    assert models._dash_verified({"verification": "verified"}) == 1
    assert models._dash_verified({"verification": "unverified"}) == 0
    assert models._dash_verified({"verified": True}) == 1
    assert models._dash_verified({}) == 0


# --------------------------------------------------------------------------- #
# 3. market: the bid-mode price field, and "unobtainable" as a signal
# --------------------------------------------------------------------------- #
def _offer(**kw):
    o = {"machine_id": 1, "geolocation": "Washington, US", "num_gpus": 2,
         "dph_total": 1.0, "min_bid": 0.4, "storage_cost": 0.2,
         "gpu_ram": 32607, "cuda_max_good": 13.2, "reliability": 0.99,
         "inet_down": 600.0, "inet_up": 650.0, "disk_space": 2700.0,
         "verification": "verified", "rentable": True}
    o.update(kw)
    return o


def test_bid_mode_prices_on_min_bid_not_the_list_price(monkeypatch):
    """In bid mode `dph_total` is the on-demand list price you do NOT pay."""
    offers = [_offer(min_bid=0.4, dph_total=1.0),
              _offer(min_bid=0.8, dph_total=2.0, machine_id=2)]
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (True, {"offers": offers}, None))
    row, orows = dashcache._dash_market_probe("RTX 5090", 2, "bid", deps=_deps())
    assert row[4] == "min_bid"
    assert row[5] == pytest.approx(0.4)        # p0 == the floor
    assert row[9] == pytest.approx(0.4)        # best_total
    assert row[10] == pytest.approx(0.2)       # best_per_gpu (2-GPU bundle)
    assert orows[0][6] == pytest.approx(0.4) and orows[0][8] == pytest.approx(1.0)

    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (True, {"offers": offers}, None))
    row, _ = dashcache._dash_market_probe("RTX 5090", 2, "ondemand", deps=_deps())
    assert row[4] == "dph_total" and row[9] == pytest.approx(1.0)


def test_zero_offers_is_a_row_not_a_gap(monkeypatch):
    """"Unobtainable at any price" is a finding; a missing row would read as
    "we never looked". The `ok_*_min` tail rides along even here: it describes
    the LENS, not the sample, and the UI labels the filter from it."""
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (True, {"offers": []}, None))
    row, orows = dashcache._dash_market_probe("B300", 8, "bid", deps=_deps())
    assert row[:5] == ("B300", 8, "bid", 0, "min_bid")
    assert len(row) == 28 and orows == []
    assert set(row[5:16]) == {None}           # permissive prices
    assert row[16] == 0                       # n_ok
    assert set(row[17:25]) == {None}          # the _ok prices and best_ok_*
    assert row[25:] == dashcache._dash_launch_floors()


# --------------------------------------------------------------------------- #
# 3b. dynamic GPU-class discovery, and the fallback that must never blank the
#     board. Owner ask 2026-08-18: "show new GPUs as they show up".
# --------------------------------------------------------------------------- #
def _census(name, n=3, ram=32607, cap=890):
    """`n` offers of one class, with the two fields discovery filters on."""
    return [_offer(gpu_name=name, gpu_ram=ram, compute_cap=cap, machine_id=i)
            for i in range(n)]


def _discover(monkeypatch, offers, ok=True, err=None):
    monkeypatch.setattr(api, "request_soft", lambda *a, **k: (ok, {"offers": offers}, err))
    deps = dataclasses.replace(_deps(), census_query=lambda: {})
    return dashcache._dash_discover_gpus(deps=deps)


def test_discovery_adds_a_class_the_builtin_set_never_heard_of(monkeypatch):
    """The whole point: silicon nobody typed into the tuple still reaches the
    board. It is APPENDED, so the audited set keeps its order and its place."""
    offers = _census("RTX 9999", 5) + _census("RTX 5090", 40)
    offers += [_offer(gpu_name="filler", machine_id=900 + i) for i in range(30)]
    gpus, note = _discover(monkeypatch, offers)
    assert gpus[:len(dashcache.DASH_GPUS_DEFAULT)] == list(dashcache.DASH_GPUS_DEFAULT)
    assert "RTX 9999" in gpus and "RTX 9999" in note


def test_discovery_never_drops_a_builtin_class(monkeypatch):
    """A class whose supply is momentarily dry must not vanish. L40 had ZERO
    offers in the 2026-08-18 census and is still ours to price."""
    gpus, _note = _discover(monkeypatch, _census("RTX 5090", 60))
    assert set(dashcache.DASH_GPUS_DEFAULT) <= set(gpus)
    assert "L40" in gpus


@pytest.mark.parametrize("kind,offers,ok,err", [
    ("read failed", [], False, "HTTP 500"),
    ("short census", [_offer(gpu_name="RTX 9999")], True, None),
])
def test_discovery_falls_back_to_the_builtin_set(monkeypatch, kind, offers, ok, err):
    """A discovery bug must not be able to empty the page. Every failure path
    returns the known-good tuple AND says why — never an exception, never []."""
    gpus, note = _discover(monkeypatch, offers, ok=ok, err=err)
    assert gpus == list(dashcache.DASH_GPUS_DEFAULT), kind
    assert "built-in set" in note, note


def test_discovery_survives_a_malformed_response(monkeypatch):
    """Garbage in the offer list is a skipped row, not a traceback."""
    offers = [None, "nonsense", {}, {"gpu_name": 42}, {"gpu_name": "x" * 99},
              {"gpu_name": "RTX 9999", "gpu_ram": None, "compute_cap": None}]
    offers += _census("RTX 5090", 40)
    gpus, note = _discover(monkeypatch, offers)
    assert set(dashcache.DASH_GPUS_DEFAULT) <= set(gpus)
    assert "x" * 99 not in gpus and 42 not in gpus


def test_discovery_rejects_pre_bf16_and_undersized_silicon(monkeypatch):
    """Two exclusions with a measured basis. `compute_cap` is sm x10, so
    sm_75 (Turing) cannot do bf16 at all; and a nominal-16 GB card is under
    the working-set floor however many of them a host bolts together."""
    offers = (_census("Tesla V100", 30, ram=32768, cap=700)
              + _census("RTX 4080", 30, ram=16376, cap=890)
              + _census("RTX 9999", 5))
    gpus, _note = _discover(monkeypatch, offers)
    assert "Tesla V100" not in gpus and "RTX 4080" not in gpus
    assert "RTX 9999" in gpus, "the control class must still land"


def test_a_multi_card_box_cannot_vouch_for_an_undersized_class(monkeypatch):
    """Measured 2026-08-18: RTX 4080 is advertised as BOTH 16376 and 32760 MiB
    — some hosts report the whole box. Taking any single offer would let a
    2x16 GB listing carry a 16 GB class over a 22 GB floor, so the class is
    judged on its SMALLEST advertised figure."""
    offers = (_census("RTX 4080", 20, ram=32760, cap=890)
              + _census("RTX 4080", 4, ram=16376, cap=890)
              + _census("RTX 9999", 5))
    gpus, _note = _discover(monkeypatch, offers)
    assert "RTX 4080" not in gpus


def test_nominal_24gb_classes_clear_the_vram_floor(monkeypatch):
    """The floor is 22, not 24, because a 24 GB part does not advertise 24 GiB:
    L4 reports 23034 MiB and A10 23028. A 24.0 floor excluded exactly the
    classes it was written to admit."""
    offers = _census("L4", 10, ram=23034) + _census("A10", 7, ram=23028, cap=860)
    offers += _census("RTX 5090", 30)
    gpus, _note = _discover(monkeypatch, offers)
    assert "L4" in gpus and "A10" in gpus


def test_discovery_is_bounded_by_the_class_cap(monkeypatch):
    """The cost bound. Discovery is driven by a third party's inventory, so a
    filter bounds nothing — only the cap does, and a deferral is NAMED."""
    offers = []
    for i in range(40):                      # far more than the cap can take
        offers += _census(f"RTX 9{i:03d}", 3)
    gpus, note = _discover(monkeypatch, offers)
    assert len(gpus) == dashcache.DASH_GPUS_MAX
    assert len(gpus) == len(set(gpus)), "the probe set must not repeat a class"
    assert "cap" in note and "deferred" in note


def test_discovery_spends_its_last_slots_on_the_deepest_supply(monkeypatch):
    """When the cap binds, rank by offer count — not alphabetical luck."""
    monkeypatch.setattr(dashcache, "DASH_GPUS_MAX",
                        len(dashcache.DASH_GPUS_DEFAULT) + 1)
    offers = _census("zzz deep", 30) + _census("aaa thin", 2)
    gpus, _note = _discover(monkeypatch, offers)
    assert "zzz deep" in gpus and "aaa thin" not in gpus


def test_discovery_probe_count_stays_inside_the_stated_budget():
    """The comment on DASH_GPUS_MAX quotes a probe count and a wall time; if
    the cap moves without the arithmetic being redone, this is the tell."""
    probes = dashcache.DASH_GPUS_MAX * len(dashcache.DASH_NUM_GPUS_DEFAULT) * 2
    seconds = probes / dashcache.DASH_MARKET_MAX_RPS
    assert probes <= 280, probes
    assert seconds <= 90, seconds          # comfortably inside a 15-min refresh


def test_an_unwired_census_degrades_to_the_builtin_set():
    """`census_query` is optional: an unwired composition root falls back to the
    old static behaviour rather than failing the whole market section."""
    gpus, note = dashcache._dash_discover_gpus(
        deps=dataclasses.replace(_deps(), census_query=None))
    assert gpus == list(dashcache.DASH_GPUS_DEFAULT)
    assert "skipped" in note


def test_market_offers_are_capped(monkeypatch):
    """The stub must exceed the cap or the assertion is vacuous."""
    n = dashcache.DASH_OFFERS_KEPT + 7
    monkeypatch.setattr(api, "request_soft", lambda *a, **k: (
        True, {"offers": [_offer(min_bid=0.1 * i, machine_id=i)
                          for i in range(1, n + 1)]}, None))
    row, orows = dashcache._dash_market_probe("RTX 4090", 2, "bid", deps=_deps())
    assert row[3] == n                        # every offer counted...
    assert len(orows) == dashcache.DASH_OFFERS_KEPT      # ...only 40 persisted
    assert [o[3] for o in orows] == list(range(dashcache.DASH_OFFERS_KEPT))


def test_market_probe_keeps_only_the_exact_gpu_count(monkeypatch):
    """`build_search_query` filters `num_gpus >= N` (a launch takes a bigger
    box), but every column here is a per-CONFIGURATION claim rendered on a row
    labelled `xN`. Percentiles over a 1/2/4/8-GPU mix describe no configuration
    at all, and a cheap 2-GPU box could become the `x1` floor `OurBid.floor` is
    graded against."""
    offers = [_offer(num_gpus=2, min_bid=0.30, machine_id=2),   # bigger box
              _offer(num_gpus=1, min_bid=0.50, machine_id=1),   # the real x1
              _offer(num_gpus=4, min_bid=0.90, machine_id=4)]
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (True, {"offers": offers}, None))
    row, orows = dashcache._dash_market_probe("RTX 5090", 1, "bid", deps=_deps())
    assert row[3] == 1                                  # n_offers: the 1-GPU one
    assert row[5] == pytest.approx(0.50)                # p0 is NOT the 2-GPU 0.30
    assert row[9] == pytest.approx(0.50)                # best_total
    assert [o[4] for o in orows] == [1]                 # machine_id 1 only

    # ...and a configuration nobody lists is "unobtainable", not a silent
    # re-label of a bigger box.
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (True, {"offers": offers}, None))
    row8, orows8 = dashcache._dash_market_probe("RTX 5090", 8, "bid", deps=_deps())
    assert row8[:5] == ("RTX 5090", 8, "bid", 0, "min_bid")
    assert set(row8[5:16]) == {None} and set(row8[17:25]) == {None}
    assert row8[16] == 0 and orows8 == []


def test_one_failed_probe_does_not_lose_the_section(db, monkeypatch):
    _path, conn = db
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return False, None, "HTTP 500"
        return True, {"offers": [_offer()]}, None

    monkeypatch.setattr(api, "request_soft", flaky)
    assert dashcache._dash_write_market(conn, ["RTX 5090"], [1, 2], deps=_deps()) == 3
    assert conn.execute("SELECT count(*) FROM market").fetchone()[0] == 3


def test_all_probes_failing_raises_so_the_old_snapshot_is_kept(db, monkeypatch):
    _path, conn = db
    conn.execute("INSERT INTO market(gpu_name,num_gpus,kind,n_offers) "
                 "VALUES('RTX 5090',1,'bid',7)")
    conn.commit()
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (False, None, "auth_error"))
    with pytest.raises(RuntimeError):
        dashcache._dash_write_market(conn, ["RTX 5090"], [1], deps=_deps())
    assert conn.execute("SELECT n_offers FROM market").fetchone()[0] == 7


# --------------------------------------------------------------------------- #
# 3b. launch parity: the floor you can actually rent at (DESIGN v10 §1)
# --------------------------------------------------------------------------- #
def _reachable(**kw):
    """An offer that clears every launch default (the bare `_offer` does not:
    its 600 Mbps downlink is below the LAUNCH_INET_DOWN_MBPS floor)."""
    return _offer(**{"cuda_max_good": 13.2, "inet_down": 2000.0,
                     "reliability": 0.99, **kw})


def _probe_into_db(monkeypatch, db, offers, num_gpus=2):
    """One bid probe, written through the real INSERTs and read back BY NAME —
    a positional read would follow the appended columns silently."""
    _path, conn = db
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (True, {"offers": offers}, None))
    dashcache._dash_write_market(conn, ["RTX 5090"], [num_gpus],
                                 kinds=("bid",), deps=_deps())
    conn.row_factory = sqlite3.Row
    return (conn.execute("SELECT * FROM market").fetchone(),
            conn.execute("SELECT * FROM market_offers ORDER BY rank").fetchall())


def test_ok_aggregates_are_taken_over_the_full_sample_not_the_kept_window(
        db, monkeypatch):
    """The kept window is a UI budget; a floor that moved when it changed would
    be a measurement artifact. So `_ok` is computed over every exact-N offer."""
    monkeypatch.setattr(dashcache, "DASH_OFFERS_KEPT", 2)
    offers = [_offer(min_bid=0.10, machine_id=1),          # unreachable, cheap
              _offer(min_bid=0.20, machine_id=2),
              _offer(min_bid=0.30, machine_id=3),
              _reachable(min_bid=0.40, machine_id=4, geolocation=", NO"),
              _reachable(min_bid=0.50, machine_id=5)]
    m, orows = _probe_into_db(monkeypatch, db, offers)
    assert len(orows) == 2                     # only the window is persisted...
    assert m["n_offers"] == 5 and m["n_ok"] == 2          # ...the counts are not
    assert m["p0"] == pytest.approx(0.10)      # the permissive floor
    assert m["p0_ok"] == pytest.approx(0.40)   # the one a launch could take
    assert m["p90_ok"] == pytest.approx(0.49)
    assert m["best_ok_total"] == pytest.approx(0.40)
    assert m["best_ok_per_gpu"] == pytest.approx(0.20)    # 2-GPU bundle
    assert m["best_ok_machine_id"] == 4 and m["best_ok_geo"] == "NO"


def test_no_reachable_offer_is_a_finding_not_a_gap(db, monkeypatch):
    """17 live configurations are in this state: offers exist, none passes the
    launch defaults. NULL `_ok` prices say so; falling back to the permissive
    floor would reprint the confident-but-unrentable number this lens exists to
    kill."""
    m, orows = _probe_into_db(monkeypatch, db,
                              [_offer(min_bid=0.10), _offer(min_bid=0.20)])
    assert m["n_offers"] == 2 and m["n_ok"] == 0
    assert m["p0"] == pytest.approx(0.10)
    assert (m["p0_ok"], m["p50_ok"], m["best_ok_total"],
            m["best_ok_machine_id"], m["best_ok_geo"]) == (None,) * 5
    # ...and the lens is still named, so the UI can say WHICH filter emptied it
    assert (m["ok_cuda_min"], m["ok_inet_down_min"], m["ok_reliability_min"]) \
        == dashcache._dash_launch_floors()
    assert [o["launch_ok"] for o in orows] == [0, 0]


def test_reliability2_is_the_fallback_the_row_writer_already_uses(db, monkeypatch):
    """The offer body spells it either way; the predicate and the persisted
    column must read the SAME one, or `launch_ok` disagrees with its own row."""
    o = _reachable(machine_id=7)
    del o["reliability"]
    o["reliability2"] = 0.995
    m, orows = _probe_into_db(monkeypatch, db, [o])
    assert m["n_ok"] == 1
    assert orows[0]["reliability"] == pytest.approx(0.995)
    assert orows[0]["launch_ok"] == 1


def test_launch_ok_agrees_with_the_row_and_with_the_aggregate(db, monkeypatch):
    """`launch_ok` is the anti-drift seam: the UI filters on the column instead
    of re-implementing three inequalities, so the column must be derivable from
    the row it sits on, and `n_ok` must be its sum over the sample."""
    offers = [_reachable(min_bid=0.10, machine_id=1),
              _offer(min_bid=0.20, machine_id=2),                  # inet 600
              _reachable(min_bid=0.30, machine_id=3, cuda_max_good=12.4),
              _reachable(min_bid=0.40, machine_id=4, reliability=0.5)]
    m, orows = _probe_into_db(monkeypatch, db, offers)
    cuda, inet, rel = dashcache._dash_launch_floors()
    for o in orows:
        expect = (o["cuda_max_good"] >= cuda and o["inet_down"] >= inet
                  and o["reliability"] >= rel)
        assert o["launch_ok"] == (1 if expect else 0), dict(o)
    assert [o["launch_ok"] for o in orows] == [1, 0, 0, 0]
    assert m["n_ok"] == sum(o["launch_ok"] for o in orows)


def test_effective_cores_is_the_offer_slice_never_the_host_count(db, monkeypatch):
    """A 384-core host handing out 1/4 of its GPUs hands out 96 cores. Picking
    on the advertised number picks the wrong box (models.effective_cores)."""
    offers = [_reachable(min_bid=0.1, machine_id=1, cpu_cores=384, gpu_frac=0.25,
                         cpu_ram=131072.0, dlperf=88.5, compute_cap=890),
              _reachable(min_bid=0.2, machine_id=2)]          # neither field
    _m, orows = _probe_into_db(monkeypatch, db, offers)
    assert orows[0]["cpu_cores_effective"] == pytest.approx(96.0)
    assert orows[0]["cpu_ram_gb"] == pytest.approx(128.0)
    assert orows[0]["dlperf"] == pytest.approx(88.5)
    assert orows[0]["compute_cap"] == 890
    # unmeasured is UNKNOWN, not zero — the UI renders an em-dash for it
    assert orows[1]["cpu_cores_effective"] is None
    assert (orows[1]["cpu_ram_gb"], orows[1]["dlperf"],
            orows[1]["compute_cap"]) == (None, None, None)


def test_a_new_driver_on_turing_silicon_passes_parity_but_not_bf16(db, monkeypatch):
    """`cuda_max_good` measures the host DRIVER, not the card: a Turing part
    (sm_75) behind a current driver reports 13.1 and is perfectly launchable —
    it just cannot do bf16. That is why `compute_cap` is persisted too."""
    m, orows = _probe_into_db(monkeypatch, db, [
        _reachable(machine_id=1, cuda_max_good=13.1, compute_cap=750)])
    assert orows[0]["launch_ok"] == 1 and m["n_ok"] == 1
    assert orows[0]["compute_cap"] == 750 and orows[0]["compute_cap"] < 800


def test_market_offers_leak_no_host_identity(db, monkeypatch):
    """`market_offers` grew five columns off the same offer dict; the positive
    allowlist is what keeps the reachability fields out, and it has to be a
    structural claim rather than a habit."""
    stuffed = _reachable(machine_id=1, public_ipaddr="203.0.113.77",
                         hostname="node-leaked.provider.example", host_id=987654,
                         cpu_name="AMD EPYC 9654 96-Core", mobo_name="H12DSG-O-CPU")
    _m, _orows = _probe_into_db(monkeypatch, db, [stuffed])
    _path, conn = db
    blob = _flatten(conn, "market_offers")
    for leak in ("203.0.113.77", "node-leaked", "987654", "EPYC", "H12DSG"):
        assert leak not in blob, leak
    cols = {r[1] for r in conn.execute("PRAGMA table_info(market_offers)")}
    assert not (cols & {"public_ipaddr", "hostname", "host_id", "cpu_name",
                        "mobo_name"})


def test_market_insert_columns_match_the_ddl_exactly(db):
    """Both market INSERTs name their columns (no dict splat). DDL order, INSERT
    order and placeholder count have to agree or a value lands in the wrong
    column — silently, since every one of them is a nullable number."""
    _path, conn = db
    for table, stmt in (("market", dashcache._DASH_MARKET_INSERT),
                        ("market_offers", dashcache._DASH_OFFERS_INSERT)):
        ddl = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
        named = [c.strip() for c in
                 stmt.split("(", 1)[1].split(")", 1)[0].split(",")]
        assert named == ddl, table
        assert stmt.count("?") == len(ddl), table


def test_percentiles():
    assert dashcache._dash_pct([], 0.5) is None
    assert dashcache._dash_pct([2.0], 0.9) == 2.0
    assert dashcache._dash_pct([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert dashcache._dash_pct([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0
    assert dashcache._dash_pct([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)


# --------------------------------------------------------------------------- #
# 4. fleet: read ops only, and a down daemon means UNKNOWN (not a stale table)
# --------------------------------------------------------------------------- #
_FLEET_STATUS = {
    "version": 1, "rev": "b0e39335", "api_ok": True, "dry_run": False,
    "tick_age_s": 9.6, "spend_total_usd": 11.3768,
    "rows": [
        {"target": "20260731T2258", "iid": 46246859, "profile": "train",
         "state": "RUNNING", "spend_usd": 3.0, "budget_usd": 12.0,
         "paused": False, "dormant": False, "adopted": True,
         "last_action": "bid_raise", "requester": SECRET_MARKERS["requester"]},
        {"target": "46193810", "iid": 46193810, "profile": "-",
         "state": "UNWATCHED", "spend_usd": None, "budget_usd": None,
         "paused": False, "last_action": "parked"},
    ],
    "alarm_records": [
        {"key": "stray:46193810", "iid": 46193810, "sticky": False,
         "msg": "unwatched box billing (/home/leakeduser/state)",
         "since_ts": 1.0, "age_s": 300.0},
        {"key": "budget:x", "sticky": True, "msg": "budget 80% consumed",
         "since_ts": 2.0, "age_s": 90.0, "count": 3},
    ],
    "intents": [{"requester": SECRET_MARKERS["requester"]}],
    "destroys": [],
}


def _fleet_ok(op, **kw):
    assert op in ("status", "spend"), f"dash-cache used a non-read op: {op}"
    if op == "status":
        return True, _FLEET_STATUS, None
    return True, {"by_box": {"46246859": 3.0, "46193810": 8.3768}}, None


def test_fleet_projection(db, monkeypatch):
    _path, conn = db
    monkeypatch.setattr(fleet_client, "fleet_request", _fleet_ok)
    n, err = fleet_client._dash_write_fleet(conn, deps=_deps())
    assert (n, err) == (2, None)
    conn.row_factory = sqlite3.Row
    f = conn.execute("SELECT * FROM fleet").fetchone()
    assert (f["daemon_up"], f["api_ok"], f["dry_run"]) == (1, 1, 0)
    assert (f["n_watches"], f["n_strays"]) == (1, 1)
    assert (f["n_alarms"], f["n_sticky_alarms"]) == (2, 1)
    assert f["tick_stale"] == 0
    w = conn.execute("SELECT * FROM fleet_watches WHERE iid=46246859").fetchone()
    assert w["budget_frac"] == pytest.approx(0.25) and w["stray"] == 0
    s = conn.execute("SELECT * FROM fleet_watches WHERE iid=46193810").fetchone()
    assert s["stray"] == 1 and s["profile"] is None
    assert conn.execute("SELECT count(*) FROM fleet_spend").fetchone()[0] == 2
    blob = " ".join(_flatten(conn, t) for t in
                    ("fleet", "fleet_watches", "fleet_alarms", "fleet_spend"))
    assert SECRET_MARKERS["requester"] not in blob
    assert "/home/" not in blob


def test_stale_tick_is_flagged(db, monkeypatch):
    _path, conn = db
    stale = dict(_FLEET_STATUS, tick_age_s=dashcache.DASH_TICK_STALE_S + 1)
    monkeypatch.setattr(fleet_client, "fleet_request",
                        lambda op, **k: (True, stale, None) if op == "status"
                        else (True, {"by_box": {}}, None))
    fleet_client._dash_write_fleet(conn, deps=_deps())
    assert conn.execute("SELECT tick_stale FROM fleet").fetchone()[0] == 1


def test_down_daemon_clears_the_watch_table(db, monkeypatch):
    """A stale watch table would claim boxes are being babysat when nothing is —
    the most expensive lie this page could tell."""
    _path, conn = db
    monkeypatch.setattr(fleet_client, "fleet_request", _fleet_ok)
    fleet_client._dash_write_fleet(conn, deps=_deps())
    monkeypatch.setattr(fleet_client, "fleet_request",
                        lambda op, **k: (False, None, "nodaemon:FileNotFound"))
    n, err = fleet_client._dash_write_fleet(conn, deps=_deps())
    assert n == 0 and "nodaemon" in err
    row = conn.execute("SELECT daemon_up, api_ok, tick_age_s FROM fleet").fetchone()
    assert row == (0, None, None)
    for t in ("fleet_watches", "fleet_alarms", "fleet_spend"):
        assert conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] == 0
    # a down daemon is still a fresh MEASUREMENT — the stamp advances
    assert conn.execute("SELECT count(*) FROM meta WHERE key='fleet'"
                        ).fetchone()[0] == 1


def test_account_writes_only_credit_and_balance(db, monkeypatch):
    _path, conn = db
    monkeypatch.setattr(api, "request_soft", lambda *a, **k: (
        True, {"id": 4242, "email": "owner@example.com", "credit": 7.11,
               "balance": 0.0, "can_pay": True}, None))
    assert dashcache._dash_write_account(conn) == 1
    blob = _flatten(conn, "account")
    assert "owner@example.com" not in blob and "4242" not in blob
    assert conn.execute("SELECT credit, balance FROM account").fetchone() == (7.11, 0.0)


# --------------------------------------------------------------------------- #
# 5. the command contract: exit 0/1 only, empty stdout, rollback journal
# --------------------------------------------------------------------------- #
def _args(path, **kw):
    ns = argparse.Namespace(sections=None, gpus=None, num_gpus=None,
                            cache_db=path, no_spot=False)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_a_failing_section_is_a_skip_not_a_nonzero_exit(db, capsys, monkeypatch):
    """`reap --json`/`guard --json` exit 2 on findings. The Node caller treats
    ANY nonzero as total failure, so that convention must not leak here."""
    path, conn = db
    conn.execute("INSERT INTO account(key,credit,balance) VALUES('account',9.0,1.0)")
    conn.commit()

    def boom(*a, **k):
        raise RuntimeError("vast API down")

    monkeypatch.setattr(dashcache, "_dash_write_account", boom)
    monkeypatch.setattr(fleet_client, "_dash_write_fleet", lambda c: (0, None))
    cli_dash_cache.run(_args(path, sections="fleet,account"))   # no SystemExit
    out = capsys.readouterr()
    assert out.out == "", "stdout must stay empty (never proxy a payload)"
    assert "SKIPPED" in out.err and "vast API down" in out.err
    # the old row survives — a stale panel beats an empty one
    assert conn.execute("SELECT credit FROM account").fetchone()[0] == 9.0


def test_unknown_section_and_bad_num_gpus_exit_1(db):
    path, _conn = db
    for kw in ({"sections": "bogus"}, {"num_gpus": "1,zz"}):
        with pytest.raises(SystemExit) as e:
            cli_dash_cache.run(_args(path, **kw))
        assert e.value.code not in (0, 2)


def test_unopenable_db_exits_1(tmp_path):
    with pytest.raises(SystemExit) as e:
        cli_dash_cache.run(_args(str(tmp_path / "no" / "such" / "d.db")))
    assert e.value.code not in (0, 2)


def test_writing_never_flips_the_journal_to_wal(gathered, db, capsys, monkeypatch):
    """`sqlite3 -readonly` — how the dashboard reads this file — CANNOT open a
    WAL database. A stray journal_mode flip breaks every page silently."""
    path, conn = db
    monkeypatch.setattr(fleet_client, "_dash_write_fleet", lambda c: (0, None))
    cli_dash_cache.run(_args(path, sections="instances,fleet"))
    capsys.readouterr()
    ro = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        assert ro.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert ro.execute("SELECT count(*) FROM instances").fetchone()[0] == 2
        assert ro.execute("SELECT fetched_at FROM meta WHERE key='instances'"
                          ).fetchone()[0].endswith("Z")
    finally:
        ro.close()


def test_section_order_is_canonical_and_deduped():
    assert dashcache._dash_parse_sections("account,instances,account") == \
        ("instances", "account")
    assert dashcache._dash_parse_sections(None) == dashcache.DASH_SECTIONS


def _dash_cache_source_blocks():
    """Every source block the read-only posture must hold over.

    Class-C repoint (plan §7, migration batch B2), completed at 6d. The
    original sliced `herdd.py` between the `def _dash_scrub(` and
    `def cmd_runs(` markers; the code now lives in
    `vastlib/storage/dashcache.py`, where the WHOLE MODULE is the block — no
    slice to get wrong, and the `_infra_cache_*` prelude the old window started
    below is covered too. The flat slice was kept while both copies existed and
    dropped with the fat body at 6d: `herdd.py` is a launcher, both anchors
    are gone, and the `index()` calls that used to be this function's tripwire
    would now raise `ValueError` on every run.
    """
    return [("vastlib/storage/dashcache.py", open(dashcache.__file__).read())]


def test_no_mutating_verb_is_reachable_from_the_dash_cache_source():
    """A grep-level backstop for the read-only posture: the dash-cache block
    must not name a destroy/park/launch/bid helper or pass a `yes=`/`-y`."""
    for where, block in _dash_cache_source_blocks():
        # Comments and docstrings DISCUSS these verbs on purpose, and substring
        # matching would trip over read-only fields like `is_bid`. So: compare
        # whole NAME tokens of executable code against the banned callables.
        names = {t.string for t in
                 tokenize.generate_tokens(io.StringIO(block).readline)
                 if t.type == tokenize.NAME}
        banned = {"_destroy_and_revoke", "_do_launch", "_destroy", "_stop", "_start",
                  "_set_bid", "_label_set", "cmd_reap", "cmd_guard", "cmd_destroy",
                  "cmd_stop", "cmd_start", "cmd_bid", "cmd_launch", "cmd_label",
                  "subprocess", "_rclone", "_ssh", "fleet_journal_path"}
        assert not (names & banned), \
            f"dash-cache reaches: {sorted(names & banned)} in {where}"
        # no mutating HTTP method, and no journal flip the readonly reader can't survive
        for lit in ('"PUT"', '"DELETE"', '"PATCH"', "PRAGMA journal_mode"):
            assert lit not in block, f"dash-cache contains {lit} in {where}"
        assert not re.search(r"journal_mode\s*=", block), where


def test_a_stray_print_in_a_section_cannot_reach_stdout(db, capsys, monkeypatch):
    """stdout is the channel an execFile caller reads. A `print()` added later
    to any transitive callee (jobs fold, market probe, health gather) must not
    be able to put a payload on it."""
    path, _conn = db

    def chatty(conn):
        print("ninja: [12/40] compiling  B2_APPLICATION_KEY=leak")
        return 0, None

    monkeypatch.setattr(fleet_client, "_dash_write_fleet", chatty)
    cli_dash_cache.run(_args(path, sections="fleet"))
    out = capsys.readouterr()
    assert out.out == ""
    assert "ninja:" in out.err
