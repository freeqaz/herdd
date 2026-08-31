"""test_hosts.py — gather_hosts event-join unit tests (no B2, no network).

Covers the boot-health Track-A additions to hosts.py:
  * the train lane's pull_throughput heartbeat join (unchanged baseline);
  * the jobs/eval lane's asset_throughput join, bridged run->IID via the
    launched event's instance_id (jobd boxes finally contribute host history);
  * the boot_killed_slow runmeta flag (SLOW/KILLED scorecard annotation +
    exclude-seed) — including a condemned machine that has no launched row.

A FakeB2 stands in for boxstate.B2, mirroring its (present, names) / (present,
text) contracts so the real gather_hosts code path runs verbatim.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hosts  # noqa: E402
import herdd  # noqa: E402


class FakeB2:
    """Dict-backed B2: keys are full object paths, values are JSON bodies.
    lsf/cat mirror boxstate.B2 (basenames; trailing-slash prefixes list one
    level; a bare object path lists just that object)."""

    def __init__(self, objects):
        self.objects = dict(objects)
        self.ok = True
        self.reason = None

    def lsf(self, path):
        if path.endswith("/"):
            names = set()
            for k in self.objects:
                if k.startswith(path):
                    rest = k[len(path):]
                    if not rest:
                        continue
                    if "/" in rest:
                        names.add(rest.split("/", 1)[0] + "/")
                    else:
                        names.add(rest)
            return bool(names), sorted(names)
        present = path in self.objects
        return present, ([path.rsplit("/", 1)[-1]] if present else [])

    def cat(self, path):
        if path in self.objects:
            return True, self.objects[path]
        return False, None


def _ev(**kw):
    return json.dumps(kw)


def _bucket():
    objects = {}
    # --- train run: launched + a pull_throughput heartbeat -------------------
    objects["runs/runT/events/20260720T000000000Z-cli_a-n1.json"] = _ev(
        event="launched", machine_id=111, instance_id=5001, gpu="RTX 4090",
        geolocation="US-West", inet_down=900, ts="20260720T000000000Z")
    objects["runs/runT/events/20260720T000030000Z-box_5001-n2.json"] = _ev(
        event="heartbeat", phase="pull_throughput", mbps=42,
        instance_id=5001, ts="20260720T000030000Z")
    # --- jobs/eval run: launched, NO heartbeat, asset_throughput on the box ---
    objects["runs/runJ/events/20260720T010000000Z-cli_a-n1.json"] = _ev(
        event="launched", machine_id=222, instance_id=5002, gpu="RTX 3090",
        geolocation="US-East", inet_down=500, ts="20260720T010000000Z")
    # a lifecycle event (must be ignored) + two asset_throughput (max wins)
    objects["jobs/nodes/5002/events/20260720T010100000Z-box_5002-n1.json"] = _ev(
        event="claimed", ts="20260720T010100000Z")
    objects["jobs/nodes/5002/events/20260720T010200000Z-box_5002-n2.json"] = _ev(
        event="asset_throughput", asset="base", bytes=700000000, secs=100,
        mbps=7, ts="20260720T010200000Z")
    objects["jobs/nodes/5002/events/20260720T010500000Z-box_5002-n3.json"] = _ev(
        event="asset_throughput", asset="ckpt", bytes=13000000000, secs=1000,
        mbps=13, ts="20260720T010500000Z")
    # --- killed run: launched on 333, but the watcher condemned 999 ----------
    objects["runs/runK/events/20260720T020000000Z-cli_a-n1.json"] = _ev(
        event="launched", machine_id=333, instance_id=5003, gpu="RTX 4090",
        ts="20260720T020000000Z")
    objects["runs/runK/events/20260720T015900000Z-cli_a-nk.json"] = _ev(
        event="boot_killed_slow", machine_id=999, mbps=2, window_s=300,
        phase="P0", ts="20260720T015900000Z")
    return FakeB2(objects)


def test_train_lane_pull_throughput_join():
    hostmap, meta = hosts.gather_hosts(_bucket())
    assert hostmap["111"]["med_mbps"] == 42
    assert hostmap["111"]["killed"] is False
    assert meta["runs_with_machine"] == 3


def test_jobs_lane_asset_throughput_join_takes_max():
    hostmap, _ = hosts.gather_hosts(_bucket())
    # bridged run->IID (5002) -> jobs/nodes/5002/events asset_throughput; MAX(7,13)
    assert hostmap["222"]["med_mbps"] == 13.0


def test_scan_asset_throughput_recovers_bytes_alongside_mbps():
    """The byte totals jobd has always emitted are the disk-sizing calibration
    signal; the host-quality fold used to read past them. One scan, both."""
    mbps, by_asset = hosts.scan_asset_throughput(_bucket(), "5002")
    assert mbps == 13.0                       # host-quality signal unchanged
    assert by_asset == {"base": 700000000, "ckpt": 13000000000}


def test_scan_asset_throughput_maxes_repulls_never_sums():
    """A park/resume re-pull re-emits the same asset name. The on-box cache
    dedupes it, so summing would over-report the box's real footprint and
    over-size every box the estimator later provisions."""
    objs = dict(_bucket().objects)
    objs["jobs/nodes/5002/events/20260720T011000000Z-box_5002-n4.json"] = _ev(
        event="asset_throughput", asset="base", bytes=700000000, secs=90,
        mbps=8, ts="20260720T011000000Z")
    _, by_asset = hosts.scan_asset_throughput(FakeB2(objs), "5002")
    assert by_asset["base"] == 700000000       # not 1.4e9


def test_scan_asset_throughput_tolerates_string_bytes_and_missing_fields():
    """jobd ships fields as shell k=v strings (`emit_box` -> `--field`), so
    `bytes` arrives as a JSON string; and an event may carry mbps without bytes
    or vice versa. Neither may raise."""
    objs = {
        "jobs/nodes/7/events/a.json": _ev(
            event="asset_throughput", asset="m", bytes="4096", mbps="5"),
        "jobs/nodes/7/events/b.json": _ev(
            event="asset_throughput", asset="nobytes", mbps=99),
        "jobs/nodes/7/events/c.json": _ev(
            event="asset_throughput", asset="nombps", bytes=17),
        "jobs/nodes/7/events/d.json": _ev(
            event="asset_throughput", asset="junk", bytes="not-a-number"),
    }
    mbps, by_asset = hosts.scan_asset_throughput(FakeB2(objs), "7")
    assert mbps == 99.0
    assert by_asset == {"m": 4096, "nombps": 17}


def test_scan_asset_throughput_absent_stream_is_not_an_error():
    assert hosts.scan_asset_throughput(FakeB2({}), "9999") == (None, {})
    assert hosts.scan_asset_throughput(FakeB2({}), "") == (None, {})


def test_boot_killed_slow_flags_condemned_machine():
    hostmap, meta = hosts.gather_hosts(_bucket())
    assert meta["killed_machines"] == 1
    # 999 was condemned but never launched-recorded -> stub SLOW/KILLED row
    assert "999" in hostmap
    k = hostmap["999"]
    assert k["killed"] is True
    assert k["kill_mbps"] == 2 and k["kill_window_s"] == 300 and k["kill_phase"] == "P0"
    assert k["n_runs"] == 0
    # 333 (the run's actual host) is NOT flagged
    assert hostmap["333"]["killed"] is False


def test_scorecard_renders_killed_marker():
    hostmap, meta = hosts.gather_hosts(_bucket())
    text = hosts.scorecard(hostmap, meta)
    assert "SLOW/KILLED" in text
    assert "999" in text
    # the killed host sinks below a healthy measured host in the table
    assert text.index("111") < text.index("999")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)


# --------------------------------------------------------------------------- #
# _offer_query — the --search alias expansion (regression: "no rentable
# verified offers" for EVERY gpu class while `herdd search` saw a full
# market). The raw alias was sent as gpu_name, which the vast API matches
# exactly, so nothing ever came back. No network: assert on the query body.
# --------------------------------------------------------------------------- #
def test_offer_query_expands_single_name_alias():
    q = hosts._offer_query("5090", 0.5)
    assert q["gpu_name"] == {"eq": "RTX 5090"}, q["gpu_name"]


def test_offer_query_uses_in_for_multi_sku_alias():
    # rtxpro6000 covers BOTH the WS and the S SKU; `eq` cannot express that.
    q = hosts._offer_query("rtxpro6000", 2.0)
    assert q["gpu_name"] == {"in": ["RTX PRO 6000 WS", "RTX PRO 6000 S"]}, q["gpu_name"]


def test_offer_query_never_sends_a_bare_alias():
    # the actual bug: the alias itself must never reach the wire.
    for alias in ("5090", "4090", "h100", "a100", "rtxpro6000", "b200"):
        q = hosts._offer_query(alias, 1.0)
        sent = q["gpu_name"].get("eq") or q["gpu_name"].get("in")
        sent = [sent] if isinstance(sent, str) else sent
        assert alias not in sent, f"raw alias {alias!r} leaked onto the wire: {sent}"


def test_offer_query_passes_through_an_exact_api_name():
    # a caller who already knows vast's spelling must not be mangled.
    q = hosts._offer_query("RTX 5090", 0.5)
    assert q["gpu_name"] == {"eq": "RTX 5090"}


def test_offer_query_omits_gpu_name_when_no_gpu_given():
    q = hosts._offer_query(None, 0.5)
    assert "gpu_name" not in q
    assert q["dph_total"] == {"lte": 0.5}


def test_offer_query_keeps_the_filters_that_were_never_broken():
    q = hosts._offer_query("5090", 0.75)
    assert q["verified"] == {"eq": True}
    assert q["rentable"] == {"eq": True}
    assert q["type"] == "ask"
    assert q["dph_total"] == {"lte": 0.75}


# --------------------------------------------------------------------------- #
# effective cores — the CPU slice an OFFER rents, not the host's advertised
# total. `cpu_cores` is whole-machine (recent 5090 hosts advertise 128/384);
# multiplied by gpu_frac it resolves to 16-55, which is what the CPU-bound
# scoring lane actually gets. No network: assert on offer dicts.
# --------------------------------------------------------------------------- #
def test_effective_cores_prefers_vasts_own_field():
    # 256 x 0.125 = 32.0, and vast publishes it directly.
    o = {"cpu_cores": 256, "gpu_frac": 0.125, "cpu_cores_effective": 32.0}
    assert herdd.effective_cores(o) == 32.0


def test_effective_cores_falls_back_to_the_multiplication():
    assert herdd.effective_cores({"cpu_cores": 384, "gpu_frac": 0.25}) == 96.0


def test_effective_cores_is_none_when_the_offer_cannot_say():
    for o in ({}, {"cpu_cores": 384}, {"gpu_frac": 0.25},
              {"cpu_cores": 0, "gpu_frac": 0.5},
              {"cpu_cores": 128, "gpu_frac": 0}):
        assert herdd.effective_cores(o) is None, o


def test_advertised_cores_are_never_mistaken_for_effective_ones():
    """The whole point: a 384-core host offering 1 of 8 GPUs is a 48-core box.
    Reading cpu_cores straight is how the eval lane picked the wrong hosts."""
    o = {"cpu_cores": 384, "gpu_frac": 0.125}
    assert herdd.effective_cores(o) == 48.0
    assert herdd.effective_cores(o) != o["cpu_cores"]


def test_min_effective_cores_is_a_no_op_unless_asked():
    """Strictly opt-in — no default search path may filter on this."""
    offers = [{"id": 1, "cpu_cores": 128, "gpu_frac": 0.125},   # 16 eff
              {"id": 2}]                                        # unknown
    for minimum in (None, 0):
        kept, dropped = hosts.filter_min_effective_cores(offers, minimum)
        assert [o["id"] for o in kept] == [1, 2] and dropped == 0


def test_min_effective_cores_drops_small_slices_and_unknowns():
    offers = [{"id": 1, "cpu_cores": 128, "gpu_frac": 0.125},   # 16 eff
              {"id": 2, "cpu_cores": 384, "gpu_frac": 0.25},    # 96 eff
              {"id": 3, "cpu_cores_effective": 56.0},           # exactly at it
              {"id": 4}]                                        # cannot say
    kept, dropped = hosts.filter_min_effective_cores(offers, 56)
    assert [o["id"] for o in kept] == [2, 3]
    assert dropped == 2
