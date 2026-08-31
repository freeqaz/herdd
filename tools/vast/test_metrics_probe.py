"""Portable tests for metrics_probe.py — pure parsers + verdict, no hardware.

Runs in the toolchain-free lane (`pytest -m "not integration"`): no nvidia-smi,
no /proc reads, no network. Every parser is fed fixture text; the sampler that
touches the real host is exercised only through the injectable-argv path.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics_probe as mp  # noqa: E402


# --- nvidia-smi --query-gpu parsing -----------------------------------------
NVSMI = (
    "0, NVIDIA GeForce RTX 3090, 95, 77, 22736, 24576, 379.25, 390.00, 83, 1890, P2, 0x0000000000000020\n"
    "1, NVIDIA GeForce RTX 3090, 12, 4, 512, 24576, 90.00, 390.00, 45, 210, P8, 0x0000000000000004\n"
)


def test_parse_nvidia_smi_two_cards():
    gpus = mp.parse_nvidia_smi(NVSMI)
    assert len(gpus) == 2
    g0 = gpus[0]
    assert g0["idx"] == 0 and g0["util"] == 95 and g0["mem_util"] == 77
    assert g0["mem_used_mb"] == 22736 and g0["mem_total_mb"] == 24576
    assert g0["power_w"] == 379.25 and g0["power_limit_w"] == 390.0
    assert g0["temp_c"] == 83 and g0["sm_clock_mhz"] == 1890 and g0["pstate"] == "P2"
    assert g0["throttle"] == ["sw_thermal"]
    assert gpus[1]["throttle"] == ["sw_power"]


def test_parse_nvidia_smi_na_and_blank():
    gpus = mp.parse_nvidia_smi(
        "0, Fake, [N/A], [N/A], [N/A], 16384, [N/A], [N/A], [N/A], [N/A], [N/A], 0x0\n"
        "\n"           # blank line skipped
        "short, row\n"  # too few columns skipped
    )
    assert len(gpus) == 1
    g = gpus[0]
    assert g["util"] is None and g["power_w"] is None and g["temp_c"] is None
    assert g["mem_total_mb"] == 16384
    assert g["throttle"] == ["none"]


def test_decode_throttle():
    assert mp.decode_throttle("0x0000000000000000") == ["none"]
    assert mp.decode_throttle("0x0000000000000001") == ["none"]        # gpu_idle dropped
    assert mp.decode_throttle("0x0000000000000004") == ["sw_power"]
    assert mp.decode_throttle("0x0000000000000020") == ["sw_thermal"]
    # multiple bits: sw_power (0x4) + sw_thermal (0x20)
    assert mp.decode_throttle("0x0000000000000024") == ["sw_power", "sw_thermal"]
    assert mp.decode_throttle("garbage") == []


def test_parse_dmon_t():
    txt = ("# gpu  rxpci  txpci \n"
           "# Idx   MB/s   MB/s \n"
           "    0     51     14 \n"
           "    1     53     14 \n")
    assert mp.parse_dmon_t(txt) == {0: (51, 14), 1: (53, 14)}


# --- cpu / mem --------------------------------------------------------------
def test_cpu_busy_pct():
    # before: busy=100 total=200 ; after: busy=150 total=300
    # -> dbusy=50 dtotal=100 -> 50%
    before = (100, 200)
    after = (150, 300)
    assert mp.cpu_busy_pct(before, after) == 50.0
    assert mp.cpu_busy_pct(None, after) is None
    assert mp.cpu_busy_pct(before, before) is None  # dtotal==0


def test_parse_proc_stat():
    txt = "cpu  100 0 100 700 100 0 0 0 0 0\ncpu0 1 2 3 4\n"
    # busy = total - idle - iowait = 1000 - 700 - 100 = 200 ; total = 1000
    assert mp.parse_proc_stat(txt) == (200, 1000)


def test_mem_used_pct():
    txt = "MemTotal:       1000 kB\nMemAvailable:    250 kB\nMemFree: 100 kB\n"
    pct, used_mb, total_mb = mp.mem_used_pct(mp.parse_meminfo(txt))
    assert pct == 75.0                       # (1000-250)/1000
    assert total_mb == 0                     # 1000 kB // 1024 == 0 MB
    # MemAvailable absent -> falls back to MemFree
    pct2, _, _ = mp.mem_used_pct({"MemTotal": 1000, "MemFree": 400})
    assert pct2 == 60.0


# --- net / disk deltas ------------------------------------------------------
NETDEV = (
    "Inter-|   Receive                    |  Transmit\n"
    " face |bytes    packets errs drop fifo frame compressed multicast|"
    "bytes    packets errs drop fifo colls carrier compressed\n"
    "    lo: 999 0 0 0 0 0 0 0 999 0 0 0 0 0 0 0\n"
    "  eth0: 1000000 0 0 0 0 0 0 0 2000000 0 0 0 0 0 0 0\n"
)


def test_parse_net_dev_excludes_lo():
    d = mp.parse_net_dev(NETDEV)
    assert "lo" not in d
    assert d["eth0"] == (1000000, 2000000)


def test_net_rates():
    before = {"eth0": (0, 0)}
    after = {"eth0": (5_000_000, 1_000_000)}  # 5 MB rx, 1 MB tx over 1s
    trx, ttx, per = mp.net_rates(before, after, 1.0)
    assert trx == 5.0 and ttx == 1.0
    assert per["eth0"] == (5.0, 1.0)
    # counter reset / first-seen iface -> no negative rate
    trx2, _, _ = mp.net_rates({}, {"eth0": (5_000_000, 0)}, 1.0)
    assert trx2 == 0.0
    assert mp.net_rates(before, after, 0)[0] == 0.0  # dt<=0 guard


DISKSTATS = (
    "   8       0 sda 10 0 2000 5 20 0 4000 8 0 0 0\n"
    "   8       1 sda1 5 0 1000 2 10 0 2000 4 0 0 0\n"   # partition -> ignored
    " 259       0 nvme0n1 1 0 1000 0 1 0 1000 0 0 0 0\n"
)


def test_parse_diskstats_whole_disks_only():
    d = mp.parse_diskstats(DISKSTATS)
    assert set(d) == {"sda", "nvme0n1"}      # sda1 partition excluded
    assert d["sda"] == (2000, 4000)          # sectors r / w


def test_disk_rates():
    before = {"sda": (0, 0)}
    after = {"sda": (2000, 4000)}            # 2000*512=1.024MB r, 4000*512=2.048MB w over 1s
    rd, wr = mp.disk_rates(before, after, 1.0)
    assert rd == 1.02 and wr == 2.05


# --- verdict ----------------------------------------------------------------
def _snap(**over):
    s = {
        "gpus": [{"util": 90, "throttle": ["none"]}],
        "gpu_util_avg": 90.0,
        "cpu": {"load_per_core": 0.2},
        "net": {"rx_mbps": 1, "tx_mbps": 1},
        "disk": {"read_mbps": 1, "write_mbps": 1},
        "errors": [],
    }
    s.update(over)
    return s


def test_verdict_gpu_bound():
    assert "GPU-bound" in mp.verdict(_snap())


def test_verdict_thermal_throttle_flagged_even_when_busy():
    s = _snap(gpus=[{"util": 90, "throttle": ["sw_thermal"]}])
    v = mp.verdict(s)
    assert "THROTTLING" in v and "sw_thermal" in v


def test_verdict_power_cap_is_expected_not_alarming():
    s = _snap(gpus=[{"util": 92, "throttle": ["sw_power"]}], gpu_util_avg=92.0)
    v = mp.verdict(s)
    assert "power-capped" in v and "THROTTLING" not in v


def test_verdict_network_bound():
    s = _snap(gpus=[{"util": 30, "throttle": ["none"]}], gpu_util_avg=30.0,
              net={"rx_mbps": 400, "tx_mbps": 10})
    v = mp.verdict(s)
    assert "under-utilized" in v and "network" in v


def test_verdict_cpu_contended():
    s = _snap(gpus=[{"util": 40, "throttle": ["none"]}], gpu_util_avg=40.0,
              cpu={"load_per_core": 2.5}, net={"rx_mbps": 1, "tx_mbps": 1})
    v = mp.verdict(s)
    assert "CPU-contended" in v


def test_verdict_idle_and_no_gpu():
    assert "idle" in mp.verdict(_snap(gpus=[{"util": 1, "throttle": ["none"]}],
                                      gpu_util_avg=1.0,
                                      net={"rx_mbps": 0, "tx_mbps": 0}))
    assert "no GPU" in mp.verdict(_snap(gpus=[], gpu_util_avg=None,
                                        errors=["nvidia-smi: not found"]))


# --- render_fields (heartbeat splice-safety) --------------------------------
def test_render_fields_is_splice_safe():
    snap = _snap(gpus=[{"util": 90, "mem_util": 70, "power_w": 380,
                        "power_limit_w": 390, "temp_c": 83, "throttle": ["sw_thermal"]}],
                 gpu_util_avg=90.0, cpu={"busy_pct": 12.8, "load_per_core": 0.15},
                 net={"rx_mbps": 0.01, "tx_mbps": 0.09})
    line = mp.render_fields(snap)
    # must be safe as a `--field host_metrics=<line>` value: no '=' and no space
    assert "=" not in line and " " not in line
    assert "gpu_util:90" in line
    assert "thr:sw_thermal" in line


# --- absolute power cap + device name (host-degradation attribution) ---------
# Box 46936034 ran a whole 27B training window at 2.13x the step time of its
# replacement while every heartbeat looked normal: gpu_pwr is a PERCENTAGE of
# the limit, so a lowered cap at 100% reads exactly like a healthy card at 100%.
# The probe sampled power.limit and threw it away in render_fields.
# (docs/plans/witness/perf/PERF_LEVERS_INVESTIGATION_2026-08-06.md §2.4/§2.5.)
def test_render_fields_carries_absolute_power_limit_and_name():
    # Shape taken from a LIVE read of box 46947265 (the healthy replacement),
    # 2026-08-06: 600 W cap, 532.76 W draw, Server Edition.
    line = mp.render_fields(_snap(
        gpus=[{"idx": 0, "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
               "util": 100, "mem_util": 64, "power_w": 532.76,
               "power_limit_w": 600.0, "temp_c": 71, "throttle": ["none"]}],
        gpu_util_avg=100.0))
    assert "gpu_plim:600" in line
    assert "gpu:RTX_PRO_6000_Blackwell_Server_Edition" in line
    assert "gpu_pwr:89" in line          # the percentage still rides along
    assert "=" not in line and " " not in line


def test_render_fields_name_and_limit_are_optional():
    # A card that reports neither must not emit an empty or bogus field — the
    # heartbeat parser splits on ':' and a bare `gpu:` would read as a null SKU.
    line = mp.render_fields(_snap(gpus=[{"util": 50, "throttle": ["none"]}],
                                  gpu_util_avg=50.0))
    assert "gpu_plim" not in line
    assert "gpu:" not in line            # `gpu_util:` must not trip this
    assert "gpu_util:50" in line


def test_min_power_limit_reports_the_binding_cap_and_flags_disagreement():
    # MIN, not mean: the slowest card paces a DDP step. When caps disagree the
    # value says so rather than averaging the anomaly away.
    assert mp._min_power_limit([{"power_limit_w": 600.0}]) == 600
    assert mp._min_power_limit([{"power_limit_w": 600.0},
                                {"power_limit_w": 450.0}]) == "450-600"
    assert mp._min_power_limit([{"util": 9}]) is None
    assert mp._min_power_limit([]) is None


def test_gpu_name_tag_is_field_safe_and_names_a_mixed_box():
    tag = mp._gpu_name_tag([{"name": "NVIDIA GeForce RTX 5090"}])
    assert tag == "GeForce_RTX_5090"     # "NVIDIA " distinguishes nothing
    mixed = mp._gpu_name_tag([{"name": "NVIDIA H100 PCIe"},
                              {"name": "NVIDIA GeForce RTX 5090"}])
    assert mixed.startswith("MIXED[") and "H100_PCIe" in mixed
    for bad in ("=", " ", ",", ":"):
        assert bad not in mixed
    assert mp._gpu_name_tag([{"name": "  "}, {}]) is None


# --- idle-card throttle artifact (Blackwell / newer drivers) -----------------
# MEASURED 2026-07-30 (BOX_SATURATION_AUDIT §1.1, §6 red flag 4): nvidia-smi
# reported 0x8C (sw_power_cap|hw_slowdown|hw_power_brake) on all four cards of a
# healthy box, including one at 8 W of a 300 W limit and 0% util for the whole
# run — so every jobd heartbeat carried `thr:hw_power_brake|hw_slowdown`.
_IDLE_CARD = {"util": 0, "power_w": 8, "power_limit_w": 300,
              "throttle": ["sw_power", "hw_slowdown", "hw_power_brake"]}
_LOADED_THROTTLED = {"util": 99, "power_w": 295, "power_limit_w": 300,
                     "throttle": ["hw_slowdown"]}


def test_card_is_idle_needs_both_readings():
    assert mp.card_is_idle(_IDLE_CARD) is True
    assert mp.card_is_idle(_LOADED_THROTTLED) is False
    # a card at 0% but drawing 60% of its limit is NOT provably idle
    assert mp.card_is_idle({"util": 0, "power_w": 180,
                            "power_limit_w": 300}) is False
    # unreadable fields => cannot prove idle => keep the raw alarm
    assert mp.card_is_idle({"util": None, "power_w": 8,
                            "power_limit_w": 300}) is False
    assert mp.card_is_idle({"util": 0, "power_w": None,
                            "power_limit_w": 300}) is False
    assert mp.card_is_idle({"util": 0, "power_w": 8,
                            "power_limit_w": None}) is False


def test_idle_card_hw_bits_are_labeled_not_alarmed():
    real, artifact = mp.split_concerning([_IDLE_CARD])
    assert real == []
    assert artifact == ["hw_power_brake", "hw_slowdown"]
    line = mp.render_fields(_snap(gpus=[_IDLE_CARD], gpu_util_avg=0.0))
    assert "thr:idle-card-artifact(hw_power_brake|hw_slowdown)" in line
    assert "=" not in line and " " not in line          # still splice-safe
    # and the verdict of an idle box must not read as throttled
    v = mp.verdict(_snap(gpus=[_IDLE_CARD], gpu_util_avg=0.0))
    assert "hw_power_brake" not in v and "THROTTLING" not in v


def test_loaded_card_still_alarms_and_outranks_an_idle_sibling():
    real, artifact = mp.split_concerning([_LOADED_THROTTLED])
    assert real == ["hw_slowdown"] and artifact == []
    v = mp.verdict(_snap(gpus=[_LOADED_THROTTLED], gpu_util_avg=99.0))
    assert "THROTTLING" in v and "hw_slowdown" in v
    # mixed box: the real hw_slowdown is NOT excused by the idle card's copy of
    # the same bit, and is not double-reported as an artifact
    real, artifact = mp.split_concerning([_IDLE_CARD, _LOADED_THROTTLED])
    assert real == ["hw_slowdown"] and artifact == ["hw_power_brake"]
    assert mp.concerning_labels([_IDLE_CARD, _LOADED_THROTTLED]) == \
        ["hw_slowdown", "idle-card-artifact(hw_power_brake)"]


def test_idle_card_thermal_bits_are_never_dismissed():
    """A thermal bit on an idle card is not the measured artifact — it could be
    an inlet/VRM problem, so it keeps its name."""
    card = dict(_IDLE_CARD, throttle=["hw_thermal", "hw_slowdown"])
    real, artifact = mp.split_concerning([card])
    assert real == ["hw_thermal"] and artifact == ["hw_slowdown"]


def test_render_table_marks_the_idle_artifact():
    snap = _snap(gpus=[dict(_IDLE_CARD, idx=3, name="RTX PRO 6000")],
                 gpu_util_avg=0.0, host="h", ts="t", window_s=1.0,
                 cpu={"busy_pct": 1.0, "load_per_core": 0.1},
                 net={"rx_mbps": 0, "tx_mbps": 0, "per_iface": {}},
                 disk={"read_mbps": 0, "write_mbps": 0})
    out = mp.render_table(snap)
    text = out if isinstance(out, str) else "\n".join(out)
    assert "idle-artifact:hw_power_brake" in text
    assert "idle-artifact:hw_slowdown" in text


# --- injectable nvidia-smi (end-to-end read_gpus, no real GPU) ---------------
def test_read_gpus_missing_binary(monkeypatch):
    monkeypatch.setenv("METRICS_NVIDIA_SMI", "/bin/false")
    gpus, err = mp.read_gpus()
    assert gpus == []
    assert err and "rc=" in err


# --------------------------------------------------------------------------- #
# util says "busy", power says "how much silicon"
#
# The two cells below are REAL measurements from 2026-08-06/07, on the same card
# class, and utilization cannot tell them apart -- which is the entire reason
# verdict() consults power. Before this, both read "saturated".
# --------------------------------------------------------------------------- #
def _card(draw, limit, throttle=("none",)):
    return {"util": 100, "power_w": draw, "power_limit_w": limit,
            "throttle": list(throttle)}


def test_verdict_high_util_low_power_is_not_saturated():
    """v9 gemma-4 12B, measured: util 100%, 403 W of a 600 W limit, roof-HFU ~30-38%."""
    s = _snap(gpus=[_card(403.0, 600.0)], gpu_util_avg=100.0)
    v = mp.verdict(s)
    assert "NOT saturated" in v
    assert "67%" in v          # the number an operator needs to see
    assert "narrow/memory-bound" in v


def test_verdict_high_util_high_power_is_saturated():
    """v7 7B dec, measured: util 100%, ~87% of a 600 W limit, roof-HFU ~41-51%."""
    s = _snap(gpus=[_card(522.0, 600.0)], gpu_util_avg=100.0)
    v = mp.verdict(s)
    assert "saturated" in v and "NOT saturated" not in v
    assert "87%" in v


def test_verdict_missing_power_is_unverified_never_saturated():
    """Absence of a power reading is 'cannot tell', NOT 'low' and NOT 'fine'.

    A driver hiding power.draw must not be silently promoted to either verdict;
    the whole defect being fixed here is a confident claim made without the
    evidence for it.
    """
    s = _snap(gpus=[{"util": 100, "throttle": ["none"]}], gpu_util_avg=100.0)
    v = mp.verdict(s)
    assert "UNVERIFIED" in v
    assert "NOT saturated" not in v


def test_power_frac_ignores_cards_that_cannot_report():
    """One reporting card + one blind card averages over the reporting one only."""
    assert mp.power_frac([_card(300.0, 600.0), {"throttle": ["none"]}]) == 0.5
    assert mp.power_frac([{"throttle": ["none"]}]) is None
    assert mp.power_frac([]) is None
    # a zero/absent limit must not divide-by-zero or count
    assert mp.power_frac([_card(300.0, 0)]) is None


def test_verdict_throttle_still_outranks_the_power_reading():
    """A real throttle is the headline even when power looks low -- the operator
    action differs (change host vs change kernel)."""
    s = _snap(gpus=[_card(300.0, 600.0, ("hw_thermal",))], gpu_util_avg=100.0)
    v = mp.verdict(s)
    assert "THROTTLING" in v and "hw_thermal" in v
