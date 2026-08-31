"""Portable tests for the boot-throughput health-check (BOOT_HEALTHCHECK_DESIGN.md,
phase P0): herdd's pure `parse_pull_progress`, the `BootThroughputSampler`
verdict math, the `boot_health_watch` poll loop, and `build_throughput_observer`.

Toolchain-free lane (`pytest -m "not integration"`): NO vast API, NO B2/rclone,
NO network. The parser runs on synthesized docker-pull `status_msg` text (the
real captured fixture's status_msg was apt/provision output — no Downloading
layer lines — so it's used only as a "provision phase = clock not started"
schema-shape case, reproduced here without its live secrets). The watcher runs
against injected get_instance/now/sleep fakes.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import herdd as v  # noqa: E402


# --- fixtures: synthesized docker-pull status_msg snapshots ------------------ #
def _dl(layer, cur, cur_u, tot, tot_u, bar="=====>    "):
    return f"{layer}: Downloading [{bar}]  {cur}{cur_u}/{tot}{tot_u}"


PULL_SNAP_1 = "\n".join([
    "a1b2c3d4e5f6: Already exists",
    "b2c3d4e5f6a7: Already exists",
    _dl("c3d4e5f6a7b8", "123.4", "MB", "2.345", "GB"),
    _dl("d4e5f6a7b8c9", "10.5", "MB", "512.0", "MB"),
    "e5f6a7b8c9d0: Waiting",
])

# a LATER snapshot: the first downloading layer completed + scrolled to a tail
# window that no longer shows the 'Already exists' layers (vast truncation),
# and a new layer is now extracting.
PULL_SNAP_2 = "\n".join([
    "c3d4e5f6a7b8: Pull complete",
    "d4e5f6a7b8c9: Downloading [========>  ]  256.0MB/512.0MB",
    "f6a7b8c9d0e1: Extracting [==>        ]  1.0GB/2.345GB",
])

# A real vast record shape during the apt/provision phase (schema mirrors the
# captured fixture 45412361 — actual_status 'loading', status_msg is apt output
# with NO docker layer lines; secrets stripped/omitted).
PROVISION_RECORD = {
    "id": 45412361, "actual_status": "loading", "cur_state": "running",
    "machine_id": 140087, "public_ipaddr": "192.0.2.233",
    "status_msg": ("#6 2.230 Get:5 http://security.ubuntu.com/ubuntu "
                   "noble-security/universe i386 Packages [1006 kB]"),
}
GONE_RECORD = None  # mirrors fixture 45375039: {"instances": null} -> gone


# --- parse_pull_progress ------------------------------------------------------ #
def test_parse_basic_downloading_and_already_exists():
    p = v.parse_pull_progress(PULL_SNAP_1, {})
    assert p["downloading"] is True
    assert p["extracting"] is False
    # Already-exists layers contribute 0 (cached, free), downloading layers
    # contribute cur-bytes (decimal SI: 123.4MB = 123.4e6).
    assert p["layers"]["a1b2c3d4e5f6"] == 0
    assert p["layers"]["c3d4e5f6a7b8"] == 123_400_000
    assert p["layers"]["d4e5f6a7b8c9"] == 10_500_000
    assert p["total_bytes"] == 123_400_000 + 10_500_000


def test_parse_provision_phase_has_no_download_evidence():
    """apt/provision status_msg (no docker layer lines) folds to zero bytes and
    downloading=False -> the throughput clock never starts on this snapshot."""
    p = v.parse_pull_progress(PROVISION_RECORD["status_msg"], {})
    assert p["downloading"] is False and p["total_bytes"] == 0 and p["layers"] == {}


def test_parse_monotonic_across_tail_truncation():
    """Per-layer high-water marks carry forward: a completed layer that scrolls
    out of the truncated tail window must NOT drop total_bytes (no phantom
    negative throughput). The new downloading + extracting bytes ADD on top."""
    p1 = v.parse_pull_progress(PULL_SNAP_1, {})
    p2 = v.parse_pull_progress(PULL_SNAP_2, p1)
    # c3.. (123.4MB while downloading) is now 'Pull complete' and only visible in
    # snap2 as a bare complete -> stays at its high-water mark, never lost.
    assert p2["layers"]["c3d4e5f6a7b8"] == 123_400_000
    # d4.. advanced 10.5MB -> 256MB (monotonic max picks the larger)
    assert p2["layers"]["d4e5f6a7b8c9"] == 256_000_000
    # a new extracting layer freezes at its total
    assert p2["layers"]["f6a7b8c9d0e1"] == 2_345_000_000
    assert p2["extracting"] is True and p2["downloading"] is True
    # total never decreases vs the prior snapshot
    assert p2["total_bytes"] >= p1["total_bytes"]


def test_parse_never_decreases_total_on_shrinking_snapshot():
    p1 = v.parse_pull_progress(PULL_SNAP_1, {})
    # a totally empty later snapshot (API blip) must not zero the byte count
    p2 = v.parse_pull_progress("", p1)
    assert p2["total_bytes"] == p1["total_bytes"]


def test_parse_short_layer_id_rejected():
    """The layer id needs >=6 hex chars; a 2-char token is not a layer line."""
    p = v.parse_pull_progress("a1: Downloading [=>] 5MB/10MB", {})
    assert p["downloading"] is False and p["layers"] == {}


def test_to_pull_bytes_units():
    assert v._to_pull_bytes("1", "B") == 1
    assert v._to_pull_bytes("1", "kB") == 1_000
    assert v._to_pull_bytes("2.5", "MB") == 2_500_000
    assert v._to_pull_bytes("1.5", "GB") == 1_500_000_000


# --- BootThroughputSampler verdicts ------------------------------------------ #
def _loading(status_msg):
    return {"actual_status": "loading", "status_msg": status_msg}


def _feed_range(s, total_at, *, layer="a1b2c3d4e5f6", tot="9000", step=20,
                span=340):
    """Feed downloading snapshots at t=0,step,...,span where the layer's
    cur-bytes at time t is `total_at(t)` MB. Returns the final verdict."""
    verd = None
    for t in range(0, span + 1, step):
        mb = total_at(t)
        verd = s.feed(_loading(f"{layer}: Downloading [=>] {mb:.4f}MB/{tot}MB"), t)
    return verd


def test_sampler_slow_condemns():
    s = v.BootThroughputSampler(min_mbps=5, window_s=300, deadline_s=1500, start_t=0)
    verd = _feed_range(s, lambda t: t * 0.001)        # ~1 kB/s: crawling
    assert verd == "slow"
    assert s.last_mbps < 5 and s.phase == "downloading"


def test_sampler_fast_never_condemns():
    s = v.BootThroughputSampler(min_mbps=5, window_s=300, deadline_s=1500, start_t=0)
    verd = _feed_range(s, lambda t: t * 10.0)          # 10 MB/s: healthy
    assert verd is None
    assert s.last_mbps >= 5


def test_sampler_window_must_be_full():
    """No verdict before a FULL window of downloading-phase samples exists, even
    if the instantaneous rate is 0 (first ~5 min are never a kill)."""
    s = v.BootThroughputSampler(min_mbps=5, window_s=300, deadline_s=1500, start_t=0)
    verd = _feed_range(s, lambda t: 1.0, span=280)     # <300s of samples, flat
    assert verd is None


def test_sampler_extract_only_excluded_from_starvation_vote():
    """After download evidence, a long extract-only phase (CPU/disk-bound) must
    NOT trip the floor: the >=50%-downloading condition fails, so no condemn."""
    s = v.BootThroughputSampler(min_mbps=5, window_s=300, deadline_s=1500, start_t=0)
    s.feed(_loading("a1b2c3d4e5f6: Downloading [=>] 1.0MB/2000MB"), 0)  # dl evidence
    verd = None
    for t in range(20, 401, 20):
        verd = s.feed(_loading("a1b2c3d4e5f6: Extracting [=>] 2000MB/2000MB"), t)
    assert verd is None


def test_sampler_clock_starts_at_first_download_not_launch():
    """Scheduling/registry-auth snapshots (no Downloading) must not count as
    0 MB/s: a long provision phase then a fast pull is NOT condemned."""
    s = v.BootThroughputSampler(min_mbps=5, window_s=300, deadline_s=10 ** 9, start_t=0)
    # 400s of provision (no download evidence) -> no verdict, clock unstarted
    for t in range(0, 401, 20):
        assert s.feed(_loading(PROVISION_RECORD["status_msg"]), t) is None
    assert s.first_dl_t is None


def test_sampler_running_and_gone_and_deadline():
    s = v.BootThroughputSampler(min_mbps=5, window_s=300, deadline_s=100, start_t=0)
    assert s.feed({"actual_status": "running"}, 10) == "running"
    assert s.feed(None, 10) == "gone"
    assert s.feed({"actual_status": "exited"}, 10) == "gone"
    # deadline backstop applies even with no download evidence
    s2 = v.BootThroughputSampler(min_mbps=5, window_s=300, deadline_s=100, start_t=0)
    assert s2.feed(_loading(""), 200) == "deadline"


# --- boot_health_watch (injected fakes) -------------------------------------- #
class _Clock:
    def __init__(self):
        self.t = 0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_watch_slow_verdict():
    clk = _Clock()
    seq = [_loading(f"a1b2c3d4e5f6: Downloading [=>] {t * 0.001:.4f}MB/2000MB")
           for t in range(0, 341, 20)]
    i = {"n": 0}

    def gi(iid):
        r = seq[min(i["n"], len(seq) - 1)]
        i["n"] += 1
        return r

    verd = v.boot_health_watch("x", min_mbps=5, window_s=300, poll_s=20,
                               deadline_s=1500, get_instance=gi, now=clk.now,
                               sleep=clk.sleep)
    assert verd == "slow"


def test_watch_running_survives_transient_poll_failures():
    """A failed poll (get_instance raising) contributes NO sample and never
    condemns; the loop keeps going and exits 'running' when the box boots."""
    clk = _Clock()
    calls = {"n": 0}

    def gi(iid):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("api blip")
        return {"actual_status": "running"}

    verd = v.boot_health_watch("x", min_mbps=5, window_s=300, poll_s=20,
                               deadline_s=1500, get_instance=gi, now=clk.now,
                               sleep=clk.sleep)
    assert verd == "running"


def test_watch_deadline_on_persistently_dead_box():
    """Persistent failed polls (None each time) never sample, but the fixed
    deadline still bounds the loop -> 'deadline', never a hang."""
    clk = _Clock()
    verd = v.boot_health_watch("x", min_mbps=5, window_s=300, poll_s=20,
                               deadline_s=100, get_instance=lambda iid: None,
                               now=clk.now, sleep=clk.sleep)
    assert verd == "deadline"


# --- knob precedence: CLI > env > herdd.yaml > constant -------------------- #
def test_boot_knob_precedence(monkeypatch):
    monkeypatch.delenv("BOOT_MIN_MBPS", raising=False)
    assert v._boot_knob("BOOT_MIN_MBPS") == 5.0                 # constant
    monkeypatch.setenv("BOOT_MIN_MBPS", "9")
    assert v._boot_knob("BOOT_MIN_MBPS") == 9.0                 # env beats constant
    assert v._boot_knob("BOOT_MIN_MBPS", cli=3) == 3.0          # CLI beats env
    monkeypatch.setenv("BOOT_MIN_MBPS", "garbage")
    assert v._boot_knob("BOOT_MIN_MBPS") == 5.0                 # malformed -> constant
    assert v._boot_knob("BOOT_MBPS_WINDOW_S", cast=int) == 300
    assert v._boot_knob("BOOT_HEALTH_POLL_S", cast=int) == 20
    assert v._boot_knob("BOOT_MAX_HOST_RETRIES", cast=int) == 3


# --- build_throughput_observer: per-tick, holds state across ticks ----------- #
def test_throughput_observer_condemns_slow_only():
    clk = _Clock()
    seq = [_loading(f"a1b2c3d4e5f6: Downloading [=>] {t * 0.001:.4f}MB/2000MB")
           for t in range(0, 341, 20)]
    i = {"n": 0}

    def gi(iid):
        r = seq[min(i["n"], len(seq) - 1)]
        i["n"] += 1
        clk.t += 20
        return r

    obs = v.build_throughput_observer(get_instance=gi, min_mbps=5, window_s=300,
                                      now=clk.now)
    res = None
    for _ in seq:
        res = obs("inst-9")
    assert isinstance(res, dict) and res["verdict"] == "slow"
    assert res["window_s"] == 300 and res["phase"] == "downloading"
    assert res["mbps"] < 5


def test_throughput_observer_none_when_fast_or_failed_poll():
    clk = _Clock()
    # fast box: never condemns
    seq = [_loading(f"a1b2c3d4e5f6: Downloading [=>] {t * 10.0:.1f}MB/9000MB")
           for t in range(0, 341, 20)]
    i = {"n": 0}

    def gi(iid):
        r = seq[min(i["n"], len(seq) - 1)]
        i["n"] += 1
        clk.t += 20
        return r

    obs = v.build_throughput_observer(get_instance=gi, min_mbps=5, window_s=300,
                                      now=clk.now)
    assert all(obs("inst-f") is None for _ in seq)
    # failed poll (None) -> None, no sample, no crash
    obs2 = v.build_throughput_observer(get_instance=lambda iid: None, min_mbps=5,
                                       window_s=300, now=clk.now)
    assert obs2("inst-x") is None
