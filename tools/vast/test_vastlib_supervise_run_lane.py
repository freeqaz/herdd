"""`vastlib.supervise.run_lane` — the first real regression net for the run tick.

Why this file exists
--------------------
At the rev this port was cut from, `supervise_init`, `supervise_tick` and
`supervise_finalize` had **zero** direct tests anywhere in `tools/vast`
(verified: the only references outside `herdd.py` are `fleetd.py`'s three
`Hooks` methods and `FLEETD_DESIGN.md`). `test_fleetd.py` and
`test_standing_watch.py` define `FakeHooks.run_init/run_tick/run_finalize`,
which pin the ARITY the daemon calls with but never reach the real bodies. So
the 156-line money tick — the one copy of the run lane's policy, driven both by
the CLI loop and by the daemon — was covered only indirectly, through unit
tests of the pieces it calls.

That is the largest single risk in the whole port: a behavior drift in the tick
would be invisible to the suite. This file closes it. It drives the real
functions with stubbed seams and asserts on what the tick *does*: the emit
sequence, the order of the fence computation, which branch each `Action` takes,
and the park-vs-destroy split on the exit path.

Four things here are money-path pins, not coverage
--------------------------------------------------
* `supervise_finalize(..., destroy_on_park_failure=True)` is the DEFAULT and
  `fleetd` passes False. The inline loop destroys when a park does not take;
  the daemon parks and alarms and NEVER destroys (FLEETD_DESIGN §3/§8). Both
  arms are asserted, and so is the signature default itself.
* `_init_state` MUTATES its argparse namespace (backfills
  `gpu/gpu_ram/cuda/num_gpus` from the captured spec). Without it a
  `train --gpu-ram 24 --cuda 12.8 --supervise` child searches the whole market
  and the understudy lands on an 8 GB GTX 1080 (handoff-canary-2, 2026-07-15).
* The fence in `supervise_tick` is **pre-OR-post** tick. `act` was computed
  against the pre-tick world, so both a pre-tick-only and a post-tick-only open
  fence must suppress it — collapsing either side fires the destroyed primary's
  `emit_evicted` against the freshly promoted understudy.
* The tri-state `None` floats stay `None` for UNKNOWN (`dph_total`,
  `on_demand`, `market_min_bid`, `first_seen_dph`, ...). A `0.0` default there
  reads as a known-zero price (defect #67), so `_init_state`'s seeds are
  asserted `is None`, not falsy.

Isolation
---------
Nothing in this file can reach the network, B2 or the vast API. `_sup_emit` is
replaced at `vastlib.supervise.journal` (module attribute — which is also the
LATE-BINDING proof the floor hooks need), every lifecycle PUT/DELETE is
replaced at `vastlib.boxes.lifecycle`, the two spec readers are replaced at
`vastlib.launch.spec`, and `_ensure_b2_remote` at `vastlib.storage.b2`. The
conftest guard over `request_soft` refuses any mutation that somehow escapes.
`monkeypatch.setattr` is used WITHOUT `raising=False` throughout, so a seam
that moves to a different module fails loudly here instead of going vacuous.

What is deliberately NOT here
-----------------------------
* No repointing of `test_supervise.py`, `test_self_floor_lag.py` or
  `test_ladder_core.py`. They stay UNEDITED under the plan §8 add-only
  amendment and keep steering the flat `herdd` copies; this file is the
  parallel net for the vastlib copy.
* No re-testing of `ladder_core.self_floor_guard`'s state machine
  (`test_ladder_core.py` owns it) — only the run lane's three wrapper facts:
  the lenient tenancy gate (D1), the hooks instance, and the
  `self_floor_at = None` ASSIGN (D5).
* No assertion that the run and jobs lanes agree. They deliberately diverge;
  `test_ladder_core.py`'s parity harness owns the shared core.

Provenance: created 2026-08-16 alongside `vastlib/supervise/run_lane.py`,
plan §8 step 4.
"""

from __future__ import annotations

import argparse
import base64
import inspect
import sys
import time
from pathlib import Path

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

import bidpolicy                                       # noqa: E402  Zone S
import ladder_core                                     # noqa: E402  Zone S

from vastlib.boxes import lifecycle                    # noqa: E402
from vastlib.launch import spec as launch_spec         # noqa: E402
from vastlib.storage import b2                         # noqa: E402
from vastlib.supervise import handoff, journal, replacement, run_lane  # noqa: E402

NOW = 1_770_000_000.0


# --------------------------------------------------------------------------- #
# fixtures — the seams, all patched by module attribute
# --------------------------------------------------------------------------- #
@pytest.fixture
def emits(monkeypatch):
    """Capture every `_sup_emit` the run lane makes, in order.

    Patched on the JOURNAL module, which is the only place the port may resolve
    it from: the floor hooks and the tick both call `journal._sup_emit(...)` at
    call time. A `from .journal import _sup_emit` in run_lane.py would make this
    fixture (and the 53 patch sites it stands in for) vacuous."""
    seen: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(journal, "_sup_emit",
                        lambda rid, ev, **kw: seen.append((rid, ev, kw)) or {})
    return seen


@pytest.fixture
def no_park(monkeypatch):
    """Refuse every real box mutation; record the calls instead."""
    calls: list[tuple] = []
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda iid, state: calls.append(("put", iid, state)) or (True, None))
    monkeypatch.setattr(lifecycle, "_wait_states_soft",
                        lambda iid, targets, timeout, **kw:
                        calls.append(("wait", iid, sorted(targets))) or (True, "stopped"))
    monkeypatch.setattr(lifecycle, "_destroy_soft",
                        lambda iid, *a, **k: calls.append(("destroy", iid)) or (True, None))
    return calls


def _ns(**kw):
    """A supervise argparse Namespace with the defaults the CLI parser gives."""
    base = dict(run_id="r-test", max_bid=None, max_relaunch=3, budget=None,
                wall_budget=None, defend_at=None, rescue_wait=600, interval=30,
                dry_run=False, handoff=True, strict_ceiling=False,
                gpu=None, gpu_ram=None, cuda=None, num_gpus=1,
                image=None, disk=None, onstart=None, runtype=None, env=None,
                price=None, boot_health=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _st(**kw):
    """A minimal post-`_observe` run state: only the keys the tick reads.

    `now` / `_last_cost_emit_t` are the REAL clock, not `NOW`: the tick compares
    `_last_cost_emit_t` against `time.time()` on a 900 s period, so a frozen
    fixture timestamp would make the periodic cost emit fire on every tick and
    the emit-sequence assertions below would drift with the calendar."""
    base = {"run_id": "r-test", "obs_status": "ok", "last_error": None,
            "relaunch_count": 0, "spend_usd": 0.0, "wall_clock_s": 0.0,
            "wall_budget_s": None, "budget_usd": None, "max_bid": None,
            "max_relaunch": 3, "defend_at": None, "present": True,
            "actual_status": "running", "instance_id": "700", "husk_id": None,
            "machine_id": 1234, "last_bid": 0.20, "on_demand": 1.0,
            "is_bid": True, "now": time.time(), "_last_cost_emit_t": time.time(),
            "backoff_deadline": 0, "rescue_deadline": 0,
            "rescue_attempted": False, "not_live_streak": 0,
            "evicted_pending": False, "decay_streak": 0, "bid_history": [],
            "launch_spec": {}}
    base.update(kw)
    return base


class _Tick:
    """Every seam `supervise_tick` reaches, stubbed and recorded.

    Patched onto the sibling modules by attribute, so a seam that lands in a
    different module than the port assumed raises AttributeError here rather
    than silently never being called."""

    def __init__(self, monkeypatch, *, act=("noop", "ok"), boot_health=None,
                 boot_sla=None, relaunch="relaunched", phase_after=None):
        self.calls: list[str] = []
        self.act = bidpolicy.Action(*act)
        self.phase_after = phase_after
        monkeypatch.setattr(run_lane, "_observe", self._observe)
        monkeypatch.setattr(run_lane, "_accrue_cost", self._accrue)
        monkeypatch.setattr(run_lane, "_emit_cost", self._emit_cost)
        monkeypatch.setattr(run_lane, "_do_bid_move", self._bid_move)
        monkeypatch.setattr(replacement, "_relaunch",
                            lambda st, a: self.calls.append("relaunch") or relaunch)
        monkeypatch.setattr(run_lane, "_supervise_boot_health",
                            lambda st, a, **k: boot_health)
        monkeypatch.setattr(replacement, "_supervise_boot_sla",
                            lambda st, a, **k: boot_sla)
        monkeypatch.setattr(bidpolicy, "_refresh_default_ceiling",
                            lambda st: self.calls.append("ceiling"))
        monkeypatch.setattr(bidpolicy, "_preferred_ceiling_alarm",
                            lambda st: (False, 0.8))
        monkeypatch.setattr(bidpolicy, "_handoff_trigger",
                            lambda st: (False, None, None, "off"))
        # (streak, since): the lane stores both, because the decay dwell is a
        # DURATION and a count alone cannot express one.
        monkeypatch.setattr(bidpolicy, "next_decay_state",
                            lambda st: bidpolicy.DecayStreak(1, st.get("now")))
        monkeypatch.setattr(bidpolicy, "poll", self._poll)
        monkeypatch.setattr(handoff, "_handoff_tick", self._handoff_tick)

    def _observe(self, st, a):
        self.calls.append("observe")
        return st

    def _accrue(self, st):
        self.calls.append("accrue")
        return st

    def _emit_cost(self, st, run_id):
        self.calls.append("emit_cost")

    def _bid_move(self, st, a, act):
        self.calls.append(f"bid_move:{act.kind}")

    def _poll(self, st):
        self.calls.append("poll")
        self.polled_fenced = st.get("handoff_fenced")
        return self.act

    def _handoff_tick(self, st, a, hf, act):
        self.calls.append("handoff_tick")
        if self.phase_after is not None:
            hf["phase"] = self.phase_after


# --------------------------------------------------------------------------- #
# _read_onstart — passthrough unless it names a file
# --------------------------------------------------------------------------- #
def test_read_onstart_passes_a_script_through_and_reads_a_path(tmp_path):
    assert run_lane._read_onstart(None) is None
    assert run_lane._read_onstart("") is None
    assert run_lane._read_onstart("echo hi") == "echo hi"
    p = tmp_path / "onstart.sh"
    p.write_text("#!/bin/bash\necho from-file\n")
    assert run_lane._read_onstart(str(p)) == "#!/bin/bash\necho from-file\n"


# --------------------------------------------------------------------------- #
# _capture_launch_spec — spec-first, then the legacy event scrape
# --------------------------------------------------------------------------- #
def _wire_spec(monkeypatch, spec_json, events=()):
    monkeypatch.setattr(launch_spec, "_read_spec_soft", lambda rid: dict(spec_json))
    monkeypatch.setattr(launch_spec, "_raw_events_soft", lambda rid: [dict(e) for e in events])


def test_capture_launch_spec_prefers_the_v1_spec(monkeypatch):
    """The good path: runs/<RUN_ID>/spec.json v=1 supplies image/disk/filters,
    the env map, the secret NAMES (never values) and the base64 onstart."""
    _wire_spec(monkeypatch, {
        "v": 1, "image": "img:tag", "disk": 80, "runtype": "train",
        "runset": "rs1", "image_login": "reg", "gpu": ["RTX_4090"],
        "gpu_ram": 24.0, "num_gpus": 2, "cuda": 12.8,
        "env": {"A": "1"}, "secret_env_keys": ["B2_KEY_ID"],
        "onstart_b64": base64.b64encode(b"echo boot").decode(),
        "bid": {"orig": 0.42},
    })
    spec, orig_bid = run_lane._capture_launch_spec("r-test", _ns())
    assert spec["image"] == "img:tag" and spec["disk"] == 80
    assert spec["gpu"] == ["RTX_4090"] and spec["gpu_ram"] == 24.0
    assert spec["num_gpus"] == 2 and spec["cuda"] == 12.8
    assert spec["env"] == {"A": "1"}
    assert spec["secret_env_keys"] == ["B2_KEY_ID"]
    assert spec["onstart"] == "echo boot"
    assert orig_bid == 0.42


def test_capture_launch_spec_falls_back_to_the_event_scrape(monkeypatch):
    """No v=1 spec (a pre-spec run, or a transient B2 failure) degrades to the
    RAW launched/supervised events — lower fidelity, first value wins."""
    _wire_spec(monkeypatch, {}, events=[
        {"event": "launched", "image": "old:tag", "disk": 40, "dph": 0.11,
         "onstart": "echo old", "runtype": "serve"},
        {"event": "supervised", "image": "newer:tag", "disk": 999},
    ])
    spec, orig_bid = run_lane._capture_launch_spec("r-test", _ns())
    assert spec["image"] == "old:tag", "first event wins (k not in spec)"
    assert spec["disk"] == 40 and spec["onstart"] == "echo old"
    assert spec["runtype"] == "serve"
    assert orig_bid == 0.11
    assert "secret_env_keys" not in spec, "the legacy path never invented one"


def test_capture_launch_spec_cli_fills_gaps_and_env_overrides_last(monkeypatch):
    _wire_spec(monkeypatch, {"v": 1, "image": "img:tag", "env": {"A": "1", "B": "2"}})
    a = _ns(disk=120, runtype="train", env=["B=override", "C=3", "malformed"])
    spec, _bid = run_lane._capture_launch_spec("r-test", a)
    assert spec["image"] == "img:tag", "a CLI flag never overrides the spec"
    assert spec["disk"] == 120, "...but it fills a gap the spec left"
    assert spec["env"] == {"A": "1", "B": "override", "C": "3"}


def test_capture_launch_spec_survives_a_corrupt_onstart_b64(monkeypatch):
    _wire_spec(monkeypatch, {"v": 1, "image": "i", "onstart_b64": "!!!not-base64!!!"})
    spec, _bid = run_lane._capture_launch_spec("r-test", _ns())
    assert "onstart" not in spec, "a corrupt blob is dropped, not raised"


def test_capture_launch_spec_takes_the_explicit_price_first(monkeypatch):
    """--price on the supervise invocation is the operator's own original bid;
    it wins over both the event dph and the spec's bid.orig."""
    _wire_spec(monkeypatch, {"v": 1, "bid": {"orig": 0.42}},
               events=[{"event": "launched", "dph": 0.11}])
    _spec, orig_bid = run_lane._capture_launch_spec("r-test", _ns(price="0.77"))
    assert orig_bid == 0.77


# --------------------------------------------------------------------------- #
# _init_state — the seeds, and the argument MUTATION that is load-bearing
# --------------------------------------------------------------------------- #
def test_init_state_backfills_the_search_filters_from_the_spec(monkeypatch):
    """MONEY PATH: without this backfill a `train --gpu-ram 24 --cuda 12.8
    --supervise` child searches the WHOLE market on relaunch, and the live
    canary's understudy landed on an 8GB GTX 1080 that could not run the cu128
    image (handoff-canary-2, 2026-07-15). The mutation of `a` IS the mechanism —
    `build_search_query(a)` drives both `_relaunch` and `_handoff_pick_offer`."""
    _wire_spec(monkeypatch, {"v": 1, "gpu": ["RTX_4090"], "gpu_ram": 24.0,
                             "cuda": 12.8, "num_gpus": 4, "bid": {"orig": 0.50}})
    a = _ns()
    st = run_lane._init_state(a)
    assert a.gpu == ["RTX_4090"] and a.gpu_ram == 24.0
    assert a.cuda == 12.8 and a.num_gpus == 4
    assert st["launch_spec"]["gpu"] == ["RTX_4090"]


def test_init_state_explicit_filters_beat_the_spec(monkeypatch):
    _wire_spec(monkeypatch, {"v": 1, "gpu": ["RTX_4090"], "gpu_ram": 24.0,
                             "cuda": 12.8, "num_gpus": 4})
    a = _ns(gpu=["H100_SXM"], gpu_ram=80.0, cuda=12.4, num_gpus=2)
    run_lane._init_state(a)
    assert a.gpu == ["H100_SXM"] and a.gpu_ram == 80.0
    assert a.cuda == 12.4 and a.num_gpus == 2


def test_init_state_seeds_the_fallback_ceiling_and_the_unknown_floats(monkeypatch):
    _wire_spec(monkeypatch, {"v": 1, "bid": {"orig": 0.40}})
    st = run_lane._init_state(_ns())
    assert st["max_bid"] == round(bidpolicy.BID_FALLBACK_DPH_MULT * 0.40, 3)
    assert st["explicit_max_bid"] is False
    assert st["first_seen_dph"] == 0.40
    # tri-state: None means UNKNOWN. A 0.0 here would read as a known-zero
    # price and the ceiling/accrual maths would believe it (defect #67).
    for k in ("dph_total", "on_demand", "market_min_bid", "num_gpus",
              "self_floor_at", "instance_id", "husk_id", "last_error",
              "boot_sampler", "boot_sampler_iid"):
        assert st[k] is None, f"{k} must stay None-for-UNKNOWN"
    assert st["obs_status"] == "ok" and st["is_bid"] is False
    assert st["floor_samples"] == [] and st["excluded_machines"] == []


def test_init_state_keeps_an_explicit_max_bid_fixed(monkeypatch):
    _wire_spec(monkeypatch, {"v": 1, "bid": {"orig": 0.40}})
    st = run_lane._init_state(_ns(max_bid=1.25))
    assert st["max_bid"] == 1.25 and st["explicit_max_bid"] is True


def test_init_state_leaves_max_bid_unknown_without_an_original_bid(monkeypatch):
    _wire_spec(monkeypatch, {"v": 1})
    st = run_lane._init_state(_ns())
    assert st["max_bid"] is None, "no original bid -> no fabricated ceiling"


# --------------------------------------------------------------------------- #
# _RunLaneFloorHooks + _self_floor_guard — D1, D5 and the LATE binding
# --------------------------------------------------------------------------- #
def test_the_hooks_resolve_the_emitter_at_call_time(emits):
    """LATE BINDING. The hook bodies must read `_sup_emit` off the journal
    MODULE when they fire — the `emits` fixture patched it after the class was
    defined, and after run_lane was imported. A `from .journal import _sup_emit`
    would capture the original at import time and this assertion would fail
    (which is the whole point: the same shape makes 53 existing patch sites
    vacuous rather than red)."""
    st = {"run_id": "r1", "last_bid": 0.20, "machine_id": 7, "instance_id": "700",
          "num_gpus": 2}
    match = bidpolicy.SelfFloor(kind="prior", price=0.016, age_s=200.0)
    run_lane._RUN_FLOOR_HOOKS.self_floor(st, market_min_bid=0.016, match=match,
                                         surviving_floor=None, visible=False)
    run_lane._RUN_FLOOR_HOOKS.floor_blind(st, since_s=1801.4)
    assert [e[1] for e in emits] == ["bid_self_floor", "bid_floor_blind"]
    assert emits[0][2]["matched"] == "prior"
    assert emits[0][2]["matched_bid"] == 0.016
    assert emits[0][2]["matched_age_s"] == 200.0
    assert emits[0][2]["standing_bid"] == 0.20
    assert emits[1][2]["since_s"] == 1801.4
    # the silent hooks stay silent
    run_lane._RUN_FLOOR_HOOKS.scaled_read(st, 0.5)
    run_lane._RUN_FLOOR_HOOKS.episode_end(st, market=0.5)
    assert len(emits) == 2


def test_the_guard_suppresses_our_own_echo_and_assigns_the_latch(emits):
    """The run lane's three wrapper facts, in one call: the hooks instance is
    wired (an emit lands), the read is suppressed (None), and D5 — this lane
    ASSIGNS `self_floor_at = None` where the jobs lane pops the key."""
    st = {"is_bid": True, "last_bid": 0.0421, "machine_id": 115469,
          "run_id": "r1", "instance_id": "700", "now": NOW,
          "bid_history": [(NOW - 200, 0.016, 115469)]}
    assert run_lane._self_floor_guard(st, 0.016, live=True) is None
    assert emits and emits[0][1] == "bid_self_floor"
    assert st["self_floor_at"] is not None, "the (value, kind) dedup latch is set"
    # D5: a REAL competing read ends the episode by ASSIGNING None — the run
    # lane keeps the key (the jobs lane pops it). Both read back None; a port
    # that "tidied" this into a pop would diverge from the jobs lane's mirror
    # in the opposite direction.
    assert run_lane._self_floor_guard(st, 0.90, live=True) == 0.90
    assert "self_floor_at" in st and st["self_floor_at"] is None


def test_the_tenancy_gate_is_the_lenient_run_lane_one(emits):
    """D1: `live` is passed IN (it is `_observe`'s lenient `_still_tenant`,
    which tolerates a running->exited->running flap) and is ANDed with
    `st['is_bid']`. Not the tenant -> the read is a real market floor."""
    st = {"is_bid": True, "last_bid": 0.0421, "machine_id": 115469,
          "run_id": "r1", "instance_id": "700", "now": NOW,
          "bid_history": [(NOW - 200, 0.016, 115469)]}
    assert run_lane._self_floor_guard(st, 0.016, live=False) == 0.016
    st["is_bid"] = False
    assert run_lane._self_floor_guard(st, 0.016, live=True) == 0.016
    assert emits == [], "a non-tenant read journals nothing"


def test_the_hooks_singleton_is_the_one_the_guard_passes(monkeypatch):
    seen = {}
    real = ladder_core.self_floor_guard

    def spy(ctx, market, **kw):
        seen.update(kw)
        return real(ctx, market, **kw)

    monkeypatch.setattr(ladder_core, "self_floor_guard", spy)
    st = {"is_bid": True, "last_bid": 0.2, "machine_id": 5, "run_id": "r1",
          "now": NOW, "bid_history": []}
    run_lane._self_floor_guard(st, 0.2, live=True, floors=[0.2], scaled=True)
    assert seen["hooks"] is run_lane._RUN_FLOOR_HOOKS
    assert seen["tenant"] is True and seen["scaled"] is True
    assert seen["floors"] == [0.2] and seen["machine_id"] == 5
    assert seen["now"] == NOW


# --------------------------------------------------------------------------- #
# supervise_tick — the observe/accrue prologue and the two hard stops
# --------------------------------------------------------------------------- #
def test_tick_returns_stop_fatal_on_a_fatal_observation(monkeypatch, emits):
    """An API outage is NOT an eviction, but a fatal (401/404/no-key) stops the
    supervisor before anything accrues or emits."""
    t = _Tick(monkeypatch)
    act = run_lane.supervise_tick(_st(obs_status="fatal", last_error="HTTP 401"),
                                  _ns(), {"phase": "IDLE"}, False)
    assert (act.kind, act.reason) == ("stop_fatal", "observe_fatal:HTTP 401")
    assert t.calls == ["observe"] and emits == []


def test_tick_holds_through_a_transient_outage_but_honors_the_wall_cap(monkeypatch, emits):
    t = _Tick(monkeypatch)
    st = _st(obs_status="transient", last_error="timeout")
    assert run_lane.supervise_tick(st, _ns(), {"phase": "IDLE"}, False) is None
    assert t.calls == ["observe", "accrue"], "no poll, no ladder, while blind"
    assert emits[-1][1] == "heartbeat"
    assert emits[-1][2]["actual_status"] == "unknown"
    # the HARD wall cap still ends the run even while the API is unreadable
    st2 = _st(obs_status="transient", wall_budget_s=100.0, wall_clock_s=101.0)
    act = run_lane.supervise_tick(st2, _ns(), {"phase": "IDLE"}, False)
    assert (act.kind, act.reason) == ("stop_budget", "wall_budget")


@pytest.mark.parametrize("verdict", ["stop_fatal", "stop_budget"])
def test_tick_lets_the_boot_watchdogs_stop_the_run(monkeypatch, emits, verdict):
    t = _Tick(monkeypatch, boot_health=verdict)
    act = run_lane.supervise_tick(_st(last_error="pull starved"), _ns(),
                                  {"phase": "IDLE"}, False)
    assert (act.kind, act.reason) == (verdict, "pull starved")
    assert "poll" not in t.calls
    t2 = _Tick(monkeypatch, boot_sla=verdict)
    act2 = run_lane.supervise_tick(_st(last_error=None), _ns(), {"phase": "IDLE"}, False)
    assert (act2.kind, act2.reason) == (verdict, verdict)
    assert "poll" not in t2.calls


def test_tick_short_circuits_on_a_condemned_box(monkeypatch, emits):
    """`condemned` means the watchdog already replaced the box: keep looping,
    but do not run the ladder against the corpse."""
    t = _Tick(monkeypatch, boot_health="condemned")
    assert run_lane.supervise_tick(_st(), _ns(), {"phase": "IDLE"}, False) is None
    assert "poll" not in t.calls and "ceiling" not in t.calls
    t2 = _Tick(monkeypatch, boot_sla="condemned")
    assert run_lane.supervise_tick(_st(), _ns(), {"phase": "IDLE"}, False) is None
    assert "poll" not in t2.calls


def test_tick_heartbeats_and_polls_on_the_healthy_path(monkeypatch, emits):
    t = _Tick(monkeypatch)
    st = _st()
    assert run_lane.supervise_tick(st, _ns(), {"phase": "IDLE"}, False) is None
    assert t.calls == ["observe", "accrue", "ceiling", "poll"]
    assert [e[1] for e in emits] == ["heartbeat"]
    assert emits[0][2]["actual_status"] == "running"
    assert st["decay_streak"] == 1, "the streak advances BEFORE poll reads it"
    assert st["backoff_ready"] is True


def test_tick_waits_out_a_rescue_window_without_polling(monkeypatch, emits):
    """SPOT_DESIGN §3.2: while a rescue bid is scheduling, the tick returns
    early — no ladder, no relaunch — until the deadline or the wall cap."""
    t = _Tick(monkeypatch)
    st = _st(present=False, actual_status="exited",
             rescue_deadline=time.time() + 300)
    assert run_lane.supervise_tick(st, _ns(), {"phase": "IDLE"}, False) is None
    assert "poll" not in t.calls
    # the same box back in a LIVE state resolves the wait and journals `rescued`
    st2 = _st(rescue_deadline=time.time() + 300, rescue_attempted=True,
              evicted_pending=True, not_live_streak=4)
    assert run_lane.supervise_tick(st2, _ns(), {"phase": "IDLE"}, False) is None
    assert [e[1] for e in emits[-2:]] == ["heartbeat", "rescued"]
    assert st2["rescue_deadline"] == 0 and st2["rescue_attempted"] is False
    assert st2["not_live_streak"] == 0 and st2["evicted_pending"] is False


# --------------------------------------------------------------------------- #
# supervise_tick — the act dispatch and the pre-OR-post fence
# --------------------------------------------------------------------------- #
def test_tick_dispatches_a_bid_move_and_restarts_the_decay_run(monkeypatch, emits):
    t = _Tick(monkeypatch, act=("lower_bid", "decay"))
    st = _st()
    assert run_lane.supervise_tick(st, _ns(), {"phase": "IDLE"}, False) is None
    assert "bid_move:lower_bid" in t.calls
    assert st["decay_streak"] == 0, "a PUT issued -> the decay run restarts"


def test_tick_emits_eviction_and_resets_the_self_floor_episode(monkeypatch, emits):
    """The run-lane twin of `_job_announce_eviction`'s reset: a frozen
    `self_floor_since` carried across a stopped gap fakes a 'continuous'
    floor-blind alarm on the replacement box."""
    t = _Tick(monkeypatch, act=("emit_evicted", "vanished"))
    st = _st(husk_id="700", relaunch_count=2, self_floor_since=NOW,
             self_floor_at=(0.2, "standing"), self_floor_sustained_said=True)
    assert run_lane.supervise_tick(st, _ns(), {"phase": "IDLE"}, False) is None
    assert [e[1] for e in emits] == ["heartbeat", "evicted"]
    assert emits[1][2] == {"instance_id": "700", "reason": "vanished"}
    assert st["evicted_pending"] is True
    assert st["backoff_deadline"] > time.time(), "120s*2^n backoff armed"
    assert st.get("self_floor_since") is None
    assert st.get("self_floor_at") is None
    assert not st.get("self_floor_sustained_said")
    assert "emit_cost" in t.calls, "a transition snapshots the cost"


def test_tick_relaunches_and_surfaces_a_terminal_verdict(monkeypatch, emits):
    t = _Tick(monkeypatch, act=("relaunch", "evicted"), relaunch="relaunched")
    assert run_lane.supervise_tick(_st(), _ns(), {"phase": "IDLE"}, False) is None
    assert "relaunch" in t.calls
    _Tick(monkeypatch, act=("relaunch", "evicted"), relaunch="stop_budget")
    act = run_lane.supervise_tick(_st(last_error="over budget"), _ns(),
                                  {"phase": "IDLE"}, False)
    assert (act.kind, act.reason) == ("stop_budget", "over budget")


def test_tick_returns_a_terminal_action_unchanged(monkeypatch, emits):
    _Tick(monkeypatch, act=("stop_terminal", "job done"))
    act = run_lane.supervise_tick(_st(), _ns(), {"phase": "IDLE"}, False)
    assert (act.kind, act.reason) == ("stop_terminal", "job done")


def test_the_fence_is_open_before_poll_reads_intended_status(monkeypatch, emits):
    """poll() must SEE the fence: the fence's own park (and the drain's own
    destroy) otherwise trip its operator-intent rows and the supervisor exits
    mid-cutover (live canary handoff-canary-2, 2026-07-15)."""
    open_phase = handoff._HANDOFF_FENCE_OPEN[0]
    t = _Tick(monkeypatch, act=("noop", "ok"))
    st = _st()
    run_lane.supervise_tick(st, _ns(), {"phase": open_phase}, True)
    assert t.polled_fenced is True
    assert st["handoff_fenced"] is True
    # ...and with handoff off the flag is False, never absent
    t2 = _Tick(monkeypatch, act=("noop", "ok"))
    st2 = _st()
    run_lane.supervise_tick(st2, _ns(), {"phase": open_phase}, False)
    assert t2.polled_fenced is False and st2["handoff_fenced"] is False


@pytest.mark.parametrize("pre_phase,phase_after", [
    (handoff._HANDOFF_FENCE_OPEN[0], "IDLE"),   # fence CLOSED by this tick
    ("IDLE", handoff._HANDOFF_FENCE_OPEN[0]),   # fence OPENED by this tick
])
def test_the_fence_is_pre_or_post_tick(monkeypatch, emits, pre_phase, phase_after):
    """`act` was computed against the PRE-tick world. On the tick where
    `complete` resets the phase to IDLE, the pre-tick DRAINING must still
    suppress the stale act — otherwise the destroyed primary's `emit_evicted`
    fires against the freshly promoted understudy. The post-tick half is the
    mirror: a fence armed during this tick suppresses the act it precedes."""
    t = _Tick(monkeypatch, act=("emit_evicted", "vanished"), phase_after=phase_after)
    st = _st(husk_id="700")
    hf = {"phase": pre_phase}
    assert run_lane.supervise_tick(st, _ns(), hf, True) is None
    assert [e[1] for e in emits] == ["heartbeat"], "no `evicted` while fenced"
    assert st["evicted_pending"] is False
    assert "handoff_tick" in t.calls


def test_a_fenced_tick_suppresses_bid_moves_and_relaunch(monkeypatch, emits):
    open_phase = handoff._HANDOFF_FENCE_OPEN[0]
    t = _Tick(monkeypatch, act=("raise_bid", "defend"))
    assert run_lane.supervise_tick(_st(), _ns(), {"phase": open_phase}, True) is None
    assert not [c for c in t.calls if c.startswith("bid_move")]
    t2 = _Tick(monkeypatch, act=("relaunch", "evicted"))
    assert run_lane.supervise_tick(_st(), _ns(), {"phase": open_phase}, True) is None
    assert "relaunch" not in t2.calls, "the primary is being retired, not replaced"


def test_the_handoff_tick_runs_only_when_handoff_is_on(monkeypatch, emits):
    t = _Tick(monkeypatch)
    run_lane.supervise_tick(_st(), _ns(), {"phase": "IDLE"}, False)
    assert "handoff_tick" not in t.calls
    t2 = _Tick(monkeypatch)
    run_lane.supervise_tick(_st(), _ns(), {"phase": "IDLE"}, True)
    assert "handoff_tick" in t2.calls


def test_the_preferred_ceiling_alarm_fires_once_per_breach(monkeypatch, emits):
    monkeypatch.setattr(bidpolicy, "_preferred_ceiling_alarm", lambda st: (True, 0.8))
    _Tick(monkeypatch)
    monkeypatch.setattr(bidpolicy, "_preferred_ceiling_alarm", lambda st: (True, 0.8))
    st = _st()
    run_lane.supervise_tick(st, _ns(), {"phase": "IDLE"}, False)
    run_lane.supervise_tick(st, _ns(), {"phase": "IDLE"}, False)
    alarms = [e for e in emits if e[1] == "bid_over_preferred_ceiling"]
    assert len(alarms) == 1 and st["_pref_alarmed"] is True
    # dropping back under the line re-arms it
    monkeypatch.setattr(bidpolicy, "_preferred_ceiling_alarm", lambda st: (False, 0.8))
    run_lane.supervise_tick(st, _ns(), {"phase": "IDLE"}, False)
    assert st["_pref_alarmed"] is False


def test_the_dwell_counter_reads_the_trigger_not_the_alarm(monkeypatch, emits):
    """2026-08-08 trigger-domain fix, both lanes: `_over_pref` (the handoff
    dwell input) is the TRIGGER's verdict, not the alarm's."""
    monkeypatch.setattr(bidpolicy, "_preferred_ceiling_alarm", lambda st: (False, 0.8))
    _Tick(monkeypatch)
    monkeypatch.setattr(bidpolicy, "_preferred_ceiling_alarm", lambda st: (False, 0.8))
    monkeypatch.setattr(bidpolicy, "_handoff_trigger",
                        lambda st: (True, None, 0.5, "cheaper box"))
    st = _st()
    run_lane.supervise_tick(st, _ns(), {"phase": "IDLE"}, False)
    assert st["_over_pref"] is True


# --------------------------------------------------------------------------- #
# supervise_finalize — the park-then-destroy MONEY PATH
# --------------------------------------------------------------------------- #
def _final_ns_st():
    return _ns(), _st(spend_usd=12.5, relaunch_count=1)


def test_finalize_parks_a_live_box_on_a_hard_cap(monkeypatch, emits, no_park):
    monkeypatch.setattr(run_lane, "_emit_cost", lambda st, rid: None)
    a, st = _final_ns_st()
    run_lane.supervise_finalize(st, a, bidpolicy.Action("stop_budget", "budget"),
                                {"phase": "IDLE"}, False)
    assert [c[0] for c in no_park] == ["put", "wait"], "parked, never destroyed"
    assert no_park[0] == ("put", "700", "stopped")
    assert [e[1] for e in emits] == ["stopping", "supervisor_exiting"]
    assert emits[0][2] == {"reason": "supervisor_budget"}


def test_finalize_destroys_when_the_park_fails_and_it_owns_the_box(monkeypatch,
                                                                   emits, no_park):
    """DEFAULT arm (the inline CLI loop): SUPERVISE_DESIGN §5 — if the park does
    not take, destroy, so the cap still guarantees the GPU bill stops."""
    monkeypatch.setattr(run_lane, "_emit_cost", lambda st, rid: None)
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda iid, state: (False, "HTTP 500"))
    a, st = _final_ns_st()
    run_lane.supervise_finalize(st, a, bidpolicy.Action("stop_budget", "budget"),
                                {"phase": "IDLE"}, False)
    assert ("destroy", "700") in no_park


def test_finalize_never_destroys_for_fleetd(monkeypatch, emits, no_park, capsys):
    """fleetd passes destroy_on_park_failure=False: the daemon parks and ALARMS
    and never destroys on its own (FLEETD_DESIGN §3/§8). This is the money-path
    contract the kwarg exists for."""
    monkeypatch.setattr(run_lane, "_emit_cost", lambda st, rid: None)
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda iid, state: (False, "HTTP 500"))
    a, st = _final_ns_st()
    run_lane.supervise_finalize(st, a, bidpolicy.Action("stop_budget", "budget"),
                                {"phase": "IDLE"}, False,
                                destroy_on_park_failure=False)
    assert [c for c in no_park if c[0] == "destroy"] == []
    out = capsys.readouterr().out
    assert "PARK FAILED" in out and "STILL BILLING" in out


def test_the_destroy_default_is_true():
    """A pin, not a behavior test: flipping this default silently hands the
    daemon a destroy it must never have (fleetd passes False explicitly)."""
    sig = inspect.signature(run_lane.supervise_finalize)
    assert sig.parameters["destroy_on_park_failure"].default is True
    assert sig.parameters["destroy_on_park_failure"].kind is inspect.Parameter.KEYWORD_ONLY


def test_finalize_leaves_a_non_budget_exit_alone(monkeypatch, emits, no_park):
    monkeypatch.setattr(run_lane, "_emit_cost", lambda st, rid: None)
    a, st = _final_ns_st()
    run_lane.supervise_finalize(st, a, bidpolicy.Action("stop_terminal", "done"),
                                {"phase": "IDLE"}, False)
    assert no_park == [], "a completed run is never parked by the supervisor"
    assert [e[1] for e in emits] == ["supervisor_exiting"]


def test_finalize_reaps_a_mid_flight_twin_only_with_handoff_on(monkeypatch,
                                                              emits, no_park):
    reaped = []
    monkeypatch.setattr(run_lane, "_emit_cost", lambda st, rid: None)
    monkeypatch.setattr(handoff, "_handoff_reap_on_exit",
                        lambda st, a, hf: reaped.append(hf))
    a, st = _final_ns_st()
    act = bidpolicy.Action("stop_terminal", "done")
    run_lane.supervise_finalize(st, a, act, {"phase": "IDLE"}, False)
    assert reaped == []
    run_lane.supervise_finalize(st, a, act, {"phase": "CUTOVER"}, True)
    assert reaped == [{"phase": "CUTOVER"}]


# --------------------------------------------------------------------------- #
# supervise_init — the bootstrap the daemon and the CLI share
# --------------------------------------------------------------------------- #
@pytest.fixture
def init_seams(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: calls.append("b2"))
    monkeypatch.setattr(handoff, "_init_handoff_state",
                        lambda: {"phase": "IDLE", "handoffs_done": 0})
    monkeypatch.setattr(handoff, "_handoff_reconcile",
                        lambda st, a, hf: calls.append("reconcile"))
    monkeypatch.setattr(launch_spec, "_read_spec_soft",
                        lambda rid: {"v": 1, "image": "img:tag", "disk": 60,
                                     "runtype": "train", "runset": "rs1",
                                     "bid": {"orig": 0.40}})
    monkeypatch.setattr(launch_spec, "_raw_events_soft", lambda rid: [])
    return calls


def test_supervise_init_emits_the_policy_snapshot(init_seams, emits):
    st, hf, handoff_on = run_lane.supervise_init(_ns(max_relaunch=5, budget=20.0))
    assert "b2" in init_seams, "the B2 remote is a precondition, not a hope"
    assert [e[1] for e in emits] == ["supervisor_started", "supervised"]
    snap = emits[1][2]
    assert snap["max_relaunch"] == 5 and snap["budget_usd"] == 20.0
    assert snap["image"] == "img:tag" and snap["disk"] == 60
    assert snap["runtype"] == "train" and snap["runset"] == "rs1"
    assert snap["rescue_wait_s"] == 600 and snap["backoff"] == "120s*2^n cap 30m"
    assert st["run_id"] == "r-test" and hf["phase"] == "IDLE"
    assert handoff_on is True and "reconcile" in init_seams


@pytest.mark.parametrize("kw,expected", [
    ({}, True),                                   # handoff is the DEFAULT
    ({"handoff": False}, False),                  # --no-handoff
    ({"strict_ceiling": True}, False),            # --strict-ceiling wins
    ({"handoff": True, "strict_ceiling": True}, False),
])
def test_strict_ceiling_beats_the_handoff_default(init_seams, emits, kw, expected):
    handoff_on = run_lane.supervise_init(_ns(**kw))[2]
    assert bool(handoff_on) is expected
    assert ("reconcile" in init_seams) is expected, \
        "a crashed-mid-flight twin is only adopted when handoff is armed"
