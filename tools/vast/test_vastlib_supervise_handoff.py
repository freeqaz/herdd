"""`vastlib.supervise.handoff` — the migration ladder's port-time regression net.

Why this file exists
--------------------
The handoff cluster is the only code in `tools/vast` that deliberately runs two
rented boxes at once and then destroys one of them, and at the rev this port was
cut from its coverage was almost entirely INDIRECT: `test_supervise.py` drives
the flat `herdd` copies through the supervise tick, and exactly ONE symbol in
the cluster (`_prefence_bid`, `test_handoff_drain_abort.py:283-285`) had a
direct-import test. That flat suite stays UNEDITED under the plan §8 add-only
amendment — it keeps steering `herdd.py`. This file is the parallel net for
the `vastlib` copy, and it is the first direct coverage the other 35 symbols in
the cluster have ever had.

What is asserted, and why those things
--------------------------------------
* **The frozen 35-key `hs` contract.** `bidpolicy.mk_handoff_state` is Zone S
  and unported; `handoff_poll` / `_handoff_fence_hold` / `_handoff_candidate_ok`
  read its result BY KEY. Both builders are asserted to emit a plain dict with
  exactly those 35 keys, passed BY KEYWORD, and the keyword sets are compared
  against `herdd.py`'s own — by `ast`, not by import — so a port that dropped
  or renamed one shows up as a set difference, not as a wrong decision three
  phases later.
* **The six pinned lane divergences** (plan §5 NOTE, v1 §7). The id guard is RAW
  `==` on the run lane and `str()==` on the jobs lane; the run builder passes 29
  kwargs and takes `driver_can_complete=True` by default while the jobs builder
  passes all 35 and fails CLOSED; only the jobs lane suppresses
  `understudy_producing` behind `retarget_incomplete`; only the jobs lane
  unwinds an OPEN fence on the exit path. Each is pinned here as behavior, so a
  future "cleanup" that unifies a money path fails a test instead of shipping.
* **In-place mutation.** `_handoff_reset` / `_job_handoff_reset` do
  `hf.clear()` + `hf.update(...)` because callers — including `fleetd`'s
  `rt["hf"]`, across a process boundary — hold that exact object across ticks.
  The tests assert on the identity of the mapping, not just its contents; a
  rebind would pass a contents-only assertion and silently stop resetting.
* **Tri-state `None` floats stay `None`.** `remaining_wall_h` with no wall
  budget, `min_running_eta_s`, and `_prefence_bid`'s "unknown" are `None` for
  UNKNOWN, never `0.0` (defect #67). `0.0` there reads as a known-zero price.
* **`_handoff_unfence_primary` never resumes at the fence pin.** A box resumed
  at `HANDOFF_PARK_BID` is a live rental that cannot win its market — the exact
  wedge the unwind exists to prevent. Both the `prefence_bid == pin` belt and
  the no-target refusal are pinned.

Isolation
---------
Nothing here can reach the network, B2, rclone or the vast API.
`journal._sup_emit` / `_job_handoff_emit` / `_job_handoff_journal`, every
`lifecycle` PUT/DELETE, `b2._b2_rcat` / `b2._rclone_soft`,
`launch_spec._raw_events_soft` and `jobmeta.read_job` are replaced by module
attribute — which is also the late-binding proof the 28 `_job_handoff_emit`
monkeypatch sites in the flat suite depend on. `monkeypatch.setattr` is used
WITHOUT `raising=False` throughout, so a seam that moves to another module fails
loudly here instead of going vacuous. `conftest.py`'s autouse guard refuses any
mutating `request_soft` that somehow escapes.

`herdd` is never imported: the parity assertions read its SOURCE with `ast`.
Importing it would execute a 17k-line module (and its `.env` discovery) for
three keyword lists.

What is deliberately NOT here
-----------------------------
* No repointing of `test_handoff_drain_abort.py`, `test_supervise.py` or
  `test_fleetd.py`. They stay UNEDITED and keep steering the flat copies; they
  migrate via the plan §7.1 rename table at steps 6-7.
* No re-testing of `bidpolicy.handoff_poll`'s state machine. Every decision in
  this cluster is that function's, `test_handoff_*.py` own it, and a second copy
  of its expectations here would be a fork of the pure core's tests.
* No assertion that the two lanes agree. They deliberately diverge.

Provenance: created 2026-08-16 alongside `vastlib/supervise/handoff.py`,
plan §8 step 4. Manifest: `tools/vast/.port_manifests/sup-handoff.json`.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Any

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

import bidpolicy                                       # noqa: E402  Zone S
import jobmeta                                         # noqa: E402  Zone S

from vastlib.boxes import health, lifecycle            # noqa: E402
from vastlib.launch import spec as launch_spec         # noqa: E402
from vastlib.storage import b2                         # noqa: E402
from vastlib.supervise import handoff, journal         # noqa: E402

NOW = 1_770_000_000.0


# --------------------------------------------------------------------------- #
# fixtures — every seam patched by module attribute
# --------------------------------------------------------------------------- #
@pytest.fixture
def emits(monkeypatch):
    """Capture `journal._sup_emit` (run lane) in order."""
    seen: list[tuple[Any, str, dict]] = []
    monkeypatch.setattr(journal, "_sup_emit",
                        lambda rid, ev, **kw: seen.append((rid, ev, kw)) or {})
    return seen


@pytest.fixture
def jemits(monkeypatch):
    """Capture `journal._job_handoff_emit` (jobs lane B2 telemetry) in order."""
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(journal, "_job_handoff_emit",
                        lambda jctx, ev, **kw: seen.append((ev, kw)) or {})
    return seen


@pytest.fixture
def jjournal(monkeypatch):
    """Capture `journal._job_handoff_journal` (the fleetd-drained decision queue)."""
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(journal, "_job_handoff_journal",
                        lambda jctx, kind, **kw: seen.append((kind, kw)))
    return seen


@pytest.fixture
def boxops(monkeypatch):
    """Refuse every real box mutation; record the calls instead."""
    calls: list[tuple] = []
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda iid, state: calls.append(("put", iid, state)) or (True, None))
    monkeypatch.setattr(lifecycle, "_put_bid_soft",
                        lambda iid, price: calls.append(("bid", iid, price)) or (True, None))
    monkeypatch.setattr(lifecycle, "_wait_states_soft",
                        lambda iid, targets, timeout, **kw:
                        calls.append(("wait", iid, sorted(targets))) or (True, "stopped"))
    monkeypatch.setattr(lifecycle, "_destroy_soft",
                        lambda iid, *a, **k: calls.append(("destroy", iid)) or (True, None))
    return calls


@pytest.fixture
def no_rclone(monkeypatch):
    """Neither B2 helper may shell out. Every call is recorded and refused."""
    calls: list[tuple] = []
    monkeypatch.setattr(b2, "_b2_rcat",
                        lambda path, body, hard=True:
                        calls.append(("rcat", path, body, hard)) or True)
    monkeypatch.setattr(b2, "_rclone_soft",
                        lambda args: calls.append(("rclone", tuple(args))) or (1, "", "no"))
    return calls


def _hf(**kw) -> dict[str, Any]:
    """A run-lane handoff sub-state, factory-fresh plus overrides."""
    hf = handoff._init_handoff_state()
    hf.update(kw)
    return hf


def _jhf(**kw) -> dict[str, Any]:
    """A jobs-lane handoff sub-state, factory-fresh plus overrides."""
    hf = handoff._init_job_handoff_state()
    hf.update(kw)
    return hf


def _st(**kw) -> dict[str, Any]:
    base = {"run_id": "r-test", "instance_id": 700, "husk_id": 700,
            "last_bid": 0.20, "dph_total": 0.25, "on_demand": 1.0,
            "spend_usd": 0.0, "budget_usd": None, "dt": 0.0, "now": NOW,
            "wall_budget_s": None, "wall_clock_s": 0.0, "evicted_pending": False,
            "not_live_streak": 0, "bid_history": [], "_instances": []}
    base.update(kw)
    return base


def _jctx(**kw) -> dict[str, Any]:
    base = {"iid": "700", "last_bid": 0.20, "dph": 0.25, "on_demand": 1.0,
            "spend_usd": 0.0, "budget_usd": None, "dt": 0.0, "now": NOW,
            "dry_run": False, "instances": [], "pending_jobs": [],
            "running_jobs": [], "remaining_wall_h": 2.0}
    base.update(kw)
    return base


def _ns(**kw) -> argparse.Namespace:
    base = dict(run_id="r-test", dry_run=False, handoff=True, max_bid=None,
                budget=None, wall_budget=None, gpu=None, gpu_ram=None,
                cuda=None, num_gpus=1, price=None)
    base.update(kw)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------- #
# ast helpers — parity against the flat file WITHOUT importing it
# --------------------------------------------------------------------------- #
def _funcs(path: str) -> dict[str, ast.FunctionDef]:
    tree = ast.parse((VAST_DIR / path).read_text())
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def _call_kwargs(fn: ast.FunctionDef, callee: str) -> list[str]:
    """Keyword names of the first call to `callee` inside `fn` (attribute or bare)."""
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name == callee:
            assert not node.args, f"{callee} must be called by KEYWORD only"
            return [k.arg for k in node.keywords if k.arg]
    raise AssertionError(f"no call to {callee} in {fn.name}")


#: The frozen key contract, read from Zone S's own signature rather than
#: transcribed — a transcription would drift the moment bidpolicy changed, and
#: silently agree with a port that dropped the same key.
MK_HANDOFF_KEYS = frozenset(
    p.arg for p in _funcs("bidpolicy.py")["mk_handoff_state"].args.kwonlyargs)


# --------------------------------------------------------------------------- #
# 1. PURE — prices and pins
# --------------------------------------------------------------------------- #
def test_prefence_bid_three_case_table() -> None:
    """The exact table `test_handoff_drain_abort.py:283-285` pins on the flat copy.

    Case 1 is the 2026-08-08 task-#62 belt: a supervisor that restarts mid-fence
    observes the primary's `dph_total` as the $0.001 PIN, and recording that as
    "the bid to restore" would unwind the box to a price it can never win a
    market with — the wedge the unwind exists to prevent, laundered through a
    restart. `None` means UNKNOWN and the policy-target fallback owns it."""
    pin = bidpolicy.HANDOFF_PARK_BID
    assert handoff._prefence_bid(pin, pin) is None
    assert handoff._prefence_bid(None, None) is None
    assert handoff._prefence_bid(2.55, 0.001) == 2.55


def test_prefence_bid_prefers_the_standing_bid_over_observed_dph() -> None:
    assert handoff._prefence_bid(0.30, 0.42) == 0.30
    assert handoff._prefence_bid(None, 0.42) == 0.42


def test_handoff_park_bid_is_the_api_minimum_and_ignores_its_argument() -> None:
    """Deliberately NOT floor-relative: a floor DROP inside the fence->drain
    window would leave a "parked" primary winnable again. Both lanes call it —
    the run lane with `st`, the jobs lane with `jctx` — and it reads neither."""
    assert handoff._handoff_park_bid(_st()) == bidpolicy.HANDOFF_PARK_BID
    assert handoff._handoff_park_bid(_jctx()) == bidpolicy.HANDOFF_PARK_BID
    assert handoff._handoff_park_bid({}) == bidpolicy.HANDOFF_PARK_BID


def test_handoff_primary_dph_prefers_last_bid_then_dph_total_then_none() -> None:
    assert handoff._handoff_primary_dph(_st(last_bid=0.2, dph_total=0.9)) == 0.2
    assert handoff._handoff_primary_dph(_st(last_bid=None, dph_total=0.9)) == 0.9
    assert handoff._handoff_primary_dph(_st(last_bid=None, dph_total=None)) is None


# --------------------------------------------------------------------------- #
# 2. PURE MUTATORS — the double-bill accruers (both lanes)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("phase", [None, "IDLE", "DONE"])
def test_accrue_early_outs_on_the_three_idle_phases(phase) -> None:
    """No second box is running in these phases, so nothing may be double-billed."""
    st, hf = _st(dt=60.0), _hf(phase=phase, understudy_iid=999,
                               understudy_status="running", understudy_dph=1.0)
    handoff._handoff_accrue(st, hf)
    assert st["spend_usd"] == 0.0 and hf["handoff_spend_usd"] == 0.0

    jctx, jhf = _jctx(dt=60.0), _jhf(phase=phase, understudy_iid="999",
                                     understudy_status="running", understudy_dph=1.0)
    handoff._handoff_job_accrue(jctx, jhf)
    assert jctx["spend_usd"] == 0.0 and jhf["handoff_spend_usd"] == 0.0


def test_accrue_early_outs_when_the_understudy_is_not_live() -> None:
    """`LIVE_STATES` membership: a `created`/`loading`/`running` box burns money,
    an `exited` one does not."""
    st, hf = _st(dt=3600.0), _hf(phase="WARMING", understudy_iid=999,
                                 understudy_status="exited", understudy_dph=1.0)
    handoff._handoff_accrue(st, hf)
    assert st["spend_usd"] == 0.0


def test_accrue_adds_the_same_amount_to_both_counters() -> None:
    """`spend_usd` feeds the budget guard, `handoff_spend_usd` the abort rule and
    the cost event. One burn, two ledgers."""
    st, hf = _st(dt=3600.0, spend_usd=5.0), _hf(phase="WARMING", understudy_iid=999,
                                                understudy_status="running",
                                                understudy_dph=0.50)
    handoff._handoff_accrue(st, hf)
    assert st["spend_usd"] == pytest.approx(5.50)
    assert hf["handoff_spend_usd"] == pytest.approx(0.50)


def test_the_id_guard_divergence_is_pinned_raw_on_the_run_lane() -> None:
    """LANE MIRRORING (pinned). The run lane compares `st['instance_id']` to
    `hf['understudy_iid']` RAW, so an int primary and a str understudy id are
    DIFFERENT boxes to it and the window keeps billing. The jobs lane spells
    every box id as a string and compares `str()==`, so the same pair is the
    SAME box and billing stops (the primary loop owns it from there).

    Do not unify: one of these is a double-bill and the other is a missed bill,
    and which is which depends on the lane's id spelling. A money-path
    unification is its own owner-called change."""
    st, hf = _st(instance_id=700, dt=3600.0), _hf(phase="CUTOVER", understudy_iid="700",
                                                  understudy_status="running",
                                                  understudy_dph=0.50)
    handoff._handoff_accrue(st, hf)
    assert st["spend_usd"] == pytest.approx(0.50)     # raw ==  -> not the same box

    jctx, jhf = _jctx(iid=700, dt=3600.0), _jhf(phase="CUTOVER", understudy_iid="700",
                                                understudy_status="running",
                                                understudy_dph=0.50)
    handoff._handoff_job_accrue(jctx, jhf)
    assert jctx["spend_usd"] == 0.0                   # str() == -> the same box


# --------------------------------------------------------------------------- #
# 3. THE FROZEN 35-KEY hs CONTRACT
# --------------------------------------------------------------------------- #
def test_mk_handoff_state_signature_is_the_35_key_contract() -> None:
    """Guards the guard: every assertion below is relative to this set, so if
    Zone S grows a 36th key the parity tests must be re-read, not silently pass."""
    assert len(MK_HANDOFF_KEYS) == 35
    for k in ("phase", "driver_can_complete", "unsafe_override", "now"):
        assert k in MK_HANDOFF_KEYS


#: Assigned onto the built dict rather than passed to the factory, because the
#: factory's key set IS the frozen contract above and `HandoffSnapshot` pins it
#: exactly. Zone S reads it with `.get()`: absent => the legacy poll COUNT.
_HS_ASSIGNED = frozenset({"over_ceiling_since"})


def test_run_builder_returns_a_plain_dict_with_exactly_those_keys() -> None:
    hs = handoff._handoff_build_state(_st(), _ns(), _hf(), bidpolicy.Action("noop", "ok"))
    assert type(hs) is dict
    assert set(hs) == set(MK_HANDOFF_KEYS) | _HS_ASSIGNED


def test_job_builder_returns_a_plain_dict_with_exactly_those_keys() -> None:
    hs = handoff._handoff_job_build_state(_jctx(), _jhf())
    assert type(hs) is dict
    assert set(hs) == set(MK_HANDOFF_KEYS) | _HS_ASSIGNED


def test_the_dwell_clock_reaches_the_pure_core_from_both_builders() -> None:
    """...and it is not decoration: the value the lane recorded has to arrive on
    the snapshot `handoff_poll` reads, or the dwell silently stays a count."""
    for hs in (handoff._handoff_build_state(
                   _st(), _ns(), dict(_hf(), over_ceiling_since=1234.0),
                   bidpolicy.Action("noop", "ok")),
               handoff._handoff_job_build_state(
                   _jctx(), dict(_jhf(), over_ceiling_since=1234.0))):
        assert hs["over_ceiling_since"] == 1234.0


def test_both_builders_pass_exactly_the_keywords_the_lanes_diverge_on() -> None:
    """The keyword sets each builder hands `mk_handoff_state`, read from SOURCE
    rather than by importing (a dropped keyword is invisible at runtime —
    `mk_handoff_state` has a default for every one of the 35 — and only shows up
    as a wrong migration decision several phases later).

    Was a parity read against `herdd.py`'s copies until step 6d, when the
    launcher stopped carrying bodies and `_funcs("herdd.py")` began raising
    `KeyError`. The comparison it made — 29 for the run lane, 35 for the jobs
    lane, every name inside `MK_HANDOFF_KEYS` — is asserted directly on the
    ported source, which is what the flat numbers were standing in for. The run
    lane passing 29 and the jobs lane 35 is the pinned divergence.
    """
    port = _funcs("vastlib/supervise/handoff.py")
    for name, expected in (("_handoff_build_state", 29),
                           ("_handoff_job_build_state", 35)):
        port_kw = _call_kwargs(port[name], "mk_handoff_state")
        assert len(port_kw) == expected, name
        assert set(port_kw) <= set(MK_HANDOFF_KEYS), name
    run_kw = _call_kwargs(port["_handoff_build_state"], "mk_handoff_state")
    job_kw = _call_kwargs(port["_handoff_job_build_state"], "mk_handoff_state")
    assert set(run_kw) < set(job_kw), (
        "the run lane's keywords must stay a strict subset of the jobs lane's — "
        "that containment is the divergence this test pins")


def test_the_run_lane_takes_driver_can_complete_true_by_default() -> None:
    """The six work-awareness keys are the jobs lane's. The run lane omits them
    and inherits `mk_handoff_state`'s defaults, where `driver_can_complete` is
    True BY DESIGN — a run watch that survives its own primary is the run lane's
    normal shape."""
    hs = handoff._handoff_build_state(_st(), _ns(), _hf(), bidpolicy.Action("noop", "ok"))
    assert hs["driver_can_complete"] is True
    assert hs["running_unresumable"] == 0
    assert hs["min_running_eta_s"] is None            # tri-state: UNKNOWN, not 0.0


def test_the_jobs_lane_fails_closed_on_driver_can_complete() -> None:
    """defect #61: a driver that has not DECLARED it can carry a migration to
    `complete` cannot arm one. Absence of the key is a refusal, not a default."""
    assert handoff._handoff_job_build_state(_jctx(), _jhf())["driver_can_complete"] is False
    jctx = _jctx(handoff_can_complete=True)
    assert handoff._handoff_job_build_state(jctx, _jhf())["driver_can_complete"] is True


def test_only_the_jobs_lane_suppresses_producing_while_a_retarget_is_stuck() -> None:
    """While a retarget delete is stuck the old ticket still names the primary, so
    DRAINING must never destroy it. The run lane has no such latch — it relabels
    a box instead of moving tickets — and passes `understudy_producing` straight
    through."""
    jhf = _jhf(understudy_producing=True, retarget_incomplete=["j-1"])
    assert handoff._handoff_job_build_state(_jctx(), jhf)["understudy_producing"] is False
    jhf["retarget_incomplete"] = None
    assert handoff._handoff_job_build_state(_jctx(), jhf)["understudy_producing"] is True

    hf = _hf(understudy_producing=True)
    hs = handoff._handoff_build_state(_st(), _ns(), hf, bidpolicy.Action("noop", "ok"))
    assert hs["understudy_producing"] is True


def test_run_builder_ors_the_polls_own_eviction_verdict_into_primary_evicted() -> None:
    """`primary_evicted` is not just the observed flag: poll()'s verdict THIS tick
    (`emit_evicted` / `relaunch`) is what the fast-cutover and abort rules key
    off, and it is known before `st['evicted_pending']` is."""
    for kind, expected in (("noop", False), ("emit_evicted", True), ("relaunch", True)):
        hs = handoff._handoff_build_state(_st(), _ns(), _hf(),
                                          bidpolicy.Action(kind, "x"))
        assert hs["primary_evicted"] is expected, kind
    hs = handoff._handoff_build_state(_st(evicted_pending=True), _ns(), _hf(),
                                      bidpolicy.Action("noop", "ok"))
    assert hs["primary_evicted"] is True


def test_run_builder_falls_back_to_st_instance_id_for_primary_iid() -> None:
    assert handoff._handoff_build_state(
        _st(instance_id=700), _ns(), _hf(), bidpolicy.Action("noop", "ok")
    )["primary_iid"] == 700
    assert handoff._handoff_build_state(
        _st(instance_id=700), _ns(), _hf(primary_iid=42), bidpolicy.Action("noop", "ok")
    )["primary_iid"] == 42


# --------------------------------------------------------------------------- #
# 4. THE IN-PLACE RESETS
# --------------------------------------------------------------------------- #
def test_handoff_reset_mutates_in_place_and_preserves_only_two_counters() -> None:
    """`fleetd` holds this exact mapping as `rt["hf"]` across ticks and reads
    `phase` / `understudy_iid` off it (defect #61 keep-alive predicate). A rebind
    would leave every holder — including another process's view — looking at the
    pre-reset object. The identity assertion is the point of the test."""
    hf = _hf(phase="DRAINING", understudy_iid=999, handoff_spend_usd=3.5,
             stall_alarmed=True, handoffs_done=1, cooldown_until=1.0)
    before = id(hf)
    handoff._handoff_reset(hf, handoffs_done=2, cooldown_until=NOW + 1800)
    assert id(hf) == before
    assert hf["phase"] == "IDLE"
    assert hf["handoffs_done"] == 2 and hf["cooldown_until"] == NOW + 1800
    assert hf["understudy_iid"] is None and hf["handoff_spend_usd"] == 0.0
    assert hf["stall_alarmed"] is False               # the once-latch must re-arm


def test_reset_drops_keys_written_by_other_modules() -> None:
    """`prefence_bid` is written by the fence in this module, `refuse_sig` /
    `defer_sig` / `pct_warned` by the jobs-lane refusal paths. `clear()` is what
    makes the reset total; a key-by-key update would leak a stale fence bid into
    the NEXT migration's unwind."""
    hf = _hf(prefence_bid=2.55, refuse_sig="checkpoint_stale")
    handoff._handoff_reset(hf, handoffs_done=0, cooldown_until=0.0)
    assert "prefence_bid" not in hf and "refuse_sig" not in hf


def test_job_reset_restores_the_three_jobs_only_keys() -> None:
    """The jobs factory seeds `pending_jobs` / `running_jobs` /
    `retarget_incomplete`; resetting from the RUN factory would delete them and
    the next `_do_job_handoff_move` would KeyError on `hf["pending_jobs"]`."""
    hf = _jhf(phase="CUTOVER", pending_jobs=["j-1"], running_jobs=["j-1"],
              retarget_incomplete=["j-1"])
    before = id(hf)
    handoff._job_handoff_reset(hf, handoffs_done=1, cooldown_until=NOW)
    assert id(hf) == before
    assert hf["pending_jobs"] == [] and hf["running_jobs"] == []
    assert hf["retarget_incomplete"] is None
    assert hf["handoffs_done"] == 1 and hf["phase"] == "IDLE"


def test_the_two_factories_differ_by_exactly_the_three_jobs_only_keys() -> None:
    assert set(handoff._init_job_handoff_state()) - set(handoff._init_handoff_state()) == {
        "pending_jobs", "running_jobs", "retarget_incomplete"}
    assert set(handoff._init_handoff_state()) - set(handoff._init_job_handoff_state()) == set()


def test_the_two_frozen_cross_process_keys_exist_from_construction() -> None:
    """`fleetd.Fleet._handoff_in_flight` does `rt["hf"].get("phase")` /
    `.get("understudy_iid")` across a process boundary while `fleetd.py` is
    unported (plan §8 step 5). Both must be present on a factory-fresh mapping
    AND after a reset — a `.get()` returning None on a missing key would read as
    "no migration in flight" and let the keep-alive lapse mid-fence."""
    for factory in (handoff._init_handoff_state, handoff._init_job_handoff_state):
        hf = factory()
        assert hf["phase"] == "IDLE" and hf["understudy_iid"] is None


# --------------------------------------------------------------------------- #
# 5. THE SHARED STALL ALARM (both lanes inject their own emit)
# --------------------------------------------------------------------------- #
def test_stall_alarm_fires_once_and_latches() -> None:
    """F2 observability, NOT a forced transition: the primary destroy stays gated
    on understudy proof-of-life (the byte-safety invariant), so a wedged
    migration is a loud alert and nothing more."""
    seen: list[dict] = []
    hf = _hf(phase="DRAINING", fence_ts=NOW, understudy_iid=999)
    late = NOW + bidpolicy.HANDOFF_DEADLINE_S + 1
    handoff._handoff_stall_alarm(hf, late, lambda **f: seen.append(f))
    assert len(seen) == 1 and hf["stall_alarmed"] is True
    assert seen[0]["phase"] == "DRAINING" and seen[0]["understudy"] == 999
    handoff._handoff_stall_alarm(hf, late + 600, lambda **f: seen.append(f))
    assert len(seen) == 1                              # latched
    assert hf["phase"] == "DRAINING"                   # and it forced nothing


def test_stall_alarm_is_silent_before_the_deadline() -> None:
    seen: list[dict] = []
    hf = _hf(phase="CUTOVER", fence_ts=NOW)
    handoff._handoff_stall_alarm(hf, NOW + bidpolicy.HANDOFF_DEADLINE_S - 1,
                                 lambda **f: seen.append(f))
    assert seen == [] and hf["stall_alarmed"] is False


@pytest.mark.parametrize("phase", ["IDLE", "ARMED", "LAUNCHING", "WARMING", "SYNCED"])
def test_stall_alarm_only_watches_the_fence_open_phases(phase) -> None:
    """CUTOVER is included on purpose (2026-07-18 review S4): it normally exits at
    HANDOFF_FENCE_TIMEOUT_S, so a CUTOVER past the deadline is already wedged."""
    seen: list[dict] = []
    handoff._handoff_stall_alarm(_hf(phase=phase, fence_ts=NOW), NOW + 10 ** 6,
                                 lambda **f: seen.append(f))
    assert seen == []


def test_stall_alarm_needs_a_fence_ts() -> None:
    seen: list[dict] = []
    handoff._handoff_stall_alarm(_hf(phase="DRAINING", fence_ts=None), NOW + 10 ** 6,
                                 lambda **f: seen.append(f))
    assert seen == []


# --------------------------------------------------------------------------- #
# 6. REFUSAL PROSE (jobs lane) + the module-exec interpolation
# --------------------------------------------------------------------------- #
def test_every_pure_core_refusal_reason_has_operator_prose() -> None:
    assert set(handoff._HANDOFF_REFUSAL_NOTES) == {
        "driver_cannot_complete", "unresumable_running_job",
        "no_resumable_checkpoint", "checkpoint_stale"}
    for reason in handoff._HANDOFF_REFUSAL_NOTES:
        assert handoff._job_handoff_refusal_note(reason) != reason


def test_the_checkpoint_stale_note_interpolates_the_zone_s_constant() -> None:
    """`_HANDOFF_REFUSAL_NOTES` is built at MODULE-EXEC time from
    `bidpolicy.HANDOFF_CKPT_FRESH_MULT`. That is why the `import bidpolicy` at the
    top of `handoff.py` must stay EAGER: making it lazy changes the text of a
    shipped operator message (or raises at import)."""
    note = handoff._job_handoff_refusal_note("checkpoint_stale")
    assert f"{bidpolicy.HANDOFF_CKPT_FRESH_MULT}x" in note


def test_eta_reasons_are_formatted_not_looked_up() -> None:
    note = handoff._job_handoff_refusal_note("eta_240s")
    assert "240s" in note
    assert f"{bidpolicy.HANDOFF_FENCE_HOLD_ETA_S}s" in note
    assert "eta_240s" not in handoff._HANDOFF_REFUSAL_NOTES


def test_an_unknown_reason_passes_through_verbatim() -> None:
    """A new pure-core reason must still be SAYABLE — an unmapped reason prints
    itself rather than vanishing into a KeyError inside the refusal path."""
    assert handoff._job_handoff_refusal_note("something_new") == "something_new"


# --------------------------------------------------------------------------- #
# 7. THE TWO CROSS-LANE CONSTANTS
# --------------------------------------------------------------------------- #
def test_fence_open_phases_are_the_two_post_fence_ones() -> None:
    """`run_lane.supervise_tick` reads this to suppress poll()'s primary churn
    while the fence is open, so the untouched bid ladder cannot fight the
    deliberate retirement of the primary."""
    assert handoff._HANDOFF_FENCE_OPEN == ("CUTOVER", "DRAINING")
    assert not set(handoff._HANDOFF_FENCE_OPEN) & set(bidpolicy._HANDOFF_PRE_CUTOVER)


def test_job_producing_event_set_matches_the_lifecycle_vocabulary() -> None:
    assert handoff._HANDOFF_JOB_PRODUCING == (
        "checkpoint", "started", "resumed", "claimed", "done")


# --------------------------------------------------------------------------- #
# 8. THE B2 MARKER WIRE CONTRACT (producer side of two Zone S shell guards)
# --------------------------------------------------------------------------- #
def test_marker_paths_are_the_frozen_prefixes_the_box_side_guards_read(
        monkeypatch, no_rclone) -> None:
    """`onstart/train.sh:286` and `onstart/jobd.sh:221` (`_handoff_epoch_stale`)
    list `runs/<RUN_ID>/handoff/` and `jobs/<JOB_ID>/handoff/` on a rented box.
    Changing a prefix here silently disables a guard running on a machine this
    process cannot see, so the paths are asserted literally."""
    monkeypatch.setenv("B2_BUCKET", "bkt")
    assert handoff._handoff_b2_write("r-1", "3.json", "{}\n") is True
    assert handoff._handoff_job_b2_write("j-1", "promoted", "{}\n") is True
    assert [c[1] for c in no_rclone] == ["b2:bkt/runs/r-1/handoff/3.json",
                                         "b2:bkt/jobs/j-1/handoff/promoted"]
    assert all(c[3] is False for c in no_rclone)       # hard=False: never fatal


def test_marker_writes_are_noops_without_a_bucket_or_under_dry_run(
        monkeypatch, no_rclone) -> None:
    monkeypatch.delenv("B2_BUCKET", raising=False)
    assert handoff._handoff_b2_write("r-1", "1.json", "x") is False
    assert handoff._handoff_job_b2_write("j-1", "1.json", "x") is False
    monkeypatch.setenv("B2_BUCKET", "bkt")
    assert handoff._handoff_b2_write("r-1", "1.json", "x", dry_run=True) is False
    assert handoff._handoff_job_b2_write("j-1", "1.json", "x", dry_run=True) is False
    assert no_rclone == []


def test_synced_epoch_takes_the_greatest_marker_and_ignores_the_rest(
        monkeypatch) -> None:
    """The SYNCED gate keys on THIS, never on API liveness: `loading` counts as a
    LIVE_STATE, so the old liveness proxy stamped SYNCED 48s after launch against
    a box with zero checkpoints staged and fenced the primary into nothing (live
    canary handoff-canary-2, 2026-07-15)."""
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(b2, "_rclone_soft",
                        lambda args: (0, "1.synced 3.synced 2.synced "
                                         "promoted junk.synced 4.json", ""))
    assert handoff._handoff_synced_epoch_soft("r-1") == 3


def test_synced_epoch_is_none_on_no_bucket_no_marker_or_a_failed_read(
        monkeypatch) -> None:
    """Fail-closed: no proof, no SYNCED, no fence."""
    monkeypatch.delenv("B2_BUCKET", raising=False)
    assert handoff._handoff_synced_epoch_soft("r-1") is None
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(b2, "_rclone_soft", lambda args: (1, "", "boom"))
    assert handoff._handoff_synced_epoch_soft("r-1") is None
    monkeypatch.setattr(b2, "_rclone_soft", lambda args: (0, "promoted", ""))
    assert handoff._handoff_synced_epoch_soft("r-1") is None


# --------------------------------------------------------------------------- #
# 9. RUN-LANE FENCE SIGNALS
# --------------------------------------------------------------------------- #
def _events(monkeypatch, evs) -> None:
    monkeypatch.setattr(launch_spec, "_raw_events_soft", lambda rid: evs)


def test_producing_needs_a_checkpoint_AFTER_the_final_flush(monkeypatch) -> None:
    _events(monkeypatch, [{"event": "checkpoint", "ts": "20260101T000000"},
                          {"event": "final_flush", "ts": "20260101T000100"}])
    assert handoff._handoff_run_signals("r-1") == {"final_flush_seen": True,
                                                   "understudy_producing": False}
    _events(monkeypatch, [{"event": "final_flush", "ts": "20260101T000100"},
                          {"event": "checkpoint", "ts": "20260101T000200"}])
    assert handoff._handoff_run_signals("r-1")["understudy_producing"] is True


def test_a_post_cutover_checkpoint_alone_proves_producing(monkeypatch) -> None:
    """Vast delivers NO SIGTERM on a fence-park (proven live twice, 2026-07-15), so
    the `flush_timeout` cutover is the NORM: no `final_flush` event ever lands and
    a flush-gated producing signal stayed False forever — every real handoff
    wedged in DRAINING. A checkpoint later than the promotion moment can only be
    the understudy's."""
    _events(monkeypatch, [{"event": "checkpoint", "ts": "20260101T000200"}])
    assert handoff._handoff_run_signals("r-1")["final_flush_seen"] is False
    assert handoff._handoff_run_signals(
        "r-1", cutover_ts="20260101T000100")["understudy_producing"] is True
    assert handoff._handoff_run_signals(
        "r-1", cutover_ts="20260101T000300")["understudy_producing"] is False


def test_signals_never_raise_through_a_broken_event_read(monkeypatch) -> None:
    def boom(rid):
        raise OSError("cache gone")
    monkeypatch.setattr(launch_spec, "_raw_events_soft", boom)
    assert handoff._handoff_run_signals("r-1") == {"final_flush_seen": False,
                                                   "understudy_producing": False}


def test_observe_understudy_adopts_the_twin_and_gates_synced_on_the_marker(
        monkeypatch) -> None:
    """Liveness is NOT the SYNCED proof — `_handoff_synced_epoch_soft` is, and it
    must be for THIS attempt's epoch or greater."""
    monkeypatch.setattr(handoff, "_handoff_synced_epoch_soft", lambda rid: None)
    inst = {"id": 900, "label": "run:r-test:handoff", "actual_status": "Running",
            "dph_total": 0.44}
    st = _st(_instances=[inst])
    hf = _hf(phase="LAUNCHING", epoch=1)
    handoff._handoff_observe_understudy(st, hf)
    assert hf["understudy_iid"] == 900 and hf["understudy_status"] == "running"
    assert hf["understudy_dph"] == 0.44 and hf["understudy_live_since"] == NOW
    assert hf["ckpt_pulled_epoch"] is None            # no marker -> no SYNCED

    monkeypatch.setattr(handoff, "_handoff_synced_epoch_soft", lambda rid: 0)
    handoff._handoff_observe_understudy(st, hf)
    assert hf["ckpt_pulled_epoch"] is None            # marker is for an older epoch
    monkeypatch.setattr(handoff, "_handoff_synced_epoch_soft", lambda rid: 1)
    handoff._handoff_observe_understudy(st, hf)
    assert hf["ckpt_pulled_epoch"] == 1


def test_observe_understudy_treats_a_missing_twin_as_transient(monkeypatch) -> None:
    """A transient absence from one listing is not a reap: the run lane leaves the
    last known status alone (the jobs lane's mirror is the one that latches
    `understudy_gone`, and only on a NON-empty listing)."""
    hf = _hf(phase="WARMING", understudy_iid=900, understudy_status="running")
    handoff._handoff_observe_understudy(_st(_instances=[]), hf)
    assert hf["understudy_status"] == "running"


# --------------------------------------------------------------------------- #
# 10. RUN-LANE TERMINALS
# --------------------------------------------------------------------------- #
def test_complete_promotes_the_understudy_and_resets_the_ladder(emits) -> None:
    """The promoted box becomes st's tracked box so the untouched poll() ladder
    supervises it from here. `last_bid` takes the promoted box's `dph_base` or
    NOTHING: the old `or understudy_dph` fallback wrote bid+storage into
    `last_bid`, permanently one storage sliver above every number vast echoes
    back, so the standing arm could never match the promoted box (review
    2026-08-10 #7). A None there is fail-closed — bid moves stay disabled until
    `_observe` re-seeds."""
    st = _st(_instances=[{"id": 900, "is_bid": True, "dph_base": 0.11,
                          "dph_total": 0.14}], evicted_pending=True)
    hf = _hf(phase="DRAINING", understudy_iid=900, understudy_dph=0.14,
             handoff_spend_usd=1.25, handoffs_done=0)
    handoff._handoff_complete(st, _ns(), hf)
    assert st["instance_id"] == 900 and st["husk_id"] == 900
    assert st["dph_total"] == 0.14 and st["last_bid"] == 0.11
    assert st["evicted_pending"] is False and st["bid_history"] == []
    assert hf["phase"] == "IDLE" and hf["handoffs_done"] == 1
    assert hf["cooldown_until"] == NOW + bidpolicy.HANDOFF_COOLDOWN_S
    assert [e[1] for e in emits] == ["handoff_complete"]
    assert emits[0][2]["handoff_spend_usd"] == 1.25


def test_complete_leaves_last_bid_none_when_the_body_has_no_dph_base(emits) -> None:
    st = _st(_instances=[{"id": 900, "is_bid": True, "dph_total": 0.14}])
    handoff._handoff_complete(st, _ns(), _hf(phase="DRAINING", understudy_iid=900,
                                             understudy_dph=0.14))
    assert st["last_bid"] is None                      # fail-closed, never bid+storage


def test_abort_reaps_the_understudy_without_bumping_handoffs_done(
        emits, boxops, monkeypatch) -> None:
    monkeypatch.setattr(handoff, "_confirm_gone", lambda iid, tries=6: True)
    hf = _hf(phase="WARMING", understudy_iid=900, handoffs_done=2)
    handoff._handoff_abort(_st(), _ns(), hf, "deadline")
    assert ("destroy", 900) in boxops
    assert hf["handoffs_done"] == 2                    # an abort is not a handoff
    assert hf["cooldown_until"] == NOW + bidpolicy.HANDOFF_COOLDOWN_S
    assert emits[0][1] == "handoff_abort" and emits[0][2]["reason"] == "deadline"


def test_reap_on_exit_reaps_pre_cutover_only(emits, boxops) -> None:
    """LANE DIVERGENCE (pinned). The run lane reaps only PRE-cutover: after the
    cutover the understudy IS the run's canonical box, and the stop path parks
    st's tracked box, which is now that understudy."""
    for phase in bidpolicy._HANDOFF_PRE_CUTOVER:
        boxops.clear()
        handoff._handoff_reap_on_exit(_st(), _ns(), _hf(phase=phase,
                                                        understudy_iid=900))
        assert ("destroy", 900) in boxops, phase
    for phase in ("CUTOVER", "DRAINING", "DONE", "IDLE"):
        boxops.clear()
        handoff._handoff_reap_on_exit(_st(), _ns(), _hf(phase=phase,
                                                        understudy_iid=900))
        assert boxops == [], phase
    boxops.clear()
    handoff._handoff_reap_on_exit(_st(), _ns(), {})     # no sub-state at all
    assert boxops == []


# --------------------------------------------------------------------------- #
# 11. THE UNFENCE (shared by both lanes' post-cutover aborts)
# --------------------------------------------------------------------------- #
def test_unfence_refuses_to_resume_a_box_it_cannot_price(boxops) -> None:
    """A box resumed at the $0.001 pin is a live rental that cannot defend itself
    — the first eviction target, and the same wedge one step later. Left PARKED
    is recoverable by hand and by the reaper, so the refusal is the safe branch
    and it is reported, not silent."""
    seen: list[dict] = []
    assert handoff._handoff_unfence_primary(700, _hf(), emit=lambda **f: seen.append(f)
                                            ) is False
    assert boxops == []
    assert seen[0]["restored_bid"] is None and seen[0]["resume_ok"] is False


def test_unfence_never_restores_the_pin_itself(boxops) -> None:
    """The belt behind `_prefence_bid`: even if a `prefence_bid` at (or below) the
    pin got recorded, it is discarded rather than restored."""
    hf = _hf(prefence_bid=bidpolicy.HANDOFF_PARK_BID)
    assert handoff._handoff_unfence_primary(700, hf, policy_target=0.42) is True
    assert ("bid", 700, 0.42) in boxops


def test_unfence_prefers_the_recorded_prefence_bid_then_resumes(boxops) -> None:
    assert handoff._handoff_unfence_primary(700, _hf(prefence_bid=2.55),
                                            policy_target=0.42) is True
    assert boxops == [("bid", 700, 2.55), ("put", 700, "running")]


def test_unfence_is_inert_under_dry_run_and_on_a_missing_iid(boxops) -> None:
    assert handoff._handoff_unfence_primary(700, _hf(prefence_bid=2.55),
                                            dry_run=True) is True
    assert handoff._handoff_unfence_primary(None, _hf(prefence_bid=2.55)) is False
    assert boxops == []


# --------------------------------------------------------------------------- #
# 12. THE RUN-LANE ACTION DISPATCHER (the money moves)
# --------------------------------------------------------------------------- #
def test_arm_stamps_the_epoch_marker_before_anything_launches(
        emits, monkeypatch) -> None:
    """T4b: the write-generation marker goes up at ARM so a stale writer (a
    resumed husk) that reads a strictly-greater epoch refuses to push."""
    wrote: list[tuple] = []
    monkeypatch.setattr(handoff, "_handoff_b2_write",
                        lambda rid, rel, body, dry_run=False:
                        wrote.append((rid, rel, dry_run)) or True)
    hf = _hf(handoffs_done=1, over_ceiling_streak=5)
    handoff._do_handoff_move(_st(), _ns(), hf, bidpolicy.HandoffAction("arm", "cheaper"))
    assert hf["phase"] == "ARMED" and hf["epoch"] == 2
    assert hf["primary_iid"] == 700 and hf["handoff_started_ts"] == NOW
    assert wrote == [("r-test", "2.json", False)]
    assert emits[0][1] == "handoff_armed" and emits[0][2]["epoch"] == 2


def test_fence_parks_the_primary_pins_its_bid_and_records_the_restore(
        emits, boxops, monkeypatch) -> None:
    """Park + pin, in that order, then CUTOVER. The pin is the two-writer belt: a
    PARKED bid box auto-resumes when the floor drops and would race the
    understudy's checkpoint writes."""
    hf = _hf(phase="SYNCED", understudy_iid=900, primary_iid=700)
    handoff._do_handoff_move(_st(last_bid=2.55, dph=2.60), _ns(), hf,
                             bidpolicy.HandoffAction("fence_primary", "synced"))
    assert boxops == [("put", 700, "stopped"), ("wait", 700, ["exited", "stopped"]),
                      ("bid", 700, bidpolicy.HANDOFF_PARK_BID)]
    assert hf["phase"] == "CUTOVER" and hf["fence_ts"] == NOW
    assert hf["prefence_bid"] == 2.55
    assert emits[0][2]["pinned_bid"] == bidpolicy.HANDOFF_PARK_BID


def test_fence_under_dry_run_touches_no_box(emits, boxops) -> None:
    hf = _hf(phase="SYNCED", understudy_iid=900, primary_iid=700)
    handoff._do_handoff_move(_st(), _ns(dry_run=True), hf,
                             bidpolicy.HandoffAction("fence_primary", "synced"))
    assert boxops == []
    assert hf["phase"] == "CUTOVER"


def test_resume_understudy_writes_promoted_before_relabelling(
        emits, monkeypatch) -> None:
    """Ordering is load-bearing: the understudy's box-side dead-man watchdog parks
    the box unless `runs/<ID>/handoff/promoted` is present, so a supervisor that
    dies mid-cutover must not self-park the true survivor."""
    order: list[str] = []
    monkeypatch.setattr(handoff, "_handoff_b2_write",
                        lambda rid, rel, body, dry_run=False:
                        order.append(f"b2:{rel}") or True)
    monkeypatch.setattr(lifecycle, "_put_label_soft",
                        lambda iid, label: order.append(f"label:{label}") or (True, None))
    hf = _hf(phase="CUTOVER", understudy_iid=900, epoch=1)
    handoff._do_handoff_move(_st(), _ns(), hf,
                             bidpolicy.HandoffAction("resume_understudy", "post_flush"))
    assert order == ["b2:promoted", "label:run:r-test"]
    assert hf["phase"] == "DRAINING" and hf["drain_ts"] == NOW
    assert hf["cutover_ts"] and hf["primary_gone"] is False


def test_a_fast_cutover_marks_the_primary_already_gone(emits, monkeypatch) -> None:
    monkeypatch.setattr(handoff, "_handoff_b2_write",
                        lambda *a, **k: True)
    monkeypatch.setattr(lifecycle, "_put_label_soft", lambda iid, label: (True, None))
    hf = _hf(phase="CUTOVER", understudy_iid=900)
    handoff._do_handoff_move(_st(), _ns(), hf,
                             bidpolicy.HandoffAction("resume_understudy", "fast_cutover"))
    assert hf["primary_gone"] is True


def test_abort_unfence_gives_the_primary_back_before_reaping(
        emits, boxops, monkeypatch) -> None:
    monkeypatch.setattr(handoff, "_confirm_gone", lambda iid, tries=6: True)
    hf = _hf(phase="DRAINING", understudy_iid=900, primary_iid=700, prefence_bid=2.55)
    handoff._do_handoff_move(_st(), _ns(), hf,
                             bidpolicy.HandoffAction("abort_unfence", "understudy_dead"))
    assert boxops == [("bid", 700, 2.55), ("put", 700, "running"), ("destroy", 900)]
    assert [e[1] for e in emits] == ["handoff_unfence", "handoff_abort"]
    assert hf["phase"] == "IDLE"


# --------------------------------------------------------------------------- #
# 13. THE RUN-LANE TICK
# --------------------------------------------------------------------------- #
@pytest.fixture
def tick_seams(monkeypatch):
    """Everything `_handoff_tick` reaches beyond the pure core."""
    calls: list[str] = []
    monkeypatch.setattr(handoff, "_handoff_observe_understudy",
                        lambda st, hf: calls.append("observe"))
    monkeypatch.setattr(handoff, "_handoff_run_signals",
                        lambda rid, cutover_ts=None:
                        calls.append("signals") or {"final_flush_seen": True,
                                                    "understudy_producing": True})
    monkeypatch.setattr(handoff, "_do_handoff_move",
                        lambda st, a, hf, act: calls.append(f"move:{act.kind}"))
    return calls


def test_tick_leaves_remaining_wall_h_none_without_a_wall_budget(
        tick_seams, emits) -> None:
    """defect #63: the flat 24.0 that stood here was the fabrication that migrated
    a healthy jobs box on 2026-08-08. No wall budget means NO horizon, and the
    candidate filter refuses — `None`, never `0.0` (defect #67)."""
    st = _st(wall_budget_s=None)
    handoff._handoff_tick(st, _ns(), _hf(), bidpolicy.Action("noop", "ok"))
    assert st["remaining_wall_h"] is None
    st = _st(wall_budget_s=7200.0, wall_clock_s=3600.0)
    handoff._handoff_tick(st, _ns(), _hf(), bidpolicy.Action("noop", "ok"))
    assert st["remaining_wall_h"] == pytest.approx(1.0)


def test_tick_dwell_counter_needs_consecutive_over_ceiling_polls(
        tick_seams, emits) -> None:
    hf = _hf()
    handoff._handoff_tick(_st(_over_pref=True), _ns(), hf, bidpolicy.Action("noop", "ok"))
    handoff._handoff_tick(_st(_over_pref=True), _ns(), hf, bidpolicy.Action("noop", "ok"))
    assert hf["over_ceiling_streak"] == 2
    handoff._handoff_tick(_st(_over_pref=False), _ns(), hf, bidpolicy.Action("noop", "ok"))
    assert hf["over_ceiling_streak"] == 0


def test_tick_reads_the_box_side_signals_only_while_the_fence_is_open(
        tick_seams, emits) -> None:
    hf = _hf(phase="WARMING")
    handoff._handoff_tick(_st(), _ns(), hf, bidpolicy.Action("noop", "ok"))
    assert "signals" not in tick_seams
    assert hf["final_flush_seen"] is False
    hf["phase"] = "DRAINING"
    handoff._handoff_tick(_st(), _ns(), hf, bidpolicy.Action("noop", "ok"))
    assert "signals" in tick_seams
    assert hf["final_flush_seen"] is True and hf["understudy_producing"] is True


def test_tick_executes_only_a_non_noop_action(tick_seams, emits, monkeypatch) -> None:
    monkeypatch.setattr(bidpolicy, "handoff_poll",
                        lambda hs: bidpolicy.HandoffAction("noop", "ok"))
    handoff._handoff_tick(_st(), _ns(), _hf(), bidpolicy.Action("noop", "ok"))
    assert not [c for c in tick_seams if c.startswith("move:")]
    monkeypatch.setattr(bidpolicy, "handoff_poll",
                        lambda hs: bidpolicy.HandoffAction("arm", "cheaper"))
    handoff._handoff_tick(_st(), _ns(), _hf(), bidpolicy.Action("noop", "ok"))
    assert "move:arm" in tick_seams


def test_tick_pins_the_primary_iid_for_an_adopted_migration(
        tick_seams, emits) -> None:
    """A reconciled handoff never saw ARM, so nothing pinned the box being
    retired. The guard is RAW (`not in (None, understudy_iid)`) — the jobs-lane
    mirror spells it with `str()` and the literal "None". Pinned divergence."""
    hf = _hf(phase="LAUNCHING", understudy_iid=900)
    handoff._handoff_tick(_st(instance_id=700), _ns(), hf, bidpolicy.Action("noop", "ok"))
    assert hf["primary_iid"] == 700
    hf2 = _hf(phase="LAUNCHING", understudy_iid=700)
    handoff._handoff_tick(_st(instance_id=700), _ns(), hf2, bidpolicy.Action("noop", "ok"))
    assert hf2["primary_iid"] is None


def test_reconcile_adopts_a_live_twin_left_by_a_crashed_supervisor(
        emits, monkeypatch) -> None:
    monkeypatch.setattr(handoff, "_handoff_synced_epoch_soft", lambda rid: None)
    monkeypatch.setattr(handoff.api, "request_soft",
                        lambda method, path, body=None, *a, **k:
                        (True, {"instances": [{"id": 900, "actual_status": "running",
                                               "dph_total": 0.44,
                                               "label": "run:r-test:handoff"}]}, None))
    hf = _hf()
    handoff._handoff_reconcile(_st(), _ns(), hf)
    assert hf["understudy_iid"] == 900 and hf["phase"] in ("LAUNCHING", "SYNCED")
    assert emits[0][1] == "handoff_reconciled"


# --------------------------------------------------------------------------- #
# 14. JOBS-LANE SIGNALS AND PROOFS
# --------------------------------------------------------------------------- #
def test_job_producing_needs_the_understudys_own_id_on_the_event(
        monkeypatch) -> None:
    """Proof-of-life gating the primary destroy: a `claimed`/`checkpoint` event
    from the PRIMARY proves nothing about the understudy, and an event kind
    outside `_HANDOFF_JOB_PRODUCING` (e.g. `failed`) is not production."""
    monkeypatch.setattr(handoff, "_raw_job_events_soft",
                        lambda jid: [{"event": "claimed", "instance_id": "900"},
                                     {"event": "final_flush"}])
    assert handoff._handoff_job_signals(["j-1"], ["j-1"], 900) == {
        "final_flush_seen": True, "understudy_producing": True}
    assert handoff._handoff_job_signals(["j-1"], ["j-1"], 901
                                        )["understudy_producing"] is False
    monkeypatch.setattr(handoff, "_raw_job_events_soft",
                        lambda jid: [{"event": "failed", "instance_id": "900"}])
    assert handoff._handoff_job_signals(["j-1"], ["j-1"], 900
                                        )["understudy_producing"] is False


def test_final_flush_needs_EVERY_job_that_was_running_at_the_fence(
        monkeypatch) -> None:
    """An empty running set (nothing was writing) is trivially fenced."""
    monkeypatch.setattr(handoff, "_raw_job_events_soft",
                        lambda jid: [{"event": "final_flush"}] if jid == "j-1" else [])
    assert handoff._handoff_job_signals(["j-1"], ["j-1", "j-2"], None
                                        )["final_flush_seen"] is True
    assert handoff._handoff_job_signals(["j-1", "j-2"], ["j-1", "j-2"], None
                                        )["final_flush_seen"] is False
    assert handoff._handoff_job_signals([], ["j-1", "j-2"], None
                                        )["final_flush_seen"] is True


def test_jobd_status_is_the_first_token_uppercased(monkeypatch) -> None:
    monkeypatch.setattr(health, "_jobd_status_line_soft",
                        lambda iid: "running 2 1234 5678")
    assert handoff._jobd_status_soft("900") == "RUNNING"
    monkeypatch.setattr(health, "_jobd_status_line_soft", lambda iid: None)
    assert handoff._jobd_status_soft("900") is None


def _synced_seams(monkeypatch, *, status="IDLE", pyhalf=None, box=None):
    monkeypatch.setattr(handoff, "_jobd_status_soft", lambda iid: status)
    monkeypatch.setattr(health, "_jobd_status_pyhalf_soft", lambda iid: pyhalf)
    monkeypatch.setattr(handoff, "_box_lifecycle_soft", lambda iid: box or {})


def test_job_synced_needs_a_live_box_with_an_affirmative_jobd_stamp(
        monkeypatch) -> None:
    """Absence-of-park is not evidence a box ever booted: a still-`loading` box
    has an empty lifecycle fold and read as healthy — the same hole the run
    lane's liveness proxy had (live canary 2026-07-15)."""
    _synced_seams(monkeypatch)
    hf = _jhf(understudy_iid="900", understudy_status="running")
    assert handoff._handoff_job_understudy_synced(_jctx(), hf) is True
    assert handoff._handoff_job_understudy_synced(
        _jctx(), _jhf(understudy_iid="900", understudy_status="exited")) is False
    assert handoff._handoff_job_understudy_synced(
        _jctx(), _jhf(understudy_iid=None, understudy_status="running")) is False
    _synced_seams(monkeypatch, status=None)
    assert handoff._handoff_job_understudy_synced(_jctx(), hf) is False
    _synced_seams(monkeypatch, box={"parked": True})
    assert handoff._handoff_job_understudy_synced(_jctx(), hf) is False


def test_only_a_true_python_half_confession_blocks_synced(monkeypatch) -> None:
    """TRI-STATE, and `is True` is load-bearing (2026-08-14). A box with a dead
    python half still stamps `IDLE` — the bash half writes the marker — and
    without this check the migration retargets the queue onto a box that can
    neither claim a ticket nor emit an event, then destroys the healthy primary.
    `None` means the bundle is older than the field, NOT "confessing"."""
    hf = _jhf(understudy_iid="900", understudy_status="running")
    _synced_seams(monkeypatch, pyhalf=None)
    assert handoff._handoff_job_understudy_synced(_jctx(), hf) is True
    _synced_seams(monkeypatch, pyhalf=False)
    assert handoff._handoff_job_understudy_synced(_jctx(), hf) is True
    _synced_seams(monkeypatch, pyhalf=True)
    assert handoff._handoff_job_understudy_synced(_jctx(), hf) is False


def test_job_observe_only_latches_gone_on_a_non_empty_listing(monkeypatch) -> None:
    """An EMPTY snapshot is an API blip, not evidence. Leaving `understudy_status`
    stale on a listing that DOES omit our id is what let a dead understudy keep
    reading as `running` while DRAINING waited on it forever."""
    _synced_seams(monkeypatch)
    hf = _jhf(phase="WARMING", understudy_iid="900", understudy_status="running")
    handoff._handoff_observe_job_understudy(_jctx(instances=[]), hf)
    assert hf["understudy_status"] == "running" and hf.get("understudy_gone") is None
    handoff._handoff_observe_job_understudy(
        _jctx(instances=[{"id": "111", "label": "job:111"}]), hf)
    assert hf["understudy_gone"] is True and hf["understudy_status"] is None


def test_job_observe_stamps_ckpt_epoch_once_the_understudy_is_synced(
        monkeypatch) -> None:
    _synced_seams(monkeypatch)
    hf = _jhf(phase="WARMING", handoffs_done=1)
    inst = {"id": "900", "label": "job:700:handoff", "actual_status": "Running",
            "dph_total": 0.44}
    handoff._handoff_observe_job_understudy(_jctx(instances=[inst]), hf)
    assert hf["understudy_iid"] == "900" and hf["understudy_dph"] == 0.44
    assert hf["understudy_live_since"] == NOW and hf["ckpt_pulled_epoch"] == 2


# --------------------------------------------------------------------------- #
# 15. JOBS-LANE REFUSALS (the half that used to be silent)
# --------------------------------------------------------------------------- #
def test_a_work_refusal_is_said_once_per_distinct_cause(jjournal) -> None:
    """Deduped on the REASON, not the tick: the condition re-tests every poll and
    would otherwise write a line a minute into `fleet log`."""
    hf = _jhf()
    handoff._job_handoff_refuse(_jctx(), hf, "checkpoint_stale")
    handoff._job_handoff_refuse(_jctx(), hf, "checkpoint_stale")
    assert len(jjournal) == 1
    assert jjournal[0][0] == "refused" and jjournal[0][1]["reason"] == "checkpoint_stale"
    handoff._job_handoff_refuse(_jctx(), hf, "unresumable_running_job")
    assert len(jjournal) == 2


def test_a_deferral_carries_the_arithmetic_and_dedupes_on_its_signature(
        jjournal) -> None:
    """The 2026-08-08 post-mortem had to reconstruct this arithmetic from the
    market by hand, because the only visible handoff decision was the one that
    fired. The numbers go IN the message."""
    jctx = _jctx(last_bid=1.00, remaining_wall_h=2.0)
    hf = _jhf(candidate_min_bid=0.20, candidate_on_demand=0.90)
    handoff._job_handoff_defer(jctx, hf)
    handoff._job_handoff_defer(jctx, hf)
    assert len(jjournal) == 1
    fields = jjournal[0][1]
    assert fields["primary_dph"] == 1.00 and fields["horizon_s"] == 7200
    assert fields["overhead_usd"] == pytest.approx(
        round((1.00 + fields["candidate_target"]) * bidpolicy.HANDOFF_WINDOW_H, 4))


def test_a_deferral_keeps_an_unknown_horizon_none_shaped(jjournal) -> None:
    """defect #67: `remaining_wall_h is None` is UNKNOWN and stays None in the
    journal field — a `0.0` there reads as a measured, expired horizon."""
    jctx = _jctx(last_bid=1.00, remaining_wall_h=None)
    handoff._job_handoff_defer(jctx, _jhf(candidate_min_bid=0.20,
                                          candidate_on_demand=0.90))
    assert jjournal[0][1]["horizon_s"] is None
    assert "UNMEASURABLE" in jjournal[0][1]["note"]


def test_a_deferral_with_no_priced_candidate_says_nothing(jjournal) -> None:
    handoff._job_handoff_defer(_jctx(last_bid=1.00), _jhf())
    handoff._job_handoff_defer(_jctx(last_bid=None, dph=None),
                               _jhf(candidate_min_bid=0.20, candidate_on_demand=0.90))
    assert jjournal == []


def test_the_progress_warning_latches_per_job_and_clears_with_the_condition(
        jjournal, monkeypatch) -> None:
    """Advisory, never a gate (task #67). The `n_checkpoints: 0` variant is the
    honest scope boundary: spot delivers no signal, so nothing here can protect
    that ticket from an eviction either."""
    from vastlib.jobs import risk
    pct = {"v": 95}
    monkeypatch.setattr(risk, "_job_pct", lambda v: pct["v"])
    view = {"display_status": "running", "job_id": "j-1", "n_checkpoints": 0}
    jctx, hf = _jctx(pending_views=[view]), _jhf()
    handoff._job_handoff_progress_warn(jctx, hf)
    handoff._job_handoff_progress_warn(jctx, hf)
    assert len(jjournal) == 1 and "NO checkpoint" in jjournal[0][1]["note"]
    view["n_checkpoints"] = 2                     # condition changed -> say it again
    handoff._job_handoff_progress_warn(jctx, hf)
    assert len(jjournal) == 2 and "NO checkpoint" not in jjournal[1][1]["note"]
    pct["v"] = 10                                 # back under the threshold -> latch clears
    handoff._job_handoff_progress_warn(jctx, hf)
    assert len(jjournal) == 2 and hf["pct_warned"] == {}


# --------------------------------------------------------------------------- #
# 16. JOBS-LANE TICKET MOVES AND THE EXIT PATH
# --------------------------------------------------------------------------- #
def test_retarget_back_is_a_noop_when_the_cutover_moved_nothing(
        jemits, monkeypatch) -> None:
    """The two-writer fence correctly refusing to retarget a job still RUNNING on
    the primary (the live 2026-08-05 shape) is the COMMON case, not an error."""
    monkeypatch.setattr(jobmeta, "list_queue", lambda iid: [])
    assert handoff._job_handoff_retarget_back(
        _jctx(), _jhf(understudy_iid="900", primary_iid="700")) == []
    assert jemits == []


def test_retarget_back_refuses_to_move_tickets_onto_the_same_box(jemits) -> None:
    assert handoff._job_handoff_retarget_back(
        _jctx(), _jhf(understudy_iid="700", primary_iid="700")) == []
    assert handoff._job_handoff_retarget_back(_jctx(), _jhf()) == []


def test_retarget_back_survives_an_unreadable_understudy_queue(
        jemits, monkeypatch) -> None:
    """A stranded ticket is a `herdd job orphans` problem; raising here would
    abandon the unfence half-done with the primary still parked."""
    def boom(iid):
        raise RuntimeError("b2 down")
    monkeypatch.setattr(jobmeta, "list_queue", boom)
    assert handoff._job_handoff_retarget_back(
        _jctx(), _jhf(understudy_iid="900", primary_iid="700")) == []


def test_retarget_back_under_dry_run_calls_no_ticket_writer(
        jemits, monkeypatch) -> None:
    monkeypatch.setattr(jobmeta, "list_queue", lambda iid: ["j-1", "j-2"])
    moved = handoff._job_handoff_retarget_back(
        _jctx(), _jhf(understudy_iid="900", primary_iid="700"), dry=True)
    assert moved == ["j-1", "j-2"]                # the seam would have raised
    assert jemits[0][0] == "handoff_retarget_back" and jemits[0][1]["jobs"] == 2


def test_the_jobs_exit_path_unwinds_an_open_fence_but_never_a_drain(
        jemits, boxops, monkeypatch) -> None:
    """LANE DIVERGENCE (pinned, 2026-08-08 task #62). CUTOVER means the primary is
    parked, pinned to an unwinnable bid, and still owns the tickets — the pin must
    not outlive the fence window on ANY path, and "the supervisor stopped" is a
    path. DRAINING is deliberately NOT unwound: the tickets are already on the
    understudy, so resuming the primary would put a second claimant on the board.
    The run lane's twin reaps PRE-cutover only and unwinds nothing."""
    monkeypatch.setattr(jobmeta, "list_queue", lambda iid: [])
    hf = _jhf(phase="CUTOVER", primary_iid="700", understudy_iid="900",
              prefence_bid=2.55)
    handoff._job_handoff_reap_on_exit(_jctx(), hf)
    assert boxops == [("bid", "700", 2.55), ("put", "700", "running"),
                      ("destroy", "900")]
    assert [e[0] for e in jemits] == ["handoff_unfence", "handoff_abort",
                                      "handoff_abort"]

    boxops.clear(); jemits.clear()
    handoff._job_handoff_reap_on_exit(
        _jctx(), _jhf(phase="DRAINING", primary_iid="700", understudy_iid="900"))
    assert boxops == [] and jemits == []


def test_the_jobs_exit_path_reaps_a_pre_cutover_understudy(jemits, boxops) -> None:
    for phase in bidpolicy._HANDOFF_PRE_CUTOVER:
        boxops.clear()
        handoff._job_handoff_reap_on_exit(
            _jctx(), _jhf(phase=phase, primary_iid="700", understudy_iid="900"))
        assert boxops == [("destroy", "900")], phase
    handoff._job_handoff_reap_on_exit(_jctx(), {})
    handoff._job_handoff_reap_on_exit(_jctx(), _jhf(phase="IDLE"))


def test_the_jobs_tick_pins_the_primary_with_the_string_shaped_guard(
        monkeypatch, jemits) -> None:
    """LANE DIVERGENCE (pinned): the jobs guard is `str(iid) not in (None, "None",
    str(understudy_iid))` — the literal "None" included, because every box id on
    this lane is spelled as a string (queue path segment, ticket `box` field,
    event `instance_id`, `--box` argv). The run lane compares raw."""
    monkeypatch.setattr(handoff, "_handoff_observe_job_understudy", lambda j, h: None)
    monkeypatch.setattr(handoff, "_do_job_handoff_move", lambda j, h, act: None)
    hf = _jhf(phase="LAUNCHING", understudy_iid="900")
    handoff._job_handoff_tick(_jctx(iid="700"), hf)
    assert hf["primary_iid"] == "700"
    hf2 = _jhf(phase="LAUNCHING", understudy_iid=900)      # int vs str: SAME box here
    handoff._job_handoff_tick(_jctx(iid="900"), hf2)
    assert hf2["primary_iid"] is None
    hf3 = _jhf(phase="LAUNCHING", understudy_iid="900")
    handoff._job_handoff_tick(_jctx(iid=None), hf3)        # the literal "None" guard
    assert hf3["primary_iid"] is None


def test_the_jobs_tick_routes_a_noop_to_the_right_explanation(
        monkeypatch, jemits, jjournal) -> None:
    """A `noop` is not silence on this lane: `candidate_reject` explains the
    ECONOMIC refusal, a `precondition:`/`fence_hold:` reason the WORK refusal, and
    a real move RETRACTS any standing refusal signature."""
    monkeypatch.setattr(handoff, "_handoff_observe_job_understudy", lambda j, h: None)
    moves: list[str] = []
    monkeypatch.setattr(handoff, "_do_job_handoff_move",
                        lambda j, h, act: moves.append(act.kind))
    jctx = _jctx(last_bid=1.00, handoff_can_complete=True)
    hf = _jhf(candidate_min_bid=0.20, candidate_on_demand=0.90)

    monkeypatch.setattr(bidpolicy, "handoff_poll",
                        lambda hs: bidpolicy.HandoffAction("noop", "candidate_reject"))
    handoff._job_handoff_tick(jctx, hf)
    assert [k for k, _ in jjournal] == ["deferred"]

    monkeypatch.setattr(bidpolicy, "handoff_poll",
                        lambda hs: bidpolicy.HandoffAction(
                            "noop", "precondition:driver_cannot_complete"))
    handoff._job_handoff_tick(jctx, hf)
    assert [k for k, _ in jjournal] == ["deferred", "refused"]
    assert hf["refuse_sig"] == "driver_cannot_complete"

    monkeypatch.setattr(bidpolicy, "handoff_poll",
                        lambda hs: bidpolicy.HandoffAction("arm", "cheaper"))
    handoff._job_handoff_tick(jctx, hf)
    assert moves == ["arm"] and "refuse_sig" not in hf


# --------------------------------------------------------------------------- #
# 17. THE SEAMS — ALL OF THEM NOW FORWARD
#
# There is no `test_every_unported_seam_raises_...` here any more. `_confirm_gone`
# was its last entry, and step 6d closed it: `boxes/lifecycle.py` owns the one
# body and this module owns the ATTRIBUTE its own callers patch. Plan §7.4
# licenses the expectation change — what changed is "not ported yet", not
# behavior — and the forwarding tests below are the stronger assertion anyway:
# "raises" only proves nothing was forked, while "a patch on the home steers
# this module" proves it positively, every run.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,home,args", [
    ("_confirm_gone", "vastlib.boxes.lifecycle", (700,)),
])
def test_the_step_6d_seam_now_forwards_to_the_boxes_ring(name, home, args,
                                                         monkeypatch) -> None:
    """Step-6d inversion of the raise assertion this file used to carry.

    `_confirm_gone` must FORWARD to `boxes.lifecycle` — not alias, not
    re-implement. A module-level `_confirm_gone = lifecycle._confirm_gone` would
    capture the function object at import and this patch would not steer it; a
    second copy of the destroy probe would not steer either. `tries` rides
    through positionally, because the four call sites here pass only `iid` and
    the retry budget is the home's contract, not this module's."""
    import importlib
    mod = importlib.import_module(home)
    seen: list[object] = []

    def _sentinel(*a, **k):
        seen.append(a)
        return "sentinel"

    monkeypatch.setattr(mod, name, _sentinel)
    assert getattr(handoff, name)(*args) == "sentinel"
    assert seen == [(*args, 6)]                      # default tries forwarded


@pytest.mark.parametrize("name,home,args", [
    ("_live_iids_set", "vastlib.jobs.view", ()),
    ("_box_lifecycle_soft", "vastlib.jobs.view", (700,)),
    ("cmd_job_retarget", "vastlib.jobs.control", (argparse.Namespace(),)),
])
def test_the_step_5_seams_now_forward_to_the_jobs_ring(name, home, args,
                                                       monkeypatch) -> None:
    """Step-5 INVERSION of the raise assertions above (plan §7.4 licenses the
    expectation change: what changed is "not ported yet", not behavior).

    Each seam must FORWARD — not alias, not re-implement. The proof is a patch on
    the home module steering `handoff`'s copy: an alias captured at import would
    not see it, and a fork would not see it either.
    """
    import importlib
    mod = importlib.import_module(home)
    seen: list[object] = []

    def _sentinel(*a, **k):
        seen.append(a)
        return "sentinel"

    monkeypatch.setattr(mod, name, _sentinel)
    assert getattr(handoff, name)(*args) == "sentinel"
    assert seen == [args]


def test_the_signal_readers_swallow_a_seam_that_has_not_landed(monkeypatch) -> None:
    """`_handoff_job_signals` calls `_live_iids_set` (unported) inside its
    try/except, so an unbound seam degrades to "cache not refreshed" — the same
    outcome a B2 read failure produces — instead of killing the fence read."""
    monkeypatch.setattr(handoff, "_raw_job_events_soft", lambda jid: [])
    assert handoff._handoff_job_signals(["j-1"], ["j-1"], "900") == {
        "final_flush_seen": False, "understudy_producing": False}
