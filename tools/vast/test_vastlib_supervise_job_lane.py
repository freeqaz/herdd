"""`vastlib.supervise.job_lane` — the port-time regression net for the JOBS tick.

Why this file exists
--------------------
`job_supervise_tick` is the money path for every rented job box, and it is
driven by TWO callers (the legacy inline `cmd_job_supervise` loop and fleetd's
`jobs`/`serve` profiles) from ONE copy of the ladder. Its heaviest existing
driver is `test_supervise.py` (32 tick references inside 426 `herdd`
references), which under the plan §8 add-only amendment stays UNEDITED this
step and therefore keeps steering the still-live FLAT copies in `herdd.py`.
That leaves the vastlib copy uncovered, which is exactly the shape a port must
not ship: the flat suite would stay green while the package copy drifted.

So this file is the parallel net. It drives the real `vastlib` functions with
hand-built `jc` dicts and stubbed module attributes — the same technique
`test_supervise.py` uses, applied to the new module homes, without touching
that file.

What it pins, and why each one is a defect and not a preference
---------------------------------------------------------------
* **The five jc-published risk metrics stay `None` for UNKNOWN.** `bidpolicy`
  (Zone S) reads `work_at_risk_h` / `running_unresumable` / `min_running_eta_s`
  / `ckpt_stale` / `remaining_wall_h` off `jc` BY KEY, with no import edge to
  catch a type change. A `0.0` there turns a migration REFUSAL into a migration
  (defect #67), so the assertions are `is None`, never `not x`.
* **`handoff_can_complete` fails CLOSED.** Defect #61: under fleetd the watch
  ended one tick before `handoff_poll` returned `complete`, the understudy
  inherited no watch and no budget cap, and the stray sweep adopted it as an
  uncapped `bare` box. Only a driver that can carry a migration to completion
  may say so; the default is that it cannot.
* **Late binding is the contract.** Every cross-module call the tick makes is
  resolved at call time from the owning module, so the tests patch
  `journal._job_ladder_journal`, `risk._ckpt_watchdog_alarm`,
  `handoff._job_handoff_tick`, `lifecycle._put_bid_soft`, ... by module
  attribute. A `from … import` in the port would make these vacuous, which is
  what `test_seams_are_reached_through_their_owning_module` exists to catch.
* **The inline `jobmeta.emit_box_event`** for `bid_over_preferred_ceiling`
  bypasses `_job_handoff_emit` and PRINTS on failure instead of returning
  `{"_emitted": False}`. It is the one jobs-lane emit a `_job_handoff_emit`
  patch does not intercept, and it ported as-is (plan §7.4).
* **The event names `jobs_bid_self_floor` / `jobs_bid_floor_blind`** are a wire
  format — `fleet log` and the journal analytics key on them, and they differ
  from the run lane's by design (different event log, different identity).
* **`_job_primary_evicted` counts THIS tick** (`streak + 1`): an off-by-one
  fast-CUTOVERs off a still-live primary and leaks it un-fenced.
* **`_job_sup_reattach(jc, iid)` keeps its 2-arg shape** — three flat tests
  stub it with a 2-arg lambda, so a signature change breaks them silently.

Isolation
---------
Nothing here reaches the network, B2, ssh or the vast API. Every instance
fetch, PUT, stop, market read, journal write and jobmeta call is replaced at
its OWNING module by `monkeypatch.setattr` WITHOUT `raising=False`, so a seam
that moves fails loudly instead of going vacuous. ONE still-unclaimed seam
(`_box_lifecycle_soft`) raises `NotImplementedError` until a later step lands
it, and every tick test stubs it explicitly. `_sticky_on_demand` and
`_serve_self_park_soft` were seams here until step 6 and are now stubbed at
`market.pricing` / `supervise.replacement`, the modules the tick calls.

What is deliberately NOT here
-----------------------------
* No edit to `test_supervise.py`, `test_eviction_blindspot.py`,
  `test_eviction_replacement.py`, `test_boot_sla.py`, `test_pull_watchdog.py`
  or `test_ladder_core.py`. They stay UNEDITED and keep steering `herdd`.
* No re-test of `ladder_core`'s self-floor state machine or of `bidpolicy`'s
  pure decisions — those have their own suites. Only this lane's wrappers.
* No assertion that the run and jobs lanes agree: the six divergences between
  them are pinned deliberately (plan §5 NOTE).

Provenance: created 2026-08-16 alongside `vastlib/supervise/job_lane.py`,
plan §8 step 4.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

import bidpolicy                                       # noqa: E402  Zone S
import jobmeta                                         # noqa: E402  Zone S

from vastlib.boxes import lifecycle                    # noqa: E402
from vastlib.core import models                        # noqa: E402
from vastlib.jobs import risk                          # noqa: E402
from vastlib.market import pricing                     # noqa: E402
from vastlib.supervise import (                        # noqa: E402
    handoff, job_lane, journal, replacement, retention,
)

IID = "9000"
MACHINE = 7
FLOOR = 0.40
ON_DEMAND = 1.20


def _inst(**over):
    """A live, bid-rented job box body, in the shape vast returns."""
    body = {"id": int(IID), "actual_status": "running",
            "intended_status": "running", "machine_id": MACHINE,
            "num_gpus": 2, "is_bid": True, "dph_total": 0.62,
            "dph_base": 0.60, "label": f"job:{IID}", "gpu_name": "RTX 4090"}
    body.update(over)
    return body


def _ns(**over):
    """The argparse namespace `job supervise` builds (fleetd seeds the same
    keys through its policy)."""
    base = dict(id=int(IID), dry_run=False, budget=100.0, max_bid=None,
                handoff=True, strict_ceiling=False, rescue_wait=None,
                keep=False, wall_budget=None, handoff_can_complete=True)
    base.update(over)
    return argparse.Namespace(**base)


class _Wire:
    """Every seam one tick reaches, stubbed at its OWNING module and recorded.

    The point of the module-attribute form is the port contract itself: the
    tick resolves each of these at call time, so a `from … import` inside
    `job_lane.py` would leave these stubs uncalled and the tests green for the
    wrong reason. `test_seams_are_reached_through_their_owning_module` asserts
    the calls actually landed."""

    def __init__(self, monkeypatch, *, inst=None, queue=(), views=(),
                 floor=FLOOR, on_demand=ON_DEMAND, listed=True,
                 parked=False, drained_pending=False, serve_parked=False,
                 bid_put_ok=True, risk_metrics=None):
        m = monkeypatch
        self.calls: list[str] = []
        self.journal: list[tuple[str, dict]] = []
        self.emits: list[tuple[str, dict]] = []
        self.box_events: list[tuple[str, str, dict]] = []
        self.puts: list[tuple] = []
        self.inst = inst
        self._views = [dict(v) for v in views]
        self._risk = risk_metrics or {}

        # --- the single per-tick instance fetch + every box mutation --------- #
        m.setattr(lifecycle, "_instances_soft",
                  lambda: [dict(self.inst)] if self.inst else [])
        m.setattr(lifecycle, "_stop_instance_soft",
                  lambda iid: self.puts.append(("stop", str(iid))) or True)
        m.setattr(lifecycle, "_put_bid_soft",
                  lambda iid, price: self.puts.append(("bid", str(iid), price))
                  or (bid_put_ok, None if bid_put_ok else "stub: refused"))
        m.setattr(lifecycle, "_put_state_soft",
                  lambda iid, state: self.puts.append(("state", str(iid), state))
                  or (True, None))
        m.setattr(lifecycle, "cmd_job_attach",
                  lambda a: self.calls.append("attach"))

        # --- the journal ring (both jobs-lane writers) ----------------------- #
        m.setattr(journal, "_job_ladder_journal",
                  lambda jc, event, **kw: self.journal.append((event, kw)))
        m.setattr(journal, "_job_handoff_emit",
                  lambda jc, event, **kw: self.emits.append((event, kw))
                  or {"_emitted": True})

        # --- Zone S jobmeta: queue reads + the INLINE box event -------------- #
        m.setattr(jobmeta, "list_queue", lambda box, **k: list(queue))
        m.setattr(jobmeta, "read_job",
                  lambda jid, **k: next(dict(v) for v in self._views
                                        if v["job_id"] == jid))
        m.setattr(jobmeta, "emit_box_event",
                  lambda iid, ev, **kw: self.box_events.append((str(iid), ev, kw))
                  or {"_emitted": True})

        # --- the market ring ------------------------------------------------- #
        m.setattr(pricing, "_market_min_bid_read",
                  lambda mid, g=None: models.MarketRead(True, listed, floor))
        m.setattr(pricing, "_market_ondemand_soft",
                  lambda mid, n=None: on_demand)

        # --- sibling supervise modules --------------------------------------- #
        m.setattr(retention, "_job_retention_sweep",
                  lambda jc, now: self.calls.append("retention"))
        m.setattr(replacement, "_job_palt_poll",
                  lambda jc, now, **k: self.calls.append("palt"))
        m.setattr(replacement, "_job_boot_sla_tick", lambda jc, inst, now: None)
        m.setattr(replacement, "_job_pull_watchdog_tick",
                  lambda jc, inst, now: None)
        m.setattr(replacement, "_serve_boot_sla_tick", lambda jc, inst, now: None)
        m.setattr(handoff, "_job_handoff_reconcile",
                  lambda jc, hf: self.calls.append("reconcile"))
        m.setattr(handoff, "_job_handoff_progress_warn",
                  lambda jc, hf: self.calls.append("progress_warn"))
        m.setattr(handoff, "_job_handoff_tick", self._handoff_tick)

        # --- the risk ring: tri-state, so the DEFAULT here is None ----------- #
        for name in ("_jobs_work_horizon_h", "_jobs_remaining_wall_h",
                     "_jobs_work_at_risk_h", "_jobs_min_running_eta_s"):
            m.setattr(risk, name, self._risk_stub(name))
        m.setattr(risk, "_jobs_unresumable_running", self._risk_stub(
            "_jobs_unresumable_running"))
        m.setattr(risk, "_jobs_ckpt_stale", self._risk_stub("_jobs_ckpt_stale"))
        m.setattr(risk, "_ckpt_watchdog_alarm", lambda v, now, **k: None)

        # --- the one seam no manifest claims --------------------------------- #
        m.setattr(job_lane, "_box_lifecycle_soft",
                  lambda iid: {"parked": parked,
                               "drained_pending": drained_pending})
        # ...and the two that stopped being seams at step 6. Both are patched at
        # their OWNING module, because that is where the tick now calls them
        # (`pricing._sticky_on_demand`, `replacement._serve_self_park_soft`) —
        # patching `job_lane` would bind a name the tick no longer reads and the
        # test would go vacuously green against the real implementations.
        m.setattr(pricing, "_sticky_on_demand", lambda jc, fresh: fresh)
        m.setattr(replacement, "_serve_self_park_soft",
                  lambda sid, **k: serve_parked)

        self.handoff_completed_iid = None
        self.handoff_phase = None

    def _risk_stub(self, name):
        def _f(views, *a, **k):
            self.calls.append(name)
            return self._risk.get(name)                # DEFAULT None = UNKNOWN
        return _f

    def _handoff_tick(self, jc, hf):
        self.calls.append("handoff_tick")
        self.handoff_jc = dict(jc)
        if self.handoff_phase is not None:
            hf["phase"] = self.handoff_phase
        if self.handoff_completed_iid is not None:
            jc["_handoff_completed_iid"] = self.handoff_completed_iid
            jc["_handoff_completed_dph"] = 9.99      # must NOT reach last_bid


def _view(job_id="job-a", status="running", display_status="running"):
    return {"job_id": job_id, "status": status, "display_status": display_status}


def _tick(monkeypatch, wire=None, a=None, **wire_kw):
    """Build `jc`/`hf` through the real constructor and run ONE tick."""
    w = wire or _Wire(monkeypatch, **wire_kw)
    ns = a or _ns()
    jc, hf = job_lane.job_supervise_init(ns)
    return w, jc, hf, job_lane.job_supervise_tick(jc, hf)


# --------------------------------------------------------------------------- #
# classify_job_box_stop — PURE, and a documented public seam (parked_lifecycle)
# --------------------------------------------------------------------------- #
def test_classify_job_box_stop_is_a_public_keyword_only_name():
    """`parked_lifecycle.py:6` names it as a seam, so the port may not rename it
    or make it positional."""
    assert job_lane.classify_job_box_stop.__name__ == "classify_job_box_stop"
    sig = inspect.signature(job_lane.classify_job_box_stop)
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY
               for p in sig.parameters.values())


def test_self_park_event_outranks_every_other_signal():
    """A `parked_self`/`drained` box event is SUCCESS, and it is consulted
    FIRST — even for a bid box that also looks displaced."""
    for parked, drained in ((True, False), (False, True)):
        assert job_lane.classify_job_box_stop(
            present=True, live=False, is_bid=True, intended_status="running",
            box_parked=parked, box_drained=drained,
            claimed_work=True) == "self_parked"


def test_operator_park_needs_an_ask_or_an_idle_on_demand_box():
    """`intended_status` is not evidence of intent (task #74). Only a journaled
    stop intent, or the pre-2026-08-09 case it was written for — an IDLE
    on-demand box — is an operator park."""
    assert job_lane.classify_job_box_stop(
        present=True, live=False, is_bid=True, intended_status="stopped",
        box_parked=False, box_drained=False,
        stop_intent=True, claimed_work=True) == "operator_park"
    assert job_lane.classify_job_box_stop(
        present=True, live=False, is_bid=False, intended_status="stopped",
        box_parked=False, box_drained=False) == "operator_park"


def test_an_unexplained_stop_falls_through_to_the_rescue_path():
    """The 2026-07-11 bakeoff-05 regression: an OUTBID box misread as an
    operator park is abandoned. Unexplained must return None (fall through),
    and an on-demand box that stopped out from under LIVE WORK is not an
    operator park either."""
    assert job_lane.classify_job_box_stop(
        present=True, live=False, is_bid=True, intended_status="stopped",
        box_parked=False, box_drained=False) is None
    assert job_lane.classify_job_box_stop(
        present=True, live=False, is_bid=False, intended_status="stopped",
        box_parked=False, box_drained=False, claimed_work=True) is None
    # a box that has left the listing entirely was never an operator park
    assert job_lane.classify_job_box_stop(
        present=False, live=False, is_bid=True, intended_status="stopped",
        box_parked=False, box_drained=False) is None


# --------------------------------------------------------------------------- #
# _job_primary_evicted — the debounce off-by-one
# --------------------------------------------------------------------------- #
def test_primary_evicted_counts_this_tick_not_the_previous_streak():
    """`not_live_streak` is the counter BEFORE this tick's increment, so the
    verdict fires at `streak + 1 >= NOT_LIVE_DEBOUNCE` — the same tick the
    rescue trigger sees it. One off in either direction either reaps a warming
    understudy or fast-CUTOVERs off a live primary."""
    d = bidpolicy.NOT_LIVE_DEBOUNCE
    assert job_lane._job_primary_evicted(True, False, d - 2) is False
    assert job_lane._job_primary_evicted(True, False, d - 1) is True
    assert job_lane._job_primary_evicted(False, False, d - 1) is True
    # a live box is never evicted, whatever the streak says
    assert job_lane._job_primary_evicted(True, True, 99) is False


# --------------------------------------------------------------------------- #
# job_supervise_init — the state.json persistence contract (plan §4, FROZEN)
# --------------------------------------------------------------------------- #
def test_init_key_names_and_container_types_are_the_persistence_contract():
    """fleetd's `_replacement_state_restore` seeds a fresh `jc` off the durable
    watch record by these exact names, and round-trips
    `evicted_machines` set<->list with STRINGIFIED `evicted_machine_ts` keys.
    `jc` is deliberately not JSON-serialisable as built (`a` is a live
    Namespace) — that asymmetry is fleetd's to own, not this module's to
    smooth over."""
    a = _ns()
    jc, hf = job_lane.job_supervise_init(a)
    assert jc["a"] is a                                   # the LIVE namespace
    assert jc["iid"] == IID and isinstance(jc["iid"], str)
    assert isinstance(jc["evicted_machines"], set)        # list on disk, set here
    assert isinstance(jc["evicted_machine_ts"], dict)
    for key in ("replacements", "replacement_history", "replacement_refused",
                "launch_dph_anchor", "launch_disk_gb", "spend_usd",
                "floor_samples", "decay_streak", "not_live", "last_bid_put"):
        assert key in jc, key
    assert hf == handoff._init_job_handoff_state()


def test_init_seeds_every_unknown_price_as_None_never_zero():
    """Tri-state discipline at construction (defect #67): `0.0` reads as a
    known-zero price to every consumer downstream."""
    jc, _ = job_lane.job_supervise_init(_ns())
    for key in ("last_bid", "first_seen_dph", "launch_dph_anchor",
                "launch_disk_gb", "rescue_deadline", "replacement_refused",
                "was_live"):
        assert jc[key] is None, key
    assert jc["spend_usd"] == 0.0                     # a MEASURED zero, not UNKNOWN


def test_init_handoff_flags_fail_closed_and_serve_forces_handoff_off():
    """`handoff_can_complete` defaults False (defect #61 — a driver has to say
    it can finish a migration). `serve_mode` forces handoff OFF entirely:
    handoff is jobd-shaped (retarget tickets) and has no serve analog."""
    jc, _ = job_lane.job_supervise_init(
        argparse.Namespace(id=1, dry_run=False, budget=None, max_bid=None))
    assert jc["handoff_can_complete"] is False
    assert jc["handoff_on"] is True                   # handoff is the DEFAULT
    assert jc["handoff_unsafe_override"] is False
    assert job_lane.job_supervise_init(_ns(serve_mode=True))[0]["handoff_on"] is False
    assert job_lane.job_supervise_init(_ns(strict_ceiling=True))[0]["handoff_on"] is False


# --------------------------------------------------------------------------- #
# the small per-tick helpers
# --------------------------------------------------------------------------- #
def test_job_sup_inst_matches_on_stringified_ids():
    """vast returns `id` as an int; `jc["iid"]` is a str for the whole watch."""
    jc = {"instances": [{"id": 9000}, {"id": "9001"}]}
    assert job_lane._job_sup_inst(jc, "9000") == {"id": 9000}
    assert job_lane._job_sup_inst(jc, "9001") == {"id": "9001"}
    assert job_lane._job_sup_inst(jc, "1") is None
    assert job_lane._job_sup_inst({}, "9000") is None      # no fetch yet


def test_job_sup_reattach_keeps_its_two_argument_shape(monkeypatch):
    """Three flat tests stub this with `lambda jc, iid: None`. A port that grew
    a third parameter would break them at call time, not at import."""
    params = list(inspect.signature(job_lane._job_sup_reattach).parameters)
    assert params == ["jc", "iid"]
    w = _Wire(monkeypatch, inst=_inst())
    job_lane._job_sup_reattach({"a": _ns()}, IID)
    assert w.calls.count("attach") == 1


def test_job_sup_reattach_is_a_no_op_under_dry_run(monkeypatch, capsys):
    w = _Wire(monkeypatch, inst=_inst())
    job_lane._job_sup_reattach({"a": _ns(dry_run=True)}, IID)
    assert "attach" not in w.calls
    assert "[dry-run]" in capsys.readouterr().out


def test_job_sup_reattach_reaches_a_REAL_body_once_the_daemon_composes(
        monkeypatch, capsys):
    """THE DEFECT THIS PINS (live, 2026-08-17, three boxes in one window).

    Every other test on this function stubs `lifecycle.cmd_job_attach` — which
    is right for them, but it means the whole file was green for a year while
    the attribute was an UNBOUND seam in every real process: fleetd re-attached
    three times (47966341 resumed in place, 47967469 and 47974737 on the
    eviction ladder) and got

        !! jobd re-attach failed (NotImplementedError: cmd_job_attach: not
           ported yet (plan §8 step 5) ...) — box onstart revives jobd

    each time. Nothing failed, which is the problem: the guard three tests down
    swallows it, so the box came back holding its LAUNCH-BAKED B2 key and the
    rotation lane (CREDENTIAL_LIFECYCLE.md) silently did not run.

    So this one does NOT stub the call site. It composes the way `fleetd.run()`
    does and patches the OWNER, which is the steering contract in
    `cli/_compose.py`: a bind that had snapshotted the body at import would
    ignore this patch and the assertion would go vacuously green against the
    real ssh-pushing implementation."""
    from vastlib.cli import _compose
    from vastlib.cli.job import attach as job_attach

    w = _Wire(monkeypatch, inst=_inst())
    seen: list[object] = []
    monkeypatch.setattr(job_attach, "cmd_job_attach", lambda a: seen.append(a.id))
    _compose.bind()      # a real process composes AFTER the module is imported,
                         # so this legitimately overwrites _Wire's call-site stub
    job_lane._job_sup_reattach({"a": _ns()}, IID)

    assert seen == [int(IID)], "the seam never reached cli/job/attach.py"
    assert "attach" not in w.calls, "still resolving to the call-site stub"
    assert "re-attach failed" not in capsys.readouterr().out


def test_job_sup_reattach_still_raises_when_NOTHING_composed(monkeypatch, capsys):
    """The other direction, and the one that was true in production: with the
    seam left unbound the daemon prints its swallow line and rotates no key.

    Kept as a test rather than deleted with the fix, because the swallow is
    deliberate and permanent — an ssh refusal must never kill the babysitter —
    so this exact silence is what a SIXTH unbound seam would also produce. The
    guard is not the bug; an unbound seam behind it is."""
    _Wire(monkeypatch, inst=_inst())

    def _unbound(a):
        raise NotImplementedError(f"cmd_job_attach: {lifecycle._SEAM_HINT}")

    monkeypatch.setattr(lifecycle, "cmd_job_attach", _unbound)
    job_lane._job_sup_reattach({"a": _ns()}, IID)      # must not raise
    out = capsys.readouterr().out
    assert "re-attach failed (NotImplementedError" in out
    assert "onstart revives jobd" in out


@pytest.mark.parametrize("boom", [SystemExit("mid-boot"),
                                  RuntimeError("pubkey rejected")])
def test_job_sup_reattach_never_kills_the_babysitter(monkeypatch, capsys, boom):
    """Box 44514902 (2026-07-11): an ssh pubkey rejection raised out of the
    attach and killed supervise mid-rescue."""
    _Wire(monkeypatch, inst=_inst())
    monkeypatch.setattr(lifecycle, "cmd_job_attach",
                        lambda a: (_ for _ in ()).throw(boom))
    job_lane._job_sup_reattach({"a": _ns()}, IID)      # must not raise
    assert "re-attach failed" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# _job_market_read — ONE read per (machine, chunk, instant)
# --------------------------------------------------------------------------- #
def test_market_read_is_memoized_per_machine_and_instant(monkeypatch):
    """Three disagreeing reads used to reach `resume_in_place` as
    `min_bid=None, listed=True`, which it refuses — skipping the cheapest rung
    on the ladder."""
    reads = []
    monkeypatch.setattr(pricing, "_market_min_bid_read",
                        lambda mid, g=None: reads.append((mid, g))
                        or models.MarketRead(True, True, FLOOR))
    jc = {"now": 1000.0}
    inst = _inst()
    first = job_lane._job_market_read(jc, inst)
    assert job_lane._job_market_read(jc, inst) is first
    assert len(reads) == 1
    # a mid-tick box swap must NOT be handed the old machine's floor
    job_lane._job_market_read(jc, _inst(machine_id=8))
    assert len(reads) == 2
    jc["now"] = 1045.0                                  # next tick re-reads
    job_lane._job_market_read(jc, inst)
    assert len(reads) == 3


def test_market_read_without_a_machine_is_a_failed_read(monkeypatch):
    """`ok=False` is ignorance, and ignorance must not read as `listed=False`
    (which is the one positive signal an outbid emits, D7)."""
    monkeypatch.setattr(pricing, "_market_min_bid_read",
                        lambda mid, g=None: pytest.fail("no machine, no query"))
    r = job_lane._job_market_read({"now": 1.0}, {})
    assert (r.ok, r.listed, r.min_bid) == (False, False, None)


# --------------------------------------------------------------------------- #
# _JobLaneFloorHooks — the jobs-lane observation surface (WIRE FORMAT)
# --------------------------------------------------------------------------- #
def test_floor_hooks_take_iid_and_inst_and_subclass_the_shared_hooks():
    """`test_ladder_core.py:289` constructs it as `_JobLaneFloorHooks("700",
    inst)` and hands it to `ladder_core.self_floor_guard`."""
    import ladder_core

    h = job_lane._JobLaneFloorHooks("700", _inst())
    assert isinstance(h, ladder_core.LaneHooks)
    assert (h.iid, h.inst["machine_id"]) == ("700", MACHINE)


def test_floor_hooks_journal_the_jobs_lane_event_names(monkeypatch):
    """`jobs_bid_self_floor` / `jobs_bid_floor_blind` are what `fleet log` and
    the journal analytics key on, and they differ from the run lane's
    `bid_self_floor` / `bid_floor_blind` BY DESIGN — different event log,
    different identity. Patched on `journal`, which is also the late-binding
    proof: the hooks resolve it as a module global inside the method body."""
    seen = []
    monkeypatch.setattr(journal, "_job_ladder_journal",
                        lambda jc, event, **kw: seen.append((event, kw)))
    h = job_lane._JobLaneFloorHooks(IID, _inst())
    jc = {"last_bid": 0.60}
    match = bidpolicy.market_floor_self_match(0.60, 0.60, bid_history=(), now=1.0)
    h.self_floor(jc, market_min_bid=0.60, match=match,
                 surviving_floor=None, visible=False)
    h.floor_blind(jc, since_s=1800.0)
    assert [e for e, _ in seen] == ["jobs_bid_self_floor", "jobs_bid_floor_blind"]
    assert seen[0][1]["machine_id"] == MACHINE
    assert seen[0][1]["matched"] == match.kind
    assert seen[1][1]["since_s"] == 1800.0
    assert seen[1][1]["iid"] == IID


# --------------------------------------------------------------------------- #
# _job_resume_in_place — RUNG ZERO
# --------------------------------------------------------------------------- #
def _resume_jc(**over):
    jc = {"a": _ns(), "iid": IID, "dry_run": False, "spend_usd": 1.0,
          "last_bid": 0.60, "instances": [_inst(actual_status="stopped")],
          "now": 1000.0}
    jc.update(over)
    return jc


def test_resume_in_place_refuses_in_serve_mode_and_during_a_live_rescue(monkeypatch):
    """The serve lane's recovery is its own SLA relaunch spec, and re-issuing a
    `start` every 45 s while one is in flight is churn — the deadline is also
    what stops `dead` from renting out from under it."""
    w = _Wire(monkeypatch, inst=_inst())
    assert job_lane._job_resume_in_place(
        _resume_jc(serve_mode=True), _ns(), IID, FLOOR, True, True, 1000.0) is False
    assert job_lane._job_resume_in_place(
        _resume_jc(rescue_deadline=2000.0), _ns(), IID, FLOOR, True, True,
        1000.0) is False
    assert w.puts == []


def test_resume_in_place_issues_the_start_arms_the_deadline_and_journals(monkeypatch):
    """Box 47226953 (2026-08-09): our bid was still winning, the chunk was
    listed, and a `start` recovered a warm 59 GB disk in ~40 s against a
    replacement's measured 11m35s of setup."""
    w = _Wire(monkeypatch, inst=_inst(actual_status="stopped"))
    jc = _resume_jc()
    now = 1000.0
    assert job_lane._job_resume_in_place(
        jc, _ns(), IID, FLOOR, True, True, now) is True
    assert w.puts == [("state", IID, "running")]
    assert jc["resume_tries"] == 1
    assert jc["rescue_deadline"] == now + job_lane.JOB_SUP_RESCUE_WAIT_S
    assert [e for e, _ in w.journal] == ["jobs_box_resumed"]
    assert [e for e, _ in w.emits] == ["box_resume_in_place"]


def test_resume_in_place_declines_without_a_market_read_and_says_so_once(
        monkeypatch, capsys):
    """Ignorance never licenses a move: a bid box with no usable floor is left
    to the bid rungs (`market_min_bid=None`)."""
    w = _Wire(monkeypatch, inst=_inst(actual_status="stopped"))
    jc = _resume_jc()
    assert job_lane._job_resume_in_place(
        jc, _ns(), IID, None, None, True, 1000.0) is False
    first = capsys.readouterr().out
    assert "resume-in-place declined" in first
    assert job_lane._job_resume_in_place(
        jc, _ns(), IID, None, None, True, 1045.0) is False
    assert "resume-in-place declined" not in capsys.readouterr().out   # latched
    assert w.puts == []


def test_resume_in_place_under_dry_run_spends_no_try(monkeypatch, capsys):
    w = _Wire(monkeypatch, inst=_inst(actual_status="stopped"))
    jc = _resume_jc(dry_run=True)
    assert job_lane._job_resume_in_place(
        jc, _ns(dry_run=True), IID, FLOOR, True, True, 1000.0) is False
    assert "[dry-run]" in capsys.readouterr().out
    assert jc.get("resume_tries") in (None, 0) and w.puts == []


# --------------------------------------------------------------------------- #
# _job_announce_eviction — say it ONCE, at the moment we decide it
# --------------------------------------------------------------------------- #
def test_announce_eviction_journals_once_per_box_and_ends_the_floor_episode(
        monkeypatch):
    """Incident 2026-08-08 / task #74: seventeen bare `print()`s and nothing in
    `fleet log` for fourteen minutes. The announcement fires on the
    CLASSIFICATION, latches per box, and resets the self-floor episode (the
    market just spoke, so 'continuous suppression' is over by definition)."""
    w = _Wire(monkeypatch, inst=_inst(actual_status="exited"))
    resets = []
    monkeypatch.setattr(pricing, "_self_floor_reset", lambda jc: resets.append(1))
    jc = {"a": _ns(), "iid": IID, "last_bid": 0.60, "spend_usd": 2.0,
          "now": 1000.0, "pending_views": [_view()]}
    inst = _inst(actual_status="exited")
    ecls = job_lane._job_announce_eviction(
        jc, IID, inst, is_bid=True, present=True, astat="exited",
        intended_status="stopped", claimed_work=True, budget=100.0)
    assert ecls is not None
    assert [e for e, _ in w.journal] == ["jobs_box_evicted"]
    assert [e for e, _ in w.emits] == ["box_evicted"]
    assert resets == [1]
    fields = w.journal[0][1]
    # the two halves of the old `None`, kept apart on purpose
    assert fields["market_read_ok"] is True and fields["market_listed"] is True
    assert fields["claimed_work"] is True
    # second call in the same eviction cycle is silent
    assert job_lane._job_announce_eviction(
        jc, IID, inst, is_bid=True, present=True, astat="exited",
        intended_status="stopped", claimed_work=True, budget=100.0) is None
    assert len(w.journal) == 1


# --------------------------------------------------------------------------- #
# the unclaimed seams
# --------------------------------------------------------------------------- #
def test_box_lifecycle_soft_delegates_to_jobs_view(monkeypatch):
    """The seam CLOSED (6f census): its body landed at jobs/view and this
    module now carries the same one-line forwarder handoff.py uses. The raising
    stub had made every `job supervise` tick on a non-serve jobs box raise —
    and the fold it guards is what keeps a self-park scored as success, so the
    stub's failure mode was a parked box getting rescue-resumed. Prove the
    routing (a re-grown local copy would be a second patch point that lies)."""
    from vastlib.jobs import view as jobs_view
    seen = []
    monkeypatch.setattr(jobs_view, "_box_lifecycle_soft",
                        lambda iid: (seen.append(iid), {"parked": True,
                                                        "drained_pending": False})[1])
    out = job_lane._box_lifecycle_soft(IID)
    assert seen == [IID] and out["parked"] is True


def test_the_two_step6_seams_are_gone_from_this_module():
    """`_sticky_on_demand` and `_serve_self_park_soft` landed at step 6 —
    `market.pricing` and `supervise.replacement` respectively — and the tick
    calls them by module attribute. This module must NOT keep a same-named
    attribute: a leftover alias is a second patch point, and a test that steered
    it would go green while the tick read the real function."""
    assert not hasattr(job_lane, "_sticky_on_demand")
    assert not hasattr(job_lane, "_serve_self_park_soft")
    assert callable(pricing._sticky_on_demand)
    assert callable(replacement._serve_self_park_soft)


# --------------------------------------------------------------------------- #
# job_supervise_tick — the verdicts (the control contract both drivers branch on)
# --------------------------------------------------------------------------- #
def test_verdict_set_is_the_control_contract():
    """`cmd_job_supervise` exits 3 on `unrecoverable` and fleetd's profiles
    branch on the same strings, so the tuple travels with the tick."""
    assert job_lane.JOB_SUP_VERDICTS == (
        "self_parked", "operator_park", "budget", "drained", "queue_empty",
        "unrecoverable", "sla_relaunched", "identity_mismatch")


def test_tick_on_a_healthy_live_box_returns_None_and_keeps_supervising(monkeypatch):
    """The baseline shape every other tick test perturbs: live bid box, one
    running ticket, nothing to decide."""
    w, jc, hf, verdict = _tick(monkeypatch, inst=_inst(),
                               queue=["job-a"], views=[_view()])
    assert verdict is None
    assert jc["was_live"] is True
    assert jc["pending_views"] == [_view()]
    assert ("stop", IID) not in w.puts


def test_tick_exits_queue_empty_when_the_box_has_no_tickets(monkeypatch):
    """`test_pull_watchdog.py` asserts this exact string off the flat copy."""
    _, _, _, verdict = _tick(monkeypatch, inst=_inst(), queue=[])
    assert verdict == "queue_empty"
    assert verdict in job_lane.JOB_SUP_VERDICTS


def _queue_boom(box, **k):
    raise jobmeta.QueueUnreadable(
        "queue listing for box 4 FAILED (rc=1) — this is not an empty queue: "
        "dial tcp: lookup example.invalid: no such host")


def test_an_unreadable_queue_never_exits_queue_empty(monkeypatch, capsys):
    """The 2026-08-22 defect: `list_queue` answered `[]` on a FAILED listing, so
    a workstation with a clobbered rclone [b2] remote read every box as having
    no work and stopped defending box 48392137 while it held a live ticket."""
    w = _Wire(monkeypatch, inst=_inst(), queue=[])
    monkeypatch.setattr(jobmeta, "list_queue", _queue_boom)   # after the wire
    jc, hf = job_lane.job_supervise_init(_ns())
    verdict = job_lane.job_supervise_tick(jc, hf)
    assert verdict != "queue_empty"
    assert ("stop", IID) not in w.puts
    out = capsys.readouterr().out
    assert "QUEUE UNREADABLE" in out and "example.invalid" in out


def test_an_unreadable_queue_never_parks_the_box_as_drained(monkeypatch):
    """The other exit that stops defending the box. Zero readable tickets is
    also zero TERMINAL tickets, so `drained` must be suppressed too."""
    w = _Wire(monkeypatch, inst=_inst(), queue=["job-a"],
              views=[_view(status="done", display_status="done")])
    monkeypatch.setattr(jobmeta, "list_queue", _queue_boom)
    jc, hf = job_lane.job_supervise_init(_ns())
    assert job_lane.job_supervise_tick(jc, hf) != "drained"
    assert ("stop", IID) not in w.puts


def test_the_refusal_tells_the_operator_where_to_look(monkeypatch, capsys):
    """A refusal belongs where the operator can still act: the failing listing
    verbatim, plus the two things that are actually wrong when it fires."""
    _Wire(monkeypatch, inst=_inst(), queue=[])
    monkeypatch.setattr(jobmeta, "list_queue", _queue_boom)
    jc, hf = job_lane.job_supervise_init(_ns())
    job_lane.job_supervise_tick(jc, hf)
    out = capsys.readouterr().out
    assert "rclone lsf b2:$B2_BUCKET/jobs/queue/" in out
    assert "rclone.conf" in out


def test_tick_parks_the_box_on_a_drained_queue_unless_keep(monkeypatch):
    """Every ticket terminal == the watch's work is done; `--keep` leaves the
    box running for a human, and neither arm is an eviction."""
    done = _view(status="done", display_status="done")
    w, _, _, verdict = _tick(monkeypatch, inst=_inst(), queue=["job-a"],
                             views=[done])
    assert verdict == "drained" and ("stop", IID) in w.puts
    w2, _, _, verdict2 = _tick(monkeypatch, a=_ns(keep=True), inst=_inst(),
                               queue=["job-a"], views=[done])
    assert verdict2 == "drained" and w2.puts == []


def test_tick_budget_exit_parks_the_box_before_any_ladder_rung(monkeypatch):
    """The hard spend cap. `spend_usd` is seeded past the cap so the exit is
    reached on the first tick rather than after an hour of accrual."""
    w = _Wire(monkeypatch, inst=_inst(), queue=["job-a"], views=[_view()])
    jc, hf = job_lane.job_supervise_init(_ns(budget=1.0))
    jc["spend_usd"] = 5.0
    assert job_lane.job_supervise_tick(jc, hf) == "budget"
    assert ("stop", IID) in w.puts


def test_tick_scores_a_self_parked_box_as_success_not_as_a_loss(monkeypatch):
    """jobd self-parked on queue drain: the box-event stream is consulted FIRST
    (SPOT_DESIGN §3.5) and no eviction is announced."""
    w, _, _, verdict = _tick(monkeypatch, inst=_inst(actual_status="stopped",
                                                     intended_status="stopped"),
                             queue=["job-a"], views=[_view()], parked=True)
    assert verdict == "self_parked"
    assert w.journal == [] and w.emits == []


def test_tick_announces_an_unexplained_stop_as_an_eviction(monkeypatch, capsys):
    """Nothing explains the stop and nobody asked — that is an EVICTION, said
    once, in the journal, at the moment it is classified."""
    w, jc, _, verdict = _tick(
        monkeypatch, inst=_inst(actual_status="exited",
                                intended_status="stopped"),
        queue=["job-a"], views=[_view()])
    assert verdict is None                       # the rescue ladder runs next
    assert [e for e, _ in w.journal if e == "jobs_box_evicted"]
    assert "EVICTION" in capsys.readouterr().out
    assert jc["evicted_announced"] == IID        # latched for this cycle


# --------------------------------------------------------------------------- #
# the five jc-published risk metrics — None-for-UNKNOWN at the KEY level
# --------------------------------------------------------------------------- #
def test_tick_publishes_unknown_risk_metrics_as_None(monkeypatch):
    """`bidpolicy` (Zone S) reads these BY KEY with no import edge, so nothing
    type-checks them. A `0.0` substitution turns a migration REFUSAL into a
    migration (defect #67) — the assertions are `is None`, not falsy."""
    _, jc, _, verdict = _tick(monkeypatch, inst=_inst(), queue=["job-a"],
                              views=[_view()])
    assert verdict is None
    for key in ("work_at_risk_h", "running_unresumable", "min_running_eta_s",
                "ckpt_stale", "remaining_wall_h", "timeout_ceiling_h"):
        assert key in jc, key
        assert jc[key] is None, f"{key} must stay None for UNKNOWN"


def test_tick_publishes_measured_risk_metrics_unchanged(monkeypatch):
    """...and a MEASURED zero is passed through as a zero: the rule is
    None-for-UNKNOWN, not never-zero."""
    metrics = {"_jobs_work_horizon_h": 0.75, "_jobs_remaining_wall_h": 9.9,
               "_jobs_work_at_risk_h": 0.0, "_jobs_unresumable_running": True,
               "_jobs_min_running_eta_s": 600.0, "_jobs_ckpt_stale": False}
    _, jc, _, _ = _tick(monkeypatch, inst=_inst(), queue=["job-a"],
                        views=[_view()], risk_metrics=metrics)
    assert jc["remaining_wall_h"] == 0.75          # the WORK horizon
    assert jc["timeout_ceiling_h"] == 9.9          # the hang-detector ceiling
    assert jc["work_at_risk_h"] == 0.0
    assert jc["running_unresumable"] is True
    assert jc["min_running_eta_s"] == 600.0
    assert jc["ckpt_stale"] is False


def test_tick_reads_the_ckpt_alarm_through_the_risk_module(monkeypatch, capsys):
    """Late binding, on the watchdog that is never money-moving but is the J1
    incident's only observable."""
    w = _Wire(monkeypatch, inst=_inst(), queue=["job-a"], views=[_view()])
    monkeypatch.setattr(risk, "_ckpt_watchdog_alarm",
                        lambda v, now, **k: "job-a silent 3x checkpoint_s")
    jc, hf = job_lane.job_supervise_init(_ns())
    assert job_lane.job_supervise_tick(jc, hf) is None
    assert "CKPT-STALL job-a silent" in capsys.readouterr().out
    assert w.calls.count("progress_warn") == 1


# --------------------------------------------------------------------------- #
# the handoff promotion — `iid` moves mid-body
# --------------------------------------------------------------------------- #
def test_tick_promotes_the_understudy_and_resets_the_per_box_latches(monkeypatch):
    """`jc.pop("_handoff_completed_iid")` moves the watch to the survivor, and
    the local `iid` must move with `jc["iid"]`. Per-box latches (echo window,
    decay streak, rebid rungs, the eviction announcement) do NOT survive a box
    swap, and `last_bid` is re-seeded from the understudy's `dph_base` or left
    None — writing the popped `dph_total` put the belief one storage sliver
    above every echo and the ladder defended against itself (review 2026-08-10,
    H1)."""
    understudy = _inst(id=9100, machine_id=8, dph_base=0.41, dph_total=0.43)
    w = _Wire(monkeypatch, inst=_inst(), queue=["job-a"], views=[_view()])
    w.handoff_completed_iid = "9100"
    jc, hf = job_lane.job_supervise_init(_ns())
    jc["evicted_announced"] = IID
    jc["decay_streak"], jc["rebid_rungs"] = 3, 2
    monkeypatch.setattr(lifecycle, "_instances_soft",
                        lambda: [_inst(), understudy])
    assert job_lane.job_supervise_tick(jc, hf) is None
    assert jc["iid"] == "9100"
    assert jc["last_bid"] == 0.41 and jc["first_seen_dph"] == 0.41
    assert jc["floor_samples"] == []
    assert (jc["not_live"], jc["rescue_deadline"], jc["was_live"]) == (0, None, None)
    assert (jc["decay_streak"], jc["rebid_rungs"]) == (0, 0)
    assert "evicted_announced" not in jc
    assert "_handoff_completed_dph" not in jc


def test_tick_leaves_last_bid_None_when_the_understudy_carries_no_dph_base(
        monkeypatch):
    """Fail-closed: moves stay disabled until the reconcile path seeds from a
    body that carries `dph_base`."""
    understudy = {"id": 9100, "actual_status": "running", "machine_id": 8,
                  "is_bid": False, "dph_total": 0.43}
    w = _Wire(monkeypatch, inst=_inst(), queue=["job-a"], views=[_view()])
    w.handoff_completed_iid = "9100"
    monkeypatch.setattr(lifecycle, "_instances_soft",
                        lambda: [_inst(), dict(understudy)])
    jc, hf = job_lane.job_supervise_init(_ns())
    assert job_lane.job_supervise_tick(jc, hf) is None
    assert jc["last_bid"] is None and jc["first_seen_dph"] is None


# --------------------------------------------------------------------------- #
# the INLINE emit — the one the _job_handoff_emit patch does not intercept
# --------------------------------------------------------------------------- #
def _over_ceiling_wire(monkeypatch, **kw):
    """A standing bid ABOVE the preferred ceiling, which is `0.75 x on_demand`.
    The bid is seeded onto `jc` rather than derived, because the current policy
    would refuse to EMIT it (the 2026-08-09 hard clamp) — the reachable shape
    is a bid from before the clamp, a hand `herdd bid --price`, or an
    on-demand price that has since fallen."""
    return _Wire(monkeypatch, inst=_inst(dph_base=1.10, dph_total=1.12),
                 queue=["job-a"], views=[_view()], on_demand=1.0, **kw)


def test_tick_emits_the_ceiling_alarm_inline_bypassing_the_handoff_emit(
        monkeypatch):
    """Ported as-is (plan §7.4): `jobmeta.emit_box_event` is called DIRECTLY, so
    a `_job_handoff_emit` monkeypatch does not intercept it. That is the whole
    reason it is called out — an emit-level regression net has to patch
    `jobmeta`, one level down."""
    w = _over_ceiling_wire(monkeypatch)
    jc, hf = job_lane.job_supervise_init(_ns())
    assert job_lane.job_supervise_tick(jc, hf) is None
    assert [(ev, kw["actor"]) for _, ev, kw in w.box_events] == [
        ("bid_over_preferred_ceiling", "job-supervise")]
    assert not [e for e, _ in w.emits if e == "bid_over_preferred_ceiling"]
    assert jc["pref_alarmed"] is True
    # latched: the second tick says nothing
    assert job_lane.job_supervise_tick(jc, hf) is None
    assert len(w.box_events) == 1


def test_a_failed_inline_emit_prints_and_never_kills_the_tick(monkeypatch, capsys):
    """The jobs lane's inline emit PRINTS its exception where the run lane's
    `_sup_emit` swallows it into `{"_emitted": False}` — a divergence, and the
    port keeps it."""
    _over_ceiling_wire(monkeypatch)
    monkeypatch.setattr(jobmeta, "emit_box_event",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("b2 down")))
    jc, hf = job_lane.job_supervise_init(_ns())
    assert job_lane.job_supervise_tick(jc, hf) is None
    assert "bid_over_preferred_ceiling emit failed (b2 down)" in \
        capsys.readouterr().out


# --------------------------------------------------------------------------- #
# serve_mode — a second personality inside ONE function
# --------------------------------------------------------------------------- #
def test_serve_mode_never_reads_the_jobd_queue(monkeypatch):
    """No queue means no `drained`/`queue_empty` exit and no horizon: the box
    itself is the workload. Splitting the tick by lane to express this would
    fork a third copy of the ladder."""
    w = _Wire(monkeypatch, inst=_inst(), queue=[], views=[])
    # patched AFTER the wire, so this is the binding the tick would reach
    monkeypatch.setattr(jobmeta, "list_queue",
                        lambda box, **k: pytest.fail("serve lane has no queue"))
    jc, hf = job_lane.job_supervise_init(_ns(serve_mode=True))
    assert job_lane.job_supervise_tick(jc, hf) is None
    assert jc["pending_views"] == []
    assert w.calls.count("handoff_tick") == 0     # handoff is OFF for serve


def test_serve_mode_reads_its_self_park_from_the_serve_status_marker(monkeypatch):
    """Without this read every MAX_HOURS watchdog park on a BID serve box is
    misread as OUTBID and rescue-resumed forever."""
    _, _, _, verdict = _tick(
        monkeypatch, a=_ns(serve_mode=True),
        inst=_inst(actual_status="stopped", intended_status="stopped"),
        serve_parked=True)
    assert verdict == "self_parked"


# --------------------------------------------------------------------------- #
# the bottom of the ladder
# --------------------------------------------------------------------------- #
def test_tick_reports_unrecoverable_only_after_every_rung_refuses(monkeypatch,
                                                                  capsys):
    """The box has left the listing entirely (host death), the re-bid ladder and
    the automatic replacement both refuse, and only THEN does the operator get
    the manual retarget checklist — `cmd_job_supervise` exits 3 on this."""
    w = _Wire(monkeypatch, inst=_inst(), queue=["job-a"], views=[_view()])
    refusals = []
    monkeypatch.setattr(replacement, "_job_rebid_ladder",
                        lambda *a, **k: refusals.append("rebid") or False)
    monkeypatch.setattr(replacement, "_job_eviction_replace",
                        lambda *a, **k: refusals.append("replace") or False)
    jc, hf = job_lane.job_supervise_init(_ns())
    assert job_lane.job_supervise_tick(jc, hf) is None          # tick 1: live
    monkeypatch.setattr(lifecycle, "_instances_soft", lambda: [])   # box gone
    w.inst = None
    verdicts = [job_lane.job_supervise_tick(jc, hf)
                for _ in range(bidpolicy.NOT_LIVE_DEBOUNCE)]
    assert verdicts[-1] == "unrecoverable"
    assert refusals == ["replace"]        # `present` is False: no re-bid rung
    out = capsys.readouterr().out
    assert "unrecoverable: box gone" in out and "job retarget job-a" in out


def test_a_replacement_that_explodes_never_kills_the_babysitter(monkeypatch,
                                                                capsys):
    """Every rung below the bid is wrapped: a raising replacement path leaves
    the refusal on `jc` and falls through to the manual instructions."""
    w = _Wire(monkeypatch, inst=_inst(), queue=["job-a"], views=[_view()])
    monkeypatch.setattr(replacement, "_job_eviction_replace",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("offers 503")))
    jc, hf = job_lane.job_supervise_init(_ns())
    assert job_lane.job_supervise_tick(jc, hf) is None
    monkeypatch.setattr(lifecycle, "_instances_soft", lambda: [])
    w.inst = None
    for _ in range(bidpolicy.NOT_LIVE_DEBOUNCE - 1):
        assert job_lane.job_supervise_tick(jc, hf) is None
    assert job_lane.job_supervise_tick(jc, hf) == "unrecoverable"
    assert jc["replacement_refused"] == "RuntimeError: offers 503"
    assert "eviction replacement errored" in capsys.readouterr().out


def test_a_replacement_that_takes_the_box_keeps_the_watch_alive(monkeypatch):
    """The rung below the bid (owner directive 2026-08-05): rent a different box
    rather than hand the operator a checklist. The tick returns None because the
    watch is now supervising the replacement."""
    w = _Wire(monkeypatch, inst=_inst(), queue=["job-a"], views=[_view()])
    monkeypatch.setattr(replacement, "_job_eviction_replace",
                        lambda *a, **k: True)
    jc, hf = job_lane.job_supervise_init(_ns())
    assert job_lane.job_supervise_tick(jc, hf) is None
    monkeypatch.setattr(lifecycle, "_instances_soft", lambda: [])
    w.inst = None
    verdicts = [job_lane.job_supervise_tick(jc, hf)
                for _ in range(bidpolicy.NOT_LIVE_DEBOUNCE)]
    assert verdicts == [None] * bidpolicy.NOT_LIVE_DEBOUNCE
    assert jc["was_live"] is False


# --------------------------------------------------------------------------- #
# the port contract itself
# --------------------------------------------------------------------------- #
def test_seams_are_reached_through_their_owning_module(monkeypatch):
    """The late-binding proof. Every one of these stubs lives on a DIFFERENT
    module than `job_lane`; if the port had written `from .journal import
    _job_ladder_journal` (or any other `from … import`), the stub would never be
    called and every assertion above would be green for the wrong reason."""
    w, jc, _, verdict = _tick(monkeypatch, inst=_inst(), queue=["job-a"],
                              views=[_view()])
    assert verdict is None
    for expected in ("retention",          # vastlib.supervise.retention
                     "reconcile",          # vastlib.supervise.handoff
                     "progress_warn",      # vastlib.supervise.handoff
                     "handoff_tick",       # vastlib.supervise.handoff
                     "palt",               # vastlib.supervise.replacement
                     "_jobs_work_at_risk_h"):   # vastlib.jobs.risk
        assert expected in w.calls, expected


def test_a_retention_sweep_that_explodes_is_swallowed(monkeypatch, capsys):
    """It follows retained boxes to a terminal outcome; it must never kill the
    babysitter, because the boxes still self-expire via their keep label."""
    _Wire(monkeypatch, inst=_inst(), queue=["job-a"], views=[_view()])
    monkeypatch.setattr(retention, "_job_retention_sweep",
                        lambda jc, now: (_ for _ in ()).throw(
                            RuntimeError("b2 list failed")))
    jc, hf = job_lane.job_supervise_init(_ns())
    assert job_lane.job_supervise_tick(jc, hf) is None
    assert "retention sweep errored" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# S2b — vast's own outbid record as EVIDENCE (peers 830579df..d5b0b773,
# re-ported onto the package copy 2026-08-16).
#
# The flat suite `test_notify_policy.py` drives the same properties against the
# still-live `herdd` copies and stays UNEDITED (plan §8 add-only amendment).
# This is the parallel net for the package copies, built on the same harness as
# everything above it.
# --------------------------------------------------------------------------- #
NOTIFY_OD = 3.0                     # a machine whose on-demand rate leaves room
ROW_BID, ROW_FLOOR = 0.45, 1.00     # the 2026-08-16 field box, in one line


def _row(iid=IID, your_bid=ROW_BID, new_min_bid=ROW_FLOOR, event_id="e1",
         created_at=None, machine_id=MACHINE):
    """A matched-evidence record, in exactly the shape the driver hands over
    (`notify.outbid_evidence`'s output, which is also what `state.json` returns
    on a restart — one matcher serves both)."""
    return {"event_id": event_id, "iid": str(iid), "machine_id": machine_id,
            "your_bid": your_bid, "new_min_bid": new_min_bid,
            "created_at": time.time() if created_at is None else created_at}


def _evict(monkeypatch, *, rows=(), anchor=None, p_alt=None, ticks=None,
           on_demand=NOTIFY_OD, views=None, **wire_kw):
    """One live tick, then enough not-live ticks to walk the whole ladder.

    Every rung BELOW the rescue is held off: rung zero spends nothing and reads
    no notification, and the re-bid / replacement rungs have their own nets. So
    what this measures is exactly the arm S2b changed."""
    # `dph_base` IS our standing bid — `ladder_core.reconcile_standing_bid`
    # rewrites `jc["last_bid"]` from it on every tick, so setting the belief by
    # hand would be reconciled away and every row would look like a bid
    # disagreement. The box is priced at the row's `your_bid`.
    w = _Wire(monkeypatch, inst=_inst(dph_base=ROW_BID, dph_total=ROW_BID),
              queue=["job-a"], views=views or [_view()],
              on_demand=on_demand, listed=False, floor=None, **wire_kw)
    monkeypatch.setattr(job_lane, "_job_resume_in_place", lambda *a, **k: False)
    monkeypatch.setattr(replacement, "_job_rebid_ladder", lambda *a, **k: False)
    monkeypatch.setattr(replacement, "_job_eviction_replace",
                        lambda *a, **k: False)
    jc, hf = job_lane.job_supervise_init(_ns())
    job_lane.job_supervise_tick(jc, hf)                 # tick 1: live
    if anchor is not None:
        jc["launch_dph_anchor"] = anchor
    if p_alt is not None:
        jc["p_alt"], jc["p_alt_ts"] = p_alt, time.time()
    if rows:
        jc["notify_rows"] = [dict(r) for r in rows]
    w.inst = _inst(actual_status="exited", intended_status="stopped",
                   dph_base=ROW_BID, dph_total=ROW_BID)
    n = ticks if ticks is not None else 2 * bidpolicy.NOT_LIVE_DEBOUNCE
    for _ in range(n):
        job_lane.job_supervise_tick(jc, hf)
        jc["last_bid_put"] = 0.0        # the rate limiter is not under test
    return w, jc, hf


def _journaled(w, name):
    got = [f for e, f in w.journal if e == name]
    return got[0] if got else None


# --- the lookaside itself: latch, consumed set, and their two lifetimes ----- #
def test_a_matched_row_is_latched_for_the_whole_eviction_cycle():
    """Seventeen not-live ticks are ONE eviction, so they are also one match:
    `_job_notify_match` returns the LATCH once one exists, and never re-matches
    a different row mid-cycle. Same discipline as `evicted_announced`, and for
    the same reason."""
    jc = {"now": time.time(), "notify_rows": [_row(event_id="e1")]}
    first = job_lane._job_notify_match(jc, IID)
    assert first is not None and first["event_id"] == "e1"
    job_lane._job_notify_consume(jc, first)
    jc["notify_rows"] = [_row(event_id="e2", new_min_bid=9.99)]
    again = job_lane._job_notify_match(jc, IID)
    assert again["event_id"] == "e1", "a newer row must not re-price the cycle"
    assert job_lane._job_notify_latched(jc, IID)["event_id"] == "e1"
    assert job_lane._job_notify_latched(jc, "other-box") is None


def test_the_consumed_set_is_bounded_and_dedupes():
    jc = {"now": time.time()}
    for i in range(job_lane.NOTIFY_CONSUMED_MAX + 5):
        job_lane._job_notify_mark_consumed(jc, f"e{i}")
    job_lane._job_notify_mark_consumed(jc, "e0")        # already spent, long ago
    assert len(jc["notify_consumed_ids"]) == job_lane.NOTIFY_CONSUMED_MAX
    assert jc["notify_consumed_ids"][-1] == "e0"        # re-added, still bounded
    job_lane._job_notify_mark_consumed(jc, None)        # a row with no id
    assert len(jc["notify_consumed_ids"]) == job_lane.NOTIFY_CONSUMED_MAX


@pytest.mark.parametrize("drift", [None, "e1", 7, {"e1": 1}])
def test_the_consumed_set_degrades_rather_than_wedging_a_watch(drift):
    """`notify_consumed_ids` is DURABLE (`REPLACEMENT_STATE_KEYS`), so a
    hand-edited or schema-drifted `state.json` can hand back any JSON shape.
    A non-iterable there used to raise INSIDE the tick, which fleetd catches
    per-watch as `watch_error` — one box wedged forever, never rescued, because
    a string was an int. The guard degrades to "less memory", never to a dead
    watch."""
    jc = {"notify_consumed_ids": drift, "now": time.time(),
          "notify_rows": [_row()]}
    assert job_lane._job_notify_consumed_ids(jc) == [], (
        "a shape that is not a list/tuple/set is LESS MEMORY, not an exception "
        "— note a bare string is iterable and would otherwise be read as a set "
        "of characters")
    assert job_lane._job_notify_match(jc, IID) is not None   # no raise
    # ...and the good shape is still read, coerced to str for comparison
    jc["notify_consumed_ids"] = ["e1"]
    assert job_lane._job_notify_consumed_ids(jc) == ["e1"]
    assert job_lane._job_notify_match(jc, IID) is None, "the row was spent"


def test_a_cycle_reset_keeps_the_consumed_set_and_a_swap_retires_it():
    """The two lifetimes, side by side. The latch clears on return-to-live; the
    consumed set does not, because the freshness window cannot see a box
    evicted, rescued and evicted again inside fifteen minutes — and cycle 2
    priced off cycle 1's row is a wrong number on a money-moving rung."""
    jc = {"notify_matched": _row(), "notify_consumed_ids": ["e1"],
          "notify_quote_said": True}
    job_lane._job_notify_cycle_reset(jc)
    assert jc["notify_consumed_ids"] == ["e1"]
    assert "notify_matched" not in jc and "notify_quote_said" not in jc
    jc["notify_matched"] = _row()
    job_lane._job_notify_box_swap_reset(jc)
    assert "notify_consumed_ids" not in jc and "notify_matched" not in jc


# --- §6.1, both halves: what a row is allowed to price --------------------- #
@pytest.mark.parametrize("nmb,od,want", [
    (1.00, NOTIFY_OD, 1.00),        # your_bid < new_min_bid < on_demand
    (0.30, NOTIFY_OD, None),        # BELOW our own bid: not a price a bid beats
    (4.00, NOTIFY_OD, None),        # at/above on-demand: displacement of
                                    # unknown class, and `outbid` shortens a
                                    # machine exclusion
    (1.00, None, 1.00),             # no on-demand read ANYWHERE: the documented
                                    # degradation, and it is PERMISSIVE — the
                                    # predicate cannot refuse on a bound it has
                                    # no number for. The money path's refusals
                                    # (`rescue_target > last_bid`, the anchor
                                    # ceiling, `notify_rescue_bound`) are the
                                    # backstop, and they still bind.
])
def test_a_row_prices_the_rescue_only_where_the_predicate_supports_it(
        nmb, od, want):
    jc = {"notify_matched": _row(new_min_bid=nmb), "last_bid": ROW_BID,
          "on_demand_last": od}
    assert job_lane._job_notify_rescue_min_bid(jc, IID) == want


def test_the_sticky_on_demand_read_is_the_default_not_a_missing_clamp():
    """A failed on-demand probe must not silently UNCLAMP the predicate — that
    is the whole lesson of the four boxes that bid over on-demand. The explicit
    argument wins where the tick has one; `on_demand_last` is the fallback."""
    jc = {"notify_matched": _row(), "last_bid": ROW_BID, "on_demand_last": 0.5}
    assert job_lane._job_notify_rescue_min_bid(jc, IID) is None
    assert job_lane._job_notify_rescue_min_bid(jc, IID, on_demand=3.0) == 1.00


# --- the late match, and the gate that says WHICH box announced ------------ #
def test_a_row_that_lands_after_the_stop_is_still_matched(monkeypatch):
    """The two observations race. The 2026-08-16 case only happened to have
    vast's row seventeen seconds early, and nothing makes that the rule — so the
    match is retried on EVERY not-live tick of the cycle, not only on the tick
    that announced it."""
    w = _Wire(monkeypatch, inst=_inst(), queue=["job-a"], views=[_view()],
              on_demand=NOTIFY_OD, listed=False, floor=None)
    monkeypatch.setattr(job_lane, "_job_resume_in_place", lambda *a, **k: False)
    monkeypatch.setattr(replacement, "_job_rebid_ladder", lambda *a, **k: False)
    monkeypatch.setattr(replacement, "_job_eviction_replace",
                        lambda *a, **k: False)
    jc, hf = job_lane.job_supervise_init(_ns())
    job_lane.job_supervise_tick(jc, hf)
    jc["last_bid"] = ROW_BID
    w.inst = _inst(actual_status="exited", intended_status="stopped")
    job_lane.job_supervise_tick(jc, hf)                 # announced, no rows yet
    assert jc["evicted_announced"] == IID
    assert _journaled(w, "notify_outbid_matched") is None
    jc["notify_rows"] = [_row()]                        # ...the row lands LATE
    for _ in range(2 * bidpolicy.NOT_LIVE_DEBOUNCE):
        job_lane.job_supervise_tick(jc, hf)
        jc["last_bid_put"] = 0.0
    m = _journaled(w, "notify_outbid_matched")
    assert m is not None and m["match_path"] == "late"
    assert m["floor_source"] == "guarded", (
        "the late path passes the SELF-FLOOR-GUARDED read; §6.5's calibration "
        "is only scoreable if the row says which floor it carries")


def test_the_late_match_is_gated_on_THIS_boxs_announcement(monkeypatch):
    """Peer 73c44cb3. `evicted_announced` carries the box id it announced, and
    testing it for mere TRUTHINESS would let a retry run against a watch whose
    ladder had already moved to a different instance. Harmless today — the match
    key is the instance id, so it would find nothing — and exactly the near-miss
    that becomes a defect the next time a seam moves."""
    w = _Wire(monkeypatch, inst=_inst(), queue=["job-a"], views=[_view()],
              on_demand=NOTIFY_OD, listed=False, floor=None)
    monkeypatch.setattr(job_lane, "_job_resume_in_place", lambda *a, **k: False)
    monkeypatch.setattr(replacement, "_job_rebid_ladder", lambda *a, **k: False)
    monkeypatch.setattr(replacement, "_job_eviction_replace",
                        lambda *a, **k: False)
    tried = []
    real = job_lane._job_notify_try_match
    monkeypatch.setattr(
        job_lane, "_job_notify_try_match",
        lambda jc, iid, cl, **k: tried.append(k["match_path"]) or real(
            jc, iid, cl, **k))
    # The ANNOUNCE path is held off so it cannot re-latch `evicted_announced`
    # to this box mid-tick and mask the gate under test. It has its own tests
    # above; what is measured here is the LATE retry's gate and nothing else.
    monkeypatch.setattr(job_lane, "_job_announce_eviction",
                        lambda jc, iid, inst, **k: None)
    jc, hf = job_lane.job_supervise_init(_ns())
    job_lane.job_supervise_tick(jc, hf)
    w.inst = _inst(actual_status="exited", intended_status="stopped",
                   dph_base=ROW_BID, dph_total=ROW_BID)
    jc["notify_rows"] = [_row()]
    # a TRUTHY announcement naming somebody else — the shape the fix rules out.
    # `bool("8888")` is True, so a truthiness test would arm the retry against a
    # watch whose ladder had already moved to a different instance.
    jc["evicted_announced"] = "8888"
    tried.clear()
    job_lane.job_supervise_tick(jc, hf)
    assert "late" not in tried, "a foreign announcement must not arm the retry"
    assert _journaled(w, "notify_outbid_matched") is None
    jc["evicted_announced"] = IID
    job_lane.job_supervise_tick(jc, hf)
    assert "late" in tried, "...and OUR announcement must"
    assert _journaled(w, "notify_outbid_matched") is not None


def test_the_announcement_journals_both_classifications(monkeypatch):
    """The field question this slice exists to answer is not "what is the class"
    but "how often does vast's own record disagree with what we inferred". A log
    carrying only the answer we ADOPTED can never decide whether §6.2 earned its
    precedence, so both are journaled off ONE set of reads."""
    w, jc, _ = _evict(monkeypatch, rows=[_row()], ticks=1)
    m = _journaled(w, "notify_outbid_matched")
    assert m is not None and m["match_path"] == "announce"
    assert m["floor_source"] == "raw"
    assert set(m) >= {"class_without_notify", "class_with_notify", "event_id"}
    assert m["class_with_notify"] == bidpolicy.EVICTION_OUTBID
    ev = _journaled(w, "jobs_box_evicted")
    assert ev["eviction_class"] == bidpolicy.EVICTION_OUTBID
    assert ev["notify_event_id"] == "e1"


def test_no_row_leaves_the_eviction_event_exactly_its_pre_s2b_shape(monkeypatch):
    """D2 in one line: no rows, no difference. `notify_event_id` is present ONLY
    when a row matched — an unconditional `None` would change the shape of every
    eviction event ever emitted on a fleet with no notifications, which is
    precisely the boundary S2b promised not to cross."""
    w, jc, _ = _evict(monkeypatch, ticks=1)
    ev = _journaled(w, "jobs_box_evicted")
    assert ev is not None and "notify_event_id" not in ev
    assert _journaled(w, "notify_outbid_matched") is None
    assert "notify_matched" not in jc


# --- M3 / round 2: the rescue quote answers to the re-bid rung's ceiling --- #
def test_the_tick_hands_the_quote_the_same_bounds_the_rebid_rung_gets(
        monkeypatch):
    """The wiring, not the arithmetic. `notify_rescue_bound` reads four fields
    off the poll state and nothing else does; the rescue PUT runs BEFORE
    `_job_rebid_ladder` on the same tick, so a bound only that rung applies is a
    bound the row walks straight past."""
    seen = {}
    real = bidpolicy.mk_poll_state
    monkeypatch.setattr(bidpolicy, "mk_poll_state",
                        lambda **kw: seen.update(kw) or real(**kw))
    _evict(monkeypatch, rows=[_row()], anchor=1.20, p_alt=0.60, ticks=1)
    assert seen["launch_dph_anchor"] == 1.20
    assert seen["rebid_ceiling_mult"] == bidpolicy.REBID_CEILING_MULT
    assert seen["defense_cap"] is not None, "a fresh p_alt arms the defense"
    assert seen["notify_min_bid"] == ROW_FLOOR
    assert seen["budget_usd"] == 100.0 and seen["spend_usd"] is not None


def test_the_defense_cap_reaches_the_quote_through_the_shared_extractor(
        monkeypatch):
    """`_job_defense_cap` lives in `replacement.py` beside the rung that shares
    its inputs, and the tick reads it BY MODULE ATTRIBUTE — so a patch there
    steers the quote, which is the whole point of the extraction."""
    monkeypatch.setattr(replacement, "_job_defense_cap", lambda jc, now: 0.606)
    seen = {}
    real = bidpolicy.mk_poll_state
    monkeypatch.setattr(bidpolicy, "mk_poll_state",
                        lambda **kw: seen.update(kw) or real(**kw))
    _evict(monkeypatch, rows=[_row()], anchor=1.20, ticks=1)
    assert seen["defense_cap"] == 0.606


def test_a_live_defense_refuses_the_quote_the_undefended_ceiling_allowed(
        monkeypatch):
    """The measured cost of M3+round 2, end to end on the package copy: anchor
    $1.20, `p_alt` $0.60, 20 h of work left. The rung's ceiling is $0.606 and it
    STOPS — replacing is rationally cheaper than holding — while the round-1
    rescue quoted $1.212 against a $2.25 ceiling and PUT it, because the rescue
    runs first. Now it is REFUSED, and the refusal is journaled."""
    views = [dict(_view(), eta_s=20 * 3600)]
    w, jc, _ = _evict(monkeypatch, rows=[_row()], anchor=1.20, p_alt=0.60,
                      views=views)
    q = _journaled(w, "notify_rescue_quote")
    assert q is not None, "the quote is journaled whether or not a bid went out"
    assert q["new_min_bid"] == ROW_FLOOR and q["row_raised"] is True
    assert q["emitted"] is None and q["quoted"] is None
    cap = replacement._job_defense_cap(jc, jc["now"])
    assert cap is not None, "the defense must actually be live, or this is a"
    assert q["ceiling"] == cap
    # ...and it is the DEFENSE that bound, not one of the pre-existing rails:
    # the undefended ceiling at this anchor is strictly higher.
    assert cap < bidpolicy.rebid_ceiling(
        launch_dph_anchor=1.20, max_bid=jc["max_bid"], on_demand=NOTIFY_OD)
    assert q["refused"], "a refusal names the line that fired"
    assert not [p for p in w.puts if p[0] == "bid"], "no money moved"


def test_without_the_defense_that_same_row_prices_and_puts(monkeypatch):
    """The CONTROL for the test above, and the proof the bound only ever
    TIGHTENS. No fresh `p_alt` means no derivable defense, so the ceiling is
    the one `rebid_ceiling` derives from the anchor / max_bid / on-demand
    triple — asserted against that function rather than against a copy of its
    arithmetic, which is the only form of the claim that cannot drift."""
    w, jc, _ = _evict(monkeypatch, rows=[_row()], anchor=1.20)
    q = _journaled(w, "notify_rescue_quote")
    assert q["ceiling"] == bidpolicy.rebid_ceiling(
        launch_dph_anchor=1.20, max_bid=jc["max_bid"], on_demand=NOTIFY_OD)
    assert q["emitted"] is not None and q["emitted"] == q["quoted"]
    assert q["quoted"] <= q["ceiling"] + 1e-9
    assert [p for p in w.puts if p[0] == "bid"]


def test_a_quote_over_the_anchor_ceiling_is_refused(monkeypatch):
    """M3's headline, on the real 2026-08-16 box: launched at $0.45, displaced
    at $1.00, so the row's $1.212 quote is 2.69x what we launched at and 1.35x
    the $0.900 ceiling the NEXT rung would have refused. Before M3 the rescue
    was the only spend rung with no anchor ceiling and no affordability floor,
    and S2b made it fire routinely. The honest answer on that box is that no
    legal bid takes it back."""
    w, jc, _ = _evict(monkeypatch, rows=[_row()], anchor=ROW_BID)
    q = _journaled(w, "notify_rescue_quote")
    assert q["ceiling"] == bidpolicy.rebid_ceiling(
        launch_dph_anchor=ROW_BID, max_bid=jc["max_bid"], on_demand=NOTIFY_OD)
    assert q["ceiling"] == 0.9
    assert q["quoted"] is None and q["emitted"] is None
    assert q["refused"], "a refusal names the line that fired"
    assert not [p for p in w.puts if p[0] == "bid"], "no money moved"


def test_a_below_bid_row_labels_the_cycle_but_never_prices_it(monkeypatch):
    """§6.1: a displacement below our own standing bid is not a price a bid can
    beat. It is still EVIDENCE and still journaled — the quote row is simply
    never emitted, because `notify_min_bid` is None."""
    w, jc, _ = _evict(monkeypatch, rows=[_row(new_min_bid=0.30)], anchor=1.20)
    assert _journaled(w, "notify_outbid_matched") is not None
    assert _journaled(w, "notify_rescue_quote") is None
    assert "notify_quote_said" not in jc
    assert not [p for p in w.puts if p[0] == "bid"]


def test_a_bid_disagreement_is_journaled_and_never_believed(monkeypatch):
    """Belief reconciliation has exactly ONE writer (`ladder_core.
    reconcile_standing_bid`, off `dph_base`). A second one re-opens the
    stranded-stale-belief class the 2026-08-10 review closed, so this row is how
    we would FIND OUT that our belief drifted — not how we would fix it."""
    w, jc, _ = _evict(monkeypatch, rows=[_row(your_bid=0.90)], anchor=1.20)
    mm = _journaled(w, "notify_bid_mismatch")
    assert mm is not None and mm["believed_bid"] == ROW_BID
    assert mm["vast_your_bid"] == 0.90
    assert jc["last_bid"] != 0.90, "the row never writes our belief"


def test_the_whole_cycles_rows_are_consumed_not_just_the_latched_one(
        monkeypatch):
    """Round 1, 2-2. Our own rescue raise is PUT against a stopped instance, and
    a raise that is itself outbid mints a SECOND row mid-cycle. Before the sweep
    that leftover stayed matchable for the full freshness window — straight into
    the next cycle, which then labelled AND priced off a row describing neither
    one."""
    _, jc, _ = _evict(monkeypatch, anchor=1.20,
                      rows=[_row(event_id="e1"),
                            _row(event_id="e2", new_min_bid=1.74)], ticks=1)
    assert sorted(jc["notify_consumed_ids"]) == ["e1", "e2"]


def test_a_row_landing_three_ticks_into_a_cycle_is_still_that_cycles(
        monkeypatch):
    """The sweep runs on the LATCHED path too, not only on the tick that
    matched. Our own rescue raise is PUT against a stopped instance; if THAT
    raise is outbid, vast mints the second row several ticks in. It belongs to
    the cycle we are living through and the cycle spends it — otherwise it is
    still matchable when the box is evicted again."""
    w = _Wire(monkeypatch, inst=_inst(dph_base=ROW_BID, dph_total=ROW_BID),
              queue=["job-a"], views=[_view()], on_demand=NOTIFY_OD,
              listed=False, floor=None)
    monkeypatch.setattr(job_lane, "_job_resume_in_place", lambda *a, **k: False)
    monkeypatch.setattr(replacement, "_job_rebid_ladder", lambda *a, **k: False)
    monkeypatch.setattr(replacement, "_job_eviction_replace",
                        lambda *a, **k: False)
    jc, hf = job_lane.job_supervise_init(_ns())
    job_lane.job_supervise_tick(jc, hf)
    jc["launch_dph_anchor"] = 1.20
    jc["notify_rows"] = [_row(event_id="e1")]
    w.inst = _inst(actual_status="exited", intended_status="stopped",
                   dph_base=ROW_BID, dph_total=ROW_BID)
    job_lane.job_supervise_tick(jc, hf)
    assert jc["notify_consumed_ids"] == ["e1"]
    # ...and the raise that was itself outbid mints the second row, mid-cycle
    jc["notify_rows"].append(_row(event_id="e2", new_min_bid=1.74))
    job_lane.job_supervise_tick(jc, hf)
    assert sorted(jc["notify_consumed_ids"]) == ["e1", "e2"]
    assert jc["notify_matched"]["event_id"] == "e1", (
        "consumed is not latched — the cycle keeps the row it journaled")


def test_rows_for_other_boxes_and_stale_rows_change_nothing(monkeypatch):
    """The match key is (instance id, freshness window). A row for a different
    box, or one older than the window, is not this cycle's evidence and may not
    label it, price it or be spent by it."""
    old = time.time() - 10 * 3600
    w, jc, _ = _evict(monkeypatch, anchor=1.20,
                      rows=[_row(iid="8888", event_id="other"),
                            _row(event_id="stale", created_at=old)])
    assert _journaled(w, "notify_outbid_matched") is None
    assert _journaled(w, "notify_rescue_quote") is None
    assert "notify_matched" not in jc
    assert not [p for p in w.puts if p[0] == "bid"]


def test_the_bare_class_is_what_reaches_the_machine_exclusion(monkeypatch):
    """F2/M2, across the module seam. `ecls` is what the ladder ACTS on and it
    may be refined by a row; `_ecls_bare` is the pre-S2b answer at the same
    reads, and it owns the one consumer a row must not touch — the evicted-
    MACHINE exclusion TTL. A row that refined `unknown -> outbid` un-excluded
    the machine at t+30m and let the very next probe re-rent it."""
    got = []
    monkeypatch.setattr(
        replacement, "_job_eviction_replace",
        lambda jc, hf, ecls, why, exclusion_class=None: got.append(
            (ecls, exclusion_class)) or False)
    # `listed=None` = the offers read itself failed, so the BARE class stays
    # `unknown` (ignorance is not evidence) while the row still refines the
    # acted-on class to `outbid`. That is the exact state the two classes
    # disagree in, and the only one where the TTL difference is reachable.
    w = _Wire(monkeypatch, inst=_inst(dph_base=ROW_BID, dph_total=ROW_BID),
              queue=["job-a"], views=[_view()],
              on_demand=NOTIFY_OD, listed=None, floor=None)
    monkeypatch.setattr(job_lane, "_job_resume_in_place", lambda *a, **k: False)
    monkeypatch.setattr(replacement, "_job_rebid_ladder", lambda *a, **k: False)
    jc, hf = job_lane.job_supervise_init(_ns())
    job_lane.job_supervise_tick(jc, hf)
    jc["notify_rows"] = [_row()]
    w.inst = _inst(actual_status="exited", intended_status="stopped",
                   dph_base=ROW_BID, dph_total=ROW_BID)
    for _ in range(3 * bidpolicy.NOT_LIVE_DEBOUNCE):
        job_lane.job_supervise_tick(jc, hf)
        jc["last_bid_put"] = 0.0
    assert got, "the replacement rung was reached"
    ecls, excl = got[0]
    assert ecls == bidpolicy.EVICTION_OUTBID, "the row refined what we ACT on"
    assert excl == bidpolicy.EVICTION_UNKNOWN, (
        "...and the MACHINE exclusion got the BARE class. `unknown` is "
        "permanent in EVICTED_TTL_CLASSES and `outbid` ages out at 30 min, so "
        "the refined class would have un-excluded a machine we were just "
        "displaced from")
    assert all(e == got[0] for e in got), "and it does not drift across ticks"


def test_the_notify_cluster_is_reached_through_its_owning_modules(monkeypatch):
    """The late-binding proof for the re-ported cluster. `_job_defense_cap` and
    `_job_eviction_replace` live in `replacement`; the notify helpers live here.
    A `from … import` on either side of that seam would make every patch above
    vacuous."""
    src = inspect.getsource(job_lane)
    assert "replacement._job_defense_cap(" in src
    assert "replacement._rebid_knob(" in src
    assert "from vastlib.supervise.replacement import" not in src
    assert not hasattr(job_lane, "_job_defense_cap")
    assert not hasattr(job_lane, "_rebid_knob")
