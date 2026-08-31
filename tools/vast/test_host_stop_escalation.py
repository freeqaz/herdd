"""The bounded host-recovery wait: a `host_stop` may not park a claimed queue.

Incident, 2026-08-28 09:59:07Z. Box 48996785 (1x A100 PCIE spot on machine
56760, standing bid $0.546 under a $0.999 max, $5.00 budget with $4.84 left) was
running a training job 38% in with checkpoint-34 already synced to B2. fleetd
journaled `jobs_box_evicted` with `eviction_class: host_stop`, `claimed_work:
true` and the note "the bid rescue / re-bid ladder / replacement rungs run
next" — and then emitted nothing but `tick` for eleven minutes.

Nothing was wedged. The rescue rung fired on the second not-live tick and armed
`rescue_deadline = now + JOB_SUP_RESCUE_WAIT_S` (900 s), which was still 200 s
in the future when the operator moved the job by hand. `dead` cannot be true
while that deadline stands, so the re-bid and replacement rungs the note
promises were fifteen minutes away, on a class no bid can win back and with the
job's whole queue parked in the meantime. Rung zero has the same shape: a
`start` that does not stick arms another 900 s, up to RESUME_MAX_TRIES times.

Owner directive the same morning: "a job that hits this case automatically moves
to a new host. we don't want to block on this case."

So the fix is a wall-clock bound on the CYCLE — `bidpolicy.host_stop_escalation`
— rather than another rung's deadline, and the class it reads is the one the
announcement published. Four things are pinned here: it fires, it waits first,
it never fires with nothing to retarget, and it still cannot spend past budget.

Toolchain-free: no vast API, no B2, no real clock. The tick harness is
`test_eviction_blindspot`'s, imported rather than copied — it stubs the same
eleven seams at the same resolution points, and a second copy would drift.
"""
import time

import pytest

import bidpolicy as bp
import fleetd
import jobmeta
from test_eviction_blindspot import (_args, _inst, _ladder_events, _MarketRead,
                                     _tick_env)
from vastlib.boxes import lifecycle
from vastlib.market import pricing
from vastlib.supervise import job_lane, replacement

#: Box 48996785, as journaled. `floor_at_stop` is the read on the eviction tick;
#: `prior_floor` is what the two reads before it said, which is why the risen
#: floor failed corroboration and the class came out `host_stop` and not
#: `outbid` (`bidpolicy.floor_rise_corroborated`).
BOX = dict(iid="48996785", machine=56760,
           standing_bid=0.546, floor_at_stop=0.569, prior_floor=0.474,
           on_demand=1.0, max_bid=0.999, budget=5.0, spend=0.1567,
           entry_floor=0.316)

NOW = 4_000_000.0
TICK_S = 30.0          # the deployed fleetd cadence (15 s + jitter, ~30 s seen)


class _Clock:
    """A hand-advanced clock. `job_supervise_tick` reads `time.time()` twice per
    tick and the escalation is a wall-clock bound, so the test owns the clock."""

    def __init__(self, t=NOW):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt
        return self.t


def _evicted_watch(monkeypatch, *, queue=("j1",), budget=None):
    """A jobs watch that has ticked LIVE with a claimed ticket and then stopped
    with nobody having asked. Returns `(clock, jc, hf, inst, calls)`.

    The live tick first is not scene-setting: `claimed_work` reads the PREVIOUS
    tick's folded views, so a watch that never ticked live cannot see its own
    tickets and the eviction would journal `claimed_work: false`.

    The floor MOVES across the stop, as it did: $0.474 while we held the box (no
    defend — it is under `DEFEND_AT x` our bid) and $0.569 on the eviction tick.
    That ordering is what puts the watch in the incident's exact posture: rung
    zero refuses ("standing bid is BELOW the live floor — raise, do not start")
    and the rescue rung arms the 900 s deadline instead.
    """
    clock = _Clock()
    monkeypatch.setattr(time, "time", clock)
    inst = _inst(iid=BOX["iid"], status="running", machine=BOX["machine"],
                 intended="running", dph_total=BOX["standing_bid"] + 0.02,
                 dph_base=BOX["standing_bid"])
    _tick_env(monkeypatch, inst, market=BOX["prior_floor"], listed=True,
              on_demand=BOX["on_demand"], queue=queue)
    floor = {"v": BOX["prior_floor"]}
    monkeypatch.setattr(pricing, "_market_min_bid_soft",
                        lambda m, n=None: floor["v"])
    monkeypatch.setattr(pricing, "_market_min_bid_read",
                        lambda m, n=None: _MarketRead(True, True, floor["v"]))
    calls = {"replace": [], "rebid": [], "start": []}
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda iid, st: (calls["start"].append((str(iid), st)),
                                         (True, None))[1])
    monkeypatch.setattr(
        replacement, "_job_rebid_ladder",
        lambda jc, a, iid, mkt, od, ecls, now: (
            calls["rebid"].append(ecls), False)[1])
    monkeypatch.setattr(
        replacement, "_job_eviction_replace",
        lambda jc, hf, ecls, why, exclusion_class=None: (
            calls["replace"].append((ecls, why)), False)[1])

    jc, hf = job_lane.job_supervise_init(
        _args(id=int(BOX["iid"]),
              budget=BOX["budget"] if budget is None else budget,
              max_bid=BOX["max_bid"],
              # the deployed default; the whole point is that the escalation
              # does not have to wait for it
              rescue_wait=job_lane.JOB_SUP_RESCUE_WAIT_S))
    assert job_lane.job_supervise_tick(jc, hf) is None      # live, ticket seen
    assert jc["last_bid"] == BOX["standing_bid"], "the live tick must not defend"
    # The reads before the stop, both under our bid. Without them the risen
    # floor has nothing to be corroborated against and `classify_eviction`
    # answers `outbid`, which is a different ladder.
    jc["floor_samples"] = [BOX["prior_floor"], BOX["prior_floor"]]
    floor["v"] = BOX["floor_at_stop"]
    inst["actual_status"], inst["intended_status"] = "exited", "stopped"
    return clock, jc, hf, inst, calls


def _tick(clock, jc, hf, n=1, dt=TICK_S):
    out = None
    for _ in range(n):
        clock.advance(dt)
        out = job_lane.job_supervise_tick(jc, hf)
    return out


def _tick_until_replaced(clock, jc, hf, calls, cap=60):
    """Tick until the replacement rung is reached, or `cap` ticks. Returns the
    verdict of the tick that reached it (a bounded loop, never a while-True:
    a wait that cannot time out is the bug this file is about)."""
    for _ in range(cap):
        out = _tick(clock, jc, hf)
        if calls["replace"]:
            return out
    return None


# --------------------------------------------------------------------------- #
# 1. the policy, pure
# --------------------------------------------------------------------------- #

def test_the_escalation_predicate_refuses_by_default():
    """Every gate is a refusal, and the refusals are the point: this decision
    ends with a rental, so anything it cannot observe must not license one."""
    ok = dict(eviction_class=bp.EVICTION_HOST_STOP, claimed_work=True,
              evicted_since=NOW, now=NOW + 300.0,
              not_live=2 * bp.NOT_LIVE_DEBOUNCE)
    assert bp.host_stop_escalation(**ok) == pytest.approx(300.0)

    # a class whose recovery is somebody else's rung, with its own bound
    for cls in (bp.EVICTION_OUTBID, bp.EVICTION_ONDEMAND, bp.EVICTION_UNKNOWN,
                bp.EVICTION_HOST_FAILURE, bp.EVICTION_NO_CREDIT, None):
        assert bp.host_stop_escalation(**{**ok, "eviction_class": cls}) is None
    # nothing to retarget => a replacement buys nothing and spends real money
    assert bp.host_stop_escalation(**{**ok, "claimed_work": False}) is None
    # no clock at all — a cycle restored from a state file that predates the key
    assert bp.host_stop_escalation(**{**ok, "evicted_since": None}) is None
    assert bp.host_stop_escalation(**{**ok, "now": None}) is None
    # a flap has not been down long enough to be a host stop
    assert bp.host_stop_escalation(**{**ok, "not_live": 1}) is None
    # the wait itself
    assert bp.host_stop_escalation(**{**ok, "now": NOW + 239.0}) is None
    assert bp.host_stop_escalation(**{**ok, "now": NOW + 240.0}) is not None
    # operator opt-out
    assert bp.host_stop_escalation(**{**ok, "escalate_after_s": 0}) is None


def test_the_wait_is_shorter_than_every_rung_deadline_it_overrides():
    """The whole defect in one assertion. A warm-disk recovery is worth waiting
    for — a replacement pays a measured 11m35s of setup on a cold disk — but it
    is not worth the 900 s a BID re-auction is given, nor the 300 s x 3 the
    re-bid ladder owns. Sized instead to rung zero's own clock: it fires at
    NOT_LIVE_DEBOUNCE polls and a `start` was measured landing in ~40 s."""
    assert bp.HOST_STOP_ESCALATE_S < job_lane.JOB_SUP_RESCUE_WAIT_S
    assert bp.HOST_STOP_ESCALATE_S < bp.REBID_MAX_RUNGS * bp.REBID_WAIT_S
    assert bp.HOST_STOP_ESCALATE_S > 2 * bp.NOT_LIVE_DEBOUNCE * TICK_S + 40.0
    assert bp.HOST_RECOVERY_CLASSES == (bp.EVICTION_HOST_STOP,)


def test_the_cycle_class_and_clock_are_durable():
    """`state.json` carries them for the reason `rebid_rungs` is carried: the
    escalation is a bound on WALL TIME, and a restart that forgot the clock
    would restart the wait — so a deploy loop could park a claimed queue for
    another `host_stop_escalate_s` every time."""
    assert "evicted_class" in fleetd.REPLACEMENT_STATE_KEYS
    assert "evicted_since" in fleetd.REPLACEMENT_STATE_KEYS


# --------------------------------------------------------------------------- #
# 2. the incident
# --------------------------------------------------------------------------- #

def test_a_host_stop_with_claimed_work_escalates_after_the_bounded_wait(monkeypatch):
    """48996785, end to end. The rescue rung arms its 900 s deadline exactly as
    it did; what changed is that the deadline no longer owns the cycle."""
    clock, jc, hf, _i, calls = _evicted_watch(monkeypatch)
    _tick(clock, jc, hf, n=bp.NOT_LIVE_DEBOUNCE + 1)

    got = {ev: f for ev, f in _ladder_events(jc)}
    assert got["jobs_box_evicted"]["eviction_class"] == bp.EVICTION_HOST_STOP
    assert got["jobs_box_evicted"]["claimed_work"] is True
    assert not calls["start"], \
        "rung zero refuses under a risen floor — this is the rescue-bid posture"
    assert jc["rescue_deadline"] is not None, \
        "the rescue rung still fires; the fix is not 'stop bidding'"
    assert not calls["replace"], "and it still gets its chance first"

    t0 = jc["evicted_since"]
    assert t0 is not None
    # ...right up to the bound, and not one tick past it.
    for _ in range(60):
        if clock.t + TICK_S - t0 >= bp.HOST_STOP_ESCALATE_S:
            break
        _tick(clock, jc, hf)
    assert clock.t - t0 < bp.HOST_STOP_ESCALATE_S
    assert not calls["replace"], \
        "escalating inside the wait throws a warm disk away for nothing"

    _tick(clock, jc, hf, n=2)
    assert clock.t < jc["rescue_deadline"], \
        "THE POINT: the old ladder is still sitting on its rescue deadline here"
    assert [c[0] for c in calls["replace"]], \
        "the replacement rung must be reached without waiting for that deadline"
    ev = {e: f for e, f in _ladder_events(jc)}["jobs_host_stop_escalated"]
    assert ev["eviction_class"] == bp.EVICTION_HOST_STOP
    assert ev["waited_s"] >= bp.HOST_STOP_ESCALATE_S
    assert ev["pending_jobs"] == ["j1"]
    assert ev["escalate_after_s"] == bp.HOST_STOP_ESCALATE_S


def test_the_escalation_skips_the_rebid_rungs(monkeypatch):
    """A re-bid buys the warm box back FROM A COMPETITOR. `host_stop` is the
    class that says there is none — and a rung would re-arm another
    `rebid_wait_s`, re-opening the stall the escalation just ended."""
    clock, jc, hf, _i, calls = _evicted_watch(monkeypatch)
    assert _tick_until_replaced(clock, jc, hf, calls) is not None
    assert not calls["rebid"], \
        f"no rung may price against a host that stopped us: {calls['rebid']}"


def test_a_box_that_comes_back_inside_the_wait_rents_nothing(monkeypatch):
    """The cheap recovery still wins. The host (or rung zero) returns the box
    before the bound, and the cycle ends with its class and clock retired — so
    the NEXT eviction is dated from itself and not from this one."""
    clock, jc, hf, inst, calls = _evicted_watch(monkeypatch)
    _tick(clock, jc, hf, n=bp.NOT_LIVE_DEBOUNCE + 1)
    assert jc.get("evicted_since") is not None
    assert clock.t - jc["evicted_since"] < bp.HOST_STOP_ESCALATE_S

    inst["actual_status"], inst["intended_status"] = "running", "running"
    _tick(clock, jc, hf, n=1)
    assert not calls["replace"], "the box is back — nothing to replace"
    assert jc.get("evicted_since") is None, "a stale clock would date the next cycle"
    assert jc.get("evicted_class") is None
    assert "jobs_box_eviction_survived" in {e for e, _ in _ladder_events(jc)}

    # ...and now well past what the OLD cycle's bound would have been, live.
    _tick(clock, jc, hf, n=int(bp.HOST_STOP_ESCALATE_S / TICK_S) + 4)
    assert not calls["replace"]
    assert "jobs_host_stop_escalated" not in {e for e, _ in _ladder_events(jc)}


def test_an_empty_queue_never_reaches_the_ladder_at_all(monkeypatch):
    """The `claimed_work` gate is a backstop, and this is why it is only that.
    A watch whose queue is empty exits `queue_empty` above the eviction ladder,
    so the tick cannot reach the escalation with nothing to retarget. The gate
    stays because the predicate is pure and its caller could move; the pure test
    above is what exercises it."""
    clock, jc, hf, _i, calls = _evicted_watch(monkeypatch, queue=("j1",))
    monkeypatch.setattr(jobmeta, "list_queue", lambda iid: [])
    assert _tick(clock, jc, hf) == "queue_empty"
    assert not calls["replace"]


def test_the_escalation_does_not_reach_past_the_spend_gates(monkeypatch):
    """It decides that the WAITING is over, never that a rental is affordable.
    `replacement_decision` still owns the money, and its refusal still lands the
    watch on the manual retarget checklist rather than on a box we cannot pay
    for."""
    clock, jc, hf, _i, calls = _evicted_watch(monkeypatch)
    out = _tick_until_replaced(clock, jc, hf, calls)
    assert calls["replace"], "the rung was reached"
    assert out == "unrecoverable", \
        "a refused replacement is still unrecoverable, exactly as before"

    # ...and the rung that refuses it, on this class, with the budget spent.
    dec = bp.replacement_decision(
        eviction_class=bp.EVICTION_HOST_STOP, replacements_used=0,
        budget_usd=BOX["budget"], spend_usd=BOX["budget"],
        launch_dph_anchor=BOX["standing_bid"], offer_min_bid=0.30,
        offer_ondemand=BOX["on_demand"])
    assert dec.action == "stop"
    assert "budget" in dec.reason.lower()


def test_the_wait_is_a_knob(monkeypatch):
    """`JOB_HOST_STOP_ESCALATE_S` (namespace > env > herdd.yaml), the same
    precedence every other ladder parameter answers to — and 0 disables it, so
    an operator who wants the old fifteen-minute posture can have it."""
    clock, jc, hf, _i, calls = _evicted_watch(monkeypatch)
    monkeypatch.setenv("JOB_HOST_STOP_ESCALATE_S", "0")
    _tick(clock, jc, hf, n=int(bp.HOST_STOP_ESCALATE_S / TICK_S) + 6)
    assert not calls["replace"], "disabled means disabled"

    monkeypatch.setenv("JOB_HOST_STOP_ESCALATE_S", "60")
    assert _tick_until_replaced(clock, jc, hf, calls) is not None
