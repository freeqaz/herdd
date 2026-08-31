"""The STANDING jobs watch — a watch that survives its queue draining.

Item 2 of `FLEET_REVIEW_2026-08-14.md`, and the "park-and-keep" option
`FLEETD_DESIGN.md` §4a filed as open-but-probably-right.

The cycle this closes, observed on box 47694876 (2026-08-14 10:02Z) and on
47511739 before it: a jobs watch ends the moment every ticket it can see is
terminal (`watch_finished {verdict: drained}`), the box drain-parks, the stray
sweep re-adopts it as unbudgeted `bare` carrying the inherited ceiling, and the
`watch:<T>:adopted` alarm says "its armed watch LAPSED ... the ceiling SURVIVED
but the ladder did not". Everything submitted after that runs on a spot box with
no outbid rescue, no eviction replacement and no drain-park until somebody
re-registers a watch by hand. 16 dead boxes' worth of durable ceilings carry
that shape.

`fleet watch <IID> --profile jobs --budget N --standing` keeps the watch: the
box still parks per policy, the watch goes dormant-but-ARMED, and the next
ticket resumes the same ladder under the same cumulative cap.

Everything here is a fake — no network, no vast API, no real clock — through the
same injected-hooks discipline `test_fleetd.py` uses. The regression half (§2)
pins the DEFAULT path unchanged: without `--standing`, a drain still ends the
watch, still hands the box to the safety net, and still raises the LAPSED alarm.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vastlib.fleet import client, daemon                         # noqa: E402
from vastlib.jobs import submit                                  # noqa: E402


# --------------------------------------------------------------------------- #
# fakes — the test_fleetd.py FakeHooks, trimmed to what this lifecycle touches
# plus a call counter on `drained` (the standing resume gate's only I/O)
# --------------------------------------------------------------------------- #
class FakeHooks:
    def __init__(self):
        self.t = 1_700_000_000.0
        self.dry_run = False
        self.boxes = {}
        self.parked, self.resumed, self.destroyed = [], [], []
        self.park_ok = True
        self.drained_map = {}           # iid -> True/False/None
        self.drained_calls = []         # every queue read the daemon made
        self.jobs_result = None         # verdict the jobs ladder returns
        self.jobs_spend = 0.0
        self.jobs_ticks = []
        self.kept = []

    # clock
    def now(self):
        return self.t

    def advance(self, s):
        self.t += s

    # fleet
    def box(self, iid, status="running", dph=0.5, label=None, **kw):
        self.boxes[str(iid)] = dict(id=int(iid), actual_status=status,
                                    intended_status=kw.pop("intended", status),
                                    dph_total=dph, label=label, **kw)
        return self.boxes[str(iid)]

    def park_box(self, iid):
        """What the jobs ladder itself does on drain (`_stop_instance_soft`),
        BEFORE its verdict reaches fleetd. `--keep` is what suppresses it."""
        b = self.boxes[str(iid)]
        b["actual_status"] = b["intended_status"] = "stopped"

    def instances(self):
        return list(self.boxes.values())

    def instance(self, iid):
        return self.boxes.get(str(iid))

    def park(self, iid):
        self.parked.append(str(iid))
        if self.park_ok:
            self.park_box(iid)
            return True, None
        return False, "park refused"

    def resume(self, iid):
        self.resumed.append(str(iid))
        b = self.boxes.get(str(iid))
        if b:
            b["actual_status"] = b["intended_status"] = "running"
        return True, None

    def destroy(self, iid):
        self.destroyed.append(str(iid))
        self.boxes.pop(str(iid), None)
        return True, None

    def drained(self, iid):
        self.drained_calls.append(str(iid))
        return self.drained_map.get(str(iid))

    def results_present(self, iid):
        return None

    def health(self, instances):
        return {}

    def keep_label(self, iid, inst):
        b = self.boxes.get(str(iid))
        label = (b or {}).get("label") or ""
        if "keep" in [t.strip() for t in label.split(":")]:
            return False, label
        new = (label + ":keep") if label else "keep:fleetd-park"
        if b is not None:
            b["label"] = new
        self.kept.append(str(iid))
        return True, new

    # profile ticks
    def jobs_init(self, a):
        return ({"a": a, "iid": str(a.id), "spend_usd": 0.0,
                 "handoff_on": True}, {"phase": "IDLE"})

    def jobs_tick(self, jc, hf):
        self.jobs_ticks.append(jc["iid"])
        jc["spend_usd"] = self.jobs_spend
        return self.jobs_result

    def run_init(self, a):
        return ({"run_id": a.run_id, "spend_usd": 0.0, "instance_id": None,
                 "actual_status": "running", "relaunch_count": 0},
                {"phase": "IDLE"}, True)

    def run_tick(self, st, a, hf, handoff_on):
        return None

    def run_finalize(self, st, a, act, hf, handoff_on):
        pass


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    monkeypatch.setenv("FLEETD_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("FLEETD_GLOBAL_BUDGET_USD", raising=False)
    monkeypatch.delenv("FLEETD_UNWATCHED_GRACE_S", raising=False)
    monkeypatch.delenv("FLEETD_UNWATCHED_GRACE_EXPENSIVE_S", raising=False)
    monkeypatch.delenv("FLEETD_EXPENSIVE_DPH", raising=False)
    return daemon.Fleet(str(tmp_path / "state"), hooks=FakeHooks())


def journal(f):
    if not os.path.exists(f.journal_path):
        return []
    with open(f.journal_path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def events(f):
    return [r["event"] for r in journal(f)]


def event(f, name):
    return [r for r in journal(f) if r["event"] == name]


def observe(f, seconds, step=None):
    """Real reconcile ticks across `seconds` of OBSERVED time (N7)."""
    step = step or daemon.MAX_OBS_DT_S
    f.tick()
    left = seconds
    while left > 0:
        f.hooks.advance(min(step, left))
        f.tick()
        left -= min(step, left)


def arm(fleet, iid=47694876, budget=5.0, standing=True, label="jobs:w8", **kw):
    """A standing (or not) jobs watch over a live box, one cycle in."""
    fleet.hooks.box(iid, label=label, **kw)
    return fleet.watch(str(iid), "jobs", budget_usd=budget,
                       policy={"id": iid, "budget": budget},
                       requester="operator@workstation", standing=standing)


def drain(fleet, iid=47694876, verdict="drained", park=True):
    """One drain tick: the ladder parks the box (unless --keep) and returns the
    drain verdict, exactly as `herdd._job_supervise_tick` does."""
    fleet.hooks.jobs_result = verdict
    if park:
        fleet.hooks.park_box(iid)
    fleet.tick()
    fleet.hooks.jobs_result = None


# --------------------------------------------------------------------------- #
# 1. a STANDING watch survives the drain
# --------------------------------------------------------------------------- #
def test_standing_watch_survives_a_drain(fleet):
    arm(fleet)
    fleet.tick()
    drain(fleet)
    w = fleet.state["watches"].get("47694876")
    assert w is not None, "a standing watch must NOT end on a drain"
    assert (w["profile"], w["budget_usd"]) == ("jobs", 5.0)
    assert w["standing"] and w["standing_dormant"]
    assert w["state"] == "standing" and w["dormant"] is True
    assert w["dormant_reason"] == "standing_drained"
    assert w["standing_cycles"] == 1
    assert "watch_finished" not in events(fleet)
    ev = event(fleet, "jobs_watch_standing_drained")
    assert len(ev) == 1 and ev[0]["verdict"] == "drained"
    assert ev[0]["remaining_usd"] == 5.0 and ev[0]["cap_usd"] == 5.0


def test_standing_drain_raises_no_lapse_alarm_and_no_bare_adoption(fleet):
    """The whole cycle, run out past the unwatched grace window: the box is
    never re-adopted, never alarms, and the ladder record stays intact."""
    arm(fleet)
    fleet.tick()
    drain(fleet)
    observe(fleet, daemon.UNWATCHED_GRACE_S * 2)
    w = fleet.state["watches"]["47694876"]
    assert (w["profile"], w["budget_usd"], w["adopted"]) == ("jobs", 5.0, False)
    assert "watch_auto_adopted" not in events(fleet)
    assert "unwatched_adopted" not in events(fleet)
    assert not any("LAPSED" in a for a in fleet.alarms)
    assert fleet.alarms == []                     # S8: a parked standing box is quiet


def test_jobd_self_park_on_drain_is_the_same_transition(fleet):
    """`self_parked` is the same fact seen from the box side (jobd parked itself
    after the queue drained). Which one fleetd sees is a race, so standing must
    survive both or it survives neither."""
    arm(fleet)
    fleet.tick()
    drain(fleet, verdict="self_parked")
    w = fleet.state["watches"].get("47694876")
    assert w is not None and w["standing_dormant"]
    assert event(fleet, "jobs_watch_standing_drained")[0]["verdict"] == "self_parked"


def test_standing_never_survives_a_budget_breach(fleet):
    """The cap is the one thing standing does not stand through: `budget` is
    the enforcement seam and still ends the watch."""
    arm(fleet)
    fleet.tick()
    fleet.hooks.jobs_result = "budget"
    fleet.tick()
    assert event(fleet, "watch_finished")[-1]["verdict"] == "budget"
    # what survives is the safety net's `bare` adoption on the ceiling, i.e.
    # exactly the pre-standing shape — the ladder is gone, as intended
    w = fleet.state["watches"].get("47694876")
    assert w is None or (w["profile"], w["adopted"]) == ("bare", True)


def test_standing_is_refused_on_profiles_with_no_queue(fleet):
    fleet.hooks.box(47694876, label="serve:eval")
    with pytest.raises(ValueError) as e:
        fleet.watch("47694876", "serve", budget_usd=5.0, standing=True)
    assert "jobs" in str(e.value)
    with pytest.raises(ValueError):
        fleet.watch("run:r1", "run", budget_usd=5.0, standing=True)


# --------------------------------------------------------------------------- #
# 2. REGRESSION — without --standing nothing moved
# --------------------------------------------------------------------------- #
def test_non_standing_drain_still_ends_the_watch(fleet):
    arm(fleet, standing=False)
    fleet.tick()
    drain(fleet)
    assert "47694876" not in fleet.state["watches"]
    fin = event(fleet, "watch_finished")
    assert len(fin) == 1 and fin[0]["verdict"] == "drained"
    assert "jobs_watch_standing_drained" not in events(fleet)


def test_non_standing_drain_still_lapses_into_a_bare_adoption(fleet):
    """The full pre-fix cycle, unchanged: watch ends -> safety net adopts the
    still-live box `bare` on the INHERITED ceiling -> "armed watch LAPSED"."""
    arm(fleet, standing=False)
    fleet.tick()
    drain(fleet, park=False)                      # `--keep`-shaped: box stays live
    fin = event(fleet, "watch_finished")
    assert len(fin) == 1 and fin[0]["verdict"] == "drained"
    observe(fleet, daemon.UNWATCHED_GRACE_S)
    w = fleet.state["watches"]["47694876"]
    assert (w["profile"], w["adopted"]) == ("bare", True)
    assert w["ceiling_source"] == "inherited"
    assert "watch_auto_adopted" in events(fleet)
    assert any("LAPSED" in a for a in fleet.alarms)


def test_a_standing_watch_is_off_by_default(fleet):
    arm(fleet, standing=False)
    assert fleet.state["watches"]["47694876"]["standing"] is False
    w = fleet.watch("999", "bare")
    assert w["standing"] is False


# --------------------------------------------------------------------------- #
# 3. persistence — the flag and the dormant phase survive a fleetd restart
# --------------------------------------------------------------------------- #
def _restart(fleet, tmp_path):
    """A new daemon over the same state dir, same clock (S2)."""
    fleet.save()
    fresh = daemon.Fleet(fleet.dir, hooks=fleet.hooks)
    return fresh


def test_standing_flag_round_trips_through_persistence(fleet, tmp_path):
    arm(fleet)
    fleet.tick()
    on_disk = json.load(open(fleet.state_path))["watches"]["47694876"]
    assert on_disk["standing"] is True
    fresh = _restart(fleet, tmp_path)
    assert fresh.state["watches"]["47694876"]["standing"] is True


def test_the_dormant_armed_phase_survives_a_restart(fleet, tmp_path):
    """A deploy/restart mid-dormancy must not turn a standing watch back into a
    lapsed one — the restart is exactly when the old cycle lost its ladder."""
    arm(fleet)
    fleet.tick()
    drain(fleet)
    fresh = _restart(fleet, tmp_path)
    w = fresh.state["watches"]["47694876"]
    assert w["standing"] and w["standing_dormant"] and w["state"] == "standing"
    # ... and it still resumes on a ticket, on the restarted daemon
    fresh.hooks.resume("47694876")
    fresh.hooks.drained_map["47694876"] = False
    fresh.hooks.advance(45)
    fresh.tick()
    assert fresh.state["watches"]["47694876"]["standing_dormant"] is False
    assert "jobs_watch_standing_resumed" in events(fresh)


def test_an_explicit_re_registration_states_the_flag(fleet):
    """`fleet watch` states the whole watch, so re-running it WITHOUT
    --standing turns standing off — and re-running it WITH --standing mid
    dormancy re-arms immediately instead of waiting for a ticket."""
    arm(fleet)
    fleet.tick()
    drain(fleet)
    fleet.watch("47694876", "jobs", budget_usd=5.0, policy={"id": 47694876},
                standing=True)
    w = fleet.state["watches"]["47694876"]
    assert w["standing"] and not w["standing_dormant"] and not w["dormant"]
    assert w["standing_cycles"] == 1              # history is not reset
    fleet.watch("47694876", "jobs", budget_usd=5.0, policy={"id": 47694876},
                standing=False)
    assert fleet.state["watches"]["47694876"]["standing"] is False
    # `standing=None` (an internal caller / an older client) leaves it alone
    fleet.watch("47694876", "jobs", budget_usd=5.0, policy={"id": 47694876})
    assert fleet.state["watches"]["47694876"]["standing"] is False


# --------------------------------------------------------------------------- #
# 4. budget — one cumulative ceiling across every cycle
# --------------------------------------------------------------------------- #
def test_the_cap_is_not_reset_by_a_drain(fleet):
    """The conservative reading, and the one the ceiling ledger already
    enforces across a lapse: cycle two spends the REMAINING headroom."""
    arm(fleet, budget=5.0)
    fleet.hooks.jobs_spend = 2.0
    fleet.tick()
    drain(fleet)
    ev = event(fleet, "jobs_watch_standing_drained")[0]
    assert (ev["ceiling_spend_usd"], ev["remaining_usd"]) == (2.0, 3.0)
    # resume on a ticket: same ceiling, same spend-to-date, no re-arm
    armed_before = len(event(fleet, "ceiling_armed"))
    fleet.hooks.resume("47694876")
    fleet.hooks.drained_map["47694876"] = False
    fleet.hooks.advance(45)
    fleet.tick()
    w = fleet.state["watches"]["47694876"]
    # the tolerance is one tick of dormant-phase accrual ($0.5/hr x 45 s), not
    # slack in the ledger: what must not happen is the counter going back to 0
    assert fleet._ceiling_spend(w) == pytest.approx(2.0, abs=0.02)
    assert w["budget_usd"] == 5.0
    assert len(event(fleet, "ceiling_armed")) == armed_before, \
        "a drain must not arm a second cap"
    ev = event(fleet, "jobs_watch_standing_resumed")[0]
    assert ev["remaining_usd"] == pytest.approx(3.0, abs=0.02)


def test_cycle_two_parks_on_the_remaining_headroom(fleet):
    """The failure this prevents: six armings of $10 = a $60 effective ceiling.
    Cycle two must breach at the ORIGINAL cap, not at cap + spend."""
    arm(fleet, budget=5.0)
    fleet.hooks.jobs_spend = 4.0
    fleet.tick()
    drain(fleet)
    fleet.hooks.resume("47694876")
    fleet.hooks.drained_map["47694876"] = False
    fleet.hooks.jobs_spend = 5.5                  # cycle two crosses the cap
    fleet.hooks.advance(45)
    fleet.tick()
    fleet.hooks.advance(45)
    fleet.tick()
    w = fleet.state["watches"]["47694876"]
    assert w["state"] == "budget_parked"
    assert "47694876" in fleet.hooks.parked
    assert any("BUDGET" in a for a in fleet.alarms)


def test_a_breach_during_dormancy_parks_once_not_every_tick(fleet):
    """A standing watch is already dormant, so the dormancy check cannot be the
    thing that stops a per-tick re-park the way it does on the normal path."""
    arm(fleet, budget=1.0)
    fleet.hooks.jobs_spend = 0.5
    fleet.tick()
    drain(fleet, park=False)                      # --keep: live and billing
    fleet.hooks.drained_map["47694876"] = True
    fleet.hooks.jobs_spend = 2.0                  # ... straight past the cap
    fleet.state["watches"]["47694876"]["spend_usd"] = 2.0
    for _ in range(5):
        fleet.hooks.advance(45)
        fleet.tick()
    assert fleet.hooks.parked == ["47694876"]
    assert len(event(fleet, "parked")) == 1
    assert fleet.state["watches"]["47694876"]["state"] == "budget_parked"
    assert any("BUDGET" in a for a in fleet.alarms)


def test_accrual_continues_through_a_keep_style_dormancy(fleet):
    """A `--keep` standing box stays LIVE and billing while dormant. The clock
    must keep running there or the cap silently stops meaning anything."""
    arm(fleet, budget=100.0, dph=3.6)             # $0.001/s
    fleet.tick()
    drain(fleet, park=False)                      # --keep: box left running
    fleet.hooks.drained_map["47694876"] = True    # queue is terminal, box live
    before = fleet.state["watches"]["47694876"]["spend_usd"]
    fleet.hooks.advance(1000)
    fleet.tick()
    w = fleet.state["watches"]["47694876"]
    assert w["standing_dormant"], "a live drained box stays dormant"
    assert w["spend_usd"] - before == pytest.approx(1.0, rel=1e-3)


def test_a_live_dormant_standing_box_alarms_and_a_parked_one_does_not(fleet):
    arm(fleet, budget=100.0)
    fleet.tick()
    drain(fleet, park=False)                      # --keep
    fleet.hooks.drained_map["47694876"] = True
    fleet.hooks.advance(600)
    fleet.tick()
    assert any("STANDING" in a and "LIVE" in a for a in fleet.alarms)
    fleet.hooks.park("47694876")                  # operator parks it after all
    fleet.hooks.advance(45)
    fleet.tick()
    assert fleet.alarms == []


# --------------------------------------------------------------------------- #
# 5. resume — a TICKET re-arms, nothing else
# --------------------------------------------------------------------------- #
def test_new_tickets_resume_the_ladder_with_no_re_arm(fleet):
    arm(fleet)
    fleet.tick()
    drain(fleet)
    n = len(fleet.hooks.jobs_ticks)
    fleet.hooks.resume("47694876")                # operator starts the box back up
    fleet.hooks.drained_map["47694876"] = False   # ... and submits a wave
    fleet.hooks.advance(45)
    fleet.tick()
    w = fleet.state["watches"]["47694876"]
    assert not w["standing_dormant"] and not w["dormant"]
    assert w["state"] == "watched"
    assert len(fleet.hooks.jobs_ticks) > n, "the SAME tick must drive the ladder"
    ev = event(fleet, "jobs_watch_standing_resumed")
    assert len(ev) == 1 and ev[0]["dormant_s"] >= 45
    assert "watch_registered" not in events(fleet)[-1:]   # nobody re-armed it


def test_a_resumed_box_with_a_still_terminal_queue_is_not_re_armed(fleet):
    """Liveness alone must NOT re-arm: re-entering the ladder against an
    all-terminal queue drain-parks the box seconds later (box 46648873,
    2026-08-03, is that same failure from the other direction). This is also
    what makes `start the box, then submit` safe."""
    arm(fleet)
    fleet.tick()
    drain(fleet)
    n = len(fleet.hooks.jobs_ticks)
    fleet.hooks.resume("47694876")
    fleet.hooks.drained_map["47694876"] = True    # nothing submitted yet
    fleet.hooks.advance(45)
    fleet.tick()
    assert fleet.state["watches"]["47694876"]["standing_dormant"] is True
    assert len(fleet.hooks.jobs_ticks) == n
    assert "jobs_watch_standing_resumed" not in events(fleet)
    assert fleet.hooks.parked == []               # and it was NOT re-parked


def test_an_unreadable_queue_holds_the_dormancy_and_says_so_once(fleet):
    """N7: an unreadable queue is not evidence of work. It must not resume on a
    guess, and it must not journal per tick either."""
    arm(fleet)
    fleet.tick()
    drain(fleet)
    fleet.hooks.resume("47694876")
    fleet.hooks.drained_map["47694876"] = None    # B2 read failed
    for _ in range(4):
        fleet.hooks.advance(45)
        fleet.tick()
    assert fleet.state["watches"]["47694876"]["standing_dormant"] is True
    assert len(event(fleet, "jobs_watch_standing_queue_unknown")) == 1
    fleet.hooks.drained_map["47694876"] = False   # B2 came back, work is waiting
    fleet.hooks.advance(45)
    fleet.tick()
    assert "jobs_watch_standing_resumed" in events(fleet)


def test_a_parked_standing_box_reads_no_queue_at_all(fleet):
    """The cost claim: the resume gate is one B2 listing per tick and ONLY
    while the box is live. The normal shape — parked between waves — is free."""
    arm(fleet)
    fleet.tick()
    drain(fleet)
    fleet.hooks.drained_calls.clear()
    observe(fleet, 600)
    assert fleet.hooks.drained_calls == []


def test_a_second_drain_cycles_the_same_watch(fleet):
    arm(fleet)
    fleet.tick()
    drain(fleet)
    fleet.hooks.resume("47694876")
    fleet.hooks.drained_map["47694876"] = False
    fleet.hooks.advance(45)
    fleet.tick()
    fleet.hooks.advance(45)
    drain(fleet)
    w = fleet.state["watches"]["47694876"]
    assert w["standing_dormant"] and w["standing_cycles"] == 2
    assert len(event(fleet, "jobs_watch_standing_drained")) == 2
    assert "watch_finished" not in events(fleet)


def test_a_standing_box_that_truly_vanishes_still_dies_via_instance_gone(fleet):
    """Scoped out, deliberately: a box that leaves the listing during the
    dormant phase has an EMPTY queue by construction, so there is nothing to
    retarget and no replacement to price. It reaps through the normal
    instance_gone path and the ceiling survives, exactly as before."""
    arm(fleet)
    fleet.tick()
    drain(fleet)
    del fleet.hooks.boxes["47694876"]
    for _ in range(daemon.GONE_CONFIRM_TICKS):
        fleet.hooks.advance(45)
        fleet.tick()
    assert "47694876" not in fleet.state["watches"]
    assert event(fleet, "watch_finished")[-1]["verdict"] == "instance_gone"


# --------------------------------------------------------------------------- #
# 6. client surface
# --------------------------------------------------------------------------- #
def test_status_rows_say_standing_and_dormant(fleet):
    arm(fleet)
    fleet.tick()
    drain(fleet)
    row = [r for r in fleet.status()["rows"] if r.get("target") == "47694876"][0]
    assert row["standing"] is True and row["standing_dormant"] is True
    assert row["state"] == "standing" and row["dormant"] is True


def test_the_watch_rpc_carries_standing_only_when_it_is_set(fleet):
    """Additive on purpose: a non-standing response is byte-identical to what
    every client before this shipped."""
    srv = daemon.Server(fleet, sock_path=os.path.join(fleet.dir, "s.sock"))
    fleet.hooks.box(222)
    ok, data, _ = srv.handle({"v": client.FLEET_PROTO_VERSION, "op": "watch",
                              "args": {"target": "222", "profile": "jobs",
                                       "budget_usd": 5.0, "standing": True,
                                       "policy": {"id": 222}}})
    assert ok and data["standing"] is True
    fleet.hooks.box(333)
    ok, data, _ = srv.handle({"v": client.FLEET_PROTO_VERSION, "op": "watch",
                              "args": {"target": "333", "profile": "jobs",
                                       "budget_usd": 5.0,
                                       "policy": {"id": 333}}})
    assert ok and "standing" not in data


def test_submit_advisory_reads_a_standing_watch_as_armed(fleet, monkeypatch,
                                                         capsys):
    """`job submit` grades supervision off state.json. A standing watch parked
    between waves must read `policy` (armed), never `lapsed` — and must say it
    is about to wake up."""
    arm(fleet)
    fleet.tick()
    drain(fleet)
    fleet.save()
    monkeypatch.setenv("FLEETD_STATE_DIR", fleet.dir)
    level, d = client.fleet_watch_supervision(47694876)
    assert level == "policy" and d["standing"] and d["standing_dormant"]
    submit._print_submit_supervision(47694876, "herdd")
    out = capsys.readouterr()
    assert "STANDING" in out.out and "re-arms it" in out.out
    assert out.err == ""


# --------------------------------------------------------------------------- #
# 7. the WAKE — `job submit|retarget|requeue` say a ticket landed
# --------------------------------------------------------------------------- #
# The queue poll in §5 is an INFERENCE, and it is silent in exactly the two
# shapes that cost money: a parked box reads no queue at all, and a B2 listing
# that will not answer is `unknown` by contract (N7). Measured on the live
# daemon 2026-08-27: `jobs_watch_standing_resumed` had fired 0 times against 84
# drains. Tickets were retargeted onto a drained standing box, the box was
# evicted minutes later, and the dormant watch journaled nothing — no rescue,
# no replacement, the work stranded on an exited box.
def _woke(fleet, iid=47694876, source="job retarget", jid="j-1"):
    return fleet.ticket_placed(str(iid), jid, source)


def test_a_wake_re_arms_a_dormant_standing_watch_without_a_queue_read(fleet):
    """The primary fix: the placement itself is the evidence, so the resume
    does not wait on — or depend on — the poll."""
    arm(fleet)
    fleet.tick()
    drain(fleet)
    fleet.hooks.resume("47694876")                # operator starts the box back up
    fleet.hooks.drained_map["47694876"] = None    # ... and the B2 listing is dark
    res = _woke(fleet)
    assert res["woken"] is True and res["watched"] is True
    fleet.hooks.drained_calls.clear()
    fleet.hooks.advance(45)
    fleet.tick()
    w = fleet.state["watches"]["47694876"]
    assert not w["standing_dormant"] and not w["dormant"]
    assert w["state"] == "watched"
    assert fleet.hooks.drained_calls == [], "the wake outranks the poll"
    ev = event(fleet, "jobs_watch_standing_resumed")
    assert len(ev) == 1 and ev[0]["trigger"] == "job retarget"
    assert event(fleet, "jobs_watch_standing_woken")[0]["job_id"] == "j-1"


def test_a_requeue_wake_is_the_same_transition(fleet):
    """`job requeue` re-opens a failed job onto a box the same way `retarget`
    moves a live one. Both place a non-terminal ticket; both must wake."""
    arm(fleet)
    fleet.tick()
    drain(fleet)
    fleet.hooks.resume("47694876")
    fleet.hooks.drained_map["47694876"] = None
    _woke(fleet, source="job requeue", jid="j-2")
    fleet.hooks.advance(45)
    fleet.tick()
    assert fleet.state["watches"]["47694876"]["standing_dormant"] is False
    assert event(fleet, "jobs_watch_standing_resumed")[0]["trigger"] == "job requeue"


def test_a_wake_never_re_arms_a_box_that_is_not_live(fleet):
    """A ticket written onto a PARKED box must not put the ladder back on it:
    the jobs ladder classifies a stopped bid box as outbid and rents against
    that, which is a money move nobody asked for. The wake is held instead."""
    arm(fleet)
    fleet.tick()
    drain(fleet)                                  # box parked by the drain
    _woke(fleet)
    for _ in range(4):
        fleet.hooks.advance(45)
        fleet.tick()
    assert fleet.state["watches"]["47694876"]["standing_dormant"] is True
    assert "jobs_watch_standing_resumed" not in events(fleet)
    assert fleet.hooks.resumed == [] and fleet.hooks.parked == []
    # ... and it is still pending, so the start alone completes the re-arm
    fleet.hooks.resume("47694876")
    fleet.hooks.advance(45)
    fleet.tick()
    assert fleet.state["watches"]["47694876"]["standing_dormant"] is False
    assert event(fleet, "jobs_watch_standing_resumed")[0]["trigger"] == "job retarget"


def test_a_wake_is_consumed_once(fleet):
    """It is a latch, not a mode: the next drain must be able to park the box
    again rather than be re-armed by a wake from two waves ago."""
    arm(fleet)
    fleet.tick()
    drain(fleet)
    fleet.hooks.resume("47694876")
    fleet.hooks.drained_map["47694876"] = None
    _woke(fleet)
    fleet.hooks.advance(45)
    fleet.tick()
    w = fleet.state["watches"]["47694876"]
    assert not w.get("standing_wake_pending")
    fleet.hooks.advance(45)
    drain(fleet)
    assert fleet.state["watches"]["47694876"]["standing_dormant"] is True
    fleet.hooks.resume("47694876")
    fleet.hooks.drained_map["47694876"] = True    # nothing new was submitted
    fleet.hooks.advance(45)
    fleet.tick()
    assert fleet.state["watches"]["47694876"]["standing_dormant"] is True


def test_an_explicit_re_registration_discards_a_pending_wake(fleet):
    """`fleet watch` states the WHOLE watch and lands armed, so a latch left
    over from a wave the operator has just re-armed past must not survive to
    undo the next drain."""
    arm(fleet)
    fleet.tick()
    drain(fleet)
    _woke(fleet)
    fleet.watch("47694876", "jobs", budget_usd=5.0, policy={"id": 47694876},
                standing=True)
    w = fleet.state["watches"]["47694876"]
    assert not w.get("standing_wake_pending") and not w["standing_dormant"]
    fleet.hooks.resume("47694876")
    fleet.hooks.advance(45)
    drain(fleet)
    fleet.hooks.resume("47694876")
    fleet.hooks.drained_map["47694876"] = True     # nothing new was submitted
    fleet.hooks.advance(45)
    fleet.tick()
    assert fleet.state["watches"]["47694876"]["standing_dormant"] is True


def test_a_wake_on_an_armed_or_unwatched_box_reports_itself_as_a_no_op(fleet):
    arm(fleet)
    fleet.tick()
    res = _woke(fleet)
    assert res["watched"] is True and res["woken"] is False
    assert res["standing"] is True and res["standing_dormant"] is False
    assert "jobs_watch_standing_woken" not in events(fleet)
    gone = fleet.ticket_placed("999999", "j-9", "job submit")
    assert gone == {"target": "999999", "watched": False, "profile": None,
                    "standing": False, "standing_dormant": False,
                    "woken": False, "note": None}


def test_the_wake_survives_a_restart(fleet, tmp_path):
    """A deploy between the placement and the next tick must not eat it."""
    arm(fleet)
    fleet.tick()
    drain(fleet)
    fleet.hooks.resume("47694876")
    fleet.hooks.drained_map["47694876"] = None
    _woke(fleet)
    fresh = _restart(fleet, tmp_path)
    fresh.hooks.advance(45)
    fresh.tick()
    assert fresh.state["watches"]["47694876"]["standing_dormant"] is False
    assert "jobs_watch_standing_resumed" in events(fresh)


def test_the_ticket_placed_rpc_is_additive_and_never_refuses(fleet):
    srv = daemon.Server(fleet, sock_path=os.path.join(fleet.dir, "s.sock"))
    arm(fleet)
    fleet.tick()
    drain(fleet)
    ok, data, err = srv.handle({"v": client.FLEET_PROTO_VERSION,
                                "op": "ticket_placed",
                                "args": {"target": "47694876", "job_id": "j-3",
                                         "source": "job retarget"}})
    assert ok and err is None and data["woken"] is True
    # an unwatched box is answered, not refused — the CLI has already written
    ok, data, err = srv.handle({"v": client.FLEET_PROTO_VERSION,
                                "op": "ticket_placed",
                                "args": {"target": "5", "source": "job submit"}})
    assert ok and data["woken"] is False


# --------------------------------------------------------------------------- #
# 8. `herdd start` must not strand the standing flags
# --------------------------------------------------------------------------- #
def test_operator_start_leaves_the_standing_dormancy_to_the_ticket(fleet):
    """`herdd start` announces `operator_intent(start)`, which cleared
    `dormant` and left `standing_dormant` set. `_standing_tick` runs off
    `dormant`, so the resume gate stopped running entirely: the ladder ticked
    while every readout still called the watch dormant, and the resume event
    never fired (0 of 84 drains on the live daemon, 2026-08-27)."""
    arm(fleet)
    fleet.tick()
    drain(fleet)
    fleet.hooks.resume("47694876")
    fleet.operator_intent("47694876", "start", requester="free@rig")
    w = fleet.state["watches"]["47694876"]
    assert w["standing_dormant"] is True and w["dormant"] is True
    fleet.hooks.drained_map["47694876"] = False   # a wave is waiting after all
    fleet.hooks.advance(45)
    fleet.tick()
    assert fleet.state["watches"]["47694876"]["standing_dormant"] is False
    assert event(fleet, "jobs_watch_standing_resumed")[0]["trigger"] == "queue"


def test_operator_start_still_re_arms_a_plain_dormant_watch(fleet):
    """Unchanged for everything that is not standing: an operator park goes
    dormant on intent and the matching start takes it straight back."""
    arm(fleet, standing=False)
    fleet.tick()
    fleet.operator_intent("47694876", "stop", requester="free@rig")
    fleet.hooks.park("47694876")
    fleet.hooks.advance(45)
    fleet.tick()
    assert fleet.state["watches"]["47694876"]["dormant"] is True
    fleet.hooks.resume("47694876")
    fleet.operator_intent("47694876", "start", requester="free@rig")
    w = fleet.state["watches"]["47694876"]
    assert w["dormant"] is False and w["dormant_reason"] is None


def test_a_stranded_standing_flag_is_repaired_on_the_next_tick(fleet):
    """Records written by the old code are already on disk in that state, so
    the tick repairs them rather than waiting for the next drain."""
    arm(fleet)
    fleet.tick()
    drain(fleet)
    w = fleet.state["watches"]["47694876"]
    w["dormant"], w["dormant_reason"] = False, None      # the pre-fix shape
    fleet.hooks.resume("47694876")
    fleet.hooks.drained_map["47694876"] = True           # queue still terminal
    n = len(fleet.hooks.jobs_ticks)
    fleet.hooks.advance(45)
    fleet.tick()
    w = fleet.state["watches"]["47694876"]
    assert w["dormant"] is True and w["state"] == "standing"
    assert len(fleet.hooks.jobs_ticks) == n, "a dormant watch does not tick the ladder"
