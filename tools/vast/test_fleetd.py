"""Portable tests for fleetd — the persistent fleet-supervision daemon.

No network, no vast API, no systemd, no real clock: every I/O touch goes
through an injected `FakeHooks` (the fake-transport discipline test_supervise.py
/ test_jobd.py established), and the systemd unit is checked by string
inspection of the generator.

Covered: every row of FLEETD_DESIGN §3's tick table (watched run/jobs/bare,
paused, unwatched, pending-destroy), pause expiry, budget breach -> PARK,
deferred-destroy conditions + exactly-once, state round-trip through a restart,
the single-instance lock, and the socket request/response protocol.
"""
import argparse
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bidpolicy                                                 # noqa: E402
import subprocess                                                # noqa: E402
import vastconf                                                  # noqa: E402
import herdd                                                   # noqa: E402
from vastlib.cli.fleet import restart as cli_fleet_restart       # noqa: E402
from vastlib.core import config as vastlib_config                # noqa: E402
from vastlib.fleet import client, daemon, deploy                 # noqa: E402
from vastlib.fleet import hooks as fleet_hooks                   # noqa: E402
from vastlib.fleet import rows as fleet_rows                     # noqa: E402
from vastlib.fleet import state as fleet_state                   # noqa: E402


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakeHooks:
    """Scripted stand-in for every fleetd I/O touch."""

    def __init__(self, dry_run=False):
        self.t = 1_700_000_000.0
        self.dry_run = dry_run
        self.boxes = {}                 # iid(str) -> instance body
        self.api_down = False
        self.parked, self.resumed, self.destroyed = [], [], []
        self.park_ok, self.destroy_ok = True, True
        self.drained_map = {}           # iid -> True/False/None
        self.run_result = None          # Action or None returned by run_tick
        self.jobs_result = None         # verdict or None
        self.run_spend, self.jobs_spend = 0.0, 0.0
        self.jobs_handoff = []          # (kind, fields) the next jobs_tick emits
        self.jobs_iid = None            # box the ladder moved the watch onto
        self.jobs_understudy = None     # handoff understudy the ladder named
        self.finalized, self.run_ticks, self.jobs_ticks = [], [], []
        self.health_map = {}            # iid -> gather_fleet_health row
        self.results_map = {}           # iid -> True/False/None
        self.kept = []                  # iids whose label got the reap keep token

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

    def instances(self):
        return None if self.api_down else list(self.boxes.values())

    def park(self, iid):
        self.parked.append(str(iid))
        if self.park_ok:
            b = self.boxes.get(str(iid))
            if b:
                b["actual_status"] = b["intended_status"] = "stopped"
            return True, None
        return False, "park refused"

    def resume(self, iid):
        self.resumed.append(str(iid))
        return True, None

    def destroy(self, iid):
        self.destroyed.append(str(iid))
        if self.destroy_ok:
            self.boxes.pop(str(iid), None)
            return True, None
        return False, "destroy refused"

    def drained(self, iid):
        return self.drained_map.get(str(iid))

    def instance(self, iid):
        return self.boxes.get(str(iid))

    def health(self, instances):
        return dict(self.health_map)

    def results_present(self, iid):
        return self.results_map.get(str(iid))

    def keep_label(self, iid, inst):
        b = self.boxes.get(str(iid))
        label = (b or {}).get("label") or ""
        if "keep" in [t.strip() for t in label.split(":")]:
            return False, label
        new_label = (label + ":keep") if label else "keep:fleetd-park"
        if b is not None:
            b["label"] = new_label
        self.kept.append(str(iid))
        return True, new_label

    # profile ticks (the imported supervise ladders, faked)
    def run_init(self, a):
        return ({"run_id": a.run_id, "spend_usd": 0.0, "instance_id": None,
                 "actual_status": "running", "relaunch_count": 0}, {"phase": "IDLE"},
                True)

    def run_tick(self, st, a, hf, handoff_on):
        self.run_ticks.append(st["run_id"])
        st["spend_usd"] = self.run_spend
        st["instance_id"] = st.get("instance_id") or 111
        return self.run_result

    def run_finalize(self, st, a, act, hf, handoff_on):
        self.finalized.append((st["run_id"], act.kind))

    def jobs_init(self, a):
        return ({"a": a, "iid": str(a.id), "spend_usd": 0.0,
                 "handoff_on": True}, {"phase": "IDLE"})

    def jobs_tick(self, jc, hf):
        self.jobs_ticks.append(jc["iid"])
        jc["spend_usd"] = self.jobs_spend
        # The ladder can swap the box under the watch (eviction replacement, SLA
        # relaunch, completed handoff) and can name a handoff UNDERSTUDY while
        # the migration is still in flight. Both are money-moving box identities
        # the daemon has to bind to the watch's ceiling, so both are scriptable
        # here — same fake-transport seam as everything else.
        if self.jobs_iid is not None:
            jc["iid"] = str(self.jobs_iid)
        if self.jobs_understudy is not None:
            hf["understudy_iid"] = str(self.jobs_understudy)
        # the jobs ladder leaves its handoff decisions on `jc` for the caller to
        # drain (herdd._do_job_handoff_move / _job_handoff_tick); scripting the
        # queue here is the same fake-transport seam every other hook uses.
        if self.jobs_handoff:
            jc.setdefault("handoff_journal", []).extend(self.jobs_handoff)
            self.jobs_handoff = []
        return self.jobs_result


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    monkeypatch.setenv("FLEETD_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("FLEETD_GLOBAL_BUDGET_USD", raising=False)
    monkeypatch.delenv("FLEETD_UNWATCHED_GRACE_S", raising=False)
    monkeypatch.delenv("FLEETD_UNWATCHED_GRACE_EXPENSIVE_S", raising=False)
    monkeypatch.delenv("FLEETD_EXPENSIVE_DPH", raising=False)
    h = FakeHooks()
    return daemon.Fleet(str(tmp_path / "state"), hooks=h)


def journal(f):
    if not os.path.exists(f.journal_path):
        return []
    with open(f.journal_path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def events(f):
    return [r["event"] for r in journal(f)]


def observe(f, seconds, step=None):
    """Drive real reconcile ticks across `seconds` of OBSERVED time (N7: grace
    clocks advance on successful observations, capped at MAX_OBS_DT_S)."""
    step = step or daemon.MAX_OBS_DT_S
    f.tick()
    left = seconds
    while left > 0:
        f.hooks.advance(min(step, left))
        f.tick()
        left -= min(step, left)


# --------------------------------------------------------------------------- #
# §3 row 1 — watched (run profile): the imported ladder drives the box
# --------------------------------------------------------------------------- #
def test_watched_run_profile_ticks_the_supervise_ladder(fleet):
    fleet.hooks.box(111, label="run:20260729-x")
    fleet.watch("run:20260729-x", "run", budget_usd=5.0, policy={"budget": 5.0})
    fleet.hooks.run_spend = 1.25
    fleet.tick()
    assert fleet.hooks.run_ticks == ["20260729-x"]
    w = fleet.state["watches"]["run:20260729-x"]
    assert w["spend_usd"] == 1.25 and w["iid"] == "111"
    assert "watch_adopted" in events(fleet)


def test_run_profile_terminal_finalizes_and_drops_the_watch(fleet):
    fleet.hooks.box(111, label="run:r1")
    fleet.watch("run:r1", "run", budget_usd=5.0)
    fleet.hooks.run_result = bidpolicy.Action("stop_terminal", "terminal:done")
    fleet.tick()
    assert fleet.hooks.finalized == [("r1", "stop_terminal")]
    assert "run:r1" not in fleet.state["watches"]
    assert "watch_finished" in events(fleet)


def test_run_watch_requires_a_budget(fleet):
    with pytest.raises(ValueError):
        fleet.watch("run:r1", "run")


# --------------------------------------------------------------------------- #
# §3 row 1 — watched (jobs profile)
# --------------------------------------------------------------------------- #
def test_watched_jobs_profile_ticks_and_finishes(fleet):
    fleet.hooks.box(222)
    fleet.watch("222", "jobs", budget_usd=5.0, policy={"budget": 5.0, "id": 222})
    fleet.hooks.jobs_spend = 0.75
    fleet.tick()
    assert fleet.hooks.jobs_ticks == ["222"]
    assert fleet.state["watches"]["222"]["spend_usd"] == 0.75
    fleet.hooks.jobs_result = "drained"
    fleet.tick()
    assert "222" not in fleet.state["watches"]
    fin = [r for r in journal(fleet) if r["event"] == "watch_finished"]
    assert fin and fin[-1]["verdict"] == "drained"


# --------------------------------------------------------------------------- #
# §3 row 1 — watched (bare/serve): accrue + budget breach -> PARK, never destroy
# --------------------------------------------------------------------------- #
def test_bare_profile_accrues_spend_from_dph(fleet):
    fleet.hooks.box(333, dph=3.6)                    # $0.001/s
    fleet.watch("333", "bare", budget_usd=100.0)
    fleet.tick()                                     # first tick seeds the clock
    fleet.hooks.advance(1000)
    fleet.tick()
    assert fleet.state["watches"]["333"]["spend_usd"] == pytest.approx(1.0, rel=1e-3)


def test_accrual_bills_storage_only_while_loading(fleet):
    """Billing-model fidelity (invoice-verified 2026-08-02, box 46633685: GPU
    hours 0.000 for an all-`loading` life): a loading box accrues the STORAGE
    rate, not dph — the old dph-while-LIVE_STATES accrual overstated a
    28-minute loading zombie 3.8x and was quoted as a billing measurement."""
    fleet.hooks.box(333, status="loading", dph=3.6, storage_total_cost=0.36)
    fleet.watch("333", "bare", budget_usd=100.0)
    fleet.tick()
    fleet.hooks.advance(1000)
    fleet.tick()
    # 1000 s at $0.36/hr storage = $0.10 — NOT $1.00 of GPU dph.
    assert fleet.state["watches"]["333"]["spend_usd"] == pytest.approx(0.1, rel=1e-3)


def test_accrual_zero_while_loading_with_unknown_storage_rate(fleet):
    """No storage rate readable -> accrue nothing during loading (never invent
    a rate), and start billing dph the moment the box reaches `running`."""
    fleet.hooks.box(333, status="loading", dph=3.6)
    fleet.watch("333", "bare", budget_usd=100.0)
    fleet.tick()
    fleet.hooks.advance(1000)
    fleet.tick()
    assert fleet.state["watches"]["333"]["spend_usd"] == pytest.approx(0.0)
    fleet.hooks.boxes["333"]["actual_status"] = "running"
    fleet.hooks.advance(1000)                        # first running interval
    fleet.tick()
    fleet.hooks.advance(1000)
    fleet.tick()
    # the flip tick bills its whole preceding interval at dph (per-tick status,
    # deliberately conservative — never undercounts running time):
    assert fleet.state["watches"]["333"]["spend_usd"] == pytest.approx(2.0, rel=1e-3)


def test_budget_breach_parks_and_alarms_never_destroys(fleet):
    fleet.hooks.box(333, dph=36.0)
    fleet.watch("333", "bare", budget_usd=1.0)
    fleet.tick()
    fleet.hooks.advance(1000)
    fleet.tick()
    assert fleet.hooks.parked == ["333"]
    assert fleet.hooks.destroyed == []
    assert "budget_parked" in events(fleet)
    assert any("BUDGET" in a for a in fleet.alarms)
    # the watch survives a budget park (resumable, not a terminal failure)
    assert fleet.state["watches"]["333"]["state"] == "budget_parked"
    assert fleet.hooks.kept == ["333"]                 # B4 reap keep token
    assert fleet.state["watches"]["333"]["dormant"] is True   # S8


def test_budget_park_failure_keeps_alarming(fleet):
    fleet.hooks.box(333, dph=36.0)
    fleet.hooks.park_ok = False
    fleet.watch("333", "bare", budget_usd=1.0)
    fleet.tick()
    fleet.hooks.advance(1000)
    fleet.tick()
    assert "budget_park_failed" in events(fleet)
    assert any("PARK FAILED" in a or "park FAILED" in a or "PARK" in a.upper()
               for a in fleet.alarms)
    assert fleet.hooks.destroyed == []


def test_global_budget_ceiling_alarms_and_freezes_but_parks_nothing(fleet, monkeypatch):
    """N3: the fleet ceiling is an alarm + a spend freeze, never a mass park."""
    monkeypatch.setenv("FLEETD_GLOBAL_BUDGET_USD", "0.5")
    fleet.hooks.box(333, dph=36.0)
    fleet.watch("333", "bare", budget_usd=None)
    fleet.tick()
    fleet.hooks.advance(100)
    fleet.tick()
    assert fleet.hooks.parked == []
    assert any("FLEET budget" in a for a in fleet.alarms)
    assert "global_budget_breached" in events(fleet)
    with pytest.raises(ValueError):                    # new spend-capable watch
        fleet.watch("999", "jobs", budget_usd=5.0)


def test_global_breach_suspends_money_moves(fleet, monkeypatch):
    monkeypatch.setenv("FLEETD_GLOBAL_BUDGET_USD", "0.01")
    fleet.hooks.box(222)
    fleet.watch("222", "jobs", budget_usd=5.0, policy={"id": 222})
    fleet.state["spend_by_box"]["222"] = 99.0
    fleet.tick()
    assert fleet.hooks.jobs_ticks == []
    assert "tick_suspended_global_budget" in events(fleet)


# --------------------------------------------------------------------------- #
# §3 row 2 — paused
# --------------------------------------------------------------------------- #
def test_pause_suspends_the_ladder_and_touches_nothing(fleet):
    fleet.hooks.box(222)
    fleet.watch("222", "jobs", budget_usd=5.0, policy={"id": 222})
    fleet.pause("222", 900, reason="interruption drill")
    fleet.tick()
    assert fleet.hooks.jobs_ticks == []               # box left alone
    assert fleet.hooks.parked == [] and fleet.hooks.destroyed == []
    assert "tick_paused" in events(fleet)
    assert fleet.state["watches"]["222"]["state"] == "paused"


def test_pause_auto_expires_back_into_supervision(fleet):
    fleet.hooks.box(222)
    fleet.watch("222", "jobs", budget_usd=5.0, policy={"id": 222})
    fleet.pause("222", 900)
    fleet.tick()
    fleet.hooks.advance(901)
    fleet.tick()
    assert "pause_expired" in events(fleet)
    assert fleet.hooks.jobs_ticks == ["222"]          # rejoined by itself


def test_pause_is_always_bounded(fleet):
    fleet.hooks.box(222)
    fleet.watch("222", "jobs", budget_usd=5.0, policy={"id": 222})
    r = fleet.pause("222", 10 ** 9)
    assert r["until"] - fleet.hooks.now() == pytest.approx(daemon.MAX_PAUSE_S)


def test_pause_zero_clears(fleet):
    fleet.hooks.box(222)
    fleet.watch("222", "jobs", budget_usd=5.0, policy={"id": 222})
    fleet.pause("222", 900)
    fleet.pause("222", 0)
    fleet.tick()
    assert fleet.hooks.jobs_ticks == ["222"]
    assert "pause_cleared" in events(fleet)


def test_pause_works_on_an_unwatched_target(fleet):
    """S1: `fleet pause` must work on a stray — the safety net skips it."""
    fleet.hooks.box(999)
    r = fleet.pause("999", 600, reason="drill")
    assert r["kind"] == "stray"
    fleet.tick()
    assert fleet.hooks.parked == []
    assert "tick_paused" in events(fleet)


def test_pause_keeps_accruing_and_alarms_on_breach_parking_at_expiry(fleet):
    """S1: a pause suspends ACTIONS; observation + accrual continue, a breach
    alarms immediately and the park happens at expiry, never mid-drill."""
    fleet.hooks.box(333, dph=36.0)
    fleet.watch("333", "bare", budget_usd=1.0)
    fleet.tick()
    fleet.pause("333", 600)
    fleet.hooks.advance(300)
    fleet.tick()
    assert fleet.hooks.parked == []                    # never mid-pause
    assert fleet.state["watches"]["333"]["spend_usd"] > 1.0
    assert any("DURING A PAUSE" in a for a in fleet.alarms)
    fleet.hooks.advance(400)                           # pause expires
    fleet.tick()
    assert fleet.hooks.parked == ["333"]


# --------------------------------------------------------------------------- #
# §3 row 3 — unwatched: alarm every tick, PARK (never destroy) after the grace
# --------------------------------------------------------------------------- #
def test_unwatched_no_evidence_box_alarms_every_tick(fleet):
    fleet.hooks.box(444, dph=1.5)                     # no label, no health row
    fleet.tick()
    assert "unwatched" in events(fleet)
    assert any("UNWATCHED" in a for a in fleet.alarms)
    assert fleet.hooks.parked == []                   # still inside the grace
    fleet.hooks.advance(60)
    fleet.tick()
    assert events(fleet).count("unwatched") == 2


# --- B1: the safety net is EVIDENCE-GATED ---------------------------------- #
def test_unwatched_but_busy_box_is_adopted_never_parked(fleet):
    fleet.hooks.box(444, dph=1.5, label="serve:eval")
    observe(fleet, daemon.UNWATCHED_GRACE_S * 2)
    assert fleet.hooks.parked == []
    assert "unwatched_adopted" in events(fleet)
    w = fleet.state["watches"]["444"]
    # `bare` and adopted, as before — but NOT uncapped. Since the ceiling ledger
    # (2026-08-09) an adoption with nothing to inherit carries the conservative
    # provisional default; `budget_usd is None` here is the defect, not the
    # contract.
    assert w["profile"] == "bare" and w["adopted"]
    assert w["budget_usd"] == vastconf.fleetd_adopt_default_budget_usd()
    assert w["ceiling_source"] == "default"
    assert any("AUTO-ADOPTED" in a.upper() for a in fleet.alarms)


def test_fresh_jobd_heartbeat_is_evidence(fleet):
    fleet.hooks.box(444)
    fleet.hooks.health_map["444"] = {"verdict": "OK",
                                     "evidence": {"jobd_hb_age_s": 30}}
    observe(fleet, daemon.UNWATCHED_GRACE_S * 2)
    assert fleet.hooks.parked == []
    assert "unwatched_adopted" in events(fleet)


def test_a_measured_zombie_is_not_saved_by_its_label(fleet):
    """A run: label must not rescue a box guard already calls dead."""
    fleet.hooks.box(444, label="run:r9")
    fleet.hooks.health_map["444"] = {"verdict": "ZOMBIE_NO_JOBD",
                                     "reason": "jobd hb stale",
                                     "evidence": {"is_jobs_box": True,
                                                  "jobd_hb_age_s": 99999}}
    observe(fleet, daemon.UNWATCHED_GRACE_S)     # ends ON the parking tick
    assert fleet.hooks.parked == ["444"]
    assert any("HEALTH ZOMBIE_NO_JOBD" in a for a in fleet.alarms)


def test_zombie_health_alarm_is_journaled_once_not_just_displayed(fleet):
    """Box 46256890 (2026-07-30) sat 3 h as a loading-stall zombie: the alarm
    lived only in the per-tick `fleet status` list, which is rebuilt every tick
    and visible only to a caller who happens to look. A zombie verdict must
    also land in the JOURNAL — once per verdict transition, not once per tick —
    and its clearing too, so the timeline survives in the record."""
    fleet.hooks.box(444, label="run:r9")
    fleet.hooks.health_map["444"] = {"verdict": "ZOMBIE_LOADING_STALL",
                                     "reason": "stuck in loading for 3h",
                                     "evidence": {"is_jobs_box": True}}
    observe(fleet, daemon.UNWATCHED_GRACE_S)          # many ticks, one verdict
    rows = [r for r in journal(fleet) if r["event"] == "health_alarm"]
    assert len(rows) == 1, rows
    assert rows[0]["iid"] == "444" and rows[0]["verdict"] == "ZOMBIE_LOADING_STALL"
    assert rows[0]["reason"]                          # auditable, not bare

    fleet.hooks.health_map["444"] = {"verdict": "OK", "evidence": {}}
    fleet.hooks.box(444, label="run:r9")              # re-add (park stopped it)
    observe(fleet, 300)
    cleared = [r for r in journal(fleet) if r["event"] == "health_alarm_cleared"]
    assert len(cleared) == 1 and cleared[0]["iid"] == "444"


def test_advisory_health_alarms_but_offers_no_destroy_remedy(fleet):
    """velvet P1: STALE_IMAGE alarms and journals like a zombie, but the remedy
    line must NOT be `guard --fix` — that destroys, and a stale-image box is
    healthy. Suggesting it would train operators to destroy warm disks."""
    fleet.hooks.box(444, label="run:r9")
    fleet.hooks.health_map["444"] = {"verdict": "STALE_IMAGE",
                                     "reason": "registry tag moved since launch",
                                     "evidence": {"image_state": "stale"}}
    observe(fleet, daemon.UNWATCHED_GRACE_S)
    alarm = [a for a in fleet.alarms if "STALE_IMAGE" in a]
    assert alarm, fleet.alarms
    assert "guard --fix" not in alarm[0]
    assert "destroy+relaunch" in alarm[0]
    rows = [r for r in journal(fleet) if r["event"] == "health_alarm"]
    assert len(rows) == 1 and rows[0]["advisory"] is True


def test_advisory_verdict_does_not_count_as_a_zombie_for_the_stray_net(fleet):
    """workload_evidence keys off _GUARD_ZOMBIE_VERDICTS. STALE_IMAGE must stay
    out of that set, or a healthy stale box loses its workload evidence and the
    stray net parks it."""
    # stays-on-flat: the rename table homes both frozensets at
    # `vastlib.boxes.health.GuardVerdict`, which absorbed them into an enum's
    # `.is_zombie` / `.is_advisory` predicates — there is no set to test `in`
    # against. Repointing would rewrite the assertion, which step 6e is not
    # licensed to do (plumbing only). Hand this to the health/GuardVerdict
    # owner with the flat frozensets.
    assert "STALE_IMAGE" not in herdd._GUARD_ZOMBIE_VERDICTS
    assert "STALE_IMAGE" in herdd._GUARD_ADVISORY_VERDICTS


def test_nofleet_label_is_exempt_from_the_safety_net(fleet):
    """B1c: workflowctl's escape hatch — fleetd ignores the box entirely."""
    fleet.hooks.box(444, label="stage:x:nofleet")
    observe(fleet, daemon.UNWATCHED_GRACE_S * 2)
    assert fleet.hooks.parked == []
    assert "unwatched" not in events(fleet)
    assert "444" not in fleet.state["watches"]


def test_workload_evidence_is_pure_and_ordered():
    assert fleet_rows.workload_evidence({"label": "run:x"}) == "label 'run:x'"
    assert fleet_rows.workload_evidence({"label": ""}) is None
    assert fleet_rows.workload_evidence(
        {"label": "run:x"}, {"verdict": "ZOMBIE_LOADING_STALL"}) is None
    assert fleet_rows.workload_evidence(
        {}, {"verdict": "OK", "evidence": {"boot_age_s": 60}}).startswith("booted")


def test_unwatched_box_is_parked_after_the_grace_never_destroyed(fleet):
    fleet.hooks.box(444, dph=1.5)
    observe(fleet, daemon.UNWATCHED_GRACE_S)
    assert fleet.hooks.parked == ["444"]
    assert fleet.hooks.destroyed == []
    assert "unwatched_parked" in events(fleet)


# --------------------------------------------------------------------------- #
# owner ruling 2026-07-29 — the unwatched-grace fuse is price-aware: more
# grace for cheap boxes ("<$2/hour is pretty cheap"), less for expensive ones
# ("more than that really needs to be managed properly").
# --------------------------------------------------------------------------- #
def test_cheap_tier_box_parks_only_after_the_base_grace(fleet):
    fleet.hooks.box(444, dph=1.5)                      # < FLEETD_EXPENSIVE_DPH
    observe(fleet, daemon.UNWATCHED_GRACE_S - 60)
    assert fleet.hooks.parked == []                     # not yet at 1800s
    rows = [r for r in journal(fleet) if r["event"] == "unwatched"]
    assert rows and rows[-1]["tier"] == "cheap"
    assert rows[-1]["grace_s"] == daemon.UNWATCHED_GRACE_S
    fleet.hooks.advance(60)
    fleet.tick()
    assert fleet.hooks.parked == ["444"]
    parked = [r for r in journal(fleet) if r["event"] == "unwatched_parked"][-1]
    assert parked["tier"] == "cheap"


def test_expensive_tier_box_parks_after_the_short_grace(fleet):
    fleet.hooks.box(444, dph=5.0)                       # >= FLEETD_EXPENSIVE_DPH
    observe(fleet, daemon.UNWATCHED_GRACE_EXPENSIVE_S - 30)
    assert fleet.hooks.parked == []                     # not yet at 300s
    rows = [r for r in journal(fleet) if r["event"] == "unwatched"]
    assert rows and rows[-1]["tier"] == "expensive"
    assert rows[-1]["grace_s"] == daemon.UNWATCHED_GRACE_EXPENSIVE_S
    fleet.hooks.advance(30)
    fleet.tick()
    assert fleet.hooks.parked == ["444"]
    parked = [r for r in journal(fleet) if r["event"] == "unwatched_parked"][-1]
    assert parked["tier"] == "expensive"
    # the short fuse must actually be shorter than the base one
    assert daemon.UNWATCHED_GRACE_EXPENSIVE_S < daemon.UNWATCHED_GRACE_S


def test_unknown_dph_is_treated_as_expensive(fleet):
    """Missing/unparseable dph fails toward the SHORT fuse — the evidence
    gate already protects a genuinely busy box regardless of tier."""
    fleet.hooks.box(444, dph=None)
    observe(fleet, daemon.UNWATCHED_GRACE_EXPENSIVE_S)
    assert fleet.hooks.parked == ["444"]
    rows = [r for r in journal(fleet) if r["event"] == "unwatched"]
    assert rows and rows[-1]["tier"] == "expensive"
    assert rows[-1].get("dph_known") is False


def test_unparseable_dph_is_treated_as_expensive(fleet):
    fleet.hooks.box(444, dph="not-a-number")
    observe(fleet, daemon.UNWATCHED_GRACE_EXPENSIVE_S)
    assert fleet.hooks.parked == ["444"]


def test_boundary_dph_exactly_threshold_is_expensive(fleet):
    """Owner: 'more than that' needs management — >= keeps the strict side
    of the $2.00 boundary safe."""
    fleet.hooks.box(444, dph=2.0)
    observe(fleet, daemon.UNWATCHED_GRACE_EXPENSIVE_S)
    assert fleet.hooks.parked == ["444"]
    rows = [r for r in journal(fleet) if r["event"] == "unwatched"]
    assert rows and rows[-1]["tier"] == "expensive"


def test_busy_expensive_box_still_auto_adopts_fuse_never_arms(fleet):
    """B1 outranks the price-aware fuse: workload evidence adopts the box
    regardless of how expensive it is."""
    fleet.hooks.box(444, dph=12.0, label="serve:eval")
    observe(fleet, daemon.UNWATCHED_GRACE_EXPENSIVE_S * 3)
    assert fleet.hooks.parked == []
    assert "unwatched_adopted" in events(fleet)
    assert "unwatched_parked" not in events(fleet)


def test_expensive_tier_clock_does_not_fast_forward_on_outage(fleet, monkeypatch):
    """N7 must hold per-tier too: an outage cannot fast-forward the SHORT
    fuse past the per-tick observation cap (MAX_OBS_DT_S). The expensive
    grace is raised above the cap here so a single capped recovery tick
    cannot itself satisfy it — otherwise this is indistinguishable from the
    boundary case where grace_s == MAX_OBS_DT_S."""
    monkeypatch.setenv("FLEETD_UNWATCHED_GRACE_EXPENSIVE_S",
                       str(daemon.MAX_OBS_DT_S * 2))
    fleet.hooks.box(444, dph=5.0)
    fleet.tick()
    fleet.hooks.api_down = True
    fleet.hooks.advance(daemon.MAX_OBS_DT_S * 20)      # a long real outage
    fleet.tick()                                       # clocks do not advance
    fleet.hooks.api_down = False
    fleet.tick()                                        # one recovery tick: capped
    assert fleet.hooks.parked == []                     # only MAX_OBS_DT_S credited


def test_expensive_dph_threshold_env_override(fleet, monkeypatch):
    monkeypatch.setenv("FLEETD_EXPENSIVE_DPH", "10")
    fleet.hooks.box(444, dph=5.0)                       # below the raised bar
    observe(fleet, daemon.UNWATCHED_GRACE_EXPENSIVE_S)
    assert fleet.hooks.parked == []                     # cheap tier now


def test_unwatched_grace_expensive_env_override(fleet, monkeypatch):
    monkeypatch.setenv("FLEETD_UNWATCHED_GRACE_EXPENSIVE_S", "60")
    fleet.hooks.box(444, dph=5.0)
    observe(fleet, 60)
    assert fleet.hooks.parked == ["444"]


def test_one_outage_cannot_fast_forward_the_grace_clock(fleet):
    """N7: age advances on OBSERVED time only."""
    fleet.hooks.box(444)
    fleet.tick()
    fleet.hooks.api_down = True
    fleet.hooks.advance(daemon.UNWATCHED_GRACE_S * 10)
    fleet.tick()
    fleet.hooks.api_down = False
    fleet.tick()
    assert fleet.hooks.parked == []


def test_safety_net_park_stamps_the_reap_keep_token(fleet):
    """B4: `herdd reap` destroys idle stopped boxes past 2h without a keep
    token — every fleetd park must survive it."""
    fleet.hooks.box(444, label="scratch")
    observe(fleet, daemon.UNWATCHED_GRACE_S)
    assert fleet.hooks.kept == ["444"]
    assert fleet.hooks.boxes["444"]["label"].endswith(":keep")
    assert "keep_label_stamped" in events(fleet)


def test_unwatched_grace_zero_is_alarm_only(fleet, monkeypatch):
    monkeypatch.setenv("FLEETD_UNWATCHED_GRACE_S", "0")
    fleet.hooks.box(444)
    observe(fleet, 4 * daemon.UNWATCHED_GRACE_S)
    assert fleet.hooks.parked == []
    assert any("UNWATCHED" in a for a in fleet.alarms)


def test_parked_stray_is_not_alarmed(fleet):
    fleet.hooks.box(444, status="stopped")
    fleet.tick()
    assert "unwatched" not in events(fleet)
    assert fleet.alarms == []


def test_watching_a_stray_clears_the_alarm(fleet):
    fleet.hooks.box(444)
    fleet.tick()
    fleet.watch("444", "bare")
    fleet.tick()
    assert "444" not in fleet.state["strays"]
    assert not any("UNWATCHED" in a for a in fleet.alarms)


def test_unwatch_is_not_amnesty(fleet):
    fleet.hooks.box(444)
    fleet.watch("444", "bare")
    fleet.tick()
    fleet.unwatch("444")
    fleet.tick()
    assert "unwatched" in events(fleet)                # the safety net re-applies


# --------------------------------------------------------------------------- #
# §3 row 4 — pending destroy (the ONLY destroy path)
# --------------------------------------------------------------------------- #
def test_destroy_requires_explicit_yes(fleet):
    with pytest.raises(ValueError):
        fleet.request_destroy("444", "now", yes=False)


def test_destroy_now_executes_and_is_journaled_with_requester(fleet):
    fleet.hooks.box(444)
    fleet.request_destroy("444", "now", reason="drill done", requester="me@host",
                          yes=True)
    fleet.tick()
    assert fleet.hooks.destroyed == ["444"]
    rec = [r for r in journal(fleet) if r["event"] == "destroyed"][0]
    assert rec["requester"] == "me@host" and rec["reason"] == "drill done"


def test_deferred_destroy_when_parked_waits_for_the_condition(fleet):
    fleet.hooks.box(444, status="running")
    fleet.request_destroy("444", "parked", yes=True)
    fleet.tick()
    assert fleet.hooks.destroyed == []
    fleet.hooks.boxes["444"]["actual_status"] = "stopped"
    fleet.tick()
    assert fleet.hooks.destroyed == []                 # S3: one tick is not enough
    assert "destroy_condition_pending" in events(fleet)
    fleet.hooks.advance(daemon.DESTROY_CONFIRM_S)      # ...nor is a fast second
    fleet.tick()
    assert fleet.hooks.destroyed == ["444"]


def test_deferred_destroy_condition_must_hold_consecutively(fleet):
    fleet.hooks.box(444, status="stopped")
    fleet.request_destroy("444", "parked", yes=True)
    fleet.tick()
    fleet.hooks.advance(daemon.DESTROY_CONFIRM_S)
    fleet.hooks.boxes["444"]["actual_status"] = "running"   # flapped back
    fleet.tick()
    fleet.hooks.boxes["444"]["actual_status"] = "stopped"
    fleet.tick()
    assert fleet.hooks.destroyed == []                 # streak AND dwell restarted
    fleet.hooks.advance(daemon.DESTROY_CONFIRM_S)
    fleet.tick()
    assert fleet.hooks.destroyed == ["444"]


def test_deferred_destroy_holds_when_results_are_missing(fleet):
    fleet.hooks.box(444, status="stopped")
    fleet.hooks.results_map["444"] = False
    fleet.request_destroy("444", "parked", yes=True)
    for _ in range(4):
        fleet.tick()
        fleet.hooks.advance(daemon.DESTROY_CONFIRM_S)
    assert fleet.hooks.destroyed == []
    assert "destroy_deferred_no_results" in events(fleet)
    fleet.request_destroy("444", "parked", yes=True, results_check=False)
    fleet.tick()
    fleet.hooks.advance(daemon.DESTROY_CONFIRM_S)
    fleet.tick()
    assert fleet.hooks.destroyed == ["444"]


def test_destroy_request_expires(fleet):
    fleet.hooks.box(444, status="running")
    fleet.request_destroy("444", "parked", yes=True)
    fleet.tick()
    fleet.hooks.advance(daemon.DESTROY_TTL_S + 1)
    fleet.tick()
    assert "destroy_expired" in events(fleet)
    assert "444" not in fleet.state["destroys"]


def test_deferred_destroy_when_drained_waits_for_the_queue(fleet):
    fleet.hooks.box(444)
    fleet.hooks.drained_map["444"] = False
    fleet.request_destroy("444", "drained", yes=True)
    fleet.tick()
    assert fleet.hooks.destroyed == []
    fleet.hooks.drained_map["444"] = True
    fleet.tick()
    fleet.hooks.advance(daemon.DESTROY_CONFIRM_S)
    fleet.tick()
    assert fleet.hooks.destroyed == ["444"]


def test_destroy_executes_at_most_once(fleet):
    fleet.hooks.box(444)
    fleet.request_destroy("444", "now", yes=True)
    fleet.tick()
    fleet.hooks.box(444)                               # box id reappears
    fleet.tick()
    assert fleet.hooks.destroyed == ["444"]            # not destroyed twice


def test_failed_destroy_is_retried_next_tick(fleet):
    fleet.hooks.box(444)
    fleet.hooks.destroy_ok = False
    fleet.request_destroy("444", "now", yes=True)
    fleet.tick()
    assert "destroy_failed" in events(fleet)
    fleet.hooks.box(444)
    fleet.hooks.destroy_ok = True
    fleet.tick()
    assert fleet.hooks.destroyed == ["444", "444"]
    assert "444" not in fleet.state["destroys"]


def test_destroy_auto_unwatches_first(fleet):
    """S4: a queued destroy on an actively-watched box unwatches it up front."""
    fleet.hooks.box(444)
    fleet.watch("444", "bare")
    r = fleet.request_destroy("444", "now", yes=True)
    assert r["unwatched"] == ["444"]
    assert "444" not in fleet.state["watches"]
    fleet.tick()
    assert fleet.hooks.destroyed == ["444"]


def test_unknown_when_is_rejected(fleet):
    with pytest.raises(ValueError):
        fleet.request_destroy("444", "someday", yes=True)


# --------------------------------------------------------------------------- #
# explicit park/resume requests
# --------------------------------------------------------------------------- #
def test_park_request_is_executed_by_the_daemon(fleet):
    fleet.hooks.box(555)
    fleet.watch("555", "bare")
    fleet.request_action("555", "park", reason="done", requester="me")
    fleet.tick()
    assert fleet.hooks.parked == ["555"]
    assert "parked" in events(fleet)


def test_resume_request_on_a_stray(fleet):
    fleet.hooks.box(555, status="running")
    fleet.tick()
    fleet.request_action("555", "resume")
    fleet.tick()
    assert fleet.hooks.resumed == ["555"]


# --------------------------------------------------------------------------- #
# api outage, restart round-trip, single-instance lock
# --------------------------------------------------------------------------- #
def test_api_outage_changes_nothing(fleet):
    fleet.hooks.box(444)
    fleet.hooks.api_down = True
    fleet.tick()
    assert fleet.hooks.parked == [] and fleet.hooks.destroyed == []
    assert events(fleet) == ["api_unavailable"]
    assert fleet.state["strays"] == {}


def test_state_survives_a_restart(tmp_path):
    h = FakeHooks()
    d = str(tmp_path / "st")
    f1 = daemon.Fleet(d, hooks=h)
    h.box(222)
    f1.watch("222", "jobs", budget_usd=5.0, policy={"id": 222})
    f1.pause("222", 900, reason="drill")
    h.jobs_spend = 2.0
    f1.tick()
    f2 = daemon.Fleet(d, hooks=h)                       # "daemon restart"
    w = f2.state["watches"]["222"]
    assert w["profile"] == "jobs" and w["budget_usd"] == 5.0
    assert w["paused_until"] == f1.state["watches"]["222"]["paused_until"]
    h.advance(901)
    f2.tick()                                           # rebuilt runtime, resumes
    assert h.jobs_ticks == ["222"]
    assert f2.state["watches"]["222"]["spend_usd"] >= 2.0


def test_state_write_is_atomic(fleet):
    fleet.watch("222", "bare")
    assert os.path.exists(fleet.state_path)
    assert not os.path.exists(fleet.state_path + ".tmp")
    json.load(open(fleet.state_path))                   # parseable


def test_single_instance_lock(tmp_path):
    d = str(tmp_path / "lockdir")
    os.makedirs(d)
    fh = fleet_state.acquire_single_instance_lock(d)
    assert fh is not None
    assert fleet_state.acquire_single_instance_lock(d) is None    # second daemon refused
    fh.close()
    fh2 = fleet_state.acquire_single_instance_lock(d)             # released -> available
    assert fh2 is not None
    fh2.close()


def test_watch_init_failure_is_retried_not_fatal(fleet):
    def boom(a):
        raise RuntimeError("B2 down")
    fleet.hooks.jobs_init = boom
    fleet.hooks.box(222)
    fleet.watch("222", "jobs", budget_usd=5.0, policy={"id": 222})
    fleet.tick()
    assert "watch_init_failed" in events(fleet)
    assert "222" in fleet.state["watches"]


def test_dry_run_hooks_mutate_nothing(monkeypatch):
    monkeypatch.setenv("FLEETD_DRY_RUN", "1")
    h = fleet_hooks.Hooks()
    assert h.dry_run is True
    assert h.park(1)[0] and h.destroy(1)[0] and h.resume(1)[0]


# --------------------------------------------------------------------------- #
# socket protocol (golden request/response, no real socket needed)
# --------------------------------------------------------------------------- #
@pytest.fixture
def server(fleet):
    return daemon.Server(fleet, sock_path=os.path.join(fleet.dir, "s.sock"))


def test_proto_ping(server):
    ok, data, err = server.handle({"v": 1, "op": "ping"})
    assert ok and data["version"] == client.FLEET_PROTO_VERSION and err is None


def test_proto_watch_then_status(server):
    server.fleet.hooks.box(222)
    ok, data, _ = server.handle({"v": 1, "op": "watch",
                                 "args": {"target": "222", "profile": "jobs",
                                          "budget_usd": 5.0,
                                          "policy": {"id": 222}}})
    assert ok and data == {"target": "222", "profile": "jobs", "iid": "222",
                           "spend_usd": 0.0, "redirected_from": None,
                           # the cap that LANDED + the headroom under it: an
                           # inherited ceiling can differ from the figure asked
                           # for, and the client prints both
                           "budget_usd": 5.0, "ceiling_id": "222",
                           "ceiling_source": "explicit", "remaining_usd": 5.0}
    ok, data, _ = server.handle({"v": 1, "op": "status"})
    assert ok and data["rows"][0]["iid"] == "222"


def test_proto_watch_without_budget_is_refused(server):
    ok, _d, err = server.handle({"v": 1, "op": "watch",
                                 "args": {"target": "222", "profile": "jobs"}})
    assert not ok and "budget" in err


def test_proto_destroy_requires_yes(server):
    ok, _d, err = server.handle({"v": 1, "op": "destroy",
                                 "args": {"target": "1", "when": "now"}})
    assert not ok and "yes" in err


def test_proto_unknown_op_and_malformed(server):
    ok, _d, err = server.handle({"v": 1, "op": "nope"})
    assert not ok and "unknown op" in err
    assert server.handle("not a dict")[2] == "malformed request"
    assert server.handle({"op": "ping", "args": []})[2] == "malformed args"
    ok, _d, err = server.handle({"v": 99, "op": "ping"})
    assert not ok and "protocol version" in err


def test_proto_pause_and_spend(server):
    server.fleet.hooks.box(222)
    server.handle({"v": 1, "op": "watch",
                   "args": {"target": "222", "profile": "bare"}})
    ok, data, _ = server.handle({"v": 1, "op": "pause",
                                 "args": {"target": "222", "seconds": 60,
                                          "reason": "drill"}})
    assert ok and data["until_iso"].endswith("Z")
    ok, data, _ = server.handle({"v": 1, "op": "spend", "args": {}})
    assert ok and "total_usd" in data


def test_proto_errors_never_raise(server):
    ok, _d, err = server.handle({"v": 1, "op": "unwatch",
                                 "args": {"target": "nope"}})
    assert not ok and "no watch" in err


def test_socket_round_trip(fleet, tmp_path):
    """One real AF_UNIX round trip (local socket only — no network)."""
    import socket as _s
    import threading
    srv = daemon.Server(fleet, sock_path=str(tmp_path / "rt.sock"))
    srv.bind()
    assert oct(os.stat(srv.sock_path).st_mode)[-3:] == "600"
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        c = _s.socket(_s.AF_UNIX, _s.SOCK_STREAM)
        c.settimeout(5)
        c.connect(srv.sock_path)
        c.sendall(json.dumps({"v": 1, "op": "ping"}).encode() + b"\n")
        resp = json.loads(c.recv(65536).decode().splitlines()[0])
        c.close()
        assert resp["ok"] and resp["data"]["version"] == client.FLEET_PROTO_VERSION
    finally:
        srv.close()
        t.join(timeout=5)


# --------------------------------------------------------------------------- #
# systemd unit generator — string inspection only (never installed in tests)
# --------------------------------------------------------------------------- #
def test_unit_text_is_generated_with_runtime_paths():
    txt = deploy.render_unit("/usr/bin/python3", "/x/tools/vast/fleetd.py", "/x")
    assert "ExecStart=/usr/bin/python3 /x/tools/vast/fleetd.py serve" in txt
    assert "WorkingDirectory=/x" in txt
    assert "Restart=always" in txt
    assert "WantedBy=default.target" in txt
    assert "FLEETD_DRY_RUN" not in txt


def test_unit_text_can_bake_dry_run():
    txt = deploy.render_unit("/p", "/s", "/r", dry_run=True)
    assert "Environment=FLEETD_DRY_RUN=1" in txt


# --------------------------------------------------------------------------- #
# the release checkout — the unit must run a KNOWN revision, not an incidental
# one. Every case below is a shape that actually mis-deployed the daemon.
# --------------------------------------------------------------------------- #
def test_repo_root_is_the_repo_not_the_tools_dir():
    """It was `dirname(_HERE)` == <repo>/tools until 2026-08-09, which put a
    doubled `tools/tools` in the generated unit and made `_env_stat` watch a
    path that never exists — so `.env` hot-reload never fired once."""
    root = daemon.repo_root()
    assert os.path.basename(root) != "tools"
    assert os.path.isfile(os.path.join(root, "tools", "vast", "fleetd.py"))


def _init_repo(path, branch="main"):
    import subprocess
    os.makedirs(path, exist_ok=True)
    for args in (["init", "-q", "-b", branch], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", path, *args], check=True,
                       capture_output=True)
    open(os.path.join(path, "f.txt"), "w").write("x")
    subprocess.run(["git", "-C", path, "add", "f.txt"], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", path, "commit", "-qm", "c"], check=True,
                   capture_output=True)
    return path


def test_audit_passes_a_clean_main_checkout_outside_scratch(tmp_path):
    ok = _init_repo(str(tmp_path / "release"))
    assert deploy.checkout_audit(ok) == []


def test_audit_refuses_a_checkout_on_another_branch(tmp_path):
    import subprocess
    p = _init_repo(str(tmp_path / "release"))
    subprocess.run(["git", "-C", p, "checkout", "-qb", "peer-wip"], check=True,
                   capture_output=True)
    bad = deploy.checkout_audit(p)
    assert any("peer-wip" in b for b in bad), bad


def test_audit_refuses_a_scratch_path(tmp_path):
    """`out/land-main` was the previous mitigation. It got swept, and with it
    the unit's ExecStart — the next restart would have crash-looped."""
    p = _init_repo(str(tmp_path / "out" / "land-main"))
    assert any("scratch" in b for b in deploy.checkout_audit(p))


def test_audit_refuses_a_linked_worktree(tmp_path):
    import subprocess
    src = _init_repo(str(tmp_path / "src"))
    wt = str(tmp_path / "wt")
    subprocess.run(["git", "-C", src, "worktree", "add", "-q", "-b", "lane", wt],
                   check=True, capture_output=True)
    bad = deploy.checkout_audit(wt)
    assert any("LINKED WORKTREE" in b for b in bad), bad


def test_audit_refuses_a_dirty_tree(tmp_path):
    p = _init_repo(str(tmp_path / "release"))
    open(os.path.join(p, "f.txt"), "w").write("edited")
    assert any("modified" in b for b in deploy.checkout_audit(p))


def test_audit_refuses_a_missing_or_non_git_path(tmp_path):
    assert deploy.checkout_audit(str(tmp_path / "nope"))
    plain = tmp_path / "plain"
    plain.mkdir()
    assert any("not a git checkout" in b
               for b in deploy.checkout_audit(str(plain)))


def test_deploy_checkout_path_is_env_overridable_and_home_relative(monkeypatch):
    monkeypatch.delenv(deploy.DEPLOY_CHECKOUT_ENV, raising=False)
    assert deploy.deploy_checkout_path() == os.path.expanduser(
        deploy.DEPLOY_CHECKOUT_DEFAULT)
    assert "~" in deploy.DEPLOY_CHECKOUT_DEFAULT, "no absolute machine path in git"
    monkeypatch.setenv(deploy.DEPLOY_CHECKOUT_ENV, "/tmp/elsewhere")
    assert deploy.deploy_checkout_path() == "/tmp/elsewhere"


def test_deploy_refuses_a_dirty_release_checkout(tmp_path, capsys):
    """A restart would still 'succeed'; `rev=` just would not describe what
    runs. Refuse before writing the unit."""
    p = _init_repo(str(tmp_path / "release"))
    open(os.path.join(p, "f.txt"), "w").write("edited")
    a = argparse.Namespace(checkout=p, source=p, ref="HEAD", python=None,
                           no_restart=True, force=False, dry_run=False,
                           verify_timeout=1.0)
    assert deploy.cmd_deploy(a) == 1
    assert "uncommitted" in capsys.readouterr().out


def test_deploy_writes_the_unit_and_links_env_without_restarting(tmp_path,
                                                                monkeypatch,
                                                                capsys):
    """The whole write path, minus systemd: unit points at the RELEASE tree, and
    the gitignored `.env` is linked so the daemon does not come up blind."""
    src = _init_repo(str(tmp_path / "src"))
    open(os.path.join(src, ".env"), "w").write("K=V\n")
    rel = str(tmp_path / "release")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    os.makedirs(os.path.join(rel, "tools", "vast"), exist_ok=True)
    _init_repo(rel)
    open(os.path.join(rel, "tools", "vast", "fleetd.py"), "w").write("#\n")
    import subprocess
    subprocess.run(["git", "-C", rel, "add", "-A"], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", rel, "commit", "-qm", "tools"], check=True,
                   capture_output=True)
    a = argparse.Namespace(checkout=rel, source=src, ref="HEAD",
                           python=sys.executable, no_restart=True, force=False,
                           dry_run=False, verify_timeout=1.0)
    assert deploy.cmd_deploy(a) == 0
    unit = os.path.join(str(tmp_path / "home"), ".config", "systemd", "user",
                        client.FLEET_UNIT_NAME)
    txt = open(unit).read()
    assert f"ExecStart={sys.executable} {rel}/tools/vast/fleetd.py serve" in txt
    assert f"WorkingDirectory={rel}" in txt
    assert os.path.realpath(os.path.join(rel, ".env")) == \
        os.path.realpath(os.path.join(src, ".env"))
    assert "restart" in capsys.readouterr().out.lower()


def test_deploy_refuses_when_execstart_would_not_exist(tmp_path, monkeypatch,
                                                       capsys):
    """The 2026-08-09 shape: a unit whose ExecStart is gone does not deploy the
    wrong code, it crash-loops. Never write one."""
    rel = _init_repo(str(tmp_path / "release"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    a = argparse.Namespace(checkout=rel, source=rel, ref="HEAD", python=None,
                           no_restart=True, force=False, dry_run=False,
                           verify_timeout=1.0)
    assert deploy.cmd_deploy(a) == 1
    assert "cannot exec" in capsys.readouterr().out


def test_verify_live_rev_fails_when_the_daemon_reports_another_rev(monkeypatch,
                                                                  capsys):
    """`systemctl restart` returning 0 is not evidence. On 2026-08-07 it was 0
    and the daemon was unchanged."""
    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: (True, {"rev": "oldrev1", "pid": 1}, None))
    assert deploy._verify_live_rev("newrev2", deadline_s=5.0) == 3
    out = capsys.readouterr().out
    assert "NOT VERIFIED" in out and "oldrev1" in out


def test_verify_live_rev_passes_on_a_match(monkeypatch, capsys):
    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: (True, {"rev": "newrev2", "pid": 7}, None))
    assert deploy._verify_live_rev("newrev2", deadline_s=5.0) == 0
    assert "VERIFIED" in capsys.readouterr().out


def test_install_unit_refuses_an_unfit_checkout(tmp_path, monkeypatch, capsys):
    """`install-unit` can only ever deploy the tree it runs from, so it is the
    one that must say no."""
    monkeypatch.setattr(daemon, "repo_root", lambda: str(tmp_path / "gone"))
    a = argparse.Namespace(no_enable=True, dry_run=False, force=False)
    assert daemon.cmd_install_unit(a) == 2
    assert "refusing to install" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# policy namespace rebuild (the daemon runs the SAME tick fns with the SAME flags)
# --------------------------------------------------------------------------- #
def test_make_policy_seeds_non_none_defaults():
    a = daemon.make_policy("run", {"budget": 5.0}, "run:r1")
    assert a.run_id == "r1" and a.budget == 5.0
    assert a.interval == 45 and a.max_relaunch == 3 and a.handoff is True
    assert a.dry_run is False and a.no_fleet is True
    assert a.something_never_defined is None            # missing != AttributeError


def test_make_policy_jobs_target_is_the_iid():
    a = daemon.make_policy("jobs", {"budget": 5.0}, "46177923")
    assert a.id == 46177923 and a.keep is False and a.strict_ceiling is False


def test_client_and_daemon_agree_on_the_socket_path(monkeypatch, tmp_path):
    monkeypatch.setenv("FLEETD_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("FLEETD_SOCK", raising=False)
    assert client.fleet_sock_path() == str(tmp_path / "fleetd.sock")
    assert client.fleet_journal_path() == str(tmp_path / "journal.ndjsonl")


def test_client_falls_back_when_no_daemon(monkeypatch, tmp_path):
    monkeypatch.setenv("FLEETD_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("FLEETD_SOCK", raising=False)
    ok, _d, err = client.fleet_request("ping")
    assert not ok and err.startswith("nodaemon:")
    assert client.fleet_daemon_up() is False


# --------------------------------------------------------------------------- #
# B2 — operator intent: the daemon must never resurrect a box a human parked
# --------------------------------------------------------------------------- #
def test_operator_stop_makes_a_jobs_watch_dormant(fleet):
    """The jobs ladder reads 'bid box stopped, no self-park' as OUTBID and
    RESCUES it. Operator intent must win over that inference."""
    fleet.hooks.box(222, is_bid=True)
    fleet.watch("222", "jobs", budget_usd=5.0, policy={"id": 222})
    fleet.tick()
    assert fleet.hooks.jobs_ticks == ["222"]
    r = fleet.operator_intent("222", "stop", requester="me@host")
    assert r["watched"] and "DORMANT" in r["note"]
    fleet.hooks.boxes["222"]["actual_status"] = "stopped"
    fleet.tick()
    fleet.tick()
    assert fleet.hooks.jobs_ticks == ["222"]           # ladder never ran again
    assert fleet.hooks.resumed == [] and fleet.hooks.parked == []
    assert "watch_dormant" in events(fleet)
    assert fleet.state["watches"]["222"]["dormant"] is True


def test_dormant_watch_does_not_alarm_every_tick(fleet):
    fleet.hooks.box(222)
    fleet.watch("222", "jobs", budget_usd=5.0, policy={"id": 222})
    fleet.operator_intent("222", "stop")
    fleet.hooks.boxes["222"]["actual_status"] = "stopped"
    fleet.tick()
    assert fleet.alarms == []                          # S8


def test_operator_start_rearms_the_watch(fleet):
    fleet.hooks.box(222)
    fleet.watch("222", "jobs", budget_usd=5.0, policy={"id": 222})
    fleet.operator_intent("222", "stop")
    fleet.hooks.boxes["222"]["actual_status"] = "stopped"
    fleet.tick()
    fleet.operator_intent("222", "start")
    fleet.hooks.boxes["222"]["actual_status"] = "running"
    fleet.tick()
    assert fleet.state["watches"]["222"]["dormant"] is False
    assert fleet.hooks.jobs_ticks[-1] == "222"


def test_fleet_resume_rearms_a_dormant_watch(fleet):
    fleet.hooks.box(222)
    fleet.watch("222", "jobs", budget_usd=5.0, policy={"id": 222})
    fleet.operator_intent("222", "stop")
    fleet.tick()
    fleet.request_action("222", "resume")
    fleet.tick()
    assert fleet.hooks.resumed == ["222"]
    assert fleet.state["watches"]["222"]["dormant"] is False


def test_unknown_intent_is_rejected(fleet):
    with pytest.raises(ValueError):
        fleet.operator_intent("1", "vaporize")


def test_fleet_park_marks_the_watch_dormant_not_evicted(fleet):
    """S8: fleetd's own park must not be re-classified as an eviction next tick."""
    fleet.hooks.box(222)
    fleet.watch("222", "jobs", budget_usd=5.0, policy={"id": 222})
    fleet.request_action("222", "park", reason="done for the night")
    fleet.tick()
    n = len(fleet.hooks.jobs_ticks)
    fleet.tick()
    assert fleet.hooks.jobs_ticks[n:] == []             # ladder stays off
    assert fleet.state["watches"]["222"]["dormant"] is True
    assert fleet.hooks.kept == ["222"]                  # B4 keep token


# --------------------------------------------------------------------------- #
# S2/S4/S6 — spend durability, watch identity, version skew
# --------------------------------------------------------------------------- #
def test_spend_is_backfilled_across_a_daemon_restart(tmp_path):
    h = FakeHooks()
    d = str(tmp_path / "st")
    h.box(222, dph=36.0)
    f1 = daemon.Fleet(d, hooks=h)
    f1.watch("222", "bare", budget_usd=100.0)
    f1.tick()
    h.advance(100)
    f1.tick()                                          # accrues, persists
    h.advance(3600)                                    # daemon DOWN for an hour
    f2 = daemon.Fleet(d, hooks=h)
    w = f2.state["watches"]["222"]
    f2._spend_backfill(w)
    assert w["spend_usd"] > 36.0                       # the downtime was charged
    assert "spend_backfilled" in events(f2)


def test_one_watch_per_instance(fleet):
    """S4: re-registering the SAME target upserts; a DIFFERENT target resolving
    to an already-watched instance is refused."""
    fleet.hooks.box(222)
    fleet.watch("222", "bare")
    fleet.watch("222", "jobs", budget_usd=1.0)          # upsert: fine
    fleet.state["watches"]["run:r1"] = dict(fleet.state["watches"]["222"],
                                            target="run:r1", profile="run",
                                            iid="222")
    del fleet.state["watches"]["222"]
    with pytest.raises(ValueError):                     # collides with run:r1
        fleet.watch("222", "bare")


def test_iid_watch_dies_with_its_instance(fleet):
    fleet.hooks.box(222)
    fleet.watch("222", "bare")
    fleet.tick()
    fleet.hooks.boxes.pop("222")
    fleet.tick()
    assert "222" in fleet.state["watches"]             # one missing tick: wait
    fleet.tick()
    assert "222" not in fleet.state["watches"]
    assert journal(fleet)[-1]["verdict"] == "instance_gone"


def test_run_watch_follows_the_label_across_a_relaunch(fleet):
    fleet.hooks.box(111, label="run:r1")
    fleet.watch("run:r1", "run", budget_usd=5.0)
    fleet.tick()
    fleet.hooks.boxes.pop("111")
    fleet.hooks.box(999, label="run:r1")               # relaunched box
    fleet.tick()
    assert fleet.state["watches"]["run:r1"]["iid"] in ("999", "111")
    assert "run:r1" in fleet.state["watches"]          # never dropped


def test_ping_reports_a_rev_for_skew_detection(server):
    ok, data, _ = server.handle({"v": 1, "op": "ping"})
    assert ok and "rev" in data


def test_policy_dict_is_overlaid_on_daemon_defaults(fleet):
    """S6: a stale state.json watch missing a newer flag must not AttributeError.

    `handoff` is False on the jobs profile since 2026-08-08 (SAFE-OFF, see
    vastconf.JOBS_HANDOFF_UNSAFE_KEY) and True on the run lane, which is
    untouched — so this pins both, and the missing-flag contract on each."""
    a = daemon.make_policy("jobs", {"id": 5, "budget": 1.0}, "5")
    assert a.keep is False and a.handoff is False and a.flag_added_next_month is None
    r = daemon.make_policy("run", {"budget": 1.0}, "run:r1")
    assert r.handoff is True and r.flag_added_next_month is None


# --------------------------------------------------------------------------- #
# B3 — the client must never be strictly worse than the pre-fleetd world
# --------------------------------------------------------------------------- #
def _ns(**kw):
    import argparse as _a
    kw.setdefault("dry_run", False)
    kw.setdefault("no_fleet", False)
    kw.setdefault("budget", 5.0)
    return _a.Namespace(**kw)


def test_daemon_refusal_is_distinguishable_from_transport_trouble(fleet, tmp_path):
    import threading
    srv = daemon.Server(fleet, sock_path=str(tmp_path / "r.sock"))
    srv.bind()
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        os.environ["FLEETD_SOCK"] = srv.sock_path
        ok, _d, err = client.fleet_request("watch", target="1", profile="jobs")
        assert not ok and err.startswith("refused:") and "budget" in err
        ok, data, err = client.fleet_request("ping")
        assert ok and err is None
    finally:
        os.environ.pop("FLEETD_SOCK", None)
        srv.close()
        t.join(timeout=5)


def test_delegation_falls_back_to_the_inline_loop_on_transport_trouble(monkeypatch):
    for err in ("nodaemon:FileNotFoundError", "timeout:no response",
                "socket:Connection reset by peer"):
        monkeypatch.setattr(client, "fleet_request",
                            lambda *a, _e=err, **k: (False, None, _e))
        assert client.fleet_delegate_job_supervise(_ns(id=42)) is False


def test_delegation_surfaces_a_real_daemon_refusal(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)   # exercise the shim
    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: (False, None, "refused:budget required"))
    with pytest.raises(SystemExit):
        client.fleet_delegate_job_supervise(_ns(id=42))


def test_delegation_is_skipped_for_dry_run_and_no_fleet(monkeypatch):
    called = []
    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: called.append(a) or (True, {}, None))
    assert client.fleet_delegate_job_supervise(_ns(id=42, dry_run=True)) is False
    assert client.fleet_delegate_job_supervise(_ns(id=42, no_fleet=True)) is False
    assert called == []


def test_a_test_run_never_delegates_to_a_live_daemon(monkeypatch):
    """PYTEST_CURRENT_TEST is set: the supervise driver tests must never
    register a watch with a real daemon on the developer's workstation."""
    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: (True, {"target": "42"}, None))
    assert client._fleet_delegation_disabled(_ns(id=42)) is True
    assert client.fleet_delegate_job_supervise(_ns(id=42)) is False


def test_delegation_registers_and_reports_true(monkeypatch, capsys):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)   # exercise the shim
    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: (True, {"target": "42"}, None))
    assert client.fleet_delegate_job_supervise(_ns(id=42)) is True
    out = capsys.readouterr().out
    assert "registered watch 42" in out and "deprecated" in out


def test_operator_intent_helper_is_best_effort(monkeypatch):
    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: (False, None, "nodaemon:x"))
    assert client.fleet_operator_intent(1, "stop") is None
    assert client.fleet_watch_best_effort(1, "bare") is False


# --------------------------------------------------------------------------- #
# 2026-07-30 live defect — an EXPLICIT watch must never be downgraded to `bare`
#
# Observed on box 46240842 (journal, verbatim order): 00:33:35 auto-adopted
# `bare` (evidence=booting) -> 00:34:31 `watch_registered` profile=jobs
# budget=$5 -> 00:34:59 `watch_adopted` -> 00:35:00 `watch_finished`
# verdict=queue_empty -> 00:35:00 `watch_auto_adopted` profile=bare. The wave
# had not submitted yet, so the jobs ladder's inline-CLI "submit first" exit
# killed the watch one second in and the stray sweep re-adopted the box with NO
# budget cap. At the ~02:0x spot preemption there was no jobs ladder left to run
# the outbid rescue and the box had to be retargeted by hand.
# --------------------------------------------------------------------------- #
def test_queue_empty_does_not_end_a_jobs_watch(fleet):
    """`queue_empty` == nothing submitted YET (pre-submission); `drained` is the
    finished verdict. Only the latter may end a watch."""
    fleet.hooks.box(46240842, label="upstream-monorepo")
    fleet.watch("46240842", "jobs", budget_usd=5.0, policy={"id": 46240842},
                requester="operator@workstation")
    fleet.hooks.jobs_result = "queue_empty"
    fleet.tick()
    w = fleet.state["watches"].get("46240842")
    assert w is not None, "an unsubmitted queue must not drop the watch"
    assert w["profile"] == "jobs" and w["budget_usd"] == 5.0
    assert w["adopted"] is False
    assert "watch_finished" not in events(fleet)
    assert "jobs_queue_empty" in events(fleet)
    assert any("QUEUE IS EMPTY" in a for a in fleet.alarms)


def test_queue_empty_never_hands_the_box_to_the_bare_safety_net(fleet):
    """The full live sequence: the same tick that ended the watch re-adopted the
    box as `bare` with no cap. Neither half may happen now."""
    fleet.hooks.box(46240842, label="upstream-monorepo")
    fleet.watch("46240842", "jobs", budget_usd=5.0, policy={"id": 46240842})
    fleet.hooks.jobs_result = "queue_empty"
    observe(fleet, daemon.UNWATCHED_GRACE_S * 2)        # many ticks, hours of it
    w = fleet.state["watches"]["46240842"]
    assert (w["profile"], w["budget_usd"]) == ("jobs", 5.0)
    assert "unwatched_adopted" not in events(fleet)
    assert "watch_auto_adopted" not in events(fleet)
    assert fleet.hooks.parked == []                     # and never parked


def test_queue_filling_resumes_the_full_jobs_ladder(fleet):
    fleet.hooks.box(46240842)
    fleet.watch("46240842", "jobs", budget_usd=5.0, policy={"id": 46240842})
    fleet.hooks.jobs_result = "queue_empty"
    fleet.tick()
    n = len(fleet.hooks.jobs_ticks)
    fleet.hooks.jobs_result = None                      # wave submitted
    fleet.hooks.advance(45)
    fleet.tick()
    assert len(fleet.hooks.jobs_ticks) > n              # ladder kept being driven
    assert "jobs_queue_filled" in events(fleet)
    assert "queue_empty_since" not in fleet.state["watches"]["46240842"]
    fleet.hooks.jobs_result = "drained"                 # a REAL terminal still ends it
    fleet.hooks.advance(45)
    fleet.tick()
    assert "46240842" not in fleet.state["watches"]
    fin = [r for r in journal(fleet) if r["event"] == "watch_finished"]
    assert len(fin) == 1 and fin[0]["verdict"] == "drained"


def test_an_outbid_jobs_box_still_reaches_the_rescue_ladder(fleet):
    """The invariant the live loss violated: a preempted box that was watched as
    `jobs` is still driven by the jobs ladder, under its own budget cap — not by
    the money-moveless `bare` path."""
    fleet.hooks.box(46240842, is_bid=True, label="upstream-monorepo")
    fleet.watch("46240842", "jobs", budget_usd=5.0, policy={"id": 46240842})
    fleet.hooks.jobs_result = "queue_empty"             # registered pre-submission
    fleet.tick()
    fleet.hooks.jobs_result = None                      # wave submits, then...
    fleet.hooks.advance(45)
    fleet.tick()
    # ... spot preemption: bid box shows stopped with no self-park event
    fleet.hooks.boxes["46240842"]["actual_status"] = "exited"
    fleet.hooks.boxes["46240842"]["intended_status"] = "stopped"
    n = len(fleet.hooks.jobs_ticks)
    fleet.hooks.advance(45)
    fleet.tick()
    assert fleet.hooks.jobs_ticks[n:] == ["46240842"], \
        "the outbid box must still be handed to the jobs ladder"
    assert fleet.runtime["46240842"]["a"].budget == 5.0
    w = fleet.state["watches"]["46240842"]
    assert w["profile"] == "jobs" and not w["adopted"] and not w.get("dormant")


# --------------------------------------------------------------------------- #
# 2026-07-31 live defect — `unrecoverable` must not end a watch on a box that
# still exists.
#
# Observed on box 46347213 (journal, verbatim order): ~04:29 spot preemption ->
# 04:33:18 `watch_finished` verdict=unrecoverable (jobs+$5 watch) -> 04:42 the
# box auto-resumed on its OWN standing bid -> 04:42:36 `watch_auto_adopted`
# profile=bare (no budget). Because the watch was gone, the was_live->live
# reattach never pushed a current jobd; onstart re-pulled the stale launch-
# pinned bundle, whose truncated-VRAM scheduler bug silently matched no ticket,
# and the box billed full GPU doing nothing until an operator intervened.
# --------------------------------------------------------------------------- #
def test_unrecoverable_keeps_the_watch_while_the_box_still_exists(fleet):
    """`unrecoverable` == the ladder gave up, NOT the box is gone. While the
    instance is still in the API listing the daemon must alarm and keep the
    watch (budget + reattach-on-resume), never end it."""
    fleet.hooks.box(46347213, label="upstream-monorepo")
    fleet.watch("46347213", "jobs", budget_usd=5.0, policy={"id": 46347213},
                requester="operator@workstation")
    fleet.hooks.jobs_result = "unrecoverable"
    fleet.tick()
    w = fleet.state["watches"].get("46347213")
    assert w is not None, "a stalled rescue must not drop the watch"
    assert w["profile"] == "jobs" and w["budget_usd"] == 5.0
    assert w["state"] == "unrecoverable"
    assert "watch_finished" not in events(fleet)
    assert "jobs_rescue_stalled" in events(fleet)
    assert any("RESCUE STALLED" in a for a in fleet.alarms)


# --------------------------------------------------------------------------- #
# A WEDGED replacement lane (REPLACEMENT_CEILING_WEDGE_2026-08-24).
#
# For 33 minutes the pull-reschedule lane emitted `jobs_box_launch_failed` every
# tick and nothing in the system escalated: `BOOT_MAX_HOST_RETRIES` counts
# SUCCESSFUL relaunches so a lane that can never launch never trips it, and
# `rescue_stalled` needs `unrecoverable`, which a pull-condemned box never
# reaches. The one alarm that did fire (ZOMBIE_LOADING_STALL, at +16 min) named
# the wrong thing — it prescribed parking the box, not the ceiling that was
# $0.013 under the only offer in the market.
# --------------------------------------------------------------------------- #
def _wedged(fleet, iid="48537477", **repl):
    fleet.hooks.box(int(iid), label="upstream-monorepo")
    fleet.watch(iid, "jobs", budget_usd=60.0, policy={"id": int(iid)})
    w = fleet.state["watches"][iid]
    w["iid"] = iid
    base = {"replacement_refusals": daemon.REPLACEMENT_WEDGE_REFUSALS,
            "replacement_refusals_since": fleet.hooks.now() - 600.0,
            "replacement_refusal_reason": "over_ceiling",
            "replacement_refusal_ceiling": 0.387,
            "replacement_market_floor": 0.40}
    base.update(repl)
    w["replacement"] = base
    return w


def test_a_repeatedly_refusing_replacement_lane_alarms(fleet):
    """A single refusal is the ladder working. A refusal that repeats every tick
    is a wedge, and it must be LOUD."""
    _wedged(fleet)
    msgs = [a for a in fleet.alarms if "REPLACEMENT WEDGED" in a]
    assert len(msgs) == 1
    m = msgs[0]
    assert "over_ceiling" in m and "$0.387" in m
    assert "$0.4000/hr" in m, "the alarm must name the MARKET GAP, not just the bound"
    assert "--replace-ceiling-mult" in m, "and the move that clears it"


def test_the_wedge_alarm_holds_its_fire_below_the_threshold(fleet):
    """Transient market blips are not wedges."""
    _wedged(fleet, replacement_refusals=daemon.REPLACEMENT_WEDGE_REFUSALS - 1)
    assert not [a for a in fleet.alarms if "REPLACEMENT WEDGED" in a]


def test_the_wedge_alarm_retracts_itself_when_a_rental_succeeds(fleet):
    """Derived, not latched: the streak counter is cleared by any rental that
    actually happened, and the alarm goes with it."""
    w = _wedged(fleet)
    assert [a for a in fleet.alarms if "REPLACEMENT WEDGED" in a]
    w["replacement"] = {k: v for k, v in w["replacement"].items()
                        if not k.startswith("replacement_refusal")}
    assert not [a for a in fleet.alarms if "REPLACEMENT WEDGED" in a]


def test_the_wedge_alarm_says_so_when_the_market_is_genuinely_empty(fleet):
    """`no_offer` with no observed price is a real state — an empty market, not
    a priced one we refused. The alarm must not invent a gap it never read."""
    _wedged(fleet, replacement_refusal_reason="no_offer",
            replacement_market_floor=None)
    m = [a for a in fleet.alarms if "REPLACEMENT WEDGED" in a][0]
    assert "no qualifying offer seen at any price" in m


def test_unrecoverable_never_hands_the_box_to_the_bare_safety_net(fleet):
    """The full live sequence: the tick after the watch ended, the stray sweep
    re-adopted the box as `bare` with no cap. Neither half may happen now."""
    fleet.hooks.box(46347213, label="upstream-monorepo")
    fleet.watch("46347213", "jobs", budget_usd=5.0, policy={"id": 46347213})
    fleet.hooks.jobs_result = "unrecoverable"
    observe(fleet, daemon.UNWATCHED_GRACE_S * 2)
    w = fleet.state["watches"]["46347213"]
    assert (w["profile"], w["budget_usd"]) == ("jobs", 5.0)
    assert "unwatched_adopted" not in events(fleet)
    assert "watch_auto_adopted" not in events(fleet)
    assert fleet.hooks.parked == []


def test_unrecoverable_recovery_rearms_the_full_jobs_ladder(fleet):
    """An auto-resume (our own standing bid regaining priority) must put the
    ladder back in charge — and a REAL terminal verdict still ends the watch."""
    fleet.hooks.box(46347213, label="upstream-monorepo")
    fleet.watch("46347213", "jobs", budget_usd=5.0, policy={"id": 46347213})
    fleet.hooks.jobs_result = "unrecoverable"
    fleet.tick()
    n = len(fleet.hooks.jobs_ticks)
    fleet.hooks.jobs_result = None                      # box came back
    fleet.hooks.advance(45)
    fleet.tick()
    assert len(fleet.hooks.jobs_ticks) > n              # ladder kept being driven
    assert "jobs_rescue_recovered" in events(fleet)
    assert "unrecoverable_since" not in fleet.state["watches"]["46347213"]
    assert not any("RESCUE STALLED" in a for a in fleet.alarms)
    fleet.hooks.jobs_result = "drained"                 # a REAL terminal still ends it
    fleet.hooks.advance(45)
    fleet.tick()
    assert "46347213" not in fleet.state["watches"]
    fin = [r for r in journal(fleet) if r["event"] == "watch_finished"]
    assert len(fin) == 1 and fin[0]["verdict"] == "drained"


def test_unrecoverable_box_truly_gone_dies_via_instance_gone(fleet):
    """A box that leaves the API listing is dead for real — the watch must
    still be reaped (through the instance_gone path, not `unrecoverable`)."""
    fleet.hooks.box(46347213, label="upstream-monorepo")
    fleet.watch("46347213", "jobs", budget_usd=5.0, policy={"id": 46347213})
    fleet.hooks.jobs_result = "unrecoverable"
    fleet.tick()
    del fleet.hooks.boxes["46347213"]                   # destroyed / host death
    for _ in range(daemon.GONE_CONFIRM_TICKS):
        fleet.hooks.advance(45)
        fleet.tick()
    assert "46347213" not in fleet.state["watches"]
    fin = [r for r in journal(fleet) if r["event"] == "watch_finished"]
    assert len(fin) == 1 and fin[0]["verdict"] == "instance_gone"


def test_auto_adopt_never_downgrades_an_explicit_watch(fleet):
    """The authoritative seam: adoption racing a `fleet watch` (accept thread vs
    reconcile thread) must not blank profile/budget."""
    fleet.hooks.box(222, label="upstream-monorepo")
    fleet.watch("222", "jobs", budget_usd=5.0, policy={"id": 222})
    got = fleet.watch("222", "bare", None, {"evidence": "booting"},
                      requester="fleetd:auto-adopt", adopted=True)
    assert (got["profile"], got["budget_usd"]) == ("jobs", 5.0)
    w = fleet.state["watches"]["222"]
    assert (w["profile"], w["budget_usd"], w["adopted"]) == ("jobs", 5.0, False)
    assert "auto_adopt_refused" in events(fleet)
    # an explicit re-registration is still a legitimate operator downgrade
    fleet.watch("222", "bare")
    assert fleet.state["watches"]["222"]["profile"] == "bare"


def test_auto_adopt_refusal_matches_on_the_resolved_iid_too(fleet):
    """A run watch is keyed by label; the box arrives by iid. The guard must see
    through the key."""
    fleet.hooks.box(111, label="run:r1")
    fleet.watch("run:r1", "run", budget_usd=5.0)
    fleet.tick()                                        # resolves iid -> 111
    assert fleet.state["watches"]["run:r1"]["iid"] == "111"
    got = fleet.watch("111", "bare", None, requester="fleetd:auto-adopt",
                      adopted=True)
    assert got["target"] == "run:r1" and got["budget_usd"] == 5.0
    assert "111" not in fleet.state["watches"]


def test_a_busy_explicitly_watched_box_is_never_swept_as_a_stray(fleet):
    """Invariant (a): an explicit watch survives auto-adopt sweeps. The live box
    showed `booting` evidence — the exact trigger for adoption."""
    fleet.hooks.box(46240842, label="upstream-monorepo")     # boot evidence
    fleet.watch("46240842", "jobs", budget_usd=5.0, policy={"id": 46240842})
    observe(fleet, daemon.UNWATCHED_GRACE_S * 2)
    assert fleet.state["strays"] == {}
    assert "unwatched_adopted" not in events(fleet)
    w = fleet.state["watches"]["46240842"]
    assert (w["profile"], w["budget_usd"]) == ("jobs", 5.0)


def test_an_explicit_watch_survives_a_restart_then_a_stray_sweep(tmp_path):
    """Invariant (b): profile+budget are durable across a daemon restart, and
    the post-restart sweep must not re-adopt the box as `bare`."""
    h = FakeHooks()
    d = str(tmp_path / "st")
    h.box(46240842, label="upstream-monorepo")
    f1 = daemon.Fleet(d, hooks=h)
    f1.watch("46240842", "jobs", budget_usd=5.0, policy={"id": 46240842},
             requester="operator@workstation")
    h.jobs_result = "queue_empty"
    f1.tick()
    f2 = daemon.Fleet(d, hooks=h)                       # daemon restart
    w = f2.state["watches"]["46240842"]
    assert (w["profile"], w["budget_usd"], w["adopted"]) == ("jobs", 5.0, False)
    h.jobs_result = None
    h.advance(45)
    f2.tick()
    assert h.jobs_ticks[-1] == "46240842"               # ladder rebuilt, jobs lane
    assert "unwatched_adopted" not in events(f2)
    w = f2.state["watches"]["46240842"]
    assert (w["profile"], w["budget_usd"]) == ("jobs", 5.0)
    assert f2.runtime["46240842"]["a"].budget == 5.0


# --------------------------------------------------------------------------- #
# the alarm channel — DERIVED vs LATCHED
#
# Live defect, 2026-07-31, box 46347213: the safety net auto-adopted a freshly
# launched box and raised "AUTO-ADOPTED ... register a real `fleet watch`". The
# operator did exactly that (`fleet watch --profile jobs --budget 5`, accepted:
# the row read adopted=false budget=$5) and the alarm string stayed lit under
# the table anyway. Alarms were a per-tick list every producer appended to, so
# a resolved condition burned until the next reconcile — unbounded across an
# API outage, which returned before the rebuild while still bumping the tick
# clock, so `tick_age_s` read fresh over a frozen list.
#
# An alarm still lit after you fixed it is worse than no alarm: it trains the
# operator to skip the block that also carries budget breaches, evictions and
# the stray fuse. These tests pin the general property (a resolved condition is
# dark on the NEXT READ, tick or no tick), not the one string.
# --------------------------------------------------------------------------- #
def _alarm(f, needle):
    return [a for a in f.alarms if needle in a]


def test_no_producer_may_append_to_an_alarm_list(fleet):
    """The defect in one line of source. Alarms are DERIVED from state (they
    retract themselves) or explicitly LATCHED (`latch_alarm`, cleared by the
    owner or an ack) — a bare append to a list nobody prunes is neither.

    Class C2 (refactor step 6): this used to read `fleetd.__file__` alone,
    which goes VACUOUSLY GREEN the moment the launcher is thinned. Step 6d
    thinned it, so the flat entry is gone and the scan is the vastlib.fleet
    modules that hold the moved alarm bodies. The needle assert is the
    non-triviality self-check and it is what makes this repoint safe: it was
    verified to fail LOUD against the thinned `fleetd.py` (`no longer contains
    'def _derive_alarms'`) rather than satisfying the absence assert by having
    no source at all. Any future move of a producer out of these three files
    trips the same needle instead of going quiet.
    """
    scanned = {
        daemon.__file__: "def _derive_alarms",
        fleet_state.__file__: "def load_state",
        fleet_rows.__file__: "def reconcile_rows",
    }
    for path, needle in scanned.items():
        with open(path) as fh:
            src = fh.read()
        assert needle in src, \
            f"{path} no longer contains {needle!r} — re-point this test"
        assert "alarms.append" not in src, \
            "append-only alarms latch forever; use _derive_alarms or latch_alarm"


def test_watching_an_auto_adopted_box_clears_its_alarm_immediately(fleet):
    """The live repro. NOTE: no tick between the fix and the re-read — the
    alarm must be gone from the very next `fleet status`, because it is derived
    from the watch table rather than appended by whichever tick noticed."""
    fleet.hooks.box(46347213, dph=0.642, label="upstream-monorepo")
    fleet.hooks.health_map["46347213"] = {"verdict": "OK",
                                          "evidence": {"boot_age_s": 12}}
    fleet.tick()                                       # safety net adopts it
    assert _alarm(fleet, "AUTO-ADOPTED")
    fleet.watch("46347213", "jobs", budget_usd=5.0,
                policy={"id": 46347213}, requester="operator@workstation")
    assert _alarm(fleet, "AUTO-ADOPTED") == [], fleet.alarms
    fleet.hooks.jobs_result = "queue_empty"
    observe(fleet, 300)                                # and it stays cleared
    assert _alarm(fleet, "AUTO-ADOPTED") == [], fleet.alarms
    assert fleet.state["watches"]["46347213"]["budget_usd"] == 5.0
    ev = events(fleet)
    assert "alarm_raised" in ev and "alarm_resolved" in ev   # both in the record


def test_status_derives_alarms_from_state_not_from_the_last_tick(fleet, tmp_path):
    """`fleetd status` (offline dump, no daemon) and a socket `status` must show
    the SAME alarms: they are a function of the persisted state, so a process
    that never ticked still reports the fleet correctly."""
    fleet.hooks.box(46347213, label="upstream-monorepo")
    fleet.hooks.health_map["46347213"] = {"verdict": "OK",
                                          "evidence": {"boot_age_s": 12}}
    fleet.tick()
    assert _alarm(fleet, "AUTO-ADOPTED")
    cold = daemon.Fleet(fleet.dir, hooks=FakeHooks())   # never ticked
    assert [a for a in cold.alarms if "AUTO-ADOPTED" in a]
    data = fleet.status()
    assert data["alarms"] == fleet.alarms
    assert all(r["sticky"] is False for r in data["alarm_records"])


def test_an_api_outage_cannot_freeze_the_alarm_block(fleet):
    """The freeze mechanism: the outage path returned before the alarm rebuild
    while bumping the tick clock, so a stale block looked freshly measured. Now
    `tick_age_s` tracks the last SUCCESSFUL reconcile, the outage is its own
    alarm, and a fix applied mid-outage still clears its alarm."""
    fleet.hooks.box(46347213, label="upstream-monorepo")
    fleet.hooks.health_map["46347213"] = {"verdict": "OK",
                                          "evidence": {"boot_age_s": 12}}
    fleet.tick()
    age0 = fleet.status()["tick_age_s"]
    fleet.hooks.api_down = True
    fleet.hooks.advance(600)
    fleet.tick()
    st = fleet.status()
    assert st["tick_age_s"] > age0 + 500               # honest staleness
    assert st["api_ok"] is False
    assert _alarm(fleet, "NOT being reconciled")
    fleet.watch("46347213", "jobs", budget_usd=5.0, policy={"id": 46347213})
    assert _alarm(fleet, "AUTO-ADOPTED") == []         # fixed mid-outage: dark
    fleet.hooks.api_down = False
    fleet.hooks.jobs_result = "queue_empty"
    fleet.tick()
    assert fleet.status()["api_ok"] is True
    assert _alarm(fleet, "NOT being reconciled") == []


def test_queue_empty_alarm_clears_when_the_wave_is_submitted(fleet):
    fleet.hooks.box(222)
    fleet.watch("222", "jobs", budget_usd=5.0, policy={"id": 222})
    fleet.hooks.jobs_result = "queue_empty"
    fleet.tick()
    assert _alarm(fleet, "QUEUE IS EMPTY")
    fleet.hooks.jobs_result = None                     # tickets appear
    fleet.hooks.advance(45)
    fleet.tick()
    assert _alarm(fleet, "QUEUE IS EMPTY") == []


def test_serve_profile_runs_the_jobs_ladder_in_serve_mode(fleet):
    """2026-08-02: serve joined the policy tier — a serve watch drives the SAME
    defend/rescue bid ladder as jobs (herdd serve_mode strips the queue
    semantics), replacing the old observe-and-alarm-only serve watch. The old
    'serve box NOT live -> herdd start it by hand' alarm is retired: the
    ladder IS the response to a not-live spot serve box."""
    fleet.hooks.box(555, status="stopped")
    fleet.watch("555", "serve", budget_usd=10.0)
    fleet.tick()
    assert fleet.hooks.jobs_ticks == ["555"]         # ladder ticked, not simple-watch
    assert fleet.runtime["555"]["a"].serve_mode is True
    assert fleet.runtime["555"]["a"].budget == 10.0  # ONE cap: ladder spends against it
    assert _alarm(fleet, "serve box NOT live") == []
    ticks = [j for j in journal(fleet) if j["event"] == "tick"]
    assert ticks and ticks[-1]["profile"] == "serve"


def test_serve_watch_requires_budget(fleet):
    """The serve profile moves money now (bid PUTs), so it inherits the run/jobs
    hard-cap registration rule."""
    fleet.hooks.box(555)
    with pytest.raises(ValueError, match="budget_usd is required"):
        fleet.watch("555", "serve")


def test_legacy_unbudgeted_serve_watch_alarms(fleet):
    """A state.json serve watch written BEFORE serve joined the policy tier has
    no budget cap but now runs the ladder — it must alarm until re-registered,
    not silently spend."""
    fleet.hooks.box(555)
    fleet.watch("555", "serve", budget_usd=10.0)
    fleet.state["watches"]["555"]["budget_usd"] = None   # simulate the old record
    fleet.tick()
    assert _alarm(fleet, "predates the bid ladder")


def test_health_alarm_clears_when_the_verdict_clears(fleet):
    fleet.hooks.box(444, label="run:r9")
    fleet.hooks.health_map["444"] = {"verdict": "ZOMBIE_NO_JOBD",
                                     "reason": "jobd hb stale", "evidence": {}}
    fleet.watch("444", "bare", budget_usd=10.0)
    fleet.tick()
    assert _alarm(fleet, "HEALTH ZOMBIE_NO_JOBD")
    fleet.hooks.health_map["444"] = {"verdict": "OK", "evidence": {}}
    observe(fleet, daemon.HEALTH_EVERY_S)
    assert _alarm(fleet, "HEALTH") == []


def test_a_parked_strays_record_cannot_alarm_forever(fleet):
    """The stray sweep skips a non-live box BEFORE it touches the record, so a
    record outlives the box it describes. Deriving off it unguarded would be
    the same latching bug: the alarm may only speak for a box this reconcile
    actually saw live."""
    fleet.hooks.box(444, dph=5.0)
    observe(fleet, daemon.UNWATCHED_GRACE_EXPENSIVE_S)
    assert fleet.hooks.parked == ["444"]
    assert fleet.state["strays"]["444"]["parked_ts"]    # record survives
    fleet.hooks.advance(45)
    fleet.tick()                                       # box is stopped now
    assert _alarm(fleet, "UNWATCHED") == [], fleet.alarms


def test_a_budget_park_keeps_alarming_on_purpose_until_the_cap_moves(fleet):
    """JUDGED LEGITIMATELY STICKY. A box fleetd parked at its cap is a standing
    condition, not an event: it outranks the S8 dormancy silence and burns
    until a human raises the cap, resumes, or drops the watch. It is still
    DERIVED — nothing is remembered, the state simply still says `capped`."""
    fleet.hooks.box(333, dph=36.0)
    fleet.watch("333", "bare", budget_usd=1.0)
    fleet.tick()
    fleet.hooks.advance(1000)
    fleet.tick()
    assert _alarm(fleet, "PARKED by fleetd")
    assert fleet.state["watches"]["333"]["dormant"] is True     # S8 silence...
    fleet.hooks.advance(1000)
    fleet.tick()
    assert _alarm(fleet, "PARKED by fleetd")                   # ...does not apply
    fleet.watch("333", "bare", budget_usd=50.0)                # operator raises it
    assert _alarm(fleet, "PARKED by fleetd") == []


def test_a_destroy_that_expired_unexecuted_latches_until_acked(fleet):
    """JUDGED LEGITIMATELY STICKY, and the reason latching exists: the request
    is popped by the tick that noticed, so nothing can re-derive it — and a
    destroy an operator asked for and never got must not scroll past once."""
    fleet.hooks.box(444)
    fleet.request_destroy("444", when="drained", yes=True, requester="free@rig")
    fleet.hooks.advance(daemon.DESTROY_TTL_S + 10)
    fleet.tick()
    assert fleet.hooks.destroyed == []
    assert _alarm(fleet, "EXPIRED unexecuted")
    rec = [r for r in fleet.alarm_records() if "EXPIRED" in r["msg"]][0]
    assert rec["sticky"] is True and rec["key"] == "destroy:444:expired"
    observe(fleet, 600)                                # survives ticks
    assert _alarm(fleet, "EXPIRED unexecuted")
    cold = daemon.Fleet(fleet.dir, hooks=fleet.hooks)  # ...and a restart
    assert [a for a in cold.alarms if "EXPIRED" in a]
    cold.ack_alarm("destroy:444:expired", requester="free@rig")
    assert [a for a in cold.alarms if "EXPIRED" in a] == []
    assert "alarm_cleared" in events(cold)


def test_acking_a_derived_alarm_is_refused_with_the_reason(fleet):
    fleet.hooks.box(46347213, label="upstream-monorepo")
    fleet.hooks.health_map["46347213"] = {"verdict": "OK",
                                          "evidence": {"boot_age_s": 12}}
    fleet.tick()
    key = [r["key"] for r in fleet.alarm_records() if "AUTO-ADOPTED" in r["msg"]][0]
    with pytest.raises(ValueError) as e:
        fleet.ack_alarm(key)
    assert "DERIVED" in str(e.value)
    with pytest.raises(KeyError):
        fleet.ack_alarm("nosuch:alarm")


def test_a_failed_park_latches_and_a_later_success_retracts_it(fleet):
    fleet.hooks.box(333)
    fleet.watch("333", "bare", budget_usd=10.0)
    fleet.hooks.park_ok = False
    fleet.request_action("333", "park", requester="free@rig")
    fleet.tick()
    assert _alarm(fleet, "park FAILED") or _alarm(fleet, "park FAILED".upper())
    assert any(r["sticky"] for r in fleet.alarm_records())
    fleet.hooks.park_ok = True
    fleet.request_action("333", "park", requester="free@rig")
    fleet.hooks.advance(45)
    fleet.tick()
    assert [a for a in fleet.alarms if "FAILED" in a] == []


def test_ack_all_clears_every_latched_alarm_and_nothing_else(fleet):
    fleet.hooks.box(444)
    fleet.hooks.box(555, dph=5.0)
    fleet.request_destroy("444", when="drained", yes=True)
    fleet.hooks.advance(daemon.DESTROY_TTL_S + 10)
    fleet.tick()
    assert any(r["sticky"] for r in fleet.alarm_records())
    out = fleet.ack_alarm(all_keys=True)
    assert out["cleared"] == ["destroy:444:expired"]
    recs = fleet.alarm_records()
    assert not any(r["sticky"] for r in recs)
    assert [r for r in recs if "UNWATCHED" in r["msg"]]      # derived: untouched


def test_proto_ack_clears_a_latched_alarm(server):
    """The wire path an operator actually uses to retire a latched alarm."""
    f = server.fleet
    f.hooks.box(444)
    f.request_destroy("444", when="drained", yes=True)
    f.hooks.advance(daemon.DESTROY_TTL_S + 10)
    f.tick()
    ok, data, _ = server.handle({"v": 1, "op": "status"})
    assert ok and any(r["sticky"] for r in data["alarm_records"])
    ok, data, err = server.handle({"v": 1, "op": "ack",
                                   "args": {"key": "destroy:444:expired",
                                            "requester": "free@rig"}})
    assert ok and data["cleared"] == ["destroy:444:expired"], err
    ok, data, _ = server.handle({"v": 1, "op": "status"})
    assert not any(r["sticky"] for r in data["alarm_records"])
    ok, _d, err = server.handle({"v": 1, "op": "ack", "args": {"key": "nope"}})
    assert not ok and "no latched alarm" in err


# --------------------------------------------------------------------------- #
# replacement linkage: the watch KEY is where a watch started, `iid` is the box
# it is on NOW (fleet_rows.watch_box_iid).
#
# Live incident 2026-08-05, twice in 20 minutes. A jobs watch whose ladder had
# auto-rented a replacement after a spot eviction was filed under the ORIGINAL
# id with `iid` -> the replacement. Both a daemon restart (46866652->46867184,
# cap $1.6) and an operator re-`fleet watch` (46867184->46867793, cap $4.0)
# rebuilt the ladder from the KEY, which pointed at the already-destroyed box:
# the tick wrote that dead id back over `iid`, the watch died `instance_gone`
# ~45s later, and the stray sweep re-adopted the LIVE replacement as `bare` with
# NO budget cap. Journal, second occurrence: 09:06:48 watch_registered budget=4
# -> 09:06:51 watch_auto_adopted profile=bare -> 09:07:36 watch_finished
# instance_gone. An explicit cap must not be losable this way.
# --------------------------------------------------------------------------- #
def _replaced(fleet, key, new_iid, spend=0.0):
    """Put an existing watch in the post-replacement shape the jobs tick writes:
    still filed under `key`, running on `new_iid`, original box destroyed."""
    w = fleet.state["watches"][key]
    w["iid"] = str(new_iid)
    w["spend_usd"] = spend
    fleet.hooks.boxes.pop(str(key), None)
    fleet.hooks.box(int(new_iid), label="upstream-monorepo")
    return w


def test_watch_by_the_replacement_iid_redirects_to_the_owning_watch(fleet):
    """`fleet status`/`ls` show the REPLACEMENT id, so that is the id an
    operator types to cap the box. It must land on the owning watch, not be
    refused with advice to unwatch a live job."""
    fleet.hooks.box(222, label="upstream-monorepo")
    fleet.watch("222", "jobs", budget_usd=1.0, policy={"id": 222, "budget": 1.0})
    created = fleet.state["watches"]["222"]["created_ts"]
    _replaced(fleet, "222", 999, spend=0.084)
    fleet.hooks.advance(300)
    got = fleet.watch("999", "jobs", budget_usd=4.0,
                      policy={"budget": 4.0}, requester="free@rig")
    assert list(fleet.state["watches"]) == ["222"]      # one watch, not two
    w = fleet.state["watches"]["222"]
    assert w["budget_usd"] == 4.0                       # the operator's intent
    assert w["spend_usd"] == 0.084                      # accounting carried over
    assert w["created_ts"] == created                   # same watch, not a new one
    assert w["iid"] == "999"                            # linkage intact
    assert got["redirected_from"] == "999"
    red = [r for r in journal(fleet) if r["event"] == "watch_redirected"]
    assert red and (red[-1]["target"], red[-1]["requested"]) == ("222", "999")


def test_a_redirect_merges_the_policy_instead_of_blanking_it(fleet):
    """A same-key upsert REPLACES `policy`; a redirect MERGES it. Raising a cap
    must not blank the ladder flags the watch was launched with — but an
    explicitly false flag (`--no-salvage`) must still win."""
    fleet.hooks.box(222, label="upstream-monorepo")
    fleet.watch("222", "jobs", budget_usd=1.0,
                policy={"id": 222, "budget": 1.0, "max_bid": 2.5,
                        "max_replacements": 1, "salvage": True, "keep": True})
    _replaced(fleet, "222", 999)
    fleet.watch("999", "jobs", budget_usd=4.0,
                policy={"budget": 4.0, "max_bid": None, "max_replacements": None,
                        "salvage": False, "keep": None})
    pol = fleet.state["watches"]["222"]["policy"]
    assert pol["max_bid"] == 2.5                    # None never blanks
    assert pol["max_replacements"] == 1
    assert pol["keep"] is True
    assert pol["salvage"] is False                  # an explicit False does win
    assert pol["budget"] == 4.0
    assert pol["id"] == 222                         # untouched keys survive


def test_a_genuine_collision_still_refuses_without_advising_unwatch(fleet):
    """Two distinct watches with no replacement relationship (here a run watch
    whose LABEL currently resolves to the target) stay refused — but never with
    "unwatch it first", which drops supervision of a live box."""
    fleet.hooks.box(111, label="run:r1")
    fleet.watch("run:r1", "run", budget_usd=5.0)
    fleet.tick()
    assert fleet.state["watches"]["run:r1"]["iid"] == "111"
    with pytest.raises(ValueError) as e:
        fleet.watch("111", "bare")
    msg = str(e.value)
    assert "run:r1" in msg                       # names the watch to address
    assert "unwatch it first" not in msg
    assert "DROPS supervision of a LIVE box" in msg   # warns about unwatch
    assert "111" not in fleet.state["watches"]


def test_a_redirect_never_downgrades_a_policy_watch_to_bare(fleet):
    """`fleet watch <replacement>` with the DEFAULT profile must not silently
    disarm the outbid rescue / replacement ladder holding a spot box."""
    fleet.hooks.box(222, label="upstream-monorepo")
    fleet.watch("222", "jobs", budget_usd=1.0, policy={"id": 222})
    _replaced(fleet, "222", 999)
    with pytest.raises(ValueError) as e:
        fleet.watch("999", "bare", budget_usd=4.0)
    assert "--profile jobs" in str(e.value)
    assert fleet.state["watches"]["222"]["profile"] == "jobs"


def test_rewatching_the_owning_key_keeps_the_watch_on_its_replacement(fleet):
    """The 09:06:48 half of the incident: re-registering the OWNING key wrote
    the key (a destroyed id) over `iid`, and the watch was gone 48s later."""
    fleet.hooks.box(46867184, label="upstream-monorepo")
    fleet.watch("46867184", "jobs", budget_usd=1.2, policy={"id": 46867184})
    fleet.tick()
    _replaced(fleet, "46867184", 46867793, spend=0.0835)
    fleet.watch("46867184", "jobs", budget_usd=4.0, policy={"id": 46867184})
    assert fleet.state["watches"]["46867184"]["iid"] == "46867793"
    observe(fleet, 4 * daemon.GONE_CONFIRM_TICKS * 45)
    w = fleet.state["watches"].get("46867184")
    assert w is not None and w["budget_usd"] == 4.0     # the cap survives
    assert fleet.hooks.jobs_ticks[-1] == "46867793"     # ladder on the LIVE box
    assert "unwatched_adopted" not in events(fleet)     # no uncapped re-adoption


def test_a_daemon_restart_keeps_a_replaced_watch_on_its_replacement(tmp_path):
    """The 08:55:18 half: the restart reloaded the right `iid` from state.json,
    then `_init_runtime` re-derived the ladder's box from the KEY and threw it
    away. `policy["id"]` is deliberately the stale original id here — that is
    what a real delegated watch carries."""
    h = FakeHooks()
    d = str(tmp_path / "st")
    h.box(46866652, label="upstream-monorepo")
    f1 = daemon.Fleet(d, hooks=h)
    f1.watch("46866652", "jobs", budget_usd=1.6, policy={"id": 46866652})
    f1.tick()
    w = f1.state["watches"]["46866652"]
    w["iid"], w["spend_usd"] = "46867184", 0.2896        # ladder replaced the box
    h.boxes.pop("46866652")
    h.box(46867184, label="upstream-monorepo")
    f1.save()

    f2 = daemon.Fleet(d, hooks=h)                        # daemon restart
    h.advance(45)
    f2.tick()
    assert f2.runtime["46866652"]["jc"]["iid"] == "46867184"
    assert h.jobs_ticks[-1] == "46867184"
    observe(f2, 4 * daemon.GONE_CONFIRM_TICKS * 45)
    w = f2.state["watches"].get("46866652")
    assert w is not None and w["budget_usd"] == 1.6
    assert "unwatched_adopted" not in events(f2)


def test_the_safety_net_cannot_adopt_a_box_an_explicit_cap_already_owns(fleet):
    """With the linkage preserved, `_explicit_owner` sees through it: the live
    replacement is never re-adopted as unbudgeted `bare`."""
    fleet.hooks.box(222, label="upstream-monorepo")
    fleet.watch("222", "jobs", budget_usd=4.0, policy={"id": 222})
    _replaced(fleet, "222", 999, spend=0.084)
    observe(fleet, daemon.UNWATCHED_GRACE_S * 2)
    assert "999" not in fleet.state["watches"]
    assert "unwatched_adopted" not in events(fleet)
    assert fleet.state["watches"]["222"]["budget_usd"] == 4.0


def test_make_policy_puts_the_ladder_on_the_current_box_not_the_key():
    a = daemon.make_policy("jobs", {"id": 46866652}, "46866652",
                           budget_usd=1.6, iid="46867184")
    assert a.id == 46867184
    b = daemon.make_policy("jobs", {"id": 46866652}, "46866652", budget_usd=1.6)
    assert b.id == 46866652                              # unchanged without iid


def test_watch_box_iid_ignores_run_watches_and_json_garbage():
    assert fleet_rows.watch_box_iid({"profile": "jobs", "iid": "999"}) == "999"
    assert fleet_rows.watch_box_iid({"profile": "run", "iid": "999"}) is None
    assert fleet_rows.watch_box_iid({"profile": "jobs", "iid": "None"}) is None
    assert fleet_rows.watch_box_iid({"profile": "jobs"}) is None
    assert fleet_rows.watch_box_iid(None) is None


# --------------------------------------------------------------------------- #
# defect #63 (P0-b): a handoff is a MONEY decision — it belongs in `fleet log`
#
# Every jobs-lane handoff event went only to B2 (`_job_handoff_emit` ->
# jobmeta.emit_box_event, keyed on the box), and the daemon journalled
# `jobs_replaced` on the EVICTION ladder only. So on 2026-08-08 an autonomous
# box rental AND the destruction of a healthy primary both happened with NOTHING
# in `fleet log` — which is why the incident first read as a spot eviction.
#
# Deliberately NOT hung on the handoff `complete` transition: under filed defect
# #61 `complete` is unreachable under fleetd (a non-`run` watch ends at
# `inst is None` before the ladder can tick again), so armed / fenced / cutover
# each have to reach the journal on their own.
# --------------------------------------------------------------------------- #
def _jobs_watch(fleet, iid=222):
    fleet.hooks.box(iid, label="upstream-monorepo")
    fleet.watch(str(iid), "jobs", budget_usd=5.0,
                policy={"budget": 5.0, "id": iid})


def test_a_jobs_handoff_reaches_the_journal_at_every_phase(fleet):
    _jobs_watch(fleet)
    for kind, fields in (("armed", {"epoch": 1, "candidate_min_bid": 0.3333,
                                    "n_jobs": 1}),
                         ("fenced", {"primary": "222", "understudy": "999"}),
                         ("cutover", {"understudy": "999", "reason": "flushed"})):
        fleet.hooks.jobs_handoff = [(kind, fields)]
        fleet.tick()
    got = [r for r in journal(fleet) if r["event"].startswith("jobs_handoff_")]
    assert [r["event"] for r in got] == ["jobs_handoff_armed",
                                         "jobs_handoff_fenced",
                                         "jobs_handoff_cutover"]
    assert got[0]["epoch"] == 1 and got[0]["candidate_min_bid"] == 0.3333
    assert got[2]["understudy"] == "999"
    assert all(r["iid"] == "222" and r["target"] == "222" for r in got)


def test_a_handoff_abort_is_journalled_too(fleet):
    _jobs_watch(fleet)
    fleet.hooks.jobs_handoff = [("abort", {"reason": "deadline",
                                           "instance_id": "999"})]
    fleet.tick()
    ab = [r for r in journal(fleet) if r["event"] == "jobs_handoff_abort"]
    assert ab and ab[-1]["reason"] == "deadline"


def test_the_journal_queue_is_drained_not_replayed(fleet):
    """A ladder decision is journalled ONCE. Re-emitting it every poll would bury
    `fleet log` under the same line for as long as the phase stands."""
    _jobs_watch(fleet)
    fleet.hooks.jobs_handoff = [("armed", {"epoch": 1})]
    fleet.tick()
    fleet.tick()
    fleet.tick()
    assert len([r for r in journal(fleet) if r["event"] == "jobs_handoff_armed"]) == 1


def test_a_deferred_handoff_latches_an_alarm_with_its_arithmetic(fleet):
    """The refusal is the loud half: `fleet status` must show WHY the ladder
    keeps declining to migrate an over-ceiling box, or the operator sees an
    expensive box and no reasoning."""
    _jobs_watch(fleet)
    note = ("!! HANDOFF DEFERRED on 222: 1 running job(s), ~2700s of horizon "
            "left — migrating would cost ~$0.61 to capture $0.43/hr; "
            "re-testing each poll")
    fleet.hooks.jobs_handoff = [("deferred", {"note": note, "horizon_s": 2700})]
    fleet.tick()
    dj = [r for r in journal(fleet) if r["event"] == "jobs_handoff_deferred"]
    assert dj and dj[-1]["horizon_s"] == 2700
    assert "alarm_latched" in events(fleet)
    rec = [r for r in fleet.alarm_records() if r["key"] == "handoff:222"]
    assert rec and "HANDOFF DEFERRED" in rec[0]["msg"] and rec[0]["sticky"]


def test_a_work_refusal_latches_its_own_alarm(fleet):
    """The 2026-08-08 work rails refuse for reasons that are not economic, and
    those have to be as visible as the economic ones: an operator looking at an
    over-ceiling box needs to see "not migrating, the running job has no
    checkpoint", not silence."""
    _jobs_watch(fleet)
    note = ("!! HANDOFF REFUSED on 222: a RUNNING job has NO checkpoint to "
            "resume from — migrating would discard the attempt, not move it "
            "(defect #62); re-testing each poll")
    fleet.hooks.jobs_handoff = [("refused", {"note": note,
                                             "reason": "unresumable_running_job"})]
    fleet.tick()
    rj = [r for r in journal(fleet) if r["event"] == "jobs_handoff_refused"]
    assert rj and rj[-1]["reason"] == "unresumable_running_job"
    rec = [r for r in fleet.alarm_records() if r["key"] == "handoff:222"]
    assert rec and "HANDOFF REFUSED" in rec[0]["msg"] and rec[0]["sticky"]


def test_a_work_warning_is_journalled_without_latching_an_alarm(fleet):
    """HANDOFF_WARN_PCT is advisory. It belongs in the log; it is not a standing
    condition an operator has to clear."""
    _jobs_watch(fleet)
    fleet.hooks.jobs_handoff = [("work_warning", {"job_id": "job-a", "pct": 92,
                                                  "n_checkpoints": 0,
                                                  "note": "job-a is 92% done"})]
    fleet.tick()
    assert [r for r in journal(fleet) if r["event"] == "jobs_handoff_work_warning"]
    assert not [r for r in fleet.alarm_records() if r["key"] == "handoff:222"]


# --- defect #61: the jobs handoff must be able to COMPLETE under fleetd -------
def test_jobs_handoff_is_safe_off_by_default_under_fleetd(monkeypatch):
    """SAFE-OFF (2026-08-08). The ladder cannot arm a migration on fleetd's jobs
    or serve profiles unless the named unsafe switch says so — and a watch stored
    with the OLD default (handoff: True, which is what every state.json written
    before today carries) cannot opt itself back in across a daemon restart."""
    monkeypatch.delenv(vastconf.JOBS_HANDOFF_UNSAFE_ENV, raising=False)
    # BOTH spellings: `vastconf` is a re-export shim over `vastlib.core.config`
    # (plan step 7) and the daemon reads `config.jobs_handoff_enabled`, which
    # resolves `load_herdd_config` in the PORT's globals. Patching only the
    # shim's binding leaves this box's real herdd.yaml being read — green,
    # today, only because that file happens to say `jobs_handoff_unsafe_enable:
    # false`, which is the very default under test.
    monkeypatch.setattr(vastconf, "load_herdd_config", lambda: {})
    monkeypatch.setattr(vastlib_config, "load_herdd_config", lambda: {})
    for profile in ("jobs", "serve"):
        a = daemon.make_policy(profile, {"id": 7, "budget": 5.0, "handoff": True}, "7")
        assert a.handoff is False, profile
    # the switch is what re-enables it, and it is named for what it is
    monkeypatch.setenv(vastconf.JOBS_HANDOFF_UNSAFE_ENV, "1")
    a = daemon.make_policy("jobs", {"id": 7, "budget": 5.0, "handoff": True}, "7")
    assert a.handoff is True
    # ...and the run lane is untouched by any of it
    monkeypatch.delenv(vastconf.JOBS_HANDOFF_UNSAFE_ENV, raising=False)
    assert daemon.make_policy("run", {"budget": 5.0}, "run:r1").handoff is True


def test_fleetd_declares_it_can_complete_a_handoff(monkeypatch):
    """The pure core refuses to arm for a driver that has not asserted it can
    finish (defect #61). fleetd asserts it because `_tick_watch` now keeps
    ticking across the primary's destroy — the two must move together, so this
    pins the assertion to the mechanism below rather than leaving it a comment."""
    monkeypatch.setenv(vastconf.JOBS_HANDOFF_UNSAFE_ENV, "1")
    a = daemon.make_policy("jobs", {"id": 7, "budget": 5.0}, "7")
    assert a.handoff_can_complete is True
    assert a.handoff_unsafe_ignore_preconditions is False


def test_a_gone_primary_ends_the_watch_when_no_handoff_is_in_flight(fleet):
    """The pre-existing contract, unchanged: an IID watch dies with its box."""
    _jobs_watch(fleet)
    fleet.tick()
    fleet.hooks.boxes.pop("222")
    for _ in range(daemon.GONE_CONFIRM_TICKS):
        fleet.tick()
    assert "222" not in fleet.state["watches"]
    assert [r for r in journal(fleet) if r["event"] == "watch_finished"]


def test_a_completing_handoff_keeps_its_watch_and_hands_over_the_budget(fleet):
    """Defect #61, at the level it actually failed.

    `handoff_poll` returns `complete` on the tick AFTER `drain_primary` destroys
    the primary. `_tick_watch` used to return at `inst is None` for every
    non-`run` profile, so that tick never happened: the watch ended
    `instance_gone` at GONE_CONFIRM_TICKS and the understudy — a live, rented,
    $2.80/hr box — was left with no watch and no budget cap for the stray sweep
    to adopt as an uncapped `bare` box. Deterministic, not a race.

    Driven through REAL `_tick_watch` passes with the primary genuinely absent
    from the listing, because that absence IS the defect; injecting a promoted
    `w["iid"]` would test the one step the real path could never reach."""
    _jobs_watch(fleet)
    fleet.hooks.box(999, label="job:222:handoff")     # the live understudy
    fleet.hooks.jobs_spend = 3.25                     # spend so far, mid-migration
    fleet.tick()
    rt = fleet.runtime["222"]
    rt["hf"].update(phase="DRAINING", understudy_iid="999", primary_iid="222")

    # the drain destroyed the primary; from here the daemon can only see the
    # understudy — the exact tick that used to kill the watch.
    fleet.hooks.boxes.pop("222")
    ticks_before = len(fleet.hooks.jobs_ticks)
    for _ in range(daemon.GONE_CONFIRM_TICKS + 1):
        fleet.tick()
    assert len(fleet.hooks.jobs_ticks) > ticks_before, "the ladder never ticked"
    assert "222" in fleet.state["watches"], "the watch died before `complete`"
    assert [r for r in journal(fleet) if r["event"] == "jobs_handoff_carryover"]

    # the ladder completes: it promotes the understudy into jc["iid"], and the
    # watch has to follow it WITH the budget cap and the spend to date.
    def _promote(jc, hf):
        jc["iid"] = "999"
        hf.update(phase="IDLE", understudy_iid=None)
        jc.setdefault("handoff_journal", []).append(
            ("complete", {"understudy": "999", "note": "migrated"}))
        return None
    fleet.hooks.jobs_tick = _promote
    fleet.tick()
    w = fleet.state["watches"]["222"]
    assert w["iid"] == "999"
    assert w["budget_usd"] == 5.0 and w["spend_usd"] >= 3.25
    assert [r for r in journal(fleet) if r["event"] == "jobs_handoff_complete"]


def test_the_carryover_does_not_keep_a_watch_alive_on_a_dead_understudy(fleet):
    """Bounded: the carryover keys on a LIVE understudy in THIS tick's listing.
    An understudy that died mid-migration must not hold a watch open forever —
    that is the failure mode `instance_gone` exists for."""
    _jobs_watch(fleet)
    fleet.hooks.box(999, label="job:222:handoff")
    fleet.tick()
    fleet.runtime["222"]["hf"].update(phase="DRAINING", understudy_iid="999")
    fleet.hooks.boxes.pop("222")
    fleet.hooks.boxes.pop("999")                       # both gone
    for _ in range(daemon.GONE_CONFIRM_TICKS):
        fleet.tick()
    assert "222" not in fleet.state["watches"]


def test_arming_retracts_the_deferral_alarm(fleet):
    """The deferral says "re-testing each poll" — when a later poll actually
    arms, `fleet status` must stop claiming the ladder is holding station."""
    _jobs_watch(fleet)
    fleet.hooks.jobs_handoff = [("deferred", {"note": "!! HANDOFF DEFERRED on 222"})]
    fleet.tick()
    assert [r for r in fleet.alarm_records() if r["key"] == "handoff:222"]
    fleet.hooks.jobs_handoff = [("armed", {"epoch": 1})]
    fleet.tick()
    assert not [r for r in fleet.alarm_records() if r["key"] == "handoff:222"]
    assert "alarm_cleared" in events(fleet)


# --------------------------------------------------------------------------- #
# RECALIBRATION 2026-08-09, items C and D
# --------------------------------------------------------------------------- #
def test_recoveries_in_flight_names_every_durable_recovery_state():
    """Item C's predicate, over a hand-built state.json — pure, so it works with
    a wedged daemon, which is exactly when `fleet restart` gets typed.

    Every kind is read from a field the ladder already persists
    (REPLACEMENT_STATE_KEYS + `unrecoverable_since` + `destroys`); none of them
    is a new piece of state, which is the point — the file already knew on
    2026-08-08 23:24Z."""
    state = {"watches": {
        "100": {"iid": "100", "replacement": {"rebid_rungs": 2}},
        "200": {"iid": "201", "replacement": {"resume_tries": 1}},
        "300": {"iid": "300", "replacement": {"retained_boxes": [
            {"iid": "299", "status": "retained", "replacement_iid": "300"},
            {"iid": "298", "status": "destroyed"}]}},
        "400": {"iid": "400", "unrecoverable_since": 1.0},
        "500": {"iid": "500", "replacement": {"rebid_rungs": 0,
                                              "resume_tries": 0}},
    }, "destroys": {"600": {"when": "drained"}}}
    got = fleet_rows.recoveries_in_flight(state)
    assert [(r["iid"], r["kind"]) for r in got] == [
        ("100", "rebid_ladder"),
        ("201", "resume_in_place"),
        ("299", "replacement"),
        ("400", "unrecoverable"),
        ("600", "destroy_queued")]
    # the watch key travels with the row: a ladder that replaced its box is still
    # FILED under its original id, and an operator has to be able to address it
    assert [r for r in got if r["kind"] == "resume_in_place"][0]["target"] == "200"
    # a settled watch contributes nothing, and neither does an empty state
    assert fleet_rows.recoveries_in_flight({}) == []
    assert fleet_rows.recoveries_in_flight({"watches": {"500": {"iid": "500"}}}) == []


def test_fleet_restart_refuses_mid_recovery_and_force_overrides(tmp_path,
                                                                monkeypatch,
                                                                capsys):
    """The guard at the CLI. 2026-08-08 23:24:37Z: a redeploy landed two minutes
    after a human destroyed 47214941, mid-chain; the restarted daemon reconciled
    the stale watch and ran its OWN condemn -> launch -> retarget -> destroy.
    ~$0.9 of duplicated recovery, two actors on one job.

    Exit 2 and NO systemctl call is the whole contract — a refusal that still
    restarted would be worse than no refusal."""
    monkeypatch.setenv("FLEETD_STATE_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(subprocess, "call",
                        lambda *a, **k: (calls.append(a), 0)[1])
    with open(os.path.join(str(tmp_path), "state.json"), "w") as f:
        json.dump({"watches": {"47214941": {"iid": "47214941",
                                            "replacement": {"rebid_rungs": 3}}}}, f)
    with pytest.raises(SystemExit) as e:
        cli_fleet_restart.run(argparse.Namespace(force=False))
    assert e.value.code == 2
    assert calls == [], "REFUSING must not also restart"
    out = capsys.readouterr().out
    assert "REFUSING to restart fleetd" in out
    assert "47214941" in out and "rebid_ladder" in out
    assert "--force" in out
    # --force is the documented escape hatch and does restart
    with pytest.raises(SystemExit) as e2:
        cli_fleet_restart.run(argparse.Namespace(force=True))
    assert e2.value.code == 0
    assert calls and list(calls[0][0])[:3] == ["systemctl", "--user", "restart"]


def test_fleet_restart_is_unblocked_by_a_settled_or_absent_state(tmp_path,
                                                                 monkeypatch):
    """The two must-not-block cases. A quiet fleet obviously restarts; so does a
    MISSING or CORRUPT state file, because refusing there would make a corrupt
    state unrecoverable-by-restart, which is the one job a restart has."""
    monkeypatch.setenv("FLEETD_STATE_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(subprocess, "call",
                        lambda *a, **k: (calls.append(a), 0)[1])
    path = os.path.join(str(tmp_path), "state.json")
    settled = json.dumps({"watches": {"1": {"iid": "1",
                                            "replacement": {"rebid_rungs": 0}}}})
    for body in (None, "{not json", json.dumps({"watches": {}}), settled):
        if body is None:
            if os.path.exists(path):
                os.remove(path)
        else:
            with open(path, "w") as f:
                f.write(body)
        assert client.fleet_recoveries_in_flight() == [], body
        with pytest.raises(SystemExit) as e:
            cli_fleet_restart.run(argparse.Namespace(force=False))
        assert e.value.code == 0
    assert len(calls) == 4


def test_a_destroyed_box_stops_rendering_as_an_unwatched_row(fleet):
    """Item D. `_tick_strays` only visits instances the API still lists, so it
    can CREATE a stray record but never prunes one for a box that has left the
    listing. `fleet status` rendered those forever as `UNWATCHED  -  $0.000`,
    padding the table with boxes that stopped existing hours ago — which is how
    a real stray gets lost in the noise."""
    fleet.hooks.box(444, dph=0.5)
    fleet.tick()
    assert fleet.state["strays"]["444"]["live_ts"]
    assert [r for r in fleet.status()["rows"] if r["iid"] == "444"]
    # the box is destroyed out of band: gone from the listing, record left behind
    fleet.hooks.boxes.pop("444")
    fleet.tick()
    assert [r for r in fleet.status()["rows"] if r["iid"] == "444"], \
        "a box that vanished one tick ago is still worth showing"
    fleet.hooks.advance(fleet_rows.UNWATCHED_STALE_S + 60)
    fleet.tick()
    assert [r for r in fleet.status()["rows"] if r["iid"] == "444"] == []
    assert "444" not in fleet.state["strays"], "the record itself is pruned too"
    assert "stray_record_pruned" in events(fleet)


def test_a_box_queued_for_destruction_is_never_offered_as_a_stray():
    """Destroyed-INTENT boxes never appear: the operator has already decided that
    box's fate, and rendering it as an unwatched stray invites a second decision
    on the same box."""
    now = 1_000.0
    state = {"strays": {"777": {"live_ts": now, "observed_s": 10.0},
                        "888": {"live_ts": now, "observed_s": 10.0}},
             "destroys": {"888": {"when": "now"}}}
    assert [r["iid"] for r in fleet_rows.stray_rows(state, now)] == ["777"]


def test_a_stray_record_with_no_live_sighting_is_history_not_a_row():
    """A record written by a daemon that predates `live_ts` cannot have been seen
    live by anything that stamps it, so it fails toward stale — the conservative
    direction for a display filter whose failure mode is a phantom row."""
    now = 1_000.0
    assert fleet_rows.stray_rows({"strays": {"1": {"observed_s": 5.0}}}, now) == []
    assert fleet_rows.stray_rows(
        {"strays": {"1": {"live_ts": now - fleet_rows.UNWATCHED_STALE_S - 1}}},
        now) == []
    rows = fleet_rows.stray_rows({"strays": {"1": {"live_ts": now - 30}}}, now)
    assert [r["iid"] for r in rows] == ["1"]
    assert rows[0]["last_seen_s"] == 30.0


# --- item E: spend reconciliation ------------------------------------------ #
def test_reconcile_surfaces_the_pre_watch_window_and_the_unwatched_box():
    """Item E, on the two shapes that made the 2026-08-08 accounting see ~$4.09
    of a ~$5.66 invoice — and neither is an arithmetic error.

      * a box adopted LATE: fleetd accrues from watch adoption, the box bills
        from `start_date`, so the boot/loading head is invisible. Here the box is
        2 h old at $1/hr, watched from the 1 h mark, and accrued $1.00.
      * a box NOBODY watched: it accrues nothing at all and its whole bill is the
        divergence. That is the single largest line the night missed.

    The estimate is `dph_total x age` and it is an UPPER BOUND — vast bills no
    GPU while `loading` and the API exposes no loading->running timestamp — so
    the row says how much of the box's billed life fleetd never watched, never
    how much is owed."""
    now = 10_000.0
    state = {"spend_by_box": {"111": 1.00},
             "watches": {"111": {"iid": "111", "created_ts": now - 3600}}}
    instances = [{"id": 111, "dph_total": 1.00, "start_date": now - 7200},
                 {"id": 222, "dph_total": 2.00, "start_date": now - 1800}]
    rows = {r["iid"]: r for r in fleet_rows.reconcile_rows(state, instances, now)}
    late = rows["111"]
    assert late["watched"] is True
    assert late["accrued_usd"] == 1.00
    assert late["upper_bound_usd"] == pytest.approx(2.00)
    assert late["divergence_usd"] == pytest.approx(1.00)
    assert late["divergence_pct"] == pytest.approx(50.0)
    assert late["unwatched_head_s"] == pytest.approx(3600.0)
    never = rows["222"]
    assert never["watched"] is False
    assert never["accrued_usd"] == 0.0
    assert never["upper_bound_usd"] == pytest.approx(1.00)   # 2.00 x 0.5 h
    assert never["divergence_usd"] == pytest.approx(1.00)
    assert never["unwatched_head_s"] is None                 # never watched


def test_reconcile_covers_boxes_that_have_left_the_listing():
    """A destroyed box keeps its accrued figure and has no live anchor to
    estimate against — `?` rather than a fabricated bound, and `present: False`
    so the renderer can say the accrual is final."""
    now = 10_000.0
    rows = fleet_rows.reconcile_rows({"spend_by_box": {"999": 3.5}}, [], now)
    assert len(rows) == 1
    assert rows[0]["present"] is False
    assert rows[0]["accrued_usd"] == 3.5
    assert rows[0]["upper_bound_usd"] is None
    assert rows[0]["divergence_usd"] is None


def test_reconcile_is_opt_in_and_degrades_on_an_unreadable_api(fleet):
    """`fleet spend` is unchanged without the flag (one file read, no API), and
    an API blip yields NO reconciliation rather than an empty fleet — the N7
    rule that an unreadable API is not evidence of anything."""
    fleet.hooks.box(555, dph=1.0)
    fleet.watch("555", "bare", budget_usd=5.0)
    fleet.tick()
    plain = fleet.spend()
    assert "reconcile" not in plain and plain["by_box"]
    assert fleet.spend(reconcile=True)["reconcile"], "opt-in must actually run"
    assert "UPPER" in fleet.spend(reconcile=True)["reconcile_basis"].upper()
    fleet.hooks.api_down = True
    degraded = fleet.spend(reconcile=True)
    assert degraded["reconcile"] is None
    assert "not an empty fleet" in degraded["reconcile_error"]
    assert degraded["by_box"], "the accrued figures survive the blip"


def test_the_spend_protocol_carries_the_reconcile_flag(server):
    ok, data, err = server.handle({"v": 1, "op": "spend",
                                   "args": {"reconcile": True}})
    assert ok and err is None and "reconcile" in data
    ok, data, err = server.handle({"v": 1, "op": "spend"})
    assert ok and "reconcile" not in data
