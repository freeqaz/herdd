"""The DURABLE CEILING — fleetd's spend cap after it stopped dying with its watch.

Defect filed 2026-08-03: three paths to a LIVE, SPENDING box with no budget
ceiling. All three are reproduced here from the real fleetd journal (fixtures in
testfixtures/fleetd_journal_*.ndjsonl, extracted verbatim from
~/.local/state/vast-fleetd/journal.ndjsonl, 2026-07-29..08-09, host identity
scrubbed), and each fixture's own recorded lines are asserted to CONTAIN the
pre-fix outcome — so these tests fail loudly if anyone ever "cleans up" a fixture
into a scenario that no longer witnesses the bug.

  PATH 1  born uncapped   — the safety net adopts a busy stray `bare`/None.
  PATH 2  handoff successor — the ladder rents `job:<pred>:handoff`, registers no
          watch, and the successor falls into path 1 with the predecessor's cap
          nowhere in sight.
  PATH 3  an ARMED watch LAPSED — `watch_finished` on a budgeted watch (44 of 51
          observed lapses were `drained`) and the sweep re-adopts the still-live
          box uncapped in the SAME reconcile pass. Silent, and later.

Plus the arithmetic that made a re-arm worse than nothing: re-arming at the
ORIGINAL figure granted a whole fresh cap, because the spend counter had been
reset by the pop. Box 46916278 was armed `--budget 10` six times.

No network, no vast API, no real clock — the FakeHooks discipline from
test_fleetd.py, which this module imports rather than re-implements.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vastconf                                                  # noqa: E402
from test_fleetd import FakeHooks, events, journal               # noqa: E402
from vastlib.core import config                                  # noqa: E402
from vastlib.fleet import daemon, rows                           # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testfixtures")


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    monkeypatch.setenv("FLEETD_STATE_DIR", str(tmp_path / "state"))
    for k in ("FLEETD_GLOBAL_BUDGET_USD", "FLEETD_UNWATCHED_GRACE_S",
              "FLEETD_UNWATCHED_GRACE_EXPENSIVE_S", "FLEETD_EXPENSIVE_DPH",
              vastconf.FLEETD_ADOPT_BUDGET_ENV):
        monkeypatch.delenv(k, raising=False)
    return daemon.Fleet(str(tmp_path / "state"), hooks=FakeHooks())


def fixture_rows(name):
    p = os.path.join(FIXTURES, f"fleetd_journal_{name}.ndjsonl")
    with open(p) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def ceiling(f, cid):
    return f.state["ceilings"][str(cid)]


def cap_of(f, target):
    return f.state["watches"][str(target)]["budget_usd"]


def busy(f, iid, **kw):
    """A box the evidence gate will adopt rather than park (a fresh boot)."""
    b = f.hooks.box(iid, **kw)
    f.hooks.health_map[str(iid)] = {"verdict": "OK", "evidence": {"boot_age_s": 30}}
    return b


def settle(f, n=5):
    """Tick until the health cache has refreshed and the accrual clock is
    seeded. `gather_fleet_health` runs every HEALTH_EVERY_S, so a box that only
    looks busy to the HEALTH probe (a fresh boot, no adoptable label) is a
    stray for that long before the sweep can adopt it — which is exactly the
    174s window understudy 47215526 sat in on 2026-08-08 before it was adopted
    on evidence "booted 215s ago"."""
    f._health_ts = None          # due the fold now; these tests freeze the clock
    for _ in range(n):
        f.tick()


# --------------------------------------------------------------------------- #
# the fail-closed core: an unreadable ceiling is the CONSERVATIVE DEFAULT
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rec,why", [
    (None, "missing record"),
    ("not a dict", "not an object"),
    ({}, "no cap_usd at all"),
    ({"cap_usd": None}, "explicit null"),
    ({"cap_usd": "banana"}, "garbage string"),
    ({"cap_usd": ""}, "empty string"),
    ({"cap_usd": 0}, "zero is not a cap"),
    ({"cap_usd": -5.0}, "negative"),
    ({"cap_usd": float("nan")}, "NaN compares false against every bound"),
    ({"cap_usd": float("inf")}, "infinity IS 'unlimited' spelled differently"),
])
def test_an_unreadable_ceiling_means_the_default_never_unlimited(rec, why):
    cap, spend, degraded = rows.normalize_ceiling(rec, 7.5)
    assert cap == 7.5, why
    assert degraded, "a substituted cap must say so, or the substitution is silent"
    assert spend == 0.0


def test_a_readable_ceiling_is_left_alone():
    cap, spend, degraded = rows.normalize_ceiling(
        {"cap_usd": 5.0, "spend_usd": 2.0}, 7.5)
    assert (cap, spend, degraded) == (5.0, 2.0, None)


def test_a_negative_or_unreadable_spend_reads_as_zero():
    # The load-bearing half of "fail closed" is the CAP. We genuinely do not
    # know that a box spent anything, so an unreadable spend must not park it.
    assert rows.normalize_ceiling({"cap_usd": 5.0, "spend_usd": "?"}, 7.5)[1] == 0.0
    assert rows.normalize_ceiling({"cap_usd": 5.0, "spend_usd": -3}, 7.5)[1] == 0.0


def test_adopt_default_is_positive_for_every_garbage_input(monkeypatch):
    for bad in ("", "  ", "banana", "0", "-1", "nan", "inf", "None"):
        monkeypatch.setenv(vastconf.FLEETD_ADOPT_BUDGET_ENV, bad)
        v = vastconf.fleetd_adopt_default_budget_usd({})
        assert v == vastconf.ADOPT_DEFAULT_BUDGET_USD, bad
        assert v > 0


def test_adopt_default_precedence_env_over_config(monkeypatch):
    monkeypatch.delenv(vastconf.FLEETD_ADOPT_BUDGET_ENV, raising=False)
    cfg = {vastconf.FLEETD_ADOPT_BUDGET_KEY: "3.25"}
    assert vastconf.fleetd_adopt_default_budget_usd(cfg) == 3.25
    monkeypatch.setenv(vastconf.FLEETD_ADOPT_BUDGET_ENV, "1.5")
    assert vastconf.fleetd_adopt_default_budget_usd(cfg) == 1.5


def test_an_unreadable_config_is_the_default_not_an_exception(monkeypatch):
    """Patch BOTH spellings, deliberately. `vastconf.py` is a re-export shim over
    `vastlib.core.config` as of plan step 7, so `vastconf.fleetd_adopt_...` IS
    the ported function and it resolves `load_herdd_config` in the PORT's
    globals — patching only the shim's binding leaves the real yaml being read,
    and this test would then pass only because this box's herdd.yaml happens
    to carry 10.0, the same number as the fail-closed default. That is a green
    for no reason. (Model: test_vastlib_fleet_rows.py's `_pin_the_adopt_default`.)
    """
    boom = lambda: (_ for _ in ()).throw(OSError("disk on fire"))  # noqa: E731
    monkeypatch.delenv(vastconf.FLEETD_ADOPT_BUDGET_ENV, raising=False)
    monkeypatch.setattr(vastconf, "load_herdd_config", boom)
    monkeypatch.setattr(config, "load_herdd_config", boom)
    assert vastconf.fleetd_adopt_default_budget_usd() == \
        vastconf.ADOPT_DEFAULT_BUDGET_USD


def test_a_ceiling_that_reads_degraded_is_repaired_and_journaled(fleet):
    fleet.state["ceilings"]["999"] = {"cap_usd": "banana", "spend_usd": 1.0}
    rec, cap, spend, degraded = fleet._ceiling_read("999")
    assert cap == vastconf.fleetd_adopt_default_budget_usd() and degraded
    assert rec["cap_usd"] == cap and rec["source"] == "degraded"
    assert "ceiling_degraded" in events(fleet)


def test_the_read_path_never_mutates(fleet):
    """`_derive_alarms` and `status()` are READ paths; a status call that
    repaired the ledger would be the latching bug in another costume."""
    fleet.state["ceilings"]["999"] = {"cap_usd": "banana"}
    before = json.dumps(fleet.state["ceilings"], sort_keys=True)
    fleet._ceiling_peek("999")
    fleet._ceiling_spend({"ceiling_id": "999", "spend_usd": 0.0})
    assert json.dumps(fleet.state["ceilings"], sort_keys=True) == before


def test_a_quarantined_state_leaves_an_empty_ledger_which_means_default(tmp_path):
    """S5 already quarantines a corrupt state.json. Empty ledger + adoption must
    land on the conservative default, never on None."""
    d = tmp_path / "state"
    d.mkdir()
    (d / "state.json").write_text("{ this is not json")
    f = daemon.Fleet(str(d), hooks=FakeHooks())
    assert f.state["ceilings"] == {} and f.state["ceiling_by_box"] == {}
    busy(f, 500, label="jobs:x")
    f.tick()
    assert cap_of(f, 500) == vastconf.fleetd_adopt_default_budget_usd()


def test_a_non_dict_ledger_section_is_replaced_not_trusted(tmp_path):
    d = tmp_path / "state"
    d.mkdir()
    (d / "state.json").write_text(json.dumps(
        {"ceilings": ["not", "a", "map"], "ceiling_by_box": 7}))
    f = daemon.Fleet(str(d), hooks=FakeHooks())
    assert f.state["ceilings"] == {} and f.state["ceiling_by_box"] == {}


# --------------------------------------------------------------------------- #
# PATH 1 — born uncapped by auto-adopt
# --------------------------------------------------------------------------- #
def test_path1_fixture_records_the_uncapped_adoption(fleet):
    """The pre-fix outcome, from the journal itself: 164 adoptions, and not one
    of them carried a budget_usd."""
    rows = fixture_rows("path1_rearm_46916278")
    adopts = [r for r in rows if r["event"] == "watch_auto_adopted"]
    assert adopts, "fixture no longer witnesses an auto-adoption"
    assert all("budget_usd" not in r for r in adopts), \
        "journal() drops None fields, so an ABSENT budget_usd IS None"
    assert any("no budget cap" in (r.get("note") or "") for r in rows)


def test_path1_an_adopted_box_now_carries_the_provisional_default(fleet):
    busy(fleet, 444, dph=1.5, label="serve:eval")
    fleet.tick()
    w = fleet.state["watches"]["444"]
    assert w["adopted"] and w["profile"] == "bare"
    assert w["budget_usd"] == vastconf.fleetd_adopt_default_budget_usd()
    assert w["ceiling_source"] == "default"
    assert fleet.state["ceiling_by_box"]["444"] == "444"
    assert ceiling(fleet, "444")["source"] == "default"


def test_path1_the_provisional_default_is_ENFORCED_not_decorative(fleet, monkeypatch):
    """A cap that never parks anything is a comment. $0.001/s at dph=3.6."""
    monkeypatch.setenv(vastconf.FLEETD_ADOPT_BUDGET_ENV, "0.5")
    busy(fleet, 444, dph=3.6, label="serve:eval")
    fleet.tick()                                   # adopt
    assert cap_of(fleet, 444) == 0.5
    fleet.tick()                                   # seeds the accrual clock
    fleet.hooks.advance(600)                       # $0.60 > $0.50
    fleet.tick()
    assert fleet.hooks.parked == ["444"]
    assert fleet.state["watches"]["444"]["state"] == "budget_parked"
    assert any("BUDGET" in a for a in fleet.alarms)


def test_path1_alarm_names_the_figure_as_nobodys_choice(fleet):
    busy(fleet, 444, label="serve:eval")
    fleet.tick()
    msg = [a for a in fleet.alarms if "AUTO-ADOPTED" in a]
    assert msg and "PROVISIONAL" in msg[0] and "nobody chose" in msg[0]


def test_path1_an_explicit_watch_still_outranks_the_default(fleet):
    busy(fleet, 444, label="serve:eval")
    fleet.tick()
    fleet.watch("444", "jobs", budget_usd=42.0, policy={"id": 444},
                requester="operator")
    w = fleet.state["watches"]["444"]
    assert w["budget_usd"] == 42.0 and not w["adopted"]
    assert w["ceiling_source"] == "explicit"
    assert ceiling(fleet, "444")["source"] == "explicit"


# --------------------------------------------------------------------------- #
# PATH 2 — the handoff successor
# --------------------------------------------------------------------------- #
def test_path2_fixture_records_a_capped_primary_and_an_uncapped_understudy():
    rows = fixture_rows("path2_handoff_47214941")
    armed = [r for r in rows
             if r["event"] == "watch_registered" and r.get("budget_usd")]
    assert armed and armed[0]["budget_usd"] == 5.0, "primary was armed at $5"
    stray = [r for r in rows
             if r["event"] == "unwatched" and r.get("label", "").endswith(":handoff")]
    assert stray, "fixture no longer witnesses the understudy going stray"
    assert stray[0]["label"] == "job:47214941:handoff"
    adopt = [r for r in rows
             if r["event"] == "unwatched_adopted" and r.get("iid") == "47215526"]
    assert adopt and "budget_usd" not in adopt[0], \
        "the understudy of a $5-capped primary was adopted with NO cap"


@pytest.mark.parametrize("label,expect", [
    ("job:47214941:handoff", "47214941"),
    ("job:123:handoff", "123"),
    (" job:123:handoff ", "123"),
    ("job:47214941:handoff:extra", None),
    ("jobs:47214941:handoff", None),
    ("job:abc:handoff", None),
    ("upstream-monorepo", None),
    ("", None),
    (None, None),
])
def test_handoff_predecessor_parsing(label, expect):
    assert rows.handoff_predecessor(label) == expect


def test_path2_understudy_inherits_via_its_label(fleet):
    """The restart-mid-migration fallback: nobody told us about the understudy,
    but its LABEL names the primary, and the primary's ceiling is durable."""
    fleet.hooks.box(47214941)
    fleet.watch("47214941", "jobs", budget_usd=5.0, policy={"id": 47214941},
                requester="operator")
    fleet.hooks.jobs_spend = 2.0
    fleet.tick()
    assert ceiling(fleet, "47214941")["spend_usd"] == pytest.approx(2.0)
    # the ladder rents the understudy; no watch is registered for it
    busy(fleet, 47215526, dph=2.804, label="job:47214941:handoff")
    settle(fleet)
    u = fleet.state["watches"]["47215526"]
    assert u["adopted"] and u["ceiling_source"] == "inherited"
    assert u["budget_usd"] == 5.0, "the PRIMARY's cap, not a provisional default"
    assert u["spend_usd"] == pytest.approx(2.0), "carrying spend-to-date"
    assert u["ceiling_id"] == "47214941"


def test_path2_understudy_is_bound_BEFORE_the_sweep_can_see_it(fleet):
    """The primary fix, ahead of the label fallback: the ladder names its
    understudy while the migration is in flight, so the box already has a
    ceiling by the time it shows up unwatched."""
    fleet.hooks.box(47214941)
    fleet.watch("47214941", "jobs", budget_usd=5.0, policy={"id": 47214941})
    fleet.hooks.jobs_handoff = [("armed", {"note": "migrating"})]
    fleet.hooks.jobs_understudy = "47215526"
    fleet.tick()
    assert fleet.state["ceiling_by_box"].get("47215526") == "47214941"
    assert "ceiling_box_bound" in events(fleet)
    # ...and an understudy with an UNRELATED label still inherits, because the
    # binding did not come from the label.
    busy(fleet, 47215526, dph=2.804, label="upstream-monorepo")
    settle(fleet)
    assert fleet.state["watches"]["47215526"]["budget_usd"] == 5.0


def test_path2_two_boxes_on_one_ceiling_ADD_they_do_not_shadow(fleet):
    """During the overlap the primary and the understudy both bill. A max()
    would let the pair spend up to 2x the cap; deltas make them share it."""
    fleet.hooks.box(47214941, dph=3.6)
    fleet.watch("47214941", "jobs", budget_usd=5.0, policy={"id": 47214941})
    fleet.hooks.jobs_spend = 1.0
    fleet.tick()
    busy(fleet, 47215526, dph=3.6, label="job:47214941:handoff")
    settle(fleet)                                  # adopt, seeds at $1.00
    fleet.hooks.advance(1000)                      # understudy accrues $1.00
    fleet.hooks.jobs_spend = 2.0                   # primary accrues $1.00 more
    fleet.tick()
    assert ceiling(fleet, "47214941")["spend_usd"] == pytest.approx(3.0, rel=1e-3)


def test_path2_a_replacement_box_stays_on_the_same_ceiling(fleet):
    """`jobs_replaced` already carried the cap; this asserts it also carries the
    ceiling identity, so a lapse on the REPLACEMENT inherits too."""
    fleet.hooks.box(47214941)
    fleet.watch("47214941", "jobs", budget_usd=5.0, policy={"id": 47214941})
    fleet.tick()
    fleet.hooks.box(47219058)
    fleet.hooks.jobs_iid = "47219058"              # the ladder swapped the box
    fleet.hooks.jobs_spend = 2.5
    fleet.tick()
    assert fleet.state["ceiling_by_box"]["47219058"] == "47214941"
    fleet.hooks.jobs_result = "drained"
    fleet.tick()
    busy(fleet, 47219058, label="upstream-monorepo")
    settle(fleet)
    w = fleet.state["watches"]["47219058"]
    assert w["budget_usd"] == 5.0 and w["spend_usd"] == pytest.approx(2.5)


# --------------------------------------------------------------------------- #
# PATH 3 — an ARMED watch LAPSED (the worst: silent, and later)
# --------------------------------------------------------------------------- #
def test_path3_fixture_records_the_same_second_lapse_to_uncapped(fleet):
    rows = fixture_rows("path3_lapse_46687567")
    fin = [r for r in rows if r["event"] == "watch_finished"]
    aa = [r for r in rows if r["event"] == "watch_auto_adopted"]
    assert fin and aa, "fixture no longer witnesses the lapse"
    assert fin[0]["verdict"] == "drained", "44 of 51 observed lapses were `drained`"
    assert fin[0]["ts_iso"] == aa[0]["ts_iso"], \
        "the re-adoption landed in the SAME reconcile pass (6ms in the journal)"
    assert "budget_usd" not in aa[0]
    # ...and the spend counter restarted at zero on the very next tick.
    after = [r for r in rows
             if r["event"] == "tick" and r["ts_iso"] > aa[0]["ts_iso"]]
    assert after and after[0]["spend_usd"] == 0.0
    assert fin[0]["spend_usd"] > 0.0, "which discarded real accrued spend"


def test_path3_a_lapse_now_keeps_the_cap_and_the_spend(fleet):
    busy(fleet, 46687567, dph=3.6, label="keep:rep-length")
    fleet.watch("46687567", "jobs", budget_usd=5.0, policy={"id": 46687567},
                requester="operator")
    fleet.hooks.jobs_spend = 2.0
    fleet.tick()
    fleet.hooks.jobs_result = "drained"
    fleet.tick()                                   # LAPSE + same-pass re-adopt
    w = fleet.state["watches"]["46687567"]
    assert w["adopted"] and w["ceiling_source"] == "inherited"
    assert w["budget_usd"] == 5.0
    assert w["spend_usd"] == pytest.approx(2.0)
    assert fleet._ceiling_spend(w) == pytest.approx(2.0)


def test_path3_the_alarm_says_LAPSED_not_merely_adopted(fleet):
    busy(fleet, 46687567, label="keep:x")
    fleet.watch("46687567", "jobs", budget_usd=5.0, policy={"id": 46687567})
    fleet.hooks.jobs_spend = 2.0
    fleet.tick()
    fleet.hooks.jobs_result = "drained"
    fleet.tick()
    msg = [a for a in fleet.alarms if "46687567" in a and "LAPSED" in a]
    assert msg, fleet.alarms
    assert "$3.00 left" in msg[0] and "$2.00 of $5.00" in msg[0]


def test_path3_the_inherited_ceiling_parks_at_the_ORIGINAL_cap(fleet):
    """The whole point: after the lapse the box may spend the REMAINDER, and
    then it parks — not another whole $5."""
    busy(fleet, 46687567, dph=3.6, label="keep:x")
    fleet.watch("46687567", "jobs", budget_usd=5.0, policy={"id": 46687567})
    fleet.hooks.jobs_spend = 4.5
    fleet.tick()
    fleet.hooks.jobs_result = "drained"
    fleet.tick()                                   # LAPSE + same-pass re-adopt
    fleet.tick()                                   # seeds the bare accrual clock
    assert fleet.hooks.parked == []
    fleet.hooks.advance(2000)                      # +$2.00 > the $0.50 remaining
    fleet.tick()
    assert fleet.hooks.parked == ["46687567"]


def test_path3_survives_a_daemon_restart(fleet, tmp_path):
    """A preempt loop crosses restarts. A ceiling that a restart forgot is a
    ceiling that a preempt loop resets."""
    busy(fleet, 46687567, dph=3.6, label="keep:x")
    fleet.watch("46687567", "jobs", budget_usd=5.0, policy={"id": 46687567})
    fleet.hooks.jobs_spend = 3.0
    fleet.tick()
    fleet.hooks.jobs_result = "drained"
    fleet.tick()
    fleet.save()
    f2 = daemon.Fleet(fleet.dir, hooks=fleet.hooks)
    assert f2.state["ceilings"]["46687567"]["cap_usd"] == 5.0
    assert f2.state["ceilings"]["46687567"]["spend_usd"] == pytest.approx(3.0)
    f2.watch("46687567", "jobs", budget_usd=5.0, policy={"id": 46687567})
    assert f2.state["watches"]["46687567"]["spend_usd"] == pytest.approx(3.0)


def test_path3_a_watch_finished_line_now_reports_the_surviving_headroom(fleet):
    busy(fleet, 46687567, label="keep:x")
    fleet.watch("46687567", "jobs", budget_usd=5.0, policy={"id": 46687567})
    fleet.hooks.jobs_spend = 2.0
    fleet.tick()
    fleet.hooks.jobs_result = "drained"
    fleet.tick()
    fin = [r for r in journal(fleet) if r["event"] == "watch_finished"][-1]
    assert fin["cap_usd"] == 5.0 and fin["remaining_usd"] == pytest.approx(3.0)
    assert "CEILING SURVIVES" in fin["note"]


# --------------------------------------------------------------------------- #
# THE RE-ARM ARITHMETIC — remaining headroom, never the original figure
# --------------------------------------------------------------------------- #
def test_rearm_fixture_records_six_caps_of_ten_dollars():
    rows = fixture_rows("path1_rearm_46916278")
    regs = [r for r in rows
            if r["event"] == "watch_registered" and r.get("budget_usd")]
    caps = [r["budget_usd"] for r in regs]
    assert caps.count(10.0) >= 5, caps
    assert sum(caps) >= 50.0, \
        "the operator believed in a $10 cap; the effective ceiling was N x $10"


def test_rearming_the_same_figure_grants_no_new_headroom(fleet):
    busy(fleet, 46916278, dph=3.6, label="keep:x")
    fleet.watch("46916278", "jobs", budget_usd=10.0, policy={"id": 46916278})
    fleet.hooks.jobs_spend = 4.0
    fleet.tick()
    fleet.hooks.jobs_result = "drained"
    fleet.tick()
    fleet.hooks.jobs_result = None
    w = fleet.watch("46916278", "jobs", budget_usd=10.0, policy={"id": 46916278})
    assert w["budget_usd"] == 10.0, "the cap is still $10"
    assert w["spend_usd"] == pytest.approx(4.0), "...with $4 already drawn"
    assert fleet._ceiling_spend(w) == pytest.approx(4.0)
    reg = [r for r in journal(fleet) if r["event"] == "watch_registered"][-1]
    assert reg["remaining_usd"] == pytest.approx(6.0)
    assert reg["spend_carried_usd"] == pytest.approx(4.0)


def test_a_preempt_loop_of_six_rearms_never_exceeds_one_cap(fleet):
    """Box 46916278's exact shape: armed $10 six times. Pre-fix that was $60 of
    real ceiling with every individual box looking compliant."""
    busy(fleet, 46916278, dph=3.6, label="keep:x")
    spend = 0.0
    for cycle in range(6):
        fleet.hooks.jobs_result = None
        fleet.watch("46916278", "jobs", budget_usd=10.0, policy={"id": 46916278})
        spend += 1.5
        fleet.hooks.jobs_spend = spend
        fleet.tick()
        fleet.hooks.jobs_result = "drained"
        fleet.tick()
        assert ceiling(fleet, "46916278")["cap_usd"] == 10.0
        assert ceiling(fleet, "46916278")["spend_usd"] == pytest.approx(spend), \
            f"cycle {cycle}: spend must accumulate ACROSS re-arms"
    assert ceiling(fleet, "46916278")["spend_usd"] == pytest.approx(9.0)
    # the seventh cycle crosses $10 and parks, instead of starting a seventh $10
    fleet.hooks.jobs_result = None
    fleet.watch("46916278", "jobs", budget_usd=10.0, policy={"id": 46916278})
    fleet.hooks.jobs_spend = 10.5
    fleet.tick()
    fleet.tick()
    assert "46916278" in fleet.hooks.parked


def test_raising_a_cap_is_naming_a_bigger_number(fleet):
    """The budget-park alarm has always said `--budget ...` raises the cap. It
    still does — and now it means +$5 of headroom, not +$10."""
    busy(fleet, 500, dph=3.6, label="keep:x")
    fleet.watch("500", "jobs", budget_usd=5.0, policy={"id": 500})
    fleet.hooks.jobs_spend = 5.0
    fleet.tick()                                   # the ladder reports $5.00
    fleet.tick()                                   # the breach test sees it
    assert fleet.hooks.parked == ["500"]
    w = fleet.watch("500", "jobs", budget_usd=10.0, policy={"id": 500})
    assert w["budget_usd"] == 10.0 and w["spend_usd"] == pytest.approx(5.0)
    assert not fleet._budget_breached(w)


def test_reset_spend_is_the_only_way_back_to_zero_and_it_is_loud(fleet):
    busy(fleet, 500, label="keep:x")
    fleet.watch("500", "jobs", budget_usd=5.0, policy={"id": 500})
    fleet.hooks.jobs_spend = 4.0
    fleet.tick()
    w = fleet.watch("500", "jobs", budget_usd=5.0, policy={"id": 500},
                    reset_spend=True, requester="operator")
    assert w["spend_usd"] == 0.0
    armed = [r for r in journal(fleet) if r["event"] == "ceiling_armed"][-1]
    assert armed["spend_discarded_usd"] == pytest.approx(4.0)
    assert "SPEND RESET" in armed["note"] and armed["requester"] == "operator"


def test_reset_spend_is_never_automatic(fleet):
    """Nothing fleetd does on its own may reset a ceiling — not an adoption, not
    a lapse, not a replacement, not a restart."""
    busy(fleet, 500, dph=3.6, label="keep:x")
    fleet.watch("500", "jobs", budget_usd=5.0, policy={"id": 500})
    fleet.hooks.jobs_spend = 2.0
    fleet.tick()
    fleet.hooks.jobs_result = "drained"
    fleet.tick()                                    # lapse -> adopt
    fleet.tick()
    fleet.save()
    f2 = daemon.Fleet(fleet.dir, hooks=fleet.hooks)
    f2.tick()
    assert f2.state["ceilings"]["500"]["spend_usd"] >= 2.0
    assert not any(r.get("reset_spend") for r in journal(fleet)
                   if r["event"] == "ceiling_armed")


# --------------------------------------------------------------------------- #
# accounting the ledger also repairs
# --------------------------------------------------------------------------- #
def test_the_fleet_ceiling_no_longer_reads_a_reset_counter(fleet, monkeypatch):
    """`spend_by_box` is written by ASSIGNMENT, so a lapse that reset the watch
    counter overwrote the higher figure with a lower one — understating the
    fleet total that FLEETD_GLOBAL_BUDGET_USD is enforced against by ~42%."""
    busy(fleet, 500, dph=3.6, label="keep:x")
    fleet.watch("500", "jobs", budget_usd=10.0, policy={"id": 500})
    fleet.hooks.jobs_spend = 3.0
    fleet.tick()
    fleet.hooks.jobs_result = "drained"
    fleet.tick()
    fleet.tick()
    assert fleet.state["spend_by_box"]["500"] >= 3.0


def test_charging_is_idempotent_within_a_tick(fleet):
    busy(fleet, 500, dph=3.6, label="keep:x")
    fleet.watch("500", "bare", budget_usd=10.0)
    fleet.tick()
    fleet.hooks.advance(1000)
    fleet.tick()
    total = ceiling(fleet, "500")["spend_usd"]
    w = fleet.state["watches"]["500"]
    fleet._charge_ceiling(w)
    fleet._charge_ceiling(w)
    assert ceiling(fleet, "500")["spend_usd"] == pytest.approx(total)


def test_status_exposes_the_ceiling_and_orphan_headroom(fleet):
    busy(fleet, 500, label="keep:x")
    fleet.watch("500", "jobs", budget_usd=5.0, policy={"id": 500})
    fleet.hooks.jobs_spend = 2.0
    fleet.tick()
    st = fleet.status()
    row = [r for r in st["rows"] if r.get("iid") == "500"][0]
    assert row["ceiling_spend_usd"] == pytest.approx(2.0)
    assert row["remaining_usd"] == pytest.approx(3.0)
    assert st["adopt_default_budget_usd"] > 0
    c = [c for c in st["ceilings"] if c["ceiling_id"] == "500"][0]
    assert c["cap_usd"] == 5.0 and c["remaining_usd"] == pytest.approx(3.0)


def test_an_explicit_bare_watch_inherits_a_ceiling_it_finds(fleet):
    """A ceiling belongs to the BOX, not to whichever watch is holding it. This
    is also the shape that dropped a serve box to `bare` mid-flight."""
    busy(fleet, 500, label="keep:x")
    fleet.watch("500", "jobs", budget_usd=5.0, policy={"id": 500})
    fleet.hooks.jobs_spend = 2.0
    fleet.tick()
    fleet.hooks.jobs_result = "drained"
    fleet.tick()
    w = fleet.watch("500", "bare", requester="operator")
    assert w["budget_usd"] == 5.0 and w["spend_usd"] == pytest.approx(2.0)


def test_a_bare_watch_on_a_box_with_no_ceiling_stays_a_deliberate_choice(fleet):
    """The ONE remaining uncapped path, and it takes a human typing it. It is
    reported, not silently equivalent to a real cap."""
    fleet.hooks.box(500)
    w = fleet.watch("500", "bare", requester="operator")
    assert w["budget_usd"] is None and w["ceiling_id"] is None
    assert w["ceiling_source"] == "uncapped"


# --------------------------------------------------------------------------- #
# JOURNAL REPLAY — the external event sequence comes from the real journal,
# fleetd's RESPONSE comes from the code under test.
# --------------------------------------------------------------------------- #
def replay(f, rows, iids, dph=1.0):
    """Drive a real Fleet through a fixture's control-plane sequence.

    Only the OPERATOR/LADDER inputs are replayed — a human `fleet watch`, the
    ladder's verdict and spend, a box appearing. Everything fleetd decides
    (adopting, capping, parking, ending a watch) is produced by the code under
    test, which is what makes this a replay and not a re-enactment.
    """
    seen = set()
    for r in rows:
        ev, target = r["event"], str(r.get("target") or r.get("iid") or "")
        iid = str(r.get("iid") or "")
        if iid and iid in iids and iid not in seen:
            seen.add(iid)
            busy(f, int(iid), dph=dph, label=r.get("label") or "keep:replay")
        if ev == "watch_registered" and r.get("requester", "").startswith("operator"):
            if r.get("budget_usd") is None:
                continue                       # a bare pre-registration
            f.hooks.jobs_result = None
            f.watch(target, r.get("profile") or "bare", r["budget_usd"],
                    policy={"id": target}, requester=r["requester"])
        elif ev == "tick" and r.get("spend_usd") is not None:
            f.hooks.jobs_spend = max(f.hooks.jobs_spend, r["spend_usd"])
            f.hooks.jobs_result = None
            f.tick()
        elif ev == "watch_finished":
            f.hooks.jobs_result = r.get("verdict") or "drained"
            f.tick()
            f.hooks.jobs_result = None
        elif ev in ("unwatched", "unwatched_adopted", "watch_auto_adopted"):
            f.tick()
    f.tick()


def test_replay_path3_the_filed_box_never_goes_uncapped(fleet):
    rows = fixture_rows("path3_lapse_46687567")
    replay(fleet, rows, {"46687567"}, dph=0.5)
    caps = [r.get("budget_usd") for r in journal(fleet)
            if r["event"] in ("watch_registered", "watch_auto_adopted")]
    assert caps and all(c is not None for c in caps), \
        "every registration in the replay carries a cap"
    for w in fleet.state["watches"].values():
        assert w["budget_usd"] is not None
    # The journal's own re-arm ledger: $5, $5, $5, $3.50 = $18.50 of effective
    # ceiling pre-fix. The replayed ceiling is a single cap.
    armed = [r["budget_usd"] for r in rows
             if r["event"] == "watch_registered" and r.get("budget_usd")]
    assert sum(armed) > max(armed), "the fixture really does re-arm"
    assert ceiling(fleet, "46687567")["cap_usd"] == armed[-1]


def test_replay_path2_the_understudy_lands_inside_the_primarys_cap(fleet):
    rows = fixture_rows("path2_handoff_47214941")
    fleet.hooks.box(47214941, dph=2.804)
    fleet.watch("47214941", "jobs", budget_usd=5.0, policy={"id": 47214941},
                requester="operator@workstation")
    fleet.hooks.jobs_spend = 1.0
    fleet.tick()
    stray = [r for r in rows if r["event"] == "unwatched"
             and (r.get("label") or "").endswith(":handoff")][0]
    busy(fleet, int(stray["iid"]), dph=stray.get("dph") or 2.804,
         label=stray["label"])
    settle(fleet)                    # the real 174s stray window, four ticks
    u = fleet.state["watches"][stray["iid"]]
    assert u["budget_usd"] == 5.0 and u["ceiling_source"] == "inherited"
    # the journal recorded the opposite for this exact box
    recorded = [r for r in rows if r["event"] == "unwatched_adopted"
                and r["iid"] == stray["iid"]][0]
    assert "budget_usd" not in recorded


def test_replay_no_watch_in_any_fixture_ever_ticks_uncapped(fleet, tmp_path):
    """The census the journal failed: 121 distinct iids ticked with a null cap.
    Replayed against the fix, that set must be empty."""
    for name, ids in (("path3_lapse_46687567", {"46687567"}),
                      ("path1_rearm_46916278", {"46916278"})):
        d = tmp_path / name
        f = daemon.Fleet(str(d), hooks=FakeHooks())
        replay(f, fixture_rows(name), ids)
        uncapped = [r for r in journal(f)
                    if r["event"] == "tick" and r.get("profile") != "run"
                    and f.state["watches"].get(str(r.get("target")), {})
                    .get("budget_usd") is None
                    and str(r.get("target")) in f.state["watches"]]
        assert uncapped == [], f"{name}: {uncapped[:2]}"
