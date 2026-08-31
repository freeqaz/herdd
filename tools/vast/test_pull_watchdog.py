"""Portable tests for the jobs-lane boot-pull watchdog (owner directive
2026-08-02): a box that cannot pull the image within BOOT_PULL_TIMEOUT_S, or
whose sustained AGGREGATE pull rate sits under BOOT_MIN_MBPS, is a bad HOST —
terminate it, reschedule its queue on a fresh box (excluding failed machines),
keep supervising the replacement. The pull phase is GPU-unbilled
(invoice-verified, BOOT_HEALTHCHECK_DESIGN.md), so the kill costs only the
wasted pull; the load-bearing requirement is that the queued jobs continue
seamlessly (the 46590907 orphaned-ticket shape must be impossible on this
lane).

Toolchain-free lane: no vast API, no B2 — every seam monkeypatched.
"""
import argparse
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobmeta  # noqa: E402
from vastlib.boxes import lifecycle  # noqa: E402
from vastlib.market import offers as market_offers  # noqa: E402
from vastlib.supervise import handoff, job_lane, journal, replacement  # noqa: E402

NOW = 2_000_000.0
PULL_TIMEOUT = 600      # BOOT_PULL_TIMEOUT_S default
MAX_RETRIES = 3         # BOOT_MAX_HOST_RETRIES default


def _args(**kw):
    base = dict(id=41, dry_run=False, budget=None, max_bid=None,
                handoff=True, strict_ceiling=False, keep=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _jc(**kw):
    jc, _hf = job_lane.job_supervise_init(_args(**kw.pop("args", {})))
    jc["now"] = NOW
    jc.update(kw)
    return jc


def _inst(iid=41, status="loading", *, age=100, machine=7, status_msg=""):
    return {"id": iid, "actual_status": status, "machine_id": machine,
            "start_date": NOW - age, "status_msg": status_msg,
            "num_gpus": 1, "gpu_name": "RTX 4090", "label": "jobs-wave"}


# --- _job_pull_watchdog_tick: verdicts ---------------------------------------- #
def test_watchdog_silent_while_pull_is_young_and_moving():
    jc = _jc()
    assert replacement._job_pull_watchdog_tick(jc, _inst(age=100), NOW) is None


def test_watchdog_deadline_condemns_past_pull_timeout():
    """The blunt 10-min backstop: not `running` by BOOT_PULL_TIMEOUT_S after
    launch (start_date clock — survives supervisor restarts) => condemned,
    regardless of whether bytes are still trickling."""
    jc = _jc()
    assert replacement._job_pull_watchdog_tick(
        jc, _inst(age=PULL_TIMEOUT + 60), NOW) == "deadline"


def test_watchdog_slow_condemns_before_the_timeout():
    """The earlier, measured kill: a full BOOT_MBPS_WINDOW_S of downloading
    samples whose AGGREGATE byte rate sits under BOOT_MIN_MBPS condemns while
    the 10-min clock is still running — slow hosts die faster than dead ones."""
    jc = _jc()
    m1 = _inst(age=10, status_msg="cafe00: Downloading [>  ]  1.0 MB / 900.0 MB")
    assert replacement._job_pull_watchdog_tick(jc, m1, NOW) is None
    # 301s later (window full, still < the 600s deadline): +1 MB total => ~0.003 MB/s
    m2 = _inst(age=311, status_msg="cafe00: Downloading [>  ]  2.0 MB / 900.0 MB")
    assert replacement._job_pull_watchdog_tick(jc, m2, NOW + 301) == "slow"


def test_watchdog_retires_nothing_on_running(monkeypatch):
    """A box that reached `running` is out of this lane's jurisdiction — the
    sampler must report 'running' (not condemn) and the tick wiring retires it."""
    jc = _jc()
    replacement._job_pull_watchdog_tick(jc, _inst(age=100), NOW)     # arm
    assert replacement._job_pull_watchdog_tick(
        jc, _inst(status="running", age=PULL_TIMEOUT + 999), NOW) is None


def test_watchdog_disabled_flag_wins():
    jc = _jc()
    jc["pull_watchdog_disabled"] = True
    assert replacement._job_pull_watchdog_tick(
        jc, _inst(age=PULL_TIMEOUT + 999), NOW) is None


# --- _job_pull_condemn: the terminate + reschedule ---------------------------- #
def _wire_condemn(monkeypatch, jc, *, launch=(77, 0.5, None),
                  retarget=(["j1", "j2"], []), destroy_fail=None):
    calls = []
    monkeypatch.setattr(replacement, "_launch_job_replacement",
                        lambda jctx, excl, **kw: (
                            calls.append(("launch", excl, kw.get("max_dph"), kw)),
                            launch)[1])
    monkeypatch.setattr(replacement, "_retarget_pending_tickets",
                        lambda old, new: (calls.append(("retarget", old, new)),
                                          retarget)[1])
    monkeypatch.setattr(lifecycle, "_destroy_and_revoke",
                        lambda ids, ins, intent, noun="": (
                            calls.append(("destroy", list(ids), intent)),
                            destroy_fail or [])[1])
    monkeypatch.setattr(journal, "_job_handoff_emit",
                        lambda jctx, event, **kw: calls.append(("emit", event, kw)))
    return calls


def test_condemn_reschedules_seamlessly(monkeypatch, capsys):
    jc = _jc(instances=[_inst()])
    replacement._job_pull_watchdog_tick(jc, _inst(age=PULL_TIMEOUT + 60), NOW)  # sampler
    calls = _wire_condemn(monkeypatch, jc)
    assert replacement._job_pull_condemn(jc, _inst(), "deadline") is None
    kinds = [c[0] for c in calls]
    assert "launch" in kinds and "retarget" in kinds and "destroy" in kinds
    # ORDER is the safety property: queue moves BEFORE the condemned box dies,
    # so no orphaned-ticket window exists even transiently.
    assert kinds.index("retarget") < kinds.index("destroy")
    assert ("retarget", "41", 77) in calls
    assert [c for c in calls if c[0] == "destroy"][0][1] == ["41"]
    # the watch now tracks the replacement with fresh anchors + fresh sampler
    assert jc["iid"] == "77"
    assert jc["pull_relaunches"] == 1
    assert 7 in jc["pull_bad_machines"]
    assert "pull_sampler" not in jc
    events = [c[1] for c in calls if c[0] == "emit"]
    assert "pull_condemned" in events and "pull_rescheduled" in events


def test_condemn_excludes_failed_machines_from_the_relaunch(monkeypatch):
    jc = _jc(instances=[_inst()])
    jc["pull_bad_machines"] = {3}
    calls = _wire_condemn(monkeypatch, jc)
    replacement._job_pull_condemn(jc, _inst(machine=7), "deadline")
    excl = [c[1] for c in calls if c[0] == "launch"][0]
    assert set(excl) == {3, 7}


def test_condemn_relaunch_carries_the_replacement_price_ceiling(monkeypatch):
    """A forced rehost is an autonomous rental too (doc 50 R3, 2026-08-05):
    "the box wouldn't pull" is not a licence to buy a 3x one. The ceiling is
    `replace_ceiling_mult` x the ORIGINAL launch price, exactly as on the
    eviction lane, and it must reach the launcher — enforcing it only in the
    decision function is what let a $3.4741/hr box be rented under a $2.164
    ceiling."""
    jc = _jc(instances=[_inst()])
    jc["launch_dph_anchor"] = 0.76
    calls = _wire_condemn(monkeypatch, jc)
    replacement._job_pull_condemn(jc, _inst(machine=7), "deadline")
    assert [c[2] for c in calls if c[0] == "launch"][0] == pytest.approx(1.52)


def test_an_over_ceiling_refusal_reprices_the_ceiling_on_the_next_tick(monkeypatch):
    """THE WEDGE (2026-08-24). The ceiling is pushed into the offer search, so
    an unaffordable market reports as an EMPTY one and the price never reaches
    the ceiling — which is why a refusal has to be treated as the market read it
    is. Anchor $0.1933 -> $0.387; the one qualifying offer billed $0.4000; the
    lane refused 36 consecutive ticks over a 3.4% gap. Here the second tick
    re-prices and rents."""
    jc = _jc(instances=[_inst()])
    jc["launch_dph_anchor"] = 0.19333333333333333
    calls = _wire_condemn(monkeypatch, jc, launch=(None, None, "over_ceiling"))
    replacement._job_pull_condemn(jc, _inst(machine=7), "deadline")
    assert [c[2] for c in calls if c[0] == "launch"][0] == pytest.approx(0.387)
    assert jc["iid"] == "41", "the condemned box is KEPT with its queue"
    assert jc["replacement_refusals"] == 1

    # the refusal itself is what the launcher recorded — replay it by hand at
    # the seam the real launcher writes, then take the next tick.
    jc["replacement_market_floor"] = 0.40
    jc["replacement_market_floor_ts"] = NOW
    calls2 = _wire_condemn(monkeypatch, jc, launch=(77, 0.40, None))
    replacement._job_pull_condemn(jc, _inst(machine=7), "deadline")
    reprice = [c[2] for c in calls2 if c[0] == "launch"][0]
    assert reprice == pytest.approx(0.44), "1.10x the observed $0.4000 floor"
    assert reprice > 0.40, "the re-derived ceiling clears the offer that was refused"
    assert jc["iid"] == "77", "the queue is rehosted"
    assert jc["launch_dph_anchor"] == 0.19333333333333333, \
        "the ANCHOR is never rewritten — that is what stops N swaps compounding"
    assert "replacement_refusals" not in jc, "a rental that happened ends the streak"


def test_the_launcher_records_the_price_it_refused(monkeypatch):
    """A refusal IS a market read, and on this lane it is the only one."""
    jc = _jc()
    jc["now"] = NOW
    monkeypatch.setattr(replacement, "_job_replacement_offer",
                        lambda *a, **kw: {"id": 9, "machine_id": 3,
                                          "min_bid": 0.35, "dph_total": 0.36})
    monkeypatch.setattr(replacement.pricing, "_market_ondemand_soft",
                        lambda *a, **kw: 2.0)
    iid, dph, reason = replacement._launch_job_replacement(jc, [], max_dph=0.30)
    assert (iid, dph, reason) == (None, None, "over_ceiling")
    assert jc["replacement_market_floor"] == pytest.approx(0.42)   # 1.2 x $0.35
    assert jc["replacement_market_floor_ts"] == NOW


def test_a_ceiling_that_moves_is_journaled_once_per_change(monkeypatch):
    """An autonomous spend bound that widens silently is not a bound. It is also
    not a per-tick event: the derivation runs every tick and only a CHANGE is
    worth a line (the 79+79 identical-refusal shape, AUTOBID_DESIGN)."""
    rows = []
    monkeypatch.setattr(journal, "_job_ladder_journal",
                        lambda jctx, event, **kw: rows.append((event, kw)))
    jc = _jc()
    jc["launch_dph_anchor"] = 0.19333333333333333
    assert replacement._job_replacement_ceiling(jc) == pytest.approx(0.387)
    assert not [r for r in rows if r[0] == "jobs_replacement_ceiling_repriced"]

    jc["replacement_market_floor"], jc["replacement_market_floor_ts"] = 0.40, NOW
    assert replacement._job_replacement_ceiling(jc) == pytest.approx(0.44)
    ev = [r for r in rows if r[0] == "jobs_replacement_ceiling_repriced"]
    assert len(ev) == 1
    k = ev[0][1]
    assert k["escalated"] is True and k["ceiling"] == pytest.approx(0.44)
    assert k["base_ceiling"] == pytest.approx(0.387)
    assert k["market_ref"] == pytest.approx(0.40) and k["market_source"] == "market_floor"
    assert k["launch_dph_anchor"] == 0.19333333333333333

    # same market next tick => same ceiling => no second line
    assert replacement._job_replacement_ceiling(jc) == pytest.approx(0.44)
    assert len([r for r in rows if r[0] == "jobs_replacement_ceiling_repriced"]) == 1


def test_a_no_offer_refusal_still_counts_toward_the_wedge(monkeypatch):
    """`no_offer` was 29 of the 36 refusals: it is what an unaffordable market
    LOOKS like through a ceiling-filtered search. It must not read as a quieter
    failure than `over_ceiling`."""
    jc = _jc(instances=[_inst()])
    jc["launch_dph_anchor"] = 0.76
    for n in (1, 2, 3):
        _wire_condemn(monkeypatch, jc, launch=(None, None, "no_offer"))
        replacement._job_pull_condemn(jc, _inst(machine=7), "deadline")
        assert jc["replacement_refusals"] == n
    assert jc["replacement_refusal_reason"] == "no_offer"
    assert jc["replacement_refusals_since"] == NOW


def test_relaunch_cap_disarms_and_never_rerents(monkeypatch):
    """The relaunch-loop guard: several slow hosts in a row (or a broken image)
    must stop and alarm, not burn money in a circle. The condemned box is KEPT
    (GPU-unbilled) so its queue is never orphaned automatically."""
    jc = _jc(instances=[_inst()])
    jc["pull_relaunches"] = MAX_RETRIES
    calls = _wire_condemn(monkeypatch, jc)
    assert replacement._job_pull_condemn(jc, _inst(), "deadline") is None
    kinds = [c[0] for c in calls]
    assert "launch" not in kinds and "destroy" not in kinds \
        and "retarget" not in kinds
    assert jc["pull_watchdog_disabled"] is True
    assert jc["iid"] == "41"
    assert [c[1] for c in calls if c[0] == "emit"] == [
        "pull_condemned", "pull_relaunch_exhausted"]
    # and the disarm sticks for future ticks
    assert replacement._job_pull_watchdog_tick(
        jc, _inst(age=PULL_TIMEOUT + 999), NOW) is None


def test_budget_rail_blocks_the_reschedule(monkeypatch):
    jc = _jc(args={"budget": 1.0}, instances=[_inst()])
    jc["spend_usd"] = 1.5
    calls = _wire_condemn(monkeypatch, jc)
    assert replacement._job_pull_condemn(jc, _inst(), "deadline") is None
    assert [c[0] for c in calls if c[0] in ("launch", "destroy", "retarget")] == []
    assert jc["iid"] == "41"


def test_failed_replacement_keeps_the_condemned_box(monkeypatch):
    """No offer / unlaunchable: the condemned box must be KEPT — destroying it
    without a replacement would orphan its queue (the 46590907 shape)."""
    jc = _jc(instances=[_inst()])
    replacement._job_pull_watchdog_tick(jc, _inst(age=PULL_TIMEOUT + 60), NOW)
    calls = _wire_condemn(monkeypatch, jc, launch=(None, None, "no_offer"))
    assert replacement._job_pull_condemn(jc, _inst(), "deadline") is None
    kinds = [c[0] for c in calls]
    assert "destroy" not in kinds and "retarget" not in kinds
    assert jc["iid"] == "41" and jc.get("pull_relaunches", 0) == 0
    assert "pull_sampler" not in jc          # re-armed fresh for the retry


def test_dry_run_touches_nothing(monkeypatch, capsys):
    jc = _jc(args={"dry_run": True}, instances=[_inst()])
    calls = _wire_condemn(monkeypatch, jc)
    assert replacement._job_pull_condemn(jc, _inst(), "deadline") is None
    assert [c[0] for c in calls if c[0] in ("launch", "destroy", "retarget")] == []
    assert "[dry-run]" in capsys.readouterr().out


# --- _retarget_pending_tickets: the seamless-queue move ----------------------- #
def test_retarget_moves_every_pending_ticket(monkeypatch):
    written, deleted, events = [], [], []
    monkeypatch.setattr(jobmeta, "list_queue", lambda box: ["j1", "j2"])
    monkeypatch.setattr(jobmeta, "read_ticket",
                        lambda box, jid: {"id": jid, "box": box,
                                          "bundle_sha256": "sha"})
    monkeypatch.setattr(jobmeta, "write_ticket",
                        lambda t: (written.append(t), (True, "k", None))[1])
    monkeypatch.setattr(jobmeta, "delete_ticket",
                        lambda box, jid: (deleted.append((box, jid)),
                                          (True, None))[1])
    monkeypatch.setattr(jobmeta, "emit_event",
                        lambda jid, ev, **kw: events.append((jid, ev, kw)))
    moved, failed = replacement._retarget_pending_tickets("41", "77")
    assert moved == ["j1", "j2"] and failed == []
    assert all(t["box"] == "77" and t["retargeted_from"] == "41"
               for t in written)
    assert deleted == [("41", "j1"), ("41", "j2")]
    assert [(j, e) for j, e, _ in events] == [("j1", "retargeted"),
                                              ("j2", "retargeted")]


def test_retarget_one_bad_ticket_never_kills_the_rest(monkeypatch):
    monkeypatch.setattr(jobmeta, "list_queue", lambda box: ["j1", "j2"])
    monkeypatch.setattr(jobmeta, "read_ticket",
                        lambda box, jid: (_ for _ in ()).throw(RuntimeError("x"))
                        if jid == "j1" else {"id": jid})
    monkeypatch.setattr(jobmeta, "write_ticket", lambda t: (True, "k", None))
    monkeypatch.setattr(jobmeta, "delete_ticket", lambda box, jid: (True, None))
    monkeypatch.setattr(jobmeta, "emit_event", lambda jid, ev, **kw: None)
    moved, failed = replacement._retarget_pending_tickets("41", "77")
    assert moved == ["j2"] and failed == ["j1"]


def test_retarget_unreadable_queue_reports_and_moves_nothing(monkeypatch, capsys):
    monkeypatch.setattr(jobmeta, "list_queue",
                        lambda box: (_ for _ in ()).throw(RuntimeError("b2 down")))
    moved, failed = replacement._retarget_pending_tickets("41", "77")
    assert moved == [] and failed == ["<queue-unreadable>"]
    assert "by hand" in capsys.readouterr().out


# --- _launch_job_replacement: machine exclusion threads to the offer pick ----- #
def test_replacement_offer_pick_excludes_failed_machines(monkeypatch):
    """The exclusion is the entire point of the reschedule, so it must reach the
    OFFER QUERY. Stubbed at `pick_offers` since 2026-08-16: the pick is a
    candidate-set search now (`_job_replacement_offers`), and
    `pick_cheapest_offer` is its limit-1 face."""
    captured = {}

    def fake_pick(**kw):
        captured.update(kw)
        return []                                     # no offer -> early return

    monkeypatch.setattr(market_offers, "pick_offers", fake_pick)
    jc = _jc(instances=[_inst()])
    cid, dph, reason = replacement._launch_job_replacement(jc, [3, 7])
    assert reason == "no_offer" and cid is None
    assert captured.get("exclude_machines") == [3, 7]


def test_pull_condemned_rehost_is_sized_by_the_LAUNCH_not_by_the_clamp(monkeypatch):
    """Box 48005604 (2026-08-18): launched `--disk 50` for a bundle staging a
    19.3 GB base model, handed 10 GB by a host with no more to give, condemned
    by this watchdog 10 minutes later — and rehosted onto 10 GB, because the
    sizing read the ALLOCATION rather than the request. This lane only ever
    fires on a box that never finished booting, so it is the one most exposed to
    a clamp nobody has noticed yet.

    Both halves are pinned: the `--disk` the rehost asks for, and the container
    floor its OFFER SEARCH filters on. Sizing the rental right buys nothing if
    the search already handed us another 10 GB machine."""
    seen = []

    def fake_pick(**kw):
        seen.append(kw)
        return []                                     # no offer -> early return

    monkeypatch.setattr(market_offers, "pick_offers", fake_pick)
    clamped = dict(_inst(age=PULL_TIMEOUT + 120), disk_space=10.0, disk_usage=-1,
                   extra_env=[["LAUNCH_DISK_GB", "50"]])
    monkeypatch.setattr(lifecycle, "_instances_soft", lambda: [clamped])
    monkeypatch.setattr(handoff, "_job_handoff_reconcile", lambda jc, hf: None)
    monkeypatch.setattr(replacement, "_job_pull_condemn", lambda *a, **k: None)
    jc, hf = job_lane.job_supervise_init(_args())
    job_lane.job_supervise_tick(jc, hf)          # anchors off the box env
    jc["instances"] = [clamped]
    replacement._launch_job_replacement(jc, [7])
    # the first call is the real search; the trailing disk_gb=0 one is the
    # shortfall probe, which re-searches UNFLOORED to name the bound
    assert seen[0].get("disk_gb") == 50.0, \
        "the replacement search inherited the clamp instead of the request"
    need, _why, known = replacement._replacement_disk_need(jc, clamped)
    assert (need, known) == (50.0, True)


def test_a_loading_box_stamps_its_disk_request_on_the_watch(monkeypatch):
    """The anchor has to be readable BEFORE the box finishes booting — a
    pull-condemned box never gets further, and its `disk_usage` is the -1
    'container not provisioned' sentinel the whole time."""
    box = dict(_inst(age=100), disk_space=10.0, disk_usage=-1,
               extra_env=[["LAUNCH_DISK_GB", "50"]])
    monkeypatch.setattr(lifecycle, "_instances_soft", lambda: [box])
    monkeypatch.setattr(handoff, "_job_handoff_reconcile", lambda jc, hf: None)
    monkeypatch.setattr(replacement, "_job_pull_condemn", lambda *a, **k: None)
    jc, hf = job_lane.job_supervise_init(_args())
    job_lane.job_supervise_tick(jc, hf)
    assert jc["launch_disk_gb"] == 50.0


# --- job_supervise_tick wiring ------------------------------------------------ #
def test_tick_routes_a_condemned_pull_into_the_reschedule(monkeypatch):
    """End-to-end through the real tick: a loading box past the pull timeout
    reaches _job_pull_condemn (and the tick returns None — keep supervising)."""
    box = _inst(age=PULL_TIMEOUT + 120)
    monkeypatch.setattr(lifecycle, "_instances_soft", lambda: [box])
    monkeypatch.setattr(handoff, "_job_handoff_reconcile", lambda jc, hf: None)
    hit = {}
    monkeypatch.setattr(replacement, "_job_pull_condemn",
                        lambda jc, inst, verdict: hit.update(
                            iid=jc["iid"], verdict=verdict) or None)
    jc, hf = job_lane.job_supervise_init(_args())
    assert job_lane.job_supervise_tick(jc, hf) is None
    assert hit == {"iid": "41", "verdict": "deadline"}


def test_tick_never_arms_the_watchdog_on_a_running_box(monkeypatch):
    """A running box must sail past the watchdog into the normal ladder (here:
    queue_empty exit) with the sampler retired."""
    box = _inst(status="running", age=PULL_TIMEOUT + 999)
    monkeypatch.setattr(lifecycle, "_instances_soft", lambda: [box])
    monkeypatch.setattr(handoff, "_job_handoff_reconcile", lambda jc, hf: None)
    monkeypatch.setattr(jobmeta, "list_queue", lambda box: [])
    monkeypatch.setattr(replacement, "_job_pull_condemn",
                        lambda *a, **k: pytest.fail("condemned a running box"))
    jc, hf = job_lane.job_supervise_init(_args())
    jc["pull_sampler"] = object()                     # stale sampler from a park
    assert job_lane.job_supervise_tick(jc, hf) == "queue_empty"
    assert "pull_sampler" not in jc


# --- rental inheritance (2026-08-08 regression) ------------------------------- #
# _launch_job_replacement defaults to rental="bid", and this lane used to call it
# without a rental at all — so a pull-condemned ON-DEMAND box was silently
# rehosted onto spot. On-demand is not a price preference the rehost may
# re-optimise: it is chosen when an interruption would CONFOUND the work rather
# than merely delay it. Box 47165024 (a serial 9-cell DDP ladder, launched
# --type ondemand for that reason) was condemned on a slow host, rehosted to
# spot, and outbid ~2 min later.
def test_condemn_rehosts_an_ondemand_box_ONTO_ONDEMAND(monkeypatch):
    jc = _jc(instances=[_inst()])
    calls = _wire_condemn(monkeypatch, jc)
    replacement._job_pull_condemn(jc, dict(_inst(), is_bid=False), "deadline")
    kw = [c[3] for c in calls if c[0] == "launch"][0]
    assert kw.get("rental") == "ondemand"


def test_condemn_rehosts_a_spot_box_onto_spot(monkeypatch):
    jc = _jc(instances=[_inst()])
    calls = _wire_condemn(monkeypatch, jc)
    replacement._job_pull_condemn(jc, dict(_inst(), is_bid=True), "deadline")
    kw = [c[3] for c in calls if c[0] == "launch"][0]
    assert kw.get("rental") == "bid"


def test_condemn_unknown_is_bid_keeps_the_cheap_rung(monkeypatch):
    """An absent is_bid must not be read as on-demand: that would double the
    bill for a spot box the moment the API changes shape."""
    jc = _jc(instances=[_inst()])
    calls = _wire_condemn(monkeypatch, jc)
    replacement._job_pull_condemn(jc, _inst(), "deadline")          # no is_bid key
    kw = [c[3] for c in calls if c[0] == "launch"][0]
    assert kw.get("rental") == "bid"
