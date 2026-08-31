"""Wiring for the job-aware one-shot defense (AUTOBID_DESIGN "Next iteration",
2026-08-09): the pre-rent `entry_floor` seed, the pre-eviction `p_alt`
replacement-market poll, what the re-bid ladder is handed, and what the learn
record (`jobs_box_evicted`) carries. The pure policy itself is
`test_defense_ceiling.py`; everything here is the plumbing that feeds it.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bidpolicy as bp  # noqa: E402
from vastlib.boxes import lifecycle  # noqa: E402
from vastlib.core import models  # noqa: E402
from vastlib.fleet import state as fleet_state  # noqa: E402
from vastlib.jobs import risk as jobs_risk  # noqa: E402
from vastlib.market import pricing  # noqa: E402
from vastlib.supervise import job_lane, replacement  # noqa: E402
from vastlib.supervise import journal as sup_journal  # noqa: E402

NOW = 1_000_000.0


# --------------------------------------------------------------------------- #
# _job_palt_poll / _job_palt_fresh
# --------------------------------------------------------------------------- #

def test_palt_poll_reads_unceilinged_and_never_its_own_machine(monkeypatch):
    """The read is the SAME query the replacement rung runs at eviction time,
    with the two deliberate differences the docstring promises: no max_dph
    (true market price, not a ceiling-filtered one) and our own machine — plus
    the watch's evicted/pull-bad sets — excluded (#73 in a different hat)."""
    calls = []

    def _offer(jctx, excl=None, rental="bid", max_dph=None, cuda=None):
        calls.append((rental, tuple(excl or ()), max_dph))
        return {"id": 9, "min_bid": 0.44, "machine_id": 501}

    monkeypatch.setattr(replacement, "_job_replacement_offer", _offer)
    jc = {"evicted_machines": {13}, "pull_bad_machines": {14}}
    replacement._job_palt_poll(jc, NOW, own_machine=7)
    assert calls == [("bid", (7, 13, 14), None)]
    assert jc["p_alt"] == 0.44
    assert jc["p_alt_ts"] == NOW
    assert jc["p_alt_machine"] == 501


def test_palt_poll_is_rate_limited(monkeypatch):
    calls = []
    monkeypatch.setattr(replacement, "_job_replacement_offer",
                        lambda *a, **k: calls.append(1) or
                        {"min_bid": 0.5, "machine_id": 1})
    jc = {}
    replacement._job_palt_poll(jc, NOW)
    replacement._job_palt_poll(jc, NOW + replacement.P_ALT_POLL_S - 1)      # inside the cadence
    assert len(calls) == 1
    replacement._job_palt_poll(jc, NOW + replacement.P_ALT_POLL_S + 1)      # past it
    assert len(calls) == 2


def test_palt_poll_failure_keeps_the_previous_read(monkeypatch):
    """A failed refresh must not erase a real read — the freshness gate retires
    it on its own schedule instead."""
    monkeypatch.setattr(replacement, "_job_replacement_offer", lambda *a, **k: None)
    jc = {"p_alt": 0.5, "p_alt_ts": NOW - replacement.P_ALT_POLL_S - 1}
    replacement._job_palt_poll(jc, NOW)
    assert jc["p_alt"] == 0.5
    assert jc["p_alt_ts"] == NOW - replacement.P_ALT_POLL_S - 1


def test_palt_fresh_gates_on_age():
    jc = {"p_alt": 0.5, "p_alt_ts": NOW - 10}
    assert replacement._job_palt_fresh(jc, NOW) == 0.5
    jc["p_alt_ts"] = NOW - replacement.P_ALT_MAX_AGE_S - 1
    assert replacement._job_palt_fresh(jc, NOW) is None, \
        "a stale p_alt is a memory, not a market read"
    assert replacement._job_palt_fresh({}, NOW) is None


# --------------------------------------------------------------------------- #
# the ladder driver hands the policy the defense inputs
# --------------------------------------------------------------------------- #

def _ladder_jc(**kw):
    jc = {"iid": "700", "last_bid": 0.80, "max_bid": None,
          "launch_dph_anchor": 0.76, "spend_usd": 1.0, "rebid_rungs": 0,
          "a": argparse.Namespace(budget=10.0, dry_run=False)}
    jc.update(kw)
    return jc


def test_driver_passes_fresh_palt_and_job_horizon_to_the_policy(monkeypatch):
    seen = {}

    def _capture(**kw):
        seen.update(kw)
        return bp.Rebid("stop", None, "captured", None, 0)

    monkeypatch.setattr(bp, "rebid_ladder", _capture)
    monkeypatch.setattr(sup_journal, "_job_handoff_emit", lambda jc, kind, **kw: None)
    monkeypatch.setattr(jobs_risk, "_jobs_work_horizon_h",
                        lambda views, now, **kw: 2.5)
    jc = _ladder_jc(p_alt=0.44, p_alt_ts=NOW - 10,
                    pending_views=[{"job_id": "j1", "checkpoint_s": 720},
                                   {"job_id": "j2", "checkpoint_s": 360}])
    replacement._job_rebid_ladder(jc, jc["a"], "700", 0.90, 2.40,
                        bp.EVICTION_OUTBID, NOW)
    assert seen["p_alt"] == 0.44
    assert seen["remaining_h"] == 2.5
    assert seen["ckpt_interval_h"] == 720 / 3600.0      # the WIDEST interval
    assert seen["setup_h"] == bp.SPOT_SETUP_H


def test_driver_passes_none_for_a_stale_palt(monkeypatch):
    """Stale read -> the policy sees no p_alt at all and keeps its pre-defense
    shape; pricing a live money move off a dead market read is the exact
    failure the freshness gate exists for."""
    seen = {}

    def _capture(**kw):
        seen.update(kw)
        return bp.Rebid("stop", None, "captured", None, 0)

    monkeypatch.setattr(bp, "rebid_ladder", _capture)
    monkeypatch.setattr(sup_journal, "_job_handoff_emit", lambda jc, kind, **kw: None)
    jc = _ladder_jc(p_alt=0.44, p_alt_ts=NOW - replacement.P_ALT_MAX_AGE_S - 1)
    replacement._job_rebid_ladder(jc, jc["a"], "700", 0.90, 2.40,
                        bp.EVICTION_OUTBID, NOW)
    assert seen["p_alt"] is None


def test_one_shot_refusal_flows_through_the_driver(monkeypatch):
    """End to end on the REAL policy: a fresh p_alt and a spent rung means the
    driver journals the one-shot refusal and falls through (returns False), so
    the replacement rung runs next — that refusal is the controller working."""
    emits = []
    monkeypatch.setattr(lifecycle, "_put_bid_soft", lambda iid, p: (True, None))
    monkeypatch.setattr(sup_journal, "_job_handoff_emit",
                        lambda jc, kind, **kw: emits.append((kind, kw)))
    jc = _ladder_jc(rebid_rungs=1, p_alt=0.44, p_alt_ts=NOW - 10)
    kept = replacement._job_rebid_ladder(jc, jc["a"], "700", 0.90, 2.40,
                               bp.EVICTION_OUTBID, NOW)
    assert kept is False
    kinds = [k for k, _ in emits]
    assert kinds == ["rebid_refused"]
    assert "one-shot job-aware defense already spent" in emits[0][1]["reason"]
    # the journal record carries the defense inputs for the postmortem
    journaled = [f for ev, f in jc.get("ladder_journal") or []
                 if ev == "jobs_rebid_refused"]
    assert journaled and journaled[0]["p_alt"] == 0.44


# --------------------------------------------------------------------------- #
# entry_floor: launch stamp -> instance env -> jc seed -> learn record
# --------------------------------------------------------------------------- #

def test_instance_env_entry_floor_seeds_jc_once():
    """The tick's seed block, exercised through its own inputs: the ENTRY_FLOOR
    the launch stamped into the box env parses back through _instance_env and
    _num_dph exactly as the seed reads it."""
    inst = {"extra_env": [["ENTRY_FLOOR", "0.1333"], ["OTHER", "x"]]}
    assert models._num_dph(models._instance_env(inst).get("ENTRY_FLOOR")) == 0.1333
    assert models._num_dph(models._instance_env({}).get("ENTRY_FLOOR")) is None


def test_eviction_event_carries_the_learn_record_fields(monkeypatch):
    """`jobs_box_evicted` is the learn record's accumulation point
    (design §4): entry_floor + p_alt ride on it, None-safe for
    pre-2026-08-09 boxes."""
    # MIGRATED (was MIGRATION-BLOCKED then MIGRATION-DEFERRED, step 6e batch B7):
    # `_sticky_on_demand` landed at `vastlib.market.pricing`, so
    # `job_lane._job_announce_eviction` reaches a real body. Each seam is stubbed
    # at the module the SUBJECT resolves it through: `_job_market_read` bare in
    # `job_lane` itself, `_market_ondemand_soft` as `pricing.<name>` (and
    # `_sticky_on_demand` beside it, left REAL — it is the arithmetic under test's
    # own input), `_job_handoff_emit` as `journal.<name>`. `models.MarketRead` is
    # the class the ported subject was written against.
    monkeypatch.setattr(job_lane, "_job_market_read",
                        lambda jc, inst: models.MarketRead(True, True, 0.30))
    monkeypatch.setattr(pricing, "_market_ondemand_soft", lambda m, n: 0.60)
    monkeypatch.setattr(sup_journal, "_job_handoff_emit", lambda jc, kind, **kw: None)
    jc = {"last_bid": 0.24, "entry_floor": 0.20, "p_alt": 0.26,
          "p_alt_ts": NOW - 30, "max_bid": 0.45, "spend_usd": 1.0}
    job_lane._job_announce_eviction(jc, "700", {"machine_id": 7, "num_gpus": 2},
                                    is_bid=True, present=True, astat="exited",
                                    intended_status="running", claimed_work=True,
                                    budget=5.0)
    ev = [f for name, f in jc["ladder_journal"]
          if name == "jobs_box_evicted"][0]
    assert ev["entry_floor"] == 0.20
    assert ev["p_alt"] == 0.26
    assert ev["p_alt_ts"] == NOW - 30


# --------------------------------------------------------------------------- #
# durability: the three observations survive a daemon restart
# --------------------------------------------------------------------------- #

def test_defense_observations_round_trip_the_watch_record():
    for k in ("entry_floor", "p_alt", "p_alt_ts"):
        assert k in fleet_state.REPLACEMENT_STATE_KEYS
    jc = {"entry_floor": 0.1333, "p_alt": 0.26, "p_alt_ts": NOW,
          "replacements": 1, "replacement_history": [],
          "launch_dph_anchor": 0.76, "launch_disk_gb": 110,
          "evicted_machines": {7}, "retained_boxes": [], "rebid_rungs": 0,
          "resume_tries": 0}
    w = {}
    fleet_state._replacement_state_persist(jc, w)
    jc2 = {}
    fleet_state._replacement_state_restore(jc2, w)
    assert jc2["entry_floor"] == 0.1333
    assert jc2["p_alt"] == 0.26
    assert jc2["p_alt_ts"] == NOW
