"""The three latch defects the 2026-08-14 field-data review found — all
observability, none economics. (1) A stuck eviction journaled its refusal
every ~50 s tick with no once-latch: box 47398836 wrote 79 identical
`jobs_rebid_refused` AND 79 identical `eviction_replacement_decision`
cap-refusals in 66 minutes. (2) The `evicted_announced` latch lived only in
memory, so each of the two deploy restarts that morning re-announced
47694876's single eviction. (3) `self_floor_since` froze across a stopped
gap (the guard is tenant-gated), so 47398836's floor-blind alarm fired ONE
tick after rescue_recovered, claiming "30 min continuous suppression" of a
box we did not hold for 67 of those minutes.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bidpolicy as bp  # noqa: E402
import fleetd  # noqa: E402
from vastlib.boxes import lifecycle  # noqa: E402
from vastlib.core import models  # noqa: E402
from vastlib.market import pricing  # noqa: E402
from vastlib.supervise import job_lane, journal, replacement  # noqa: E402

NOW = 1_000_000.0


def _ladder_events(jc, name=None):
    evs = [(ev, f) for ev, f in (jc.get("ladder_journal") or [])]
    return [e for e in evs if name is None or e[0] == name]


def _jc(**kw):
    jc = {"iid": "700", "last_bid": 0.48, "machine_id": 7,
          "ladder_journal": [],
          "a": argparse.Namespace(budget=10.0, dry_run=False)}
    jc.update(kw)
    return jc


# --------------------------------------------------------------------------- #
# 1a. jobs_rebid_refused: journal on CHANGE, not per tick
# --------------------------------------------------------------------------- #
def _refuse(reason):
    def _capture(**kw):
        return bp.Rebid("stop", None, reason, 1.0, 0)
    return _capture


def test_a_repeated_rebid_refusal_journals_once(monkeypatch):
    """47398836, 2026-08-10 20:05-21:11: 79 byte-identical refusals. The
    decision is re-made every tick (that retry is how rung zero eventually
    recovered the box); only the ANNOUNCEMENT dedups."""
    emits = []
    monkeypatch.setattr(bp, "rebid_ladder",
                        _refuse("one-shot job-aware defense already spent"))
    monkeypatch.setattr(journal, "_job_handoff_emit",
                        lambda jc, ev, **kw: emits.append(ev))
    jc = _jc()
    for _ in range(5):
        assert replacement._job_rebid_ladder(jc, jc["a"], "700", 0.90, 2.40,
                                   bp.EVICTION_OUTBID, NOW) is False
    assert len(_ladder_events(jc, "jobs_rebid_refused")) == 1
    assert emits.count("rebid_refused") == 1


def test_a_changed_rebid_refusal_reason_journals_again(monkeypatch):
    """A refusal whose numbers moved is a different bound binding — that is
    new information and must journal."""
    monkeypatch.setattr(journal, "_job_handoff_emit", lambda jc, ev, **kw: None)
    jc = _jc()
    monkeypatch.setattr(bp, "rebid_ladder", _refuse("bound A"))
    replacement._job_rebid_ladder(jc, jc["a"], "700", 0.90, 2.40, bp.EVICTION_OUTBID, NOW)
    replacement._job_rebid_ladder(jc, jc["a"], "700", 0.90, 2.40, bp.EVICTION_OUTBID, NOW)
    monkeypatch.setattr(bp, "rebid_ladder", _refuse("bound B"))
    replacement._job_rebid_ladder(jc, jc["a"], "700", 0.90, 2.40, bp.EVICTION_OUTBID, NOW)
    reasons = [f["reason"] for _, f in _ladder_events(jc, "jobs_rebid_refused")]
    assert reasons == ["bound A", "bound B"]


def test_a_successful_rebid_reopens_the_refusal_latch(monkeypatch):
    """rebid -> refused -> rebid clears `rebid_refused` (existing behavior),
    so the NEXT refusal episode journals from its first tick."""
    monkeypatch.setattr(journal, "_job_handoff_emit", lambda jc, ev, **kw: None)
    monkeypatch.setattr(lifecycle, "_put_bid_soft", lambda iid, p: (True, None))
    jc = _jc()
    monkeypatch.setattr(bp, "rebid_ladder", _refuse("bound A"))
    replacement._job_rebid_ladder(jc, jc["a"], "700", 0.90, 2.40, bp.EVICTION_OUTBID, NOW)
    monkeypatch.setattr(
        bp, "rebid_ladder",
        lambda **kw: bp.Rebid("rebid", 0.60, "rung 1", 1.0, 1))
    replacement._job_rebid_ladder(jc, jc["a"], "700", 0.90, 2.40, bp.EVICTION_OUTBID, NOW)
    assert jc["rebid_refused"] is None
    monkeypatch.setattr(bp, "rebid_ladder", _refuse("bound A"))
    replacement._job_rebid_ladder(jc, jc["a"], "700", 0.90, 2.40, bp.EVICTION_OUTBID, NOW)
    assert len(_ladder_events(jc, "jobs_rebid_refused")) == 2


# --------------------------------------------------------------------------- #
# 1b. eviction_replacement_decision: refusals dedup, rents never do
# --------------------------------------------------------------------------- #
def _replace_env(monkeypatch, decision):
    monkeypatch.setattr(bp, "replacement_decision", lambda **kw: decision)
    monkeypatch.setattr(replacement, "_job_replacement_offer",
                        lambda jc, excl, **kw: None)
    monkeypatch.setattr(pricing, "_market_ondemand_soft", lambda mid, n: None)
    monkeypatch.setattr(replacement, "_job_observed_lifetime_h", lambda jc: None)
    emits = []
    monkeypatch.setattr(journal, "_job_handoff_emit",
                        lambda jc, ev, **kw: emits.append(ev))
    return emits


def test_a_repeated_replacement_refusal_journals_once(monkeypatch):
    dec = bp.Replacement("skip", None, None,
                         "replacement cap reached (3/3) — not re-renting in a "
                         "loop", 1.0, 0.5)
    emits = _replace_env(monkeypatch, dec)
    jc = _jc(replacements=3)
    for _ in range(5):
        assert replacement._job_eviction_replace(jc, None, bp.EVICTION_OUTBID,
                                       "test") is False
    assert len(_ladder_events(jc, "eviction_replacement_decision")) == 1
    assert emits.count("eviction_replacement_decision") == 1
    assert jc["replacement_refused"] == dec.reason


def test_a_changed_replacement_refusal_reason_journals_again(monkeypatch):
    dec_a = bp.Replacement("skip", None, None, "reason A", 1.0, 0.5)
    dec_b = bp.Replacement("skip", None, None, "reason B", 1.0, 0.5)
    _replace_env(monkeypatch, dec_a)
    jc = _jc()
    replacement._job_eviction_replace(jc, None, bp.EVICTION_OUTBID, "test")
    replacement._job_eviction_replace(jc, None, bp.EVICTION_OUTBID, "test")
    monkeypatch.setattr(bp, "replacement_decision", lambda **kw: dec_b)
    replacement._job_eviction_replace(jc, None, bp.EVICTION_OUTBID, "test")
    reasons = [f["reason"] for _, f in
               _ladder_events(jc, "eviction_replacement_decision")]
    assert reasons == ["reason A", "reason B"]


# --------------------------------------------------------------------------- #
# 2. evicted_announced survives a daemon restart
# --------------------------------------------------------------------------- #
def test_the_eviction_announce_latch_is_persisted_state():
    """Two deploy restarts on 2026-08-14 re-announced 47694876's one eviction
    three times: the latch was memory-only. The announce docstring promises
    'seventeen ticks are one event' — that must include ticks on either side
    of a restart."""
    assert "evicted_announced" in fleetd.REPLACEMENT_STATE_KEYS


# --------------------------------------------------------------------------- #
# 3. an eviction ends the self-floor suppression episode (both lanes)
# --------------------------------------------------------------------------- #
def test_the_jobs_eviction_announce_resets_the_self_floor_episode(monkeypatch):
    """The floor-blind clock must not span a stopped gap: the eviction that
    opened the gap is a REAL market read (someone displaced us), so the
    suppression episode is over the moment it is classified."""
    # MIGRATED (was MIGRATION-BLOCKED, step 6e batch B3): `_sticky_on_demand`
    # landed at `vastlib.market.pricing` — exactly the home job_lane's seam
    # comment named — so `job_lane._job_announce_eviction` reaches a real body.
    # Seams go where the SUBJECT resolves them: `_job_market_read` is job_lane's
    # own module global, `_market_ondemand_soft` is read as `pricing.<name>`.
    monkeypatch.setattr(job_lane, "_job_market_read",
                        lambda jc, inst: models.MarketRead(False, None, None))
    monkeypatch.setattr(pricing, "_market_ondemand_soft", lambda mid, n: None)
    jc = _jc(self_floor_at=[0.48, "standing"], self_floor_since=NOW - 4000,
             self_floor_sustained_said=True)
    job_lane._job_announce_eviction(jc, "700", {"machine_id": 7}, is_bid=True,
                                    present=True, astat="exited",
                                    intended_status="running", claimed_work=False,
                                    budget=10.0)
    assert "self_floor_at" not in jc
    assert "self_floor_since" not in jc
    assert "self_floor_sustained_said" not in jc
    # ...and the eviction itself was journaled (the reset must not eat it)
    assert len(_ladder_events(jc, "jobs_box_evicted")) == 1
