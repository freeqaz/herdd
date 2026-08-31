"""Portable regression tests for the 2026-08-08 night's fleetd defect cluster.

Every number in here is measured — from `journalctl --user -u vast-fleetd` for
the night of 2026-08-08, from the `show` snapshot of box 47218938, and from
read-only vast API reads taken 2026-08-09 against two live bid instances. The
incident record is `FLEETD_INCIDENT_2026-08-08.md`.

Four defects, four sections:

  #74  fleetd never CLASSIFIED the displacement of box 47214941 as an eviction
       and said nothing structured about it for fourteen minutes. Two causes:
       `intended_status: stopped` was read as evidence of intent (it is not —
       vast reports it for real displacements), and `classify_eviction` cannot
       observe an outbid because the machine stops listing rentable bid offers
       at the moment it is taken (defect D7: 0 `outbid` classifications in
       production, ever).
  #78  the whole autonomous replacement / pull-reschedule chain — launch,
       condemn, retarget, destroy — reached `herdd fleet log` as NOTHING.
  #73  the defend ladder was handed OUR OWN STANDING BID as "the market floor",
       because on a chunk we hold vast lists `min_bid` as the price to displace
       the current tenant. Self-referential ratchet toward `max_bid`.
  #76  the $2.9/hr box that launched at a printed $1.338 bid. Not a launch bug:
       the launch price was correct and PUT correctly; #73 walked it there in
       eleven seconds, amplified by `BID_TARGET_MULT` 1.20 -> 2.00 and by
       `jc["last_bid"]` being seeded from `dph_total` (bid + storage) instead of
       `dph_base` (the standing bid).

Toolchain-free lane: no vast API, no B2, no clock. Every seam monkeypatched.
"""
import argparse
import collections
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bidpolicy as bp  # noqa: E402
import fleetd  # noqa: E402
import jobmeta  # noqa: E402
from vastlib.boxes import lifecycle  # noqa: E402
from vastlib.core import api, models  # noqa: E402
from vastlib.jobs import risk as jobs_risk  # noqa: E402
from vastlib.market import pricing  # noqa: E402
from vastlib.supervise import handoff, job_lane, journal, replacement  # noqa: E402

NOW = 3_000_000.0

#: Local stand-in for `models.MarketRead` so the tick harness below can be
#: monkeypatched onto a build that predates it — see `_tick_env`, whose two
#: `raising=False` probe patches exist for the same fail-first reason. Carries
#: the row-level fields (2026-08-10, F3) with defaults, so it still quacks
#: like the 3-field original.
_MarketRead = collections.namedtuple("_MarketRead",
                                     ["ok", "listed", "min_bid",
                                      "floors", "scaled"])
_MarketRead.__new__.__defaults__ = ((), False)

# --------------------------------------------------------------------------- #
# The incident, as numbers.
# --------------------------------------------------------------------------- #
#: Box 47214941, H200 SXM, `jobs` watch with a $5 budget and one RUNNING eval
#: ticket. Priced off its GPU at 23:03Z: the machine's min_bid rose past our
#: standing bid and the offer went `avail: no`. The last floor fleetd ever read
#: for it was $1.315789 at 22:18:34Z — the machine's TRUE floor, 25/19.
BOX_A = dict(
    iid="47214941", machine=45768,
    standing_bid=2.55,          # our bid at displacement
    risen_min_bid=2.81,         # what the machine's min_bid rose to
    true_floor=1.3158,          # journal 22:18:34Z, and the handoff's own
                                # `candidate_min_bid: 1.3333333333333333`
    on_demand=3.876315789473685,   # journal 22:17:41Z `jobs_handoff_armed`
    # The defend ladder's four steps, 22:10:07 -> 22:14:55Z. Each is exactly
    # BID_MIN_CUSHION_MULT (1.10) x the floor the PREVIOUS step had just been
    # handed, and each of those floors was our own preceding bid read back.
    ladder=[2.697, 2.818, 3.100, 3.410],
    # The re-bid ladder, 23:06:25 -> 23:11:28Z, on a `floor $None` read.
    rebid=[2.134, 2.667, 3.216],
)

#: Box 47226953, RTX PRO 6000 96GB, machine 138918, `jobs` watch with a $3
#: budget registered 01:25:50Z and one live claimed ticket. Stopped on its own
#: at ~01:31Z with NO price cause at all — min_bid $0.3333 against our standing
#: $0.667 and `avail: yes` — and fleetd emitted no decision event: three ticks
#: at a FLAT spend_usd 0.0831 (the box was no longer billing GPU) and then
#: `operator_intent_start`, a human running `herdd start`.
#:
#: FAIRNESS: the operator intervened after ~2 minutes / 2 ticks, which may be
#: inside a legitimate debounce window, so this box alone does NOT establish the
#: fourteen-minute silence — 47214941 does. What it establishes is (a) that a box
#: can stop with our bid comfortably winning, so recovery cannot key on a price
#: signal, and (b) that resume-in-place beats renting: the husk still held 59 GB
#: of a 104 GiB base+merged pull and `start` recovered it in ~40 s.
BOX_C = dict(
    iid="47226953", machine=138918,
    standing_bid=0.667,         # dph_base
    dph_total=0.7197777777777778,
    min_bid=0.3333333333333333,  # STILL BELOW our bid, and still listed
    on_demand=1.2666666666666666,
    budget=3.0, spend=0.0831,
)

#: Box 47218938, H200 NVL 1/8 chunk, machine 37586. `herdd launch --gpu h200
#: --type bid --max-dph 2.0` printed "auto bid price $1.338 = 1.2x floor $1.333,
#: capped below on-demand $1.339" and PUT exactly that. Six minutes later it
#: billed $2.944/hr. Numbers from the committed `show` snapshot.
BOX_B = dict(
    iid="47218938", machine=37586,
    launch_bid=1.338,           # what launch printed AND PUT
    disk_hour=0.13722222222222222,  # instance.diskHour
    dph_total_at_adoption=1.4752222222222222,   # journal 23:22:17Z "defending
                                # bid $1.4752222222222222 -> $2.676" == launch
                                # bid + diskHour, i.e. dph_total, NOT the bid
    inst_min_bid=1.333333,      # the machine's TRUE floor, on the instance body,
                                # unchanged throughout
    defended=[2.676, 2.944],    # 23:22:17Z and 23:28:00Z; 2.676 == 2.00 x 1.338
    dph_base_after=2.944,       # snapshot: dph_base == our last PUT
    dph_total_after=3.0812222222222223,   # snapshot: dph_base + diskHour
)


# --------------------------------------------------------------------------- #
# 1. #74 — the eviction was never classified
# --------------------------------------------------------------------------- #
def test_d7_an_unlisted_machine_is_an_outbid_not_unknown():
    """THE defect. `classify_eviction` has never once returned `outbid` in
    production (AUTOBID_AUDIT_2026-08-08 §4: 0 of 15) — not because we are never
    outbid but because the discriminator is blind at exactly the moment it is
    asked. `market_min_bid` comes from a `rentable: {eq: True}` query, and a
    machine that has just been TAKEN lists none, so the read is None and the
    `mmb > lb` test can never fire.

    With the read's evidence preserved (`market_listed=False` — vast ANSWERED
    and the machine is no longer purchasable) that is a displacement."""
    assert bp.classify_eviction(
        present=True, actual_status="exited",
        market_min_bid=None, market_listed=False,
        on_demand=BOX_A["on_demand"], last_bid=BOX_A["standing_bid"],
    ) == bp.EVICTION_OUTBID


def test_a_failed_market_read_is_still_unknown_never_outbid():
    """SPOT_DESIGN §5 rule 1 — transient != eviction. `market_listed=None` is
    "nobody asked / the request failed", and it must stay ignorance. The whole
    point of the tri-state is that `False` can only come from a request that
    SUCCEEDED."""
    assert bp.classify_eviction(
        present=True, actual_status="exited",
        market_min_bid=None, market_listed=None,
        on_demand=BOX_A["on_demand"], last_bid=BOX_A["standing_bid"],
    ) == bp.EVICTION_UNKNOWN
    # and the pre-existing default (no kwarg at all) is unchanged
    assert bp.classify_eviction(
        present=True, actual_status="exited", market_min_bid=None,
        on_demand=BOX_A["on_demand"], last_bid=BOX_A["standing_bid"],
    ) == bp.EVICTION_UNKNOWN


def test_ondemand_displacement_still_outranks_the_unlisted_evidence():
    """ORDER. An on-demand renter also empties the bid listing, so the two
    signals co-occur. "No bid can win this back" is the more actionable class
    and keeps precedence — the v7 eviction-2 shape ($1.05 bid, $1.0017
    on-demand) must not become `outbid` now that `market_listed` exists."""
    assert bp.classify_eviction(
        present=True, actual_status="exited",
        market_min_bid=None, market_listed=False,
        on_demand=1.0017, last_bid=1.05,
    ) == bp.EVICTION_ONDEMAND


def test_market_listed_probe_splits_a_failed_read_from_an_empty_book(monkeypatch):
    """The probe that produces the evidence. Three answers, and the two that
    used to be indistinguishable are now distinct."""
    def _resp(ok, body):
        return lambda m, p, b=None, **kw: (ok, body, None if ok else "HTTP 500")

    monkeypatch.setattr(api, "request_soft",
                        _resp(True, {"offers": [{"min_bid": 1.3158,
                                                 "num_gpus": 2}]}))
    assert pricing._market_bid_listed_soft(BOX_A["machine"], 2) is True

    monkeypatch.setattr(api, "request_soft", _resp(True, {"offers": []}))
    assert pricing._market_bid_listed_soft(BOX_A["machine"], 2) is False, \
        "vast ANSWERED and this machine lists nothing — that is the outbid"

    monkeypatch.setattr(api, "request_soft", _resp(False, None))
    assert pricing._market_bid_listed_soft(BOX_A["machine"], 2) is None
    assert pricing._market_bid_listed_soft(None, 2) is None

    # ...and the two-state contract every other caller depends on is untouched.
    monkeypatch.setattr(api, "request_soft", _resp(True, {"offers": []}))
    assert pricing._market_min_bid_soft(BOX_A["machine"], 2) is None


def test_intended_status_stopped_is_not_evidence_of_intent():
    """#74's rule. Vast reported `intended_status: stopped` for 47214941 while
    it was being competitively displaced, so that field describes the box, not
    an operator's ask. A box that stopped under a NON-TERMINAL ticket with no
    journaled stop intent is an eviction whatever its rental type."""
    common = dict(present=True, live=False, intended_status="stopped",
                  box_parked=False, box_drained=False)
    # the incident's shape (bid box) — unchanged, still an eviction
    assert job_lane.classify_job_box_stop(is_bid=True, claimed_work=True,
                                   stop_intent=False, **common) is None
    # the NEW half: an ON-DEMAND box under live work is an eviction too. Before
    # 2026-08-09 this returned "operator_park" and supervise exited.
    assert job_lane.classify_job_box_stop(is_bid=False, claimed_work=True,
                                   stop_intent=False, **common) is None


def test_a_journaled_stop_intent_still_reads_as_an_operator_park():
    """The half that must NOT regress: `fleet park`, `herdd stop`, `guard` and
    `reap` all record intent before the PUT, and a genuine park must never be
    rescue-resumed. Intent outranks everything below it, INCLUDING a bid box
    with live work — which is the case the old classifier could not express at
    all."""
    common = dict(present=True, live=False, intended_status="stopped",
                  box_parked=False, box_drained=False)
    assert job_lane.classify_job_box_stop(is_bid=True, claimed_work=True,
                                   stop_intent=True, **common) == "operator_park"
    assert job_lane.classify_job_box_stop(is_bid=False, claimed_work=False,
                                   stop_intent=True, **common) == "operator_park"
    # an IDLE on-demand box keeps the pre-2026-08-09 answer with no intent
    assert job_lane.classify_job_box_stop(is_bid=False, claimed_work=False,
                                   stop_intent=False, **common) == "operator_park"
    # a self-park still wins over everything
    assert job_lane.classify_job_box_stop(
        present=True, live=False, is_bid=True, intended_status="stopped",
        box_parked=True, box_drained=False, claimed_work=True) == "self_parked"


# --------------------------------------------------------------------------- #
# tick-level harness
# --------------------------------------------------------------------------- #
def _args(**kw):
    base = dict(id=int(BOX_A["iid"]), dry_run=False, budget=5.0, max_bid=None,
                handoff=False, strict_ceiling=False, keep=False,
                max_replacements=None, replace_ceiling_mult=None,
                replacement_retention_hours=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _inst(iid=BOX_A["iid"], status="exited", machine=BOX_A["machine"],
          dph_total=2.6875, dph_base=None, intended="stopped", is_bid=True):
    return {"id": int(iid), "actual_status": status, "machine_id": machine,
            "intended_status": intended,
            "dph_total": dph_total,
            "dph_base": dph_base if dph_base is not None else dph_total,
            "num_gpus": 1, "gpu_name": "H200 SXM", "label": "upstream-monorepo",
            "start_date": NOW - 3600, "is_bid": is_bid}


def _tick_env(monkeypatch, inst, *, market=None, on_demand=None, listed=None,
              queue=("j1",), bid_put=True):
    # MIGRATED (was MIGRATION-BLOCKED then MIGRATION-DEFERRED, step 6e batch B3):
    # `_sticky_on_demand` landed at `vastlib.market.pricing` and the tick reaches
    # it as `pricing._sticky_on_demand`, so nothing this group drives raises. Each
    # seam is stubbed at the module `job_lane.job_supervise_tick` RESOLVES it
    # through, which is not always its home: `_instances_soft`/`_put_bid_soft` are
    # read as `lifecycle.<name>`, the market reads as `pricing.<name>`,
    # `_job_handoff_reconcile` as `handoff.<name>`, `_ckpt_watchdog_alarm` as
    # `risk.<name>`, `_job_handoff_emit` as `journal.<name>`, the two replacement
    # rungs as `replacement.<name>`, and `_box_lifecycle_soft`/`_job_sup_reattach`
    # bare in `job_lane` itself. `job_lane._box_lifecycle_soft` is still a raising
    # SEAM stub (its body lives at `vastlib.jobs.view`); this fixture stubs it, as
    # it always did, so the tick never reaches the raise.
    puts = []
    monkeypatch.setattr(lifecycle, "_instances_soft", lambda: ([inst] if inst else []))
    monkeypatch.setattr(handoff, "_job_handoff_reconcile", lambda jc, hf: None)
    monkeypatch.setattr(job_lane, "_box_lifecycle_soft",
                        lambda iid: {"parked": False, "drained_pending": False})
    monkeypatch.setattr(jobmeta, "list_queue", lambda iid: list(queue))
    monkeypatch.setattr(jobmeta, "read_job",
                        lambda j, **kw: {"job_id": j, "display_status": "running",
                                         "status": "running"})
    monkeypatch.setattr(pricing, "_market_min_bid_soft", lambda m, n=None: market)
    # `raising=False` on the two evidence probes ON PURPOSE: it keeps this
    # harness runnable against the PRE-FIX herdd, so the fail-first check
    # (`git checkout <pre-sha> -- tools/vast/*.py` + rerun) exercises the old
    # BEHAVIOR and fails on the assertion, instead of dying in the fixture with
    # an AttributeError that proves only that a new symbol is new.
    monkeypatch.setattr(pricing, "_market_min_bid_read", lambda m, n=None:
                        _MarketRead(listed is not None, bool(listed), market),
                        raising=False)
    monkeypatch.setattr(pricing, "_market_bid_listed_soft", lambda m, n=None: listed,
                        raising=False)
    monkeypatch.setattr(pricing, "_market_ondemand_soft", lambda m, n=None: on_demand)
    monkeypatch.setattr(
        lifecycle, "_put_bid_soft",
        lambda iid, p: (puts.append((str(iid), p)), (bid_put, None))[1])
    monkeypatch.setattr(job_lane, "_job_sup_reattach", lambda jc, iid: None)
    monkeypatch.setattr(jobs_risk, "_ckpt_watchdog_alarm", lambda vw, now: None)
    monkeypatch.setattr(journal, "_job_handoff_emit", lambda jc, ev, **kw: None)
    monkeypatch.setattr(replacement, "_job_rebid_ladder",
                        lambda *a_, **k_: False)      # tested separately
    monkeypatch.setattr(replacement, "_job_eviction_replace",
                        lambda jc, hf, ecls, why, exclusion_class=None: False)
    return puts


def _ladder_events(jc):
    return [(ev, f) for ev, f in (jc.get("ladder_journal") or [])]


def test_the_eviction_is_journaled_on_the_first_not_live_tick(monkeypatch):
    """THE fourteen minutes. On 2026-08-08 the ladder DID fall through to the
    rescue path at 23:03:12Z and printed `treating as OUTBID` seventeen times —
    but every one of those was a bare print(). `fleet log` carried nothing but
    `tick` until 23:17:16Z, so the eviction was found by hand, by polling job
    status, and that hand-rescue then collided with fleetd's own.

    The event now lands on the FIRST tick the box is actually down, carries the
    class, and carries the evidence the class was made on.

    The watch ticks LIVE first, as it had for 54 minutes before 23:03Z. That is
    not scene-setting: `claimed_work` reads the PREVIOUS tick's folded views (the
    same "last tick's state" rule the handoff fence runs on), because this branch
    sits above the queue read and hoisting a B2 round trip in front of every stop
    classification of every box is not worth it. A watch that has never ticked
    live therefore cannot see its own tickets — noted in FLEETD_INCIDENT and
    harmless under fleetd, where a genuine park is caught by the intent journal
    long before this."""
    inst = _inst(dph_base=BOX_A["standing_bid"], status="running",
                 intended="running")
    _tick_env(monkeypatch, inst, market=None, listed=False,
              on_demand=BOX_A["on_demand"])
    jc, hf = job_lane.job_supervise_init(_args())
    assert job_lane.job_supervise_tick(jc, hf) is None       # live; ticket observed
    assert not _ladder_events(jc)

    inst["actual_status"], inst["intended_status"] = "exited", "stopped"
    assert job_lane.job_supervise_tick(jc, hf) is None

    evicted = [f for ev, f in _ladder_events(jc) if ev == "jobs_box_evicted"]
    assert len(evicted) == 1, f"expected ONE eviction event, got {evicted}"
    f = evicted[0]
    assert f["eviction_class"] == bp.EVICTION_OUTBID
    assert f["iid"] == BOX_A["iid"]
    assert f["intended_status"] == "stopped"
    assert f["standing_bid"] == BOX_A["standing_bid"]
    assert f["market_read_ok"] is True and f["market_listed"] is False
    assert f["claimed_work"] is True and f["pending_jobs"] == ["j1"]
    assert f["budget_usd"] == 5.0


def test_the_eviction_event_is_latched_not_repeated(monkeypatch):
    """Seventeen not-live ticks are ONE eviction. A per-tick event would bury
    `fleet log` under exactly the noise that makes an operator stop reading it,
    which is the failure mode this whole change exists to fix."""
    _tick_env(monkeypatch, _inst(dph_base=BOX_A["standing_bid"]),
              market=None, listed=False, on_demand=BOX_A["on_demand"])
    jc, hf = job_lane.job_supervise_init(_args())
    for _ in range(6):
        job_lane.job_supervise_tick(jc, hf)
    assert len([1 for ev, _f in _ladder_events(jc)
                if ev == "jobs_box_evicted"]) == 1


def test_a_box_that_comes_back_retracts_its_eviction(monkeypatch):
    """A rescued box is a resolved condition, and the log has to say so — the
    re-bid ladder exists precisely so that "EVICTED" is often not the end of the
    story. It also re-arms: the NEXT displacement is a new event."""
    inst = _inst(dph_base=BOX_A["standing_bid"])
    _tick_env(monkeypatch, inst, market=None, listed=False,
              on_demand=BOX_A["on_demand"])
    jc, hf = job_lane.job_supervise_init(_args())
    job_lane.job_supervise_tick(jc, hf)
    assert jc.get("evicted_announced") == BOX_A["iid"]

    inst["actual_status"], inst["intended_status"] = "running", "running"
    job_lane.job_supervise_tick(jc, hf)
    assert jc.get("evicted_announced") is None
    assert [ev for ev, _f in _ladder_events(jc)] \
        .count("jobs_box_eviction_survived") == 1


def test_an_operator_park_never_produces_an_eviction_event(monkeypatch):
    """The guard rail on the whole change. `fleet park` / `herdd stop` /
    `reap` record intent BEFORE the vast PUT; under fleetd such a watch never
    reaches this ladder at all, and when a driver says so explicitly the ladder
    exits clean and journals nothing."""
    _tick_env(monkeypatch, _inst(dph_base=BOX_A["standing_bid"]),
              market=None, listed=False, on_demand=BOX_A["on_demand"])
    jc, hf = job_lane.job_supervise_init(_args())
    jc["stop_intent"] = True
    assert job_lane.job_supervise_tick(jc, hf) == "operator_park"
    assert not _ladder_events(jc)


# --------------------------------------------------------------------------- #
# 2. #78 — the ladder's money moves reach `fleet log`
# --------------------------------------------------------------------------- #
def test_fleetd_drains_the_ladder_journal_under_the_events_own_name():
    """The seam. `_job_handoff_journal` is hard-prefixed `jobs_handoff_` by
    fleetd's drain — right for a migration, a lie for a pull-reschedule — so the
    ladder gets a second queue whose names survive verbatim, plus a per-event
    `iid` so an event about the CONDEMNED box is not filed under the
    replacement's id."""
    f = fleetd.Fleet.__new__(fleetd.Fleet)
    rows = []
    f.journal = lambda ev, **kw: rows.append((ev, kw))
    jc = {}
    journal._job_ladder_journal(jc, "jobs_box_condemned", iid="47219058", dph=1.909)
    journal._job_ladder_journal(jc, "jobs_box_launched", dph=2.1)

    w = {"iid": "47219872"}
    for ev, fields in jc.pop("ladder_journal"):
        fields = dict(fields)
        f.journal(ev, iid=str(fields.pop("iid", None) or w["iid"]),
                  target="47214941", **fields)
    assert rows[0][0] == "jobs_box_condemned"
    assert rows[0][1]["iid"] == "47219058"       # the box the event is ABOUT
    assert rows[0][1]["target"] == "47214941"    # the watch identity
    assert rows[1][1]["iid"] == "47219872"       # defaulted to the watch's box


def _pull_env(monkeypatch, *, launch=(47219872, 1.909, None),
              retarget=(["j1"], []), destroy_fail=None):
    calls = []
    monkeypatch.setattr(journal, "_job_handoff_emit",
                        lambda jc, ev, **kw: calls.append(("emit", ev)))
    monkeypatch.setattr(
        replacement, "_launch_job_replacement",
        lambda jc, excl, offer=None, rental="bid", price=None, max_dph=None: (
            calls.append(("launch", max_dph)), launch)[1])
    monkeypatch.setattr(
        replacement, "_retarget_pending_tickets",
        lambda old, new, reason="pull_condemned": (
            calls.append(("retarget", old, new)), retarget)[1])
    monkeypatch.setattr(
        lifecycle, "_destroy_and_revoke",
        lambda ids, ins, intent, noun="": (
            calls.append(("destroy", list(ids), intent)), destroy_fail or [])[1])
    return calls


def test_the_pull_reschedule_chain_is_visible_in_fleet_log(monkeypatch):
    """23:27:56Z. fleetd condemned 47219058 for a stalled image pull, launched
    47219872, moved the ticket and destroyed the condemned box — and `herdd
    fleet log` had ZERO events for either id. Grepping it for `replac|
    jobs_replaced|eviction` returned nothing while a human was hand-rescuing the
    same job from the other side.

    Every money-moving step now carries the prices and the watch identity."""
    jc, hf = job_lane.job_supervise_init(_args())
    jc["iid"] = "47219058"
    jc["launch_dph_anchor"] = 1.909
    jc["instances"] = []
    _pull_env(monkeypatch)
    inst = _inst(iid="47219058", status="loading", machine=45768,
                 intended="running")
    assert replacement._job_pull_condemn(jc, inst, "deadline") is None

    got = {ev: f for ev, f in _ladder_events(jc)}
    assert set(got) >= {"jobs_box_condemned", "jobs_box_launched",
                        "jobs_queue_retargeted", "jobs_box_destroyed"}, \
        f"the chain is still invisible: {sorted(got)}"
    assert got["jobs_box_condemned"]["iid"] == "47219058"
    assert got["jobs_box_condemned"]["machine_id"] == 45768
    assert got["jobs_box_launched"]["iid"] == "47219872"
    assert got["jobs_box_launched"]["from_box"] == "47219058"
    assert got["jobs_box_launched"]["dph"] == 1.909      # THE PRICE
    assert got["jobs_queue_retargeted"]["moved_jobs"] == 1
    assert got["jobs_box_destroyed"]["iid"] == "47219058"
    assert got["jobs_box_destroyed"]["ok"] is True
    assert got["jobs_box_destroyed"]["actor"] == "fleetd:pull-watchdog"


def test_a_refused_replacement_launch_is_journaled_too(monkeypatch):
    """A rung that DIDN'T spend is as much a part of the audit trail as one that
    did — "why is this box still condemned" has to be answerable from the log."""
    jc, hf = job_lane.job_supervise_init(_args())
    jc["iid"] = "47219058"
    jc["launch_dph_anchor"] = 1.909
    jc["instances"] = []
    jc["last_error"] = "no offers under $3.818"
    _pull_env(monkeypatch, launch=(None, None, "no_offer"))
    assert replacement._job_pull_condemn(jc, _inst(iid="47219058", status="loading",
                                         intended="running"), "slow") is None
    got = {ev: f for ev, f in _ladder_events(jc)}
    assert got["jobs_box_launch_failed"]["reason"] == "no_offer"
    assert "jobs_box_destroyed" not in got, \
        "a failed launch must leave the condemned box and its queue alone"


def test_the_rebid_rung_that_walked_the_bid_to_3216_is_journaled(monkeypatch):
    """23:06:25Z and 23:11:28Z: the re-bid ladder raised 47214941's standing bid
    $2.134 -> $2.667 -> $3.216 against `floor $None`. Raising a standing bid is
    moving money, and `fleet log` recorded none of it."""
    puts = []
    monkeypatch.setattr(lifecycle, "_put_bid_soft",
                        lambda iid, p: (puts.append(p), (True, None))[1])
    monkeypatch.setattr(journal, "_job_handoff_emit", lambda jc, ev, **kw: None)
    jc, hf = job_lane.job_supervise_init(_args())
    jc["iid"] = BOX_A["iid"]
    jc["last_bid"] = BOX_A["rebid"][0]
    jc["launch_dph_anchor"] = BOX_A["rebid"][0]
    jc["max_bid"] = 3.216
    assert replacement._job_rebid_ladder(jc, jc["a"], BOX_A["iid"], None,
                               BOX_A["on_demand"], bp.EVICTION_OUTBID, NOW)
    rung = [f for ev, f in _ladder_events(jc) if ev == "jobs_rebid_rung"]
    assert len(rung) == 1
    assert rung[0]["old_bid"] == BOX_A["rebid"][0]
    assert rung[0]["new_bid"] == puts[0]
    assert rung[0]["eviction_class"] == bp.EVICTION_OUTBID


# --------------------------------------------------------------------------- #
# 3. #73 — the self-referential floor
# --------------------------------------------------------------------------- #
def test_market_floor_is_self_on_the_incidents_own_reads():
    """Every floor the two ladders were handed was a number we had just PUT."""
    for bid in BOX_A["ladder"][1:]:              # 2.818, 3.100, 3.410
        assert bp.market_floor_is_self(bid, bid), \
            f"floor ${bid} == our own bid ${bid} must read as SELF"
    assert bp.market_floor_is_self(BOX_B["launch_bid"], BOX_B["launch_bid"])
    assert bp.market_floor_is_self(BOX_B["defended"][0], BOX_B["defended"][0])


def test_market_floor_is_self_never_swallows_a_real_competing_bidder():
    """The dangerous direction. A floor strictly ABOVE our bid is the genuine
    outbid signal `classify_eviction` keys on (`mmb > lb`), and suppressing it
    would trade a money bug for an availability bug — the box would die rather
    than be defended. The incident's own competing read, min_bid $2.81 against
    our standing $2.55, must NOT read as self."""
    assert not bp.market_floor_is_self(BOX_A["risen_min_bid"],
                                       BOX_A["standing_bid"])
    assert not bp.market_floor_is_self(BOX_A["true_floor"],
                                       BOX_A["standing_bid"])
    # one price-grid step away is still the market, not us
    assert not bp.market_floor_is_self(2.551, 2.55)
    assert not bp.market_floor_is_self(None, 2.55)
    assert not bp.market_floor_is_self(2.55, None)


def test_an_out_of_band_bid_move_is_reconciled_from_the_box(monkeypatch):
    """Review 2026-08-10 (M3): `jc["last_bid"]` was written only by our own
    successful PUTs, so a `herdd bid --price` typed by a human — or a PUT
    vast applied but answered 5xx — left the lane's belief stale with no path
    back. The guard's standing arm then compared the echo against a price we
    no longer hold, and the covering `prior` entry silently aged out of the
    echo window while the price still stood (the H1 mechanism). The belief now
    reconciles to the observed dph_base whenever no PUT of ours is in flight."""
    inst = _inst(status="running", intended="running", dph_base=0.50,
                 dph_total=0.55)
    _tick_env(monkeypatch, inst, market=None, listed=True, on_demand=3.0)
    jc, hf = job_lane.job_supervise_init(_args())
    job_lane.job_supervise_tick(jc, hf)
    assert jc["last_bid"] == 0.50            # seeded from dph_base, not total
    inst["dph_base"] = 0.60                  # a human moves the bid via CLI
    job_lane.job_supervise_tick(jc, hf)
    assert jc["last_bid"] == 0.60, "the lane's belief must follow the box"
    # ...and the CLI price's echo now reads as OURS through the standing arm
    assert bp.market_floor_self_match(
        0.60, jc["last_bid"],
        bid_history=pricing._bid_history_for(jc, inst["machine_id"]),
        now=jc["t_prev"]) is not None
    # but a price we JUST PUT is not clobbered by a body fetched pre-PUT
    jc["last_bid"], jc["last_bid_put"] = 0.72, jc["t_prev"]
    job_lane.job_supervise_tick(jc, hf)
    assert jc["last_bid"] == 0.72, "an in-flight PUT wins over a stale body"


def test_a_self_referential_floor_does_not_move_the_bid(monkeypatch):
    """THE money bug, replayed on box 47218938's numbers. Live bid box, standing
    bid $1.338, and the offers read hands back $1.338 — the price to displace
    OURSELVES. `_bid_target` would answer 2.00 x 1.338 = $2.676 and the ladder
    would PUT it, which is exactly what happened at 23:22:17Z, eleven seconds
    after the watch was registered."""
    inst = _inst(iid=BOX_B["iid"], status="running", machine=BOX_B["machine"],
                 intended="running",
                 dph_total=BOX_B["dph_total_at_adoption"],
                 dph_base=BOX_B["launch_bid"])
    puts = _tick_env(monkeypatch, inst, market=BOX_B["launch_bid"],
                     listed=True, on_demand=4.258)
    jc, hf = job_lane.job_supervise_init(_args(id=int(BOX_B["iid"])))
    for _ in range(4):
        job_lane.job_supervise_tick(jc, hf)

    assert puts == [], f"the ladder chased its own bid: PUT {puts}"
    assert jc["last_bid"] == BOX_B["launch_bid"]
    assert BOX_B["defended"][0] not in [p for _iid, p in puts]
    selfy = [f for ev, f in _ladder_events(jc) if ev == "jobs_bid_self_floor"]
    assert len(selfy) == 1, "say it once, in the journal, with both numbers"
    assert selfy[0]["market_min_bid"] == BOX_B["launch_bid"]
    assert selfy[0]["standing_bid"] == BOX_B["launch_bid"]
    # ...and the self-read never enters the median-floor fallback for max_bid,
    # or the ceiling ratchets along with the bid.
    assert BOX_B["launch_bid"] not in jc["floor_samples"]


def test_a_real_floor_still_gets_defended(monkeypatch):
    """The suppression is EXACT-equality and tenant-gated, so a genuine market
    move on the same box still moves the bid. Without this the fix would be a
    silent disarming of the defend ladder."""
    inst = _inst(iid=BOX_B["iid"], status="running", machine=BOX_B["machine"],
                 intended="running", dph_total=1.4752, dph_base=1.338)
    puts = _tick_env(monkeypatch, inst, market=1.60, listed=True,
                     on_demand=4.258)
    jc, hf = job_lane.job_supervise_init(_args(id=int(BOX_B["iid"])))
    job_lane.job_supervise_tick(jc, hf)
    assert puts, "a real floor above our bid must still be defended"
    assert puts[0][1] > 1.338


def test_the_self_floor_guard_is_tenant_gated(monkeypatch):
    """On a STOPPED box the same equality means the opposite thing: somebody
    else holds the chunk now, at a price that happens to match what we were
    paying. That is a real market read and the rescue ladder's whole input, so
    the guard must not fire there."""
    inst = _inst(iid=BOX_B["iid"], status="exited", machine=BOX_B["machine"],
                 intended="stopped", dph_total=1.4752, dph_base=1.338)
    _tick_env(monkeypatch, inst, market=1.338, listed=True, on_demand=4.258)
    jc, hf = job_lane.job_supervise_init(_args(id=int(BOX_B["iid"])))
    job_lane.job_supervise_tick(jc, hf)
    assert not [1 for ev, _f in _ladder_events(jc) if ev == "jobs_bid_self_floor"]
    assert jc["floor_samples"] == [1.338]


# --------------------------------------------------------------------------- #
# 4. #76 — the $1.338 launch that billed $2.9/hr
# --------------------------------------------------------------------------- #
def test_the_standing_bid_is_dph_base_not_dph_total():
    """`dph_total` is the bid PLUS storage. Verified against the incident box's
    own snapshot and against two live instances read 2026-08-09."""
    assert models._instance_standing_bid(
        {"dph_base": BOX_B["dph_base_after"],
         "dph_total": BOX_B["dph_total_after"]}) == BOX_B["dph_base_after"]
    assert round(BOX_B["dph_base_after"] + BOX_B["disk_hour"], 6) == \
        round(BOX_B["dph_total_after"], 6), "dph_total == dph_base + diskHour"
    assert models._instance_standing_bid({"dph_total": 3.08}) is None
    assert models._instance_standing_bid(None) is None


def test_the_ladder_seeds_last_bid_from_the_standing_bid(monkeypatch):
    """Why the storage sliver is load-bearing rather than cosmetic. Seeded from
    `dph_total`, `jc["last_bid"]` is $0.137 ABOVE the number vast reports back as
    the chunk's `min_bid` — so `market_floor_is_self`, which is an exact-equality
    test by design, cannot recognise our own bid, and the ratchet walks straight
    through the guard. That is precisely the 22:10:07Z step: bid 2.697
    (dph_total) against floor 2.562 (dph_base), read as a market $0.135 below us
    and "defended" to $2.818."""
    inst = _inst(iid=BOX_B["iid"], status="running", machine=BOX_B["machine"],
                 intended="running",
                 dph_total=BOX_B["dph_total_at_adoption"],
                 dph_base=BOX_B["launch_bid"])
    _tick_env(monkeypatch, inst, market=None, listed=True, on_demand=4.258)
    jc, hf = job_lane.job_supervise_init(_args(id=int(BOX_B["iid"])))
    job_lane.job_supervise_tick(jc, hf)
    assert jc["last_bid"] == BOX_B["launch_bid"]
    assert jc["last_bid"] != BOX_B["dph_total_at_adoption"]
    assert jc["first_seen_dph"] == BOX_B["launch_bid"]
    # the BILLED rate is still dph_total — the replacement ceiling prices the
    # invoice, not the bid, and moving that anchor would silently shrink it.
    assert jc["launch_dph_anchor"] == BOX_B["dph_total_at_adoption"]


def test_the_full_76_ratchet_is_dead(monkeypatch):
    """End to end on the real sequence. Launch PUT $1.338 (correct, and the
    printed formula matched what was sent — #76 is not a launch defect). Then
    the tick reads dph_total 1.4752 as our bid, reads the chunk floor as 1.338,
    sees a market "below" us, and defends to 2.00 x 1.338 = 2.676; the next poll
    reads 2.676 back and defends to 2.944. Six minutes, $1.47/hr -> $2.94/hr, on
    a machine whose instance body said min_bid 1.333333 the whole time."""
    inst = _inst(iid=BOX_B["iid"], status="running", machine=BOX_B["machine"],
                 intended="running",
                 dph_total=BOX_B["dph_total_at_adoption"],
                 dph_base=BOX_B["launch_bid"])
    puts = _tick_env(monkeypatch, inst, market=BOX_B["launch_bid"], listed=True,
                     on_demand=4.258)

    jc, hf = job_lane.job_supervise_init(_args(id=int(BOX_B["iid"])))
    for _ in range(3):
        job_lane.job_supervise_tick(jc, hf)
        # a real driver would see the new bid come back as the next floor
        if puts:
            inst["dph_base"] = puts[-1][1]

    priced = [p for _iid, p in puts]
    for step in BOX_B["defended"]:
        assert step not in priced, f"the ratchet still reaches ${step}"
    assert not priced, f"nothing should have been PUT at all, got {priced}"


# --------------------------------------------------------------------------- #
# 5. #74, second instance — a stop with NO price cause, and rung ZERO
# --------------------------------------------------------------------------- #
def test_a_stop_with_our_bid_still_winning_is_host_stop_not_ondemand():
    """Box 47226953. `min_bid` $0.3333 against our standing $0.667 and the chunk
    still listed `avail: yes` — nobody outbid us and no on-demand renter can be
    sitting on GPUs that are still purchasable as spot. The pre-2026-08-09 code
    answered `ondemand_displaced` here, purely because an on-demand PRICE
    existed, and `ondemand_displaced` is the one class the re-bid ladder refuses
    by name — so the next rung was renting a cold replacement for a box that was
    one `start` away."""
    assert bp.classify_eviction(
        present=True, actual_status="exited",
        market_min_bid=BOX_C["min_bid"], market_listed=True,
        on_demand=BOX_C["on_demand"], last_bid=BOX_C["standing_bid"],
    ) == bp.EVICTION_HOST_STOP
    # with no listing evidence the old, weaker answer is unchanged
    assert bp.classify_eviction(
        present=True, actual_status="exited",
        market_min_bid=BOX_C["min_bid"],
        on_demand=BOX_C["on_demand"], last_bid=BOX_C["standing_bid"],
    ) == bp.EVICTION_ONDEMAND


def test_resume_in_place_is_the_first_rung_when_our_bid_still_clears():
    """The pure decision, on 47226953's numbers."""
    d = bp.resume_in_place(present=True, is_bid=True,
                           market_min_bid=BOX_C["min_bid"], market_listed=True,
                           last_bid=BOX_C["standing_bid"],
                           budget_usd=BOX_C["budget"], spend_usd=BOX_C["spend"])
    assert d.action == "start"
    assert "clears the live floor" in d.reason


def test_resume_in_place_refuses_every_case_a_start_cannot_fix():
    """Rung zero must never mask a real displacement — each refusal hands the
    box straight to the bid rungs that CAN act on it."""
    base = dict(present=True, is_bid=True, market_listed=True,
                market_min_bid=BOX_C["min_bid"], last_bid=BOX_C["standing_bid"],
                budget_usd=BOX_C["budget"], spend_usd=BOX_C["spend"])
    # gone from the listing -> host failure, the replacement rung's job
    assert bp.resume_in_place(**{**base, "present": False}).action == "skip"
    # the machine lists nothing: 47214941's shape, a real displacement
    assert bp.resume_in_place(**{**base, "market_listed": False,
                                 "market_min_bid": None}).action == "skip"
    # outbid: the answer is a higher bid, not a start vast will refuse
    assert bp.resume_in_place(**{**base, "market_min_bid": 2.81,
                                 "last_bid": 2.55}).action == "skip"
    # no usable market read: ignorance never licenses a move
    assert bp.resume_in_place(**{**base, "market_min_bid": None,
                                 "market_listed": None}).action == "skip"
    # budget consumed — a resume restarts the meter
    assert bp.resume_in_place(**{**base, "spend_usd": 3.0}).action == "skip"
    # bounded per eviction cycle
    assert bp.resume_in_place(**{**base, "tries_used": 2}).action == "skip"
    # an ON-DEMAND box has no bid to lose: always start first
    assert bp.resume_in_place(present=True, is_bid=False,
                              budget_usd=3.0, spend_usd=0.1).action == "start"


def test_the_ladder_starts_the_warm_box_before_it_rents_anything(monkeypatch):
    """End to end on 47226953. The box is stopped under a live claimed ticket
    with our bid still winning; the ladder must issue a `start` on the box we
    already rent and must NOT reach the replacement rung."""
    states, rented = [], []
    inst = _inst(iid=BOX_C["iid"], status="running", machine=BOX_C["machine"],
                 intended="running", dph_total=BOX_C["dph_total"],
                 dph_base=BOX_C["standing_bid"])
    _tick_env(monkeypatch, inst, market=BOX_C["min_bid"], listed=True,
              on_demand=BOX_C["on_demand"])
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda iid, st: (states.append((str(iid), st)),
                                         (True, None))[1])
    monkeypatch.setattr(
        replacement, "_job_eviction_replace",
        lambda jc, hf, ecls, why, exclusion_class=None: (
            rented.append((ecls, why)), False)[1])

    jc, hf = job_lane.job_supervise_init(_args(id=int(BOX_C["iid"]),
                                        budget=BOX_C["budget"]))
    assert job_lane.job_supervise_tick(jc, hf) is None       # live: ticket observed
    inst["actual_status"], inst["intended_status"] = "exited", "stopped"
    for _ in range(bp.NOT_LIVE_DEBOUNCE + 1):
        assert job_lane.job_supervise_tick(jc, hf) is None

    assert states == [(BOX_C["iid"], "running")], \
        f"expected exactly one in-place start, got {states}"
    assert not rented, f"rung zero must pre-empt the replacement rung: {rented}"
    got = {ev: f for ev, f in _ladder_events(jc)}
    assert "jobs_box_evicted" in got, "the stop is still journaled as a decision"
    assert got["jobs_box_evicted"]["eviction_class"] == bp.EVICTION_HOST_STOP
    assert got["jobs_box_evicted"]["claimed_work"] is True
    assert got["jobs_box_resumed"]["ok"] is True
    assert got["jobs_box_resumed"]["standing_bid"] == BOX_C["standing_bid"]
    assert got["jobs_box_resumed"]["market_min_bid"] == BOX_C["min_bid"]


def test_the_displaced_box_does_not_get_a_pointless_start(monkeypatch):
    """The other half: 47214941's shape (machine lists nothing) must skip rung
    zero entirely and go on to the bid rungs, or the fix would have traded a
    missing recovery for a stalled one."""
    states = []
    inst = _inst(dph_base=BOX_A["standing_bid"], status="running",
                 intended="running")
    _tick_env(monkeypatch, inst, market=None, listed=False,
              on_demand=BOX_A["on_demand"])
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda iid, st: (states.append((str(iid), st)),
                                         (True, None))[1])
    jc, hf = job_lane.job_supervise_init(_args())
    job_lane.job_supervise_tick(jc, hf)
    inst["actual_status"], inst["intended_status"] = "exited", "stopped"
    for _ in range(2 * bp.NOT_LIVE_DEBOUNCE + 2):
        job_lane.job_supervise_tick(jc, hf)
    assert states == [], f"a displaced box must not be `start`ed: {states}"


def test_a_refused_start_falls_through_to_the_bid_rungs(monkeypatch):
    """`herdd start` is refused while another renter holds the GPUs. That
    refusal is information, not a dead end — it is journaled and the ladder
    carries on."""
    inst = _inst(iid=BOX_C["iid"], status="running", machine=BOX_C["machine"],
                 intended="running", dph_total=BOX_C["dph_total"],
                 dph_base=BOX_C["standing_bid"])
    _tick_env(monkeypatch, inst, market=BOX_C["min_bid"], listed=True,
              on_demand=BOX_C["on_demand"])
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda iid, st: (False, "no_such_instance/gpus busy"))
    jc, hf = job_lane.job_supervise_init(_args(id=int(BOX_C["iid"]),
                                        budget=BOX_C["budget"]))
    job_lane.job_supervise_tick(jc, hf)
    inst["actual_status"], inst["intended_status"] = "exited", "stopped"
    for _ in range(bp.NOT_LIVE_DEBOUNCE + 1):
        job_lane.job_supervise_tick(jc, hf)
    got = {ev: f for ev, f in _ladder_events(jc)}
    assert got["jobs_box_resume_failed"]["ok"] is False
    assert "busy" in got["jobs_box_resume_failed"]["error"]
    assert int(jc["resume_tries"]) <= bp.RESUME_MAX_TRIES


# --------------------------------------------------------------------------- #
# 6. ONE MarketRead per tick
#
# An eviction tick used to issue THREE independent offers queries against the
# same machine: the per-tick floor (`_market_min_bid_soft`), the not-live listed
# probe (`_market_bid_listed_soft`, which computes a `min_bid` and discards it),
# and the announcement's own `_market_min_bid_read`. Three reads of a moving
# market can disagree, and the disagreement reaches a MONEY decision — see the
# pure demonstration below.
# --------------------------------------------------------------------------- #
def test_disagreeing_reads_skip_the_cheapest_rung_on_the_ladder():
    """The consequence, stated purely. `listed=True` from one read next to
    `min_bid=None` from another is a pair the real API cannot produce in a
    single answer — `MarketRead(ok=True, listed=True)` always carries a floor —
    yet the three-read tick could assemble it, and `resume_in_place` refuses it
    (correctly: ignorance never licenses a move). The box stays down and the
    ladder walks on to rungs that spend."""
    consistent = dict(present=True, is_bid=True, market_listed=True,
                      market_min_bid=BOX_C["min_bid"],
                      last_bid=BOX_C["standing_bid"],
                      budget_usd=BOX_C["budget"], spend_usd=BOX_C["spend"])
    assert bp.resume_in_place(**consistent).action == "start"
    torn = {**consistent, "market_min_bid": None}       # two reads, one instant
    assert bp.resume_in_place(**torn).action == "skip"


def _count_reads(monkeypatch, *, market, listed):
    """Wrap `_market_min_bid_read` in a counter and BOOBY-TRAP the two probes it
    replaced, so a second query of any shape fails the test loudly."""
    reads = []

    def _read(mid, g=None):
        reads.append((mid, g))
        return _MarketRead(listed is not None, bool(listed), market)

    # ONE namespace now (step 6e, batch B3 unblocked): every subject this
    # fixture serves — `job_lane._job_market_read` and the two tick tests below
    # — resolves the market reads through `pricing`, so patching it there is
    # complete. It used to be patched in `herdd` as well, because the tick
    # lane was still flat and one target would have left the other live.
    monkeypatch.setattr(pricing, "_market_min_bid_read", _read)

    def _trap(name):
        def _boom(*a_, **k_):
            raise AssertionError(f"{name} is a SECOND market query on this tick")
        return _boom

    for _n in ("_market_bid_listed_soft", "_market_min_bid_soft"):
        monkeypatch.setattr(pricing, _n, _trap(_n))
    return reads


def test_one_market_query_per_tick_on_a_live_box(monkeypatch):
    inst = _inst(iid=BOX_C["iid"], status="running", machine=BOX_C["machine"],
                 intended="running", dph_total=BOX_C["dph_total"],
                 dph_base=BOX_C["standing_bid"])
    _tick_env(monkeypatch, inst, market=BOX_C["min_bid"], listed=True,
              on_demand=BOX_C["on_demand"])
    reads = _count_reads(monkeypatch, market=BOX_C["min_bid"], listed=True)
    jc, hf = job_lane.job_supervise_init(_args(id=int(BOX_C["iid"]),
                                        budget=BOX_C["budget"]))
    assert job_lane.job_supervise_tick(jc, hf) is None
    assert len(reads) == 1, f"one read per tick, got {reads}"
    assert reads[0] == (BOX_C["machine"], 1)          # machine AND chunk size


def test_one_market_query_on_the_eviction_tick_and_it_is_consistent(monkeypatch):
    """THE tick that used to read three times: the box goes down, the
    announcement classifies it, and the not-live path asks whether the machine
    still lists anything. All three now share one answer — so the floor that
    reaches `resume_in_place` is the same floor that proved `listed`."""
    states = []
    inst = _inst(iid=BOX_C["iid"], status="running", machine=BOX_C["machine"],
                 intended="running", dph_total=BOX_C["dph_total"],
                 dph_base=BOX_C["standing_bid"])
    _tick_env(monkeypatch, inst, market=BOX_C["min_bid"], listed=True,
              on_demand=BOX_C["on_demand"])
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda iid, st: (states.append((str(iid), st)),
                                         (True, None))[1])
    reads = _count_reads(monkeypatch, market=BOX_C["min_bid"], listed=True)
    jc, hf = job_lane.job_supervise_init(_args(id=int(BOX_C["iid"]),
                                        budget=BOX_C["budget"]))
    job_lane.job_supervise_tick(jc, hf)                      # live tick
    inst["actual_status"], inst["intended_status"] = "exited", "stopped"
    before = len(reads)
    for _ in range(bp.NOT_LIVE_DEBOUNCE):
        job_lane.job_supervise_tick(jc, hf)
    per_tick = (len(reads) - before) / float(bp.NOT_LIVE_DEBOUNCE)
    assert per_tick == 1.0, f"{per_tick} market queries per eviction tick"
    got = {ev: f for ev, f in _ladder_events(jc)}
    # the announcement and rung zero read the SAME numbers
    assert got["jobs_box_evicted"]["market_min_bid"] == BOX_C["min_bid"]
    assert got["jobs_box_evicted"]["market_listed"] is True
    assert got["jobs_box_resumed"]["market_min_bid"] == BOX_C["min_bid"]
    assert got["jobs_box_resumed"]["market_listed"] is True
    assert states == [(BOX_C["iid"], "running")]


def test_the_per_tick_read_is_not_reused_across_boxes(monkeypatch):
    """The memo is keyed on (machine, chunk, tick) and not merely 'once per
    watch': the boot watchdogs and the replacement rungs can move a watch to a
    DIFFERENT box mid-tick, and a floor read for the old machine must never be
    handed to the new one."""
    jc = {"now": NOW}
    reads = _count_reads(monkeypatch, market=0.5, listed=True)
    a_box = {"machine_id": 111, "num_gpus": 1}
    assert job_lane._job_market_read(jc, a_box).min_bid == 0.5
    assert job_lane._job_market_read(jc, a_box).min_bid == 0.5          # memo hit
    assert len(reads) == 1
    job_lane._job_market_read(jc, {"machine_id": 222, "num_gpus": 1})   # other machine
    job_lane._job_market_read(jc, {"machine_id": 111, "num_gpus": 2})   # other chunk
    jc["now"] = NOW + 45
    job_lane._job_market_read(jc, a_box)                                # next tick
    assert len(reads) == 4
    # no machine == no query, and ignorance, never a floor
    assert job_lane._job_market_read(jc, {}) == models.MarketRead(False, False, None)
    assert len(reads) == 4


# --------------------------------------------------------------------------- #
# RECALIBRATION 2026-08-09, item A — the hard ceiling, at the tick
#
# The self-floor guard above removes the SELF-REFERENTIAL trigger for the
# 47214941 ratchet. The precedence defect underneath it is separate and survives
# on genuine floors: the survival cushion outranks the cost cap, so on a machine
# whose floor is a large fraction of on-demand the cushion set the price and the
# only thing under it was `on_demand - EPS`. `BID_CEILING_ONDEMAND_FRAC` is now a
# HARD clamp (`bidpolicy.effective_bid_ceiling`), and over it the answer is an
# ESCALATION rather than a bigger bid.
#
# These two drive the REAL tick, because the pure half is only half the fix: a
# `None` target is a SILENT no-op on the defend path, and "a money decision
# nobody was told about" is exactly defect #78.
# --------------------------------------------------------------------------- #
def test_a_live_box_over_the_ceiling_holds_its_bid_and_says_so(monkeypatch):
    """Floor $3.10 against an on-demand rate of $3.876 — the machine 47214941
    actually was, priced 80% of list. The cushion wants $3.41 (the bid the ladder
    reached that night); the hard ceiling is 0.75 x 3.876 = $2.907.

    Three things must all hold, and only the first was true before:
      * NO bid is PUT (the standing bid is HELD — never raise into a dominated
        price, and never DECAY into a more displaceable one either);
      * the decision reaches `fleet log`, once, with its arithmetic;
      * it names the escalation, not a failure — the rungs that answer this are
        replacement / on-demand, and an operator has to be able to act before the
        eviction rather than fourteen minutes after it."""
    inst = _inst(iid=BOX_A["iid"], status="running", machine=BOX_A["machine"],
                 intended="running", dph_total=2.6875, dph_base=2.55)
    puts = _tick_env(monkeypatch, inst, market=3.10, listed=True,
                     on_demand=3.876315789473685)
    jc, hf = job_lane.job_supervise_init(_args())
    job_lane.job_supervise_tick(jc, hf)
    assert puts == [], f"a bid was moved on an over-ceiling machine: {puts}"
    assert jc["last_bid"] == 2.55, "the standing bid must be HELD, not lowered"
    got = [f for ev, f in _ladder_events(jc) if ev == "jobs_bid_over_ceiling"]
    assert len(got) == 1, f"expected ONE ceiling escalation, got {got}"
    assert got[0]["ceiling"] == pytest.approx(2.907, abs=1e-3)
    assert got[0]["market_min_bid"] == 3.10
    assert "escalate_over_ceiling" in got[0]["reason"]
    assert "structurally unsafe" in got[0]["reason"]
    # latched: a second tick in the same condition must not re-emit
    job_lane.job_supervise_tick(jc, hf)
    assert len([1 for ev, _ in _ladder_events(jc)
                if ev == "jobs_bid_over_ceiling"]) == 1


def test_an_ordinary_machine_is_defended_exactly_as_before(monkeypatch):
    """The must-not-regress guard, and the reason the rail is cheap: the measured
    floor/on-demand distribution is 0.36-0.53 across 51 real bid records, and the
    escalation frontier is 0.6818. An ordinary machine sees no change at all — it
    is defended, and nothing is journaled."""
    inst = _inst(iid=BOX_B["iid"], status="running", machine=BOX_B["machine"],
                 intended="running", dph_total=1.4752, dph_base=1.338)
    puts = _tick_env(monkeypatch, inst, market=1.60, listed=True,
                     on_demand=4.258)
    jc, hf = job_lane.job_supervise_init(_args(id=int(BOX_B["iid"])))
    job_lane.job_supervise_tick(jc, hf)
    assert puts and puts[0][1] > 1.338
    assert puts[0][1] <= bp.effective_bid_ceiling(4.258)
    assert [ev for ev, _ in _ladder_events(jc) if ev == "jobs_bid_over_ceiling"] == []
