"""The extracted ladder core (`ladder_core.py`), and the lane-parity contract.

`FLEET_REVIEW_2026-08-14.md` item 1: the run/serve lane and the jobs lane drove
the SAME pure `bidpolicy` decisions through TWO hand-written copies of the state
transitions around them, and three of the nine 2026-08-10 review defects plus
two of the three 2026-08-14 defects were fixes that landed in one copy and had
to be remembered into the other.

This file is the regression bar for the single copy. It asserts two different
kinds of thing, and the difference matters:

  * that the shared state machine BEHAVES the same when driven from either
    lane's adapter — the tests that would fail if a future edit re-forked it;
  * that the places the two lanes genuinely still DIVERGE keep diverging
    exactly as they do today. Those are pins, not endorsements: D1 is
    intentional, D2/D3 look like unfixed parity gaps, and each pin exists so
    that a repair is a deliberate change with a test that flips rather than a
    silent behavior drift inside a refactor. The full list, with the reading of
    each, is AUTOBID_DESIGN.md §"One core, two lanes".

`test_self_floor_lag.py` and `test_ladder_latch_hygiene.py` still own the
guard's policy semantics end-to-end through `herdd`; nothing there was
touched, and the overlap is deliberate — those pin the LANE, these pin the
CORE.
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bidpolicy as bp  # noqa: E402
import ladder_core as lc  # noqa: E402
import herdd as v  # noqa: E402

# The run lane's event sink is patched at its OWNER. Since plan §8 step 6d
# `run_lane`/`handoff` resolve it as `journal._sup_emit` at call time, so a
# patch of the `herdd` re-export steers nothing and the drive below would
# emit for real.
from vastlib.supervise import journal  # noqa: E402

NOW = 1_000_000.0


class _Recorder(lc.LaneHooks):
    """A hook set that records instead of journalling — the third 'lane', and
    the proof that the state machine needs no I/O to run."""

    def __init__(self):
        self.calls = []

    def scaled_read(self, ctx, market):
        self.calls.append(("scaled_read", market))

    def self_floor(self, ctx, *, market_min_bid, match, surviving_floor,
                   visible):
        self.calls.append(("self_floor", market_min_bid, match.kind,
                           surviving_floor))

    def floor_blind(self, ctx, *, since_s):
        self.calls.append(("floor_blind", round(since_s, 1)))

    def episode_end(self, ctx, *, market):
        self.calls.append(("episode_end", market))

    def names(self):
        return [c[0] for c in self.calls]


def _ctx(**kw):
    """A minimal lane state dict — the intersection of `st` and `jc`."""
    ctx = {"last_bid": 0.50, "machine_id": 7, "now": NOW,
           "bid_history": [[NOW - 10, 0.50, 7, NOW - 10]]}
    ctx.update(kw)
    return ctx


# --------------------------------------------------------------------------- #
# 1. the echo window — same recorder, same reader, one copy
# --------------------------------------------------------------------------- #
def test_the_lanes_and_the_core_share_ONE_echo_recorder():
    """`herdd._note_standing_bid` is an ALIAS, not a second implementation.
    An alias is what keeps every existing call site and every
    `herdd.<name>` reference in the suite working; it is also what makes
    'a fix here is a fix' true, so it is worth asserting rather than assuming."""
    assert v._note_standing_bid is lc.note_standing_bid
    assert v._bid_history_for is lc.bid_history_for
    assert v._hist_field is lc.hist_field
    assert v._self_floor_reset is lc.self_floor_reset
    assert v._num_dph is lc.num_dph
    assert v.BID_HISTORY_MAX == lc.BID_HISTORY_MAX


def test_the_echo_window_survives_the_json_round_trip_fleetd_does():
    """`bid_history` is durable state (fleetd REPLACEMENT_STATE_KEYS /
    RUN_STATE_KEYS) and JSON has no tuples, so the recorder must read back its
    own output after a daemon restart."""
    ctx = {}
    lc.note_standing_bid(ctx, 0.20, 7, NOW)
    ctx = json.loads(json.dumps(ctx))
    lc.note_standing_bid(ctx, 0.20, 7, NOW + 60)          # dedupe still works
    assert len(ctx["bid_history"]) == 1
    assert lc.hist_field(ctx["bid_history"][0], 3) == NOW + 60
    assert [e[1] for e in lc.bid_history_for(ctx, 7)] == [0.20]
    assert lc.bid_history_for(ctx, 9) == []               # per MACHINE


# --------------------------------------------------------------------------- #
# 2. the seam resets fire identically from either lane's adapter
# --------------------------------------------------------------------------- #
def test_the_box_swap_seam_clears_the_same_three_things_on_both_lanes():
    """Five call sites, two lanes, one seam. A swap is a relaunch, an eviction
    replacement, a pull reschedule or a handoff promotion — different rungs,
    same fact: the watch points at a different contract, and the episode latch,
    the echo window and the per-MACHINE on-demand clamp are all properties of
    the box we no longer hold."""
    st = _ctx(self_floor_at=[0.5, "standing"], self_floor_since=NOW - 99,
              self_floor_sustained_said=True, on_demand_last=3.21,
              evicted_pending=True)
    jc = _ctx(self_floor_at=[0.5, "standing"], self_floor_since=NOW - 99,
              self_floor_sustained_said=True, on_demand_last=3.21,
              evicted_announced="700")
    for ctx in (st, jc):
        lc.box_swap_reset(ctx)
        assert ctx.get("self_floor_at") is None
        assert "self_floor_since" not in ctx
        assert "self_floor_sustained_said" not in ctx
        assert ctx["bid_history"] == []
        assert "on_demand_last" not in ctx
    # ...and it touches NOTHING else: the lane-specific latches beside it are
    # each lane's own business and stay at the call site.
    assert st["evicted_pending"] is True
    assert jc["evicted_announced"] == "700"


def test_an_eviction_ends_the_self_floor_episode_on_both_lanes():
    """The fix that had to be made twice (merge 168d2d0f). The guard is
    tenant-gated, so nothing else clears the clock while the box sits stopped:
    without this reset 47398836's floor-blind alarm fired ONE tick after
    rescue_recovered off a `self_floor_since` frozen across a 67-minute stopped
    gap, and the "30 min continuous" mostly measured a box we did not hold."""
    for ctx in (_ctx(self_floor_since=NOW - 4000,
                     self_floor_sustained_said=True,
                     self_floor_at=[0.5, "standing"]),
                _ctx(self_floor_since=NOW - 4000,
                     self_floor_sustained_said=True,
                     self_floor_at=[0.5, "prior"])):
        lc.self_floor_reset(ctx)
        assert ctx.get("self_floor_since") is None
        assert ctx.get("self_floor_sustained_said") is None
        assert ctx.get("self_floor_at") is None
        # the echo window is NOT an episode: an eviction does not invalidate
        # the record of what this chunk's bid has been (only a box swap does)
        assert ctx["bid_history"]


# --------------------------------------------------------------------------- #
# 3. the guard's episode bookkeeping, driven with no lane at all
# --------------------------------------------------------------------------- #
def test_a_prior_echo_is_suppressed_and_journalled_once_per_value_and_kind():
    """The (value, kind) dedup key. On a machine whose every listed chunk is
    ours the market value never changes across an episode, so a value-only key
    swallowed the standing->prior transition — the one event carrying a real
    `matched_age_s`, which is how the lag window gets sized by measurement (it
    already re-sized it once, 900 s -> 3600 s)."""
    h = _Recorder()
    ctx = _ctx(last_bid=0.0421,
               bid_history=[[NOW - 200, 0.016, 7, NOW - 200]])
    # `machine_id` is the echo window's key and is passed IN, not read off the
    # state dict — a floor read is per machine, and a replacement inherits no
    # echoes (review 2026-08-10, F7).
    guard = lambda: lc.self_floor_guard(ctx, 0.016, tenant=True,
                                        machine_id=7, hooks=h)
    assert guard() is None
    assert h.calls == [("self_floor", 0.016, "prior", None)]
    # same value, same kind, next tick: no second journal entry
    assert guard() is None
    assert len(h.calls) == 1
    # the same state through a daemon restart must still dedupe (json turns
    # the tuple-shaped key into a list, and list != tuple)
    ctx = json.loads(json.dumps(ctx))
    assert guard() is None
    assert len(h.calls) == 1
    # ...and the SAME price on a different machine is not our echo at all
    assert lc.self_floor_guard(ctx, 0.016, tenant=True, machine_id=9,
                               hooks=h) == 0.016


def test_a_failed_read_and_a_not_tenant_tick_do_not_flap_the_episode():
    """Review 2026-08-10 #8/L6. `self_match is None` also covers a FAILED offers
    read and every not-tenant tick; clearing the latch there printed "$None is
    a real competing read" and re-journalled a phantom episode start into the
    matched_age_s distribution on the next match."""
    h = _Recorder()
    ctx = _ctx(last_bid=0.032, bid_history=[[NOW - 10, 0.032, 7, NOW - 10]])
    assert lc.self_floor_guard(ctx, 0.032, tenant=True, hooks=h) is None
    assert lc.self_floor_guard(ctx, None, tenant=True, hooks=h) is None
    assert ctx.get("self_floor_at") is not None
    assert lc.self_floor_guard(ctx, 0.032, tenant=False, hooks=h) == 0.032
    assert ctx.get("self_floor_at") is not None
    assert h.names() == ["self_floor"]
    # a REAL competing read while we are still the tenant DOES end it
    assert lc.self_floor_guard(ctx, 0.050, tenant=True, hooks=h) == 0.050
    assert ctx.get("self_floor_at") is None
    assert h.names() == ["self_floor", "episode_end"]


def test_a_sibling_floor_survives_and_stops_the_floor_blind_clock():
    """Row-level suppression (F3): one offers query can list BOTH our rented
    chunk (the echo) and a free sibling chunk (a genuine floor). Dropping the
    collapsed min() hid a genuine floor that had risen ABOVE our bid for as
    long as our lower echo was listed. And with a sibling present there IS a
    market signal, so the floor-blind clock must not run."""
    h = _Recorder()
    ctx = _ctx(last_bid=0.50, self_floor_since=NOW - 9999,
               self_floor_sustained_said=True)
    assert lc.self_floor_guard(ctx, 0.50, tenant=True, floors=[0.50, 0.90],
                               hooks=h) == 0.90
    assert h.calls == [("self_floor", 0.50, "standing", 0.90)]
    assert "self_floor_since" not in ctx
    assert "self_floor_sustained_said" not in ctx


def test_full_suppression_starts_the_clock_and_alarms_once():
    """Every listed chunk is ours: defend AND decay are parked, which is
    correct but must not be silent (review 2026-08-10, #6). Once per episode."""
    h = _Recorder()
    ctx = _ctx(last_bid=0.50)
    assert lc.self_floor_guard(ctx, 0.50, tenant=True, hooks=h) is None
    assert ctx["self_floor_since"] == NOW
    late = NOW + bp.BID_SELF_FLOOR_SUSTAINED_S + 1
    ctx["bid_history"] = [[late - 10, 0.50, 7, late - 10]]
    assert lc.self_floor_guard(ctx, 0.50, tenant=True, now=late, hooks=h) is None
    assert ("floor_blind", round(bp.BID_SELF_FLOOR_SUSTAINED_S + 1.0, 1)) in h.calls
    n = len(h.calls)
    ctx["bid_history"] = [[late, 0.50, 7, late]]
    assert lc.self_floor_guard(ctx, 0.50, tenant=True, now=late + 1,
                               hooks=h) is None
    assert len(h.calls) == n, "the floor-blind alarm is once per EPISODE"


def test_a_rescaled_read_while_tenant_is_a_failed_read():
    """F8: no offer matched our exact GPU count, so the floor is a different
    chunk size's price stretched to ours. While we are the tenant our own chunk
    IS a listing at our count, so its absence means the listing is mid-flap —
    and the rescaled number can never match our history, so it would read as a
    market 1.25-2x above us: a defend trigger by construction."""
    h = _Recorder()
    ctx = _ctx(last_bid=0.50, bid_history=[])
    assert lc.self_floor_guard(ctx, 1.00, tenant=True, floors=[1.00],
                               scaled=True, hooks=h) is None
    assert h.names() == ["scaled_read"]
    assert lc.self_floor_guard(ctx, 1.00, tenant=True, floors=[1.00],
                               scaled=True, hooks=h) is None
    assert h.names() == ["scaled_read"], "said once, latched"
    # ...and passed through on a stopped box: the rescue path keeps the only
    # number it has
    assert lc.self_floor_guard(ctx, 1.00, tenant=False, floors=[1.00],
                               scaled=True, hooks=h) == 1.00
    assert "_scaled_floor_said" not in ctx


def test_the_guard_never_raises_without_hooks():
    """A caller that wants the state machine and no observation surface (a
    probe, a replay, a future third lane) gets silent no-ops, not an
    AttributeError."""
    ctx = _ctx(last_bid=0.50)
    assert lc.self_floor_guard(ctx, 0.50, tenant=True) is None
    assert lc.self_floor_guard(ctx, 0.90, tenant=True) == 0.90


# --------------------------------------------------------------------------- #
# 4. the same inputs through BOTH lanes' real adapters
# --------------------------------------------------------------------------- #
def _drive_run_lane(monkeypatch, market, *, live, floors=None, scaled=False,
                    st=None):
    emits = []
    monkeypatch.setattr(journal, "_sup_emit",
                        lambda rid, ev, **kw: emits.append((ev, kw)))
    st = st if st is not None else _ctx(
        is_bid=True, last_bid=0.50, run_id="r1", instance_id="700")
    out = v._self_floor_guard(st, market, live=live, floors=floors,
                              scaled=scaled)
    return st, out, emits


def _drive_jobs_lane(market, *, tenant, floors=None, scaled=False, jc=None):
    jc = jc if jc is not None else _ctx(last_bid=0.50, ladder_journal=[])
    inst = {"machine_id": 7, "num_gpus": 1}
    out = lc.self_floor_guard(
        jc, market, tenant=tenant, floors=floors, scaled=scaled,
        machine_id=7, now=jc["now"], hooks=v._JobLaneFloorHooks("700", inst),
        clear_latch_by_pop=True)
    return jc, out, jc["ladder_journal"]


def test_both_lanes_reach_the_same_state_on_the_same_market(monkeypatch):
    """The parity test proper. Identical inputs, identical resulting STATE —
    the latch, the clock and the returned floor. Only the observation surface
    differs, and it differs on purpose (the run lane writes the RUN's event
    log, the jobs lane the BOX's), so the event NAMES are asserted apart."""
    st, run_out, emits = _drive_run_lane(monkeypatch, 0.50, live=True)
    jc, job_out, journal = _drive_jobs_lane(0.50, tenant=True)

    assert run_out is job_out is None
    for key in ("self_floor_at", "self_floor_since"):
        assert st.get(key) == jc.get(key), key
    assert st["self_floor_at"] == [0.50, "standing"]

    assert [e for e, _ in emits] == ["bid_self_floor"]
    assert [e for e, _ in journal] == ["jobs_bid_self_floor"]
    for _, f in emits + journal:                 # the shared payload
        assert f["market_min_bid"] == 0.50
        assert f["matched"] == "standing"
        assert f["surviving_floor"] is None


def test_both_lanes_keep_the_sibling_floor_and_both_end_the_episode(monkeypatch):
    st, run_out, _e = _drive_run_lane(monkeypatch, 0.50, live=True,
                                      floors=[0.50, 0.90])
    jc, job_out, _j = _drive_jobs_lane(0.50, tenant=True, floors=[0.50, 0.90])
    assert run_out == job_out == 0.90

    st, run_out, _e = _drive_run_lane(monkeypatch, 0.99, live=True, st=st)
    jc, job_out, _j = _drive_jobs_lane(0.99, tenant=True, jc=jc)
    assert run_out == job_out == 0.99
    assert st.get("self_floor_at") is None and jc.get("self_floor_at") is None


def test_neither_lane_ends_an_episode_on_a_not_tenant_tick(monkeypatch):
    """The tenancy gate is where the two lanes genuinely differ (D1) — but the
    CORE's response to `tenant=False` must be the same on both, or the
    divergence stops being one knob and becomes two behaviors."""
    st, run_out, _e = _drive_run_lane(monkeypatch, 0.50, live=True)
    jc, job_out, _j = _drive_jobs_lane(0.50, tenant=True)
    st2, run_out2, _e2 = _drive_run_lane(monkeypatch, 0.50, live=False, st=st)
    jc2, job_out2, _j2 = _drive_jobs_lane(0.50, tenant=False, jc=jc)
    assert run_out2 == job_out2 == 0.50          # a stopped box: a real market
    assert st2.get("self_floor_at") is not None
    assert jc2.get("self_floor_at") is not None


# --------------------------------------------------------------------------- #
# 5. the standing-bid reconcile — one path, two clock names
# --------------------------------------------------------------------------- #
def test_the_reconcile_seeds_then_corrects_the_lane_belief():
    """Review 2026-08-10 F2/M3, written twice a month apart before this. The
    standing bid can move without a successful PUT of ours (an out-of-band
    `herdd bid`, a PUT vast applied but answered 5xx, a handoff pin, a
    restart), and `last_bid` drives defend_at, the rebid ladder and the guard's
    standing arm."""
    ctx = {"last_bid": None, "first_seen_dph": None}
    got = lc.reconcile_standing_bid(
        ctx, is_bid=True, true_bid=0.20, dph=0.2463, machine_id=7, now=NOW,
        put_ts_key="last_bid_put")
    assert got == 0.20, "the ANCHOR is dph_base, never the billed dph_total"
    assert ctx["last_bid"] == 0.20 and ctx["first_seen_dph"] == 0.20
    assert [e[1] for e in ctx["bid_history"]] == [0.20]

    seen = []
    lc.reconcile_standing_bid(
        ctx, is_bid=True, true_bid=0.30, dph=0.3463, machine_id=7,
        now=NOW + bp.BID_RATE_LIMIT_S + 1, put_ts_key="last_bid_put",
        on_reconcile=lambda old, new: seen.append((old, new)))
    assert seen == [(0.20, 0.30)] and ctx["last_bid"] == 0.30
    assert ctx["first_seen_dph"] == 0.20, "the ceiling anchor is written ONCE"


def test_the_reconcile_will_not_race_our_own_in_flight_PUT():
    """Inside the rate-limit window the box may still be reporting the OLD
    price; correcting the belief to it would undo the rung we just issued."""
    ctx = {"last_bid": 0.30, "first_seen_dph": 0.20, "last_bid_put": NOW}
    seen = []
    lc.reconcile_standing_bid(
        ctx, is_bid=True, true_bid=0.20, dph=0.2463, machine_id=7,
        now=NOW + bp.BID_RATE_LIMIT_S - 1, put_ts_key="last_bid_put",
        on_reconcile=lambda old, new: seen.append((old, new)))
    assert seen == [] and ctx["last_bid"] == 0.30


def test_an_on_demand_box_records_no_bid_and_no_echo():
    """`is_bid=False` is a box that cannot be outbid: writing a price into
    `last_bid` would arm the defend/decay moves against it."""
    ctx = {"last_bid": None, "first_seen_dph": None}
    lc.reconcile_standing_bid(ctx, is_bid=False, true_bid=None, dph=3.21,
                              machine_id=7, now=NOW,
                              put_ts_key="last_bid_put_ts")
    assert ctx["last_bid"] is None and ctx["first_seen_dph"] is None
    assert not ctx.get("bid_history")


# --------------------------------------------------------------------------- #
# 6. the DIVERGENCE pins — current behavior, not endorsed behavior
#    (AUTOBID_DESIGN.md §"One core, two lanes")
# --------------------------------------------------------------------------- #
def test_D1_the_tenancy_gate_is_the_lane_s_to_compute():
    """INTENTIONAL. The run lane passes `_observe`'s `_still_tenant`, which
    tolerates a running->exited->running flap with intended_status still
    `running` (2026-08-10 #3); the jobs lane passes a strict `live and is_bid`
    because it has a resume-in-place rung that consumes the floor while
    not-live. The core must therefore take the verdict, never derive it — this
    pins that it has no opinion of its own."""
    ctx = _ctx(last_bid=0.50, is_bid=True)
    assert lc.self_floor_guard(dict(ctx), 0.50, tenant=True) is None
    assert lc.self_floor_guard(dict(ctx), 0.50, tenant=False) == 0.50


def test_D2_the_run_handoff_seam_keeps_the_stale_sticky_on_demand(monkeypatch):
    """LOOKS LIKE AN UNFIXED PARITY GAP — pinned, not fixed. Four of the five
    box-swap seams clear `on_demand_last` because the sticky on-demand price is
    per MACHINE; the run lane's `_handoff_complete` never has, so the retired
    primary's machine keeps clamping the promoted understudy's rails until the
    next probe re-seeds. Behavior-preserving refactor: recorded and pinned. If
    a future change clears it, THIS TEST SHOULD FLIP, deliberately."""
    monkeypatch.setattr(journal, "_sup_emit", lambda *a, **kw: None)
    st = _ctx(run_id="r1", on_demand_last=3.21, self_floor_since=NOW - 99,
              _instances=[{"id": "u9", "is_bid": True, "dph_base": 0.40}],
              spend_usd=0.0)
    hf = {"understudy_iid": "u9", "understudy_dph": 0.4463,
          "handoff_spend_usd": 0.1, "handoffs_done": 0}
    v._handoff_complete(st, argparse.Namespace(dry_run=True), hf)
    assert st["bid_history"] == []                    # the seam DID run
    assert st.get("self_floor_since") is None
    assert st["on_demand_last"] == 3.21, \
        "unfixed parity gap D2: the run handoff seam keeps the OLD machine's " \
        "sticky on-demand price"
    # ...where the jobs lane's promotion clears it
    jc = _ctx(on_demand_last=3.21)
    lc.box_swap_reset(jc)
    assert "on_demand_last" not in jc


def test_D3_only_the_jobs_lane_warns_about_a_missing_dph_base():
    """LOOKS LIKE AN UNFIXED PARITY GAP, observability only. A bid box whose
    body carries no `dph_base` makes every anchor run one storage sliver off,
    and the self-floor guard's exact-equality standing arm stops matching. The
    jobs lane says so once per watch; the run lane silently falls back to
    `dph_total`. The core fires the hook whenever one is supplied — which lane
    supplies one is the divergence."""
    said = []
    ctx = {"last_bid": None, "first_seen_dph": None}
    lc.reconcile_standing_bid(ctx, is_bid=True, true_bid=None, dph=0.2463,
                              machine_id=7, now=NOW,
                              put_ts_key="last_bid_put",
                              on_missing_base=lambda: said.append(1))
    assert said == [1] and ctx["last_bid"] == 0.2463
    # the run lane passes no hook, and is silent — today's behavior
    ctx2 = {"last_bid": None, "first_seen_dph": None}
    lc.reconcile_standing_bid(ctx2, is_bid=True, true_bid=None, dph=0.2463,
                              machine_id=7, now=NOW,
                              put_ts_key="last_bid_put_ts")
    assert ctx2["last_bid"] == 0.2463


def test_D4_the_two_lanes_name_the_rate_limit_clock_differently():
    """The run lane's key is `last_bid_put_ts`, the jobs lane's is
    `last_bid_put`. Both are persisted (fleetd state.json), so neither can be
    renamed by a refactor — the core takes the name instead."""
    for key in ("last_bid_put_ts", "last_bid_put"):
        ctx = {"last_bid": 0.30, "first_seen_dph": 0.20, key: NOW}
        lc.reconcile_standing_bid(ctx, is_bid=True, true_bid=0.20, dph=0.2463,
                                  machine_id=7,
                                  now=NOW + bp.BID_RATE_LIMIT_S - 1,
                                  put_ts_key=key)
        assert ctx["last_bid"] == 0.30, f"{key} must gate the reconcile"


def test_D5_the_episode_end_clears_the_latch_in_each_lane_s_own_shape():
    """The run lane ASSIGNS `self_floor_at = None`; the jobs lane POPS the key.
    Both read back None, and both dedupe identically — but the state.json shape
    differs, and a refactor does not get to change a persisted shape."""
    run = _ctx(last_bid=0.50, self_floor_at=[0.50, "standing"])
    lc.self_floor_guard(run, 0.90, tenant=True, clear_latch_by_pop=False)
    assert "self_floor_at" in run and run["self_floor_at"] is None

    jobs = _ctx(last_bid=0.50, self_floor_at=[0.50, "standing"])
    lc.self_floor_guard(jobs, 0.90, tenant=True, clear_latch_by_pop=True)
    assert "self_floor_at" not in jobs
