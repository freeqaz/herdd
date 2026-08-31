"""The self-floor guard's LAG WINDOW (measured 2026-08-09, box 47297871).

The 2026-08-08 guard asks one question — "is this floor equal to the bid we
hold RIGHT NOW?" — and that question goes false the instant a defend, decay or
ladder rung moves the bid. The chunk's `min_bid` does not follow the move: it
was measured still reporting our FIRST bid ($0.016) while the standing bid had
walked to $0.0421 over three PUTs. For that whole window the guard sees a stale
echo of our own money and calls it a competing bidder.

Two shapes ratchet through the gap, and both are pinned here as the tests that
would have failed before the fix:

  * decay -> defend: lower to L, read the old B > L, defend fires because
    B >= 0.9 x L, and the target is 1.2 x B — ABOVE the bid decay just left.
  * any sub-11% raise (the 1.10 survival cushion is one): after B -> B', the
    stale B still clears `defend_at x B'`, so the defend re-fires every poll.

`test_a_genuine_competitor_is_still_seen` is the other half: this guard is
suppressive, and a guard that suppresses a real competing bidder trades a money
bug for an availability bug.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bidpolicy as bp  # noqa: E402
from vastlib.market import pricing  # noqa: E402
from vastlib.supervise import journal, run_lane  # noqa: E402

NOW = 1_000_000.0


# --------------------------------------------------------------------------- #
# the pure predicate
# --------------------------------------------------------------------------- #

def test_the_current_bid_still_matches_exactly_as_before():
    m = bp.market_floor_self_match(0.20, 0.20)
    assert m.kind == "standing" and m.price == 0.20
    assert bp.market_floor_is_self(0.20, 0.20) is True
    assert bp.market_floor_is_self(0.21, 0.20) is False


def test_a_bid_we_held_minutes_ago_is_still_ourselves():
    """THE FIX. Floor $0.016 against a standing bid of $0.0421 — the measured
    47297871 shape. The old predicate answered 'competing bidder'."""
    hist = [(NOW - 200, 0.016, 115469), (NOW - 100, 0.0312, 115469)]
    assert bp.market_floor_is_self(0.016, 0.0421) is False, \
        "control: the current-bid-only question is blind here"
    m = bp.market_floor_self_match(0.016, 0.0421, bid_history=hist, now=NOW)
    assert m is not None
    assert (m.kind, m.price) == ("prior", 0.016)
    assert m.age_s == 200


def test_the_window_expires():
    hist = [(NOW - bp.BID_SELF_FLOOR_LAG_S - 1, 0.016, 7)]
    assert bp.market_floor_self_match(0.016, 0.05, bid_history=hist,
                                      now=NOW) is None, \
        "a bid older than the window is history, not an echo"


def test_the_newest_matching_bid_explains_the_echo():
    hist = [(NOW - 400, 0.02, 7), (NOW - 30, 0.02, 7)]
    m = bp.market_floor_self_match(0.02, 0.05, bid_history=hist, now=NOW)
    assert m.age_s == 30


def test_no_clock_matches_on_price_alone():
    """`now=None` is the conservative reading — suppress rather than chase."""
    m = bp.market_floor_self_match(0.016, 0.05,
                                   bid_history=[(0.0, 0.016, 7)], now=None)
    assert m.kind == "prior" and m.age_s is None


def test_the_three_decimal_echo_of_a_four_decimal_bid_still_matches():
    """Measured on probe v2 (2026-08-10, machine 52305): vast stores the bid at
    4 decimals (`dph_base` reads back 0.0336) but the rented chunk's min_bid
    echo is QUANTIZED TO 3 (0.034). The eps (0.0005) is exactly the rounding
    radius — this test is what makes tightening it a red suite."""
    m = bp.market_floor_self_match(0.034, 0.0336)
    assert m is not None and m.kind == "standing"
    hist = [(NOW - 90, 0.0353, 52305)]
    m = bp.market_floor_self_match(0.035, 0.032, bid_history=hist, now=NOW)
    assert m is not None and (m.kind, m.price) == ("prior", 0.0353)
    # worst case of round-to-nearest-0.001 sits exactly ON the eps
    assert bp.market_floor_is_self(0.034, 0.0335) is True


def test_a_genuine_competitor_is_still_seen():
    """One price-grid step above anything of ours is a REAL bidder, and the
    rescue/defend path must still get it."""
    hist = [(NOW - 60, 0.020, 7), (NOW - 30, 0.024, 7)]
    assert bp.market_floor_self_match(0.025, 0.024, bid_history=hist,
                                      now=NOW) is None
    assert bp.market_floor_self_match(0.021, 0.024, bid_history=hist,
                                      now=NOW) is None


def test_garbage_and_missing_inputs_never_raise():
    assert bp.market_floor_self_match(None, 0.2) is None
    assert bp.market_floor_self_match("x", 0.2) is None
    assert bp.market_floor_self_match(0.2, None) is None
    assert bp.market_floor_self_match(0.2, None, bid_history=[("a", "b"), (1,),
                                                              None, ()],
                                      now=NOW) is None
    # a mapping-shaped entry (future schema / hand-edited state.json) indexes
    # by KEY — e[0] raises KeyError, which the PURE matcher must swallow like
    # the other shapes (_hist_field already does; the matcher did not)
    assert bp.market_floor_self_match(0.2, None, bid_history=[{"ts": 1}],
                                      now=NOW) is None


# --------------------------------------------------------------------------- #
# the two ratchets, end to end through _bid_action
# --------------------------------------------------------------------------- #

def _state(**kw):
    s = bp.mk_poll_state(present=True, actual_status="running",
                         market_min_bid=None, last_bid=None, max_bid=1.00,
                         on_demand=1.00, now=NOW, last_bid_put_ts=0.0)
    s.update(kw)
    return s


def test_decay_then_stale_echo_would_ratchet_above_where_decay_left_it():
    """The decay shape. Guard OFF (floor passed through) the tick raises to
    1.2 x our own PREVIOUS bid — strictly above the bid decay had just set.
    Guard ON the floor is None and no bid move is priced at all."""
    # decay lowered us 0.50 -> 0.40; the chunk still echoes 0.50
    act = bp._bid_action(_state(market_min_bid=0.50, last_bid=0.40))
    assert act is not None and act.kind == "raise_bid"
    assert float(act.reason.split(":")[1]) == 0.60 > 0.50, \
        "the echo does not merely undo the decay, it ratchets past the old bid"
    # with the fix the floor never reaches _bid_action
    hist = [(NOW - 120, 0.50, 7)]
    assert bp.market_floor_self_match(0.50, 0.40, bid_history=hist, now=NOW)
    assert bp._bid_action(_state(market_min_bid=None, last_bid=0.40)) is None


def test_a_sub_eleven_percent_raise_leaves_the_echo_defendable():
    """The cushion shape: a 1.10x raise keeps the stale echo above
    `defend_at x last_bid`, so the defend re-fires on our own money."""
    assert 1.10 < 1.0 / bp.DEFEND_AT, "the cushion sits inside the blind window"
    act = bp._bid_action(_state(market_min_bid=0.50, last_bid=0.55))
    assert act is not None and act.kind == "raise_bid"     # 0.50 >= 0.9 x 0.55
    hist = [(NOW - 90, 0.50, 7)]
    assert bp.market_floor_self_match(0.50, 0.55, bid_history=hist, now=NOW)


# --------------------------------------------------------------------------- #
# the recorder + the two lane guards
# --------------------------------------------------------------------------- #

def test_recorder_collapses_a_still_bid_and_refreshes_ts_last():
    ctx = {}
    pricing._note_standing_bid(ctx, 0.20, 7, NOW)
    pricing._note_standing_bid(ctx, 0.20, 7, NOW + 60)
    pricing._note_standing_bid(ctx, 0.20, 7, NOW + 120)
    assert len(ctx["bid_history"]) == 1
    assert ctx["bid_history"][0][0] == NOW, \
        "ts_first keeps the OLDEST sighting of the price (telemetry)"
    assert ctx["bid_history"][0][3] == NOW + 120, \
        "ts_last must track the NEWEST sighting — the echo window runs from it"
    pricing._note_standing_bid(ctx, 0.25, 7, NOW + 180)
    assert [e[1] for e in ctx["bid_history"]] == [0.20, 0.25]


def test_a_long_held_price_is_still_suppressed_after_the_bid_moves():
    """THE steady-state ratchet reopener (review 2026-08-10, F1): the original
    recorder kept only the FIRST-seen ts and pruned on it, so a bid that sat
    still for >= lag_s was dropped from history the instant a rung moved it —
    and its echo (measured to persist up to ~222 s after a lower) re-armed the
    defend on our own money. The window must run from when the price STOPPED
    standing, not from when it started."""
    ctx = {}
    t = NOW
    while t <= NOW + 3600:                       # an hour at one price
        pricing._note_standing_bid(ctx, 0.50, 7, t)
        t += 60
    pricing._note_standing_bid(ctx, 0.55, 7, NOW + 3660)   # a defend/decay rung moves it
    hist = pricing._bid_history_for(ctx, 7)
    assert [e[1] for e in hist] == [0.50, 0.55], \
        "the hour-held price must survive the prune after the move"
    m = bp.market_floor_self_match(0.50, 0.55, bid_history=hist,
                                   now=NOW + 3660 + 222)
    assert m is not None and m.kind == "prior" and m.price == 0.50, \
        "the echo of the long-held price must be suppressed at +222s"
    assert m.age_s == 282.0, \
        "age runs from the LAST tick the price was seen standing (the move " \
        "tick minus one poll interval), not from the first sighting an hour ago"
    # ...and once the echo window has truly elapsed, the guard lets go
    assert bp.market_floor_self_match(
        0.50, 0.55, bid_history=hist,
        now=NOW + 3660 + bp.BID_SELF_FLOOR_LAG_S + 1) is None


def test_a_backwards_wall_clock_step_does_not_unsuppress():
    """Wall-clock time can step backwards (NTP). A history ts in the 'future'
    made the old matcher skip the entry (negative age), failing OPEN into the
    ratchet. Negative age clamps to 0: a future-dated entry is still ours."""
    hist = [[NOW + 300, 0.50, 7, NOW + 300]]     # recorded 'later' than now
    m = bp.market_floor_self_match(0.50, 0.55, bid_history=hist, now=NOW)
    assert m is not None and m.age_s == 0.0


def test_recorder_trims_to_the_window_and_the_cap():
    ctx = {}
    pricing._note_standing_bid(ctx, 0.10, 7, NOW)
    pricing._note_standing_bid(ctx, 0.20, 7, NOW + bp.BID_SELF_FLOOR_LAG_S + 1)
    assert [e[1] for e in ctx["bid_history"]] == [0.20]
    ctx = {}
    for i in range(pricing.BID_HISTORY_MAX + 10):
        pricing._note_standing_bid(ctx, 0.10 + i / 1000.0, 7, NOW + i)
    assert len(ctx["bid_history"]) == pricing.BID_HISTORY_MAX


def test_the_entry_cap_cannot_evict_an_in_window_price():
    """The rate limiter bounds distinct prices per window at
    lag_s / BID_RATE_LIMIT_S; if BID_HISTORY_MAX falls below that, the cap
    — not the window — silently decides what the guard remembers, during
    exactly the aggressive ladder activity the guard exists for. Broke
    latently when the window widened 900 s -> 3600 s (2026-08-14): 24 kept
    entries vs 60 possible per window."""
    assert pricing.BID_HISTORY_MAX >= bp.BID_SELF_FLOOR_LAG_S / bp.BID_RATE_LIMIT_S
    # and behaviorally: prices PUT at the rate limit across a whole window
    # must all still be in history at the end of it
    ctx = {}
    steps = int(bp.BID_SELF_FLOOR_LAG_S / bp.BID_RATE_LIMIT_S)
    for i in range(steps):
        pricing._note_standing_bid(ctx, 0.10 + i / 1000.0, 7,
                             NOW + i * bp.BID_RATE_LIMIT_S)
    end = NOW + (steps - 1) * bp.BID_RATE_LIMIT_S
    kept = [e for e in ctx["bid_history"]
            if end - e[3] <= bp.BID_SELF_FLOOR_LAG_S]
    assert len(kept) == steps


def test_history_is_per_machine_so_a_replacement_inherits_no_echo():
    ctx = {}
    pricing._note_standing_bid(ctx, 0.20, 7, NOW)          # old box
    pricing._note_standing_bid(ctx, 0.30, 9, NOW + 10)     # replacement, machine 9
    assert [e[1] for e in pricing._bid_history_for(ctx, 9)] == [0.30]
    assert bp.market_floor_self_match(0.20, 0.30,
                                      bid_history=pricing._bid_history_for(ctx, 9),
                                      now=NOW + 20) is None, \
        "the old machine's price is not the new chunk's echo"


def test_recorder_ignores_missing_prices():
    ctx = {}
    pricing._note_standing_bid(ctx, None, 7, NOW)
    pricing._note_standing_bid(ctx, 0, 7, NOW)
    assert ctx.get("bid_history") in (None, [])


def test_history_survives_the_json_round_trip_fleetd_does():
    """fleetd persists the watch record as JSON, which has no tuples — the
    entries come back as LISTS. Every accessor must tolerate both."""
    import json
    import fleetd
    assert "bid_history" in fleetd.REPLACEMENT_STATE_KEYS
    ctx = {}
    pricing._note_standing_bid(ctx, 0.20, 7, NOW)
    round_tripped = json.loads(json.dumps({"bid_history": ctx["bid_history"]}))
    ctx2 = dict(round_tripped)
    assert isinstance(ctx2["bid_history"][0], list)
    pricing._note_standing_bid(ctx2, 0.20, 7, NOW + 60)       # dedupe still works
    assert len(ctx2["bid_history"]) == 1
    pricing._note_standing_bid(ctx2, 0.25, 7, NOW + 90)
    assert len(ctx2["bid_history"]) == 2
    hist = pricing._bid_history_for(ctx2, 7)
    assert bp.market_floor_self_match(0.20, 0.25, bid_history=hist,
                                      now=NOW + 100) is not None


def test_run_lane_guard_suppresses_a_prior_bid_echo(monkeypatch):
    st = {"is_bid": True, "last_bid": 0.0421, "machine_id": 115469,
          "run_id": "r1", "instance_id": "700", "now": NOW,
          "bid_history": [(NOW - 200, 0.016, 115469)]}
    emits = []
    monkeypatch.setattr(journal, "_sup_emit",
                        lambda rid, ev, **kw: emits.append((ev, kw)))
    assert run_lane._self_floor_guard(st, 0.016, live=True) is None
    assert emits and emits[0][0] == "bid_self_floor"
    assert emits[0][1]["matched"] == "prior"
    assert emits[0][1]["matched_bid"] == 0.016
    assert emits[0][1]["matched_age_s"] == 200.0
    # ...and a STOPPED box is the opposite case: somebody else holds the chunk
    assert run_lane._self_floor_guard(st, 0.016, live=False) == 0.016


def test_the_standing_to_prior_transition_is_not_deduped_away(monkeypatch):
    """On a machine whose every listed chunk is ours, the market value never
    changes across an echo episode — the OLD value-only dedup key journaled the
    first match and swallowed the standing->prior transition after a bid move,
    which is the event whose `matched_age_s` measures the real echo duration.
    The key is now (value, kind); same value + new kind must re-emit."""
    st = {"is_bid": True, "last_bid": 0.032, "machine_id": 52305,
          "run_id": "r1", "instance_id": "700", "now": NOW,
          "bid_history": [(NOW - 10, 0.032, 52305)]}
    emits = []
    monkeypatch.setattr(journal, "_sup_emit",
                        lambda rid, ev, **kw: emits.append((ev, kw)))
    assert run_lane._self_floor_guard(st, 0.032, live=True) is None
    assert [e[1]["matched"] for e in emits] == ["standing"]
    # a rung raises the bid; the echo (same VALUE) is now a prior price
    st["last_bid"] = 0.0353
    assert run_lane._self_floor_guard(st, 0.032, live=True) is None
    assert [e[1]["matched"] for e in emits] == ["standing", "prior"], \
        "the kind change must journal even though the market value did not move"
    # a repeat of the SAME (value, kind) still dedupes — no per-poll spam
    assert run_lane._self_floor_guard(st, 0.032, live=True) is None
    assert len(emits) == 2


def test_a_sibling_floor_survives_row_level_suppression(monkeypatch):
    """Review 2026-08-10 (F3): one offers query can list BOTH our rented chunk
    (min_bid = the echo of our own bid) and a free sibling chunk (a genuine
    floor). The old scalar guard suppressed the collapsed min() — so a genuine
    sibling floor that rose ABOVE our bid stayed invisible for as long as our
    lower echo was listed, unboundedly for a standing match: we would hold
    $0.50 into a $0.90 market and lose the box without a defend. Row-level:
    only our row is dropped, the sibling stays the market."""
    st = {"is_bid": True, "last_bid": 0.50, "machine_id": 52305,
          "run_id": "r1", "instance_id": "700", "now": NOW,
          "bid_history": [[NOW - 10, 0.50, 52305, NOW - 10]]}
    emits = []
    monkeypatch.setattr(journal, "_sup_emit",
                        lambda rid, ev, **kw: emits.append((ev, kw)))
    out = run_lane._self_floor_guard(st, 0.50, live=True, floors=[0.50, 0.90])
    assert out == 0.90, "the genuine sibling floor must remain the market"
    assert emits and emits[0][0] == "bid_self_floor"
    assert emits[0][1]["market_min_bid"] == 0.50      # the suppressed row
    assert emits[0][1]["surviving_floor"] == 0.90
    # a machine whose every listed chunk is ours: full suppression, as before
    st2 = {"is_bid": True, "last_bid": 0.50, "machine_id": 52305,
           "run_id": "r1", "instance_id": "700", "now": NOW,
           "bid_history": [[NOW - 10, 0.50, 52305, NOW - 10]]}
    assert run_lane._self_floor_guard(st2, 0.50, live=True, floors=[0.50]) is None


def test_a_rescaled_floor_is_not_a_defend_trigger_while_tenant(monkeypatch):
    """Review 2026-08-10 (F8): no offer matched our exact GPU count, so the
    floor was synthesized by per-GPU rescale of a DIFFERENT chunk size. While
    we are the live tenant our own rented chunk IS a listing at our count —
    its absence means the listing is mid-flap (a measured transient), and the
    rescaled number can never match our bid history, reading as a market
    1.25-2x above us: a defend trigger by construction. Treated as a failed
    read while tenant; passed through on a stopped box (the rescue path
    keeps the only number it has)."""
    st = {"is_bid": True, "last_bid": 0.50, "machine_id": 52305, "num_gpus": 4,
          "run_id": "r1", "instance_id": "700", "now": NOW, "bid_history": []}
    monkeypatch.setattr(journal, "_sup_emit", lambda rid, ev, **kw: None)
    assert run_lane._self_floor_guard(st, 1.00, live=True, floors=[1.00],
                               scaled=True) is None
    assert run_lane._self_floor_guard(st, 1.00, live=False, floors=[1.00],
                               scaled=True) == 1.00


def test_a_suppressed_echo_does_not_masquerade_as_an_operator_park():
    """Review 2026-08-10 (#1): poll() rule 2a's underbid-park carve-out read
    the GUARDED floor, and a suppressed `prior` echo collapses to the same
    None a failed offers read produces. The shape: decay lowers us, vast
    underbid-parks the box, and the floor read echoes the bid we decayed FROM
    — with the carve-out blind, the park read as operator intent and the
    supervisor exited `operator_stop`, terminally abandoning a recoverable
    box. The carve-out now reads the RAW floor kept beside the guarded one
    (a diagnostic, not a price input — no move is priced off it)."""
    s = bp.mk_poll_state(present=True, actual_status="running",
                         intended_status="stopped", market_min_bid=None,
                         last_bid=0.40, max_bid=1.0, now=NOW)
    s["market_min_bid_raw"] = 0.50       # the echo of the bid decay left behind
    act = bp.poll(s)
    assert act.kind != "stop_terminal", \
        f"an underbid park with a suppressed echo must stay rescuable: {act}"
    # ...while a genuinely FAILED read (raw None too) keeps the conservative
    # operator-intent read — the fail-closed direction is unchanged
    s2 = bp.mk_poll_state(present=True, actual_status="running",
                          intended_status="stopped", market_min_bid=None,
                          last_bid=0.40, max_bid=1.0, now=NOW)
    act2 = bp.poll(s2)
    assert act2.kind == "stop_terminal" and act2.reason == "operator_stop"


def test_a_failed_read_does_not_end_the_suppression_episode(monkeypatch):
    """Review 2026-08-10 (#8/L6): `_self is None` also covers a FAILED offers
    read and every not-live tick. Clearing the dedup latch there printed
    '$None is a real competing read' and re-journaled a phantom episode start
    on the next match — polluting the matched_age_s distribution that sizes
    the lag window (and did: 900 s -> 3600 s on field data, 2026-08-14)."""
    st = {"is_bid": True, "last_bid": 0.032, "machine_id": 52305,
          "run_id": "r1", "instance_id": "700", "now": NOW,
          "bid_history": [[NOW - 10, 0.032, 52305, NOW - 10]]}
    emits = []
    monkeypatch.setattr(journal, "_sup_emit", lambda rid, ev, **kw: emits.append(ev))
    assert run_lane._self_floor_guard(st, 0.032, live=True) is None
    assert run_lane._self_floor_guard(st, None, live=True) is None    # failed read
    assert st.get("self_floor_at") is not None, \
        "a failed read must not flap the episode latch"
    assert run_lane._self_floor_guard(st, 0.032, live=False) == 0.032  # eviction tick
    assert st.get("self_floor_at") is not None, \
        "a not-live tick must not flap the episode latch either"
    assert run_lane._self_floor_guard(st, 0.032, live=True) is None
    assert emits == ["bid_self_floor"], "one episode, one journal entry"
    # a REAL competing read while we are the live tenant DOES end the episode
    assert run_lane._self_floor_guard(st, 0.050, live=True) == 0.050
    assert st.get("self_floor_at") is None


def test_the_dedup_key_survives_a_state_json_round_trip(monkeypatch):
    """The suppression state persists through state.json. json returns tuples
    as LISTS, so a tuple dedup key mis-compares once after every daemon restart
    and re-journals a duplicate event into the matched_age_s field data. The
    key must therefore be JSON-stable: guard -> serialize -> deserialize ->
    guard again under identical conditions must NOT re-emit."""
    import json
    st = {"is_bid": True, "last_bid": 0.032, "machine_id": 52305,
          "run_id": "r1", "instance_id": "700", "now": NOW,
          "bid_history": [[NOW - 10, 0.032, 52305]]}
    emits = []
    monkeypatch.setattr(journal, "_sup_emit",
                        lambda rid, ev, **kw: emits.append((ev, kw)))
    assert run_lane._self_floor_guard(st, 0.032, live=True) is None
    assert len(emits) == 1
    st = json.loads(json.dumps(st))          # the daemon restart
    assert run_lane._self_floor_guard(st, 0.032, live=True) is None
    assert len(emits) == 1, \
        "an unchanged (value, kind) must stay deduped across a restart"
