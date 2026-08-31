"""Durable host reputation: the scoring rules, and the seams that consume them.

The tests that matter here are the ones pinning a THRESHOLD RELATIONSHIP rather
than a number. The layer's whole claim is that failures spread across days mean
something a burst inside one session does not, and that claim lives in the gap
between two scores and a constant — so `test_two_days_blocks_where_one_session_
does_not` is the module's real specification. The absolute 3.26 is incidental
and would survive a retune; the ordering must not.

Fail-open is the other property under test, and it is tested by BREAKING things
(a corrupt file, an unwritable dir, an exploding store) and asserting the market
still answers. A reputation layer that can make a launch impossible is worse
than no reputation layer.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vastlib.market import hostrep, offers  # noqa: E402

NOW = 1_800_000_000.0
DAY = 86400.0


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A private store for one test, addressed the way callers address it."""
    p = tmp_path / "host_reputation.json"
    monkeypatch.setenv(hostrep.PATH_ENV, str(p))
    monkeypatch.delenv(hostrep.DISABLE_ENV, raising=False)
    hostrep._cache.update({"path": None, "t": 0.0, "data": None})
    return str(p)


def _offer(mid, price, oid=None):
    return {"id": oid if oid is not None else mid, "machine_id": mid,
            "min_bid": price, "dph_total": price}


# ------------------------------------------------------------------ scoring

def test_one_strike_scores_one_and_prices_at_the_documented_multiplier(store):
    assert hostrep.note_strike(7, "pull_timeout", now=NOW) == pytest.approx(1.0)
    assert hostrep.penalty(7, now=NOW) == pytest.approx(1.35)
    assert hostrep.blocked_machines(NOW) == set()


def test_two_days_blocks_where_one_session_does_not(store, tmp_path, monkeypatch):
    """THE rule this module exists for. Same two failures, same host; only the
    spacing differs, and only the spread-out pair is a host verdict."""
    hostrep.note_strike(1, "pull_timeout", now=NOW - 600)
    burst = hostrep.note_strike(1, "pull_timeout", now=NOW)

    monkeypatch.setenv(hostrep.PATH_ENV, str(tmp_path / "b.json"))
    hostrep._cache.update({"path": None, "t": 0.0, "data": None})
    hostrep.note_strike(2, "pull_timeout", now=NOW - 3 * DAY)
    spread = hostrep.note_strike(2, "pull_timeout", now=NOW)

    block = 3.0  # HOSTREP_BLOCK_SCORE default
    assert burst < block < spread, (burst, spread)
    assert hostrep.blocked_machines(NOW) == {2}


def test_decay_forgives_an_old_strike(store):
    hostrep.note_strike(7, "pull_timeout", now=NOW - 28 * DAY)   # two half-lives
    assert hostrep.penalty(7, now=NOW) == pytest.approx(1.0 + 0.35 * 0.25)


def test_a_block_outlives_the_score_that_earned_it(store):
    """A block bought by recurrence decays back under its own threshold in under
    two days. Without the cooldown the next launch re-rents the host we just
    condemned — which is the exact behaviour this whole module replaces."""
    hostrep.note_strike(3, "pull_timeout", now=NOW - 3 * DAY)
    hostrep.note_strike(3, "pull_timeout", now=NOW)
    later = NOW + 3 * DAY
    rec = hostrep.load(store)["machines"]["3"]
    assert hostrep.score(rec, later) < 3.0, "score must have decayed under it"
    assert hostrep.blocked_machines(later) == {3}, "cooldown still holds"
    assert hostrep.blocked_machines(NOW + 15 * DAY) == set(), "and then expires"


def test_every_block_expires_so_a_fixed_host_is_retried(store):
    """No path to a permanent block. A host that fixes itself has to be able to
    come back, or the store is a shrinking supply list rather than a memory —
    which is the failure mode a hand-maintained blocklist has.

    Walks past the cooldown from a THREE-strike block, so it also covers the
    case where a later strike extended `blocked_until`."""
    for d in (6, 3, 0):
        hostrep.note_strike(3, "pull_timeout", now=NOW - d * DAY)
    assert hostrep.blocked_machines(NOW) == {3}
    horizon = NOW + 400 * DAY
    assert hostrep.blocked_machines(horizon) == set()
    assert hostrep.penalty(3, now=horizon) == pytest.approx(1.0, abs=0.01), \
        "and the penalty has decayed to nothing too"


def test_the_retry_is_probation_not_a_clean_slate(store):
    """The host comes back PENALISED, and one more failure re-blocks it.

    This is the pair that makes a bounded cooldown safe: retrying a host we
    still hold evidence against is only sane if the retry is cheap to lose and
    a repeat is expensive. Both halves are asserted here so a retune that
    breaks either one fails.
    """
    hostrep.note_strike(3, "pull_timeout", now=NOW - 3 * DAY)
    hostrep.note_strike(3, "pull_timeout", now=NOW)
    retry = NOW + 14 * DAY
    assert hostrep.blocked_machines(retry) == set(), "rentable again"
    assert hostrep.penalty(3, now=retry) > 1.4, "but far from a clean slate"
    hostrep.note_strike(3, "pull_timeout", now=retry)
    assert hostrep.blocked_machines(retry) == {3}, "a third bad day re-blocks"


def test_a_success_discounts_earlier_strikes_but_does_not_clear_a_block(store):
    hostrep.note_strike(4, "pull_timeout", now=NOW - 3 * DAY)
    hostrep.note_strike(4, "pull_timeout", now=NOW - DAY)
    blocked_before = hostrep.blocked_machines(NOW)
    hostrep.note_ok(4, now=NOW)
    rec = hostrep.load(store, fresh=True)["machines"]["4"]
    # Discounted, not erased: a host that fails, works, then fails again is the
    # flaky host this exists to find.
    assert 0.0 < hostrep.score(rec, NOW) < 2.0
    assert hostrep.blocked_machines(NOW) == blocked_before == {4}


def test_a_second_host_stop_in_one_session_blocks_the_host(store):
    """The rule `host_stop` exists to express. A host that stops boxes stops
    them minutes apart, so the recurrence term — which counts distinct DAYS —
    never sees it, and only the WEIGHT can answer inside one night.

    Both halves are pinned: one stop must not block (a popular host that
    hiccups once stays rentable) and the second must, without waiting for a
    second day the way `pull_timeout` does.

    The gap matters and is why the weight is not exactly half the threshold:
    the first strike DECAYS before the second lands, so a weight of 3.0/2 puts
    the pair a hair under the bar for any gap at all."""
    block = 3.0  # HOSTREP_BLOCK_SCORE default
    first = hostrep.note_strike(1, "host_stop", now=NOW - 600)
    assert first < block, first
    assert hostrep.blocked_machines(NOW - 600) == set()
    second = hostrep.note_strike(1, "host_stop", now=NOW)
    assert second >= block, second
    assert hostrep.blocked_machines(NOW) == {1}

    # ...and the same pair a DAY apart still blocks, so the rule survives a
    # watch that spans a night rather than an hour.
    hostrep.note_strike(2, "host_stop", now=NOW - DAY)
    hostrep.note_strike(2, "host_stop", now=NOW)
    assert 2 in hostrep.blocked_machines(NOW)


def test_one_host_stop_ranks_the_host_down_without_writing_it_off(store):
    """"One host_stop is enough — move" as arithmetic: the stopper has to be a
    third cheaper before price wins it back, and it is still in the market."""
    hostrep.note_strike(2, "host_stop", now=NOW)
    assert hostrep.penalty(2, now=NOW) == pytest.approx(1.0 + 0.35 * 1.6)
    assert hostrep.blocked_machines(NOW) == set()
    ranked, _notes = hostrep.rank_offers([_offer(2, 0.40), _offer(3, 0.50)],
                                         "min_bid", now=NOW)
    assert [o["machine_id"] for o in ranked] == [3, 2]


def test_a_single_host_stop_decays_away_over_days(store):
    """The other half of "not a blacklist": one stop halves every half-life, so
    a host we met on one bad night is effectively clean a month later."""
    hostrep.note_strike(4, "host_stop", now=NOW - 28 * DAY)   # two half-lives
    rec = hostrep.load(store, fresh=True)["machines"]["4"]
    assert hostrep.score(rec, NOW) == pytest.approx(1.6 * 0.25)
    assert hostrep.penalty(4, now=NOW) == pytest.approx(1.0 + 0.35 * 0.4)
    assert hostrep.blocked_machines(NOW) == set()


def test_an_unknown_strike_kind_still_counts(store):
    """Forward compatibility in the direction that matters: a newer fleetd
    writing a kind this reader does not know must not score it as zero."""
    assert hostrep.note_strike(9, "some_future_kind", now=NOW) == pytest.approx(1.0)


# ------------------------------------------------------------ operator verbs

def test_hold_blocks_and_release_lifts_it_without_dropping_evidence(store):
    hostrep.note_strike(5, "pull_timeout", now=NOW)
    hostrep.hold(5, days=7, reason="ate three pulls while I watched", now=NOW)
    assert hostrep.blocked_machines(NOW) == {5}
    assert hostrep.release(5)
    assert hostrep.blocked_machines(NOW) == set()
    assert hostrep.load(store, fresh=True)["machines"]["5"]["strikes"], "kept"


def test_forget_drops_the_record_entirely(store):
    hostrep.note_strike(6, "pull_timeout", now=NOW)
    assert hostrep.forget(6)
    assert hostrep.load(store, fresh=True)["machines"] == {}


def test_prune_keeps_a_machine_that_is_still_held(store):
    hostrep.note_strike(8, "pull_timeout", now=NOW - 200 * DAY)
    hostrep.hold(8, days=30, reason="known bad", now=NOW)
    hostrep.prune(older_than_d=90.0, now=NOW)
    assert "8" in hostrep.load(store, fresh=True)["machines"]
    assert hostrep.blocked_machines(NOW) == {8}


def test_strike_history_is_bounded(store):
    for i in range(60):
        hostrep.note_strike(11, "pull_timeout", now=NOW - i * 60)
    assert len(hostrep.load(store, fresh=True)["machines"]["11"]["strikes"]) == 40


# ---------------------------------------------------------------- selection

def test_ranking_prefers_a_dearer_clean_host_at_the_pk8_spread(store):
    """The measured case this shipped for: the failing host was $0.361 and the
    one that worked $0.444 (+23%). One strike has to be enough to flip it."""
    hostrep.note_strike(1, "pull_timeout", now=NOW)
    ranked, notes = hostrep.rank_offers([_offer(1, 0.361), _offer(2, 0.444)],
                                        "min_bid", now=NOW)
    assert [o["machine_id"] for o in ranked] == [2, 1]
    assert any("preferring machine 2" in n for n in notes)


def test_a_penalty_is_a_preference_not_a_veto(store):
    """A strike must not buy a host at any price. At 2.5x the penalized host is
    still the better deal and still wins."""
    hostrep.note_strike(1, "pull_timeout", now=NOW)
    ranked, _ = hostrep.rank_offers([_offer(1, 0.361), _offer(2, 0.900)],
                                    "min_bid", now=NOW)
    assert [o["machine_id"] for o in ranked] == [1, 2]


def test_a_blocked_machine_is_dropped_and_the_reason_is_printed(store):
    hostrep.note_strike(1, "pull_timeout", now=NOW - 3 * DAY)
    hostrep.note_strike(1, "pull_timeout", now=NOW)
    ranked, notes = hostrep.rank_offers([_offer(1, 0.10), _offer(2, 0.99)],
                                        "min_bid", now=NOW)
    assert [o["machine_id"] for o in ranked] == [2]
    assert notes and "skipped machine 1" in notes[0]


def test_an_empty_store_leaves_the_market_order_untouched(store):
    rows = [_offer(1, 0.5), _offer(2, 0.1), _offer(3, 0.9)]
    ranked, notes = hostrep.rank_offers(rows, "min_bid", now=NOW)
    assert [o["machine_id"] for o in ranked] == [1, 2, 3], "input order kept"
    assert notes == []


def test_with_blocked_unions_rather_than_replaces(store):
    hostrep.note_strike(1, "pull_timeout", now=NOW - 3 * DAY)
    hostrep.note_strike(1, "pull_timeout", now=NOW)
    assert hostrep.with_blocked([99]) == [1, 99]
    assert hostrep.with_blocked(None) == [1]


def test_with_blocked_never_narrows_the_callers_own_exclusion(store):
    """It sits in front of callers that already had a working list. Dropping an
    entry it could not parse would re-rent the exact machine they excluded."""
    assert hostrep.with_blocked([99, "weird-id", None]) == [99, "weird-id", None]


# --------------------------------------------------------------- fail-open

def test_a_corrupt_store_reads_as_no_evidence(store):
    with open(store, "w", encoding="utf-8") as fh:
        fh.write("{not json at all")
    assert hostrep.load(store, fresh=True)["machines"] == {}
    assert hostrep.blocked_machines(NOW) == set()


def test_a_foreign_schema_version_reads_as_no_evidence(store):
    with open(store, "w", encoding="utf-8") as fh:
        json.dump({"version": hostrep.SCHEMA_VERSION + 1,
                   "machines": {"1": {"strikes": [{"ts": NOW, "kind": "x"}]}}}, fh)
    assert hostrep.load(store, fresh=True)["machines"] == {}


def test_an_unwritable_store_loses_the_strike_instead_of_raising(tmp_path,
                                                                monkeypatch):
    monkeypatch.setenv(hostrep.PATH_ENV, str(tmp_path / "no" / "such" / "d.json"))
    hostrep._cache.update({"path": None, "t": 0.0, "data": None})
    monkeypatch.setattr(hostrep.os, "makedirs",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
    assert hostrep.note_strike(1, "pull_timeout", now=NOW) is None


def test_pick_offers_survives_a_reputation_layer_that_explodes(monkeypatch):
    """The property that makes this safe to put on the launch path."""
    def _boom(*a, **k):
        raise RuntimeError("store on fire")
    monkeypatch.setattr(hostrep, "rank_offers", _boom)
    rows = [_offer(1, 0.5), _offer(2, 0.1)]
    assert offers._hostrep_rerank(rows, "min_bid", True) == rows


def test_the_disable_switch_reverts_to_cheapest_first(store, monkeypatch):
    hostrep.note_strike(1, "pull_timeout", now=NOW - 3 * DAY)
    hostrep.note_strike(1, "pull_timeout", now=NOW)
    monkeypatch.setenv(hostrep.DISABLE_ENV, "1")
    ranked, notes = hostrep.rank_offers([_offer(1, 0.1), _offer(2, 0.9)],
                                        "min_bid", now=NOW)
    assert [o["machine_id"] for o in ranked] == [1, 2] and notes == []
    assert hostrep.blocked_machines(NOW) == set()
    assert hostrep.note_strike(2, "pull_timeout", now=NOW) is None


# ------------------------------------------------------------------ display

def test_summary_reports_the_distinct_day_count(store):
    hostrep.note_strike(1, "pull_timeout", now=NOW - 3 * DAY)
    hostrep.note_strike(1, "boot_sla", now=NOW)
    row = hostrep.summary(NOW)[0]
    assert row["machine_id"] == "1" and row["strikes"] == 2
    assert row["distinct_days"] == 2
    assert row["kinds"] == ["boot_sla", "pull_timeout"]
    assert "BLOCK" not in str(row["blocked_reason"]) or row["blocked_reason"]


def test_a_block_reason_says_when_we_will_retry(store):
    """Every block here is temporary, and a message that says only "blocked"
    invites an operator to treat it as permanent and clear it by hand."""
    hostrep.note_strike(3, "pull_timeout", now=NOW - 3 * DAY)
    hostrep.note_strike(3, "pull_timeout", now=NOW)
    reason = hostrep.verdicts(NOW)["3"]["blocked_reason"]
    assert "retry in 14" in reason, reason
    assert "retry in" in hostrep.summary(NOW)[0]["blocked_reason"]


def test_an_operator_hold_also_says_when_it_lifts(store):
    hostrep.hold(3, days=30, reason="known bad", now=NOW)
    reason = hostrep.verdicts(NOW)["3"]["blocked_reason"]
    assert "known bad" in reason and "retry in 30d" in reason, reason


def test_retuning_the_cooldown_moves_a_LIVE_block(store, monkeypatch):
    """A retry policy an operator changes has to apply to the hosts currently
    blocked, or the change reads as a no-op on the only records it is about.

    Found live: HOSTREP_BLOCK_COOLDOWN_D went 7d -> 14d and the one blocked
    machine kept reporting "retry in 6.4d", because the deadline had been
    materialized at write time.
    """
    hostrep.note_strike(3, "pull_timeout", now=NOW - 3 * DAY)
    hostrep.note_strike(3, "pull_timeout", now=NOW)
    assert hostrep.blocked_machines(NOW + 10 * DAY) == {3}, "14d default holds"
    monkeypatch.setenv("HOSTREP_BLOCK_COOLDOWN_D", "2")
    assert hostrep.blocked_machines(NOW + 10 * DAY) == set(), "shortened knob applies"
    monkeypatch.setenv("HOSTREP_BLOCK_COOLDOWN_D", "40")
    assert hostrep.blocked_machines(NOW + 30 * DAY) == {3}, "lengthened knob too"


def test_a_pre_2026_08_20_record_keeps_its_materialized_deadline(store):
    """`blocked_until` was the decision actually made; re-deriving it from a
    crossing we never recorded would be an invention."""
    import json
    with open(store, "w", encoding="utf-8") as fh:
        json.dump({"version": hostrep.SCHEMA_VERSION, "machines": {"3": {
            "strikes": [{"ts": NOW, "kind": "pull_timeout"}],
            "blocked_until": NOW + 3 * DAY}}}, fh)
    hostrep._cache.update({"path": None, "t": 0.0, "data": None})
    assert hostrep.blocked_machines(NOW + 2 * DAY) == {3}
    assert hostrep.blocked_machines(NOW + 4 * DAY) == set(), "not extended to 14d"
