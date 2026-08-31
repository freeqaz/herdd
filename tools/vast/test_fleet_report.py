"""Tests for `fleet_report` — the fleetd journal review as a command.

The fixtures below are SYNTHETIC but shaped from the real journal (field names
and values were read off `~/.local/state/vast-fleetd/journal.ndjsonl` before the
schema was written, not guessed): the 79-events-in-66-min refusal spam, the
887.6 s prior-bid echo against a 900 s window, and the evicted -> rescued ->
evicted -> replaced ladder are the three shapes the 2026-08-14 review actually
had to reason about.

The load-bearing test in here is `test_event_schema_covers_every_touched_event`:
a report that quietly stops counting an event it used to count is worse than one
that crashes, because the aggregate keeps printing a smaller number.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fleet_report as fr  # noqa: E402


T0 = 1786000000.0          # arbitrary epoch anchor for the fixtures


def _iso(ts):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _row(event, ts, **kw):
    rec = {"event": event, "ts": ts, "ts_iso": _iso(ts)}
    rec.setdefault("target", kw.get("iid"))
    rec.update(kw)
    return rec


def write_journal(tmp_path, rows, name="journal.ndjsonl"):
    """rows: dicts (written as JSON) or raw strings (written verbatim)."""
    p = tmp_path / name
    with open(p, "w") as fh:
        for r in rows:
            fh.write(r if isinstance(r, str) else json.dumps(r))
            fh.write("\n")
    return str(p)


# --------------------------------------------------------------------------- #
# schema pin
# --------------------------------------------------------------------------- #
def test_event_schema_covers_every_touched_event():
    """Every event name the aggregation constants name is in EVENT_SCHEMA."""
    missing = sorted(fr.TOUCHED_EVENTS - set(fr.EVENT_SCHEMA))
    assert not missing, (
        f"aggregates touch events with no EVENT_SCHEMA entry: {missing} — the "
        "schema pin is the only thing that turns a journal rename into a "
        "counted anomaly instead of a silently smaller number")
    for const in ("SELF_FLOOR_EVENT", "FLOOR_BLIND_EVENT", "EVICTED_EVENT",
                  "SURVIVED_EVENT", "REPLACED_EVENT", "REBID_RUNG_EVENT",
                  "REBID_REFUSED_EVENT", "REPLACEMENT_DECISION_EVENT",
                  "WATCH_REGISTERED_EVENT", "WATCH_AUTO_ADOPTED_EVENT",
                  "WATCH_FINISHED_EVENT", "WATCH_DORMANT_EVENT"):
        assert getattr(fr, const) in fr.EVENT_SCHEMA, f"{const} unpinned"


def test_no_function_body_hardcodes_an_event_name():
    """Event names may only be referenced through the module-level constants.

    A bare `if e["event"] == "jobs_rebid_refused"` inside a function would work
    today and go stale invisibly on a rename, because nothing links it to the
    schema. This walks the AST rather than grepping so a name in a comment or a
    docstring is not a false positive."""
    src = pathlib.Path(fr.__file__).read_text()
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body[1:] if (node.body and isinstance(node.body[0], ast.Expr)
                                 and isinstance(node.body[0].value, ast.Constant)
                                 and isinstance(node.body[0].value.value, str)
                                 ) else node.body
        for sub in body:
            for lit in ast.walk(sub):
                if (isinstance(lit, ast.Constant)
                        and isinstance(lit.value, str)
                        and lit.value in fr.EVENT_SCHEMA):
                    offenders.append(f"{node.name}:{lit.lineno} {lit.value!r}")
    assert not offenders, (
        "event names hardcoded inside function bodies (use the module-level "
        f"*_EVENT constants so the schema pin covers them): {offenders}")


def test_unknown_events_and_bad_rows_are_counted_never_fatal(tmp_path):
    """A heterogeneous journal is the normal case, not the error case."""
    p = write_journal(tmp_path, [
        _row("tick", T0),                                  # unknown to us
        _row("spend_backfilled", T0 + 1, iid="1"),         # ditto
        "{not json at all",                                # unparseable
        "",                                                # blank, skipped
        json.dumps({"ts": T0 + 2, "note": "no event key"}),
        _row("watch_finished", T0 + 3, iid="1"),           # missing `verdict`
        _row("jobs_box_evicted", T0 + 4, iid="1", eviction_class="outbid"),
    ])
    rep = fr.build_report(path=p)
    sch = rep["schema"]
    assert sch["unknown_events"] == {"tick": 1, "spend_backfilled": 1}
    assert sch["malformed_lines"] == 1
    assert sch["rows_without_event"] == 1
    assert sch["missing_fields"] == {"watch_finished.verdict": 1}
    # the degraded watch_finished row is excluded from the aggregate...
    assert rep["watches"]["finished_by_verdict"] == {}
    # ...while the healthy rows still report
    assert rep["evictions"]["episodes"] == 1
    buf = io.StringIO()
    fr.render_text(rep, buf)                     # must not raise
    assert "unparseable lines: 1" in buf.getvalue()
    assert "missing field watch_finished.verdict" in buf.getvalue()


def test_missing_journal_reports_rather_than_raises(tmp_path):
    rep = fr.build_report(path=str(tmp_path / "nope.ndjsonl"))
    assert "no journal at" in rep["schema"]["error"]
    buf = io.StringIO()
    fr.render_text(rep, buf)
    assert "!!" in buf.getvalue()
    assert fr.run(argparse.Namespace(journal=str(tmp_path / "nope.ndjsonl"),
                                     since=None, box=None, json=False)) == 1


# --------------------------------------------------------------------------- #
# (a) self-floor suppression ages
# --------------------------------------------------------------------------- #
def _self_floor_rows():
    rows = []
    for i, age in enumerate((0.0, 0.0, 0.0, 0.0)):
        rows.append(_row("jobs_bid_self_floor", T0 + i, iid="47511739",
                         machine_id=56779, matched="standing",
                         matched_age_s=age, matched_bid=0.902,
                         standing_bid=0.902, market_min_bid=0.902))
    # the real post-fix prior-echo sequence off box 47511739 / machine 56779
    for i, age in enumerate((43.0, 450.0, 695.0, 887.6)):
        rows.append(_row("jobs_bid_self_floor", T0 + 10 + i, iid="47511739",
                         machine_id=56779, matched="prior", matched_age_s=age,
                         matched_bid=0.832, standing_bid=0.902,
                         market_min_bid=0.832))
    # a pre-`matched_age_s` row: counted, never dropped
    rows.append(_row("jobs_bid_self_floor", T0 + 20, iid="47205562",
                     machine_id=22078, standing_bid=0.2, market_min_bid=0.2))
    return rows


def test_self_floor_ages_by_match_kind(tmp_path):
    p = write_journal(tmp_path, _self_floor_rows())
    sf = fr.build_report(path=p)["self_floor"]
    assert sf["events"] == 9
    assert sf["by_kind"]["standing"] == {"count": 4, "min": 0.0, "median": 0.0,
                                         "p90": 0.0, "max": 0.0}
    prior = sf["by_kind"]["prior"]
    assert prior["count"] == 4
    assert prior["median"] == 450.0        # nearest rank, no interpolation
    assert prior["p90"] == 887.6
    assert prior["max"] == 887.6
    assert sf["by_kind"]["unspecified"]["no_age_rows"] == 1
    assert sf["no_age_rows"] == 1


def test_censoring_warning_fires_at_the_window_edge(tmp_path):
    """887.6 s against the OLD 900 s window is 98.7% of a censoring boundary —
    the observation that widened the window 4x. The report has to say so."""
    p = write_journal(tmp_path, _self_floor_rows())
    events, _ = fr.load_events(p)
    at_900 = fr.self_floor_ages(events, lag_s=900.0)
    assert at_900["censored"] is True
    assert "CENSORED" in at_900["warning"] and "98.6" in at_900["warning"]
    at_3600 = fr.self_floor_ages(events, lag_s=3600.0)   # the shipped window
    assert at_3600["censored"] is False
    assert at_3600["warning"] is None
    buf = io.StringIO()
    fr.render_text({"schema": {"journal": "x", "lines": 1, "rows_in_window": 1,
                               "malformed_lines": 0, "rows_without_event": 0,
                               "unknown_events": {}, "missing_fields": {}},
                    "since": None, "box": None, "self_floor": at_900,
                    "refusals": fr.refusal_episodes([]),
                    "evictions": fr.eviction_outcomes([]),
                    "watches": fr.watch_lifecycle([]),
                    "notifications": fr.notifications([])}, buf)
    assert "CENSORED" in buf.getvalue()


def test_censoring_default_reads_the_live_ladder_constant(tmp_path):
    """The threshold is not a local copy of 3600 — it comes from bidpolicy, so
    re-tuning the window re-tunes the warning."""
    import bidpolicy
    p = write_journal(tmp_path, _self_floor_rows())
    sf = fr.build_report(path=p)["self_floor"]
    assert sf["lag_s"] == float(bidpolicy.BID_SELF_FLOOR_LAG_S)


# --------------------------------------------------------------------------- #
# (b) refusal episodes
# --------------------------------------------------------------------------- #
_SPENT = ("one-shot job-aware defense already spent (1 rung used) — one priced "
          "re-bid per eviction cycle, never a bidding war")
_CAP = "replacement cap reached (3/3) — not re-renting in a loop"


def _refusal_spam(n=79, step=50.0, iid="47398836"):
    rows = []
    for i in range(n):
        ts = T0 + i * step
        rows.append(_row("jobs_rebid_refused", ts, iid=iid, reason=_SPENT,
                         eviction_class="outbid", last_bid=0.6, rungs_used=1))
        rows.append(_row("eviction_replacement_decision", ts, iid=iid,
                         action="stop", reason=_CAP, eviction_class="outbid",
                         replacements_used=3, max_replacements=3))
    return rows


def test_refusal_episodes_collapse_the_tick_spam(tmp_path):
    p = write_journal(tmp_path, _refusal_spam())
    rf = fr.build_report(path=p)["refusals"]
    assert rf["events"] == 158            # the 2026-08-14 finding, verbatim
    assert rf["episodes"] == 2            # ...announcing exactly 2 facts
    assert rf["amplification"] == 79.0
    fams = {r["family"]: r for r in rf["reasons"]}
    assert set(fams) == {"rebid", "replacement"}
    assert fams["rebid"]["events"] == 79 and fams["rebid"]["episodes"] == 1
    assert fams["rebid"]["worst_episode"]["events"] == 79
    assert fams["rebid"]["worst_episode"]["span_s"] == pytest.approx(78 * 50.0)


def test_refusal_episodes_split_on_the_gap(tmp_path):
    """Two refusal cycles an hour apart are two episodes, not one."""
    rows = _refusal_spam(n=3) + _refusal_spam(n=3, step=50.0)
    for r in rows[6:]:
        r["ts"] += 3600.0
        r["ts_iso"] = _iso(r["ts"])
    p = write_journal(tmp_path, rows)
    rf = fr.build_report(path=p)["refusals"]
    assert rf["events"] == 12 and rf["episodes"] == 4


def test_refusal_reasons_group_by_class_not_by_price(tmp_path):
    """Prices differ per rung; the REASON does not. Grouping on the raw string
    would report one 'reason' per bid and hide the amplification entirely."""
    rows = [
        _row("jobs_rebid_refused", T0, iid="a", reason="ceiling $0.930 at 1/1"),
        _row("jobs_rebid_refused", T0 + 4000, iid="b",
             reason="ceiling $1.212 at 1/1"),
    ]
    p = write_journal(tmp_path, rows)
    rf = fr.build_report(path=p)["refusals"]
    assert len(rf["reasons"]) == 1
    assert rf["reasons"][0]["events"] == 2
    assert rf["reasons"][0]["episodes"] == 2      # different boxes
    assert fr.normalize_reason("ceiling $0.930 at 1/1") == "ceiling $N at N/N"
    assert fr.normalize_reason(None) == "(none)"


def test_accepted_replacement_decisions_are_not_refusals(tmp_path):
    p = write_journal(tmp_path, [
        _row("eviction_replacement_decision", T0, iid="a", action="rent",
             reason="spot rung: $0.6090/hr within the $1.093 ceiling",
             eviction_class="outbid", rental="bid", price=0.609),
        _row("eviction_replacement_decision", T0 + 1, iid="b", action="stop",
             reason=_CAP, eviction_class="outbid"),
    ])
    rf = fr.build_report(path=p)["refusals"]
    assert rf["events"] == 1
    assert rf["replacements_accepted"] == 1


# --------------------------------------------------------------------------- #
# (c) evictions by class and outcome
# --------------------------------------------------------------------------- #
def _eviction_journal():
    """evicted(host_stop) -> rescued, evicted(outbid) -> rung -> refused ->
    replaced, plus a still-open eviction at the end."""
    return [
        _row("jobs_box_evicted", T0, iid="47398836", eviction_class="host_stop",
             machine_id=98261, is_bid=True, claimed_work=True),
        # a re-announcement of the SAME stuck eviction, 40 s later
        _row("jobs_box_evicted", T0 + 40, iid="47398836",
             eviction_class="host_stop", machine_id=98261, is_bid=True,
             claimed_work=True),
        _row("jobs_box_eviction_survived", T0 + 130, iid="47398836",
             standing_bid=0.51),
        _row("jobs_box_evicted", T0 + 300, iid="47398836",
             eviction_class="outbid", machine_id=98261, is_bid=True,
             claimed_work=True),
        _row("jobs_rebid_rung", T0 + 320, iid="47398836",
             eviction_class="outbid", old_bid=0.48, new_bid=0.6, ceiling=0.607),
        _row("jobs_rebid_refused", T0 + 700, iid="47398836", reason=_SPENT,
             eviction_class="outbid", last_bid=0.6, rungs_used=1),
        _row("jobs_replaced", T0 + 900, iid="47399999", from_box="47398836",
             to_box="47399999", eviction_class="outbid", rental="bid",
             replacements_used=1, target="47398836"),
        _row("jobs_box_evicted", T0 + 99000, iid="47694876",
             eviction_class="outbid", machine_id=43934, is_bid=True),
    ]


def test_eviction_outcomes_by_class(tmp_path):
    p = write_journal(tmp_path, _eviction_journal())
    ev = fr.build_report(path=p)["evictions"]
    assert ev["raw_events"] == 4          # announcements
    assert ev["episodes"] == 3            # ...for three real evictions
    assert ev["by_class"] == {
        "host_stop": {"rescued": 1},
        "outbid": {"replaced": 1, "unresolved": 1},
    }
    assert ev["by_outcome"] == {"rescued": 1, "replaced": 1, "unresolved": 1}
    first = ev["rows"][0]
    assert first["announcements"] == 2 and first["outcome"] == "rescued"
    replaced = [r for r in ev["rows"] if r["outcome"] == "replaced"][0]
    assert replaced["signals"] == ["rebid", "refused", "replaced"]


def test_a_terminal_outcome_ends_the_episode(tmp_path):
    """evicted -> rescued -> evicted 60 s later is TWO evictions. A gap-only
    rule would merge them and report one rescue where there were two cycles."""
    rows = [
        _row("jobs_box_evicted", T0, iid="b", eviction_class="host_stop"),
        _row("jobs_box_eviction_survived", T0 + 30, iid="b"),
        _row("jobs_box_evicted", T0 + 60, iid="b", eviction_class="host_stop"),
        _row("jobs_box_eviction_survived", T0 + 90, iid="b"),
    ]
    ev = fr.build_report(path=write_journal(tmp_path, rows))["evictions"]
    assert ev["episodes"] == 2
    assert ev["by_class"] == {"host_stop": {"rescued": 2}}


def test_an_eviction_class_change_ends_the_episode(tmp_path):
    rows = [
        _row("jobs_box_evicted", T0, iid="b", eviction_class="host_stop"),
        _row("jobs_box_evicted", T0 + 30, iid="b", eviction_class="outbid"),
    ]
    ev = fr.build_report(path=write_journal(tmp_path, rows))["evictions"]
    assert ev["episodes"] == 2
    assert ev["by_outcome"] == {"unresolved": 2}


def test_outcome_precedence_prefers_the_terminal_signal(tmp_path):
    rows = [
        _row("jobs_box_evicted", T0, iid="b", eviction_class="outbid"),
        _row("jobs_rebid_rung", T0 + 10, iid="b"),
        _row("jobs_rebid_refused", T0 + 20, iid="b", reason=_SPENT),
        _row("jobs_box_eviction_survived", T0 + 30, iid="b"),
    ]
    ev = fr.build_report(path=write_journal(tmp_path, rows))["evictions"]
    assert ev["rows"][0]["outcome"] == "rescued"
    assert ev["rows"][0]["signals"] == ["rebid", "refused", "rescued"]


# --------------------------------------------------------------------------- #
# (d) per-box timeline + the --box filter
# --------------------------------------------------------------------------- #
def test_box_timeline_and_filter(tmp_path):
    p = write_journal(tmp_path, _eviction_journal())
    rep = fr.build_report(path=p, box="47398836")
    assert rep["box"] == "47398836"
    tl = rep["timeline"]
    assert tl["events"] == 7              # everything on the box, incl. the
    assert [r["event"] for r in tl["rows"]][:2] == ["jobs_box_evicted"] * 2
    # the replacement row names the box as `from_box`, so it belongs to it
    assert "jobs_replaced" in [r["event"] for r in tl["rows"]]
    # the other box's eviction is filtered out of EVERY aggregate
    assert rep["evictions"]["boxes"] == 1
    assert rep["schema"]["rows_other_box"] == 1
    detail = [r["detail"] for r in tl["rows"] if r["event"] == "jobs_rebid_rung"][0]
    assert "new_bid=0.6" in detail and "ceiling=0.607" in detail
    buf = io.StringIO()
    fr.render_text(rep, buf)
    assert "LADDER TIMELINE 47398836" in buf.getvalue()


# --------------------------------------------------------------------------- #
# (e) watch lifecycle
# --------------------------------------------------------------------------- #
def test_watch_lifecycle_counts_the_lapsed_cycle(tmp_path):
    rows = [
        _row("watch_registered", T0, iid="1", profile="jobs", budget_usd=5.0,
             requester="free@rig"),
        _row("watch_finished", T0 + 100, iid="1", profile="jobs",
             verdict="drained", spend_usd=1.5),
        _row("watch_auto_adopted", T0 + 160, iid="1", profile="bare",
             requester="fleetd:auto-adopt"),
        # a bare adoption with NO prior policy watch is not a lapse
        _row("watch_auto_adopted", T0 + 200, iid="2", profile="bare",
             requester="fleetd:auto-adopt"),
        _row("watch_finished", T0 + 300, iid="2", profile="bare",
             verdict="instance_gone", spend_usd=0.3),
        _row("watch_dormant", T0 + 400, iid="3", reason="operator_stop",
             requester="free@rig"),
    ]
    wl = fr.build_report(path=write_journal(tmp_path, rows))["watches"]
    assert wl["finished_by_verdict"] == {"drained": 1, "instance_gone": 1}
    assert wl["drained"] == 1
    assert wl["registered_by_profile"] == {"jobs": 1}
    assert wl["auto_adopted_by_profile"] == {"bare": 2}
    assert wl["dormant_by_reason"] == {"operator_stop": 1}
    assert wl["bare_adoptions_after_policy_watch"] == 1
    lapse = wl["lapses"][0]
    assert lapse["iid"] == "1" and lapse["from_profile"] == "jobs"
    assert lapse["verdict"] == "drained" and lapse["gap_s"] == 60.0


def test_an_explicit_rewatch_is_not_a_lapse(tmp_path):
    rows = [
        _row("watch_finished", T0, iid="1", profile="jobs", verdict="drained"),
        _row("watch_registered", T0 + 10, iid="1", profile="jobs",
             budget_usd=5.0),
        _row("watch_auto_adopted", T0 + 20, iid="1", profile="bare"),
    ]
    wl = fr.build_report(path=write_journal(tmp_path, rows))["watches"]
    assert wl["bare_adoptions_after_policy_watch"] == 0


# --------------------------------------------------------------------------- #
# --since, paths, CLI
# --------------------------------------------------------------------------- #
# (f) vast's notification channel (NOTIFY_DESIGN S2a)
# --------------------------------------------------------------------------- #
def _notify_rows():
    """`notify_seen` rows built from the REAL captured inbox, in the journal
    shape fleetd writes (`notify.journal_fields` + the common envelope), mixed
    with the ladder events they explain."""
    import notify as _notify
    fix = (pathlib.Path(__file__).resolve().parent / "testfixtures" / "notify"
           / "inbox_2026-08-16.json")
    with open(fix) as fh:
        inbox = json.load(fh)
    rows, i = [], 0
    for r in sorted(inbox["notifications"], key=lambda r: r["created_at"]):
        if r["notif_type"] != "outbid":
            continue
        i += 1
        rows.append(_row(_notify.SEEN_EVENT, T0 + i, **_notify.journal_fields(r)))
    return rows


def test_notifications_aggregate_over_the_real_feed(tmp_path):
    p = write_journal(tmp_path, _notify_rows() + _eviction_journal())
    rep = fr.build_report(path=p)
    nt = rep["notifications"]
    assert nt["rows"] == nt["outbids"] == 16
    assert nt["by_type"] == {"outbid": 16}
    # 16 outbids across 15 boxes: one instance was displaced TWICE in the
    # window, which is why a per-eviction-cycle match (S2b §6.3) needs a
    # freshness bound rather than a per-instance lookup.
    assert nt["priced"] == 16 and nt["boxes"] == 15
    # the displacing price is the number that exists nowhere else
    assert nt["spread_max"] == pytest.approx(2.26)          # $1.60 -> $3.86
    assert nt["worst"][0]["machine_id"] == 43532
    # §1.4: displacement BELOW our own bid is real and must be called out
    assert [r["iid"] for r in nt["below_our_bid"]] == ["47840057"]
    # per-machine concentration is the "we keep losing this host" read
    assert sum(nt["outbids_by_machine"].values()) == 16
    assert nt["gaps"] == 0 and nt["poll_error_episodes"] == 0
    # the eviction section is untouched by the notification rows
    assert rep["evictions"]["episodes"] == 3
    buf = io.StringIO()
    fr.render_text(rep, buf)
    out = buf.getvalue()
    assert "NOTIFICATIONS — 16 row(s)" in out
    assert "displaced BELOW our bid: 47840057" in out
    assert "EVICTIONS" in out


def test_notification_rows_land_on_the_box_timeline(tmp_path):
    """An outbid row must show up in `--box`, beside the eviction it explains —
    that adjacency is the whole point of journaling it (D4)."""
    rows = _notify_rows()
    iid = [r["iid"] for r in rows if r.get("iid")][0]
    p = write_journal(tmp_path, rows + [
        _row("jobs_box_evicted", T0 + 100, iid=iid, eviction_class="outbid")])
    rep = fr.build_report(path=p, box=iid)
    names = [r["event"] for r in rep["timeline"]["rows"]]
    assert names == [fr.NOTIFY_SEEN_EVENT, "jobs_box_evicted"]
    detail = rep["timeline"]["rows"][0]["detail"]
    assert "notif_type=outbid" in detail and "new_min_bid" in detail


def test_gaps_and_poll_error_episodes_are_counted(tmp_path):
    p = write_journal(tmp_path, [
        _row(fr.NOTIFY_GAP_EVENT, T0, rows=50, window=50),
        _row(fr.NOTIFY_POLL_ERROR_EVENT, T0 + 1, state="failing",
             error="HTTP 404 on GET v0/notifications/inbox/: {}", gone=True),
        _row(fr.NOTIFY_POLL_ERROR_EVENT, T0 + 2, state="ok", failures=3),
    ] + _notify_rows()[:2])
    nt = fr.build_report(path=p)["notifications"]
    assert nt["gaps"] == 1
    assert nt["poll_error_episodes"] == 1 and nt["poll_last_state"] == "ok"
    assert nt["endpoint_gone"] is True
    buf = io.StringIO()
    fr.render_text(fr.build_report(path=p), buf)
    out = buf.getvalue()
    assert "1 gap(s)" in out and "ENDPOINT GONE" in out


def test_an_unknown_notif_type_is_counted_not_crashed_on(tmp_path):
    """The catalog has 30 keys and the feed has shown 3. An unknown type is the
    expected case; a row with no `associated_id` at all is the documented
    webhook shape (§1.7)."""
    p = write_journal(tmp_path, [
        _row(fr.NOTIFY_SEEN_EVENT, T0, event_id="e" * 32,
             notif_type="upcoming_downtime", created_at=T0,
             associated_id={"machine_id": 7, "when": "soon"}),
        _row(fr.NOTIFY_SEEN_EVENT, T0 + 1, event_id="f" * 32,
             notif_type="low_credit", created_at=T0 + 1),
        # a degraded row: no notif_type at all -> counted, never placed
        _row(fr.NOTIFY_SEEN_EVENT, T0 + 2, event_id="0" * 32),
    ])
    rep = fr.build_report(path=p)
    nt = rep["notifications"]
    assert nt["rows"] == 2 and nt["outbids"] == 0
    assert nt["by_type"] == {"upcoming_downtime": 1, "low_credit": 1}
    assert rep["schema"]["missing_fields"] == {"notify_seen.notif_type": 1}
    buf = io.StringIO()
    fr.render_text(rep, buf)
    assert "upcoming_downtimex1" in buf.getvalue()


def test_notify_event_names_come_from_the_shared_leaf():
    """fleetd WRITES these names and this report READS them; both import them
    from `notify`, so a rename cannot leave the schema pin behind."""
    import notify as _notify
    assert fr.NOTIFY_SEEN_EVENT is _notify.SEEN_EVENT
    assert fr.NOTIFY_GAP_EVENT is _notify.GAP_EVENT
    assert fr.NOTIFY_POLL_ERROR_EVENT is _notify.POLL_ERROR_EVENT
    assert set(fr.NOTIFY_EVENTS) <= set(fr.EVENT_SCHEMA)


def test_parse_since_shorthand_and_iso():
    now = 1786000000.0
    assert fr.parse_since(None) is None
    assert fr.parse_since("") is None
    assert fr.parse_since("4d", now=now) == now - 4 * 86400
    assert fr.parse_since("36h", now=now) == now - 36 * 3600
    assert fr.parse_since("90m", now=now) == now - 90 * 60
    assert fr.parse_since("2026-08-10") == 1786320000.0
    assert fr.parse_since("2026-08-10T00:00:00Z") == 1786320000.0
    for bad in ("yesterday", "4w", "2026-13-01", "d"):
        with pytest.raises(ValueError):
            fr.parse_since(bad, now=now)


def test_since_filters_the_window(tmp_path):
    rows = [
        _row("jobs_box_evicted", T0, iid="old", eviction_class="outbid"),
        _row("jobs_box_evicted", T0 + 10000, iid="new", eviction_class="outbid"),
    ]
    p = write_journal(tmp_path, rows)
    rep = fr.build_report(path=p, since=T0 + 5000)
    assert rep["evictions"]["boxes"] == 1
    assert rep["evictions"]["rows"][0]["iid"] == "new"
    assert rep["schema"]["rows_before_since"] == 1
    assert rep["since"].startswith("2026-")
    # the string form goes through parse_since
    rep2 = fr.build_report(path=p, since="1h", now=T0 + 10000)
    assert rep2["evictions"]["boxes"] == 1
    assert fr.build_report(path=p, since="4d", now=T0 + 10000)[
        "evictions"]["boxes"] == 2


def test_journal_path_default_and_override(monkeypatch, tmp_path):
    monkeypatch.delenv("FLEETD_STATE_DIR", raising=False)
    assert fr.journal_path().endswith("/.local/state/vast-fleetd/journal.ndjsonl")
    assert not fr.journal_path().startswith("~")
    monkeypatch.setenv("FLEETD_STATE_DIR", str(tmp_path))
    assert fr.journal_path() == str(tmp_path / "journal.ndjsonl")
    assert fr.journal_path("~/x.ndjsonl") == os.path.expanduser("~/x.ndjsonl")


def test_cli_json_and_text(tmp_path, capsys):
    p = write_journal(tmp_path, _eviction_journal() + _self_floor_rows())
    assert fr.main(["--journal", p]) == 0
    text = capsys.readouterr().out
    assert "SELF-FLOOR SUPPRESSION" in text and "EVICTIONS" in text
    assert "WATCH LIFECYCLE" in text and "REFUSALS" in text
    assert fr.main(["--journal", p, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data) >= {"schema", "self_floor", "refusals", "evictions",
                         "watches"}
    assert data["evictions"]["episodes"] == 3
    assert fr.main(["--journal", p, "--since", "not-a-date"]) == 2
    assert "bad --since" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# herdd wiring
# --------------------------------------------------------------------------- #
def test_herdd_fleet_report_help_ends_with_a_docs_list():
    """Every herdd subcommand's -h carries a `docs:` epilog so an agent
    mid-run can jump straight to the runbook. Shelling out is the only way to
    reach the parser (herdd builds its argparse tree inline in main())."""
    import subprocess
    here = pathlib.Path(__file__).resolve().parent
    r = subprocess.run([sys.executable, str(here / "herdd.py"),
                        "fleet", "report", "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "docs:" in out, out
    tail = out[out.index("docs:"):].strip().splitlines()
    assert len(tail) >= 2 and all(ln.startswith("  ") for ln in tail[1:])
    assert "FLEET_REVIEW_2026-08-14.md" in out
    assert "FLEETD_DESIGN.md" in out and "AUTOBID_DESIGN.md" in out
    for flag in ("--since", "--box", "--journal", "--json"):
        assert flag in out
