#!/usr/bin/env python3
"""fleet_report — the fleetd journal review, as a command instead of a script.

Every productive fleet review so far (2026-08-10, 2026-08-14) was a hand-written
mining loop over `~/.local/state/vast-fleetd/journal.ndjsonl`, and the SAME four
loops paid for themselves twice: self-floor suppression ages by match kind (found
the echo 12 s from the 900 s window edge, which is why `BID_SELF_FLOOR_LAG_S` is
now 3600), refusal episodes against raw refusal counts (found 158 identical
events announcing 2 facts, which is why both refusal sites now journal on reason
change), eviction class x outcome, and the per-box ladder timeline.
`FLEET_REVIEW_2026-08-14.md` item 6 asks for exactly those as a routine command;
this module is the pure half of it and `herdd fleet report` is the CLI.

Two properties are load-bearing, both because a review runs on FIELD data:

  * **Heterogeneous rows never crash the report.** The journal is an
    append-only log written by many emitters across many daemon revisions:
    `tick` rows carry no market fields, `since_s` lives on one event and
    `matched_age_s` on another, and 10 of the 67 self-floor rows on this box
    predate `matched_age_s` existing at all. So unknown event names, unparseable
    lines, and rows missing an expected field are COUNTED and reported in a
    `schema` block — never dropped silently and never raised.
  * **The schema is pinned here, in `EVENT_SCHEMA`.** It is the only place that
    names journal fields, so a rename in `fleetd`/`herdd` shows up as a
    missing-field count in every report rather than as a quietly empty section.
    `test_fleet_report.py` asserts the aggregation code touches no event name
    absent from it.

Pure + stdlib-only: no network, no vast API, no daemon socket. It reads the
journal file and nothing else, so it works with the daemon down — which is
exactly when a review happens.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
from collections import Counter, defaultdict, namedtuple
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import bidpolicy  # noqa: E402  (pure leaf: the ladder constants, no I/O)
import notify  # noqa: E402  (pure leaf: the notification event names + accessors)


DEFAULT_JOURNAL = "~/.local/state/vast-fleetd/journal.ndjsonl"

# A refusal re-announced every ~50 s tick for an hour is ONE fact, not 79 (the
# 2026-08-14 latch defect: 79 + 79 identical events in 66 min on 47398836). Two
# refusals of the same reason on the same box further apart than this are
# separate episodes — comfortably above the ~45-50 s reconcile tick, so a
# re-announcement is merged and a genuine second refusal cycle is not.
REFUSAL_EPISODE_GAP_S = 600.0

# The same collapse for eviction announcements, wider: an eviction's ladder
# (rescue poll -> re-bid rung -> replacement rung) legitimately runs for many
# minutes while the box stays evicted, and the pre-latch-fix daemon re-announced
# through all of it. Observed re-announce spacings on this journal: 17 s-3190 s.
EVICTION_EPISODE_GAP_S = 1800.0

# How far past an eviction episode to look for its outcome, when no later
# eviction on the same box bounds the search.
EVICTION_OUTCOME_WINDOW_S = 6 * 3600.0

# Suppression ages cannot exceed the echo window by construction (an echo older
# than the window reads as a market floor and is never journaled as a match), so
# a max sitting near the edge is a CENSORED sample, not a measured tail. 887.6 s
# against the old 900 s window was 98.7% of the edge, and the window was widened
# 4x off it. Warn at this fraction of the window.
CENSORING_FRACTION = 0.90


# --------------------------------------------------------------------------- #
# schema pin
# --------------------------------------------------------------------------- #
EventSpec = namedtuple("EventSpec", "required optional doc")


def _spec(required=(), optional=(), doc=""):
    return EventSpec(frozenset(required), frozenset(optional), doc)


#: fields every journal row carries; not repeated per event.
COMMON_FIELDS = frozenset({"event", "ts", "ts_iso", "target", "note"})

#: Known event names -> the fields THIS REPORT reads. `required` is what an
#: aggregate needs to place the row at all (a row missing one is counted under
#: `schema.missing_fields` and skipped by the aggregates); `optional` is read
#: when present. Fields the journal carries but the report never reads are
#: deliberately absent — this pins the report's contract, not the journal's.
EVENT_SCHEMA = {
    "jobs_bid_self_floor": _spec(
        required=("iid",),
        optional=("machine_id", "market_min_bid", "standing_bid",
                  "matched", "matched_age_s", "matched_bid"),
        doc="a market read that was OUR OWN bid echoing back; suppressed"),
    "jobs_bid_floor_blind": _spec(
        required=("iid",), optional=("machine_id", "since_s", "standing_bid"),
        doc="suppression sustained past BID_SELF_FLOOR_SUSTAINED_S — no market"),
    "jobs_bid_over_ceiling": _spec(
        required=("iid",),
        optional=("ceiling", "market_min_bid", "on_demand", "reason",
                  "standing_bid"),
        doc="surviving this floor costs more than the hard ceiling allows"),
    "jobs_rebid_refused": _spec(
        required=("iid", "reason"),
        optional=("eviction_class", "last_bid", "on_demand", "p_alt",
                  "rungs_used", "ceiling", "market_min_bid"),
        doc="the one-shot job-aware defense declined to bid"),
    "jobs_rebid_rung": _spec(
        required=("iid",),
        optional=("eviction_class", "old_bid", "new_bid", "ceiling",
                  "on_demand", "p_alt", "reason", "rungs_left", "rungs_used"),
        doc="a priced re-bid rung was issued"),
    "eviction_replacement_decision": _spec(
        required=("iid", "action", "reason"),
        optional=("eviction_class", "machine_id", "price", "rental", "ceiling",
                  "budget_left", "budget_usd", "replacements_used",
                  "max_replacements", "lifetime_basis", "offer_min_bid",
                  "offer_ondemand", "ondemand_under_ceiling",
                  # candidate-set selection (2026-08-16): which class was
                  # walked, how many survived the per-offer safety rail, and
                  # what the survivors were ranked on
                  "spot_candidates", "spot_survivors", "ranked_by", "spot_gpu",
                  "spot_machine", "spot_ondemand", "ondemand_candidates",
                  "ondemand_gpu", "why", "spend_usd", "launch_dph_anchor",
                  # the container-disk FIT requirement (2026-08-16): the floor
                  # the candidate class was searched under, and whether that
                  # floor — not price — is what emptied the market
                  "disk_floor_gb", "disk_blocked"),
        doc="rent-a-replacement vs stop; action='stop' IS the refusal"),
    "jobs_box_evicted": _spec(
        required=("iid",),
        optional=("eviction_class", "machine_id", "actual_status",
                  "intended_status", "is_bid", "claimed_work", "market_listed",
                  "market_min_bid", "market_read_ok", "on_demand", "max_bid",
                  "entry_floor", "p_alt", "spend_usd", "standing_bid"),
        doc="box stopped with no self-park and no stop intent — EVICTION"),
    "jobs_box_eviction_survived": _spec(
        required=("iid",), optional=("standing_bid",),
        doc="the evicted box is live again — the ladder won it back"),
    "jobs_replaced": _spec(
        required=("from_box", "to_box"),
        optional=("iid", "eviction_class", "rental", "dph",
                  "replacements_used", "spend_usd", "budget_usd",
                  "old_box_fate"),
        doc="evicted box auto-replaced; queue retargeted"),
    "jobs_box_retention": _spec(
        required=("iid",),
        optional=("eviction_class", "deadline", "retention_h", "keep_labeled",
                  "status", "est_cost_usd", "est_cost_hi_usd"),
        doc="evicted box held for salvage of state that never reached B2"),
    "jobs_box_quiesced": _spec(
        required=("iid",),
        optional=("eviction_class", "to_box", "stopped", "bid_pinned",
                  "prior_bid", "errors", "actor"),
        doc="retained box stopped and bid-pinned so vast cannot resume it"),
    "jobs_retained_box_resurrected": _spec(
        required=("iid",),
        optional=("eviction_class", "replacement_iid", "dph", "live_cost_usd",
                  "live_multiple", "resurrections", "requiesces", "deadline",
                  "actor"),
        doc="a RETAINED box is running again, billing GPU rate unqueued"),
    "jobs_box_condemned": _spec(
        required=("iid",),
        optional=("machine_id", "phase", "verdict", "boot_age_s", "dph",
                  "spend_usd"),
        doc="boot/pull deadline blown — the box is written off"),
    "jobs_rescue_stalled": _spec(
        required=("iid",),
        optional=("profile", "budget_usd", "replacement_refused"),
        doc="rescue gave up but the instance still exists"),
    "jobs_rescue_recovered": _spec(
        required=("iid",), optional=(),
        doc="box back under its watch — ladder + budget re-armed"),
    "jobs_box_resumed": _spec(
        required=("iid",), optional=("profile",),
        doc="a parked/stopped box was resumed in place"),
    "jobs_queue_empty": _spec(
        required=("iid",), optional=("profile", "budget_usd"),
        doc="nothing submitted — the watch is KEPT, ladder stays armed"),
    "watch_registered": _spec(
        required=("iid", "profile"),
        optional=("requester", "budget_usd", "remaining_usd", "ceiling_id",
                  "ceiling_source", "spend_carried_usd"),
        doc="an explicit `fleet watch` upsert"),
    "watch_adopted": _spec(
        required=("iid", "profile"), optional=("spend_usd",),
        doc="the daemon took over an existing watch record"),
    "watch_auto_adopted": _spec(
        required=("iid", "profile"),
        optional=("requester", "budget_usd", "remaining_usd", "ceiling_id",
                  "ceiling_source", "spend_carried_usd"),
        doc="fleetd adopted an unwatched box on its own (usually `bare`)"),
    "watch_finished": _spec(
        required=("iid", "verdict"),
        optional=("profile", "spend_usd", "cap_usd", "ceiling_id",
                  "ceiling_spend_usd", "remaining_usd", "reason"),
        doc="a watch ended; verdict='drained' starts the LAPSED cycle"),
    "watch_dormant": _spec(
        required=("iid", "reason"), optional=("requester",),
        doc="watch parked by operator intent — the ladder will NOT rescue"),
    "watch_removed": _spec(
        required=("iid",), optional=("profile", "reason"),
        doc="watch deleted"),
    # --- vast's notification channel (NOTIFY_DESIGN S2a) -------------------- #
    # Journaled by fleetd's tick as EVIDENCE ONLY: no classifier, ladder or
    # park reads them yet (that is S2b). `notif_type` is the only required
    # field beyond identity — a row whose `associated_id` we cannot read is
    # still a row we saw, and the report says so rather than dropping it.
    notify.SEEN_EVENT: _spec(
        required=("event_id", "notif_type"),
        optional=("iid", "machine_id", "created_at", "associated_id"),
        doc="one notification row consumed from vast's inbox"),
    notify.GAP_EVENT: _spec(
        required=(), optional=("rows", "window"),
        doc="a full inbox window, none of it known — rows may have been missed"),
    notify.POLL_ERROR_EVENT: _spec(
        required=("state",), optional=("error", "gone", "failures", "failing_s"),
        doc="notification poll health CHANGED (never announced per tick)"),
    # --- S2b: rows as EVIDENCE in the classifier + rescue quote ------------- #
    # Pinned here so the timeline prints them beside the eviction they explain
    # and `schema.unknown_events` stays a real alarm. No aggregate reads them
    # yet: the calibration these rows exist for is a field question (does the
    # notification ever change the verdict?) that needs field rows to answer,
    # and inventing a percentage over an empty journal would answer it wrong.
    notify.MATCHED_EVENT: _spec(
        required=("iid", "event_id"),
        optional=("your_bid", "new_min_bid", "created_at",
                  "class_without_notify", "class_with_notify",
                  # WHICH call site matched, and which of the two floors it
                  # carries (review round 1, 2-3). The announce path passes the
                  # RAW market read and the late path the self-floor-GUARDED
                  # one; without these the §6.5 dataset mixes two quantities
                  # under one field name.
                  "match_path", "floor_source"),
        doc="an outbid row matched an eviction; BOTH classes, with and without"),
    notify.BID_MISMATCH_EVENT: _spec(
        required=("iid",),
        optional=("event_id", "believed_bid", "vast_your_bid", "delta"),
        doc="vast's record of our standing bid disagrees with ours (log only)"),
    notify.RESCUE_QUOTE_EVENT: _spec(
        required=("iid",),
        optional=("new_min_bid", "market_floor", "proposed_floor", "emitted",
                  "standing_bid", "max_bid", "rescue_attempted",
                  # the BOUNDS the quote was held to (review round 1, M3), so
                  # the field record can score whether the bound ever bound
                  "ceiling", "launch_dph_anchor", "budget_left", "quoted",
                  "refused", "row_raised"),
        doc="the rescue rung priced off the row; emitted=null = rails refused"),
    notify.FLOOR_CHECK_EVENT: _spec(
        required=("iid",),
        optional=("machine_id", "listing_floor_at_stop", "market_listed",
                  "new_min_bid", "standing_bid", "match_path", "floor_source"),
        doc="listing floor at the stop vs the authoritative displacing price"),
}

# --- the event names the aggregates below are allowed to touch -------------- #
# Every aggregate reads events THROUGH these constants; test_fleet_report.py
# asserts their union is a subset of EVENT_SCHEMA and that no bare event-name
# literal sneaks into the module.
SELF_FLOOR_EVENT = "jobs_bid_self_floor"
FLOOR_BLIND_EVENT = "jobs_bid_floor_blind"
REBID_REFUSED_EVENT = "jobs_rebid_refused"
REPLACEMENT_DECISION_EVENT = "eviction_replacement_decision"
EVICTED_EVENT = "jobs_box_evicted"
SURVIVED_EVENT = "jobs_box_eviction_survived"
REPLACED_EVENT = "jobs_replaced"
REBID_RUNG_EVENT = "jobs_rebid_rung"
WATCH_REGISTERED_EVENT = "watch_registered"
WATCH_AUTO_ADOPTED_EVENT = "watch_auto_adopted"
WATCH_FINISHED_EVENT = "watch_finished"
WATCH_DORMANT_EVENT = "watch_dormant"

NOTIFY_SEEN_EVENT = notify.SEEN_EVENT
NOTIFY_GAP_EVENT = notify.GAP_EVENT
NOTIFY_POLL_ERROR_EVENT = notify.POLL_ERROR_EVENT

REFUSAL_EVENTS = (REBID_REFUSED_EVENT, REPLACEMENT_DECISION_EVENT)
EVICTION_OUTCOME_EVENTS = (SURVIVED_EVENT, REPLACED_EVENT, REBID_RUNG_EVENT,
                           REBID_REFUSED_EVENT, REPLACEMENT_DECISION_EVENT)
WATCH_EVENTS = (WATCH_REGISTERED_EVENT, WATCH_AUTO_ADOPTED_EVENT,
                WATCH_FINISHED_EVENT, WATCH_DORMANT_EVENT)
NOTIFY_EVENTS = (NOTIFY_SEEN_EVENT, NOTIFY_GAP_EVENT, NOTIFY_POLL_ERROR_EVENT)
#: the timeline prints every schema'd event, so a per-box read is never missing
#: the one row that explains the rest.
TIMELINE_EVENTS = tuple(sorted(EVENT_SCHEMA))

TOUCHED_EVENTS = frozenset(
    (SELF_FLOOR_EVENT, FLOOR_BLIND_EVENT, EVICTED_EVENT)
    + REFUSAL_EVENTS + EVICTION_OUTCOME_EVENTS + WATCH_EVENTS + NOTIFY_EVENTS
    + TIMELINE_EVENTS)

#: the notification row types the report breaks out by name. Everything else is
#: counted under its own slug in `by_type` — the catalog has 30 keys and we have
#: only ever observed 3, so an unknown type is the expected case, not an error.
NOTIFY_OUTBID_TYPE = "outbid"

#: the "stop" action on a replacement decision IS the replacement refusal; the
#: same event with action="rent" is an acceptance and is reported separately.
REPLACEMENT_REFUSAL_ACTION = "stop"

#: eviction outcome precedence, most terminal first. `rescued` outranks
#: `replaced` because a box that came back needed no replacement; `refused`
#: outranks `rebid` because a rung followed by a refusal ended in the refusal.
OUTCOME_PRECEDENCE = ("rescued", "replaced", "refused", "rebid", "unresolved")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def journal_path(path=None):
    """Default `~/.local/state/vast-fleetd/journal.ndjsonl`; FLEETD_STATE_DIR is
    honoured so a soak/test state dir reports on its own journal."""
    if path:
        return os.path.expanduser(path)
    state_dir = os.environ.get("FLEETD_STATE_DIR")
    if state_dir:
        return os.path.join(os.path.expanduser(state_dir), "journal.ndjsonl")
    return os.path.expanduser(DEFAULT_JOURNAL)


def _now():
    return datetime.now(timezone.utc).timestamp()


def parse_since(spec, now=None):
    """`--since` -> epoch seconds. Accepts `Nd`/`Nh`/`Nm` back from now, or an
    ISO date/datetime (`2026-08-10`, `2026-08-10T12:00:00Z`). Returns None for an
    empty spec; raises ValueError on anything else — a mistyped window that
    silently reported ALL of history would be worse than an error."""
    if spec in (None, ""):
        return None
    s = str(spec).strip()
    now = float(now if now is not None else _now())
    if len(s) > 1 and s[-1] in "dhm":
        try:
            n = float(s[:-1])
        except ValueError:
            raise ValueError(
                f"bad --since {spec!r}: expected Nd/Nh/Nm or an ISO date")
        return now - n * {"d": 86400.0, "h": 3600.0, "m": 60.0}[s[-1]]
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(
            f"bad --since {spec!r}: expected Nd/Nh/Nm or an ISO date")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _row_boxes(rec):
    """Every box id a row is ABOUT. `iid` is the subject and `target` the watch
    it belongs to — they differ (a floor-blind row on 47393050 targets watch
    47389375) — and a replacement names both its old and its new box."""
    out = []
    for key in ("iid", "target", "from_box", "to_box"):
        v = rec.get(key)
        if v not in (None, ""):
            out.append(str(v))
    return out


def load_events(path=None, since=None, box=None):
    """Read the journal into `(events, schema_report)`.

    Never raises on content: an unparseable line, a row with no `event`, an event
    name absent from EVENT_SCHEMA, and a row missing a required field are each
    COUNTED into the schema report and the read continues."""
    p = journal_path(path)
    schema = {
        "journal": p,
        "lines": 0,
        "malformed_lines": 0,
        "rows_without_event": 0,
        "unknown_events": Counter(),
        "missing_fields": Counter(),
        "rows_in_window": 0,
        "rows_before_since": 0,
        "rows_other_box": 0,
        "known_rows": 0,
    }
    events = []
    if not os.path.exists(p):
        schema["error"] = f"no journal at {p}"
        schema["unknown_events"] = {}
        schema["missing_fields"] = {}
        return events, schema
    box = str(box) if box not in (None, "") else None
    with open(p, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            schema["lines"] += 1
            try:
                rec = json.loads(line)
            except ValueError:
                schema["malformed_lines"] += 1
                continue
            if not isinstance(rec, dict):
                schema["malformed_lines"] += 1
                continue
            name = rec.get("event")
            if not name:
                schema["rows_without_event"] += 1
                continue
            spec = EVENT_SCHEMA.get(name)
            if spec is None:
                # `tick` and the rest of the daemon vocabulary: known to exist,
                # not read by this report. Counted, so a NEW event that ought to
                # be here is visible rather than invisible.
                schema["unknown_events"][name] += 1
                continue
            try:
                ts = float(rec.get("ts"))
            except (TypeError, ValueError):
                schema["missing_fields"][f"{name}.ts"] += 1
                continue
            if since is not None and ts < since:
                schema["rows_before_since"] += 1
                continue
            if box is not None and box not in _row_boxes(rec):
                schema["rows_other_box"] += 1
                continue
            missing = [f for f in sorted(spec.required) if rec.get(f) is None]
            for f in missing:
                schema["missing_fields"][f"{name}.{f}"] += 1
            rec["_ts"] = ts
            rec["_degraded"] = bool(missing)
            events.append(rec)
            schema["known_rows"] += 1
    events.sort(key=lambda r: r["_ts"])
    schema["rows_in_window"] = len(events)
    schema["unknown_events"] = dict(schema["unknown_events"])
    schema["missing_fields"] = dict(schema["missing_fields"])
    if events:
        schema["window"] = {
            "first": events[0].get("ts_iso"),
            "last": events[-1].get("ts_iso"),
            "span_s": round(events[-1]["_ts"] - events[0]["_ts"], 1),
        }
    return events, schema


def _of(events, *names):
    """Rows of these event names that carry every required field. A degraded row
    is already counted in `schema.missing_fields`; using it would put a
    half-known row into a distribution."""
    want = set(names)
    return [e for e in events
            if e.get("event") in want and not e.get("_degraded")]


# --------------------------------------------------------------------------- #
# (a) self-floor suppression ages
# --------------------------------------------------------------------------- #
def _pct(sorted_vals, q):
    """Nearest-rank percentile — no interpolation, so every number reported is a
    number the fleet actually observed. (An interpolated p90 of a CENSORED sample
    invents a value on the side of the distribution we cannot see.)"""
    if not sorted_vals:
        return None
    return sorted_vals[max(1, math.ceil(q * len(sorted_vals))) - 1]


def self_floor_ages(events, lag_s=None):
    """Suppression age distribution BY MATCH KIND, with the censoring warning.

    `matched="standing"` should sit at age 0 (a price that still stands never
    ages); `matched="prior"` is the echo of a bid we have since moved, and its
    tail is what the window has to cover. Rows predating `matched_age_s` are
    counted under `no_age_rows` rather than dropped."""
    lag_s = float(lag_s if lag_s is not None
                  else bidpolicy.BID_SELF_FLOOR_LAG_S)
    by_kind = defaultdict(list)
    no_age = Counter()
    boxes = set()
    rows = _of(events, SELF_FLOOR_EVENT)
    for e in rows:
        kind = e.get("matched") or "unspecified"
        boxes.update(_row_boxes(e))
        try:
            # float() FIRST: `by_kind[kind]` is a defaultdict access and would
            # mint an empty bucket for a row that turns out to carry no age.
            age = float(e.get("matched_age_s"))
        except (TypeError, ValueError):
            no_age[kind] += 1
        else:
            by_kind[kind].append(age)
    out = {}
    overall = []
    for kind, vals in by_kind.items():
        vals.sort()
        overall.extend(vals)
        out[kind] = {"count": len(vals), "min": vals[0],
                     "median": _pct(vals, 0.5), "p90": _pct(vals, 0.9),
                     "max": vals[-1]}
    for kind, n in no_age.items():
        out.setdefault(kind, {"count": 0, "min": None, "median": None,
                              "p90": None, "max": None})
        out[kind]["no_age_rows"] = n
    overall.sort()
    res = {
        "events": len(rows),
        "boxes": len(boxes),
        "lag_s": lag_s,
        "by_kind": out,
        "no_age_rows": sum(no_age.values()),
        "overall": ({"count": len(overall), "median": _pct(overall, 0.5),
                     "p90": _pct(overall, 0.9), "max": overall[-1]}
                    if overall else None),
        "floor_blind_episodes": len(_of(events, FLOOR_BLIND_EVENT)),
        "censored": False,
        "warning": None,
    }
    if overall and overall[-1] >= CENSORING_FRACTION * lag_s:
        res["censored"] = True
        res["warning"] = (
            f"CENSORED: max suppression age {overall[-1]:.1f}s is "
            f"{100.0 * overall[-1] / lag_s:.1f}% of BID_SELF_FLOOR_LAG_S "
            f"({lag_s:.0f}s). An echo older than the window reads as a MARKET "
            f"floor and never journals as a match, so this sample CANNOT show "
            f"an age above the window: a max at the edge means the true tail "
            f"almost certainly crosses it. Widen by a MULTIPLE, not a margin "
            f"(AUTOBID_DESIGN 'The field data answered'), or measure the echo "
            f"outright with an active probe (FLEET_REVIEW item 3).")
    return res


# --------------------------------------------------------------------------- #
# (b) refusal episodes
# --------------------------------------------------------------------------- #
def normalize_reason(reason):
    """Collapse a reason string to its CLASS by blanking the numbers it embeds.
    `rung 1/1: $0.64 -> $0.8000 (ceiling $0.930...)` and the same refusal at
    other prices are ONE reason, which is the whole point: 158 events announced
    2 facts, and only a class-level grouping shows that."""
    if reason in (None, ""):
        return "(none)"
    out, in_num = [], False
    for ch in str(reason):
        if ch.isdigit() or (ch == "." and in_num):
            if not in_num:
                out.append("N")
            in_num = True
        else:
            in_num = False
            out.append(ch)
    return "".join(out).strip()


def _refusal_family(e):
    name = e.get("event")
    if name == REBID_REFUSED_EVENT:
        return "rebid"
    if name == REPLACEMENT_DECISION_EVENT:
        return ("replacement"
                if e.get("action") == REPLACEMENT_REFUSAL_ACTION else None)
    return None


def refusal_episodes(events, gap_s=REFUSAL_EPISODE_GAP_S):
    """Rebid + replacement refusals grouped by reason CLASS: raw events against
    distinct episodes. The ratio is the review's spam detector — the ladder
    re-decides every tick by design, but ANNOUNCING every re-decision is a defect
    (fixed 2026-08-14: both sites now journal on reason change)."""
    rows = []
    for e in _of(events, *REFUSAL_EVENTS):
        fam = _refusal_family(e)
        if fam:
            rows.append((fam, e))
    groups = defaultdict(list)
    for fam, e in rows:
        groups[(fam, normalize_reason(e.get("reason")))].append(e)

    out = []
    for (fam, reason), rs in groups.items():
        per_box = defaultdict(list)
        for e in rs:
            per_box[str(e.get("iid"))].append(e)
        episodes = []
        for iid, box_rows in per_box.items():
            box_rows.sort(key=lambda r: r["_ts"])
            cur = [box_rows[0]]
            for prev, nxt in zip(box_rows, box_rows[1:]):
                if nxt["_ts"] - prev["_ts"] > gap_s:
                    episodes.append((iid, cur))
                    cur = [nxt]
                else:
                    cur.append(nxt)
            episodes.append((iid, cur))
        worst = max(episodes, key=lambda ep: len(ep[1]))
        out.append({
            "family": fam,
            "reason": reason,
            "sample_reason": str(rs[0].get("reason") or "")[:200],
            "events": len(rs),
            "episodes": len(episodes),
            "boxes": sorted(per_box),
            "worst_episode": {
                "iid": worst[0],
                "events": len(worst[1]),
                "span_s": round(worst[1][-1]["_ts"] - worst[1][0]["_ts"], 1),
                "first": worst[1][0].get("ts_iso"),
            },
        })
    out.sort(key=lambda r: (-r["events"], r["family"], r["reason"]))
    accepted = [e for e in _of(events, REPLACEMENT_DECISION_EVENT)
                if e.get("action") != REPLACEMENT_REFUSAL_ACTION]
    tot_e = sum(r["events"] for r in out)
    tot_ep = sum(r["episodes"] for r in out)
    return {
        "reasons": out,
        "events": tot_e,
        "episodes": tot_ep,
        "amplification": (round(tot_e / tot_ep, 1) if tot_ep else None),
        "replacements_accepted": len(accepted),
    }


# --------------------------------------------------------------------------- #
# (c) evictions by class and outcome
# --------------------------------------------------------------------------- #
#: an eviction episode is CLOSED by one of these outcomes — the box is back, or
#: a successor is running. A later eviction row after one of these is a NEW
#: eviction, however few seconds later it lands.
TERMINAL_OUTCOMES = ("rescued", "replaced")


def _eviction_episodes(rows, resolvers, gap_s, window_s):
    """One box's eviction rows -> episodes, each with the outcome signals seen
    inside it.

    Three things start a new episode, and all three are needed: the gap (a
    re-announcement of a still-stuck eviction is not a second eviction), a
    TERMINAL outcome since the last announcement (47398836 on 2026-08-10 was
    evicted, rescued, evicted, rescued, evicted 17 min apart — one gap-merged
    episode would report a 3-in-1 as a single rescue), and a change of
    `eviction_class` (a host_stop and an outbid are different events by
    definition, whatever their spacing)."""
    merged = ([(r["_ts"], "evicted", r) for r in rows]
              + [(ts, tag, None) for ts, tag in resolvers])
    merged.sort(key=lambda t: (t[0], t[1] != "evicted"))
    episodes = []
    cur = None
    last_evicted_ts = None
    terminal_since = False
    for ts, tag, rec in merged:
        if tag == "evicted":
            klass = rec.get("eviction_class") or "unknown"
            if (cur is None or terminal_since
                    or ts - last_evicted_ts > gap_s
                    or klass != cur["eviction_class"]):
                cur = {"rows": [rec], "eviction_class": klass, "signals": set()}
                episodes.append(cur)
            else:
                cur["rows"].append(rec)
            last_evicted_ts = ts
            terminal_since = False
        else:
            if cur is None or ts - last_evicted_ts > window_s:
                continue          # a resolver with no eviction to resolve
            cur["signals"].add(tag)
            if tag in TERMINAL_OUTCOMES:
                terminal_since = True
    return episodes


def eviction_outcomes(events, gap_s=EVICTION_EPISODE_GAP_S,
                      window_s=EVICTION_OUTCOME_WINDOW_S):
    """Eviction episodes per box, each resolved to ONE outcome.

    Outcome signals are the ladder events on the SAME box inside the episode,
    folded through OUTCOME_PRECEDENCE. `unresolved` is a real answer, not a hole
    in the data: it is an eviction the ladder never closed inside `window_s`,
    which is precisely the row an operator wants surfaced."""
    per_box = defaultdict(list)
    for e in _of(events, EVICTED_EVENT):
        per_box[str(e.get("iid"))].append(e)

    resolvers = defaultdict(list)          # box -> [(ts, outcome_tag)]
    for e in _of(events, *EVICTION_OUTCOME_EVENTS):
        name = e.get("event")
        if name == REPLACED_EVENT:
            key, tag = str(e.get("from_box")), "replaced"
        elif name == SURVIVED_EVENT:
            key, tag = str(e.get("iid")), "rescued"
        elif name == REBID_RUNG_EVENT:
            key, tag = str(e.get("iid")), "rebid"
        elif name == REBID_REFUSED_EVENT:
            key, tag = str(e.get("iid")), "refused"
        else:                              # replacement decision
            if e.get("action") != REPLACEMENT_REFUSAL_ACTION:
                continue
            key, tag = str(e.get("iid")), "refused"
        resolvers[key].append((e["_ts"], tag))

    rows_out, by_class = [], defaultdict(Counter)
    raw_by_class = Counter()
    for iid, rows in sorted(per_box.items()):
        rows.sort(key=lambda r: r["_ts"])
        for e in rows:
            raw_by_class[e.get("eviction_class") or "unknown"] += 1
        for ep in _eviction_episodes(rows, sorted(resolvers.get(iid, ())),
                                     gap_s, window_s):
            outcome = next((o for o in OUTCOME_PRECEDENCE
                            if o in ep["signals"]), "unresolved")
            by_class[ep["eviction_class"]][outcome] += 1
            rows_out.append({
                "iid": iid,
                "eviction_class": ep["eviction_class"],
                "outcome": outcome,
                "at": ep["rows"][0].get("ts_iso"),
                "announcements": len(ep["rows"]),
                "span_s": round(ep["rows"][-1]["_ts"] - ep["rows"][0]["_ts"], 1),
                "signals": sorted(ep["signals"]),
            })
    rows_out.sort(key=lambda r: (r["at"] or "", r["iid"]))
    return {
        "episodes": len(rows_out),
        "raw_events": sum(raw_by_class.values()),
        "boxes": len(per_box),
        "by_class": {k: dict(v) for k, v in sorted(by_class.items())},
        "raw_by_class": dict(raw_by_class),
        "by_outcome": dict(Counter(r["outcome"] for r in rows_out)),
        "rows": rows_out,
    }


# --------------------------------------------------------------------------- #
# (d) per-box ladder timeline
# --------------------------------------------------------------------------- #
def _timeline_detail(e):
    spec = EVENT_SCHEMA.get(e.get("event"))
    if spec is None:
        return ""
    bits = []
    for k in sorted((spec.required | spec.optional) - {"iid", "target"}):
        v = e.get(k)
        if v in (None, "", [], {}):
            continue
        if isinstance(v, float):
            v = f"{v:.4g}"
        s = str(v)
        bits.append(f"{k}={s[:87] + '...' if len(s) > 90 else s}")
    return " ".join(bits)


def box_timeline(events, box):
    """Every schema'd row touching one box, in order — the review's `--box` loop.
    Deliberately NOT narrowed to the ladder: the row that explains a weird rung
    is usually the watch or eviction row sitting next to it."""
    box = str(box)
    rows = []
    for e in events:
        if e.get("event") not in EVENT_SCHEMA or box not in _row_boxes(e):
            continue
        rows.append({"ts": e["_ts"], "ts_iso": e.get("ts_iso"),
                     "event": e.get("event"), "iid": str(e.get("iid") or "-"),
                     "detail": _timeline_detail(e),
                     "degraded": bool(e.get("_degraded"))})
    return {"box": box, "events": len(rows), "rows": rows}


# --------------------------------------------------------------------------- #
# (e) watch lifecycle
# --------------------------------------------------------------------------- #
def watch_lifecycle(events):
    """Drain / lapse / dormant counts.

    The LAPSED cycle (FLEET_REVIEW item 2) is a POLICY watch finishing — usually
    `drained` — and the box then being auto-adopted `bare`: still observed and
    still accruing against its ceiling, but with NO armed ladder until someone
    re-registers a watch. Counting the re-adoptions is the only way to see how
    often the fleet is running on a bare watch nobody chose."""
    finished, registered = Counter(), Counter()
    auto_adopted, dormant = Counter(), Counter()
    lapses, pending = [], {}
    for e in events:
        name = e.get("event")
        if e.get("_degraded"):
            continue
        if name == WATCH_FINISHED_EVENT:
            finished[e.get("verdict") or "unknown"] += 1
            if (e.get("profile") or "bare") != "bare":
                pending[str(e.get("iid"))] = e
        elif name == WATCH_REGISTERED_EVENT:
            registered[e.get("profile")] += 1
            pending.pop(str(e.get("iid")), None)
        elif name == WATCH_AUTO_ADOPTED_EVENT:
            auto_adopted[e.get("profile")] += 1
            prior = pending.pop(str(e.get("iid")), None)
            if prior is not None and e.get("profile") == "bare":
                lapses.append({
                    "iid": str(e.get("iid")),
                    "from_profile": prior.get("profile"),
                    "verdict": prior.get("verdict"),
                    "finished": prior.get("ts_iso"),
                    "adopted": e.get("ts_iso"),
                    "gap_s": round(e["_ts"] - prior["_ts"], 1),
                })
        elif name == WATCH_DORMANT_EVENT:
            dormant[e.get("reason") or "unknown"] += 1
    return {
        "finished_by_verdict": dict(finished.most_common()),
        "drained": finished.get("drained", 0),
        "registered_by_profile": dict(registered.most_common()),
        "auto_adopted_by_profile": dict(auto_adopted.most_common()),
        "dormant_by_reason": dict(dormant.most_common()),
        "bare_adoptions_after_policy_watch": len(lapses),
        "lapses": lapses,
    }


# --------------------------------------------------------------------------- #
# (f) vast's notification channel (NOTIFY_DESIGN S2a)
# --------------------------------------------------------------------------- #
def notifications(events):
    """What the notification feed said, and whether we heard all of it.

    The one number here that exists nowhere else in the fleet's telemetry is the
    **displacement spread**: an outbid row's `new_min_bid` is the price that
    actually took the box, read from vast rather than inferred from a listing
    that echoes our own bid back (FLEET_REVIEW item 3). Reporting it per machine
    is what turns "we keep losing this host" into a price.

    Two shapes are reported and NOT hidden: rows whose `new_min_bid` is at or
    BELOW our bid (real — §1.4 — and not an outbid we could have won by bidding
    more), and gaps/poll-error episodes, which bound how much of the feed the
    rest of this section is actually summarizing."""
    seen = _of(events, NOTIFY_SEEN_EVENT)
    by_type, by_machine = Counter(), defaultdict(list)
    spreads, below, boxes = [], [], set()
    for e in seen:
        kind = e.get("notif_type") or "unknown"
        by_type[kind] += 1
        if e.get("iid"):
            boxes.add(str(e["iid"]))
        if kind != NOTIFY_OUTBID_TYPE:
            continue
        assoc = e.get("associated_id")
        assoc = assoc if isinstance(assoc, dict) else {}
        yb, nmb = _num_field(assoc.get("your_bid")), _num_field(assoc.get("new_min_bid"))
        mid = assoc.get("machine_id", e.get("machine_id"))
        row = {"iid": str(e.get("iid") or "-"), "machine_id": mid,
               "your_bid": yb, "new_min_bid": nmb, "at": e.get("ts_iso")}
        if mid is not None:
            by_machine[str(mid)].append(row)
        if yb is not None and nmb is not None:
            row["spread"] = round(nmb - yb, 4)
            row["multiple"] = round(nmb / yb, 3) if yb else None
            spreads.append(row)
            if nmb <= yb:
                below.append(row)

    ranked = sorted(spreads, key=lambda r: r["spread"], reverse=True)
    vals = sorted(r["spread"] for r in spreads)
    gaps = _of(events, NOTIFY_GAP_EVENT)
    health = _of(events, NOTIFY_POLL_ERROR_EVENT)
    episodes = [e for e in health if e.get("state") == "failing"]
    return {
        "rows": len(seen),
        "by_type": dict(by_type.most_common()),
        "boxes": len(boxes),
        "outbids": by_type.get(NOTIFY_OUTBID_TYPE, 0),
        "outbids_by_machine": {m: len(rows) for m, rows in
                               sorted(by_machine.items(),
                                      key=lambda kv: -len(kv[1]))},
        "priced": len(spreads),
        "spread_median": _pct(vals, 0.5) if vals else None,
        "spread_max": vals[-1] if vals else None,
        "worst": ranked[:5],
        "below_our_bid": below,
        "gaps": len(gaps),
        "poll_error_episodes": len(episodes),
        "poll_last_state": health[-1].get("state") if health else None,
        "endpoint_gone": any(e.get("gone") for e in episodes),
    }


def _num_field(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def build_report(path=None, since=None, box=None, now=None):
    """The whole review as one dict. `since` is an epoch float or a `--since`
    spec string; `box` restricts EVERY aggregate and adds the timeline."""
    if isinstance(since, str):
        since = parse_since(since, now=now)
    events, schema = load_events(path, since=since, box=box)
    rep = {
        "schema": schema,
        "since": (datetime.fromtimestamp(since, timezone.utc)
                  .strftime("%Y-%m-%dT%H:%M:%SZ") if since else None),
        "box": str(box) if box else None,
        "self_floor": self_floor_ages(events),
        "refusals": refusal_episodes(events),
        "evictions": eviction_outcomes(events),
        "watches": watch_lifecycle(events),
        "notifications": notifications(events),
    }
    if box:
        rep["timeline"] = box_timeline(events, box)
    return rep


def _fmt(v, unit="s"):
    return "-" if v is None else f"{v:.1f}{unit}"


def render_text(rep, out=None):
    """Operator-facing rendering. Every section prints its denominator: a review
    that says "3 refusals" without saying "3 of how many events" is how the
    158-events-for-2-facts spam survived two review passes."""
    w = (out or sys.stdout).write
    sch = rep["schema"]
    w(f"fleet report — {sch.get('journal')}\n")
    if sch.get("error"):
        w(f"  !! {sch['error']}\n")
        return
    win = sch.get("window") or {}
    w(f"  window: {rep.get('since') or 'all history'} -> now   "
      f"rows {sch['rows_in_window']} of {sch['lines']} journal lines"
      + (f"   ({win.get('first')} .. {win.get('last')})" if win else "") + "\n")
    if rep.get("box"):
        w(f"  box filter: {rep['box']}\n")

    sf = rep["self_floor"]
    w(f"\n(a) SELF-FLOOR SUPPRESSION — {sf['events']} events, {sf['boxes']} "
      f"box(es), window BID_SELF_FLOOR_LAG_S={sf['lag_s']:.0f}s\n")
    if not sf["events"]:
        w("    none in window\n")
    else:
        w(f"    {'match kind':<14}{'n':>5}{'median':>10}{'p90':>10}{'max':>10}\n")
        for kind, st in sorted(sf["by_kind"].items()):
            w(f"    {kind:<14}{st['count']:>5}{_fmt(st['median']):>10}"
              f"{_fmt(st['p90']):>10}{_fmt(st['max']):>10}"
              + (f"   (+{st['no_age_rows']} row(s) with no age)"
                 if st.get("no_age_rows") else "") + "\n")
        if sf["floor_blind_episodes"]:
            w(f"    floor-blind alarms (sustained full suppression): "
              f"{sf['floor_blind_episodes']}\n")
        if sf["warning"]:
            w(f"    !! {sf['warning']}\n")

    rf = rep["refusals"]
    w(f"\n(b) REFUSALS — {rf['events']} events / {rf['episodes']} distinct "
      f"episodes"
      + (f" (x{rf['amplification']} announcement amplification)"
         if rf["amplification"] else "")
      + f"; {rf['replacements_accepted']} replacement decision(s) ACCEPTED\n")
    if not rf["reasons"]:
        w("    none in window\n")
    for r in rf["reasons"]:
        w(f"    {r['family']:<12}{r['events']:>5} ev {r['episodes']:>4} ep  "
          f"{len(r['boxes'])} box(es)\n")
        # WRAPPED, not truncated: the two ceiling refusals differ only in
        # their tail ("the binding bound is the HARD on-demand ceiling" vs
        # "the JOB-AWARE defense ceiling"), and that difference is the whole
        # finding — a 110-char cut renders them as two identical rows.
        for ln in textwrap.wrap(r["reason"], 96, max_lines=4,
                                placeholder=" ..."):
            w(f"        {ln}\n")
        we = r["worst_episode"]
        if we["events"] > 1:
            w(f"        worst: {we['events']} events over "
              f"{we['span_s'] / 60.0:.0f} min on {we['iid']} "
              f"from {we['first']}\n")

    ev = rep["evictions"]
    w(f"\n(c) EVICTIONS — {ev['episodes']} episodes ({ev['raw_events']} raw "
      f"announcements) across {ev['boxes']} box(es)\n")
    if ev["episodes"]:
        w(f"    {'class':<20}"
          + "".join(f"{o:>12}" for o in OUTCOME_PRECEDENCE) + "\n")
        for klass, counts in ev["by_class"].items():
            w(f"    {klass:<20}"
              + "".join(f"{counts.get(o, 0):>12}" for o in OUTCOME_PRECEDENCE)
              + "\n")
        w(f"    {'TOTAL':<20}"
          + "".join(f"{ev['by_outcome'].get(o, 0):>12}"
                    for o in OUTCOME_PRECEDENCE) + "\n")
        for r in ev["rows"]:
            if r["outcome"] == "unresolved":
                w(f"      UNRESOLVED {r['iid']:<12}{r['eviction_class']:<20}"
                  f"{r['at']}\n")

    tl = rep.get("timeline")
    if tl is not None:
        w(f"\n(d) LADDER TIMELINE {tl['box']} — {tl['events']} events\n")
        for r in tl["rows"]:
            w(f"    {str(r['ts_iso']):<21}{r['event']:<32}{r['detail']}\n")

    wl = rep["watches"]
    w(f"\n(e) WATCH LIFECYCLE — {sum(wl['finished_by_verdict'].values())} "
      f"watch(es) finished\n")
    for verdict, n in wl["finished_by_verdict"].items():
        w(f"    finished {verdict:<22}{n:>6}\n")
    for profile, n in wl["auto_adopted_by_profile"].items():
        w(f"    auto-adopted {str(profile):<18}{n:>6}\n")
    for reason, n in wl["dormant_by_reason"].items():
        w(f"    dormant {reason:<23}{n:>6}\n")
    w(f"    LAPSED cycles (policy watch ended -> bare auto-adoption): "
      f"{wl['bare_adoptions_after_policy_watch']}\n")
    if wl["bare_adoptions_after_policy_watch"]:
        w("      each is a box left with a surviving ceiling and NO armed "
          "ladder until a watch is re-registered (FLEET_REVIEW item 2)\n")

    nt = rep["notifications"]
    w(f"\n(f) NOTIFICATIONS — {nt['rows']} row(s) consumed from vast's inbox "
      f"across {nt['boxes']} box(es); {nt['outbids']} outbid(s)\n")
    if not nt["rows"]:
        w("    none in window\n")
    else:
        w("    by type: "
          + ", ".join(f"{k}x{v}" for k, v in nt["by_type"].items()) + "\n")
    if nt["priced"]:
        w(f"    displacing price ({nt['priced']} priced outbid(s)): "
          f"spread median {_fmt(nt['spread_median'], '')} "
          f"max {_fmt(nt['spread_max'], '')} $/hr over our bid\n")
        for r in nt["worst"]:
            w(f"      {r['iid']:<12}machine {str(r['machine_id']):<10}"
              f"${r['your_bid']:.2f} -> ${r['new_min_bid']:.2f}"
              f"  (x{r['multiple']})   {r['at']}\n")
    if nt["outbids_by_machine"]:
        top = list(nt["outbids_by_machine"].items())[:8]
        w("    outbids by machine: "
          + ", ".join(f"{m}x{n}" for m, n in top)
          + (", ..." if len(nt["outbids_by_machine"]) > len(top) else "") + "\n")
    for r in nt["below_our_bid"]:
        # §1.4: displacement BELOW our bid is real (on-demand taker or host
        # action). Printed loud because reading it as an ordinary outbid would
        # invite a rescue bid that cannot win the box back.
        w(f"    !! displaced BELOW our bid: {r['iid']} machine "
          f"{r['machine_id']} ${r['your_bid']:.2f} -> ${r['new_min_bid']:.2f} "
          f"— not an outbid we could have won by bidding more\n")
    if nt["gaps"]:
        w(f"    !! {nt['gaps']} gap(s): a full inbox window with nothing known "
          f"in it — rows above are a LOWER BOUND on what the feed carried\n")
    if nt["poll_error_episodes"]:
        w(f"    poll-error episodes: {nt['poll_error_episodes']}"
          + (f" (last state: {nt['poll_last_state']})"
             if nt["poll_last_state"] else "")
          + ("   ENDPOINT GONE — the hidden inbox was retired; fleetd is back "
             "to pre-notify behavior" if nt["endpoint_gone"] else "") + "\n")

    if (sch["malformed_lines"] or sch["rows_without_event"]
            or sch["missing_fields"]):
        w("\nSCHEMA\n")
        if sch["malformed_lines"]:
            w(f"    unparseable lines: {sch['malformed_lines']}\n")
        if sch["rows_without_event"]:
            w(f"    rows with no `event`: {sch['rows_without_event']}\n")
        for key, n in sorted(sch["missing_fields"].items()):
            w(f"    missing field {key}: {n} row(s)\n")
    if sch["unknown_events"]:
        top = sorted(sch["unknown_events"].items(), key=lambda kv: -kv[1])
        w(f"\n    events not read by this report ({len(top)} names, "
          f"{sum(sch['unknown_events'].values())} rows): "
          + ", ".join(f"{k}x{v}" for k, v in top[:8])
          + (", ..." if len(top) > 8 else "") + "\n")


def add_args(p):
    """Flags, shared by the standalone CLI and `herdd fleet report`."""
    p.add_argument("--since", default=None, metavar="WHEN",
                   help="ISO date/datetime, or Nd/Nh/Nm back from now "
                        "(e.g. 4d, 36h, 2026-08-10)")
    p.add_argument("--box", default=None, metavar="IID",
                   help="restrict every aggregate to ONE box and print its "
                        "full ladder timeline")
    p.add_argument("--journal", default=None, metavar="PATH",
                   help=f"journal file (default {DEFAULT_JOURNAL})")
    p.add_argument("--json", action="store_true",
                   help="emit the whole report as JSON")
    return p


def run(a):
    """argparse namespace -> exit code. `herdd fleet report` calls this."""
    try:
        since = parse_since(getattr(a, "since", None))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rep = build_report(path=getattr(a, "journal", None), since=since,
                       box=getattr(a, "box", None))
    if getattr(a, "json", False):
        print(json.dumps(rep, indent=2, sort_keys=True, default=str))
    else:
        render_text(rep)
    return 1 if rep["schema"].get("error") else 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="fleet_report",
        description="aggregate the fleetd journal (the 2026-08-14 fleet "
                    "review, as a command)")
    add_args(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
