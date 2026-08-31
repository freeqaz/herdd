#!/usr/bin/env python3
"""notify — vast's notification channel, the PURE half.

Design of record + verified ground truths: `NOTIFY_DESIGN.md`. This module owns
everything that can be decided without touching the network: which rows a poll
has never seen (the cursor, D3), when a poll implies a labeling HOLE (D4's
`notify_gap`), and how the three read-only views render. `herdd notify` does
the HTTP and prints these strings; `fleetd`'s reconcile tick does the HTTP and
journals what `poll()` returns. Neither owns any of the logic below, so both are
testable against the same captured rows and neither can drift from the other.

Three properties are load-bearing, all three because the feed is a HIDDEN,
undocumented endpoint carrying rows written by a service we do not control:

  * **Nothing here raises on content.** An unknown `notif_type`, a missing
    `associated_id`, a string where a float belongs, extra keys we have never
    seen — each degrades to "less detail", never to an exception. A tick that
    died on an unexpected notification row would take the whole fleet's
    reconcile with it, and these rows are EVIDENCE ONLY (D2): they may never
    cost us a reconcile pass.
  * **The cursor is ours, and we never PUT `seen_through_at`** (D3). Server-side
    seen-state is useless for dedup under at-least-once delivery, and the
    console UI may someday depend on it.
  * **`created_at` is a floor, `event_id` is the authority.** The id set is
    bounded at `RECENT_IDS_MAX` (4x the observed 50-row window), so every row
    currently in the window is covered by ids alone; `last_created_at` exists
    for the case where the id set has rolled or been lost, so a truncated
    cursor can never resurrect three days of history as "new".
"""
from __future__ import annotations

from collections import namedtuple

# --------------------------------------------------------------------------- #
# endpoints (NOTIFY_DESIGN §1)
# --------------------------------------------------------------------------- #
#: HIDDEN: commented out of vast's published OpenAPI spec ("no console UI yet"),
#: i.e. undocumented and revocable. Every caller must degrade on 404 (D2).
INBOX_PATH = "v0/notifications/inbox/"
TYPES_PATH = "v0/notification-types"
WEBHOOKS_PATH = "v0/webhooks/"

#: Observed server-side inbox window: exactly 50 rows (~3 days, 2026-08-16).
#: A poll that returns this many rows of which NONE were known is the gap
#: signal — see `poll()`.
WINDOW = 50

#: Bounded recent-`event_id` set carried in the cursor (D3). 4x the window, so
#: the entire current feed is always covered by ids; growing it buys nothing.
RECENT_IDS_MAX = 200

#: How far BELOW the high-water `created_at` an unseen `event_id` may sit and
#: still count as new. Rows are stamped by the producing service, so a poll can
#: legitimately deliver a row a little older than the newest one we already
#: have. Beyond this we treat an unseen id as history the cursor already passed
#: (its id having rolled out of the bounded set), not as a new event.
REORDER_SLOP_S = 300.0

#: vast's documented max webhooks per user. Printed by `render_webhooks` so
#: "0 webhooks" reads as headroom rather than as an error.
MAX_WEBHOOKS = 4

#: The unprefixed slug of the one type S2b acts on. `full_key()` makes it
#: `client:outbid`; the inbox ships the bare form.
OUTBID_TYPE = "outbid"

#: How far a row's `created_at` may sit from the tick that observed the stop and
#: still describe THAT stop (S2b, §6.3). +/-, because a row can land either side
#: of the observation: the poll runs once per reconcile tick, and the classifier
#: may reach an eviction a tick before or after the row is journaled.
#:
#: 15 minutes is not a round number picked for comfort — the captured feed holds
#: instance 47833510 evicted TWICE (02:40:08Z and 04:48:47Z, bids $0.96 and
#: $1.20). Any window wide enough to glue those two together would let cycle N's
#: row label cycle N+1, which is a wrong price on a money-moving rung. The
#: consumed-id exclusion (`exclude_ids`) covers the tighter case the window
#: cannot: a box evicted, rescued and re-evicted inside one window.
FRESH_WINDOW_S = 900.0

#: How long fleetd retains a consumed outbid row as matchable evidence, and how
#: many. Retention is deliberately wider than the freshness window (a row must
#: still be there when a stop is observed at the far edge of it) and the count
#: is small: this is a lookaside for evictions in flight, not a second copy of
#: the feed — the journal already holds every row verbatim (D4).
RETAIN_S = 3 * FRESH_WINDOW_S
RETAIN_MAX = 32

# --------------------------------------------------------------------------- #
# journal event names (S2a, S2b)
# --------------------------------------------------------------------------- #
# Named HERE, in the leaf both writer and reader import, so `fleetd`'s emit and
# `fleet_report`'s schema pin cannot drift apart on a rename — the failure the
# pin exists to catch, made structurally impossible for these.
SEEN_EVENT = "notify_seen"            #: one consumed row (D4)
GAP_EVENT = "notify_gap"              #: a full window, none of it known (D4)
POLL_ERROR_EVENT = "notify_poll_error"  #: poll health CHANGED (never per-tick)
#: S2b (§6.3): a row was matched to an eviction. Carries BOTH classifications —
#: with and without the row — because the whole question S2b exists to answer in
#: the field is how often the notification changes the verdict.
MATCHED_EVENT = "notify_outbid_matched"
#: S2b (§6.3): vast's record of our standing bid disagrees with ours.
#: JOURNAL-ONLY, forever: belief reconciliation has exactly one writer.
BID_MISMATCH_EVENT = "notify_bid_mismatch"
#: S2b (§6.4): the rescue rung priced off a row. `emitted: null` means the rails
#: refused the quote — which is the outcome that proves they still bind.
RESCUE_QUOTE_EVENT = "notify_rescue_quote"
#: S2b (§6.5): the offers-listing floor at the stop beside the authoritative
#: displacing price. JOURNAL-ONLY — it calibrates the echo guard, it is not one.
FLOOR_CHECK_EVENT = "notify_floor_check"


# --------------------------------------------------------------------------- #
# tolerant row accessors
# --------------------------------------------------------------------------- #
def inbox_rows(payload):
    """The `notifications` list out of an inbox payload — [] for anything else.

    Accepts the envelope, a bare list, or junk, so no caller has to ask whether
    the endpoint answered in the shape it expected."""
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("notifications")
    else:
        rows = None
    return [r for r in (rows or []) if isinstance(r, dict)]


def event_id(row):
    v = (row or {}).get("event_id")
    return str(v) if v not in (None, "") else None


def notif_type(row):
    """The UNPREFIXED slug (`outbid`), as the inbox ships it. The webhook API
    and `notification-types` use the full key (`client:outbid`) — do not compare
    the two without `full_key()`."""
    v = (row or {}).get("notif_type")
    return str(v) if v not in (None, "") else None


def full_key(row):
    """`client:outbid` from an inbox row — the form webhooks subscribe to."""
    slug, ctx = notif_type(row), (row or {}).get("user_context")
    if not slug:
        return None
    return f"{ctx}:{slug}" if ctx else slug


def created_at(row):
    """Float epoch, or None. The field is a float in every observed row, but a
    string epoch would be a silent sort corruption, so it is coerced."""
    try:
        return float((row or {}).get("created_at"))
    except (TypeError, ValueError):
        return None


def associated(row):
    """`associated_id` as a dict — {} when absent or not an object.

    Absent is REAL: the documented webhook payload has no structured id at all
    (§1.7), and an unknown future notif_type may ship anything here."""
    v = (row or {}).get("associated_id")
    return v if isinstance(v, dict) else {}


def _assoc_int(row, key):
    v = associated(row).get(key)
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _assoc_float(row, key):
    v = associated(row).get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def instance_id(row):
    return _assoc_int(row, "instance_id")


def machine_id(row):
    return _assoc_int(row, "machine_id")


def your_bid(row):
    return _assoc_float(row, "your_bid")


def new_min_bid(row):
    """The DISPLACING price (§1.4) — an authoritative floor read that cannot be
    our own bid echoing back. Evidence about the winning price, NOT a guarantee
    that `new_min_bid + eps` wins the box back: instance 47840057 was displaced
    with your_bid 0.16 against new_min_bid 0.15."""
    return _assoc_float(row, "new_min_bid")


def is_gone(err):
    """True when a request error says the hidden endpoint is GONE (404).

    A 404 here is the expected end-of-life for an endpoint vast never published;
    it is not a bug and not a transient, so callers report it as its own outcome
    instead of retrying or crashing."""
    return "HTTP 404" in str(err or "")


# --------------------------------------------------------------------------- #
# the cursor (D3)
# --------------------------------------------------------------------------- #
Poll = namedtuple("Poll", "new cursor gap rows_seen initialized")
Poll.__doc__ = """One poll's verdict.
  new         — rows never journaled before, oldest first
  cursor      — the cursor to persist (the input cursor is never mutated)
  gap         — a full window of rows, NONE of them known: events may have been
                missed (D4). Never true on the first poll, where "all new" is
                initialization rather than a hole.
  rows_seen   — how many rows the poll returned
  initialized — this poll seeded an empty cursor"""


def empty_cursor():
    """The zero cursor: nothing seen, no floor."""
    return {"last_created_at": None, "recent_ids": []}


def normalize_cursor(cur):
    """Any persisted shape (missing, half-written, wrong types) -> a usable
    cursor. State files outlive schemas; a cursor that raised on load would
    crash-loop the daemon on a field rename."""
    out = empty_cursor()
    if not isinstance(cur, dict):
        return out
    try:
        lca = cur.get("last_created_at")
        out["last_created_at"] = None if lca is None else float(lca)
    except (TypeError, ValueError):
        out["last_created_at"] = None
    ids = cur.get("recent_ids")
    if isinstance(ids, (list, tuple)):
        out["recent_ids"] = [str(i) for i in ids
                             if i not in (None, "")][-RECENT_IDS_MAX:]
    return out


def _stamp(row):
    ts = created_at(row)
    return 0.0 if ts is None else ts


def poll(payload, cursor=None, window=WINDOW):
    """Split an inbox payload against a cursor. PURE — no clock, no I/O.

    Rows with no `event_id` are unjournalable (nothing could dedup them on the
    next poll, so they would re-emit forever) and are dropped."""
    cur = normalize_cursor(cursor)
    first_poll = not cur["recent_ids"] and cur["last_created_at"] is None
    seen = set(cur["recent_ids"])
    floor = cur["last_created_at"]

    rows = inbox_rows(payload)
    # oldest first: the journal then reads as a timeline, and the id list's
    # truncated head is the OLDEST ids — which is what the bound wants to drop.
    keyed = sorted((r for r in rows if event_id(r)), key=_stamp)

    new = []
    for r in keyed:
        if event_id(r) in seen:
            continue
        ts = created_at(r)
        if floor is not None and ts is not None and ts < floor - REORDER_SLOP_S:
            continue                      # older than the floor: settled history
        new.append(r)

    stamps = [t for t in (created_at(r) for r in keyed) if t is not None]
    floor_out = floor if not stamps else (max(stamps) if floor is None
                                          else max(floor, max(stamps)))
    ids_out = list(cur["recent_ids"])
    for r in keyed:                       # EVERY row the server showed us, new
        eid = event_id(r)                 # or not: dedup must not depend on
        if eid not in seen:               # whether an aggregate liked the row
            ids_out.append(eid)
            seen.add(eid)
    ids_out = ids_out[-RECENT_IDS_MAX:]

    gap = (not first_poll and len(keyed) >= window
           and len(new) == len(keyed) and len(keyed) > 0)
    return Poll(new=new,
                cursor={"last_created_at": floor_out, "recent_ids": ids_out},
                gap=gap, rows_seen=len(rows), initialized=first_poll)


def journal_fields(row):
    """The `notify_seen` payload for one row (D4): identity, label, the
    structured id VERBATIM, and the stamp. Nothing derived, nothing dropped — a
    row we cannot interpret today must still be minable tomorrow."""
    out = {
        "event_id": event_id(row),
        "notif_type": notif_type(row),
        "created_at": created_at(row),
        "associated_id": associated(row) or None,
    }
    iid, mid = instance_id(row), machine_id(row)
    if iid is not None:
        out["iid"] = str(iid)             # so `fleet report --box` finds it
    if mid is not None:
        out["machine_id"] = mid
    return out


# --------------------------------------------------------------------------- #
# outbid evidence: normalize, retain, match (S2b, NOTIFY_DESIGN §6.3)
# --------------------------------------------------------------------------- #
# The rule this whole section serves: a notification is EVIDENCE (D2). Nothing
# below decides anything — it turns a feed row into a small, JSON-safe record
# and answers "is there a fresh, unconsumed outbid row for THIS box?". What that
# answer is worth is `bidpolicy`'s call, and what to do about it is the driver's.
def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def is_outbid(row):
    return notif_type(row) == OUTBID_TYPE


def outbid_evidence(row):
    """One inbox row -> the retainable evidence record, or None.

    None for anything we could not act on anyway: a non-outbid type, a row with
    no `event_id` (nothing could dedup it), or one whose `associated_id` names
    no instance (the DOCUMENTED webhook payload has no structured id at all,
    §1.7 — so "no instance" is a real shape, not a corruption).

    The prices are carried as read, including the below-our-bid case: §6.1's
    whole point is that `new_min_bid <= your_bid` is a displacement of unknown
    class, which is a thing to record and refuse to price off, not a thing to
    drop on the floor."""
    if not is_outbid(row):
        return None
    iid, eid = instance_id(row), event_id(row)
    if iid is None or eid is None:
        return None
    return {"event_id": eid, "iid": str(iid), "machine_id": machine_id(row),
            "your_bid": your_bid(row), "new_min_bid": new_min_bid(row),
            "created_at": created_at(row)}


def _as_evidence(item):
    """Either shape -> evidence: a raw inbox row, or an already-normalized
    record read back out of `state.json`. One matcher serves the daemon (which
    retains records) and any caller holding rows, and neither can drift."""
    if not isinstance(item, dict):
        return None
    if "notif_type" in item or "associated_id" in item:
        return outbid_evidence(item)
    eid, iid = item.get("event_id"), item.get("iid", item.get("instance_id"))
    if not eid or iid in (None, ""):
        return None
    return {"event_id": str(eid), "iid": str(iid),
            "machine_id": item.get("machine_id"),
            "your_bid": _as_float(item.get("your_bid")),
            "new_min_bid": _as_float(item.get("new_min_bid")),
            "created_at": _as_float(item.get("created_at"))}


def outbid_evidence_rows(rows):
    """The outbid evidence in a batch of rows, oldest first. Junk is dropped."""
    out = [ev for ev in (outbid_evidence(r) for r in (rows or []))
           if ev is not None]
    return sorted(out, key=lambda e: e["created_at"] or 0.0)


def retain_outbid(kept, new_rows, now, *, keep_s=RETAIN_S, max_rows=RETAIN_MAX):
    """PURE. The retained outbid lookaside after this poll: previously kept
    records plus this poll's, de-duped by `event_id`, aged out past `keep_s`,
    newest `max_rows`. Tolerant of any persisted shape — a state file that
    outlives a schema must degrade to "less evidence", never to a crash."""
    seen, out = set(), []
    for item in list(kept or []) + list(new_rows or []):
        ev = _as_evidence(item)
        if ev is None or ev["event_id"] in seen:
            continue
        ts = ev["created_at"]
        if now is not None and ts is not None and now - ts > keep_s:
            continue
        seen.add(ev["event_id"])
        out.append(ev)
    out.sort(key=lambda e: e["created_at"] or 0.0)
    return out[-max_rows:]


def match_outbid(rows, iid, now, *, window_s=FRESH_WINDOW_S, exclude_ids=()):
    """PURE. The NEWEST fresh, unconsumed outbid record for `iid` — or None.

    Match key is `instance_id`, never `machine_id` (§6.3): a machine hosts
    sibling chunks, and a rehost lands a NEW instance id, so matching by
    instance is box-swap-safe by construction.

    `exclude_ids` is the consumed set. It is what the freshness window alone
    cannot do: a box evicted, rescued, and evicted again inside 15 minutes would
    otherwise have cycle 2 labelled by cycle 1's row — at cycle 1's price."""
    if iid in (None, "") or now is None:
        return None
    excl = {str(e) for e in (exclude_ids or [])}
    best = None
    for item in (rows or []):
        ev = _as_evidence(item)
        if ev is None or ev["iid"] != str(iid) or ev["event_id"] in excl:
            continue
        ts = ev["created_at"]
        if ts is None or abs(now - ts) > window_s:
            continue
        if best is None or ts > (best["created_at"] or 0.0):
            best = ev
    return best


def fresh_outbid_ids(rows, iid, now, *, window_s=FRESH_WINDOW_S):
    """PURE. Every fresh in-window outbid `event_id` for `iid`, matched or not.

    `match_outbid` answers "which row explains this eviction"; this answers
    "which rows BELONG to it" — the whole cycle, not the one the latch took
    (review round 1, 2-2). One eviction cycle can mint more than one row: our
    rescue raise is PUT against a stopped instance, and a raise that is itself
    outbid before the box resumes mints a second. Left unconsumed, that second
    row stays matchable for the rest of the freshness window and labels — and
    prices — the NEXT cycle, off a row describing neither.

    Deliberately NOT filtered by `exclude_ids`: the caller is building that set,
    and re-adding an id it already holds is a no-op. Order is feed order; the
    caller only needs the ids."""
    if iid in (None, "") or now is None:
        return []
    out = []
    for item in (rows or []):
        ev = _as_evidence(item)
        if ev is None or ev["iid"] != str(iid):
            continue
        ts = ev["created_at"]
        if ts is None or abs(now - ts) > window_s:
            continue
        out.append(ev["event_id"])
    return out


# --------------------------------------------------------------------------- #
# rendering (`herdd notify`, S1)
# --------------------------------------------------------------------------- #
def age_str(sec):
    """Mirrors `herdd._age_str`. Duplicated on purpose: this module is a pure
    leaf (like `bidpolicy`) that `fleetd` and `herdd` both import, so it may
    import neither."""
    try:
        sec = max(0, int(sec))
    except (TypeError, ValueError):
        return "?"
    if sec < 90:
        return f"{sec}s"
    if sec < 5400:
        return f"{sec // 60}m"
    if sec < 172800:
        return f"{sec // 3600}h"
    return f"{sec // 86400}d"


def _detail(row):
    """The per-type one-liner. `outbid` prints the displacement (the number this
    whole channel is worth reading for); everything else prints what structure
    it has, falling back to the subject so an UNKNOWN type is still legible."""
    kind = notif_type(row)
    if kind == "outbid":
        yb, nmb = your_bid(row), new_min_bid(row)
        if yb is not None and nmb is not None:
            tail = ("   (BELOW our bid — on-demand taker or host action)"
                    if nmb <= yb else "")
            return f"bid ${yb:.2f}/hr -> min ${nmb:.2f}/hr{tail}"
    extra = {k: v for k, v in associated(row).items()
             if k not in ("instance_id", "machine_id")}
    if extra:
        return " ".join(f"{k}={v}" for k, v in sorted(extra.items()))
    return (row.get("subject") or "").strip()


def render_inbox(payload, now, limit=0):
    """The inbox table, newest first. `now` is passed in so the view is a pure
    function of (payload, clock) and testable to the character."""
    rows = sorted(inbox_rows(payload), key=_stamp, reverse=True)
    shown = rows[:limit] if limit and limit > 0 else rows
    env = payload if isinstance(payload, dict) else {}
    out = [f"inbox: {len(rows)} row(s)"
           + (f", showing {len(shown)}" if len(shown) != len(rows) else "")
           + f"   unread={env.get('unread_count')}"
           + f"  last_seen_at={env.get('last_seen_at')}"]
    if not rows:
        out.append("  (empty — no notifications in the server's window)")
        return "\n".join(out) + "\n"
    out.append(f"  {'age':<6}{'type':<20}{'instance':<11}{'machine':<10}detail")
    for r in shown:
        ts = created_at(r)
        iid, mid = instance_id(r), machine_id(r)
        out.append(f"  {(age_str(now - ts) if ts is not None else '?'):<6}"
                   f"{(notif_type(r) or '?'):<20}"
                   f"{(str(iid) if iid is not None else '-'):<11}"
                   f"{(str(mid) if mid is not None else '-'):<10}{_detail(r)}")
    return "\n".join(out) + "\n"


def type_rows(payload):
    """`notification_types` as a list — [] for any other shape."""
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("notification_types")
    else:
        rows = None
    return [r for r in (rows or []) if isinstance(r, dict)]


def _channels(spec):
    prefs = spec.get("default_preferences")
    if not isinstance(prefs, dict):
        return "-"
    on = sorted(k for k, v in prefs.items() if v is True)
    return ",".join(on) if on else "-"


def render_types(payload):
    """key / topic / category / default channels. The channel column is the
    account's DEFAULT preference set as the API reports it — `webhooks` showing
    up there is what a `POST /webhooks/` flips on for its keys (§1.5)."""
    rows = type_rows(payload)
    out = [f"notification types: {len(rows)}"]
    if not rows:
        out.append("  (none reported)")
        return "\n".join(out) + "\n"
    out.append(f"  {'key':<34}{'topic':<16}{'category':<16}channels(default)")
    for s in sorted(rows, key=lambda r: (str(r.get("context")), str(r.get("key")))):
        out.append(f"  {str(s.get('key') or '?'):<34}"
                   f"{str(s.get('topic') or '-'):<16}"
                   f"{str(s.get('category') or '-'):<16}{_channels(s)}")
    return "\n".join(out) + "\n"


def webhook_rows(payload):
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("webhooks")
    else:
        rows = None
    return [r for r in (rows or []) if isinstance(r, dict)]


def _webhook_events(w):
    """A webhook's subscribed keys. The live list is EMPTY today (§1.5), so the
    field name is unverified — every plausible spelling is accepted rather than
    printing '-' against a real subscription."""
    for k in ("event_types", "notification_keys", "notification_types",
              "events", "keys"):
        v = w.get(k)
        if isinstance(v, (list, tuple)) and v:
            return ",".join(str(x) for x in v)
        if isinstance(v, str) and v:
            return v
    return "-"


def render_webhooks(payload):
    """id / name / url / subscribed keys. Zero webhooks is the CURRENT, correct
    state (D5: the slice is deferred behind an owner ingress decision) and must
    read as such, not as a failure."""
    rows = webhook_rows(payload)
    out = [f"webhooks: {len(rows)} of {MAX_WEBHOOKS} slot(s) used"]
    if not rows:
        out.append("  (none — nothing is subscribed today; the webhook slice is "
                   "deferred behind public ingress, NOTIFY_DESIGN D5)")
        return "\n".join(out) + "\n"
    out.append(f"  {'id':<10}{'name':<24}{'url':<52}events")
    for w in rows:
        out.append(f"  {str(w.get('id') or '?'):<10}"
                   f"{str(w.get('name') or '-'):<24}"
                   f"{str(w.get('url') or '-'):<52}{_webhook_events(w)}")
    return "\n".join(out) + "\n"
