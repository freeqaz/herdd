"""The supervise journal leaves — four emitters, two destinations, one contract.

Why this module exists
----------------------
These four functions were misfiled inside the biggest cluster in `herdd.py`
(supervise-run-lane) and called from four others — 29 + 33 + 22 + 8 call lines
across 33 distinct callers, plus one live external caller in `fleetd.py`. They
are pure leaves: nothing here calls back into the supervise knot, so they come
out FIRST (plan §8 step 2) and everything that follows can be ported without
dragging ~5,400 lines of mutually recursive lane code along with it.

Two destinations, deliberately not unified:

* **B2, durable.** `_sup_emit` and `_job_handoff_emit` append ONE immutable
  object per event to the append-only run log (`runs/<run_id>/events/`) and the
  box-lifecycle log (`jobs/nodes/<IID>/events/`) respectively, via the Zone S
  leaves `runmeta` / `jobmeta`. Every byte of those records is a wire contract
  (plan §4): the v1 envelope, the None-valued-field drop, the
  `<ts>-<actor_slug>-<nonce>.json` key, `json.dumps(separators=(",", ":"))`
  plus a trailing newline. Both swallow transport failures — a failed emit must
  never kill a supervision loop — and the two swallow shapes are NOT identical
  (see below), which is also contract.
* **An in-memory queue on the caller's context dict.** `_job_ladder_journal`
  and `_job_handoff_journal` write nothing and touch no I/O; they append a
  `(name, fields)` tuple to `jctx["ladder_journal"]` / `jctx["handoff_journal"]`
  and truncate in place. fleetd drains the first verbatim into `fleet log` and
  the second under a hard `jobs_handoff_` prefix. The in-place `del q[:-MAX]`
  is load-bearing: the caller and fleetd alias the same list object, so a
  "clearer" `q = q[-MAX:]` rebind would silently orphan the drain.

Details that are silently load-bearing, recorded so a later cleanup cannot
quietly undo them:

* `_sup_emit`'s own failure record is exactly ``{"_emitted": False, "_error":
  ...}`` — NO ``_key``, unlike `runmeta.emit_event`'s own rc!=0 path, which
  returns the full event plus `_key`. Callers only ever read `_emitted`, but
  the two shapes differ and both are reproduced here verbatim.
* `_iso_z` renders ``%Y-%m-%dT%H:%M:%SZ``. That is a DIFFERENT format from the
  event envelope's `runmeta.now_ts()` compact basic-ISO (no colons, millisecond
  precision, sorts lexicographically as an object key). Both are contract; they
  are not to be unified.
* Late binding is the contract. All 53 `monkeypatch.setattr(herdd, "<name>",
  ...)` sites in the suite work only because callers resolve these emitters as
  a module global at CALL time. Any caller ported into `vastlib` must keep
  addressing them in module-attribute form (`journal._sup_emit(...)`), never
  `from ... import _sup_emit` — a call bound at import time is invisible to the
  patch and the test goes vacuously green (plan §10, "vacuous test patches").

What is deliberately NOT here
-----------------------------
* **No new transport abstraction.** The rclone/B2 write stays exactly where it
  was: inside `runmeta.emit_event` / `jobmeta.emit_box_event` (Zone S), which
  own the `runner` seam, the `b2:` vs `b2w:` split-key choice and the bucket
  read. This module is the four wrappers and nothing else; inventing a seam
  here would put a second implementation of a wire contract in the tree.
* **No environment reads.** `B2_BUCKET`, `B2_WRITE_KEY_ID` and the actor
  identity vars are read inside `runmeta`/`jobmeta`. With no `.env` and no
  `B2_BUCKET` (the test baseline), an unpatched `_sup_emit` raises inside
  `runmeta._bucket` and the swallow turns it into `{"_emitted": False, ...}` —
  that swallow is the only thing keeping unpatched run-lane tests off the
  network, since conftest's API guard covers `request_soft`, not B2 writes.
* **No flush/atexit/signal handling.** Each emit is one synchronous subprocess;
  there is nothing buffered to lose. The flush semantics live in the callers
  (`_handoff_run_signals`' flush-timeout path), which stay in `herdd.py` for
  now and move in plan step 4.
* **No folding, no reading.** This is the write side only. Reading a run or a
  box log is `runmeta.read_run` / the jobmeta box fold.
* **`_serve_sla_emit`** (a fourth wrapper of the same shape, actor
  ``boot-sla``) is not in the plan §5 list and is not ported here; it moves
  with its cluster — which it has: it lives in `supervise/replacement.py`
  beside `_serve_boot_sla_tick` / `_serve_boot_sla_condemn`, its only callers,
  and `_serve_self_park_soft` + `SERVE_SELF_PARK_FRESH_S` joined it there at
  step 6 (leftovers). Nothing serve-lane is expected here.

Provenance: moved verbatim from `tools/vast/herdd.py` (plan §8 step 2,
2026-08-16). Behavior-preserving: bodies copied, annotations added, nothing
else. `_job_handoff_journal` travels with its identical twin
`_job_ladder_journal` — porting one and leaving the other would split the pair
and force `JOB_HANDOFF_JOURNAL_MAX` to exist in two places.
"""

from __future__ import annotations

import datetime
from typing import Any, MutableMapping

import jobmeta
import runmeta

# Cap on the undrained jobs-lane handoff decision queue (_job_handoff_journal).
# fleetd drains it every poll and never approaches this; the legacy inline
# `job supervise` loop never drains it at all, so the list needs a ceiling.
# moved-from: herdd.JOB_HANDOFF_JOURNAL_MAX
JOB_HANDOFF_JOURNAL_MAX = 200

# The queue element both journal functions append: (name, fields). fleetd's
# drain unpacks exactly this pair, so the shape is part of the drain contract.
JournalEntry = tuple[str, dict[str, Any]]


# moved-from: herdd._sup_emit
def _sup_emit(run_id: str, event: str, **fields: Any) -> dict[str, Any]:  # noqa: ANN401
    """Best-effort supervisor event (actor='supervisor'). A transport failure is
    swallowed — a failed emit must NEVER kill the loop."""
    # `**fields` is Any and stays Any: these are arbitrary JSON-serializable
    # event payload values forwarded straight into the v1 envelope, and the set
    # of keys is open by design (26 event names, each with its own fields).
    try:
        return runmeta.emit_event(run_id, event, actor="supervisor", **fields)
    except Exception as e:
        return {"_emitted": False, "_error": str(e)}


# moved-from: herdd._job_handoff_emit
def _job_handoff_emit(jctx: MutableMapping[str, Any], event: str,
                      **fields: Any) -> dict[str, Any]:  # noqa: ANN401
    """Best-effort jobs-lane handoff telemetry -> the box lifecycle log
    (jobs/nodes/<IID>/events/, keyed on the primary box the way the run lane keys
    on run_id). A failed emit never kills the loop."""
    try:
        return jobmeta.emit_box_event(str(jctx.get("iid")), event,
                                      actor="job-supervise", **fields)
    except Exception as e:
        return {"_emitted": False, "_error": str(e)}


# moved-from: herdd._job_handoff_journal
def _job_handoff_journal(jctx: MutableMapping[str, Any], kind: str,
                         **fields: Any) -> None:  # noqa: ANN401
    """Queue ONE handoff decision for whoever is driving this ladder to surface.

    `_job_handoff_emit` above writes the same decisions to B2 (jobs/nodes/<IID>/
    events/), which is durable but invisible where an operator actually looks.
    fleetd drains this list into its own journal as `jobs_handoff_<kind>`, so
    `fleet log` shows the migration — the 2026-08-08 incident renting a second
    box and destroying a healthy primary produced NOTHING there, which is why it
    first read as a spot eviction. The legacy inline `job supervise` loop prints
    to a terminal and simply leaves the list unread.

    Kinds are the PHASES, never just the terminal one: under filed defect #61 the
    handoff `complete` transition is unreachable under fleetd (a non-`run` watch
    ends at `inst is None` before the ladder ticks again), so `armed`, `fenced`
    and `cutover` each have to reach the journal on their own.

    Bounded, because one of the two drivers never drains it: a multi-day inline
    `job supervise` would otherwise accumulate a deferral per market move for the
    life of the process. Oldest entries fall off — a drained reader (fleetd, every
    poll) never sees the cap, and an undrained one only loses backlog it was
    never going to read."""
    q: list[JournalEntry] = jctx.setdefault("handoff_journal", [])
    q.append((kind, fields))
    del q[:-JOB_HANDOFF_JOURNAL_MAX]


# moved-from: herdd._job_ladder_journal
def _job_ladder_journal(jctx: MutableMapping[str, Any], event: str,
                        **fields: Any) -> None:  # noqa: ANN401
    """Queue ONE ladder event under its OWN name for the driver to surface.

    Sibling of `_job_handoff_journal`, and it exists because that one is
    hard-prefixed `jobs_handoff_` by fleetd's drain — the right name for a
    migration and a lie for a pull-reschedule. fleetd drains this list into
    `fleet log` verbatim; the legacy inline loop leaves it unread, bounded the
    same way.

    **Why every money-moving rung now goes through here (task #78, incident
    2026-08-08 23:27Z).** After a redeploy, fleetd reconciled a stale watch and
    ran a full replacement chain on its own initiative — launched 47219058,
    condemned it for a stalled image pull, launched 47219872, moved the ticket,
    destroyed the condemned box. Every step of that was a bare `print()` to the
    daemon's stdout. `herdd fleet log` — the surface the runbooks tell an
    operator to read — contained **zero** events for either box id, so a human
    who was hand-rescuing the same job from the other side had no way to see that
    an autonomous actor was already on it. Two recovery actors, one job, ~$0.9
    wasted.

    The bar this encodes: a rung that LAUNCHES, CONDEMNS, RETARGETS or DESTROYS
    a box emits a journal event carrying the watch identity and the prices. A
    print is a courtesy to whoever is tailing the daemon; the journal is the
    record. Pass `iid=` when the event is about a box that is not the watch's
    current one (the condemned box, the box being destroyed) — the drain uses
    it and falls back to the watch's live iid."""
    q: list[JournalEntry] = jctx.setdefault("ladder_journal", [])
    q.append((event, fields))
    del q[:-JOB_HANDOFF_JOURNAL_MAX]


# moved-from: herdd._iso_z
def _iso_z(ts: float | None) -> str | None:
    """`2026-08-05T18:30:00Z` for a journal field; None passes through."""
    if not ts:
        return None
    return datetime.datetime.fromtimestamp(
        ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
