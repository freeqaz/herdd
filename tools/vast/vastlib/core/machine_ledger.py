"""`instance_id -> machine_id`, written down while the mapping still exists.

WHY THIS FILE EXISTS
--------------------
vast's instances API is the only place that mapping lives, and a row leaves it
the moment the box is destroyed. `hostfacts.py ingest` therefore resolves only
boxes that are alive AT THE MOMENT SOMEBODY RUNS IT. Measured 2026-08-24, with
`ingest` last run on 2026-08-07: **3 of 202** box-written records resolved and
199 did not. Those 199 are not lost — they group under `iid:<IID>` — but they
can never aggregate per machine, which is the only question host acceptance
asks.

Two other sources were checked and neither closes it: box-written hostfacts
carry `instance_id` and no `machine_id` (the box genuinely does not know it),
and jobs-v2 events carry `instance_id` alone. `hosts.py`'s launched-event join
covers only the older `runs/` train lane.

But fleetd already reads every instance every 45 s and holds `machine_id` in
hand. The mapping flows past us constantly and was simply never written down.
So this is a ledger fleetd appends to, and the cost is one file write per tick
that observes something new.

APPEND-ONLY BY CONTRACT
-----------------------
An entry is a historical fact and is never removed, never overwritten with a
different machine, and never expired: the whole point is to outlive the box.
`record()` merges — it refreshes `last_seen` (lazily: see LAST_SEEN_REFRESH_S,
because a to-the-second one is a whole-file rewrite per tick) and fills gaps,
and a CONFLICTING
machine for a known instance is kept as `conflicts` rather than silently
replacing what was there, because vast reusing an instance id across machines
would otherwise mis-attribute every record that instance ever wrote.

READ SIDE
---------
`hostfacts.py` reads this file directly, with no vastlib import: it is a Zone S
flat leaf shipped in the jobd bundle and must import bare-name under `python3
-P`. That means the default path is spelled in two places, so
`test_machine_ledger.py` asserts the two agree rather than a comment asking
politely.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from typing import Any

from vastlib.core import config

PATH_ENV = "VAST_MACHINE_LEDGER_PATH"

#: Mirrored in `hostfacts.py:_ledger_path()`. Pinned equal by a test.
LEDGER_FILENAME = "machine_ledger.json"


def store_path() -> str:
    """`$VAST_MACHINE_LEDGER_PATH` wins; otherwise it sits beside fleetd's other
    state, since fleetd is what writes it."""
    override = os.environ.get(PATH_ENV)
    if override:
        return os.path.expanduser(override)
    return os.path.join(config.fleet_state_dir(), LEDGER_FILENAME)


def load(path: str | None = None) -> dict[str, Any]:
    """`{iid: {machine_id, first_seen, last_seen, conflicts?}}`.

    A missing or corrupt file reads as empty rather than raising: this is a
    convenience index that can always be rebuilt going forward, and a resolver
    that dies on a truncated write would take an `ingest` down with it.
    """
    p = path or store_path()
    try:
        with open(p) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save(data: dict[str, Any], path: str) -> bool:
    """Atomic by tmp-then-rename — fleetd writes this from its tick loop and a
    reader can arrive at any moment."""
    d = os.path.dirname(path) or "."
    try:
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".machine_ledger.")
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        return False
    return True


#: How stale `last_seen` may get before a re-sighting is worth a write. The
#: field answers "roughly when did we last see this box", which nothing reads to
#: the second; at tick cadence an exact one costs a whole-file rewrite every
#: ~50 s forever on a fleet where nothing changed.
LAST_SEEN_REFRESH_S = 3600.0


def record(pairs: Iterable[tuple[Any, Any]], *, now: float,
           path: str | None = None) -> int:
    """Merge `(instance_id, machine_id)` pairs in. Returns entries CHANGED — a
    new mapping or a new conflict, which is what a caller journals.

    Returns 0 without touching the disk when nothing is new, so a steady fleet
    costs one read per tick and no write. Re-seeing a box we already know is not
    "new": it refreshes `last_seen` in memory but only reaches disk once the
    stored value is LAST_SEEN_REFRESH_S stale, and never counts as changed.
    """
    p = path or store_path()
    data = load(p)
    changed = 0
    refreshed = 0
    for iid, mid in pairs:
        if not iid or not mid:
            continue
        key, mid = str(iid), str(mid)
        cur = data.get(key)
        if not isinstance(cur, dict):
            data[key] = {"machine_id": mid, "first_seen": now, "last_seen": now}
            changed += 1
            continue
        if cur.get("machine_id") != mid:
            # Never overwrite: an instance id that has named two machines makes
            # every record it wrote ambiguous, and losing the first answer would
            # hide that rather than fix it.
            conflicts = cur.setdefault("conflicts", [])
            if mid not in conflicts:
                conflicts.append(mid)
                changed += 1
            continue
        if _stale(cur.get("last_seen"), now):
            cur["last_seen"] = now
            refreshed += 1
    if changed or refreshed:
        _save(data, p)
    return changed


def _stale(seen: object, now: float) -> bool:
    """Is a stored `last_seen` far enough behind `now` to be worth persisting?
    Unreadable or absent is stale — a record with no clock should get one."""
    try:
        return abs(float(now) - float(seen)) >= LAST_SEEN_REFRESH_S  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return True


def resolve(instance_id: object, path: str | None = None) -> str | None:
    """`machine_id` for an instance, or None. A conflicted entry resolves to
    None — an ambiguous attribution is worse than an absent one."""
    entry = load(path).get(str(instance_id))
    if not isinstance(entry, dict) or entry.get("conflicts"):
        return None
    mid = entry.get("machine_id")
    return str(mid) if mid else None


def stats(path: str | None = None) -> dict[str, int]:
    data = load(path)
    return {
        "instances": len(data),
        "machines": len({e.get("machine_id") for e in data.values()
                         if isinstance(e, dict) and e.get("machine_id")}),
        "conflicted": sum(1 for e in data.values()
                          if isinstance(e, dict) and e.get("conflicts")),
    }
