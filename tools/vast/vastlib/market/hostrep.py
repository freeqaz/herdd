"""vastlib.market.hostrep — durable per-machine host reputation.

Why this exists
---------------
Every host-quality memory in this codebase was, until now, EPISODIC. The pull
watchdog's `pull_bad_machines` and the eviction lane's `evicted_machines` both
live on the watch context `jc`, so they are forgotten when the watch ends — and
the boot-SLA relaunch lane forgets even sooner, passing its exclusions to one
`launch_serve.sh` invocation and nothing else. The consequence is measurable:
machine 72425 was condemned for a stalled image pull on 2026-08-17 and rented
again, from a clean slate, on 2026-08-20, where it stalled the same way. Nothing
in the system was capable of noticing that.

The reason it re-rents is not stubbornness, it is price. Automatic lanes take
the cheapest qualifying offer, and the cheapest tier of a popular card is
routinely one operator whose cluster cannot feed our image fast enough. Cheap is
exactly the correlate of slow here, so "cheapest first" reliably selects the
host we have the most evidence against.

Owner directive, 2026-08-20: *"a cheap host that doesn't work is not worth using
for us -- paying more to have a fast boot time is worth it."*

What it does
------------
One JSON file under the fleetd state dir records a decaying STRIKE per condemned
boot, keyed by vast `machine_id` and durable across sessions, watches and
reboots. From it two numbers fall out:

* a **penalty** — a multiplier applied to a machine's price when RANKING offers
  (never when paying: see `penalty`). One strike reads as +35%, so the host
  loses to any clean host up to 35% dearer and still beats one that is 2x. This
  is the directive expressed as arithmetic rather than as a rule someone has to
  remember.
* a **block** — an outright exclusion, earned at `HOSTREP_BLOCK_SCORE`.

Recurrence is the load-bearing signal
-------------------------------------
Four condemns inside one bad hour can be one transient event: a backbone hiccup,
a registry blip, an operator rebooting a rack. A host that fails again three
days later is failing because of what it IS. So the score multiplies by
`1 + HOSTREP_RECURRENCE_BONUS * (distinct_days - 1)`, and the default thresholds
are set so that **two strikes on two different days block a host while two
strikes in one session do not** (3.26 vs 2.00 against a 3.0 threshold). This is
the one term that can tell those two stories apart, and it is why the store has
to be durable to work at all.

Decay, not curation
-------------------
A strike's weight halves every `HOSTREP_HALF_LIFE_D` days, so the file forgives
without anyone pruning it: a host that had one bad night three weeks ago is
scored at 0.35 and effectively clean. A block is held for
`HOSTREP_BLOCK_COOLDOWN_D` regardless, because a block earned by recurrence
decays back under its own threshold in under two days and the next launch would
re-rent the host we just condemned. Successes are recorded too, and a strike
older than the machine's last good boot is discounted — evidence superseded by
newer, better evidence.

EVERY BLOCK EXPIRES. That is a requirement, not a side effect: hosts get fixed
(a bad NIC, a saturated uplink, a dockerd pinned to one concurrent download are
all operator-side changes), and a store that only accumulates is a shrinking
supply list rather than a memory. So the cooldown is the RETRY CLOCK — 14 d,
matched to the half-life, so the retry lands when the evidence has halved and
the host returns on PROBATION rather than clean: still penalised, therefore
picked only where it is clearly cheapest, and re-blocked hard if that probe
fails (a third distinct day multiplies the recurrence term). Nothing here can
produce a permanent exclusion except an operator `hold`, which carries its own
explicit expiry.

Only HOST fault is admissible
-----------------------------
A strike is a claim about a machine, so a failure that is true of every machine
at once must not become one. `note_strike` refuses outright while
`core.acctfault` holds a live account-level fault (no credit, a rejected key, a
suspended account): those cannot be fixed by preferring a different host, and a
mark charged under one distorts every later ranking with nothing to correct it.
Market states are excluded at the caller instead, where the class is known — see
`STRIKE_FREE_EVICTION_CLASSES` in `supervise/replacement.py`.

Fail-open, everywhere
---------------------
This module can only ever make a pick WORSE-ORDERED, never impossible. Every
public entry point swallows its own errors and degrades to "no reputation
known": a missing file, a corrupt file, an unwritable state dir and a stale
schema all read as an empty store. A launch must never fail because an advisory
score could not be loaded — reputation is a preference layered on top of the
market, and the market is the thing that has to keep working.

Reads are process-cached for `_CACHE_TTL_S` so a candidate walk that prices
thirty offers does not stat the file thirty times; writes invalidate it.

Operator surface: `herdd fleet hosts` (list / block / allow / forget).
Design note: `tools/vast/FLEETD_DESIGN.md` §9.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from typing import Any

from vastlib.core import acctfault, config

#: Bumped when a stored record's shape changes incompatibly. A file carrying
#: anything else is treated as absent (fail-open), never migrated in place —
#: the store is rebuildable from live operation within days, so a migration
#: helper here would be more code than the data is worth.
SCHEMA_VERSION = 1

#: Env override for the store path. Exists so the test suite can sandbox the
#: store without moving `FLEETD_STATE_DIR`, which several tests assert the
#: default of; `tools/vast/conftest.py` sets it at conftest import.
PATH_ENV = "VAST_HOSTREP_PATH"

#: Set to 1 to disable reputation entirely — ranking reverts to cheapest-first
#: and nothing is excluded. The escape hatch for "the store is wrong and I need
#: a box now", and what `--no-hostrep` sets on the lanes that expose it.
DISABLE_ENV = "VAST_HOSTREP_DISABLE"

#: What a strike is worth before decay. Every one of these is a boot that cost
#: us a rental and delivered nothing, so they weigh the same by default; the
#: kinds are kept distinct because the operator listing names them and because
#: an operator hold should outweigh a single automatic condemnation.
#:
#: `host_stop` is the one weighted above the default. It is the only kind that
#: recurs on a MINUTE scale (a host that dumps boxes dumps the replacement too),
#: so the recurrence term — which needs distinct DAYS — cannot see it, and 1.0
#: would let a host take the fleet all night unblocked. At 1.6 the SECOND stop
#: blocks while one sits at half the bar and halves again every half-life, so a
#: single stop still forgives itself. Not 1.5 (2 x 1.5 == the threshold
#: exactly): decay over the gap between two stops puts that pair a hair UNDER,
#: so the rule would read as blocking and never fire.
STRIKE_WEIGHTS: dict[str, float] = {
    "pull_timeout": 1.0,    # never finished the image pull inside the deadline
    "pull_slow": 1.0,       # sustained throughput under the BOOT_MIN_MBPS floor
    "boot_sla": 1.0,        # reached `running`, never stamped JOBD_STATUS
    "host_failure": 1.0,    # the host took the box away — not the market
    "host_stop": 1.6,       # the host stopped a box nobody had outbid
    "operator": 2.0,        # a human condemned this host by hand
}

#: A strike of an unknown kind still counts. Forward compatibility in the
#: direction that matters: a newer fleetd writing a kind this reader does not
#: know must not silently score it as zero.
DEFAULT_STRIKE_WEIGHT = 1.0

#: Weight multiplier for a strike that predates the machine's last GOOD boot.
#: Not zero: a host that fails, works, then fails again is exactly the flaky
#: host this module exists to find, and zeroing superseded strikes would make
#: it indistinguishable from a clean one.
SUPERSEDED_DISCOUNT = 0.4

_CACHE_TTL_S = 5.0
_cache: dict[str, Any] = {"path": None, "t": 0.0, "data": None}

_DAY_S = 86400.0


# --------------------------------------------------------------------- paths

def store_path() -> str:
    """Where the reputation file lives. `$VAST_HOSTREP_PATH` wins; otherwise it
    sits beside fleetd's own state, because fleetd is what writes most of it."""
    override = os.environ.get(PATH_ENV)
    if override:
        return os.path.expanduser(override)
    # `core.config`, not `fleet.client`: the ring order forbids market -> fleet,
    # and the formula was moved down for exactly this caller. Never re-spell the
    # FLEETD_STATE_DIR default here — one home, or the two drift.
    return os.path.join(config.fleet_state_dir(), "host_reputation.json")


def enabled() -> bool:
    return os.environ.get(DISABLE_ENV, "").strip() not in ("1", "true", "yes")


# --------------------------------------------------------------------- store

def _empty() -> dict[str, Any]:
    return {"version": SCHEMA_VERSION, "machines": {}}


def load(path: str | None = None, *, fresh: bool = False) -> dict[str, Any]:
    """The whole store, or an empty one. Never raises, never migrates: a file
    that is missing, unreadable, not JSON, not a dict or of another schema
    version all read as "we know nothing about any host"."""
    try:
        p = path or store_path()
    except Exception:
        return _empty()
    now = time.monotonic()
    if (not fresh and _cache["data"] is not None and _cache["path"] == p
            and (now - float(_cache["t"])) < _CACHE_TTL_S):
        cached: dict[str, Any] = _cache["data"]
        return cached
    data: dict[str, Any] = _empty()
    try:
        with open(p, encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict) and raw.get("version") == SCHEMA_VERSION \
                and isinstance(raw.get("machines"), dict):
            data = raw
    except FileNotFoundError:
        pass
    except Exception:
        # Corrupt or half-written: fall back to empty rather than to a guess.
        pass
    _cache.update({"path": p, "t": now, "data": data})
    return data


def _save(data: Mapping[str, Any], path: str) -> bool:
    """Atomic replace, so a reader never sees a half-written store. Returns
    False (silently) when the state dir is unwritable — losing a strike is a
    worse pick later, which is survivable; raising into a condemn path is not."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".",
                                   prefix=".host_reputation.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp, path)
        except Exception:
            with _suppress():
                os.unlink(tmp)
            raise
    except Exception:
        return False
    _cache.update({"path": path, "t": time.monotonic(), "data": dict(data)})
    return True


class _suppress:
    """Tiny `contextlib.suppress(Exception)` without the import — this module is
    on the launch path and its import cost is paid by every CLI invocation."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return True


def _update(path: str | None,
            fn: Callable[[MutableMapping[str, Any]], None]) -> bool:
    """Read-modify-write the store through `fn(data)`.

    Deliberately NOT locked. Two fleetd lanes condemning two hosts in the same
    millisecond can lose one strike, and that is the right trade: the loser is
    re-earned by the next failure on that host (which is what the store is
    about — recurrence), whereas a lock file in a condemn path is one more thing
    that can wedge a supervisor. The write itself is atomic, so no reader ever
    sees a torn file; only the interleaving is racy."""
    try:
        p = path or store_path()
    except Exception:
        return False
    data = load(p, fresh=True)
    data = json.loads(json.dumps(data))          # never mutate the cached copy
    try:
        fn(data)
    except Exception:
        return False
    return _save(data, p)


# ---------------------------------------------------------------- recording

def note_strike(machine: object, kind: str, *, iid: object = None,
                note: str | None = None, now: float | None = None,
                path: str | None = None) -> float | None:
    """Record one failed boot against `machine` and return its new score (None
    if nothing was recorded). Idempotent enough for a supervise loop: the caller
    is expected to call this once per condemnation, not once per tick.

    Crossing `HOSTREP_BLOCK_SCORE` here is what arms the block cooldown — the
    threshold is evaluated at WRITE time so that a block, once earned, survives
    the score decaying back under it (see the module docstring).

    REFUSES while an account-level fault is latched. Every kind above is a claim
    about a machine, and an account that cannot rent is true of every machine at
    once — so a strike written under one is a permanent mark on whichever host
    happened to be under the cursor. One guard here rather than at each caller:
    the callers are condemn paths that never see the API refusal, and a fourth
    caller added later inherits the protection instead of having to remember it.
    """
    if machine is None or not enabled():
        return None
    now = time.time() if now is None else float(now)
    # Against the STRIKE's clock, not the wall's: a caller replaying a condemn
    # at an explicit `now` must get the same verdict either way round.
    acct = acctfault.recent(now=now)
    if acct:
        # Not silent: the operator who later asks why a bad host has no strike
        # needs this line, and it is the only place the two facts meet.
        print(f">> host reputation: NOT charging machine {machine} a "
              f"'{kind}' strike — {acctfault.describe(acct['code'])} "
              f"(seen {int(acct['age_s'])}s ago). The host is not the reason.")
        return None
    key = str(machine)
    out: dict[str, float | None] = {"score": None}

    def _apply(data: MutableMapping[str, Any]) -> None:
        rec = data.setdefault("machines", {}).setdefault(key, {})
        strikes = rec.setdefault("strikes", [])
        strikes.append({"ts": now, "kind": str(kind),
                        **({"iid": str(iid)} if iid is not None else {}),
                        **({"note": note} if note else {})})
        strikes.sort(key=lambda s: -float(s.get("ts") or 0.0))
        keep = int(config._boot_knob("HOSTREP_MAX_STRIKES_KEPT", cast=int))
        del strikes[max(1, keep):]
        sc = score(rec, now)
        out["score"] = sc
        if sc >= config._boot_knob("HOSTREP_BLOCK_SCORE"):
            # Store WHEN the threshold was crossed, not when the block ends.
            # A materialized deadline freezes the cooldown knob's value at
            # write time, so retuning it leaves every live block on the old
            # policy — measured when HOSTREP_BLOCK_COOLDOWN_D went 7d -> 14d
            # and the one blocked host kept reporting "retry in 6.4d". The
            # knob is policy and policy belongs at READ time; the fact worth
            # persisting is the crossing.
            rec["blocked_at"] = max(float(rec.get("blocked_at") or 0.0), now)

    if not _update(path, _apply):
        return None
    return out["score"]


def note_ok(machine: object, *, iid: object = None, now: float | None = None,
            path: str | None = None) -> bool:
    """Record that `machine` booted our image and reached a working jobd.

    Cheap and worth it: this is what lets `score` discount strikes a host has
    since disproved, and it is the only positive evidence the store would
    otherwise hold. It does NOT clear a block — a cooldown that any single good
    boot could cancel is not a cooldown, and a host that alternates is precisely
    the one we want held out."""
    if machine is None or not enabled():
        return False
    now = time.time() if now is None else float(now)
    key = str(machine)

    def _apply(data: MutableMapping[str, Any]) -> None:
        rec = data.setdefault("machines", {}).setdefault(key, {})
        rec["last_ok_ts"] = max(float(rec.get("last_ok_ts") or 0.0), now)
        rec["ok_count"] = int(rec.get("ok_count") or 0) + 1
        if iid is not None:
            rec["last_ok_iid"] = str(iid)

    return _update(path, _apply)


def hold(machine: object, *, days: float, reason: str = "",
         now: float | None = None, path: str | None = None) -> bool:
    """Operator hold: block `machine` for `days` regardless of its score.

    Separate from a strike because it is separate evidence — "I watched this
    host eat three pulls" is not the same claim as an automatic condemnation,
    and an operator must be able to lift it (`release`) without the automatic
    layer having an opinion."""
    if machine is None:
        return False
    now = time.time() if now is None else float(now)

    def _apply(data: MutableMapping[str, Any]) -> None:
        rec = data.setdefault("machines", {}).setdefault(str(machine), {})
        rec["hold"] = {"until": now + float(days) * _DAY_S,
                       "reason": reason or "operator hold", "ts": now}

    return _update(path, _apply)


def release(machine: object, *, path: str | None = None) -> bool:
    """Lift an operator hold AND any earned block cooldown, leaving the strike
    history intact. The "I know why it failed and it is fixed" verb — the score
    still decays from real evidence, but the host is rentable again now."""
    def _apply(data: MutableMapping[str, Any]) -> None:
        rec = (data.get("machines") or {}).get(str(machine))
        if isinstance(rec, dict):
            rec.pop("hold", None)
            rec.pop("blocked_at", None)
            rec.pop("blocked_until", None)      # pre-2026-08-20 records

    return _update(path, _apply)


def forget(machine: object, *, path: str | None = None) -> bool:
    """Drop a machine's record entirely — the "that was our bug, not the host's"
    verb. Use after a defect on OUR side (a bad image, a broken onstart) charged
    strikes to hosts that did nothing wrong."""
    def _apply(data: MutableMapping[str, Any]) -> None:
        (data.get("machines") or {}).pop(str(machine), None)

    return _update(path, _apply)


def prune(*, older_than_d: float = 90.0, now: float | None = None,
          path: str | None = None) -> int:
    """Drop strikes older than `older_than_d` and any machine left with no
    evidence at all. Returns machines removed. Housekeeping only — the score
    already treats a 90-day-old strike as worth 0.012 — so nothing depends on
    this ever being run."""
    now = time.time() if now is None else float(now)
    cut = now - float(older_than_d) * _DAY_S
    removed = 0

    def _apply(data: MutableMapping[str, Any]) -> None:
        nonlocal removed
        machines = data.get("machines") or {}
        for key in list(machines):
            rec = machines[key]
            if not isinstance(rec, dict):
                machines.pop(key, None)
                removed += 1
                continue
            rec["strikes"] = [s for s in (rec.get("strikes") or [])
                              if float(s.get("ts") or 0.0) >= cut]
            live_hold = float((rec.get("hold") or {}).get("until") or 0.0) > now
            live_block = _block_until(rec) > now
            if not rec["strikes"] and not live_hold and not live_block:
                machines.pop(key, None)
                removed += 1

    _update(path, _apply)
    return removed


# ------------------------------------------------------------------ scoring

def score(rec: Mapping[str, Any] | None, now: float | None = None) -> float:
    """Decayed, recurrence-weighted evidence against one machine.

        score = (Σ weight_i · 2^(-age_i/half_life) · superseded_i)
                · (1 + recurrence_bonus · (distinct_strike_days - 1))

    `distinct_strike_days` counts calendar-independent 24 h buckets from now, so
    the term cannot be gamed by a strike landing either side of midnight UTC.
    """
    if not isinstance(rec, Mapping):
        return 0.0
    now = time.time() if now is None else float(now)
    half = max(1e-9, config._boot_knob("HOSTREP_HALF_LIFE_D")) * _DAY_S
    last_ok = float(rec.get("last_ok_ts") or 0.0)
    base, days = 0.0, set()
    for s in (rec.get("strikes") or []):
        if not isinstance(s, Mapping):
            continue
        try:
            ts = float(s.get("ts") or 0.0)
        except (TypeError, ValueError):
            continue
        age = max(0.0, now - ts)
        w = STRIKE_WEIGHTS.get(str(s.get("kind")), DEFAULT_STRIKE_WEIGHT)
        w *= 2.0 ** (-age / half)
        if last_ok and ts < last_ok:
            w *= SUPERSEDED_DISCOUNT
        base += w
        days.add(int(age // _DAY_S))
    if base <= 0.0:
        return 0.0
    bonus = config._boot_knob("HOSTREP_RECURRENCE_BONUS")
    return base * (1.0 + bonus * max(0, len(days) - 1))


def _block_until(rec: Mapping[str, Any] | None) -> float:
    """When an earned block lifts, DERIVED from the crossing plus the current
    cooldown knob — so retuning `HOSTREP_BLOCK_COOLDOWN_D` moves every live
    block, which is what an operator changing a retry policy means by it.

    `blocked_until` is the pre-2026-08-20 shape, where the deadline was
    materialized at write time. Honoured verbatim for records that still carry
    it: it is the decision that was actually made, and re-deriving it from a
    crossing we did not record would be an invention.
    """
    if not isinstance(rec, Mapping):
        return 0.0
    at = rec.get("blocked_at")
    if at is not None:
        try:
            return (float(at)
                    + config._boot_knob("HOSTREP_BLOCK_COOLDOWN_D") * _DAY_S)
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(rec.get("blocked_until") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _blocked_rec(rec: Mapping[str, Any] | None, now: float) -> str | None:
    """Why this machine is excluded right now, AND when we will retry it, or
    None. Returns the reason so every caller can SAY it — an offer that silently
    vanishes from a market is the shape that makes an empty market
    unexplainable.

    The retry date is part of the reason, not a detail: every block here is
    temporary by design, and a message that says only "blocked" invites the
    operator to treat it as permanent and go clear it by hand."""
    if not isinstance(rec, Mapping):
        return None
    h = rec.get("hold")
    if isinstance(h, Mapping) and float(h.get("until") or 0.0) > now:
        return (f"operator hold ({h.get('reason') or 'no reason given'}), "
                f"retry in {_days(float(h['until']) - now)}")
    until = _block_until(rec)
    if until > now:
        n = len(rec.get("strikes") or [])
        return (f"{n} strike{'s' if n != 1 else ''}, score "
                f"{score(rec, now):.2f} >= "
                f"{config._boot_knob('HOSTREP_BLOCK_SCORE'):.2f}; "
                f"retry in {_days(until - now)}")
    return None


def _days(seconds: float) -> str:
    d = seconds / _DAY_S
    return f"{d:.1f}d" if d < 10 else f"{d:.0f}d"


def penalty_for_score(sc: float) -> float:
    """Score -> price multiplier, >= 1.0 and bounded by HOSTREP_PENALTY_MAX.

    A RANKING multiplier only. Nothing in this module ever feeds a bid, a
    ceiling or a budget: inflating a price we then pay would be a real cost
    increase bought with an advisory signal. Its only job is to answer "which
    of these two offers do we want", which is exactly where the directive
    lives — a host that boots is worth paying more for."""
    if sc <= 0.0:
        return 1.0
    per = config._boot_knob("HOSTREP_PENALTY_PER_POINT")
    return min(config._boot_knob("HOSTREP_PENALTY_MAX"), 1.0 + per * float(sc))


def verdicts(now: float | None = None,
             path: str | None = None) -> dict[str, dict[str, Any]]:
    """`{machine_id_str: {score, penalty, blocked_reason}}` for every machine we
    hold evidence on. One store read; the per-offer helpers below take this so a
    candidate walk does not re-derive it per row."""
    if not enabled():
        return {}
    now = time.time() if now is None else float(now)
    out: dict[str, dict[str, Any]] = {}
    for key, rec in (load(path).get("machines") or {}).items():
        sc = score(rec, now)
        reason = _blocked_rec(rec, now)
        if sc <= 0.0 and reason is None:
            continue
        out[str(key)] = {"score": sc, "penalty": penalty_for_score(sc),
                         "blocked_reason": reason}
    return out


def blocked_machines(now: float | None = None,
                     path: str | None = None) -> set[int]:
    """Machine ids no automatic lane may pick. Ints, because that is what the
    bundles query's `machine_id notin` filter and every `exclude_machines`
    caller already carry; an id that will not parse is dropped rather than
    poisoning the filter."""
    out: set[int] = set()
    for key, v in verdicts(now, path).items():
        if v.get("blocked_reason"):
            try:
                out.add(int(key))
            except (TypeError, ValueError):
                continue
    return out


def penalty(machine: object, *, verd: Mapping[str, Any] | None = None,
            now: float | None = None, path: str | None = None) -> float:
    """One machine's ranking multiplier. `verd` reuses a `verdicts()` read."""
    if machine is None:
        return 1.0
    v = verd if verd is not None else verdicts(now, path)
    rec = v.get(str(machine))
    if not isinstance(rec, Mapping):
        return 1.0
    try:
        return float(rec.get("penalty") or 1.0)
    except (TypeError, ValueError):
        return 1.0


# -------------------------------------------------------------- offer lanes

def rank_offers(offers: Sequence[Mapping[str, Any]], price_field: str, *,
                now: float | None = None, path: str | None = None,
                ) -> tuple[list[Mapping[str, Any]], list[str]]:
    """Reorder an already-cheapest-first offer list by REPUTATION-ADJUSTED price
    and drop blocked machines. Returns `(offers, notes)`; notes are lines the
    caller should print, because a pick that is not the cheapest one has to be
    explainable at the moment it happens.

    Stable in the absence of evidence: with an empty store this returns the
    input order unchanged, which is what keeps every existing caller's
    "cheapest-first" contract true. An offer whose price will not parse keeps
    its position rather than being scored against a guess."""
    rows = list(offers or [])
    if not rows or not enabled():
        return rows, []
    try:
        verd = verdicts(now, path)
    except Exception:
        return rows, []
    if not verd:
        return rows, []
    kept: list[tuple[float, int, Mapping[str, Any]]] = []
    dropped: list[str] = []
    moved = 0
    for i, o in enumerate(rows):
        mid = o.get("machine_id")
        v = verd.get(str(mid)) if mid is not None else None
        if v and v.get("blocked_reason"):
            dropped.append(f"machine {mid} ({v['blocked_reason']})")
            continue
        try:
            price = float(o.get(price_field))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            kept.append((float("inf"), i, o))
            continue
        mult = float(v.get("penalty") or 1.0) if v else 1.0
        if mult > 1.0:
            moved += 1
        kept.append((price * mult, i, o))
    kept.sort(key=lambda t: (t[0], t[1]))
    notes: list[str] = []
    if dropped:
        notes.append(">> host reputation: skipped " + ", ".join(dropped[:4])
                     + (f" (+{len(dropped) - 4} more)" if len(dropped) > 4
                        else ""))
    out = [o for _, _, o in kept]
    if moved and out and rows and out[0] is not rows[0]:
        first = out[0]
        old = rows[0]
        notes.append(
            f">> host reputation: preferring machine {first.get('machine_id')} "
            f"at ${_num(first.get(price_field)):.4f}/hr over the cheaper "
            f"machine {old.get('machine_id')} at "
            f"${_num(old.get(price_field)):.4f}/hr "
            f"(penalty x{penalty(old.get('machine_id'), verd=verd):.2f} — "
            f"a host that boots is worth the difference)")
    return out, notes


def _num(v: object) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def with_blocked(exclude: Iterable[object] | None,
                 now: float | None = None,
                 path: str | None = None) -> list[object] | None:
    """A caller's `exclude_machines` unioned with the durable block list.

    The seam for lanes that pass exclusions into the bundles QUERY rather than
    filtering a returned list — excluding server-side is strictly better there,
    because a blocked machine that fills the first page of a cheapest-first
    query would otherwise crowd out offers we can actually use.

    An entry that will not parse as an int is KEPT VERBATIM, not dropped. This
    function sits on the launch path in front of callers that already had a
    working exclusion list, and silently narrowing someone else's exclusion is
    the one failure mode a helper like this must not have — a dropped entry
    re-rents the exact machine the caller had evidence against."""
    ints: set[int] = set()
    other: list[object] = []
    for m in (exclude or []):
        try:
            ints.add(int(m))  # type: ignore[call-overload]
        except (TypeError, ValueError):
            other.append(m)
    try:
        ints |= blocked_machines(now, path)
    except Exception:
        pass
    out: list[object] = [*sorted(ints), *other]
    return out or None


# ------------------------------------------------------------------ display

def summary(now: float | None = None,
            path: str | None = None) -> list[dict[str, Any]]:
    """Worst-first rows for `herdd fleet hosts`."""
    now = time.time() if now is None else float(now)
    rows: list[dict[str, Any]] = []
    for key, rec in (load(path).get("machines") or {}).items():
        if not isinstance(rec, Mapping):
            continue
        strikes = [s for s in (rec.get("strikes") or []) if isinstance(s, Mapping)]
        sc = score(rec, now)
        rows.append({
            "machine_id": key,
            "score": round(sc, 3),
            "penalty": round(penalty_for_score(sc), 3),
            "blocked_reason": _blocked_rec(rec, now),
            "strikes": len(strikes),
            "kinds": sorted({str(s.get("kind")) for s in strikes}),
            "last_strike_ts": max((float(s.get("ts") or 0.0)
                                   for s in strikes), default=None),
            "distinct_days": len({int(max(0.0, now - float(s.get("ts") or 0.0))
                                      // _DAY_S) for s in strikes}),
            "last_ok_ts": float(rec.get("last_ok_ts") or 0.0) or None,
            "ok_count": int(rec.get("ok_count") or 0),
        })
    rows.sort(key=lambda r: (-float(r["score"]), str(r["machine_id"])))
    return rows
