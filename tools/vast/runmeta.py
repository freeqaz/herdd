#!/usr/bin/env python3
"""runmeta.py — safe, append-only run-metadata store on Backblaze B2.

The source of truth for training-run history. Design (see AUTOMATION_PLAN.md):

  * B2 has **no** conditional writes / CAS (405/NotImplemented) but **is**
    strongly consistent (read-after-write). So the ONLY safe pattern is
    **append-only, one immutable object per event**: never a shared mutable
    object (two writers GET->edit->PUT lose updates, and B2 can't catch it).
  * Layout: b2:$B2_BUCKET/runs/<RUN_ID>/events/<ts>-<actor>-<nonce>.json
    Every key is unique (urandom nonce) so concurrent writers never collide.
  * The current state of a run is a **fold** over its events. Filenames carry
    the *writer's* clock (box/laptop/supervisor are unsynchronized) so filename
    order is NOT a status oracle — status is computed by event SEMANTICS
    (relaunch epochs + a fixed precedence lattice + max for monotone fields).
  * Liveness ("running *now*?") is NOT in B2 — it is injected from the vast API
    (a live instance labelled run:<RUN_ID>). B2 records history; vast records
    reality.

Module boundary (load-bearing for the portable test lane):
  * `fold_events()` and `derive_status()` are **pure** (no I/O) — importable and
    callable with zero deps (no rclone/B2/net).
  * Transport is injectable: `emit_event`/`read_run` take a `runner` callable;
    `_default_runner` is the only thing that shells out to `rclone`.
  * This module imports NOTHING that shells out and NEVER imports `herdd`
    (whose helpers `sys.exit` on failure).
"""
from __future__ import annotations

import datetime
import json
import os
import re
import subprocess

# --- frozen schema (v1) ------------------------------------------------------
# Objects are immutable, so any format shipped lives forever. Freeze before the
# first emitter ships. The fold must tolerate unknown events + unknown fields.
SCHEMA_VERSION = 1

EVENTS = frozenset({
    "launched", "running", "checkpoint", "evicted", "relaunched", "done",
    "failed", "cost", "eval_done", "stopping", "heartbeat", "supervised",
    "supervisor_started", "supervisor_exiting",
    # v1 extension (2026-07-10, fold-neutral): `resumed` — a parked (operator-
    # stopped) instance was started again via `herdd start`. Emitted by the
    # CLI; clears the operator-intent read (herdd._last_stopping_actor) so a
    # later supervisor doesn't treat the old park as intent-to-stay-dead. The
    # fold ignores it (tolerates-unknown by design), so old readers are safe.
    "resumed",
    # v1 extension (2026-07-10, SPOT_DESIGN §3.2/§3.3): `preempted` — the box's
    # own SIGTERM trap fired (a preemption it saw coming). Kept OUT of `status`/
    # CHURN (that would make poll()'s emit_evicted gate skip its own confirming
    # `evicted` event) — instead it's a display-only `preempted_pending` field
    # (see fold_events) that reads as the `evicted` status tier, non-terminal,
    # until something confirms otherwise. `rescued` / `bid_raised` — supervisor
    # bid-defense events, informational/fold-neutral (liveness, not event
    # presence, already restores `running` once the SAME box is confirmed live
    # again).
    "preempted", "rescued", "bid_raised",
})
TERMINAL = frozenset({"done", "failed"})
CHURN = frozenset({"evicted", "relaunched"})          # each opens a new epoch
_CORE_KEYS = ("ts", "event", "run_id")                 # I1: required to be valid
_INF = float("inf")

# --- field-carrier tables (fold tolerance across emitter generations) --------
# Cumulative-spend carriers. `cost` is the canonical event, but the supervisor
# stamps the SAME vast-API figure onto every heartbeat (and onto `relaunched` /
# `handoff_complete`, the latter spelled `spend_usd`) far more often than it
# emits `cost` — folding `cost` alone understated supervised runs and left every
# unsupervised run null. All are cumulative snapshots, so MAX (I6), never sum.
_SPEND_KEYS = ("cost_usd", "spent_usd", "spend_usd")
# Events that price the CURRENT box. `relaunched` spells $/hr `bid_price` and
# carries no `dph` at all, so newest-launch-wins blanked the price on any run
# that was ever relaunched.
_PRICED_EVENTS = ("launched", "relaunched", "handoff_launch", "handoff_complete")
# Events that identify the current box without being a launch. Runs started
# outside `herdd train` (the workflow generate/score arms) emit no `launched`
# whatsoever, yet every stopping/resumed/evicted event they write carries the
# box id. Deliberately EXCLUDES handoff_launch/handoff_synced — those name the
# understudy, which is not the current box until handoff_complete.
_IID_EVENTS = ("launched", "relaunched", "resumed", "stopping", "evicted",
               "running")
# Fields the `emit --field K=V` CLI must coerce to numbers before writing.
_NUMERIC_FIELDS = ("dph", "bid_price", "step", "final_step", "cost_usd",
                   "spent_usd", "spend_usd", "relaunch_count", "rc", "disk",
                   "inet_down", "gpu_ram_gb")

RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RunmetaError(ValueError):
    pass


def validate_run_id(run_id: str) -> str:
    """RUN_ID is used raw in object keys, event filenames, env vars, and the
    `run:<id>` vast label — reject anything that would corrupt those."""
    if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
        raise RunmetaError(
            f"invalid RUN_ID {run_id!r}: must match {RUN_ID_RE.pattern}")
    return run_id


# --- time / identity ---------------------------------------------------------
def now_ts() -> str:
    """UTC compact basic-ISO with milliseconds: YYYYMMDDTHHMMSSmmmZ.

    Fixed-width -> lexicographic sort == chronological; NO colons (clean object
    key). Mirrors the box-side bash `date -u +%Y%m%dT%H%M%S%3NZ`. Deliberately
    NOT the STATUS marker's %FT%TZ (its colons are a key/sort wart)."""
    n = datetime.datetime.now(datetime.timezone.utc)
    return n.strftime("%Y%m%dT%H%M%S") + f"{n.microsecond // 1000:03d}Z"


def ts_succ(ts: str) -> str:
    """The smallest `now_ts`-shaped timestamp STRICTLY GREATER than `ts` (+1 ms).

    For the rare writer that must be provably NEWEST in the log rather than
    merely concurrent. `now_ts` is millisecond-resolution, so two events written
    in the same millisecond TIE, and every (ts, nonce) comparison in this system
    then falls through to a random 6-byte nonce — i.e. a coin flip. A writer
    whose whole purpose is to outrank an existing event (jobmeta's requeue
    `resumed` un-sticking a `failed`) must not depend on that: it steps past the
    newest ts it can see instead. Malformed input is returned unchanged; callers
    max() against `now_ts()` anyway, so the fallback is the ordinary path."""
    try:
        dt = datetime.datetime.strptime(ts, "%Y%m%dT%H%M%S%fZ")
    except (ValueError, TypeError):
        return ts
    dt += datetime.timedelta(milliseconds=1)
    return dt.strftime("%Y%m%dT%H%M%S") + f"{dt.microsecond // 1000:03d}Z"


def nonce() -> str:
    """6 bytes hex from urandom — never $RANDOM (15-bit, collides)."""
    return os.urandom(6).hex()


def _actor_slug(actor: str) -> str:
    # filename-safe actor: box:123 -> box_123 (matches onstart/train.sh emit_event)
    return re.sub(r"[^A-Za-z0-9._]", "_", actor or "unknown")


def make_event(run_id: str, event: str, actor: str, ts: str | None = None,
               **fields) -> dict:
    """Build an event dict with the v1 envelope. Unknown `event` values are
    allowed (the fold tolerates them) but warned by the CLI layer."""
    validate_run_id(run_id)
    ev = {
        "v": SCHEMA_VERSION,
        "ts": ts or now_ts(),
        "actor": actor,
        "event": event,
        "run_id": run_id,
        "nonce": fields.pop("nonce", None) or nonce(),
    }
    # drop None-valued fields so "opt" fields stay absent, not null
    ev.update({k: v for k, v in fields.items() if v is not None})
    return ev


def event_key(ev: dict) -> str:
    """Object key (path under runs/<run_id>/events/) for an event."""
    return f"{ev['ts']}-{_actor_slug(ev.get('actor', 'unknown'))}-{ev['nonce']}.json"


# --- parse / validate (I1) ---------------------------------------------------
def _coerce(raw) -> dict | None:
    """Return a valid event dict, or None (caller counts it as parse_errors).

    Accepts a dict (already parsed) or JSON str/bytes. Rejects empty, non-JSON,
    non-object, and any event missing a core key. One bad object NEVER breaks a
    fold (I1)."""
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return None
    if not isinstance(raw, dict):
        return None
    if not all(raw.get(k) for k in _CORE_KEYS):
        return None
    return raw


def _num(x):
    """Numeric coercion for fold fields — tolerant of STRING numerics.

    Events are immutable, so every format ever shipped lives forever and the fold
    is the ONLY place a typing mistake can be repaired. The `launched` emitter
    (herdd) goes through `runmeta.py emit --field K=V`, which stringifies every
    value, so `dph` has always landed as `"0.25"`; an isinstance-only check
    dropped it silently and `dph` read null on 100% of runs. Ints stay ints (a
    stringified `step` folds to 200, not 200.0). bool is NOT a number here."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return x if x == x and x not in (_INF, -_INF) else None
    if isinstance(x, str):
        s = x.strip()
        try:
            return int(s)
        except ValueError:
            pass
        try:
            f = float(s)
        except ValueError:
            return None
        return f if f == f and f not in (_INF, -_INF) else None
    return None


def _ts_epoch(ts):
    """runmeta ts (`YYYYMMDDTHHMMSSmmmZ`, see now_ts) -> unix seconds, else None.

    Span arithmetic only (derived cost); NEVER ordering — ordering is
    lexicographic on the fixed-width string by construction."""
    if not isinstance(ts, str) or len(ts) < 15:
        return None
    s = ts.rstrip("Z")
    for fmt in ("%Y%m%dT%H%M%S%f", "%Y%m%dT%H%M%S"):
        try:
            d = datetime.datetime.strptime(s, fmt)
        except ValueError:
            continue
        return d.replace(tzinfo=datetime.timezone.utc).timestamp()
    return None


# --- the fold (PURE — no I/O) ------------------------------------------------
def fold_events(raw_events, live_iids=()) -> dict:
    """Fold an unordered multiset of events into a run view (I7: tolerant to
    missing/extra/duplicate/out-of-order files; never mutates or deletes).

    `live_iids` is the INJECTED set of instance ids vast reports live (I5) — the
    only source of "running now"; NEVER inferred from event recency.

    Status is computed by SEMANTICS, not last-filename-wins (I3):
      1. split into relaunch epochs (each evicted/relaunched opens one);
      2. only the LATEST epoch decides status, by the precedence lattice
         terminal(done|failed) > evicted|relaunched > running > launched;
         a terminal is STICKY (a later-sorting running never resurrects it);
      3. monotone numeric fields fold by MAX across all events (I6: cost is
         cumulative snapshots -> max, never sum; step -> max).
    """
    live = set(live_iids or ())
    evs, parse_errors = [], 0
    for r in raw_events:
        e = _coerce(r)
        if e is None:
            parse_errors += 1
        else:
            evs.append(e)

    # deterministic order for within-epoch reasoning + display (NOT a status
    # oracle): (ts, nonce). nonce tiebreak keeps identical-ts events stable.
    evs.sort(key=lambda e: (e.get("ts", ""), e.get("nonce", "")))

    view = {
        "run_id": evs[-1]["run_id"] if evs else None,
        "status": "unknown", "live": False, "display_status": "unknown",
        "instance_id": None, "gpu": None, "offer_id": None, "dph": None,
        "config_hash": None, "runset": None, "base_model": None,
        "latest_step": None, "cost_usd": None, "cost_source": None,
        "relaunch_count": 0,
        "fail_reason": None, "fail_rc": None, "final_step": None,
        "started_at": None, "ended_at": None,
        "last_event": None, "last_event_ts": None,
        "n_events": len(evs), "parse_errors": parse_errors,
        "preempted_pending": False,
    }
    if not evs:
        return view

    # --- latest relaunch epoch --------------------------------------------
    last_churn = max((i for i, e in enumerate(evs)
                      if e.get("event") in CHURN), default=-1)
    latest = evs[last_churn:] if last_churn >= 0 else evs
    types = {e.get("event") for e in latest}

    if "failed" in types:                 # terminal, sticky; failed beats done
        status = "failed"
    elif "done" in types:
        status = "done"
    elif "running" in types:
        status = "running"
    elif "relaunched" in types:           # new box booting, not yet training
        status = "launched"
    elif "evicted" in types:              # evicted, not yet relaunched
        status = "evicted"
    elif "launched" in types:
        status = "launched"
    else:
        status = "unknown"
    view["status"] = status

    # --- monotone / provenance folds (across ALL events) ------------------
    steps = [s for e in evs if e.get("event") == "checkpoint"
             for s in (_num(e.get("step")),) if s is not None]
    if steps:
        view["latest_step"] = max(steps)
    costs = [c for e in evs for k in _SPEND_KEYS
             for c in (_num(e.get(k)),) if c is not None]
    if costs:
        view["cost_usd"] = max(costs)
        view["cost_source"] = "event"
    view["relaunch_count"] = max(
        sum(1 for e in evs if e.get("event") == "relaunched"),
        max((e["relaunch_count"] for e in evs
             if isinstance(e.get("relaunch_count"), int)), default=0),
    )

    # current instance from the newest launched|relaunched
    launch_like = [e for e in evs if e.get("event") in ("launched", "relaunched")]
    if launch_like:
        view["instance_id"] = launch_like[-1].get("instance_id")
    # gpu/offer_id: newest launch-like event that actually CARRIES one.
    # `relaunched` omits both, so plain newest-wins blanked them on relaunched
    # runs even though the opening `launched` recorded them.
    for key in ("gpu", "offer_id"):
        for e in reversed(launch_like):
            if e.get(key) not in (None, ""):
                view[key] = e[key]
                break
    # instance_id fallback: the newest non-launch event that names a box (see
    # _IID_EVENTS) — the only id source for runs that never emit `launched`.
    if view["instance_id"] in (None, ""):
        for e in reversed(evs):
            if (e.get("event") in _IID_EVENTS
                    and e.get("instance_id") not in (None, "")):
                view["instance_id"] = e["instance_id"]
                break
    # $/hr: newest event that prices the box, under either spelling.
    for e in reversed(evs):
        if e.get("event") not in _PRICED_EVENTS:
            continue
        d = _num(e.get("dph"))
        if d is None:
            d = _num(e.get("bid_price"))
        if d is not None:
            view["dph"] = d
            break
    # config is set at first launch
    first_launch = next((e for e in evs if e.get("event") == "launched"), None)
    if first_launch:
        view["config_hash"] = first_launch.get("config_hash")
        view["runset"] = first_launch.get("runset")
        view["base_model"] = first_launch.get("base_model")

    # terminal detail (from the actual terminal event, any epoch)
    failed = [e for e in evs if e.get("event") == "failed"]
    if failed:
        view["fail_reason"] = failed[-1].get("reason")
        view["fail_rc"] = failed[-1].get("rc")
    done = [e for e in evs if e.get("event") == "done"]
    if done:
        fs = _num(done[-1].get("final_step"))
        view["final_step"] = fs
        if fs is not None and (view["latest_step"] is None or fs > view["latest_step"]):
            view["latest_step"] = fs

    starts = [e["ts"] for e in evs if e.get("event") in ("launched", "running")]
    # Fallback to the earliest event in the log: a run whose log opens on a
    # `stopping`/`heartbeat` (no launch emitter) still has a real first-observed
    # timestamp, and `started_at` is contractually "first event" (lib/types.ts).
    view["started_at"] = min(starts) if starts else evs[0].get("ts")
    terms = [e["ts"] for e in evs if e.get("event") in TERMINAL]
    if terms:
        view["ended_at"] = max(terms)
    elif evs[-1].get("event") == "stopping":
        # No terminal event, but the newest thing that happened is an explicit
        # stop (operator park / workflow-arm teardown): that IS where it ended.
        view["ended_at"] = evs[-1].get("ts")
    view["last_event"] = evs[-1].get("event")
    view["last_event_ts"] = evs[-1].get("ts")

    # Derived cost — honest, and MARKED. Unsupervised runs emit no cost/heartbeat
    # at all, but a known $/hr times the observed span of the log is a real
    # estimate computed from real events, not an invention. It never overrides an
    # event-sourced figure, and `cost_source` tells every consumer to render it
    # as an approximation. Approximate by construction: it assumes one price for
    # the whole span and that the box stopped when the log did.
    if view["cost_usd"] is None and view["dph"]:
        t0 = _ts_epoch(view["started_at"])
        t1 = _ts_epoch(view["ended_at"] or view["last_event_ts"])
        if t0 is not None and t1 is not None and t1 > t0:
            view["cost_usd"] = round(view["dph"] * (t1 - t0) / 3600.0, 4)
            view["cost_source"] = "derived"

    # `preempted` (SPOT_DESIGN §3.3, box's own SIGTERM trap) is deliberately kept
    # OUT of `status`/CHURN above: `status` feeds poll()'s emit_evicted gate
    # (herdd.py) and folding preempted into it would make poll() skip its own
    # confirming `evicted` event (and the backoff-deadline init that rides on
    # it). This is a separate, display-only signal: true once a `preempted` has
    # fired with nothing chronologically after it (across ALL events, any
    # epoch) from {running, evicted, relaunched, done, failed} to contradict
    # it — i.e. the box saw the eviction coming and nothing has confirmed it
    # resumed or was formally evicted/relaunched yet.
    preempts = [i for i, e in enumerate(evs) if e.get("event") == "preempted"]
    view["preempted_pending"] = bool(preempts) and not any(
        e.get("event") in ("running", "evicted", "relaunched", "done", "failed")
        for e in evs[preempts[-1] + 1:])

    # --- liveness combine (I5): vast is the ONLY "running now" source -----
    # Compare as STRINGS: the vast API hands back int instance ids while the
    # CLI `--field instance_id=…` emitter writes them as str, so an `in` on the
    # raw values read False for every CLI-launched run (silently forcing
    # display_status to "stopped" on live boxes in the `runs --run` detail view).
    _live_s = {str(x) for x in live}
    view["live"] = bool(view["instance_id"] not in (None, "")
                        and str(view["instance_id"]) in _live_s)
    if status in TERMINAL:
        view["display_status"] = status
    elif view["live"]:
        view["display_status"] = "running"
    elif status == "evicted" or view["preempted_pending"]:
        # gone, and either the fold already says evicted or the box's own
        # preemption trap fired with nothing confirming it since: distinct
        # from an operator-parked box — supervise is expected to be
        # mid-rescue/relaunch, not "nothing to see here".
        view["display_status"] = "evicted"
    else:
        # non-terminal, no live instance, no eviction on record: STOPPED,
        # never "running" (I5)
        view["display_status"] = "stopped"
    return view


# --- legacy STATUS-marker fallback (pure) ------------------------------------
def status_marker_to_view(status_str: str | None, run_id: str | None = None,
                          live_iids=()) -> dict:
    """Minimal view derived from a legacy free-text checkpoints/<id>/STATUS
    marker (runs with no event log). Vocabulary per AUTOMATION_PLAN.md."""
    v = fold_events([], live_iids)          # empty skeleton
    v["run_id"] = run_id
    s = (status_str or "").strip()
    head = s.split(None, 1)[0].upper() if s else ""
    if head == "DONE":
        v["status"] = "done"
    elif head in ("FAILED", "STAGED"):
        v["status"] = "failed" if head == "FAILED" else "staged"
        v["fail_reason"] = s
    elif head in ("RUNNING", "LAUNCHED", "STARTING", ""):
        v["status"] = "running" if head == "RUNNING" else "launched"
    else:
        v["status"] = "unknown"
    v["last_event"] = f"STATUS:{s}" if s else None
    live = set(live_iids or ())
    v["live"] = bool(run_id and run_id in live)   # STATUS runs have no iid; label match handled by caller
    if v["status"] in ("done", "failed", "staged"):
        v["display_status"] = "done" if v["status"] == "done" else "failed"
    elif v["live"]:
        v["display_status"] = "running"
    else:
        v["display_status"] = "stopped"
    return v


def derive_status(view: dict) -> str:
    """Back-compat: render a view as a checkpoints/<id>/STATUS string that stays
    inside the vocabulary babysit parses (`herdd train` globs). Non-terminal
    events MUST map to RUNNING/token-matching-no-terminal-glob, else babysit
    would falsely tear down a live run the supervisor is about to relaunch."""
    s = view.get("status")
    ts = view.get("ended_at") or view.get("last_event_ts") or now_ts()
    if s == "done":
        return f"DONE {ts}"
    if s == "failed":
        reason = view.get("fail_reason")
        # keep it FAILED*-glob-matching; don't double-prefix if reason already is
        if reason and reason.upper().startswith("FAILED"):
            return reason
        return f"FAILED {reason + ' ' if reason else ''}{ts}"
    if s == "staged":
        return view.get("fail_reason") or f"STAGED {ts}"
    # launched / running / evicted / relaunched / unknown -> non-terminal
    return f"RUNNING {ts}"


def final_status(view: dict, *, status_marker: str | None = None,
                 instance_live: bool = False) -> dict:
    """Combine the event-fold view with the legacy STATUS marker + vast liveness
    into a terminal/display decision (I4). The event log wins when it is
    terminal. When the fold is non-terminal but the instance is GONE and the
    STATUS marker (or a published artifact) is terminal, the box was destroyed
    and can never emit the missing `done` — infer terminal so a reader never
    hangs forever. A live instance with a non-terminal fold is `running`."""
    s = view.get("status")
    if s in TERMINAL:                      # event log is authoritative
        return {"status": s, "terminal": True, "display": s}
    if not instance_live:
        # guard: "".split() -> [] so [0] would IndexError; status_marker is None
        # for any events-only run whose box is gone (the common non-terminal case).
        _parts = (status_marker or "").strip().split(None, 1)
        head = _parts[0].upper() if _parts else ""
        if head == "DONE":
            return {"status": "done", "terminal": True, "display": "done (inferred)"}
        if head in ("FAILED", "STAGED"):
            return {"status": "failed", "terminal": True, "display": "failed (inferred)"}
        if s == "evicted" or view.get("preempted_pending"):
            # gone, and either the fold says evicted or the box's own
            # preemption trap fired with nothing confirming it since (SPOT_DESIGN
            # §3.3): read as mid-recovery, not the generic "stopped" an
            # operator park also shows as.
            return {"status": s, "terminal": False, "display": "evicted"}
        return {"status": s, "terminal": False, "display": "stopped"}
    return {"status": s, "terminal": False, "display": "running"}


# --- transport seam (the ONLY thing that shells out) -------------------------
# Injectable-runner contract: runner(args, input=None) -> (rc, stdout, stderr).
# Tests pass a fake in-memory runner and never touch rclone.
def _default_runner(args, input=None):
    """Run rclone; return (rc, stdout, stderr). Never raises for a nonzero rc —
    callers decide. Missing rclone surfaces as rc=127 (soft), never sys.exit."""
    try:
        p = subprocess.run(["rclone", *args], capture_output=True, text=True,
                           input=input)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", "rclone not found on PATH"


def _bucket(bucket=None):
    b = bucket or os.environ.get("B2_BUCKET")
    if not b:
        raise RunmetaError("B2_BUCKET not set (env or bucket=)")
    return b


def _hostname():
    try:
        import socket
        return socket.gethostname()
    except Exception:
        return "unknown"


def _default_actor():
    iid = os.environ.get("VAST_INSTANCE_ID") or os.environ.get("INSTANCE_ID")
    if iid:
        return f"box:{iid}"
    return f"cli:{os.environ.get('HOSTNAME') or _hostname()}"


def emit_event(run_id, event, *, actor=None, runner=_default_runner,
               bucket=None, **fields) -> dict:
    """Append one immutable event object to runs/<run_id>/events/. Safe for any
    number of concurrent writers (unique nonce key). Best-effort: a transport
    failure returns the event with `_emitted=False` rather than raising, so a
    dying box's final emit can't crash the caller. `rclone rcat` reads the body
    from stdin (input=)."""
    ev = make_event(run_id, event, actor or _default_actor(), **fields)
    b = _bucket(bucket)
    key = f"runs/{run_id}/events/{event_key(ev)}"
    body = json.dumps(ev, separators=(",", ":")) + "\n"
    rc, _, err = runner(["rcat", f"b2:{b}/{key}"], input=body)
    ev["_key"] = key
    ev["_emitted"] = (rc == 0)
    if rc != 0:
        ev["_error"] = (err or "").strip()
    return ev


def read_run(run_id, *, runner=_default_runner, live_iids=(), bucket=None,
             cache_dir=None) -> dict:
    """Fold one run's event log into a view. Uses an incremental local cache of
    the immutable event bodies (rclone copy --include '*/events/*.json') so a
    warm read is pure directory listings + GETs for NEW events only — never
    re-GETs an event (keys are immutable). Falls back to the legacy STATUS
    marker when the run has no event log."""
    validate_run_id(run_id)
    b = _bucket(bucket)
    cache = cache_dir or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "vast-runmeta")
    dst = os.path.join(cache, run_id, "events")
    os.makedirs(dst, exist_ok=True)
    # incremental: rclone skips files already present locally (immutable keys)
    runner(["copy", f"b2:{b}/runs/{run_id}/events/", dst,
            "--transfers", "16", "--checkers", "32", "--fast-list"])
    bodies = []
    try:
        for name in os.listdir(dst):
            if name.endswith(".json"):
                try:
                    with open(os.path.join(dst, name), "rb") as fh:
                        bodies.append(fh.read())
                except OSError:
                    pass
    except OSError:
        pass
    if bodies:
        return fold_events(bodies, live_iids)
    # legacy fallback: read the STATUS marker
    rc, out, _ = runner(["cat", f"b2:{b}/checkpoints/{run_id}/STATUS"])
    return status_marker_to_view(out if rc == 0 else None, run_id, live_iids)


if __name__ == "__main__":                       # tiny CLI for manual poking
    import argparse
    ap = argparse.ArgumentParser(description="runmeta — B2 run-metadata")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("emit"); pe.add_argument("run_id"); pe.add_argument("event")
    pe.add_argument("--field", action="append", default=[], metavar="K=V")
    pr = sub.add_parser("read"); pr.add_argument("run_id")
    a = ap.parse_args()
    if a.cmd == "emit":
        f = dict(kv.split("=", 1) for kv in a.field)
        # `--field K=V` is inherently textual, so numerics arrived as strings and
        # the fold used to drop them (see _num). The fold now tolerates that for
        # the immutable history; coerce here so NEW events are correctly typed.
        # Allow-listed on purpose — a blanket json.loads would retype opaque
        # string fields (ids, reasons, geolocations) as a side effect.
        for k in _NUMERIC_FIELDS:
            if k in f:
                n = _num(f[k])
                if n is not None:
                    f[k] = n
        print(json.dumps(emit_event(a.run_id, a.event, **f), indent=2))
    elif a.cmd == "read":
        print(json.dumps(read_run(a.run_id), indent=2))
