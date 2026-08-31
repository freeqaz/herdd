#!/usr/bin/env python3
"""Bid-echo probe — measure, outright, how long vast echoes OUR OWN bid back to
us as a machine chunk's `min_bid` (FLEET_REVIEW_2026-08-14 item 3).

The measurement problem it answers
----------------------------------
On a chunk we are the tenant of, vast lists that chunk's `min_bid` as the price
to displace the current tenant — us (AUTOBID_DESIGN, "The market floor can be
OUR OWN BID"). The echo is not synchronous with our bid moves, so the self-floor
guard matches the floor against the recent standing-bid series inside
`bidpolicy.BID_SELF_FLOOR_LAG_S` (3600 s). That window is sized off PASSIVE
journal data, and passive data is **censored at the window by construction**: an
echo older than `lag_s` reads as a market floor and never journals as a match,
so the observed `matched_age_s` max can never exceed the window it is being used
to justify (the 2026-08-14 field review found a max at 98.7% of the old 900 s
edge — a censoring signature, not a tail).

This probe removes the censoring by moving the bid deliberately and watching the
floor with a clock: bid B -> B', then poll the machine's offer floor until it
reads B' on two consecutive polls. The elapsed time since the PUT is the echo
lag for that direction. Direction matters — probe v2 (2026-08-10, machine 52305)
measured a *lower* echoing 2-4x longer than a raise, and a lower is exactly the
decay-then-defend ratchet's precondition — so the default plan measures both:
raise, then return to the original bid, so the box ends at the price it started
at.

It is an ACTIVE probe on an ALREADY-RENTED spot box. It only ever PUTs bid
prices (never destroy/stop/start), never bids above `--max-price`, and restores
the original bid on any exit path, including SIGINT/SIGTERM.

Operator notes (this probe deliberately does NOT integrate with fleetd)
----------------------------------------------------------------------
* **Expect self-floor suppression events in the fleetd journal for the whole
  run.** Moving our own bid is precisely the shape `market_floor_self_match`
  exists to suppress; `*_bid_self_floor` events with `matched="prior"` and a
  growing `matched_age_s` are the guard working, not a fault. A sustained
  episode may also trip the 1800 s `*_bid_floor_blind` alarm — that too is
  expected while probing, and is the probe's finding rather than an incident.
* **Label the box `keep:<why>` before probing.** The idle reaper destroys a
  stopped box past 2 h, and nothing here holds the box for you.
* Any watch/ladder armed on this box will see the probe's moves and may move the
  bid itself, which corrupts the measurement (the floor would be echoing a price
  no phase of this probe chose). Probe a box with no armed ladder, or pause the
  watch first. The per-poll `observed_bid` column exists to catch exactly that:
  if it diverges from `our_bid`, somebody else is moving the price.

Output
------
NDJSON to `--out`, one row per read plus per-phase summaries, and a human
summary on stdout. Phase outcomes:

  flipped                 the floor reached the new bid on 2 consecutive polls;
                          `lag_s` = first read of that confirming run - the PUT
  censored-at-max-wait    it never did, within `--max-wait`. A RESULT (the tail
                          is at least this long), not an error — exit code 0
  no-echo-observed        the floor never matched our standing bid at all during
                          the baseline: this machine may not echo. Also a
                          finding; the probe stops without moving the bid
  bid-move-failed         the PUT was refused; nothing was measured

Usage
-----
    python3 tools/vast/bid_echo_probe.py \
        --box <IID> --out <run-dir>/echo-probe.ndjson --max-price <DPH>

See also: AUTOBID_DESIGN.md ("The echo has a lag window", "Measured again, with
attribution", "The field data answered — wider, not narrower"), bidpolicy.py
(`BID_SELF_FLOOR_LAG_S`, `BID_SELF_FLOOR_EPS`, `market_floor_self_match`).
"""
import argparse
import collections
import datetime
import json
import os
import signal
import sys
import time

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
DEFAULT_INTERVAL_S = 60.0
INTERVAL_FLOOR_S = 30.0        # vast API politeness: never poll faster than this
DEFAULT_MAX_WAIT_S = 5400.0    # 1.5 h per phase — 1.5x the pre-widening window
DEFAULT_BASELINE_WAIT_S = 900.0
DEFAULT_DELTA_PCT = 5.0
CONFIRM_READS = 2              # consecutive new-bid reads that confirm a flip

# Mirrors bidpolicy.BID_SELF_FLOOR_EPS, and for the same reason: the echo comes
# back QUANTIZED TO 3 DECIMALS while a standing bid round-trips at 4, so 0.0005
# is exactly the rounding radius of a 3-decimal echo. Kept as a local literal so
# the pure classifier needs no herdd import; the value must not drift from
# bidpolicy's (test_bid_echo_probe pins them equal).
ECHO_EPS = 0.0005

# One 3-decimal price-grid step. A bid move smaller than this cannot be told
# apart from its own quantized echo, so the probe refuses to start rather than
# measure a flip it could never see.
MIN_MOVE = 0.001

# Mirrors herdd.BID_RATE_LIMIT_S — the minimum spacing the probe keeps between
# its own PUTs, so a fast flip followed by an immediate restore does not race.
BID_RATE_LIMIT_S = 60.0

RESTORE_ATTEMPTS = 3
RESTORE_BACKOFF_S = 10.0

# `actual_status` values the probe will start on. The floor's self-echo
# semantics are TENANT-gated (AUTOBID_DESIGN: on a stopped box the same equality
# means somebody else now holds the chunk), so a probe of a non-running box
# would be measuring a different thing entirely.
RUNNING = "running"

PHASE_NAMES = ("raise", "lower")


class ProbeError(Exception):
    """A refusal or an abort. Carries no bid state — the caller restores."""


class ProbeAborted(Exception):
    """SIGTERM/SIGINT arrived; unwind through the restore path."""


# --------------------------------------------------------------------------- #
# seams — every vast API touch goes through one of these three functions, so the
# tests monkeypatch three names and no network exists for them. Each is a thin
# wrapper over herdd's existing plumbing: NO API logic is duplicated here.
# --------------------------------------------------------------------------- #
_HERDD = None


def _herdd():
    """Import herdd the way its own test files do (path-insert + import), and
    lazily, so `--help`, the pure helpers and the tests never load it."""
    global _HERDD
    if _HERDD is None:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        import herdd
        # herdd only auto-loads .env inside its own main(); a module import
        # skips it, so every soft API read fails keyless and read_box(None)s a
        # live box (first live run, 2026-08-14). load_env() is idempotent.
        herdd.load_env()
        _HERDD = herdd
    return _HERDD


def read_box(iid):
    """SEAM. One instance record, or None on any API failure/gone.

    Wraps `herdd._get_instance_soft` (soft GET v0/instances/<id>/) and
    `herdd._instance_standing_bid` — the latter because the STANDING BID is
    `dph_base`, never `dph_total` (= bid + storage): reading the total is the
    field confusion that made the exact-equality self-floor test unable to
    recognise our own bid on 2026-08-08."""
    v = _herdd()
    inst = v._get_instance_soft(iid)
    if not inst:
        return None
    return {
        "id": inst.get("id"),
        "machine_id": inst.get("machine_id"),
        "num_gpus": inst.get("num_gpus"),
        "is_bid": bool(inst.get("is_bid")),
        "actual_status": (inst.get("actual_status") or "").lower() or None,
        "standing_bid": v._instance_standing_bid(inst),
        "dph_total": v._num_dph(inst.get("dph_total")),
    }


def read_floor(machine_id, num_gpus=None):
    """SEAM. The machine's bid-offer floor for OUR gpu-count chunk.

    Wraps `herdd._market_min_bid_read` (POST v0/bundles/, type=bid, rentable),
    keeping its evidence intact: `ok=False` is ignorance (a failed read), while
    `ok=True, listed=False` is evidence the machine lists no rentable bid offer.
    `floors` is the per-chunk row list, NOT the min() collapse — on a machine we
    are a tenant of, one query returns both our rented chunk (min_bid = the
    echo) and any free sibling chunk (a genuine floor), and the probe logs both
    so an 'other' classification can be attributed after the fact."""
    v = _herdd()
    r = v._market_min_bid_read(machine_id, num_gpus)
    return {"ok": bool(r.ok), "listed": bool(r.listed), "min_bid": r.min_bid,
            "floors": list(r.floors or ()), "scaled": bool(r.scaled)}


def put_bid(iid, price):
    """SEAM. Change the standing bid in place; returns (ok, err).

    Wraps `herdd.set_bid` -> `_put_bid_soft` (PUT v0/instances/bid_price/<id>/),
    which is soft by contract: a 429 or a 200-with-`success: false` comes back as
    (False, err) rather than an exception."""
    return _herdd().set_bid(iid, float(price))


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def quantize(price):
    """The 3-decimal grid every ladder rung rounds to before a PUT, so our moved
    bids echo back bit-exact (AUTOBID_DESIGN, probe v2 finding 3)."""
    return round(float(price), 3)


def classify_floor(floor, old_bid, new_bid, eps=ECHO_EPS):
    """PURE. What is this floor read? -> "new" | "old" | "other" | "unread".

    `new` wins ties: after the move, the question the probe asks is "has the
    echo caught up yet", and a floor that matches both prices is a move too
    small to measure (refused up front by `build_plan`/`preflight`)."""
    if floor is None:
        return "unread"
    f = float(floor)
    if new_bid is not None and abs(f - float(new_bid)) <= eps:
        return "new"
    if old_bid is not None and abs(f - float(old_bid)) <= eps:
        return "old"
    return "other"


Phase = collections.namedtuple("Phase", ["name", "from_bid", "to_bid"])


def build_plan(orig_bid, delta_pct, phases):
    """PURE. The sequential bid plan: [(name, from, to), ...].

    `raise` moves up by `delta_pct` from wherever the bid currently is; `lower`
    returns to the ORIGINAL bid when a previous phase raised us off it (so the
    default `raise,lower` run ends the box at the price it started at), and
    otherwise moves down by `delta_pct`."""
    orig = quantize(orig_bid)
    cur, plan = orig, []
    for name in phases:
        if name not in PHASE_NAMES:
            raise ProbeError(f"unknown phase {name!r} "
                             f"(want: {'/'.join(PHASE_NAMES)})")
        if name == "raise":
            tgt = quantize(cur * (1.0 + delta_pct / 100.0))
        elif cur > orig + 1e-9:
            tgt = orig
        else:
            tgt = quantize(cur * (1.0 - delta_pct / 100.0))
        plan.append(Phase(name, cur, tgt))
        cur = tgt
    return plan


def preflight(box, plan, max_price, orig_bid=None):
    """PURE. Every reason to refuse to start, as a list of strings (empty = go).

    The `--max-price` cap is checked here AND again inside every PUT: a cap that
    only guards the plan cannot stop a bug in the move path from spending."""
    errs = []
    if not box:
        errs.append("box not found (or the instance read failed)")
        return errs
    if not box.get("is_bid"):
        errs.append("box is not a bid (spot) box — there is no standing bid to "
                    "move, and the floor of an on-demand box is not an echo")
    if box.get("actual_status") != RUNNING:
        errs.append(f"box actual_status is {box.get('actual_status')!r}, want "
                    f"{RUNNING!r}: the self-echo is tenant-gated, so a "
                    f"non-running box measures a different thing")
    if not box.get("machine_id"):
        errs.append("no machine_id on the instance record — cannot read the "
                    "machine's offer floor")
    bid = orig_bid if orig_bid is not None else box.get("standing_bid")
    if not bid or float(bid) <= 0:
        errs.append("no standing bid (dph_base) on the instance record")
    elif max_price is not None and float(bid) > float(max_price) + 1e-9:
        errs.append(f"the box's CURRENT bid ${float(bid):.4f} already exceeds "
                    f"--max-price ${float(max_price):.4f}")
    for ph in plan:
        if max_price is not None and ph.to_bid > float(max_price) + 1e-9:
            errs.append(f"phase {ph.name!r} would bid ${ph.to_bid:.4f} > "
                        f"--max-price ${float(max_price):.4f}")
        if abs(ph.to_bid - ph.from_bid) < MIN_MOVE - 1e-12:
            errs.append(
                f"phase {ph.name!r} moves ${ph.from_bid:.4f} -> "
                f"${ph.to_bid:.4f}, under one ${MIN_MOVE} price-grid step: a "
                f"3-decimal echo could not distinguish the two prices. Raise "
                f"--delta-pct")
    return errs


# --------------------------------------------------------------------------- #
# event log
# --------------------------------------------------------------------------- #
class EventLog:
    """Append-only NDJSON. Every row carries ts/iso/event plus the box+machine
    identity, so a concatenation of several probes stays attributable."""

    def __init__(self, path, *, box=None, machine=None, now=time.time):
        self.path = path
        self.box = box
        self.machine = machine
        self._now = now
        self.rows = []
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def emit(self, event, **fields):
        ts = float(self._now())
        row = {"ts": round(ts, 3),
               "iso": datetime.datetime.fromtimestamp(
                   ts, datetime.timezone.utc).isoformat(),
               "event": event, "box": self.box, "machine": self.machine}
        row.update(fields)
        self._fh.write(json.dumps(row, sort_keys=False) + "\n")
        self._fh.flush()
        self.rows.append(row)
        return row

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# the probe
# --------------------------------------------------------------------------- #
PhaseResult = collections.namedtuple(
    "PhaseResult", ["name", "from_bid", "to_bid", "outcome", "lag_s",
                    "confirm_lag_s", "polls", "counts", "baseline"])

BaselineResult = collections.namedtuple(
    "BaselineResult", ["outcome", "polls", "matches", "target_seen"])


class EchoProbe:
    """The per-phase state machine. Holds no argparse and no printing beyond the
    per-poll progress line, so tests drive it directly with fake seams and a
    fake clock."""

    def __init__(self, *, box, machine, num_gpus, orig_bid, log, max_price,
                 interval=DEFAULT_INTERVAL_S, max_wait=DEFAULT_MAX_WAIT_S,
                 baseline_wait=DEFAULT_BASELINE_WAIT_S, confirm=CONFIRM_READS,
                 eps=ECHO_EPS, now=time.time, sleep=time.sleep, quiet=False):
        self.box = box
        self.machine = machine
        self.num_gpus = num_gpus
        self.orig_bid = quantize(orig_bid)
        self.log = log
        self.max_price = float(max_price)
        self.interval = float(interval)
        self.max_wait = float(max_wait)
        self.baseline_wait = float(baseline_wait)
        self.confirm = int(confirm)
        self.eps = float(eps)
        self._now = now
        self._sleep = sleep
        self.quiet = quiet
        self.current_bid = self.orig_bid    # what we believe we hold
        self.bid_uncertain = False          # a PUT whose outcome we don't know
        self.last_put_ts = None

    # -- reads ------------------------------------------------------------- #
    def _poll(self, phase, stage, old_bid, new_bid, since=None):
        """One market read + one box read, classified and journaled."""
        fr = read_floor(self.machine, self.num_gpus) or {}
        floor = fr.get("min_bid") if fr.get("ok") else None
        matched = classify_floor(floor, old_bid, new_bid, self.eps)
        obs = None
        b = read_box(self.box)
        if b:
            obs = b.get("standing_bid")
        row = self.log.emit(
            "poll", phase=phase, stage=stage, our_bid=self.current_bid,
            observed_bid=obs, floor_read=floor, matched=matched,
            listed=fr.get("listed"), read_ok=fr.get("ok"),
            floors=fr.get("floors"), scaled=fr.get("scaled"),
            old_bid=old_bid, new_bid=new_bid,
            since_move_s=(None if since is None
                          else round(float(self._now()) - since, 1)))
        if not self.quiet:
            print(f"  [{phase}/{stage}] floor={floor} bid={self.current_bid} "
                  f"observed={obs} -> {matched}")
        return matched, floor, row

    # -- writes ------------------------------------------------------------ #
    def _move_bid(self, price, *, why):
        """The ONLY place a price is PUT. Re-checks --max-price (a cap that only
        guards the plan cannot stop a bug in here) and keeps the vast-side rate
        limit spacing between our own PUTs."""
        price = quantize(price)
        if price > self.max_price + 1e-9:
            raise ProbeError(f"refusing to bid ${price:.4f} > --max-price "
                             f"${self.max_price:.4f} ({why})")
        if self.last_put_ts is not None:
            wait = BID_RATE_LIMIT_S - (float(self._now()) - self.last_put_ts)
            if wait > 0:
                self._sleep(wait)
        was = self.current_bid
        self.bid_uncertain = True
        ok, err = put_bid(self.box, price)
        self.last_put_ts = float(self._now())
        if ok:
            self.current_bid = price
            self.bid_uncertain = False
        self.log.emit("bid_move", why=why, target=price, from_bid=was,
                      ok=bool(ok), error=(None if ok else str(err)))
        return bool(ok), err

    def restore(self):
        """Put the box back on its starting price. Idempotent, and safe to call
        from a finally: a no-op when we never moved and never lost track."""
        if abs(self.current_bid - self.orig_bid) < 1e-9 and not self.bid_uncertain:
            return True
        for attempt in range(1, RESTORE_ATTEMPTS + 1):
            try:
                if self.last_put_ts is not None:
                    wait = BID_RATE_LIMIT_S - (float(self._now())
                                               - self.last_put_ts)
                    if wait > 0:
                        self._sleep(wait)
                ok, err = put_bid(self.box, self.orig_bid)
                self.last_put_ts = float(self._now())
            except Exception as e:                       # a seam that raised
                ok, err = False, f"{type(e).__name__}: {e}"
            self.log.emit("bid_restore", target=self.orig_bid, attempt=attempt,
                          ok=bool(ok), error=(None if ok else str(err)))
            if ok:
                self.current_bid = self.orig_bid
                self.bid_uncertain = False
                if not self.quiet:
                    print(f"  restored bid to ${self.orig_bid}")
                return True
            if attempt < RESTORE_ATTEMPTS:
                self._sleep(RESTORE_BACKOFF_S)
        print(f"WARNING: could not restore the bid on {self.box} to "
              f"${self.orig_bid} — check it by hand "
              f"(`herdd bid <id> --price {self.orig_bid}`)", file=sys.stderr)
        return False

    # -- phases ------------------------------------------------------------ #
    def _baseline(self, phase, cur_bid, target):
        """Poll until the floor STABLY echoes the bid we currently hold.

        `confirm` consecutive matches = stable. Zero matches inside
        `baseline_wait` = "no-echo-observed": this machine's listing may simply
        never surface our rented chunk (the offers query alternates between our
        chunk and free sibling chunks — probe v2 finding 1), which is itself the
        result. Matches that never land consecutively are "unstable": the echo
        is intermittent, the phase still runs, and the summary says so."""
        t0 = float(self._now())
        streak = matches = polls = 0
        target_seen = False
        while True:
            matched, _floor, _row = self._poll(phase, "baseline", cur_bid, target)
            polls += 1
            if matched == "old":
                matches += 1
                streak += 1
            elif matched == "new":
                # the floor already reads the price this phase is about to move
                # to — a coincident competitor at exactly our target makes the
                # flip signal ambiguous. Recorded, not fatal.
                target_seen = True
                streak = 0
            elif matched == "other":
                streak = 0
            if streak >= self.confirm:
                return BaselineResult("stable", polls, matches, target_seen)
            if float(self._now()) - t0 >= self.baseline_wait:
                outcome = "unstable" if matches else "no-echo-observed"
                return BaselineResult(outcome, polls, matches, target_seen)
            self._sleep(self.interval)

    def run_phase(self, phase):
        """Baseline -> move -> poll for the flip. Returns a PhaseResult and
        journals a `phase_summary` row."""
        self.log.emit("phase_start", phase=phase.name, from_bid=phase.from_bid,
                      to_bid=phase.to_bid, interval_s=self.interval,
                      max_wait_s=self.max_wait)
        if not self.quiet:
            print(f"phase {phase.name}: ${phase.from_bid} -> ${phase.to_bid}")
        base = self._baseline(phase.name, phase.from_bid, phase.to_bid)
        if base.outcome == "no-echo-observed":
            return self._summary(phase, "no-echo-observed", None, None, 0,
                                 collections.Counter(), base)

        ok, err = self._move_bid(phase.to_bid, why=f"phase:{phase.name}")
        if not ok:
            if not self.quiet:
                print(f"  bid move refused: {err}")
            return self._summary(phase, "bid-move-failed", None, None, 0,
                                 collections.Counter(), base)
        t_move = float(self._now())

        counts = collections.Counter()
        streak, first_new, polls = 0, None, 0
        while True:
            matched, _floor, row = self._poll(phase.name, "flip", phase.from_bid,
                                              phase.to_bid, since=t_move)
            counts[matched] += 1
            polls += 1
            if matched == "new":
                streak += 1
                if first_new is None:
                    first_new = row["ts"]
            elif matched == "old":
                # the echo is still stale: the confirming run restarts.
                streak, first_new = 0, None
            # "other" (a real competitor / a sibling chunk floor) and "unread"
            # (a failed read) are evidence about neither price: recorded, but
            # they neither confirm nor refute the flip, so the streak stands.
            if streak >= self.confirm:
                return self._summary(phase, "flipped",
                                     round(first_new - t_move, 1),
                                     round(row["ts"] - t_move, 1),
                                     polls, counts, base)
            if float(self._now()) - t_move >= self.max_wait:
                return self._summary(phase, "censored-at-max-wait", None, None,
                                     polls, counts, base)
            self._sleep(self.interval)

    def _summary(self, phase, outcome, lag_s, confirm_lag_s, polls, counts, base):
        res = PhaseResult(phase.name, phase.from_bid, phase.to_bid, outcome,
                          lag_s, confirm_lag_s, polls, dict(counts),
                          dict(base._asdict()))
        self.log.emit("phase_summary", phase=phase.name,
                      from_bid=phase.from_bid, to_bid=phase.to_bid,
                      outcome=outcome, lag_s=lag_s, confirm_lag_s=confirm_lag_s,
                      polls=polls, counts=dict(counts),
                      baseline=dict(base._asdict()), max_wait_s=self.max_wait,
                      interval_s=self.interval)
        return res

    def run(self, plan):
        results = []
        for phase in plan:
            res = self.run_phase(phase)
            results.append(res)
            if res.outcome in ("no-echo-observed", "bid-move-failed"):
                break                       # nothing later can measure better
        return results


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def format_summary(results, orig_bid, out_path):
    lines = ["", "bid-echo probe — results", "-" * 40]
    for r in results:
        head = f"{r.name:<6} ${r.from_bid} -> ${r.to_bid}: {r.outcome}"
        if r.outcome == "flipped":
            head += (f" — echo lag {r.lag_s}s "
                     f"(confirmed at {r.confirm_lag_s}s, {r.polls} polls)")
        elif r.outcome == "censored-at-max-wait":
            head += (f" — the old bid still echoed after {r.polls} polls; the "
                     f"true lag is > the --max-wait ceiling")
        elif r.outcome == "no-echo-observed":
            head += (" — the floor never matched our bid; this machine may not "
                     "echo")
        lines.append(head)
        if r.counts:
            lines.append(f"       reads: {r.counts}")
        if r.baseline:
            lines.append(f"       baseline: {r.baseline}")
    if not results:
        lines.append("(no phase ran)")
    lines.append(f"original bid: ${orig_bid}")
    lines.append(f"events: {out_path}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        prog="bid_echo_probe.py",
        description="Measure vast's self-bid echo lag on an already-rented spot "
                    "box, per direction (raise/lower).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The probe only ever PUTs bid prices, never above --max-price, "
               "and restores the original bid on every exit path. Label the box "
               "keep:<why> first, and expect self-floor suppressions in the "
               "fleetd journal while it runs.")
    p.add_argument("--box", required=True, type=int, metavar="IID",
                   help="instance id of an ALREADY-RENTED, running spot box")
    p.add_argument("--out", required=True, metavar="PATH",
                   help="ndjson event log (appended)")
    p.add_argument("--max-price", required=True, type=float, metavar="DPH",
                   help="hard cap: the probe never bids above this $/hr")
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S,
                   help=f"poll cadence, seconds (default {DEFAULT_INTERVAL_S:g}, "
                        f"floored at {INTERVAL_FLOOR_S:g} for API politeness)")
    p.add_argument("--max-wait", type=float, default=DEFAULT_MAX_WAIT_S,
                   help=f"per-phase ceiling, seconds (default "
                        f"{DEFAULT_MAX_WAIT_S:g}); reaching it is the outcome "
                        f"censored-at-max-wait, not an error")
    p.add_argument("--baseline-wait", type=float, default=DEFAULT_BASELINE_WAIT_S,
                   help=f"per-phase baseline ceiling, seconds (default "
                        f"{DEFAULT_BASELINE_WAIT_S:g})")
    p.add_argument("--delta-pct", type=float, default=DEFAULT_DELTA_PCT,
                   help=f"bid move size, %% (default {DEFAULT_DELTA_PCT:g})")
    p.add_argument("--phases", default=",".join(PHASE_NAMES),
                   help="comma list of raise/lower (default raise,lower — raise "
                        "first, then back to the original bid)")
    p.add_argument("--quiet", action="store_true", help="no per-poll progress")
    return p


def _install_signal_handlers():
    def _handler(signum, _frame):
        raise ProbeAborted(f"signal {signum}")
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):        # not the main thread
            pass


def main(argv=None, *, now=time.time, sleep=time.sleep, install_signals=True):
    a = build_parser().parse_args(argv)
    phases = [x.strip() for x in a.phases.split(",") if x.strip()]
    if not phases:
        print("error: --phases is empty", file=sys.stderr)
        return 2
    interval = float(a.interval)
    if interval < INTERVAL_FLOOR_S:
        print(f"note: --interval {interval:g}s raised to the "
              f"{INTERVAL_FLOOR_S:g}s politeness floor", file=sys.stderr)
        interval = INTERVAL_FLOOR_S

    box = read_box(a.box)
    orig = (quantize(box["standing_bid"])
            if (box and box.get("standing_bid")) else None)
    try:
        plan = build_plan(orig if orig is not None else 0.0, a.delta_pct, phases)
    except ProbeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    errs = preflight(box, plan, a.max_price, orig_bid=orig)
    if errs:
        for e in errs:
            print(f"error: {e}", file=sys.stderr)
        return 2

    log = EventLog(a.out, box=a.box, machine=box.get("machine_id"), now=now)
    probe = EchoProbe(box=a.box, machine=box.get("machine_id"),
                      num_gpus=box.get("num_gpus"), orig_bid=orig, log=log,
                      max_price=a.max_price, interval=interval,
                      max_wait=a.max_wait, baseline_wait=a.baseline_wait,
                      now=now, sleep=sleep, quiet=a.quiet)
    log.emit("probe_start", orig_bid=orig, max_price=a.max_price,
             interval_s=interval, max_wait_s=a.max_wait,
             baseline_wait_s=a.baseline_wait, delta_pct=a.delta_pct,
             phases=[dict(p._asdict()) for p in plan],
             num_gpus=box.get("num_gpus"))
    if install_signals:
        _install_signal_handlers()

    results, rc, failure = [], 0, None
    try:
        results = probe.run(plan)
    except (ProbeAborted, KeyboardInterrupt) as e:
        failure, rc = f"aborted: {e or 'interrupt'}", 130
    except Exception as e:
        failure, rc = f"{type(e).__name__}: {e}", 1
    finally:
        restored = probe.restore()
        log.emit("probe_end", outcomes=[r.outcome for r in results],
                 failure=failure, restored=bool(restored),
                 final_bid=probe.current_bid)
    if failure:
        print(f"error: {failure}", file=sys.stderr)
    print(format_summary(results, orig, a.out))
    log.close()
    return rc


if __name__ == "__main__":
    sys.exit(main() or 0)
