"""vastlib.fleet.state — the fleetd state document and its journal. SCHEMA FROZEN.

Why this exists
---------------
`~/.local/state/vast-fleetd/state.json` is the only thing that survives a
`fleet deploy`. Every durable spend bound the eviction ladder owns lives in it:
the ceiling ledger, the per-box accrual, the replacement counters, the retention
deadlines. A daemon that fails to LOAD it does not error — it cold-re-adopts,
which reads as a working fleet while every cap resets to a provisional default
and every ladder restarts its budget of autonomous rentals. That failure is
silent by construction, which is why the schema is a plan §4 frozen contract and
why this module is separable from the tick loop at all.

The three contracts that bind AT DEPLOY (not at test time)
-----------------------------------------------------------
1. **Key names and the serialization flags.** `save_state` writes
   `json.dump(indent=1, sort_keys=True, default=str)`. `default=str` is what
   silently stringifies a stray `set`; `indent=1` is why the step-0 snapshot can
   be byte-compared at all. Neither is cosmetic.
2. **The rollback direction.** fleetd runs live from a checkout and rolls back by
   redeploying the PRIOR revision, so the new writer must emit a document the
   OLD reader accepts. Additive keys are safe (the old `_load` ignores what it
   does not setdefault). A RENAMED or RETYPED key is not — and a non-dict where
   the old code `isinstance`-checks (`notify` / `ceilings` / `ceiling_by_box`)
   is silently RESET to empty, which for the ceiling ledger means "provisional
   default cap for every adoption": a spend-policy change wearing the costume of
   a load.
3. **`LOCK_NAME`.** If the ported daemon flocks a differently-named file, it
   does not see the running daemon's lock, and two reconcilers drive one fleet.

`STATE_NAME` and `JOURNAL_NAME` are additionally DUPLICATED outside this package
— `herdd.fleet_state_path` / `fleet_journal_path` hardcode the same two
literals, and `tools/vast/fleet_report.py` carries a third copy of the journal
name. They are frozen here rather than imported from `fleet.client` so that a
`client` that has not landed yet cannot make this module unimportable; the two
places must agree, and `test_vastlib_fleet_state.py` asserts they do.

The conventions this module is not allowed to "clean up"
--------------------------------------------------------
* **A corrupt state file is quarantined, never raised.** `load_state` moves it
  aside to `<path>.corrupt-<epoch>` and rebuilds empty (S5): a daemon that
  crash-loops on its own state file is a fleet with no supervisor at all. Every
  `OSError` on the move is swallowed — the rebuild matters, the tidy-up does
  not.
* **Key ABSENCE is load-bearing on the restore side.** `_replacement_state_restore`
  skips keys the record does not carry, leaving them to `job_supervise_init`'s
  zeros, so a watch written before a feature existed reads as "this watch has
  rented nothing yet". Pre-populating a default there would hand an old watch a
  fabricated history.
* **`if out:` on both persist halves.** An empty projection writes NOTHING, so
  an untouched watch keeps its prior record rather than having it blanked.
* **The two lanes persist under DIFFERENT watch keys** — `w["replacement"]` for
  the jobs lane, `w["run_state"]` for the run lane. Two sub-documents, one
  schema; the asymmetry is deliberate and the names are wire format.
* **`iso` raises on a falsy timestamp.** It is NOT
  `supervise.journal._iso_z`, which answers `None`. Every caller here and in
  `fleet.rows` guards first; merging the two would turn a would-be crash into a
  silent `None` in a journal field nobody reads until they need it.
* **A journal write failure never stops a tick.** It prints and continues, and
  the line still goes to stdout — journald is the second sink, and the daemon's
  log IS the journal.

What is deliberately NOT here
-----------------------------
* **No module-level `vastlib` import, on purpose.** `supervise/state.py`
  re-exports `REPLACEMENT_STATE_KEYS` / `RUN_STATE_KEYS` from here (integrator
  ruling, 2026-08-16: this file is THE DEFINITION, so three copies cannot
  drift). `fleet.client` — whose `Hooks` reach `supervise.run_lane` — would
  close that into an import cycle, so `state_dir()` imports `client` inside the
  function. That is the whole reason for the local import; do not hoist it.
* **No runtime state.** `Store` holds the persisted document, the two paths, a
  lock and a clock. `Fleet.runtime`, `_alarm_since`, `_health`, the tick
  counters and the alarm latch machinery stay in `fleet/daemon.py`: they are
  what the daemon knows between ticks, and none of it is durable.
* **No alarm derivation and no `_Policy`.** `_Policy`'s None-for-missing
  `__getattr__` IS a state.json compat device (an old watch's policy dict must
  never `AttributeError` a tick) and its port in `daemon.py` cites this
  module's schema note — but it is watch-registration policy, not persistence.
* **`VERSION` is the state document's `version` field.** fleetd spends the same
  literal on its socket protocol check; `fleet/client.py` owns that half as
  `FLEET_PROTO_VERSION`. Today they are one constant serving two contracts,
  which is a collapse to decide at step 6 — not a rename to make here, because
  bumping either one is a schema break.

Provenance: verbatim-with-types move from `tools/vast/fleetd.py` (plan §8 step
5, 2026-08-16), each symbol carrying its `# moved-from:` marker. ADD-ONLY:
`fleetd.py` keeps its live copies until step 6. Round-trip evidence:
`tools/vast/test_vastlib_fleet_state.py` replays the step-0 live snapshot at
`tools/vast/testfixtures/fleetd_state_snapshot_2026-08-16/`.
"""

from __future__ import annotations

import datetime
import fcntl
import json
import os
import shutil
import sys
import threading
import time
from typing import Any, Callable, Final, TextIO

# --------------------------------------------------------------------------- #
# frozen names (plan §4). Changing any literal below is a deploy-visible schema
# break, not a refactor: an external reader keeps reading a file nobody writes.
#
# DUPLICATE RULINGS (2026-08-16, wave 6a; fleetd-reexports H4). `fleetd` spelled
# the protocol version and the two filenames once each, and both `fleet/client.py`
# and this module ported them — two vastlib targets for one flat name, which the
# test migration cannot rewrite. Split by what each name IS:
#   * `fleetd.VERSION` -> `client.FLEET_PROTO_VERSION`. The flat constant is the
#     WIRE protocol version the client and the daemon negotiate, and `client.py`
#     is the one definition of the wire protocol. `state.VERSION` below is the
#     state.json SCHEMA version — same integer today, different fact, no rename
#     claim on the flat symbol, so it is deliberately marker-less.
#   * `fleetd.STATE_NAME` / `fleetd.JOURNAL_NAME` -> HERE. They are on-disk
#     names owned by the state writer; `client.py` dropped its markers.
# --------------------------------------------------------------------------- #
VERSION: Final = 1                 # state.json `version`. No reader branches on it
                                   # today; bumping it is the schema break.
# moved-from: fleetd.JOURNAL_MAX_BYTES
JOURNAL_MAX_BYTES: Final = 32 * 1024 * 1024
# moved-from: fleetd.JOURNAL_NAME
JOURNAL_NAME: Final = "journal.ndjsonl"
# moved-from: fleetd.STATE_NAME
STATE_NAME: Final = "state.json"
# moved-from: fleetd.LOCK_NAME
LOCK_NAME: Final = "fleetd.lock"


# moved-from: fleetd.iso
def iso(ts: float) -> str:
    """`%Y-%m-%dT%H:%M:%SZ` UTC. RAISES on None — see the module docstring on
    why this is not `supervise.journal._iso_z`."""
    return datetime.datetime.fromtimestamp(
        ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# moved-from: fleetd.state_dir
def state_dir() -> str:
    """The daemon's state directory, created if missing.

    `client.fleet_state_dir()` is the ONE place the `FLEETD_STATE_DIR` env knob
    is read; this function only adds the `makedirs`. The import is function-local
    deliberately (module docstring: `supervise.state` re-exports from this module,
    and `fleet.client` reaches `supervise` through its hooks — a module-level
    import here would close the cycle)."""
    from vastlib.fleet import client
    d = client.fleet_state_dir()
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# the two durable PROJECTIONS — the only parts of the supervise knot's `jc`/`st`
# that outlive the process.
#
# THIS IS THE DEFINITION (integrator ruling, 2026-08-16). `fleetd.py` holds a
# live copy until plan step 6 (add-only), and `vastlib/supervise/state.py`
# RE-EXPORTS from here rather than carrying a third literal;
# `test_vastlib_supervise_state.py` binds all of them together, AST-parsing
# `fleetd.py` and asserting object identity with the supervise re-export. Both
# tuples are written as PLAIN assignments of tuple literals — no `Final`, no
# annotation — so that AST-parity helper can read this file the same way it
# reads `fleetd.py`.
# --------------------------------------------------------------------------- #
# jc keys the eviction-replacement ladder accumulates that MUST survive a daemon
# restart. Without them a restart hands the ladder a fresh budget of autonomous
# rentals (the count cap resets to 0) and a fresh price anchor (the ceiling
# re-derives from the REPLACEMENT's price, ratcheting 2x per restart) — both of
# them spend bounds, so both are durable state, not runtime state.
# `retained_boxes` joins them for the same reason: a retention DEADLINE that a
# restart forgot is a box that nobody follows to a terminal outcome. (The keep
# label on the box still self-expires — that is the primary mechanism and it is
# deliberately independent of this file — but the backstop, the outcome
# classification, and `fleet status` all read from here.)
# `launch_disk_gb` joins them for the price anchor's exact reason (task #69,
# 2026-08-08): it is the disk the WORKLOAD was launched at, and a restart that
# forgot it sends the next rehost back to re-deriving the size from whatever box
# it happens to be holding — which is how driftr3 went 110 -> 110 -> 60 GB and
# died on its own disk guard.
#
# `rebid_rungs` (autobid audit 2026-08-08) is durable for the mirror-image
# reason and one extra one: it is a bound on WALL TIME as well as money
# (REBID_MAX_RUNGS x REBID_WAIT_S = one replacement's setup cost), and a restart
# that forgot it would let a box that has already spent its whole ladder start
# another one — turning a bounded 15-minute stall into an unbounded loop of
# them. Unlike `replacements` it is per EVICTION CYCLE, so the supervise tick
# clears it on any return to live and on every replacement.
# moved-from: fleetd.REPLACEMENT_STATE_KEYS
REPLACEMENT_STATE_KEYS = ("replacements", "replacement_history",
                          "launch_dph_anchor", "launch_disk_gb",
                          # `launch_cc_allow` (2026-08-18) is the sm allowlist
                          # the launch declared, durable for the disk anchor's
                          # exact reason: the replacement lane cannot read it
                          # off an EVICTED box, and a restart that forgot it
                          # rehosts the workload onto any architecture the
                          # market is cheapest on — which is how two runs were
                          # voided by an sm_120 box in two days.
                          "launch_cc_allow",
                          # `launch_env_pin` (2026-08-17) is the allowlisted
                          # launch env — EVAL_ENV_VER — durable for the same
                          # reason the disk anchor is: a replacement cannot read
                          # it off the evicted box, and an unpinned eval box
                          # provisions eval-env/LATEST and grades on the wrong
                          # instrument.
                          "launch_env_pin",
                          "evicted_machines",
                          # `evicted_machine_ts` (2026-08-16) is the sidecar
                          # that makes an eviction exclusion EXPIRE: machine id
                          # (as a string, because state.json is JSON) ->
                          # {"ts", "class"}. Durable for the same reason the
                          # set is — a restart that forgot the timestamps would
                          # silently promote every TTL'd exclusion to permanent
                          # (the set survives, the clock does not), which is the
                          # pre-2026-08-16 behaviour it exists to end. A watch
                          # persisted before this key existed restores the set
                          # with no sidecar and every entry reads permanent:
                          # degraded, never wrong.
                          "evicted_machine_ts",
                          "retained_boxes", "rebid_rungs",
                          # `resume_tries` (2026-08-09) is the resume-in-place
                          # rung's per-eviction-cycle counter. A `start` spends
                          # nothing, so a restart that forgot it would not cost
                          # money the way a forgotten `rebid_rungs` would — but
                          # the ladder's shape should not change across a
                          # restart either, and both clear on return-to-live.
                          "resume_tries",
                          # `evicted_announced` (2026-08-14): the once-per-
                          # eviction-cycle announce latch. Its docstring
                          # promises "seventeen ticks are one event", but the
                          # latch lived only in memory — the two deploy
                          # restarts on 2026-08-14 re-announced 47694876's one
                          # eviction three times. The box-swap seams still
                          # pop it explicitly; persistence only stops a
                          # restart from re-announcing the SAME box's cycle.
                          "evicted_announced",
                          # `evicted_class` / `evicted_since` (2026-08-28) are
                          # the announce latch's other two halves: the class the
                          # cycle was journaled with, and when. They carry the
                          # host-stop escalation's bounded wait, which is the one
                          # deadline on the ladder measured from the EVICTION
                          # rather than armed by whichever rung last spent — so a
                          # restart that forgot them would restart the wait, and
                          # a claimed queue would sit parked for another
                          # `host_stop_escalate_s` per deploy. Popped together at
                          # all four cycle/box-swap seams
                          # (`job_lane._job_evicted_latch_reset`).
                          "evicted_class", "evicted_since",
                          # `notify_matched` / `notify_consumed_ids` (2026-08-16,
                          # NOTIFY_DESIGN S2b) are the announce latch's twins for
                          # vast's own outbid record: WHICH row explained this
                          # eviction cycle, and which rows a later cycle may no
                          # longer use. Durable for `evicted_announced`'s reason
                          # and one sharper one — the consumed set is what stops
                          # cycle N's row from PRICING cycle N+1's rescue, and a
                          # restart that forgot it would re-consume a row this
                          # process had already spent (instance 47833510 was
                          # evicted twice in one night). The rows themselves are
                          # NOT here: they live in the daemon's `notify` state
                          # and are re-fed every tick.
                          "notify_matched", "notify_consumed_ids",
                          # `rescue_deadline` / `rescue_put_failures`
                          # (2026-08-16, S2b review round 1 finding 2-4). The
                          # rescue rung's ONE-SHOT latch per eviction cycle is
                          # `rescue_deadline is not None` — and it lived only in
                          # memory while `notify_matched` beside it did not, so
                          # a restart mid-cycle re-armed a rescue the cycle had
                          # already spent and S2b handed it a price to re-spend
                          # it at. Probed: the PUT-ok arm self-corrects (the
                          # standing bid reconciles up, so the raise is no longer
                          # a raise), the PUT-429 arm re-fires at the same price.
                          # The behaviour change is strictly spend-REDUCING and
                          # it also closes a pre-S2b duplicate-rescue-after-
                          # restart shape that nobody had noticed, because
                          # pre-S2b that state usually had no price to fire on.
                          # Both clear on return-to-live and on every box swap,
                          # exactly like the latches above.
                          "rescue_deadline", "rescue_put_failures",
                          # Defense-controller observations (AUTOBID_DESIGN
                          # "Next iteration", 2026-08-09). `entry_floor` is the
                          # PRE-RENT market floor — unrecoverable after launch
                          # (post-rent the machine's min_bid reflects our own
                          # bid back, #73), so losing it to a restart loses the
                          # learn record's entry anchor for the whole rental.
                          # `p_alt`/`p_alt_ts` are the pre-eviction
                          # replacement-market read the one-shot defense prices
                          # against; the ts travels WITH the price so the
                          # freshness gate (_job_palt_fresh) keeps working
                          # across a restart instead of trusting a read of
                          # unknown age.
                          "entry_floor", "p_alt", "p_alt_ts",
                          # `bid_history` is the self-floor guard's echo window
                          # (2026-08-09): the standing-bid series this chunk has
                          # shown, which is what tells a stale echo of OUR bid
                          # apart from a competing bidder. Durable because the
                          # window (900 s) is far longer than a daemon restart,
                          # and a restart that forgot it would hand the defend
                          # ladder a self-referential floor to chase — the exact
                          # money bug the guard exists to stop, re-opened by a
                          # `fleet deploy`.
                          "bid_history",
                          # The replacement-ceiling re-pricing state
                          # (2026-08-24, REPLACEMENT_CEILING_WEDGE). All four
                          # are durable for the anchor's own reason — they are
                          # spend-BOUND inputs and a restart that forgot them
                          # would re-open the wedge. `replacement_market_floor`
                          # /`_ts` is the price a qualifying offer was last seen
                          # to bill (an over-ceiling refusal IS the market read
                          # on the pull lane); `replacement_refusals`/`_since`
                          # is the consecutive-refusal streak the derived
                          # `replacement_wedged` alarm fires on, and a restart
                          # that reset it would re-arm exactly the 33 minutes of
                          # silence the alarm exists to end.
                          "replacement_market_floor",
                          "replacement_market_floor_ts",
                          "replacement_refusals", "replacement_refusals_since",
                          "replacement_refusal_reason",
                          "replacement_refusal_ceiling")

#: Run-lane durable state (review 2026-08-10, F2). ONLY the echo window:
#: `last_bid` is deliberately NOT persisted — a restart re-derives it from the
#: box's own dph_base via the reconcile path, which cannot go stale, while a
#: persisted belief could. The echo window is the one thing only history
#: knows: without it, a bid moved shortly before a deploy re-arms the defend
#: against its own still-echoing prior price on the daemon's first ticks.
# moved-from: fleetd.RUN_STATE_KEYS
RUN_STATE_KEYS = ("bid_history",)


# moved-from: fleetd._replacement_state_restore
def _replacement_state_restore(jc: dict[str, Any], w: dict[str, Any] | None) -> None:
    """Seed a freshly-initialised `jc` with the eviction-replacement state the
    watch record carries (durable half of REPLACEMENT_STATE_KEYS).

    `evicted_machines` round-trips as a LIST because state.json is JSON and the
    ladder wants a set; everything else round-trips as-is, including
    `evicted_machine_ts`, whose keys are machine ids STRINGIFIED for the same
    JSON reason (`_job_excluded_machines` looks them up with `str(m)`, so an
    int-keyed in-memory dict and a string-keyed restored one behave
    identically). A watch that predates this feature has none of the keys and
    gets `job_supervise_init`'s zeros, which is the correct reading of "this
    watch has rented nothing yet"."""
    repl = (w or {}).get("replacement") or {}
    if not isinstance(repl, dict):
        return
    for k in REPLACEMENT_STATE_KEYS:
        if k not in repl:
            continue
        v = repl[k]
        jc[k] = set(v) if k == "evicted_machines" and isinstance(v, list) else v


# moved-from: fleetd._run_lane_state_restore
def _run_lane_state_restore(st: dict[str, Any], w: dict[str, Any] | None) -> None:
    """Seed a freshly-initialised run-lane `st` with the durable self-floor
    state the watch record carries. A watch that predates this has no record
    and starts with an empty window — degraded to standing-bid-only matching
    for the first lag_s, exactly the pre-persistence behaviour."""
    rec = (w or {}).get("run_state") or {}
    if not isinstance(rec, dict):
        return
    for k in RUN_STATE_KEYS:
        if k in rec:
            st[k] = rec[k]


# moved-from: fleetd._run_lane_state_persist
def _run_lane_state_persist(st: dict[str, Any], w: dict[str, Any] | None) -> None:
    """Mirror the run lane's durable self-floor state onto the watch record
    after every tick (the jobs lane's `_replacement_state_persist` pattern)."""
    if not isinstance(w, dict):
        return
    out = {k: st[k] for k in RUN_STATE_KEYS if k in st}
    if out:
        w["run_state"] = out


#: The serve lane's identity verdict + its condemn latch (P3, 2026-08-24). Its
#: own section rather than a REPLACEMENT_STATE_KEYS entry, because it is not
#: replacement state and a state.json a human reads should not say it is.
SERVE_IDENTITY_KEYS = ("serve_identity", "serve_identity_condemned")


def _serve_identity_persist(jc: dict[str, Any], w: dict[str, Any] | None) -> None:
    """Mirror the serve lane's identity verdict onto the watch record.

    fleetd DERIVES the identity alarms from `w["serve_identity"]` rather than
    latching them, so this write is what keeps the alarm burning tick after
    tick — and what makes a mismatch survive a daemon restart. The condemn
    latch rides with it for the same reason `replacements` does: a restart that
    forgot it would hand the ladder a fresh licence to rescue the very box it
    withdrew, which is the one outcome this whole check exists to prevent.

    Absent keys are POPPED, not left standing. A watch re-registered without
    `--expect-ident` is an operator saying "stop checking", and a verdict that
    outlived the pin it was made against would alarm forever with nothing able
    to retract it.
    """
    if not isinstance(w, dict):
        return
    for k in SERVE_IDENTITY_KEYS:
        if k in jc:
            w[k] = jc[k]
        else:
            w.pop(k, None)


def _serve_identity_restore(jc: dict[str, Any], w: dict[str, Any] | None) -> None:
    """Seed a rebuilt serve ladder with the identity verdict + condemn latch."""
    for k in SERVE_IDENTITY_KEYS:
        v = (w or {}).get(k)
        if v is not None:
            jc[k] = v


# moved-from: fleetd._replacement_state_persist
def _replacement_state_persist(jc: dict[str, Any], w: dict[str, Any] | None) -> None:
    """Mirror the ladder's replacement counters back onto the watch record after
    every tick, so `Store.save()`'s atomic write carries them. Cheap (four keys)
    and unconditional — a replacement can happen on any tick, and a counter that
    is only persisted on the tick that changed it is a counter that a crash
    between ticks silently rolls back."""
    if not isinstance(w, dict):
        return
    out = {}
    for k in REPLACEMENT_STATE_KEYS:
        if k not in jc:
            continue
        v = jc[k]
        out[k] = sorted(v) if isinstance(v, set) else v
    if out:
        w["replacement"] = out


# --------------------------------------------------------------------------- #
# persistence. The four `Fleet` methods, split into free functions (so a reader
# — or a test — can load, save and build a journal record without owning a
# daemon) plus the thin `Store` that binds them to a directory, a lock and a
# clock. The DECOMPOSITION is the only change: every body below is verbatim.
#
# DELIBERATELY MARKER-LESS (ruled 2026-08-16, wave 6a; fleetd-reexports H4):
# these four carried `# moved-from: fleetd.Fleet.{_load,save,_rotate_journal,
# journal}` markers and so did `daemon.Fleet`'s four delegating methods — two
# targets per flat name, which the test migration cannot rewrite. `daemon.Fleet`
# owns those mappings: it keeps the METHOD call shape every test that drives the
# tick loop patches (`monkeypatch.setattr(Fleet, "save", …)` has nowhere to land
# on a free function). What lives here is the split-out body, reached through
# the `Store`; the rename claim stays with the method.
# --------------------------------------------------------------------------- #
def load_state(state_path: str) -> dict[str, Any]:
    """Read `state.json` and apply the schema defaults. Never raises.

    A missing file is a fresh install. A CORRUPT file is moved aside to
    `<path>.corrupt-<epoch>` and rebuilt empty (S5: a corrupt state file must
    never crash-loop the daemon) — and note which sections are RESET rather than
    merged when they arrive with the wrong type: `notify`, `ceilings` and
    `ceiling_by_box`. An empty ceiling ledger means "every adoption gets the
    conservative provisional default", never "unlimited"."""
    st: dict[str, Any] = {}
    try:
        with open(state_path) as f:
            st = json.load(f)
        if not isinstance(st, dict):
            raise ValueError("state is not an object")
    except FileNotFoundError:
        st = {}
    except (ValueError, OSError) as e:
        # S5: a corrupt state file must never crash-loop the daemon.
        quarantine = f"{state_path}.corrupt-{int(time.time())}"
        try:
            shutil.move(state_path, quarantine)
            print(f"!! corrupt state.json quarantined to {quarantine} ({e}); "
                  f"rebuilding from the API + journal", file=sys.stderr)
        except OSError:
            pass
        st = {}
    st.setdefault("version", VERSION)
    st.setdefault("watches", {})
    st.setdefault("strays", {})
    st.setdefault("destroys", {})
    st.setdefault("intents", {})
    st.setdefault("spend_by_box", {})
    st.setdefault("meta", {})
    st.setdefault("alarms", {})       # LATCHED alarms only (see _derive_alarms)
    # The notification cursor + poll health (NOTIFY_DESIGN D3). Ours alone:
    # we never PUT `seen_through_at`, so this file is the only thing that
    # knows what we have consumed. A missing/garbage section is an EMPTY
    # cursor, which re-reads the server's window once — the safe direction,
    # since `notify_seen` rows are evidence and a duplicate costs nothing
    # but a line.
    if not isinstance(st.get("notify"), dict):
        st["notify"] = {}
    # The durable ceiling ledger. A missing section (fresh install, or the
    # S5 corrupt-state quarantine above) is EMPTY, and empty means "every
    # adoption gets the conservative provisional default" — never
    # "unlimited". See the ceiling-ledger block in fleet/rows.py.
    if not isinstance(st.get("ceilings"), dict):
        st["ceilings"] = {}
    if not isinstance(st.get("ceiling_by_box"), dict):
        st["ceiling_by_box"] = {}
    return st


def save_state(state: dict[str, Any], state_path: str) -> None:
    """Atomic temp+fsync+rename — a killed daemon never leaves a half-written
    state file (restart must lose nothing, S2).

    The three `json.dump` flags are WIRE FORMAT, not style: `indent=1` and
    `sort_keys=True` are what the step-0 snapshot was captured under and what a
    byte-comparison round-trip depends on, and `default=str` is what keeps a
    stray non-JSON object (a `set`, a `Namespace`) from turning one tick's save
    into a `TypeError` that loses the whole document."""
    tmp = state_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1, sort_keys=True, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, state_path)


def rotate_journal(journal_path: str) -> None:
    """Single-generation rotation past `JOURNAL_MAX_BYTES`: the previous `.1` is
    discarded. Every `OSError` is swallowed — a journal that cannot rotate is
    still a journal."""
    try:
        if os.path.getsize(journal_path) > JOURNAL_MAX_BYTES:
            os.replace(journal_path, journal_path + ".1")
    except OSError:
        pass


def journal_record(event: str, now: float, iid: object = None,
                   **fields: Any) -> dict[str, Any]:  # noqa: ANN401 — arbitrary event body
    """The FROZEN journal record: `ts` (3 dp), `ts_iso`, `event`, `iid` (a
    string, and only when one was given) and the caller's fields with every
    None-valued one DROPPED.

    Split out of `Store.journal` so the shape can be pinned without a
    filesystem; `herdd fleet journal`, `fleet_report.py` and the step-6
    cutover continuity check all read lines built here."""
    rec: dict[str, Any] = {"ts": round(now, 3), "ts_iso": iso(now), "event": event}
    if iid is not None:
        rec["iid"] = str(iid)
    rec.update({k: v for k, v in fields.items() if v is not None})
    return rec


class Store:
    """The persisted fleet document, its journal, and nothing else.

    The daemon owns one of these. `lock` is INJECTABLE and defaults to a private
    `RLock`: `fleetd.Fleet.save()` takes the daemon's own structural lock, so a
    `Store` with a lock of its own would quietly stop excluding the daemon's
    mutations from a save. `now` is the same seam `Fleet.hooks.now` is — pass
    `hooks.now`, not `time.time`, or a test's frozen clock stops reaching
    `meta.saved_ts` and every journal `ts`.
    """

    def __init__(self, dirpath: str | None = None,
                 now: Callable[[], float] | None = None,
                 lock: threading.RLock | None = None) -> None:
        self.dir = dirpath or state_dir()
        os.makedirs(self.dir, exist_ok=True)
        self.state_path = os.path.join(self.dir, STATE_NAME)
        self.journal_path = os.path.join(self.dir, JOURNAL_NAME)
        self.lock = lock if lock is not None else threading.RLock()
        self.now = now if now is not None else time.time
        self.state = self._load()

    def _load(self) -> dict[str, Any]:
        return load_state(self.state_path)

    def save(self) -> None:
        """Stamp `meta.saved_ts` and write atomically, under the lock."""
        with self.lock:
            self.state["meta"]["saved_ts"] = self.now()
            save_state(self.state, self.state_path)

    def _rotate_journal(self) -> None:
        rotate_journal(self.journal_path)

    def journal(self, event: str, iid: object = None,
                **fields: Any) -> dict[str, Any]:  # noqa: ANN401 — arbitrary event body
        """Append one journal line, ALSO print it to stdout, and return the
        record (callers consume the returned dict).

        journald is the second sink — the daemon's log IS the journal — so the
        print happens whether or not the file write did. A write failure prints
        to stderr and never stops a tick."""
        rec = journal_record(event, self.now(), iid, **fields)
        line = json.dumps(rec, sort_keys=True, default=str)
        try:
            self._rotate_journal()
            with open(self.journal_path, "a") as f:
                f.write(line + "\n")
        except OSError as e:                       # journal failure never stops a tick
            print(f"!! journal write failed: {e}", file=sys.stderr)
        print(line, flush=True)
        return rec


# moved-from: fleetd.acquire_single_instance_lock
def acquire_single_instance_lock(dirpath: str) -> TextIO | None:
    """flock the state dir — a stray second daemon must REFUSE to start (two
    reconcilers fighting over the same fleet is the worst possible bug).

    THE CALLER MUST HOLD THE RETURNED HANDLE for the process lifetime: a
    returned-and-dropped file object closes, and closing releases the flock."""
    path = os.path.join(dirpath, LOCK_NAME)
    fh = open(path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    fh.write(str(os.getpid()))
    fh.flush()
    return fh
